"""
Lambda entry point for the test runner.

This is the thinnest possible handler — it receives the trigger
event, assembles the components, and delegates immediately to
the core logic. No business logic lives here.

Two invocation patterns:
1. Direct invoke — triggered manually or by orchestrator
   event: { "persona": {...}, "scenario": {...}, "config": {...} }

2. WebSocket route — triggered by Twilio media stream
   event: { "requestContext": { "routeKey": "$connect" | "message" | "$disconnect" } }

API Gateway WebSocket routes Twilio's stream to this Lambda.
The same handler manages both the test setup and the live audio stream.
"""

import json
import base64
import asyncio
import boto3
from src.infrastructure.config import (
    ANTHROPIC_API_KEY,
    BEDROCK_MODEL_ID,
    RESULTS_BUCKET,
    RESULTS_TABLE,
    MAX_TURNS,
    MAX_CALL_DURATION
)
from src.backends.anthropic_backend import AnthropicBackend
from src.backends.bedrock_backend   import BedrockBackend
from src.personas.persona_generator import Persona
from src.personas.scenario_generator import Scenario, ScenarioStep
from src.lambdas.test_runner.caller_agent  import (
    CallerTurn, get_opening_utterance, respond_to_connect
)
from src.lambdas.test_runner.voice_caller  import TwilioVoiceCaller, AudioChunk
from src.lambdas.test_runner.audio_bridge  import AudioBridge
from src.backends.base_backend import Message


# ── WebSocket connection registry ──────────────────────────────────────────
# Maps connection_id → active test session
# In production this would be DynamoDB — Lambda is stateless
# For now in-memory works for single-instance testing
_active_sessions = {}


def lambda_handler(event: dict, context) -> dict:
    """
    Main Lambda entry point.
    Routes to the correct handler based on event type.
    """
    # WebSocket event from API Gateway
    if "requestContext" in event:
        route = event["requestContext"].get("routeKey", "")

        if route == "$connect":
            return _handle_ws_connect(event)
        elif route == "message":
            return _handle_ws_message(event)
        elif route == "$disconnect":
            return _handle_ws_disconnect(event)

    # Direct invocation — start a new test run
    return _handle_test_run(event, context)


# ── Direct invocation ──────────────────────────────────────────────────────

def _handle_test_run(event: dict, context) -> dict:
    """
    Start a complete voice test run.

    Expects event:
    {
        "persona":  { ... persona fields ... },
        "scenario": { ... scenario fields ... },
        "backend":  "anthropic" | "bedrock",
        "stream_url": "wss://your-endpoint/stream"
    }
    """
    try:
        # ── Assemble components ────────────────────────────────
        backend_name = event.get("backend", "anthropic")
        backend      = _get_backend(backend_name)

        persona  = _deserialize_persona(event["persona"])
        scenario = _deserialize_scenario(event["scenario"])

        stream_url = event.get("stream_url", "")
        if not stream_url:
            return _error("stream_url is required for voice testing")

        # ── Initialize components ──────────────────────────────
        audio_bridge = AudioBridge(emotional_state=persona.emotional_state)
        voice_caller = TwilioVoiceCaller(stream_url=stream_url)

        print(f"[Handler] Starting test run — persona: {persona.name}, backend: {backend_name}")

        # ── Run the test ───────────────────────────────────────
        result = _run_voice_test(
            persona      = persona,
            scenario     = scenario,
            backend      = backend,
            audio_bridge = audio_bridge,
            voice_caller = voice_caller
        )

        # ── Store result ───────────────────────────────────────
        _store_result(result)

        return {
            "statusCode": 200,
            "body":       json.dumps(result)
        }

    except Exception as e:
        print(f"[Handler] Test run failed: {e}")
        return _error(str(e))


def _run_voice_test(
    persona:      Persona,
    scenario:     Scenario,
    backend,
    audio_bridge: AudioBridge,
    voice_caller: TwilioVoiceCaller
) -> dict:
    """
    Execute the full voice test — dial, converse, hang up.
    Returns a complete test result dict for the evaluator.
    """
    transcript    = []
    turn_number   = 0
    current_step  = 0
    caller_turns  = []

    # ── Step 1: Dial Connect ───────────────────────────────────
    session = voice_caller.initiate_call()
    answered = voice_caller.wait_for_answer(timeout=30)

    if not answered:
        return _failed_result(persona, scenario, "Call not answered")

    # ── Step 2: Opening utterance ──────────────────────────────
    # Caller speaks first when call connects
    opening = get_opening_utterance(persona, scenario, backend)
    caller_turns.append(opening)

    audio_segment = audio_bridge.synthesize_speech(
        text           = opening.utterance,
        emotional_state = persona.emotional_state
    )
    voice_caller.send_audio(audio_segment.audio_bytes)

    transcript.append({
        "turn":    turn_number,
        "speaker": "caller",
        "text":    opening.utterance,
        "sentiment": opening.sentiment,
        "internal_thought": opening.internal_thought
    })
    turn_number += 1

    # ── Step 3: Conversation loop ──────────────────────────────
    conversation_history = [
        Message(role="user", content=opening.utterance)
    ]

    while turn_number < MAX_TURNS:
        # Wait for Connect to respond
        connect_audio = _wait_for_connect_utterance(voice_caller)

        if connect_audio is None:
            print(f"[Handler] No audio from Connect — ending call")
            break

        # Transcribe what Connect said
        connect_text = audio_bridge.transcribe_utterance(
            audio_bytes = connect_audio,
            temp_bucket = RESULTS_BUCKET
        )

        if not connect_text.text.strip():
            print(f"[Handler] Empty transcription — skipping turn")
            continue

        transcript.append({
            "turn":       turn_number,
            "speaker":    "connect",
            "text":       connect_text.text,
            "confidence": connect_text.confidence
        })
        turn_number += 1

        # Add Connect's response to conversation history
        conversation_history.append(
            Message(role="assistant", content=connect_text.text)
        )

        # Get current scenario step
        step = scenario.steps[current_step] if current_step < len(scenario.steps) else scenario.steps[-1]

        # Caller agent responds
        caller_turn = respond_to_connect(
            connect_utterance    = connect_text.text,
            conversation_history = conversation_history,
            persona              = persona,
            scenario             = scenario,
            current_step         = step,
            turn_number          = turn_number,
            backend              = backend
        )
        caller_turns.append(caller_turn)

        # Advance scenario step if completed
        if caller_turn.current_step_done and current_step < len(scenario.steps) - 1:
            current_step += 1
            print(f"[Handler] Step {current_step} completed — advancing")

        # Synthesize and send caller's response
        audio_segment = audio_bridge.synthesize_speech(
            text            = caller_turn.utterance,
            emotional_state = caller_turn.sentiment
        )
        voice_caller.send_audio(audio_segment.audio_bytes)

        transcript.append({
            "turn":             turn_number,
            "speaker":          "caller",
            "text":             caller_turn.utterance,
            "sentiment":        caller_turn.sentiment,
            "internal_thought": caller_turn.internal_thought,
            "step_completed":   caller_turn.current_step_done
        })
        turn_number += 1

        # Add caller's response to history
        conversation_history.append(
            Message(role="user", content=caller_turn.utterance)
        )

        # Check if caller decided to end the call
        if caller_turn.should_end_call:
            print(f"[Handler] Caller ending call — reason: {caller_turn.end_reason}")
            break

    # ── Step 4: Hang up ────────────────────────────────────────
    call_session = voice_caller.end_call()

    # ── Step 5: Build result ───────────────────────────────────
    steps_completed = current_step
    total_steps     = len(scenario.steps)
    end_reason      = caller_turns[-1].end_reason if caller_turns else "max_turns"

    return {
        "test_id":        f"{persona.name}-{scenario.name}-{int(__import__('time').time())}",
        "persona":        persona.name,
        "scenario":       scenario.name,
        "backend":        backend.get_backend_name(),
        "transcript":     transcript,
        "steps_completed": steps_completed,
        "total_steps":    total_steps,
        "total_turns":    turn_number,
        "end_reason":     end_reason,
        "call_duration":  call_session.duration_seconds if call_session else 0,
        "status":         "completed"
    }


# ── WebSocket handlers ─────────────────────────────────────────────────────

def _handle_ws_connect(event: dict) -> dict:
    """
    API Gateway WebSocket $connect route.
    Called when Twilio opens the media stream connection.
    Registers the connection for the active test session.
    """
    connection_id = event["requestContext"]["connectionId"]
    print(f"[Handler] WebSocket connected — ID: {connection_id}")

    _active_sessions[connection_id] = {
        "connected_at": __import__("time").time(),
        "audio_buffer": [],
        "pending_send": None
    }

    return {"statusCode": 200}


def _handle_ws_message(event: dict) -> dict:
    """
    API Gateway WebSocket message route.
    Called for every audio packet Twilio sends.

    Twilio media stream messages are JSON with a base64-encoded
    audio payload. We decode and buffer the audio for the test loop.
    """
    connection_id = event["requestContext"]["connectionId"]
    body          = json.loads(event.get("body", "{}"))
    event_type    = body.get("event", "")

    if event_type == "media":
        # Decode base64 audio payload from Twilio
        payload    = base64.b64decode(body["media"]["payload"])
        track      = body["media"].get("track", "inbound")
        sequence   = body["media"].get("sequenceNumber", 0)

        chunk = AudioChunk(
            payload   = payload,
            timestamp = __import__("time").time(),
            track     = track,
            sequence  = int(sequence)
        )

        # Buffer the audio chunk for the test loop to consume
        if connection_id in _active_sessions:
            _active_sessions[connection_id]["audio_buffer"].append(chunk)

    elif event_type == "stop":
        print(f"[Handler] Twilio stream stopped — connection: {connection_id}")

    return {"statusCode": 200}


def _handle_ws_disconnect(event: dict) -> dict:
    """
    API Gateway WebSocket $disconnect route.
    Called when Twilio closes the media stream.
    Clean up the session.
    """
    connection_id = event["requestContext"]["connectionId"]
    print(f"[Handler] WebSocket disconnected — ID: {connection_id}")

    if connection_id in _active_sessions:
        del _active_sessions[connection_id]

    return {"statusCode": 200}


# ── Helpers ────────────────────────────────────────────────────────────────

def _wait_for_connect_utterance(
    voice_caller: TwilioVoiceCaller,
    timeout:      int = 15
) -> bytes | None:
    """
    Wait for Connect to finish speaking and return the audio.
    Polls voice_caller for a complete utterance (silence detected).
    Returns None if timeout reached with no audio.
    """
    import time
    start = time.time()

    while time.time() - start < timeout:
        audio = voice_caller.get_utterance_audio(silence_threshold_ms=800)
        if audio:
            return audio
        time.sleep(0.1)

    return None


def _get_backend(backend_name: str):
    """Return the appropriate backend instance."""
    if backend_name == "bedrock":
        return BedrockBackend()
    return AnthropicBackend()


def _store_result(result: dict) -> None:
    """Store test result to DynamoDB and S3."""
    try:
        # DynamoDB — for fast lookup and querying
        dynamodb = boto3.resource("dynamodb")
        table    = dynamodb.Table(RESULTS_TABLE)
        table.put_item(Item={
            "test_id":         result["test_id"],
            "persona":         result["persona"],
            "scenario":        result["scenario"],
            "backend":         result["backend"],
            "steps_completed": result["steps_completed"],
            "total_steps":     result["total_steps"],
            "total_turns":     result["total_turns"],
            "end_reason":      result["end_reason"],
            "status":          result["status"]
        })

        # S3 — full transcript and result JSON for deep analysis
        s3  = boto3.client("s3")
        key = f"results/{result['test_id']}.json"
        s3.put_object(
            Bucket      = RESULTS_BUCKET,
            Key         = key,
            Body        = json.dumps(result, indent=2),
            ContentType = "application/json"
        )

        print(f"[Handler] Result stored — test_id: {result['test_id']}")

    except Exception as e:
        print(f"[Handler] Failed to store result: {e}")


def _deserialize_persona(data: dict) -> Persona:
    """Reconstruct a Persona dataclass from a dict."""
    return Persona(
        name                = data["name"],
        age                 = data["age"],
        emotional_state     = data["emotional_state"],
        patience_level      = data["patience_level"],
        tech_savviness      = data["tech_savviness"],
        communication_style = data["communication_style"],
        goal                = data["goal"],
        background          = data["background"],
        speech_patterns     = data.get("speech_patterns", []),
        edge_case_traits    = data.get("edge_case_traits", [])
    )


def _deserialize_scenario(data: dict) -> Scenario:
    """Reconstruct a Scenario dataclass from a dict."""
    from src.personas.scenario_generator import ScenarioStep
    steps = [
        ScenarioStep(
            intent           = s["intent"],
            expected_outcome = s["expected_outcome"],
            fallback_phrase  = s["fallback_phrase"],
            is_edge_case     = s.get("is_edge_case", False)
        )
        for s in data["steps"]
    ]
    return Scenario(
        name                    = data["name"],
        description             = data["description"],
        flow_type               = data["flow_type"],
        persona_name            = data["persona_name"],
        steps                   = steps,
        success_criteria        = data["success_criteria"],
        expected_duration_turns = data["expected_duration_turns"],
        tags                    = data.get("tags", [])
    )


def _failed_result(persona: Persona, scenario: Scenario, reason: str) -> dict:
    """Build a failed test result."""
    return {
        "test_id":         f"{persona.name}-{scenario.name}-failed",
        "persona":         persona.name,
        "scenario":        scenario.name,
        "backend":         "unknown",
        "transcript":      [],
        "steps_completed": 0,
        "total_steps":     len(scenario.steps),
        "total_turns":     0,
        "end_reason":      reason,
        "call_duration":   0,
        "status":          "failed"
    }


def _error(message: str) -> dict:
    """Return a Lambda error response."""
    return {
        "statusCode": 500,
        "body":       json.dumps({"error": message})
    }


"""
The handler is the most complex file so far because it does three different jobs depending on how Lambda is invoked
- direct test run, WebSocket connect, or WebSocket message.
The _active_sessions dict is the in-memory bridge between the WebSocket audio stream and the test loop.
In production that would move to DynamoDB since Lambda can spin up multiple instances and they wouldn't share memory.
"""