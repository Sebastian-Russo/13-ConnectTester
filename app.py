"""
Local Flask server for development and testing.

In production, Lambda handles everything. Locally, this Flask
server lets you run the full test suite, trigger individual tests,
and view reports without deploying to AWS.

Same pattern as every other project — thin HTTP layer that
delegates immediately to the orchestrator.
"""

import json
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from src.orchestrator import Orchestrator, TestSuiteConfig
from src.backends.anthropic_backend import AnthropicBackend
from src.backends.bedrock_backend import BedrockBackend
from src.personas.persona_generator import generate_persona, generate_persona_batch
from src.personas.scenario_generator import generate_scenario

app = Flask(__name__, static_folder="dashboard")
CORS(app)


# ── Dashboard ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("dashboard", "index.html")


# ── Test suite ─────────────────────────────────────────────────────────────

@app.route("/suite/run", methods=["POST"])
def run_suite():
    """
    Trigger a full test suite run.

    Body:
    {
        "flow_type":     "hotel_reservation",
        "flow_context":  "handles room booking, check-in times, pet policy...",
        "stream_url":    "wss://your-ngrok-url/stream",
        "persona_count": 3,
        "backends":      ["anthropic", "bedrock"],
        "trait_matrix":  [["frustrated"], ["calm"], ["confused"]],
        "tag_matrix":    [["happy_path"], ["edge_case"], ["angry_caller"]]
    }
    """
    data = request.get_json()

    if not data.get("flow_type"):
        return jsonify({"error": "flow_type is required"}), 400
    if not data.get("flow_context"):
        return jsonify({"error": "flow_context is required"}), 400
    if not data.get("stream_url"):
        return jsonify({"error": "stream_url is required for voice testing"}), 400

    config = TestSuiteConfig(
        flow_type     = data["flow_type"],
        flow_context  = data["flow_context"],
        stream_url    = data["stream_url"],
        persona_count = data.get("persona_count", 3),
        backends      = data.get("backends", ["anthropic", "bedrock"]),
        trait_matrix  = data.get("trait_matrix", []),
        tag_matrix    = data.get("tag_matrix", [])
    )

    try:
        orchestrator = Orchestrator(config)
        result       = orchestrator.execute()
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Single test ────────────────────────────────────────────────────────────

@app.route("/test/run", methods=["POST"])
def run_single_test():
    """
    Run a single voice test with a specific persona and scenario.
    Useful for debugging a specific flow failure.

    Body:
    {
        "persona":    { ... persona fields ... },
        "scenario":   { ... scenario fields ... },
        "backend":    "anthropic" | "bedrock",
        "stream_url": "wss://your-ngrok-url/stream"
    }
    """
    data = request.get_json()

    if not data.get("persona"):
        return jsonify({"error": "persona is required"}), 400
    if not data.get("scenario"):
        return jsonify({"error": "scenario is required"}), 400
    if not data.get("stream_url"):
        return jsonify({"error": "stream_url is required"}), 400

    from src.lambdas.test_runner.handler import (
        _run_voice_test, _deserialize_persona, _deserialize_scenario
    )
    from src.lambdas.test_runner.voice_caller import TwilioVoiceCaller
    from src.lambdas.test_runner.audio_bridge import AudioBridge

    persona      = _deserialize_persona(data["persona"])
    scenario     = _deserialize_scenario(data["scenario"])
    backend_name = data.get("backend", "anthropic")
    backend      = AnthropicBackend() if backend_name == "anthropic" else BedrockBackend()

    try:
        audio_bridge = AudioBridge(emotional_state=persona.emotional_state)
        voice_caller = TwilioVoiceCaller(stream_url=data["stream_url"])

        result = _run_voice_test(
            persona      = persona,
            scenario     = scenario,
            backend      = backend,
            audio_bridge = audio_bridge,
            voice_caller = voice_caller
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Persona and scenario generation ───────────────────────────────────────

@app.route("/personas/generate", methods=["POST"])
def generate_personas():
    """
    Generate personas without running a test.
    Useful for previewing personas before committing to a full suite run.

    Body:
    {
        "scenario_type": "hotel_reservation",
        "count":         3,
        "backend":       "anthropic",
        "trait_matrix":  [["frustrated"], ["calm"], ["confused"]]
    }
    """
    data         = request.get_json()
    scenario_type = data.get("scenario_type", "hotel_reservation")
    count        = data.get("count", 3)
    backend_name = data.get("backend", "anthropic")
    backend      = AnthropicBackend() if backend_name == "anthropic" else BedrockBackend()

    try:
        import dataclasses
        personas = generate_persona_batch(
            backend       = backend,
            scenario_type = scenario_type,
            count         = count,
            trait_matrix  = data.get("trait_matrix", [])
        )
        return jsonify([dataclasses.asdict(p) for p in personas])

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/scenarios/generate", methods=["POST"])
def generate_scenarios():
    """
    Generate a scenario for a specific persona.
    Useful for previewing scenarios before a suite run.

    Body:
    {
        "persona":       { ... persona fields ... },
        "flow_type":     "hotel_reservation",
        "flow_context":  "handles room booking, check-in times, pet policy...",
        "backend":       "anthropic",
        "tags":          ["happy_path"]
    }
    """
    data         = request.get_json()
    backend_name = data.get("backend", "anthropic")
    backend      = AnthropicBackend() if backend_name == "anthropic" else BedrockBackend()

    from src.lambdas.test_runner.handler import _deserialize_persona
    from src.personas.scenario_generator import generate_scenario
    import dataclasses

    try:
        persona  = _deserialize_persona(data["persona"])
        scenario = generate_scenario(
            backend      = backend,
            persona      = persona,
            flow_type    = data.get("flow_type", "hotel_reservation"),
            flow_context = data.get("flow_context", ""),
            tags         = data.get("tags", [])
        )
        return jsonify(dataclasses.asdict(scenario))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Health ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":   "ok",
        "backends": ["anthropic", "bedrock"],
        "mode":     "voice"
    })


# ── Twilio WebSocket stream endpoint ──────────────────────────────────────

@app.route("/stream", methods=["POST"])
def twilio_stream():
    """
    Twilio sends a POST to this endpoint when a call connects.
    Returns TwiML instructing Twilio to open a media stream
    to our WebSocket endpoint.

    In development: use ngrok to expose this endpoint publicly
    so Twilio can reach it.
    """
    from twilio.twiml.voice_response import VoiceResponse, Start
    from src.infrastructure.config import MAX_CALL_DURATION

    host     = request.host
    response = VoiceResponse()
    start    = Start()
    start.stream(url=f"wss://{host}/stream/ws", track="both_tracks")
    response.append(start)
    response.pause(length=MAX_CALL_DURATION)

    return str(response), 200, {"Content-Type": "text/xml"}


if __name__ == "__main__":
    app.run(debug=True, port=5000)
