"""
The AI that plays the customer role during a test call.

Think of this like an actor who has been given a character brief
and a loose scene outline — they know who they are, what they want,
and roughly what needs to happen, but they respond naturally to
whatever the other actor says rather than reading from a script.

This agent receives what Connect just said, the full conversation
history, the persona it's embodying, and the current scenario step.
It decides what to say next and whether the step was completed.
"""

import json
from dataclasses import dataclass
from src.backends.base_backend import BaseBackend, Message
from src.personas.persona_generator import Persona
from src.personas.scenario_generator import Scenario, ScenarioStep


@dataclass
class CallerTurn:
    """
    The caller agent's response for a single conversation turn.
    Not just what to say — also state tracking so the orchestrator
    knows how the test is progressing.
    """
    utterance:         str    # what the caller says out loud
    current_step_done: bool   # did this turn complete the current scenario step
    sentiment:         str    # "positive" | "neutral" | "frustrated" | "angry"
    should_end_call:   bool   # caller decides to hang up (success or frustration)
    end_reason:        str    # "goal_achieved" | "frustrated_hangup" | "max_turns" | ""
    internal_thought:  str    # caller's internal state — not spoken, used for evaluation


CALLER_SYSTEM_PROMPT = """You are simulating a real customer calling into an automated contact center.
You must stay completely in character as the persona you are given.
You are NOT a helpful AI assistant — you are a real person with a specific goal,
emotional state, and communication style.

Critical rules:
- Never break character or reveal you are an AI
- Respond only as the persona would — match their communication style exactly
- React authentically to frustration triggers (repetition, misunderstanding, long waits)
- Do not be overly cooperative — real customers push back, mishear things, go off topic
- Keep utterances to 1-3 sentences — real phone conversations are short turns"""


def build_persona_context(persona: Persona, scenario: Scenario, current_step: ScenarioStep, turn_number: int) -> str:
    """
    Build the full context prompt for the caller agent.
    Called every turn with the current state of the conversation.
    """
    return f"""You are {persona.name}, {persona.age} years old.

Your emotional state: {persona.emotional_state}
Your patience level: {persona.patience_level}
Your communication style: {persona.communication_style}
Your speech patterns: {', '.join(persona.speech_patterns)}
Your edge case traits: {', '.join(persona.edge_case_traits)}

Your overall goal: {persona.goal}
Your background: {persona.background}

Current scenario step ({turn_number} turns elapsed):
- You are trying to: {current_step.intent}
- If not understood, you can try saying: "{current_step.fallback_phrase}"
- This is {"an edge case step" if current_step.is_edge_case else "a standard step"}

Respond with ONLY a JSON object:
{{
  "utterance":         "exactly what you say out loud — stay in character",
  "current_step_done": true | false,
  "sentiment":         "positive" | "neutral" | "frustrated" | "angry",
  "should_end_call":   true | false,
  "end_reason":        "goal_achieved" | "frustrated_hangup" | "",
  "internal_thought":  "your internal state in one sentence — not spoken aloud"
}}

No markdown, no explanation outside the JSON."""


def get_opening_utterance(persona: Persona, scenario: Scenario, backend: BaseBackend) -> CallerTurn:
    """
    Generate the caller's very first utterance when the call connects.
    Before Connect has said anything — the caller speaks first.
    """
    prompt = f"""You are {persona.name}, {persona.age} years old, calling a contact center.
Your goal: {persona.goal}
Your emotional state: {persona.emotional_state}
Your communication style: {persona.communication_style}

Generate your opening statement when the call connects.

Respond with ONLY a JSON object:
{{
  "utterance":         "your opening words",
  "current_step_done": false,
  "sentiment":         "positive" | "neutral" | "frustrated" | "angry",
  "should_end_call":   false,
  "end_reason":        "",
  "internal_thought":  "your internal state"
}}

No markdown, no explanation outside the JSON."""

    response = backend.generate(
        system_prompt = CALLER_SYSTEM_PROMPT,
        messages      = [Message(role="user", content=prompt)],
        max_tokens    = 300,
        temperature   = 0.8
    )

    return _parse_caller_turn(response.text)


def respond_to_connect(
    connect_utterance: str,
    conversation_history: list[Message],
    persona:          Persona,
    scenario:         Scenario,
    current_step:     ScenarioStep,
    turn_number:      int,
    backend:          BaseBackend
) -> CallerTurn:
    """
    Generate the caller's response to what Connect just said.

    connect_utterance:    what the contact flow just said (transcribed from audio)
    conversation_history: full transcript so far as Message list
    persona:              the character being played
    scenario:             the test scenario
    current_step:         which scenario step we're currently on
    turn_number:          how many turns have elapsed
    backend:              which AI backend to use
    """
    context = build_persona_context(persona, scenario, current_step, turn_number)

    # Add Connect's latest utterance to history
    messages = conversation_history + [
        Message(role="user", content=f"The contact center just said: \"{connect_utterance}\"\n\n{context}")
    ]

    response = backend.generate(
        system_prompt = CALLER_SYSTEM_PROMPT,
        messages      = messages,
        max_tokens    = 400,
        temperature   = 0.7
    )

    return _parse_caller_turn(response.text)


def _parse_caller_turn(raw_text: str) -> CallerTurn:
    """Parse the JSON response from the caller agent into a CallerTurn."""
    raw = raw_text.strip().replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
        return CallerTurn(
            utterance         = data["utterance"],
            current_step_done = data.get("current_step_done", False),
            sentiment         = data.get("sentiment", "neutral"),
            should_end_call   = data.get("should_end_call", False),
            end_reason        = data.get("end_reason", ""),
            internal_thought  = data.get("internal_thought", "")
        )
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[CallerAgent] Parse failed: {e} — raw: {raw[:200]}")
        # Return a safe default that keeps the call going
        return CallerTurn(
            utterance         = "I'm sorry, could you repeat that?",
            current_step_done = False,
            sentiment         = "neutral",
            should_end_call   = False,
            end_reason        = "",
            internal_thought  = "parse error — defaulting to repeat request"
        )


"""
The internal_thought field is important —
it never gets spoken but gets logged with the transcript.
The evaluator uses it later to understand why the caller behaved a certain way,
which produces much richer test reports than just looking at the raw conversation.
"""
