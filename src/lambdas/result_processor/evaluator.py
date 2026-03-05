"""
Scores a completed test conversation against the scenario's success criteria.

Think of this like a QA manager reviewing a call recording —
they watch the full transcript, score each dimension independently,
and produce a structured assessment of what worked and what didn't.

The evaluator never participates in the call. It only reads the
completed transcript and produces objective scores. This separation
means the same evaluator works for both voice mode and API mode.
"""

import json
from dataclasses import dataclass, field
from src.backends.base_backend import BaseBackend, Message
from src.personas.scenario_generator import Scenario


@dataclass
class StepResult:
    """
    Evaluation of a single scenario step.
    Was the intent accomplished? How many turns did it take?
    """
    intent:          str
    completed:       bool
    turns_taken:     int
    first_attempt:   bool     # completed on first try or required fallback
    notes:           str      = ""


@dataclass
class EvaluationResult:
    """
    Complete evaluation of one test run.
    Structured scores across every dimension we care about.
    """
    test_id:              str
    persona:              str
    scenario:             str
    backend:              str

    # ── Scores (0-10) ──────────────────────────────────────────
    goal_completion:      float   # did the caller accomplish their overall goal
    turn_efficiency:      float   # how quickly goals were accomplished
    intent_accuracy:      float   # did the flow understand intents correctly
    sentiment_trajectory: float   # did sentiment stay stable or deteriorate
    fallout_risk:         float   # how likely a real customer would have hung up

    # ── Detailed findings ──────────────────────────────────────
    overall_score:        float
    grade:                str     # A | B | C | D | F
    step_results:         list[StepResult] = field(default_factory=list)
    fallout_points:       list[str]        = field(default_factory=list)
    strengths:            list[str]        = field(default_factory=list)
    weaknesses:           list[str]        = field(default_factory=list)
    recommendations:      list[str]        = field(default_factory=list)
    summary:              str              = ""


EVALUATOR_SYSTEM_PROMPT = """You are an expert contact center quality analyst.
You evaluate automated contact flow test runs objectively and specifically.

You identify:
- Where the flow failed to understand the caller
- Where the caller had to repeat themselves
- Where sentiment deteriorated
- Where a real customer would likely have abandoned the call
- What worked well and should be preserved

Be specific and actionable. Vague feedback is not useful."""


def evaluate_test_run(
    test_result: dict,
    scenario:    Scenario,
    backend:     BaseBackend
) -> EvaluationResult:
    """
    Evaluate a completed test run and produce structured scores.

    test_result: the dict produced by handler._run_voice_test()
    scenario:    the original scenario definition for context
    backend:     AI backend for analysis
    """
    transcript     = test_result.get("transcript", [])
    steps_completed = test_result.get("steps_completed", 0)
    total_steps    = test_result.get("total_steps", len(scenario.steps))
    total_turns    = test_result.get("total_turns", 0)
    end_reason     = test_result.get("end_reason", "")

    # ── Format transcript for the evaluator prompt ─────────────
    transcript_text = _format_transcript(transcript)

    # ── Format scenario steps for context ──────────────────────
    steps_text = "\n".join(
        f"  Step {i+1}: {s.intent} (expected: {s.expected_outcome})"
        for i, s in enumerate(scenario.steps)
    )

    # ── Format success criteria ─────────────────────────────────
    criteria_text = "\n".join(
        f"  - {c}" for c in scenario.success_criteria
    )

    prompt = f"""Evaluate this contact center test run.

Scenario: {scenario.name}
Description: {scenario.description}
Expected steps:
{steps_text}

Success criteria:
{criteria_text}

Steps completed: {steps_completed}/{total_steps}
Total turns: {total_turns}
Call end reason: {end_reason}

Full transcript:
{transcript_text}

Score each dimension 0-10 and provide detailed findings.

Respond with ONLY a JSON object:
{{
  "goal_completion":      <0-10, did caller accomplish overall goal>,
  "turn_efficiency":      <0-10, 10=completed in minimum turns>,
  "intent_accuracy":      <0-10, did flow understand intents correctly>,
  "sentiment_trajectory": <0-10, 10=sentiment stayed positive throughout>,
  "fallout_risk":         <0-10, 10=very low risk of real customer abandoning>,
  "overall_score":        <0-10, weighted average>,
  "grade":                "A" | "B" | "C" | "D" | "F",
  "step_results": [
    {{
      "intent":        "step intent",
      "completed":     true | false,
      "turns_taken":   <integer>,
      "first_attempt": true | false,
      "notes":         "specific observation about this step"
    }}
  ],
  "fallout_points": [
    "specific moment where a real customer might have abandoned"
  ],
  "strengths": [
    "specific thing the flow handled well"
  ],
  "weaknesses": [
    "specific thing the flow handled poorly"
  ],
  "recommendations": [
    "specific actionable improvement"
  ],
  "summary": "2-3 paragraph assessment of overall flow quality"
}}

No markdown, no explanation outside the JSON."""

    response = backend.generate(
        system_prompt = EVALUATOR_SYSTEM_PROMPT,
        messages      = [Message(role="user", content=prompt)],
        max_tokens    = 2000,
        temperature   = 0.3    # low temperature for consistent, objective scoring
    )

    raw = response.text.strip().replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)

        step_results = [
            StepResult(
                intent        = s["intent"],
                completed     = s["completed"],
                turns_taken   = s["turns_taken"],
                first_attempt = s["first_attempt"],
                notes         = s.get("notes", "")
            )
            for s in data.get("step_results", [])
        ]

        return EvaluationResult(
            test_id              = test_result.get("test_id", ""),
            persona              = test_result.get("persona", ""),
            scenario             = test_result.get("scenario", ""),
            backend              = test_result.get("backend", ""),
            goal_completion      = data["goal_completion"],
            turn_efficiency      = data["turn_efficiency"],
            intent_accuracy      = data["intent_accuracy"],
            sentiment_trajectory = data["sentiment_trajectory"],
            fallout_risk         = data["fallout_risk"],
            overall_score        = data["overall_score"],
            grade                = data["grade"],
            step_results         = step_results,
            fallout_points       = data.get("fallout_points", []),
            strengths            = data.get("strengths", []),
            weaknesses           = data.get("weaknesses", []),
            recommendations      = data.get("recommendations", []),
            summary              = data.get("summary", "")
        )

    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Evaluator] Parse failed: {e}")
        return _default_evaluation(test_result)


def _format_transcript(transcript: list[dict]) -> str:
    """
    Format transcript turns into readable text for the evaluator prompt.
    Includes internal thoughts so evaluator understands caller intent.
    """
    lines = []
    for turn in transcript:
        speaker  = turn["speaker"].upper()
        text     = turn["text"]
        thought  = turn.get("internal_thought", "")
        sentiment = turn.get("sentiment", "")

        line = f"[Turn {turn['turn']}] {speaker}: {text}"
        if thought:
            line += f"\n  (internal: {thought})"
        if sentiment and speaker == "CALLER":
            line += f"  [{sentiment}]"

        lines.append(line)

    return "\n".join(lines)


def _default_evaluation(test_result: dict) -> EvaluationResult:
    """Return a safe default when evaluation parsing fails."""
    return EvaluationResult(
        test_id              = test_result.get("test_id", ""),
        persona              = test_result.get("persona", ""),
        scenario             = test_result.get("scenario", ""),
        backend              = test_result.get("backend", ""),
        goal_completion      = 5.0,
        turn_efficiency      = 5.0,
        intent_accuracy      = 5.0,
        sentiment_trajectory = 5.0,
        fallout_risk         = 5.0,
        overall_score        = 5.0,
        grade                = "C",
        summary              = "Evaluation parse failed — manual review required."
    )