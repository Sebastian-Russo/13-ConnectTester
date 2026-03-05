"""
Think of this like a screenplay writer for a phone call.
Given a persona and a contact flow type, it generates the
ordered sequence of intents the caller will try to accomplish —
including how they'll phrase things, where they might go off
script, and what success looks like for this specific caller.

The scenario is the test plan. The persona is who executes it.
Together they define one complete test case.
"""

import json
from dataclasses import dataclass, field
from src.backends.base_backend import BaseBackend, Message
from src.personas.persona_generator import Persona


@dataclass
class ScenarioStep:
    """
    A single intent or action in the test scenario.
    The caller agent uses this as a guide — not a script.
    It knows what it wants to accomplish but decides how
    to phrase it based on the persona's communication style.
    """
    intent:          str          # what the caller wants e.g. "book a room"
    expected_outcome: str         # what a successful response looks like
    fallback_phrase:  str         # what to say if the flow doesn't understand
    is_edge_case:     bool = False # whether this step intentionally tests a boundary


@dataclass
class Scenario:
    """
    A complete test scenario — the full arc of one phone call.
    """
    name:            str
    description:     str
    flow_type:       str                      # e.g. "hotel_reservation"
    persona_name:    str                      # which persona runs this scenario
    steps:           list[ScenarioStep]       # ordered list of intents
    success_criteria: list[str]               # what defines a passing test
    expected_duration_turns: int              # how many turns this should take
    tags:            list[str] = field(default_factory=list)  # e.g. ["happy_path", "edge_case"]


SCENARIO_SYSTEM_PROMPT = """You are an expert at designing contact center test scenarios.
You create realistic, specific test cases that expose real weaknesses in automated flows.

Your scenarios must:
- Reflect how real customers actually communicate — not idealized interactions
- Include natural variation in how intents are expressed
- Test both happy paths and realistic edge cases
- Be specific enough to evaluate objectively"""


def generate_scenario(
    backend:     BaseBackend,
    persona:     Persona,
    flow_type:   str,
    flow_context: str,
    tags:        list[str] = None
) -> Scenario:
    """
    Generate a complete test scenario for a given persona and flow.

    persona:      the Persona who will execute this scenario
    flow_type:    e.g. "hotel_reservation", "flight_booking"
    flow_context: description of what the flow can handle
                  e.g. "handles room booking, check-in times, pet policy, cancellations"
    tags:         optional scenario tags e.g. ["happy_path", "edge_case", "angry_caller"]
    """
    tags_instruction = ""
    if tags:
        tags_instruction = f"\nThis scenario should focus on: {', '.join(tags)}"

    prompt = f"""Generate a test scenario for a {flow_type} contact flow.{tags_instruction}

Flow capabilities: {flow_context}

Caller persona:
- Name: {persona.name}
- Age: {persona.age}
- Emotional state: {persona.emotional_state}
- Patience level: {persona.patience_level}
- Communication style: {persona.communication_style}
- Goal: {persona.goal}
- Background: {persona.background}
- Speech patterns: {', '.join(persona.speech_patterns)}
- Edge case traits: {', '.join(persona.edge_case_traits)}

Generate a realistic scenario this caller would experience.

Respond with ONLY a JSON object in this exact format:
{{
  "name":        "short scenario name",
  "description": "one sentence description",
  "steps": [
    {{
      "intent":           "what the caller wants to accomplish in this step",
      "expected_outcome": "what a successful flow response looks like",
      "fallback_phrase":  "what the caller says if not understood first time",
      "is_edge_case":     false
    }}
  ],
  "success_criteria": [
    "criterion 1",
    "criterion 2"
  ],
  "expected_duration_turns": <integer>,
  "tags": ["tag1", "tag2"]
}}

Include 3-7 steps. At least one step should reflect the persona's edge case traits.
No markdown, no explanation outside the JSON."""

    response = backend.generate(
        system_prompt = SCENARIO_SYSTEM_PROMPT,
        messages      = [Message(role="user", content=prompt)],
        max_tokens    = 1000,
        temperature   = 0.8
    )

    raw = response.text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    data = json.loads(raw)

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
        name                     = data["name"],
        description              = data["description"],
        flow_type                = flow_type,
        persona_name             = persona.name,
        steps                    = steps,
        success_criteria         = data["success_criteria"],
        expected_duration_turns  = data["expected_duration_turns"],
        tags                     = data.get("tags", tags or [])
    )


def generate_scenario_batch(
    backend:      BaseBackend,
    personas:     list[Persona],
    flow_type:    str,
    flow_context: str,
    tag_matrix:   list[list[str]] = None
) -> list[Scenario]:
    """
    Generate one scenario per persona for a full test suite run.

    tag_matrix: optional list of tag sets to distribute across scenarios
    e.g. [["happy_path"], ["edge_case"], ["angry_caller", "repeat_caller"]]
    Ensures the suite covers a range of scenario types.
    """
    scenarios = []

    for i, persona in enumerate(personas):
        tags = None
        if tag_matrix and i < len(tag_matrix):
            tags = tag_matrix[i]

        scenario = generate_scenario(
            backend      = backend,
            persona      = persona,
            flow_type    = flow_type,
            flow_context = flow_context,
            tags         = tags
        )
        scenarios.append(scenario)
        print(f"[ScenarioGenerator] Generated scenario {i + 1}/{len(personas)}: {scenario.name}")

    return scenarios



"""
The relationship between persona and scenario is one-to-one
— each persona gets their own scenario tailored to their traits.
A frustrated low-patience caller gets a scenario that tests how the flow handles interruptions and repeated requests.
A confused elderly caller gets a scenario that tests how the flow handles unclear input and requests for clarification.
Same flow, completely different test coverage.
"""
