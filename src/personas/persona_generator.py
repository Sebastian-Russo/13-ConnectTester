"""
Think of this like a casting director for a contact center.
Given a test scenario type, it generates realistic caller personas —
complete characters with names, personalities, emotional states,
and communication styles that the caller agent will embody
during the test call.

The more realistic the persona, the more valuable the test.
A calm, patient caller will expose different flow weaknesses
than a frustrated, rushed one.
"""

import json
from dataclasses import dataclass
from src.backends.base_backend import BaseBackend, Message


@dataclass
class Persona:
    """
    A complete caller persona for a test run.
    Every field influences how the caller agent behaves
    during the conversation.
    """
    name:              str          # caller's name
    age:               int          # influences communication style
    emotional_state:   str          # calm | frustrated | confused | urgent
    patience_level:    str          # high | medium | low
    tech_savviness:    str          # high | medium | low
    communication_style: str        # formal | casual | terse | verbose
    goal:              str          # what they want to accomplish
    background:        str          # relevant context about their situation
    speech_patterns:   list[str]    # phrases or habits e.g. ["speaks quickly", "interrupts"]
    edge_case_traits:  list[str]    # e.g. ["will ask off-topic questions", "gives wrong info initially"]


PERSONA_SYSTEM_PROMPT = """You are an expert at creating realistic customer personas for
contact center testing. Your personas must be specific, believable, and varied enough
to expose real weaknesses in automated contact flows.

Focus on creating personas that represent real customer segments — not caricatures.
Include edge case traits that real customers actually exhibit."""


def generate_persona(
    backend:       BaseBackend,
    scenario_type: str,
    focus_traits:  list[str] = None
) -> Persona:
    """
    Generate a single realistic caller persona for a given scenario type.

    scenario_type: e.g. "hotel_reservation", "flight_booking", "tech_support"
    focus_traits:  optional list of traits to emphasize e.g. ["frustrated", "elderly"]
    """
    focus_instructions = ""
    if focus_traits:
        focus_instructions = f"\nEmphasize these traits: {', '.join(focus_traits)}"

    prompt = f"""Generate a realistic caller persona for testing a {scenario_type} contact flow.{focus_instructions}

Respond with ONLY a JSON object in this exact format:
{{
  "name":               "full name",
  "age":                <integer>,
  "emotional_state":    "calm" | "frustrated" | "confused" | "urgent",
  "patience_level":     "high" | "medium" | "low",
  "tech_savviness":     "high" | "medium" | "low",
  "communication_style": "formal" | "casual" | "terse" | "verbose",
  "goal":               "specific thing they want to accomplish",
  "background":         "2-3 sentences of relevant context about their situation",
  "speech_patterns":    ["pattern 1", "pattern 2"],
  "edge_case_traits":   ["trait 1", "trait 2"]
}}

No markdown, no explanation outside the JSON."""

    response = backend.generate(
        system_prompt = PERSONA_SYSTEM_PROMPT,
        messages      = [Message(role="user", content=prompt)],
        max_tokens    = 600,
        temperature   = 0.9    # high temperature = more varied personas
    )

    raw = response.text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    data = json.loads(raw)

    return Persona(
        name               = data["name"],
        age                = data["age"],
        emotional_state    = data["emotional_state"],
        patience_level     = data["patience_level"],
        tech_savviness     = data["tech_savviness"],
        communication_style = data["communication_style"],
        goal               = data["goal"],
        background         = data["background"],
        speech_patterns    = data.get("speech_patterns", []),
        edge_case_traits   = data.get("edge_case_traits", [])
    )


def generate_persona_batch(
    backend:       BaseBackend,
    scenario_type: str,
    count:         int,
    trait_matrix:  list[list[str]] = None
) -> list[Persona]:
    """
    Generate multiple personas for a full test suite run.

    trait_matrix: optional list of trait sets to distribute across personas
    e.g. [["frustrated", "elderly"], ["calm", "tech-savvy"], ["urgent", "confused"]]
    Ensures the suite covers a range of customer types rather than generating
    similar personas repeatedly.
    """
    personas = []

    for i in range(count):
        focus = None
        if trait_matrix and i < len(trait_matrix):
            focus = trait_matrix[i]

        persona = generate_persona(
            backend       = backend,
            scenario_type = scenario_type,
            focus_traits  = focus
        )
        personas.append(persona)
        print(f"[PersonaGenerator] Generated persona {i + 1}/{count}: {persona.name} — {persona.emotional_state}")

    return personas

"""
The temperature=0.9 on persona generation is intentional —
higher than anything in the previous projects.
You want varied, unpredictable personas, not similar ones.
Low temperature here would generate the same calm middle-aged caller every time
and your test suite would miss entire categories of real customer behavior.
"""
