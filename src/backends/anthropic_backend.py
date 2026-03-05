"""
Anthropic backend implementation.
Implements the BaseBackend contract using the Anthropic Python SDK.

base_backend.py defines the contract that all backends must follow.
This file implements that contract for Anthropic's API.

BaseBackend > AnthropicBackend

This is the familiar pattern from all previous projects —
the only difference is it now conforms to a standard interface
so the rest of the codebase can treat it identically to Bedrock.
"""

import anthropic
from src.backends.base_backend import BaseBackend, BackendResponse, Message
from src.infrastructure.config import ANTHROPIC_API_KEY


# Model IDs — Haiku for caller agent (fast, cheap, many parallel calls)
# Sonnet for evaluator and reporter (stronger reasoning for synthesis)
HAIKU_MODEL  = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"


class AnthropicBackend(BaseBackend):
    """
    Claude via the Anthropic API.
    Used when running locally or when AWS credentials aren't available.
    """

    def __init__(self, model_id: str = HAIKU_MODEL):
        self.client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model_id = model_id

    def generate(
        self,
        system_prompt: str,
        messages:      list[Message],
        max_tokens:    int   = 1000,
        temperature:   float = 0.7
    ) -> BackendResponse:
        """
        Call the Anthropic API and return a standardized BackendResponse.

        Converts our internal Message dataclass list into the dict format
        Anthropic expects, then converts the response back to our standard shape.
        """
        # Convert Message dataclasses to Anthropic's expected dict format
        formatted_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        response = self.client.messages.create(
            model      = self.model_id,
            max_tokens = max_tokens,
            system     = system_prompt,
            messages   = formatted_messages
        )

        return BackendResponse(
            text          = response.content[0].text,
            input_tokens  = response.usage.input_tokens,
            output_tokens = response.usage.output_tokens,
            model         = self.model_id,
            backend       = self.get_backend_name()
        )

    def get_model_id(self) -> str:
        return self.model_id

    def get_backend_name(self) -> str:
        return "anthropic"
