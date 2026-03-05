"""
Think of this like a job contract that both AI backends must sign.
It defines exactly what every backend must be able to do —
without caring how they do it internally.

Anthropic backend uses the Anthropic SDK.
Bedrock backend uses boto3.
The rest of the codebase only ever talks to this contract —
so swapping backends requires zero changes anywhere else.
"""

from abc import ABC, abstractmethod # this is how we define an abstract base class
from dataclasses import dataclass # this is how we define a dataclass


@dataclass
class Message:
    """
    A single turn in a conversation.
    role:    "user" or "assistant"
    content: the text of the message
    """
    role:    str
    content: str


@dataclass
class BackendResponse:
    """
    Standardized response from any backend.
    Both Anthropic and Bedrock return different structures —
    this flattens them into one consistent shape the rest
    of the codebase can rely on.
    """
    text:         str            # the generated text
    input_tokens:  int           # tokens consumed by the prompt
    output_tokens: int           # tokens consumed by the response
    model:        str            # which model actually ran
    backend:      str            # "anthropic" or "bedrock"


class BaseBackend(ABC):
    """
    Abstract base class for all AI backends.

    Any class that inherits from this MUST implement
    generate() — Python will raise a TypeError at
    instantiation time if it doesn't.
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        messages:      list[Message],
        max_tokens:    int = 1000,
        temperature:   float = 0.7
    ) -> BackendResponse:
        """
        Generate a response given a system prompt and conversation history.

        system_prompt: the role/persona instructions
        messages:      full conversation history so far
        max_tokens:    cap on response length
        temperature:   0.0 = deterministic, 1.0 = creative

        Returns a BackendResponse with text + token counts + metadata.
        """
        pass

    @abstractmethod
    def get_model_id(self) -> str:
        """Return the model identifier string for this backend."""
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return 'anthropic' or 'bedrock'."""
        pass

"""
Three things defined here —
1. Message (one conversation turn)
2. BackendResponse (standardized output)
3. BaseBackend (the contract)
Every other file in the project imports Message and BackendResponse from here rather than defining their own shapes. That's how the whole system stays consistent regardless of which backend is running.
"""
