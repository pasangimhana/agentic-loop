from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    result: str


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    raw_response: object = None


class BaseLLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Send messages to the LLM and return a parsed response."""
        ...

    @abstractmethod
    def build_assistant_message(self, response: LLMResponse) -> dict:
        """Convert an LLMResponse into a provider-native assistant message for conversation history."""
        ...

    @abstractmethod
    def build_tool_results_message(self, results: list[ToolResult]) -> dict | list[dict]:
        """Convert tool results into provider-native message(s) for conversation history.
        Returns a single dict or list of dicts to append to messages."""
        ...


def get_provider(name: str) -> BaseLLMProvider:
    if name == "anthropic":
        from providers.anthropic import AnthropicProvider
        return AnthropicProvider()
    elif name == "openai":
        from providers.openai import OpenAIProvider
        return OpenAIProvider()
    else:
        raise ValueError(f"Unknown provider: {name}")
