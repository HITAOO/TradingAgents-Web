import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_DEEPSEEK_API_BASE = "https://api.deepseek.com"

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "callbacks",
)


class DeepSeekChatOpenAI(ChatOpenAI):
    """ChatOpenAI configured for DeepSeek with reasoning_content support.

    deepseek-reasoner (thinking mode) includes reasoning_content in responses.
    For multi-turn conversations with tool calls, this field must be echoed back
    in subsequent requests — langchain-openai does not do this automatically.
    This subclass captures reasoning_content from responses and re-injects it
    into outgoing assistant messages.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        # Capture reasoning_content from raw response into additional_kwargs
        choices = getattr(response, "choices", None)
        if choices:
            for gen, choice in zip(result.generations, choices):
                rc = getattr(getattr(choice, "message", None), "reasoning_content", None)
                if rc:
                    gen.message.additional_kwargs["reasoning_content"] = rc
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # Re-inject reasoning_content from original messages into serialized payload
        if isinstance(input_, list):
            for orig_msg, dict_msg in zip(input_, payload["messages"]):
                rc = getattr(orig_msg, "additional_kwargs", {}).get("reasoning_content")
                if rc and dict_msg.get("role") == "assistant":
                    dict_msg["reasoning_content"] = rc
        return payload


class DeepSeekClient(BaseLLMClient):
    """Client for DeepSeek models with thinking mode (reasoning_content) support."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        llm_kwargs = {
            "model": self.model,
            "base_url": self.base_url or _DEEPSEEK_API_BASE,
        }

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if api_key:
            llm_kwargs["api_key"] = api_key

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return DeepSeekChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("deepseek", self.model)

    def get_provider_name(self) -> str:
        return "deepseek"
