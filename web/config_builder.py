"""Translates web AnalysisRequest into a TradingAgents config dict."""

from tradingagents.default_config import DEFAULT_CONFIG
from web.models import AnalysisRequest

# Canonical API base URLs for each provider (mirrors cli/utils.py PROVIDERS)
PROVIDER_URLS: dict[str, str | None] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/",
    "google": None,
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "glm": "https://api.z.ai/api/paas/v4/",
    "minimax": "https://api.minimaxi.com/v1?GroupId=2035277407077798827",
    "openrouter": "https://openrouter.ai/api/v1",
    "azure": None,
    "ollama": "http://localhost:11434/v1",
}

# Human-readable provider names for the UI
PROVIDER_DISPLAY_NAMES: list[tuple[str, str]] = [
    ("OpenAI", "openai"),
    ("Google", "google"),
    ("Anthropic", "anthropic"),
    ("xAI", "xai"),
    ("DeepSeek", "deepseek"),
    ("Qwen", "qwen"),
    ("GLM", "glm"),
    ("MiniMax", "minimax"),
    ("OpenRouter", "openrouter"),
    ("Azure OpenAI", "azure"),
    ("Ollama", "ollama"),
]


def build_config(req: AnalysisRequest) -> dict:
    """Build TradingAgentsGraph config dict from an AnalysisRequest."""
    config = DEFAULT_CONFIG.copy()
    provider = req.llm_provider.lower()

    config["llm_provider"] = provider
    config["quick_think_llm"] = req.quick_model
    config["deep_think_llm"] = req.deep_model
    config["max_debate_rounds"] = req.research_depth
    config["max_risk_discuss_rounds"] = req.research_depth
    config["output_language"] = req.output_language
    config["backend_url"] = req.backend_url or PROVIDER_URLS.get(provider)

    if req.google_thinking_level:
        config["google_thinking_level"] = req.google_thinking_level
    if req.openai_reasoning_effort:
        config["openai_reasoning_effort"] = req.openai_reasoning_effort
    if req.anthropic_effort:
        config["anthropic_effort"] = req.anthropic_effort

    return config
