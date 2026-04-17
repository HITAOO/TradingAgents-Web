"""Pydantic models for the TradingAgents web API."""

from typing import List, Optional
from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    ticker: str
    date: str
    analysts: List[str]  # e.g. ["market", "social", "news", "fundamentals"]
    research_depth: int = 1  # maps to max_debate_rounds: 1, 3, or 5
    llm_provider: str = "openai"
    quick_model: str = "gpt-5.4-mini"
    deep_model: str = "gpt-5.4"
    backend_url: Optional[str] = None
    output_language: str = "English"
    google_thinking_level: Optional[str] = None   # "high" | "minimal"
    openai_reasoning_effort: Optional[str] = None  # "low" | "medium" | "high"
    anthropic_effort: Optional[str] = None         # "low" | "medium" | "high"
