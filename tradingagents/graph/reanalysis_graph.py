"""Lightweight re-analysis graph.

Runs only Market Analyst + News Analyst, then the full research/trading/risk
pipeline.  The previous analysis context is injected as a human message so
every agent sees it alongside the fresh data it fetches.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


class ReanalysisGraph:
    """Re-analysis graph — focused subset of agents with previous context."""

    # Only technical + news analysts; skips fundamentals & social
    ANALYSTS = ["market", "news"]

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        callbacks: Optional[List] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG.copy()
        self.callbacks = callbacks or []
        self._inner = TradingAgentsGraph(
            selected_analysts=self.ANALYSTS,
            config=self.config,
            debug=False,
            callbacks=self.callbacks,
        )

    def stream(
        self,
        ticker: str,
        current_date: str,
        current_price: str,
        previous_context: str,
    ) -> Iterator[dict]:
        """Stream LangGraph chunks with previous context injected."""
        init_state = self._inner.propagator.create_initial_state(ticker, current_date)

        context_msg = (
            "REANALYSIS MODE\n\n"
            f"Current Market Price: {current_price}\n\n"
            "The following is the previous analysis report for this ticker. "
            "Your task is to assess whether the current price action and any new "
            "developments change the previous assessment. "
            "Pay special attention to whether key support/resistance levels identified "
            "previously have been breached and what that implies.\n\n"
            f"{previous_context}"
        )
        # Inject context as a second human message so all agents can reference it
        init_state["messages"] = [
            ("human", ticker),
            ("human", context_msg),
        ]

        args = self._inner.propagator.get_graph_args(callbacks=self.callbacks)
        yield from self._inner.graph.stream(init_state, **args)

    def process_signal(self, full_signal: str) -> str:
        return self._inner.process_signal(full_signal)
