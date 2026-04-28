# TradingAgents/graph/trading_graph.py

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf
from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client
from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_transactions,
    get_global_news,
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor

logger = logging.getLogger(__name__)


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        set_config(self.config)

        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        llm_kwargs = self._get_provider_kwargs()
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm  = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()

        # Persistent decision memory log (replaces per-agent BM25 memory)
        self.memory_log = TradingMemoryLog(self.config)

        self.tool_nodes = self._create_tool_nodes()

        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
        )
        self.propagator      = Propagator()
        self.reflector       = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        self.curr_state      = None
        self.ticker          = None
        self.log_states_dict: Dict[str, Any] = {}

        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()
        if provider == "google":
            level = self.config.get("google_thinking_level")
            if level:
                kwargs["thinking_level"] = level
        elif provider == "openai":
            effort = self.config.get("openai_reasoning_effort")
            if effort:
                kwargs["reasoning_effort"] = effort
        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort
        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        return {
            "market": ToolNode([get_stock_data, get_indicators]),
            "social": ToolNode([get_news]),
            "news":   ToolNode([get_news, get_global_news, get_insider_transactions]),
            "fundamentals": ToolNode([
                get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement,
            ]),
        }

    # ---------------------------------------------------------------------------
    # Auto Phase B: resolve pending entries at the start of each same-ticker run
    # ---------------------------------------------------------------------------

    def _fetch_returns(
        self, ticker: str, trade_date: str, holding_days: int = 5
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """Fetch raw and alpha (vs SPY) return for ticker over holding_days from trade_date."""
        try:
            start   = datetime.strptime(trade_date, "%Y-%m-%d")
            end_str = (start + timedelta(days=holding_days + 7)).strftime("%Y-%m-%d")

            stock = yf.Ticker(ticker).history(start=trade_date, end=end_str)
            spy   = yf.Ticker("SPY").history(start=trade_date, end=end_str)

            if len(stock) < 2 or len(spy) < 2:
                return None, None, None

            actual_days = min(holding_days, len(stock) - 1, len(spy) - 1)
            raw   = float((stock["Close"].iloc[actual_days] - stock["Close"].iloc[0]) / stock["Close"].iloc[0])
            alpha = raw - float((spy["Close"].iloc[actual_days] - spy["Close"].iloc[0]) / spy["Close"].iloc[0])
            return raw, alpha, actual_days

        except Exception as e:
            logger.warning("Could not fetch returns for %s on %s: %s", ticker, trade_date, e)
            return None, None, None

    def _resolve_pending_entries(self, ticker: str) -> None:
        """Auto-resolve pending memory log entries for this ticker when price data is available."""
        pending = [e for e in self.memory_log.get_pending_entries() if e["ticker"] == ticker]
        if not pending:
            return

        updates = []
        for entry in pending:
            raw, alpha, days = self._fetch_returns(ticker, entry["date"])
            if raw is None:
                continue
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha,
            )
            updates.append({
                "ticker":       ticker,
                "trade_date":   entry["date"],
                "raw_return":   raw,
                "alpha_return": alpha,
                "holding_days": days,
                "reflection":   reflection,
            })

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

    # ---------------------------------------------------------------------------
    # Main analysis pipeline
    # ---------------------------------------------------------------------------

    def propagate(self, company_name: str, trade_date: str):
        """Run the trading agents graph for a company on a specific date."""
        self.ticker = company_name

        # Auto-resolve pending memory entries for this ticker before the new run
        self._resolve_pending_entries(company_name)

        # Inject historical context into initial state
        past_context     = self.memory_log.get_past_context(company_name)
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, past_context=past_context
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if chunk.get("messages"):
                    chunk["messages"][-1].pretty_print()
                trace.append(chunk)
            final_state = trace[-1]
        else:
            final_state = self.graph.invoke(init_agent_state, **args)

        self.curr_state = final_state
        self._log_state(trade_date, final_state)

        # Store decision as pending in the memory log (Phase A)
        self.memory_log.store_decision(
            ticker=company_name,
            trade_date=str(trade_date),
            final_trade_decision=final_state["final_trade_decision"],
        )

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date: str, final_state: Dict[str, Any]) -> None:
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date":          final_state["trade_date"],
            "market_report":       final_state["market_report"],
            "sentiment_report":    final_state["sentiment_report"],
            "news_report":         final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history":     final_state["investment_debate_state"]["bull_history"],
                "bear_history":     final_state["investment_debate_state"]["bear_history"],
                "history":          final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"]["current_response"],
                "judge_decision":   final_state["investment_debate_state"]["judge_decision"],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history":   final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history":      final_state["risk_debate_state"]["neutral_history"],
                "history":              final_state["risk_debate_state"]["history"],
                "judge_decision":       final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan":      final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        directory = Path(self.config["results_dir"]) / self.ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def process_signal(self, full_signal: str) -> str:
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
