"""
Core streaming engine for the TradingAgents web frontend.

Runs TradingAgentsGraph in the current thread (safe inside Gradio worker threads)
and yields structured event dicts. UI layers (Gradio / FastAPI SSE) consume these
events and render them without caring about LangGraph internals.

Ported from cli/main.py chunk-processing logic (not imported, to avoid module-level
side effects like typer.Typer() and MessageBuffer() instantiation).
"""

from __future__ import annotations

import ast
import datetime
import threading
from collections import deque
from typing import Iterator

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.reanalysis_graph import ReanalysisGraph
from cli.stats_handler import StatsCallbackHandler
from web.models import AnalysisRequest, ReanalysisRequest
from web.config_builder import build_config

# ---------------------------------------------------------------------------
# Constants (mirrors cli/main.py)
# ---------------------------------------------------------------------------

ANALYST_ORDER = ["market", "social", "news", "fundamentals"]

ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}

ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}

FIXED_AGENTS = {
    "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
    "Trading Team": ["Trader"],
    "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
    "Portfolio Management": ["Portfolio Manager"],
}

REPORT_SECTIONS: dict[str, tuple[str | None, str]] = {
    "market_report":          ("market",        "Market Analyst"),
    "sentiment_report":       ("social",         "Social Analyst"),
    "news_report":            ("news",           "News Analyst"),
    "fundamentals_report":    ("fundamentals",   "Fundamentals Analyst"),
    "investment_plan":        (None,             "Research Manager"),
    "trader_investment_plan": (None,             "Trader"),
    "final_trade_decision":   (None,             "Portfolio Manager"),
}

# ---------------------------------------------------------------------------
# Minimal state tracker (no Rich, no disk I/O)
# ---------------------------------------------------------------------------

class _State:
    """Internal state tracker for one analysis run."""

    def __init__(self) -> None:
        self.selected_analysts: list[str] = []
        self.agent_status: dict[str, str] = {}
        self.report_sections: dict[str, str | None] = {}
        self._processed_ids: set = set()

    def init(self, analysts: list[str]) -> None:
        self.selected_analysts = [a.lower() for a in analysts]
        self.agent_status = {}
        for key in self.selected_analysts:
            if key in ANALYST_AGENT_NAMES:
                self.agent_status[ANALYST_AGENT_NAMES[key]] = "pending"
        for agents in FIXED_AGENTS.values():
            for agent in agents:
                self.agent_status[agent] = "pending"
        self.report_sections = {
            section: None
            for section, (analyst_key, _) in REPORT_SECTIONS.items()
            if analyst_key is None or analyst_key in self.selected_analysts
        }
        self._processed_ids.clear()

    def set_agent(self, agent: str, status: str) -> None:
        if agent in self.agent_status:
            self.agent_status[agent] = status

    def set_section(self, section: str, content: str) -> None:
        if section in self.report_sections:
            self.report_sections[section] = content


# ---------------------------------------------------------------------------
# Message helpers (ported from cli/main.py)
# ---------------------------------------------------------------------------

def _extract_content(content) -> str | None:
    def empty(v) -> bool:
        if v is None or v == "":
            return True
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False
        return not bool(v)

    if empty(content):
        return None
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text", "")
        return text.strip() if not empty(text) else None
    if isinstance(content, list):
        parts = [
            item.get("text", "").strip()
            if isinstance(item, dict) and item.get("type") == "text"
            else (item.strip() if isinstance(item, str) else "")
            for item in content
        ]
        result = " ".join(t for t in parts if t and not empty(t))
        return result if result else None
    return str(content).strip() if not empty(content) else None


def _classify_message(message) -> tuple[str, str | None]:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    content = _extract_content(getattr(message, "content", None))
    if isinstance(message, HumanMessage):
        return ("Control" if content and content.strip() == "Continue" else "User", content)
    if isinstance(message, ToolMessage):
        return ("Data", content)
    if isinstance(message, AIMessage):
        return ("Agent", content)
    return ("System", content)


def _fmt_args(args, max_len: int = 120) -> str:
    result = str(args)
    return result[:max_len - 3] + "..." if len(result) > max_len else result


# ---------------------------------------------------------------------------
# Analyst status logic (ported from cli/main.py update_analyst_statuses)
# ---------------------------------------------------------------------------

def _update_analyst_statuses(state: _State, chunk: dict) -> list[dict]:
    """Update analyst statuses from chunk. Returns list of change events."""
    events: list[dict] = []
    found_active = False

    for key in ANALYST_ORDER:
        if key not in state.selected_analysts:
            continue
        agent = ANALYST_AGENT_NAMES[key]
        report_key = ANALYST_REPORT_MAP[key]

        if chunk.get(report_key):
            state.set_section(report_key, chunk[report_key])

        has_report = bool(state.report_sections.get(report_key))
        old = state.agent_status.get(agent)

        if has_report:
            new_status = "completed"
        elif not found_active:
            new_status = "in_progress"
            found_active = True
        else:
            new_status = "pending"

        if old != new_status:
            state.set_agent(agent, new_status)
            events.append({"event": "agent_status", "data": {"agent": agent, "status": new_status}})

    if not found_active and state.selected_analysts:
        if state.agent_status.get("Bull Researcher") == "pending":
            state.set_agent("Bull Researcher", "in_progress")
            events.append({"event": "agent_status", "data": {"agent": "Bull Researcher", "status": "in_progress"}})

    return events


def _set_research_status(state: _State, status: str) -> list[dict]:
    events = []
    for agent in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
        if state.agent_status.get(agent) != status:
            state.set_agent(agent, status)
            events.append({"event": "agent_status", "data": {"agent": agent, "status": status}})
    return events


# ---------------------------------------------------------------------------
# Report markdown builder
# ---------------------------------------------------------------------------

def build_full_report_markdown(sections: dict, ticker: str = "", date: str = "") -> str:
    """Assemble a complete markdown report from all section contents."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"# Trading Analysis Report{': ' + ticker if ticker else ''}"
    parts = [f"{title}\n\nGenerated: {now}\n"]

    analyst_map = [
        ("market_report",       "Market Analysis"),
        ("sentiment_report",    "Social Sentiment"),
        ("news_report",         "News Analysis"),
        ("fundamentals_report", "Fundamentals Analysis"),
    ]
    analyst_parts = [(title, sections[k]) for k, title in analyst_map if sections.get(k)]
    if analyst_parts:
        parts.append("## I. Analyst Team Reports")
        for title, content in analyst_parts:
            parts.append(f"### {title}\n{content}")

    if sections.get("investment_plan"):
        parts.append(f"## II. Research Team Decision\n{sections['investment_plan']}")
    if sections.get("trader_investment_plan"):
        parts.append(f"## III. Trading Team Plan\n{sections['trader_investment_plan']}")
    if sections.get("final_trade_decision"):
        parts.append(f"## IV. Portfolio Management Decision\n{sections['final_trade_decision']}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main streaming handler
# ---------------------------------------------------------------------------

class AnalysisStreamHandler:
    """
    Runs TradingAgentsGraph synchronously and yields structured event dicts.

    Safe to call from any thread (including Gradio's worker thread).
    Pass a threading.Event as stop_event to enable cooperative cancellation.
    """

    def __init__(self, request: AnalysisRequest, stop_event: threading.Event | None = None) -> None:
        self.request = request
        self.config = build_config(request)
        self.stop_event = stop_event or threading.Event()

    def iter_events(self) -> Iterator[dict]:
        stats = StatsCallbackHandler()
        graph = TradingAgentsGraph(
            self.request.analysts,
            config=self.config,
            debug=False,
            callbacks=[stats],
        )

        state = _State()
        state.init(self.request.analysts)

        # Emit initial pending statuses so the UI can render the full table immediately
        for agent, status in state.agent_status.items():
            yield {"event": "agent_status", "data": {"agent": agent, "status": status}}

        # Inject historical memory context and auto-resolve any pending entries
        graph._resolve_pending_entries(self.request.ticker)
        past_context     = graph.memory_log.get_past_context(self.request.ticker)
        init_agent_state = graph.propagator.create_initial_state(
            self.request.ticker, self.request.date, past_context=past_context
        )
        args = graph.propagator.get_graph_args(callbacks=[stats])

        trace: list[dict] = []

        try:
            for chunk in graph.graph.stream(init_agent_state, **args):
                if self.stop_event.is_set():
                    break

                # --- Messages & tool calls ---
                for message in chunk.get("messages", []):
                    msg_id = getattr(message, "id", None)
                    if msg_id is not None:
                        if msg_id in state._processed_ids:
                            continue
                        state._processed_ids.add(msg_id)

                    msg_type, content = _classify_message(message)
                    if content and content.strip():
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        yield {"event": "message",
                               "data": {"type": msg_type, "content": content, "timestamp": ts}}

                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tc in message.tool_calls:
                            if isinstance(tc, dict):
                                name, tc_args = tc["name"], tc["args"]
                            else:
                                name, tc_args = tc.name, tc.args
                            ts = datetime.datetime.now().strftime("%H:%M:%S")
                            yield {"event": "tool_call",
                                   "data": {"tool": name, "args": _fmt_args(tc_args), "timestamp": ts}}

                # Snapshot old state for diffing
                old_statuses = dict(state.agent_status)
                old_sections = dict(state.report_sections)

                # --- Analyst statuses ---
                yield from _update_analyst_statuses(state, chunk)

                # --- Research Team ---
                if chunk.get("investment_debate_state"):
                    debate = chunk["investment_debate_state"]
                    bull  = debate.get("bull_history", "").strip()
                    bear  = debate.get("bear_history", "").strip()
                    judge = debate.get("judge_decision", "").strip()

                    if bull or bear:
                        yield from _set_research_status(state, "in_progress")
                    if bull:
                        state.set_section("investment_plan",
                                          f"### Bull Researcher Analysis\n{bull}")
                    if bear:
                        state.set_section("investment_plan",
                                          f"### Bear Researcher Analysis\n{bear}")
                    if judge:
                        state.set_section("investment_plan",
                                          f"### Research Manager Decision\n{judge}")
                        yield from _set_research_status(state, "completed")
                        if state.agent_status.get("Trader") == "pending":
                            state.set_agent("Trader", "in_progress")
                            yield {"event": "agent_status",
                                   "data": {"agent": "Trader", "status": "in_progress"}}

                # --- Trading Team ---
                if chunk.get("trader_investment_plan"):
                    state.set_section("trader_investment_plan",
                                      chunk["trader_investment_plan"])
                    if state.agent_status.get("Trader") != "completed":
                        state.set_agent("Trader", "completed")
                        yield {"event": "agent_status",
                               "data": {"agent": "Trader", "status": "completed"}}
                        state.set_agent("Aggressive Analyst", "in_progress")
                        yield {"event": "agent_status",
                               "data": {"agent": "Aggressive Analyst", "status": "in_progress"}}

                # --- Risk Management ---
                if chunk.get("risk_debate_state"):
                    risk  = chunk["risk_debate_state"]
                    agg   = risk.get("aggressive_history", "").strip()
                    con   = risk.get("conservative_history", "").strip()
                    neu   = risk.get("neutral_history", "").strip()
                    judge = risk.get("judge_decision", "").strip()

                    if agg:
                        if state.agent_status.get("Aggressive Analyst") != "completed":
                            state.set_agent("Aggressive Analyst", "in_progress")
                            yield {"event": "agent_status",
                                   "data": {"agent": "Aggressive Analyst", "status": "in_progress"}}
                        state.set_section("final_trade_decision",
                                          f"### Aggressive Analyst\n{agg}")
                    if con:
                        if state.agent_status.get("Conservative Analyst") != "completed":
                            state.set_agent("Conservative Analyst", "in_progress")
                            yield {"event": "agent_status",
                                   "data": {"agent": "Conservative Analyst", "status": "in_progress"}}
                        state.set_section("final_trade_decision",
                                          f"### Conservative Analyst\n{con}")
                    if neu:
                        if state.agent_status.get("Neutral Analyst") != "completed":
                            state.set_agent("Neutral Analyst", "in_progress")
                            yield {"event": "agent_status",
                                   "data": {"agent": "Neutral Analyst", "status": "in_progress"}}
                        state.set_section("final_trade_decision",
                                          f"### Neutral Analyst\n{neu}")
                    if judge:
                        if state.agent_status.get("Portfolio Manager") != "completed":
                            state.set_agent("Portfolio Manager", "in_progress")
                            state.set_section("final_trade_decision",
                                              f"### Portfolio Manager Decision\n{judge}")
                            for agent in ["Aggressive Analyst", "Conservative Analyst",
                                          "Neutral Analyst", "Portfolio Manager"]:
                                state.set_agent(agent, "completed")
                                yield {"event": "agent_status",
                                       "data": {"agent": agent, "status": "completed"}}

                # --- Emit section changes ---
                for section, content in state.report_sections.items():
                    if content and content != old_sections.get(section):
                        yield {"event": "report_section",
                               "data": {"section": section, "content": content}}

                # --- Stats snapshot ---
                yield {"event": "stats", "data": stats.get_stats()}

                trace.append(chunk)

        except Exception as exc:
            import traceback
            yield {"event": "error",
                   "data": {"message": str(exc), "detail": traceback.format_exc()}}
            return

        if not trace:
            return

        # --- Final cleanup ---
        final_state = trace[-1]
        decision = graph.process_signal(final_state.get("final_trade_decision", ""))

        # Store decision as pending in the memory log (Phase A)
        if final_state.get("final_trade_decision"):
            graph.memory_log.store_decision(
                ticker=self.request.ticker,
                trade_date=self.request.date,
                final_trade_decision=final_state["final_trade_decision"],
            )

        # Mark all agents completed
        for agent in list(state.agent_status):
            if state.agent_status[agent] != "completed":
                state.set_agent(agent, "completed")
                yield {"event": "agent_status",
                       "data": {"agent": agent, "status": "completed"}}

        # Sync final section values from the complete state
        for section in list(state.report_sections):
            if section in final_state and final_state[section]:
                state.set_section(section, final_state[section])
                yield {"event": "report_section",
                       "data": {"section": section, "content": final_state[section]}}

        yield {"event": "done",
               "data": {
                   "decision": decision,
                   "all_sections": {k: v for k, v in state.report_sections.items() if v},
               }}


# ---------------------------------------------------------------------------
# Re-analysis stream handler
# ---------------------------------------------------------------------------

# Re-analysis uses only market + news analysts
_REANALYSIS_ANALYSTS = ["market", "news"]

_REANALYSIS_REPORT_SECTIONS: dict[str, tuple[str | None, str]] = {
    "market_report":          ("market", "Market Analyst"),
    "news_report":            ("news",   "News Analyst"),
    "investment_plan":        (None,     "Research Manager"),
    "trader_investment_plan": (None,     "Trader"),
    "final_trade_decision":   (None,     "Portfolio Manager"),
}


class ReanalysisStreamHandler:
    """Runs ReanalysisGraph and yields the same event dicts as AnalysisStreamHandler."""

    def __init__(
        self,
        request: ReanalysisRequest,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.request = request
        self.config = build_config(request)
        self.stop_event = stop_event or threading.Event()

    def iter_events(self) -> Iterator[dict]:
        stats = StatsCallbackHandler()
        graph = ReanalysisGraph(config=self.config, callbacks=[stats])

        state = _State()
        state.init(_REANALYSIS_ANALYSTS)
        # Use the re-analysis section set (no sentiment/fundamentals)
        state.report_sections = {k: None for k in _REANALYSIS_REPORT_SECTIONS}

        for agent, status in state.agent_status.items():
            yield {"event": "agent_status", "data": {"agent": agent, "status": status}}

        trace: list[dict] = []

        try:
            for chunk in graph.stream(
                ticker=self.request.ticker,
                current_date=self.request.current_date,
                current_price=self.request.current_price,
                previous_context=self.request.previous_context,
            ):
                if self.stop_event.is_set():
                    break

                # Messages & tool calls
                for message in chunk.get("messages", []):
                    msg_id = getattr(message, "id", None)
                    if msg_id is not None:
                        if msg_id in state._processed_ids:
                            continue
                        state._processed_ids.add(msg_id)

                    msg_type, content = _classify_message(message)
                    if content and content.strip():
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        yield {"event": "message",
                               "data": {"type": msg_type, "content": content, "timestamp": ts}}

                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tc in message.tool_calls:
                            if isinstance(tc, dict):
                                name, tc_args = tc["name"], tc["args"]
                            else:
                                name, tc_args = tc.name, tc.args
                            ts = datetime.datetime.now().strftime("%H:%M:%S")
                            yield {"event": "tool_call",
                                   "data": {"tool": name, "args": _fmt_args(tc_args), "timestamp": ts}}

                old_sections = dict(state.report_sections)

                # Analyst statuses (market + news only)
                yield from _update_analyst_statuses(state, chunk)

                # Research Team
                if chunk.get("investment_debate_state"):
                    debate = chunk["investment_debate_state"]
                    bull  = debate.get("bull_history", "").strip()
                    bear  = debate.get("bear_history", "").strip()
                    judge = debate.get("judge_decision", "").strip()

                    if bull or bear:
                        yield from _set_research_status(state, "in_progress")
                    if bull:
                        state.set_section("investment_plan",
                                          f"### Bull Researcher Analysis\n{bull}")
                    if bear:
                        state.set_section("investment_plan",
                                          f"### Bear Researcher Analysis\n{bear}")
                    if judge:
                        state.set_section("investment_plan",
                                          f"### Research Manager Decision\n{judge}")
                        yield from _set_research_status(state, "completed")
                        if state.agent_status.get("Trader") == "pending":
                            state.set_agent("Trader", "in_progress")
                            yield {"event": "agent_status",
                                   "data": {"agent": "Trader", "status": "in_progress"}}

                # Trading Team
                if chunk.get("trader_investment_plan"):
                    state.set_section("trader_investment_plan",
                                      chunk["trader_investment_plan"])
                    if state.agent_status.get("Trader") != "completed":
                        state.set_agent("Trader", "completed")
                        yield {"event": "agent_status",
                               "data": {"agent": "Trader", "status": "completed"}}
                        state.set_agent("Aggressive Analyst", "in_progress")
                        yield {"event": "agent_status",
                               "data": {"agent": "Aggressive Analyst", "status": "in_progress"}}

                # Risk Management
                if chunk.get("risk_debate_state"):
                    risk  = chunk["risk_debate_state"]
                    agg   = risk.get("aggressive_history", "").strip()
                    con   = risk.get("conservative_history", "").strip()
                    neu   = risk.get("neutral_history", "").strip()
                    judge = risk.get("judge_decision", "").strip()

                    if agg:
                        state.set_agent("Aggressive Analyst", "in_progress")
                        yield {"event": "agent_status",
                               "data": {"agent": "Aggressive Analyst", "status": "in_progress"}}
                        state.set_section("final_trade_decision",
                                          f"### Aggressive Analyst\n{agg}")
                    if con:
                        state.set_agent("Conservative Analyst", "in_progress")
                        yield {"event": "agent_status",
                               "data": {"agent": "Conservative Analyst", "status": "in_progress"}}
                        state.set_section("final_trade_decision",
                                          f"### Conservative Analyst\n{con}")
                    if neu:
                        state.set_agent("Neutral Analyst", "in_progress")
                        yield {"event": "agent_status",
                               "data": {"agent": "Neutral Analyst", "status": "in_progress"}}
                        state.set_section("final_trade_decision",
                                          f"### Neutral Analyst\n{neu}")
                    if judge:
                        state.set_section("final_trade_decision",
                                          f"### Portfolio Manager Decision\n{judge}")
                        for agent in ["Aggressive Analyst", "Conservative Analyst",
                                      "Neutral Analyst", "Portfolio Manager"]:
                            state.set_agent(agent, "completed")
                            yield {"event": "agent_status",
                                   "data": {"agent": agent, "status": "completed"}}

                # Emit section changes
                for section, content in state.report_sections.items():
                    if content and content != old_sections.get(section):
                        yield {"event": "report_section",
                               "data": {"section": section, "content": content}}

                yield {"event": "stats", "data": stats.get_stats()}
                trace.append(chunk)

        except Exception as exc:
            import traceback
            yield {"event": "error",
                   "data": {"message": str(exc), "detail": traceback.format_exc()}}
            return

        if not trace:
            return

        final_state = trace[-1]
        decision = graph.process_signal(final_state.get("final_trade_decision", ""))

        for agent in list(state.agent_status):
            if state.agent_status[agent] != "completed":
                state.set_agent(agent, "completed")
                yield {"event": "agent_status",
                       "data": {"agent": agent, "status": "completed"}}

        for section in list(state.report_sections):
            if section in final_state and final_state[section]:
                state.set_section(section, final_state[section])
                yield {"event": "report_section",
                       "data": {"section": section, "content": final_state[section]}}

        yield {"event": "done",
               "data": {
                   "decision": decision,
                   "all_sections": {k: v for k, v in state.report_sections.items() if v},
               }}
