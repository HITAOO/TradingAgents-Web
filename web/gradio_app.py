"""
Gradio frontend for TradingAgents.

Call create_demo() to get a gr.Blocks instance, then .launch() it.
"""

from __future__ import annotations

import datetime
import queue
import tempfile
import threading
import time
from typing import Generator

import gradio as gr
import yfinance as yf

from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS, get_model_options
from web.config_builder import PROVIDER_DISPLAY_NAMES
from web.models import AnalysisRequest, ReanalysisRequest
from web.report_scanner import ReportEntry, extract_key_sections, scan_reports
from web.stream_handler import (
    AnalysisStreamHandler,
    ReanalysisStreamHandler,
    build_full_report_markdown,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANALYST_CHOICES = [
    ("Market Analyst",       "market"),
    ("Social Media Analyst", "social"),
    ("News Analyst",         "news"),
    ("Fundamentals Analyst", "fundamentals"),
]

LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean", "Hindi",
    "Spanish", "Portuguese", "French", "German", "Arabic", "Russian",
]

# Map Gradio checkbox label → analyst key
_ANALYST_LABEL_TO_KEY = {label: key for label, key in ANALYST_CHOICES}

# ---------------------------------------------------------------------------
# Helpers: dynamic model selection
# ---------------------------------------------------------------------------

def _model_choices(provider: str, mode: str) -> list[tuple[str, str]]:
    p = provider.lower()
    if p not in MODEL_OPTIONS:
        return []
    return list(get_model_options(p, mode))


def update_models_for_provider(provider: str):
    """
    Called when provider dropdown changes.
    Returns updated quick/deep model dropdowns + visibility for the
    provider-specific option rows.
    """
    p = provider.lower()
    free_entry = p in ("azure", "openrouter")

    if free_entry:
        quick = gr.Dropdown(choices=[], value="", allow_custom_value=True,
                            label="Quick-Thinking Model")
        deep  = gr.Dropdown(choices=[], value="", allow_custom_value=True,
                            label="Deep-Thinking Model")
    elif p in MODEL_OPTIONS:
        qc = _model_choices(p, "quick")
        dc = _model_choices(p, "deep")
        quick = gr.Dropdown(choices=qc, value=qc[0][1] if qc else "",
                            allow_custom_value=False, label="Quick-Thinking Model")
        deep  = gr.Dropdown(choices=dc, value=dc[0][1] if dc else "",
                            allow_custom_value=False, label="Deep-Thinking Model")
    else:
        quick = gr.Dropdown(choices=[], value="", allow_custom_value=True,
                            label="Quick-Thinking Model")
        deep  = gr.Dropdown(choices=[], value="", allow_custom_value=True,
                            label="Deep-Thinking Model")

    return (
        quick,
        deep,
        gr.update(visible=(p == "openai")),
        gr.update(visible=(p == "google")),
        gr.update(visible=(p == "anthropic")),
    )


# ---------------------------------------------------------------------------
# Helpers: HTML builders
# ---------------------------------------------------------------------------

_STATUS_COLOR = {
    "pending":     "#9CA3AF",
    "in_progress": "#3B82F6",
    "completed":   "#10B981",
    "error":       "#EF4444",
}
_STATUS_ICON = {
    "pending":     "○",
    "in_progress": "⟳",
    "completed":   "✓",
    "error":       "✗",
}
_TEAMS = {
    "Analyst Team":       ["Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst"],
    "Research Team":      ["Bull Researcher", "Bear Researcher", "Research Manager"],
    "Trading Team":       ["Trader"],
    "Risk Management":    ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
    "Portfolio Mgmt":     ["Portfolio Manager"],
}


def _progress_html(agent_statuses: dict) -> str:
    if not agent_statuses:
        return (
            '<div style="padding:16px;text-align:center;color:#6B7280;font-size:13px">'
            'Configure settings and click <b>Start Analysis</b> to begin.</div>'
        )

    rows_html = ""
    for team, agents in _TEAMS.items():
        active = [a for a in agents if a in agent_statuses]
        if not active:
            continue
        for i, agent in enumerate(active):
            status = agent_statuses.get(agent, "pending")
            color = _STATUS_COLOR.get(status, "#9CA3AF")
            icon  = _STATUS_ICON.get(status, "○")
            team_td = (
                f'<td style="padding:5px 10px;color:#6B7280;font-size:11px;'
                f'white-space:nowrap">{team if i == 0 else ""}</td>'
            )
            rows_html += (
                f"<tr>{team_td}"
                f'<td style="padding:5px 10px;font-size:13px">{agent}</td>'
                f'<td style="padding:5px 10px;color:{color};font-weight:600;font-size:13px">'
                f'{icon} {status}</td></tr>'
            )

    return f"""
<div style="border:1px solid #E5E7EB;border-radius:8px;overflow:auto">
<table style="width:100%;border-collapse:collapse">
  <thead>
    <tr style="border-bottom:1px solid #E5E7EB;background:#F9FAFB">
      <th style="padding:6px 10px;text-align:left;font-size:11px;color:#9CA3AF;font-weight:500">TEAM</th>
      <th style="padding:6px 10px;text-align:left;font-size:11px;color:#9CA3AF;font-weight:500">AGENT</th>
      <th style="padding:6px 10px;text-align:left;font-size:11px;color:#9CA3AF;font-weight:500">STATUS</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</div>"""


def _decision_html(decision: str) -> str:
    if not decision:
        return (
            '<div style="padding:16px;text-align:center;color:#9CA3AF;font-size:13px">'
            'Final decision will appear here after analysis completes.</div>'
        )
    _colors = {
        "BUY":         "#10B981",
        "OVERWEIGHT":  "#34D399",
        "HOLD":        "#F59E0B",
        "UNDERWEIGHT": "#F87171",
        "SELL":        "#EF4444",
    }
    color = _colors.get(decision.upper(), "#6B7280")
    return f"""
<div style="text-align:center;padding:24px 16px;border:1px solid #E5E7EB;border-radius:8px">
  <div style="font-size:52px;font-weight:700;color:{color};letter-spacing:2px">
    {decision.upper()}
  </div>
  <div style="color:#9CA3AF;margin-top:8px;font-size:12px;letter-spacing:1px">
    FINAL TRADING DECISION
  </div>
</div>"""


def _stats_text(stats: dict, elapsed: str) -> str:
    if not stats:
        return f"⏱ {elapsed}"
    def _fmt(n: int) -> str:
        return f"{n/1000:.1f}k" if n >= 1000 else str(n)
    return (
        f"⏱ {elapsed}  |  "
        f"LLM calls: {stats.get('llm_calls', 0)}  |  "
        f"Tool calls: {stats.get('tool_calls', 0)}  |  "
        f"Tokens: {_fmt(stats.get('tokens_in', 0))}↑ {_fmt(stats.get('tokens_out', 0))}↓"
    )


# ---------------------------------------------------------------------------
# Streaming analysis generator
# ---------------------------------------------------------------------------

# Number of outputs yielded on every iteration — must match the outputs= list
# in run_btn.click().  Keep in sync if you add/remove output components.
#
#  0  progress_html
#  1  market_md
#  2  social_md
#  3  news_md
#  4  fundamentals_md
#  5  research_md
#  6  trading_md
#  7  risk_md
#  8  decision_html
#  9  stats_display
# 10  log_md
# 11  download_btn   (visibility)
# 12  report_state   (gr.State)

_N_OUTPUTS = 13


def stream_analysis(
    ticker: str,
    date: str,
    analysts: list[str],
    depth: int,
    provider: str,
    quick_model: str,
    deep_model: str,
    language: str,
    openai_effort: str,
    google_thinking: str,
    anthropic_effort: str,
) -> Generator:
    """Gradio generator function driving the streaming UI updates."""

    # --- Validation ---
    if not ticker.strip():
        gr.Warning("Please enter a ticker symbol.")
        return
    if not analysts:
        gr.Warning("Please select at least one analyst.")
        return
    if not quick_model or not deep_model:
        gr.Warning("Please select models for Quick and Deep thinking.")
        return

    analyst_keys = [_ANALYST_LABEL_TO_KEY.get(a, a.lower().split()[0]) for a in analysts]

    p = provider.lower()
    req = AnalysisRequest(
        ticker=ticker.strip().upper(),
        date=date.strip(),
        analysts=analyst_keys,
        research_depth=int(depth),
        llm_provider=p,
        quick_model=quick_model.strip(),
        deep_model=deep_model.strip(),
        output_language=language,
        openai_reasoning_effort=openai_effort  if p == "openai"    and openai_effort    else None,
        google_thinking_level=google_thinking   if p == "google"    and google_thinking  else None,
        anthropic_effort=anthropic_effort       if p == "anthropic" and anthropic_effort else None,
    )

    stop_event = threading.Event()
    handler = AnalysisStreamHandler(req, stop_event)

    # Accumulate state across events
    agent_statuses: dict[str, str] = {}
    sections:       dict[str, str] = {}
    log_entries:    list[dict]     = []
    stats:          dict           = {}
    decision:       str            = ""
    start_time = time.time()

    def _elapsed() -> str:
        e = time.time() - start_time
        return f"{int(e // 60):02d}:{int(e % 60):02d}"

    def _make_yield():
        log_lines = []
        for e in log_entries[-60:]:
            ts      = e.get("timestamp", "")
            etype   = e.get("type", "")
            content = e.get("content", "")[:200]
            log_lines.append(f"`{ts}` **[{etype}]** {content}")
        log_text = "\n\n".join(log_lines) if log_lines else "_No activity yet._"

        return (
            _progress_html(agent_statuses),              # 0
            sections.get("market_report", ""),            # 1
            sections.get("sentiment_report", ""),         # 2
            sections.get("news_report", ""),              # 3
            sections.get("fundamentals_report", ""),      # 4
            sections.get("investment_plan", ""),          # 5
            sections.get("trader_investment_plan", ""),   # 6
            sections.get("final_trade_decision", ""),     # 7
            _decision_html(decision),                     # 8
            _stats_text(stats, _elapsed()),               # 9
            log_text,                                     # 10
            gr.update(visible=bool(decision)),            # 11  download_btn
            sections,                                     # 12  report_state
        )

    # Run iter_events() in a background thread so the generator can yield
    # periodic timer updates while waiting for slow LLM responses.
    event_queue: queue.Queue = queue.Queue()

    def _run_handler():
        try:
            for ev in handler.iter_events():
                event_queue.put(ev)
        finally:
            event_queue.put(None)  # sentinel: analysis finished

    bg_thread = threading.Thread(target=_run_handler, daemon=True)
    bg_thread.start()

    try:
        while True:
            try:
                ev = event_queue.get(timeout=0.5)
            except queue.Empty:
                # No event yet — yield a heartbeat so the timer keeps ticking
                yield _make_yield()
                continue

            if ev is None:
                # Analysis finished
                break

            etype = ev["event"]
            data  = ev["data"]

            if etype == "agent_status":
                agent_statuses[data["agent"]] = data["status"]
            elif etype == "report_section":
                sections[data["section"]] = data["content"]
            elif etype == "message":
                log_entries.append(data)
            elif etype == "tool_call":
                log_entries.append({
                    "timestamp": data["timestamp"],
                    "type":      "Tool",
                    "content":   f"{data['tool']}: {data['args']}",
                })
            elif etype == "stats":
                stats = data
            elif etype == "done":
                decision = data.get("decision", "")
                if data.get("all_sections"):
                    sections.update({k: v for k, v in data["all_sections"].items() if v})
            elif etype == "error":
                gr.Warning(f"Analysis error: {data['message']}")
                break

            yield _make_yield()

    except GeneratorExit:
        # User clicked Cancel — signal the background thread to stop
        stop_event.set()
        return

    # Final yield to ensure last state is shown
    yield _make_yield()


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def prepare_download(sections: dict, ticker: str, date: str, save_dir: str):
    if not sections:
        return gr.update(visible=False), ""
    content = build_full_report_markdown(sections, ticker=ticker, date=date)
    suffix = f"_{ticker}_{date}" if ticker else ""
    filename = f"tradingagents_report{suffix}.md"

    # Save to user-specified directory if provided
    save_msg = ""
    if save_dir and save_dir.strip():
        import pathlib
        dest = pathlib.Path(save_dir.strip()).expanduser()
        dest.mkdir(parents=True, exist_ok=True)
        out_path = dest / filename
        out_path.write_text(content, encoding="utf-8")
        save_msg = f"✅ Report saved to: `{out_path}`"

    # Also write to temp file for browser download
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False,
        prefix=f"tradingagents_report{suffix}_",
        encoding="utf-8",
    ) as f:
        f.write(content)
        return gr.update(value=f.name, visible=True), save_msg


# ---------------------------------------------------------------------------
# Re-analysis helpers
# ---------------------------------------------------------------------------

# Stores ReportEntry list for the current ticker scan so we can look up
# the selected entry's directory without re-scanning.
_reanalysis_report_cache: list[ReportEntry] = []


def fetch_current_price(ticker: str) -> str:
    """Return the latest price for ticker via yfinance fast_info."""
    ticker = ticker.strip().upper()
    if not ticker:
        return ""
    try:
        price = yf.Ticker(ticker).fast_info.last_price
        return f"{price:.2f}" if price else ""
    except Exception:
        return ""


def scan_ticker_reports(library_dir: str, ticker: str):
    """Scan library for reports matching ticker; return dropdown update."""
    global _reanalysis_report_cache
    ticker = ticker.strip().upper()
    lib    = library_dir.strip() if library_dir.strip() else "reports"

    if not ticker:
        _reanalysis_report_cache = []
        return gr.update(choices=[], value=None, interactive=False), \
               gr.update(visible=False)

    entries = scan_reports(lib, ticker)
    _reanalysis_report_cache = entries

    if not entries:
        _reanalysis_report_cache = []
        return gr.update(choices=[], value=None, interactive=False,
                         placeholder="No reports found for this ticker"), \
               gr.update(visible=False)

    labels = [e.label for e in entries]
    return gr.update(choices=labels, value=labels[0], interactive=True), \
           gr.update(visible=True)


def stream_reanalysis(
    library_dir: str,
    ra_ticker: str,
    report_label: str,
    current_price: str,
    ra_depth: int,
    ra_provider: str,
    ra_quick_model: str,
    ra_deep_model: str,
    ra_language: str,
    ra_openai_effort: str,
    ra_google_thinking: str,
    ra_anthropic_effort: str,
) -> Generator:
    """Gradio generator for the Re-analysis tab."""
    ra_ticker = ra_ticker.strip().upper()

    if not ra_ticker:
        gr.Warning("Please enter a ticker symbol.")
        return
    if not report_label:
        gr.Warning("Please scan and select a report first.")
        return
    if not current_price.strip():
        gr.Warning("Please enter the current price.")
        return
    if not ra_quick_model or not ra_deep_model:
        gr.Warning("Please select models for Quick and Deep thinking.")
        return

    # Find matching ReportEntry
    entry = next((e for e in _reanalysis_report_cache if e.label == report_label), None)
    if entry is None:
        gr.Warning("Report not found. Please re-scan.")
        return

    previous_context = extract_key_sections(entry.directory)
    if not previous_context:
        gr.Warning("Could not extract key sections from the selected report.")
        return

    p = ra_provider.lower()
    req = ReanalysisRequest(
        ticker=ra_ticker,
        current_date=datetime.date.today().strftime("%Y-%m-%d"),
        current_price=current_price.strip(),
        previous_context=previous_context,
        research_depth=int(ra_depth),
        llm_provider=p,
        quick_model=ra_quick_model.strip(),
        deep_model=ra_deep_model.strip(),
        output_language=ra_language,
        openai_reasoning_effort=ra_openai_effort  if p == "openai"    and ra_openai_effort    else None,
        google_thinking_level=ra_google_thinking  if p == "google"    and ra_google_thinking  else None,
        anthropic_effort=ra_anthropic_effort      if p == "anthropic" and ra_anthropic_effort else None,
    )

    stop_event = threading.Event()
    handler = ReanalysisStreamHandler(req, stop_event)

    agent_statuses: dict[str, str] = {}
    sections:       dict[str, str] = {}
    log_entries:    list[dict]     = []
    stats:          dict           = {}
    decision:       str            = ""
    start_time = time.time()

    def _elapsed() -> str:
        e = time.time() - start_time
        return f"{int(e // 60):02d}:{int(e % 60):02d}"

    # Re-analysis has no social/fundamentals sections
    _RA_SECTION_KEYS = [
        "market_report", "news_report",
        "investment_plan", "trader_investment_plan", "final_trade_decision",
    ]

    def _make_yield():
        log_lines = []
        for e in log_entries[-60:]:
            ts      = e.get("timestamp", "")
            etype   = e.get("type", "")
            content = e.get("content", "")[:200]
            log_lines.append(f"`{ts}` **[{etype}]** {content}")
        log_text = "\n\n".join(log_lines) if log_lines else "_No activity yet._"

        return (
            _progress_html(agent_statuses),                    # 0
            sections.get("market_report", ""),                  # 1
            sections.get("news_report", ""),                    # 2
            sections.get("investment_plan", ""),                # 3
            sections.get("trader_investment_plan", ""),         # 4
            sections.get("final_trade_decision", ""),           # 5
            _decision_html(decision),                           # 6
            _stats_text(stats, _elapsed()),                     # 7
            log_text,                                           # 8
        )

    event_queue: queue.Queue = queue.Queue()

    def _run():
        try:
            for ev in handler.iter_events():
                event_queue.put(ev)
        finally:
            event_queue.put(None)

    threading.Thread(target=_run, daemon=True).start()

    try:
        while True:
            try:
                ev = event_queue.get(timeout=0.5)
            except queue.Empty:
                yield _make_yield()
                continue

            if ev is None:
                break

            etype = ev["event"]
            data  = ev["data"]

            if etype == "agent_status":
                agent_statuses[data["agent"]] = data["status"]
            elif etype == "report_section":
                sections[data["section"]] = data["content"]
            elif etype == "message":
                log_entries.append(data)
            elif etype == "tool_call":
                log_entries.append({
                    "timestamp": data["timestamp"],
                    "type":      "Tool",
                    "content":   f"{data['tool']}: {data['args']}",
                })
            elif etype == "stats":
                stats = data
            elif etype == "done":
                decision = data.get("decision", "")
                if data.get("all_sections"):
                    sections.update({k: v for k, v in data["all_sections"].items() if v})
            elif etype == "error":
                gr.Warning(f"Re-analysis error: {data['message']}")
                break

            yield _make_yield()

    except GeneratorExit:
        stop_event.set()
        return

    yield _make_yield()


# ---------------------------------------------------------------------------
# Landing page HTML & CSS
# ---------------------------------------------------------------------------

_LANDING_BG_HTML = """
<div style="
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f172a 100%);
    border-radius: 16px;
    padding: 72px 48px 56px;
    text-align: center;
    position: relative;
    overflow: hidden;
    min-height: 46vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-bottom: 8px;
">
    <!-- subtle grid -->
    <div style="
        position:absolute; inset:0; border-radius:16px;
        background-image:
            linear-gradient(rgba(148,163,184,.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(148,163,184,.05) 1px, transparent 1px);
        background-size: 52px 52px;
    "></div>
    <!-- decorative chart line -->
    <svg style="position:absolute;bottom:0;left:0;right:0;width:100%;height:90px;opacity:.07"
         viewBox="0 0 800 90" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
        <polyline
            points="0,70 80,55 160,62 240,38 320,48 400,22 480,34 560,14 640,26 720,8 800,18"
            fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linejoin="round"/>
        <polyline
            points="0,80 80,68 160,74 240,52 320,60 400,36 480,46 560,28 640,40 720,20 800,30"
            fill="none" stroke="#34d399" stroke-width="1.5" stroke-dasharray="6 4" stroke-linejoin="round"/>
    </svg>
    <!-- content -->
    <div style="position:relative; z-index:1;">
        <div style="font-size:54px; margin-bottom:14px; line-height:1;">📈</div>
        <h1 style="
            color:#f1f5f9; font-size:42px; font-weight:700; margin:0;
            letter-spacing:-1.5px; line-height:1.1;
        ">TradingAgents</h1>
        <p style="color:#64748b; font-size:15px; margin:12px 0 0; letter-spacing:.3px;">
            Multi-Agent LLM Financial Trading Analysis
        </p>
    </div>
</div>
"""

_APP_CSS = """
footer { display: none !important; }

/* System sans-serif everywhere */
*, *::before, *::after {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, "Noto Sans", "PingFang SC",
                 "Microsoft YaHei", sans-serif !important;
}
h1, h2, h3, h4, h5, h6 { font-weight: 600 !important; letter-spacing: 0 !important; }

/* Hide the main top-level tab navigation — we drive it programmatically */
#main-nav > .tab-nav { display: none !important; }

/* ── Landing cards ── */
#card-first > button, #card-re > button {
    min-height: 200px !important;
    border-radius: 20px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    white-space: pre-line !important;
    line-height: 1.7 !important;
    padding: 28px 24px !important;
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease !important;
    cursor: pointer !important;
}

/* First Analysis card — blue */
#card-first > button {
    background: linear-gradient(145deg, #1e3a5f 0%, #162d4a 100%) !important;
    border: 2px solid rgba(59,130,246,.35) !important;
    color: #93c5fd !important;
    box-shadow: 0 4px 24px rgba(59,130,246,.10) !important;
}
#card-first > button:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 14px 40px rgba(59,130,246,.28) !important;
    border-color: rgba(59,130,246,.75) !important;
}

/* Re-Analysis card — emerald */
#card-re > button {
    background: linear-gradient(145deg, #1a3d2e 0%, #132e22 100%) !important;
    border: 2px solid rgba(16,185,129,.35) !important;
    color: #6ee7b7 !important;
    box-shadow: 0 4px 24px rgba(16,185,129,.10) !important;
}
#card-re > button:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 14px 40px rgba(16,185,129,.28) !important;
    border-color: rgba(16,185,129,.75) !important;
}

/* Back button on sub-pages */
#back-btn-fa > button, #back-btn-ra > button {
    background: transparent !important;
    border: 1px solid #374151 !important;
    color: #9CA3AF !important;
    font-size: 13px !important;
    padding: 6px 14px !important;
    border-radius: 8px !important;
}
#back-btn-fa > button:hover, #back-btn-ra > button:hover {
    border-color: #6B7280 !important;
    color: #D1D5DB !important;
}

/* Inner report tab nav */
.tab-nav button { font-size: 13px !important; }
"""

# ---------------------------------------------------------------------------
# Gradio Blocks definition
# ---------------------------------------------------------------------------

def create_demo() -> gr.Blocks:
    default_provider = "openai"
    initial_quick = _model_choices(default_provider, "quick")
    initial_deep  = _model_choices(default_provider, "deep")
    today = datetime.date.today().strftime("%Y-%m-%d")

    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="TradingAgents",
        css=_APP_CSS,
    ) as demo:

        # ═══════════════════════════════════════════════════════════════════
        # Main navigation — tab bar hidden, driven by buttons
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tabs(selected=0, elem_id="main-nav") as main_nav:

            # ── Tab 0: Landing page ──────────────────────────────────────
            with gr.Tab("Home", id=0):
                gr.HTML(_LANDING_BG_HTML)

                with gr.Row(equal_height=True):
                    card_first = gr.Button(
                        "📊\n\nFirst Analysis\n\nFull multi-agent analysis\nfor any ticker",
                        elem_id="card-first",
                        scale=1,
                    )
                    card_re = gr.Button(
                        "🔄\n\nRe-Analysis\n\nUpdate assessment with\ncurrent price data",
                        elem_id="card-re",
                        scale=1,
                    )

            # ── Tab 1: First Analysis ────────────────────────────────────
            with gr.Tab("First Analysis", id=1):
                with gr.Row():
                    back_fa = gr.Button("← Back", elem_id="back-btn-fa", scale=0, min_width=80)
                    gr.Markdown("## First Analysis", elem_id="page-title-fa")

                with gr.Row(equal_height=False):

                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### Configuration")
                        ticker_input = gr.Textbox(
                            label="Ticker Symbol",
                            placeholder="e.g. NVDA  SPY  0700.HK  7203.T",
                            value="", max_lines=1,
                        )
                        date_input = gr.Textbox(
                            label="Analysis Date", value=today,
                            placeholder="YYYY-MM-DD", max_lines=1,
                        )
                        analysts_check = gr.CheckboxGroup(
                            choices=[label for label, _ in ANALYST_CHOICES],
                            value=[label for label, _ in ANALYST_CHOICES],
                            label="Analyst Team",
                        )
                        depth_radio = gr.Radio(
                            choices=[
                                ("Shallow — 1 debate round (fast)", 1),
                                ("Medium — 3 debate rounds",        3),
                                ("Deep — 5 debate rounds (thorough)", 5),
                            ],
                            value=1, label="Research Depth",
                        )
                        gr.Markdown("### LLM Settings")
                        provider_dropdown = gr.Dropdown(
                            choices=[(name, key) for name, key in PROVIDER_DISPLAY_NAMES],
                            value=default_provider, label="Provider",
                        )
                        quick_model_dropdown = gr.Dropdown(
                            choices=initial_quick,
                            value=initial_quick[0][1] if initial_quick else "",
                            label="Quick-Thinking Model", allow_custom_value=False,
                        )
                        deep_model_dropdown = gr.Dropdown(
                            choices=initial_deep,
                            value=initial_deep[0][1] if initial_deep else "",
                            label="Deep-Thinking Model", allow_custom_value=False,
                        )
                        with gr.Row(visible=False) as openai_effort_row:
                            openai_effort = gr.Dropdown(
                                choices=[("Medium (default)","medium"),("High","high"),("Low","low")],
                                value="medium", label="Reasoning Effort",
                            )
                        with gr.Row(visible=False) as google_thinking_row:
                            google_thinking = gr.Dropdown(
                                choices=[("Enable thinking","high"),("Minimal / disable","minimal")],
                                value="high", label="Gemini Thinking Mode",
                            )
                        with gr.Row(visible=False) as anthropic_effort_row:
                            anthropic_effort = gr.Dropdown(
                                choices=[("High (recommended)","high"),("Medium","medium"),("Low","low")],
                                value="high", label="Claude Effort Level",
                            )
                        language_dropdown = gr.Dropdown(
                            choices=LANGUAGES, value="English",
                            label="Output Language", allow_custom_value=True,
                        )
                        with gr.Row():
                            run_btn    = gr.Button("▶  Start Analysis", variant="primary", scale=3)
                            cancel_btn = gr.Button("✕ Cancel", variant="stop", scale=1)
                        stats_display = gr.Markdown(
                            "_Ready. Configure settings above and click Start._", label="Stats",
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Agent Progress")
                        progress_html = gr.HTML(value=_progress_html({}))
                        gr.Markdown("### Reports")
                        with gr.Tabs():
                            with gr.Tab("Market"):       market_md       = gr.Markdown()
                            with gr.Tab("Social"):       social_md       = gr.Markdown()
                            with gr.Tab("News"):         news_md         = gr.Markdown()
                            with gr.Tab("Fundamentals"): fundamentals_md = gr.Markdown()
                            with gr.Tab("Research"):     research_md     = gr.Markdown()
                            with gr.Tab("Trading"):      trading_md      = gr.Markdown()
                            with gr.Tab("Risk & Portfolio"): risk_md     = gr.Markdown()
                        gr.Markdown("### Final Decision")
                        decision_html = gr.HTML(value=_decision_html(""))
                        with gr.Row():
                            save_dir_input = gr.Textbox(
                                label="Save report to directory (optional)",
                                placeholder="e.g. ~/reports  or  /Users/you/Desktop",
                                max_lines=1, scale=3,
                            )
                            download_btn = gr.Button(
                                "⬇  Export Report", variant="secondary", visible=False, scale=1,
                            )
                        save_msg_md   = gr.Markdown(visible=False)
                        download_file = gr.File(visible=False, label="Report File")

                with gr.Accordion("Live Activity Log", open=False):
                    log_md = gr.Markdown("_Activity log will stream here during analysis._")

                report_state = gr.State({})

            # ── Tab 2: Re-Analysis ───────────────────────────────────────
            with gr.Tab("Re-Analysis", id=2):
                with gr.Row():
                    back_ra = gr.Button("← Back", elem_id="back-btn-ra", scale=0, min_width=80)
                    gr.Markdown("## Re-Analysis", elem_id="page-title-ra")

                with gr.Row(equal_height=False):

                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### Report Selection")
                        ra_library_input = gr.Textbox(
                            label="Report Library Directory", value="reports",
                            placeholder="e.g. reports  or  ~/my-reports", max_lines=1,
                        )
                        ra_ticker_input = gr.Textbox(
                            label="Ticker Symbol", placeholder="e.g. AAPL", max_lines=1,
                        )
                        ra_scan_btn = gr.Button("🔍  Scan Reports", variant="secondary")
                        ra_report_dropdown = gr.Dropdown(
                            label="Select Report", choices=[], value=None,
                            interactive=False, allow_custom_value=False,
                        )
                        gr.Markdown("### Current Price")
                        with gr.Row():
                            ra_price_input = gr.Textbox(
                                label="Current Price", placeholder="e.g. 272.50",
                                max_lines=1, scale=3,
                            )
                            ra_fetch_price_btn = gr.Button("📡 Fetch", variant="secondary", scale=1)
                        gr.Markdown("### Settings")
                        ra_depth_radio = gr.Radio(
                            choices=[
                                ("Shallow — 1 debate round (fast)", 1),
                                ("Medium — 3 debate rounds",        3),
                                ("Deep — 5 debate rounds (thorough)", 5),
                            ],
                            value=1, label="Research Depth",
                        )
                        gr.Markdown("### LLM Settings")
                        ra_provider = gr.Dropdown(
                            choices=[(name, key) for name, key in PROVIDER_DISPLAY_NAMES],
                            value=default_provider, label="Provider",
                        )
                        ra_quick_model = gr.Dropdown(
                            choices=initial_quick,
                            value=initial_quick[0][1] if initial_quick else "",
                            label="Quick-Thinking Model", allow_custom_value=False,
                        )
                        ra_deep_model = gr.Dropdown(
                            choices=initial_deep,
                            value=initial_deep[0][1] if initial_deep else "",
                            label="Deep-Thinking Model", allow_custom_value=False,
                        )
                        with gr.Row(visible=False) as ra_openai_effort_row:
                            ra_openai_effort = gr.Dropdown(
                                choices=[("Medium (default)","medium"),("High","high"),("Low","low")],
                                value="medium", label="Reasoning Effort",
                            )
                        with gr.Row(visible=False) as ra_google_thinking_row:
                            ra_google_thinking = gr.Dropdown(
                                choices=[("Enable thinking","high"),("Minimal","minimal")],
                                value="high", label="Gemini Thinking Mode",
                            )
                        with gr.Row(visible=False) as ra_anthropic_effort_row:
                            ra_anthropic_effort = gr.Dropdown(
                                choices=[("High","high"),("Medium","medium"),("Low","low")],
                                value="high", label="Claude Effort Level",
                            )
                        ra_language = gr.Dropdown(
                            choices=LANGUAGES, value="English",
                            label="Output Language", allow_custom_value=True,
                        )
                        with gr.Row():
                            ra_run_btn    = gr.Button("▶  Re-Analyze", variant="primary", scale=3)
                            ra_cancel_btn = gr.Button("✕ Cancel", variant="stop", scale=1)
                        ra_stats = gr.Markdown(
                            "_Configure above and click Re-Analyze._", label="Stats",
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Agent Progress")
                        ra_progress_html = gr.HTML(value=_progress_html({}))
                        gr.Markdown("### Re-Analysis Reports")
                        with gr.Tabs():
                            with gr.Tab("Technical"): ra_market_md   = gr.Markdown()
                            with gr.Tab("News"):      ra_news_md     = gr.Markdown()
                            with gr.Tab("Research"):  ra_research_md = gr.Markdown()
                            with gr.Tab("Trading"):   ra_trading_md  = gr.Markdown()
                            with gr.Tab("Risk & Decision"): ra_risk_md = gr.Markdown()
                        gr.Markdown("### Re-Analysis Decision")
                        ra_decision_html = gr.HTML(value=_decision_html(""))

                with gr.Accordion("Live Activity Log", open=False):
                    ra_log_md = gr.Markdown("_Activity log will stream here._")

        # ═══════════════════════════════════════════════════════════════════
        # Event wiring
        # ═══════════════════════════════════════════════════════════════════

        # ── Landing page navigation ──────────────────────────────────────
        card_first.click(fn=lambda: gr.update(selected=1), outputs=[main_nav])
        card_re.click(   fn=lambda: gr.update(selected=2), outputs=[main_nav])
        back_fa.click(   fn=lambda: gr.update(selected=0), outputs=[main_nav])
        back_ra.click(   fn=lambda: gr.update(selected=0), outputs=[main_nav])

        # ── First Analysis ───────────────────────────────────────────────
        ticker_input.change(
            fn=lambda x: x.upper(), inputs=[ticker_input], outputs=[ticker_input],
            show_progress=False,
        )
        provider_dropdown.change(
            fn=update_models_for_provider, inputs=[provider_dropdown],
            outputs=[quick_model_dropdown, deep_model_dropdown,
                     openai_effort_row, google_thinking_row, anthropic_effort_row],
            show_progress=False,
        )
        run_event = run_btn.click(
            fn=stream_analysis,
            inputs=[ticker_input, date_input, analysts_check, depth_radio,
                    provider_dropdown, quick_model_dropdown, deep_model_dropdown,
                    language_dropdown, openai_effort, google_thinking, anthropic_effort],
            outputs=[progress_html, market_md, social_md, news_md, fundamentals_md,
                     research_md, trading_md, risk_md, decision_html,
                     stats_display, log_md, download_btn, report_state],
        )
        cancel_btn.click(fn=None, cancels=[run_event])
        download_btn.click(
            fn=prepare_download,
            inputs=[report_state, ticker_input, date_input, save_dir_input],
            outputs=[download_file, save_msg_md],
        ).then(
            fn=lambda msg: gr.update(value=msg, visible=bool(msg)),
            inputs=[save_msg_md], outputs=[save_msg_md],
        )

        # ── Re-Analysis ──────────────────────────────────────────────────
        ra_ticker_input.change(
            fn=lambda x: x.upper(), inputs=[ra_ticker_input], outputs=[ra_ticker_input],
            show_progress=False,
        )
        ra_provider.change(
            fn=update_models_for_provider, inputs=[ra_provider],
            outputs=[ra_quick_model, ra_deep_model,
                     ra_openai_effort_row, ra_google_thinking_row, ra_anthropic_effort_row],
            show_progress=False,
        )
        ra_scan_btn.click(
            fn=scan_ticker_reports,
            inputs=[ra_library_input, ra_ticker_input],
            outputs=[ra_report_dropdown, ra_run_btn],
            show_progress=True,
        )
        ra_fetch_price_btn.click(
            fn=fetch_current_price, inputs=[ra_ticker_input], outputs=[ra_price_input],
            show_progress=True,
        )
        ra_run_event = ra_run_btn.click(
            fn=stream_reanalysis,
            inputs=[ra_library_input, ra_ticker_input, ra_report_dropdown, ra_price_input,
                    ra_depth_radio, ra_provider, ra_quick_model, ra_deep_model,
                    ra_language, ra_openai_effort, ra_google_thinking, ra_anthropic_effort],
            outputs=[ra_progress_html, ra_market_md, ra_news_md, ra_research_md,
                     ra_trading_md, ra_risk_md, ra_decision_html, ra_stats, ra_log_md],
        )
        ra_cancel_btn.click(fn=None, cancels=[ra_run_event])

    return demo
