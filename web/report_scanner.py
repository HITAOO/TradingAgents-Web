"""Report library scanner.

Scans a directory for TradingAgents report folders matching the naming pattern
{TICKER}_{YYYYMMDD}_{HHMMSS} and extracts the key sections needed for re-analysis.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import NamedTuple


class ReportEntry(NamedTuple):
    ticker: str
    timestamp: datetime
    directory: Path
    label: str          # display label for dropdown


def scan_reports(library_dir: str, ticker: str) -> list[ReportEntry]:
    """Return all reports for *ticker* in *library_dir*, newest first.

    Matches subdirectory names of the form  AAPL_20260417_103000.
    Non-matching entries and empty/broken directories are silently skipped.
    """
    library = Path(library_dir).expanduser().resolve()
    if not library.exists():
        return []

    pattern = re.compile(r"^([A-Z0-9.\-]+)_(\d{8})_(\d{6})$", re.IGNORECASE)
    entries: list[ReportEntry] = []

    for item in library.iterdir():
        if not item.is_dir():
            continue
        m = pattern.match(item.name)
        if not m:
            continue
        if m.group(1).upper() != ticker.upper():
            continue
        try:
            ts = datetime.strptime(f"{m.group(2)}_{m.group(3)}", "%Y%m%d_%H%M%S")
        except ValueError:
            continue

        label = f"{ticker.upper()}  ·  {ts.strftime('%Y-%m-%d  %H:%M:%S')}"
        entries.append(ReportEntry(
            ticker=m.group(1).upper(),
            timestamp=ts,
            directory=item,
            label=label,
        ))

    return sorted(entries, key=lambda e: e.timestamp, reverse=True)


def extract_key_sections(report_dir: Path) -> str:
    """Extract technical analysis + trading recommendation + risk summary.

    Returns a formatted markdown string ready to be injected into the
    re-analysis agent context.  Returns an empty string if the directory
    contains no recognisable sections.
    """
    parts: list[str] = []

    # 1. Technical analysis (Market Analyst)
    market_file = report_dir / "1_analysts" / "market.md"
    if market_file.exists():
        content = market_file.read_text(encoding="utf-8").strip()
        if content:
            parts.append("## Previous Technical Analysis\n\n" + content)

    # 2. Trading recommendation (Trader)
    trader_file = report_dir / "3_trading" / "trader.md"
    if trader_file.exists():
        content = trader_file.read_text(encoding="utf-8").strip()
        if content:
            parts.append("## Previous Trading Recommendation\n\n" + content)

    # 3. Risk summary — neutral first, then conservative/aggressive as supplements
    risk_dir = report_dir / "4_risk"
    risk_parts: list[str] = []
    for fname, label in [("neutral.md", "Neutral"), ("conservative.md", "Conservative"), ("aggressive.md", "Aggressive")]:
        rf = risk_dir / fname
        if rf.exists():
            c = rf.read_text(encoding="utf-8").strip()
            if c:
                risk_parts.append(f"### {label} Risk Assessment\n\n{c}")
    if risk_parts:
        parts.append("## Previous Risk Assessment\n\n" + "\n\n---\n\n".join(risk_parts))

    return "\n\n---\n\n".join(parts)
