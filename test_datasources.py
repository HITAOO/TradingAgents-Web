#!/usr/bin/env python3
"""Test all data source interfaces (yfinance + Alpha Vantage)."""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

SYMBOL = "AAPL"
START_DATE = "2025-01-01"
END_DATE = "2025-01-31"
CURR_DATE = "2025-01-31"

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

results = []

def test(name, fn):
    try:
        result = fn()
        if result and not str(result).startswith("Error") and "No data" not in str(result):
            snippet = str(result)[:80].replace("\n", " ")
            print(f"[{PASS}] {name}\n         {snippet}...")
            results.append((name, True, None))
        else:
            print(f"[{FAIL}] {name}\n         empty or error response: {str(result)[:120]}")
            results.append((name, False, str(result)[:120]))
    except Exception as e:
        print(f"[{FAIL}] {name}\n         {type(e).__name__}: {e}")
        results.append((name, False, f"{type(e).__name__}: {e}"))

print("=" * 60)
print("  TradingAgents Data Source Test")
print(f"  Symbol: {SYMBOL}  |  {START_DATE} ~ {END_DATE}")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# yfinance
# ──────────────────────────────────────────────────────────────
print("\n── yfinance ──────────────────────────────────────────────")

from tradingagents.dataflows.y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals as yf_fundamentals,
    get_balance_sheet as yf_balance_sheet,
    get_cashflow as yf_cashflow,
    get_income_statement as yf_income_stmt,
    get_insider_transactions as yf_insider,
)
from tradingagents.dataflows.yfinance_news import get_news_yfinance, get_global_news_yfinance

test("yfinance | stock (OHLCV)",
     lambda: get_YFin_data_online(SYMBOL, START_DATE, END_DATE))

test("yfinance | technical indicator (rsi)",
     lambda: get_stock_stats_indicators_window(SYMBOL, "rsi", CURR_DATE, 14))

test("yfinance | fundamentals",
     lambda: yf_fundamentals(SYMBOL, CURR_DATE))

test("yfinance | balance sheet",
     lambda: yf_balance_sheet(SYMBOL, "quarterly", CURR_DATE))

test("yfinance | cash flow",
     lambda: yf_cashflow(SYMBOL, "quarterly", CURR_DATE))

test("yfinance | income statement",
     lambda: yf_income_stmt(SYMBOL, "quarterly", CURR_DATE))

test("yfinance | insider transactions",
     lambda: yf_insider(SYMBOL))

test("yfinance | stock news",
     lambda: get_news_yfinance(SYMBOL, START_DATE, END_DATE))

test("yfinance | global news",
     lambda: get_global_news_yfinance(CURR_DATE, look_back_days=7))

# ──────────────────────────────────────────────────────────────
# Alpha Vantage
# ──────────────────────────────────────────────────────────────
print("\n── Alpha Vantage ─────────────────────────────────────────")

from tradingagents.dataflows.alpha_vantage_stock import get_stock as av_stock
from tradingagents.dataflows.alpha_vantage_indicator import get_indicator as av_indicator
from tradingagents.dataflows.alpha_vantage_fundamentals import (
    get_fundamentals as av_fundamentals,
    get_balance_sheet as av_balance_sheet,
    get_cashflow as av_cashflow,
    get_income_statement as av_income_stmt,
)
from tradingagents.dataflows.alpha_vantage_news import (
    get_news as av_news,
    get_global_news as av_global_news,
    get_insider_transactions as av_insider,
)

test("alpha_vantage | stock (OHLCV)",
     lambda: av_stock(SYMBOL, START_DATE, END_DATE))

test("alpha_vantage | technical indicator (rsi)",
     lambda: av_indicator(SYMBOL, "rsi", CURR_DATE, look_back_days=14))

test("alpha_vantage | fundamentals",
     lambda: av_fundamentals(SYMBOL, CURR_DATE))

test("alpha_vantage | balance sheet",
     lambda: av_balance_sheet(SYMBOL, "quarterly", CURR_DATE))

test("alpha_vantage | cash flow",
     lambda: av_cashflow(SYMBOL, "quarterly", CURR_DATE))

test("alpha_vantage | income statement",
     lambda: av_income_stmt(SYMBOL, "quarterly", CURR_DATE))

test("alpha_vantage | stock news",
     lambda: av_news(SYMBOL, START_DATE, END_DATE))

test("alpha_vantage | global news",
     lambda: av_global_news(CURR_DATE, look_back_days=7))

test("alpha_vantage | insider transactions",
     lambda: av_insider(SYMBOL))

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"  Result: {passed}/{total} passed")

if passed < total:
    print("\n  Failed:")
    for name, ok, err in results:
        if not ok:
            print(f"    ✗ {name}")
            print(f"      {err}")

print("=" * 60)
sys.exit(0 if passed == total else 1)
