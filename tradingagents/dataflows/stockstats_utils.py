import time
import logging

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config

logger = logging.getLogger(__name__)


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. When yfinance is rate-limited it can also silently
    return None (causing TypeError downstream) — that case is treated as
    a rate-limit and retried, then re-raised as YFRateLimitError so the
    vendor fallback chain in interface.py can switch to Alpha Vantage.
    """
    last_type_error = None
    for attempt in range(max_retries + 1):
        try:
            result = func()
            if result is None:
                raise TypeError("yfinance returned None (possible rate limit or network issue)")
            return result
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise
        except TypeError as e:
            last_type_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance returned None/TypeError, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise YFRateLimitError() from e
        except Exception as e:
            msg = str(e).lower()
            if any(kw in msg for kw in ("timed out", "timeout", "curl", "connection", "network error", "failed to perform")):
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Yahoo Finance network error, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(delay)
                else:
                    raise YFRateLimitError() from e
            else:
                raise


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Uses a stable per-symbol cache file. On each call:
    - Cache hit (covers curr_date): returned as-is, no network request.
    - Cache exists but is stale: only the missing days are fetched and
      appended, avoiding a full 5-year re-download.
    - No cache: full 5-year download is performed.
    Rows after curr_date are always filtered out to prevent look-ahead bias.
    """
    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)

    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(config["data_cache_dir"], f"{symbol}-YFin-data.csv")

    cached = None

    if os.path.exists(data_file):
        try:
            cached = pd.read_csv(data_file, on_bad_lines="skip")
            if cached.empty or "Date" not in cached.columns:
                raise ValueError("Cache file is empty or malformed")
            cached["Date"] = pd.to_datetime(cached["Date"], errors="coerce")
            max_cached = cached["Date"].max()

            if pd.notna(max_cached) and max_cached >= curr_date_dt:
                logger.debug(
                    "OHLCV cache hit for %s (cached up to %s, requested %s)",
                    symbol, max_cached.date(), curr_date_dt.date(),
                )
                data = cached
            else:
                # Fetch only the missing days and append to cache
                fetch_start = (max_cached + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if pd.notna(max_cached) else start_str
                logger.info(
                    "OHLCV cache for %s only reaches %s; fetching %s → %s",
                    symbol,
                    max_cached.date() if pd.notna(max_cached) else "N/A",
                    fetch_start,
                    end_str,
                )
                raw = yf_retry(lambda: yf.download(
                    symbol,
                    start=fetch_start,
                    end=end_str,
                    multi_level_index=False,
                    progress=False,
                    auto_adjust=True,
                ))
                if raw.empty:
                    raise YFRateLimitError("yf.download returned empty data (likely rate limited)")
                new_rows = raw.reset_index()
                data = pd.concat([cached, new_rows], ignore_index=True)
                data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
                data = data.drop_duplicates(subset=["Date"]).sort_values("Date")
                data.to_csv(data_file, index=False)
        except Exception as e:
            logger.warning("Failed to read OHLCV cache for %s (%s); re-fetching.", symbol, e)
            cached = None

    if cached is None:
        logger.info("Downloading OHLCV data for %s (%s → %s)", symbol, start_str, end_str)
        raw = yf_retry(lambda: yf.download(
            symbol,
            start=start_str,
            end=end_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        ))
        if raw.empty:
            raise YFRateLimitError("yf.download returned empty data (likely rate limited)")
        data = raw.reset_index()
        data.to_csv(data_file, index=False)

    data = _clean_dataframe(data)

    # Filter to curr_date to prevent look-ahead bias in backtesting
    data = data[data["Date"] <= curr_date_dt]

    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop financial statement columns (fiscal period timestamps) after curr_date.

    yfinance financial statements use fiscal period end dates as columns.
    Columns after curr_date represent future data and are removed to
    prevent look-ahead bias.
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        data = load_ohlcv(symbol, curr_date)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
