import logging
import os
import pandas as pd
from io import StringIO
from datetime import datetime
from .alpha_vantage_common import _make_api_request, _filter_csv_by_date_range
from .config import get_config

logger = logging.getLogger(__name__)


def get_stock(
    symbol: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Returns raw daily OHLCV values filtered to the specified date range.

    Results are cached per symbol in the configured data_cache_dir.  On
    subsequent calls the cache is checked first: if the stored data already
    covers end_date the API is skipped entirely; otherwise a full-history
    download is performed and the cache is updated.

    Args:
        symbol: The name of the equity. For example: symbol=IBM
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        CSV string containing the daily time series data filtered to the date range.
    """
    config = get_config()
    cache_dir = config["data_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"{symbol}-AV-data.csv")
    end_dt = pd.Timestamp(end_date)

    # --- Cache check ---
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file)
            date_col = cached_df.columns[0]
            cached_df[date_col] = pd.to_datetime(cached_df[date_col], errors="coerce")
            max_cached = cached_df[date_col].max()
            if pd.notna(max_cached) and max_cached >= end_dt:
                logger.debug(
                    "AV OHLCV cache hit for %s (cached up to %s, requested %s)",
                    symbol, max_cached.date(), end_dt.date(),
                )
                return _filter_csv_by_date_range(cached_df.to_csv(index=False), start_date, end_date)
            else:
                logger.info(
                    "AV OHLCV cache for %s only reaches %s; refreshing to cover %s",
                    symbol,
                    max_cached.date() if pd.notna(max_cached) else "N/A",
                    end_dt.date(),
                )
        except Exception as e:
            logger.warning("Failed to read AV OHLCV cache for %s (%s); re-fetching.", symbol, e)

    # --- Fetch from API (always full history so the cache stays useful) ---
    logger.info("Downloading AV OHLCV data for %s (full history)", symbol)
    params = {
        "symbol": symbol,
        "outputsize": "full",
        "datatype": "csv",
    }
    response = _make_api_request("TIME_SERIES_DAILY", params)

    # Persist to cache
    try:
        df = pd.read_csv(StringIO(response))
        df.to_csv(cache_file, index=False)
    except Exception as e:
        logger.warning("Failed to write AV OHLCV cache for %s: %s", symbol, e)

    return _filter_csv_by_date_range(response, start_date, end_date)