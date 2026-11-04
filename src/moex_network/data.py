"""Downloading and cleaning daily closing prices from the MOEX ISS API."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from . import config


def fetch_close(session, ticker: str, start: str, end: str, board: str = config.BOARD) -> pd.Series:
    """Daily ``LEGALCLOSEPRICE`` for one ticker.

    ``LEGALCLOSEPRICE`` is the official MOEX closing price (volume-weighted over
    the closing auction / last minutes of trading). It is more robust than the
    last-trade ``CLOSE`` and is defined consistently for every share.
    """
    import apimoex  # imported lazily: only needed when refreshing data

    rows = apimoex.get_board_history(
        session, ticker, start=start, end=end, board=board,
        columns=("TRADEDATE", "LEGALCLOSEPRICE"),
    )
    if not rows:
        return pd.Series(name=ticker, dtype=float)
    df = pd.DataFrame(rows)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    return df.set_index("TRADEDATE").sort_index()["LEGALCLOSEPRICE"].rename(ticker)


def download_prices(
    tickers: Iterable[str] = config.TICKERS,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
) -> tuple[pd.DataFrame, list[str]]:
    """Download closing prices for all tickers. Returns (prices, failed_tickers)."""
    import requests

    series, failed = {}, []
    with requests.Session() as session:
        for t in tickers:
            try:
                s = fetch_close(session, t, start, end)
            except Exception as exc:  # network / API errors: skip ticker, report
                print(f"  x {t}: {exc}")
                failed.append(t)
                continue
            if s.empty:
                failed.append(t)
            else:
                series[t] = s
    prices = pd.DataFrame(series)
    prices.index.name = "TRADEDATE"
    return prices, failed


def load_prices(path=config.RAW_PRICES) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def adjust_for_splits(prices: pd.DataFrame, splits: list[dict] = config.SPLITS) -> pd.DataFrame:
    """Back-adjust prices before each split so the series is continuous."""
    adjusted = prices.copy()
    for s in splits:
        if s["ticker"] in adjusted.columns:
            before = adjusted.index < pd.Timestamp(s["date"])
            adjusted.loc[before, s["ticker"]] *= s["factor"]
    return adjusted


def clean_prices(
    prices: pd.DataFrame,
    min_coverage_ticker: float = 0.5,
    min_coverage_day: float = 0.8,
) -> tuple[pd.DataFrame, dict]:
    """Drop thin tickers/days, align to a common start and forward-fill gaps.

    1. Drop tickers with less than ``min_coverage_ticker`` of days observed
       (e.g. recent IPOs).
    2. Drop days on which less than ``min_coverage_day`` of tickers traded.
    3. Start the sample on the first date on which *every* ticker has a price,
       so no history is invented by back-filling.
    4. Forward-fill remaining isolated gaps (trading halts).
    """
    report: dict = {"shape_before": prices.shape}
    n_days = len(prices)
    kept = prices.dropna(axis=1, thresh=int(n_days * min_coverage_ticker))
    report["dropped_tickers"] = sorted(set(prices.columns) - set(kept.columns))

    kept = kept.dropna(axis=0, thresh=int(kept.shape[1] * min_coverage_day))
    common_start = kept.apply(lambda col: col.first_valid_index()).max()
    report["common_start"] = common_start
    kept = kept.loc[common_start:].ffill().dropna()

    report["shape_after"] = kept.shape
    return kept, report


def find_price_jumps(prices: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """List days with |log-return| > ``threshold`` (splits, dividend gaps, data errors)."""
    log_ret = np.log(prices / prices.shift(1))
    stacked = log_ret.stack()
    jumps = stacked[stacked.abs() > threshold]
    out = jumps.rename("log_return").reset_index()
    out.columns = ["date", "ticker", "log_return"]
    out["price_ratio"] = np.exp(out["log_return"])
    return out.sort_values("date").reset_index(drop=True)


def flat_runs(prices: pd.DataFrame, min_run: int = 3) -> pd.DataFrame:
    """Runs of >= ``min_run`` identical consecutive prices (typical of trading halts)."""
    rows = []
    for t in prices.columns:
        s = prices[t]
        run_id = s.ne(s.shift()).cumsum()
        for _, grp in s.groupby(run_id):
            if len(grp) >= min_run:
                rows.append({"ticker": t, "start": grp.index[0].date(),
                             "end": grp.index[-1].date(), "days": len(grp)})
    return pd.DataFrame(rows, columns=["ticker", "start", "end", "days"])
