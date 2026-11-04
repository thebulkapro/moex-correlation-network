import numpy as np
import pandas as pd

from moex_network.data import adjust_for_splits, clean_prices, find_price_jumps


def test_split_adjustment_removes_jump():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    prices = pd.DataFrame({"X": [1000.0, 1010, 1005, 10.1, 10.2, 10.0]}, index=idx)
    adj = adjust_for_splits(prices, [{"ticker": "X", "date": str(idx[3].date()), "factor": 0.01}])
    assert find_price_jumps(adj).empty
    assert adj["X"].iloc[0] == 10.0
    assert not find_price_jumps(prices).empty


def test_clean_prices_common_start_and_ffill():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    prices = pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 6],
        "B": [np.nan, np.nan, 1, np.nan, 3, 4],
        "C": [np.nan] * 5 + [1.0],  # < 50% coverage -> dropped
    }, index=idx, dtype=float)
    clean, report = clean_prices(prices)
    assert report["dropped_tickers"] == ["C"]
    assert clean.index[0] == idx[2]
    assert clean.loc[idx[3], "B"] == 1
    assert not clean.isna().any().any()
