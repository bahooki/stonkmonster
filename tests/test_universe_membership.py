from __future__ import annotations

import pandas as pd

from stonkmodel.data.universe import filter_frames_by_membership_intervals, reconstruct_sp500_membership_intervals


def test_reconstruct_sp500_membership_intervals_backfills_history() -> None:
    changes = pd.DataFrame(
        {
            "Effective Date": ["2025-01-05", "2024-01-03"],
            "Added Ticker": ["CCC", "BBB"],
            "Removed Ticker": ["BBB", "AAA"],
        }
    )

    intervals = reconstruct_sp500_membership_intervals(current_symbols=["CCC"], changes_table=changes)
    assert not intervals.empty

    by_symbol = {symbol: group.reset_index(drop=True) for symbol, group in intervals.groupby("symbol")}
    assert set(by_symbol) >= {"AAA", "BBB", "CCC"}

    bbb = by_symbol["BBB"].iloc[0]
    assert str(pd.to_datetime(bbb["start_date"], utc=True).date()) == "2024-01-03"
    assert str(pd.to_datetime(bbb["end_date"], utc=True).date()) == "2025-01-05"

    ccc = by_symbol["CCC"].iloc[0]
    assert str(pd.to_datetime(ccc["start_date"], utc=True).date()) == "2025-01-05"



def test_filter_frames_by_membership_intervals_keeps_only_valid_dates() -> None:
    intervals = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "start_date": pd.to_datetime(["2024-01-01", "2024-03-01"], utc=True),
            "end_date": pd.to_datetime(["2024-02-01", "2024-04-01"], utc=True),
        }
    )

    frames = {
        "AAA": pd.DataFrame(
            {
                "symbol": ["AAA", "AAA", "AAA", "AAA"],
                "datetime": pd.to_datetime(
                    ["2024-01-10", "2024-02-15", "2024-03-10", "2024-04-10"],
                    utc=True,
                ),
                "close": [1.0, 2.0, 3.0, 4.0],
            }
        )
    }

    filtered = filter_frames_by_membership_intervals(frames, intervals)
    assert "AAA" in filtered
    dates = pd.to_datetime(filtered["AAA"]["datetime"], utc=True).dt.strftime("%Y-%m-%d").tolist()
    assert dates == ["2024-01-10", "2024-03-10"]
