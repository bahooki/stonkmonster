from __future__ import annotations

import pandas as pd

from stonkmodel.features.indicators import infer_feature_columns
from stonkmodel.features.labels import add_forward_labels


def test_forward_labels_include_excess_and_rank_targets() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "datetime": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                    "2024-01-03T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                    "2024-01-03T00:00:00Z",
                ],
                utc=True,
            ),
            "close": [100.0, 110.0, 121.0, 100.0, 90.0, 81.0],
        }
    )

    out = add_forward_labels(frame, horizon_bars=1, threshold=0.0, label_mode="excess")
    day1 = out.loc[out["datetime"] == pd.Timestamp("2024-01-01T00:00:00Z")]
    assert not day1.empty
    # AAA +10%, BBB -10%, market 0%, so excess should be +10% and -10%.
    assert float(day1.loc[day1["symbol"] == "AAA", "future_excess_return"].iloc[0]) > 0
    assert float(day1.loc[day1["symbol"] == "BBB", "future_excess_return"].iloc[0]) < 0
    assert int(day1.loc[day1["symbol"] == "AAA", "future_direction"].iloc[0]) == 1
    assert int(day1.loc[day1["symbol"] == "BBB", "future_direction"].iloc[0]) == 0

    out_rank = add_forward_labels(frame, horizon_bars=1, threshold=0.5, label_mode="cross_sectional")
    day1_rank = out_rank.loc[out_rank["datetime"] == pd.Timestamp("2024-01-01T00:00:00Z")]
    assert int(day1_rank.loc[day1_rank["symbol"] == "AAA", "future_direction"].iloc[0]) == 1
    assert int(day1_rank.loc[day1_rank["symbol"] == "BBB", "future_direction"].iloc[0]) == 0


def test_infer_feature_columns_excludes_new_label_columns() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "datetime": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "future_return": [0.01],
            "future_market_return": [0.005],
            "future_excess_return": [0.005],
            "future_rank_pct": [0.8],
            "future_direction": [1],
            "feature_x": [123.0],
        }
    )
    cols = infer_feature_columns(frame)
    assert "feature_x" in cols
    assert "future_market_return" not in cols
    assert "future_excess_return" not in cols
    assert "future_rank_pct" not in cols


def test_forward_labels_use_adjusted_close_when_present() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "datetime": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                    "2024-01-03T00:00:00Z",
                ],
                utc=True,
            ),
            "close": [100.0, 50.0, 50.0],  # raw split-distorted close
            "adj_close": [100.0, 100.0, 110.0],  # adjusted close has true path
        }
    )

    out = add_forward_labels(frame, horizon_bars=1, threshold=0.0, label_mode="absolute")
    first = out.loc[out["datetime"] == pd.Timestamp("2024-01-01T00:00:00Z", tz="UTC")].iloc[0]

    # With adjusted close, first future return is 0% (100 -> 100), not -50%.
    assert float(first["future_return"]) == 0.0
