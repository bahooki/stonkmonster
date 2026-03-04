from __future__ import annotations

import pandas as pd

from stonkmodel.features.labels import add_train_test_split


def test_default_split_is_chronological_by_datetime() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"] * 10 + ["BBB"] * 10,
            "datetime": list(dates) + list(dates),
            "close": [100 + i for i in range(20)],
        }
    ).sort_values(["symbol", "datetime"], ignore_index=True)

    out = add_train_test_split(frame, split_date=None)
    train_dates = pd.to_datetime(out.loc[out["split"] == "train", "datetime"], utc=True, errors="coerce")
    test_dates = pd.to_datetime(out.loc[out["split"] == "test", "datetime"], utc=True, errors="coerce")

    assert not train_dates.empty
    assert not test_dates.empty
    assert train_dates.max() < test_dates.min()
    assert out.loc[(out["symbol"] == "AAA") & (out["datetime"] == dates[-1]), "split"].iloc[0] == "test"
    assert out.loc[(out["symbol"] == "BBB") & (out["datetime"] == dates[-1]), "split"].iloc[0] == "test"


def test_explicit_split_date_handles_string_datetime_column() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "datetime": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "close": [100.0, 101.0, 102.0],
        }
    )
    out = add_train_test_split(frame, split_date="2024-01-02")
    assert out["split"].tolist() == ["train", "test", "test"]
