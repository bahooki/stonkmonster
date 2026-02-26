from __future__ import annotations

import pandas as pd

from stonkmodel.features.patterns import add_candlestick_patterns


def test_bearish_engulfing_detection() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["ABC", "ABC"],
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "open": [10.0, 11.2],
            "high": [11.5, 11.3],
            "low": [9.5, 9.1],
            "close": [11.0, 9.2],
            "volume": [1000, 1200],
        }
    )

    out = add_candlestick_patterns(frame)
    assert int(out.loc[out.index[-1], "pattern_bearish_engulfing"]) == 1


def test_doji_detection() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["XYZ"],
            "datetime": pd.to_datetime(["2024-01-01"], utc=True),
            "open": [10.0],
            "high": [10.5],
            "low": [9.5],
            "close": [10.02],
            "volume": [1000],
        }
    )

    out = add_candlestick_patterns(frame)
    assert int(out.loc[0, "pattern_doji"]) == 1
