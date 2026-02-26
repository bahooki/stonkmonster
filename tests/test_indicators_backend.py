from __future__ import annotations

import pandas as pd

import stonkmodel.features.indicators as indicators


class _FakeTrend:
    @staticmethod
    def sma_indicator(close: pd.Series, window: int) -> pd.Series:
        return close.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def ema_indicator(close: pd.Series, window: int) -> pd.Series:
        return close.ewm(span=window, adjust=False).mean()

    @staticmethod
    def wma_indicator(close: pd.Series, window: int) -> pd.Series:
        return close.rolling(window=window, min_periods=1).mean()


class _FakeMomentum:
    @staticmethod
    def rsi(close: pd.Series, window: int) -> pd.Series:
        return close.pct_change().rolling(window=window, min_periods=1).mean().fillna(0.0)


class _FakeTa:
    trend = _FakeTrend()
    momentum = _FakeMomentum()


# Other namespaces intentionally omitted to ensure fallback gracefully tolerates partial backend.


def test_ta_fallback_backend(monkeypatch) -> None:
    monkeypatch.setattr(indicators, "pandas_ta_lib", None)
    monkeypatch.setattr(indicators, "ta_lib", _FakeTa())

    rows = 40
    frame = pd.DataFrame(
        {
            "symbol": ["ABC"] * rows,
            "datetime": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "open": [100 + i * 0.1 for i in range(rows)],
            "high": [101 + i * 0.1 for i in range(rows)],
            "low": [99 + i * 0.1 for i in range(rows)],
            "close": [100 + i * 0.2 for i in range(rows)],
            "volume": [1000 + i * 5 for i in range(rows)],
        }
    )

    out = indicators.add_indicators(frame)

    assert "sma_5" in out.columns
    assert "ema_5" in out.columns
    assert "rsi_14" in out.columns
    assert "dist_sma_5" in out.columns
