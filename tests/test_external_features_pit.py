from __future__ import annotations

import pandas as pd

from stonkmodel.data.external_features import merge_external_features


def test_merge_external_features_uses_point_in_time_fundamentals() -> None:
    prices = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "datetime": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z", "2024-03-01T00:00:00Z"],
                utc=True,
            ),
            "close": [10.0, 11.0, 12.0],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "asof_datetime": pd.to_datetime(["2024-01-15T00:00:00Z", "2024-02-15T00:00:00Z"], utc=True),
            "market_cap": [100.0, 200.0],
        }
    )

    out = merge_external_features(prices, fundamental_table=fundamentals, politician_features=None)
    assert "fundamental_asof_datetime" in out.columns

    first = out.iloc[0]
    second = out.iloc[1]
    third = out.iloc[2]
    assert pd.isna(first["market_cap"])
    assert str(second["fundamental_asof_datetime"]) == "2024-01-15 00:00:00+00:00"
    assert float(second["market_cap"]) == 100.0
    assert str(third["fundamental_asof_datetime"]) == "2024-02-15 00:00:00+00:00"
    assert float(third["market_cap"]) == 200.0


def test_merge_external_features_adds_financial_factor_derivatives() -> None:
    prices = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "datetime": pd.to_datetime(["2024-03-01T00:00:00Z", "2024-03-02T00:00:00Z"], utc=True),
            "close": [10.0, 11.0],
            "volume": [1000.0, 1200.0],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "asof_datetime": pd.to_datetime(["2024-02-15T00:00:00Z"], utc=True),
            "market_cap": [1000.0],
            "trailing_pe": [20.0],
            "price_to_book": [2.0],
            "price_to_sales_trailing_12m": [5.0],
            "free_cashflow": [100.0],
            "operating_cashflow": [120.0],
            "shares_outstanding": [200.0],
            "float_shares": [150.0],
            "analyst_net_score": [0.4],
            "rating_score": [3.0],
            "return_on_assets": [0.08],
            "return_on_equity": [0.16],
            "profit_margins": [0.22],
            "operating_margins": [0.18],
            "debt_to_equity": [0.5],
            "current_ratio": [2.0],
            "earnings_eps_surprise_pct": [0.1],
        }
    )

    out = merge_external_features(prices, fundamental_table=fundamentals, politician_features=None)

    assert "earnings_yield" in out.columns
    assert "book_to_price" in out.columns
    assert "free_cashflow_yield" in out.columns
    assert "turnover_ratio" in out.columns
    assert "analyst_rating_blend" in out.columns
    assert "quality_composite" in out.columns
    assert "surprise_sentiment_interaction" in out.columns

    first = out.iloc[0]
    assert float(first["earnings_yield"]) == 0.05
    assert float(first["book_to_price"]) == 0.5
    assert float(first["free_cashflow_yield"]) == 0.1
    assert float(first["turnover_ratio"]) == 5.0


def test_merge_external_features_adds_macro_asof_features() -> None:
    prices = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "datetime": pd.to_datetime(["2024-03-01T00:00:00Z", "2024-03-02T00:00:00Z"], utc=True),
            "close": [10.0, 11.0],
            "ret_5": [0.02, 0.03],
            "alpha_ret_5": [0.01, 0.015],
            "volatility_20": [0.2, 0.25],
        }
    )
    macro = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-02-29T00:00:00Z", "2024-03-02T00:00:00Z"], utc=True),
            "macro_risk_off_score": [1.2, 0.4],
            "macro_surprise_abs_index": [0.9, 0.2],
            "macro_days_to_next_us_major_event": [1.0, 3.0],
        }
    )

    out = merge_external_features(prices, fundamental_table=None, politician_features=None, macro_features=macro)
    assert "macro_risk_off_score" in out.columns
    assert "alpha_ret_5_macro_adj" in out.columns
    assert "ret_5_macro_shock_adj" in out.columns
    assert "event_risk_tension" in out.columns

    first = out.iloc[0]
    assert float(first["macro_risk_off_score"]) == 1.2


def test_merge_external_features_ignores_fundamentals_without_asof_datetime() -> None:
    prices = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "datetime": pd.to_datetime(["2024-03-01T00:00:00Z", "2024-03-02T00:00:00Z"], utc=True),
            "close": [10.0, 11.0],
            "volume": [1000.0, 1200.0],
        }
    )
    malformed_fundamentals = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "market_cap": [1000.0],
        }
    )

    out = merge_external_features(prices, fundamental_table=malformed_fundamentals, politician_features=None)
    assert not out.empty
    assert "market_cap" in out.columns
    assert pd.isna(out["market_cap"]).all()


def test_merge_external_features_politician_join_keeps_symbol_column() -> None:
    prices = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "datetime": pd.to_datetime(
                ["2024-03-01T00:00:00Z", "2024-03-02T00:00:00Z", "2024-03-02T00:00:00Z"],
                utc=True,
            ),
            "close": [10.0, 11.0, 20.0],
            "volume": [100.0, 110.0, 150.0],
        }
    )
    politician = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "trade_date": pd.to_datetime(["2024-02-28T00:00:00Z", "2024-03-01T00:00:00Z"], utc=True),
            "politician_signed_flow": [5000.0, -2500.0],
            "politician_flow_30d": [5000.0, -2500.0],
            "politician_trade_count": [1.0, 1.0],
        }
    )

    out = merge_external_features(prices, fundamental_table=None, politician_features=politician)
    assert "symbol" in out.columns
    assert "symbol_x" not in out.columns
    assert "symbol_y" not in out.columns
    assert len(out) == 3
    assert float(out.loc[out["symbol"] == "AAA", "politician_flow_30d"].iloc[-1]) == 5000.0
