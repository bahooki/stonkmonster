from __future__ import annotations

import pandas as pd

import stonkmodel.data.external_features as ext


class _DummyEventClient:
    def get_analyst_stock_recommendations(self, symbol: str, limit: int = 240):
        return [
            {
                "date": "2025-01-02",
                "analystRatingsStrongBuy": 3,
                "analystRatingsBuy": 5,
                "analystRatingsHold": 2,
                "analystRatingsSell": 1,
                "analystRatingsStrongSell": 0,
            }
        ]

    def get_historical_rating(self, symbol: str, limit: int = 240):
        return []

    def get_historical_earning_calendar(self, symbol: str, limit: int = 240):
        return []

    def get_grade_history(self, symbol: str, limit: int = 240):
        return []

    def get_analyst_estimates(self, symbol: str, period: str = "annual", page: int = 0, limit: int = 40):
        return [
            {
                "date": "2025-01-03",
                "estimatedEpsAvg": 6.0,
                "estimatedEpsHigh": 7.5,
                "estimatedEpsLow": 4.5,
                "estimatedRevenueAvg": 120.0,
                "estimatedRevenueHigh": 140.0,
                "estimatedRevenueLow": 100.0,
            }
        ]

    def get_price_target_summary(self, symbol: str):
        return {
            "date": "2025-01-04",
            "targetMean": 150.0,
            "targetHigh": 170.0,
            "targetLow": 120.0,
            "currentPrice": 100.0,
            "analystCount": 12,
        }

    def get_price_target_consensus(self, symbol: str):
        return {}

    def get_grades_consensus(self, symbol: str):
        return {
            "date": "2025-01-05",
            "buy": 6,
            "hold": 3,
            "sell": 1,
            "consensusScore": 0.42,
        }

    def search_insider_trades(self, symbol: str | None = None, page: int = 0, limit: int = 100):
        return [
            {
                "symbol": symbol,
                "transactionDate": "2025-01-10",
                "transactionType": "Purchase",
                "shares": 1000,
                "price": 10,
            },
            {
                "symbol": symbol,
                "transactionDate": "2025-01-12",
                "transactionType": "Sale",
                "shares": 400,
                "price": 12,
            },
        ]

    def get_latest_insider_trades(self, page: int = 0, limit: int = 100):
        return []


def test_fetch_symbol_event_features_fmp_adds_new_indicator_columns() -> None:
    frame = ext.fetch_symbol_event_features_fmp(symbol="AAA", client=_DummyEventClient(), limit=32)
    assert not frame.empty

    expected_cols = {
        "analyst_est_eps_avg",
        "price_target_upside",
        "grades_consensus_score",
        "insider_net_value_30d",
        "insider_buy_sell_ratio_30d",
    }
    assert expected_cols.issubset(set(frame.columns))

    latest = frame.sort_values("asof_datetime").iloc[-1]
    assert pd.notna(latest["insider_net_value_30d"])


class _DummyPoliticalClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_senate_trades(self, symbol: str | None = None, page: int = 0, limit: int = 100):
        if page > 0:
            return []
        return [
            {
                "symbol": "AAA",
                "transactionDate": "2025-01-03",
                "type": "Purchase",
                "amount": "$1,001 - $15,000",
                "representative": "Jane Doe",
                "disclosureDate": "2025-01-08",
            }
        ]

    def get_house_trades(self, symbol: str | None = None, page: int = 0, limit: int = 100):
        if page > 0:
            return []
        return [
            {
                "symbol": "AAA",
                "transactionDate": "2025-01-04",
                "type": "Sale (Full)",
                "amount": "$15,001 - $50,000",
                "representative": "John Doe",
                "disclosureDate": "2025-01-10",
            }
        ]

    def get_latest_senate_disclosures(self, page: int = 0, limit: int = 100):
        return []

    def get_latest_house_disclosures(self, page: int = 0, limit: int = 100):
        return []


def test_fetch_politician_trades_fmp_parses_ranges_and_side(monkeypatch) -> None:
    monkeypatch.setattr(ext, "FMPClient", _DummyPoliticalClient)

    trades = ext.fetch_politician_trades_fmp(
        symbols=["AAA"],
        fmp_api_key="test-key",
        start_datetime=pd.Timestamp("2025-01-01", tz="UTC"),
        end_datetime=pd.Timestamp("2025-02-01", tz="UTC"),
        request_workers=2,
        max_pages=2,
        page_size=50,
    )
    assert len(trades) == 2
    assert set(trades["side"].tolist()) == {"buy", "sell"}
    assert "amount_low_usd" in trades.columns
    assert "amount_high_usd" in trades.columns


def test_engineer_politician_features_adds_chamber_and_rolling_metrics() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "trade_date": pd.to_datetime(["2025-01-03", "2025-01-10", "2025-01-15"], utc=True),
            "amount_usd": [1000.0, 800.0, 500.0],
            "side": ["buy", "sell", "buy"],
            "chamber": ["senate", "house", "senate"],
            "politician": ["A", "B", "C"],
            "disclosure_lag_days": [2.0, 3.0, 4.0],
        }
    )

    features = ext.engineer_politician_features(trades)
    assert not features.empty
    required = {
        "politician_flow_30d",
        "politician_buy_sell_ratio_30d",
        "politician_senate_flow_30d",
        "politician_house_flow_30d",
    }
    assert required.issubset(set(features.columns))
