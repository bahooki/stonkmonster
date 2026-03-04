from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stonkmodel.analysis.ticker_analysis import TickerAnalyzer
from stonkmodel.ui.ticker_analysis_mode import _parse_symbols


class _FakeFMPClient:
    def get_quote(self, symbol: str):
        return {
            "symbol": symbol,
            "price": 100.0,
            "marketCap": 1_000_000_000.0,
            "sharesOutstanding": 10_000_000.0,
            "yearHigh": 130.0,
            "yearLow": 70.0,
            "priceAvg50": 102.0,
            "priceAvg200": 95.0,
        }

    def get_income_statement(self, symbol: str, period: str = "quarter", limit: int = 40):
        if period == "annual":
            return [
                {"revenue": 2_000_000_000.0, "grossProfit": 1_000_000_000.0},
                {"revenue": 1_700_000_000.0, "grossProfit": 850_000_000.0},
                {"revenue": 1_450_000_000.0, "grossProfit": 725_000_000.0},
                {"revenue": 1_250_000_000.0, "grossProfit": 625_000_000.0},
            ]
        q = []
        for i in range(8):
            growth = 1.04 if i < 4 else 0.96
            q.append(
                {
                    "revenue": 500_000_000.0 * growth,
                    "grossProfit": 260_000_000.0 * growth,
                    "netIncome": 70_000_000.0 * growth,
                    "interestExpense": 4_000_000.0,
                    "depreciationAndAmortization": 6_000_000.0,
                    "ebitda": 95_000_000.0 * growth,
                    "operatingExpenses": 140_000_000.0,
                    "incomeTaxExpense": 12_000_000.0,
                    "incomeBeforeTax": 82_000_000.0,
                }
            )
        return q

    def get_balance_sheet_statement(self, symbol: str, period: str = "quarter", limit: int = 40):
        return [
            {
                "date": "2025-12-31",
                "totalLiabilities": 450_000_000.0,
                "totalStockholdersEquity": 650_000_000.0,
                "totalDebt": 220_000_000.0,
                "cashAndCashEquivalents": 110_000_000.0,
                "preferredStock": 0.0,
            }
        ]

    def get_cash_flow_statement(self, symbol: str, period: str = "quarter", limit: int = 40):
        if period == "annual":
            return [
                {"netCashProvidedByOperatingActivities": 300_000_000.0},
                {"netCashProvidedByOperatingActivities": 270_000_000.0},
                {"netCashProvidedByOperatingActivities": 240_000_000.0},
                {"netCashProvidedByOperatingActivities": 215_000_000.0},
            ]
        return [{"netCashProvidedByOperatingActivities": 80_000_000.0} for _ in range(8)]

    def get_historical_price_eod(self, symbol: str, start: date, end: date):
        idx = pd.date_range(start=start, end=end, freq="5D")
        base = 100.0 if symbol != "VOO" else 90.0
        slope = 0.13 if symbol != "VOO" else 0.10
        cyc = np.sin(np.arange(len(idx)) / 8.0) * 1.2
        values = base + np.arange(len(idx)) * slope + cyc
        return [{"date": d.date().isoformat(), "adjClose": float(v)} for d, v in zip(idx, values)]

    def get_market_risk_premium(self):
        return [{"riskFreeRate": 4.25}]


def test_parse_symbols_textbox_input():
    raw = " TSLA, aapl\nNVDA ; tsla ; MSFT "
    assert _parse_symbols(raw) == ["TSLA", "AAPL", "NVDA", "MSFT"]


def test_ticker_analyzer_single_ticker_shape():
    analyzer = TickerAnalyzer(api_key="x")
    analyzer.client = _FakeFMPClient()  # type: ignore[assignment]

    row = analyzer.analyze_ticker("TSLA")
    assert row["symbol"] == "TSLA"
    assert "target_price" in row
    assert row["target_price"] >= 0.0
    assert np.isfinite(float(row["price"]))
    assert "error" in row
    assert row["error"] == ""


def test_ticker_analyzer_many_dedupes_symbols():
    analyzer = TickerAnalyzer(api_key="x")
    analyzer.client = _FakeFMPClient()  # type: ignore[assignment]

    frame = analyzer.analyze_many(["TSLA", "AAPL", "tsla"], workers=2)
    assert isinstance(frame, pd.DataFrame)
    assert set(frame["symbol"].tolist()) == {"TSLA", "AAPL"}
    assert "upside_pct" in frame.columns

