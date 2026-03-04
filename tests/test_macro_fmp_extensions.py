from __future__ import annotations

import pandas as pd

import stonkmodel.data.macro as macro


def test_derive_macro_features_builds_sector_and_erp_composites() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
            "macro_proxy_spy": [100, 101, 102, 101, 103],
            "macro_proxy_tlt": [90, 90, 91, 91, 92],
            "macro_proxy_iwm": [50, 50.5, 51, 50.8, 51.2],
            "macro_proxy_qqq": [200, 201, 202, 201, 203],
            "macro_proxy_hyg": [80, 80.2, 80.5, 80.1, 80.7],
            "macro_proxy_lqd": [100, 100.1, 100.3, 100.2, 100.4],
            "macro_fmp_market_risk_premium": [0.05, 0.051, 0.052, 0.0515, 0.053],
            "macro_fmp_risk_free_rate": [0.02, 0.02, 0.021, 0.021, 0.022],
            "macro_sector_perf_information_technology": [0.2, 0.3, -0.1, 0.15, 0.25],
            "macro_sector_perf_financials": [0.1, -0.05, 0.2, 0.05, 0.12],
            "macro_sector_pe_information_technology": [24, 24.2, 24.5, 24.6, 24.8],
            "macro_sector_pe_financials": [15, 15.1, 15.2, 15.0, 15.3],
        }
    )

    out = macro._derive_macro_features(frame)
    assert "macro_sector_perf_mean" in out.columns
    assert "macro_sector_perf_std" in out.columns
    assert "macro_sector_pe_mean" in out.columns
    assert "macro_sector_perf_to_pe" in out.columns
    assert "macro_equity_risk_premium_total" in out.columns


def test_fetch_fmp_market_risk_premium_table_parses_payload(monkeypatch) -> None:
    def _fake_get(url: str, params: dict[str, object], timeout: float = 30.0):
        return [
            {
                "date": "2025-01-01",
                "marketRiskPremium": 0.05,
                "riskFreeRate": 0.02,
                "expectedMarketReturn": 0.07,
            }
        ]

    monkeypatch.setattr(macro, "_http_get_json_once", _fake_get)
    out = macro.fetch_fmp_market_risk_premium_table(api_key="k")
    assert not out.empty
    assert "macro_fmp_market_risk_premium" in out.columns
    assert float(out["macro_fmp_market_risk_premium"].iloc[0]) == 0.05


def test_fetch_fmp_economic_indicators_table_parses_multiple_series(monkeypatch) -> None:
    def _fake_get(url: str, params: dict[str, object], timeout: float = 30.0):
        name = params.get("name")
        if name == "GDP":
            return [{"date": "2025-01-01", "value": "28700"}]
        if name == "CPI":
            return [{"date": "2025-01-01", "value": "312.4"}]
        return []

    monkeypatch.setattr(macro, "_http_get_json_once", _fake_get)
    out = macro.fetch_fmp_economic_indicators_table(
        api_key="k",
        indicator_map={"GDP": "macro_fmp_gdp", "CPI": "macro_fmp_cpi"},
        request_workers=2,
    )
    assert not out.empty
    assert "macro_fmp_gdp" in out.columns
    assert "macro_fmp_cpi" in out.columns
