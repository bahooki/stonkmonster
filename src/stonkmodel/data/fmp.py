from __future__ import annotations

from datetime import date
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class FMPClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://financialmodelingprep.com/stable",
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("FMP API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.v3_base_url = self._derive_v3_base_url(self.base_url)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(4))
    def _get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        base_url: str | None = None,
    ) -> Any:
        merged = dict(params or {})
        merged["apikey"] = self.api_key
        base = (base_url or self.base_url).rstrip("/")
        url = f"{base}{path if path.startswith('/') else '/' + path}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=merged)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _derive_v3_base_url(base_url: str) -> str:
        base = base_url.rstrip("/")
        if "/api/v3" in base:
            return base
        if "/stable" in base:
            return base.replace("/stable", "/api/v3")
        return "https://financialmodelingprep.com/api/v3"

    def get_sp500_constituents(self) -> list[dict[str, Any]]:
        payload = self._get_json("/sp500-constituent")
        return self._payload_to_rows(payload)

    def get_historical_price_eod(self, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        payload = self._get_json(
            "/historical-price-eod/full",
            params={"symbol": symbol, "from": start.isoformat(), "to": end.isoformat()},
        )
        return self._payload_to_rows(payload)

    def get_historical_chart(
        self,
        symbol: str,
        interval_token: str,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        payload = self._get_json(
            f"/historical-chart/{interval_token}",
            params={"symbol": symbol, "from": start.isoformat(), "to": end.isoformat()},
        )
        return self._payload_to_rows(payload)

    def get_quote(self, symbol: str) -> dict[str, Any]:
        payload = self._get_json("/quote", params={"symbol": symbol})
        rows = self._payload_to_rows(payload)
        if rows:
            return rows[0]
        if isinstance(payload, dict):
            return payload
        return {}

    def get_income_statement(
        self,
        symbol: str,
        period: str = "quarter",
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        payload = self._get_json(
            "/income-statement",
            params={"symbol": symbol, "period": period, "limit": int(limit)},
        )
        return self._payload_to_rows(payload)

    def get_balance_sheet_statement(
        self,
        symbol: str,
        period: str = "quarter",
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        payload = self._get_json(
            "/balance-sheet-statement",
            params={"symbol": symbol, "period": period, "limit": int(limit)},
        )
        return self._payload_to_rows(payload)

    def get_cash_flow_statement(
        self,
        symbol: str,
        period: str = "quarter",
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        payload = self._get_json(
            "/cash-flow-statement",
            params={"symbol": symbol, "period": period, "limit": int(limit)},
        )
        return self._payload_to_rows(payload)

    def get_profile(self, symbol: str) -> dict[str, Any]:
        payload = self._get_json("/profile", params={"symbol": symbol})
        if isinstance(payload, list):
            return payload[0] if payload else {}
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            data = payload["data"]
            return data[0] if data else {}
        return payload if isinstance(payload, dict) else {}

    def get_ratios_ttm(self, symbol: str) -> dict[str, Any]:
        payload = self._get_json("/ratios-ttm", params={"symbol": symbol})
        if isinstance(payload, list):
            return payload[0] if payload else {}
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            data = payload["data"]
            return data[0] if data else {}
        return payload if isinstance(payload, dict) else {}

    def get_key_metrics_ttm(self, symbol: str) -> dict[str, Any]:
        payload = self._get_json("/key-metrics-ttm", params={"symbol": symbol})
        if isinstance(payload, list):
            return payload[0] if payload else {}
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            data = payload["data"]
            return data[0] if data else {}
        return payload if isinstance(payload, dict) else {}

    def get_ratios_history(self, symbol: str, limit: int = 120, period: str = "quarter") -> list[dict[str, Any]]:
        payload = self._get_json("/ratios", params={"symbol": symbol, "limit": int(limit), "period": period})
        return self._payload_to_rows(payload)

    def get_key_metrics_history(self, symbol: str, limit: int = 120, period: str = "quarter") -> list[dict[str, Any]]:
        payload = self._get_json("/key-metrics", params={"symbol": symbol, "limit": int(limit), "period": period})
        return self._payload_to_rows(payload)

    def get_analyst_stock_recommendations(self, symbol: str, limit: int = 120) -> list[dict[str, Any]]:
        payload = self._get_json(
            f"/analyst-stock-recommendations/{symbol}",
            params={"limit": int(limit)},
            base_url=self.v3_base_url,
        )
        return self._payload_to_rows(payload)

    def get_historical_rating(self, symbol: str, limit: int = 240) -> list[dict[str, Any]]:
        payload = self._get_json(
            f"/historical-rating/{symbol}",
            params={"limit": int(limit)},
            base_url=self.v3_base_url,
        )
        return self._payload_to_rows(payload)

    def get_historical_earning_calendar(self, symbol: str, limit: int = 120) -> list[dict[str, Any]]:
        payload = self._get_json(
            f"/historical/earning_calendar/{symbol}",
            params={"limit": int(limit)},
            base_url=self.v3_base_url,
        )
        return self._payload_to_rows(payload)

    def get_grade_history(self, symbol: str, limit: int = 240) -> list[dict[str, Any]]:
        payload = self._get_json(
            f"/grade/{symbol}",
            params={"limit": int(limit)},
            base_url=self.v3_base_url,
        )
        return self._payload_to_rows(payload)

    def get_analyst_estimates(
        self,
        symbol: str,
        period: str = "annual",
        page: int = 0,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        payload = self._get_json(
            "/analyst-estimates",
            params={"symbol": symbol, "period": period, "page": int(page), "limit": int(limit)},
        )
        return self._payload_to_rows(payload)

    def get_price_target_summary(self, symbol: str) -> dict[str, Any]:
        payload = self._get_json("/price-target-summary", params={"symbol": symbol})
        rows = self._payload_to_rows(payload)
        return rows[0] if rows else (payload if isinstance(payload, dict) else {})

    def get_price_target_consensus(self, symbol: str) -> dict[str, Any]:
        payload = self._get_json("/price-target-consensus", params={"symbol": symbol})
        rows = self._payload_to_rows(payload)
        return rows[0] if rows else (payload if isinstance(payload, dict) else {})

    def get_grades_consensus(self, symbol: str) -> dict[str, Any]:
        payload = self._get_json("/grades-consensus", params={"symbol": symbol})
        rows = self._payload_to_rows(payload)
        return rows[0] if rows else (payload if isinstance(payload, dict) else {})

    def get_insider_trading_statistics(self, symbol: str) -> list[dict[str, Any]]:
        payload = self._get_json("/insider-trading/statistics", params={"symbol": symbol})
        return self._payload_to_rows(payload)

    def search_insider_trades(self, symbol: str | None = None, page: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"page": int(page), "limit": int(limit)}
        if symbol:
            params["symbol"] = symbol
        payload = self._get_json("/insider-trading/search", params=params)
        return self._payload_to_rows(payload)

    def get_latest_insider_trades(self, page: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        payload = self._get_json("/insider-trading/latest", params={"page": int(page), "limit": int(limit)})
        return self._payload_to_rows(payload)

    def get_insider_transaction_types(self) -> list[dict[str, Any]]:
        payload = self._get_json("/insider-trading-transaction-type")
        return self._payload_to_rows(payload)

    def get_senate_trades(self, symbol: str | None = None, page: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"page": int(page), "limit": int(limit)}
        if symbol:
            params["symbol"] = symbol
        payload = self._get_json("/senate-trades", params=params)
        return self._payload_to_rows(payload)

    def get_house_trades(self, symbol: str | None = None, page: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"page": int(page), "limit": int(limit)}
        if symbol:
            params["symbol"] = symbol
        payload = self._get_json("/house-trades", params=params)
        return self._payload_to_rows(payload)

    def get_latest_senate_disclosures(self, page: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        payload = self._get_json("/senate-latest", params={"page": int(page), "limit": int(limit)})
        return self._payload_to_rows(payload)

    def get_latest_house_disclosures(self, page: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        payload = self._get_json("/house-latest", params={"page": int(page), "limit": int(limit)})
        return self._payload_to_rows(payload)

    def get_senate_trades_by_name(self, name: str, page: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        payload = self._get_json(
            "/senate-trades-by-name",
            params={"name": name, "page": int(page), "limit": int(limit)},
        )
        return self._payload_to_rows(payload)

    def get_economic_indicators(self, name: str) -> list[dict[str, Any]]:
        payload = self._get_json("/economic-indicators", params={"name": name})
        return self._payload_to_rows(payload)

    def get_market_risk_premium(self) -> list[dict[str, Any]]:
        payload = self._get_json("/market-risk-premium")
        return self._payload_to_rows(payload)

    def get_sector_performance_snapshot(self, date_value: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if date_value:
            params["date"] = date_value
        payload = self._get_json("/sector-performance-snapshot", params=params)
        return self._payload_to_rows(payload)

    def get_industry_performance_snapshot(self, date_value: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if date_value:
            params["date"] = date_value
        payload = self._get_json("/industry-performance-snapshot", params=params)
        return self._payload_to_rows(payload)

    def get_sector_pe_snapshot(self, date_value: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if date_value:
            params["date"] = date_value
        payload = self._get_json("/sector-pe-snapshot", params=params)
        return self._payload_to_rows(payload)

    def get_industry_pe_snapshot(self, date_value: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if date_value:
            params["date"] = date_value
        payload = self._get_json("/industry-pe-snapshot", params=params)
        return self._payload_to_rows(payload)

    def get_historical_sector_performance(self, sector: str) -> list[dict[str, Any]]:
        payload = self._get_json("/historical-sector-performance", params={"sector": sector})
        return self._payload_to_rows(payload)

    def get_historical_sector_pe(self, sector: str) -> list[dict[str, Any]]:
        payload = self._get_json("/historical-sector-pe", params={"sector": sector})
        return self._payload_to_rows(payload)

    def get_available_sectors(self) -> list[dict[str, Any]]:
        payload = self._get_json("/available-sectors")
        return self._payload_to_rows(payload)

    def get_available_industries(self) -> list[dict[str, Any]]:
        payload = self._get_json("/available-industries")
        return self._payload_to_rows(payload)

    @staticmethod
    def _payload_to_rows(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("historical", "data"):
                values = payload.get(key)
                if isinstance(values, list):
                    return [row for row in values if isinstance(row, dict)]
        return []
