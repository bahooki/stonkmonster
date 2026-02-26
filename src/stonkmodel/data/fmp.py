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

    @retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(4))
    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        merged = dict(params or {})
        merged["apikey"] = self.api_key
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=merged)
            resp.raise_for_status()
            return resp.json()

    def get_sp500_constituents(self) -> list[dict[str, Any]]:
        payload = self._get_json("/sp500-constituent")
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return payload["data"]
        return []

    def get_historical_price_eod(self, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        payload = self._get_json(
            "/historical-price-eod/full",
            params={"symbol": symbol, "from": start.isoformat(), "to": end.isoformat()},
        )
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if isinstance(payload.get("historical"), list):
                return payload["historical"]
            if isinstance(payload.get("data"), list):
                return payload["data"]
        return []

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
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if isinstance(payload.get("historical"), list):
                return payload["historical"]
            if isinstance(payload.get("data"), list):
                return payload["data"]
        return []

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
