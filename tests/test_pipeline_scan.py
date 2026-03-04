from __future__ import annotations

from pathlib import Path

import pandas as pd

import stonkmodel.pipeline as pipeline_mod
from stonkmodel.config import Settings
from stonkmodel.pipeline import StonkService


def _service(tmp_path: Path) -> StonkService:
    settings = Settings(
        data_dir=tmp_path / "data",
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        models_dir=tmp_path / "models",
    )
    for path in (settings.data_dir, settings.raw_data_dir, settings.processed_data_dir, settings.models_dir):
        path.mkdir(parents=True, exist_ok=True)
    return StonkService(settings)


def test_scan_requests_fundamentals_for_full_loaded_universe(monkeypatch, tmp_path: Path) -> None:
    service = _service(tmp_path)
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    frames = {
        sym: pd.DataFrame(
            {
                "symbol": [sym],
                "datetime": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True),
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )
        for sym in symbols
    }
    service.load_history = lambda **kwargs: frames  # type: ignore[method-assign]

    captured: dict[str, int] = {}

    def _fake_build_fundamental_table(*, symbols, cache_path, refresh, max_symbols, provider, fmp_api_key, fmp_base_url, request_workers):
        captured["max_symbols"] = int(max_symbols) if max_symbols is not None else -1
        return pd.DataFrame()

    monkeypatch.setattr(pipeline_mod, "build_fundamental_table", _fake_build_fundamental_table)
    monkeypatch.setattr(pipeline_mod, "build_macro_feature_table", lambda **kwargs: pd.DataFrame())
    service.scanner.scan = lambda **kwargs: pd.DataFrame({"ok": [1]})  # type: ignore[method-assign]

    out = service.scan(interval="1d", years=1, refresh_prices=False)
    assert not out.empty
    assert int(captured["max_symbols"]) == len(symbols)
