from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "StonkModel"
    app_env: Literal["dev", "prod", "test"] = "dev"

    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")

    universe_source: Literal["sp500", "sp100", "custom"] = "sp500"
    custom_universe_csv: Path | None = None

    default_interval: str = "1d"
    history_years: int = 5

    # Keep this comfortably above S&P 500 constituent count (~503 tickers)
    # so we don't accidentally truncate the universe on membership changes.
    max_symbols: int = 1000
    request_workers: int = 12
    batch_size: int = 64

    market_data_provider: Literal["auto", "fmp", "polygon", "yfinance"] = "auto"
    fundamentals_provider: Literal["auto", "fmp", "yfinance"] = "auto"

    fmp_api_key: str | None = None
    fmp_base_url: str = "https://financialmodelingprep.com/stable"

    # Optional market data vendor keys for higher-resolution history.
    polygon_api_key: str | None = None
    alphavantage_api_key: str | None = None

    # Model settings
    train_test_split_date: str | None = None
    forward_horizon_bars: int = 1
    return_threshold: float = 0.0

    # Scanner settings
    min_pattern_count: int = 100
    top_n_signals: int = 50


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return settings
