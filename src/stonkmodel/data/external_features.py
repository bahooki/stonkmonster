from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yfinance as yf

from stonkmodel.data.fmp import FMPClient


FUNDAMENTAL_COLUMNS = [
    "market_cap",
    "float_shares",
    "shares_outstanding",
    "trailing_pe",
    "forward_pe",
    "peg_ratio",
    "price_to_book",
    "price_to_sales_trailing_12m",
    "enterprise_to_revenue",
    "enterprise_to_ebitda",
    "beta",
    "debt_to_equity",
    "current_ratio",
    "quick_ratio",
    "profit_margins",
    "gross_margins",
    "operating_margins",
    "ebitda_margins",
    "return_on_assets",
    "return_on_equity",
    "earnings_growth",
    "revenue_growth",
    "free_cashflow",
    "operating_cashflow",
]


def _first_valid(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def fetch_symbol_fundamentals_yfinance(symbol: str) -> dict[str, float | str | datetime | None]:
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}

    row: dict[str, float | str | datetime | None] = {
        "symbol": symbol,
        "asof_datetime": datetime.now(timezone.utc),
        "market_cap": info.get("marketCap"),
        "float_shares": info.get("floatShares"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg_ratio": info.get("pegRatio"),
        "price_to_book": info.get("priceToBook"),
        "price_to_sales_trailing_12m": info.get("priceToSalesTrailing12Months"),
        "enterprise_to_revenue": info.get("enterpriseToRevenue"),
        "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
        "beta": info.get("beta"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "quick_ratio": info.get("quickRatio"),
        "profit_margins": info.get("profitMargins"),
        "gross_margins": info.get("grossMargins"),
        "operating_margins": info.get("operatingMargins"),
        "ebitda_margins": info.get("ebitdaMargins"),
        "return_on_assets": info.get("returnOnAssets"),
        "return_on_equity": info.get("returnOnEquity"),
        "earnings_growth": info.get("earningsGrowth"),
        "revenue_growth": info.get("revenueGrowth"),
        "free_cashflow": info.get("freeCashflow"),
        "operating_cashflow": info.get("operatingCashflow"),
    }

    return row


def fetch_symbol_fundamentals_fmp(symbol: str, client: FMPClient) -> dict[str, float | str | datetime | None]:
    profile = client.get_profile(symbol)
    ratios = client.get_ratios_ttm(symbol)
    metrics = client.get_key_metrics_ttm(symbol)

    row: dict[str, float | str | datetime | None] = {
        "symbol": symbol,
        "asof_datetime": datetime.now(timezone.utc),
        "market_cap": _first_valid(profile.get("marketCap"), profile.get("mktCap"), metrics.get("marketCapTTM")),
        "float_shares": _first_valid(profile.get("floatShares"), profile.get("freeFloat")),
        "shares_outstanding": _first_valid(profile.get("sharesOutstanding"), profile.get("sharesOutStanding")),
        "trailing_pe": _first_valid(profile.get("pe"), ratios.get("peRatioTTM"), metrics.get("peRatioTTM")),
        "forward_pe": _first_valid(profile.get("forwardPE"), ratios.get("forwardPE")),
        "peg_ratio": _first_valid(profile.get("pegRatio"), ratios.get("pegRatioTTM"), metrics.get("pegRatioTTM")),
        "price_to_book": _first_valid(profile.get("priceToBook"), ratios.get("priceToBookRatioTTM")),
        "price_to_sales_trailing_12m": _first_valid(
            profile.get("priceToSalesTrailing12Months"),
            ratios.get("priceToSalesRatioTTM"),
            metrics.get("priceToSalesRatioTTM"),
        ),
        "enterprise_to_revenue": _first_valid(profile.get("enterpriseToRevenue"), metrics.get("enterpriseValueOverRevenueTTM")),
        "enterprise_to_ebitda": _first_valid(profile.get("enterpriseToEbitda"), metrics.get("enterpriseValueOverEBITDATTM")),
        "beta": profile.get("beta"),
        "debt_to_equity": _first_valid(ratios.get("debtEquityRatioTTM"), metrics.get("debtToEquityTTM")),
        "current_ratio": _first_valid(ratios.get("currentRatioTTM"), metrics.get("currentRatioTTM")),
        "quick_ratio": _first_valid(ratios.get("quickRatioTTM"), metrics.get("quickRatioTTM")),
        "profit_margins": _first_valid(ratios.get("netProfitMarginTTM"), profile.get("profitMargins")),
        "gross_margins": _first_valid(ratios.get("grossProfitMarginTTM"), profile.get("grossMargins")),
        "operating_margins": _first_valid(ratios.get("operatingProfitMarginTTM"), profile.get("operatingMargins")),
        "ebitda_margins": _first_valid(ratios.get("ebitdaMarginTTM"), profile.get("ebitdaMargins")),
        "return_on_assets": _first_valid(ratios.get("returnOnAssetsTTM"), profile.get("returnOnAssets")),
        "return_on_equity": _first_valid(ratios.get("returnOnEquityTTM"), profile.get("returnOnEquity")),
        "earnings_growth": _first_valid(profile.get("earningsGrowth"), metrics.get("earningsGrowthTTM")),
        "revenue_growth": _first_valid(profile.get("revenueGrowth"), metrics.get("revenueGrowthTTM")),
        "free_cashflow": _first_valid(profile.get("freeCashflow"), metrics.get("freeCashFlowTTM")),
        "operating_cashflow": _first_valid(profile.get("operatingCashflow"), metrics.get("operatingCashFlowTTM")),
    }
    return row


def build_fundamental_table(
    symbols: list[str],
    cache_path: Path,
    refresh: bool = False,
    max_symbols: int | None = None,
    provider: Literal["auto", "fmp", "yfinance"] = "auto",
    fmp_api_key: str | None = None,
    fmp_base_url: str = "https://financialmodelingprep.com/stable",
    request_workers: int = 8,
) -> pd.DataFrame:
    """Fetch fundamentals for symbols and persist a denormalized table."""
    if cache_path.exists() and not refresh:
        cached = pd.read_parquet(cache_path)
    else:
        cached = pd.DataFrame(columns=["symbol", "asof_datetime", *FUNDAMENTAL_COLUMNS])

    existing = set(cached["symbol"].unique()) if not cached.empty else set()
    todo = [s for s in symbols if refresh or s not in existing]
    if max_symbols is not None:
        todo = todo[:max_symbols]

    resolved_provider = provider
    if provider == "auto":
        resolved_provider = "fmp" if fmp_api_key else "yfinance"

    rows: list[dict[str, Any]] = []
    if resolved_provider == "fmp" and fmp_api_key:
        client = FMPClient(api_key=fmp_api_key, base_url=fmp_base_url)
        with ThreadPoolExecutor(max_workers=request_workers) as pool:
            futures = {pool.submit(fetch_symbol_fundamentals_fmp, symbol, client): symbol for symbol in todo}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception:
                    continue
                rows.append(row)
    else:
        with ThreadPoolExecutor(max_workers=request_workers) as pool:
            futures = {pool.submit(fetch_symbol_fundamentals_yfinance, symbol): symbol for symbol in todo}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception:
                    continue
                rows.append(row)

    if rows:
        new = pd.DataFrame(rows)
        if cached.empty:
            frame = new
        else:
            frame = pd.concat([cached, new], ignore_index=True)
        frame = frame.sort_values("asof_datetime").drop_duplicates(subset=["symbol"], keep="last")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)
        return frame

    return cached


def load_politician_trades(csv_path: Path) -> pd.DataFrame:
    """Load politician trade data from a normalized CSV.

    Expected fields: symbol, trade_date, amount_usd, side, politician (optional).
    """
    if not csv_path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(csv_path)
    required = {"symbol", "trade_date", "amount_usd", "side"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Politician trade CSV missing columns: {required - set(frame.columns)}")

    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
    out["trade_date"] = pd.to_datetime(out["trade_date"], utc=True, errors="coerce")
    out["amount_usd"] = pd.to_numeric(out["amount_usd"], errors="coerce").fillna(0.0)
    out["side"] = out["side"].astype(str).str.lower()
    out = out.dropna(subset=["trade_date", "symbol"])
    return out


def engineer_politician_features(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades

    t = trades.copy()
    t["signed_amount"] = np.where(t["side"].str.startswith("b"), t["amount_usd"], -t["amount_usd"])
    grouped = (
        t.groupby(["symbol", "trade_date"], as_index=False)
        .agg(
            politician_signed_flow=("signed_amount", "sum"),
            politician_abs_flow=("amount_usd", "sum"),
            politician_trade_count=("amount_usd", "count"),
        )
        .sort_values(["symbol", "trade_date"])
    )

    per_symbol: list[pd.DataFrame] = []
    for _, g in grouped.groupby("symbol", sort=False):
        x = g.copy()
        x["politician_flow_7d"] = x["politician_signed_flow"].rolling(7, min_periods=1).sum()
        x["politician_flow_30d"] = x["politician_signed_flow"].rolling(30, min_periods=1).sum()
        x["politician_count_30d"] = x["politician_trade_count"].rolling(30, min_periods=1).sum()
        per_symbol.append(x)

    return pd.concat(per_symbol, ignore_index=True)


def merge_external_features(
    price_frame: pd.DataFrame,
    fundamental_table: pd.DataFrame | None = None,
    politician_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if price_frame.empty:
        return price_frame

    out = price_frame.copy().sort_values(["symbol", "datetime"]).reset_index(drop=True)

    if fundamental_table is not None and not fundamental_table.empty:
        f = fundamental_table.copy()
        for col in FUNDAMENTAL_COLUMNS:
            if col not in f.columns:
                f[col] = np.nan
        out = out.merge(f[["symbol", *FUNDAMENTAL_COLUMNS]], on="symbol", how="left")

    if politician_features is not None and not politician_features.empty:
        p = politician_features.copy().sort_values(["symbol", "trade_date"])
        merged: list[pd.DataFrame] = []
        for symbol, g in out.groupby("symbol", sort=False):
            p_symbol = p.loc[p["symbol"] == symbol]
            if p_symbol.empty:
                merged.append(g)
                continue
            aligned = pd.merge_asof(
                g.sort_values("datetime"),
                p_symbol,
                left_on="datetime",
                right_on="trade_date",
                direction="backward",
            )
            merged.append(aligned.drop(columns=["trade_date"], errors="ignore"))
        out = pd.concat(merged, ignore_index=True)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out
