from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import httpx
import numpy as np
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential


FRED_BASE_URL = "https://api.stlouisfed.org/fred"
DEFAULT_FRED_SERIES: dict[str, str] = {
    "DGS10": "macro_fred_dgs10",
    "DGS2": "macro_fred_dgs2",
    "DGS3MO": "macro_fred_dgs3m",
    "T10Y2Y": "macro_fred_curve_10y2y",
    "T10Y3M": "macro_fred_curve_10y3m",
    "VIXCLS": "macro_fred_vix",
    "DFF": "macro_fred_fed_funds",
    "BAMLH0A0HYM2": "macro_fred_hy_oas",
    "CPIAUCSL": "macro_fred_cpi",
    "UNRATE": "macro_fred_unrate",
    "PAYEMS": "macro_fred_payems",
    "INDPRO": "macro_fred_indpro",
    "M2SL": "macro_fred_m2",
    "WALCL": "macro_fred_walcl",
}

DEFAULT_YF_PROXIES: dict[str, str] = {
    "SPY": "macro_proxy_spy",
    "QQQ": "macro_proxy_qqq",
    "IWM": "macro_proxy_iwm",
    "TLT": "macro_proxy_tlt",
    "HYG": "macro_proxy_hyg",
    "LQD": "macro_proxy_lqd",
    "GLD": "macro_proxy_gld",
    "USO": "macro_proxy_uso",
    "UUP": "macro_proxy_uup",
    "^VIX": "macro_proxy_vix",
    "^TNX": "macro_proxy_tnx",
    "^IRX": "macro_proxy_irx",
}

DEFAULT_FMP_ECONOMIC_INDICATORS: dict[str, str] = {
    "GDP": "macro_fmp_gdp",
    "realGDP": "macro_fmp_real_gdp",
    "nominalPotentialGDP": "macro_fmp_nominal_potential_gdp",
    "CPI": "macro_fmp_cpi",
    "inflationRate": "macro_fmp_inflation_rate",
    "federalFunds": "macro_fmp_fed_funds",
    "unemploymentRate": "macro_fmp_unemployment_rate",
    "retailSales": "macro_fmp_retail_sales",
    "consumerSentiment": "macro_fmp_consumer_sentiment",
    "durableGoods": "macro_fmp_durable_goods",
    "industrialProductionTotalIndex": "macro_fmp_industrial_production_index",
}

DEFAULT_FMP_SECTORS: tuple[str, ...] = (
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
)


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(4))
def _http_get_json(url: str, params: dict[str, Any], timeout: float = 30.0) -> Any:
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def _http_get_json_once(url: str, params: dict[str, Any], timeout: float = 30.0) -> Any:
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def _fmp_payload_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                rows.append(item)
            elif isinstance(item, str):
                rows.append({"name": item})
        return rows
    if isinstance(payload, dict):
        for key in ("historical", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def _slugify_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window=window, min_periods=max(5, window // 3)).std().replace(0, np.nan)
    return (series - mean) / std


def _consecutive_true_counts(mask: pd.Series) -> pd.Series:
    out = np.zeros(len(mask), dtype=float)
    run = 0
    for i, flag in enumerate(mask.astype(bool).to_numpy()):
        if flag:
            run += 1
        else:
            run = 0
        out[i] = float(run)
    return pd.Series(out, index=mask.index, dtype=float)


def _parse_numeric_maybe(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        out = float(value)
        if np.isnan(out) or np.isinf(out):
            return None
        return out

    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "null", "n/a"}:
        return None

    pct = text.endswith("%")
    if pct:
        text = text[:-1]

    mult = 1.0
    if text and text[-1].upper() in {"K", "M", "B", "T"}:
        suffix = text[-1].upper()
        text = text[:-1]
        if suffix == "K":
            mult = 1e3
        elif suffix == "M":
            mult = 1e6
        elif suffix == "B":
            mult = 1e9
        elif suffix == "T":
            mult = 1e12

    try:
        out = float(text) * mult
    except Exception:
        return None
    if pct:
        out /= 100.0
    if np.isnan(out) or np.isinf(out):
        return None
    return out


def _fetch_fred_series(
    api_key: str,
    series_id: str,
    observation_start: str,
    observation_end: str,
) -> pd.DataFrame:
    payload = _http_get_json(
        f"{FRED_BASE_URL}/series/observations",
        params={
            "api_key": api_key,
            "series_id": series_id,
            "observation_start": observation_start,
            "observation_end": observation_end,
            "sort_order": "asc",
            "file_type": "json",
        },
    )
    observations = payload.get("observations", []) if isinstance(payload, dict) else []
    if not observations:
        return pd.DataFrame(columns=["datetime", series_id])

    frame = pd.DataFrame(observations)
    if "date" not in frame.columns or "value" not in frame.columns:
        return pd.DataFrame(columns=["datetime", series_id])

    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(frame["date"], utc=True, errors="coerce"),
            series_id: pd.to_numeric(frame["value"].replace(".", np.nan), errors="coerce"),
        }
    )
    out = out.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return out


def fetch_fred_macro_table(
    api_key: str | None,
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
    series_map: dict[str, str] | None = None,
    request_workers: int = 8,
) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()

    mapping = series_map or DEFAULT_FRED_SERIES
    if not mapping:
        return pd.DataFrame()

    start = pd.to_datetime(start_datetime, utc=True, errors="coerce")
    end = pd.to_datetime(end_datetime, utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame()

    series_ids = list(mapping.keys())
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, int(request_workers))) as pool:
        futures = {
            pool.submit(
                _fetch_fred_series,
                api_key,
                series_id,
                start.date().isoformat(),
                end.date().isoformat(),
            ): series_id
            for series_id in series_ids
        }
        for future in as_completed(futures):
            series_id = futures[future]
            try:
                frame = future.result()
            except Exception:
                continue
            if frame.empty:
                continue
            renamed = frame.rename(columns={series_id: mapping.get(series_id, f"macro_fred_{series_id.lower()}")})
            frames.append(renamed)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="datetime", how="outer")
    return merged.sort_values("datetime").reset_index(drop=True)


def fetch_fmp_treasury_rates_table(
    api_key: str | None,
    base_url: str = "https://financialmodelingprep.com/stable",
) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()

    payload = _http_get_json(
        f"{base_url.rstrip('/')}/treasury-rates",
        params={"apikey": api_key, "limit": 5000},
    )
    rows = [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame.get("date"), utc=True, errors="coerce")
    frame = frame.dropna(subset=["datetime"]).copy()
    rename_map = {
        "month1": "macro_tsy_1m",
        "month2": "macro_tsy_2m",
        "month3": "macro_tsy_3m",
        "month6": "macro_tsy_6m",
        "year1": "macro_tsy_1y",
        "year2": "macro_tsy_2y",
        "year3": "macro_tsy_3y",
        "year5": "macro_tsy_5y",
        "year7": "macro_tsy_7y",
        "year10": "macro_tsy_10y",
        "year20": "macro_tsy_20y",
        "year30": "macro_tsy_30y",
    }
    available = [k for k in rename_map if k in frame.columns]
    out = frame[["datetime", *available]].rename(columns=rename_map)
    for col in [c for c in out.columns if c != "datetime"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)


def fetch_fmp_economic_calendar_table(
    api_key: str | None,
    base_url: str = "https://financialmodelingprep.com/stable",
) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()

    payload = _http_get_json(
        f"{base_url.rstrip('/')}/economic-calendar",
        params={"apikey": api_key, "limit": 12000},
    )
    rows = [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame.get("date"), utc=True, errors="coerce")
    frame = frame.dropna(subset=["datetime"]).copy()
    return frame.sort_values("datetime").reset_index(drop=True)


def fetch_fmp_market_risk_premium_table(
    api_key: str | None,
    base_url: str = "https://financialmodelingprep.com/stable",
) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    try:
        payload = _http_get_json_once(
            f"{base_url.rstrip('/')}/market-risk-premium",
            params={"apikey": api_key},
        )
    except Exception:
        return pd.DataFrame()

    rows = _fmp_payload_rows(payload)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame.get("date"), utc=True, errors="coerce")
    frame = frame.dropna(subset=["datetime"]).copy()
    if frame.empty:
        return pd.DataFrame()

    col_map = {
        "marketRiskPremium": "macro_fmp_market_risk_premium",
        "riskFreeRate": "macro_fmp_risk_free_rate",
        "expectedMarketReturn": "macro_fmp_expected_market_return",
        "countryRiskPremium": "macro_fmp_country_risk_premium",
    }
    existing = [c for c in col_map if c in frame.columns]
    out = frame[["datetime", *existing]].rename(columns=col_map)
    for col in out.columns:
        if col == "datetime":
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)


def _parse_fmp_indicator_value(row: dict[str, Any]) -> float | None:
    preferred = (
        "value",
        "close",
        "price",
        "indicator",
        "marketRiskPremium",
        "riskFreeRate",
        "expectedMarketReturn",
    )
    for key in preferred:
        if key in row:
            parsed = _parse_numeric_maybe(row.get(key))
            if parsed is not None:
                return parsed

    for key, value in row.items():
        if str(key).lower() in {"date", "symbol", "name", "country"}:
            continue
        parsed = _parse_numeric_maybe(value)
        if parsed is not None:
            return parsed
    return None


def fetch_fmp_economic_indicators_table(
    api_key: str | None,
    base_url: str = "https://financialmodelingprep.com/stable",
    indicator_map: dict[str, str] | None = None,
    request_workers: int = 8,
) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()

    mapping = indicator_map or DEFAULT_FMP_ECONOMIC_INDICATORS
    if not mapping:
        return pd.DataFrame()

    def _fetch_indicator(name: str, out_col: str) -> pd.DataFrame:
        try:
            payload = _http_get_json_once(
                f"{base_url.rstrip('/')}/economic-indicators",
                params={"apikey": api_key, "name": name},
            )
        except Exception:
            return pd.DataFrame(columns=["datetime", out_col])

        rows = _fmp_payload_rows(payload)
        if not rows:
            return pd.DataFrame(columns=["datetime", out_col])

        frame = pd.DataFrame(rows)
        dt = pd.to_datetime(
            frame.get("date", frame.get("datetime", frame.get("period"))),
            utc=True,
            errors="coerce",
        )
        if dt.isna().all():
            return pd.DataFrame(columns=["datetime", out_col])

        values = frame.apply(lambda r: _parse_fmp_indicator_value(dict(r)), axis=1)
        out = pd.DataFrame({"datetime": dt, out_col: pd.to_numeric(values, errors="coerce")})
        out = out.dropna(subset=["datetime"]).sort_values("datetime")
        out = out.drop_duplicates(subset=["datetime"], keep="last")
        return out.reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(request_workers), len(mapping)))) as pool:
        futures = {
            pool.submit(_fetch_indicator, indicator_name, out_col): out_col
            for indicator_name, out_col in mapping.items()
        }
        for future in as_completed(futures):
            try:
                frame = future.result()
            except Exception:
                continue
            if not frame.empty:
                frames.append(frame)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="datetime", how="outer")
    return merged.sort_values("datetime").reset_index(drop=True)


def _fetch_fmp_available_sectors(
    api_key: str | None,
    base_url: str = "https://financialmodelingprep.com/stable",
) -> list[str]:
    if not api_key:
        return list(DEFAULT_FMP_SECTORS)
    try:
        payload = _http_get_json_once(
            f"{base_url.rstrip('/')}/available-sectors",
            params={"apikey": api_key},
        )
    except Exception:
        return list(DEFAULT_FMP_SECTORS)

    sectors: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                sectors.append(item.strip())
            elif isinstance(item, dict):
                candidate = item.get("sector") or item.get("name") or item.get("value")
                if candidate is not None:
                    sectors.append(str(candidate).strip())
    elif isinstance(payload, dict):
        rows = _fmp_payload_rows(payload)
        for row in rows:
            candidate = row.get("sector") or row.get("name") or row.get("value")
            if candidate is not None:
                sectors.append(str(candidate).strip())

    cleaned = [s for s in sectors if s]
    return cleaned or list(DEFAULT_FMP_SECTORS)


def fetch_fmp_sector_history_tables(
    api_key: str | None,
    base_url: str = "https://financialmodelingprep.com/stable",
    request_workers: int = 8,
) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    sectors = _fetch_fmp_available_sectors(api_key=api_key, base_url=base_url)
    if not sectors:
        return pd.DataFrame()

    def _fetch_sector_perf(sector: str) -> pd.DataFrame:
        try:
            payload = _http_get_json_once(
                f"{base_url.rstrip('/')}/historical-sector-performance",
                params={"apikey": api_key, "sector": sector},
            )
        except Exception:
            return pd.DataFrame()
        rows = _fmp_payload_rows(payload)
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        dt = pd.to_datetime(frame.get("date"), utc=True, errors="coerce")
        raw_value = frame.get("changesPercentage", frame.get("changePercentage", frame.get("value")))
        if raw_value is None:
            raw_value = pd.Series(index=frame.index, dtype=object)
        value = pd.Series(raw_value, index=frame.index).map(_parse_numeric_maybe)
        col = f"macro_sector_perf_{_slugify_token(sector)}"
        out = pd.DataFrame({"datetime": dt, col: value})
        return out.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"], keep="last")

    def _fetch_sector_pe(sector: str) -> pd.DataFrame:
        try:
            payload = _http_get_json_once(
                f"{base_url.rstrip('/')}/historical-sector-pe",
                params={"apikey": api_key, "sector": sector},
            )
        except Exception:
            return pd.DataFrame()
        rows = _fmp_payload_rows(payload)
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        dt = pd.to_datetime(frame.get("date"), utc=True, errors="coerce")
        raw_value = frame.get("pe", frame.get("value"))
        if raw_value is None:
            raw_value = pd.Series(index=frame.index, dtype=object)
        value = pd.Series(raw_value, index=frame.index).map(_parse_numeric_maybe)
        col = f"macro_sector_pe_{_slugify_token(sector)}"
        out = pd.DataFrame({"datetime": dt, col: value})
        return out.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"], keep="last")

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(request_workers), 10))) as pool:
        perf_futures = {pool.submit(_fetch_sector_perf, sector): sector for sector in sectors}
        pe_futures = {pool.submit(_fetch_sector_pe, sector): sector for sector in sectors}
        for future in as_completed({**perf_futures, **pe_futures}):
            try:
                frame = future.result()
            except Exception:
                continue
            if not frame.empty:
                frames.append(frame)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="datetime", how="outer")
    return merged.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)


def fetch_yfinance_macro_proxies(
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
    ticker_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    mapping = ticker_map or DEFAULT_YF_PROXIES
    if not mapping:
        return pd.DataFrame()

    start = pd.to_datetime(start_datetime, utc=True, errors="coerce")
    end = pd.to_datetime(end_datetime, utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame()

    tickers = list(mapping.keys())
    raw = yf.download(
        tickers=" ".join(tickers),
        interval="1d",
        start=start.date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        group_by="ticker",
        auto_adjust=True,
        threads=False,
        progress=False,
    )
    if raw.empty:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(raw.index, utc=True, errors="coerce")
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        for ticker, col_name in mapping.items():
            if ticker not in lvl0:
                continue
            close = pd.to_numeric(raw[(ticker, "Close")], errors="coerce")
            out[col_name] = close.to_numpy()
    else:
        # Single ticker fallback.
        ticker = tickers[0]
        close = pd.to_numeric(raw["Close"], errors="coerce")
        out[mapping[ticker]] = close.to_numpy()

    out = out.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return out


def _build_macro_event_features(
    calendar: pd.DataFrame,
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
) -> pd.DataFrame:
    if calendar.empty:
        return pd.DataFrame(columns=["datetime"])

    start = pd.to_datetime(start_datetime, utc=True, errors="coerce").floor("D")
    end = pd.to_datetime(end_datetime, utc=True, errors="coerce").ceil("D")
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame(columns=["datetime"])

    work = calendar.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
    work = work.dropna(subset=["datetime"]).copy()
    if work.empty:
        return pd.DataFrame(columns=["datetime"])

    work["date"] = work["datetime"].dt.floor("D")
    country = work.get("country", pd.Series(index=work.index, dtype=object)).astype(str).str.upper()
    work = work.loc[country.isin({"US", "USA", "UNITED STATES"})].copy()
    if work.empty:
        return pd.DataFrame(columns=["datetime"])

    event_text = work.get("event", pd.Series(index=work.index, dtype=object)).astype(str).str.lower()
    major_regex = re.compile(r"(cpi|pce|fomc|fed|interest rate|non farm|payroll|gdp|unemployment|retail sales|pmi)")
    work["is_major"] = event_text.str.contains(major_regex, regex=True)

    impact_text = work.get("impact", pd.Series(index=work.index, dtype=object)).astype(str).str.lower()
    impact_score = pd.Series(1.0, index=work.index, dtype=float)
    impact_score.loc[impact_text.str.contains("high")] = 3.0
    impact_score.loc[impact_text.str.contains("med")] = 2.0
    work["impact_score"] = impact_score

    actual = work.get("actual", pd.Series(index=work.index, dtype=object)).map(_parse_numeric_maybe)
    estimate = work.get("estimate", pd.Series(index=work.index, dtype=object)).map(_parse_numeric_maybe)
    previous = work.get("previous", pd.Series(index=work.index, dtype=object)).map(_parse_numeric_maybe)
    base = estimate.where(pd.notna(estimate), previous)
    eps = 1e-9
    work["surprise_raw"] = np.where(
        pd.notna(actual) & pd.notna(base) & (np.abs(base) > eps),
        (actual - base) / np.abs(base),
        np.nan,
    )
    work["abs_surprise_raw"] = np.abs(work["surprise_raw"])

    grouped = (
        work.groupby("date", as_index=False)
        .agg(
            macro_us_event_count=("event", "count"),
            macro_us_high_impact_count=("impact_score", lambda s: float((pd.to_numeric(s, errors="coerce") >= 3).sum())),
            macro_us_major_event_count=("is_major", lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
            macro_us_event_surprise_mean=("surprise_raw", "mean"),
            macro_us_event_abs_surprise_mean=("abs_surprise_raw", "mean"),
        )
        .rename(columns={"date": "datetime"})
    )
    grouped["datetime"] = pd.to_datetime(grouped["datetime"], utc=True, errors="coerce")
    grouped = grouped.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    daily = pd.DataFrame({"datetime": pd.date_range(start=start, end=end, freq="D", tz="UTC")})
    out = daily.merge(grouped, on="datetime", how="left")
    for col in ("macro_us_event_count", "macro_us_high_impact_count", "macro_us_major_event_count"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["macro_us_event_surprise_mean"] = pd.to_numeric(out["macro_us_event_surprise_mean"], errors="coerce")
    out["macro_us_event_abs_surprise_mean"] = pd.to_numeric(out["macro_us_event_abs_surprise_mean"], errors="coerce")

    out["macro_us_event_count_5d"] = out["macro_us_event_count"].rolling(5, min_periods=1).sum()
    out["macro_us_high_impact_count_5d"] = out["macro_us_high_impact_count"].rolling(5, min_periods=1).sum()
    out["macro_us_major_event_count_5d"] = out["macro_us_major_event_count"].rolling(5, min_periods=1).sum()
    out["macro_us_abs_surprise_5d"] = out["macro_us_event_abs_surprise_mean"].rolling(5, min_periods=1).mean()
    out["macro_us_abs_surprise_21d"] = out["macro_us_event_abs_surprise_mean"].rolling(21, min_periods=3).mean()

    major_flag = out["macro_us_major_event_count"] > 0
    last_major = out["datetime"].where(major_flag).ffill()
    next_major = out["datetime"].where(major_flag).bfill()
    out["macro_days_since_us_major_event"] = (out["datetime"] - last_major).dt.days.astype(float)
    out["macro_days_to_next_us_major_event"] = (next_major - out["datetime"]).dt.days.astype(float)

    out["macro_days_since_us_major_event"] = out["macro_days_since_us_major_event"].fillna(999.0).clip(lower=0, upper=999)
    out["macro_days_to_next_us_major_event"] = out["macro_days_to_next_us_major_event"].fillna(999.0).clip(lower=0, upper=999)
    return out


def _derive_macro_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.copy().sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
    numeric_cols = [c for c in out.columns if c != "datetime"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    y10 = out.get("macro_fred_dgs10")
    if y10 is None:
        y10 = out.get("macro_tsy_10y")
    y2 = out.get("macro_fred_dgs2")
    if y2 is None:
        y2 = out.get("macro_tsy_2y")
    y3m = out.get("macro_fred_dgs3m")
    if y3m is None:
        y3m = out.get("macro_tsy_3m")
    y5 = out.get("macro_tsy_5y")

    if y10 is not None and y2 is not None:
        out["macro_curve_10y_2y"] = y10 - y2
        out["macro_curve_inversion_10y_2y"] = (out["macro_curve_10y_2y"] < 0).astype(float)
        out["macro_curve_inversion_days_10y_2y"] = _consecutive_true_counts(out["macro_curve_10y_2y"] < 0)
    if y10 is not None and y3m is not None:
        out["macro_curve_10y_3m"] = y10 - y3m
        out["macro_curve_inversion_10y_3m"] = (out["macro_curve_10y_3m"] < 0).astype(float)
    if y10 is not None and y2 is not None and y5 is not None:
        out["macro_curve_curvature_5s10s2s"] = (2.0 * y5) - y2 - y10

    if "macro_proxy_spy" in out.columns and "macro_proxy_tlt" in out.columns:
        out["macro_equity_bond_ratio"] = out["macro_proxy_spy"] / out["macro_proxy_tlt"].replace(0, np.nan)
    if "macro_proxy_iwm" in out.columns and "macro_proxy_qqq" in out.columns:
        out["macro_small_large_ratio"] = out["macro_proxy_iwm"] / out["macro_proxy_qqq"].replace(0, np.nan)
    if "macro_proxy_hyg" in out.columns and "macro_proxy_lqd" in out.columns:
        out["macro_credit_risk_ratio"] = out["macro_proxy_hyg"] / out["macro_proxy_lqd"].replace(0, np.nan)
    if "macro_fmp_market_risk_premium" in out.columns and "macro_fmp_risk_free_rate" in out.columns:
        out["macro_equity_risk_premium_total"] = (
            out["macro_fmp_market_risk_premium"] + out["macro_fmp_risk_free_rate"]
        )
    if "macro_fmp_market_risk_premium" in out.columns and "macro_proxy_spy" in out.columns:
        spy_mom_20 = pd.to_numeric(out["macro_proxy_spy"], errors="coerce").pct_change(20)
        out["macro_erp_vs_equity_momentum"] = out["macro_fmp_market_risk_premium"] - spy_mom_20

    sector_perf_cols = [c for c in out.columns if c.startswith("macro_sector_perf_")]
    if sector_perf_cols:
        perf = pd.concat([pd.to_numeric(out[c], errors="coerce") for c in sector_perf_cols], axis=1)
        out["macro_sector_perf_mean"] = perf.mean(axis=1, skipna=True)
        out["macro_sector_perf_std"] = perf.std(axis=1, skipna=True)
        out["macro_sector_perf_spread"] = perf.max(axis=1, skipna=True) - perf.min(axis=1, skipna=True)
        out["macro_sector_positive_share"] = (perf > 0).mean(axis=1, skipna=True)

    sector_pe_cols = [c for c in out.columns if c.startswith("macro_sector_pe_")]
    if sector_pe_cols:
        pe = pd.concat([pd.to_numeric(out[c], errors="coerce") for c in sector_pe_cols], axis=1)
        out["macro_sector_pe_mean"] = pe.mean(axis=1, skipna=True)
        out["macro_sector_pe_std"] = pe.std(axis=1, skipna=True)
        out["macro_sector_pe_spread"] = pe.max(axis=1, skipna=True) - pe.min(axis=1, skipna=True)

    if {"macro_sector_perf_mean", "macro_sector_pe_mean"}.issubset(out.columns):
        out["macro_sector_perf_to_pe"] = out["macro_sector_perf_mean"] / out["macro_sector_pe_mean"].replace(0, np.nan)
    if {"macro_sector_perf_std", "macro_sector_pe_std"}.issubset(out.columns):
        out["macro_sector_dispersion_blend"] = out[["macro_sector_perf_std", "macro_sector_pe_std"]].mean(
            axis=1,
            skipna=True,
        )

    proxy_price_cols = [
        c
        for c in out.columns
        if c.startswith("macro_proxy_")
        and c not in {"macro_proxy_tnx", "macro_proxy_irx"}
    ]
    for col in proxy_price_cols:
        out[f"{col}_ret_1"] = out[col].pct_change(1)
        out[f"{col}_ret_5"] = out[col].pct_change(5)
        out[f"{col}_ret_20"] = out[col].pct_change(20)

    macro_state_cols = [
        c
        for c in (
            "macro_fred_vix",
            "macro_proxy_vix",
            "macro_fred_hy_oas",
            "macro_curve_10y_2y",
            "macro_curve_10y_3m",
            "macro_proxy_spy_ret_5",
            "macro_sector_perf_std",
            "macro_sector_positive_share",
            "macro_fmp_market_risk_premium",
        )
        if c in out.columns
    ]
    z_components: list[pd.Series] = []
    for col in macro_state_cols:
        s = pd.to_numeric(out[col], errors="coerce")
        if col in {
            "macro_curve_10y_2y",
            "macro_curve_10y_3m",
            "macro_proxy_spy_ret_5",
            "macro_sector_positive_share",
        }:
            s = -s
        z_components.append(_rolling_zscore(s, 63))
    if z_components:
        joined = pd.concat(z_components, axis=1)
        out["macro_risk_off_score"] = joined.mean(axis=1, skipna=True)
        out["macro_regime_risk_off"] = (out["macro_risk_off_score"] > 0.5).astype(float)
        out["macro_regime_high_stress"] = (out["macro_risk_off_score"] > 1.0).astype(float)

    surprise_bases = [
        c
        for c in (
            "macro_fred_cpi",
            "macro_fred_unrate",
            "macro_fred_payems",
            "macro_fred_indpro",
            "macro_fred_m2",
            "macro_fred_walcl",
        )
        if c in out.columns
    ]
    surprise_cols: list[str] = []
    for col in surprise_bases:
        delta = pd.to_numeric(out[col], errors="coerce").diff(1)
        expected = delta.rolling(12, min_periods=4).median().shift(1)
        shock = delta - expected
        scale = delta.rolling(24, min_periods=6).std().shift(1).replace(0, np.nan)
        out[f"{col}_surprise"] = shock / scale
        surprise_cols.append(f"{col}_surprise")

    if surprise_cols:
        out["macro_surprise_index"] = pd.concat([out[c] for c in surprise_cols], axis=1).mean(axis=1, skipna=True)
        out["macro_surprise_abs_index"] = pd.concat([out[c].abs() for c in surprise_cols], axis=1).mean(axis=1, skipna=True)

    for col in ("macro_risk_off_score", "macro_surprise_index", "macro_surprise_abs_index"):
        if col in out.columns:
            out[f"{col}_z_63"] = _rolling_zscore(pd.to_numeric(out[col], errors="coerce"), 63)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_macro_feature_table(
    cache_path: Path,
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
    refresh: bool = False,
    fmp_api_key: str | None = None,
    fmp_base_url: str = "https://financialmodelingprep.com/stable",
    fred_api_key: str | None = None,
    request_workers: int = 8,
) -> pd.DataFrame:
    start = pd.to_datetime(start_datetime, utc=True, errors="coerce")
    end = pd.to_datetime(end_datetime, utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame()

    start = start.floor("D")
    end = end.ceil("D")
    if end < start:
        start, end = end, start

    if cache_path.exists() and not refresh:
        try:
            cached = pd.read_parquet(cache_path)
            cached["datetime"] = pd.to_datetime(cached["datetime"], utc=True, errors="coerce")
            cached = cached.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            if not cached.empty:
                cached_start = cached["datetime"].min()
                cached_end = cached["datetime"].max()
                if pd.notna(cached_start) and pd.notna(cached_end) and cached_start <= start and cached_end >= end:
                    return cached.loc[(cached["datetime"] >= start) & (cached["datetime"] <= end)].reset_index(drop=True)
        except Exception:
            pass

    fetch_start = start - pd.Timedelta(days=400)
    fetch_end = end
    base = pd.DataFrame({"datetime": pd.date_range(start=fetch_start, end=fetch_end, freq="D", tz="UTC")})

    frames: list[pd.DataFrame] = [base]

    fred = fetch_fred_macro_table(
        api_key=fred_api_key,
        start_datetime=fetch_start,
        end_datetime=fetch_end,
        request_workers=request_workers,
    )
    if not fred.empty:
        frames.append(fred)

    fmp_tsy = fetch_fmp_treasury_rates_table(api_key=fmp_api_key, base_url=fmp_base_url)
    if not fmp_tsy.empty:
        frames.append(fmp_tsy)

    fmp_risk_premium = fetch_fmp_market_risk_premium_table(api_key=fmp_api_key, base_url=fmp_base_url)
    if not fmp_risk_premium.empty:
        frames.append(fmp_risk_premium)

    fmp_econ = fetch_fmp_economic_indicators_table(
        api_key=fmp_api_key,
        base_url=fmp_base_url,
        request_workers=request_workers,
    )
    if not fmp_econ.empty:
        frames.append(fmp_econ)

    fmp_sector_hist = fetch_fmp_sector_history_tables(
        api_key=fmp_api_key,
        base_url=fmp_base_url,
        request_workers=request_workers,
    )
    if not fmp_sector_hist.empty:
        frames.append(fmp_sector_hist)

    proxies = fetch_yfinance_macro_proxies(start_datetime=fetch_start, end_datetime=fetch_end)
    if not proxies.empty:
        frames.append(proxies)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="datetime", how="left")
    merged = merged.sort_values("datetime").reset_index(drop=True)

    # Forward-fill slow macro series to daily; avoid leaking too far if source is stale.
    macro_cols = [c for c in merged.columns if c != "datetime"]
    merged[macro_cols] = merged[macro_cols].ffill(limit=366)

    events = fetch_fmp_economic_calendar_table(api_key=fmp_api_key, base_url=fmp_base_url)
    event_features = _build_macro_event_features(events, start_datetime=fetch_start, end_datetime=fetch_end)
    if not event_features.empty:
        merged = merged.merge(event_features, on="datetime", how="left")

    derived = _derive_macro_features(merged)
    derived = derived.loc[(derived["datetime"] >= start) & (derived["datetime"] <= end)].reset_index(drop=True)
    if derived.empty:
        return derived

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        derived.to_parquet(cache_path, index=False)
    except Exception:
        pass
    return derived
