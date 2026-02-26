from __future__ import annotations

from io import StringIO
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from stonkmodel.data.fmp import FMPClient

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_SP100_URL = "https://en.wikipedia.org/wiki/S%26P_100"
GITHUB_SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

FMP_SP500_CONSTITUENTS_PATH = "/sp500_constituent"
FMP_SP100_PROXY_ETF_HOLDINGS_PATH = "/etf-holder/OEF"

HTTP_USER_AGENT = "StonkModel/1.0"
DEFAULT_MEMBERSHIP_START = pd.Timestamp("1900-01-01", tz="UTC")


def _normalize_symbol(symbol: str) -> str:
    # Yahoo-style symbols map dots to dashes (BRK.B -> BRK-B).
    return symbol.strip().upper().replace(".", "-")


def _is_equity_symbol(symbol: str) -> bool:
    return bool(symbol) and symbol[0].isalpha()


def _apply_limit(symbols: list[str], limit: int | None) -> list[str]:
    if limit is None:
        return symbols
    return symbols[:limit]


def _fetch_html(url: str) -> str:
    response = requests.get(url, timeout=30, headers={"User-Agent": HTTP_USER_AGENT})
    response.raise_for_status()
    return response.text


def _parse_symbols_from_tables(html: str, required_column: str = "symbol") -> list[str]:
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise RuntimeError("No HTML tables were parsed")

    for table in tables:
        for column in table.columns:
            if str(column).strip().lower() != required_column.lower():
                continue
            symbols = sorted({_normalize_symbol(s) for s in table[column].dropna().astype(str)})
            if symbols:
                return symbols

    raise RuntimeError(f"No `{required_column}` column found in parsed tables")


def _flatten_columns(columns: pd.Index) -> list[str]:
    if isinstance(columns, pd.MultiIndex):
        flat: list[str] = []
        for col in columns.to_flat_index():
            parts = [str(part).strip() for part in col if str(part).strip() and str(part).strip().lower() != "nan"]
            flat.append(" ".join(parts).strip())
        return flat
    return [str(col).strip() for col in columns]


def _pick_column(columns: list[str], include_terms: tuple[str, ...]) -> str | None:
    lowered = {c: c.lower() for c in columns}
    for column, lowered_name in lowered.items():
        if all(term in lowered_name for term in include_terms):
            return column
    return None


def _parse_sp500_change_events(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(columns=["effective_date", "added_symbol", "removed_symbol"])

    work = table.copy()
    work.columns = _flatten_columns(work.columns)

    date_col = _pick_column(list(work.columns), include_terms=("effective", "date"))
    added_col = _pick_column(list(work.columns), include_terms=("added", "ticker"))
    removed_col = _pick_column(list(work.columns), include_terms=("removed", "ticker"))
    if date_col is None or added_col is None or removed_col is None:
        return pd.DataFrame(columns=["effective_date", "added_symbol", "removed_symbol"])

    events = pd.DataFrame(
        {
            "effective_date": pd.to_datetime(work[date_col], utc=True, errors="coerce").dt.normalize(),
            "added_symbol": work[added_col].astype(str).map(_normalize_symbol),
            "removed_symbol": work[removed_col].astype(str).map(_normalize_symbol),
        }
    )
    for col in ["added_symbol", "removed_symbol"]:
        events.loc[events[col].isin({"", "NAN", "NONE"}), col] = pd.NA
    events = events.dropna(subset=["effective_date"]).copy()
    if events.empty:
        return pd.DataFrame(columns=["effective_date", "added_symbol", "removed_symbol"])
    return events


def _merge_membership_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    if intervals.empty:
        return intervals

    work = intervals.copy().sort_values(["symbol", "start_date", "end_date"]).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for symbol, group in work.groupby("symbol", sort=False):
        current_start = None
        current_end = None
        for _, row in group.iterrows():
            start = pd.to_datetime(row["start_date"], utc=True, errors="coerce")
            end = pd.to_datetime(row["end_date"], utc=True, errors="coerce")
            if pd.isna(start) or pd.isna(end) or end <= start:
                continue
            if current_start is None:
                current_start = start
                current_end = end
                continue
            if start <= current_end:
                current_end = max(current_end, end)
                continue
            rows.append({"symbol": symbol, "start_date": current_start, "end_date": current_end})
            current_start = start
            current_end = end

        if current_start is not None and current_end is not None and current_end > current_start:
            rows.append({"symbol": symbol, "start_date": current_start, "end_date": current_end})

    if not rows:
        return pd.DataFrame(columns=["symbol", "start_date", "end_date"])
    out = pd.DataFrame(rows)
    out["start_date"] = pd.to_datetime(out["start_date"], utc=True, errors="coerce")
    out["end_date"] = pd.to_datetime(out["end_date"], utc=True, errors="coerce")
    return out.sort_values(["symbol", "start_date", "end_date"]).reset_index(drop=True)


def reconstruct_sp500_membership_intervals(
    current_symbols: list[str],
    changes_table: pd.DataFrame,
    asof_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    end_ts = asof_end or (pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1))
    end_ts = pd.to_datetime(end_ts, utc=True, errors="coerce")
    if pd.isna(end_ts):
        end_ts = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)

    active: set[str] = {s for s in (_normalize_symbol(x) for x in current_symbols) if _is_equity_symbol(s)}
    active_interval_end: dict[str, pd.Timestamp] = {symbol: end_ts for symbol in active}
    events = _parse_sp500_change_events(changes_table).sort_values("effective_date", ascending=False, kind="stable")

    intervals: list[dict[str, object]] = []
    for effective_date, group in events.groupby("effective_date", sort=False):
        date_ts = pd.to_datetime(effective_date, utc=True, errors="coerce")
        if pd.isna(date_ts):
            continue

        for added_symbol in group["added_symbol"].dropna().astype(str):
            symbol = _normalize_symbol(added_symbol)
            if symbol not in active:
                continue
            interval_end = active_interval_end.get(symbol, end_ts)
            if interval_end > date_ts:
                intervals.append({"symbol": symbol, "start_date": date_ts, "end_date": interval_end})
            active.remove(symbol)
            active_interval_end.pop(symbol, None)

        for removed_symbol in group["removed_symbol"].dropna().astype(str):
            symbol = _normalize_symbol(removed_symbol)
            if symbol in active:
                continue
            active.add(symbol)
            active_interval_end[symbol] = date_ts

    for symbol in sorted(active):
        interval_end = active_interval_end.get(symbol, end_ts)
        if interval_end > DEFAULT_MEMBERSHIP_START:
            intervals.append({"symbol": symbol, "start_date": DEFAULT_MEMBERSHIP_START, "end_date": interval_end})

    if not intervals:
        return pd.DataFrame(columns=["symbol", "start_date", "end_date"])
    return _merge_membership_intervals(pd.DataFrame(intervals))


def _fmp_v3_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}/api/v3"
    return "https://financialmodelingprep.com/api/v3"


def _fmp_v3_get_list(path: str, api_key: str, base_url: str) -> list[dict[str, object]]:
    v3_base_url = _fmp_v3_base_url(base_url)
    url = f"{v3_base_url}{path if path.startswith('/') else '/' + path}"
    with httpx.Client(timeout=30) as client:
        response = client.get(url, params={"apikey": api_key}, headers={"User-Agent": HTTP_USER_AGENT})
        response.raise_for_status()
        payload = response.json()

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    return []


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(4))
def get_sp500_symbols(limit: int | None = None) -> list[str]:
    html = _fetch_html(WIKI_SP500_URL)
    symbols = _parse_symbols_from_tables(html, required_column="Symbol")
    return _apply_limit(symbols, limit)


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(4))
def get_sp100_symbols(limit: int | None = None) -> list[str]:
    html = _fetch_html(WIKI_SP100_URL)
    symbols = _parse_symbols_from_tables(html, required_column="Symbol")
    symbols = [s for s in symbols if _is_equity_symbol(s)]
    return _apply_limit(symbols, limit)


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(4))
def get_sp500_symbols_github(limit: int | None = None) -> list[str]:
    with httpx.Client(timeout=30) as client:
        response = client.get(GITHUB_SP500_CSV_URL, headers={"User-Agent": HTTP_USER_AGENT})
        response.raise_for_status()
        csv_text = response.text

    frame = pd.read_csv(StringIO(csv_text))
    if "Symbol" not in frame.columns:
        raise RuntimeError("GitHub S&P 500 CSV missing `Symbol` column")

    symbols = sorted({_normalize_symbol(s) for s in frame["Symbol"].astype(str)})
    return _apply_limit(symbols, limit)


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(4))
def get_sp500_symbols_fmp(api_key: str, base_url: str = "https://financialmodelingprep.com/stable", limit: int | None = None) -> list[str]:
    errors: list[str] = []
    try:
        client = FMPClient(api_key=api_key, base_url=base_url)
        rows = client.get_sp500_constituents()
        symbols = sorted({_normalize_symbol(str(r.get("symbol", ""))) for r in rows if r.get("symbol")})
        if symbols:
            return _apply_limit(symbols, limit)
        errors.append("stable endpoint returned no symbols")
    except Exception as exc:
        errors.append(f"stable endpoint failed: {exc}")

    rows = _fmp_v3_get_list(FMP_SP500_CONSTITUENTS_PATH, api_key=api_key, base_url=base_url)
    symbols = sorted({_normalize_symbol(str(r.get("symbol", ""))) for r in rows if r.get("symbol")})
    if symbols:
        return _apply_limit(symbols, limit)

    detail = " | ".join(errors) if errors else "no symbols returned"
    raise RuntimeError(f"FMP SP500 constituents unavailable. {detail}")


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(4))
def get_sp100_symbols_fmp(api_key: str, base_url: str = "https://financialmodelingprep.com/stable", limit: int | None = None) -> list[str]:
    # FMP does not expose a stable S&P100 constituents endpoint, so use OEF ETF holdings as proxy.
    rows = _fmp_v3_get_list(FMP_SP100_PROXY_ETF_HOLDINGS_PATH, api_key=api_key, base_url=base_url)
    symbols = sorted({_normalize_symbol(str(r.get("asset", ""))) for r in rows if r.get("asset")})
    symbols = [s for s in symbols if _is_equity_symbol(s)]
    if not symbols:
        raise RuntimeError("FMP OEF holdings returned no symbols")
    return _apply_limit(symbols, limit)


def get_custom_symbols(csv_path: Path) -> list[str]:
    frame = pd.read_csv(csv_path)
    if "symbol" not in frame.columns:
        raise ValueError("Custom universe CSV must include a `symbol` column")
    return sorted({_normalize_symbol(s) for s in frame["symbol"].dropna().astype(str)})


def load_cached_symbols(cache_path: Path) -> list[str]:
    if not cache_path.exists():
        return []
    frame = pd.read_csv(cache_path)
    if "symbol" not in frame.columns:
        return []
    return sorted({_normalize_symbol(s) for s in frame["symbol"].dropna().astype(str)})


def save_cached_symbols(symbols: list[str], cache_path: Path) -> None:
    if not symbols:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": sorted({_normalize_symbol(s) for s in symbols})}).to_csv(cache_path, index=False)


def get_sp500_symbols_resilient(
    limit: int | None = None,
    fmp_api_key: str | None = None,
    fmp_base_url: str = "https://financialmodelingprep.com/stable",
    cache_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []

    if fmp_api_key:
        try:
            symbols = get_sp500_symbols_fmp(api_key=fmp_api_key, base_url=fmp_base_url, limit=None)
            if symbols:
                if cache_path is not None:
                    save_cached_symbols(symbols, cache_path)
                return _apply_limit(symbols, limit)
        except Exception as exc:
            errors.append(f"FMP failed: {exc}")

    try:
        symbols = get_sp500_symbols_github(limit=None)
        if symbols:
            if cache_path is not None:
                save_cached_symbols(symbols, cache_path)
            return _apply_limit(symbols, limit)
    except Exception as exc:
        errors.append(f"GitHub CSV failed: {exc}")

    try:
        symbols = get_sp500_symbols(limit=None)
        if symbols:
            if cache_path is not None:
                save_cached_symbols(symbols, cache_path)
            return _apply_limit(symbols, limit)
    except Exception as exc:
        errors.append(f"Wikipedia failed: {exc}")

    if cache_path is not None:
        cached = load_cached_symbols(cache_path)
        if cached:
            return _apply_limit(cached, limit)

    detail = " | ".join(errors) if errors else "No upstream sources available"
    raise RuntimeError(f"Unable to resolve S&P 500 universe. {detail}")


def get_sp100_symbols_resilient(
    limit: int | None = None,
    fmp_api_key: str | None = None,
    fmp_base_url: str = "https://financialmodelingprep.com/stable",
    cache_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []

    try:
        symbols = get_sp100_symbols(limit=None)
        if symbols:
            if cache_path is not None:
                save_cached_symbols(symbols, cache_path)
            return _apply_limit(symbols, limit)
    except Exception as exc:
        errors.append(f"Wikipedia failed: {exc}")

    if fmp_api_key:
        try:
            symbols = get_sp100_symbols_fmp(api_key=fmp_api_key, base_url=fmp_base_url, limit=None)
            if symbols:
                if cache_path is not None:
                    save_cached_symbols(symbols, cache_path)
                return _apply_limit(symbols, limit)
        except Exception as exc:
            errors.append(f"FMP ETF holdings failed: {exc}")

    if cache_path is not None:
        cached = load_cached_symbols(cache_path)
        if cached:
            return _apply_limit(cached, limit)

    detail = " | ".join(errors) if errors else "No upstream sources available"
    raise RuntimeError(f"Unable to resolve S&P 100 universe. {detail}")


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(4))
def get_sp500_membership_intervals() -> pd.DataFrame:
    html = _fetch_html(WIKI_SP500_URL)
    tables = pd.read_html(StringIO(html))
    if len(tables) < 2:
        raise RuntimeError("Wikipedia S&P 500 page missing changes table")

    current = tables[0]
    if "Symbol" not in current.columns:
        raise RuntimeError("Wikipedia S&P 500 constituents table missing `Symbol`")
    current_symbols = sorted({_normalize_symbol(s) for s in current["Symbol"].dropna().astype(str)})
    intervals = reconstruct_sp500_membership_intervals(current_symbols=current_symbols, changes_table=tables[1])
    if intervals.empty:
        raise RuntimeError("Failed to reconstruct S&P 500 membership intervals")
    return intervals


def build_static_membership_intervals(
    symbols: list[str],
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_date, utc=True, errors="coerce") if start_date is not None else DEFAULT_MEMBERSHIP_START
    if pd.isna(start_ts):
        start_ts = DEFAULT_MEMBERSHIP_START
    end_ts = pd.to_datetime(end_date, utc=True, errors="coerce")
    if pd.isna(end_ts):
        end_ts = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)

    clean_symbols = sorted({_normalize_symbol(s) for s in symbols if _is_equity_symbol(_normalize_symbol(s))})
    if not clean_symbols:
        return pd.DataFrame(columns=["symbol", "start_date", "end_date"])
    return pd.DataFrame({"symbol": clean_symbols, "start_date": start_ts, "end_date": end_ts})


def save_membership_intervals(intervals: pd.DataFrame, cache_path: Path) -> None:
    if intervals.empty:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out = intervals.copy()
    out["start_date"] = pd.to_datetime(out["start_date"], utc=True, errors="coerce").astype("string")
    out["end_date"] = pd.to_datetime(out["end_date"], utc=True, errors="coerce").astype("string")
    out.to_csv(cache_path, index=False)


def load_membership_intervals(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=["symbol", "start_date", "end_date"])
    frame = pd.read_csv(cache_path)
    expected = {"symbol", "start_date", "end_date"}
    if not expected.issubset(frame.columns):
        return pd.DataFrame(columns=["symbol", "start_date", "end_date"])
    frame = frame[["symbol", "start_date", "end_date"]].copy()
    frame["symbol"] = frame["symbol"].astype(str).map(_normalize_symbol)
    frame["start_date"] = pd.to_datetime(frame["start_date"], utc=True, errors="coerce")
    frame["end_date"] = pd.to_datetime(frame["end_date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["symbol", "start_date", "end_date"])
    frame = frame.loc[frame["end_date"] > frame["start_date"]].copy()
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "start_date", "end_date"])
    return _merge_membership_intervals(frame)


def get_sp500_membership_intervals_resilient(
    cache_path: Path | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    errors: list[str] = []

    try:
        intervals = get_sp500_membership_intervals()
        if not intervals.empty:
            if cache_path is not None:
                save_membership_intervals(intervals, cache_path)
            if symbols:
                allowed = {_normalize_symbol(s) for s in symbols}
                intervals = intervals.loc[intervals["symbol"].isin(allowed)].copy()
            return intervals.reset_index(drop=True)
    except Exception as exc:
        errors.append(f"Wikipedia membership history failed: {exc}")

    if cache_path is not None:
        cached = load_membership_intervals(cache_path)
        if not cached.empty:
            if symbols:
                allowed = {_normalize_symbol(s) for s in symbols}
                cached = cached.loc[cached["symbol"].isin(allowed)].copy()
            return cached.reset_index(drop=True)

    detail = " | ".join(errors) if errors else "No membership interval sources available"
    raise RuntimeError(f"Unable to resolve S&P 500 membership history. {detail}")


def filter_frames_by_membership_intervals(
    frames: dict[str, pd.DataFrame],
    membership_intervals: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if not frames:
        return {}
    if membership_intervals.empty:
        return frames

    intervals = membership_intervals.copy()
    intervals["symbol"] = intervals["symbol"].astype(str).map(_normalize_symbol)
    intervals["start_date"] = pd.to_datetime(intervals["start_date"], utc=True, errors="coerce")
    intervals["end_date"] = pd.to_datetime(intervals["end_date"], utc=True, errors="coerce")
    intervals = intervals.dropna(subset=["symbol", "start_date", "end_date"]).copy()
    intervals = intervals.loc[intervals["end_date"] > intervals["start_date"]].copy()
    if intervals.empty:
        return {}

    grouped: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for symbol, group in intervals.groupby("symbol", sort=False):
        grouped[symbol] = [(row.start_date, row.end_date) for row in group.itertuples(index=False)]

    out: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        key = _normalize_symbol(symbol)
        intervals_for_symbol = grouped.get(key)
        if not intervals_for_symbol:
            continue
        if frame.empty or "datetime" not in frame.columns:
            continue

        dt = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        mask = np.zeros(len(frame), dtype=bool)
        for start_dt, end_dt in intervals_for_symbol:
            mask |= (dt >= start_dt) & (dt < end_dt)

        filtered = frame.loc[mask].copy()
        if filtered.empty:
            continue
        out[symbol] = filtered
    return out
