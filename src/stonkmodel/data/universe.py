from __future__ import annotations

from io import StringIO
from pathlib import Path
from urllib.parse import urlparse

import httpx
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
