from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import re
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
    "analyst_total_ratings",
    "analyst_buy_ratio",
    "analyst_hold_ratio",
    "analyst_sell_ratio",
    "analyst_net_score",
    "rating_score",
    "rating_dcf_score",
    "rating_roe_score",
    "rating_roa_score",
    "rating_de_score",
    "rating_pe_score",
    "rating_pb_score",
    "earnings_eps_surprise",
    "earnings_eps_surprise_pct",
    "earnings_revenue_surprise_pct",
    "earnings_estimate_age_days",
    "analyst_grade_net_30d",
    "analyst_grade_up_30d",
    "analyst_grade_down_30d",
    "analyst_est_eps_avg",
    "analyst_est_eps_high",
    "analyst_est_eps_low",
    "analyst_est_eps_std",
    "analyst_est_eps_dispersion",
    "analyst_est_revenue_avg",
    "analyst_est_revenue_high",
    "analyst_est_revenue_low",
    "analyst_est_revenue_std",
    "analyst_est_revenue_dispersion",
    "price_target_mean",
    "price_target_high",
    "price_target_low",
    "price_target_upside",
    "price_target_dispersion",
    "price_target_analyst_count",
    "grades_consensus_score",
    "grades_consensus_buy_ratio",
    "grades_consensus_sell_ratio",
    "insider_net_shares_30d",
    "insider_net_value_30d",
    "insider_trade_count_30d",
    "insider_buy_sell_ratio_30d",
]


def _empty_fundamental_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "asof_datetime", *FUNDAMENTAL_COLUMNS])


def _normalize_fundamental_table(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_fundamental_frame()

    out = frame.copy()
    if "symbol" not in out.columns:
        return _empty_fundamental_frame()

    out["symbol"] = out["symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
    if "asof_datetime" not in out.columns:
        out["asof_datetime"] = pd.NaT
    out["asof_datetime"] = pd.to_datetime(out["asof_datetime"], utc=True, errors="coerce")

    for col in FUNDAMENTAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out = out[["symbol", "asof_datetime", *FUNDAMENTAL_COLUMNS]].copy()
    out = out.dropna(subset=["symbol", "asof_datetime"]).copy()
    if out.empty:
        return _empty_fundamental_frame()

    if out.duplicated(subset=["symbol", "asof_datetime"]).any():
        grouped = out.sort_values(["symbol", "asof_datetime"]).groupby(["symbol", "asof_datetime"], as_index=False)
        out = grouped[FUNDAMENTAL_COLUMNS].agg(_last_non_null)

    return out.sort_values(["symbol", "asof_datetime"]).reset_index(drop=True)


def _first_valid(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def _last_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[-1]


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out) or np.isinf(out):
        return None
    return out


def _safe_ratio(numerator: object, denominator: object) -> float | None:
    num = _safe_float(numerator)
    den = _safe_float(denominator)
    if num is None or den is None or den == 0:
        return None
    return num / den


def _normalize_symbol_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper().replace(".", "-")
    if not text or text in {"NAN", "NONE", "NULL"}:
        return None
    # Keep simple equity-like symbols only to avoid contaminating joins.
    if not re.fullmatch(r"[A-Z0-9\-]{1,10}", text):
        return None
    return text


def _extract_datetime(row: dict[str, Any], *candidates: str) -> pd.Timestamp:
    for key in candidates:
        if key not in row:
            continue
        dt = pd.to_datetime(row.get(key), utc=True, errors="coerce")
        if pd.notna(dt):
            return dt
    return pd.NaT


def _parse_amount_range(value: object) -> tuple[float | None, float | None, float | None]:
    if value is None:
        return None, None, None
    if isinstance(value, (int, float, np.number)):
        v = _safe_float(value)
        return v, v, v

    text = str(value).strip()
    if not text:
        return None, None, None
    text = text.replace("$", "").replace(",", "")
    text = text.replace("USD", "").replace("usd", "")
    text = text.strip()

    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return None, None, None
    parsed = [_safe_float(n) for n in nums]
    parsed = [p for p in parsed if p is not None]
    if not parsed:
        return None, None, None

    if len(parsed) == 1:
        only = parsed[0]
        return only, only, only

    lo = float(min(parsed[0], parsed[1]))
    hi = float(max(parsed[0], parsed[1]))
    mid = (lo + hi) / 2.0
    return lo, hi, mid


def _parse_trade_side(value: object) -> str:
    text = str(value or "").strip().lower()
    if any(token in text for token in ("purchase", "buy", "acquire", "acquisition", "received")):
        return "buy"
    if any(token in text for token in ("sale", "sell", "sold", "dispose", "disposition")):
        return "sell"
    return "other"


def _row_numeric(row: dict[str, Any], *candidates: str) -> float | None:
    for key in candidates:
        if key in row:
            value = _safe_float(row.get(key))
            if value is not None:
                return value
    return None


def _grade_label_to_score(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().lower()
    if not text:
        return 0.0
    if any(token in text for token in ("strong buy",)):
        return 2.0
    if any(token in text for token in ("strong sell",)):
        return -2.0
    if any(token in text for token in ("outperform", "overweight")):
        return 2.0
    if any(token in text for token in ("buy", "positive", "accumulate")):
        return 1.0
    if any(token in text for token in ("underperform", "underweight", "negative", "reduce")):
        return -1.0
    if any(token in text for token in ("sell",)):
        return -1.0
    if any(token in text for token in ("hold", "neutral", "market perform", "equal-weight", "in-line")):
        return 0.0
    return 0.0


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


def fetch_symbol_event_features_fmp(
    symbol: str,
    client: FMPClient,
    limit: int = 240,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    symbol_norm = _normalize_symbol_value(symbol) or symbol

    try:
        recommendations = client.get_analyst_stock_recommendations(symbol, limit=int(limit))
    except Exception:
        recommendations = []
    for row in recommendations:
        asof = _extract_datetime(row, "date", "publishedDate", "updatedAt")
        if pd.isna(asof):
            continue
        strong_buy = _safe_float(row.get("analystRatingsStrongBuy")) or 0.0
        buy = _safe_float(_first_valid(row.get("analystRatingsBuy"), row.get("analystRatingsbuy"))) or 0.0
        hold = _safe_float(row.get("analystRatingsHold")) or 0.0
        sell = _safe_float(row.get("analystRatingsSell")) or 0.0
        strong_sell = _safe_float(row.get("analystRatingsStrongSell")) or 0.0
        buy_total = strong_buy + buy
        sell_total = sell + strong_sell
        total = buy_total + hold + sell_total
        rows.append(
            {
                "symbol": symbol,
                "asof_datetime": asof,
                "analyst_total_ratings": total,
                "analyst_buy_ratio": _safe_ratio(buy_total, total),
                "analyst_hold_ratio": _safe_ratio(hold, total),
                "analyst_sell_ratio": _safe_ratio(sell_total, total),
                "analyst_net_score": _safe_ratio(buy_total - sell_total, total),
            }
        )

    try:
        ratings = client.get_historical_rating(symbol, limit=int(limit))
    except Exception:
        ratings = []
    for row in ratings:
        asof = _extract_datetime(row, "date", "publishedDate", "updatedAt")
        if pd.isna(asof):
            continue
        recommendation_score = _grade_label_to_score(row.get("ratingRecommendation"))
        rows.append(
            {
                "symbol": symbol,
                "asof_datetime": asof,
                "rating_score": _first_valid(_safe_float(row.get("ratingScore")), recommendation_score),
                "rating_dcf_score": _safe_float(row.get("ratingDetailsDCFScore")),
                "rating_roe_score": _safe_float(row.get("ratingDetailsROEScore")),
                "rating_roa_score": _safe_float(row.get("ratingDetailsROAScore")),
                "rating_de_score": _safe_float(row.get("ratingDetailsDEScore")),
                "rating_pe_score": _safe_float(row.get("ratingDetailsPEScore")),
                "rating_pb_score": _safe_float(row.get("ratingDetailsPBScore")),
            }
        )

    try:
        earnings = client.get_historical_earning_calendar(symbol, limit=int(limit))
    except Exception:
        earnings = []
    for row in earnings:
        asof = _extract_datetime(row, "date", "publishedDate", "updatedAt")
        if pd.isna(asof):
            continue
        eps = _safe_float(row.get("eps"))
        eps_est = _safe_float(row.get("epsEstimated"))
        revenue = _safe_float(row.get("revenue"))
        revenue_est = _safe_float(row.get("revenueEstimated"))
        surprise = None if eps is None or eps_est is None else eps - eps_est
        updated_dt = pd.to_datetime(row.get("updatedFromDate"), utc=True, errors="coerce")
        rows.append(
            {
                "symbol": symbol,
                "asof_datetime": asof,
                "earnings_eps_surprise": surprise,
                "earnings_eps_surprise_pct": _safe_ratio(surprise, abs(eps_est) if eps_est is not None else None),
                "earnings_revenue_surprise_pct": _safe_ratio(
                    (revenue - revenue_est) if revenue is not None and revenue_est is not None else None,
                    abs(revenue_est) if revenue_est is not None else None,
                ),
                "earnings_estimate_age_days": _safe_float((asof - updated_dt).days if pd.notna(updated_dt) else None),
            }
        )

    try:
        grade_rows = client.get_grade_history(symbol, limit=int(limit))
    except Exception:
        grade_rows = []
    grade_events: list[dict[str, object]] = []
    for row in grade_rows:
        asof = _extract_datetime(row, "date", "publishedDate", "updatedAt")
        if pd.isna(asof):
            continue
        previous = _grade_label_to_score(row.get("previousGrade"))
        new = _grade_label_to_score(row.get("newGrade"))
        delta = float(new - previous)
        grade_events.append(
            {
                "asof_datetime": asof,
                "grade_delta": delta,
                "grade_up": 1.0 if delta > 0 else 0.0,
                "grade_down": 1.0 if delta < 0 else 0.0,
            }
        )

    if grade_events:
        grade = pd.DataFrame(grade_events)
        grade = grade.groupby("asof_datetime", as_index=False).agg(
            grade_delta=("grade_delta", "sum"),
            grade_up=("grade_up", "sum"),
            grade_down=("grade_down", "sum"),
        )
        grade = grade.sort_values("asof_datetime").set_index("asof_datetime")
        grade["analyst_grade_net_30d"] = grade["grade_delta"].rolling("30D", min_periods=1).sum()
        grade["analyst_grade_up_30d"] = grade["grade_up"].rolling("30D", min_periods=1).sum()
        grade["analyst_grade_down_30d"] = grade["grade_down"].rolling("30D", min_periods=1).sum()
        grade = grade.reset_index()
        for _, row in grade.iterrows():
            rows.append(
                {
                    "symbol": symbol,
                    "asof_datetime": row["asof_datetime"],
                    "analyst_grade_net_30d": row["analyst_grade_net_30d"],
                    "analyst_grade_up_30d": row["analyst_grade_up_30d"],
                    "analyst_grade_down_30d": row["analyst_grade_down_30d"],
                }
            )

    try:
        analyst_estimates = client.get_analyst_estimates(
            symbol=symbol,
            period="annual",
            page=0,
            limit=max(20, int(limit)),
        )
    except Exception:
        analyst_estimates = []
    for row in analyst_estimates:
        asof = _extract_datetime(row, "date", "publishedDate", "fiscalDateEnding", "calendarYear")
        if pd.isna(asof):
            continue
        eps_avg = _row_numeric(row, "estimatedEpsAvg", "epsAvg", "estimatedEPSAvg", "estimatedEPS")
        eps_high = _row_numeric(row, "estimatedEpsHigh", "epsHigh", "estimatedEPSHigh")
        eps_low = _row_numeric(row, "estimatedEpsLow", "epsLow", "estimatedEPSLow")
        eps_std = _row_numeric(row, "estimatedEpsStdDev", "epsStdDev", "estimatedEPSStdDev")

        rev_avg = _row_numeric(
            row,
            "estimatedRevenueAvg",
            "revenueAvg",
            "estimatedRevenue",
        )
        rev_high = _row_numeric(row, "estimatedRevenueHigh", "revenueHigh")
        rev_low = _row_numeric(row, "estimatedRevenueLow", "revenueLow")
        rev_std = _row_numeric(row, "estimatedRevenueStdDev", "revenueStdDev")

        rows.append(
            {
                "symbol": symbol,
                "asof_datetime": asof,
                "analyst_est_eps_avg": eps_avg,
                "analyst_est_eps_high": eps_high,
                "analyst_est_eps_low": eps_low,
                "analyst_est_eps_std": eps_std,
                "analyst_est_eps_dispersion": _safe_ratio(
                    (eps_high - eps_low) if eps_high is not None and eps_low is not None else None,
                    abs(eps_avg) if eps_avg is not None else None,
                ),
                "analyst_est_revenue_avg": rev_avg,
                "analyst_est_revenue_high": rev_high,
                "analyst_est_revenue_low": rev_low,
                "analyst_est_revenue_std": rev_std,
                "analyst_est_revenue_dispersion": _safe_ratio(
                    (rev_high - rev_low) if rev_high is not None and rev_low is not None else None,
                    abs(rev_avg) if rev_avg is not None else None,
                ),
            }
        )

    consensus_sources: list[dict[str, Any]] = []
    try:
        price_target_summary = client.get_price_target_summary(symbol)
    except Exception:
        price_target_summary = {}
    if isinstance(price_target_summary, dict) and price_target_summary:
        consensus_sources.append(price_target_summary)

    try:
        price_target_consensus = client.get_price_target_consensus(symbol)
    except Exception:
        price_target_consensus = {}
    if isinstance(price_target_consensus, dict) and price_target_consensus:
        consensus_sources.append(price_target_consensus)

    for row in consensus_sources:
        asof = _extract_datetime(row, "date", "publishedDate", "updatedAt", "lastUpdated")
        if pd.isna(asof):
            asof = pd.Timestamp.now(tz="UTC")
        target_mean = _row_numeric(
            row,
            "targetMean",
            "targetConsensus",
            "priceTargetAverage",
            "targetMedian",
            "priceTarget",
        )
        target_high = _row_numeric(row, "targetHigh", "priceTargetHigh")
        target_low = _row_numeric(row, "targetLow", "priceTargetLow")
        current_price = _row_numeric(row, "price", "currentPrice", "stockPrice", "lastPrice")
        rows.append(
            {
                "symbol": symbol,
                "asof_datetime": asof,
                "price_target_mean": target_mean,
                "price_target_high": target_high,
                "price_target_low": target_low,
                "price_target_upside": _safe_ratio(
                    (target_mean - current_price) if target_mean is not None and current_price is not None else None,
                    abs(current_price) if current_price is not None else None,
                ),
                "price_target_dispersion": _safe_ratio(
                    (target_high - target_low) if target_high is not None and target_low is not None else None,
                    abs(target_mean) if target_mean is not None else None,
                ),
                "price_target_analyst_count": _row_numeric(
                    row,
                    "analystCount",
                    "numberAnalystOpinions",
                    "analystsCount",
                ),
            }
        )

    try:
        grades_consensus = client.get_grades_consensus(symbol)
    except Exception:
        grades_consensus = {}
    if isinstance(grades_consensus, dict) and grades_consensus:
        asof = _extract_datetime(grades_consensus, "date", "publishedDate", "updatedAt")
        if pd.isna(asof):
            asof = pd.Timestamp.now(tz="UTC")
        buy = (
            (_row_numeric(grades_consensus, "strongBuy") or 0.0)
            + (_row_numeric(grades_consensus, "buy") or 0.0)
            + (_row_numeric(grades_consensus, "outperform") or 0.0)
            + (_row_numeric(grades_consensus, "overweight") or 0.0)
        )
        hold = (_row_numeric(grades_consensus, "hold") or 0.0) + (_row_numeric(grades_consensus, "neutral") or 0.0)
        sell = (
            (_row_numeric(grades_consensus, "sell") or 0.0)
            + (_row_numeric(grades_consensus, "strongSell") or 0.0)
            + (_row_numeric(grades_consensus, "underperform") or 0.0)
            + (_row_numeric(grades_consensus, "underweight") or 0.0)
        )
        total = buy + hold + sell
        rows.append(
            {
                "symbol": symbol,
                "asof_datetime": asof,
                "grades_consensus_score": _first_valid(
                    _row_numeric(grades_consensus, "consensusScore", "ratingScore", "gradeScore"),
                    _safe_ratio(buy - sell, total),
                ),
                "grades_consensus_buy_ratio": _safe_ratio(buy, total),
                "grades_consensus_sell_ratio": _safe_ratio(sell, total),
            }
        )

    insider_events: list[dict[str, float | pd.Timestamp]] = []
    try:
        insider_rows = client.search_insider_trades(symbol=symbol, page=0, limit=max(int(limit), 300))
    except Exception:
        insider_rows = []
    if not insider_rows:
        try:
            insider_rows = client.get_latest_insider_trades(page=0, limit=max(int(limit), 300))
        except Exception:
            insider_rows = []
    for row in insider_rows:
        row_symbol = _normalize_symbol_value(_first_valid(row.get("symbol"), row.get("ticker")))
        if row_symbol is not None and row_symbol != symbol_norm:
            continue
        asof = _extract_datetime(row, "transactionDate", "filingDate", "date")
        if pd.isna(asof):
            continue
        side = _parse_trade_side(
            _first_valid(
                row.get("transactionType"),
                row.get("acquistionOrDisposition"),
                row.get("type"),
                row.get("action"),
            )
        )
        shares = _row_numeric(
            row,
            "securitiesTransacted",
            "shares",
            "sharesTraded",
            "numberOfShares",
            "securityCount",
        )
        trade_value = _row_numeric(row, "value", "transactionValue", "amount", "securityValue")
        trade_price = _row_numeric(row, "price", "securityPrice", "pricePerShare")
        if trade_value is None and shares is not None and trade_price is not None:
            trade_value = shares * trade_price

        sign = 1.0 if side == "buy" else (-1.0 if side == "sell" else 0.0)
        insider_events.append(
            {
                "asof_datetime": asof,
                "signed_shares": float(sign * (shares or 0.0)),
                "signed_value": float(sign * (trade_value or 0.0)),
                "trade_count": 1.0,
                "buy_count": 1.0 if side == "buy" else 0.0,
                "sell_count": 1.0 if side == "sell" else 0.0,
            }
        )

    if insider_events:
        insider = pd.DataFrame(insider_events)
        insider = (
            insider.groupby("asof_datetime", as_index=False)
            .agg(
                signed_shares=("signed_shares", "sum"),
                signed_value=("signed_value", "sum"),
                trade_count=("trade_count", "sum"),
                buy_count=("buy_count", "sum"),
                sell_count=("sell_count", "sum"),
            )
            .sort_values("asof_datetime")
        )
        insider = insider.set_index("asof_datetime")
        insider["insider_net_shares_30d"] = insider["signed_shares"].rolling("30D", min_periods=1).sum()
        insider["insider_net_value_30d"] = insider["signed_value"].rolling("30D", min_periods=1).sum()
        insider["insider_trade_count_30d"] = insider["trade_count"].rolling("30D", min_periods=1).sum()
        buy_30d = insider["buy_count"].rolling("30D", min_periods=1).sum()
        sell_30d = insider["sell_count"].rolling("30D", min_periods=1).sum()
        insider["insider_buy_sell_ratio_30d"] = buy_30d / (sell_30d + 1.0)
        insider = insider.reset_index()
        for _, row in insider.iterrows():
            rows.append(
                {
                    "symbol": symbol,
                    "asof_datetime": row["asof_datetime"],
                    "insider_net_shares_30d": row["insider_net_shares_30d"],
                    "insider_net_value_30d": row["insider_net_value_30d"],
                    "insider_trade_count_30d": row["insider_trade_count_30d"],
                    "insider_buy_sell_ratio_30d": row["insider_buy_sell_ratio_30d"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["symbol", "asof_datetime", *FUNDAMENTAL_COLUMNS])

    out = pd.DataFrame(rows)
    for col in FUNDAMENTAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[["symbol", "asof_datetime", *FUNDAMENTAL_COLUMNS]].copy()


def fetch_symbol_fundamentals_history_fmp(
    symbol: str,
    client: FMPClient,
    limit: int = 120,
) -> pd.DataFrame:
    profile = client.get_profile(symbol)
    ratios_rows = client.get_ratios_history(symbol, limit=int(limit), period="quarter")
    metrics_rows = client.get_key_metrics_history(symbol, limit=int(limit), period="quarter")

    ratios = pd.DataFrame(ratios_rows) if ratios_rows else pd.DataFrame()
    metrics = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()

    if ratios.empty and metrics.empty:
        base = pd.DataFrame([fetch_symbol_fundamentals_fmp(symbol, client)])
        event_features = fetch_symbol_event_features_fmp(symbol=symbol, client=client, limit=max(int(limit), 240))
        if event_features.empty:
            return _normalize_fundamental_table(base)
        return _normalize_fundamental_table(pd.concat([base, event_features], ignore_index=True, sort=False))

    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        if "date" not in out.columns:
            for alt in ("fillingDate", "acceptedDate", "calendarYear"):
                if alt in out.columns:
                    out["date"] = out[alt]
                    break
        out["asof_datetime"] = pd.to_datetime(out.get("date"), utc=True, errors="coerce")
        out = out.dropna(subset=["asof_datetime"]).copy()
        return out

    ratios = _prepare(ratios)
    metrics = _prepare(metrics)

    if ratios.empty and metrics.empty:
        base = pd.DataFrame([fetch_symbol_fundamentals_fmp(symbol, client)])
        event_features = fetch_symbol_event_features_fmp(symbol=symbol, client=client, limit=max(int(limit), 240))
        if event_features.empty:
            return _normalize_fundamental_table(base)
        return _normalize_fundamental_table(pd.concat([base, event_features], ignore_index=True, sort=False))

    if not ratios.empty and not metrics.empty:
        merged = ratios.merge(metrics, on="asof_datetime", how="outer", suffixes=("_ratio", "_metric"))
    elif not ratios.empty:
        merged = ratios.copy()
    else:
        merged = metrics.copy()

    def pick(row: pd.Series, *candidates: str):
        for candidate in candidates:
            for key in (candidate, f"{candidate}_ratio", f"{candidate}_metric"):
                if key in row.index:
                    value = row[key]
                    if pd.notna(value):
                        return value
        return None

    rows: list[dict[str, Any]] = []
    for _, row in merged.sort_values("asof_datetime").iterrows():
        rows.append(
            {
                "symbol": symbol,
                "asof_datetime": row.get("asof_datetime"),
                "market_cap": _first_valid(
                    pick(row, "marketCapTTM", "marketCap", "mktCap"),
                    profile.get("marketCap"),
                    profile.get("mktCap"),
                ),
                "float_shares": _first_valid(
                    pick(row, "floatShares", "freeFloat"),
                    profile.get("floatShares"),
                    profile.get("freeFloat"),
                ),
                "shares_outstanding": _first_valid(
                    pick(row, "sharesOutstanding", "sharesOutStanding"),
                    profile.get("sharesOutstanding"),
                    profile.get("sharesOutStanding"),
                ),
                "trailing_pe": _first_valid(pick(row, "peRatioTTM", "pe"), profile.get("pe")),
                "forward_pe": _first_valid(pick(row, "forwardPE"), profile.get("forwardPE")),
                "peg_ratio": _first_valid(pick(row, "pegRatioTTM", "pegRatio"), profile.get("pegRatio")),
                "price_to_book": _first_valid(pick(row, "priceToBookRatioTTM", "priceToBook"), profile.get("priceToBook")),
                "price_to_sales_trailing_12m": _first_valid(
                    pick(row, "priceToSalesRatioTTM", "priceToSalesTrailing12Months"),
                    profile.get("priceToSalesTrailing12Months"),
                ),
                "enterprise_to_revenue": _first_valid(
                    pick(row, "enterpriseValueOverRevenueTTM", "enterpriseToRevenue"),
                    profile.get("enterpriseToRevenue"),
                ),
                "enterprise_to_ebitda": _first_valid(
                    pick(row, "enterpriseValueOverEBITDATTM", "enterpriseToEbitda"),
                    profile.get("enterpriseToEbitda"),
                ),
                "beta": _first_valid(pick(row, "beta"), profile.get("beta")),
                "debt_to_equity": _first_valid(
                    pick(row, "debtEquityRatioTTM", "debtToEquityTTM", "debtToEquity"),
                    profile.get("debtToEquity"),
                ),
                "current_ratio": _first_valid(pick(row, "currentRatioTTM", "currentRatio"), profile.get("currentRatio")),
                "quick_ratio": _first_valid(pick(row, "quickRatioTTM", "quickRatio"), profile.get("quickRatio")),
                "profit_margins": _first_valid(
                    pick(row, "netProfitMarginTTM", "profitMargins"),
                    profile.get("profitMargins"),
                ),
                "gross_margins": _first_valid(
                    pick(row, "grossProfitMarginTTM", "grossMargins"),
                    profile.get("grossMargins"),
                ),
                "operating_margins": _first_valid(
                    pick(row, "operatingProfitMarginTTM", "operatingMargins"),
                    profile.get("operatingMargins"),
                ),
                "ebitda_margins": _first_valid(
                    pick(row, "ebitdaMarginTTM", "ebitdaMargins"),
                    profile.get("ebitdaMargins"),
                ),
                "return_on_assets": _first_valid(
                    pick(row, "returnOnAssetsTTM", "returnOnAssets"),
                    profile.get("returnOnAssets"),
                ),
                "return_on_equity": _first_valid(
                    pick(row, "returnOnEquityTTM", "returnOnEquity"),
                    profile.get("returnOnEquity"),
                ),
                "earnings_growth": _first_valid(
                    pick(row, "earningsGrowthTTM", "earningsGrowth"),
                    profile.get("earningsGrowth"),
                ),
                "revenue_growth": _first_valid(
                    pick(row, "revenueGrowthTTM", "revenueGrowth"),
                    profile.get("revenueGrowth"),
                ),
                "free_cashflow": _first_valid(
                    pick(row, "freeCashFlowTTM", "freeCashflow"),
                    profile.get("freeCashflow"),
                ),
                "operating_cashflow": _first_valid(
                    pick(row, "operatingCashFlowTTM", "operatingCashflow"),
                    profile.get("operatingCashflow"),
                ),
            }
        )

    if not rows:
        base = pd.DataFrame([fetch_symbol_fundamentals_fmp(symbol, client)])
    else:
        base = pd.DataFrame(rows)

    event_features = fetch_symbol_event_features_fmp(symbol=symbol, client=client, limit=max(int(limit), 240))
    if event_features.empty:
        return _normalize_fundamental_table(base)

    merged = pd.concat([base, event_features], ignore_index=True, sort=False)
    return _normalize_fundamental_table(merged)


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
    resolved_provider = provider
    if provider == "auto":
        resolved_provider = "fmp" if fmp_api_key else "yfinance"

    if cache_path.exists() and not refresh:
        try:
            cached = _normalize_fundamental_table(pd.read_parquet(cache_path))
        except Exception:
            cached = _empty_fundamental_frame()
    else:
        cached = _empty_fundamental_frame()

    if resolved_provider == "fmp":
        # For point-in-time joins we want a small history window per symbol, not a single latest snapshot.
        signal_cols = [c for c in ("analyst_net_score", "rating_score", "earnings_eps_surprise_pct") if c in cached.columns]
        if signal_cols:
            has_signals = cached[signal_cols].notna().any(axis=1)
        else:
            has_signals = pd.Series(False, index=cached.index)
        eligible = cached.loc[cached["asof_datetime"].notna() & has_signals].copy()
        existing = set(eligible.groupby("symbol").size().loc[lambda s: s >= 2].index.tolist())
    else:
        existing = set(cached["symbol"].dropna().astype(str).tolist())

    todo = [s for s in symbols if refresh or s not in existing]
    if max_symbols is not None:
        todo = todo[:max_symbols]
    if not todo:
        return cached

    fetched_frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    if resolved_provider == "fmp" and fmp_api_key:
        client = FMPClient(api_key=fmp_api_key, base_url=fmp_base_url)
        with ThreadPoolExecutor(max_workers=request_workers) as pool:
            futures = {pool.submit(fetch_symbol_fundamentals_history_fmp, symbol, client): symbol for symbol in todo}
            for future in as_completed(futures):
                try:
                    history = future.result()
                except Exception:
                    continue
                if isinstance(history, pd.DataFrame) and not history.empty:
                    fetched_frames.append(history)
    else:
        with ThreadPoolExecutor(max_workers=request_workers) as pool:
            futures = {pool.submit(fetch_symbol_fundamentals_yfinance, symbol): symbol for symbol in todo}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception:
                    continue
                rows.append(row)

    if fetched_frames:
        new = pd.concat(fetched_frames, ignore_index=True)
    elif rows:
        new = pd.DataFrame(rows)
    else:
        new = _empty_fundamental_frame()

    new = _normalize_fundamental_table(new)
    if not new.empty:
        if cached.empty:
            frame = new
        else:
            frame = _normalize_fundamental_table(pd.concat([cached, new], ignore_index=True))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)
        return frame

    return cached


def fetch_politician_trades_fmp(
    symbols: list[str],
    fmp_api_key: str | None,
    fmp_base_url: str = "https://financialmodelingprep.com/stable",
    start_datetime: pd.Timestamp | None = None,
    end_datetime: pd.Timestamp | None = None,
    request_workers: int = 8,
    max_pages: int = 20,
    page_size: int = 100,
) -> pd.DataFrame:
    if not fmp_api_key:
        return pd.DataFrame()

    symbol_set = {_normalize_symbol_value(s) for s in symbols}
    symbol_set = {s for s in symbol_set if s is not None}
    start = pd.to_datetime(start_datetime, utc=True, errors="coerce")
    end = pd.to_datetime(end_datetime, utc=True, errors="coerce")

    client = FMPClient(api_key=fmp_api_key, base_url=fmp_base_url)
    raw_batches: list[tuple[str, list[dict[str, Any]]]] = []

    def _fetch_paginated_chamber(chamber: str) -> list[tuple[str, list[dict[str, Any]]]]:
        fetcher = client.get_senate_trades if chamber == "senate" else client.get_house_trades
        batches: list[tuple[str, list[dict[str, Any]]]] = []
        for page in range(max(1, int(max_pages))):
            try:
                rows = fetcher(symbol=None, page=page, limit=page_size)
            except Exception:
                if page == 0:
                    return batches
                break
            if not rows:
                break
            batches.append((chamber, rows))
            if len(rows) < int(page_size):
                break
        return batches

    with ThreadPoolExecutor(max_workers=max(1, min(int(request_workers), 2))) as pool:
        futures = {pool.submit(_fetch_paginated_chamber, chamber): chamber for chamber in ("senate", "house")}
        for future in as_completed(futures):
            try:
                raw_batches.extend(future.result())
            except Exception:
                continue

    for chamber, fetcher in (
        ("senate", client.get_latest_senate_disclosures),
        ("house", client.get_latest_house_disclosures),
    ):
        try:
            rows = fetcher(page=0, limit=page_size)
        except Exception:
            rows = []
        if rows:
            raw_batches.append((chamber, rows))

    parsed_rows: list[dict[str, Any]] = []
    for chamber, batch in raw_batches:
        for row in batch:
            symbol = _normalize_symbol_value(
                _first_valid(
                    row.get("symbol"),
                    row.get("ticker"),
                    row.get("assetTicker"),
                    row.get("assetSymbol"),
                )
            )
            if symbol is None:
                continue
            if symbol_set and symbol not in symbol_set:
                continue

            trade_dt = _extract_datetime(
                row,
                "transactionDate",
                "tradeDate",
                "dateReceived",
                "date",
            )
            if pd.isna(trade_dt):
                continue
            if pd.notna(start) and trade_dt < start:
                continue
            if pd.notna(end) and trade_dt > end:
                continue

            disclosure_dt = _extract_datetime(row, "disclosureDate", "filingDate", "dateReceived", "dateRecieved")
            lag_days = (
                float((disclosure_dt - trade_dt).days)
                if pd.notna(disclosure_dt) and pd.notna(trade_dt)
                else np.nan
            )
            amount_low, amount_high, amount_mid = _parse_amount_range(
                _first_valid(
                    row.get("amount"),
                    row.get("amountRange"),
                    row.get("transactionValue"),
                    row.get("value"),
                )
            )
            if amount_mid is None:
                amount_mid = _row_numeric(row, "amountUsd", "amount", "transactionValue", "value")
            if amount_mid is None:
                amount_mid = 0.0

            side = _parse_trade_side(
                _first_valid(
                    row.get("type"),
                    row.get("transactionType"),
                    row.get("transaction"),
                    row.get("acquistionOrDisposition"),
                )
            )
            parsed_rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_dt,
                    "amount_usd": float(amount_mid),
                    "side": side,
                    "politician": str(
                        _first_valid(
                            row.get("representative"),
                            row.get("senator"),
                            row.get("office"),
                            row.get("name"),
                            " ".join(
                                [
                                    str(_first_valid(row.get("firstName"), "") or "").strip(),
                                    str(_first_valid(row.get("lastName"), "") or "").strip(),
                                ]
                            ).strip(),
                            row.get("firstName"),
                        )
                        or ""
                    ),
                    "chamber": chamber,
                    "disclosure_lag_days": lag_days,
                    "amount_low_usd": amount_low,
                    "amount_high_usd": amount_high,
                }
            )

    if not parsed_rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "trade_date",
                "amount_usd",
                "side",
                "politician",
                "chamber",
                "disclosure_lag_days",
                "amount_low_usd",
                "amount_high_usd",
            ]
        )

    out = pd.DataFrame(parsed_rows)
    out["trade_date"] = pd.to_datetime(out["trade_date"], utc=True, errors="coerce")
    out["amount_usd"] = pd.to_numeric(out["amount_usd"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["symbol", "trade_date"]).copy()
    out = out.drop_duplicates(
        subset=["symbol", "trade_date", "amount_usd", "side", "politician", "chamber"],
        keep="last",
    ).sort_values(["symbol", "trade_date"])
    return out.reset_index(drop=True)


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
    out["side"] = out["side"].map(_parse_trade_side)
    out = out.dropna(subset=["trade_date", "symbol"])
    return out


def engineer_politician_features(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades

    t = trades.copy()
    t["symbol"] = t["symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
    t["trade_date"] = pd.to_datetime(t["trade_date"], utc=True, errors="coerce")
    t["amount_usd"] = pd.to_numeric(t["amount_usd"], errors="coerce").fillna(0.0)
    t["side"] = t.get("side", pd.Series("other", index=t.index, dtype=object)).map(_parse_trade_side)
    t["signed_amount"] = np.where(t["side"] == "buy", t["amount_usd"], np.where(t["side"] == "sell", -t["amount_usd"], 0.0))
    t["buy_count"] = (t["side"] == "buy").astype(float)
    t["sell_count"] = (t["side"] == "sell").astype(float)
    if "disclosure_lag_days" in t.columns:
        t["disclosure_lag_days"] = pd.to_numeric(t["disclosure_lag_days"], errors="coerce")
    else:
        t["disclosure_lag_days"] = np.nan
    if "politician" not in t.columns:
        t["politician"] = ""
    if "chamber" not in t.columns:
        t["chamber"] = "unknown"
    t["chamber"] = t["chamber"].astype(str).str.lower()

    t = t.dropna(subset=["symbol", "trade_date"])
    if t.empty:
        return pd.DataFrame()

    t["senate_signed_flow"] = np.where(t["chamber"].str.contains("senate"), t["signed_amount"], 0.0)
    t["house_signed_flow"] = np.where(t["chamber"].str.contains("house"), t["signed_amount"], 0.0)
    t["senate_trade_count"] = np.where(t["chamber"].str.contains("senate"), 1.0, 0.0)
    t["house_trade_count"] = np.where(t["chamber"].str.contains("house"), 1.0, 0.0)

    grouped = (
        t.groupby(["symbol", "trade_date"], as_index=False)
        .agg(
            politician_signed_flow=("signed_amount", "sum"),
            politician_abs_flow=("amount_usd", "sum"),
            politician_trade_count=("amount_usd", "count"),
            politician_buy_count=("buy_count", "sum"),
            politician_sell_count=("sell_count", "sum"),
            politician_unique_count=("politician", "nunique"),
            politician_disclosure_lag_days=("disclosure_lag_days", "mean"),
            politician_senate_signed_flow=("senate_signed_flow", "sum"),
            politician_house_signed_flow=("house_signed_flow", "sum"),
            politician_senate_trade_count=("senate_trade_count", "sum"),
            politician_house_trade_count=("house_trade_count", "sum"),
        )
        .sort_values(["symbol", "trade_date"])
    )

    per_symbol: list[pd.DataFrame] = []
    for _, g in grouped.groupby("symbol", sort=False):
        x = g.copy().sort_values("trade_date")
        x = x.set_index("trade_date")
        x["politician_flow_7d"] = x["politician_signed_flow"].rolling("7D", min_periods=1).sum()
        x["politician_flow_30d"] = x["politician_signed_flow"].rolling("30D", min_periods=1).sum()
        x["politician_flow_90d"] = x["politician_signed_flow"].rolling("90D", min_periods=1).sum()
        x["politician_abs_flow_30d"] = x["politician_abs_flow"].rolling("30D", min_periods=1).sum()
        x["politician_count_30d"] = x["politician_trade_count"].rolling("30D", min_periods=1).sum()
        x["politician_buy_count_30d"] = x["politician_buy_count"].rolling("30D", min_periods=1).sum()
        x["politician_sell_count_30d"] = x["politician_sell_count"].rolling("30D", min_periods=1).sum()
        x["politician_unique_count_30d"] = x["politician_unique_count"].rolling("30D", min_periods=1).sum()
        x["politician_buy_sell_ratio_30d"] = x["politician_buy_count_30d"] / (x["politician_sell_count_30d"] + 1.0)
        x["politician_flow_imbalance_30d"] = x["politician_flow_30d"] / x["politician_abs_flow_30d"].replace(0, np.nan)
        x["politician_disclosure_lag_30d"] = x["politician_disclosure_lag_days"].rolling("30D", min_periods=1).mean()
        x["politician_senate_flow_30d"] = x["politician_senate_signed_flow"].rolling("30D", min_periods=1).sum()
        x["politician_house_flow_30d"] = x["politician_house_signed_flow"].rolling("30D", min_periods=1).sum()
        x["politician_senate_count_30d"] = x["politician_senate_trade_count"].rolling("30D", min_periods=1).sum()
        x["politician_house_count_30d"] = x["politician_house_trade_count"].rolling("30D", min_periods=1).sum()
        per_symbol.append(x.reset_index())

    return pd.concat(per_symbol, ignore_index=True)


def merge_external_features(
    price_frame: pd.DataFrame,
    fundamental_table: pd.DataFrame | None = None,
    politician_features: pd.DataFrame | None = None,
    macro_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if price_frame.empty:
        return price_frame

    out = price_frame.copy().sort_values(["symbol", "datetime"]).reset_index(drop=True)
    if "symbol" not in out.columns or "datetime" not in out.columns:
        return out
    out["symbol"] = out["symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out = out.dropna(subset=["symbol", "datetime"]).copy()
    if out.empty:
        return out

    if fundamental_table is not None and not fundamental_table.empty:
        f = _normalize_fundamental_table(fundamental_table)
        left = out.sort_values(["datetime", "symbol"]).reset_index(drop=True)
        if f.empty:
            left["fundamental_asof_datetime"] = pd.NaT
            for col in FUNDAMENTAL_COLUMNS:
                left[col] = np.nan
            out = left
        else:
            right = f[["symbol", "asof_datetime", *FUNDAMENTAL_COLUMNS]].rename(
                columns={"asof_datetime": "fundamental_asof_datetime"}
            )
            right["fundamental_asof_datetime"] = pd.to_datetime(
                right["fundamental_asof_datetime"], utc=True, errors="coerce"
            )
            right = right.dropna(subset=["symbol", "fundamental_asof_datetime"]).sort_values(
                ["fundamental_asof_datetime", "symbol"]
            )
            if right.empty:
                left["fundamental_asof_datetime"] = pd.NaT
                for col in FUNDAMENTAL_COLUMNS:
                    left[col] = np.nan
                out = left
            else:
                out = pd.merge_asof(
                    left,
                    right,
                    left_on="datetime",
                    right_on="fundamental_asof_datetime",
                    by="symbol",
                    direction="backward",
                )

    if politician_features is not None and not politician_features.empty:
        p = politician_features.copy()
        p["symbol"] = p["symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
        p["trade_date"] = pd.to_datetime(p["trade_date"], utc=True, errors="coerce")
        p = p.dropna(subset=["symbol", "trade_date"]).sort_values(["symbol", "trade_date"])
        p = p.drop_duplicates(subset=["symbol", "trade_date"], keep="last").reset_index(drop=True)
        if not p.empty:
            out = pd.merge_asof(
                out.sort_values(["datetime", "symbol"]).reset_index(drop=True),
                p.sort_values(["trade_date", "symbol"]).reset_index(drop=True),
                left_on="datetime",
                right_on="trade_date",
                by="symbol",
                direction="backward",
            ).drop(columns=["trade_date"], errors="ignore")

    if macro_features is not None and not macro_features.empty:
        m = macro_features.copy()
        if "datetime" not in m.columns and "date" in m.columns:
            m = m.rename(columns={"date": "datetime"})
        if "datetime" in m.columns:
            m["datetime"] = pd.to_datetime(m["datetime"], utc=True, errors="coerce")
            m = m.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
            m_cols = [c for c in m.columns if c != "datetime"]
            for col in m_cols:
                m[col] = pd.to_numeric(m[col], errors="coerce")
            if not m.empty and m_cols:
                out = pd.merge_asof(
                    out.sort_values("datetime"),
                    m.sort_values("datetime"),
                    on="datetime",
                    direction="backward",
                ).sort_values(["symbol", "datetime"]).reset_index(drop=True)

    for col in FUNDAMENTAL_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if {"trailing_pe"}.issubset(out.columns):
        out["earnings_yield"] = 1.0 / out["trailing_pe"].replace(0, np.nan)
    if {"price_to_book"}.issubset(out.columns):
        out["book_to_price"] = 1.0 / out["price_to_book"].replace(0, np.nan)
    if {"price_to_sales_trailing_12m"}.issubset(out.columns):
        out["sales_yield"] = 1.0 / out["price_to_sales_trailing_12m"].replace(0, np.nan)
    if {"market_cap", "free_cashflow"}.issubset(out.columns):
        out["free_cashflow_yield"] = out["free_cashflow"] / out["market_cap"].replace(0, np.nan)
    if {"market_cap", "operating_cashflow"}.issubset(out.columns):
        out["operating_cashflow_yield"] = out["operating_cashflow"] / out["market_cap"].replace(0, np.nan)
    if {"shares_outstanding", "float_shares"}.issubset(out.columns):
        out["free_float_ratio"] = out["float_shares"] / out["shares_outstanding"].replace(0, np.nan)
    if {"volume", "shares_outstanding"}.issubset(out.columns):
        out["turnover_ratio"] = out["volume"] / out["shares_outstanding"].replace(0, np.nan)
    if {"close", "volume"}.issubset(out.columns):
        out["dollar_volume"] = pd.to_numeric(out["close"], errors="coerce") * pd.to_numeric(
            out["volume"], errors="coerce"
        )
        out["log_dollar_volume"] = np.log1p(out["dollar_volume"].clip(lower=0))

    quality_cols = [c for c in ("return_on_assets", "return_on_equity", "profit_margins", "operating_margins") if c in out.columns]
    if quality_cols:
        out["quality_composite"] = out[quality_cols].mean(axis=1, skipna=True)

    if {"debt_to_equity", "current_ratio"}.issubset(out.columns):
        out["leverage_pressure"] = out["debt_to_equity"] / out["current_ratio"].replace(0, np.nan)

    if {"analyst_net_score", "rating_score"}.issubset(out.columns):
        out["analyst_rating_blend"] = out[["analyst_net_score", "rating_score"]].mean(axis=1, skipna=True)
    if {"analyst_rating_blend", "grades_consensus_score"}.issubset(out.columns):
        out["analyst_grade_consensus_blend"] = out[["analyst_rating_blend", "grades_consensus_score"]].mean(
            axis=1,
            skipna=True,
        )
    if {"analyst_net_score", "quality_composite"}.issubset(out.columns):
        out["analyst_quality_interaction"] = out["analyst_net_score"] * out["quality_composite"]
    if {"earnings_eps_surprise_pct", "analyst_net_score"}.issubset(out.columns):
        out["surprise_sentiment_interaction"] = out["earnings_eps_surprise_pct"] * out["analyst_net_score"]
    if {"price_target_upside", "analyst_net_score"}.issubset(out.columns):
        out["price_target_sentiment_alignment"] = out["price_target_upside"] * out["analyst_net_score"]
    if {"analyst_est_eps_dispersion", "price_target_dispersion"}.issubset(out.columns):
        out["analyst_uncertainty_index"] = out[["analyst_est_eps_dispersion", "price_target_dispersion"]].mean(
            axis=1,
            skipna=True,
        )
    if {"insider_net_value_30d", "market_cap"}.issubset(out.columns):
        out["insider_flow_to_mcap_30d"] = out["insider_net_value_30d"] / out["market_cap"].replace(0, np.nan)
    if {"insider_buy_sell_ratio_30d", "analyst_net_score"}.issubset(out.columns):
        out["insider_analyst_alignment"] = out["insider_buy_sell_ratio_30d"] * out["analyst_net_score"]
    if {"politician_flow_30d", "market_cap"}.issubset(out.columns):
        out["politician_flow_to_mcap_30d"] = out["politician_flow_30d"] / out["market_cap"].replace(0, np.nan)
    if {"politician_flow_30d", "insider_net_value_30d"}.issubset(out.columns):
        out["politician_insider_flow_alignment"] = np.sign(out["politician_flow_30d"]) * np.sign(out["insider_net_value_30d"])
    if {"alpha_ret_5", "macro_risk_off_score"}.issubset(out.columns):
        out["alpha_ret_5_macro_adj"] = out["alpha_ret_5"] * (1.0 - out["macro_risk_off_score"].clip(-3.0, 3.0) / 3.0)
    if {"ret_5", "macro_surprise_abs_index"}.issubset(out.columns):
        out["ret_5_macro_shock_adj"] = out["ret_5"] * (1.0 - out["macro_surprise_abs_index"].fillna(0.0).clip(0.0, 3.0) / 3.0)
    if {"volatility_20", "macro_days_to_next_us_major_event"}.issubset(out.columns):
        out["event_risk_tension"] = out["volatility_20"] / (1.0 + out["macro_days_to_next_us_major_event"].clip(lower=0.0))

    if "symbol" in out.columns:
        enhanced: list[pd.DataFrame] = []
        for _, g in out.groupby("symbol", sort=False):
            x = g.sort_values("datetime").copy()
            if "turnover_ratio" in x.columns:
                x["turnover_z_20"] = (
                    (x["turnover_ratio"] - x["turnover_ratio"].rolling(20, min_periods=5).mean())
                    / x["turnover_ratio"].rolling(20, min_periods=5).std().replace(0, np.nan)
                )
            if "analyst_net_score" in x.columns:
                x["analyst_net_score_delta_20"] = x["analyst_net_score"] - x["analyst_net_score"].shift(20)
            if "rating_score" in x.columns:
                x["rating_score_delta_20"] = x["rating_score"] - x["rating_score"].shift(20)
            if "earnings_eps_surprise_pct" in x.columns:
                x["earnings_surprise_decay_20"] = x["earnings_eps_surprise_pct"].ewm(
                    span=20,
                    adjust=False,
                    min_periods=1,
                ).mean()
            if "price_target_upside" in x.columns:
                x["price_target_upside_delta_20"] = x["price_target_upside"] - x["price_target_upside"].shift(20)
            if "insider_net_value_30d" in x.columns:
                x["insider_net_value_30d_z_63"] = (
                    (x["insider_net_value_30d"] - x["insider_net_value_30d"].rolling(63, min_periods=10).mean())
                    / x["insider_net_value_30d"].rolling(63, min_periods=10).std().replace(0, np.nan)
                )
            if "politician_flow_30d" in x.columns:
                x["politician_flow_30d_z_63"] = (
                    (x["politician_flow_30d"] - x["politician_flow_30d"].rolling(63, min_periods=10).mean())
                    / x["politician_flow_30d"].rolling(63, min_periods=10).std().replace(0, np.nan)
                )
            enhanced.append(x)
        out = pd.concat(enhanced, ignore_index=True)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out
