from __future__ import annotations

import os
import re
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from stonkmodel.analysis.ticker_analysis import TickerAnalyzer
from stonkmodel.config import Settings


def _parse_symbols(raw: str) -> list[str]:
    parts = re.split(r"[\s,;]+", str(raw or "").strip())
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        sym = re.sub(r"[^A-Za-z0-9.\-]", "", part.strip().upper())
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _resolve_fmp_api_key(settings: Settings) -> str:
    candidates: list[str] = []

    configured = str(getattr(settings, "fmp_api_key", "") or "").strip()
    if configured:
        candidates.append(configured)

    env_upper = str(os.getenv("FMP_API_KEY", "") or "").strip()
    if env_upper:
        candidates.append(env_upper)

    env_lower = str(os.getenv("fmp_api_key", "") or "").strip()
    if env_lower:
        candidates.append(env_lower)

    try:
        secrets_key = str(st.secrets.get("FMP_API_KEY", "")).strip()
    except Exception:
        secrets_key = ""
    if secrets_key:
        candidates.append(secrets_key)

    try:
        secrets_alt = str(st.secrets.get("fmp_api_key", "")).strip()
    except Exception:
        secrets_alt = ""
    if secrets_alt:
        candidates.append(secrets_alt)

    return candidates[0] if candidates else ""


def render_ticker_analysis_mode(
    settings: Settings,
    run_inline: Callable[[Callable[..., Any]], Any],
) -> None:
    st.subheader("Ticker Analysis")
    st.caption(
        "Fundamental target-price analysis ported from your legacy ticker analysis flow. "
        "Enter tickers directly instead of loading from `fundTargets.txt`."
    )
    resolved_key = _resolve_fmp_api_key(settings)

    cpu_count = max(1, int(os.cpu_count() or 1))
    default_workers = min(8, cpu_count)

    with st.form("ticker_analysis_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            ticker_text = st.text_area(
                "Tickers",
                value="TSLA",
                height=140,
                help=(
                    "Enter one or more tickers separated by commas, spaces, or new lines. "
                    "Example: `TSLA, AAPL, NVDA`."
                ),
            )
        with c2:
            benchmark_symbol = st.text_input(
                "Benchmark for beta",
                value="VOO",
                help="Used for 5-year beta and CAPM/WACC pieces of the valuation model.",
            )
            workers = st.number_input(
                "Parallel workers",
                min_value=1,
                max_value=64,
                value=default_workers,
                step=1,
                help=(
                    f"How many tickers to analyze at once. Detected logical CPU cores: {cpu_count}. "
                    "Start around 4-8; reduce if FMP rate limits or memory pressure appear."
                ),
            )
        with c3:
            use_auto_rf = st.checkbox(
                "Auto risk-free rate",
                value=True,
                help="Uses FMP market-risk-premium endpoint when available, with fallback default.",
            )
            manual_rf_pct = st.number_input(
                "Manual risk-free rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=4.25,
                step=0.05,
                help="Used when Auto risk-free is unchecked. Example: 4.25 for 4.25%.",
                disabled=use_auto_rf,
            )

        with st.expander("API key override (optional)", expanded=False):
            fmp_key_override = st.text_input(
                "FMP API key",
                value="",
                type="password",
                help=(
                    "Only needed if the app cannot resolve `FMP_API_KEY` from settings, environment, or Streamlit secrets."
                ),
            )
            st.caption(
                "Current key detected from runtime sources: "
                + ("yes" if bool(resolved_key) else "no")
            )

        submitted = st.form_submit_button("Run Ticker Analysis", width="stretch")

    if submitted:
        effective_key = str(fmp_key_override or resolved_key).strip()
        if not effective_key:
            st.error(
                "FMP API key not found. Add `FMP_API_KEY` in `.env`, export it in shell, set Streamlit secrets, "
                "or paste it in API key override."
            )
            return
        symbols = _parse_symbols(ticker_text)
        if not symbols:
            st.warning("Please enter at least one valid ticker symbol.")
        else:
            risk_free_rate = None if use_auto_rf else float(manual_rf_pct) / 100.0
            analyzer = TickerAnalyzer(
                api_key=effective_key,
                base_url=str(settings.fmp_base_url),
                request_workers=int(workers),
            )

            def _task(progress: Callable[[float, str], None] | None = None) -> pd.DataFrame:
                return analyzer.analyze_many(
                    symbols=symbols,
                    workers=int(workers),
                    risk_free_rate=risk_free_rate,
                    benchmark_symbol=benchmark_symbol,
                    progress_callback=progress,
                )

            frame = run_inline(_task)
            st.session_state["ticker_analysis_last"] = frame

    if "ticker_analysis_last" not in st.session_state:
        return

    frame = st.session_state["ticker_analysis_last"]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        st.info("No ticker analysis output yet.")
        return

    view = frame.copy()
    numeric_pct_cols = [
        "upside_pct",
        "gross_margin_ttm",
        "operating_profit_margin_ttm",
        "yoy_revenue_growth",
        "yoy_gross_margin_change",
        "revenue_geometric_growth",
        "ocf_geometric_growth",
        "risk_free_rate",
        "wacc",
    ]
    for col in numeric_pct_cols:
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce") * 100.0
    rename_map = {
        "symbol": "Symbol",
        "price": "Price",
        "target_price": "Target Price",
        "upside_pct": "Upside (%)",
        "signal": "Signal",
        "signal_strength": "Signal Strength",
        "latest_quarter_date": "Latest Quarter",
        "revenue_ttm": "Revenue TTM",
        "gross_margin_ttm": "Gross Margin TTM (%)",
        "operating_profit_margin_ttm": "Operating Margin TTM (%)",
        "yoy_revenue_growth": "YoY Revenue Growth (%)",
        "yoy_gross_margin_change": "YoY Gross Margin Change (%)",
        "revenue_geometric_growth": "Revenue Geo Growth (%)",
        "ocf_geometric_growth": "OCF Geo Growth (%)",
        "operating_cashflow_ttm": "Operating Cashflow TTM",
        "revenue_growth_years": "Revenue Growth Years",
        "ocf_growth_years": "OCF Growth Years",
        "pe_ttm": "P/E TTM",
        "interest_coverage": "Interest Coverage",
        "beta_5y": "Beta (5Y)",
        "risk_free_rate": "Risk-Free Rate (%)",
        "wacc": "WACC (%)",
        "rsi_14": "RSI-14",
        "ppo_12_26": "PPO(12,26)",
        "gdx_ratio": "G/Dx Ratio",
        "benchmark_symbol": "Benchmark",
        "error": "Error",
    }
    core_cols = [
        "symbol",
        "price",
        "target_price",
        "upside_pct",
        "signal",
        "signal_strength",
        "rsi_14",
        "ppo_12_26",
        "revenue_ttm",
        "gross_margin_ttm",
        "operating_profit_margin_ttm",
        "yoy_revenue_growth",
        "yoy_gross_margin_change",
        "operating_cashflow_ttm",
        "revenue_geometric_growth",
        "ocf_geometric_growth",
        "revenue_growth_years",
        "ocf_growth_years",
        "gdx_ratio",
        "pe_ttm",
        "interest_coverage",
        "error",
    ]
    core_cols = [c for c in core_cols if c in view.columns]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Tickers analyzed", int(len(frame)))
    with c2:
        errs = int((frame.get("error", pd.Series(dtype=str)).fillna("").astype(str) != "").sum())
        st.metric("Errors", errs)
    with c3:
        avg_upside = pd.to_numeric(frame.get("upside_pct"), errors="coerce").mean()
        st.metric("Avg upside", f"{avg_upside * 100:.2f}%" if np.isfinite(avg_upside) else "N/A")
    with c4:
        buy_count = int((frame.get("signal", pd.Series(dtype=str)).fillna("").astype(str) == "Buy").sum())
        st.metric("Buy signals", buy_count)

    show_details = st.checkbox(
        "Show additional/detailed columns",
        value=False,
        help=(
            "Off: shows the core ticker-analysis output. "
            "On: appends all extra raw output columns from the analyzer."
        ),
        key="ticker_analysis_show_details",
    )
    if show_details:
        extra_cols = [c for c in view.columns if c not in core_cols]
        selected_cols = core_cols + extra_cols
    else:
        selected_cols = core_cols
    table = view[selected_cols].rename(columns=rename_map)
    st.dataframe(table, width="stretch")
