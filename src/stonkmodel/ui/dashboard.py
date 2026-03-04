from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st

from stonkmodel.config import get_settings
from stonkmodel.pipeline import StonkService
from stonkmodel.ui.groundup_mode import render_groundup_mode
from stonkmodel.ui.ticker_analysis_mode import render_ticker_analysis_mode

st.set_page_config(page_title="StonkModel Control Center", layout="wide")

settings = get_settings()
service = StonkService(settings)

INTERVAL_OPTIONS = ["1d", "1h", "30m", "15m", "5m", "1m"]
UNIVERSE_LABEL_TO_VALUE = {
    "S&P 500": "sp500",
    "S&P 100": "sp100",
}
BACKTEST_MODE_LABEL_TO_VALUE = {
    "Saved Models (strict OOS)": "saved_models",
    "Walk-Forward Retrain": "walk_forward_retrain",
}
LOGICAL_CORES = max(1, int(os.cpu_count() or 1))
PARALLEL_WORKER_HELP = (
    "How many tasks to run in parallel on this machine. "
    f"Detected logical CPU cores: {LOGICAL_CORES}. "
    "Start with 2-4 workers; increase gradually if CPU and memory stay healthy. "
    "If runs stall, memory spikes, or API calls get throttled, lower this value."
)

st.title("StonkModel Control Center")
st.caption("Dataset updates, training, model registry, backtests, scanner, and analytics")
ui_mode = st.radio(
    "Mode",
    options=["Classic", "Mixture of Experts", "Ticker Analysis"],
    index=0,
    horizontal=True,
    help=(
        "Classic mode is your existing workflow. Mixture of Experts mode is a separate "
        "regime-aware champion/challenger workflow. Ticker Analysis runs fundamental "
        "target-price analysis."
    ),
)


@st.cache_data(ttl=30)
def _load_dataset_registry() -> pd.DataFrame:
    return service.list_datasets()


@st.cache_data(ttl=30)
def _load_model_registry() -> pd.DataFrame:
    return service.model_registry()


def _dataset_names(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    return frame["dataset_name"].dropna().astype(str).tolist()


def _model_files(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    return frame["model_file"].dropna().astype(str).tolist()


def _pattern_names(frame: pd.DataFrame) -> list[str]:
    if frame.empty or "pattern" not in frame.columns:
        return []
    return sorted(frame["pattern"].dropna().astype(str).unique().tolist())


def _accepts_progress_callback(fn: Callable[..., Any]) -> bool:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            return True
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
    return False


def _call_with_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return fn(**kwargs)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(**filtered)


def _run_inline(fn: Callable[..., Any]) -> Any:
    if _accepts_progress_callback(fn):
        progress_slot = st.empty()
        bar = progress_slot.progress(0.0, text="0.0% - Starting")

        def _inline_progress(pct: float, message: str) -> None:
            bar.progress(max(0.0, min(1.0, float(pct) / 100.0)), text=f"{float(pct):.1f}% - {message}")

        try:
            result = fn(_inline_progress)
        finally:
            progress_slot.empty()
        return result
    return fn()


with st.sidebar:
    st.header("Runtime")
    st.write(f"Market data provider: `{settings.market_data_provider}`")
    st.write(f"Fundamentals provider: `{settings.fundamentals_provider}`")
    st.write(f"FMP key configured: `{bool(settings.fmp_api_key)}`")
    st.write(f"FRED key configured: `{bool(settings.fred_api_key)}`")
    st.write(f"Polygon key configured: `{bool(settings.polygon_api_key)}`")

    default_universe_label = "S&P 100" if settings.universe_source == "sp100" else "S&P 500"
    selected_universe_label = st.selectbox(
        "Universe (for market pulls)",
        options=list(UNIVERSE_LABEL_TO_VALUE.keys()),
        index=list(UNIVERSE_LABEL_TO_VALUE.keys()).index(default_universe_label),
        help=(
            "Controls which stock list is pulled when building datasets or scanning. "
            "S&P 100 is faster and good for rapid experimentation; S&P 500 is broader but heavier."
        ),
    )
    selected_universe = UNIVERSE_LABEL_TO_VALUE[selected_universe_label]


if ui_mode == "Mixture of Experts":
    render_groundup_mode(
        service=service,
        settings=settings,
        selected_universe=selected_universe,
        interval_options=INTERVAL_OPTIONS,
        run_inline=_run_inline,
        load_dataset_registry=_load_dataset_registry,
        load_model_registry=_load_model_registry,
    )
    st.stop()

if ui_mode == "Ticker Analysis":
    render_ticker_analysis_mode(
        settings=settings,
        run_inline=_run_inline,
    )
    st.stop()


tab_data, tab_train, tab_backtest, tab_scanner, tab_models, tab_analytics, tab_config = st.tabs(
    ["Data", "Train", "Backtest", "Scanner", "Models", "Analytics", "Config"]
)

with tab_data:
    st.subheader("Build or Update Dataset")
    st.caption("Create or refresh a processed training dataset from raw market + auxiliary inputs.")
    with st.form("build_dataset_form"):
        st.write("#### Core")
        col1, col2, col3 = st.columns(3)
        with col1:
            dataset_name = st.text_input(
                "Dataset name",
                value="model_dataset",
                help=(
                    "Name of the processed dataset file used later by training/backtesting/scanning. "
                    "Use descriptive names so you can compare runs, e.g. `sp500_1d_10y_v2`."
                ),
            )
        with col2:
            interval = st.selectbox(
                "Interval",
                options=INTERVAL_OPTIONS,
                index=0,
                help=(
                    "Candle timeframe for price history. "
                    "Example: `1d` is best for longer history and stability; intraday intervals are noisier and heavier."
                ),
            )
        with col3:
            years_ago_range = st.slider(
                "History window (years ago)",
                min_value=0,
                max_value=25,
                value=(0, min(15, settings.history_years)),
                help=(
                    "Choose a historical slice by years-ago. "
                    "Example: `(5, 10)` means keep data from 10 years ago up to 5 years ago (excluding recent 0-5 years)."
                ),
            )

        with st.expander("Advanced dataset options", expanded=False):
            col4, col5 = st.columns(2)
            with col4:
                refresh_prices = st.checkbox(
                    "Refresh prices from provider",
                    value=True,
                    help=(
                        "If enabled, re-downloads raw price bars from the provider and updates local cache. "
                        "Turn off to reuse cached files for faster reruns."
                    ),
                )
            with col5:
                politician_path_raw = st.text_input(
                    "Politician trades CSV (optional)",
                    value="",
                    help=(
                        "Optional local CSV path with politician-trade features to merge into the dataset. "
                        "If left blank and FMP API is configured, the app will auto-fetch congressional trades "
                        "(Senate/House) from FMP as a fallback source."
                    ),
                )

        submitted = st.form_submit_button("Build / Update Dataset", width="stretch")

    if submitted:
        politician_path = Path(politician_path_raw).expanduser() if politician_path_raw else None

        def _task(progress: Callable[[float, str], None] | None = None) -> dict[str, object]:
            result = service.build_dataset(
                interval=interval,
                years=years_ago_range[1],
                refresh_prices=refresh_prices,
                dataset_name=dataset_name,
                politician_trades_csv=politician_path,
                universe=selected_universe,
                years_ago_start=years_ago_range[0],
                years_ago_end=years_ago_range[1],
                progress_callback=progress,
            )
            return {
                "universe": result.universe,
                "years_ago_start": result.years_ago_start,
                "years_ago_end": result.years_ago_end,
                "symbols_requested": result.symbols_requested,
                "symbols_loaded": result.symbols_loaded,
                "rows": result.rows,
                "dataset_path": result.dataset_path,
            }

        result_payload = _run_inline(_task)
        st.session_state["last_build_result"] = result_payload
        _load_dataset_registry.clear()

    if "last_build_result" in st.session_state:
        st.success("Dataset build complete")
        st.json(st.session_state["last_build_result"])

    st.subheader("Dataset Registry")
    datasets = _load_dataset_registry()
    if datasets.empty:
        st.info("No datasets found in data/processed yet.")
    else:
        st.dataframe(datasets, width="stretch")

        choices = _dataset_names(datasets)
        selected = st.selectbox(
            "Inspect dataset",
            options=choices,
            help=(
                "View row counts, feature counts, date span, and other dataset metadata before training/backtesting."
            ),
        )
        summary = service.dataset_summary(selected)
        st.json(summary)

        if st.button("Preview first 30 rows", key="preview_dataset_rows"):
            frame = service.dataset_builder.load_dataset(selected).head(30)
            st.dataframe(frame, width="stretch")

        st.write("#### Delete Dataset")
        with st.form("delete_dataset_form"):
            delete_dataset_name = st.selectbox(
                "Dataset to delete",
                options=choices,
                key="delete_dataset_name",
                help="Permanently deletes the selected processed dataset from local storage.",
            )
            delete_dataset_confirm = st.text_input(
                "Type dataset name to confirm",
                value="",
                key="delete_dataset_confirm",
                help="Safety check: must match the dataset name exactly to prevent accidental deletion.",
            )
            delete_dataset_submitted = st.form_submit_button("Delete Dataset", width="stretch")

        if delete_dataset_submitted:
            if delete_dataset_confirm != delete_dataset_name:
                st.error("Confirmation mismatch. Dataset was not deleted.")
            else:
                deleted = service.delete_dataset(delete_dataset_name)
                if deleted:
                    st.success(f"Deleted dataset `{delete_dataset_name}`")
                    _load_dataset_registry.clear()
                else:
                    st.warning(f"Dataset `{delete_dataset_name}` was not found.")

with tab_train:
    st.subheader("Train Pattern Models")
    datasets = _load_dataset_registry()
    choices = _dataset_names(datasets)
    if not choices:
        st.info("Build a dataset first.")
    else:
        with st.form("train_form"):
            st.write("#### Core")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                train_dataset = st.selectbox(
                    "Dataset",
                    options=choices,
                    index=0,
                    help="Processed dataset to train models from (features + labels).",
                )
            with col2:
                train_interval = st.selectbox(
                    "Interval",
                    options=INTERVAL_OPTIONS,
                    index=0,
                    help=(
                        "Interval to train for. Usually keep this aligned with dataset interval "
                        "so pattern behavior matches the bar timeframe."
                    ),
                )
            with col3:
                min_pattern_rows = st.number_input(
                    "Min rows per pattern",
                    min_value=10,
                    max_value=100000,
                    value=settings.min_pattern_count,
                    help=(
                        "Minimum examples required to train a pattern model. "
                        "Higher values reduce overfitting. Example: 200+ is often more stable than 50."
                    ),
                )
            with col4:
                train_model_name = st.text_input(
                    "Model name (optional)",
                    value="",
                    help=(
                        "Optional custom name prefix for saved model files. "
                        "Useful for tracking experiments, e.g. `sp500_1d_fee1bps_v3`."
                    ),
                )

            with st.expander("Advanced training options", expanded=False):
                col5, col6, col7 = st.columns(3)
                with col5:
                    train_fast_mode = st.checkbox(
                        "Fast mode",
                        value=True,
                        help=(
                            "Prioritizes speed over exhaustive search. "
                            "Good for iteration; disable for more thorough training runs."
                        ),
                    )
                with col6:
                    train_parallel_patterns = st.number_input(
                        "Parallel patterns",
                        min_value=1,
                        max_value=16,
                        value=4,
                        step=1,
                        help=(
                            f"{PARALLEL_WORKER_HELP} "
                            "Example: if you have 8 logical cores, start around 4 and test 6-8."
                        ),
                    )
                with col7:
                    train_candidate_models = st.number_input(
                        "Candidate models/pattern",
                        min_value=1,
                        max_value=6,
                        value=2,
                        step=1,
                        help=(
                            "Number of model types tested per candlestick pattern. "
                            "Higher can improve quality but increases train time. Example: 2-3 is a common balance."
                        ),
                    )
                train_max_rows_per_pattern = st.number_input(
                    "Max rows per pattern (0 = no cap)",
                    min_value=0,
                    max_value=2_000_000,
                    value=120000,
                    step=10000,
                    help=(
                        "Maximum training rows per pattern. "
                        "Use caps to control runtime/memory on large datasets. `0` means use all available rows."
                    ),
                )
                train_pattern_filter_raw = st.text_input(
                    "Pattern filter (optional, comma-separated)",
                    value="",
                    help=(
                        "Train only selected patterns (comma-separated). "
                        "Example: `doji,hammer,bearish_engulfing`."
                    ),
                )
            submitted = st.form_submit_button("Train Models", width="stretch")

        if submitted:
            include_patterns = {p.strip() for p in str(train_pattern_filter_raw).split(",") if p.strip()} or None

            def _task(progress: Callable[[float, str], None] | None = None) -> pd.DataFrame:
                return service.train(
                    dataset_name=train_dataset,
                    interval=train_interval,
                    min_pattern_rows=int(min_pattern_rows),
                    model_name=train_model_name or None,
                    progress_callback=progress,
                    parallel_patterns=int(train_parallel_patterns),
                    fast_mode=bool(train_fast_mode),
                    max_rows_per_pattern=None if int(train_max_rows_per_pattern) <= 0 else int(train_max_rows_per_pattern),
                    include_patterns=include_patterns,
                    candidate_models_per_pattern=int(train_candidate_models),
                )

            table = _run_inline(_task)
            st.session_state["last_train_table"] = table
            _load_model_registry.clear()

        if "last_train_table" in st.session_state:
            table = st.session_state["last_train_table"]
            if table.empty:
                st.warning("No models trained. Increase history or lower min rows.")
            else:
                st.success(f"Trained {len(table)} models")
                st.dataframe(table, width="stretch")

        st.write("### Recursive Return Optimization")
        st.caption(
            "Iteratively retrains patterns, re-optimizes thresholds from strict OOS backtests, "
            "and keeps top performers each round."
        )
        with st.form("recursive_train_form"):
            st.write("#### Core")
            colr1, colr2, colr3, colr4, colr5 = st.columns(5)
            with colr1:
                rec_dataset = st.selectbox(
                    "Dataset",
                    options=choices,
                    index=0,
                    key="rec_dataset",
                    help="Processed dataset used for each recursive train/backtest round.",
                )
            with colr2:
                rec_interval = st.selectbox(
                    "Interval",
                    options=INTERVAL_OPTIONS,
                    index=0,
                    key="rec_interval",
                    help=(
                        "Interval used in recursive rounds. "
                        "Example: `1d` for swing-style models, intraday for shorter-horizon signals."
                    ),
                )
            with colr3:
                rec_rounds = st.number_input(
                    "Rounds",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                    help=(
                        "Number of recursive cycles. Each cycle retrains and re-ranks patterns by OOS performance. "
                        "Example: 3 rounds is a good starting point."
                    ),
                )
            with colr4:
                rec_keep_top = st.number_input(
                    "Keep top patterns/round",
                    min_value=1,
                    max_value=50,
                    value=6,
                    step=1,
                    help=(
                        "How many top patterns survive each round. "
                        "Lower values are more selective; higher values keep more diversification."
                    ),
                )
            with colr5:
                rec_model_base_name = st.text_input(
                    "Base model name (optional)",
                    value="recursive",
                    help=(
                        "Name prefix for models created in recursive rounds, e.g. `recursive_q1_2026`."
                    ),
                )

            with st.expander("Advanced training controls", expanded=False):
                colr6, colr7, colr8, colr9 = st.columns(4)
                with colr6:
                    rec_candidate_models = st.number_input(
                        "Candidate models/pattern",
                        min_value=1,
                        max_value=6,
                        value=2,
                        step=1,
                        key="rec_candidates",
                        help=(
                            "Model types evaluated per pattern per round. "
                            "Higher may improve fit but increases total runtime."
                        ),
                    )
                with colr7:
                    rec_parallel_patterns = st.number_input(
                        "Parallel patterns",
                        min_value=1,
                        max_value=16,
                        value=4,
                        step=1,
                        key="rec_parallel",
                        help=(
                            f"{PARALLEL_WORKER_HELP} "
                            "Keep this moderate if memory usage is high."
                        ),
                    )
                with colr8:
                    rec_min_pattern_rows = st.number_input(
                        "Min rows per pattern",
                        min_value=10,
                        max_value=100000,
                        value=settings.min_pattern_count,
                        step=10,
                        key="rec_min_rows",
                        help=(
                            "Ignore patterns with too few historical occurrences. "
                            "More rows generally means more reliable statistics."
                        ),
                    )
                with colr9:
                    rec_fast_mode = st.checkbox(
                        "Fast mode",
                        value=True,
                        key="rec_fast",
                        help=(
                            "Uses faster settings for each round. "
                            "Useful while searching; disable for a deeper final run."
                        ),
                    )

                colr10, colr11 = st.columns(2)
                with colr10:
                    rec_max_rows = st.number_input(
                        "Max rows per pattern (0 = no cap)",
                        min_value=0,
                        max_value=2_000_000,
                        value=120000,
                        step=10000,
                        key="rec_max_rows",
                        help=(
                            "Caps training rows per pattern in each round to limit runtime. "
                            "`0` uses full history."
                        ),
                    )
                with colr11:
                    rec_max_eval = st.number_input(
                        "Max eval rows per pattern",
                        min_value=1000,
                        max_value=20_000_000,
                        value=250000,
                        step=10000,
                        key="rec_max_eval",
                        help=(
                            "Caps strict out-of-sample rows used when scoring each pattern between rounds."
                        ),
                    )

            with st.expander("Execution realism and threshold controls", expanded=False):
                colr12, colr13, colr14, colr15, colr16 = st.columns(5)
                with colr12:
                    rec_min_trades_keep = st.number_input(
                        "Min trades to keep pattern",
                        min_value=1,
                        max_value=200000,
                        value=50,
                        step=5,
                        help=(
                            "Pattern must produce at least this many OOS trades to remain in the pool. "
                            "Prevents keeping high-return patterns with tiny sample size."
                        ),
                    )
                with colr13:
                    rec_fee_bps = st.number_input(
                        "Fee bps",
                        min_value=0.0,
                        max_value=1000.0,
                        value=1.0,
                        step=0.1,
                        key="rec_fee",
                        help=(
                            "Commission/fee per trade in basis points. "
                            "1 bps = 0.01%. Example: 5 bps = 0.05% per trade."
                        ),
                    )
                with colr14:
                    rec_spread_bps = st.number_input(
                        "Spread bps",
                        min_value=0.0,
                        max_value=1000.0,
                        value=0.0,
                        step=0.1,
                        key="rec_spread",
                        help=(
                            "Bid/ask spread cost assumption in bps. "
                            "Example: 2 bps means ~0.02% slippage from spread."
                        ),
                    )
                with colr15:
                    rec_slippage_bps = st.number_input(
                        "Slippage bps",
                        min_value=0.0,
                        max_value=1000.0,
                        value=0.0,
                        step=0.1,
                        key="rec_slip",
                        help=(
                            "Extra market-impact slippage in bps on top of spread. "
                            "Set higher for less-liquid names or aggressive fills."
                        ),
                    )
                with colr16:
                    rec_short_borrow = st.number_input(
                        "Short borrow bps/day",
                        min_value=0.0,
                        max_value=1000.0,
                        value=0.0,
                        step=0.1,
                        key="rec_borrow",
                        help=(
                            "Daily short borrow financing in bps/day. "
                            "Example: 10 bps/day = 0.10% cost each day a short stays open."
                        ),
                    )

                colr17, colr18 = st.columns(2)
                with colr17:
                    rec_latency_bars = st.number_input(
                        "Latency bars",
                        min_value=0,
                        max_value=100,
                        value=1,
                        step=1,
                        key="rec_latency",
                        help=(
                            "Execution delay after signal generation (in bars). "
                            "Use 1+ for more realistic bar-close systems."
                        ),
                    )
                with colr18:
                    rec_min_opt_trades = st.number_input(
                        "Min trades for threshold optimizer",
                        min_value=1,
                        max_value=200000,
                        value=40,
                        step=5,
                        help=(
                            "Only tune long/short thresholds when at least this many trades exist, "
                            "to reduce unstable threshold choices."
                        ),
                    )
            recursive_submitted = st.form_submit_button("Run Recursive Optimization", width="stretch")

        if recursive_submitted:

            def _recursive_task(progress: Callable[[float, str], None] | None = None) -> pd.DataFrame:
                return service.recursive_train_from_backtest(
                    dataset_name=rec_dataset,
                    interval=rec_interval,
                    min_pattern_rows=int(rec_min_pattern_rows),
                    base_model_name=rec_model_base_name or None,
                    rounds=int(rec_rounds),
                    keep_top_patterns=int(rec_keep_top),
                    min_trades_to_keep=int(rec_min_trades_keep),
                    parallel_patterns=int(rec_parallel_patterns),
                    fast_mode=bool(rec_fast_mode),
                    max_rows_per_pattern=None if int(rec_max_rows) <= 0 else int(rec_max_rows),
                    candidate_models_per_pattern=int(rec_candidate_models),
                    fee_bps=float(rec_fee_bps),
                    spread_bps=float(rec_spread_bps),
                    slippage_bps=float(rec_slippage_bps),
                    short_borrow_bps_per_day=float(rec_short_borrow),
                    latency_bars=int(rec_latency_bars),
                    min_threshold_opt_trades=int(rec_min_opt_trades),
                    max_eval_rows_per_pattern=int(rec_max_eval),
                    progress_callback=progress,
                )

            history = _run_inline(_recursive_task)
            st.session_state["recursive_history_table"] = history
            _load_model_registry.clear()

        if "recursive_history_table" in st.session_state:
            hist = st.session_state["recursive_history_table"]
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                st.success("Recursive optimization complete")
                st.dataframe(hist, width="stretch")

        st.write("### Autopilot Improvement Loop")
        st.caption(
            "Runs repeated train -> threshold-optimize -> strict-OOS backtest cycles automatically."
        )
        with st.form("autopilot_loop_form"):
            cola1, cola2, cola3, cola4 = st.columns(4)
            with cola1:
                auto_dataset = st.selectbox(
                    "Dataset",
                    options=choices,
                    index=0,
                    key="auto_dataset",
                    help="Dataset used for all autopilot iterations.",
                )
            with cola2:
                auto_interval = st.selectbox(
                    "Interval",
                    options=INTERVAL_OPTIONS,
                    index=0,
                    key="auto_interval",
                    help="Model/backtest interval used in each iteration.",
                )
            with cola3:
                auto_iterations = st.number_input(
                    "Iterations",
                    min_value=1,
                    max_value=100,
                    value=8,
                    step=1,
                    help="Maximum search iterations before stopping.",
                )
            with cola4:
                auto_max_minutes = st.number_input(
                    "Max runtime minutes",
                    min_value=5,
                    max_value=24 * 60,
                    value=180,
                    step=5,
                    help="Hard wall-clock stop for the loop.",
                )

            cola5, cola6, cola7 = st.columns(3)
            with cola5:
                auto_patience = st.number_input(
                    "Patience",
                    min_value=1,
                    max_value=20,
                    value=3,
                    step=1,
                    help="Stop after this many non-improving iterations.",
                )
            with cola6:
                auto_min_sig = st.number_input(
                    "Min significant improvement",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.10,
                    step=0.01,
                    help=(
                        "Relative objective improvement required to count as a real step-up. "
                        "0.10 means a new run must beat current best by 10%."
                    ),
                )
            with cola7:
                auto_min_trades = st.number_input(
                    "Min trades to qualify",
                    min_value=0,
                    max_value=500000,
                    value=40,
                    step=5,
                    help=(
                        "An iteration can only become the current best if its selected backtest row has at least "
                        "this many trades. Set 0 to disable this gate."
                    ),
                )

            cola9, cola10, cola11, cola12 = st.columns(4)
            with cola9:
                auto_fee_bps = st.number_input("Fee bps", min_value=0.0, max_value=1000.0, value=1.0, step=0.1, key="auto_fee_bps")
            with cola10:
                auto_spread_bps = st.number_input(
                    "Spread bps", min_value=0.0, max_value=1000.0, value=0.5, step=0.1, key="auto_spread_bps"
                )
            with cola11:
                auto_slippage_bps = st.number_input(
                    "Slippage bps", min_value=0.0, max_value=1000.0, value=0.5, step=0.1, key="auto_slippage_bps"
                )
            with cola12:
                auto_latency_bars = st.number_input(
                    "Latency bars", min_value=0, max_value=100, value=1, step=1, key="auto_latency_bars"
                )

            cola13, cola14, cola15, cola16 = st.columns(4)
            with cola13:
                auto_parallel_patterns = st.number_input(
                    "Parallel patterns",
                    min_value=1,
                    max_value=32,
                    value=4,
                    step=1,
                    key="auto_parallel_patterns",
                )
            with cola14:
                auto_include_spreads = st.checkbox(
                    "Include spread overlays",
                    value=True,
                    key="auto_include_spreads",
                )
            with cola15:
                auto_random_seed = st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=10_000_000,
                    value=42,
                    step=1,
                    key="auto_seed",
                )
            with cola16:
                auto_short_borrow_bps = st.number_input(
                    "Short borrow bps/day",
                    min_value=0.0,
                    max_value=1000.0,
                    value=0.0,
                    step=0.1,
                    key="auto_short_borrow",
                )

            auto_submitted = st.form_submit_button("Run Autopilot Loop", width="stretch")

        if auto_submitted:

            def _autopilot_task(progress: Callable[[float, str], None] | None = None) -> pd.DataFrame:
                return service.auto_improve(
                    dataset_name=auto_dataset,
                    interval=auto_interval,
                    iterations=int(auto_iterations),
                    max_minutes=int(auto_max_minutes),
                    patience=int(auto_patience),
                    min_significant_improvement=float(auto_min_sig),
                    min_iteration_trades=int(auto_min_trades),
                    fee_bps=float(auto_fee_bps),
                    spread_bps=float(auto_spread_bps),
                    slippage_bps=float(auto_slippage_bps),
                    short_borrow_bps_per_day=float(auto_short_borrow_bps),
                    latency_bars=int(auto_latency_bars),
                    parallel_patterns=int(auto_parallel_patterns),
                    include_spread_strategies=bool(auto_include_spreads),
                    random_seed=int(auto_random_seed),
                    progress_callback=progress,
                )

            try:
                autopilot_table = _run_inline(_autopilot_task)
            except Exception as exc:
                st.error(f"Autopilot failed: {exc}")
                st.exception(exc)
            else:
                st.session_state["autopilot_history_table"] = autopilot_table
                _load_model_registry.clear()

        if "autopilot_history_table" in st.session_state:
            auto_hist = st.session_state["autopilot_history_table"]
            if isinstance(auto_hist, pd.DataFrame) and not auto_hist.empty:
                st.success("Autopilot loop complete")
                st.dataframe(auto_hist, width="stretch")

with tab_backtest:
    st.subheader("Backtest Models")
    datasets = _load_dataset_registry()
    models = _load_model_registry()
    dataset_choices = _dataset_names(datasets)

    if not dataset_choices:
        st.info("Build a dataset first.")
    else:
        with st.form("backtest_form"):
            st.write("#### Core")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                bt_dataset = st.selectbox(
                    "Dataset",
                    options=dataset_choices,
                    index=0,
                    help="Processed dataset to evaluate strategy performance on out-of-sample history.",
                )
            with col2:
                bt_interval = st.selectbox(
                    "Interval",
                    options=INTERVAL_OPTIONS,
                    index=0,
                    help="Bar timeframe used for model scoring and PnL simulation.",
                )
            with col3:
                bt_mode_label = st.selectbox(
                    "Mode",
                    options=list(BACKTEST_MODE_LABEL_TO_VALUE.keys()),
                    index=0,
                    help=(
                        "`Saved Models` is faster and uses your stored trained models. "
                        "`Walk-Forward Retrain` retrains in rolling windows for stricter time-based validation."
                    ),
                )
            with col4:
                fee_bps = st.number_input(
                    "Fee bps",
                    min_value=0.0,
                    max_value=1000.0,
                    value=1.0,
                    step=0.1,
                    help=(
                        "Commission/fee per trade in basis points. "
                        "1 bps = 0.01%. Example: 1.5 bps means 0.015% per trade."
                    ),
                )
            bt_mode = BACKTEST_MODE_LABEL_TO_VALUE[bt_mode_label]
            bt_threshold_rec = service.recommend_thresholds(interval=bt_interval)
            rec_long = float(bt_threshold_rec.get("recommended_long_threshold", 0.65))
            rec_short = float(bt_threshold_rec.get("recommended_short_threshold", 0.35))
            st.caption(
                (
                    f"Suggested thresholds for `{bt_interval}`: long >= {rec_long:.2f}, short <= {rec_short:.2f} "
                    f"(from {int(bt_threshold_rec.get('models_considered', 0))} model(s))."
                )
            )
            st.caption(str(bt_threshold_rec.get("explanation", "")))

            use_model_thresholds = st.checkbox(
                "Use tuned model thresholds",
                value=(bt_mode == "saved_models"),
                help=(
                    "Uses each model's own saved long/short thresholds from prior optimization. "
                    "Disable to force one global threshold pair for all models."
                ),
            )

            long_threshold = None
            short_threshold = None
            use_recommended_thresholds = False
            if not use_model_thresholds:
                col5, col6, col7 = st.columns(3)
                with col5:
                    use_recommended_thresholds = st.checkbox(
                        "Use suggested thresholds above",
                        value=False,
                        help=(
                            "Applies interval-level recommended thresholds automatically. "
                            "Good starting point before manual tuning."
                        ),
                    )
                with col6:
                    long_threshold = st.slider(
                        "Long threshold",
                        min_value=0.5,
                        max_value=0.95,
                        value=float(np.clip(round(rec_long, 2), 0.5, 0.95)),
                        step=0.01,
                        help=(
                            "Enter long when model probability of up move is at/above this value. "
                            "Example: 0.60 means only higher-conviction longs."
                        ),
                    )
                with col7:
                    short_threshold = st.slider(
                        "Short threshold",
                        min_value=0.05,
                        max_value=0.5,
                        value=float(np.clip(round(rec_short, 2), 0.05, 0.5)),
                        step=0.01,
                        help=(
                            "Enter short when model probability of up move is at/below this value. "
                            "Example: 0.40 means short when downside is more likely."
                        ),
                    )

            with st.expander("Execution realism", expanded=True):
                col8, col9, col10, col11, col12 = st.columns(5)
                with col8:
                    spread_bps = st.number_input(
                        "Spread bps",
                        min_value=0.0,
                        max_value=1000.0,
                        value=0.0,
                        step=0.1,
                        help=(
                            "Bid/ask spread cost assumption in bps. "
                            "Example: 2 bps approximates 0.02% round impact."
                        ),
                    )
                with col9:
                    slippage_bps = st.number_input(
                        "Slippage bps",
                        min_value=0.0,
                        max_value=1000.0,
                        value=0.0,
                        step=0.1,
                        help=(
                            "Additional execution slippage/impact beyond spread in bps. "
                            "Use higher values for volatile or less-liquid periods."
                        ),
                    )
                with col10:
                    short_borrow_bps_per_day = st.number_input(
                        "Short borrow bps/day",
                        min_value=0.0,
                        max_value=1000.0,
                        value=0.0,
                        step=0.1,
                        help=(
                            "Daily short borrow financing in bps/day. "
                            "Example: 8 bps/day = 0.08% daily cost while short is open."
                        ),
                    )
                with col11:
                    latency_bars = st.number_input(
                        "Latency bars",
                        min_value=0,
                        max_value=100,
                        value=1,
                        step=1,
                        help=(
                            "Delay between signal and fill (in bars). "
                            "0 is optimistic; 1+ is typically more realistic."
                        ),
                    )
                    st.caption("Use `1` for bar-close signals to avoid optimistic same-bar execution assumptions.")
                with col12:
                    embargo_bars = st.number_input(
                        "Embargo bars",
                        min_value=0,
                        max_value=100,
                        value=1,
                        step=1,
                        help=(
                            "Gap after train cutoff before evaluation starts. "
                            "Use 1-3 bars to reduce near-boundary leakage."
                        ),
                    )

            with st.expander("Performance and limits", expanded=False):
                col_fast, col_workers = st.columns(2)
                with col_fast:
                    bt_fast_mode = st.checkbox(
                        "Fast mode",
                        value=True,
                        help=(
                            "Reduces runtime with speed-focused settings. "
                            "Useful for parameter sweeps, then confirm top runs with stricter settings."
                        ),
                    )
                with col_workers:
                    bt_parallel_models = st.number_input(
                        "Parallel model workers",
                        min_value=1,
                        max_value=32,
                        value=4,
                        step=1,
                        help=(
                            f"{PARALLEL_WORKER_HELP} "
                            "For large S&P 500 jobs, try 4-8 first, then tune up/down."
                        ),
                    )
                bt_max_eval_rows = st.number_input(
                    "Max eval rows per pattern (saved-model mode, 0 = no cap)",
                    min_value=0,
                    max_value=20_000_000,
                    value=250000,
                    step=10000,
                    help=(
                        "Upper limit for evaluation rows per pattern in saved-model mode. "
                        "Use lower values for quick checks and higher values for final validation."
                    ),
                )

            with st.expander("Portfolio construction", expanded=False):
                col_portfolio1, col_portfolio2, col_portfolio3, col_portfolio4 = st.columns(4)
                with col_portfolio1:
                    bt_include_portfolio = st.checkbox(
                        "Include portfolio-combined row",
                        value=True,
                        help=(
                            "Adds one combined long/short portfolio line, not just per-pattern results. "
                            "Useful for comparing a tradable aggregate strategy."
                        ),
                    )
                with col_portfolio2:
                    bt_portfolio_top_k = st.number_input(
                        "Top K per side",
                        min_value=1,
                        max_value=500,
                        value=3,
                        step=1,
                        help=(
                            "At each rebalance, select top K longs and top K shorts by score. "
                            "Lower K usually means fewer trades and higher signal concentration."
                        ),
                    )
                with col_portfolio3:
                    bt_portfolio_gross = st.number_input(
                        "Max gross exposure",
                        min_value=0.0,
                        max_value=10.0,
                        value=1.0,
                        step=0.1,
                        help=(
                            "Gross exposure cap (sum of absolute weights). "
                            "1.0 is roughly 100% gross; 2.0 is roughly 200% gross."
                        ),
                    )
                with col_portfolio4:
                    bt_portfolio_pattern_selection_label = st.selectbox(
                        "Pattern combination mode",
                        options=[
                            "All selected patterns",
                            "Best patterns only",
                            "Both (compare both rows)",
                        ],
                        index=1,
                        help=(
                            "How to build the combined portfolio row. "
                            "`Best patterns only` ranks patterns by return/risk/win-rate quality and uses the top subset."
                        ),
                    )
                    bt_portfolio_pattern_selection = {
                        "All selected patterns": "all",
                        "Best patterns only": "best",
                        "Both (compare both rows)": "both",
                    }[bt_portfolio_pattern_selection_label]

                col_portfolio5, col_portfolio6, col_portfolio7 = st.columns(3)
                with col_portfolio5:
                    bt_portfolio_best_patterns_top_n = st.number_input(
                        "Best patterns to keep",
                        min_value=1,
                        max_value=50,
                        value=6,
                        step=1,
                        help=(
                            "Used when combination mode includes `Best patterns`. "
                            "Only the top N ranked patterns are included in the portfolio."
                        ),
                    )
                with col_portfolio6:
                    bt_portfolio_min_pattern_trades = st.number_input(
                        "Min pattern trades (for best ranking)",
                        min_value=0,
                        max_value=200000,
                        value=40,
                        step=5,
                        help=(
                            "Pattern must have at least this many trades to be considered for `Best patterns` ranking."
                        ),
                    )
                with col_portfolio7:
                    bt_portfolio_min_pattern_win_rate_trade = st.slider(
                        "Min pattern trade win rate",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.55,
                        step=0.01,
                        help=(
                            "Pattern-level filter for `Best patterns` mode. "
                            "Example: 0.55 keeps only patterns with >=55% trade win rate."
                        ),
                    )

                col_portfolio8, col_portfolio9, col_portfolio10 = st.columns(3)
                with col_portfolio8:
                    bt_portfolio_min_abs_score = st.slider(
                        "Min confidence edge |p-0.5|",
                        min_value=0.0,
                        max_value=0.45,
                        value=0.15,
                        step=0.01,
                        help=(
                            "Only take portfolio trades when signal edge is strong enough. "
                            "Higher values usually reduce trades and can improve win rate."
                        ),
                    )
                with col_portfolio9:
                    bt_portfolio_rebalance_every_n_bars = st.number_input(
                        "Rebalance every N bars",
                        min_value=1,
                        max_value=50,
                        value=3,
                        step=1,
                        help=(
                            "Trade only every N bars (1 = every bar). "
                            "Higher N reduces turnover."
                        ),
                    )
                with col_portfolio10:
                    bt_portfolio_symbol_cooldown_bars = st.number_input(
                        "Symbol cooldown bars",
                        min_value=0,
                        max_value=100,
                        value=5,
                        step=1,
                        help=(
                            "After trading a symbol, wait this many bars before trading it again. "
                            "Useful to cut churning."
                        ),
                    )

                col_portfolio11, col_portfolio12 = st.columns(2)
                with col_portfolio11:
                    bt_portfolio_volatility_scaling = st.checkbox(
                        "Volatility-scaled sizing",
                        value=True,
                        help=(
                            "Scales position size by signal quality and inverse volatility. "
                            "Helps reduce risk concentration in choppy symbols."
                        ),
                    )
                with col_portfolio12:
                    bt_portfolio_max_symbol_weight = st.slider(
                        "Max weight per symbol",
                        min_value=0.05,
                        max_value=1.0,
                        value=0.35,
                        step=0.01,
                        help=(
                            "Hard cap on any single symbol's absolute weight in the combined portfolio. "
                            "Lower caps reduce single-name blowup risk."
                        ),
                    )

                bt_initial_investment = st.number_input(
                    "Hypothetical starting investment ($)",
                    min_value=100.0,
                    max_value=10_000_000.0,
                    value=10_000.0,
                    step=100.0,
                    help=(
                        "Starting capital for the hypothetical equity curve chart. "
                        "Example: set 10000 to view growth/loss of a $10,000 account."
                    ),
                )

            with st.expander("Spread strategies (research-backed)", expanded=False):
                st.caption(
                    "Implements relative-value long/short overlays inspired by pairs/stat-arb and regime-aware allocation research."
                )
                st.markdown(
                    (
                        "References: "
                        "[Gatev et al. (2006)](https://www.nber.org/papers/w7032), "
                        "[Frazzini & Pedersen (2014)](https://www.nber.org/papers/w16601), "
                        "[Hamilton (1989)](https://econpapers.repec.org/article/ecmemetrp/v_3a57_3ay_3a1989_3ai_3a2_3ap_3a357-84.htm), "
                        "[Moreira & Muir (2017)](https://www.nber.org/papers/w22208), "
                        "[Jegadeesh & Titman (1993)](https://afajof.org/issue/volume-48-issue-1/)."
                    )
                )
                col_spread1, col_spread2, col_spread3, col_spread4 = st.columns(4)
                with col_spread1:
                    bt_include_spread_strategies = st.checkbox(
                        "Include spread overlays",
                        value=True,
                        help=(
                            "Adds model-vs-model and pattern-vs-pattern relative-value strategies. "
                            "These trade the spread between stronger and weaker components, not just outright direction."
                        ),
                    )
                with col_spread2:
                    bt_spread_lookback_bars = st.number_input(
                        "Spread lookback bars",
                        min_value=10,
                        max_value=5000,
                        value=63,
                        step=1,
                        help=(
                            "History length used to rank components before placing spread trades. "
                            "63 bars is about one quarter on daily data."
                        ),
                    )
                with col_spread3:
                    bt_spread_top_components = st.number_input(
                        "Top/bottom components",
                        min_value=1,
                        max_value=50,
                        value=3,
                        step=1,
                        help=(
                            "Number of strongest and weakest components considered each rebalance. "
                            "Lower values are more concentrated; higher values are more diversified."
                        ),
                    )
                with col_spread4:
                    bt_spread_min_edge = st.slider(
                        "Min spread edge",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.02,
                        step=0.01,
                        help=(
                            "Only trade when strength difference between long and short components exceeds this edge. "
                            "Higher values reduce trade count and can improve selectivity."
                        ),
                    )

                col_spread5, col_spread6, col_spread7, col_spread8 = st.columns(4)
                with col_spread5:
                    bt_spread_switch_cost_bps = st.number_input(
                        "Spread switch cost bps",
                        min_value=0.0,
                        max_value=1000.0,
                        value=0.0,
                        step=0.1,
                        help=(
                            "Extra friction paid when the selected spread pair changes. "
                            "Use this to penalize excessive pair turnover."
                        ),
                    )
                with col_spread6:
                    bt_spread_include_neutral_overlay = st.checkbox(
                        "Include beta/size neutral overlays",
                        value=True,
                        help=(
                            "Adds neutralized variants that rebalance long/short weights to reduce market beta and size tilts "
                            "(aligned with low-beta and market-neutral literature)."
                        ),
                    )
                with col_spread7:
                    bt_spread_include_regime_switch = st.checkbox(
                        "Include regime switching spread",
                        value=True,
                        help=(
                            "Adds a macro/volatility-conditioned switch between model-spread and pattern-spread variants "
                            "(inspired by regime-switching research)."
                        ),
                    )
                with col_spread8:
                    bt_spread_target_vol_annual = st.number_input(
                        "Spread vol target (annual, 0-1)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.01,
                        help=(
                            "Optional volatility targeting for spread returns. "
                            "Example: 0.15 targets roughly 15% annualized spread volatility."
                        ),
                    )

            if bt_mode == "walk_forward_retrain":
                with st.expander("Walk-forward settings", expanded=False):
                    col12, col13, col14, col15, col16, col17 = st.columns(6)
                    with col12:
                        train_window_days = st.number_input(
                            "Train window days",
                            min_value=30,
                            max_value=3650,
                            value=504,
                            step=21,
                            help=(
                                "Length of each rolling training window. "
                                "Example: 504 trading days is about 2 years."
                            ),
                        )
                    with col13:
                        test_window_days = st.number_input(
                            "Test window days",
                            min_value=5,
                            max_value=730,
                            value=63,
                            step=7,
                            help=(
                                "Out-of-sample test window after each retrain. "
                                "Example: 63 trading days is roughly 1 quarter."
                            ),
                        )
                    with col14:
                        step_days = st.number_input(
                            "Step days",
                            min_value=1,
                            max_value=365,
                            value=21,
                            step=1,
                            help=(
                                "How far windows move forward each cycle. "
                                "Example: 21 means about one trading month per step."
                            ),
                        )
                    with col15:
                        min_pattern_rows_bt = st.number_input(
                            "Min rows per pattern",
                            min_value=10,
                            max_value=100000,
                            value=settings.min_pattern_count,
                            step=10,
                            help="Skip sparse patterns inside each rolling window to reduce noisy fits.",
                        )
                    with col16:
                        bt_max_windows = st.number_input(
                            "Max windows/pattern (0 = no cap)",
                            min_value=0,
                            max_value=20000,
                            value=80,
                            step=10,
                            help="Hard cap on windows processed per pattern to control runtime.",
                        )
                    with col17:
                        bt_max_train_rows = st.number_input(
                            "Max train rows/window (0 = no cap)",
                            min_value=0,
                            max_value=20_000_000,
                            value=120000,
                            step=10000,
                            help="Cap rows used in each retraining window for speed/memory control.",
                        )
                selected_model_files: list[str] = []
            else:
                train_window_days = 504
                test_window_days = 63
                step_days = 21
                min_pattern_rows_bt = settings.min_pattern_count
                bt_max_windows = 0
                bt_max_train_rows = 0
                selected_model_files = []

            with st.expander("Pattern/model filters", expanded=False):
                pattern_options = _pattern_names(models)
                selected_patterns = st.multiselect(
                    "Filter patterns (optional)",
                    options=pattern_options,
                    default=[],
                    help=(
                        "Only backtest selected candlestick patterns. "
                        "Leave empty to evaluate all available patterns."
                    ),
                )
                if bt_mode != "walk_forward_retrain":
                    model_options = _model_files(models.loc[models["interval"] == bt_interval]) if not models.empty else []
                    selected_model_files = st.multiselect(
                        "Filter model files (optional)",
                        options=model_options,
                        default=[],
                        help=(
                            "Limit backtest to specific saved model files. "
                            "Useful for comparing named experiments only."
                        ),
                    )

            submitted = st.form_submit_button("Run Backtest", width="stretch")

        if submitted:
            if bt_mode == "walk_forward_retrain" and int(step_days) < int(test_window_days):
                st.info(
                    "Step days is smaller than test window days. "
                    "For leakage-safe walk-forward scoring, step is auto-clamped to test window days."
                )
            if int(latency_bars) == 0:
                st.warning(
                    "Latency bars is set to 0. This assumes same-bar execution and can overstate backtest quality. "
                    "Set latency to 1+ for stricter realism."
                )

            def _task() -> tuple[pd.DataFrame, pd.DataFrame]:
                backtest_kwargs: dict[str, Any] = {
                    "dataset_name": bt_dataset,
                    "interval": bt_interval,
                    "mode": bt_mode,
                    "long_threshold": (
                        float(rec_long)
                        if (not use_model_thresholds and use_recommended_thresholds)
                        else (float(long_threshold) if long_threshold is not None else None)
                    ),
                    "short_threshold": (
                        float(rec_short)
                        if (not use_model_thresholds and use_recommended_thresholds)
                        else (float(short_threshold) if short_threshold is not None else None)
                    ),
                    "fee_bps": float(fee_bps),
                    "include_patterns": set(selected_patterns) or None,
                    "include_model_files": set(selected_model_files) or None,
                    "use_model_thresholds": bool(use_model_thresholds),
                    "spread_bps": float(spread_bps),
                    "slippage_bps": float(slippage_bps),
                    "short_borrow_bps_per_day": float(short_borrow_bps_per_day),
                    "latency_bars": int(latency_bars),
                    "embargo_bars": int(embargo_bars),
                    "train_window_days": int(train_window_days),
                    "test_window_days": int(test_window_days),
                    "step_days": int(step_days),
                    "min_pattern_rows": int(min_pattern_rows_bt),
                    "fast_mode": bool(bt_fast_mode),
                    "parallel_models": int(bt_parallel_models),
                    "max_eval_rows_per_pattern": None if int(bt_max_eval_rows) <= 0 else int(bt_max_eval_rows),
                    "max_windows_per_pattern": None if int(bt_max_windows) <= 0 else int(bt_max_windows),
                    "max_train_rows_per_window": None if int(bt_max_train_rows) <= 0 else int(bt_max_train_rows),
                    "include_portfolio": bool(bt_include_portfolio),
                    "portfolio_top_k_per_side": int(bt_portfolio_top_k),
                    "portfolio_max_gross_exposure": float(bt_portfolio_gross),
                    "portfolio_pattern_selection": str(bt_portfolio_pattern_selection),
                    "portfolio_best_patterns_top_n": int(bt_portfolio_best_patterns_top_n),
                    "portfolio_min_pattern_trades": int(bt_portfolio_min_pattern_trades),
                    "portfolio_min_pattern_win_rate_trade": float(bt_portfolio_min_pattern_win_rate_trade),
                    "portfolio_min_abs_score": float(bt_portfolio_min_abs_score),
                    "portfolio_rebalance_every_n_bars": int(bt_portfolio_rebalance_every_n_bars),
                    "portfolio_symbol_cooldown_bars": int(bt_portfolio_symbol_cooldown_bars),
                    "portfolio_volatility_scaling": bool(bt_portfolio_volatility_scaling),
                    "portfolio_max_symbol_weight": float(bt_portfolio_max_symbol_weight),
                    "include_spread_strategies": bool(bt_include_spread_strategies),
                    "spread_lookback_bars": int(bt_spread_lookback_bars),
                    "spread_top_components": int(bt_spread_top_components),
                    "spread_min_edge": float(bt_spread_min_edge),
                    "spread_switch_cost_bps": float(bt_spread_switch_cost_bps),
                    "spread_include_neutral_overlay": bool(bt_spread_include_neutral_overlay),
                    "spread_include_regime_switch": bool(bt_spread_include_regime_switch),
                    "spread_target_vol_annual": float(bt_spread_target_vol_annual),
                    "initial_investment": float(bt_initial_investment),
                }
                return _call_with_supported_kwargs(service.backtest_with_details, backtest_kwargs)

            with st.spinner("Running backtest..."):
                bt, bt_curve = _run_inline(_task)
            st.session_state["last_backtest_table"] = bt
            st.session_state["last_backtest_curve"] = bt_curve

        if "last_backtest_table" in st.session_state:
            bt = st.session_state["last_backtest_table"]
            if bt.empty:
                st.warning("No backtest rows returned for the selected filters.")
            else:
                if {"backtest_start_datetime", "backtest_end_datetime"}.issubset(bt.columns):
                    bt_start = pd.to_datetime(bt["backtest_start_datetime"], utc=True, errors="coerce").min()
                    bt_end = pd.to_datetime(bt["backtest_end_datetime"], utc=True, errors="coerce").max()
                    if pd.notna(bt_start) and pd.notna(bt_end):
                        st.caption(f"Backtest range: {bt_start.isoformat()} to {bt_end.isoformat()}")

                preferred_cols = [
                    "pattern",
                    "model_file",
                    "backtest_start_datetime",
                    "backtest_end_datetime",
                    "windows_used",
                    "trades",
                    "portfolio_selection_mode",
                    "portfolio_pattern_count",
                    "portfolio_patterns_used",
                    "win_rate",
                    "win_rate_period",
                    "win_rate_trade",
                    "avg_trade_return",
                    "cumulative_return",
                    "sharpe",
                    "sortino",
                    "max_drawdown",
                    "profit_factor",
                    "annualized_return",
                    "annualized_volatility",
                ]
                ordered_cols = [c for c in preferred_cols if c in bt.columns] + [c for c in bt.columns if c not in preferred_cols]
                st.dataframe(bt[ordered_cols], width="stretch")

                bt_curve = st.session_state.get("last_backtest_curve")
                if isinstance(bt_curve, pd.DataFrame) and not bt_curve.empty:
                    st.write("### Hypothetical Equity Curve")
                    bt_curve_work = bt_curve.copy()
                    bt_curve_work["datetime"] = pd.to_datetime(bt_curve_work["datetime"], utc=True, errors="coerce")
                    bt_curve_work = bt_curve_work.dropna(subset=["datetime"]).copy()
                    bt_curve_work["curve_key"] = (
                        bt_curve_work["model_file"].astype(str) + " | " + bt_curve_work["pattern"].astype(str)
                    )
                    curve_keys = sorted(bt_curve_work["curve_key"].dropna().astype(str).unique().tolist())
                    if curve_keys:
                        selected_curve_key = st.selectbox(
                            "Curve",
                            options=curve_keys,
                            index=0,
                            key="backtest_curve_selector",
                        )
                        curve_selected = (
                            bt_curve_work.loc[bt_curve_work["curve_key"] == selected_curve_key]
                            .sort_values("datetime")
                            .copy()
                        )
                        if not curve_selected.empty:
                            if "curve_variant" not in curve_selected.columns:
                                curve_selected["curve_variant"] = "ml_model"
                            curve_selected["curve_variant"] = curve_selected["curve_variant"].astype(str).str.strip().replace("", "ml_model")
                            variant_label_map = {
                                "ml_model": "ML Strategy",
                                "baseline_blind_pattern": "Blind Pattern",
                                "baseline_universe_eqw": "Universe Benchmark (Equal-Weight)",
                            }
                            curve_selected["curve_label"] = curve_selected["curve_variant"].map(variant_label_map).fillna(
                                curve_selected["curve_variant"]
                            )

                            chart_frame = (
                                curve_selected.pivot_table(
                                    index="datetime",
                                    columns="curve_label",
                                    values="equity_value",
                                    aggfunc="last",
                                )
                                .sort_index()
                            )
                            preferred_order = [
                                "ML Strategy",
                                "Blind Pattern",
                                "Universe Benchmark (Equal-Weight)",
                            ]
                            ordered_cols = [c for c in preferred_order if c in chart_frame.columns] + [
                                c for c in chart_frame.columns if c not in preferred_order
                            ]
                            st.line_chart(chart_frame[ordered_cols], width="stretch")

                            ml_curve = curve_selected.loc[curve_selected["curve_variant"] == "ml_model"].sort_values("datetime")
                            focus_curve = ml_curve if not ml_curve.empty else curve_selected.sort_values("datetime")
                            start_equity = (
                                float(focus_curve["initial_investment"].iloc[0])
                                if "initial_investment" in focus_curve.columns
                                else float(focus_curve["equity_value"].iloc[0])
                            )
                            end_equity = float(focus_curve["equity_value"].iloc[-1])
                            total_return = float(focus_curve["cumulative_return"].iloc[-1])
                            st.caption(
                                f"ML Strategy | Start: ${start_equity:,.2f} | End: ${end_equity:,.2f} | Total return: {total_return * 100:.2f}%"
                            )

                            variant_returns: list[str] = []
                            for label in ordered_cols:
                                label_slice = curve_selected.loc[curve_selected["curve_label"] == label].sort_values("datetime")
                                if label_slice.empty:
                                    continue
                                label_ret = float(label_slice["cumulative_return"].iloc[-1])
                                variant_returns.append(f"{label}: {label_ret * 100:.2f}%")
                            if variant_returns:
                                st.caption("Comparison returns | " + " | ".join(variant_returns))

        st.write("### Optimize Saved-Model Thresholds")
        st.caption("Return-first threshold search over strict OOS data for currently selected dataset/interval.")
        with st.form("threshold_optimizer_form"):
            st.write("#### Core")
            colto1, colto2, colto3, colto4, colto5 = st.columns(5)
            with colto1:
                to_dataset = st.selectbox(
                    "Dataset",
                    options=dataset_choices,
                    index=0,
                    key="to_dataset",
                    help=(
                        "Dataset used to tune long/short thresholds on strict out-of-sample data."
                    ),
                )
            with colto2:
                to_interval = st.selectbox(
                    "Interval",
                    options=INTERVAL_OPTIONS,
                    index=0,
                    key="to_interval",
                    help="Only saved models matching this interval are optimized.",
                )
            with colto3:
                to_min_trades = st.number_input(
                    "Min trades",
                    min_value=1,
                    max_value=200000,
                    value=40,
                    step=5,
                    help=(
                        "Require at least this many trades before accepting optimized thresholds. "
                        "Higher values reduce unstable overfit threshold picks."
                    ),
                )
            with colto4:
                to_parallel = st.number_input(
                    "Parallel workers",
                    min_value=1,
                    max_value=64,
                    value=4,
                    step=1,
                    help=(
                        f"{PARALLEL_WORKER_HELP} "
                        "For threshold search, 4-8 is a common starting range."
                    ),
                )
            with colto5:
                to_max_eval = st.number_input(
                    "Max eval rows/pattern",
                    min_value=1000,
                    max_value=20_000_000,
                    value=250000,
                    step=10000,
                    help=(
                        "Row cap for strict OOS evaluation per pattern during threshold search. "
                        "Increase for final runs, lower for faster iteration."
                    ),
                )

            with st.expander("Execution assumptions", expanded=False):
                colto6, colto7, colto8, colto9, colto10 = st.columns(5)
                with colto6:
                    to_fee = st.number_input(
                        "Fee bps",
                        min_value=0.0,
                        max_value=1000.0,
                        value=1.0,
                        step=0.1,
                        key="to_fee",
                        help="Per-trade fee in bps (1 bps = 0.01%) applied during threshold search.",
                    )
                with colto7:
                    to_spread = st.number_input(
                        "Spread bps",
                        min_value=0.0,
                        max_value=1000.0,
                        value=0.0,
                        step=0.1,
                        key="to_spread",
                        help="Bid/ask spread assumption in bps for simulated execution.",
                    )
                with colto8:
                    to_slip = st.number_input(
                        "Slippage bps",
                        min_value=0.0,
                        max_value=1000.0,
                        value=0.0,
                        step=0.1,
                        key="to_slip",
                        help="Extra slippage/market-impact cost in bps beyond spread.",
                    )
                with colto9:
                    to_borrow = st.number_input(
                        "Short borrow bps/day",
                        min_value=0.0,
                        max_value=1000.0,
                        value=0.0,
                        step=0.1,
                        key="to_borrow",
                        help="Daily short borrow cost in bps/day used in threshold evaluation.",
                    )
                with colto10:
                    to_latency = st.number_input(
                        "Latency bars",
                        min_value=0,
                        max_value=100,
                        value=1,
                        step=1,
                        key="to_latency",
                        help="Signal-to-execution delay in bars; use 1+ for more realistic fills.",
                    )

            with st.expander("Model filters and persistence", expanded=False):
                to_patterns_raw = st.text_input(
                    "Pattern filter (optional, comma-separated)",
                    value="",
                    help=(
                        "Only optimize selected patterns. "
                        "Example: `doji,hammer,bearish_engulfing`."
                    ),
                    key="to_patterns_raw",
                )
                to_persist = st.checkbox(
                    "Persist tuned thresholds to model files",
                    value=True,
                    help=(
                        "Save optimized thresholds into each model file so scanner/backtest can reuse them automatically."
                    ),
                )
            to_submitted = st.form_submit_button("Run Threshold Optimizer", width="stretch")

        if to_submitted:
            to_patterns = {p.strip() for p in str(to_patterns_raw).split(",") if p.strip()} or None

            def _threshold_task() -> pd.DataFrame:
                return service.optimize_model_thresholds_from_backtest(
                    dataset_name=to_dataset,
                    interval=to_interval,
                    fee_bps=float(to_fee),
                    spread_bps=float(to_spread),
                    slippage_bps=float(to_slip),
                    short_borrow_bps_per_day=float(to_borrow),
                    latency_bars=int(to_latency),
                    include_patterns=to_patterns,
                    min_trades=int(to_min_trades),
                    max_eval_rows_per_pattern=int(to_max_eval),
                    parallel_models=int(to_parallel),
                    persist=bool(to_persist),
                )

            with st.spinner("Optimizing thresholds..."):
                to_table = _run_inline(_threshold_task)
            st.session_state["threshold_optimizer_table"] = to_table
            _load_model_registry.clear()

        if "threshold_optimizer_table" in st.session_state:
            to_table = st.session_state["threshold_optimizer_table"]
            if isinstance(to_table, pd.DataFrame) and not to_table.empty:
                st.success(f"Threshold optimizer processed {len(to_table)} models")
                st.dataframe(to_table, width="stretch")
            else:
                st.info("No models matched the threshold optimizer filters.")

with tab_scanner:
    st.subheader("Run Scanner")
    models = _load_model_registry()

    with st.form("scan_form"):
        st.write("#### Core")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            scan_interval = st.selectbox(
                "Interval",
                options=INTERVAL_OPTIONS,
                index=0,
                help=(
                    "Interval used for today's scan signals. "
                    "Choose the same interval your best models were trained on."
                ),
            )
        with col2:
            scan_years = st.slider(
                "History years for scan context",
                min_value=1,
                max_value=10,
                value=2,
                help=(
                    "How much recent history to load for feature creation before scoring current signals. "
                    "More years may improve context but increases runtime."
                ),
            )
        with col3:
            scan_top_n = st.slider(
                "Top N",
                min_value=1,
                max_value=300,
                value=settings.top_n_signals,
                help=(
                    "Maximum number of signals returned after ranking by model confidence/score."
                ),
            )
        with col4:
            scan_min_conf = st.slider(
                "Min confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help=(
                    "Minimum confidence score (0-1) required to keep a signal. "
                    "Example: 0.60 filters out lower-conviction trades."
                ),
            )

        with st.expander("Data refresh and alternative inputs", expanded=False):
            refresh_prices = st.checkbox(
                "Refresh prices before scanning",
                value=True,
                help="Re-pulls recent bars before scanning so signals use freshest data.",
            )
            pol_path_raw = st.text_input(
                "Politician trades CSV (optional)",
                value="",
                help=(
                    "Optional local CSV path for politician-trade features during scan build. "
                    "If blank and FMP API is configured, scan uses auto-fetched congressional trades from FMP."
                ),
            )

        with st.expander("Thresholds and model filters", expanded=False):
            col5, col6, col7 = st.columns(3)
            with col5:
                scan_use_model_thresholds = st.checkbox(
                    "Use tuned model thresholds",
                    value=True,
                    help=(
                        "Use each model's stored long/short thresholds from optimizer output. "
                        "Disable to force one global threshold pair."
                    ),
                )
            with col6:
                scan_long_threshold = st.number_input(
                    "Long threshold override",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.65,
                    step=0.01,
                    help=(
                        "Global long trigger (used only if tuned thresholds are disabled). "
                        "Example: 0.58 means only stronger long signals pass."
                    ),
                )
            with col7:
                scan_short_threshold = st.number_input(
                    "Short threshold override",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.35,
                    step=0.01,
                    help=(
                        "Global short trigger (used only if tuned thresholds are disabled). "
                        "Example: 0.42 requires stronger downside conviction."
                    ),
                )

            model_options = _model_files(models.loc[models["interval"] == scan_interval]) if not models.empty else []
            pattern_options = _pattern_names(models)
            scan_model_files = st.multiselect(
                "Filter model files (optional)",
                options=model_options,
                default=[],
                help="Optional subset of model artifacts to score in this scan run.",
            )
            scan_patterns = st.multiselect(
                "Filter patterns (optional)",
                options=pattern_options,
                default=[],
                help="Optional subset of candlestick patterns to score (leave blank for all).",
            )

        submitted = st.form_submit_button("Run Scanner", width="stretch")

    if submitted:
        pol_path = Path(pol_path_raw).expanduser() if pol_path_raw else None

        def _task() -> pd.DataFrame:
            return service.scan(
                interval=scan_interval,
                years=int(scan_years),
                top_n=int(scan_top_n),
                refresh_prices=refresh_prices,
                politician_trades_csv=pol_path,
                include_patterns=set(scan_patterns) or None,
                include_model_files=set(scan_model_files) or None,
                min_confidence=float(scan_min_conf),
                universe=selected_universe,
                use_model_thresholds=bool(scan_use_model_thresholds),
                long_threshold=None if scan_use_model_thresholds else float(scan_long_threshold),
                short_threshold=None if scan_use_model_thresholds else float(scan_short_threshold),
            )

        with st.spinner("Running scanner..."):
            scan_table = _run_inline(_task)
        st.session_state["last_scan_table"] = scan_table

    if "last_scan_table" in st.session_state:
        scan_table = st.session_state["last_scan_table"]
        if scan_table.empty:
            st.warning("No scan signals for current filters.")
        else:
            st.dataframe(scan_table, width="stretch")

with tab_models:
    st.subheader("Historical Model Registry")
    models = _load_model_registry()
    if models.empty:
        st.info("No saved models yet.")
    else:
        work = models.copy()
        if "modified_utc" in work.columns:
            work["modified_utc"] = pd.to_datetime(work["modified_utc"], utc=True, errors="coerce")
        for col in ("roc_auc", "train_rows", "test_rows", "size_kb", "horizon_bars"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

        st.write("### Registry Filters")
        mf1, mf2, mf3, mf4 = st.columns(4)
        with mf1:
            model_search = st.text_input(
                "Search text",
                value="",
                help="Filter by model file, model name, or pattern.",
            )
        with mf2:
            interval_opts = sorted(work["interval"].dropna().astype(str).unique().tolist()) if "interval" in work.columns else []
            selected_intervals = st.multiselect(
                "Intervals",
                options=interval_opts,
                default=[],
                help="Optional interval filter.",
            )
        with mf3:
            pattern_opts = sorted(work["pattern"].dropna().astype(str).unique().tolist()) if "pattern" in work.columns else []
            selected_patterns = st.multiselect(
                "Patterns",
                options=pattern_opts,
                default=[],
                help="Optional pattern filter.",
            )
        with mf4:
            model_name_opts = sorted(work["model_name"].dropna().astype(str).unique().tolist()) if "model_name" in work.columns else []
            selected_model_names = st.multiselect(
                "Model names",
                options=model_name_opts,
                default=[],
                help="Optional run-name filter.",
            )

        mf5, mf6, mf7, mf8 = st.columns(4)
        with mf5:
            latest_only = st.checkbox(
                "Latest only per group",
                value=False,
                help="Keep only the newest model for each (pattern, interval, horizon, model_name) group.",
            )
        with mf6:
            min_roc_auc_filter = st.number_input(
                "Min ROC-AUC",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                help="Filter out lower-ROC-AUC models (0 keeps all).",
            )
        with mf7:
            sort_field_options = [c for c in ["modified_utc", "roc_auc", "train_rows", "size_kb"] if c in work.columns]
            sort_field = st.selectbox(
                "Sort by",
                options=sort_field_options,
                index=0 if sort_field_options else None,
            )
        with mf8:
            sort_desc = st.checkbox("Sort descending", value=True)

        filtered = work.copy()
        if model_search.strip():
            q = model_search.strip().lower()
            tokens = pd.Series([""] * len(filtered), index=filtered.index, dtype="string")
            for col in ("model_file", "model_name", "pattern"):
                if col in filtered.columns:
                    tokens = tokens + " " + filtered[col].astype("string").fillna("")
            filtered = filtered.loc[tokens.str.lower().str.contains(q, na=False)].copy()
        if selected_intervals and "interval" in filtered.columns:
            filtered = filtered.loc[filtered["interval"].astype(str).isin(selected_intervals)].copy()
        if selected_patterns and "pattern" in filtered.columns:
            filtered = filtered.loc[filtered["pattern"].astype(str).isin(selected_patterns)].copy()
        if selected_model_names and "model_name" in filtered.columns:
            filtered = filtered.loc[filtered["model_name"].astype(str).isin(selected_model_names)].copy()
        if float(min_roc_auc_filter) > 0.0 and "roc_auc" in filtered.columns:
            filtered = filtered.loc[pd.to_numeric(filtered["roc_auc"], errors="coerce") >= float(min_roc_auc_filter)].copy()

        if latest_only and not filtered.empty:
            group_cols = [c for c in ("pattern", "interval", "horizon_bars", "model_name") if c in filtered.columns]
            if group_cols and "modified_utc" in filtered.columns:
                filtered = filtered.sort_values("modified_utc", ascending=False).drop_duplicates(subset=group_cols, keep="first").copy()

        if not filtered.empty and sort_field in filtered.columns:
            filtered = filtered.sort_values(sort_field, ascending=not bool(sort_desc), na_position="last").copy()

        st.write("### Registry Summary")
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Models (filtered)", value=int(len(filtered)))
        sm2.metric("Patterns", value=int(filtered["pattern"].nunique()) if "pattern" in filtered.columns and not filtered.empty else 0)
        if "model_name" in filtered.columns and not filtered.empty:
            names = filtered["model_name"].fillna("").astype(str)
            unique_names = names.replace("", "__unnamed__").nunique()
        else:
            unique_names = 0
        sm3.metric("Model Names", value=int(unique_names))
        avg_roc = float(pd.to_numeric(filtered.get("roc_auc"), errors="coerce").mean()) if not filtered.empty else np.nan
        sm4.metric("Avg ROC-AUC", value=f"{avg_roc:.4f}" if np.isfinite(avg_roc) else "n/a")

        if filtered.empty:
            st.info("No models match current filters.")
        else:
            run_summary = filtered.copy()
            if "model_name" in run_summary.columns:
                run_summary["model_name"] = run_summary["model_name"].fillna("").astype(str).replace("", "__unnamed__")
            group_cols = [c for c in ("model_name", "interval", "horizon_bars") if c in run_summary.columns]
            if group_cols:
                group_ops: dict[str, tuple[str, str]] = {"model_count": ("model_file", "count")}
                if "pattern" in run_summary.columns:
                    group_ops["pattern_count"] = ("pattern", "nunique")
                if "roc_auc" in run_summary.columns:
                    group_ops["mean_roc_auc"] = ("roc_auc", "mean")
                if "train_rows" in run_summary.columns:
                    group_ops["median_train_rows"] = ("train_rows", "median")
                if "modified_utc" in run_summary.columns:
                    group_ops["latest_modified_utc"] = ("modified_utc", "max")
                grouped = run_summary.groupby(group_cols, dropna=False).agg(**group_ops).reset_index()
                if "latest_modified_utc" in grouped.columns:
                    grouped = grouped.sort_values("latest_modified_utc", ascending=False)
                    grouped["latest_modified_utc"] = pd.to_datetime(grouped["latest_modified_utc"], utc=True, errors="coerce").dt.strftime(
                        "%Y-%m-%d %H:%M:%S UTC"
                    )
                st.write("### Run-Level Summary")
                st.dataframe(grouped, width="stretch")

            display_cols = [
                "model_file",
                "model_name",
                "pattern",
                "interval",
                "horizon_bars",
                "roc_auc",
                "train_rows",
                "test_rows",
                "size_kb",
                "modified_utc",
                "tuned_long_threshold",
                "tuned_short_threshold",
                "meta_filter_enabled",
                "importance_file",
            ]
            display_cols = [c for c in display_cols if c in filtered.columns]
            model_table = filtered[display_cols].copy()
            if "modified_utc" in model_table.columns:
                model_table["modified_utc"] = pd.to_datetime(model_table["modified_utc"], utc=True, errors="coerce").dt.strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
            st.write("### Model Table")
            st.dataframe(model_table, width="stretch")

            inspect_options = _model_files(filtered)
            inspect_idx = 0
            if "model_details" in st.session_state:
                selected_details_file = str(st.session_state["model_details"].get("model_file", ""))
                if selected_details_file in inspect_options:
                    inspect_idx = inspect_options.index(selected_details_file)
            model_file = st.selectbox(
                "Inspect model file",
                options=inspect_options,
                index=inspect_idx,
                help=(
                    "Choose a saved model artifact to inspect metadata, thresholds, and feature importance."
                ),
            )
            top_n_imp = st.slider(
                "Top importance rows",
                min_value=5,
                max_value=100,
                value=30,
                help=(
                    "Number of highest-importance features shown for the selected model. "
                    "Example: 30 gives a concise diagnostic view."
                ),
            )
            if st.button("Load Model Details", width="stretch"):
                details = service.model_details(model_file=model_file, top_n_importance=int(top_n_imp))
                st.session_state["model_details"] = details

            st.write("### Delete Single Model")
            with st.form("delete_model_form"):
                delete_model_file = st.selectbox(
                    "Model file to delete",
                    options=inspect_options,
                    key="delete_model_file",
                    help="Permanently delete the selected saved model file from local storage.",
                )
                delete_model_confirm = st.text_input(
                    "Type model filename to confirm",
                    value="",
                    key="delete_model_confirm",
                    help="Safety check: must exactly match the model filename before deletion.",
                )
                delete_model_submitted = st.form_submit_button("Delete Model", width="stretch")

            if delete_model_submitted:
                if delete_model_confirm != delete_model_file:
                    st.error("Confirmation mismatch. Model was not deleted.")
                else:
                    result = service.delete_model(delete_model_file)
                    if bool(result.get("deleted")):
                        st.success(f"Deleted model `{delete_model_file}`")
                        if "model_details" in st.session_state and st.session_state["model_details"].get("model_file") == delete_model_file:
                            st.session_state.pop("model_details", None)
                        _load_model_registry.clear()
                        st.rerun()
                    else:
                        st.warning(f"Model `{delete_model_file}` was not found.")

            st.write("### Bulk Delete Models")
            st.caption("Select models with checkboxes, then type `delete` to confirm bulk deletion.")
            bulk_cols = [c for c in ["model_file", "model_name", "pattern", "interval", "horizon_bars", "roc_auc", "train_rows", "modified_utc"] if c in filtered.columns]
            bulk_frame = filtered[bulk_cols].copy()
            if "modified_utc" in bulk_frame.columns:
                bulk_frame["modified_utc"] = pd.to_datetime(bulk_frame["modified_utc"], utc=True, errors="coerce").dt.strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
            bulk_frame.insert(0, "delete", False)
            edited_bulk = st.data_editor(
                bulk_frame,
                hide_index=True,
                disabled=[c for c in bulk_frame.columns if c != "delete"],
                column_config={
                    "delete": st.column_config.CheckboxColumn(
                        "Delete",
                        help="Mark this model for deletion.",
                    )
                },
                key="bulk_delete_editor",
            )
            selected_for_delete: list[str] = []
            if isinstance(edited_bulk, pd.DataFrame) and not edited_bulk.empty:
                selected_for_delete = (
                    edited_bulk.loc[edited_bulk["delete"].astype(bool), "model_file"].dropna().astype(str).tolist()
                )
            st.write(f"Selected models: `{len(selected_for_delete)}`")

            with st.form("bulk_delete_models_form"):
                bulk_delete_confirm = st.text_input(
                    'Type "delete" to confirm bulk delete',
                    value="",
                    help="Safety check: type delete exactly before the selected models are removed.",
                )
                bulk_delete_submitted = st.form_submit_button("Delete Selected Models", width="stretch")

            if bulk_delete_submitted:
                if not selected_for_delete:
                    st.warning("No models selected for bulk delete.")
                elif bulk_delete_confirm.strip().lower() != "delete":
                    st.error('Confirmation mismatch. Type "delete" exactly to proceed.')
                else:
                    result = service.delete_models(selected_for_delete)
                    deleted_models = [str(x) for x in list(result.get("deleted_models", []))]
                    missing_models = [str(x) for x in list(result.get("missing_or_invalid", []))]
                    st.success(
                        f"Bulk delete finished. Deleted `{int(result.get('deleted_count', 0))}` / "
                        f"`{int(result.get('requested_count', 0))}` selected models."
                    )
                    if missing_models:
                        st.warning(f"Missing or invalid: {', '.join(missing_models[:10])}")
                    if "model_details" in st.session_state:
                        current_file = str(st.session_state["model_details"].get("model_file", ""))
                        if current_file in set(deleted_models):
                            st.session_state.pop("model_details", None)
                    _load_model_registry.clear()
                    st.rerun()

    if "model_details" in st.session_state:
        details = st.session_state["model_details"]
        st.write("### Model Metadata")
        st.json(
            {
                "model_file": details.get("model_file"),
                "pattern": details.get("pattern"),
                "interval": details.get("interval"),
                "horizon_bars": details.get("horizon_bars"),
                "model_name": details.get("model_name"),
                "metrics": details.get("metrics"),
                "train_rows": details.get("train_rows"),
                "test_rows": details.get("test_rows"),
                "feature_count": details.get("feature_count"),
                "tuned_thresholds": details.get("tuned_thresholds"),
                "probability_calibration": details.get("probability_calibration"),
                "meta_filter": details.get("meta_filter"),
            }
        )

        importance = pd.DataFrame(details.get("importance", []))
        if not importance.empty:
            st.write("### Feature Importance")
            st.dataframe(importance, width="stretch")

with tab_analytics:
    st.subheader("Analytics")

    datasets = _load_dataset_registry()
    dataset_choices = _dataset_names(datasets)

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Pattern Coverage")
        if dataset_choices:
            coverage_dataset = st.selectbox(
                "Dataset for coverage",
                options=dataset_choices,
                key="coverage_dataset",
                help=(
                    "Dataset used to show how frequently each candlestick pattern appears. "
                    "Useful for spotting patterns with weak sample size."
                ),
            )
            coverage = service.coverage(coverage_dataset)
            if coverage.empty:
                st.info("No coverage data available.")
            else:
                st.dataframe(coverage, width="stretch")
                st.bar_chart(coverage.set_index("pattern")["count"])
        else:
            st.info("No datasets available.")

    with col2:
        st.write("#### Feature Importance Summary")
        imp_interval = st.selectbox(
            "Interval",
            options=["all", *INTERVAL_OPTIONS],
            index=1,
            help=(
                "Filter feature-importance summaries by interval, or choose `all` to compare across intervals."
            ),
        )
        imp_top_n = st.slider(
            "Top N per pattern",
            min_value=5,
            max_value=100,
            value=20,
            help=(
                "Maximum number of top features retained per pattern in the summary."
            ),
        )
        table = service.feature_importance(
            interval=None if imp_interval == "all" else imp_interval,
            horizon_bars=settings.forward_horizon_bars,
            top_n_per_pattern=int(imp_top_n),
        )
        if table.empty:
            st.info("No feature importance reports yet.")
        else:
            st.dataframe(table, width="stretch")

    st.write("#### Interval Sweep")
    with st.form("sweep_form"):
        sweep_intervals = st.multiselect(
            "Intervals",
            options=INTERVAL_OPTIONS,
            default=["1d", "1h", "30m", "15m"],
            help=(
                "Intervals to benchmark end-to-end (build, train, backtest) to find where signal is strongest."
            ),
        )
        sweep_years = st.slider(
            "Years",
            min_value=1,
            max_value=15,
            value=min(10, settings.history_years),
            help=(
                "Historical depth used for each interval in the sweep. "
                "Example: 10 means each interval run uses ~10 years of history."
            ),
        )
        sweep_dataset_prefix = st.text_input(
            "Dataset prefix",
            value="model_dataset",
            help=(
                "Prefix for auto-generated sweep dataset names, e.g. `model_dataset_1d`, `model_dataset_1h`."
            ),
        )
        sweep_refresh = st.checkbox(
            "Refresh prices",
            value=False,
            help="Refresh raw price data before each interval run in the sweep.",
        )
        submitted = st.form_submit_button("Run Interval Sweep", width="stretch")

    if submitted:

        def _task() -> pd.DataFrame:
            return service.sweep_intervals(
                intervals=sweep_intervals,
                years=sweep_years,
                refresh_prices=sweep_refresh,
                base_dataset_name=sweep_dataset_prefix,
                universe=selected_universe,
            )

        with st.spinner("Running interval sweep..."):
            sweep = _run_inline(_task)
        st.session_state["last_sweep_table"] = sweep

    if "last_sweep_table" in st.session_state:
        sweep = st.session_state["last_sweep_table"]
        if sweep.empty:
            st.warning("No sweep results returned.")
        else:
            st.dataframe(sweep, width="stretch")

with tab_config:
    st.subheader("Runtime Configuration")
    cfg = service.settings_dict()
    cfg["fmp_api_key_set"] = bool(settings.fmp_api_key)
    cfg["fred_api_key_set"] = bool(settings.fred_api_key)
    cfg["polygon_api_key_set"] = bool(settings.polygon_api_key)
    cfg.pop("fmp_api_key", None)
    cfg.pop("fred_api_key", None)
    cfg.pop("polygon_api_key", None)
    st.json(cfg)

    st.caption("Set environment variables in .env to control providers and credentials.")
