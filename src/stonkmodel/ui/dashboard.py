from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

from stonkmodel.config import get_settings
from stonkmodel.pipeline import StonkService
from stonkmodel.ui.jobs import AsyncJobManager

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

if "job_manager" not in st.session_state:
    st.session_state["job_manager"] = AsyncJobManager(settings.data_dir / "jobs", max_workers=1)
job_manager: AsyncJobManager = st.session_state["job_manager"]

st.title("StonkModel Control Center")
st.caption("Dataset updates, training, model registry, backtests, scanner, and analytics")


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


def _run_or_queue(name: str, run_async: bool, fn: Callable[..., Any]) -> tuple[str | None, Any | None]:
    if run_async:
        job_id = job_manager.submit(name=name, fn=fn)
        return job_id, None
    if _accepts_progress_callback(fn):
        progress_slot = st.empty()
        bar = progress_slot.progress(0.0, text="0.0% - Starting")

        def _inline_progress(pct: float, message: str) -> None:
            bar.progress(max(0.0, min(1.0, float(pct) / 100.0)), text=f"{float(pct):.1f}% - {message}")

        try:
            result = fn(_inline_progress)
        finally:
            progress_slot.empty()
        return None, result
    return None, fn()


def _show_job_message(job_id: str) -> None:
    st.success(f"Job queued: `{job_id}`. Check the Jobs tab for progress and results.")


with st.sidebar:
    st.header("Runtime")
    st.write(f"Market data provider: `{settings.market_data_provider}`")
    st.write(f"Fundamentals provider: `{settings.fundamentals_provider}`")
    st.write(f"FMP key configured: `{bool(settings.fmp_api_key)}`")
    st.write(f"Polygon key configured: `{bool(settings.polygon_api_key)}`")

    run_async_jobs = st.checkbox("Run heavy actions as background jobs", value=True)

    default_universe_label = "S&P 100" if settings.universe_source == "sp100" else "S&P 500"
    selected_universe_label = st.selectbox(
        "Universe (for market pulls)",
        options=list(UNIVERSE_LABEL_TO_VALUE.keys()),
        index=list(UNIVERSE_LABEL_TO_VALUE.keys()).index(default_universe_label),
    )
    selected_universe = UNIVERSE_LABEL_TO_VALUE[selected_universe_label]


tab_data, tab_train, tab_backtest, tab_scanner, tab_models, tab_analytics, tab_jobs, tab_config = st.tabs(
    ["Data", "Train", "Backtest", "Scanner", "Models", "Analytics", "Jobs", "Config"]
)

with tab_data:
    st.subheader("Build or Update Dataset")
    with st.form("build_dataset_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            dataset_name = st.text_input("Dataset name", value="model_dataset")
        with col2:
            interval = st.selectbox("Interval", options=INTERVAL_OPTIONS, index=0)
        with col3:
            years_ago_range = st.slider(
                "History window (years ago)",
                min_value=0,
                max_value=25,
                value=(0, min(15, settings.history_years)),
            )
        st.caption("Example: selecting 5 and 10 keeps data between 10 and 5 years ago.")

        col4, col5 = st.columns(2)
        with col4:
            refresh_prices = st.checkbox("Refresh prices from provider", value=True)
        with col5:
            politician_path_raw = st.text_input("Politician trades CSV (optional)", value="")

        submitted = st.form_submit_button("Build / Update Dataset", use_container_width=True)

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

        job_id, result_payload = _run_or_queue("build_dataset", run_async_jobs, _task)
        if job_id is not None:
            _show_job_message(job_id)
        else:
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
        st.dataframe(datasets, use_container_width=True)

        choices = _dataset_names(datasets)
        selected = st.selectbox("Inspect dataset", options=choices)
        summary = service.dataset_summary(selected)
        st.json(summary)

        if st.button("Preview first 30 rows", key="preview_dataset_rows"):
            frame = service.dataset_builder.load_dataset(selected).head(30)
            st.dataframe(frame, use_container_width=True)

        st.write("#### Delete Dataset")
        with st.form("delete_dataset_form"):
            delete_dataset_name = st.selectbox("Dataset to delete", options=choices, key="delete_dataset_name")
            delete_dataset_confirm = st.text_input("Type dataset name to confirm", value="", key="delete_dataset_confirm")
            delete_dataset_submitted = st.form_submit_button("Delete Dataset", use_container_width=True)

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
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                train_dataset = st.selectbox("Dataset", options=choices, index=0)
            with col2:
                train_interval = st.selectbox("Interval", options=INTERVAL_OPTIONS, index=0)
            with col3:
                min_pattern_rows = st.number_input(
                    "Min rows per pattern",
                    min_value=10,
                    max_value=100000,
                    value=settings.min_pattern_count,
                )
            with col4:
                train_model_name = st.text_input("Model name (optional)", value="")
            with col5:
                train_fast_mode = st.checkbox("Fast mode", value=True)
            with col6:
                train_parallel_patterns = st.number_input(
                    "Parallel patterns",
                    min_value=1,
                    max_value=16,
                    value=4,
                    step=1,
                )
            train_max_rows_per_pattern = st.number_input(
                "Max rows per pattern (0 = no cap)",
                min_value=0,
                max_value=2_000_000,
                value=120000,
                step=10000,
            )
            submitted = st.form_submit_button("Train Models", use_container_width=True)

        if submitted:

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
                )

            job_id, table = _run_or_queue("train_models", run_async_jobs, _task)
            if job_id is not None:
                _show_job_message(job_id)
            else:
                st.session_state["last_train_table"] = table
                _load_model_registry.clear()

        if "last_train_table" in st.session_state:
            table = st.session_state["last_train_table"]
            if table.empty:
                st.warning("No models trained. Increase history or lower min rows.")
            else:
                st.success(f"Trained {len(table)} models")
                st.dataframe(table, use_container_width=True)

with tab_backtest:
    st.subheader("Backtest Models")
    datasets = _load_dataset_registry()
    models = _load_model_registry()
    dataset_choices = _dataset_names(datasets)

    if not dataset_choices:
        st.info("Build a dataset first.")
    else:
        with st.form("backtest_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                bt_dataset = st.selectbox("Dataset", options=dataset_choices, index=0)
            with col2:
                bt_interval = st.selectbox("Interval", options=INTERVAL_OPTIONS, index=0)
            with col3:
                bt_mode_label = st.selectbox("Mode", options=list(BACKTEST_MODE_LABEL_TO_VALUE.keys()), index=0)
            bt_mode = BACKTEST_MODE_LABEL_TO_VALUE[bt_mode_label]

            col4, col5 = st.columns(2)
            with col4:
                use_model_thresholds = st.checkbox("Use tuned model thresholds", value=(bt_mode == "saved_models"))
            with col5:
                fee_bps = st.number_input("Fee bps", min_value=0.0, max_value=1000.0, value=1.0, step=0.1)

            long_threshold = None
            short_threshold = None
            if not use_model_thresholds:
                col6, col7 = st.columns(2)
                with col6:
                    long_threshold = st.slider("Long threshold", min_value=0.5, max_value=0.95, value=0.55, step=0.01)
                with col7:
                    short_threshold = st.slider("Short threshold", min_value=0.05, max_value=0.5, value=0.45, step=0.01)

            st.write("Execution realism")
            col8, col9, col10, col11 = st.columns(4)
            with col8:
                spread_bps = st.number_input("Spread bps", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
            with col9:
                slippage_bps = st.number_input("Slippage bps", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
            with col10:
                short_borrow_bps_per_day = st.number_input(
                    "Short borrow bps/day",
                    min_value=0.0,
                    max_value=1000.0,
                    value=0.0,
                    step=0.1,
                )
            with col11:
                latency_bars = st.number_input("Latency bars", min_value=0, max_value=100, value=0, step=1)

            st.write("Performance")
            col_fast, col_workers = st.columns(2)
            with col_fast:
                bt_fast_mode = st.checkbox("Fast mode", value=True)
            with col_workers:
                bt_parallel_models = st.number_input(
                    "Parallel model workers",
                    min_value=1,
                    max_value=32,
                    value=4,
                    step=1,
                )
            bt_max_eval_rows = st.number_input(
                "Max eval rows per pattern (saved-model mode, 0 = no cap)",
                min_value=0,
                max_value=20_000_000,
                value=250000,
                step=10000,
            )

            if bt_mode == "walk_forward_retrain":
                st.write("Walk-forward settings")
                col12, col13, col14, col15, col16, col17 = st.columns(6)
                with col12:
                    train_window_days = st.number_input("Train window days", min_value=30, max_value=3650, value=504, step=21)
                with col13:
                    test_window_days = st.number_input("Test window days", min_value=5, max_value=730, value=63, step=7)
                with col14:
                    step_days = st.number_input("Step days", min_value=1, max_value=365, value=21, step=1)
                with col15:
                    min_pattern_rows_bt = st.number_input(
                        "Min rows per pattern",
                        min_value=10,
                        max_value=100000,
                        value=settings.min_pattern_count,
                        step=10,
                    )
                with col16:
                    bt_max_windows = st.number_input(
                        "Max windows/pattern (0 = no cap)",
                        min_value=0,
                        max_value=20000,
                        value=80,
                        step=10,
                    )
                with col17:
                    bt_max_train_rows = st.number_input(
                        "Max train rows/window (0 = no cap)",
                        min_value=0,
                        max_value=20_000_000,
                        value=120000,
                        step=10000,
                    )
                selected_model_files: list[str] = []
            else:
                train_window_days = 504
                test_window_days = 63
                step_days = 21
                min_pattern_rows_bt = settings.min_pattern_count
                bt_max_windows = 0
                bt_max_train_rows = 0
                model_options = _model_files(models.loc[models["interval"] == bt_interval]) if not models.empty else []
                selected_model_files = st.multiselect("Filter model files (optional)", options=model_options, default=[])

            pattern_options = _pattern_names(models)
            selected_patterns = st.multiselect("Filter patterns (optional)", options=pattern_options, default=[])

            submitted = st.form_submit_button("Run Backtest", use_container_width=True)

        if submitted:

            def _task() -> pd.DataFrame:
                return service.backtest(
                    dataset_name=bt_dataset,
                    interval=bt_interval,
                    mode=bt_mode,
                    long_threshold=float(long_threshold) if long_threshold is not None else None,
                    short_threshold=float(short_threshold) if short_threshold is not None else None,
                    fee_bps=float(fee_bps),
                    include_patterns=set(selected_patterns) or None,
                    include_model_files=set(selected_model_files) or None,
                    use_model_thresholds=bool(use_model_thresholds),
                    spread_bps=float(spread_bps),
                    slippage_bps=float(slippage_bps),
                    short_borrow_bps_per_day=float(short_borrow_bps_per_day),
                    latency_bars=int(latency_bars),
                    train_window_days=int(train_window_days),
                    test_window_days=int(test_window_days),
                    step_days=int(step_days),
                    min_pattern_rows=int(min_pattern_rows_bt),
                    fast_mode=bool(bt_fast_mode),
                    parallel_models=int(bt_parallel_models),
                    max_eval_rows_per_pattern=None if int(bt_max_eval_rows) <= 0 else int(bt_max_eval_rows),
                    max_windows_per_pattern=None if int(bt_max_windows) <= 0 else int(bt_max_windows),
                    max_train_rows_per_window=None if int(bt_max_train_rows) <= 0 else int(bt_max_train_rows),
                )

            job_id, bt = _run_or_queue("backtest", run_async_jobs, _task)
            if job_id is not None:
                _show_job_message(job_id)
            else:
                st.session_state["last_backtest_table"] = bt

        if "last_backtest_table" in st.session_state:
            bt = st.session_state["last_backtest_table"]
            if bt.empty:
                st.warning("No backtest rows returned for the selected filters.")
            else:
                st.dataframe(bt, use_container_width=True)

with tab_scanner:
    st.subheader("Run Scanner")
    models = _load_model_registry()

    with st.form("scan_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            scan_interval = st.selectbox("Interval", options=INTERVAL_OPTIONS, index=0)
        with col2:
            scan_years = st.slider("History years for scan context", min_value=1, max_value=10, value=2)
        with col3:
            scan_top_n = st.slider("Top N", min_value=1, max_value=300, value=settings.top_n_signals)
        with col4:
            scan_min_conf = st.slider("Min confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        refresh_prices = st.checkbox("Refresh prices before scanning", value=True)
        pol_path_raw = st.text_input("Politician trades CSV (optional)", value="")

        col5, col6, col7 = st.columns(3)
        with col5:
            scan_use_model_thresholds = st.checkbox("Use tuned model thresholds", value=True)
        with col6:
            scan_long_threshold = st.number_input("Long threshold override", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
        with col7:
            scan_short_threshold = st.number_input("Short threshold override", min_value=0.0, max_value=1.0, value=0.45, step=0.01)

        model_options = _model_files(models.loc[models["interval"] == scan_interval]) if not models.empty else []
        pattern_options = _pattern_names(models)
        scan_model_files = st.multiselect("Filter model files (optional)", options=model_options, default=[])
        scan_patterns = st.multiselect("Filter patterns (optional)", options=pattern_options, default=[])

        submitted = st.form_submit_button("Run Scanner", use_container_width=True)

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

        job_id, scan_table = _run_or_queue("scan", run_async_jobs, _task)
        if job_id is not None:
            _show_job_message(job_id)
        else:
            st.session_state["last_scan_table"] = scan_table

    if "last_scan_table" in st.session_state:
        scan_table = st.session_state["last_scan_table"]
        if scan_table.empty:
            st.warning("No scan signals for current filters.")
        else:
            st.dataframe(scan_table, use_container_width=True)

with tab_models:
    st.subheader("Historical Model Registry")
    models = _load_model_registry()
    if models.empty:
        st.info("No saved models yet.")
    else:
        st.dataframe(models, use_container_width=True)

        model_file = st.selectbox("Inspect model file", options=_model_files(models), index=0)
        top_n_imp = st.slider("Top importance rows", min_value=5, max_value=100, value=30)
        if st.button("Load Model Details", use_container_width=True):
            details = service.model_details(model_file=model_file, top_n_importance=int(top_n_imp))
            st.session_state["model_details"] = details

        st.write("### Delete Model")
        with st.form("delete_model_form"):
            delete_model_file = st.selectbox("Model file to delete", options=_model_files(models), key="delete_model_file")
            delete_model_confirm = st.text_input("Type model filename to confirm", value="", key="delete_model_confirm")
            delete_model_submitted = st.form_submit_button("Delete Model", use_container_width=True)

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
                else:
                    st.warning(f"Model `{delete_model_file}` was not found.")

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
            }
        )

        importance = pd.DataFrame(details.get("importance", []))
        if not importance.empty:
            st.write("### Feature Importance")
            st.dataframe(importance, use_container_width=True)

with tab_analytics:
    st.subheader("Analytics")

    datasets = _load_dataset_registry()
    dataset_choices = _dataset_names(datasets)

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Pattern Coverage")
        if dataset_choices:
            coverage_dataset = st.selectbox("Dataset for coverage", options=dataset_choices, key="coverage_dataset")
            coverage = service.coverage(coverage_dataset)
            if coverage.empty:
                st.info("No coverage data available.")
            else:
                st.dataframe(coverage, use_container_width=True)
                st.bar_chart(coverage.set_index("pattern")["count"])
        else:
            st.info("No datasets available.")

    with col2:
        st.write("#### Feature Importance Summary")
        imp_interval = st.selectbox("Interval", options=["all", *INTERVAL_OPTIONS], index=1)
        imp_top_n = st.slider("Top N per pattern", min_value=5, max_value=100, value=20)
        table = service.feature_importance(
            interval=None if imp_interval == "all" else imp_interval,
            horizon_bars=settings.forward_horizon_bars,
            top_n_per_pattern=int(imp_top_n),
        )
        if table.empty:
            st.info("No feature importance reports yet.")
        else:
            st.dataframe(table, use_container_width=True)

    st.write("#### Interval Sweep")
    with st.form("sweep_form"):
        sweep_intervals = st.multiselect("Intervals", options=INTERVAL_OPTIONS, default=["1d", "1h", "30m", "15m"])
        sweep_years = st.slider("Years", min_value=1, max_value=15, value=min(10, settings.history_years))
        sweep_dataset_prefix = st.text_input("Dataset prefix", value="model_dataset")
        sweep_refresh = st.checkbox("Refresh prices", value=False)
        submitted = st.form_submit_button("Run Interval Sweep", use_container_width=True)

    if submitted:

        def _task() -> pd.DataFrame:
            return service.sweep_intervals(
                intervals=sweep_intervals,
                years=sweep_years,
                refresh_prices=sweep_refresh,
                base_dataset_name=sweep_dataset_prefix,
                universe=selected_universe,
            )

        job_id, sweep = _run_or_queue("interval_sweep", run_async_jobs, _task)
        if job_id is not None:
            _show_job_message(job_id)
        else:
            st.session_state["last_sweep_table"] = sweep

    if "last_sweep_table" in st.session_state:
        sweep = st.session_state["last_sweep_table"]
        if sweep.empty:
            st.warning("No sweep results returned.")
        else:
            st.dataframe(sweep, use_container_width=True)

with tab_jobs:
    st.subheader("Background Jobs")
    jobs = job_manager.list_jobs()
    if jobs.empty:
        st.info("No jobs yet. Queue one from Data, Train, Backtest, Scanner, or Analytics tabs.")
    else:
        jobs_display = jobs.copy()
        if "progress_pct" not in jobs_display.columns:
            jobs_display["progress_pct"] = 0.0
        if "status_message" not in jobs_display.columns:
            jobs_display["status_message"] = None
        jobs_display["progress_pct"] = jobs_display["progress_pct"].fillna(0.0).astype(float).round(1)
        st.dataframe(
            jobs_display[
                [
                    "job_id",
                    "name",
                    "status",
                    "progress_pct",
                    "status_message",
                    "created_at",
                    "started_at",
                    "finished_at",
                    "error",
                ]
            ],
            use_container_width=True,
        )

        selected_job_id = st.selectbox(
            "Inspect job",
            options=jobs["job_id"].astype(str).tolist(),
            format_func=lambda x: f"{x[:10]}...",
        )
        if selected_job_id:
            details = job_manager.get_job(selected_job_id) or {}
            pct = float(details.get("progress_pct", 0.0) or 0.0)
            msg = details.get("status_message") or "Working"
            st.progress(max(0.0, min(1.0, pct / 100.0)), text=f"{pct:.1f}% - {msg}")
            st.json(
                {
                    "job_id": details.get("job_id"),
                    "name": details.get("name"),
                    "status": details.get("status"),
                    "progress_pct": details.get("progress_pct"),
                    "status_message": details.get("status_message"),
                    "created_at": details.get("created_at"),
                    "started_at": details.get("started_at"),
                    "finished_at": details.get("finished_at"),
                    "result_type": details.get("result_type"),
                    "result_path": details.get("result_path"),
                    "error": details.get("error"),
                }
            )

            if details.get("traceback"):
                st.code(str(details.get("traceback")), language="text")

            progress_log = details.get("progress_log")
            if isinstance(progress_log, list) and progress_log:
                st.write("### Progress Log")
                log_frame = pd.DataFrame(progress_log)
                if not log_frame.empty:
                    st.dataframe(log_frame.tail(40), use_container_width=True)

            result = job_manager.load_result(selected_job_id)
            if isinstance(result, pd.DataFrame):
                st.write("### Result")
                st.dataframe(result, use_container_width=True)
            elif result is not None:
                st.write("### Result")
                st.json(result)

            if st.button("Delete Job Record", key=f"delete_job_{selected_job_id}"):
                deleted = job_manager.delete_job(selected_job_id)
                if deleted:
                    st.success("Job deleted")
                else:
                    st.warning("Cannot delete running job")

with tab_config:
    st.subheader("Runtime Configuration")
    cfg = service.settings_dict()
    cfg["fmp_api_key_set"] = bool(settings.fmp_api_key)
    cfg["polygon_api_key_set"] = bool(settings.polygon_api_key)
    cfg.pop("fmp_api_key", None)
    cfg.pop("polygon_api_key", None)
    st.json(cfg)

    st.caption("Set environment variables in .env to control providers and credentials.")
