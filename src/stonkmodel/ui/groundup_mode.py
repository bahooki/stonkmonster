from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import re

import numpy as np
import pandas as pd
import streamlit as st

from stonkmodel.config import Settings
from stonkmodel.pipeline import StonkService


def _dataset_names(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    return frame["dataset_name"].dropna().astype(str).tolist()


def _clean_run_name(text: str) -> str:
    out = re.sub(r"\s+", "_", str(text or "").strip().lower())
    out = re.sub(r"[^a-z0-9_-]+", "", out)
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "groundup"


def render_groundup_mode(
    service: StonkService,
    settings: Settings,
    selected_universe: str,
    interval_options: list[str],
    run_inline: Callable[[Callable[..., Any]], Any],
    load_dataset_registry: Callable[[], pd.DataFrame],
    load_model_registry: Callable[[], pd.DataFrame],
) -> None:
    st.subheader("Mixture of Experts (MoE) Architecture")
    st.caption(
        "Regime-aware mixture-of-experts workflow with explicit champion/challenger deployment controls."
    )
    (
        tab_data,
        tab_train,
        tab_backtest,
        tab_deploy,
        tab_diagnostics,
    ) = st.tabs(["Data", "Train", "Backtest", "Deploy", "Diagnostics"])

    with tab_data:
        st.write("### MoE Dataset Builder")
        with st.form("gx_build_dataset_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                gx_dataset_name = st.text_input(
                    "Dataset name",
                    value="gx_sp500_5y_1d",
                    help="Dedicated dataset name for Mixture of Experts experiments.",
                )
            with c2:
                gx_interval = st.selectbox("Interval", options=interval_options, index=0, key="gx_data_interval")
            with c3:
                gx_years_ago_range = st.slider(
                    "History window (years ago)",
                    min_value=0,
                    max_value=25,
                    value=(0, min(10, int(settings.history_years))),
                )
            c4, c5 = st.columns(2)
            with c4:
                gx_refresh_prices = st.checkbox("Refresh prices", value=False)
            with c5:
                gx_politician_path_raw = st.text_input("Politician CSV (optional)", value="")
            gx_build_submitted = st.form_submit_button("Build MoE Dataset", width="stretch")

        if gx_build_submitted:
            gx_pol_path = Path(gx_politician_path_raw).expanduser() if gx_politician_path_raw else None

            def _task(progress: Callable[[float, str], None] | None = None) -> dict[str, object]:
                result = service.build_dataset(
                    interval=gx_interval,
                    years=gx_years_ago_range[1],
                    refresh_prices=bool(gx_refresh_prices),
                    dataset_name=gx_dataset_name,
                    politician_trades_csv=gx_pol_path,
                    universe=selected_universe,
                    years_ago_start=gx_years_ago_range[0],
                    years_ago_end=gx_years_ago_range[1],
                    progress_callback=progress,
                )
                return {
                    "dataset_name": gx_dataset_name,
                    "universe": result.universe,
                    "rows": result.rows,
                    "symbols_loaded": result.symbols_loaded,
                    "dataset_path": result.dataset_path,
                }

            result_payload = run_inline(_task)
            st.session_state["gx_last_build"] = result_payload
            if hasattr(load_dataset_registry, "clear"):
                load_dataset_registry.clear()

        if "gx_last_build" in st.session_state:
            st.success("MoE dataset build complete")
            st.json(st.session_state["gx_last_build"])

        datasets = load_dataset_registry()
        if datasets.empty:
            st.info("No datasets available.")
        else:
            st.write("### Dataset Registry")
            st.dataframe(datasets, width="stretch")

    with tab_train:
        st.write("### Train Challenger Run")
        datasets = load_dataset_registry()
        choices = _dataset_names(datasets)
        if not choices:
            st.info("Build at least one dataset first.")
        else:
            with st.form("gx_train_form"):
                t1, t2, t3, t4 = st.columns(4)
                with t1:
                    gx_train_dataset = st.selectbox("Dataset", options=choices, index=0, key="gx_train_dataset")
                with t2:
                    gx_train_interval = st.selectbox("Interval", options=interval_options, index=0, key="gx_train_interval")
                with t3:
                    gx_run_name = st.text_input("Run name", value="regime_moe_v1")
                with t4:
                    gx_min_pattern_rows = st.number_input("Min pattern rows", min_value=10, max_value=100000, value=120, step=10)

                t5, t6, t7, t8 = st.columns(4)
                with t5:
                    gx_parallel = st.number_input("Parallel patterns", min_value=1, max_value=32, value=2, step=1)
                with t6:
                    gx_fast_mode = st.checkbox("Fast mode", value=True)
                with t7:
                    gx_max_rows = st.number_input(
                        "Max rows/pattern (0=auto)",
                        min_value=0,
                        max_value=2_000_000,
                        value=120000,
                        step=10000,
                    )
                with t8:
                    gx_candidates = st.number_input("Candidate models/pattern", min_value=1, max_value=6, value=2, step=1)

                t9, t10, t11, t12 = st.columns(4)
                with t9:
                    gx_fee_bps = st.number_input("Fee bps", min_value=0.0, max_value=1000.0, value=1.0, step=0.1)
                with t10:
                    gx_spread_bps = st.number_input("Spread bps", min_value=0.0, max_value=1000.0, value=0.5, step=0.1)
                with t11:
                    gx_slippage_bps = st.number_input("Slippage bps", min_value=0.0, max_value=1000.0, value=0.5, step=0.1)
                with t12:
                    gx_latency_bars = st.number_input("Latency bars", min_value=0, max_value=100, value=1, step=1)

                gx_train_submitted = st.form_submit_button("Train Challenger", width="stretch")

            if gx_train_submitted:
                run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                run_key = f"gx_{_clean_run_name(gx_run_name)}_{run_stamp}"
                model_name = f"gx__{_clean_run_name(gx_run_name)}__{run_stamp}"

                def _task(progress: Callable[[float, str], None] | None = None) -> dict[str, object]:
                    if progress:
                        progress(1.0, "Training run models")
                    train_table = service.train(
                        dataset_name=gx_train_dataset,
                        interval=gx_train_interval,
                        min_pattern_rows=int(gx_min_pattern_rows),
                        model_name=model_name,
                        parallel_patterns=int(gx_parallel),
                        fast_mode=bool(gx_fast_mode),
                        max_rows_per_pattern=None if int(gx_max_rows) <= 0 else int(gx_max_rows),
                        candidate_models_per_pattern=int(gx_candidates),
                        progress_callback=progress,
                    )
                    model_files = set(train_table["model_file"].dropna().astype(str).tolist()) if not train_table.empty else set()
                    if progress:
                        progress(75.0, "Optimizing thresholds and backtesting challenger")
                    threshold_table = pd.DataFrame()
                    bt = pd.DataFrame()
                    if model_files:
                        threshold_table = service.optimize_model_thresholds_from_backtest(
                            dataset_name=gx_train_dataset,
                            interval=gx_train_interval,
                            fee_bps=float(gx_fee_bps),
                            spread_bps=float(gx_spread_bps),
                            slippage_bps=float(gx_slippage_bps),
                            latency_bars=int(gx_latency_bars),
                            include_model_files=model_files,
                            min_trades=40,
                            parallel_models=max(1, int(gx_parallel)),
                            persist=True,
                        )
                        bt = service.backtest(
                            dataset_name=gx_train_dataset,
                            interval=gx_train_interval,
                            mode="saved_models",
                            fee_bps=float(gx_fee_bps),
                            spread_bps=float(gx_spread_bps),
                            slippage_bps=float(gx_slippage_bps),
                            latency_bars=int(gx_latency_bars),
                            include_model_files=model_files,
                            use_model_thresholds=True,
                            include_portfolio=True,
                            parallel_models=max(1, int(gx_parallel)),
                        )
                    run_row = service.groundup_register_run(
                        run_id=run_key,
                        run_name=gx_run_name,
                        dataset_name=gx_train_dataset,
                        interval=gx_train_interval,
                        model_files=model_files,
                        train_table=train_table,
                        backtest_table=bt,
                        status="challenger",
                    )
                    if progress:
                        progress(100.0, "Challenger run registered")
                    return {
                        "run_row": run_row,
                        "train_table": train_table,
                        "threshold_table": threshold_table,
                        "backtest_table": bt,
                    }

                out = run_inline(_task)
                st.session_state["gx_last_train_run"] = out
                if hasattr(load_model_registry, "clear"):
                    load_model_registry.clear()

        if "gx_last_train_run" in st.session_state:
            out = st.session_state["gx_last_train_run"]
            st.success(f"Run registered: `{out['run_row'].get('run_id')}`")
            st.json(out["run_row"])
            train_table = out.get("train_table", pd.DataFrame())
            threshold_table = out.get("threshold_table", pd.DataFrame())
            backtest_table = out.get("backtest_table", pd.DataFrame())
            if isinstance(train_table, pd.DataFrame) and not train_table.empty:
                st.write("#### Train Summary")
                st.dataframe(train_table, width="stretch")
            if isinstance(threshold_table, pd.DataFrame) and not threshold_table.empty:
                st.write("#### Threshold Optimization")
                st.dataframe(threshold_table, width="stretch")
            if isinstance(backtest_table, pd.DataFrame) and not backtest_table.empty:
                st.write("#### Initial Challenger Backtest")
                st.dataframe(backtest_table, width="stretch")

    with tab_backtest:
        st.write("### Backtest MoE Runs")
        runs = service.groundup_runs()
        if runs.empty:
            st.info("No MoE runs registered yet.")
        else:
            run_summary_cols = [c for c in ["run_id", "run_name", "dataset_name", "interval", "model_count", "status"] if c in runs.columns]
            if run_summary_cols:
                st.dataframe(runs[run_summary_cols], width="stretch")

            dataset_choices = _dataset_names(load_dataset_registry())
            if not dataset_choices:
                st.info("No datasets available for backtest.")
            else:
                run_options = runs["run_id"].astype(str).tolist()
                default_runs = run_options[: min(3, len(run_options))]
                latest_run = runs.iloc[0] if not runs.empty else None
                latest_dataset = str(latest_run.get("dataset_name", "")) if latest_run is not None else ""
                latest_interval = str(latest_run.get("interval", "")) if latest_run is not None else ""
                dataset_idx = dataset_choices.index(latest_dataset) if latest_dataset in dataset_choices else 0
                interval_idx = interval_options.index(latest_interval) if latest_interval in interval_options else 0
                with st.form("gx_backtest_form"):
                    b1, b2, b3, b4 = st.columns(4)
                    with b1:
                        gx_bt_dataset = st.selectbox("Dataset", options=dataset_choices, index=dataset_idx, key="gx_bt_dataset")
                    with b2:
                        gx_bt_interval = st.selectbox("Interval", options=interval_options, index=interval_idx, key="gx_bt_interval")
                    with b3:
                        gx_bt_mode = st.selectbox("Mode", options=["saved_models", "walk_forward_retrain"], index=0)
                    with b4:
                        gx_bt_use_thresholds = st.checkbox("Use model thresholds", value=True)

                    b5, b6, b7, b8 = st.columns(4)
                    with b5:
                        gx_bt_fee = st.number_input("Fee bps", min_value=0.0, max_value=1000.0, value=1.0, step=0.1, key="gx_bt_fee")
                    with b6:
                        gx_bt_spread = st.number_input("Spread bps", min_value=0.0, max_value=1000.0, value=0.5, step=0.1, key="gx_bt_spread")
                    with b7:
                        gx_bt_slippage = st.number_input("Slippage bps", min_value=0.0, max_value=1000.0, value=0.5, step=0.1, key="gx_bt_slip")
                    with b8:
                        gx_bt_latency = st.number_input("Latency bars", min_value=0, max_value=100, value=1, step=1, key="gx_bt_latency")

                    selected_run_ids = st.multiselect(
                        "Runs to include",
                        options=run_options,
                        default=default_runs,
                        help="Backtest only models from selected runs.",
                    )
                    gx_bt_submit = st.form_submit_button("Run MoE Backtest", width="stretch")

                if gx_bt_submit:
                    if not selected_run_ids:
                        st.warning("Select at least one run to backtest.")
                    else:
                        model_files_raw: set[str] = set()
                        runs_without_models: list[str] = []
                        for rid in selected_run_ids:
                            files = service.groundup_models_for_run(str(rid))
                            if not files:
                                runs_without_models.append(str(rid))
                            for item in files:
                                safe = Path(str(item)).name
                                if safe:
                                    model_files_raw.add(safe)

                        existing_model_files = {p.name for p in service.model_io.list_models()}
                        resolved_model_files = sorted(f for f in model_files_raw if f in existing_model_files)
                        missing_model_files = sorted(f for f in model_files_raw if f not in existing_model_files)

                        if runs_without_models:
                            st.warning(
                                "Some selected runs had no saved model files: "
                                + ", ".join(runs_without_models)
                            )
                        if missing_model_files:
                            preview = ", ".join(missing_model_files[:6])
                            suffix = " ..." if len(missing_model_files) > 6 else ""
                            st.warning(
                                f"{len(missing_model_files)} model files were missing on disk and skipped: "
                                f"{preview}{suffix}"
                            )

                        if not resolved_model_files:
                            st.error(
                                "No usable model files were resolved for this backtest. "
                                "Try retraining the run or selecting a different run."
                            )
                        else:

                            def _task(progress: Callable[[float, str], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
                                if progress:
                                    progress(3.0, "Starting MoE backtest")
                                out = service.backtest_with_details(
                                    dataset_name=gx_bt_dataset,
                                    interval=gx_bt_interval,
                                    mode=gx_bt_mode,
                                    fee_bps=float(gx_bt_fee),
                                    spread_bps=float(gx_bt_spread),
                                    slippage_bps=float(gx_bt_slippage),
                                    latency_bars=int(gx_bt_latency),
                                    include_model_files=set(resolved_model_files),
                                    use_model_thresholds=bool(gx_bt_use_thresholds),
                                    include_portfolio=True,
                                    include_spread_strategies=True,
                                    initial_investment=10000.0,
                                )
                                if progress:
                                    progress(100.0, "Ground-up backtest complete")
                                return out

                            try:
                                bt, curve = run_inline(_task)
                            except Exception as exc:
                                st.error(f"MoE backtest failed: {exc}")
                                st.exception(exc)
                            else:
                                st.session_state["gx_last_backtest"] = {
                                    "table": bt,
                                    "curve": curve,
                                    "runs": selected_run_ids,
                                    "dataset_name": gx_bt_dataset,
                                    "interval": gx_bt_interval,
                                    "mode": gx_bt_mode,
                                    "requested_model_files": int(len(model_files_raw)),
                                    "used_model_files": int(len(resolved_model_files)),
                                    "missing_model_files": int(len(missing_model_files)),
                                }
                                st.success(
                                    "MoE backtest complete: "
                                    f"{len(bt)} summary rows, {len(curve)} curve points."
                                )

        if "gx_last_backtest" in st.session_state:
            payload = st.session_state["gx_last_backtest"]
            bt = payload.get("table", pd.DataFrame())
            curve = payload.get("curve", pd.DataFrame())
            st.write(f"Selected runs: `{', '.join([str(x) for x in payload.get('runs', [])])}`")
            st.caption(
                "Dataset: "
                f"`{payload.get('dataset_name', 'n/a')}` | "
                "Interval: "
                f"`{payload.get('interval', 'n/a')}` | "
                "Mode: "
                f"`{payload.get('mode', 'n/a')}` | "
                "Models used: "
                f"`{payload.get('used_model_files', 'n/a')}`"
            )
            if isinstance(bt, pd.DataFrame) and not bt.empty:
                st.dataframe(bt, width="stretch")
            else:
                st.warning(
                    "Backtest returned no summary rows. Common causes: dataset/interval mismatch with selected runs, "
                    "or models filtered out by strict OOS windowing."
                )
            if isinstance(curve, pd.DataFrame) and not curve.empty:
                chart = curve.copy()
                if "datetime" in chart.columns:
                    chart["datetime"] = pd.to_datetime(chart["datetime"], utc=True, errors="coerce")
                    chart = chart.dropna(subset=["datetime"]).sort_values("datetime")
                if {"datetime", "equity_value", "curve_variant", "model_file", "pattern"}.issubset(chart.columns):
                    chart["curve_variant"] = chart["curve_variant"].astype(str).str.strip().replace("", "ml_model")
                    chart["curve_key"] = chart["model_file"].astype(str) + " | " + chart["pattern"].astype(str)
                    curve_keys = sorted(chart["curve_key"].dropna().astype(str).unique().tolist())
                    if curve_keys:
                        default_key = curve_keys[0]
                        preferred_key = None
                        if isinstance(bt, pd.DataFrame) and not bt.empty and {"model_file", "pattern"}.issubset(bt.columns):
                            top = bt.iloc[0]
                            preferred_key = f"{str(top.get('model_file', ''))} | {str(top.get('pattern', ''))}"
                        portfolio_keys = [k for k in curve_keys if "portfolio_combined" in k]
                        if portfolio_keys:
                            default_key = portfolio_keys[0]
                        elif preferred_key and preferred_key in curve_keys:
                            default_key = preferred_key
                        selected_curve_key = st.selectbox(
                            "MoE curve",
                            options=curve_keys,
                            index=curve_keys.index(default_key),
                            key="gx_backtest_curve_selector",
                            help="Pick one strategy/model curve to compare ML vs blind-pattern vs universe benchmark.",
                        )
                        selected = chart.loc[chart["curve_key"] == selected_curve_key].sort_values("datetime").copy()
                        selected["curve_variant"] = (
                            selected["curve_variant"].astype(str).str.strip().replace("", "ml_model")
                        )
                        label_map = {
                            "ml_model": "ML Strategy",
                            "baseline_blind_pattern": "Blind Pattern",
                            "baseline_universe_eqw": "Universe Benchmark (Equal-Weight)",
                        }
                        selected["curve_label"] = selected["curve_variant"].map(label_map).fillna(selected["curve_variant"])
                        pivot = (
                            selected.pivot_table(
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
                        ordered_cols = [c for c in preferred_order if c in pivot.columns] + [
                            c for c in pivot.columns if c not in preferred_order
                        ]
                        st.line_chart(pivot[ordered_cols], width="stretch")

                        variant_returns: list[str] = []
                        for label in ordered_cols:
                            part = selected.loc[selected["curve_label"] == label].sort_values("datetime")
                            if part.empty:
                                continue
                            ret = float(pd.to_numeric(part["cumulative_return"], errors="coerce").iloc[-1])
                            variant_returns.append(f"{label}: {ret * 100:.2f}%")
                        if variant_returns:
                            st.caption("Comparison returns | " + " | ".join(variant_returns))

                        ml_slice = selected.loc[selected["curve_variant"] == "ml_model"].sort_values("datetime")
                        if not ml_slice.empty and "period_return" in ml_slice.columns:
                            pr = pd.to_numeric(ml_slice["period_return"], errors="coerce").dropna()
                            if not pr.empty:
                                st.caption(
                                    "ML period stats | "
                                    f"N={len(pr)} | "
                                    f"mean={float(pr.mean()) * 100:.2f}% | "
                                    f"std={float(pr.std()) * 100:.2f}% | "
                                    f"min={float(pr.min()) * 100:.2f}% | "
                                    f"max={float(pr.max()) * 100:.2f}%"
                                )

                    st.write("### Multi-Pattern Portfolio Curve")
                    pattern_options = sorted(
                        chart.loc[chart["curve_variant"] == "ml_model", "pattern"]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    if pattern_options:
                        default_patterns = pattern_options[: min(3, len(pattern_options))]
                        if isinstance(bt, pd.DataFrame) and not bt.empty and "pattern" in bt.columns:
                            bt_default = (
                                bt["pattern"]
                                .dropna()
                                .astype(str)
                                .tolist()
                            )
                            bt_default = [p for p in bt_default if p in pattern_options]
                            if bt_default:
                                default_patterns = bt_default[: min(3, len(bt_default))]
                        selected_patterns = st.multiselect(
                            "Patterns to combine as one portfolio",
                            options=pattern_options,
                            default=default_patterns,
                            key="gx_multi_pattern_portfolio_select",
                            help=(
                                "Combines selected models by averaging their period returns on each date "
                                "(equal-weight across active selected models)."
                            ),
                        )

                        if selected_patterns:
                            selected_set = set(str(x) for x in selected_patterns)
                            selected_frame = chart.loc[chart["pattern"].astype(str).isin(selected_set)].copy()

                            init_series = pd.to_numeric(selected_frame.get("initial_investment"), errors="coerce").dropna()
                            initial_investment = float(init_series.iloc[0]) if not init_series.empty else 10000.0

                            def _aggregate_variant_period(variant: str) -> pd.Series:
                                part = selected_frame.loc[selected_frame["curve_variant"] == variant].copy()
                                if part.empty or "period_return" not in part.columns:
                                    return pd.Series(dtype=float)
                                period = pd.to_numeric(part["period_return"], errors="coerce")
                                dt = pd.to_datetime(part["datetime"], utc=True, errors="coerce")
                                valid = dt.notna() & period.notna()
                                if not bool(valid.any()):
                                    return pd.Series(dtype=float)
                                rows = pd.DataFrame(
                                    {
                                        "datetime": dt.loc[valid],
                                        "period_return": period.loc[valid].astype(float),
                                    }
                                )
                                return rows.groupby("datetime", sort=True)["period_return"].mean()

                            def _equity_frame(period_series: pd.Series, label: str) -> pd.DataFrame:
                                if period_series.empty:
                                    return pd.DataFrame(columns=["datetime", "curve_label", "equity_value", "cumulative_return"])
                                clean = pd.to_numeric(period_series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                                if clean.empty:
                                    return pd.DataFrame(columns=["datetime", "curve_label", "equity_value", "cumulative_return"])
                                growth = np.exp(np.log1p(clean.clip(lower=-0.999999999)).cumsum())
                                return pd.DataFrame(
                                    {
                                        "datetime": pd.to_datetime(clean.index, utc=True, errors="coerce"),
                                        "curve_label": str(label),
                                        "equity_value": float(initial_investment) * growth.to_numpy(dtype=float),
                                        "cumulative_return": (growth - 1.0).to_numpy(dtype=float),
                                    }
                                )

                            ml_period = _aggregate_variant_period("ml_model")
                            blind_period = _aggregate_variant_period("baseline_blind_pattern")
                            bench_period = _aggregate_variant_period("baseline_universe_eqw")

                            portfolio_curves = pd.concat(
                                [
                                    _equity_frame(ml_period, "ML Portfolio (Selected Patterns)"),
                                    _equity_frame(blind_period, "Blind Portfolio (Selected Patterns)"),
                                    _equity_frame(bench_period, "Universe Benchmark (Equal-Weight)"),
                                ],
                                ignore_index=True,
                            )
                            portfolio_curves = portfolio_curves.dropna(subset=["datetime"]).sort_values("datetime")
                            if portfolio_curves.empty:
                                st.info("No combined portfolio curve was produced for the selected patterns.")
                            else:
                                pf_pivot = (
                                    portfolio_curves.pivot_table(
                                        index="datetime",
                                        columns="curve_label",
                                        values="equity_value",
                                        aggfunc="last",
                                    )
                                    .sort_index()
                                )
                                pf_order = [
                                    "ML Portfolio (Selected Patterns)",
                                    "Blind Portfolio (Selected Patterns)",
                                    "Universe Benchmark (Equal-Weight)",
                                ]
                                ordered_pf = [c for c in pf_order if c in pf_pivot.columns] + [
                                    c for c in pf_pivot.columns if c not in pf_order
                                ]
                                st.line_chart(pf_pivot[ordered_pf], width="stretch")

                                comp: list[str] = []
                                for label in ordered_pf:
                                    part = portfolio_curves.loc[portfolio_curves["curve_label"] == label].sort_values("datetime")
                                    if part.empty:
                                        continue
                                    ret = float(pd.to_numeric(part["cumulative_return"], errors="coerce").iloc[-1])
                                    comp.append(f"{label}: {ret * 100:.2f}%")
                                if comp:
                                    st.caption("Combined portfolio returns | " + " | ".join(comp))
                        else:
                            st.info("Select at least one pattern to build a combined portfolio curve.")
            else:
                st.info("No curve points were produced for this run.")

    with tab_deploy:
        st.write("### Champion / Challenger Deployment")
        runs = service.groundup_runs()
        if runs.empty:
            st.info("No runs available yet.")
        else:
            st.dataframe(runs, width="stretch")
            deployment = service.groundup_get_deployment()
            run_ids = runs["run_id"].astype(str).tolist()

            def _idx_or_zero(value: object) -> int:
                text = str(value) if value is not None else ""
                return run_ids.index(text) if text in run_ids else 0

            with st.form("gx_deployment_form"):
                d1, d2 = st.columns(2)
                with d1:
                    champion_run_id = st.selectbox(
                        "Champion run",
                        options=run_ids,
                        index=_idx_or_zero(deployment.get("champion_run_id")),
                    )
                with d2:
                    challenger_run_id = st.selectbox(
                        "Challenger run",
                        options=["", *run_ids],
                        index=(["", *run_ids].index(str(deployment.get("challenger_run_id"))) if str(deployment.get("challenger_run_id")) in ["", *run_ids] else 0),
                    )
                d3, d4, d5 = st.columns(3)
                policy = dict(deployment.get("policy", {}))
                with d3:
                    min_rel = st.number_input("Min relative improvement", min_value=0.0, max_value=2.0, value=float(policy.get("min_relative_improvement", 0.05)), step=0.01)
                with d4:
                    min_trade_count = st.number_input("Min challenger trades", min_value=0, max_value=1_000_000, value=int(policy.get("min_trade_count", 80)), step=5)
                with d5:
                    max_age_days = st.number_input("Max champion age (days)", min_value=1, max_value=3650, value=int(policy.get("max_champion_age_days", 45)), step=1)
                gx_save_deploy = st.form_submit_button("Save Deployment State", width="stretch")

            if gx_save_deploy:
                new_state = service.groundup_set_deployment(
                    champion_run_id=champion_run_id,
                    challenger_run_id=challenger_run_id or None,
                    min_relative_improvement=float(min_rel),
                    min_trade_count=int(min_trade_count),
                    max_champion_age_days=int(max_age_days),
                )
                st.success("Deployment state saved")
                st.json(new_state)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Evaluate Promotion Decision", width="stretch"):
                    decision = service.groundup_promotion_decision()
                    st.session_state["gx_promotion_decision"] = decision
            with c2:
                if st.button("Promote Challenger Now (if eligible)", width="stretch"):
                    promoted = service.groundup_promote_challenger()
                    st.session_state["gx_promotion_decision"] = promoted
                    if bool(promoted.get("promoted")):
                        st.success(f"Promoted new champion: `{promoted.get('new_champion_run_id')}`")
                    else:
                        st.warning(f"Promotion blocked: {promoted.get('reason')}")

            if "gx_promotion_decision" in st.session_state:
                st.json(st.session_state["gx_promotion_decision"])

            st.write("### Champion Scanner")
            s1, s2 = st.columns(2)
            with s1:
                gx_scan_years = st.number_input("Scan lookback years", min_value=1, max_value=20, value=2, step=1)
            with s2:
                gx_scan_top_n = st.number_input("Top N signals", min_value=1, max_value=500, value=30, step=1)
            if st.button("Scan with Active Champion Models", width="stretch"):
                state = service.groundup_get_deployment()
                champion = state.get("champion_run_id")
                if not champion:
                    st.warning("No champion configured.")
                else:
                    champion_models = service.groundup_models_for_run(str(champion))
                    if not champion_models:
                        st.warning("Champion run has no model files.")
                    else:
                        scan = service.scan(
                            interval=settings.default_interval,
                            years=int(gx_scan_years),
                            top_n=int(gx_scan_top_n),
                            refresh_prices=False,
                            include_model_files=champion_models,
                            min_confidence=0.5,
                            universe=selected_universe,
                            use_model_thresholds=True,
                        )
                        st.session_state["gx_champion_scan"] = scan
            if "gx_champion_scan" in st.session_state:
                scan = st.session_state["gx_champion_scan"]
                if isinstance(scan, pd.DataFrame) and not scan.empty:
                    st.dataframe(scan, width="stretch")
                else:
                    st.info("No champion scan signals.")

    with tab_diagnostics:
        st.write("### Run Diagnostics")
        runs = service.groundup_runs()
        if runs.empty:
            st.info("No diagnostics yet. Train at least one challenger run.")
        else:
            work = runs.copy()
            if "created_utc" in work.columns:
                work["created_utc"] = pd.to_datetime(work["created_utc"], utc=True, errors="coerce")
            for col in ("objective_score", "cumulative_return", "trades", "sharpe"):
                if col in work.columns:
                    work[col] = pd.to_numeric(work[col], errors="coerce")
            st.dataframe(work, width="stretch")
            if "created_utc" in work.columns and "objective_score" in work.columns:
                chart = work.dropna(subset=["created_utc"]).set_index("created_utc")[["objective_score", "cumulative_return"]]
                if not chart.empty:
                    st.line_chart(chart, width="stretch")
            if "status" in work.columns:
                status_counts = work["status"].astype(str).value_counts()
                st.bar_chart(status_counts)

        st.write("### Regime Snapshot")
        datasets = load_dataset_registry()
        choices = _dataset_names(datasets)
        if choices:
            gx_diag_dataset = st.selectbox("Dataset for regime snapshot", options=choices, key="gx_diag_dataset")
            regime = service.groundup_regime_snapshot(gx_diag_dataset)
            if regime.empty:
                st.info("No regime columns found in selected dataset.")
            else:
                st.dataframe(regime, width="stretch")
                if {"regime", "rows"}.issubset(regime.columns):
                    st.bar_chart(regime.set_index("regime")["rows"])
        else:
            st.info("No dataset available for regime diagnostics.")

        st.write("### MoE Model Artifacts")
        registry = load_model_registry()
        if registry.empty:
            st.info("No saved models.")
        else:
            reg = registry.copy()
            if "model_name" in reg.columns:
                name_col = reg["model_name"].fillna("").astype(str)
                gx_only = reg.loc[name_col.str.startswith("gx__", na=False)].copy()
            else:
                gx_only = reg.head(0).copy()
            st.dataframe(gx_only if not gx_only.empty else reg.head(0), width="stretch")
