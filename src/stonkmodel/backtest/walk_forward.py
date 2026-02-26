from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stonkmodel.features.dataset import split_xy
from stonkmodel.features.patterns import PATTERN_COLUMNS
from stonkmodel.models.calibration import (
    apply_probability_calibration,
    fit_probability_calibration,
    optimize_thresholds_from_validation,
    resolve_thresholds,
)
from stonkmodel.models.feature_selection import FeatureSelectionConfig, select_features
from stonkmodel.models.stacking import PatternModelIO, build_stacking_classifier


@dataclass
class BacktestResult:
    pattern: str
    model_file: str
    trades: int
    win_rate: float
    avg_trade_return: float
    cumulative_return: float
    sharpe: float
    backtest_start_datetime: str | None = None
    backtest_end_datetime: str | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "pattern": self.pattern,
            "model_file": self.model_file,
            "backtest_start_datetime": self.backtest_start_datetime,
            "backtest_end_datetime": self.backtest_end_datetime,
            "trades": self.trades,
            "win_rate": self.win_rate,
            "avg_trade_return": self.avg_trade_return,
            "cumulative_return": self.cumulative_return,
            "sharpe": self.sharpe,
        }


def _bars_per_day(interval: str) -> float:
    mapping = {
        "1d": 1.0,
        "1h": 6.5,
        "60m": 6.5,
        "30m": 13.0,
        "15m": 26.0,
        "5m": 78.0,
        "1m": 390.0,
    }
    return float(mapping.get(interval, 1.0))


def _holding_days(interval: str, horizon_bars: int, latency_bars: int) -> float:
    return float(max(1, horizon_bars + max(0, latency_bars)) / _bars_per_day(interval))


def _periods_per_year(interval: str) -> float:
    return float(252.0 * _bars_per_day(interval))


def _aggregate_returns_by_datetime(subset: pd.DataFrame, column: str = "strategy_return") -> pd.Series:
    if subset.empty or column not in subset.columns:
        return pd.Series(dtype=float)

    returns = pd.to_numeric(subset[column], errors="coerce")
    if "datetime" not in subset.columns:
        return returns.dropna().astype(float).reset_index(drop=True)

    dt = pd.to_datetime(subset["datetime"], utc=True, errors="coerce")
    valid = dt.notna() & returns.notna()
    if not bool(valid.any()):
        return pd.Series(dtype=float)

    frame = pd.DataFrame({"datetime": dt.loc[valid], "strategy_return": returns.loc[valid].astype(float)})
    # Aggregate concurrent symbols/signals into one portfolio return per timestamp.
    return frame.groupby("datetime", sort=True)["strategy_return"].mean()


def _safe_cumulative_return(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    # Guard log1p domain while preserving normal return magnitudes.
    clean = clean.clip(lower=-0.999999999)
    return float(np.expm1(np.log1p(clean).sum()))


def _safe_sharpe_for_interval(series: pd.Series, interval: str) -> float:
    std = series.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((series.mean() / std) * np.sqrt(_periods_per_year(interval)))


def _to_iso_utc(value: object) -> str | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.isoformat()


def _extract_pattern_subset(frame: pd.DataFrame, pattern: str) -> pd.DataFrame:
    if pattern == "none":
        return frame.loc[frame["pattern"].isna()].copy()
    pattern_col = f"pattern_{pattern}"
    if pattern_col not in frame.columns:
        return pd.DataFrame()
    return frame.loc[frame[pattern_col] == 1].copy()


def _cap_rows_for_speed(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None:
        return frame
    cap = int(max_rows)
    if cap <= 0 or len(frame) <= cap:
        return frame
    if "datetime" not in frame.columns:
        return frame.tail(cap).copy()
    return frame.sort_values("datetime").tail(cap).copy()


def _compute_realized_return(
    frame: pd.DataFrame,
    horizon_bars: int,
    latency_bars: int,
) -> pd.Series:
    if "close" not in frame.columns:
        if "future_return" in frame.columns and int(latency_bars) == 0:
            return frame["future_return"].astype(float)
        return pd.Series(np.nan, index=frame.index, dtype=float)

    out = pd.Series(index=frame.index, dtype=float)
    lag = max(0, int(latency_bars))
    horizon = max(1, int(horizon_bars))
    for _, g in frame.groupby("symbol", sort=False):
        if "datetime" in g.columns:
            g = g.sort_values("datetime")
        entry = g["close"].shift(-lag)
        exit_ = g["close"].shift(-(lag + horizon))
        out.loc[g.index] = (exit_ / entry) - 1.0
    return out


def _apply_execution_realism(
    subset: pd.DataFrame,
    interval: str,
    horizon_bars: int,
    fee_bps: float,
    spread_bps: float,
    slippage_bps: float,
    short_borrow_bps_per_day: float,
    latency_bars: int,
) -> pd.Series:
    holding_days = _holding_days(interval, horizon_bars=horizon_bars, latency_bars=latency_bars)
    fee = float(fee_bps) / 10000.0
    spread = float(spread_bps) / 10000.0
    slippage = float(slippage_bps) / 10000.0
    borrow = (float(short_borrow_bps_per_day) / 10000.0) * holding_days

    trade_cost = fee + spread + (2.0 * slippage)
    short_cost = np.where(subset["position"] < 0, borrow, 0.0)
    return subset["position"] * subset["realized_return"] - trade_cost - short_cost


def run_pattern_backtests(
    dataset: pd.DataFrame,
    model_io: PatternModelIO,
    interval: str,
    horizon_bars: int,
    long_threshold: float | None = 0.55,
    short_threshold: float | None = 0.45,
    fee_bps: float = 1.0,
    include_patterns: set[str] | None = None,
    include_model_files: set[str] | None = None,
    use_model_thresholds: bool = False,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    short_borrow_bps_per_day: float = 0.0,
    latency_bars: int = 0,
    parallel_models: int = 1,
    max_eval_rows_per_pattern: int | None = None,
) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()

    full = dataset.copy()
    full_dt = pd.to_datetime(full["datetime"], utc=True, errors="coerce")

    default_test = full.loc[full["split"] == "test"].copy() if "split" in full.columns else pd.DataFrame()
    if default_test.empty:
        cutoff = int(len(full) * 0.8)
        default_test = full.iloc[cutoff:].copy()

    default_train_end = pd.NaT
    if "split" in full.columns:
        train_dt = pd.to_datetime(full.loc[full["split"] == "train", "datetime"], utc=True, errors="coerce")
        if not train_dt.empty:
            default_train_end = train_dt.max()

    candidate_models: list[tuple] = []
    for model_path in model_io.list_models():
        model_file = model_path.name
        if include_model_files and model_file not in include_model_files:
            continue

        parsed = model_io.parse_model_filename(model_file)
        parsed_interval = parsed.get("interval")
        parsed_horizon = parsed.get("horizon_bars")
        parsed_pattern = parsed.get("pattern")
        if parsed_interval is not None and str(parsed_interval) != str(interval):
            continue
        if parsed_horizon is not None and int(parsed_horizon) != int(horizon_bars):
            continue
        if include_patterns and parsed_pattern is not None and str(parsed_pattern) not in include_patterns:
            continue
        candidate_models.append((model_path, model_file))

    if not candidate_models:
        return pd.DataFrame(
            columns=[
                "pattern",
                "model_file",
                "backtest_start_datetime",
                "backtest_end_datetime",
                "trades",
                "win_rate",
                "avg_trade_return",
                "cumulative_return",
                "sharpe",
            ]
        )

    def evaluate_model(model_path, model_file: str) -> dict[str, float | int | str | None] | None:
        try:
            payload = model_io.load_from_path(model_path)
        except Exception:
            return None

        if payload.get("interval") != interval or int(payload.get("horizon_bars", -1)) != int(horizon_bars):
            return None

        pattern = str(payload["pattern"])
        if include_patterns and pattern not in include_patterns:
            return None

        model_train_end = pd.to_datetime(payload.get("train_end_datetime"), utc=True, errors="coerce")
        if pd.isna(model_train_end):
            model_train_end = default_train_end

        if pd.notna(model_train_end):
            eval_pool = full.loc[full_dt > model_train_end].copy()
        else:
            eval_pool = default_test.copy()

        if eval_pool.empty:
            eval_pool = default_test.copy()
        if eval_pool.empty:
            return None

        eval_dt = pd.to_datetime(eval_pool["datetime"], utc=True, errors="coerce")
        eval_start_iso = _to_iso_utc(eval_dt.min())
        eval_end_iso = _to_iso_utc(eval_dt.max())

        eval_pool = eval_pool.sort_values(["symbol", "datetime"]).copy()
        eval_pool["realized_return"] = _compute_realized_return(
            eval_pool,
            horizon_bars=horizon_bars,
            latency_bars=latency_bars,
        )

        model = payload["model"]
        feature_cols = payload["feature_columns"]

        subset = _extract_pattern_subset(eval_pool, pattern=pattern)
        subset = _cap_rows_for_speed(subset, max_rows=max_eval_rows_per_pattern)
        if subset.empty:
            return None

        x = subset.reindex(columns=feature_cols).fillna(subset.median(numeric_only=True)).fillna(0.0)
        prob_raw = model.predict_proba(x)[:, 1]
        prob = apply_probability_calibration(prob_raw, payload.get("probability_calibration"))
        long_t, short_t = resolve_thresholds(
            payload=payload,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            use_model_thresholds=use_model_thresholds,
        )

        subset["pred_prob_up"] = prob
        subset["position"] = 0
        subset.loc[subset["pred_prob_up"] >= long_t, "position"] = 1
        subset.loc[subset["pred_prob_up"] <= short_t, "position"] = -1
        subset = subset.loc[subset["position"] != 0].copy()
        if subset.empty:
            return None

        subset = subset.dropna(subset=["realized_return"]).copy()
        if subset.empty:
            return None

        subset["strategy_return"] = _apply_execution_realism(
            subset=subset,
            interval=interval,
            horizon_bars=horizon_bars,
            fee_bps=fee_bps,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            short_borrow_bps_per_day=short_borrow_bps_per_day,
            latency_bars=latency_bars,
        )

        period_returns = _aggregate_returns_by_datetime(subset, column="strategy_return")
        result = BacktestResult(
            pattern=pattern,
            model_file=model_file,
            trades=int(len(subset)),
            win_rate=float((subset["strategy_return"] > 0).mean()),
            avg_trade_return=float(subset["strategy_return"].mean()),
            cumulative_return=_safe_cumulative_return(period_returns),
            sharpe=_safe_sharpe_for_interval(period_returns, interval=interval),
            backtest_start_datetime=eval_start_iso,
            backtest_end_datetime=eval_end_iso,
        )
        return result.to_dict()

    rows: list[dict[str, float | int | str | None]] = []
    worker_count = max(1, int(parallel_models))
    if worker_count <= 1:
        for model_path, model_file in candidate_models:
            row = evaluate_model(model_path, model_file=model_file)
            if row is not None:
                rows.append(row)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(evaluate_model, model_path, model_file): model_file
                for model_path, model_file in candidate_models
            }
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception:
                    continue
                if row is not None:
                    rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "pattern",
                "model_file",
                "backtest_start_datetime",
                "backtest_end_datetime",
                "trades",
                "win_rate",
                "avg_trade_return",
                "cumulative_return",
                "sharpe",
            ]
        )
    return pd.DataFrame(rows).sort_values("cumulative_return", ascending=False)


def run_walk_forward_retraining_backtests(
    dataset: pd.DataFrame,
    interval: str,
    horizon_bars: int,
    train_window_days: int = 504,
    test_window_days: int = 63,
    step_days: int = 21,
    min_pattern_rows: int = 120,
    include_patterns: set[str] | None = None,
    fee_bps: float = 1.0,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    short_borrow_bps_per_day: float = 0.0,
    latency_bars: int = 0,
    fast_mode: bool = False,
    max_windows_per_pattern: int | None = None,
    max_train_rows_per_window: int | None = None,
) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()

    work = dataset.copy().sort_values(["datetime", "symbol"]).reset_index(drop=True)
    work["datetime"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
    work = work.dropna(subset=["datetime"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["_realized_return_full"] = _compute_realized_return(
        work,
        horizon_bars=horizon_bars,
        latency_bars=latency_bars,
    )

    min_dt = work["datetime"].min()
    max_dt = work["datetime"].max()
    test_start = min_dt + pd.Timedelta(days=int(train_window_days))
    if test_start >= max_dt:
        return pd.DataFrame()

    pattern_names = [c.replace("pattern_", "") for c in PATTERN_COLUMNS] + ["none"]
    if include_patterns:
        pattern_names = [p for p in pattern_names if p in include_patterns]

    rows: list[dict[str, float | int | str | None]] = []
    for pattern in pattern_names:
        subset_pattern = _extract_pattern_subset(work, pattern=pattern)
        if subset_pattern.empty:
            continue
        subset_pattern["realized_return"] = pd.to_numeric(
            work.loc[subset_pattern.index, "_realized_return_full"],
            errors="coerce",
        )

        windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        cursor = test_start
        while cursor < max_dt:
            train_start = cursor - pd.Timedelta(days=int(train_window_days))
            test_end = cursor + pd.Timedelta(days=int(test_window_days))
            windows.append((train_start, cursor, test_end))
            cursor = cursor + pd.Timedelta(days=int(step_days))
        if max_windows_per_pattern is not None and int(max_windows_per_pattern) > 0 and len(windows) > int(max_windows_per_pattern):
            windows = windows[-int(max_windows_per_pattern) :]

        trade_returns: list[float] = []
        trade_datetimes: list[pd.Timestamp] = []
        used_window_starts: list[pd.Timestamp] = []
        used_window_ends: list[pd.Timestamp] = []
        trades = 0
        windows_used = 0
        for train_start, window_start, test_end in windows:
            train_win = subset_pattern.loc[
                (subset_pattern["datetime"] >= train_start) & (subset_pattern["datetime"] < window_start)
            ].copy()
            test_win = subset_pattern.loc[
                (subset_pattern["datetime"] >= window_start) & (subset_pattern["datetime"] < test_end)
            ].copy()

            train_win = _cap_rows_for_speed(train_win, max_rows=max_train_rows_per_window)
            if len(train_win) < int(min_pattern_rows) or test_win.empty:
                continue

            train_win = train_win.sort_values("datetime").reset_index(drop=True)
            x_train_raw, y_train, _ = split_xy(train_win)
            selection = select_features(
                x_train=x_train_raw,
                y_train=y_train,
                config=FeatureSelectionConfig(
                    min_features=15 if fast_mode else 25,
                    max_features=120 if fast_mode else 260,
                    random_state=42,
                ),
            )
            feature_cols = selection.selected_features
            if not feature_cols:
                continue

            split_idx = int(round(len(train_win) * 0.8))
            split_idx = max(1, min(split_idx, len(train_win) - 1))
            fit_win = train_win.iloc[:split_idx].copy()
            cal_win = train_win.iloc[split_idx:].copy()

            x_fit = fit_win.reindex(columns=feature_cols).fillna(x_train_raw.median(numeric_only=True)).fillna(0.0)
            y_fit = fit_win["future_direction"].astype(int)
            if y_fit.nunique() < 2:
                continue

            model = build_stacking_classifier(random_state=42, n_jobs=1, fast_mode=fast_mode)
            model.fit(x_fit, y_fit)

            calibration = None
            tuned_long = 0.55
            tuned_short = 0.45
            if not cal_win.empty and not fast_mode:
                x_cal = cal_win.reindex(columns=feature_cols).fillna(x_fit.median(numeric_only=True)).fillna(0.0)
                y_cal = cal_win["future_direction"].astype(int)
                if y_cal.nunique() >= 2:
                    prob_cal_raw = model.predict_proba(x_cal)[:, 1]
                    calibration = fit_probability_calibration(prob_cal_raw, y_cal)
                    prob_cal = apply_probability_calibration(prob_cal_raw, calibration)
                    tuned = optimize_thresholds_from_validation(
                        prob_up=prob_cal,
                        future_return=cal_win["future_return"],
                        min_trades=max(10, int(len(cal_win) * 0.05)),
                    )
                    tuned_long = float(tuned.long_threshold)
                    tuned_short = float(tuned.short_threshold)

            x_test = test_win.reindex(columns=feature_cols).fillna(x_fit.median(numeric_only=True)).fillna(0.0)
            prob_test_raw = model.predict_proba(x_test)[:, 1]
            prob_test = apply_probability_calibration(prob_test_raw, calibration)

            test_win["pred_prob_up"] = prob_test
            test_win["position"] = 0
            test_win.loc[test_win["pred_prob_up"] >= tuned_long, "position"] = 1
            test_win.loc[test_win["pred_prob_up"] <= tuned_short, "position"] = -1
            test_win = test_win.loc[test_win["position"] != 0].copy()
            if test_win.empty:
                continue

            test_win = test_win.dropna(subset=["realized_return"]).copy()
            if test_win.empty:
                continue

            test_win["strategy_return"] = _apply_execution_realism(
                subset=test_win,
                interval=interval,
                horizon_bars=horizon_bars,
                fee_bps=fee_bps,
                spread_bps=spread_bps,
                slippage_bps=slippage_bps,
                short_borrow_bps_per_day=short_borrow_bps_per_day,
                latency_bars=latency_bars,
            )

            trade_datetimes.extend(pd.to_datetime(test_win["datetime"], utc=True, errors="coerce").tolist())
            trade_returns.extend(test_win["strategy_return"].astype(float).tolist())
            used_window_starts.append(window_start)
            used_window_ends.append(test_end)
            trades += int(len(test_win))
            windows_used += 1

        if trades == 0:
            continue

        strat = pd.Series(trade_returns, dtype=float)
        period_returns = _aggregate_returns_by_datetime(
            pd.DataFrame({"datetime": trade_datetimes, "strategy_return": trade_returns}),
            column="strategy_return",
        )
        backtest_start_iso = _to_iso_utc(min(used_window_starts)) if used_window_starts else _to_iso_utc(min(trade_datetimes))
        backtest_end_iso = _to_iso_utc(max(used_window_ends)) if used_window_ends else _to_iso_utc(max(trade_datetimes))
        rows.append(
            {
                "pattern": pattern,
                "model_file": "walk_forward_retrain",
                "backtest_start_datetime": backtest_start_iso,
                "backtest_end_datetime": backtest_end_iso,
                "windows_used": int(windows_used),
                "trades": int(trades),
                "win_rate": float((strat > 0).mean()),
                "avg_trade_return": float(strat.mean()),
                "cumulative_return": _safe_cumulative_return(period_returns),
                "sharpe": _safe_sharpe_for_interval(period_returns, interval=interval),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "pattern",
                "model_file",
                "backtest_start_datetime",
                "backtest_end_datetime",
                "windows_used",
                "trades",
                "win_rate",
                "avg_trade_return",
                "cumulative_return",
                "sharpe",
            ]
        )
    return pd.DataFrame(rows).sort_values("cumulative_return", ascending=False).reset_index(drop=True)


def summarize_pattern_coverage(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame(columns=["pattern", "count", "share"])

    total = len(dataset)
    rows: list[dict[str, float | int | str]] = []
    for col in PATTERN_COLUMNS:
        count = int(dataset[col].sum()) if col in dataset.columns else 0
        rows.append({"pattern": col.replace("pattern_", ""), "count": count, "share": count / total})

    none_count = int(dataset["pattern"].isna().sum()) if "pattern" in dataset.columns else 0
    rows.append({"pattern": "none", "count": none_count, "share": none_count / total})
    return pd.DataFrame(rows).sort_values("count", ascending=False)
