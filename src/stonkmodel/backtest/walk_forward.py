from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stonkmodel.features.patterns import PATTERN_COLUMNS
from stonkmodel.models.stacking import PatternModelIO


@dataclass
class BacktestResult:
    pattern: str
    model_file: str
    trades: int
    win_rate: float
    avg_trade_return: float
    cumulative_return: float
    sharpe: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "pattern": self.pattern,
            "model_file": self.model_file,
            "trades": self.trades,
            "win_rate": self.win_rate,
            "avg_trade_return": self.avg_trade_return,
            "cumulative_return": self.cumulative_return,
            "sharpe": self.sharpe,
        }


def _safe_sharpe(series: pd.Series) -> float:
    std = series.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((series.mean() / std) * np.sqrt(252))


def run_pattern_backtests(
    dataset: pd.DataFrame,
    model_io: PatternModelIO,
    interval: str,
    horizon_bars: int,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    fee_bps: float = 1.0,
    include_patterns: set[str] | None = None,
    include_model_files: set[str] | None = None,
) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()

    test = dataset.loc[dataset["split"] == "test"].copy()
    if test.empty:
        cutoff = int(len(dataset) * 0.8)
        test = dataset.iloc[cutoff:].copy()

    rows: list[dict[str, float | int | str]] = []

    for model_path in model_io.list_models():
        model_file = model_path.name
        if include_model_files and model_file not in include_model_files:
            continue

        try:
            payload = model_io.load_from_path(model_path)
        except Exception:
            continue

        if payload.get("interval") != interval or int(payload.get("horizon_bars", -1)) != int(horizon_bars):
            continue

        pattern = str(payload["pattern"])
        if include_patterns and pattern not in include_patterns:
            continue

        model = payload["model"]
        feature_cols = payload["feature_columns"]

        pattern_col = f"pattern_{pattern}"
        if pattern == "none":
            subset = test.loc[test["pattern"].isna()].copy()
        elif pattern_col in test.columns:
            subset = test.loc[test[pattern_col] == 1].copy()
        else:
            continue

        if subset.empty:
            continue

        x = subset.reindex(columns=feature_cols).fillna(subset.median(numeric_only=True)).fillna(0.0)
        prob = model.predict_proba(x)[:, 1]

        subset["pred_prob_up"] = prob
        subset["position"] = 0
        subset.loc[subset["pred_prob_up"] >= long_threshold, "position"] = 1
        subset.loc[subset["pred_prob_up"] <= short_threshold, "position"] = -1
        subset = subset.loc[subset["position"] != 0].copy()

        if subset.empty:
            continue

        fee = fee_bps / 10000.0
        subset["strategy_return"] = subset["position"] * subset["future_return"] - fee

        result = BacktestResult(
            pattern=pattern,
            model_file=model_file,
            trades=int(len(subset)),
            win_rate=float((subset["strategy_return"] > 0).mean()),
            avg_trade_return=float(subset["strategy_return"].mean()),
            cumulative_return=float((1.0 + subset["strategy_return"]).prod() - 1.0),
            sharpe=_safe_sharpe(subset["strategy_return"]),
        )
        rows.append(result.to_dict())

    if not rows:
        return pd.DataFrame(
            columns=["pattern", "model_file", "trades", "win_rate", "avg_trade_return", "cumulative_return", "sharpe"]
        )

    return pd.DataFrame(rows).sort_values("cumulative_return", ascending=False)


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
