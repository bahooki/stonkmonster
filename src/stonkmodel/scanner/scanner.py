from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stonkmodel.data.external_features import merge_external_features
from stonkmodel.features.indicators import add_indicators
from stonkmodel.features.patterns import add_candlestick_patterns
from stonkmodel.models.calibration import apply_probability_calibration, resolve_thresholds
from stonkmodel.models.stacking import PatternModelIO


@dataclass
class ScanConfig:
    interval: str
    horizon_bars: int
    top_n: int = 50
    min_confidence: float = 0.5
    use_model_thresholds: bool = True
    long_threshold: float | None = None
    short_threshold: float | None = None


class SignalScanner:
    def __init__(self, model_io: PatternModelIO) -> None:
        self.model_io = model_io

    def scan(
        self,
        latest_frames: dict[str, pd.DataFrame],
        config: ScanConfig,
        fundamentals: pd.DataFrame | None = None,
        politician_features: pd.DataFrame | None = None,
        include_patterns: set[str] | None = None,
        include_model_files: set[str] | None = None,
    ) -> pd.DataFrame:
        if not latest_frames:
            return pd.DataFrame()

        frame = pd.concat(latest_frames.values(), ignore_index=True)
        if frame.empty:
            return frame

        features = add_candlestick_patterns(frame)
        features = add_indicators(features)
        features = merge_external_features(features, fundamentals, politician_features)

        # Keep latest bar per symbol for scanning.
        latest = features.sort_values(["symbol", "datetime"]).groupby("symbol", as_index=False).tail(1)

        rows: list[dict[str, float | str]] = []
        for model_path in self.model_io.list_models():
            if include_model_files and model_path.name not in include_model_files:
                continue

            payload = self.model_io.load_from_path(model_path)
            if payload.get("interval") != config.interval or int(payload.get("horizon_bars", -1)) != int(config.horizon_bars):
                continue

            pattern = str(payload["pattern"])
            if include_patterns and pattern not in include_patterns:
                continue

            model = payload["model"]
            feature_cols: list[str] = payload["feature_columns"]
            metrics = payload.get("metrics", {})
            long_t, short_t = resolve_thresholds(
                payload=payload,
                long_threshold=config.long_threshold,
                short_threshold=config.short_threshold,
                use_model_thresholds=config.use_model_thresholds,
            )

            if pattern == "none":
                subset = latest.loc[latest["pattern"].isna()].copy()
            else:
                col = f"pattern_{pattern}"
                if col not in latest.columns:
                    continue
                subset = latest.loc[latest[col] == 1].copy()

            if subset.empty:
                continue

            x = subset.reindex(columns=feature_cols).fillna(subset.median(numeric_only=True)).fillna(0.0)
            prob_up_raw = model.predict_proba(x)[:, 1]
            prob_up = apply_probability_calibration(prob_up_raw, payload.get("probability_calibration"))
            prob_down = 1.0 - prob_up
            subset["prob_up"] = prob_up
            subset["prob_down"] = prob_down
            subset["signal"] = "flat"
            subset.loc[subset["prob_up"] >= long_t, "signal"] = "up"
            subset.loc[subset["prob_up"] <= short_t, "signal"] = "down"
            subset = subset.loc[subset["signal"] != "flat"].copy()
            if subset.empty:
                continue

            subset["confidence"] = np.where(
                subset["signal"] == "up",
                subset["prob_up"],
                subset["prob_down"],
            )
            subset["edge"] = np.where(
                subset["signal"] == "up",
                subset["prob_up"] - long_t,
                short_t - subset["prob_up"],
            )

            subset = subset.loc[subset["confidence"] >= float(config.min_confidence)].copy()
            if subset.empty:
                continue

            for _, row in subset.iterrows():
                rows.append(
                    {
                        "symbol": row["symbol"],
                        "datetime": row["datetime"],
                        "pattern": pattern,
                        "model_file": model_path.name,
                        "signal": row["signal"],
                        "prob_up": float(row["prob_up"]),
                        "prob_down": float(row["prob_down"]),
                        "confidence": float(row["confidence"]),
                        "edge": float(row["edge"]),
                        "model_roc_auc": float(metrics.get("roc_auc", np.nan)),
                        "long_threshold": float(long_t),
                        "short_threshold": float(short_t),
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "datetime",
                    "pattern",
                    "model_file",
                    "signal",
                    "prob_up",
                    "prob_down",
                    "confidence",
                    "edge",
                    "model_roc_auc",
                    "long_threshold",
                    "short_threshold",
                ]
            )

        out = pd.DataFrame(rows)
        out = out.sort_values(["confidence", "model_roc_auc", "edge"], ascending=False).drop_duplicates(
            subset=["symbol"], keep="first"
        )
        return out.head(config.top_n).reset_index(drop=True)
