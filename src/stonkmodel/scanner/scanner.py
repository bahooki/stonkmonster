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
        macro_features: pd.DataFrame | None = None,
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
        features = merge_external_features(
            features,
            fundamental_table=fundamentals,
            politician_features=politician_features,
            macro_features=macro_features,
        )

        # Keep latest bar per symbol for scanning.
        latest = features.sort_values(["symbol", "datetime"]).groupby("symbol", as_index=False).tail(1)

        rows: list[dict[str, float | str]] = []
        for model_path in self.model_io.list_models():
            if include_model_files and model_path.name not in include_model_files:
                continue

            try:
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
                subset["meta_prob_trade"] = np.nan

                meta = payload.get("meta_filter", {}) if isinstance(payload, dict) else {}
                if isinstance(meta, dict) and bool(meta.get("enabled", False)):
                    active_mask = subset["signal"] != "flat"
                    model_meta = meta.get("model")
                    meta_cols = [str(c) for c in list(meta.get("feature_columns", []))]
                    if bool(active_mask.any()) and model_meta is not None and meta_cols:
                        side = np.where(subset["signal"] == "up", 1, np.where(subset["signal"] == "down", -1, 0))
                        meta_frame = subset.reindex(columns=feature_cols).copy()
                        meta_frame["meta_prob_up"] = subset["prob_up"].to_numpy(dtype=float)
                        meta_frame["meta_abs_edge"] = (subset["prob_up"] - 0.5).abs().to_numpy(dtype=float)
                        meta_frame["meta_side"] = pd.Series(side, index=subset.index, dtype=float)
                        meta_frame["meta_signed_edge"] = np.where(side >= 0, subset["prob_up"] - 0.5, 0.5 - subset["prob_up"])
                        x_meta = meta_frame.loc[active_mask].reindex(columns=meta_cols).copy()
                        x_meta = x_meta.fillna(x_meta.median(numeric_only=True)).fillna(0.0)
                        try:
                            meta_prob = model_meta.predict_proba(x_meta)[:, 1]
                            threshold = float(np.clip(float(meta.get("threshold", 0.55)), 0.5, 0.95))
                            keep_active = meta_prob >= threshold
                            subset.loc[active_mask, "meta_prob_trade"] = meta_prob
                            idx_active = subset.index[active_mask]
                            drop_idx = idx_active[~keep_active]
                            subset.loc[drop_idx, "signal"] = "flat"
                        except Exception:
                            pass
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
                            "meta_prob_trade": float(row["meta_prob_trade"]) if pd.notna(row["meta_prob_trade"]) else np.nan,
                        }
                    )
            except Exception:
                # Skip invalid/corrupt model artifacts and continue scanning remaining models.
                continue

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
                    "meta_prob_trade",
                ]
            )

        out = pd.DataFrame(rows)
        out = out.sort_values(["confidence", "model_roc_auc", "edge"], ascending=False).drop_duplicates(
            subset=["symbol"], keep="first"
        )
        return out.head(config.top_n).reset_index(drop=True)
