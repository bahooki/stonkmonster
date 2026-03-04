from __future__ import annotations

from pathlib import Path

import pandas as pd

from stonkmodel.config import Settings
from stonkmodel.pipeline import StonkService


def _build_service(tmp_path: Path) -> StonkService:
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    models_dir = tmp_path / "models"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        app_env="test",
        data_dir=data_dir,
        raw_data_dir=raw_dir,
        processed_data_dir=processed_dir,
        models_dir=models_dir,
    )
    return StonkService(settings)


def test_auto_improve_runs_loop(monkeypatch, tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    dataset_stub = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "datetime": pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"], utc=True),
            "future_return": [0.01, 0.02],
            "future_direction": [1, 1],
        }
    )
    monkeypatch.setattr(service.dataset_builder, "load_dataset", lambda name="model_dataset": dataset_stub.copy())

    call_counts = {"train": 0, "bt": 0}

    def _fake_train(**kwargs) -> pd.DataFrame:
        call_counts["train"] += 1
        return pd.DataFrame(
            [
                {
                    "model_file": f"autoloop_fake_{call_counts['train']}.joblib",
                    "pattern": "doji",
                    "roc_auc": 0.55 + (0.01 * call_counts["train"]),
                }
            ]
        )

    def _fake_opt(**kwargs) -> pd.DataFrame:
        return pd.DataFrame([{"threshold_updated": True}])

    def _fake_backtest(**kwargs) -> pd.DataFrame:
        call_counts["bt"] += 1
        c = call_counts["bt"]
        cum = 0.03 * c
        return pd.DataFrame(
            [
                {
                    "pattern": "portfolio_combined",
                    "model_file": "portfolio_combined",
                    "cumulative_return": cum,
                    "sharpe": 0.8 + (0.1 * c),
                    "max_drawdown": -0.10,
                    "win_rate_trade": 0.55,
                    "win_rate": 0.55,
                    "trades": 120,
                },
                {
                    "pattern": "doji",
                    "model_file": f"pattern_model_{c}",
                    "cumulative_return": cum * 0.8,
                    "sharpe": 0.6 + (0.05 * c),
                    "max_drawdown": -0.12,
                    "win_rate_trade": 0.53,
                    "win_rate": 0.53,
                    "trades": 80,
                },
            ]
        )

    monkeypatch.setattr(service, "train", _fake_train)
    monkeypatch.setattr(service, "optimize_model_thresholds_from_backtest", _fake_opt)
    monkeypatch.setattr(service, "backtest", _fake_backtest)

    out = service.auto_improve(
        dataset_name="model_dataset",
        interval="1d",
        iterations=4,
        max_minutes=15,
        patience=4,
        min_significant_improvement=0.01,
        random_seed=7,
    )

    assert not out.empty
    assert "stage" in out.columns
    assert (out["stage"].astype(str) == "baseline").any()
    assert (out["stage"].astype(str) == "search").any()


def test_auto_improve_negative_objective_requires_true_improvement(monkeypatch, tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    dataset_stub = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "datetime": pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"], utc=True),
            "future_return": [0.01, 0.02],
            "future_direction": [1, 1],
        }
    )
    monkeypatch.setattr(service.dataset_builder, "load_dataset", lambda name="model_dataset": dataset_stub.copy())
    monkeypatch.setattr(
        service,
        "train",
        lambda **kwargs: pd.DataFrame([{"model_file": "autoloop_fake.joblib", "pattern": "doji", "roc_auc": 0.51}]),
    )
    monkeypatch.setattr(service, "optimize_model_thresholds_from_backtest", lambda **kwargs: pd.DataFrame([{"threshold_updated": True}]))
    monkeypatch.setattr(
        service,
        "backtest",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "pattern": "doji",
                    "model_file": "pattern_model",
                    "cumulative_return": -0.01,
                    "sharpe": -0.1,
                    "max_drawdown": -0.15,
                    "win_rate_trade": 0.45,
                    "win_rate": 0.45,
                    "trades": 100,
                }
            ]
        ),
    )

    objective_seq = iter([-1.0, -1.05, -1.05])

    def _fake_objective(_: pd.DataFrame, min_trades: int | None = None) -> dict[str, object]:
        score = float(next(objective_seq))
        return {
            "objective_score": score,
            "cumulative_return": score,
            "sharpe": -0.1,
            "max_drawdown": -0.15,
            "trades": 100,
        }

    monkeypatch.setattr(service, "_objective_row_from_backtest", _fake_objective)

    out = service.auto_improve(
        dataset_name="model_dataset",
        interval="1d",
        iterations=1,
        max_minutes=10,
        patience=2,
        min_significant_improvement=0.10,
        random_seed=11,
    )

    search_rows = out.loc[out["stage"].astype(str) == "search"].copy()
    assert not search_rows.empty
    assert int(pd.to_numeric(search_rows["improved"], errors="coerce").fillna(0).iloc[0]) == 0


def test_auto_improve_trade_gate_blocks_low_trade_winner(monkeypatch, tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    dataset_stub = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "datetime": pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"], utc=True),
            "future_return": [0.01, 0.02],
            "future_direction": [1, 1],
        }
    )
    monkeypatch.setattr(service.dataset_builder, "load_dataset", lambda name="model_dataset": dataset_stub.copy())
    monkeypatch.setattr(
        service,
        "train",
        lambda **kwargs: pd.DataFrame([{"model_file": "autoloop_fake.joblib", "pattern": "doji", "roc_auc": 0.51}]),
    )
    monkeypatch.setattr(service, "optimize_model_thresholds_from_backtest", lambda **kwargs: pd.DataFrame([{"threshold_updated": True}]))
    monkeypatch.setattr(
        service,
        "backtest",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "pattern": "portfolio_combined",
                    "model_file": "portfolio_combined",
                    "cumulative_return": 0.10,
                    "sharpe": 1.2,
                    "max_drawdown": -0.05,
                    "win_rate_trade": 0.60,
                    "win_rate": 0.60,
                    "trades": 3,
                }
            ]
        ),
    )

    out = service.auto_improve(
        dataset_name="model_dataset",
        interval="1d",
        iterations=1,
        max_minutes=10,
        patience=2,
        min_significant_improvement=0.10,
        min_iteration_trades=10,
        random_seed=11,
    )

    blocked = out.loc[out["status"].astype(str) == "below_min_trades_gate"].copy()
    assert not blocked.empty
    assert float(pd.to_numeric(blocked["trades"], errors="coerce").iloc[0]) == 3.0
