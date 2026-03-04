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


def test_groundup_register_run_and_lookup(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    train_table = pd.DataFrame(
        [
            {"model_file": "m1.joblib", "roc_auc": 0.58, "train_rows": 1000, "thresholds_updated": 1},
            {"model_file": "m2.joblib", "roc_auc": 0.61, "train_rows": 900, "thresholds_updated": 1},
        ]
    )
    bt = pd.DataFrame(
        [
            {
                "pattern": "portfolio_combined",
                "model_file": "portfolio_combined",
                "cumulative_return": 0.12,
                "sharpe": 1.1,
                "max_drawdown": -0.08,
                "win_rate_trade": 0.58,
                "win_rate": 0.58,
                "trades": 120,
            }
        ]
    )

    row = service.groundup_register_run(
        run_id="run_001",
        run_name="alpha",
        dataset_name="dset",
        interval="1d",
        model_files={"m1.joblib", "m2.joblib"},
        train_table=train_table,
        backtest_table=bt,
        status="challenger",
    )
    assert row["run_id"] == "run_001"
    assert int(row["model_count"]) == 2

    runs = service.groundup_runs()
    assert not runs.empty
    assert (runs["run_id"].astype(str) == "run_001").any()

    files = service.groundup_models_for_run("run_001")
    assert files == {"m1.joblib", "m2.joblib"}


def test_groundup_promotion_flow(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    train = pd.DataFrame([{"model_file": "champ.joblib", "roc_auc": 0.56, "train_rows": 800}])
    bt_champion = pd.DataFrame(
        [
            {
                "pattern": "portfolio_combined",
                "model_file": "portfolio_combined",
                "cumulative_return": 0.05,
                "sharpe": 0.8,
                "max_drawdown": -0.10,
                "win_rate_trade": 0.54,
                "win_rate": 0.54,
                "trades": 100,
            }
        ]
    )
    bt_challenger = pd.DataFrame(
        [
            {
                "pattern": "portfolio_combined",
                "model_file": "portfolio_combined",
                "cumulative_return": 0.11,
                "sharpe": 1.0,
                "max_drawdown": -0.08,
                "win_rate_trade": 0.57,
                "win_rate": 0.57,
                "trades": 130,
            }
        ]
    )

    service.groundup_register_run(
        run_id="champion_run",
        run_name="champ",
        dataset_name="dset",
        interval="1d",
        model_files={"champ.joblib"},
        train_table=train,
        backtest_table=bt_champion,
        status="champion",
    )
    service.groundup_register_run(
        run_id="challenger_run",
        run_name="chall",
        dataset_name="dset",
        interval="1d",
        model_files={"chall.joblib"},
        train_table=train,
        backtest_table=bt_challenger,
        status="challenger",
    )
    service.groundup_set_deployment(
        champion_run_id="champion_run",
        challenger_run_id="challenger_run",
        min_relative_improvement=0.01,
        min_trade_count=80,
        max_champion_age_days=365,
    )

    decision = service.groundup_promotion_decision()
    assert bool(decision.get("promote")) is True

    promoted = service.groundup_promote_challenger()
    assert bool(promoted.get("promoted")) is True
    assert str(promoted.get("new_champion_run_id")) == "challenger_run"

    state = service.groundup_get_deployment()
    assert str(state.get("champion_run_id")) == "challenger_run"


def test_groundup_run_model_file_normalization(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    train = pd.DataFrame([{"model_file": "foo.joblib", "roc_auc": 0.55, "train_rows": 500}])
    bt = pd.DataFrame(
        [
            {
                "pattern": "portfolio_combined",
                "model_file": "portfolio_combined",
                "cumulative_return": 0.02,
                "sharpe": 0.4,
                "max_drawdown": -0.12,
                "win_rate_trade": 0.51,
                "win_rate": 0.51,
                "trades": 50,
            }
        ]
    )

    service.groundup_register_run(
        run_id="run_paths",
        run_name="path-normalize",
        dataset_name="dset",
        interval="1d",
        model_files={
            "/tmp/models/a.joblib",
            "nested/b.joblib",
            "c.joblib",
        },
        train_table=train,
        backtest_table=bt,
        status="challenger",
    )

    files = service.groundup_models_for_run("run_paths")
    assert files == {"a.joblib", "b.joblib", "c.joblib"}
