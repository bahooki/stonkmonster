from __future__ import annotations

from pathlib import Path

import pandas as pd

from stonkmodel.features.dataset import DatasetBuilder


def test_delete_dataset(tmp_path: Path) -> None:
    builder = DatasetBuilder(tmp_path)
    dataset = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "future_return": [0.01, -0.02],
            "future_direction": [1, 0],
        }
    )
    path = builder.save_dataset(dataset, name="demo")
    assert path.exists()

    deleted = builder.delete_dataset("demo")
    assert deleted is True
    assert not path.exists()

    deleted_again = builder.delete_dataset("demo")
    assert deleted_again is False
