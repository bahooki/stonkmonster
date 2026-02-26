from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
import traceback
from typing import Any, Callable
from uuid import uuid4

import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    if is_dataclass(value):
        return _to_json_safe(asdict(value))
    return str(value)


class AsyncJobManager:
    def __init__(self, jobs_dir: Path, max_workers: int = 1) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir = self.jobs_dir / "meta"
        self.result_dir = self.jobs_dir / "results"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix="stonk-jobs")
        self._futures: dict[str, Future] = {}
        self._lock = Lock()

    def _meta_path(self, job_id: str) -> Path:
        return self.meta_dir / f"{job_id}.json"

    def _write_meta(self, meta: dict[str, Any]) -> None:
        path = self._meta_path(str(meta["job_id"]))
        path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")

    def _read_meta(self, job_id: str) -> dict[str, Any] | None:
        path = self._meta_path(job_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _update_meta(self, job_id: str, **updates: Any) -> dict[str, Any] | None:
        with self._lock:
            meta = self._read_meta(job_id)
            if meta is None:
                return None
            meta.update(updates)
            self._write_meta(meta)
            return meta

    def submit(self, name: str, fn: Callable[[], Any]) -> str:
        job_id = uuid4().hex
        meta = {
            "job_id": job_id,
            "name": str(name),
            "status": "queued",
            "created_at": _utc_now_iso(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "traceback": None,
            "result_type": None,
            "result_path": None,
            "result_preview": None,
        }
        self._write_meta(meta)

        future = self._executor.submit(self._run_job, job_id, fn)
        with self._lock:
            self._futures[job_id] = future
        return job_id

    def _persist_result(self, job_id: str, result: Any) -> dict[str, Any]:
        if isinstance(result, pd.DataFrame):
            result_path = self.result_dir / f"{job_id}.parquet"
            result.to_parquet(result_path, index=False)
            preview = {
                "rows": int(len(result)),
                "columns": [str(c) for c in result.columns],
                "head": (
                    result.head(25)
                    .astype(object)
                    .where(pd.notna(result.head(25)), None)
                    .to_dict(orient="records")
                ),
            }
            return {
                "result_type": "dataframe",
                "result_path": str(result_path),
                "result_preview": preview,
            }

        payload = _to_json_safe(result)
        result_path = self.result_dir / f"{job_id}.json"
        result_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        return {
            "result_type": "json",
            "result_path": str(result_path),
            "result_preview": payload,
        }

    def _run_job(self, job_id: str, fn: Callable[[], Any]) -> None:
        self._update_meta(job_id, status="running", started_at=_utc_now_iso(), error=None, traceback=None)
        try:
            result = fn()
            result_payload = self._persist_result(job_id, result)
            self._update_meta(
                job_id,
                status="succeeded",
                finished_at=_utc_now_iso(),
                **result_payload,
            )
        except Exception as exc:
            self._update_meta(
                job_id,
                status="failed",
                finished_at=_utc_now_iso(),
                error=str(exc),
                traceback=traceback.format_exc(limit=30),
            )
        finally:
            with self._lock:
                self._futures.pop(job_id, None)

    def list_jobs(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.meta_dir.glob("*.json")):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append(meta)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "job_id",
                    "name",
                    "status",
                    "created_at",
                    "started_at",
                    "finished_at",
                    "error",
                    "result_type",
                    "result_path",
                ]
            )

        table = pd.DataFrame(rows)
        return table.sort_values("created_at", ascending=False).reset_index(drop=True)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._read_meta(job_id)

    def load_result(self, job_id: str) -> Any:
        meta = self.get_job(job_id)
        if meta is None:
            return None

        result_type = meta.get("result_type")
        result_path = meta.get("result_path")
        if not result_path:
            return meta.get("result_preview")

        path = Path(str(result_path))
        if not path.exists():
            return meta.get("result_preview")

        if result_type == "dataframe":
            try:
                return pd.read_parquet(path)
            except Exception:
                return meta.get("result_preview")

        if result_type == "json":
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return meta.get("result_preview")

        return meta.get("result_preview")

    def delete_job(self, job_id: str) -> bool:
        meta = self.get_job(job_id)
        if meta is None:
            return False

        status = str(meta.get("status", ""))
        with self._lock:
            fut = self._futures.get(job_id)
            if fut is not None and not fut.done():
                return False

        result_path = meta.get("result_path")
        if result_path:
            path = Path(str(result_path))
            if path.exists():
                path.unlink()

        meta_path = self._meta_path(job_id)
        if meta_path.exists():
            meta_path.unlink()

        return status in {"queued", "running", "succeeded", "failed"}
