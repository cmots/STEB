"""Helpers for sparse shard resume, alignment, and aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .scorers.base import EvalRecord
except ImportError:  # pragma: no cover - script entrypoint path
    from scorers.base import EvalRecord


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_sparse_rows(row_groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for rows in row_groups:
        for row in rows:
            row_id = row.get("id")
            if not row_id:
                continue
            current = merged.setdefault(row_id, {"id": row_id})
            for key, value in row.items():
                if key != "id":
                    current[key] = value
    return [merged[row_id] for row_id in sorted(merged)]


def align_sparse_rows_to_records(
    records: list[EvalRecord],
    sparse_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sparse_by_id = {row["id"]: row for row in sparse_rows if row.get("id")}
    aligned: list[dict[str, Any]] = []
    for record in records:
        row = {"id": record.id}
        row.update({k: v for k, v in sparse_by_id.get(record.id, {}).items() if k != "id"})
        aligned.append(row)
    return aligned


def collect_completed_ids_from_rows(
    rows: list[dict[str, Any]],
    *,
    required_keys: set[str] | None = None,
) -> set[str]:
    if not required_keys:
        return {row["id"] for row in rows if row.get("id")}
    return {
        row["id"]
        for row in rows
        if row.get("id") and all(key in row and row[key] is not None for key in required_keys)
    }


def build_pending_shard_records(
    records: list[EvalRecord],
    *,
    completed_ids: set[str],
    rank: int,
    world_size: int,
) -> list[EvalRecord]:
    pending = [record for record in records if record.id not in completed_ids]
    pending.sort(key=lambda record: record.id)
    return pending[rank::world_size]


def load_sparse_group_rows(
    *,
    group_results_path: str | Path | None = None,
    shard_results_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    row_groups: list[list[dict[str, Any]]] = []

    if group_results_path is not None:
        group_results_file = Path(group_results_path)
        if group_results_file.exists():
            row_groups.append(load_json(group_results_file))

    if shard_results_dir is not None:
        shard_dir = Path(shard_results_dir)
        if shard_dir.exists():
            for path in sorted(shard_dir.glob("shard_*.json")):
                rows = load_json(path)
                if isinstance(rows, list):
                    row_groups.append(rows)

    return merge_sparse_rows(row_groups)
