"""Input and output helpers for the file-based competition harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_tasks(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("tasks.json must contain a JSON array")

    tasks: list[dict[str, str]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"task at index {index} must be an object")
        task_id = item.get("task_id")
        prompt = item.get("prompt")
        if task_id is None:
            raise ValueError(f"task at index {index} is missing task_id")
        if prompt is None:
            raise ValueError(f"task {task_id!r} is missing prompt")
        tasks.append({"task_id": str(task_id), "prompt": str(prompt)})

    return tasks


def write_results(path: str, results: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, separators=(",", ":"))

    temp_path.replace(output_path)
