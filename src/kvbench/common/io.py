"""File helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
