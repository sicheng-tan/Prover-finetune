import json
from pathlib import Path
from typing import Any


def _load_local_json(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return rows


def _load_local_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_from_lean_dir(root_dir: str | Path, split_file: str | Path) -> list[dict[str, Any]]:
    root = Path(root_dir)
    with open(split_file, "r", encoding="utf-8") as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    out: list[dict[str, Any]] = []
    for rel in rel_paths:
        lean_path = root / rel
        statement = lean_path.read_text(encoding="utf-8")
        out.append({"id": rel, "statement": statement})
    return out


def load_minif2f(minif2f_cfg: dict[str, Any], split: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    source_type = minif2f_cfg.get("source_type", "local_jsonl")
    if source_type == "local_json":
        rows = _load_local_json(minif2f_cfg["json_path"])
    elif source_type == "local_jsonl":
        rows = _load_local_jsonl(minif2f_cfg["jsonl_path"])
    elif source_type == "local_lean_dir":
        rows = _load_from_lean_dir(minif2f_cfg["root_dir"], minif2f_cfg["split_file"])
    else:
        raise NotImplementedError(f"Unsupported miniF2F source_type: {source_type}")

    filtered = [r for r in rows if r.get("split", split) == split or "split" not in r]
    if max_samples is not None:
        filtered = filtered[:max_samples]
    return filtered

