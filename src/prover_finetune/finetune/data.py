import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


def _load_jsonl(path: str | Path) -> Dataset:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def _format_example(
    row: dict[str, Any],
    text_field: str | None,
    target_field: str | None,
    template: str,
) -> dict[str, str]:
    prompt = str(row.get(text_field, "")) if text_field else ""
    completion = str(row.get(target_field, "")) if target_field else ""
    text = template.format(prompt=prompt, completion=completion, **row)
    return {"text": text}


def _load_hf_split(data_cfg: dict[str, Any], split_key: str) -> Dataset | None:
    split_name = data_cfg.get(split_key)
    if not split_name:
        return None

    dataset_name = data_cfg["dataset_name"]
    dataset_config = data_cfg.get("dataset_config")
    data_files = data_cfg.get("data_files")
    streaming = bool(data_cfg.get("streaming", False))

    ds = load_dataset(
        path=dataset_name,
        name=dataset_config,
        split=split_name,
        data_files=data_files,
        streaming=streaming,
    )
    if streaming:
        ds = Dataset.from_list(list(ds))
    return ds


def load_and_process_dataset(data_cfg: dict[str, Any]) -> tuple[Dataset, Dataset | None]:
    source_type = data_cfg.get("source_type", "jsonl")
    text_field = data_cfg.get("text_field")
    target_field = data_cfg.get("target_field")
    template = data_cfg.get("template", "{prompt}\n{completion}")
    if source_type == "jsonl":
        train_path = data_cfg["train_path"]
        eval_path = data_cfg.get("eval_path")
        train_ds = _load_jsonl(train_path)
        eval_ds = _load_jsonl(eval_path) if eval_path else None
    elif source_type == "huggingface":
        train_ds = _load_hf_split(data_cfg, "train_split")
        if train_ds is None:
            raise ValueError("For source_type=huggingface, 'train_split' must be provided.")
        eval_ds = _load_hf_split(data_cfg, "eval_split")
    else:
        raise NotImplementedError(f"Unsupported source_type for now: {source_type}")

    train_ds = train_ds.map(lambda x: _format_example(x, text_field, target_field, template), remove_columns=None)
    if eval_ds is not None:
        eval_ds = eval_ds.map(lambda x: _format_example(x, text_field, target_field, template), remove_columns=None)

    return train_ds, eval_ds

