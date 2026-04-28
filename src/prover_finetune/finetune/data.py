import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

DEEPSEEK_PROVER_V2_PROMPT = """Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""


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


def _normalize_reasoning_steps(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines()]
        return [line for line in lines if line]
    return []


def _format_deepseek_prover_v2_example(row: dict[str, Any], data_cfg: dict[str, Any]) -> dict[str, str]:
    formal_statement_field = data_cfg.get("formal_statement_field", "formal_statement")
    reasoning_steps_field = data_cfg.get("reasoning_steps_field", "reasoning_steps")
    proof_field = data_cfg.get("proof_field", "formal_proof_no_comments")
    prompt_template = data_cfg.get("deepseek_prompt_template", DEEPSEEK_PROVER_V2_PROMPT)

    formal_statement = str(row.get(formal_statement_field, "")).strip()
    reasoning_steps = _normalize_reasoning_steps(row.get(reasoning_steps_field, []))
    formal_proof = str(row.get(proof_field, "")).strip()

    if reasoning_steps:
        plan_lines = "\n".join([f"{idx}. {step}" for idx, step in enumerate(reasoning_steps, start=1)])
    else:
        plan_lines = "1. Introduce the goal and key assumptions.\n2. Apply intermediate lemmas to transform the target.\n3. Close the goal with algebraic/simp tactics."

    assistant_completion = (
        "Proof plan:\n"
        f"{plan_lines}\n\n"
        "Lean 4 code:\n"
        "```lean4\n"
        f"{formal_proof}\n"
        "```"
    )
    text = f"{prompt_template.format(formal_statement)}\n\n{assistant_completion}"
    return {"text": text}


def _resolve_formatter_type(data_cfg: dict[str, Any], model_cfg: dict[str, Any] | None) -> str:
    configured = data_cfg.get("formatter_type", "auto")
    if configured != "auto":
        return str(configured)

    model_name = ""
    if model_cfg is not None:
        model_name = str(model_cfg.get("name_or_path", "")).lower()
    if "deepseek-ai/deepseek-prover-v2" in model_name:
        return "deepseek_prover_v2"
    return "generic"


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


def load_and_process_dataset(
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any] | None = None,
) -> tuple[Dataset, Dataset | None]:
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

    formatter_type = _resolve_formatter_type(data_cfg, model_cfg)
    if formatter_type == "deepseek_prover_v2":
        formatter = lambda x: _format_deepseek_prover_v2_example(x, data_cfg)
    elif formatter_type == "generic":
        formatter = lambda x: _format_example(x, text_field, target_field, template)
    else:
        raise ValueError(f"Unsupported formatter_type: {formatter_type}")

    train_ds = train_ds.map(formatter, remove_columns=None)
    if eval_ds is not None:
        eval_ds = eval_ds.map(formatter, remove_columns=None)

    return train_ds, eval_ds

