import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


def _safe_get(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None:
        return ""
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter AI-MO/NuminaMath-LEAN with fixed quality conditions."
    )
    parser.add_argument(
        "--dataset-name",
        default="AI-MO/NuminaMath-LEAN",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional Hugging Face dataset config name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to filter.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        default="data/hf_cache/datasets",
        help="Local cache directory for downloaded Hugging Face datasets.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="gpt2",
        help="Tokenizer model name used for token counting.",
    )
    parser.add_argument(
        "--tokenizer-max-length",
        type=int,
        default=131072,
        help="Tokenizer max_length override for long formal proofs.",
    )
    parser.add_argument(
        "--author-field",
        default="author",
        help="Field name for author.",
    )
    parser.add_argument(
        "--ground-truth-type-field",
        default="ground_truth_type",
        help="Field name for ground truth type.",
    )
    parser.add_argument(
        "--ground-truth-type-value",
        default="complete",
        help="ground_truth_type value used for filtering.",
    )
    parser.add_argument(
        "--formal-field",
        default="formal_ground_truth",
        help="Field name for formal ground truth text.",
    )
    parser.add_argument(
        "--max-formal-tokens",
        type=int,
        default=4096,
        help="Keep samples with formal_ground_truth token count <= this value.",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/numinamath_lean_filtered_train.jsonl",
        help="Output JSONL path for filtered samples.",
    )
    args = parser.parse_args()

    dataset_cache_dir = Path(args.dataset_cache_dir)
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        path=args.dataset_name,
        name=args.dataset_config,
        split=args.split,
        cache_dir=str(dataset_cache_dir),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    tokenizer.model_max_length = args.tokenizer_max_length

    total_samples = 0
    matched_samples = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            total_samples += 1
            author = str(_safe_get(row, args.author_field))
            gt_type = str(_safe_get(row, args.ground_truth_type_field))
            if author != "human" or gt_type != args.ground_truth_type_value:
                continue

            formal_text = str(_safe_get(row, args.formal_field))
            token_count = len(tokenizer.encode(formal_text, add_special_tokens=False))
            if token_count > args.max_formal_tokens:
                continue

            row_with_stats = dict(row)
            row_with_stats["formal_ground_truth_token_count"] = token_count
            f.write(json.dumps(row_with_stats, ensure_ascii=False) + "\n")
            matched_samples += 1

    print(f"dataset={args.dataset_name}")
    print(f"split={args.split}")
    print(f"tokenizer={args.tokenizer_name}")
    print(
        "filter="
        f"{args.author_field}=human AND "
        f"{args.ground_truth_type_field}={args.ground_truth_type_value} AND "
        f"{args.formal_field}_tokens<={args.max_formal_tokens}"
    )
    print(f"total_samples={total_samples}")
    print(f"matched_samples={matched_samples}")
    print(f"output_path={output_path}")


if __name__ == "__main__":
    main()
