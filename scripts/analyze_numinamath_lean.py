import argparse
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
        description="Analyze AI-MO/NuminaMath-LEAN dataset statistics."
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
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-Math-7B-Instruct",
        help="Tokenizer name for counting formal_ground_truth tokens.",
    )
    parser.add_argument(
        "--author-field",
        default="author",
        help="Field name for author.",
    )
    parser.add_argument(
        "--formal-field",
        default="formal_ground_truth",
        help="Field name for formal ground truth text.",
    )
    args = parser.parse_args()

    dataset = load_dataset(
        path=args.dataset_name,
        name=args.dataset_config,
        split=args.split,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    human_count = 0
    max_formal_tokens = 0
    max_formal_idx = -1

    for idx, row in enumerate(dataset):
        author = str(_safe_get(row, args.author_field))
        if author == "human":
            human_count += 1

        formal_text = str(_safe_get(row, args.formal_field))
        token_count = len(tokenizer.encode(formal_text, add_special_tokens=False))
        if token_count > max_formal_tokens:
            max_formal_tokens = token_count
            max_formal_idx = idx

    print(f"dataset={args.dataset_name}")
    print(f"split={args.split}")
    print(f"total_samples={len(dataset)}")
    print(f"author_human_count={human_count}")
    print(f"formal_ground_truth_max_tokens={max_formal_tokens}")
    print(f"formal_ground_truth_max_tokens_sample_index={max_formal_idx}")


if __name__ == "__main__":
    main()
