import argparse
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


def _safe_get(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None:
        return ""
    return value


def _build_token_counter(tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    return (
        lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        f"llm_tokenizer:{tokenizer_name}",
    )


def _bucket_label(token_count: int) -> str:
    buckets = [
        (0, 256),
        (257, 512),
        (513, 1024),
        (1025, 2048),
        (2049, 4096),
    ]
    for lo, hi in buckets:
        if lo <= token_count <= hi:
            return f"{lo}-{hi}"
    return ">4096"


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
        default="compete",
        help="ground_truth_type value used for filtering.",
    )
    parser.add_argument(
        "--formal-field",
        default="formal_ground_truth",
        help="Field name for formal ground truth text.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        default="data/hf_cache/datasets",
        help="Local cache directory for downloaded Hugging Face datasets.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="gpt2",
        help="Tokenizer model name for token counting.",
    )
    args = parser.parse_args()

    dataset_cache_dir = Path(args.dataset_cache_dir)
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        path=args.dataset_name,
        name=args.dataset_config,
        split=args.split,
        cache_dir=str(dataset_cache_dir),
    )
    count_tokens, tokenizer_mode = _build_token_counter(tokenizer_name=args.tokenizer_name)

    filtered_count = 0
    bucket_counts: dict[str, int] = {
        "0-256": 0,
        "257-512": 0,
        "513-1024": 0,
        "1025-2048": 0,
        "2049-4096": 0,
        ">4096": 0,
    }
    max_formal_tokens = 0
    max_formal_idx = -1

    for idx, row in enumerate(dataset):
        author = str(_safe_get(row, args.author_field))
        gt_type = str(_safe_get(row, args.ground_truth_type_field))
        if author != "human" or gt_type != args.ground_truth_type_value:
            continue

        filtered_count += 1

        formal_text = str(_safe_get(row, args.formal_field))
        token_count = count_tokens(formal_text)
        bucket_counts[_bucket_label(token_count)] += 1
        if token_count > max_formal_tokens:
            max_formal_tokens = token_count
            max_formal_idx = idx

    print(f"dataset={args.dataset_name}")
    print(f"split={args.split}")
    print(f"total_samples={len(dataset)}")
    print("filter=author=human AND ground_truth_type=compete")
    print(f"tokenizer_mode={tokenizer_mode}")
    print(f"filtered_samples={filtered_count}")
    print(f"formal_ground_truth_max_tokens={max_formal_tokens}")
    print(f"formal_ground_truth_max_tokens_sample_index={max_formal_idx}")
    print("formal_ground_truth_token_bucket_distribution:")
    for label in ["0-256", "257-512", "513-1024", "1025-2048", "2049-4096", ">4096"]:
        print(f"  {label}: {bucket_counts[label]}")


if __name__ == "__main__":
    main()
