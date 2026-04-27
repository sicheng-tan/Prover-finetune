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


def _build_token_counter(tokenizer_name: str, tokenizer_max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.model_max_length = tokenizer_max_length
    return (
        lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        f"llm_tokenizer:{tokenizer_name},max_length={tokenizer_max_length}",
    )


def _bucket_label(token_count: int) -> str:
    buckets = [
        (0, 256),
        (257, 512),
        (513, 1024),
        (1025, 2048),
        (2049, 4096),
        (4097, 8192),
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
        default="complete",
        help="ground_truth_type value used for filtering.",
    )
    parser.add_argument(
        "--formal-field",
        default="formal_ground_truth",
        help="Field name for formal ground truth text.",
    )
    parser.add_argument(
        "--problem-field",
        default="problem",
        help="Field name for the natural language problem text.",
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
    parser.add_argument(
        "--tokenizer-max-length",
        type=int,
        default=131072,
        help="Tokenizer max_length override for long formal proofs.",
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
    count_tokens, tokenizer_mode = _build_token_counter(
        tokenizer_name=args.tokenizer_name,
        tokenizer_max_length=args.tokenizer_max_length,
    )

    filtered_count = 0
    bucket_counts: dict[str, int] = {
        "0-256": 0,
        "257-512": 0,
        "513-1024": 0,
        "1025-2048": 0,
        "2049-4096": 0,
        "4097-8192": 0,
        ">8192": 0,
    }
    max_formal_tokens = 0
    max_formal_idx = -1
    min_formal_tokens = 10**18
    min_formal_idx = -1
    min_problem_text = ""
    min_formal_text = ""

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
        if token_count < min_formal_tokens:
            min_formal_tokens = token_count
            min_formal_idx = idx
            min_problem_text = str(_safe_get(row, args.problem_field))
            min_formal_text = formal_text

    print(f"dataset={args.dataset_name}")
    print(f"split={args.split}")
    print(f"total_samples={len(dataset)}")
    print(
        "filter="
        f"{args.author_field}=human AND "
        f"{args.ground_truth_type_field}={args.ground_truth_type_value}"
    )
    print(f"tokenizer_mode={tokenizer_mode}")
    print(f"filtered_samples={filtered_count}")
    print(f"formal_ground_truth_max_tokens={max_formal_tokens}")
    print(f"formal_ground_truth_max_tokens_sample_index={max_formal_idx}")
    if filtered_count > 0:
        print(f"formal_ground_truth_min_tokens={min_formal_tokens}")
        print(f"formal_ground_truth_min_tokens_sample_index={min_formal_idx}")
        print("formal_ground_truth_min_tokens_problem:")
        print(min_problem_text)
        print("formal_ground_truth_min_tokens_proof:")
        print(min_formal_text)
    print("formal_ground_truth_token_bucket_distribution:")
    for label in ["0-256", "257-512", "513-1024", "1025-2048", "2049-4096", "4097-8192", ">8192"]:
        print(f"  {label}: {bucket_counts[label]}")


if __name__ == "__main__":
    main()
