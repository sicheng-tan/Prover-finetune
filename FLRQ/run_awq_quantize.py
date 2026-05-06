import argparse
import inspect
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

try:
    # autoawq>=0.2.x
    from awq import AutoAWQForCausalLM
except ImportError:
    # Compatibility fallback for variants that expose different casing.
    from awq import AutoAwqForCausalLM as AutoAWQForCausalLM


def get_wikitext2_calib_texts(tokenizer, n_samples: int = 512, block_size: int = 512) -> list[str]:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test").shuffle(seed=42)
    texts: list[str] = []
    for row in ds:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 0:
            continue
        if len(token_ids) > block_size:
            token_ids = token_ids[:block_size]
        clipped = tokenizer.decode(token_ids, skip_special_tokens=True)
        if clipped.strip():
            texts.append(clipped)
        if len(texts) >= n_samples:
            break
    if not texts:
        raise RuntimeError("No calibration texts collected from wikitext2.")
    return texts


def _build_quantize_kwargs(model, tokenizer, quant_config: dict, calib_texts: list[str], max_calib_seq_len: int) -> dict:
    sig = inspect.signature(model.quantize)
    params = sig.parameters
    kwargs: dict = {}
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    if "quant_config" in params:
        kwargs["quant_config"] = quant_config
    if "calib_data" in params:
        kwargs["calib_data"] = calib_texts
    if "max_calib_seq_len" in params:
        kwargs["max_calib_seq_len"] = max_calib_seq_len
    if "max_calib_samples" in params:
        kwargs["max_calib_samples"] = len(calib_texts)
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="AWQ quantization with FLRQ-like defaults.")
    parser.add_argument("--model_path", type=str, required=True, help="HF model id or local path.")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for AWQ model.")
    parser.add_argument("--qbit", type=int, default=4, help="Weight bit width (default: 4).")
    parser.add_argument("--groupsize", type=int, default=128, help="Group size (default: 128).")
    parser.add_argument("--n_samples", type=int, default=512, help="Calibration sample count (default: 512).")
    parser.add_argument("--block_size", type=int, default=512, help="Max calibration sequence length (default: 512).")
    parser.add_argument(
        "--awq_version",
        type=str,
        default="GEMM",
        choices=["GEMM", "gemm"],
        help="AWQ kernel version field in quant_config.",
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True.")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    calib_texts = get_wikitext2_calib_texts(
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        block_size=args.block_size,
    )

    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path,
        safetensors=True,
        trust_remote_code=args.trust_remote_code,
    )

    quant_config = {
        "zero_point": True,
        "q_group_size": int(args.groupsize),
        "w_bit": int(args.qbit),
        "version": args.awq_version,
    }

    quantize_kwargs = _build_quantize_kwargs(
        model=model,
        tokenizer=tokenizer,
        quant_config=quant_config,
        calib_texts=calib_texts,
        max_calib_seq_len=int(args.block_size),
    )
    model.quantize(**quantize_kwargs)

    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"AWQ model saved to: {output_path}")


if __name__ == "__main__":
    main()

