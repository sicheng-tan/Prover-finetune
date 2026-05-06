import argparse
import inspect
from pathlib import Path

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

try:
    # autoawq>=0.2.x
    from awq import AutoAWQForCausalLM
except ImportError:
    # Compatibility fallback for variants that expose different casing.
    from awq import AutoAwqForCausalLM as AutoAWQForCausalLM

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
except ImportError:
    AutoGPTQForCausalLM = None
    BaseQuantizeConfig = None


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


def _to_gptq_examples(tokenizer, calib_texts: list[str], block_size: int) -> list[dict]:
    examples: list[dict] = []
    for text in calib_texts:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=block_size,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        if input_ids.numel() == 0:
            continue
        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )
    if not examples:
        raise RuntimeError("No GPTQ calibration examples were produced.")
    return examples


def _run_awq(args: argparse.Namespace, tokenizer, calib_texts: list[str], output_path: Path) -> None:
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
    try:
        model.quantize(**quantize_kwargs)
    except AssertionError as exc:
        raise RuntimeError(
            "AWQ quantization hit an internal assertion. "
            "Common causes: (1) input model is already quantized, "
            "(2) current AutoAWQ/torch/transformers combo is incompatible, "
            "(3) q_group_size is unsupported for this model/hardware. "
            "Try `--groupsize 64` (or `--groupsize -1` for no grouping), "
            "and ensure model_path points to full-precision weights."
        ) from exc

    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))


def _run_gptq(args: argparse.Namespace, tokenizer, calib_texts: list[str], output_path: Path) -> None:
    if AutoGPTQForCausalLM is None or BaseQuantizeConfig is None:
        raise ImportError(
            "auto-gptq is not installed in this environment. "
            "Install `auto-gptq` to use `--method gptq`."
        )

    quant_config = BaseQuantizeConfig(
        bits=int(args.qbit),
        group_size=int(args.groupsize),
        desc_act=bool(args.gptq_desc_act),
        damp_percent=float(args.gptq_damp_percent),
        sym=bool(args.gptq_sym),
        true_sequential=bool(args.gptq_true_sequential),
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_path,
        quantize_config=quant_config,
        trust_remote_code=args.trust_remote_code,
    )
    examples = _to_gptq_examples(tokenizer, calib_texts, int(args.block_size))
    model.quantize(examples)
    model.save_quantized(str(output_path), use_safetensors=True)
    tokenizer.save_pretrained(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="AWQ/GPTQ quantization with FLRQ-like defaults.")
    parser.add_argument("--model_path", type=str, required=True, help="HF model id or local path.")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for AWQ model.")
    parser.add_argument(
        "--method",
        type=str,
        default="awq",
        choices=["awq", "gptq"],
        help="Quantization backend.",
    )
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
    parser.add_argument(
        "--gptq_damp_percent",
        type=float,
        default=0.01,
        help="GPTQ damp percent (default: 0.01).",
    )
    parser.add_argument(
        "--gptq_sym",
        action="store_true",
        help="Use symmetric GPTQ quantization (default: asymmetric).",
    )
    parser.add_argument(
        "--gptq_desc_act",
        action="store_true",
        help="Enable GPTQ desc_act (default: off).",
    )
    parser.add_argument(
        "--gptq_true_sequential",
        action="store_true",
        default=True,
        help="Use GPTQ true_sequential (default: on).",
    )
    parser.add_argument(
        "--no_gptq_true_sequential",
        action="store_false",
        dest="gptq_true_sequential",
        help="Disable GPTQ true_sequential.",
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    # AWQ here expects a full-precision HF checkpoint as input.
    if getattr(cfg, "quantization_config", None) is not None:
        raise ValueError(
            "Input model already appears quantized (quantization_config exists). "
            "Please pass the original/merged full-precision model (e.g. deepseek-ai/DeepSeek-Prover-V2-7B), "
            "not a GPTQ/AWQ checkpoint."
        )

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

    if args.method == "awq":
        _run_awq(args, tokenizer, calib_texts, output_path)
    else:
        _run_gptq(args, tokenizer, calib_texts, output_path)
    print(f"{args.method.upper()} model saved to: {output_path}")


if __name__ == "__main__":
    main()

