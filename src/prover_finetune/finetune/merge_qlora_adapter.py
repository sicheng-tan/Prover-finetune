import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import FinetuneConfig


def _to_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _resolve_paths(args: argparse.Namespace) -> tuple[str, str, Path, bool]:
    if args.config:
        cfg = FinetuneConfig.load(args.config)
        model_cfg = cfg.section("model")
        train_cfg = cfg.section("training")

        base_model_path = model_cfg["name_or_path"]
        trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
        adapter_path = args.adapter_path or str(
            Path(train_cfg.get("output_dir", "outputs/qlora")) / "adapter"
        )
        output_path = Path(args.output_path or (Path(train_cfg.get("output_dir", "outputs/qlora")) / "merged"))
        return base_model_path, adapter_path, output_path, trust_remote_code

    if not args.base_model_path:
        raise ValueError("`--base-model-path` is required when `--config` is not provided.")
    if not args.adapter_path:
        raise ValueError("`--adapter-path` is required when `--config` is not provided.")
    if not args.output_path:
        raise ValueError("`--output-path` is required when `--config` is not provided.")

    return args.base_model_path, args.adapter_path, Path(args.output_path), False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a QLoRA adapter into the base model and save merged weights."
    )
    parser.add_argument(
        "--config",
        help="Optional finetune yaml config. If provided, base model and default paths are inferred.",
    )
    parser.add_argument("--base-model-path", help="Base model name or local path.")
    parser.add_argument("--adapter-path", help="LoRA adapter directory path.")
    parser.add_argument("--output-path", help="Merged model output directory path.")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype used when loading the base model for merge.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        help="Save model using safetensors format.",
    )
    args = parser.parse_args()

    base_model_path, adapter_path, output_path, config_trust_remote_code = _resolve_paths(args)
    output_path.mkdir(parents=True, exist_ok=True)
    trust_remote_code = args.trust_remote_code or config_trust_remote_code

    torch_dtype = _to_torch_dtype(args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    peft_model = PeftModel.from_pretrained(model, adapter_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(str(output_path), safe_serialization=args.safe_serialization)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    tokenizer.save_pretrained(str(output_path))

    print(f"Merged model saved to: {output_path}")


if __name__ == "__main__":
    main()
