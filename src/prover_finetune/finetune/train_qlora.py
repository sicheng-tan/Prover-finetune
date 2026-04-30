import argparse
import inspect
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from .config import FinetuneConfig
from .data import load_and_process_dataset


def _resolve_eval_strategy_key() -> str:
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in params:
        return "evaluation_strategy"
    if "eval_strategy" in params:
        return "eval_strategy"
    return "evaluation_strategy"


def build_sft_trainer(
    *,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    peft_config,
    args: TrainingArguments,
    dataset_text_field: str,
    max_seq_length: int,
) -> SFTTrainer:
    params = inspect.signature(SFTTrainer.__init__).parameters
    kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
        "args": args,
    }
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = dataset_text_field
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = max_seq_length
    return SFTTrainer(**kwargs)


def _to_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def build_model_and_tokenizer(model_cfg: dict):
    quant_cfg = None
    torch_dtype = None
    use_4bit = model_cfg.get("use_4bit", False)
    use_8bit = model_cfg.get("use_8bit", False)

    if use_4bit and use_8bit:
        raise ValueError("Only one quantization mode can be enabled: use_4bit or use_8bit.")

    if use_4bit:
        compute_dtype = _to_torch_dtype(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif use_8bit:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    else:
        # Keep full-precision weights in low-memory floating format when quantization is disabled.
        torch_dtype = _to_torch_dtype(model_cfg.get("torch_dtype", "bfloat16"))

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        quantization_config=quant_cfg,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_lora_config(model_cfg: dict) -> LoraConfig:
    lora_cfg = model_cfg.get("lora", {})
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get("target_modules"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to finetune yaml config")
    args = parser.parse_args()

    cfg = FinetuneConfig.load(args.config)
    model_cfg = cfg.section("model")
    data_cfg = cfg.section("data")
    train_cfg = cfg.section("training")

    model, tokenizer = build_model_and_tokenizer(model_cfg)
    train_ds, eval_ds = load_and_process_dataset(data_cfg, model_cfg=model_cfg)
    peft_config = build_lora_config(model_cfg)

    output_dir = Path(train_cfg.get("output_dir", "outputs/qlora"))
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_strategy_key = _resolve_eval_strategy_key()
    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": train_cfg.get("num_train_epochs", 1),
        "per_device_train_batch_size": train_cfg.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": train_cfg.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps", 1),
        "learning_rate": train_cfg.get("learning_rate", 2e-4),
        "logging_steps": train_cfg.get("logging_steps", 10),
        "save_steps": train_cfg.get("save_steps", 100),
        "eval_steps": train_cfg.get("eval_steps", 100),
        eval_strategy_key: "steps" if eval_ds is not None else "no",
        "warmup_ratio": train_cfg.get("warmup_ratio", 0.03),
        "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
        "bf16": train_cfg.get("bf16", True),
        "report_to": train_cfg.get("report_to", "none"),
        "seed": train_cfg.get("seed", 42),
    }
    training_args = TrainingArguments(**training_kwargs)

    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=data_cfg.get("max_seq_length", 1024),
        peft_config=peft_config,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))


if __name__ == "__main__":
    main()

