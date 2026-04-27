# Prover Finetune + Lean Eval Framework

This repository provides two configurable Python frameworks:

1. QLoRA finetuning for small LLMs.
2. Lean proving experiments on miniF2F with Lean checker validation.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or run one-shot bootstrap:

```bash
python scripts/bootstrap_env.py --experiment-config configs/experiment.example.yaml
```

### Finetune

```bash
python -m src.prover_finetune.finetune.train_qlora --config configs/finetune.example.yaml
```

### Experiment

```bash
python -m src.prover_finetune.experiments.run_experiment --config configs/experiment.example.yaml
```

### Lean/Mathlib Version Config

See `docs/lean-mathlib-config.md` for detailed setup and version switching instructions.

## Notes

- Data processing in finetuning is intentionally pluggable and driven by config fields.
- Lean and Mathlib versions are controlled by experiment config and applied dynamically.
- miniF2F can be loaded from local files (`jsonl` or `lean` files + split lists).
