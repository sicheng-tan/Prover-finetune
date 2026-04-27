# Lean + Mathlib Configuration Guide

This project already supports dynamic Lean and Mathlib version configuration.

The implementation is in:

- `src/prover_finetune/experiments/lean_checker.py`

## How It Works

At runtime, `LeanChecker.setup_project()` will:

1. Create a Lean runner project directory (`lean.project_dir`).
2. Write `lean-toolchain` from `lean.lean_version`.
3. Write `lakefile.lean` with Mathlib git dependency pinned by `lean.mathlib_ref`.
4. Run `lake update` to fetch dependencies for the selected versions.

Then each generated proof is checked by:

- `lake env lean Main.lean`

## Config Fields

In your experiment config (for example `configs/experiment.example.yaml`), use:

```yaml
lean:
  project_dir: .lean_runner
  lean_version: leanprover/lean4:v4.11.0
  mathlib_ref: v4.11.0
  timeout_sec: 30
  lake_exe: lake
```

### Field Meanings

- `project_dir`: working Lean project for verification files.
- `lean_version`: toolchain string written into `lean-toolchain`.
- `mathlib_ref`: branch/tag/commit used in `lakefile.lean`.
- `timeout_sec`: timeout for a single Lean check.
- `lake_exe`: `lake` executable name/path.

## Change Versions

Just change `lean.lean_version` and `lean.mathlib_ref` in your config and rerun:

```bash
python -m src.prover_finetune.experiments.run_experiment --config configs/experiment.example.yaml
```

## Compatibility Notes

- Lean and Mathlib versions should be compatible with each other.
- If `lake update` fails, first check version compatibility and local Lean toolchain availability.
- You can use a commit hash for `mathlib_ref` for stricter reproducibility.
