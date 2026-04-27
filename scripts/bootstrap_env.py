#!/usr/bin/env python3
"""
One-shot environment bootstrap for this project.

What it does:
1) Create/refresh Python virtualenv and install dependencies.
2) Initialize Lean/Mathlib environment from experiment config.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def bootstrap_python_env(
    repo_root: Path,
    python_bin: str,
    venv_path: Path,
    requirements_path: Path,
    install_editable: bool,
) -> Path:
    if not venv_path.exists():
        run([python_bin, "-m", "venv", str(venv_path)], cwd=repo_root)

    pip_bin = venv_path / "bin" / "pip"
    run([str(pip_bin), "install", "--upgrade", "pip"], cwd=repo_root)
    run([str(pip_bin), "install", "-r", str(requirements_path)], cwd=repo_root)
    if install_editable:
        run([str(pip_bin), "install", "-e", "."], cwd=repo_root)
    return pip_bin


def bootstrap_lean_env(repo_root: Path, experiment_config_path: Path) -> None:
    # Ensure local package import works when running this script directly.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.prover_finetune.experiments.config import ExperimentConfig
    from src.prover_finetune.experiments.lean_checker import LeanChecker

    cfg = ExperimentConfig.load(experiment_config_path)
    lean_cfg = cfg.section("lean")
    checker = LeanChecker(lean_cfg)
    checker.setup_project()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Python + Lean/Mathlib environment.")
    parser.add_argument("--repo-root", default=".", help="Project root path.")
    parser.add_argument("--python-bin", default="python3", help="Python binary used to create venv.")
    parser.add_argument("--venv-path", default=".venv", help="Virtualenv path.")
    parser.add_argument("--requirements", default="requirements.txt", help="Requirements file path.")
    parser.add_argument(
        "--experiment-config",
        default="configs/experiment.example.yaml",
        help="Experiment config used for Lean/Mathlib setup.",
    )
    parser.add_argument("--skip-python-deps", action="store_true", help="Skip Python venv + pip install.")
    parser.add_argument("--skip-lean", action="store_true", help="Skip Lean/Mathlib setup.")
    parser.add_argument("--install-editable", action="store_true", help="Also run `pip install -e .`.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    venv_path = (repo_root / args.venv_path).resolve()
    requirements_path = (repo_root / args.requirements).resolve()
    experiment_config_path = (repo_root / args.experiment_config).resolve()

    if not args.skip_python_deps:
        bootstrap_python_env(
            repo_root=repo_root,
            python_bin=args.python_bin,
            venv_path=venv_path,
            requirements_path=requirements_path,
            install_editable=args.install_editable,
        )
        print(f"[ok] Python deps ready in: {venv_path}")

    if not args.skip_lean:
        bootstrap_lean_env(repo_root=repo_root, experiment_config_path=experiment_config_path)
        print(f"[ok] Lean/Mathlib ready from config: {experiment_config_path}")

    print("[done] Environment bootstrap complete.")


if __name__ == "__main__":
    main()

