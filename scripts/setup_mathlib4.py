#!/usr/bin/env python3
"""
Setup mathlib4 using a lean project config file.

Example:
  python scripts/setup_mathlib4.py --config configs/lean_project.example.yaml
"""

import argparse
import subprocess
from pathlib import Path

import yaml

def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def git_clone_if_missing(repo_url: str, target_dir: Path) -> None:
    if target_dir.exists():
        if not (target_dir / ".git").exists():
            raise ValueError(
                f"Target directory exists but is not a git repo: {target_dir}. "
                "Please remove it or choose another target directory."
            )
        print(f"[skip] target directory already exists, skip download: {target_dir}")
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", repo_url, str(target_dir)])


def checkout_ref(target_dir: Path, ref: str) -> None:
    run(["git", "checkout", ref], cwd=target_dir)


def build_mathlib(target_dir: Path, skip_cache_get: bool, skip_build: bool) -> None:
    if not skip_cache_get:
        run(["lake", "exe", "cache", "get"], cwd=target_dir)
    if not skip_build:
        run(["lake", "build"], cwd=target_dir)


def normalize_lean_version(version: str) -> str:
    v = version.strip()
    if v.startswith("leanprover/lean4:"):
        return v
    if v.startswith("v"):
        return f"leanprover/lean4:{v}"
    return f"leanprover/lean4:v{v}"


def set_local_lean_default(version: str) -> None:
    toolchain = normalize_lean_version(version)
    run(["elan", "default", toolchain])


def _resolve_target_dir(cfg: dict, config_path: Path) -> tuple[Path, str, str]:
    setup_cfg = cfg.get("mathlib_setup", {})
    if not isinstance(setup_cfg, dict):
        raise ValueError("mathlib_setup must be a mapping in config file.")

    repo_url = str(setup_cfg.get("repo_url", "https://github.com/leanprover-community/mathlib4.git"))
    ref = str(setup_cfg.get("ref", cfg.get("lean_version", "v4.27.0")))
    clone_root = Path(str(setup_cfg.get("clone_root", "external")))
    dir_template = str(setup_cfg.get("dir_template", "mathlib4-{ref}"))

    target_dir_name = dir_template.format(ref=ref)
    target_dir = (config_path.parent.parent / clone_root / target_dir_name).resolve()
    return target_dir, repo_url, ref


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must be a YAML mapping.")
    return data


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup mathlib4 from lean project config.")
    parser.add_argument(
        "--config",
        default="configs/lean_project.example.yaml",
        help="Path to lean project yaml config.",
    )
    parser.add_argument(
        "--skip-cache-get",
        action="store_true",
        help="Skip `lake exe cache get`.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip `lake build`.",
    )
    parser.add_argument(
        "--sync-mathlib-path",
        action="store_true",
        help="Write resolved target directory back into config as mathlib_path.",
    )
    parser.add_argument(
        "--set-elan-default",
        action="store_true",
        help="Also run `elan default <lean_version>` from config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)
    target_dir, repo_url, ref = _resolve_target_dir(cfg, config_path)

    git_clone_if_missing(repo_url, target_dir)
    checkout_ref(target_dir, ref)
    build_mathlib(target_dir, skip_cache_get=args.skip_cache_get, skip_build=args.skip_build)
    if args.set_elan_default:
        set_local_lean_default(str(cfg.get("lean_version", ref)))

    if args.sync_mathlib_path:
        rel = target_dir.relative_to(config_path.parent.parent)
        cfg["mathlib_path"] = str(rel)
        _write_yaml(config_path, cfg)

    print(f"[ok] mathlib4 ready at: {target_dir}")
    print(f"[ok] checked out ref: {ref}")
    if args.set_elan_default:
        print(f"[ok] set elan default to: {normalize_lean_version(str(cfg.get('lean_version', ref)))}")
    if args.sync_mathlib_path:
        print(f"[ok] updated mathlib_path in config: {cfg['mathlib_path']}")
    else:
        print("[next] set mathlib_path in config to this directory (or use --sync-mathlib-path).")


if __name__ == "__main__":
    main()

