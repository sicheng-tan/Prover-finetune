import json
import subprocess
from pathlib import Path

import yaml


class LeanChecker:
    def __init__(self, lean_cfg: dict):
        merged_cfg = dict(lean_cfg)
        project_cfg_path = merged_cfg.get("project_config_path")
        if project_cfg_path:
            with open(project_cfg_path, "r", encoding="utf-8") as f:
                file_cfg = yaml.safe_load(f) or {}
            if not isinstance(file_cfg, dict):
                raise ValueError("Lean project config must be a YAML mapping.")
            # File config is the base; explicit lean section values can override it.
            merged_cfg = {**file_cfg, **merged_cfg}

        self.project_dir = Path(merged_cfg.get("project_dir", ".lean_runner"))
        self.lean_version = merged_cfg["lean_version"]
        self.mathlib_ref = merged_cfg["mathlib_ref"]
        self.package_name = merged_cfg.get("package_name", "runner")
        self.extra_dependencies = merged_cfg.get("extra_dependencies", [])
        self.timeout_sec = int(merged_cfg.get("timeout_sec", 30))
        self.lake_exe = merged_cfg.get("lake_exe", "lake")
        self.header_imports = merged_cfg.get("header_imports", ["Mathlib", "Aesop"])
        self.header_set_options = merged_cfg.get("header_set_options", ["maxHeartbeats 0"])
        self.header_open_scopes = merged_cfg.get(
            "header_open_scopes", ["BigOperators", "Real", "Nat", "Topology", "Rat"]
        )
        self.cache_get_timeout_sec = int(merged_cfg.get("cache_get_timeout_sec", 300))

    def _version_stamp_path(self) -> Path:
        return self.project_dir / ".version_stamp"

    def _render_lakefile(self) -> str:
        base = (
            "import Lake\n"
            "open Lake DSL\n\n"
            f"package {self.package_name}\n\n"
            "require mathlib from git\n"
            '  "https://github.com/leanprover-community/mathlib4.git" @ '
            f'"{self.mathlib_ref}"\n'
        )
        extra_lines = []
        for dep in self.extra_dependencies:
            if not isinstance(dep, dict):
                raise ValueError("Each item in extra_dependencies must be a mapping.")
            name = dep.get("name")
            git = dep.get("git")
            ref = dep.get("ref")
            if not name or not git or not ref:
                raise ValueError("Each extra dependency needs name, git, and ref.")
            extra_lines.append(f'\nrequire {name} from git\n  "{git}" @ "{ref}"\n')
        return base + "".join(extra_lines)

    def _target_stamp(self) -> str:
        stamp = {
            "lean_version": self.lean_version,
            "lakefile": self._render_lakefile(),
            "lake_exe": self.lake_exe,
        }
        return json.dumps(stamp, ensure_ascii=True, sort_keys=True, indent=2)

    def _version_is_already_configured(self) -> bool:
        stamp_path = self._version_stamp_path()
        if not stamp_path.exists():
            return False
        return stamp_path.read_text(encoding="utf-8") == self._target_stamp()

    def _write_version_stamp(self) -> None:
        self._version_stamp_path().write_text(self._target_stamp(), encoding="utf-8")

    def setup_project(self) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)

        if self._version_is_already_configured():
            # Config matches local setup; skip rebuild workflow.
            return

        toolchain = self.project_dir / "lean-toolchain"
        toolchain.write_text(f"{self.lean_version}\n", encoding="utf-8")

        lakefile = self.project_dir / "lakefile.lean"
        lakefile.write_text(self._render_lakefile(), encoding="utf-8")

        (self.project_dir / "Main.lean").write_text("-- placeholder\n", encoding="utf-8")
        subprocess.run([self.lake_exe, "update"], cwd=self.project_dir, check=True)
        # Pull prebuilt cache after dependency update; faster than full local rebuild.
        subprocess.run(
            [self.lake_exe, "exe", "cache", "get"],
            cwd=self.project_dir,
            check=True,
            timeout=self.cache_get_timeout_sec,
        )
        self._write_version_stamp()

    def check_proof(self, theorem_block: str, proof: str) -> tuple[bool, str]:
        test_file = self.project_dir / "Main.lean"
        imports_block = "\n".join(f"import {name}" for name in self.header_imports)
        options_block = "\n".join(f"set_option {opt}" for opt in self.header_set_options)
        open_block = (
            f"open {' '.join(self.header_open_scopes)}" if self.header_open_scopes else ""
        )
        content = (
            f"{imports_block}\n\n"
            f"{options_block}\n\n"
            f"{open_block}\n\n"
            f"{theorem_block}\n\n"
            f"{proof}\n"
        )
        test_file.write_text(content, encoding="utf-8")

        try:
            proc = subprocess.run(
                [self.lake_exe, "env", "lean", "Main.lean"],
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout_sec,
            )
            ok = proc.returncode == 0
            return ok, proc.stdout
        except subprocess.TimeoutExpired as exc:
            timeout_log = (
                f"Lean check timeout after {self.timeout_sec}s.\n"
                f"Command: {' '.join(exc.cmd) if exc.cmd else f'{self.lake_exe} env lean Main.lean'}\n"
            )
            if exc.stdout:
                timeout_log += f"\nPartial output:\n{exc.stdout}"
            return False, timeout_log

