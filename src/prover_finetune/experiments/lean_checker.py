import json
import re
import subprocess
from pathlib import Path

import yaml

try:
    from lean_interact import AutoLeanServer, Command, LeanREPLConfig, LocalProject
except Exception:  # pragma: no cover - optional runtime dependency
    AutoLeanServer = None
    Command = None
    LeanREPLConfig = None
    LocalProject = None


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
        self.header_imports = merged_cfg.get("header_imports", ["Mathlib"])
        self.header_set_options = merged_cfg.get("header_set_options", ["maxHeartbeats 200000"])
        self.header_open_scopes = merged_cfg.get(
            "header_open_scopes", ["BigOperators", "Real", "Nat", "Topology", "Rat"]
        )
        self.cache_get_timeout_sec = int(merged_cfg.get("cache_get_timeout_sec", 300))
        self.use_lean_interact = bool(merged_cfg.get("use_lean_interact", True))
        self.memory_limit_mb = merged_cfg.get("memory_limit_mb")
        self.lean_interact_verbose = bool(merged_cfg.get("lean_interact_verbose", False))
        self.strict_project_setup = bool(merged_cfg.get("strict_project_setup", True))
        self._lean_server = None

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

    def _mathlib_smoke_test_passes(self) -> bool:
        smoke_file = self.project_dir / "Smoke.lean"
        smoke_content = (
            "import Mathlib\n\n"
            "example : (1 : Nat) + 1 = 2 := by\n"
            "  norm_num\n"
        )
        smoke_file.write_text(smoke_content, encoding="utf-8")
        try:
            proc = subprocess.run(
                [self.lake_exe, "env", "lean", str(smoke_file.name)],
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout_sec,
            )
            return proc.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    def setup_project(self) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)

        if self.strict_project_setup and self._version_is_already_configured():
            # Config matches local setup; skip rebuild workflow.
            return

        # In non-strict mode, keep a fast path for already usable environments.
        if (not self.strict_project_setup) and self._mathlib_smoke_test_passes():
            self._write_version_stamp()
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
        # Recreate lean-interact server so it binds to the refreshed project env.
        self._lean_server = None

    def _build_check_content(self, theorem_block: str, proof: str) -> str:
        imports_block = "\n".join(f"import {name}" for name in self.header_imports)
        options_block = "\n".join(f"set_option {opt}" for opt in self.header_set_options)
        open_block = (
            f"open {' '.join(self.header_open_scopes)}" if self.header_open_scopes else ""
        )
        return (
            f"{imports_block}\n\n"
            f"{options_block}\n\n"
            f"{open_block}\n\n"
            f"{theorem_block}\n\n"
            f"{proof}\n"
        )

    def _init_lean_interact_server(self) -> bool:
        if not self.use_lean_interact:
            return False
        if self._lean_server is not None:
            return True
        if not all([AutoLeanServer, Command, LeanREPLConfig, LocalProject]):
            return False
        try:
            project = LocalProject(directory=self.project_dir.resolve(), auto_build=False)
            config = LeanREPLConfig(
                project=project,
                verbose=self.lean_interact_verbose,
                memory_hard_limit_mb=self.memory_limit_mb,
            )
            self._lean_server = AutoLeanServer(config)
            return True
        except Exception:
            self._lean_server = None
            return False

    def _check_with_lean_interact(self, content: str) -> tuple[bool, str]:
        if not self._init_lean_interact_server():
            return False, "lean-interact unavailable; fallback to subprocess."
        try:
            response = self._lean_server.run(
                Command(cmd=content),
                timeout=self.timeout_sec if self.timeout_sec > 0 else None,
            )
            errors = [msg for msg in response.messages if msg.severity == "error"]
            warnings = [msg for msg in response.messages if msg.severity == "warning"]
            error_lines = [
                f"{err.data} (at line {getattr(err.start_pos, 'line', '?')})" for err in errors
            ]
            warning_lines = [
                f"{w.data} (at line {getattr(w.start_pos, 'line', '?')})" for w in warnings
            ]
            sorries = []
            if getattr(response, "sorries", None):
                for s in response.sorries:
                    sorries.append(
                        f"Incomplete proof at line {getattr(s.start_pos, 'line', '?')}: {s.goal[:120]}"
                    )
            has_sorry_warning = any(
                re.search(r"declaration uses 'sorry'", line, flags=re.IGNORECASE)
                for line in warning_lines
            )
            ok = not error_lines and not sorries and not has_sorry_warning
            output_lines = []
            if error_lines:
                output_lines.append("Errors:")
                output_lines.extend(error_lines)
            if warning_lines:
                output_lines.append("Warnings:")
                output_lines.extend(warning_lines)
            if sorries:
                output_lines.append("Sorries:")
                output_lines.extend(sorries)
            if not output_lines:
                output_lines.append("Verification successful")
            return ok, "\n".join(output_lines)
        except Exception as exc:
            return False, f"lean-interact verification error: {exc}"

    def check_proof(self, theorem_block: str, proof: str) -> tuple[bool, str]:
        test_file = self.project_dir / "Main.lean"
        content = self._build_check_content(theorem_block, proof)
        test_file.write_text(content, encoding="utf-8")

        if self.use_lean_interact:
            interact_ok, interact_log = self._check_with_lean_interact(content)
            # If lean-interact runs, trust its result/log and skip subprocess.
            if "fallback to subprocess" not in interact_log.lower():
                return interact_ok, interact_log

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
            # Treat Lean's "declaration uses 'sorry'" warning as failure.
            # This avoids false positives from naive "sorry" substring matching.
            has_sorry_warning = bool(
                re.search(
                    r"warning:.*declaration uses 'sorry'",
                    proc.stdout or "",
                    flags=re.IGNORECASE,
                )
            )
            if has_sorry_warning:
                ok = False
            return ok, proc.stdout
        except subprocess.TimeoutExpired as exc:
            timeout_log = (
                f"Lean check timeout after {self.timeout_sec}s.\n"
                f"Command: {' '.join(exc.cmd) if exc.cmd else f'{self.lake_exe} env lean Main.lean'}\n"
            )
            if exc.stdout:
                timeout_log += f"\nPartial output:\n{exc.stdout}"
            return False, timeout_log

