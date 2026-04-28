import re
from pathlib import Path

import yaml

try:
    from lean_interact import AutoLeanServer, Command, LeanREPLConfig, LeanServer, LocalProject
except Exception:  # pragma: no cover - optional runtime dependency
    AutoLeanServer = None
    Command = None
    LeanREPLConfig = None
    LeanServer = None
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
            merged_cfg = {**file_cfg, **merged_cfg}

        self.mathlib_path = Path(merged_cfg.get("mathlib_path", "mathlib4"))
        self.lean_version = str(merged_cfg.get("lean_version", "v4.27.0"))
        self.timeout_sec = int(merged_cfg.get("timeout_sec", 30))
        self.header_imports = merged_cfg.get("header_imports", ["Mathlib"])
        self.header_set_options = merged_cfg.get("header_set_options", ["maxHeartbeats 200000"])
        self.header_open_scopes = merged_cfg.get(
            "header_open_scopes", ["BigOperators", "Real", "Nat", "Topology", "Rat"]
        )
        self.use_auto_server = bool(merged_cfg.get("use_auto_server", True))
        self.use_lean_interact = bool(merged_cfg.get("use_lean_interact", True))
        self.memory_limit_mb = merged_cfg.get("memory_limit_mb")
        self.lean_interact_verbose = bool(merged_cfg.get("lean_interact_verbose", False))
        self._lean_server = None

    def setup_project(self) -> None:
        if not self.mathlib_path.exists():
            raise FileNotFoundError(
                f"mathlib_path not found: {self.mathlib_path}. "
                "Please initialize submodule and run lake manually."
            )
        toolchain_file = self.mathlib_path / "lean-toolchain"
        if toolchain_file.exists():
            self.lean_version = toolchain_file.read_text(encoding="utf-8").strip().replace(
                "leanprover/lean4:", ""
            )

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
        if not all([Command, LeanREPLConfig, LocalProject]):
            return False
        try:
            project = LocalProject(directory=self.mathlib_path.resolve(), auto_build=False)
            config = LeanREPLConfig(
                project=project,
                verbose=self.lean_interact_verbose,
                memory_hard_limit_mb=self.memory_limit_mb,
            )
            if self.use_auto_server and AutoLeanServer is not None:
                self._lean_server = AutoLeanServer(config)
            elif LeanServer is not None:
                self._lean_server = LeanServer(config)
            else:
                self._lean_server = None
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
        content = self._build_check_content(theorem_block, proof)
        return self._check_with_lean_interact(content)

