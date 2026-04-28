import argparse
import copy
import concurrent.futures
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from .config import ExperimentConfig
from .lean_checker import LeanChecker
from .minif2f import load_minif2f
from .prover import LLMGenerationTimeoutError, build_prover_generator


def _extract_theorem_block(problem: dict) -> str:
    # Expect miniF2F row to include Lean theorem statement.
    if "definition" in problem:
        return str(problem["definition"])
    if "statement" in problem:
        return str(problem["statement"])
    if "theorem" in problem:
        return str(problem["theorem"])
    raise ValueError("miniF2F sample must contain 'definition', 'statement', or 'theorem' field.")


def _setup_logger(verbose_logging: bool) -> logging.Logger:
    logger = logging.getLogger("prover_finetune.experiments")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Keep default stream settings if runtime does not support reconfigure.
            pass
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose_logging else logging.WARNING)
    logger.propagate = False
    return logger


def _attach_file_logger(logger: logging.Logger, output_dir: Path) -> None:
    log_path = output_dir / "experiment.log"
    target = str(log_path.resolve())
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == target:
            return
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)


def _extract_lean_code_from_generation(generation: str) -> tuple[str, str]:
    def _extract_last_fence(text: str, language: str) -> str | None:
        pattern = rf"```{language}\b[\s\S]*?```"
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if not matches:
            return None
        block = matches[-1].group(0)
        block = re.sub(rf"^```{language}\b", "", block, flags=re.IGNORECASE).strip()
        block = re.sub(r"```$", "", block).strip()
        return block.strip()

    lean4_code = _extract_last_fence(generation, "lean4")
    if lean4_code:
        return lean4_code, "lean4_fence"

    lean_code = _extract_last_fence(generation, "lean")
    if lean_code:
        return lean_code, "lean_fence"

    return generation.strip(), "raw_fallback"


def _discover_gpu_ids(exp_cfg: dict, model_cfg: dict, logger: logging.Logger) -> list[int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run GPU-parallel experiment.")
    visible = torch.cuda.device_count()
    if visible <= 0:
        raise RuntimeError("No visible CUDA devices found.")

    configured = exp_cfg.get("gpu_ids")
    if configured is None:
        raise ValueError(
            "experiment.gpu_ids must be explicitly configured, e.g. gpu_ids: [0, 1]."
        )
    if isinstance(configured, str):
        gpu_ids = [int(x.strip()) for x in configured.split(",") if x.strip()]
    elif isinstance(configured, list):
        gpu_ids = [int(x) for x in configured]
    else:
        raise ValueError("experiment.gpu_ids must be a list[int] or comma-separated string.")
    if not gpu_ids:
        raise ValueError("experiment.gpu_ids cannot be empty.")
    invalid = [gid for gid in gpu_ids if gid < 0 or gid >= visible]
    if invalid:
        raise ValueError(
            f"experiment.gpu_ids has invalid entries {invalid}; visible device count is {visible}."
        )
    logger.info("Using %d parallel GPU workers on configured devices: %s", len(gpu_ids), gpu_ids)
    return gpu_ids


def _append(lines: list[str], msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"[{ts}] {msg}")


def _process_one_problem(
    row_idx: int,
    total_rows: int,
    row: dict,
    pass_k: int,
    prover,
    checker,
) -> tuple[dict, int, list[str]]:
    logs: list[str] = []
    theorem_block = _extract_theorem_block(row)
    sample_id = row.get("id", row.get("name", f"sample_{row_idx}"))
    _append(logs, "=" * 80)
    _append(logs, f"SAMPLE {row_idx}/{total_rows} - id={sample_id}")
    _append(logs, "-" * 80)
    _append(logs, "THEOREM BLOCK:")
    _append(logs, theorem_block)
    _append(logs, "=" * 80)

    ok = False
    lean_log = ""
    first_prediction = ""
    first_prediction_lean_code = ""
    first_extraction_mode = "none"
    candidates = []

    for idx in range(1, pass_k + 1):
        retries_used = idx - 1
        retries_remaining = max(0, pass_k - idx)
        _append(logs, "=" * 80)
        _append(logs, f"ATTEMPT {idx}/{pass_k} - GENERATION")
        _append(logs, "-" * 80)
        _append(
            logs,
            f"Sample id={sample_id} | retries_used={retries_used} | retries_remaining={retries_remaining}",
        )
        try:
            pred_proof = prover.generate_proof(theorem_block)
        except LLMGenerationTimeoutError as exc:
            timeout_log = str(exc)
            _append(logs, f"LLM generation timeout: {timeout_log}")
            if idx == 1:
                first_extraction_mode = "llm_timeout"
                lean_log = timeout_log
            candidates.append(
                {
                    "sample_idx": idx,
                    "ok": False,
                    "prediction": "",
                    "prediction_lean_code": "",
                    "extraction_mode": "llm_timeout",
                    "lean_output": timeout_log,
                }
            )
            _append(logs, "=" * 80)
            continue

        if idx == 1:
            first_prediction = pred_proof
        extracted_proof, extraction_mode = _extract_lean_code_from_generation(pred_proof)
        if idx == 1:
            first_prediction_lean_code = extracted_proof
            first_extraction_mode = extraction_mode

        _append(logs, "RAW MODEL OUTPUT:")
        _append(logs, pred_proof)
        _append(logs, "-" * 80)
        _append(logs, f"EXTRACTED LEAN CODE (mode={extraction_mode}):")
        _append(logs, extracted_proof)
        _append(logs, "=" * 80)
        _append(logs, f"ATTEMPT {idx}/{pass_k} - LEAN VERIFICATION")
        _append(logs, "-" * 80)
        cur_ok, cur_log = checker.check_proof(theorem_block, extracted_proof)
        _append(logs, "LEAN OUTPUT:")
        _append(logs, cur_log)
        _append(logs, "-" * 80)
        _append(logs, f"LEAN CHECK RESULT: ok={cur_ok}")
        _append(logs, "=" * 80)

        candidates.append(
            {
                "sample_idx": idx,
                "ok": cur_ok,
                "prediction": pred_proof,
                "prediction_lean_code": extracted_proof,
                "extraction_mode": extraction_mode,
                "lean_output": cur_log,
            }
        )
        if cur_ok:
            ok = True
            lean_log = cur_log
            _append(
                logs,
                f"SUCCESS - sample id={sample_id} solved at attempt {idx}/{pass_k} (retries_used={retries_used}).",
            )
            _append(logs, "=" * 80)
            break
        if idx == 1:
            lean_log = cur_log

    if not ok:
        _append(logs, f"FAILED - sample id={sample_id} after {pass_k} attempts.")
        _append(logs, "=" * 80)

    result = {
        "row_idx": row_idx,
        "id": sample_id,
        "ok": ok,
        "theorem": theorem_block,
        "comment": row.get("comment"),
        "prediction": first_prediction,
        "prediction_lean_code": first_prediction_lean_code,
        "extraction_mode": first_extraction_mode,
        "lean_output": lean_log,
        "pass_k": pass_k,
        "candidates": candidates,
    }
    return result, int(ok), logs


def _run_worker(
    worker_idx: int,
    gpu_id: int,
    shard: list[tuple[int, dict]],
    total_rows: int,
    pass_k: int,
    model_cfg: dict,
    lean_cfg: dict,
    problem_log_dir: Path,
) -> list[tuple[int, dict, int]]:
    worker_model_cfg = copy.deepcopy(model_cfg)
    worker_model_cfg["device_map"] = f"cuda:{gpu_id}"
    prover = build_prover_generator(worker_model_cfg)

    worker_lean_cfg = copy.deepcopy(lean_cfg)
    base_project_dir = worker_lean_cfg.get("project_dir", ".lean_runner")
    worker_lean_cfg["project_dir"] = f"{base_project_dir}_worker_{worker_idx}"
    checker = LeanChecker(worker_lean_cfg)
    checker.setup_project()

    out: list[tuple[int, dict, int]] = []
    for row_idx, row in shard:
        result, ok_int, logs = _process_one_problem(row_idx, total_rows, row, pass_k, prover, checker)
        log_name_raw = row.get("name") or row.get("id") or f"sample_{row_idx}"
        safe_name = str(log_name_raw).replace("/", "_")
        (problem_log_dir / f"{safe_name}.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
        out.append((row_idx, result, ok_int))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment yaml config")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    exp_cfg = cfg.section("experiment")
    model_cfg = cfg.section("model")
    minif2f_cfg = cfg.section("minif2f")
    lean_cfg = cfg.section("lean")
    verbose_logging = bool(exp_cfg.get("verbose_logging", True))
    logger = _setup_logger(verbose_logging)

    split = exp_cfg.get("split", "valid")
    max_samples = exp_cfg.get("max_samples")
    output_dir = Path(exp_cfg.get("output_dir", "outputs/experiments/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _attach_file_logger(logger, output_dir)

    dataset = load_minif2f(minif2f_cfg, split=split, max_samples=max_samples)
    pass_k = int(exp_cfg.get("pass_k", 1))
    total_rows = len(dataset)
    if total_rows == 0:
        summary = {"total": 0, "pass": 0, f"pass@{pass_k}": 0.0}
        (output_dir / "results.json").write_text(
            json.dumps({"summary": summary, "results": []}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False))
        return

    problem_log_dir = output_dir / "problem_logs"
    problem_log_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = _discover_gpu_ids(exp_cfg, model_cfg, logger)
    worker_count = min(len(gpu_ids), total_rows)
    gpu_ids = gpu_ids[:worker_count]
    indexed_rows = list(enumerate(dataset, start=1))
    shards: list[list[tuple[int, dict]]] = [[] for _ in range(worker_count)]
    for i, item in enumerate(indexed_rows):
        shards[i % worker_count].append(item)

    packed_results: list[tuple[int, dict, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _run_worker,
                worker_idx=i,
                gpu_id=gpu_ids[i],
                shard=shards[i],
                total_rows=total_rows,
                pass_k=pass_k,
                model_cfg=model_cfg,
                lean_cfg=lean_cfg,
                problem_log_dir=problem_log_dir,
            )
            for i in range(worker_count)
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="workers"):
            packed_results.extend(future.result())

    packed_results.sort(key=lambda x: x[0])
    results = [item[1] for item in packed_results]
    num_pass = sum(item[2] for item in packed_results)

    total = len(results)
    pass_at_k = (num_pass / total) if total > 0 else 0.0
    summary = {"total": total, "pass": num_pass, f"pass@{pass_k}": pass_at_k}
    logger.info("Experiment finished. Summary: %s", json.dumps(summary, ensure_ascii=False))

    (output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

