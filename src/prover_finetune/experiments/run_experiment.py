import argparse
import copy
import concurrent.futures
import json
import logging
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from .config import ExperimentConfig
from .lean_checker import LeanChecker
from .minif2f import load_minif2f
from .prover import LLMGenerationTimeoutError, build_prover_generator

_LEAN_CHECKER_WORKER = None


def _resolve_tensor_parallel_size(model_cfg: dict, default: int = 1) -> int:
    value = model_cfg.get("tensor_parallel_size")
    if value is None:
        return max(1, int(default))
    return max(1, int(value))


def _resolve_lean_parallel_workers(model_cfg: dict, lean_cfg: dict) -> int:
    configured = lean_cfg.get("parallel_workers")
    if configured is None:
        configured = model_cfg.get("lean_parallel_workers")
    if configured is None:
        batch_size = int(model_cfg.get("vllm_batch_size", 16))
        return max(1, min(batch_size, os.cpu_count() or 1))
    return max(1, int(configured))


def _init_parallel_lean_checker(lean_cfg: dict) -> None:
    global _LEAN_CHECKER_WORKER
    worker_cfg = copy.deepcopy(lean_cfg)
    base_project_dir = worker_cfg.get("project_dir", ".lean_runner")
    worker_cfg["project_dir"] = f"{base_project_dir}_parallel_{os.getpid()}"
    checker = LeanChecker(worker_cfg)
    checker.setup_project()
    _LEAN_CHECKER_WORKER = checker


def _parallel_lean_check_task(payload: tuple[str, str]) -> tuple[bool, str]:
    theorem_block, extracted_proof = payload
    if _LEAN_CHECKER_WORKER is None:
        raise RuntimeError("Parallel Lean checker worker is not initialized.")
    return _LEAN_CHECKER_WORKER.check_proof(theorem_block, extracted_proof)


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


def _safe_config_for_log(value):
    if isinstance(value, dict):
        redacted = {}
        for k, v in value.items():
            key_lower = str(k).lower()
            if any(token in key_lower for token in ("key", "token", "secret", "password")):
                redacted[k] = "***REDACTED***"
            else:
                redacted[k] = _safe_config_for_log(v)
        return redacted
    if isinstance(value, list):
        return [_safe_config_for_log(v) for v in value]
    return value


def _log_run_configuration(
    logger: logging.Logger, exp_cfg: dict, model_cfg: dict, minif2f_cfg: dict, lean_cfg: dict
) -> None:
    payload = {
        "experiment": _safe_config_for_log(exp_cfg),
        "model": _safe_config_for_log(model_cfg),
        "minif2f": _safe_config_for_log(minif2f_cfg),
        "lean": _safe_config_for_log(lean_cfg),
    }
    logger.info("Run configuration:\n%s", json.dumps(payload, ensure_ascii=False, indent=2))


def _log_stage(logger: logging.Logger, stage: str, **kwargs) -> None:
    if kwargs:
        detail = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info("[stage] %s | %s", stage, detail)
    else:
        logger.info("[stage] %s", stage)


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
    total_generation_ms = 0.0
    total_verify_ms = 0.0

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
        gen_start = time.perf_counter()
        try:
            pred_proof = prover.generate_proof(theorem_block)
        except LLMGenerationTimeoutError as exc:
            gen_ms = (time.perf_counter() - gen_start) * 1000
            total_generation_ms += gen_ms
            timeout_log = str(exc)
            _append(
                logs,
                f"LLM generation timeout: {timeout_log} | gen_ms={gen_ms:.1f} | gen_ms(total)={total_generation_ms:.1f}",
            )
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
                    "generation_ms": gen_ms,
                    "generation_ms_total": total_generation_ms,
                    "verify_ms": 0.0,
                    "verify_ms_total": total_verify_ms,
                }
            )
            _append(logs, "=" * 80)
            continue
        gen_ms = (time.perf_counter() - gen_start) * 1000
        total_generation_ms += gen_ms

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
        verify_start = time.perf_counter()
        cur_ok, cur_log = checker.check_proof(theorem_block, extracted_proof)
        verify_ms = (time.perf_counter() - verify_start) * 1000
        total_verify_ms += verify_ms
        _append(logs, "LEAN OUTPUT:")
        _append(logs, cur_log)
        _append(logs, "-" * 80)
        _append(
            logs,
            (
                f"LEAN CHECK RESULT: ok={cur_ok} | gen_ms={gen_ms:.1f} | verify_ms={verify_ms:.1f} "
                f"| gen_ms(total)={total_generation_ms:.1f} | verify_ms(total)={total_verify_ms:.1f}"
            ),
        )
        _append(logs, "=" * 80)

        candidates.append(
            {
                "sample_idx": idx,
                "ok": cur_ok,
                "prediction": pred_proof,
                "prediction_lean_code": extracted_proof,
                "extraction_mode": extraction_mode,
                "lean_output": cur_log,
                "generation_ms": gen_ms,
                "generation_ms_total": total_generation_ms,
                "verify_ms": verify_ms,
                "verify_ms_total": total_verify_ms,
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

    attempts_used = len(candidates)
    _append(
        logs,
        (
            f"SAMPLE SUMMARY: attempts={attempts_used}/{pass_k} | "
            f"gen_ms(total)={total_generation_ms:.1f} | verify_ms(total)={total_verify_ms:.1f}"
        ),
    )

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
        "attempts_used": attempts_used,
        "generation_ms_total": total_generation_ms,
        "verify_ms_total": total_verify_ms,
        "pass_k": pass_k,
        "candidates": candidates,
    }
    return result, int(ok), logs


def _run_vllm_batch_mode(
    dataset: list[dict],
    pass_k: int,
    model_cfg: dict,
    lean_cfg: dict,
    problem_log_dir: Path,
    logger: logging.Logger,
    prover_override=None,
    verify_fn=None,
) -> list[tuple[int, dict, int]]:
    prover = prover_override or build_prover_generator(model_cfg)
    batch_size = max(1, int(model_cfg.get("vllm_batch_size", 16)))
    lean_parallel_workers = _resolve_lean_parallel_workers(model_cfg, lean_cfg)
    logger.info(
        "Lean verification parallel workers=%d (batch_size=%d).",
        lean_parallel_workers,
        batch_size,
    )
    _log_stage(
        logger,
        "vllm_batch_mode_init",
        batch_size=batch_size,
        lean_parallel_workers=lean_parallel_workers,
        dataset_size=len(dataset),
        pass_k=pass_k,
    )

    checker = None
    verify_executor = None
    verify_callable = verify_fn
    if verify_callable is None:
        if lean_parallel_workers <= 1:
            _log_stage(logger, "lean_checker_init_single")
            checker = LeanChecker(copy.deepcopy(lean_cfg))
            checker.setup_project()
        else:
            _log_stage(logger, "lean_checker_init_pool", workers=lean_parallel_workers)
            verify_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=lean_parallel_workers,
                initializer=_init_parallel_lean_checker,
                initargs=(lean_cfg,),
            )
            verify_callable = _parallel_lean_check_task
    elif lean_parallel_workers > 1:
        # For injected verify function (tests/mocks), use threads to parallelize.
        verify_executor = concurrent.futures.ThreadPoolExecutor(max_workers=lean_parallel_workers)

    states: list[dict] = []
    total_rows = len(dataset)
    for row_idx, row in enumerate(dataset, start=1):
        theorem_block = _extract_theorem_block(row)
        sample_id = row.get("id", row.get("name", f"sample_{row_idx}"))
        logs: list[str] = []
        _append(logs, "=" * 80)
        _append(logs, f"SAMPLE {row_idx}/{total_rows} - id={sample_id}")
        _append(logs, "-" * 80)
        _append(logs, "THEOREM BLOCK:")
        _append(logs, theorem_block)
        _append(logs, "=" * 80)
        states.append(
            {
                "row_idx": row_idx,
                "row": row,
                "id": sample_id,
                "theorem": theorem_block,
                "ok": False,
                "lean_log": "",
                "first_prediction": "",
                "first_prediction_lean_code": "",
                "first_extraction_mode": "none",
                "candidates": [],
                "total_generation_ms": 0.0,
                "total_verify_ms": 0.0,
                "logs": logs,
            }
        )

    pending = list(range(len(states)))
    done_count = 0
    try:
        for attempt in range(1, pass_k + 1):
            if not pending:
                break
            logger.info(
                "vLLM batch generation attempt %d/%d for %d pending samples.",
                attempt,
                pass_k,
                len(pending),
            )
            next_pending = []
            for chunk_start in range(0, len(pending), batch_size):
                chunk = pending[chunk_start : chunk_start + batch_size]
                chunk_ids = [states[i]["id"] for i in chunk]
                _log_stage(
                    logger,
                    "vllm_chunk_start",
                    attempt=attempt,
                    chunk_start=chunk_start,
                    chunk_size=len(chunk),
                    chunk_ids="|".join(chunk_ids),
                )
                statements = [states[i]["theorem"] for i in chunk]
                gen_start = time.perf_counter()
                try:
                    batch_preds = prover.generate_proofs_batch(statements, num_samples=1)
                except LLMGenerationTimeoutError as exc:
                    batch_elapsed_ms = (time.perf_counter() - gen_start) * 1000
                    timeout_log = str(exc)
                    per_sample_gen_ms = batch_elapsed_ms / max(1, len(chunk))
                    for state_idx in chunk:
                        state = states[state_idx]
                        state["total_generation_ms"] += per_sample_gen_ms
                        _append(state["logs"], "=" * 80)
                        _append(state["logs"], f"ATTEMPT {attempt}/{pass_k} - GENERATION")
                        _append(state["logs"], "-" * 80)
                        _append(
                            state["logs"],
                            (
                                f"LLM generation timeout: {timeout_log} | gen_ms={per_sample_gen_ms:.1f} | "
                                f"gen_ms(total)={state['total_generation_ms']:.1f}"
                            ),
                        )
                        if attempt == 1:
                            state["first_extraction_mode"] = "llm_timeout"
                            state["lean_log"] = timeout_log
                        state["candidates"].append(
                            {
                                "sample_idx": attempt,
                                "ok": False,
                                "prediction": "",
                                "prediction_lean_code": "",
                                "extraction_mode": "llm_timeout",
                                "lean_output": timeout_log,
                                "generation_ms": per_sample_gen_ms,
                                "generation_ms_total": state["total_generation_ms"],
                                "verify_ms": 0.0,
                                "verify_ms_total": state["total_verify_ms"],
                            }
                        )
                        _append(state["logs"], "=" * 80)
                        if attempt >= pass_k:
                            done_count += 1
                        else:
                            next_pending.append(state_idx)
                    continue
                except Exception:
                    logger.exception(
                        "Unexpected exception during vLLM generation batch. attempt=%d chunk_ids=%s",
                        attempt,
                        chunk_ids,
                    )
                    raise

                batch_elapsed_ms = (time.perf_counter() - gen_start) * 1000
                per_sample_gen_ms = batch_elapsed_ms / max(1, len(chunk))
                prepared = []
                for state_idx, preds in zip(chunk, batch_preds):
                    state = states[state_idx]
                    pred_proof = preds[0] if preds else ""
                    retries_used = attempt - 1
                    retries_remaining = max(0, pass_k - attempt)
                    state["total_generation_ms"] += per_sample_gen_ms

                    _append(state["logs"], "=" * 80)
                    _append(state["logs"], f"ATTEMPT {attempt}/{pass_k} - GENERATION")
                    _append(state["logs"], "-" * 80)
                    _append(
                        state["logs"],
                        f"Sample id={state['id']} | retries_used={retries_used} | retries_remaining={retries_remaining}",
                    )
                    if attempt == 1:
                        state["first_prediction"] = pred_proof
                    extracted_proof, extraction_mode = _extract_lean_code_from_generation(pred_proof)
                    if attempt == 1:
                        state["first_prediction_lean_code"] = extracted_proof
                        state["first_extraction_mode"] = extraction_mode

                    _append(state["logs"], "RAW MODEL OUTPUT:")
                    _append(state["logs"], pred_proof)
                    _append(state["logs"], "-" * 80)
                    _append(state["logs"], f"EXTRACTED LEAN CODE (mode={extraction_mode}):")
                    _append(state["logs"], extracted_proof)
                    _append(state["logs"], "=" * 80)
                    _append(state["logs"], f"ATTEMPT {attempt}/{pass_k} - LEAN VERIFICATION")
                    _append(state["logs"], "-" * 80)
                    prepared.append((state_idx, pred_proof, extracted_proof, extraction_mode, retries_used))

                verify_start = time.perf_counter()
                verify_payloads = [(states[i]["theorem"], extracted) for i, _, extracted, _, _ in prepared]
                if verify_executor is not None and verify_callable is not None:
                    try:
                        verify_results = list(verify_executor.map(verify_callable, verify_payloads))
                    except Exception:
                        logger.exception(
                            "Unexpected exception in parallel Lean verification. attempt=%d chunk_ids=%s",
                            attempt,
                            chunk_ids,
                        )
                        raise
                elif verify_callable is not None:
                    try:
                        verify_results = [verify_callable(payload) for payload in verify_payloads]
                    except Exception:
                        logger.exception(
                            "Unexpected exception in custom verify function. attempt=%d chunk_ids=%s",
                            attempt,
                            chunk_ids,
                        )
                        raise
                else:
                    verify_results = [checker.check_proof(theorem, extracted) for theorem, extracted in verify_payloads]
                verify_elapsed_ms = (time.perf_counter() - verify_start) * 1000
                per_sample_verify_ms = verify_elapsed_ms / max(1, len(prepared))

                for prepared_item, verify_result in zip(prepared, verify_results):
                    state_idx, pred_proof, extracted_proof, extraction_mode, retries_used = prepared_item
                    cur_ok, cur_log = verify_result
                    state = states[state_idx]
                    state["total_verify_ms"] += per_sample_verify_ms

                    _append(state["logs"], "LEAN OUTPUT:")
                    _append(state["logs"], cur_log)
                    _append(state["logs"], "-" * 80)
                    _append(
                        state["logs"],
                        (
                            f"LEAN CHECK RESULT: ok={cur_ok} | gen_ms={per_sample_gen_ms:.1f} | "
                            f"verify_ms={per_sample_verify_ms:.1f} | "
                            f"gen_ms(total)={state['total_generation_ms']:.1f} | "
                            f"verify_ms(total)={state['total_verify_ms']:.1f}"
                        ),
                    )
                    _append(state["logs"], "=" * 80)

                    state["candidates"].append(
                        {
                            "sample_idx": attempt,
                            "ok": cur_ok,
                            "prediction": pred_proof,
                            "prediction_lean_code": extracted_proof,
                            "extraction_mode": extraction_mode,
                            "lean_output": cur_log,
                            "generation_ms": per_sample_gen_ms,
                            "generation_ms_total": state["total_generation_ms"],
                            "verify_ms": per_sample_verify_ms,
                            "verify_ms_total": state["total_verify_ms"],
                        }
                    )

                    if cur_ok:
                        state["ok"] = True
                        state["lean_log"] = cur_log
                        done_count += 1
                        _append(
                            state["logs"],
                            f"SUCCESS - sample id={state['id']} solved at attempt {attempt}/{pass_k} (retries_used={retries_used}).",
                        )
                        _append(state["logs"], "=" * 80)
                    else:
                        if attempt == 1:
                            state["lean_log"] = cur_log
                        if attempt >= pass_k:
                            done_count += 1
                        else:
                            next_pending.append(state_idx)
                    logger.info(
                        "[progress] %d/%d | sample=%s | %s | attempts=%d/%d | gen_ms(total)=%.1f | verify_ms(total)=%.1f",
                        done_count,
                        total_rows,
                        state["id"],
                        "PASS" if cur_ok else "FAIL",
                        attempt,
                        pass_k,
                        float(state["total_generation_ms"]),
                        float(state["total_verify_ms"]),
                    )
            pending = next_pending
    finally:
        if verify_executor is not None:
            verify_executor.shutdown(wait=True, cancel_futures=False)

    packed_results: list[tuple[int, dict, int]] = []
    for state in states:
        if not state["ok"]:
            _append(state["logs"], f"FAILED - sample id={state['id']} after {pass_k} attempts.")
            _append(state["logs"], "=" * 80)
        attempts_used = len(state["candidates"])
        _append(
            state["logs"],
            (
                f"SAMPLE SUMMARY: attempts={attempts_used}/{pass_k} | "
                f"gen_ms(total)={state['total_generation_ms']:.1f} | "
                f"verify_ms(total)={state['total_verify_ms']:.1f}"
            ),
        )

        result = {
            "row_idx": state["row_idx"],
            "id": state["id"],
            "ok": state["ok"],
            "theorem": state["theorem"],
            "comment": state["row"].get("comment"),
            "prediction": state["first_prediction"],
            "prediction_lean_code": state["first_prediction_lean_code"],
            "extraction_mode": state["first_extraction_mode"],
            "lean_output": state["lean_log"],
            "attempts_used": attempts_used,
            "generation_ms_total": state["total_generation_ms"],
            "verify_ms_total": state["total_verify_ms"],
            "pass_k": pass_k,
            "candidates": state["candidates"],
        }
        log_name_raw = state["row"].get("name") or state["row"].get("id") or f"sample_{state['row_idx']}"
        safe_name = str(log_name_raw).replace("/", "_")
        (problem_log_dir / f"{safe_name}.log").write_text("\n".join(state["logs"]) + "\n", encoding="utf-8")
        packed_results.append((state["row_idx"], result, int(state["ok"])))
    return packed_results


def _run_worker(
    worker_idx: int,
    gpu_id: int,
    task_queue: "queue.Queue[tuple[int, dict]]",
    total_rows: int,
    pass_k: int,
    model_cfg: dict,
    lean_cfg: dict,
    problem_log_dir: Path,
    logger: logging.Logger,
    load_state: dict,
    progress_state: dict,
) -> list[tuple[int, dict, int]]:
    worker_model_cfg = copy.deepcopy(model_cfg)
    worker_model_cfg["tensor_parallel_size"] = _resolve_tensor_parallel_size(worker_model_cfg, default=1)
    prover = build_prover_generator(worker_model_cfg)
    with load_state["lock"]:
        load_state["loaded"] += 1
        logger.info("GPU %s model loaded.", gpu_id)
        if load_state["loaded"] == load_state["total"]:
            logger.info("All worker models have finished loading.")

    worker_lean_cfg = copy.deepcopy(lean_cfg)
    base_project_dir = worker_lean_cfg.get("project_dir", ".lean_runner")
    worker_lean_cfg["project_dir"] = f"{base_project_dir}_worker_{worker_idx}"
    checker = LeanChecker(worker_lean_cfg)
    checker.setup_project()

    out: list[tuple[int, dict, int]] = []
    logger.info("worker_%d started on gpu=%s", worker_idx, gpu_id)
    while True:
        try:
            row_idx, row = task_queue.get_nowait()
        except queue.Empty:
            break
        queue_left = task_queue.qsize()
        logger.info(
            "worker_%d dequeued sample_id=%s row_idx=%d queue_remaining≈%d",
            worker_idx,
            row.get("id", row.get("name", f"sample_{row_idx}")),
            row_idx,
            queue_left,
        )
        result, ok_int, logs = _process_one_problem(row_idx, total_rows, row, pass_k, prover, checker)
        log_name_raw = row.get("name") or row.get("id") or f"sample_{row_idx}"
        safe_name = str(log_name_raw).replace("/", "_")
        (problem_log_dir / f"{safe_name}.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
        out.append((row_idx, result, ok_int))
        with progress_state["lock"]:
            progress_state["done"] += 1
            done = progress_state["done"]
            total = progress_state["total"]
            status = "PASS" if ok_int else "FAIL"
            logger.info(
                "[progress] %d/%d | gpu=%s | sample=%s | %s | attempts=%d/%d | gen_ms(total)=%.1f | verify_ms(total)=%.1f",
                done,
                total,
                gpu_id,
                result["id"],
                status,
                int(result.get("attempts_used", 0)),
                int(result.get("pass_k", pass_k)),
                float(result.get("generation_ms_total", 0.0)),
                float(result.get("verify_ms_total", 0.0)),
            )
    logger.info("worker_%d finished on gpu=%s processed=%d", worker_idx, gpu_id, len(out))
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
    _log_stage(logger, "config_loaded", config_path=args.config)

    split = exp_cfg.get("split", "valid")
    max_samples = exp_cfg.get("max_samples")
    output_dir = Path(exp_cfg.get("output_dir", "outputs/experiments/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _attach_file_logger(logger, output_dir)
    _log_stage(logger, "output_dir_ready", output_dir=str(output_dir.resolve()))
    _log_run_configuration(logger, exp_cfg, model_cfg, minif2f_cfg, lean_cfg)
    # Suppress model loading/download progress bars in multi-thread mode.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:
        pass

    dataset = load_minif2f(minif2f_cfg, split=split, max_samples=max_samples)
    _log_stage(logger, "dataset_loaded", split=split, max_samples=max_samples, dataset_size=len(dataset))
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
    _log_stage(logger, "gpu_ids_discovered", gpu_ids=gpu_ids)
    use_vllm = bool(model_cfg.get("use_vllm", True))
    if use_vllm:
        # vLLM manages intra-engine parallelism itself via tensor parallel.
        # Keep a single engine process and use all configured GPUs.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
        model_cfg = copy.deepcopy(model_cfg)
        model_cfg["tensor_parallel_size"] = _resolve_tensor_parallel_size(model_cfg, default=len(gpu_ids))
        worker_count = 1
        gpu_ids = [0]
        logger.info(
            "vLLM enabled: CUDA_VISIBLE_DEVICES=%s, tensor_parallel_size=%d, worker_count=%d",
            os.environ["CUDA_VISIBLE_DEVICES"],
            int(model_cfg["tensor_parallel_size"]),
            worker_count,
        )
        _log_stage(
            logger,
            "vllm_mode_configured",
            cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
            tensor_parallel_size=int(model_cfg["tensor_parallel_size"]),
            worker_count=worker_count,
        )
    else:
        worker_count = min(len(gpu_ids), total_rows)
        gpu_ids = gpu_ids[:worker_count]
    if use_vllm:
        logger.info(
            "Starting vLLM batched run with dynamic backfill: total_samples=%d, pass_k=%d",
            total_rows,
            pass_k,
        )
        try:
            packed_results = _run_vllm_batch_mode(
                dataset=dataset,
                pass_k=pass_k,
                model_cfg=model_cfg,
                lean_cfg=lean_cfg,
                problem_log_dir=problem_log_dir,
                logger=logger,
            )
        except Exception:
            logger.exception("Fatal error in vLLM batch mode.")
            raise
    else:
        indexed_rows = list(enumerate(dataset, start=1))
        task_queue: "queue.Queue[tuple[int, dict]]" = queue.Queue()
        for item in indexed_rows:
            task_queue.put(item)

        packed_results: list[tuple[int, dict, int]] = []
        load_state = {"loaded": 0, "total": worker_count, "lock": threading.Lock()}
        progress_state = {"done": 0, "total": total_rows, "lock": threading.Lock()}
        logger.info(
            "Starting dynamic parallel run: total_samples=%d, workers=%d, gpus=%s",
            total_rows,
            worker_count,
            gpu_ids,
        )
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _run_worker,
                        worker_idx=i,
                        gpu_id=gpu_ids[i],
                        task_queue=task_queue,
                        total_rows=total_rows,
                        pass_k=pass_k,
                        model_cfg=model_cfg,
                        lean_cfg=lean_cfg,
                        problem_log_dir=problem_log_dir,
                        logger=logger,
                        load_state=load_state,
                        progress_state=progress_state,
                    )
                    for i in range(worker_count)
                ]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="workers"):
                    packed_results.extend(future.result())
        except Exception:
            logger.exception("Fatal error in legacy threaded worker mode.")
            raise

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

