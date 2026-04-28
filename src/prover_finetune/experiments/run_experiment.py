import argparse
import json
import logging
import re
from pathlib import Path

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
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose_logging else logging.WARNING)
    logger.propagate = False
    return logger


def _extract_lean_code_from_generation(generation: str) -> tuple[str, str]:
    lean4_match = re.search(r"```lean4\b[^\n\r]*[\r]?\n(.*?)```", generation, flags=re.DOTALL | re.IGNORECASE)
    if lean4_match:
        return lean4_match.group(1).strip(), "lean4_fence"

    lean_match = re.search(r"```lean\b[^\n\r]*[\r]?\n(.*?)```", generation, flags=re.DOTALL | re.IGNORECASE)
    if lean_match:
        return lean_match.group(1).strip(), "lean_fence"

    return generation.strip(), "raw_fallback"


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

    dataset = load_minif2f(minif2f_cfg, split=split, max_samples=max_samples)
    prover = build_prover_generator(model_cfg)
    checker = LeanChecker(lean_cfg)
    checker.setup_project()
    logger.info("Lean project setup completed.")

    pass_k = int(exp_cfg.get("pass_k", 1))
    results = []
    num_pass = 0
    total_rows = len(dataset)
    for row_idx, row in enumerate(tqdm(dataset, desc="miniF2F eval"), start=1):
        theorem_block = _extract_theorem_block(row)
        sample_id = row.get("id", row.get("name", f"sample_{row_idx}"))
        logger.info("=" * 80)
        logger.info("Sample %d/%d | id=%s", row_idx, total_rows, sample_id)
        logger.info("Theorem block:\n%s", theorem_block)
        ok = False
        lean_log = ""
        first_prediction = ""
        first_prediction_lean_code = ""
        first_extraction_mode = "none"
        candidates = []
        for idx in range(1, pass_k + 1):
            retries_used = idx - 1
            retries_remaining = max(0, pass_k - idx)
            try:
                pred_proof = prover.generate_proof(theorem_block)
            except LLMGenerationTimeoutError as exc:
                timeout_log = str(exc)
                logger.warning(
                    "Sample id=%s | attempt %d/%d | LLM timeout: %s",
                    sample_id,
                    idx,
                    pass_k,
                    timeout_log,
                )
                if idx == 1:
                    first_prediction = ""
                    first_prediction_lean_code = ""
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
                continue
            if idx == 1:
                first_prediction = pred_proof
            extracted_proof, extraction_mode = _extract_lean_code_from_generation(pred_proof)
            if idx == 1:
                first_prediction_lean_code = extracted_proof
                first_extraction_mode = extraction_mode
            logger.info(
                "Sample id=%s | attempt %d/%d | retries_used=%d | retries_remaining=%d",
                sample_id,
                idx,
                pass_k,
                retries_used,
                retries_remaining,
            )
            logger.info("Full LLM generation (attempt %d):\n%s", idx, pred_proof)
            logger.info(
                "Lean code extraction (attempt %d): mode=%s\nExtracted Lean code:\n%s",
                idx,
                extraction_mode,
                extracted_proof,
            )
            cur_ok, cur_log = checker.check_proof(theorem_block, extracted_proof)
            logger.info(
                "Lean check result (attempt %d): ok=%s\nLean output:\n%s",
                idx,
                cur_ok,
                cur_log,
            )
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
                logger.info(
                    "Sample id=%s solved at attempt %d/%d (retries_used=%d).",
                    sample_id,
                    idx,
                    pass_k,
                    retries_used,
                )
                break
            # Keep the first failure log for easier debugging.
            if idx == 1:
                lean_log = cur_log

        num_pass += int(ok)
        if not ok:
            logger.info(
                "Sample id=%s failed after %d attempts (retries_used=%d).",
                sample_id,
                pass_k,
                max(0, pass_k - 1),
            )
        results.append(
            {
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
        )

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

