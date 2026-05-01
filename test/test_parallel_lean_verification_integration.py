import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from prover_finetune.experiments.run_experiment import _run_vllm_batch_mode


class MockProver:
    """Return deterministic Lean proof snippets for integration verification."""

    def generate_proofs_batch(self, statements, num_samples=1):
        del num_samples
        outputs = []
        for stmt in statements:
            if "bad_" in stmt:
                outputs.append(["by\n  sorry"])
            else:
                outputs.append(["by\n  trivial"])
        return outputs


def _build_dataset():
    return [
        {"id": "s1", "statement": "theorem good_1 : True :="},
        {"id": "s2", "statement": "theorem bad_1 : True :="},
        {"id": "s3", "statement": "theorem good_2 : True :="},
        {"id": "s4", "statement": "theorem bad_2 : True :="},
        {"id": "s5", "statement": "theorem good_3 : True :="},
        {"id": "s6", "statement": "theorem bad_3 : True :="},
        {"id": "s7", "statement": "theorem good_4 : True :="},
        {"id": "s8", "statement": "theorem bad_4 : True :="},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lean-project-config",
        default="configs/lean_project.example.yaml",
        help="Path to lean project config yaml used by LeanChecker.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=4,
        help="Lean checker parallel worker count.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="vLLM batch size used by scheduler (with MockProver).",
    )
    args = parser.parse_args()

    logger = logging.getLogger("parallel-lean-integration")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = False

    dataset = _build_dataset()
    model_cfg = {
        "vllm_batch_size": int(args.batch_size),
        "lean_parallel_workers": int(args.parallel_workers),
    }
    lean_cfg = {
        "project_config_path": args.lean_project_config,
        "use_lean_interact": True,
        "use_auto_server": True,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        start = time.perf_counter()
        packed = _run_vllm_batch_mode(
            dataset=dataset,
            pass_k=1,
            model_cfg=model_cfg,
            lean_cfg=lean_cfg,
            problem_log_dir=Path(tmpdir),
            logger=logger,
            prover_override=MockProver(),
            verify_fn=None,  # IMPORTANT: use real LeanChecker path
        )
        elapsed = time.perf_counter() - start

        # Keep a lightweight artifact for remote debugging.
        payload = {
            "elapsed_sec": elapsed,
            "parallel_workers": args.parallel_workers,
            "batch_size": args.batch_size,
            "results": [r for _, r, _ in packed],
        }
        out_path = Path(tmpdir) / "integration_results.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] wrote integration result artifact: {out_path}")

    expected_ok_by_id = {
        "s1": True,
        "s2": False,
        "s3": True,
        "s4": False,
        "s5": True,
        "s6": False,
        "s7": True,
        "s8": False,
    }
    result_by_id = {result["id"]: result for _, result, _ in packed}
    for sample_id, expected_ok in expected_ok_by_id.items():
        assert sample_id in result_by_id, f"Missing result for sample_id={sample_id}"
        item = result_by_id[sample_id]
        assert item["ok"] is expected_ok, (
            f"Unexpected verification status for {sample_id}: got={item['ok']} expected={expected_ok}"
        )
        assert int(item["attempts_used"]) == 1, f"Expected single attempt for {sample_id}"
        assert len(item["candidates"]) == 1, f"Expected exactly one candidate for {sample_id}"

    print(
        f"[PASS] lean-interact parallel verification integration test, "
        f"elapsed={elapsed:.3f}s workers={args.parallel_workers}"
    )


if __name__ == "__main__":
    main()

