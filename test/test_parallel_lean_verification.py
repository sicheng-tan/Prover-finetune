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
    def generate_proofs_batch(self, statements, num_samples=1):
        del num_samples
        return [[f"by -- mock proof for {stmt}"] for stmt in statements]


def _mock_verify(payload):
    theorem, proof = payload
    del proof
    time.sleep(0.1)
    return ("good" in theorem), ("Verification successful" if "good" in theorem else "mock failure")


def main():
    logger = logging.getLogger("parallel-lean-test")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = False

    dataset = [
        {"id": "s1", "statement": "theorem good_1 : True := by trivial"},
        {"id": "s2", "statement": "theorem bad_1 : True := by trivial"},
        {"id": "s3", "statement": "theorem good_2 : True := by trivial"},
        {"id": "s4", "statement": "theorem bad_2 : True := by trivial"},
        {"id": "s5", "statement": "theorem good_3 : True := by trivial"},
        {"id": "s6", "statement": "theorem bad_3 : True := by trivial"},
        {"id": "s7", "statement": "theorem good_4 : True := by trivial"},
        {"id": "s8", "statement": "theorem bad_4 : True := by trivial"},
    ]
    model_cfg = {
        "vllm_batch_size": 8,
        "lean_parallel_workers": 4,
    }
    lean_cfg = {}

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
            verify_fn=_mock_verify,
        )
        elapsed = time.perf_counter() - start

    # With 8 samples * 0.1s verify each, serial runtime would be ~0.8s.
    # We expect clear speedup from 4-way parallel verify.
    assert elapsed < 0.7, f"Parallel verify did not speed up enough: elapsed={elapsed:.3f}s"
    assert len(packed) == len(dataset)

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
        cand = item["candidates"][0]
        assert cand["ok"] is expected_ok, f"Candidate ok mismatch for {sample_id}"
        assert isinstance(cand.get("lean_output", ""), str) and cand.get("lean_output", "") != ""

    passed = sum(1 for sample_id, ok in expected_ok_by_id.items() if result_by_id[sample_id]["ok"] == ok)
    assert passed == len(expected_ok_by_id), f"Expected all per-sample checks to pass, got {passed}"
    print(f"[PASS] parallel lean verification test, elapsed={elapsed:.3f}s")


if __name__ == "__main__":
    main()

