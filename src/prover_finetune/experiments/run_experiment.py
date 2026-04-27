import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .config import ExperimentConfig
from .lean_checker import LeanChecker
from .minif2f import load_minif2f
from .prover import ProverGenerator


def _extract_theorem_block(problem: dict) -> str:
    # Expect miniF2F row to include Lean theorem statement.
    if "definition" in problem:
        return str(problem["definition"])
    if "statement" in problem:
        return str(problem["statement"])
    if "theorem" in problem:
        return str(problem["theorem"])
    raise ValueError("miniF2F sample must contain 'definition', 'statement', or 'theorem' field.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment yaml config")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    exp_cfg = cfg.section("experiment")
    model_cfg = cfg.section("model")
    minif2f_cfg = cfg.section("minif2f")
    lean_cfg = cfg.section("lean")

    split = exp_cfg.get("split", "valid")
    max_samples = exp_cfg.get("max_samples")
    output_dir = Path(exp_cfg.get("output_dir", "outputs/experiments/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_minif2f(minif2f_cfg, split=split, max_samples=max_samples)
    prover = ProverGenerator(model_cfg)
    checker = LeanChecker(lean_cfg)
    checker.setup_project()

    results = []
    num_pass = 0
    for row in tqdm(dataset, desc="miniF2F eval"):
        theorem_block = _extract_theorem_block(row)
        pred_proof = prover.generate_proof(theorem_block)
        ok, lean_log = checker.check_proof(theorem_block, pred_proof)
        num_pass += int(ok)
        results.append(
            {
                "id": row.get("id", row.get("name")),
                "ok": ok,
                "theorem": theorem_block,
                "comment": row.get("comment"),
                "prediction": pred_proof,
                "lean_output": lean_log,
            }
        )

    total = len(results)
    pass_at_1 = (num_pass / total) if total > 0 else 0.0
    summary = {"total": total, "pass": num_pass, "pass@1": pass_at_1}

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

