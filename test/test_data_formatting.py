import json
import importlib
import sys
import tempfile
import traceback
from pathlib import Path
from types import ModuleType


DATA_PATH = Path("data/processed/numinamath_lean_reasoning_train.jsonl")
PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))


class FakeDataset:
    def __init__(self, rows):
        self._rows = rows

    @classmethod
    def from_list(cls, rows):
        return cls(list(rows))

    def map(self, formatter, remove_columns=None):
        mapped_rows = []
        for row in self._rows:
            new_row = dict(row)
            new_row.update(formatter(row))
            mapped_rows.append(new_row)
        return FakeDataset(mapped_rows)

    @property
    def column_names(self):
        if not self._rows:
            return []
        return list(self._rows[0].keys())

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


def _install_mock_dependencies():
    fake_datasets = ModuleType("datasets")
    fake_datasets.Dataset = FakeDataset
    fake_datasets.load_dataset = lambda *args, **kwargs: FakeDataset.from_list([])
    sys.modules["datasets"] = fake_datasets


def _import_target_functions():
    _install_mock_dependencies()
    data_module = importlib.import_module("prover_finetune.finetune.data")
    return data_module._format_deepseek_prover_v2_example, data_module.load_and_process_dataset


class DataFormattingTests:
    def __init__(self):
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Missing dataset file: {DATA_PATH}")
        first_line = DATA_PATH.read_text(encoding="utf-8").splitlines()[0]
        self.sample = json.loads(first_line)
        self._format_deepseek_prover_v2_example, self.load_and_process_dataset = _import_target_functions()
        self._log(
            "Loaded sample",
            {
                "formal_statement": self.sample.get("formal_statement", ""),
                "reasoning_steps_count": len(self.sample.get("reasoning_steps", [])),
                "proof": self.sample.get("formal_proof_no_comments", ""),
            },
        )

    @staticmethod
    def _log(title, payload):
        print(f"[INFO] {title}: {payload}")

    def test_format_single_example_has_expected_structure(self):
        data_cfg = {
            "formal_statement_field": "formal_statement",
            "reasoning_steps_field": "reasoning_steps",
            "proof_field": "formal_proof_no_comments",
        }
        formatted = self._format_deepseek_prover_v2_example(self.sample, data_cfg)["text"]
        self._log(
            "Formatted text",
            {
                "text_length": len(formatted),
                "text": formatted,
            },
        )

        assert formatted.startswith("Complete the following Lean 4 code:")
        assert "### Proof Plan" in formatted
        assert "Lean 4 code:\n```lean4\n" in formatted
        assert "theorem" in formatted
        assert "by" in formatted
        assert "User:\n" not in formatted
        # 验证 reasoning steps 已拼接到 plan（不再要求编号前缀）。
        reasoning_steps = self.sample.get("reasoning_steps", [])
        if reasoning_steps:
            assert str(reasoning_steps[0]).strip() in formatted

    def test_load_and_process_dataset_builds_text_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.jsonl"
            train_path.write_text(
                json.dumps(self.sample, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            data_cfg = {
                "source_type": "jsonl",
                "train_path": str(train_path),
                "formatter_type": "deepseek_prover_v2",
                "formal_statement_field": "formal_statement",
                "reasoning_steps_field": "reasoning_steps",
                "proof_field": "formal_proof_no_comments",
            }
            train_ds, eval_ds = self.load_and_process_dataset(data_cfg)
            self._log(
                "Dataset map result",
                {
                    "dataset_size": len(train_ds),
                    "column_names": train_ds.column_names,
                    "first_text": train_ds[0]["text"],
                },
            )

            assert eval_ds is None
            assert len(train_ds) == 1
            assert "text" in train_ds.column_names
            assert "### Proof Plan" in train_ds[0]["text"]
            assert "Lean 4 code:" in train_ds[0]["text"]


def main():
    tests = DataFormattingTests()
    test_methods = [
        tests.test_format_single_example_has_expected_structure,
        tests.test_load_and_process_dataset_builds_text_field,
    ]

    passed = 0
    failed = 0
    for test_method in test_methods:
        test_name = test_method.__name__
        try:
            test_method()
            print(f"[PASS] {test_name}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {test_name}: {exc}")
            print(traceback.format_exc())
            failed += 1

    total = len(test_methods)
    print(f"\nResult: {passed}/{total} passed, {failed} failed.")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
