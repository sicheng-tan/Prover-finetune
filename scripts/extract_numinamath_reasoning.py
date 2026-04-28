import argparse
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CommentSpan:
    start: int
    end: int
    text: str
    kind: str  # line | block


def _extract_theorem_pos(code: str) -> int:
    theorem_pos = code.find("theorem ")
    if theorem_pos != -1:
        return theorem_pos
    lemma_pos = code.find("lemma ")
    if lemma_pos != -1:
        return lemma_pos
    return 0


def _parse_comments(code: str) -> list[CommentSpan]:
    comments: list[CommentSpan] = []
    i = 0
    n = len(code)
    while i < n:
        if i + 1 < n and code[i] == "-" and code[i + 1] == "-":
            start = i
            i += 2
            while i < n and code[i] != "\n":
                i += 1
            comments.append(CommentSpan(start=start, end=i, text=code[start + 2 : i].strip(), kind="line"))
            continue

        if i + 1 < n and code[i] == "/" and code[i + 1] == "-":
            start = i
            i += 2
            depth = 1
            while i < n and depth > 0:
                if i + 1 < n and code[i] == "/" and code[i + 1] == "-":
                    depth += 1
                    i += 2
                elif i + 1 < n and code[i] == "-" and code[i + 1] == "/":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            end = i
            block_text = code[start + 2 : end - 2].strip() if end >= start + 4 else ""
            comments.append(CommentSpan(start=start, end=end, text=block_text, kind="block"))
            continue
        i += 1
    return comments


def _remove_comment_ranges(code: str, comments: list[CommentSpan]) -> str:
    if not comments:
        return code.strip()

    parts: list[str] = []
    cursor = 0
    for comment in comments:
        remove_end = comment.end
        # Remove the newline that directly follows a comment.
        if remove_end < len(code) and code[remove_end] == "\n":
            remove_end += 1
        if cursor < comment.start:
            parts.append(code[cursor : comment.start])
        cursor = remove_end
    if cursor < len(code):
        parts.append(code[cursor:])
    return "".join(parts).strip()


def _normalize_reasoning_step(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    clean_lines = [line for line in lines if line]
    normalized = "\n".join(clean_lines)
    if ":" in normalized.replace("\n", ""):
        normalized = normalized.replace(":", "")
    return normalized


def _to_sorry_statement(formal_statement: str) -> str:
    text = formal_statement.strip()
    if not text:
        return ""
    if "sorry" in text:
        return text
    if text.endswith(":= by"):
        return text + "\n  sorry"
    return text + "\n:= by\n  sorry"


def process_record(record: dict) -> dict:
    formal_code = str(record.get("formal_ground_truth", ""))
    theorem_pos = _extract_theorem_pos(formal_code)
    comments = _parse_comments(formal_code)

    problem_comment_blocks: list[str] = []
    reasoning_steps: list[str] = []
    for comment in comments:
        normalized = _normalize_reasoning_step(comment.text)
        if not normalized:
            continue
        if comment.kind == "block" and comment.start < theorem_pos:
            problem_comment_blocks.append(normalized)
        elif comment.start >= theorem_pos:
            reasoning_steps.append(normalized)

    proof_no_comments = _remove_comment_ranges(formal_code, comments)
    return {
        "formal_statement": _to_sorry_statement(str(record.get("formal_statement", ""))),
        "reasoning_steps": reasoning_steps,
        "formal_proof_no_comments": proof_no_comments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract reasoning comments and clean Lean proof from NuminaMath JSONL."
    )
    parser.add_argument(
        "--input-path",
        default="data/processed/numinamath_lean_filtered_train.jsonl",
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/numinamath_lean_reasoning_train.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            raw = line.strip()
            if not raw:
                continue
            total += 1
            record = json.loads(raw)
            updated = process_record(record)
            dst.write(json.dumps(updated, ensure_ascii=False) + "\n")
            written += 1

    print(f"input_path={input_path}")
    print(f"output_path={output_path}")
    print(f"total_samples={total}")
    print(f"written_samples={written}")


if __name__ == "__main__":
    main()
