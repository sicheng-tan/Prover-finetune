import json
import re
from pathlib import Path


THEOREM_START_RE = re.compile(r"^theorem\s+([A-Za-z0-9_'.]+)", re.MULTILINE)
DOC_BLOCK_RE = re.compile(r"/--(.*?)-/", re.DOTALL)


def _nearest_doc_comment_before(lines: list[str], theorem_line_idx: int) -> str:
    i = theorem_line_idx - 1
    while i >= 0 and (lines[i].strip() == "" or lines[i].lstrip().startswith("--")):
        i -= 1

    if i < 0 or not lines[i].strip().endswith("-/"):
        return ""

    end = i
    while i >= 0 and "/--" not in lines[i]:
        i -= 1
    if i < 0:
        return ""

    block = "\n".join(lines[i : end + 1])
    m = DOC_BLOCK_RE.search(block)
    if not m:
        return ""
    return m.group(1).strip()


def parse_lean_theorems(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    line_starts = [0]
    for ln in lines:
        line_starts.append(line_starts[-1] + len(ln) + 1)

    def pos_to_line(pos: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    matches = list(THEOREM_START_RE.finditer(text))
    out = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        theorem_chunk = text[start:end].strip()

        # Keep only theorem declaration (without proof script) where possible.
        split_marker = ":= by"
        marker_pos = theorem_chunk.find(split_marker)
        theorem_def = theorem_chunk[:marker_pos].rstrip() if marker_pos != -1 else theorem_chunk

        theorem_name = m.group(1)
        if ".variants." in theorem_name:
            continue
        line_idx = pos_to_line(start)
        comment = _nearest_doc_comment_before(lines, line_idx)

        out.append(
            {
                "name": theorem_name,
                "definition": theorem_def,
                "comment": comment,
            }
        )

    return out


def main() -> None:
    root = Path("data/minif2f/miniF2F-main/MiniF2F")
    out_dir = Path("data/minif2f/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    valid = parse_lean_theorems(root / "Valid.lean")
    test = parse_lean_theorems(root / "Test.lean")

    (out_dir / "valid.json").write_text(json.dumps(valid, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "test.json").write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"valid_count={len(valid)}")
    print(f"test_count={len(test)}")
    print(f"total_count={len(valid) + len(test)}")


if __name__ == "__main__":
    main()

