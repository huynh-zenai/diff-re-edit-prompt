"""Unit tests for utils.search_replace. Run: `python3 tests/test_search_replace.py`."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline_service"))

from utils.search_replace import (  # noqa: E402
    SearchReplaceError,
    apply_blocks,
    parse_blocks,
)


GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

_fails = 0


def _check(name: str, ok: bool, detail: str = "") -> None:
    global _fails
    if ok:
        print(f"  {GREEN}PASS{RESET} {name}")
    else:
        _fails += 1
        print(f"  {RED}FAIL{RESET} {name}  {detail}")


SAMPLE = """\
export default function generate(THREE) {
  const seat = new THREE.Mesh(seatGeom, woodMat);
  seat.position.y = 0.2;

  const leg = new THREE.Mesh(legGeom, woodMat);
  leg.position.y = -0.1;

  const group = new THREE.Group();
  group.add(seat);
  group.add(leg);
  return group;
}
"""


def test_parse_single_block() -> None:
    text = """<<<<<<< SEARCH
seat.position.y = 0.2;
=======
seat.position.y = 0.25;
>>>>>>> REPLACE"""
    blocks, full = parse_blocks(text)
    _check("parse: single block", len(blocks) == 1 and full is None)
    _check("parse: search content", blocks[0][0] == "seat.position.y = 0.2;")
    _check("parse: replace content", blocks[0][1] == "seat.position.y = 0.25;")


def test_parse_multi_block() -> None:
    text = """preamble that should be ignored
<<<<<<< SEARCH
a
=======
A
>>>>>>> REPLACE

some text in between

<<<<<<< SEARCH
b
=======
B
>>>>>>> REPLACE
"""
    blocks, full = parse_blocks(text)
    _check(
        "parse: multi block count",
        len(blocks) == 2 and full is None,
        detail=f"got {len(blocks)} blocks",
    )
    _check("parse: block 1", blocks[0] == ("a", "A"))
    _check("parse: block 2", blocks[1] == ("b", "B"))


def test_parse_empty_replace() -> None:
    text = """<<<<<<< SEARCH
delete me
=======
>>>>>>> REPLACE"""
    blocks, full = parse_blocks(text)
    _check("parse: empty replace", len(blocks) == 1 and blocks[0][1] == "")


def test_parse_full_rewrite() -> None:
    body = "export default function generate(THREE) { return new THREE.Group(); }"
    text = f"<<<<<<< FULL_REWRITE\n{body}\n>>>>>>> END_REWRITE"
    blocks, full = parse_blocks(text)
    _check("parse: full rewrite", full == body and not blocks)


def test_parse_outer_fence_stripped() -> None:
    text = "```\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n```"
    blocks, full = parse_blocks(text)
    _check("parse: outer fence stripped", len(blocks) == 1 and blocks[0] == ("x", "y"))


def test_parse_no_markers() -> None:
    blocks, full = parse_blocks("just some text without markers")
    _check("parse: no markers -> ([], None)", not blocks and full is None)


def test_apply_single_block() -> None:
    blocks = [("seat.position.y = 0.2;", "seat.position.y = 0.25;")]
    result = apply_blocks(SAMPLE, blocks)
    _check(
        "apply: single block",
        "seat.position.y = 0.25;" in result and "seat.position.y = 0.2;" not in result,
    )


def test_apply_multi_block_sequential() -> None:
    blocks = [
        (
            "seat.position.y = 0.2;",
            "seat.position.y = 0.25;\n  seat.scale.set(1.2,1,1);",
        ),
        ("group.add(leg);", "group.add(leg);\n  group.add(armrest);"),
    ]
    result = apply_blocks(SAMPLE, blocks)
    _check(
        "apply: multi block both applied",
        "seat.scale.set(1.2,1,1);" in result and "group.add(armrest);" in result,
    )


def test_apply_deletion() -> None:
    blocks = [("  leg.position.y = -0.1;\n", "")]
    result = apply_blocks(SAMPLE, blocks)
    _check("apply: deletion removes span", "leg.position.y" not in result)


def test_apply_addition_via_anchor() -> None:
    anchor = "  group.add(leg);"
    blocks = [(anchor, f"{anchor}\n  group.add(armrest_left);\n  group.add(armrest_right);")]
    result = apply_blocks(SAMPLE, blocks)
    _check(
        "apply: addition via anchor",
        "armrest_left" in result
        and "armrest_right" in result
        and result.count("group.add(leg);") == 1,
    )


def test_apply_search_miss_raises() -> None:
    blocks = [("nonexistent line", "anything")]
    try:
        apply_blocks(SAMPLE, blocks)
        _check("apply: search miss raises", False, detail="did not raise")
    except SearchReplaceError as exc:
        _check("apply: search miss raises", True)
        _check("apply: miss carries hint", bool(exc.hint))


def test_apply_search_ambiguous_raises() -> None:
    src = "x = 1;\ny = 2;\nx = 1;\n"
    try:
        apply_blocks(src, [("x = 1;", "x = 3;")])
        _check("apply: ambiguous raises", False, detail="did not raise")
    except SearchReplaceError as exc:
        _check("apply: ambiguous raises", "matches 2" in str(exc) or "matches" in str(exc))


def test_apply_empty_search_raises() -> None:
    try:
        apply_blocks(SAMPLE, [("", "anything")])
        _check("apply: empty search raises", False, detail="did not raise")
    except SearchReplaceError:
        _check("apply: empty search raises", True)


def test_apply_whitespace_tolerant() -> None:
    src = "const seat = new THREE.Mesh(g, m);   \nseat.position.y = 0.2;\n"
    blocks = [
        (
            "const seat = new THREE.Mesh(g, m);",
            "const seat = new THREE.Mesh(g2, m);",
        )
    ]
    result = apply_blocks(src, blocks)
    _check(
        "apply: whitespace-tolerant fallback",
        "new THREE.Mesh(g2, m)" in result,
        detail=repr(result),
    )


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    print(f"Running {len(tests)} tests...")
    for test in tests:
        test()
    print()
    if _fails:
        print(f"{RED}{_fails} test(s) failed.{RESET}")
        return 1
    print(f"{GREEN}All tests passed.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
