from __future__ import annotations

import re
from difflib import SequenceMatcher

_BLOCK_RE = re.compile(
    r"<{7}\s*SEARCH[ \t]*\n(.*?)\n?={7}[ \t]*\n(.*?)\n?>{7}\s*REPLACE",
    re.DOTALL,
)
_FULL_REWRITE_RE = re.compile(
    r"<{7}\s*FULL_REWRITE[ \t]*\n(.*?)\n?>{7}\s*END_REWRITE",
    re.DOTALL,
)
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\n(.*?)\n```\s*$", re.DOTALL)


class SearchReplaceError(Exception):
    """Raised when a SEARCH block cannot be applied to the current source."""

    def __init__(self, message: str, *, search: str = "", hint: str = "") -> None:
        super().__init__(message)
        self.search = search
        self.hint = hint


def _strip_outer_fence(text: str) -> str:
    text = text.strip()
    m = _CODE_FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


def parse_blocks(text: str) -> tuple[list[tuple[str, str]], str | None]:
    """Parse model output into SEARCH/REPLACE blocks or a FULL_REWRITE body."""
    stripped = _strip_outer_fence(text)
    full = _FULL_REWRITE_RE.search(stripped)
    if full:
        return [], full.group(1)
    blocks = [(m.group(1), m.group(2)) for m in _BLOCK_RE.finditer(stripped)]
    return blocks, None


def apply_blocks(source: str, blocks: list[tuple[str, str]]) -> str:
    """Apply SEARCH/REPLACE blocks sequentially to `source`."""
    out = source
    for idx, (search, replace) in enumerate(blocks):
        if search == "":
            raise SearchReplaceError(
                f"Block #{idx + 1}: empty SEARCH not allowed",
                search=search,
            )
        count = out.count(search)
        if count == 1:
            out = out.replace(search, replace, 1)
            continue
        if count > 1:
            raise SearchReplaceError(
                f"Block #{idx + 1}: SEARCH matches {count} locations (must be unique)",
                search=search,
                hint="Add more surrounding lines to the SEARCH block to make it unique.",
            )
        span = _whitespace_tolerant_locate(out, search)
        if span is None:
            raise SearchReplaceError(
                f"Block #{idx + 1}: SEARCH not found in source",
                search=search,
                hint=_closest_snippet(out, search),
            )
        start, end = span
        out = out[:start] + replace + out[end:]
    return out


def _whitespace_tolerant_locate(source: str, search: str) -> tuple[int, int] | None:
    """Find `search` in `source` ignoring per-line trailing whitespace."""
    src_lines = source.splitlines(keepends=True)
    needle_lines = search.splitlines()
    if not needle_lines:
        return None
    needle_stripped = [ln.rstrip() for ln in needle_lines]
    n = len(needle_stripped)
    matches: list[tuple[int, int]] = []
    for i in range(len(src_lines) - n + 1):
        window = [src_lines[j].rstrip() for j in range(i, i + n)]
        if window == needle_stripped:
            start = sum(len(src_lines[k]) for k in range(i))
            end = start + sum(len(src_lines[k]) for k in range(i, i + n))
            matches.append((start, end))
            if len(matches) > 1:
                return None
    if len(matches) != 1:
        return None
    return matches[0]


def _closest_snippet(source: str, needle: str, context: int = 3) -> str:
    """Return the source span most similar to the first line of `needle`."""
    src_lines = source.splitlines()
    needle_lines = needle.splitlines()
    if not needle_lines or not src_lines:
        return ""
    head = needle_lines[0]
    best_idx = -1
    best_ratio = 0.0
    for i, line in enumerate(src_lines):
        ratio = SequenceMatcher(None, line, head).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
    if best_idx < 0:
        return ""
    start = max(0, best_idx - context)
    end = min(len(src_lines), best_idx + len(needle_lines) + context)
    return "\n".join(src_lines[start:end])
