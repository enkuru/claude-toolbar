"""Minimal subset of :mod:`jaraco.text` required by :mod:`pkg_resources`.

The full `jaraco.text` package pulls in a significant dependency graph which
py2app's bootstrap imports very early.  When building a frozen app we only need
basic helpers used by ``pkg_resources`` to parse requirement metadata.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator


def drop_comment(line: str) -> str:
    """Strip trailing ``" #..."`` segments used for comments."""

    head, sep, _tail = line.partition(" #")
    return head if sep else line


def join_continuation(lines: Iterable[str]) -> Iterator[str]:
    """Collapse lines ending in a continuation backslash.

    Mirrors the tiny subset of behaviour relied upon by ``pkg_resources``.
    The final character preceding the backslash is trimmed, matching the
    upstream helper.
    """

    pending = []
    for raw in lines:
        line = raw.rstrip()
        if line.endswith("\\"):
            pending.append(line[:-1].rstrip())
            continue
        if pending:
            pending.append(line)
            yield "".join(pending)
            pending.clear()
        else:
            yield line
    if pending:
        # Unterminated continuation – emulate the upstream helper by
        # discarding the buffered fragment.
        pending.clear()


def yield_lines(source: Iterable[str] | str) -> Iterator[str]:
    """Yield cleaned, non-empty lines from strings or nested iterables."""

    for line in _iter_lines(source):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        yield text


def _iter_lines(obj: Iterable[str] | str) -> Iterator[str]:
    if isinstance(obj, str):
        for line in obj.splitlines():
            yield line
        return

    for item in obj:
        if isinstance(item, str):
            yield from _iter_lines(item)
        elif isinstance(item, Iterable):
            yield from _iter_lines(item)
        else:
            yield str(item)
