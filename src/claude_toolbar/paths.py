from __future__ import annotations

import os
from pathlib import Path
from typing import List

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "claude"
DEFAULT_CODE_DIR = Path.home() / ".claude"
PROJECTS_SUBDIR = "projects"
CLAUDE_CONFIG_ENV = "CLAUDE_CONFIG_DIR"


def discover_claude_paths(explicit_paths: List[Path] | None = None) -> List[Path]:
    paths: List[Path] = []
    seen = set()

    if explicit_paths:
        for path in explicit_paths:
            candidate = path.expanduser().resolve()
            if _is_valid_claude_dir(candidate) and str(candidate) not in seen:
                paths.append(candidate)
                seen.add(str(candidate))
        if paths:
            return paths

    env_value = os.getenv(CLAUDE_CONFIG_ENV, "").strip()
    if env_value:
        for part in env_value.split(","):
            candidate = Path(part).expanduser().resolve()
            if _is_valid_claude_dir(candidate) and str(candidate) not in seen:
                paths.append(candidate)
                seen.add(str(candidate))
        if paths:
            return paths

    for candidate in (DEFAULT_CONFIG_DIR, DEFAULT_CODE_DIR):
        candidate = candidate.expanduser().resolve()
        if _is_valid_claude_dir(candidate) and str(candidate) not in seen:
            paths.append(candidate)
            seen.add(str(candidate))

    return paths


def _is_valid_claude_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    projects_dir = path / PROJECTS_SUBDIR
    return projects_dir.exists() and projects_dir.is_dir()


__all__ = ["discover_claude_paths", "DEFAULT_CONFIG_DIR", "DEFAULT_CODE_DIR", "PROJECTS_SUBDIR"]
