from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import psutil


@dataclass
class ProcessSnapshot:
    pid: int
    status: str
    cmdline: List[str]
    cwd: str | None


class ProcessMonitor:
    def list_relevant(self) -> List[ProcessSnapshot]:
        matches: List[ProcessSnapshot] = []
        for proc in psutil.process_iter([
            "pid",
            "name",
            "cmdline",
            "cwd",
            "status",
        ]):
            try:
                name = (proc.info.get("name") or "").lower()
                cmdline = proc.info.get("cmdline") or []
                if not _looks_like_claude_process(name, cmdline):
                    continue
                matches.append(
                    ProcessSnapshot(
                        pid=proc.info.get("pid", proc.pid),
                        status=proc.info.get("status", "unknown"),
                        cmdline=list(cmdline),
                        cwd=proc.info.get("cwd"),
                    )
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return matches

    def map_by_cwd(self, processes: Iterable[ProcessSnapshot]) -> Dict[str, List[ProcessSnapshot]]:
        mapping: Dict[str, List[ProcessSnapshot]] = {}
        for snap in processes:
            cwd = snap.cwd
            if not cwd:
                continue
            mapping.setdefault(cwd, []).append(snap)
        return mapping


def _looks_like_claude_process(name: str, cmdline: List[str]) -> bool:
    if "claude" in name:
        return True
    return any("claude" in part.lower() for part in cmdline)


__all__ = ["ProcessMonitor", "ProcessSnapshot"]
