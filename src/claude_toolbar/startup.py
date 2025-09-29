from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path


AGENT_IDENTIFIER = "com.enes.claude-toolbar"
AGENT_FILENAME = f"{AGENT_IDENTIFIER}.plist"
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
PLIST_PATH = LAUNCH_AGENTS_DIR / AGENT_FILENAME


def _program_arguments() -> list[str]:
    executable = sys.executable
    return [executable, "-m", "claude_toolbar.app"]


def enable_launch_agent() -> bool:
    try:
        LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    plist_data = {
        "Label": AGENT_IDENTIFIER,
        "ProgramArguments": _program_arguments(),
        "RunAtLoad": True,
        "KeepAlive": False,
        "EnvironmentVariables": {
            "PATH": os.environ.get("PATH", ""),
        },
    }

    try:
        with PLIST_PATH.open("wb") as handle:
            plistlib.dump(plist_data, handle)
    except OSError:
        return False

    _reload_launch_agent()
    return True


def disable_launch_agent() -> bool:
    removed = False
    if PLIST_PATH.exists():
        try:
            PLIST_PATH.unlink()
            removed = True
        except OSError:
            removed = False
    _bootout_launch_agent()
    return removed


def is_launch_agent_enabled() -> bool:
    return PLIST_PATH.exists()


def _reload_launch_agent() -> None:
    if not PLIST_PATH.exists():
        return
    uid = os.getuid()
    try:
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(PLIST_PATH)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return
    try:
        subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(PLIST_PATH)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return


def _bootout_launch_agent() -> None:
    uid = os.getuid()
    try:
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(PLIST_PATH)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return


__all__ = [
    "enable_launch_agent",
    "disable_launch_agent",
    "is_launch_agent_enabled",
]
