from __future__ import annotations

from pathlib import Path

from setuptools import setup

try:
    from py2app import util as py2app_util
except Exception:  # pragma: no cover - py2app may not be available
    py2app_util = None
else:  # pragma: no cover - used only during app builds
    _orig_is_platform_file = py2app_util.is_platform_file

    def _patched_is_platform_file(path: str) -> bool:
        """Treat extra binary-like files as signable for ad-hoc codesign."""

        if path.endswith((".a", ".sh")):
            return True
        return _orig_is_platform_file(path)

    py2app_util.is_platform_file = _patched_is_platform_file

APP = ["src/claude_toolbar/app.py"]
RESOURCES_DIR = Path("src/claude_toolbar/assets")

VERSION = "0.1.0"
if (Path(__file__).parent / "pyproject.toml").exists():
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:  # pragma: no cover - fallback for Python <3.11
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            tomllib = None  # type: ignore
    if tomllib is not None:
        try:
            with (Path(__file__).parent / "pyproject.toml").open("rb") as handle:
                data = tomllib.load(handle)
            VERSION = data.get("project", {}).get("version", VERSION)
        except Exception:  # pragma: no cover - best effort
            pass

OPTIONS = {
    "argv_emulation": False,
    "plist": {
        "LSUIElement": True,
        "CFBundleName": "Claude Toolbar",
        "CFBundleIdentifier": "com.enes.claude-toolbar",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
    },
    "resources": [str(RESOURCES_DIR)] if RESOURCES_DIR.exists() else [],
}

setup(
    app=APP,
    options={"py2app": OPTIONS},
    setup_requires=["py2app>=0.28"],
)
