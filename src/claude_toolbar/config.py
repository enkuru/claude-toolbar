from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_PATH = Path(
    os.getenv(
        "CLAUDE_TOOLBAR_CONFIG",
        Path.home() / ".config" / "claude_toolbar" / "config.json",
    )
)

DEFAULT_CONFIG = {
    "claude_paths": [],
    "refresh_interval": 5.0,
    "idle_seconds": 120,
    "history_days": 60,
    "session_duration_hours": 5,
    "enable_process_monitor": True,
    "ccusage_refresh_interval": 120,
    "enable_ccusage_prices": True,
    "limit_reset_override": None,
}


@dataclass
class ToolbarConfig:
    claude_paths: List[Path] = field(default_factory=list)
    refresh_interval: float = DEFAULT_CONFIG["refresh_interval"]
    idle_seconds: int = DEFAULT_CONFIG["idle_seconds"]
    history_days: int = DEFAULT_CONFIG["history_days"]
    session_duration_hours: int = DEFAULT_CONFIG["session_duration_hours"]
    enable_process_monitor: bool = DEFAULT_CONFIG["enable_process_monitor"]
    ccusage_refresh_interval: int = DEFAULT_CONFIG["ccusage_refresh_interval"]
    enable_ccusage_prices: bool = DEFAULT_CONFIG["enable_ccusage_prices"]
    limit_reset_override: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolbarConfig":
        raw_paths = data.get("claude_paths") or []
        paths = [Path(p).expanduser() for p in raw_paths if isinstance(p, str)]
        refresh_interval = float(
            data.get("refresh_interval", DEFAULT_CONFIG["refresh_interval"])
        )
        idle_seconds = int(data.get("idle_seconds", DEFAULT_CONFIG["idle_seconds"]))
        history_days = int(data.get("history_days", DEFAULT_CONFIG["history_days"]))
        session_duration_hours = int(
            data.get("session_duration_hours", DEFAULT_CONFIG["session_duration_hours"])
        )
        enable_process_monitor = bool(
            data.get(
                "enable_process_monitor", DEFAULT_CONFIG["enable_process_monitor"]
            )
        )
        ccusage_refresh_interval = int(
            data.get("ccusage_refresh_interval", DEFAULT_CONFIG["ccusage_refresh_interval"])
        )
        enable_ccusage_prices = bool(
            data.get("enable_ccusage_prices", DEFAULT_CONFIG["enable_ccusage_prices"])
        )
        override_raw = data.get("limit_reset_override")
        limit_reset_override = _parse_datetime(override_raw) if override_raw else None

        return cls(
            claude_paths=paths,
            refresh_interval=refresh_interval,
            idle_seconds=idle_seconds,
            history_days=history_days,
            session_duration_hours=session_duration_hours,
            enable_process_monitor=enable_process_monitor,
            ccusage_refresh_interval=ccusage_refresh_interval,
            enable_ccusage_prices=enable_ccusage_prices,
            limit_reset_override=limit_reset_override,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claude_paths": [str(path) for path in self.claude_paths],
            "refresh_interval": self.refresh_interval,
            "idle_seconds": self.idle_seconds,
            "history_days": self.history_days,
            "session_duration_hours": self.session_duration_hours,
            "enable_process_monitor": self.enable_process_monitor,
            "ccusage_refresh_interval": self.ccusage_refresh_interval,
            "enable_ccusage_prices": self.enable_ccusage_prices,
            "limit_reset_override": self.limit_reset_override.isoformat()
            if self.limit_reset_override
            else None,
        }


def ensure_config_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_config() -> ToolbarConfig:
    ensure_config_dir(CONFIG_PATH)
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}
    return ToolbarConfig.from_dict(data)


def save_config(config: ToolbarConfig) -> None:
    payload = config.to_dict()
    ensure_config_dir(CONFIG_PATH)
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _parse_datetime(value: str) -> Optional[datetime]:
    try:
        if value.endswith("Z"):
            return datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


__all__ = ["ToolbarConfig", "load_config", "save_config", "CONFIG_PATH"]
