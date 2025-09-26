from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def to_local(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone()


def format_tokens(tokens: int) -> str:
    if tokens >= 1_000_000_000:
        return f"{tokens / 1_000_000_000:.2f}B"
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.2f}M"
    if tokens >= 1000:
        return f"{tokens / 1000:.1f}k"
    return str(tokens)


def format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def format_ts(dt: Optional[datetime]) -> str:
    if dt is None:
        return "—"
    local = to_local(dt)
    return local.strftime("%Y-%m-%d %H:%M")


def format_relative(dt: Optional[datetime], now: Optional[datetime] = None) -> str:
    if dt is None:
        return "unknown"
    now = now or datetime.now(timezone.utc)
    delta = abs((now - dt).total_seconds())
    if delta < 60:
        return "just now" if dt <= now else "in <1m"
    minutes = int(delta // 60)
    if minutes < 60:
        return f"{minutes}m ago" if dt <= now else f"in {minutes}m"
    hours = int(delta // 3600)
    if hours < 24:
        return f"{hours}h ago" if dt <= now else f"in {hours}h"
    days = int(delta // 86400)
    return f"{days}d ago" if dt <= now else f"in {days}d"


def format_currency(value: Optional[float]) -> str:
    if value is None:
        return "$0.00"
    return f"${value:.2f}"


__all__ = [
    "format_currency",
    "format_elapsed",
    "format_relative",
    "format_tokens",
    "format_ts",
    "parse_timestamp",
    "to_local",
]
