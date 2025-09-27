from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional


class SessionStatus(str, Enum):
    RUNNING = "running"
    IDLE = "idle"
    WAITING = "waiting"


@dataclass
class UsageTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    def add_usage(self, usage: Dict[str, int]) -> None:
        self.input_tokens += int(usage.get("input_tokens", 0))
        self.output_tokens += int(usage.get("output_tokens", 0))
        self.cache_creation_tokens += int(usage.get("cache_creation_input_tokens", 0))
        self.cache_read_tokens += int(usage.get("cache_read_input_tokens", 0))

    def merge(self, other: "UsageTotals") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.cache_read_tokens += other.cache_read_tokens

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
        )


@dataclass
class PendingTool:
    tool_id: str
    name: str
    started_at: Optional[datetime]
    description: Optional[str] = None


@dataclass
class ProcessInfo:
    pid: int
    status: str
    cmdline: List[str] = field(default_factory=list)


@dataclass
class WindowInfo:
    active_start: Optional[datetime] = None
    active_end: Optional[datetime] = None
    last_start: Optional[datetime] = None
    last_end: Optional[datetime] = None
    duration: timedelta = timedelta(hours=5)


@dataclass
class SessionState:
    session_id: str
    project: str
    file_path: str
    cwd: Optional[str] = None
    first_activity: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    totals: UsageTotals = field(default_factory=UsageTotals)
    last_model: Optional[str] = None
    pending_tools: Dict[str, PendingTool] = field(default_factory=dict)
    awaiting_approval: bool = False
    cost_usd: float = 0.0
    awaiting_message: Optional[str] = None
    last_event_type: Optional[str] = None
    last_event_role: Optional[str] = None
    processes: List[ProcessInfo] = field(default_factory=list)
    last_tool_name: Optional[str] = None

    def clone_summary(self, status: SessionStatus) -> "SessionSummary":
        return SessionSummary(
            session_id=self.session_id,
            project=self.project,
            cwd=self.cwd,
            status=status,
            last_activity=self.last_activity,
            totals=UsageTotals(
                input_tokens=self.totals.input_tokens,
                output_tokens=self.totals.output_tokens,
                cache_creation_tokens=self.totals.cache_creation_tokens,
                cache_read_tokens=self.totals.cache_read_tokens,
            ),
            last_model=self.last_model,
            awaiting_message=self.awaiting_message,
            processes=list(self.processes),
            cost_usd=self.cost_usd,
            file_path=self.file_path,
            awaiting_approval=self.awaiting_approval,
            pending_tool_count=len(self.pending_tools),
        )


@dataclass
class SessionSummary:
    session_id: str
    project: str
    cwd: Optional[str]
    status: SessionStatus
    last_activity: Optional[datetime]
    totals: UsageTotals
    last_model: Optional[str]
    awaiting_message: Optional[str]
    processes: List[ProcessInfo] = field(default_factory=list)
    cost_usd: float = 0.0
    file_path: Optional[str] = None
    awaiting_approval: bool = False
    pending_tool_count: int = 0


@dataclass
class LimitInfo:
    timestamp: Optional[datetime]
    source: str


@dataclass
class UsageSummary:
    today: UsageTotals
    seven_day: UsageTotals
    month: UsageTotals
    total_sessions: int
    limit_info: LimitInfo
    window_info: WindowInfo
    today_cost: float = 0.0
    seven_day_cost: float = 0.0
    month_cost: float = 0.0


__all__ = [
    "LimitInfo",
    "PendingTool",
    "ProcessInfo",
    "SessionState",
    "SessionStatus",
    "SessionSummary",
    "UsageSummary",
    "UsageTotals",
    "WindowInfo",
]
