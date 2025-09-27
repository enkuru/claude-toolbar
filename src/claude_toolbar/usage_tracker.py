from __future__ import annotations

import json
import re
import subprocess
import time
from collections import defaultdict, OrderedDict
import logging
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import ToolbarConfig
from .models import (
    LimitInfo,
    PendingTool,
    ProcessInfo,
    SessionState,
    SessionStatus,
    SessionSummary,
    UsageSummary,
    UsageTotals,
    WindowInfo,
)
from .paths import PROJECTS_SUBDIR
from .process_monitor import ProcessMonitor
from .utils import parse_timestamp, to_local

APPROVAL_PATTERNS = [
    "requires approval",
    "awaiting approval",
    "needs approval",
    "needs your approval",
    "requires manual approval",
    "requires permission",
    "needs permission",
    "do you want to proceed",
    "enter your selection",
    "choose an option",
]

SCAN_INTERVAL_SECONDS = 30.0

FILES_PER_TICK = 50
INITIAL_FILES_PER_TICK = 100
SECOND_PASS_FILES_PER_TICK = 200
SNAPSHOT_SAVE_INTERVAL = 5.0
SNAPSHOT_PATH = Path.home() / ".config" / "claude_toolbar" / "session_snapshot.json"
SESSION_CACHE_PATH = Path.home() / ".config" / "claude_toolbar" / "session_cache.json"

DEBUG_MODE = os.getenv("CLAUDE_TOOLBAR_DEBUG")
logger = logging.getLogger("claude_toolbar")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[claude-toolbar] %(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)


@dataclass
class FileState:
    path: Path
    project: str
    position: int = 0
    size: int = 0
    session_id_hint: Optional[str] = None
    scanned_initially: bool = False


class UsageTracker:
    def __init__(self, config: ToolbarConfig, claude_paths: List[Path]):
        self.config = config
        self.claude_paths = claude_paths
        self.file_states: OrderedDict[Path, FileState] = OrderedDict()
        self.sessions: Dict[str, SessionState] = {}
        self.daily_totals: Dict[str, UsageTotals] = defaultdict(UsageTotals)
        self.daily_costs: Dict[str, float] = defaultdict(float)
        self.limit_candidates: set[datetime] = set()
        self.session_costs: Dict[str, float] = {}
        self._last_scan = 0.0
        self._last_price_refresh = 0.0
        self._process_monitor = ProcessMonitor() if config.enable_process_monitor else None
        self._snapshots: Dict[str, Dict[str, float]] = {}
        self._dirty_snapshot = False
        self._last_snapshot_save = 0.0
        self._max_files_per_tick = FILES_PER_TICK
        self._initializing = True
        self._activity_history: Dict[str, List[datetime]] = defaultdict(list)
        self._last_limit_notice: Optional[datetime] = None
        self._load_snapshots()
        self._load_session_cache()
        if self.file_states:
            self._initializing = len(self.sessions) == 0
        self._activity_history: Dict[str, List[datetime]] = defaultdict(list)
        logger.debug(
            "UsageTracker initialized: sessions=%s, files=%s, initializing=%s",
            len(self.sessions),
            len(self.file_states),
            self._initializing,
        )
        self._ccusage_thread: Optional[threading.Thread] = None
        self._ccusage_lock = threading.Lock()
        self._ccusage_thread: Optional[threading.Thread] = None

    def _load_session_cache(self) -> None:
        try:
            with SESSION_CACHE_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        for session_id, payload in data.items():
            totals = UsageTotals(
                input_tokens=int(payload.get("input_tokens", 0)),
                output_tokens=int(payload.get("output_tokens", 0)),
                cache_creation_tokens=int(payload.get("cache_creation_tokens", 0)),
                cache_read_tokens=int(payload.get("cache_read_tokens", 0)),
            )
            session = SessionState(
                session_id=session_id,
                project=payload.get("project") or session_id,
                file_path=payload.get("file_path") or "",
                first_activity=parse_timestamp(payload.get("first_activity")),
                last_activity=parse_timestamp(payload.get("last_activity")),
                totals=totals,
            )
            session.cost_usd = float(payload.get("cost_usd", 0.0))
            self.sessions[session_id] = session
            if session.last_activity:
                history = self._activity_history.setdefault(session_id, [])
                history.append(session.last_activity)
        if self.sessions:
            logger.debug("Loaded %s sessions from cache", len(self.sessions))
    def _load_snapshots(self) -> None:
        try:
            with SNAPSHOT_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            self._snapshots = {}
            return
        except (OSError, json.JSONDecodeError):
            self._snapshots = {}
            return
        if isinstance(data, dict):
            self._snapshots = data
        else:
            self._snapshots = {}

    def _persist_snapshots(self) -> None:
        try:
            SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                str(path): {
                    "position": state.position,
                    "size": state.size,
                    "session_id": state.session_id_hint,
                }
                for path, state in self.file_states.items()
            }
            with SNAPSHOT_PATH.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            self._snapshots = payload
            self._dirty_snapshot = False
            self._last_snapshot_save = time.time()
        except OSError:
            return

    def _persist_session_cache(self) -> None:
        try:
            SESSION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {}
            for session_id, session in self.sessions.items():
                payload[session_id] = {
                    "project": session.project,
                    "file_path": session.file_path,
                    "first_activity": session.first_activity.isoformat() if session.first_activity else None,
                    "last_activity": session.last_activity.isoformat() if session.last_activity else None,
                    "input_tokens": session.totals.input_tokens,
                    "output_tokens": session.totals.output_tokens,
                    "cache_creation_tokens": session.totals.cache_creation_tokens,
                    "cache_read_tokens": session.totals.cache_read_tokens,
                    "cost_usd": session.cost_usd,
                }
            with SESSION_CACHE_PATH.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except OSError:
            return

    def is_initializing(self) -> bool:
        return self._initializing


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self) -> None:
        update_start = time.perf_counter()
        now = time.time()
        if now - self._last_scan > SCAN_INTERVAL_SECONDS or not self.file_states:
            self._discover_files()
            self._last_scan = now
        limit = self._max_files_per_tick
        if self._initializing and self.file_states:
            if not any(state.scanned_initially for state in self.file_states.values()):
                limit = max(limit, min(INITIAL_FILES_PER_TICK, len(self.file_states)))
            else:
                limit = max(limit, min(SECOND_PASS_FILES_PER_TICK, len(self.file_states)))
        processed = 0
        for path, state in list(self.file_states.items()):
            if processed >= limit:
                break
            self._process_file(state)
            self.file_states.move_to_end(path)
            processed += 1
        if self._process_monitor:
            self._refresh_processes()
        if self._initializing:
            if not self.file_states:
                self._initializing = False
            else:
                all_scanned = all(state.scanned_initially for state in self.file_states.values())
                if all_scanned:
                    self._initializing = False
        if self._dirty_snapshot and time.time() - self._last_snapshot_save > SNAPSHOT_SAVE_INTERVAL:
            self._persist_snapshots()
            self._persist_session_cache()
        self._maybe_refresh_ccusage_async()
        logger.debug(
            "update complete in %.3fs (sessions=%s processed_limit=%s, initializing=%s)",
            time.perf_counter() - update_start,
            len(self.sessions),
            limit,
            self._initializing,
        )

    def get_session_summaries(self) -> List[SessionSummary]:
        now = datetime.now(timezone.utc)
        summaries: List[SessionSummary] = []
        for state in self.sessions.values():
            status = self._determine_status(state, now)
            summaries.append(state.clone_summary(status))
        summaries.sort(key=_session_sort_key, reverse=True)
        return summaries

    def get_usage_summary(self) -> UsageSummary:
        today_local = datetime.now().astimezone().date()
        seven_days_ago = today_local - timedelta(days=6)
        month_start = today_local.replace(day=1)

        today_totals = UsageTotals()
        seven_day_totals = UsageTotals()
        month_totals = UsageTotals()

        for day_key, totals in self.daily_totals.items():
            try:
                day = date.fromisoformat(day_key)
            except ValueError:
                continue
            if day == today_local:
                today_totals.merge(totals)
            if seven_days_ago <= day <= today_local:
                seven_day_totals.merge(totals)
            if day >= month_start:
                month_totals.merge(totals)

        limit_info = self._resolve_limit_info()
        window_info = self._compute_window_info()

        today_cost = self.daily_costs.get(today_local.strftime("%Y-%m-%d"), 0.0)
        seven_day_cost = 0.0
        month_cost = 0.0
        for day_key, cost in self.daily_costs.items():
            try:
                day = date.fromisoformat(day_key)
            except ValueError:
                continue
            if seven_days_ago <= day <= today_local:
                seven_day_cost += cost
            if day >= month_start:
                month_cost += cost

        return UsageSummary(
            today=today_totals,
            seven_day=seven_day_totals,
            month=month_totals,
            total_sessions=len(self.sessions),
            limit_info=limit_info,
            window_info=window_info,
            today_cost=today_cost,
            seven_day_cost=seven_day_cost,
            month_cost=month_cost,
        )

    def sessions_ready(self, session_ids: Iterable[str]) -> bool:
        for session_id in session_ids:
            state = self.sessions.get(session_id)
            if state is None:
                continue
            if state.file_path:
                path = Path(state.file_path)
                file_state = self.file_states.get(path)
                if file_state is not None and not file_state.scanned_initially:
                    return False
            if state.last_activity is None and not state.totals.total_tokens:
                return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _discover_files(self) -> None:
        for base_path in self.claude_paths:
            projects_dir = base_path / PROJECTS_SUBDIR
            if not projects_dir.exists():
                continue
            try:
                candidates = sorted(
                    projects_dir.rglob("*.jsonl"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            except FileNotFoundError:
                continue
            for path in candidates:
                if path in self.file_states:
                    continue
                project = _extract_project(path)
                state = FileState(path=path, project=project)
                snapshot = self._snapshots.get(str(path))
                if snapshot:
                    state.session_id_hint = snapshot.get("session_id")
                self.file_states[path] = state
                self.file_states.move_to_end(path, last=False)
        logger.debug("discovered %s files", len(self.file_states))

        existing_paths = {state.path for state in self.file_states.values() if state.path.exists()}
        for path in list(self.file_states.keys()):
            if path not in existing_paths:
                self.file_states.pop(path, None)
                self._dirty_snapshot = True
        if not self.file_states:
            self._initializing = False

    def _process_file(self, state: FileState) -> None:
        try:
            current_stat = state.path.stat()
        except FileNotFoundError:
            self.file_states.pop(state.path, None)
            return

        if current_stat.st_size < state.position:
            state.position = 0

        try:
            with state.path.open("r", encoding="utf-8") as handle:
                handle.seek(state.position)
                while True:
                    start_pos = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        state.position = handle.tell()
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        handle.seek(start_pos)
                        break
                    state.position = handle.tell()
                    self._process_entry(state, data)
        except OSError:
            return

        state.size = current_stat.st_size
        self._dirty_snapshot = True
        if not state.scanned_initially:
            state.scanned_initially = True

    def _process_entry(self, state: FileState, data: Dict) -> None:
        session_id = data.get("sessionId") or state.session_id_hint or state.path.stem
        state.session_id_hint = session_id
        session = self.sessions.get(session_id)
        if session is None:
            session = SessionState(
                session_id=session_id,
                project=state.project,
                file_path=str(state.path),
            )
            if state.project in self.session_costs:
                session.cost_usd = self.session_costs[state.project]
            self.sessions[session_id] = session

        timestamp = parse_timestamp(data.get("timestamp"))
        if timestamp:
            if session.first_activity is None or timestamp < session.first_activity:
                session.first_activity = timestamp
            if session.last_activity is None or timestamp > session.last_activity:
                session.last_activity = timestamp
            self._record_activity(session_id, timestamp)

        if data.get("cwd"):
            session.cwd = data.get("cwd")

        session.last_event_type = data.get("type")
        message = data.get("message") if isinstance(data.get("message"), dict) else None
        if message:
            session.last_event_role = message.get("role")
            if message.get("model"):
                session.last_model = message.get("model")
            usage = message.get("usage")
            if isinstance(usage, dict) and timestamp:
                self._register_usage(session, usage, timestamp)
            self._handle_tool_use(session, message, timestamp)
            limit_ts = _extract_limit_timestamp(message, timestamp)
            if limit_ts:
                self.limit_candidates.add(limit_ts)
                self._last_limit_notice = limit_ts

        if data.get("type") == "user":
            self._handle_user_content(session, data.get("message"))

    def _register_usage(
        self, session: SessionState, usage: Dict[str, int], timestamp: datetime
    ) -> None:
        session.totals.add_usage(usage)
        local_dt = to_local(timestamp)
        day_key = local_dt.strftime("%Y-%m-%d")
        self.daily_totals[day_key].add_usage(usage)

    def _handle_tool_use(
        self, session: SessionState, message: Dict, timestamp: Optional[datetime]
    ) -> None:
        content = message.get("content")
        if not isinstance(content, list):
            return
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tool_use":
                tool_id = item.get("id")
                if tool_id:
                    session.pending_tools[tool_id] = PendingTool(
                        tool_id=tool_id,
                        name=item.get("name", "tool"),
                        started_at=timestamp,
                        description=_summarize_tool_input(item.get("input")),
                    )
                session.awaiting_approval = False
                session.awaiting_message = None

    def _handle_user_content(self, session: SessionState, message: Optional[Dict]) -> None:
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if not isinstance(content, list):
            return
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "tool_result":
                continue
            tool_id = item.get("tool_use_id")
            if tool_id and tool_id in session.pending_tools:
                session.pending_tools.pop(tool_id, None)
            text_blob = item.get("content")
            is_error = bool(item.get("is_error"))
            if _looks_like_approval_request(text_blob):
                session.awaiting_approval = True
                session.awaiting_message = _as_text(text_blob)
            elif not is_error:
                session.awaiting_approval = False
                session.awaiting_message = None

    def _record_activity(self, session_id: str, timestamp: datetime) -> None:
        history = self._activity_history.setdefault(session_id, [])
        history.append(timestamp)
        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=max(self.config.session_duration_hours * 2, 6)
        )
        while history and history[0] < cutoff:
            history.pop(0)

    def _refresh_processes(self) -> None:
        assert self._process_monitor is not None
        snapshots = self._process_monitor.list_relevant()
        by_cwd = self._process_monitor.map_by_cwd(snapshots)

        latest_session_by_cwd: Dict[str, SessionState] = {}
        for session in self.sessions.values():
            if not session.cwd:
                continue
            current = latest_session_by_cwd.get(session.cwd)
            if current is None:
                latest_session_by_cwd[session.cwd] = session
            else:
                current_ts = current.last_activity.timestamp() if current.last_activity else 0
                session_ts = session.last_activity.timestamp() if session.last_activity else 0
                if session_ts >= current_ts:
                    latest_session_by_cwd[session.cwd] = session

        for session in self.sessions.values():
            session.processes.clear()

        for cwd, processes in by_cwd.items():
            session = latest_session_by_cwd.get(cwd)
            if not session:
                continue
            for snap in processes:
                session.processes.append(
                    ProcessInfo(
                        pid=snap.pid,
                        status=snap.status,
                        cmdline=list(snap.cmdline),
                    )
                )

    def _maybe_refresh_ccusage_async(self) -> None:
        if not self.config.enable_ccusage_prices:
            return
        if self._ccusage_thread and self._ccusage_thread.is_alive():
            return
        now_ts = time.time()
        if now_ts - self._last_price_refresh < self.config.ccusage_refresh_interval:
            return

        def worker() -> None:
            try:
                self._refresh_ccusage_prices_blocking()
            finally:
                self._ccusage_thread = None

        logger.debug("scheduling ccusage refresh thread")
        self._ccusage_thread = threading.Thread(target=worker, name="ccusage-refresh", daemon=True)
        self._ccusage_thread.start()

    def _refresh_ccusage_prices_blocking(self) -> None:
        if not self.config.enable_ccusage_prices:
            return
        logger.debug("refreshing ccusage prices…")
        with self._ccusage_lock:
            now_ts = time.time()
            if now_ts - self._last_price_refresh < self.config.ccusage_refresh_interval:
                return
            try:
                session_proc = subprocess.run(
                    ["ccusage", "session", "--json", "-O"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if session_proc.returncode == 0 and session_proc.stdout:
                    session_data = json.loads(session_proc.stdout)
                    sessions = session_data.get("sessions", [])
                    mapping: Dict[str, float] = {}
                    for item in sessions:
                        key = str(item.get("sessionId"))
                        if not key:
                            continue
                        mapping[key] = float(item.get("totalCost", 0.0))
                    self.session_costs = mapping
                    for session in self.sessions.values():
                        if session.project in mapping:
                            session.cost_usd = mapping[session.project]

                daily_proc = subprocess.run(
                    ["ccusage", "daily", "--json", "-O"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if daily_proc.returncode == 0 and daily_proc.stdout:
                    daily_data = json.loads(daily_proc.stdout)
                    daily_entries = daily_data.get("daily", [])
                    costs: Dict[str, float] = {}
                    totals_map: Dict[str, UsageTotals] = {}
                    for item in daily_entries:
                        date_str = item.get("date")
                        if not date_str:
                            continue
                        total_cost = float(item.get("totalCost", 0.0))
                        costs[date_str] = total_cost
                        totals = UsageTotals()
                        totals.input_tokens = int(item.get("inputTokens", 0))
                        totals.output_tokens = int(item.get("outputTokens", 0))
                        totals.cache_creation_tokens = int(item.get("cacheCreationTokens", 0))
                        totals.cache_read_tokens = int(item.get("cacheReadTokens", 0))
                        totals_map[date_str] = totals
                    if totals_map:
                        self.daily_totals.update(totals_map)
                    self.daily_costs = costs

                self._persist_session_cache()
                self._last_price_refresh = time.time()
                logger.debug(
                    "ccusage refresh complete: sessions=%s daily=%s",
                    len(self.session_costs),
                    len(self.daily_costs),
                )
            except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
                logger.debug("ccusage refresh failed: %s", exc, exc_info=bool(DEBUG_MODE))
                return

    def _determine_status(
        self, session: SessionState, now: datetime
    ) -> SessionStatus:
        if session.awaiting_approval:
            return SessionStatus.WAITING

        last_activity = session.last_activity
        recent = False
        if last_activity is not None:
            recent = (now - last_activity).total_seconds() <= self.config.idle_seconds

        if session.processes and any(proc.status == "running" for proc in session.processes):
            return SessionStatus.RUNNING
        if session.pending_tools and recent:
            return SessionStatus.RUNNING
        if recent:
            return SessionStatus.RUNNING
        return SessionStatus.IDLE

    def _compute_window_info(self) -> WindowInfo:
        duration = timedelta(hours=max(self.config.session_duration_hours, 1))
        now = datetime.now(timezone.utc)
        cutoff = now - duration

        recent_events: List[datetime] = []
        all_events: List[datetime] = []

        for session_id, history in self._activity_history.items():
            if not history:
                continue
            for ts in history:
                if ts is None:
                    continue
                all_events.append(ts)
                if ts >= cutoff:
                    recent_events.append(ts)

        if not all_events:
            for session in self.sessions.values():
                if session.last_activity:
                    all_events.append(session.last_activity)
                    if session.last_activity >= cutoff:
                        recent_events.append(session.last_activity)
                elif session.first_activity and session.first_activity >= cutoff:
                    recent_events.append(session.first_activity)

        active_start = None
        active_end = None
        last_start = None
        last_end = None

        if recent_events:
            recent_events.sort()
            active_start = recent_events[0]
            active_end = active_start + duration

        if all_events:
            all_events.sort()
            last_start = all_events[-1]
            last_end = last_start + duration
            if active_start is None and last_start >= cutoff and now < last_end:
                active_start = last_start
                active_end = last_end

        return WindowInfo(
            active_start=active_start,
            active_end=active_end,
            last_start=last_start,
            last_end=last_end,
            duration=duration,
        )

    def _resolve_limit_info(self) -> LimitInfo:
        now = datetime.now(timezone.utc)
        if self.config.limit_reset_override:
            return LimitInfo(
                timestamp=self.config.limit_reset_override,
                source="config override",
                reached=False,
            )

        valid_candidates: List[datetime] = []
        for ts in list(self.limit_candidates):
            if ts is None:
                continue
            if ts < now - timedelta(hours=12):
                self.limit_candidates.discard(ts)
                continue
            valid_candidates.append(ts)
        valid_candidates.sort()
        self.limit_candidates = set(valid_candidates)

        if self._last_limit_notice and self._last_limit_notice < now - timedelta(hours=12):
            self._last_limit_notice = None

        future_candidates = [ts for ts in valid_candidates if ts >= now]
        if future_candidates:
            target = future_candidates[0]
            reached = now < target
            return LimitInfo(timestamp=target, source="recent limit notice", reached=reached)

        if valid_candidates:
            target = valid_candidates[-1]
            reached = self._last_limit_notice is not None and now < target
            return LimitInfo(timestamp=target, source="latest limit notice", reached=reached)

        fallback = _next_monthly_reset(now)
        return LimitInfo(timestamp=fallback, source="assumed monthly reset", reached=False)


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def _extract_project(path: Path) -> str:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == PROJECTS_SUBDIR and idx + 1 < len(parts):
            return parts[idx + 1]
    return path.stem


def _session_sort_key(summary: SessionSummary) -> tuple:
    status_weight = {
        SessionStatus.WAITING: 3,
        SessionStatus.RUNNING: 2,
        SessionStatus.IDLE: 1,
    }
    weight = status_weight.get(summary.status, 0)
    last_activity = summary.last_activity or datetime.fromtimestamp(0, tz=timezone.utc)
    return (weight, last_activity.timestamp(), summary.session_id)


def _summarize_tool_input(value: object) -> Optional[str]:
    if isinstance(value, dict):
        command = value.get("command")
        description = value.get("description")
        if command and description:
            return f"{command} — {description}"
        if command:
            return str(command)
        if description:
            return str(description)
    if isinstance(value, str):
        return value.strip()
    return None


def _looks_like_approval_request(text: object) -> bool:
    blob = _as_text(text).lower()
    if not blob:
        return False
    return any(pattern in blob for pattern in APPROVAL_PATTERNS)


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_as_text(item) for item in value)
    if isinstance(value, dict):
        parts = []
        for item in value.values():
            parts.append(_as_text(item))
        return "\n".join(part for part in parts if part)
    return ""


def _extract_limit_timestamp(message: Dict, reference: Optional[datetime] = None) -> Optional[datetime]:
    content = message.get("content")
    if not isinstance(content, list):
        return None

    reference = reference or datetime.now(timezone.utc)
    duration_hint = timedelta(hours=5)

    for item in content:
        if not isinstance(item, dict):
            continue
        text_value = item.get("text")
        if not isinstance(text_value, str):
            continue
        lower = text_value.lower()
        if "usage limit" in lower:
            match = re.search(r"\|(\d+)", text_value)
            if match:
                try:
                    stamp = int(match.group(1))
                    if stamp > 0:
                        return datetime.fromtimestamp(stamp, tz=timezone.utc)
                except ValueError:
                    pass
        reset_match = re.search(r"resets\s+([0-9]{1,2})(?::([0-9]{2}))?\s*(am|pm)", lower)
        if reset_match:
            hour = int(reset_match.group(1))
            minute = int(reset_match.group(2) or 0)
            meridiem = reset_match.group(3) or ""
            if meridiem == "am":
                hour = 0 if hour == 12 else hour
            elif meridiem == "pm":
                hour = hour if hour == 12 else hour + 12
            local_ref = to_local(reference)
            reset_local = local_ref.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if reset_local < local_ref:
                reset_local = reset_local + timedelta(days=1)
            return reset_local.astimezone(timezone.utc)
    return reference + duration_hint


def _next_monthly_reset(reference: datetime) -> datetime:
    local = reference.astimezone()
    if local.month == 12:
        target = local.replace(year=local.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        target = local.replace(month=local.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return target.astimezone(timezone.utc)


__all__ = ["UsageTracker"]
