from __future__ import annotations

import json
import re
import subprocess
import time
from collections import defaultdict, OrderedDict
import shutil
import logging
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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

import calendar

MONTH_PATTERN = re.compile(r"(\d{4}-\d{2})")


def _normalize_daily_key(raw_key: Optional[str]) -> Optional[str]:
    """Convert ccusage keys into YYYY-MM-DD strings."""
    if not raw_key:
        return None
    key = raw_key.strip()
    if not key:
        return None
    if MONTH_PATTERN.fullmatch(key):
        try:
            month_start = date.fromisoformat(f"{key}-01")
        except ValueError:
            return None
        return month_start.strftime("%Y-%m-%d")
    try:
        day = date.fromisoformat(key)
    except ValueError:
        iso_value = key
        if iso_value.endswith("Z"):
            iso_value = iso_value[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(iso_value)
        except ValueError:
            parsed = parse_timestamp(key)
            if parsed is None:
                return None
            dt = parsed
        if isinstance(dt, datetime):
            return dt.date().strftime("%Y-%m-%d")
        return None
    return day.strftime("%Y-%m-%d")


@dataclass
class AggregatedUsage:
    totals: UsageTotals
    cost: float = 0.0


def _safe_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("$", "").replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


def _safe_int(entry: Dict[str, object], *keys: str) -> int:
    for key in keys:
        if key in entry and entry[key] is not None:
            value = entry[key]
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                cleaned = value.strip().replace(",", "")
                try:
                    return int(float(cleaned))
                except ValueError:
                    continue
    return 0


def _usage_totals_from_entry(entry: Dict[str, object]) -> UsageTotals:
    totals = UsageTotals()
    totals.input_tokens = _safe_int(entry, 'inputTokens', 'input')
    totals.output_tokens = _safe_int(entry, 'outputTokens', 'output')
    totals.cache_creation_tokens = _safe_int(entry, 'cacheCreationTokens', 'cacheCreation')
    totals.cache_read_tokens = _safe_int(entry, 'cacheReadTokens', 'cacheRead')
    if totals.total_tokens == 0:
        total_tokens = _safe_int(entry, 'totalTokens', 'tokens')
        if total_tokens:
            totals.input_tokens = total_tokens
    return totals


def _clone_totals(totals: UsageTotals) -> UsageTotals:
    return UsageTotals(
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        cache_creation_tokens=totals.cache_creation_tokens,
        cache_read_tokens=totals.cache_read_tokens,
    )


def _extract_entry_date(entry: Dict[str, object], candidate_keys: List[str]) -> Optional[date]:
    for key in candidate_keys:
        value = entry.get(key)
        normalized = _normalize_daily_key(value) if isinstance(value, str) else None
        if normalized:
            try:
                return date.fromisoformat(normalized)
            except ValueError:
                continue
    return None


def _extract_period(entry: Dict[str, object]) -> tuple[Optional[date], Optional[date]]:
    start = _extract_entry_date(
        entry,
        ['start', 'startDate', 'rangeStart', 'periodStart', 'weekStart', 'week', 'monthStart'],
    )
    end = _extract_entry_date(
        entry,
        ['end', 'endDate', 'rangeEnd', 'periodEnd', 'weekEnd', 'monthEnd'],
    )
    month_key = entry.get('month')
    if start and not end:
        if isinstance(month_key, str) or entry.get('monthStart'):
            last_day = calendar.monthrange(start.year, start.month)[1]
            end = date(start.year, start.month, last_day)
        elif entry.get('week') or entry.get('weekStart') or entry.get('weekEnd'):
            end = start + timedelta(days=6)
    if end and not start:
        start = end
    if start and end and end < start:
        start, end = end, start
    return start, end


def _extract_month_key(entry: Dict[str, object]) -> Optional[str]:
    for key in ('month', 'monthStart', 'period', 'label', 'name'):
        value = entry.get(key)
        if isinstance(value, str):
            match = MONTH_PATTERN.search(value)
            if match:
                return match.group(1)
    return None


def _entry_covers_date(entry: Dict[str, object], target: date) -> bool:
    start, end = _extract_period(entry)
    if start and end:
        return start <= target <= end
    if start:
        return start <= target
    if end:
        return target <= end
    month_key = _extract_month_key(entry)
    if month_key:
        try:
            month_start = date.fromisoformat(f"{month_key}-01")
        except ValueError:
            return False
        last_day = calendar.monthrange(month_start.year, month_start.month)[1]
        month_end = date(month_start.year, month_start.month, last_day)
        return month_start <= target <= month_end
    return False


def _pick_latest_entry(entries: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not entries:
        return None
    best_entry = None
    best_end: Optional[date] = None
    for entry in entries:
        end_date = _extract_entry_date(
            entry,
            ['end', 'endDate', 'rangeEnd', 'periodEnd', 'weekEnd', 'monthEnd', 'date'],
        )
        if end_date is None:
            end_date = _extract_entry_date(
                entry,
                ['start', 'startDate', 'rangeStart', 'periodStart', 'weekStart', 'week', 'monthStart'],
            )
        if end_date is None:
            continue
        if best_end is None or end_date > best_end:
            best_end = end_date
            best_entry = entry
    return best_entry or entries[0]


def _candidate_ccusage_paths(explicit: Optional[str] = None) -> List[Path]:
    home = Path.home()
    direct_env = os.environ.get("CCUSAGE_PATH")
    resolved = shutil.which("ccusage")
    candidates = [
        explicit,
        direct_env,
        resolved,
        "/usr/local/bin/ccusage",
        "/opt/homebrew/bin/ccusage",
        str(home / "go" / "bin" / "ccusage"),
        str(home / ".local" / "bin" / "ccusage"),
    ]
    unique: List[Path] = []
    seen = set()
    for item in candidates:
        if not item:
            continue
        try:
            path = Path(item).expanduser()
        except OSError:
            continue
        norm = str(path.resolve()) if path.exists() else str(path)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(path)
    return unique

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
        self._limit_override: Optional[datetime] = None
        self._ccusage_rollups: Dict[str, AggregatedUsage] = {}
        self._ccusage_path: Optional[str] = (
            str(config.ccusage_path) if config.ccusage_path else None
        )
        self._ccusage_warned: bool = False
        self._ccusage_logged: bool = False
        self._load_snapshots()
        self._load_session_cache()
        if self.file_states:
            self._initializing = len(self.sessions) == 0
        logger.debug(
            "UsageTracker initialized: sessions=%s, files=%s, initializing=%s",
            len(self.sessions),
            len(self.file_states),
            self._initializing,
        )
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

        normalized_totals: Dict[str, UsageTotals] = {}
        for raw_key, totals in list(self.daily_totals.items()):
            normalized_key = _normalize_daily_key(raw_key)
            if not normalized_key:
                continue
            bucket = normalized_totals.get(normalized_key)
            if bucket is None:
                bucket = UsageTotals()
                normalized_totals[normalized_key] = bucket
            bucket.merge(totals)
        self.daily_totals = defaultdict(UsageTotals, normalized_totals)

        normalized_costs: Dict[str, float] = {}
        for raw_key, cost in list(self.daily_costs.items()):
            normalized_key = _normalize_daily_key(raw_key)
            if not normalized_key:
                continue
            normalized_costs[normalized_key] = normalized_costs.get(normalized_key, 0.0) + cost
        self.daily_costs = defaultdict(float, normalized_costs)

        today_roll = self._ccusage_rollups.get("today")
        seven_day_roll = self._ccusage_rollups.get("seven_day")
        month_roll = self._ccusage_rollups.get("month")
        last_month_roll = self._ccusage_rollups.get("last_month")
        all_time_roll = self._ccusage_rollups.get("all_time")

        today_totals = _clone_totals(today_roll.totals) if today_roll else UsageTotals()
        seven_day_totals = _clone_totals(seven_day_roll.totals) if seven_day_roll else UsageTotals()
        month_totals = _clone_totals(month_roll.totals) if month_roll else UsageTotals()
        last_month_totals = _clone_totals(last_month_roll.totals) if last_month_roll else UsageTotals()
        all_time_totals = _clone_totals(all_time_roll.totals) if all_time_roll else UsageTotals()

        today_cost = today_roll.cost if today_roll else None
        seven_day_cost = seven_day_roll.cost if seven_day_roll else None
        month_cost = month_roll.cost if month_roll else None
        last_month_cost = last_month_roll.cost if last_month_roll else None
        all_time_cost = all_time_roll.cost if all_time_roll else None

        fallback_today = UsageTotals()
        fallback_seven = UsageTotals()
        fallback_month = UsageTotals()
        fallback_last_month = UsageTotals()
        fallback_all = UsageTotals()

        prev_month_start = (month_start - timedelta(days=1)).replace(day=1)

        for day_key, totals in self.daily_totals.items():
            try:
                day = date.fromisoformat(day_key)
            except ValueError:
                continue
            fallback_all.merge(totals)
            if day == today_local:
                fallback_today.merge(totals)
            if seven_days_ago <= day <= today_local:
                fallback_seven.merge(totals)
            if day >= month_start:
                fallback_month.merge(totals)
            if prev_month_start <= day < month_start:
                fallback_last_month.merge(totals)

        if today_totals.total_tokens == 0 and fallback_today.total_tokens:
            today_totals = _clone_totals(fallback_today)
        if seven_day_totals.total_tokens == 0 and fallback_seven.total_tokens:
            seven_day_totals = _clone_totals(fallback_seven)
        if month_totals.total_tokens == 0 and fallback_month.total_tokens:
            month_totals = _clone_totals(fallback_month)
        if last_month_totals.total_tokens == 0 and fallback_last_month.total_tokens:
            last_month_totals = _clone_totals(fallback_last_month)
        if all_time_totals.total_tokens == 0 and fallback_all.total_tokens:
            all_time_totals = _clone_totals(fallback_all)

        def _sum_range(predicate) -> Optional[float]:
            total = 0.0
            found = False
            for day_key, cost in self.daily_costs.items():
                try:
                    day = date.fromisoformat(day_key)
                except ValueError:
                    continue
                if predicate(day):
                    total += cost
                    found = True
            return total if found else None

        if today_cost is None:
            today_cost = _sum_range(lambda d: d == today_local)
        if seven_day_cost is None:
            seven_day_cost = _sum_range(lambda d: seven_days_ago <= d <= today_local)
        if month_cost is None:
            month_cost = _sum_range(lambda d: d >= month_start)
        if last_month_cost is None:
            last_month_cost = _sum_range(lambda d: prev_month_start <= d < month_start)
        if all_time_cost is None:
            all_time_cost = _sum_range(lambda _d: True)

        today_cost = today_cost or 0.0
        seven_day_cost = seven_day_cost or 0.0
        month_cost = month_cost or 0.0
        last_month_cost = last_month_cost or 0.0
        all_time_cost = all_time_cost or 0.0

        limit_info = self._resolve_limit_info()
        window_info = self._compute_window_info()

        return UsageSummary(
            today=today_totals,
            seven_day=seven_day_totals,
            month=month_totals,
            last_month=last_month_totals,
            all_time=all_time_totals,
            total_sessions=len(self.sessions),
            limit_info=limit_info,
            window_info=window_info,
            today_cost=today_cost,
            seven_day_cost=seven_day_cost,
            month_cost=month_cost,
            last_month_cost=last_month_cost,
            all_time_cost=all_time_cost,
        )

    def refresh_active_processes(self, session_ids: Iterable[str]) -> None:
        if not session_ids:
            return
        if self._process_monitor:
            self._refresh_processes()

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
            cached_cost = self.session_costs.get(session_id)
            if cached_cost is None and state.project:
                cached_cost = self.session_costs.get(state.project)
            if cached_cost is not None:
                session.cost_usd = cached_cost
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
        limit_detected = False
        if message:
            session.last_event_role = message.get("role")
            if message.get("model"):
                session.last_model = message.get("model")
            usage = message.get("usage")
            if isinstance(usage, dict) and timestamp:
                self._register_usage(session, usage, timestamp)
            self._handle_tool_use(session, message, timestamp)
            limit_ts, limit_matched = _extract_limit_timestamp(message, timestamp)
            if limit_matched and limit_ts:
                self.limit_candidates.add(limit_ts)
                self._last_limit_notice = timestamp
                self._limit_override = limit_ts
                session.limit_blocked = True
                session.limit_reset_at = limit_ts
                limit_detected = True

        if data.get("type") == "user":
            self._handle_user_content(session, data.get("message"))

        if message and not limit_detected and session.limit_blocked:
            session.limit_blocked = False
            session.limit_reset_at = None

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

    def _resolve_ccusage_path(self) -> Optional[str]:
        if self._ccusage_path:
            candidate = Path(self._ccusage_path)
            if candidate.exists() and os.access(candidate, os.X_OK):
                return self._ccusage_path
            self._ccusage_path = None
        explicit = None
        if self.config.ccusage_path:
            explicit = str(self.config.ccusage_path)
        for path in _candidate_ccusage_paths(explicit):
            try:
                if path.exists() and os.access(path, os.X_OK):
                    resolved = str(path.resolve())
                    self._ccusage_path = resolved
                    return resolved
            except OSError:
                continue
        if not self._ccusage_warned:
            logger.info(
                "ccusage binary not found; set CCUSAGE_PATH or add it to PATH before launching the toolbar"
            )
            self._ccusage_warned = True
        return None

    def _subprocess_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        if "PATH" not in env or not env["PATH"]:
            env["PATH"] = "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin"
        return env

    def _refresh_ccusage_prices_blocking(self) -> None:
        if not self.config.enable_ccusage_prices:
            return
        logger.debug("refreshing ccusage prices…")
        with self._ccusage_lock:
            now_ts = time.time()
            if now_ts - self._last_price_refresh < self.config.ccusage_refresh_interval:
                return
            self._ccusage_rollups = {}
            ccusage_bin = self._resolve_ccusage_path()
            if not ccusage_bin:
                return
            if not self._ccusage_logged:
                logger.info("using ccusage binary at %s", ccusage_bin)
                self._ccusage_logged = True
            try:
                session_proc = subprocess.run(
                    [ccusage_bin, "session", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=self._subprocess_env(),
                )
                if session_proc.returncode == 0 and session_proc.stdout:
                    session_data = json.loads(session_proc.stdout)
                    sessions = session_data.get("sessions", [])
                    by_session: Dict[str, float] = {}
                    by_project: Dict[str, float] = {}
                    for item in sessions:
                        cost = _safe_float(item.get("totalCost"))
                        if cost <= 0.0:
                            continue
                        session_key = item.get("sessionId")
                        if isinstance(session_key, str) and session_key:
                            by_session[session_key] = cost
                        for alt_key in (
                            "project",
                            "projectId",
                            "projectName",
                            "workspaceId",
                            "workspace",
                            "path",
                        ):
                            alt_value = item.get(alt_key)
                            if isinstance(alt_value, str) and alt_value:
                                by_project.setdefault(alt_value, cost)
                    combined_mapping = {**by_project, **by_session}
                    if combined_mapping:
                        self.session_costs = combined_mapping
                        for session in self.sessions.values():
                            cost = combined_mapping.get(session.session_id)
                            if cost is None and session.project:
                                cost = combined_mapping.get(session.project)
                            if cost is not None:
                                session.cost_usd = cost
                else:
                    logger.info(
                        "ccusage session command failed rc=%s", session_proc.returncode
                    )

                today_local = datetime.now().astimezone().date()

                daily_proc = subprocess.run(
                    [ccusage_bin, "daily", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=self._subprocess_env(),
                )
                if daily_proc.returncode == 0 and daily_proc.stdout:
                    daily_data = json.loads(daily_proc.stdout)
                    daily_entries = daily_data.get("daily", [])
                    costs: Dict[str, float] = {}
                    totals_map: Dict[str, UsageTotals] = {}
                    for item in daily_entries:
                        raw_key = item.get("date")
                        normalized_key = _normalize_daily_key(raw_key)
                        if not normalized_key:
                            continue
                        total_cost = _safe_float(item.get("totalCost") or item.get("cost"))
                        costs[normalized_key] = total_cost
                        totals_map[normalized_key] = _usage_totals_from_entry(item)
                    if totals_map:
                        self.daily_totals.update(totals_map)
                    if costs:
                        self.daily_costs = defaultdict(float, costs)
                    today_key = today_local.strftime("%Y-%m-%d")
                    today_totals = totals_map.get(today_key)
                    if today_totals:
                        self._ccusage_rollups["today"] = AggregatedUsage(
                            totals=_clone_totals(today_totals),
                            cost=costs.get(today_key, 0.0),
                        )
                else:
                    logger.info(
                        "ccusage daily command failed rc=%s", daily_proc.returncode
                    )

                weekly_proc = subprocess.run(
                    [ccusage_bin, "weekly", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=self._subprocess_env(),
                )
                if weekly_proc.returncode == 0 and weekly_proc.stdout:
                    weekly_data = json.loads(weekly_proc.stdout)
                    weekly_entries = (
                        weekly_data.get("weekly")
                        or weekly_data.get("weeks")
                        or weekly_data.get("data")
                        or []
                    )
                    target_week = None
                    for entry in weekly_entries:
                        if _entry_covers_date(entry, today_local):
                            target_week = entry
                            break
                    if target_week is None:
                        target_week = _pick_latest_entry(weekly_entries)
                    if target_week:
                        week_totals = _usage_totals_from_entry(target_week)
                        week_cost = _safe_float(
                            target_week.get("totalCost") or target_week.get("cost")
                        )
                        self._ccusage_rollups["seven_day"] = AggregatedUsage(
                            totals=_clone_totals(week_totals),
                            cost=week_cost,
                        )
                else:
                    logger.info(
                        "ccusage weekly command failed rc=%s", weekly_proc.returncode
                    )

                monthly_proc = subprocess.run(
                    [ccusage_bin, "monthly", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=self._subprocess_env(),
                )
                if monthly_proc.returncode == 0 and monthly_proc.stdout:
                    monthly_data = json.loads(monthly_proc.stdout)
                    monthly_entries = (
                        monthly_data.get("monthly")
                        or monthly_data.get("months")
                        or monthly_data.get("data")
                        or []
                    )
                    target_month = None
                    target_month_key = today_local.strftime("%Y-%m")
                    for entry in monthly_entries:
                        month_key = _extract_month_key(entry)
                        if month_key == target_month_key or _entry_covers_date(
                            entry, today_local
                        ):
                            target_month = entry
                            break
                    if target_month is None:
                        target_month = _pick_latest_entry(monthly_entries)
                    if target_month:
                        month_totals = _usage_totals_from_entry(target_month)
                        month_cost = _safe_float(
                            target_month.get("totalCost") or target_month.get("cost")
                        )
                        self._ccusage_rollups["month"] = AggregatedUsage(
                            totals=_clone_totals(month_totals),
                            cost=month_cost,
                        )

                    prev_month_key = (
                        (today_local.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
                    )
                    last_month_entry = None
                    last_month_key = None
                    for entry in monthly_entries:
                        month_key = _extract_month_key(entry)
                        if not month_key:
                            continue
                        if month_key == prev_month_key:
                            last_month_entry = entry
                            last_month_key = month_key
                            break
                        if month_key < target_month_key:
                            if last_month_key is None or month_key > last_month_key:
                                last_month_entry = entry
                                last_month_key = month_key
                    if last_month_entry:
                        last_totals = _usage_totals_from_entry(last_month_entry)
                        last_cost = _safe_float(
                            last_month_entry.get("totalCost")
                            or last_month_entry.get("cost")
                        )
                        self._ccusage_rollups["last_month"] = AggregatedUsage(
                            totals=_clone_totals(last_totals),
                            cost=last_cost,
                        )

                    totals_payload = monthly_data.get("totals")
                    if isinstance(totals_payload, dict):
                        aggregate_totals = _usage_totals_from_entry(totals_payload)
                        aggregate_cost = _safe_float(totals_payload.get("totalCost"))
                        self._ccusage_rollups["all_time"] = AggregatedUsage(
                            totals=aggregate_totals,
                            cost=aggregate_cost,
                        )
                    elif monthly_entries:
                        aggregate_totals = UsageTotals()
                        aggregate_cost = 0.0
                        for entry in monthly_entries:
                            aggregate_totals.merge(_usage_totals_from_entry(entry))
                            aggregate_cost += _safe_float(
                                entry.get("totalCost") or entry.get("cost")
                            )
                        self._ccusage_rollups["all_time"] = AggregatedUsage(
                            totals=aggregate_totals,
                            cost=aggregate_cost,
                        )
                else:
                    logger.info(
                        "ccusage monthly command failed rc=%s", monthly_proc.returncode
                    )

                self._persist_session_cache()
                self._last_price_refresh = time.time()
                logger.debug(
                    "ccusage refresh complete: sessions=%s daily=%s rollups=%s",
                    len(self.session_costs),
                    len(self.daily_costs),
                    list(self._ccusage_rollups.keys()),
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
        if self._limit_override and now > self._limit_override + timedelta(hours=6):
            self._limit_override = None
        if self._limit_override is not None:
            reached = now < self._limit_override
            return LimitInfo(timestamp=self._limit_override, source="limit notice", reached=reached)
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
            reached = self._last_limit_notice is not None and self._last_limit_notice <= now
            return LimitInfo(timestamp=target, source="recent limit notice", reached=reached)

        if valid_candidates:
            target = valid_candidates[-1]
            reached = self._last_limit_notice is not None and self._last_limit_notice <= now
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


def _extract_limit_timestamp(
    message: Dict, reference: Optional[datetime] = None
) -> Tuple[Optional[datetime], bool]:
    content = message.get("content")
    if not isinstance(content, list):
        return None, False

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
                        return datetime.fromtimestamp(stamp, tz=timezone.utc), True
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
            return reset_local.astimezone(timezone.utc), True
    return None, False


def _next_monthly_reset(reference: datetime) -> datetime:
    local = reference.astimezone()
    if local.month == 12:
        target = local.replace(year=local.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        target = local.replace(month=local.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return target.astimezone(timezone.utc)


__all__ = ["UsageTracker"]
