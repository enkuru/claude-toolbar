from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import rumps

try:
    import AppKit
    from Foundation import NSBundle
except ImportError:  # pragma: no cover - macOS only integration
    AppKit = None
    NSBundle = None

from .config import CONFIG_PATH, ToolbarConfig, load_config, save_config
from .models import SessionStatus, SessionSummary, UsageSummary
from .paths import discover_claude_paths
from .usage_tracker import UsageTracker
from .utils import format_currency, format_relative, format_tokens, format_ts

STATUS_EMOJI = {
    SessionStatus.RUNNING: "🟢",
    SessionStatus.IDLE: "⚪️",
    SessionStatus.WAITING: "🟡",
}

GREEN_DOT = "🟢"
YELLOW_DOT = "🟡"
WHITE_DOT = "⚪️"
BLUE_DOT = "🔵"
ORANGE_DOT = "🟠"
MAX_SESSIONS = 20
ICON_PATH = Path(__file__).resolve().parent / "assets" / "claude_toolbar_icon.png"
MIN_LOADING_DISPLAY_SECONDS = 0.8
MAX_LOADING_DISPLAY_SECONDS = 6.0
FAST_REFRESH_INTERVAL = 0.5


def _summary_sort_key(summary: SessionSummary) -> tuple[int, float]:
    weight_map = {
        SessionStatus.WAITING: 3,
        SessionStatus.RUNNING: 2,
        SessionStatus.IDLE: 1,
    }
    weight = weight_map.get(summary.status, 0)
    last_ts = summary.last_activity.timestamp() if summary.last_activity else 0.0
    return weight, last_ts


def _load_icon_path() -> Optional[str]:
    if ICON_PATH.exists():
        return str(ICON_PATH)
    return None


class ClaudeToolbarApp(rumps.App):
    def __init__(self, config: Optional[ToolbarConfig] = None):
        self.config = config or load_config()
        claude_paths = discover_claude_paths(self.config.claude_paths)
        self.tracker = UsageTracker(self.config, claude_paths)

        super().__init__("", icon=_load_icon_path(), quit_button=None)

        self.usage_today_item = rumps.MenuItem("Today: …")
        self.usage_week_item = rumps.MenuItem("Last 7 days: …")
        self.usage_month_item = rumps.MenuItem("This month: …")
        self.limit_item = rumps.MenuItem("Limit reset: …")
        self.window_item = rumps.MenuItem("5h window: —")

        self.refresh_item = rumps.MenuItem("Refresh Now", callback=self.refresh_now)
        self.open_config_item = rumps.MenuItem("Open Config", callback=self.open_config)
        self.quit_item = rumps.MenuItem("Quit", callback=rumps.quit_application)

        self.menu = []
        self.session_lookup: Dict[str, SessionSummary] = {}
        self.loading_sessions_item = rumps.MenuItem("Loading sessions…", callback=None)

        self._loading_frames = ["⏳ Claude", "⌛ Claude"]
        self._loading_frame_index = 0
        self._loading_anim_timer: Optional[rumps.Timer] = None
        self._pending_usage_summary: Optional[UsageSummary] = None
        self._pending_sessions: Optional[List[SessionSummary]] = None
        self._loading_started_at = time.monotonic()
        self._fast_refresh_timer: Optional[rumps.Timer] = None
        self._in_fast_refresh = False
        self._active_sessions_cache: List[str] = []

        self._render_loading_state()

        self._refresh_interval = self.config.refresh_interval
        self.refresh_timer = rumps.Timer(self.refresh_timer_tick, self._refresh_interval)
        self.refresh_timer.start()
        self._initial_timer = rumps.Timer(self._initial_refresh, 0.1)
        self._initial_timer.start()

    # ------------------------------------------------------------------
    # Menu rendering
    # ------------------------------------------------------------------
    def refresh_timer_tick(self, _):
        if self._loading_anim_timer is not None or self.tracker.is_initializing():
            if self._should_show_loading():
                self._render_loading_state()

        self.tracker.update()

        if self.tracker.is_initializing() or self._should_show_loading():
            self._set_refresh_interval(min(self.config.refresh_interval, FAST_REFRESH_INTERVAL))
        else:
            self._set_refresh_interval(self.config.refresh_interval)
            self._ensure_fast_refresh_timer()
            self._ensure_fast_refresh_timer()

        usage_summary = self.tracker.get_usage_summary()
        raw_sessions = self.tracker.get_session_summaries()
        grouped_sessions = self._group_sessions(raw_sessions)

        self._pending_usage_summary = usage_summary
        self._pending_sessions = grouped_sessions

        if self._should_show_loading(grouped_sessions):
            return

        self._stop_loading_animation()
        self._render_pending_menu()

    def _render_pending_menu(self) -> None:
        usage_summary = self._pending_usage_summary or self.tracker.get_usage_summary()
        sessions = self._pending_sessions or self._group_sessions(self.tracker.get_session_summaries())
        self._pending_usage_summary = None
        self._pending_sessions = None
        self._render_menu(usage_summary, sessions)
        if not self._in_fast_refresh:
            self._toggle_fast_refresh_timer(False)

    def _should_show_loading(self, sessions: Optional[List[SessionSummary]] = None) -> bool:
        if self.tracker.is_initializing():
            return True
        elapsed = time.monotonic() - self._loading_started_at
        if elapsed < MIN_LOADING_DISPLAY_SECONDS:
            return True
        if sessions is None:
            sessions = self._pending_sessions
        ready = True
        if sessions:
            visible = sessions[:MAX_SESSIONS]
            ready = self.tracker.sessions_ready(summary.session_id for summary in visible)
        if not ready:
            return elapsed < MAX_LOADING_DISPLAY_SECONDS
        return False

    def _start_loading_animation(self) -> None:
        if self._loading_anim_timer is None:
            self._loading_frame_index = 0
            self._loading_started_at = time.monotonic()
            self._loading_anim_timer = rumps.Timer(self._tick_loading_animation, 0.35)
            self._loading_anim_timer.start()
        self._update_loading_titles()

    def _stop_loading_animation(self) -> None:
        if self._loading_anim_timer is not None:
            self._loading_anim_timer.stop()
            self._loading_anim_timer = None

    def _set_refresh_interval(self, interval: float) -> None:
        if abs(interval - self._refresh_interval) < 1e-6:
            return
        self._refresh_interval = interval
        self.refresh_timer.stop()
        self.refresh_timer.interval = interval
        self.refresh_timer.start()

    def _tick_loading_animation(self, _):
        if not self._should_show_loading():
            self._stop_loading_animation()
            self._render_pending_menu()
            return
        self._loading_frame_index = (self._loading_frame_index + 1) % len(self._loading_frames)
        self._update_loading_titles()

    def _update_loading_titles(self) -> None:
        suffix = self._loading_suffix()
        self.title = self._loading_frames[self._loading_frame_index]
        self.usage_today_item.title = f"Today: loading{suffix}"
        self.usage_week_item.title = f"Last 7 days: loading{suffix}"
        self.usage_month_item.title = f"This month: loading{suffix}"
        self.limit_item.title = f"Limit reset: loading{suffix}"
        self.window_item.title = f"Loading session data{suffix}"
        self.loading_sessions_item.title = f"Loading sessions{suffix}"

    def _loading_suffix(self) -> str:
        dots = (self._loading_frame_index % 3) + 1
        return "." * dots

    def _initial_refresh(self, timer: rumps.Timer) -> None:
        timer.stop()
        self.refresh_timer_tick(None)

    def _render_loading_state(self) -> None:
        self._start_loading_animation()
        self._update_loading_titles()
        self.menu.clear()
        self.menu.add(self.usage_today_item)
        self.menu.add(self.usage_week_item)
        self.menu.add(self.usage_month_item)
        self.menu.add(rumps.separator)
        self.menu.add(self.limit_item)
        self.menu.add(self.window_item)
        self.menu.add(rumps.separator)
        self.menu.add(self.loading_sessions_item)
        self.menu.add(rumps.separator)
        self.menu.add(self.refresh_item)
        self.menu.add(self.open_config_item)
        self.menu.add(self.quit_item)

    def _group_sessions(self, sessions: List[SessionSummary]) -> List[SessionSummary]:
        grouped: Dict[str, SessionSummary] = {}
        for summary in sessions:
            key = summary.project or summary.session_id
            current = grouped.get(key)
            if current is None or _summary_sort_key(summary) > _summary_sort_key(current):
                grouped[key] = summary
        ordered = sorted(grouped.values(), key=_summary_sort_key, reverse=True)
        return ordered

    def _render_menu(
        self, usage_summary: UsageSummary, sessions: List[SessionSummary]
    ) -> None:
        if self.tracker.is_initializing():
            self._render_loading_state()
            return
        self.menu.clear()

        self._update_usage_section(usage_summary, sessions)
        self.menu.add(self.usage_today_item)
        self.menu.add(self.usage_week_item)
        self.menu.add(self.usage_month_item)
        self.menu.add(rumps.separator)
        self.menu.add(self.limit_item)
        self.menu.add(self.window_item)
        self.menu.add(rumps.separator)

        self._populate_sessions(sessions)
        self._ensure_fast_refresh_timer()

        self.menu.add(rumps.separator)
        self.menu.add(self.refresh_item)
        self.menu.add(self.open_config_item)
        self.menu.add(self.quit_item)

    def _update_usage_section(
        self, summary: UsageSummary, sessions: List[SessionSummary]
    ) -> None:
        if self.tracker.is_initializing():
            self.usage_today_item.title = "Today: loading…"
            self.usage_week_item.title = "Last 7 days: loading…"
            self.usage_month_item.title = "This month: loading…"
            self.limit_item.title = "Limit reset: loading…"
            self.window_item.title = "Loading session data…"
            self.title = "⏳"
            return
        self.usage_today_item.title = (
            f"Today: {format_tokens(summary.today.total_tokens)} tokens ({format_currency(summary.today_cost)})"
        )
        self.usage_week_item.title = (
            f"Last 7 days: {format_tokens(summary.seven_day.total_tokens)} tokens ({format_currency(summary.seven_day_cost)})"
        )
        self.usage_month_item.title = (
            f"This month: {format_tokens(summary.month.total_tokens)} tokens ({format_currency(summary.month_cost)})"
        )

        window = summary.window_info
        if window.active_start and window.active_end:
            start_text = format_ts(window.active_start)
            end_text = format_ts(window.active_end)
            remaining = format_relative(window.active_end)
            self.window_item.title = f"5h window: {start_text} → {end_text} (ends {remaining})"
        elif window.last_end:
            end_text = format_ts(window.last_end)
            since = format_relative(window.last_end)
            self.window_item.title = f"No active 5h window (last ended {end_text}, {since})"
        else:
            self.window_item.title = "No active 5h window today"

        limit_ts = summary.limit_info.timestamp
        prefix = "Limit reset"
        if limit_ts is None and window.active_end:
            limit_ts = window.active_end
        elif summary.limit_info.timestamp is not None:
            prefix = "Limit reached" if summary.limit_info.reached else "Limit reset"

        if limit_ts:
            ts_text = format_ts(limit_ts)
            rel = format_relative(limit_ts)
            self.limit_item.title = f"{prefix}: {ts_text} ({rel})"
        else:
            self.limit_item.title = "Limit reset: unknown"

        working = []
        waiting = []
        running = []
        for summary in sessions:
            icon = _session_status_icon(summary, self.config.idle_seconds)
            if icon == ORANGE_DOT:
                waiting.append(summary)
            elif icon == BLUE_DOT:
                working.append(summary)
            elif icon == GREEN_DOT:
                running.append(summary)

        active = waiting + working + running
        if len(active) == 1:
            first = active[0]
            icon = _session_status_icon(first, self.config.idle_seconds)
            text = _session_status_text(first, self.config.idle_seconds)
            self.title = f"{icon} {text}"
        elif active:
            parts: List[str] = []
            if waiting:
                parts.append(f"{ORANGE_DOT}{len(waiting)}")
            if working:
                parts.append(f"{BLUE_DOT}{len(working)}")
            if running:
                parts.append(f"{GREEN_DOT}{len(running)}")
            self.title = " ".join(parts)
        else:
            self.title = f"{WHITE_DOT} Idle"

    def _populate_sessions(self, sessions: List[SessionSummary]) -> None:
        self.session_lookup.clear()
        if not sessions:
            message = "Loading sessions…" if self.tracker.is_initializing() else "No recent sessions"
            empty_item = rumps.MenuItem(message, callback=None)
            self.menu.add(empty_item)
            self._active_sessions_cache = []
            self._toggle_fast_refresh_timer(False)
            return

        limited = sessions[:MAX_SESSIONS]
        if not limited:
            message = "Loading sessions…" if self.tracker.is_initializing() else "No recent sessions"
            empty_item = rumps.MenuItem(message, callback=None)
            self.menu.add(empty_item)
            self._active_sessions_cache = []
            self._toggle_fast_refresh_timer(False)
            return

        buckets = self._bucket_sessions(limited, self.config.idle_seconds)
        active_ids: List[str] = []
        for bucket, entries in buckets:
            if not entries:
                continue
            header = rumps.MenuItem(bucket, callback=None)
            self.menu.add(header)
            for summary in entries:
                label = self._session_label(summary)
                item = rumps.MenuItem(f"  {label}", callback=self._on_session_clicked)
                item._session_id = summary.session_id  # type: ignore[attr-defined]
                self.session_lookup[summary.session_id] = summary
                self.menu.add(item)
                if summary.processes:
                    active_ids.append(summary.session_id)
        self._active_sessions_cache = active_ids


    def _session_label(self, summary: SessionSummary) -> str:
        emoji = _session_status_icon(summary, self.config.idle_seconds)
        tokens = format_tokens(summary.totals.total_tokens)
        relative = (
            format_relative(summary.last_activity)
            if summary.last_activity
            else "unknown"
        )
        project_display = _pretty_project(summary)
        cost_text = format_currency(summary.cost_usd) if summary.cost_usd else "$0.00"
        status_text = _session_status_text(summary, self.config.idle_seconds)
        return f"{emoji} {project_display} — {tokens} tokens / {cost_text} — {status_text} — {relative}"

    def _bucket_sessions(
        self, sessions: List[SessionSummary], idle_seconds: int
    ) -> List[tuple[str, List[SessionSummary]]]:
        now_local = datetime.now().astimezone()
        today = now_local.date()
        buckets: Dict[str, List[SessionSummary]] = {
            "Today": [],
            "Yesterday": [],
            "Earlier this week": [],
            "Earlier this month": [],
            "Older": [],
            "No activity": [],
        }

        for summary in sessions:
            if _session_is_recent(summary, idle_seconds, 1.0):
                buckets["Today"].append(summary)
                continue

            last_activity = summary.last_activity
            if last_activity is None:
                buckets["No activity"].append(summary)
                continue
            local_dt = last_activity.astimezone()
            session_date = local_dt.date()
            delta_days = (today - session_date).days
            if delta_days <= 0:
                buckets["Today"].append(summary)
            elif delta_days == 1:
                buckets["Yesterday"].append(summary)
            elif 1 < delta_days <= 6:
                buckets["Earlier this week"].append(summary)
            elif session_date.month == today.month and session_date.year == today.year:
                buckets["Earlier this month"].append(summary)
            else:
                buckets["Older"].append(summary)

        ordered_labels = [
            "Today",
            "Yesterday",
            "Earlier this week",
            "Earlier this month",
            "Older",
            "No activity",
        ]
        return [(label, buckets[label]) for label in ordered_labels]

    def _ensure_fast_refresh_timer(self) -> None:
        if not self._active_sessions_cache or self.tracker.is_initializing():
            self._toggle_fast_refresh_timer(False)
            return
        self._toggle_fast_refresh_timer(True)

    def _toggle_fast_refresh_timer(self, enabled: bool) -> None:
        if not enabled:
            if self._fast_refresh_timer is not None:
                self._fast_refresh_timer.stop()
                self._fast_refresh_timer = None
            return
        if self._fast_refresh_timer is None:
            self._fast_refresh_timer = rumps.Timer(self._fast_refresh_tick, 1.0)
            self._fast_refresh_timer.start()

    def _fast_refresh_tick(self, _):
        if self.tracker.is_initializing() or not self._active_sessions_cache:
            self._toggle_fast_refresh_timer(False)
            return
        self.tracker.refresh_active_processes(self._active_sessions_cache)
        usage_summary = self.tracker.get_usage_summary()
        grouped_sessions = self._group_sessions(self.tracker.get_session_summaries())
        self._pending_usage_summary = usage_summary
        self._pending_sessions = grouped_sessions
        self._in_fast_refresh = True
        try:
            self._render_pending_menu()
        finally:
            self._in_fast_refresh = False

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def refresh_now(self, _):
        self._stop_loading_animation()
        self._pending_usage_summary = None
        self._pending_sessions = None
        self._render_loading_state()
        self.refresh_timer_tick(None)

    def open_config(self, _):
        save_config(self.config)
        subprocess.run([
            "open",
            str(CONFIG_PATH.parent),
        ], check=False)

    def _on_session_clicked(self, sender: rumps.MenuItem):
        session_id = getattr(sender, "_session_id", None)
        summary = self.session_lookup.get(session_id)
        if not summary:
            return
        details = _format_session_details(summary)
        rumps.alert(title="Claude Session", message=details)


def _pretty_project(summary: SessionSummary) -> str:
    if summary.cwd:
        return Path(summary.cwd).name
    return summary.project


def _format_session_details(summary: SessionSummary) -> str:
    tokens = summary.totals
    totals_text = (
        f"Tokens: {format_tokens(tokens.total_tokens)}\n"
        f"  input: {tokens.input_tokens:,}\n"
        f"  output: {tokens.output_tokens:,}\n"
        f"  cache create: {tokens.cache_creation_tokens:,}\n"
        f"  cache read: {tokens.cache_read_tokens:,}"
    )
    cost_text = format_currency(summary.cost_usd)
    last_activity = (
        format_ts(summary.last_activity) if summary.last_activity else "Unknown"
    )
    process_lines = []
    for proc in summary.processes:
        process_lines.append(f"PID {proc.pid} — {proc.status}")
    processes_text = "\n".join(process_lines) if process_lines else "None"

    awaiting = summary.awaiting_message or "None"
    session_id_text = summary.session_id

    return (
        f"Session ID: {session_id_text}\n"
        f"Project: {_pretty_project(summary)}\n"
        f"Model: {summary.last_model or 'Unknown'}\n"
        f"Last activity: {last_activity} ({format_relative(summary.last_activity)})\n"
        f"Cost: {cost_text}\n\n"
        f"{totals_text}\n\n"
        f"Processes: {processes_text}\n"
        f"Waiting message: {awaiting}"
    )


def _session_is_recent(summary: SessionSummary, idle_seconds: int, multiplier: float = 1.0) -> bool:
    if summary.processes:
        return True
    if summary.last_activity is None:
        return False
    threshold = max(idle_seconds, 60) * max(multiplier, 1.0)
    now = datetime.now(timezone.utc)
    delta = now - summary.last_activity.astimezone(timezone.utc)
    return delta.total_seconds() <= threshold


def _session_status_icon(summary: SessionSummary, idle_seconds: int) -> str:
    if summary.awaiting_approval or summary.awaiting_message:
        return ORANGE_DOT
    if summary.pending_tool_count and _session_is_recent(summary, idle_seconds, 2.0):
        return BLUE_DOT
    if summary.status == SessionStatus.RUNNING and _session_is_recent(summary, idle_seconds, 1.0):
        return GREEN_DOT
    if summary.status == SessionStatus.WAITING and _session_is_recent(summary, idle_seconds, 1.0):
        return YELLOW_DOT
    return WHITE_DOT


def _session_status_text(summary: SessionSummary, idle_seconds: int) -> str:
    if summary.awaiting_approval or summary.awaiting_message:
        return "Awaiting approval"
    if summary.pending_tool_count:
        if _session_is_recent(summary, idle_seconds, 2.0):
            count = summary.pending_tool_count
            return "Running tool" if count == 1 else f"Running {count} tools"
        return "Idle"
    if summary.status == SessionStatus.RUNNING and _session_is_recent(summary, idle_seconds, 1.0):
        return "Running"
    if summary.status == SessionStatus.WAITING and _session_is_recent(summary, idle_seconds, 1.0):
        return "Waiting"
    return "Idle"


def main() -> None:
    if AppKit is not None and NSBundle is not None:
        info = NSBundle.mainBundle().infoDictionary()
        if info is not None:
            info["LSUIElement"] = "1"
        ns_app = AppKit.NSApplication.sharedApplication()
        ns_app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    elif AppKit is not None:
        ns_app = AppKit.NSApplication.sharedApplication()
        ns_app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    app = ClaudeToolbarApp()
    app.run()


__all__ = ["main", "ClaudeToolbarApp"]
