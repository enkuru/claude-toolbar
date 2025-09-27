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

        if summary.limit_info.timestamp:
            ts_text = format_ts(summary.limit_info.timestamp)
            rel = format_relative(summary.limit_info.timestamp)
            self.limit_item.title = f"Limit reset: {ts_text} ({rel})"
        else:
            self.limit_item.title = "Limit reset: unknown"

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

        active_sessions = [
            summary
            for summary in sessions
            if summary.status in (SessionStatus.RUNNING, SessionStatus.WAITING)
        ]

        if len(active_sessions) == 1:
            first = active_sessions[0]
            status_span = STATUS_EMOJI.get(first.status, WHITE_DOT)
            project = _pretty_project(first)
            self.title = f"{status_span} {project}"
        elif len(active_sessions) > 1:
            running_count = sum(1 for s in active_sessions if s.status == SessionStatus.RUNNING)
            waiting_count = len(active_sessions) - running_count
            parts: List[str] = []
            if running_count:
                parts.append(f"{GREEN_DOT}{running_count}")
            if waiting_count:
                parts.append(f"{YELLOW_DOT}{waiting_count}")
            self.title = " ".join(parts)
        else:
            self.title = ""

    def _populate_sessions(self, sessions: List[SessionSummary]) -> None:
        self.session_lookup.clear()
        if not sessions:
            message = "Loading sessions…" if self.tracker.is_initializing() else "No recent sessions"
            empty_item = rumps.MenuItem(message, callback=None)
            self.menu.add(empty_item)
            return

        for summary in sessions[:MAX_SESSIONS]:
            label = self._session_label(summary)
            item = rumps.MenuItem(label, callback=self._on_session_clicked)
            item._session_id = summary.session_id  # type: ignore[attr-defined]
            self.session_lookup[summary.session_id] = summary
            self.menu.add(item)

    def _session_label(self, summary: SessionSummary) -> str:
        emoji = STATUS_EMOJI.get(summary.status, "•")
        tokens = format_tokens(summary.totals.total_tokens)
        relative = (
            format_relative(summary.last_activity)
            if summary.last_activity
            else "unknown"
        )
        project_display = _pretty_project(summary)
        cost_text = format_currency(summary.cost_usd) if summary.cost_usd else "$0.00"
        return f"{emoji} {project_display} — {tokens} tokens / {cost_text} — {relative}"

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
