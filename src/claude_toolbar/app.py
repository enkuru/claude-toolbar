from __future__ import annotations

import subprocess
import time
from datetime import datetime, timedelta, timezone
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
from .pty_manager import PtyManager, SessionLaunchInfo
from .scheduler import ScheduledJob, SessionScheduler
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

        self.pty_manager = PtyManager(self.config.launch_command)
        self.scheduler = SessionScheduler(self.pty_manager)
        self._scheduled_jobs_cache: Dict[str, ScheduledJob] = {}

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

        for summary in grouped_sessions:
            self.pty_manager.auto_approve_from_summary(summary)

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
        self._render_scheduled_runs_section()
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
            project = _pretty_project(first)
            text = _session_status_text(first, self.config.idle_seconds)
            self.title = f"{icon} {project} ({text.lower()})"
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

    def _render_scheduled_runs_section(self) -> None:
        jobs = self._refresh_scheduled_jobs()
        if not jobs:
            self.menu.add(rumps.MenuItem("Scheduled runs: none", callback=None))
            return

        self.menu.add(rumps.MenuItem("Scheduled runs", callback=None))
        for job in jobs:
            label = self._scheduled_job_label(job)
            item = rumps.MenuItem(label, callback=self._on_scheduled_job_clicked)
            item._scheduled_job_id = job.job_id  # type: ignore[attr-defined]
            self.menu.add(item)

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

    def _refresh_scheduled_jobs(self) -> List[ScheduledJob]:
        jobs = sorted(self.scheduler.list_jobs(), key=lambda job: job.run_at)
        self._scheduled_jobs_cache = {job.job_id: job for job in jobs}
        return jobs

    def _scheduled_job_label(self, job: ScheduledJob) -> str:
        run_at = job.run_at
        if run_at.tzinfo is not None:
            run_at = run_at.astimezone()
        time_text = run_at.strftime("%Y-%m-%d %H:%M")
        command_text = (job.initial_command or "(no command)").strip()
        if len(command_text) > 40:
            command_text = f"{command_text[:37]}…"
        project_name = job.launch_info.project_name
        return f"{time_text} — {project_name} — {command_text}"

    def _format_scheduled_job_details(self, job: ScheduledJob) -> str:
        run_at = job.run_at
        if run_at.tzinfo is not None:
            run_at = run_at.astimezone()
        time_text = run_at.strftime("%Y-%m-%d %H:%M:%S")
        command_text = job.initial_command or "(no command)"
        auto_text = "enabled" if job.auto_approve else "disabled"
        return (
            f"Session: {job.launch_info.session_id}\n"
            f"Project: {job.launch_info.project_name}\n"
            f"Working dir: {job.launch_info.resolve_working_directory()}\n"
            f"Runs at: {time_text}\n"
            f"Command: {command_text}\n"
            f"Auto approval: {auto_text} (choice {job.approval_choice})"
        )

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
        can_schedule = True
        if can_schedule:
            response = rumps.alert(
                title="Claude Session",
                message=details,
                ok="Close",
                cancel="Send Command…",
                other="Schedule Run…",
            )
        else:
            response = rumps.alert(
                title="Claude Session",
                message=details,
                ok="Close",
                cancel="Send Command…",
            )

        if response == 0:
            self._prompt_send_command(summary)
        elif response == 2 and can_schedule:
            self._prompt_schedule_run(summary)

    def _on_scheduled_job_clicked(self, sender: rumps.MenuItem) -> None:
        job_id = getattr(sender, "_scheduled_job_id", None)
        job = self._scheduled_jobs_cache.get(job_id)
        if not job:
            return
        message = self._format_scheduled_job_details(job)
        response = rumps.alert(
            title="Scheduled Run",
            message=message,
            ok="Close",
            cancel="Cancel Job",
            other="Run Now",
        )
        if response == 1:
            if self.scheduler.cancel_job(job.job_id):
                rumps.notification(
                    title="Claude Toolbar",
                    subtitle="Scheduled run cancelled",
                    message=self._scheduled_job_label(job),
                )
        elif response == 2:
            removed = self.scheduler.cancel_job(job.job_id)
            try:
                self.pty_manager.start_session(
                    job.launch_info,
                    initial_command=job.initial_command,
                    auto_approve=job.auto_approve,
                    approval_choice=job.approval_choice,
                )
            except Exception:
                rumps.notification(
                    title="Claude Toolbar",
                    subtitle="Unable to launch session",
                    message=self._scheduled_job_label(job),
                )
            else:
                subtitle = "Scheduled run launched" if removed else "Session launched"
                rumps.notification(
                    title="Claude Toolbar",
                    subtitle=subtitle,
                    message=self._scheduled_job_label(job),
                )
        self._render_pending_menu()

    def _prompt_send_command(self, summary: SessionSummary) -> None:
        prompt = rumps.Window(
            title="Send Command",
            message=f"Send a command to session {summary.session_id}\nProject: {_pretty_project(summary)}",
            default_text="/resume",
            ok="Send",
            cancel="Cancel",
        )
        response = prompt.run()
        if response.clicked != 0 or not response.text:
            return
        command = response.text.strip()
        if not command:
            return
        auto_approve, choice = self._ask_auto_approve()
        info = self._build_launch_info(summary)
        try:
            self.pty_manager.start_session(
                info,
                initial_command=command,
                auto_approve=auto_approve,
                approval_choice=choice,
            )
        except Exception:
            rumps.notification(
                title="Claude Toolbar",
                subtitle="Unable to deliver command",
                message=_pretty_project(summary),
            )
            return
        rumps.notification(
            title="Claude Toolbar",
            subtitle="Command sent",
            message=f"{_pretty_project(summary)} — {command}",
        )

    def _prompt_schedule_run(self, summary: SessionSummary) -> None:
        command_prompt = rumps.Window(
            title="Schedule Run",
            message=f"Command for session {summary.session_id}\nProject: {_pretty_project(summary)}",
            default_text="/resume",
            ok="Next",
            cancel="Cancel",
        )
        command_response = command_prompt.run()
        if command_response.clicked != 0 or not command_response.text:
            return
        command = command_response.text.strip()
        if not command:
            return

        timing_prompt = rumps.Window(
            title="Schedule Run",
            message="Start after how many minutes? (0 for immediate)",
            default_text="0",
            ok="Schedule",
            cancel="Cancel",
        )
        timing_response = timing_prompt.run()
        if timing_response.clicked != 0 or not timing_response.text:
            return
        try:
            minutes = float(timing_response.text.strip())
        except ValueError:
            rumps.notification(
                title="Claude Toolbar",
                subtitle="Invalid schedule delay",
                message=timing_response.text.strip(),
            )
            return
        run_at = datetime.now() + timedelta(minutes=max(0.0, minutes))

        auto_approve, choice = self._ask_auto_approve()
        info = self._build_launch_info(summary)
        job = self.scheduler.schedule_job(
            info,
            run_at=run_at,
            initial_command=command,
            auto_approve=auto_approve,
            approval_choice=choice,
        )
        rumps.notification(
            title="Claude Toolbar",
            subtitle="Run scheduled",
            message=self._scheduled_job_label(job),
        )
        self._render_pending_menu()

    def _ask_auto_approve(self) -> tuple[bool, str]:
        response = rumps.alert(
            title="Auto Approve",
            message="Automatically approve tool requests (sends a numeric choice)?",
            ok="Yes",
            cancel="No",
        )
        if response != 0:
            return False, "1"
        choice_prompt = rumps.Window(
            title="Auto Approve",
            message="Enter the numeric response to send when approval is required",
            default_text="1",
            ok="Save",
            cancel="Cancel",
        )
        choice_response = choice_prompt.run()
        if choice_response.clicked != 0 or not choice_response.text:
            return True, "1"
        choice = choice_response.text.strip() or "1"
        return True, choice

    def _build_launch_info(self, summary: SessionSummary) -> SessionLaunchInfo:
        return SessionLaunchInfo(
            session_id=summary.session_id,
            project=summary.project,
            cwd=summary.cwd,
            file_path=summary.file_path,
        )


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
