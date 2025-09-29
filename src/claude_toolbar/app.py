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

if __package__ in (None, ""):
    # Handle execution as a top-level script inside the py2app bundle.
    from claude_toolbar.config import CONFIG_PATH, ToolbarConfig, load_config, save_config
    from claude_toolbar.models import (
        SessionStatus,
        SessionSummary,
        UsageSummary,
        UsageTotals,
        WindowInfo,
    )
    from claude_toolbar.paths import discover_claude_paths
    from claude_toolbar.preferences import PreferencesController
    from claude_toolbar.startup import (
        disable_launch_agent,
        enable_launch_agent,
        is_launch_agent_enabled,
    )
    from claude_toolbar.usage_tracker import UsageTracker
    from claude_toolbar.utils import (
        format_currency,
        format_relative,
        format_tokens,
        format_ts,
    )
else:
    from .config import CONFIG_PATH, ToolbarConfig, load_config, save_config
    from .models import (
        SessionStatus,
        SessionSummary,
        UsageSummary,
        UsageTotals,
        WindowInfo,
    )
    from .paths import discover_claude_paths
    from .preferences import PreferencesController
    from .startup import (
        disable_launch_agent,
        enable_launch_agent,
        is_launch_agent_enabled,
    )
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
RED_DOT = "🔴"
ICON_TODAY = "🟢"
ICON_WEEK = "🗓️"
ICON_MONTH = "📆"
ICON_LIMIT_OK = "✅"
ICON_LIMIT_WAIT = "⏳"
ICON_LIMIT_BLOCKED = "🚫"
ICON_WINDOW = "⏱️"
ICON_PATH = Path(__file__).resolve().parent / "assets" / "claude_toolbar_icon.png"
MIN_LOADING_DISPLAY_SECONDS = 0.8
MAX_LOADING_DISPLAY_SECONDS = 6.0
FAST_REFRESH_INTERVAL = 0.5

if AppKit is not None:  # pragma: no branch - macOS only styling
    def _nscolor(name: str, default):
        attr = getattr(AppKit.NSColor, name, None)
        return attr() if attr else default


    _MENU_FONT = AppKit.NSFont.menuFontOfSize_(0) or AppKit.NSFont.systemFontOfSize_(13)
    _MENU_BOLD_FONT = AppKit.NSFont.boldSystemFontOfSize_(_MENU_FONT.pointSize())
    COLOR_LABEL = _nscolor("labelColor", AppKit.NSColor.blackColor())
    COLOR_SECONDARY = _nscolor("secondaryLabelColor", AppKit.NSColor.grayColor())
    COLOR_TOKEN = _nscolor("systemTealColor", COLOR_LABEL)
    COLOR_COST = _nscolor("systemGreenColor", COLOR_LABEL)
    COLOR_WARNING = _nscolor("systemOrangeColor", COLOR_LABEL)
    COLOR_ALERT = _nscolor("systemRedColor", COLOR_LABEL)
    COLOR_OK = _nscolor("systemGreenColor", COLOR_LABEL)
    COLOR_INFO = _nscolor("systemBlueColor", COLOR_LABEL)
else:  # pragma: no cover - non-mac fallback
    _MENU_FONT = None
    _MENU_BOLD_FONT = None
    COLOR_LABEL = None
    COLOR_SECONDARY = None
    COLOR_TOKEN = None
    COLOR_COST = None
    COLOR_WARNING = None
    COLOR_ALERT = None
    COLOR_OK = None
    COLOR_INFO = None


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
        self.preferences_controller: Optional[PreferencesController] = None

        super().__init__("", icon=_load_icon_path(), quit_button=None)

        self.usage_header_item = rumps.MenuItem("📊 Usage Snapshot", callback=None)
        self.usage_today_item = rumps.MenuItem("Today: …")
        self.usage_week_item = rumps.MenuItem("Last 7 days: …")
        self.usage_month_item = rumps.MenuItem("This month: …")
        self.limit_header_item = rumps.MenuItem("🚦 Limits", callback=None)
        self.limit_item = rumps.MenuItem("Limit reset: …")
        self.window_item = rumps.MenuItem("5h window: —")

        self.refresh_item = rumps.MenuItem("Refresh Now", callback=self.refresh_now)
        self.open_config_item = rumps.MenuItem("Preferences…", callback=self.open_config)
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
        self._sync_launch_agent_preference()

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

    def _sync_launch_agent_preference(self) -> None:
        try:
            desired = bool(self.config.launch_at_login)
            current = is_launch_agent_enabled()
            if desired and not current:
                enable_launch_agent()
            elif not desired and current:
                disable_launch_agent()
        except Exception:
            return

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
            visible = sessions[: max(1, self.config.max_sessions)]
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
        self.menu.add(self.usage_header_item)
        self.menu.add(self.usage_today_item)
        self.menu.add(self.usage_week_item)
        self.menu.add(self.usage_month_item)
        self.menu.add(rumps.separator)
        self.menu.add(self.limit_header_item)
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
        self.menu.add(self.usage_header_item)
        self.menu.add(self.usage_today_item)
        self.menu.add(self.usage_week_item)
        self.menu.add(self.usage_month_item)
        self.menu.add(rumps.separator)
        self.menu.add(self.limit_header_item)
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
            self.usage_header_item.title = "📊 Usage Snapshot"
            self.limit_header_item.title = "🚦 Limits"
            self.usage_today_item.title = "Today: loading…"
            self.usage_week_item.title = "Last 7 days: loading…"
            self.usage_month_item.title = "This month: loading…"
            self.limit_item.title = "Limit reset: loading…"
            self.window_item.title = "Loading session data…"
            self.title = "⏳"
            return

        self._style_header_item(self.usage_header_item, "📊 Usage Snapshot")
        self._style_header_item(self.limit_header_item, "🚦 Limits")
        self._set_usage_item(
            self.usage_today_item, ICON_TODAY, "Today", summary.today, summary.today_cost
        )
        self._set_usage_item(
            self.usage_week_item,
            ICON_WEEK,
            "Last 7 days",
            summary.seven_day,
            summary.seven_day_cost,
        )
        self._set_usage_item(
            self.usage_month_item,
            ICON_MONTH,
            "This month",
            summary.month,
            summary.month_cost,
        )

        window = summary.window_info
        self._set_window_item(window)

        limit_ts = summary.limit_info.timestamp
        prefix = "Limit reset"
        if limit_ts is None and window.active_end:
            limit_ts = window.active_end
        elif summary.limit_info.timestamp is not None:
            prefix = "Limit reached" if summary.limit_info.reached else "Limit reset"

        self._set_limit_item(limit_ts, prefix, summary, window)

        working = []
        waiting = []
        running = []
        limited = []
        for summary in sessions:
            icon = _session_status_icon(summary, self.config.idle_seconds)
            if icon == ORANGE_DOT:
                waiting.append(summary)
            elif icon == BLUE_DOT:
                working.append(summary)
            elif icon == GREEN_DOT:
                running.append(summary)
            elif icon == RED_DOT:
                limited.append(summary)

        active = waiting + working + running + limited
        if len(active) == 1:
            first = active[0]
            icon = _session_status_icon(first, self.config.idle_seconds)
            text = _session_status_text(first, self.config.idle_seconds)
            self.title = f"{icon} {text}"
        elif active:
            parts: List[str] = []
            if limited:
                parts.append(f"{RED_DOT}{len(limited)}")
            if waiting:
                parts.append(f"{ORANGE_DOT}{len(waiting)}")
            if working:
                parts.append(f"{BLUE_DOT}{len(working)}")
            if running:
                parts.append(f"{GREEN_DOT}{len(running)}")
            self.title = " ".join(parts)
        else:
            self.title = f"{WHITE_DOT} Idle"

    def _style_header_item(self, item: rumps.MenuItem, text: str) -> None:
        _apply_menu_style(
            item,
            text,
            [(text, COLOR_SECONDARY if COLOR_SECONDARY else None, True)],
        )

    def _set_usage_item(
        self,
        item: rumps.MenuItem,
        icon: str,
        label: str,
        totals: "UsageTotals",
        cost: float,
    ) -> None:
        metrics: List[str] = []
        segments: List[tuple[str, Optional[object], bool]] = [
            (f"{icon} {label}: ", COLOR_LABEL if COLOR_LABEL else None, True)
        ]
        if self.config.show_tokens:
            token_text = f"{format_tokens(totals.total_tokens)} tokens"
            metrics.append(token_text)
            segments.append((token_text, COLOR_TOKEN if COLOR_TOKEN else None, False))
        if self.config.show_costs:
            if metrics:
                segments.append((" • ", None, False))
            cost_text = format_currency(cost)
            metrics.append(cost_text)
            segments.append((cost_text, COLOR_COST if COLOR_COST else None, False))
        if not metrics:
            metrics.append("hidden")
            segments.append(("hidden", COLOR_SECONDARY if COLOR_SECONDARY else None, False))
        fallback = f"{icon} {label}: {' • '.join(metrics)}"
        _apply_menu_style(item, fallback, segments)

    def _set_limit_item(
        self,
        limit_ts: Optional[datetime],
        prefix: str,
        summary: UsageSummary,
        window: "WindowInfo",
    ) -> None:
        candidate_ts = limit_ts
        candidate_label = prefix
        window_candidate = False
        now_utc = datetime.now(timezone.utc)
        if window.active_end and window.active_end >= now_utc:
            if candidate_ts is None or window.active_end < candidate_ts:
                candidate_ts = window.active_end
                candidate_label = "Window reset"
                window_candidate = True

        if candidate_ts:
            ts_text = format_ts(candidate_ts)
            now_utc = datetime.now(timezone.utc)
            rel = _format_time_remaining(candidate_ts)
            suffix = ""
            icon = ICON_LIMIT_WAIT
            color = COLOR_WARNING if COLOR_WARNING else None
            if candidate_ts <= now_utc:
                icon = ICON_LIMIT_OK
                color = COLOR_OK if COLOR_OK else None
                suffix = "available now"
            elif summary.limit_info.reached and not window_candidate:
                icon = ICON_LIMIT_BLOCKED
                color = COLOR_ALERT if COLOR_ALERT else None
                suffix = f"in {rel}" if rel else "soon"
            elif rel:
                suffix = f"in {rel}"
            fallback = f"{icon} {candidate_label}: {ts_text}" + (f" ({suffix})" if suffix else "")
            segments: List[tuple[str, Optional[object], bool]] = [
                (f"{icon} {candidate_label}: ", COLOR_LABEL if COLOR_LABEL else None, True),
                (ts_text, color, False),
            ]
            if suffix:
                segments.append((f" ({suffix})", COLOR_SECONDARY if COLOR_SECONDARY else None, False))
        else:
            fallback = f"{ICON_LIMIT_WAIT} Limit reset: unknown"
            segments = [
                (fallback, COLOR_SECONDARY if COLOR_SECONDARY else None, False)
            ]
        _apply_menu_style(self.limit_item, fallback, segments)

    def _set_window_item(self, window: "WindowInfo") -> None:
        if window.active_start and window.active_end:
            start_text = format_ts(window.active_start)
            end_text = format_ts(window.active_end)
            remaining = format_relative(window.active_end)
            fallback = (
                f"{ICON_WINDOW} 5h window: {start_text} → {end_text} (ends {remaining})"
            )
            segments: List[tuple[str, Optional[object], bool]] = [
                (f"{ICON_WINDOW} 5h window: ", COLOR_LABEL if COLOR_LABEL else None, True),
                (f"{start_text} → {end_text}", COLOR_INFO if COLOR_INFO else None, False),
                (f" (ends {remaining})", COLOR_SECONDARY if COLOR_SECONDARY else None, False),
            ]
        elif window.last_end:
            end_text = format_ts(window.last_end)
            since = format_relative(window.last_end)
            fallback = (
                f"{ICON_WINDOW} No active 5h window (last ended {end_text}, {since})"
            )
            segments = [
                (f"{ICON_WINDOW} No active 5h window ", COLOR_LABEL if COLOR_LABEL else None, True),
                ("(last ended ", COLOR_SECONDARY if COLOR_SECONDARY else None, False),
                (end_text, COLOR_INFO if COLOR_INFO else None, False),
                (f", {since})", COLOR_SECONDARY if COLOR_SECONDARY else None, False),
            ]
        else:
            fallback = f"{ICON_WINDOW} No active 5h window today"
            segments = [
                (fallback, COLOR_SECONDARY if COLOR_SECONDARY else None, False)
            ]
        _apply_menu_style(self.window_item, fallback, segments)

    def _populate_sessions(self, sessions: List[SessionSummary]) -> None:
        self.session_lookup.clear()
        if not sessions:
            message = "Loading sessions…" if self.tracker.is_initializing() else "No recent sessions"
            empty_item = rumps.MenuItem(message, callback=None)
            self.menu.add(empty_item)
            self._active_sessions_cache = []
            self._toggle_fast_refresh_timer(False)
            return

        limited = sessions[: max(1, self.config.max_sessions)]
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
        relative = (
            format_relative(summary.last_activity)
            if summary.last_activity
            else "unknown"
        )
        project_display = _pretty_project(summary)
        status_text = _session_status_text(summary, self.config.idle_seconds)
        metrics: List[str] = []
        if self.config.show_tokens:
            metrics.append(f"{format_tokens(summary.totals.total_tokens)} tokens")
        if self.config.show_costs:
            metrics.append(format_currency(summary.cost_usd))
        segments = [f"{emoji} {project_display}"]
        if metrics:
            segments.append(" • ".join(metrics))
        segments.extend([status_text, relative])
        return " — ".join(segments)

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
        if AppKit is None:
            save_config(self.config)
            subprocess.run([
                "open",
                str(CONFIG_PATH.parent),
            ], check=False)
            return
        if self.preferences_controller is None:
            try:
                self.preferences_controller = PreferencesController(
                    self.config, self._apply_preferences
                )
            except RuntimeError:
                save_config(self.config)
                subprocess.run([
                    "open",
                    str(CONFIG_PATH.parent),
                ], check=False)
                return
        self.preferences_controller.show()

    def _apply_preferences(self, config: ToolbarConfig) -> None:
        self.config = config
        save_config(self.config)
        self._sync_launch_agent_preference()
        self.refresh_timer_tick(None)

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


def _apply_menu_style(menu_item: rumps.MenuItem, fallback: str, segments: List[tuple[str, Optional[object], bool]]) -> None:
    menu_item.title = fallback
    if AppKit is None or not hasattr(menu_item, "_menuitem") or _MENU_FONT is None:
        return
    try:
        attributed = AppKit.NSMutableAttributedString.alloc().initWithString_("")
        for text, color, bold in segments:
            attrs = {
                AppKit.NSFontAttributeName: _MENU_BOLD_FONT if bold else _MENU_FONT
            }
            if color is not None:
                attrs[AppKit.NSForegroundColorAttributeName] = color
            fragment = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                text,
                attrs,
            )
            attributed.appendAttributedString_(fragment)
        menu_item._menuitem.setAttributedTitle_(attributed)
    except Exception:
        menu_item.title = fallback


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
    if summary.limit_blocked and summary.processes:
        if _limit_reset_available(summary):
            return ORANGE_DOT
        return RED_DOT
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
    if summary.limit_blocked and summary.processes:
        if summary.limit_reset_at:
            if _limit_reset_available(summary):
                return "Limit reset available"
            remaining = _format_time_remaining(summary.limit_reset_at)
            suffix = f"in {remaining}" if remaining else "soon"
            return f"Waiting for limit reset ({suffix})"
        return "Waiting for limit reset"
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


def _format_time_remaining(target: datetime) -> str:
    now = datetime.now(timezone.utc)
    delta_seconds = int((target - now).total_seconds())
    if delta_seconds <= 0:
        return ""
    hours, remainder = divmod(delta_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    parts: List[str] = []
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def _limit_reset_available(summary: SessionSummary) -> bool:
    if not summary.limit_reset_at:
        return False
    now = datetime.now(timezone.utc)
    return summary.limit_reset_at <= now


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
