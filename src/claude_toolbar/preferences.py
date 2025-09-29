from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple, Sequence

try:
    import AppKit
except ImportError:  # pragma: no cover - macOS only UI
    AppKit = None  # type: ignore

from .config import ToolbarConfig
from .utils import parse_timestamp


NS_ON_STATE = 1
NS_OFF_STATE = 0
if AppKit is not None:  # pragma: no branch - macOS only constants
    NS_ON_STATE = getattr(AppKit, "NSControlStateValueOn", getattr(AppKit, "NSOnState", 1))
    NS_OFF_STATE = getattr(AppKit, "NSControlStateValueOff", getattr(AppKit, "NSOffState", 0))


def _make_rect(frame) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Normalize Cocoa frame inputs to ((x, y), (width, height))."""

    if hasattr(frame, "origin") and hasattr(frame, "size"):
        origin = frame.origin
        size = frame.size
        return (
            (float(getattr(origin, "x", 0.0)), float(getattr(origin, "y", 0.0))),
            (float(getattr(size, "width", 0.0)), float(getattr(size, "height", 0.0))),
        )

    if isinstance(frame, Sequence):
        if len(frame) == 2 and all(isinstance(part, Sequence) for part in frame):
            (x, y), (w, h) = frame  # type: ignore[misc]
            return ((float(x), float(y)), (float(w), float(h)))
        if len(frame) == 4:
            x, y, w, h = frame  # type: ignore[misc]
            return ((float(x), float(y)), (float(w), float(h)))

    raise ValueError(f"Unsupported frame format: {frame!r}")


class PreferencesController:  # pragma: no cover - UI heavy
    def __init__(self, config: ToolbarConfig, on_apply: Callable[[ToolbarConfig], None]):
        if AppKit is None:
            raise RuntimeError("Preferences UI requires macOS AppKit")
        self.config = config
        self.on_apply = on_apply

        self.window: Optional[AppKit.NSWindow] = None
        self.show_costs_checkbox = None
        self.show_tokens_checkbox = None
        self.enable_process_checkbox = None
        self.enable_ccusage_checkbox = None
        self.launch_login_checkbox = None
        self.refresh_interval_field = None
        self.idle_seconds_field = None
        self.history_days_field = None
        self.session_duration_field = None
        self.ccusage_refresh_field = None
        self.limit_override_field = None
        self.max_sessions_field = None
        self.claude_paths_view = None

        self._build_window()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_window(self) -> None:
        frame = ((0.0, 0.0), (420.0, 600.0))
        style = (
            getattr(AppKit, "NSWindowStyleMaskTitled", AppKit.NSTitledWindowMask)
            | getattr(AppKit, "NSWindowStyleMaskClosable", AppKit.NSClosableWindowMask)
            | getattr(AppKit, "NSWindowStyleMaskMiniaturizable", AppKit.NSMiniaturizableWindowMask)
        )
        window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            style,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        window.setTitle_("Claude Toolbar Preferences")
        window.center()

        content_width = int(frame[1][0])
        content_height = int(frame[1][1])
        content = AppKit.NSView.alloc().initWithFrame_(((0, 0), (content_width, content_height)))
        window.setContentView_(content)

        padding = 20
        label_width = 210
        field_width = content_width - (2 * padding) - label_width - 10
        y = content_height - padding - 24

        self.show_tokens_checkbox = self._add_checkbox(
            content,
            title="Show tokens",
            frame=(padding, y, 200, 24),
            value=self.config.show_tokens,
        )
        y -= 28

        self.show_costs_checkbox = self._add_checkbox(
            content,
            title="Show costs",
            frame=(padding, y, 200, 24),
            value=self.config.show_costs,
        )
        y -= 28

        self.enable_process_checkbox = self._add_checkbox(
            content,
            title="Enable process monitor",
            frame=(padding, y, 220, 24),
            value=self.config.enable_process_monitor,
        )
        y -= 28

        self.enable_ccusage_checkbox = self._add_checkbox(
            content,
            title="Enable ccusage pricing",
            frame=(padding, y, 220, 24),
            value=self.config.enable_ccusage_prices,
        )
        y -= 28

        self.launch_login_checkbox = self._add_checkbox(
            content,
            title="Launch at login",
            frame=(padding, y, 200, 24),
            value=self.config.launch_at_login,
        )
        y -= 36

        def add_field(title: str, value: str):
            nonlocal y
            self._add_label(
                content,
                title,
                (padding, y, label_width, 22),
            )
            field = self._add_text_field(
                content,
                (padding + label_width + 10, y, field_width, 24),
                value,
            )
            y -= 32
            return field

        self.refresh_interval_field = add_field(
            "Refresh interval (seconds):",
            f"{self.config.refresh_interval}",
        )
        self.idle_seconds_field = add_field(
            "Idle timeout (seconds):",
            str(self.config.idle_seconds),
        )
        self.history_days_field = add_field(
            "History window (days):",
            str(self.config.history_days),
        )
        self.session_duration_field = add_field(
            "Session window (hours):",
            str(self.config.session_duration_hours),
        )
        self.ccusage_refresh_field = add_field(
            "ccusage refresh interval (seconds):",
            str(self.config.ccusage_refresh_interval),
        )
        self.max_sessions_field = add_field(
            "Max sessions in menu:",
            str(self.config.max_sessions),
        )
        self.limit_override_field = add_field(
            "Limit reset override (ISO 8601):",
            self._format_limit_override(),
        )

        paths_label_y = y
        self._add_label(
            content,
            "Claude paths (one per line):",
            (padding, paths_label_y, content_width - 2 * padding, 22),
        )
        y -= 26

        paths_height = 140
        scroll_frame = (padding, y - paths_height, content_width - 2 * padding, paths_height)
        scroll_view = AppKit.NSScrollView.alloc().initWithFrame_(_make_rect(scroll_frame))
        scroll_view.setBorderType_(AppKit.NSBezelBorder)
        scroll_view.setHasVerticalScroller_(True)
        scroll_view.setAutohidesScrollers_(True)

        text_view_frame = _make_rect((0, 0, scroll_frame[2], scroll_frame[3]))
        self.claude_paths_view = AppKit.NSTextView.alloc().initWithFrame_(text_view_frame)
        self.claude_paths_view.setRichText_(False)
        self.claude_paths_view.setFont_(AppKit.NSFont.systemFontOfSize_(12))
        self.claude_paths_view.setString_(self._format_claude_paths())
        self.claude_paths_view.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)

        scroll_view.setDocumentView_(self.claude_paths_view)
        content.addSubview_(scroll_view)
        y = scroll_frame[1] - 16

        save_button = AppKit.NSButton.alloc().initWithFrame_(_make_rect((padding, 20, 140, 32)))
        save_button.setTitle_("Save")
        save_button.setBezelStyle_(
            getattr(AppKit, "NSBezelStyleRounded", AppKit.NSRoundedBezelStyle)
        )
        save_button.setTarget_(self)
        save_button.setAction_("saveClicked:")
        content.addSubview_(save_button)

        cancel_button = AppKit.NSButton.alloc().initWithFrame_(_make_rect((padding + 160, 20, 140, 32)))
        cancel_button.setTitle_("Cancel")
        cancel_button.setBezelStyle_(
            getattr(AppKit, "NSBezelStyleRounded", AppKit.NSRoundedBezelStyle)
        )
        cancel_button.setTarget_(self)
        cancel_button.setAction_("cancelClicked:")
        content.addSubview_(cancel_button)

        self.window = window

    def _add_checkbox(self, parent, title: str, frame, value: bool):
        checkbox = AppKit.NSButton.alloc().initWithFrame_(_make_rect(frame))
        checkbox.setButtonType_(AppKit.NSSwitchButton)
        checkbox.setTitle_(title)
        checkbox.setState_(NS_ON_STATE if value else NS_OFF_STATE)
        parent.addSubview_(checkbox)
        return checkbox

    def _add_label(self, parent, title: str, frame):
        label = AppKit.NSTextField.alloc().initWithFrame_(_make_rect(frame))
        label.setStringValue_(title)
        label.setBordered_(False)
        label.setEditable_(False)
        label.setDrawsBackground_(False)
        parent.addSubview_(label)
        return label

    def _add_text_field(self, parent, frame, value: str):
        field = AppKit.NSTextField.alloc().initWithFrame_(_make_rect(frame))
        field.setStringValue_(value)
        parent.addSubview_(field)
        return field

    def _format_limit_override(self) -> str:
        override = self.config.limit_reset_override
        if override is None:
            return ""
        return override.isoformat()

    def _format_claude_paths(self) -> str:
        if not self.config.claude_paths:
            return ""
        return "\n".join(str(path) for path in self.config.claude_paths)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def show(self) -> None:
        if self.window is None:
            return
        self.window.makeKeyAndOrderFront_(None)
        AppKit.NSApp.activateIgnoringOtherApps_(True)

    def saveClicked_(self, sender) -> None:  # noqa: N802 - Cocoa selector
        if self.window is None:
            return

        def coerce_float(field, default: float, minimum: Optional[float] = None) -> float:
            try:
                value = float(field.stringValue())
            except (TypeError, ValueError, AttributeError):
                return default
            if minimum is not None:
                value = max(minimum, value)
            return value

        def coerce_int(field, default: int, minimum: int = 1) -> int:
            try:
                value = int(field.stringValue())
            except (TypeError, ValueError, AttributeError):
                return default
            return max(minimum, value)

        self.config.show_costs = self.show_costs_checkbox.state() == NS_ON_STATE
        self.config.show_tokens = self.show_tokens_checkbox.state() == NS_ON_STATE
        self.config.enable_process_monitor = (
            self.enable_process_checkbox.state() == NS_ON_STATE
        )
        self.config.enable_ccusage_prices = (
            self.enable_ccusage_checkbox.state() == NS_ON_STATE
        )
        self.config.launch_at_login = self.launch_login_checkbox.state() == NS_ON_STATE

        self.config.refresh_interval = coerce_float(
            self.refresh_interval_field,
            self.config.refresh_interval,
            minimum=0.5,
        )
        self.config.idle_seconds = coerce_int(
            self.idle_seconds_field,
            self.config.idle_seconds,
            minimum=30,
        )
        self.config.history_days = coerce_int(
            self.history_days_field,
            self.config.history_days,
            minimum=1,
        )
        self.config.session_duration_hours = coerce_int(
            self.session_duration_field,
            self.config.session_duration_hours,
            minimum=1,
        )
        self.config.ccusage_refresh_interval = coerce_int(
            self.ccusage_refresh_field,
            self.config.ccusage_refresh_interval,
            minimum=30,
        )
        self.config.max_sessions = coerce_int(
            self.max_sessions_field,
            self.config.max_sessions,
            minimum=1,
        )

        override_raw = ""
        if self.limit_override_field is not None:
            override_raw = self.limit_override_field.stringValue().strip()
        if override_raw:
            parsed = parse_timestamp(override_raw)
            if parsed is not None:
                self.config.limit_reset_override = parsed
        else:
            self.config.limit_reset_override = None

        if self.claude_paths_view is not None:
            raw_paths = self.claude_paths_view.string() or ""
            parsed_paths = []
            for line in raw_paths.splitlines():
                candidate = line.strip()
                if candidate:
                    parsed_paths.append(Path(candidate).expanduser())
            self.config.claude_paths = parsed_paths

        self.on_apply(self.config)
        self.window.orderOut_(None)

    def cancelClicked_(self, sender) -> None:  # noqa: N802 - Cocoa selector
        if self.window is not None:
            self.window.orderOut_(None)


__all__ = ["PreferencesController"]
