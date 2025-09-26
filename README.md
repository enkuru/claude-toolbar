# Claude Toolbar

A lightweight macOS menu bar utility that surfaces Claude Code usage metrics and live session status without relying on the Anthropic API. It reads the locally cached session transcripts that Claude CLI already writes under `~/.claude/projects`, aggregates usage, and shows a quick view of active sessions, throttling status, and limit reset estimates.

## Features

- **Usage tracking** – rolling totals for today, the last seven days, and the current month (token counts include cache reads/creation so they align with billing).
- **Limit reset hint** – uses any recorded "usage limit" messages when available, otherwise falls back to the first day of the next month; you can override the timestamp in the config file.
- **5-hour window tracking** – calculates the current Claude Code 5h throttle window from session start times, even if you haven’t hit the limit message yet, and shows when the window ends.
- **Cost estimates** – shells out to `ccusage` for per-session and per-day pricing so you see dollars alongside tokens (requires `ccusage` on your PATH).
- **Session presence** – lists the most recent Claude sessions with status icons (`🟢` running, `⚪️` idle, `🟡` waiting on approval) and relative last-activity times.
- **Approval awareness** – detects "This command requires approval" results inside the transcript so you can see which session is stuck.
- **Process linkage** – optionally associates running `claude` CLI processes (via `psutil`) with each session and shows their PIDs in the details popup.

## Installation

1. Ensure you have Python 3.10+ on macOS. Install the dependencies:

   ```bash
   pip install -e .
   ```

   The project depends on `rumps` for menu bar integration and `psutil` for lightweight process inspection.

2. (Optional) Create a config file at `~/.config/claude_toolbar/config.json` if you want to tweak paths, refresh interval, or override the limit reset timestamp. The defaults typically work out of the box.

3. Start the toolbar:

   ```bash
   claude-toolbar
   ```

   or run directly via:

   ```bash
   python -m claude_toolbar.app
   ```

The menu bar title displays `Claude <tokens today>` so you can keep an eye on burn without opening the drop-down.

## Configuration

Configuration lives at `~/.config/claude_toolbar/config.json`. All fields are optional; omit anything you do not need.

```json
{
  "claude_paths": ["/Users/you/.claude"],
  "refresh_interval": 5.0,
  "idle_seconds": 120,
  "history_days": 60,
  "session_duration_hours": 5,
  "enable_process_monitor": true,
  "ccusage_refresh_interval": 120,
  "enable_ccusage_prices": true,
  "limit_reset_override": "2025-10-01T00:00:00+00:00"
}
```

- `claude_paths`: Explicit locations that contain the `projects/` transcripts. If omitted the tool falls back to `$CLAUDE_CONFIG_DIR`, then `~/.config/claude` and `~/.claude`.
- `refresh_interval`: Seconds between background scans.
- `idle_seconds`: How long (in seconds) of inactivity before a session is treated as idle.
- `history_days`: Window of daily usage cached in memory for summaries.
- `session_duration_hours`: The rolling window that defines an "active" session block (mirrors ccusage defaults).
- `enable_process_monitor`: Set to `false` if you do not want the app to touch `psutil` or inspect running processes.
- `ccusage_refresh_interval`: Seconds between reusing `ccusage` output (defaults to 120).
- `enable_ccusage_prices`: Disable if you do not have `ccusage` installed or don’t want dollar estimates.
- `limit_reset_override`: Optional ISO timestamp to use when the CLI has not logged a reset hint (or when you want to correct it manually).

After editing the configuration you can select **Refresh Now** from the menu or restart the app to reload it.

## Session details popup

Clicking any session entry reveals:

- Total tokens and the individual input/output/cache components
- Last activity timestamp and relative time
- Any queued approval/error message (e.g. "This command requires approval")
- The associated `claude` process IDs if the process monitor is enabled

Inactive (older) sessions are automatically ordered to the bottom; only the 15 most recent entries are shown to keep the menu tidy.

## Notes & Limitations

- The toolbar only reads local transcript files. It will match whatever latency Claude CLI has for flushing those JSONL entries.
- Limit reset times are best-effort. If the CLI has never displayed a "usage limit" message, the toolbar assumes a monthly reset and labels the source accordingly.
- The app keeps everything in-memory; no telemetry or external network calls are performed.

## Development

Run unit checks quickly with:

```bash
PYTHONPATH=src python -m compileall src/claude_toolbar
```

You can also invoke `UsageTracker` in isolation for debugging:

```bash
PYTHONPATH=src python - <<'PY'
from claude_toolbar.config import load_config
from claude_toolbar.paths import discover_claude_paths
from claude_toolbar.usage_tracker import UsageTracker

config = load_config()
tracker = UsageTracker(config, discover_claude_paths(config.claude_paths))
tracker.update()
print(tracker.get_usage_summary())
PY
```

To trace startup behaviour, enable debug logging by launching with:

```bash
CLAUDE_TOOLBAR_DEBUG=1 claude-toolbar
```

or run the measurement helper:

```bash
PYTHONPATH=src python scripts/measure_startup.py
```

---

Made for Claude CLI power users who want a quick sanity check on usage caps and long-running sessions right in the menu bar.
