import time
from datetime import datetime, timezone

from claude_toolbar.config import load_config
from claude_toolbar.paths import discover_claude_paths
from claude_toolbar.usage_tracker import UsageTracker


def main() -> None:
    config = load_config()
    tracker = UsageTracker(config, discover_claude_paths(config.claude_paths))

    t0 = time.perf_counter()
    tracker.update()
    first_elapsed = time.perf_counter() - t0

    sessions = tracker.get_session_summaries()
    now = datetime.now(timezone.utc)
    active = [s for s in sessions if s.last_activity and (now - s.last_activity).total_seconds() <= config.idle_seconds]

    print(f"first update: {first_elapsed:.3f}s, sessions={len(sessions)}, recent_active={len(active)}")

    t1 = time.perf_counter()
    tracker.update()
    second_elapsed = time.perf_counter() - t1
    print(f"second update: {second_elapsed:.3f}s")

    summary = tracker.get_usage_summary()
    print(f"totals → today_tokens={summary.today.total_tokens}, seven_day_cost={summary.seven_day_cost}")


if __name__ == "__main__":
    main()
