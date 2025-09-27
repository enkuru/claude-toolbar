from __future__ import annotations

import heapq
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .pty_manager import PtyManager, SessionLaunchInfo


@dataclass(order=True)
class ScheduledJob:
    run_at: datetime
    job_id: str = field(compare=False, default_factory=lambda: uuid.uuid4().hex)
    launch_info: SessionLaunchInfo = field(compare=False, default=None)
    initial_command: Optional[str] = field(compare=False, default=None)
    auto_approve: bool = field(compare=False, default=False)
    approval_choice: str = field(compare=False, default="1")

    def seconds_until(self) -> float:
        return max(0.0, (self.run_at - datetime.now(self.run_at.tzinfo)).total_seconds())


class SessionScheduler:
    def __init__(self, pty_manager: PtyManager):
        self._pty_manager = pty_manager
        self._jobs: List[ScheduledJob] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._runner = threading.Thread(target=self._run_loop, name="session-scheduler", daemon=True)
        self._runner.start()

    def schedule_job(
        self,
        launch_info: SessionLaunchInfo,
        run_at: datetime,
        initial_command: Optional[str],
        auto_approve: bool,
        approval_choice: str,
    ) -> ScheduledJob:
        job = ScheduledJob(
            run_at=run_at,
            launch_info=launch_info,
            initial_command=initial_command,
            auto_approve=auto_approve,
            approval_choice=approval_choice or "1",
        )
        with self._condition:
            heapq.heappush(self._jobs, job)
            self._condition.notify()
        return job

    def cancel_job(self, job_id: str) -> bool:
        with self._condition:
            for idx, job in enumerate(self._jobs):
                if job.job_id == job_id:
                    self._jobs.pop(idx)
                    heapq.heapify(self._jobs)
                    return True
        return False

    def list_jobs(self) -> List[ScheduledJob]:
        with self._lock:
            return list(self._jobs)

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                while not self._jobs:
                    self._condition.wait()
                job = self._jobs[0]
                delay = job.seconds_until()
                if delay > 0:
                    self._condition.wait(timeout=delay)
                    continue
                heapq.heappop(self._jobs)
            self._execute(job)

    def _execute(self, job: ScheduledJob) -> None:
        try:
            self._pty_manager.start_session(
                job.launch_info,
                initial_command=job.initial_command,
                auto_approve=job.auto_approve,
                approval_choice=job.approval_choice,
            )
        except Exception:
            # Runtime errors are swallowed here; logging could be added later if desired.
            return


__all__ = ["ScheduledJob", "SessionScheduler"]
