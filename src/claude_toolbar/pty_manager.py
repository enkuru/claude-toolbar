from __future__ import annotations

import os
import pty
import selectors
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence


@dataclass
class SessionLaunchInfo:
    session_id: str
    project: str
    cwd: Optional[str] = None
    file_path: Optional[str] = None

    def resolve_working_directory(self) -> Path:
        if self.cwd:
            candidate = Path(self.cwd)
            if candidate.exists():
                return candidate
        if self.file_path:
            candidate = Path(self.file_path).expanduser().resolve().parent
            if candidate.exists():
                return candidate
        return Path.home()

    @property
    def project_name(self) -> str:
        if self.cwd:
            return Path(self.cwd).name
        if self.file_path:
            return Path(self.file_path).parent.name
        return self.project


@dataclass
class PtySession:
    launch_info: SessionLaunchInfo
    command: Sequence[str]
    auto_approve: bool = False
    approval_choice: str = "1"
    startup_timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._master_fd: Optional[int] = None
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._output_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._last_auto_message: Optional[str] = None
        self._exceptions: List[str] = []

    def start(self) -> None:
        if self._running.is_set():
            return
        try:
            master_fd, slave_fd = pty.openpty()
            cwd = self.launch_info.resolve_working_directory()
            self._process = subprocess.Popen(
                list(self.command),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=str(cwd),
                close_fds=True,
                start_new_session=True,
            )
            os.close(slave_fd)
            self._master_fd = master_fd
            self._running.set()
            self._output_thread = threading.Thread(
                target=self._consume_output,
                name=f"pty-session-{self.launch_info.session_id[:8]}-out",
                daemon=True,
            )
            self._output_thread.start()
            self._monitor_thread = threading.Thread(
                target=self._monitor_process,
                name=f"pty-session-{self.launch_info.session_id[:8]}-monitor",
                daemon=True,
            )
            self._monitor_thread.start()
        except Exception as exc:  # pragma: no cover - defensive
            self._exceptions.append(str(exc))
            self._running.clear()
            if self._master_fd is not None:
                os.close(self._master_fd)
                self._master_fd = None
            raise

    def _consume_output(self) -> None:
        if self._master_fd is None:
            return
        sel = selectors.DefaultSelector()
        sel.register(self._master_fd, selectors.EVENT_READ)
        try:
            while self._running.is_set():
                events = sel.select(timeout=0.5)
                if not events:
                    if self._process and self._process.poll() is not None:
                        break
                    continue
                for key, _ in events:
                    try:
                        data = os.read(key.fd, 4096)
                    except OSError:
                        self._running.clear()
                        return
                    if not data:
                        self._running.clear()
                        return
        finally:
            sel.close()

    def _monitor_process(self) -> None:
        if not self._process:
            return
        self._process.wait()
        self._running.clear()

    def send_text(self, text: str, append_newline: bool = True) -> None:
        if not self._running.is_set() or self._master_fd is None:
            return
        payload = text
        if append_newline and not text.endswith("\n"):
            payload += "\n"
        with self._lock:
            try:
                os.write(self._master_fd, payload.encode("utf-8"))
            except OSError:
                self._running.clear()

    def stop(self) -> None:
        self._running.clear()
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
            except OSError:
                pass
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None

    def is_running(self) -> bool:
        return self._running.is_set()

    def record_auto_message(self, message: Optional[str]) -> bool:
        if not message:
            return False
        normalized = message.strip()
        if not normalized:
            return False
        if normalized == self._last_auto_message:
            return False
        self._last_auto_message = normalized
        return True


class PtyManager:
    def __init__(self, launch_command: Sequence[str]):
        if not launch_command:
            raise ValueError("launch_command must contain at least one element")
        self._launch_command = list(launch_command)
        self._sessions: Dict[str, PtySession] = {}
        self._lock = threading.Lock()

    def _format_command(self, info: SessionLaunchInfo) -> List[str]:
        mapping = {
            "session_id": info.session_id,
            "project": info.project,
            "project_name": info.project_name,
            "project_path": str(info.resolve_working_directory()),
        }
        return [part.format(**mapping) for part in self._launch_command]

    def start_session(
        self,
        info: SessionLaunchInfo,
        initial_command: Optional[str] = None,
        auto_approve: bool = False,
        approval_choice: str = "1",
    ) -> PtySession:
        approval_choice = approval_choice or "1"
        with self._lock:
            controller = self._sessions.get(info.session_id)
            if controller and controller.is_running():
                controller.auto_approve = auto_approve
                controller.approval_choice = approval_choice
            else:
                command = self._format_command(info)
                controller = PtySession(
                    launch_info=info,
                    command=command,
                    auto_approve=auto_approve,
                    approval_choice=approval_choice,
                )
                try:
                    controller.start()
                except Exception:
                    self._sessions.pop(info.session_id, None)
                    raise
                self._sessions[info.session_id] = controller
        if initial_command:
            # Give the session a short moment to boot before sending the first command.
            threading.Timer(0.5, controller.send_text, args=(initial_command, True)).start()
        return controller

    def send_command(self, session_id: str, command: str) -> None:
        with self._lock:
            controller = self._sessions.get(session_id)
        if not controller:
            return
        controller.send_text(command, append_newline=True)

    def stop_session(self, session_id: str) -> None:
        with self._lock:
            controller = self._sessions.pop(session_id, None)
        if controller:
            controller.stop()

    def active_sessions(self) -> List[str]:
        with self._lock:
            return [sid for sid, controller in self._sessions.items() if controller.is_running()]

    def auto_approve_from_summary(self, summary) -> None:
        session_id = getattr(summary, "session_id", None)
        if not session_id:
            return
        with self._lock:
            controller = self._sessions.get(session_id)
        if not controller or not controller.auto_approve:
            return
        if not summary.awaiting_message:
            controller.record_auto_message(None)
            return
        if not controller.record_auto_message(summary.awaiting_message):
            return
        controller.send_text(controller.approval_choice, append_newline=True)

    def set_auto_approve(self, session_id: str, enabled: bool, choice: str = "1") -> None:
        with self._lock:
            controller = self._sessions.get(session_id)
            if not controller:
                return
            controller.auto_approve = enabled
            controller.approval_choice = choice or "1"

    def get_controller(self, session_id: str) -> Optional[PtySession]:
        with self._lock:
            return self._sessions.get(session_id)


__all__ = [
    "SessionLaunchInfo",
    "PtySession",
    "PtyManager",
]
