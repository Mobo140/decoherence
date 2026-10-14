"""One-at-a-time background runner for `python -m experiments.<name>`."""
from __future__ import annotations

import dataclasses
import pathlib
import subprocess
import sys
import threading
import time
from collections.abc import Sequence

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "experiments"


@dataclasses.dataclass
class JobState:
    """State of the current or last job.

    Attributes:
        module: Module run with `python -m`.
        args: Extra command-line arguments.
        status: One of idle, running, done, failed, cancelled.
        log_path: File that receives stdout and stderr.
        started_at: Start time in seconds since the epoch.
        returncode: Exit code once the process has ended.
        pid: Process id.
    """

    module: str = ""
    args: list[str] = dataclasses.field(default_factory=list)
    status: str = "idle"
    log_path: pathlib.Path | None = None
    started_at: float = 0.0
    returncode: int | None = None
    pid: int | None = None


class JobQueue:
    """Runs one experiment process at a time in the background."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.state = JobState()

    def start(self, module: str, extra_args: Sequence[str] = ()) -> str:
        """Starts `python -m module` unless a job is already running.

        Args:
            module: Module to run, e.g. "experiments.e8_improved_2qubit".
            extra_args: Arguments after the module name.

        Returns:
            A message saying what happened.
        """
        extra_args = list(extra_args)
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return (f"already running: {self.state.module} "
                        f"(pid {self.state.pid})")
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_path = LOG_DIR / "ui_job.log"
            command = [sys.executable, "-u", "-m", module, *extra_args]
            with open(log_path, "w") as log_file:
                self._proc = subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            self.state = JobState(
                module=module,
                args=extra_args,
                status="running",
                log_path=log_path,
                started_at=time.time(),
                pid=self._proc.pid,
            )
        return f"started {module} {' '.join(extra_args)} pid={self._proc.pid}"

    def cancel(self) -> str:
        """Terminates the running job, if any.

        Returns:
            A message saying what happened.
        """
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return "nothing to cancel"
            self._proc.terminate()
            self.state.status = "cancelled"
        return f"cancelled pid={self.state.pid}"

    def refresh(self) -> JobState:
        """Updates and returns the job state from the process."""
        with self._lock:
            if self._proc is not None:
                code = self._proc.poll()
                if code is None:
                    self.state.status = "running"
                else:
                    self.state.returncode = code
                    self.state.status = "done" if code == 0 else "failed"
            return self.state

    def log_tail(self, n: int = 40) -> str:
        """Returns the last `n` lines of the job log."""
        path = self.state.log_path
        if path is None or not path.exists():
            return "(no log)"
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:]) if lines else "(empty log)"

    def as_dict(self) -> dict[str, object]:
        """Returns the job state as a JSON-ready dict."""
        state = self.refresh()
        elapsed = time.time() - state.started_at if state.started_at else 0.0
        return {
            "status": state.status,
            "module": state.module,
            "args": list(state.args),
            "pid": state.pid,
            "elapsed_min": round(elapsed / 60.0, 2),
            "returncode": state.returncode,
            "log": self.log_tail(),
        }
