"""One-at-a-time background runner for `python -m experiments.eN`."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "experiments"


@dataclass
class JobState:
    module: str = ""
    args: List[str] = field(default_factory=list)
    status: str = "idle"  # idle | running | done | failed | cancelled
    log_path: Optional[Path] = None
    started_at: float = 0.0
    returncode: Optional[int] = None
    pid: Optional[int] = None


class JobQueue:
    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self.state = JobState()

    def start(self, module: str, extra_args: Optional[List[str]] = None) -> str:
        extra_args = extra_args or []
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return f"already running: {self.state.module} (pid {self.state.pid})"
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_path = LOG_DIR / "ui_job.log"
            cmd = [sys.executable, "-u", "-m", module, *extra_args]
            log_f = open(log_path, "w")
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=log_f,
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
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return "nothing to cancel"
            self._proc.terminate()
            self.state.status = "cancelled"
        return f"cancelled pid={self.state.pid}"

    def refresh(self) -> JobState:
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
        path = self.state.log_path
        if path is None or not path.exists():
            return "(no log)"
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:]) if lines else "(empty log)"

    def as_dict(self) -> dict:
        st = self.refresh()
        elapsed = time.time() - st.started_at if st.started_at else 0.0
        return {
            "status": st.status,
            "module": st.module,
            "args": list(st.args),
            "pid": st.pid,
            "elapsed_min": round(elapsed / 60.0, 2),
            "returncode": st.returncode,
            "log": self.log_tail(),
        }

    def status_markdown(self) -> str:
        st = self.as_dict()
        args = " ".join(st["args"])
        return (
            f"**status:** `{st['status']}`  \n"
            f"**job:** `{st['module']} {args}`  \n"
            f"**pid:** {st['pid'] or '—'}  ·  **elapsed:** {st['elapsed_min']:.1f} min  \n"
            f"**returncode:** {st['returncode'] if st['returncode'] is not None else '—'}\n\n"
            f"```\n{st['log']}\n```"
        )
