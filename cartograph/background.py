"""Run a long `carto` command detached from the terminal (and from whatever agent session started it),
so closing the window or ending the session doesn't kill a day-long index build."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from .config import home


def log_dir() -> Path:
    d = home() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def spawn(args: list[str], name: str) -> Path:
    """Start `carto <args>` in its own session; stdout/stderr go to ~/.cartograph/logs/<name>.log."""
    log = log_dir() / f"{name}.log"
    fh = open(log, "a", encoding="utf-8")
    fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  carto {' '.join(args)}\n")
    fh.flush()
    cmd = [sys.executable, "-m", "cartograph", *args]
    kw: dict = {"stdout": fh, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
                "cwd": str(Path.home()), "env": {**os.environ, "PYTHONUNBUFFERED": "1"}}
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200       # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    subprocess.Popen(cmd, **kw)
    return log
