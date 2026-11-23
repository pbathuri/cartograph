"""Terminal view of a semantic-index build: `carto index --status [--watch]`."""
from __future__ import annotations

import time
from typing import Callable

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

MOLE = ("  ▄█▀▀▀█▄  ", " █ ◕ ᴥ ◕ █ ", "  ▀█▄▄▄█▀ ⛏")


def _dur(sec: float | None) -> str:
    if sec is None or sec != sec or sec < 0:
        return "—"
    sec = int(sec)
    d, sec = divmod(sec, 86400)
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    return f"{d}d {h:02d}h {m:02d}m" if d else f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def render(st: dict, live_rate: float | None = None) -> Panel:
    state = st.get("state", "none")
    done, total = int(st.get("done", 0)), int(st.get("total", 0))
    frac = done / total if total else (1.0 if state == "complete" else 0.0)
    rate = live_rate if live_rate else float(st.get("rate") or 0)
    eta = (total - done) / rate if rate > 0 and total else None
    if state == "complete":
        headline, colour = "index complete — semantic search is on", "green"
    elif state == "building" and st.get("running"):
        headline, colour = "digging through your chunks…", "yellow"
    elif state == "building":
        headline, colour = "paused — run `carto index` to resume where it stopped", "red"
    else:
        headline, colour = "no index yet — run `carto index` (or `carto index -b`)", "dim"
    mole = Text("\n".join(MOLE), style="bold #c08457")
    head = Table.grid(padding=(0, 2))
    head.add_row(mole, Text.assemble(("Cartograph", "bold"), ("  semantic index\n", "dim"), (headline, colour)))
    stats = Table.grid(padding=(0, 3))
    stats.add_row(Text("vectors", style="dim"), f"{done:,} / {total:,}", Text("done", style="dim"), f"{frac:6.2%}")
    stats.add_row(Text("rate", style="dim"), f"{rate:,.0f} vec/s" if rate else "—",
                  Text("ETA", style="dim"), _dur(eta) if state == "building" else "—")
    if st.get("model"):
        stats.add_row(Text("model", style="dim"), str(st["model"]), Text("pid", style="dim"), str(st.get("pid", "—")))
    bar = ProgressBar(total=max(total, 1), completed=done, width=None,
                      complete_style=colour if colour != "dim" else "white", finished_style="green")
    return Panel(Group(head, Text(""), bar, Text(""), stats), border_style=colour, padding=(1, 2))


def show_index_status(get_status: Callable[[], dict], *, watch: bool = False, console: Console | None = None,
                      interval: float = 2.0) -> dict:
    console = console or Console()
    st = get_status()
    if not watch:
        console.print(render(st))
        return st
    hist: list[tuple[float, int]] = []
    with Live(render(st), console=console, refresh_per_second=4, transient=False) as live:
        while True:
            st = get_status()
            now = time.time()
            hist = [(t, d) for t, d in hist if now - t <= 120] + [(now, int(st.get("done", 0)))]
            rate = None
            if len(hist) >= 2 and hist[-1][0] > hist[0][0] and hist[-1][1] > hist[0][1]:
                rate = (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0])
            live.update(render(st, rate))
            if st.get("state") != "building" or not st.get("running"):
                return st
            time.sleep(interval)
