"""Shared terminal-output helpers for the Pulsim benchmark suite.

This module is the dev-side visual layer ("Onda 1" of the bench-tool
refresh). It uses `rich` when available for colored progress bars,
status streaming, and a final results table. When `rich` is not
installed, **everything falls back to plain `print()` so CI and
minimal environments keep working without complaint**.

Public surface (consumed by `benchmark_runner.py`, the other legacy
runners in `benchmarks/`, and `pulsim-bench` subcommands in
`tools/bench/`):

    make_console(record: bool = False, force_plain: bool = False)
    print_environment_header(console, *, package_version=None, extra=None)
    BenchProgress(console, total, *, enabled=True)        # context manager
    format_status(status: str) -> str                     # rich markup or plain
    print_results_table(console, results, *, title=...)
    print_results_summary(console, results, *, runtime_s=None)

Behavior flags (env vars, evaluated lazily at console-construction time):
    PULSIM_BENCH_PLAIN=1   force plain text (no rich, no color)
    NO_COLOR=1             standard — rich respects this natively

Design notes:
    * No mandatory rich import. Everything that touches rich is guarded.
    * `BenchProgress.case_done(...)` streams a colored one-liner per
      completed case, so users see incremental progress instead of a
      silent run that dumps a JSON at the end.
    * Status colors: green=passed, red=failed, yellow=baseline,
      dim=skipped, magenta=anything else.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

try:
    from rich.console import Console as _RichConsole
    from rich.panel import Panel as _RichPanel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress as _RichProgress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table as _RichTable
    from rich.text import Text as _RichText

    _HAVE_RICH = True
except ImportError:  # pragma: no cover - exercised in plain-mode runs
    _HAVE_RICH = False
    if TYPE_CHECKING:
        from rich.console import Console as _RichConsole
        from rich.panel import Panel as _RichPanel
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress as _RichProgress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
        from rich.table import Table as _RichTable
        from rich.text import Text as _RichText


# =============================================================================
# Status taxonomy
# =============================================================================

_STATUS_COLOR = {
    "passed": "green",
    "pass": "green",
    "failed": "red",
    "fail": "red",
    "baseline": "yellow",
    "base": "yellow",
    "skipped": "dim",
    "skip": "dim",
}

_STATUS_GLYPH = {
    "passed": "✓",
    "pass": "✓",
    "failed": "✗",
    "fail": "✗",
    "baseline": "△",
    "base": "△",
    "skipped": "⊘",
    "skip": "⊘",
}

_STATUS_LABEL = {
    "passed": "PASS",
    "pass": "PASS",
    "failed": "FAIL",
    "fail": "FAIL",
    "baseline": "BASE",
    "base": "BASE",
    "skipped": "SKIP",
    "skip": "SKIP",
}


def _norm_status(status: Any) -> str:
    return str(status or "").strip().lower()


def format_status(status: Any, *, rich: bool = True) -> str:
    """Return a status cell as rich markup (`[green]✓ PASS[/green]`) or plain
    text (`PASS`). `rich=False` is used when emitting to non-TTY sinks."""
    s = _norm_status(status)
    label = _STATUS_LABEL.get(s, s.upper() if s else "-")
    glyph = _STATUS_GLYPH.get(s, "")
    color = _STATUS_COLOR.get(s)
    cell = f"{glyph} {label}".strip()
    if rich and color and _HAVE_RICH:
        return f"[{color}]{cell}[/{color}]"
    return cell


# =============================================================================
# Console factory
# =============================================================================


def _plain_mode_requested() -> bool:
    return os.environ.get("PULSIM_BENCH_PLAIN", "0") == "1"


class _PlainConsole:
    """Tiny stand-in for rich.console.Console used when rich is absent
    or `PULSIM_BENCH_PLAIN=1`. Only the methods our helpers actually
    call are implemented."""

    def __init__(self) -> None:
        self.is_terminal = sys.stdout.isatty()
        self.record = False

    def print(self, *args: Any, **kwargs: Any) -> None:
        # Strip kwargs that don't map onto plain print().
        kwargs.pop("style", None)
        kwargs.pop("highlight", None)
        kwargs.pop("markup", None)
        kwargs.pop("overflow", None)
        kwargs.pop("no_wrap", None)
        end = kwargs.pop("end", "\n")
        # Replace rich.text.Text or other renderables with str().
        rendered = [str(a) for a in args]
        sys.stdout.write(" ".join(rendered) + end)
        sys.stdout.flush()

    def rule(self, title: str = "", **_: Any) -> None:
        cols = shutil.get_terminal_size((80, 20)).columns
        if title:
            pad = max(2, (cols - len(title) - 2) // 2)
            sys.stdout.write("─" * pad + " " + title + " " + "─" * pad + "\n")
        else:
            sys.stdout.write("─" * cols + "\n")
        sys.stdout.flush()

    # Methods used by the rich-only paths — no-op here.
    def log(self, *args: Any, **kwargs: Any) -> None:
        self.print(*args, **kwargs)


def make_console(*, record: bool = False, force_plain: bool = False):
    """Construct a console suitable for the current environment.

    Returns either a `rich.console.Console` or a `_PlainConsole`
    stub. Callers don't need to special-case: every helper in this
    module accepts both.
    """
    if force_plain or _plain_mode_requested() or not _HAVE_RICH:
        return _PlainConsole()
    # rich respects NO_COLOR and FORCE_COLOR natively; we just pass
    # `record=True` through when the caller wants to export an HTML
    # transcript later.
    return _RichConsole(record=record, highlight=False)


def is_rich_console(console: Any) -> bool:
    return _HAVE_RICH and isinstance(console, _RichConsole)


# =============================================================================
# Environment header
# =============================================================================


def _git_describe(repo: Optional[str] = None) -> Dict[str, str]:
    """Return {branch, sha, dirty} or empty dict if not in a git repo."""
    out: Dict[str, str] = {}
    cwd = repo if repo else None
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=cwd,
        ).strip()
        out["sha"] = sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return out
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=cwd,
        ).strip()
        out["branch"] = branch
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=cwd,
        ).strip()
        out["dirty"] = "yes" if dirty else "no"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return out


def _detect_pulsim_version() -> Optional[str]:
    try:
        import pulsim  # type: ignore[import-not-found]

        return getattr(pulsim, "__version__", None)
    except Exception:
        return None


def print_environment_header(
    console: Any,
    *,
    title: str = "Pulsim Bench",
    package_version: Optional[str] = None,
    extra: Optional[Dict[str, str]] = None,
) -> None:
    """Print a tidy panel summarizing git/host/runtime context."""
    git = _git_describe()
    rows: List[tuple[str, str]] = []
    branch = git.get("branch", "?")
    sha = git.get("sha", "?")
    dirty = git.get("dirty", "?")
    rows.append(("git", f"{branch} @ {sha}" + (" (dirty)" if dirty == "yes" else "")))
    rows.append(("host", f"{socket.gethostname()}  {platform.system()} {platform.release()}  {os.cpu_count()} cores"))
    rows.append(("python", f"{platform.python_version()}  ({sys.executable})"))
    version = package_version or _detect_pulsim_version() or "?"
    rows.append(("pulsim", f"{version}"))
    if extra:
        for k, v in extra.items():
            rows.append((k, str(v)))

    if is_rich_console(console):
        body = _RichText()
        keylen = max(len(k) for k, _ in rows)
        for k, v in rows:
            body.append(f"{k.ljust(keylen)} ", style="bold cyan")
            body.append(f"{v}\n", style="white")
        console.print(_RichPanel(body, title=title, title_align="left", border_style="cyan"))
    else:
        console.rule(title)
        for k, v in rows:
            console.print(f"  {k:<8} {v}")
        console.rule()


# =============================================================================
# Progress
# =============================================================================


@dataclass
class _CaseRow:
    benchmark_id: str
    scenario: str
    status: str
    runtime_s: float
    max_error: Optional[float] = None
    message: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)


class BenchProgress:
    """Context manager that drives a live progress bar (rich) or
    incremental `[i/N]` lines (plain). Use it like:

        with BenchProgress(console, total=89) as prog:
            for case in cases:
                prog.case_start(f"{case.bench} · {case.scenario}")
                result = run(case)
                prog.case_done(
                    benchmark_id=case.bench,
                    scenario=case.scenario,
                    status=result.status,
                    runtime_s=result.runtime_s,
                    max_error=result.max_error,
                )
    """

    def __init__(
        self,
        console: Any,
        total: int,
        *,
        enabled: bool = True,
        description: str = "Running",
    ) -> None:
        self.console = console
        self.total = max(0, int(total))
        self.enabled = enabled and self.total > 0
        self.description = description
        self._rich: Optional[Any] = None
        self._overall_task: Optional[int] = None
        self._current_task: Optional[int] = None
        self._done = 0
        self._start_time: float = 0.0
        self._rows: List[_CaseRow] = []

    # ------------------------------------------------------------- enter/exit
    def __enter__(self) -> "BenchProgress":
        self._start_time = time.monotonic()
        if self.enabled and is_rich_console(self.console):
            self._rich = _RichProgress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("ETA"),
                TimeRemainingColumn(),
                console=self.console,
                transient=False,
            )
            self._rich.__enter__()
            self._overall_task = self._rich.add_task(self.description, total=self.total)
            self._current_task = self._rich.add_task("idle", total=1, start=False)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._rich is not None:
            self._rich.__exit__(*exc)
            self._rich = None

    # ------------------------------------------------------------------ API
    def case_start(self, label: str) -> None:
        if self._rich is not None and self._current_task is not None:
            self._rich.reset(self._current_task, total=1, description=label, start=True)
        # Plain mode doesn't print anything here — `case_done` carries the line.

    def case_done(
        self,
        *,
        benchmark_id: str,
        scenario: str,
        status: str,
        runtime_s: float,
        max_error: Optional[float] = None,
        message: str = "",
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._done += 1
        row = _CaseRow(
            benchmark_id=benchmark_id,
            scenario=scenario,
            status=_norm_status(status),
            runtime_s=float(runtime_s or 0.0),
            max_error=max_error,
            message=message or "",
            extras=dict(extras or {}),
        )
        self._rows.append(row)
        line = self._format_case_line(row, rich=is_rich_console(self.console))
        if self._rich is not None and self._overall_task is not None:
            self._rich.console.print(line)
            self._rich.advance(self._overall_task, 1)
            if self._current_task is not None:
                self._rich.update(self._current_task, completed=1)
        else:
            prefix = f"[{self._done:>3}/{self.total}]"
            self.console.print(f"{prefix} {line}")

    @property
    def rows(self) -> List[_CaseRow]:
        return list(self._rows)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._start_time if self._start_time else 0.0

    # -------------------------------------------------------------- internals
    @staticmethod
    def _format_case_line(row: _CaseRow, *, rich: bool) -> str:
        status_cell = format_status(row.status, rich=rich)
        name = f"{row.benchmark_id} · {row.scenario}"
        runtime = f"{row.runtime_s:7.3f}s"
        err = "" if row.max_error is None else f"  max_err {row.max_error:.2e}"
        msg = ""
        if row.message and row.status in {"failed", "fail"}:
            short = row.message.strip().splitlines()[0]
            if len(short) > 60:
                short = short[:57] + "..."
            msg = f"  [dim]{short}[/dim]" if rich else f"  {short}"
        if rich:
            return f"  {status_cell}  [bold]{name:<48}[/bold]  {runtime}{err}{msg}"
        return f"  {status_cell}  {name:<48}  {runtime}{err}{msg}"


# =============================================================================
# Final table + summary
# =============================================================================


_DEFAULT_COLUMNS: Sequence[tuple[str, str]] = (
    ("benchmark_id", "benchmark"),
    ("scenario", "scenario"),
    ("status", "status"),
    ("runtime_s", "runtime[s]"),
    ("max_error", "max_err"),
    ("steps", "steps"),
)


def _as_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "__dict__"):
        return dict(item.__dict__)
    if hasattr(item, "_asdict"):
        return dict(item._asdict())
    return {}


def _fmt_cell(value: Any, *, kind: str = "auto") -> str:
    if value is None or value == "":
        return "-"
    if kind == "sci":
        try:
            return f"{float(value):.2e}"
        except (TypeError, ValueError):
            return str(value)
    if kind == "float":
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)
    if kind == "int":
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def print_results_table(
    console: Any,
    results: Iterable[Any],
    *,
    title: str = "Results",
    columns: Optional[Sequence[tuple[str, str]]] = None,
) -> None:
    """Render a final results table. `results` may be a list of dicts,
    dataclass instances, or any object with `__dict__`."""
    columns = columns or _DEFAULT_COLUMNS
    rows = [_as_dict(r) for r in results]

    if is_rich_console(console):
        table = _RichTable(title=title, header_style="bold cyan", border_style="dim")
        for field_name, header in columns:
            justify = "right" if field_name in {"runtime_s", "max_error", "steps"} else "left"
            table.add_column(header, justify=justify, no_wrap=(field_name == "status"))
        for r in rows:
            cells = []
            for field_name, _header in columns:
                raw = r.get(field_name)
                if field_name == "status":
                    cells.append(format_status(raw, rich=True))
                elif field_name == "runtime_s":
                    cells.append(_fmt_cell(raw, kind="float"))
                elif field_name == "max_error":
                    cells.append(_fmt_cell(raw, kind="sci"))
                elif field_name == "steps":
                    cells.append(_fmt_cell(raw, kind="int"))
                else:
                    cells.append(_fmt_cell(raw))
            table.add_row(*cells)
        console.print(table)
        return

    # Plain fallback: column widths computed once.
    headers = [h for _, h in columns]
    rendered_rows: List[List[str]] = []
    for r in rows:
        cells: List[str] = []
        for field_name, _h in columns:
            raw = r.get(field_name)
            if field_name == "status":
                cells.append(format_status(raw, rich=False))
            elif field_name == "runtime_s":
                cells.append(_fmt_cell(raw, kind="float"))
            elif field_name == "max_error":
                cells.append(_fmt_cell(raw, kind="sci"))
            elif field_name == "steps":
                cells.append(_fmt_cell(raw, kind="int"))
            else:
                cells.append(_fmt_cell(raw))
        rendered_rows.append(cells)
    widths = [len(h) for h in headers]
    for row in rendered_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    bar = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    console.print("")
    console.print(title)
    console.print("=" * len(title))
    console.print(bar)
    console.print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    console.print(bar.replace("-", "="))
    for row in rendered_rows:
        console.print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |")
    console.print(bar)


def print_results_summary(
    console: Any,
    results: Iterable[Any],
    *,
    runtime_s: Optional[float] = None,
    max_failures: int = 5,
) -> None:
    """Print pass/fail counts + top-N failure reasons. Always emitted,
    rich or plain."""
    rows = [_as_dict(r) for r in results]
    counts = {"passed": 0, "failed": 0, "baseline": 0, "skipped": 0}
    for r in rows:
        key = _norm_status(r.get("status"))
        if key in counts:
            counts[key] += 1
    total = len(rows)
    failed_rows = [r for r in rows if _norm_status(r.get("status")) in {"failed", "fail"}]

    if is_rich_console(console):
        text = _RichText()
        text.append("✓ passed  ", style="bold")
        text.append(f"{counts['passed']:>4}", style="green")
        text.append("    ✗ failed  ", style="bold")
        text.append(
            f"{counts['failed']:>4}",
            style="red" if counts["failed"] else "dim",
        )
        text.append("    △ baseline  ", style="bold")
        text.append(f"{counts['baseline']:>4}", style="yellow")
        text.append("    ⊘ skipped  ", style="bold")
        text.append(f"{counts['skipped']:>4}", style="dim")
        text.append(f"    total {total}", style="dim")
        if runtime_s is not None:
            text.append(f"    elapsed {runtime_s:.1f}s", style="dim")
        console.print(_RichPanel(text, title="Summary", title_align="left", border_style="cyan"))
        if failed_rows:
            console.print("[bold red]Failures[/bold red]:")
            for r in failed_rows[:max_failures]:
                name = f"{r.get('benchmark_id', '?')} · {r.get('scenario', '?')}"
                msg = (str(r.get("message", "")) or "").strip().splitlines()[:1]
                msg_str = msg[0] if msg else ""
                console.print(f"  [red]✗[/red] [bold]{name}[/bold]  [dim]{msg_str}[/dim]")
            if len(failed_rows) > max_failures:
                console.print(f"  [dim]… and {len(failed_rows) - max_failures} more[/dim]")
    else:
        console.print("")
        console.print("Summary")
        console.print("-------")
        console.print(
            f"  passed={counts['passed']}  failed={counts['failed']}  "
            f"baseline={counts['baseline']}  skipped={counts['skipped']}  total={total}"
        )
        if runtime_s is not None:
            console.print(f"  elapsed {runtime_s:.1f}s")
        if failed_rows:
            console.print("Failures:")
            for r in failed_rows[:max_failures]:
                name = f"{r.get('benchmark_id', '?')} · {r.get('scenario', '?')}"
                msg = (str(r.get("message", "")) or "").strip().splitlines()[:1]
                msg_str = msg[0] if msg else ""
                console.print(f"  - {name}  {msg_str}")
            if len(failed_rows) > max_failures:
                console.print(f"  … and {len(failed_rows) - max_failures} more")


# =============================================================================
# Optional context manager for one-shot scripts
# =============================================================================


@contextmanager
def bench_session(
    *,
    title: str = "Pulsim Bench",
    extra_env: Optional[Dict[str, str]] = None,
    record: bool = False,
):
    """Convenience: yields (console, start_time). Used by scripts that
    want the standard header + final elapsed line without writing the
    boilerplate."""
    console = make_console(record=record)
    print_environment_header(console, title=title, extra=extra_env)
    started = time.monotonic()
    try:
        yield console, started
    finally:
        elapsed = time.monotonic() - started
        if is_rich_console(console):
            console.print(f"[dim]Total wall time: {elapsed:.1f}s[/dim]")
        else:
            console.print(f"Total wall time: {elapsed:.1f}s")


# =============================================================================
# HTML / text export of recorded sessions
# =============================================================================


def save_html(console: Any, path: Any) -> bool:
    """Persist a recorded console session to `path` as HTML.

    Returns True on success, False when the console wasn't constructed
    with `record=True` or rich is unavailable. Callers are encouraged
    to check the return value and fall back to plain-text export if
    needed.

    `path` is accepted as `str | Path`; the directory is created if
    missing.
    """
    from pathlib import Path as _Path

    out = _Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not is_rich_console(console):
        return False
    if not getattr(console, "record", False):
        return False
    try:
        # `inline_styles=True` keeps the HTML self-contained — no
        # external CSS needed for the captured colors to render.
        console.save_html(str(out), inline_styles=True, theme=None)
        return True
    except Exception:
        return False


def save_text(console: Any, path: Any) -> bool:
    """Persist a recorded console session to `path` as plain text.

    Useful as a fallback when HTML export isn't possible. Same return
    semantics as `save_html`."""
    from pathlib import Path as _Path

    out = _Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not is_rich_console(console):
        return False
    if not getattr(console, "record", False):
        return False
    try:
        console.save_text(str(out))
        return True
    except Exception:
        return False
