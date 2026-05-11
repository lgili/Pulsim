#!/usr/bin/env python3
"""Pretty terminal dashboard for the Pulsim vs SPICE parity bench.

Wraps ``benchmarks/benchmark_ngspice.py``: enumerates the manifest, runs
each circuit individually, and renders a live progress view with status
icons + a final summary table. Nothing under the hood is new — the
existing JSON / CSV artifacts are still produced (one per circuit, in
``--output-dir/<id>/``), so CI integrations keep working.

Usage::

    # Run the whole manifest with progress UI:
    python scripts/parity_dashboard.py

    # Filter to a few benchmarks:
    python scripts/parity_dashboard.py --only rc_step rlc_step

    # Quiet mode (exit code only — for CI / scripts):
    python scripts/parity_dashboard.py --quiet

    # Replay results without re-running:
    python scripts/parity_dashboard.py --output-dir benchmarks/parity_out --replay

The script auto-detects ``rich``; if it isn't installed we fall back to
plain ANSI output.

Designed to also handle the case where Pulsim is not the system pulsim
(set ``PYTHONPATH`` to the local build before running)::

    PYTHONPATH=build_py/python python scripts/parity_dashboard.py --only rc_step
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# ---- optional rich integration -------------------------------------------
try:
    from rich.console import Console
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text

    RICH = True
except ImportError:
    RICH = False
    Console = Live = Progress = Table = Text = None  # type: ignore

# ---- repo paths ----------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "benchmarks.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "parity_dashboard_out"
RUNNER = REPO_ROOT / "benchmarks" / "benchmark_ngspice.py"


@dataclass
class DashboardRow:
    """Per-circuit result we render in the live view + summary."""
    benchmark_id: str
    status: str = "pending"          # pending | running | passed | failed | skipped | error
    scenario: str = ""
    pulsim_runtime_s: float = 0.0
    spice_runtime_s: float = 0.0
    speedup: float | None = None
    max_error: float | None = None
    rms_error: float | None = None
    threshold_max: float | None = None
    failure_reason: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# manifest enumeration — kept self-contained so we don't have to import
# the giant benchmark_ngspice module just to list benchmarks.
# ---------------------------------------------------------------------------

def _read_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required to read the benchmark manifest. "
            "Install with: pip install pyyaml"
        ) from exc
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    with open(path) as fp:
        manifest = yaml.safe_load(fp)
    return list(manifest.get("benchmarks", []))


def _benchmark_id(entry: dict[str, Any], manifest_path: Path) -> str:
    """The benchmark id is in the circuit YAML's ``benchmark.id``; fall
    back to the file stem if absent."""
    rel = entry.get("path")
    if not rel:
        return "<unknown>"
    circuit_path = (manifest_path.parent / rel).resolve()
    try:
        import yaml
        with open(circuit_path) as fp:
            net = yaml.safe_load(fp)
        return str(net.get("benchmark", {}).get("id", circuit_path.stem))
    except Exception:
        return circuit_path.stem


def _circuit_threshold(entry: dict[str, Any], manifest_path: Path) -> float | None:
    """Pull ``benchmark.expectations.metrics.max_error`` from the circuit
    YAML so the dashboard can color-code "tight" vs "loose" passes."""
    try:
        import yaml
        circuit_path = (manifest_path.parent / entry["path"]).resolve()
        with open(circuit_path) as fp:
            net = yaml.safe_load(fp)
        m = net.get("benchmark", {}).get("expectations", {}).get("metrics", {})
        v = m.get("max_error")
        return float(v) if v is not None else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# subprocess invocation — one circuit per call, summary JSON parsed back.
# ---------------------------------------------------------------------------

def _run_one(
    benchmark_id: str,
    output_dir: Path,
    backend: str,
    extra_runner_args: list[str],
    verbose: bool,
) -> DashboardRow:
    """Invoke benchmark_ngspice.py for a single benchmark and return
    the parsed row.  Captures the runner's stdout/stderr (echoed when
    --verbose); the JSON file is the source of truth for metrics."""
    item_dir = output_dir / benchmark_id
    item_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(RUNNER),
        "--only", benchmark_id,
        "--output-dir", str(item_dir),
        "--backend", backend,
        *extra_runner_args,
    ]
    env = os.environ.copy()
    # Make the local build_py importable transparently — but ONLY if
    # the .so in it matches the current Python's ABI tag. Stale
    # cross-version artifacts (e.g. cp314 .so left over while running
    # Python 3.13) would shadow a working installed pulsim and break
    # the runner with "Python package 'pulsim' is not available".
    if "PYTHONPATH" not in env:
        local_build = REPO_ROOT / "build_py" / "python"
        if local_build.exists():
            import sysconfig
            ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
            compatible_sos = list((local_build / "pulsim").glob(f"_pulsim*{ext_suffix}"))
            if compatible_sos:
                env["PYTHONPATH"] = str(local_build)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    if verbose:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)

    summary_path = item_dir / f"{backend}_summary.json"
    results_path = item_dir / f"{backend}_results.json"

    # --- happy path: results JSON exists ---
    if results_path.exists():
        try:
            payload = json.loads(results_path.read_text())
        except json.JSONDecodeError:
            return DashboardRow(
                benchmark_id=benchmark_id, status="error",
                message="malformed results JSON",
            )
        items = payload.get("results") or []
        if not items:
            return DashboardRow(
                benchmark_id=benchmark_id, status="skipped",
                message="no results emitted (likely missing SPICE netlist)",
            )
        # If the user requested matrix mode, multiple scenarios may run.
        # Pick the worst (highest max_error among non-skipped) as the
        # representative; show the count in the message.
        non_skipped = [it for it in items if it.get("status") != "skipped"]
        candidates = non_skipped if non_skipped else items
        worst = max(
            candidates,
            key=lambda it: (it.get("max_error") or 0.0)
            if it.get("status") != "passed" else (it.get("max_error") or 0.0),
        )
        row = DashboardRow(
            benchmark_id=benchmark_id,
            status=worst.get("status", "error"),
            scenario=worst.get("scenario", ""),
            pulsim_runtime_s=float(worst.get("pulsim_runtime_s", 0.0) or 0.0),
            spice_runtime_s=float(
                worst.get("reference_runtime_s")
                or worst.get("ngspice_runtime_s")
                or 0.0
            ),
            speedup=worst.get("speedup"),
            max_error=worst.get("max_error"),
            rms_error=worst.get("rms_error"),
            failure_reason=worst.get("failure_reason") or "",
            message=worst.get("message", ""),
        )
        if len(items) > 1:
            row.message = f"{len(non_skipped)}/{len(items)} scenarios — {row.scenario}"
        return row

    # --- runner failed before producing JSON ---
    msg = (proc.stderr or proc.stdout).strip().splitlines()[-1] if (proc.stderr or proc.stdout).strip() else "no output"
    return DashboardRow(
        benchmark_id=benchmark_id,
        status="error",
        message=msg[:120],
        failure_reason=f"runner_exited_{proc.returncode}",
    )


# ---------------------------------------------------------------------------
# rendering: rich path + plain-ASCII fallback.
# ---------------------------------------------------------------------------

STATUS_ICON = {
    "pending":  ("·", "dim"),
    "running":  ("…", "yellow"),
    "passed":   ("✓", "green"),
    "failed":   ("✗", "red"),
    "skipped":  ("○", "yellow"),
    "error":    ("!", "red"),
}


def _build_summary_table(rows: Iterable[DashboardRow], backend: str) -> Any:
    if not RICH:
        return None
    table = Table(title=f"Pulsim vs {backend} — parity summary",
                  title_justify="left",
                  expand=True)
    table.add_column("status", justify="center", width=6)
    table.add_column("benchmark", style="bold")
    table.add_column("scenario")
    table.add_column("max_error", justify="right")
    table.add_column("rms_error", justify="right")
    table.add_column("threshold", justify="right")
    table.add_column("Pulsim (ms)", justify="right")
    table.add_column(f"{backend} (ms)", justify="right")
    table.add_column("speedup", justify="right")
    table.add_column("note", overflow="fold")

    for r in rows:
        icon, color = STATUS_ICON.get(r.status, ("?", "white"))
        threshold = ("—" if r.threshold_max is None
                     else f"{r.threshold_max:.2e}")
        max_err = "—" if r.max_error is None else f"{r.max_error:.3e}"
        rms_err = "—" if r.rms_error is None else f"{r.rms_error:.3e}"
        speedup = "—" if r.speedup is None else f"{r.speedup:.2f}×"
        # Color-code max_error vs threshold
        max_text = Text(max_err)
        if r.max_error is not None and r.threshold_max is not None:
            ratio = r.max_error / max(r.threshold_max, 1e-30)
            if ratio >= 1.0:
                max_text.stylize("red bold")
            elif ratio >= 0.5:
                max_text.stylize("yellow")
            else:
                max_text.stylize("green")
        table.add_row(
            Text(icon, style=color),
            r.benchmark_id,
            r.scenario or "—",
            max_text,
            rms_err,
            threshold,
            f"{r.pulsim_runtime_s*1e3:.1f}" if r.pulsim_runtime_s else "—",
            f"{r.spice_runtime_s*1e3:.1f}" if r.spice_runtime_s else "—",
            speedup,
            r.failure_reason or r.message,
            style=color if r.status in ("failed", "error", "skipped") else None,
        )
    return table


def _ascii_summary(rows: list[DashboardRow], backend: str) -> str:
    cols = [
        ("",          1,   "<"),
        ("benchmark", 32,  "<"),
        ("status",    8,   "<"),
        ("max_err",   11,  ">"),
        ("rms_err",   11,  ">"),
        ("threshold", 10,  ">"),
        ("Pulsim ms", 10,  ">"),
        (f"{backend} ms", 10, ">"),
        ("speedup",   8,   ">"),
        ("note",      30,  "<"),
    ]
    out = []
    out.append("=" * sum(w for _, w, _ in cols) + "=" * (len(cols) - 1))
    out.append(f"Pulsim vs {backend} — parity summary")
    out.append("-" * (sum(w for _, w, _ in cols) + len(cols) - 1))
    header = " ".join(f"{name:{align}{w}}" for name, w, align in cols)
    out.append(header)
    out.append("-" * len(header))
    for r in rows:
        icon, _ = STATUS_ICON.get(r.status, ("?", ""))
        line = " ".join([
            icon,
            r.benchmark_id[:32].ljust(32),
            r.status.ljust(8),
            ("—" if r.max_error is None else f"{r.max_error:.3e}").rjust(11),
            ("—" if r.rms_error is None else f"{r.rms_error:.3e}").rjust(11),
            ("—" if r.threshold_max is None else f"{r.threshold_max:.2e}").rjust(10),
            (f"{r.pulsim_runtime_s*1e3:.1f}" if r.pulsim_runtime_s else "—").rjust(10),
            (f"{r.spice_runtime_s*1e3:.1f}" if r.spice_runtime_s else "—").rjust(10),
            ("—" if r.speedup is None else f"{r.speedup:.2f}x").rjust(8),
            (r.failure_reason or r.message)[:30],
        ])
        out.append(line)
    out.append("-" * len(header))
    return "\n".join(out)


def _aggregate(rows: list[DashboardRow]) -> dict:
    total = len(rows)
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    passed = by_status.get("passed", 0)
    pass_rate = (passed / total * 100.0) if total else 0.0
    max_errors = [r.max_error for r in rows if r.max_error is not None]
    return {
        "total":     total,
        "passed":    passed,
        "failed":    by_status.get("failed", 0),
        "skipped":   by_status.get("skipped", 0),
        "errors":    by_status.get("error", 0),
        "pass_rate": pass_rate,
        "p50_max_error": (
            sorted(max_errors)[len(max_errors) // 2] if max_errors else None
        ),
        "p99_max_error": (
            sorted(max_errors)[max(0, int(0.99 * len(max_errors)) - 1)]
            if max_errors else None
        ),
    }


# ---------------------------------------------------------------------------
# main flow.
# ---------------------------------------------------------------------------

def _run_with_rich(
    rows: list[DashboardRow],
    output_dir: Path,
    backend: str,
    extra_args: list[str],
    verbose: bool,
) -> None:
    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    overall = progress.add_task(
        f"Pulsim vs {backend} parity bench", total=len(rows))
    with progress:
        for row in rows:
            row.status = "running"
            progress.update(
                overall,
                description=f"Pulsim vs {backend} — running [{row.benchmark_id}]",
            )
            result = _run_one(
                row.benchmark_id, output_dir, backend, extra_args, verbose,
            )
            # preserve threshold (from manifest pre-population)
            result.threshold_max = row.threshold_max
            row.__dict__.update(result.__dict__)
            progress.advance(overall)
            icon, color = STATUS_ICON.get(row.status, ("?", "white"))
            console.print(
                f"  [{color}]{icon}[/{color}] {row.benchmark_id} "
                f"[dim]({row.scenario or '—'})[/dim]   "
                f"max_err={row.max_error if row.max_error is not None else '—':<14}"
                f"  {row.failure_reason or row.message}",
                highlight=False,
            )
    console.print()
    console.print(_build_summary_table(rows, backend))


def _run_with_ascii(
    rows: list[DashboardRow],
    output_dir: Path,
    backend: str,
    extra_args: list[str],
    verbose: bool,
) -> None:
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        sys.stdout.write(
            f"[{i:>2}/{total:>2}] running {row.benchmark_id} ... ")
        sys.stdout.flush()
        result = _run_one(
            row.benchmark_id, output_dir, backend, extra_args, verbose,
        )
        result.threshold_max = row.threshold_max
        row.__dict__.update(result.__dict__)
        icon, _ = STATUS_ICON.get(row.status, ("?", ""))
        max_err = "—" if row.max_error is None else f"{row.max_error:.3e}"
        sys.stdout.write(
            f"{icon} {row.status:<8} max_err={max_err:<12}  "
            f"{row.failure_reason or row.message}\n",
        )
        sys.stdout.flush()
    print()
    print(_ascii_summary(rows, backend))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="Path to benchmarks.yaml manifest")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Where per-benchmark JSON / CSV gets written")
    parser.add_argument("--backend", choices=["ngspice", "ltspice"],
                        default="ngspice")
    parser.add_argument("--only", nargs="+", metavar="ID",
                        help="Only run these benchmark ids")
    parser.add_argument("--matrix", action="store_true",
                        help="Run every scenario for each benchmark "
                             "(passed through to the runner)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress live UI; print only the final "
                             "summary line + use exit code")
    parser.add_argument("--verbose", action="store_true",
                        help="Echo runner stdout/stderr per benchmark")
    parser.add_argument("--rich", dest="force_rich", action="store_true",
                        help="Force the rich UI even if the terminal "
                             "looks non-interactive")
    parser.add_argument("--ascii", dest="force_ascii", action="store_true",
                        help="Force the plain ASCII UI")
    args = parser.parse_args()

    manifest = _read_manifest(args.manifest)
    if args.only:
        wanted = set(args.only)
        manifest = [
            entry for entry in manifest
            if _benchmark_id(entry, args.manifest) in wanted
        ]
        if not manifest:
            print(f"no manifest entries matched: {args.only}", file=sys.stderr)
            return 1
    rows = [
        DashboardRow(
            benchmark_id=_benchmark_id(entry, args.manifest),
            threshold_max=_circuit_threshold(entry, args.manifest),
        )
        for entry in manifest
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    extra_runner_args: list[str] = []
    if args.matrix:
        extra_runner_args.append("--matrix")

    use_rich = (RICH and not args.force_ascii
                and (args.force_rich or sys.stdout.isatty())
                and not args.quiet)

    t0 = time.perf_counter()
    if use_rich:
        _run_with_rich(rows, args.output_dir, args.backend,
                       extra_runner_args, args.verbose)
    elif args.quiet:
        for row in rows:
            r = _run_one(row.benchmark_id, args.output_dir, args.backend,
                         extra_runner_args, verbose=False)
            r.threshold_max = row.threshold_max
            row.__dict__.update(r.__dict__)
    else:
        _run_with_ascii(rows, args.output_dir, args.backend,
                        extra_runner_args, args.verbose)
    elapsed = time.perf_counter() - t0

    agg = _aggregate(rows)
    one_line = (
        f"Result: {agg['passed']}/{agg['total']} passed   "
        f"failed={agg['failed']}   skipped={agg['skipped']}   "
        f"errors={agg['errors']}   pass_rate={agg['pass_rate']:.1f}%   "
        f"wall={elapsed:.1f}s"
    )
    if args.quiet:
        print(one_line)
    elif RICH and use_rich:
        Console().print(f"\n[bold]{one_line}[/bold]")
    else:
        print()
        print(one_line)

    # Persist a top-level dashboard summary (replay-friendly).
    summary_path = args.output_dir / "dashboard_summary.json"
    summary_path.write_text(json.dumps({
        "backend":    args.backend,
        "elapsed_s":  elapsed,
        "aggregate":  agg,
        "rows": [row.__dict__ for row in rows],
    }, indent=2))

    # Exit code: pass iff every non-skipped row passed.
    non_skipped = [r for r in rows if r.status not in ("skipped", "pending")]
    bad = [r for r in non_skipped if r.status not in ("passed",)]
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
