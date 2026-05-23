"""Pulsim developer benchmark CLI — `pulsim-bench`.

A single typer-based entry point that unifies the historical
collection of standalone scripts under `benchmarks/`. Each subcommand
reuses the already-patched runner modules (Onda 1 + 1.5) so business
logic is shared and the legacy `benchmarks/*.py` invocations keep
working unchanged.

Subcommands (MVP):
    run            wraps benchmarks/benchmark_runner.py
    parity         wraps benchmarks/benchmark_ngspice.py
    stress         wraps benchmarks/stress_suite.py
    local-limit    wraps benchmarks/local_limit_suite.py

Usage examples:
    pulsim-bench run --only rc_step
    pulsim-bench parity --backend ngspice
    pulsim-bench stress --tier A
    pulsim-bench local-limit --mode both
    pulsim-bench --help
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import typer

from pulsim_bench._bootstrap import (
    benchmarks_dir,
    ensure_benchmarks_on_path,
)

app = typer.Typer(
    name="pulsim-bench",
    help="Pulsim developer benchmark CLI (run, parity, stress, local-limit).",
    no_args_is_help=True,
    add_completion=False,
)


# =============================================================================
# Helpers shared across subcommands
# =============================================================================


def _resolve_benchmarks_yaml(explicit: Optional[Path]) -> Path:
    """Return the YAML manifest to feed runners. Default = repo's
    `benchmarks/benchmarks.yaml`."""
    if explicit is not None:
        return explicit.resolve()
    return benchmarks_dir() / "benchmarks.yaml"


def _ensure_pulsim_backend() -> None:
    """Fail fast if the Pulsim runtime extension isn't importable.
    The legacy runners do the same; we mirror the message here so
    users get the error before any UI scaffolding renders."""
    ensure_benchmarks_on_path()
    try:
        from pulsim_python_backend import (  # type: ignore[import-not-found]
            availability_error,
        )
    except ImportError as exc:
        raise typer.BadParameter(
            f"Could not import pulsim_python_backend: {exc}. "
            "Make sure `benchmarks/` is reachable and the Pulsim C++ "
            "bindings are built."
        ) from exc
    reason = None
    try:
        reason = availability_error()
    except Exception as exc:
        reason = f"{exc.__class__.__name__}: {exc}"
    if reason:
        raise typer.BadParameter(
            "Pulsim Python runtime backend unavailable. Build bindings "
            f"and expose build/python on PYTHONPATH. Reason: {reason}"
        )


# =============================================================================
# `pulsim-bench run`
# =============================================================================


@app.command("run", help="Run the standard benchmark suite (per-scenario, with KPIs).")
def cmd_run(
    only: Optional[List[str]] = typer.Option(
        None, "--only", "-o", help="Restrict to these benchmark ids."
    ),
    scenario_filter: Optional[List[str]] = typer.Option(
        None, "--scenario", "-s", help="Restrict to these scenario names (repeatable)."
    ),
    matrix: bool = typer.Option(
        False, "--matrix", help="Run the full validation matrix instead of default scenarios."
    ),
    output_dir: Path = typer.Option(
        Path("benchmarks/out"), "--output-dir", "-O", help="Where artifacts are written."
    ),
    benchmarks: Optional[Path] = typer.Option(
        None, "--benchmarks", help="Custom benchmarks manifest YAML."
    ),
    generate_baselines: bool = typer.Option(
        False, "--generate-baselines", help="Create missing reference baselines."
    ),
    force_adaptive: bool = typer.Option(
        False, "--force-adaptive", help="Override simulation.step_mode=variable."
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Disable rich UI (header, progress, table)."
    ),
) -> None:
    _ensure_pulsim_backend()
    import benchmark_runner as br  # type: ignore[import-not-found]
    from _console import (  # type: ignore[import-not-found]
        BenchProgress,
        make_console,
        print_environment_header,
        print_results_summary,
        print_results_table,
    )

    bench_yaml = _resolve_benchmarks_yaml(benchmarks)
    console = None if quiet else make_console()
    if console is not None:
        print_environment_header(
            console,
            title="Pulsim Bench — run",
            extra={"manifest": str(bench_yaml)},
        )

    try:
        total = br.count_scenarios(
            bench_yaml,
            selected=only,
            matrix=matrix,
            scenario_filter=scenario_filter,
        )
    except Exception:
        total = 0

    elapsed = None
    if console is not None:
        with BenchProgress(console, total=total, description="benchmarks") as prog:
            results = br.run_benchmarks(
                bench_yaml,
                output_dir,
                selected=only,
                matrix=matrix,
                generate_baselines=generate_baselines,
                simulation_overrides=(
                    {"step_mode": "variable"} if force_adaptive else None
                ),
                scenario_filter=scenario_filter,
                progress=prog,
            )
            elapsed = prog.elapsed_s
    else:
        results = br.run_benchmarks(
            bench_yaml,
            output_dir,
            selected=only,
            matrix=matrix,
            generate_baselines=generate_baselines,
            simulation_overrides=(
                {"step_mode": "variable"} if force_adaptive else None
            ),
            scenario_filter=scenario_filter,
        )

    br.write_results(output_dir, results)

    if console is not None:
        print_results_table(console, results, title="Benchmark Results")
        print_results_summary(console, results, runtime_s=elapsed)

    fails = sum(1 for r in results if r.status == "failed")
    raise typer.Exit(code=1 if fails else 0)


# =============================================================================
# `pulsim-bench parity`
# =============================================================================


@app.command("parity", help="Compare Pulsim against an external SPICE backend (ngspice or LTspice).")
def cmd_parity(
    backend: str = typer.Option(
        "ngspice", "--backend", "-b", help="External backend.", case_sensitive=False,
    ),
    only: Optional[List[str]] = typer.Option(
        None, "--only", "-o", help="Restrict to these benchmark ids."
    ),
    matrix: bool = typer.Option(False, "--matrix", help="Run all scenarios."),
    scenario: Optional[str] = typer.Option(
        None, "--scenario", "-s", help="Run only this scenario."
    ),
    output_dir: Path = typer.Option(
        Path("benchmarks/parity_out"), "--output-dir", "-O", help="Where artifacts are written."
    ),
    benchmarks: Optional[Path] = typer.Option(
        None, "--benchmarks", help="Custom benchmarks manifest YAML."
    ),
    ngspice_exe: Optional[Path] = typer.Option(
        None, "--ngspice-exe", help="Path to ngspice executable."
    ),
    ltspice_exe: Optional[Path] = typer.Option(
        None, "--ltspice-exe", help="Path to LTspice executable."
    ),
    backend_arg: Optional[List[str]] = typer.Option(
        None, "--backend-arg", help="Extra args passed to the backend (repeatable)."
    ),
    observable: Optional[List[str]] = typer.Option(
        None, "--observable", help="Observable column to compare (repeatable)."
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Disable rich UI."
    ),
) -> None:
    if backend.lower() not in ("ngspice", "ltspice"):
        raise typer.BadParameter("--backend must be ngspice or ltspice")
    _ensure_pulsim_backend()
    import benchmark_ngspice as bp  # type: ignore[import-not-found]
    from _console import (  # type: ignore[import-not-found]
        BenchProgress,
        make_console,
        print_environment_header,
        print_results_summary,
        print_results_table,
    )

    bench_yaml = _resolve_benchmarks_yaml(benchmarks)
    backend_config = bp.resolve_backend_config(
        backend=backend,
        manifest=bp.load_yaml(bench_yaml),
        ngspice_executable=ngspice_exe,
        ltspice_executable=ltspice_exe,
        backend_args=backend_arg,
    )

    console = None if quiet else make_console()
    if console is not None:
        print_environment_header(
            console,
            title=f"Pulsim Bench — parity ({backend})",
            extra={
                "backend": backend,
                "executable": str(backend_config.executable) if backend_config.executable else "—",
            },
        )

    try:
        total = bp.count_parity_scenarios(
            bench_yaml, only=only, matrix=matrix, force_scenario=scenario,
        )
    except Exception:
        total = 0

    elapsed = None
    if console is not None:
        with BenchProgress(console, total=total, description=f"parity-{backend}") as prog:
            results = bp.run_manifest(
                manifest_path=bench_yaml,
                output_dir=output_dir,
                only=only,
                matrix=matrix,
                force_scenario=scenario,
                cli_observables=observable,
                backend=backend,
                ngspice_executable=ngspice_exe,
                ltspice_executable=ltspice_exe,
                backend_args=backend_arg,
                progress=prog,
            )
            elapsed = prog.elapsed_s
    else:
        results = bp.run_manifest(
            manifest_path=bench_yaml,
            output_dir=output_dir,
            only=only,
            matrix=matrix,
            force_scenario=scenario,
            cli_observables=observable,
            backend=backend,
            ngspice_executable=ngspice_exe,
            ltspice_executable=ltspice_exe,
            backend_args=backend_arg,
        )

    bp.write_results(
        output_dir=output_dir,
        results=results,
        backend=backend,
        executable=backend_config.executable,
    )

    if console is not None:
        rows = [
            {
                "benchmark_id": r.benchmark_id,
                "scenario": r.scenario,
                "status": r.status,
                "runtime_s": float(r.pulsim_runtime_s or 0.0),
                "max_error": r.max_error,
                "steps": int(r.pulsim_steps or 0),
                "message": r.message,
            }
            for r in results
        ]
        print_results_table(console, rows, title=f"Parity Results — {backend}")
        print_results_summary(console, rows, runtime_s=elapsed)

    fails = sum(1 for r in results if r.status == "failed")
    raise typer.Exit(code=1 if fails else 0)


# =============================================================================
# `pulsim-bench stress`
# =============================================================================


@app.command("stress", help="Run tiered stress validation (tiers A/B/C with pass criteria).")
def cmd_stress(
    tier: Optional[List[str]] = typer.Option(
        None, "--tier", "-t", help="Run only these tiers (repeatable)."
    ),
    output_dir: Path = typer.Option(
        Path("benchmarks/stress_out"), "--output-dir", "-O"
    ),
    benchmarks: Optional[Path] = typer.Option(
        None, "--benchmarks", help="Custom benchmarks manifest YAML."
    ),
    catalog: Optional[Path] = typer.Option(
        None, "--catalog", help="Custom stress catalog YAML."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    _ensure_pulsim_backend()
    import stress_suite as ss  # type: ignore[import-not-found]
    from _console import (  # type: ignore[import-not-found]
        BenchProgress,
        make_console,
        print_environment_header,
        print_results_summary,
        print_results_table,
    )

    bench_yaml = _resolve_benchmarks_yaml(benchmarks)
    catalog_yaml = catalog.resolve() if catalog else benchmarks_dir() / "stress_catalog.yaml"

    console = None if quiet else make_console()
    if console is not None:
        print_environment_header(
            console,
            title="Pulsim Bench — stress",
            extra={
                "catalog": str(catalog_yaml),
                "tiers": ",".join(tier) if tier else "all",
            },
        )

    try:
        total = ss.count_stress_scenarios(bench_yaml, catalog_yaml, selected_tiers=tier)
    except Exception:
        total = 0

    elapsed = None
    if console is not None:
        with BenchProgress(console, total=total, description="stress") as prog:
            tier_runs = ss.run_stress_suite(
                benchmarks_manifest_path=bench_yaml,
                stress_catalog_path=catalog_yaml,
                output_dir=output_dir,
                selected_tiers=tier,
                progress=prog,
            )
            elapsed = prog.elapsed_s
    else:
        tier_runs = ss.run_stress_suite(
            benchmarks_manifest_path=bench_yaml,
            stress_catalog_path=catalog_yaml,
            output_dir=output_dir,
            selected_tiers=tier,
        )
    ss.write_stress_artifacts(output_dir, tier_runs)

    if console is not None:
        tier_rows = [
            {
                "benchmark_id": tr.tier,
                "scenario": "(tier)",
                "status": tr.evaluation.status,
                "runtime_s": tr.evaluation.max_runtime_s_observed or 0.0,
                "max_error": tr.evaluation.max_max_error_observed,
                "steps": tr.evaluation.total,
                "message": tr.evaluation.message,
            }
            for tr in tier_runs
        ]
        print_results_table(console, tier_rows, title="Stress Tiers")
        all_rows = [
            {
                "benchmark_id": r.benchmark_id,
                "scenario": r.scenario,
                "status": r.status,
                "runtime_s": r.runtime_s,
                "max_error": r.max_error,
                "message": r.message,
            }
            for tr in tier_runs
            for r in tr.results
        ]
        print_results_summary(console, all_rows, runtime_s=elapsed)

    fails = sum(1 for tr in tier_runs if tr.evaluation.status == "failed")
    raise typer.Exit(code=1 if fails else 0)


# =============================================================================
# `pulsim-bench local-limit`
# =============================================================================


@app.command("show", help="Render a previous benchmark run from disk (file or directory).")
def cmd_show(
    path: Path = typer.Argument(..., help="Run directory or JSON artifact file."),
    title: Optional[str] = typer.Option(None, "--title", help="Override the table title."),
    export_html: Optional[Path] = typer.Option(
        None, "--export-html", help="Save the rendered output as a self-contained HTML file."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Reads `results.json` (or `parity_results.json` / `stress_results.json`)
    and re-renders the standard rich table + summary panel. Useful for
    inspecting an old run without re-executing it."""
    ensure_benchmarks_on_path()
    from _console import (  # type: ignore[import-not-found]
        make_console,
        print_environment_header,
        print_results_summary,
        print_results_table,
        save_html as _save_html,
    )

    from pulsim_bench._diff import load_results

    try:
        records, kind = load_results(path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    record_console = export_html is not None and not quiet
    console = make_console(force_plain=quiet, record=record_console)
    print_environment_header(
        console,
        title=title or f"Pulsim Bench — show ({kind})",
        extra={"source": str(Path(path).resolve())},
    )
    if not records:
        console.print(
            "[yellow]No comparable records in the artifact "
            "(empty `results` list).[/yellow]"
        )
    else:
        print_results_table(console, records, title=title or f"{kind.capitalize()} Results")
        print_results_summary(console, records)

    if export_html is not None:
        if _save_html(console, export_html):
            console.print(f"[dim]HTML written to {export_html}[/dim]")
        else:
            console.print(
                "[yellow]HTML export skipped — needs rich + record mode "
                "(install rich and avoid --quiet).[/yellow]"
            )


@app.command("compare", help="Show the delta between two benchmark runs.")
def cmd_compare(
    run_a: Path = typer.Argument(..., help="First run (file or directory)."),
    run_b: Path = typer.Argument(..., help="Second run (file or directory)."),
    label_a: str = typer.Option("A", "--label-a", help="Display label for run_a."),
    label_b: str = typer.Option("B", "--label-b", help="Display label for run_b."),
    only_changes: bool = typer.Option(
        False, "--only-changes",
        help="Hide rows that didn't change (still-passing / still-failing).",
    ),
    export_html: Optional[Path] = typer.Option(
        None, "--export-html", help="Save the rendered diff as a self-contained HTML file."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Compares two artifact files / directories and prints a colored
    delta table sorted with regressions on top. Exit code = 1 when
    any regression is detected (PASS/BASE → FAIL)."""
    ensure_benchmarks_on_path()
    from _console import (  # type: ignore[import-not-found]
        make_console,
        print_environment_header,
        save_html as _save_html,
    )

    from pulsim_bench._diff import (
        Transition,
        compute_diff,
        load_results,
    )

    try:
        records_a, kind_a = load_results(run_a)
        records_b, kind_b = load_results(run_b)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    rows, summary = compute_diff(records_a, records_b)
    if only_changes:
        rows = [
            r for r in rows
            if r.transition not in (Transition.STILL_PASSING,)
        ]

    record_console = export_html is not None and not quiet
    console = make_console(force_plain=quiet, record=record_console)
    print_environment_header(
        console,
        title=f"Pulsim Bench — compare ({label_a} → {label_b})",
        extra={
            f"{label_a}": f"{run_a}  [{kind_a}]",
            f"{label_b}": f"{run_b}  [{kind_b}]",
        },
    )

    if not rows:
        console.print("[yellow]No comparable records found.[/yellow]")
    else:
        _render_diff_table(console, rows, label_a, label_b)
    _render_diff_summary(console, summary, label_a, label_b)

    if export_html is not None:
        if _save_html(console, export_html):
            console.print(f"[dim]HTML written to {export_html}[/dim]")
        else:
            console.print(
                "[yellow]HTML export skipped — needs rich + record mode.[/yellow]"
            )

    raise typer.Exit(code=1 if summary.regressed > 0 else 0)


# ---------------------------------------------------------------------------
# Diff rendering helpers (compare-specific; small enough to live inline)
# ---------------------------------------------------------------------------


_TRANSITION_STYLE = {
    "regressed":      ("red",     "✗→", "REGRESSED"),
    "fixed":          ("green",   "→✓", "FIXED"),
    "still_failing":  ("red",     "==", "FAIL=="),
    "still_passing":  ("dim",     "==", "OK=="),
    "skipped_now":    ("yellow",  "⊘",  "SKIP"),
    "new":            ("cyan",    "+",  "NEW"),
    "removed":        ("magenta", "-",  "REMOVED"),
    "other":          ("white",   "~",  "OTHER"),
}


def _fmt_runtime_delta(pct: Optional[float]) -> str:
    if pct is None:
        return "-"
    if abs(pct) < 0.05:
        return "  ~0%"
    arrow = "↑" if pct > 0 else "↓"
    color = "red" if pct > 5 else ("green" if pct < -5 else "dim")
    return f"[{color}]{arrow}{abs(pct):5.1f}%[/{color}]"


def _fmt_value(v: Optional[float], kind: str) -> str:
    if v is None:
        return "-"
    if kind == "runtime":
        return f"{v:6.3f}"
    if kind == "err":
        return f"{v:.2e}"
    return str(v)


def _render_diff_table(console: Any, rows: list, label_a: str, label_b: str) -> None:
    # Reuse rich Table; the typer dep already pulls rich in.
    from _console import _RichTable, format_status  # type: ignore[import-not-found]

    if not is_rich_console_safe(console):
        # Plain fallback: condensed list per row.
        for r in rows:
            color, glyph, label = _TRANSITION_STYLE.get(r.transition.value, _TRANSITION_STYLE["other"])
            pct = r.runtime_delta_pct
            pct_str = "n/a" if pct is None else f"{pct:+.1f}%"
            console.print(
                f"  {glyph} {label:<10} {r.benchmark_id} · {r.scenario}  "
                f"{label_a}={r.status_a}/{_fmt_value(r.runtime_a, 'runtime')}  "
                f"{label_b}={r.status_b}/{_fmt_value(r.runtime_b, 'runtime')}  "
                f"Δrt={pct_str}"
            )
        return

    table = _RichTable(title="Diff", header_style="bold cyan", border_style="dim")
    table.add_column("change", no_wrap=True)
    table.add_column("benchmark", no_wrap=True)
    table.add_column("scenario", no_wrap=True)
    table.add_column(f"{label_a} status", no_wrap=True)
    table.add_column(f"{label_b} status", no_wrap=True)
    table.add_column(f"{label_a} runtime", justify="right")
    table.add_column(f"{label_b} runtime", justify="right")
    table.add_column("Δ runtime", justify="right")
    table.add_column(f"{label_a} max_err", justify="right")
    table.add_column(f"{label_b} max_err", justify="right")

    for r in rows:
        color, glyph, label = _TRANSITION_STYLE.get(r.transition.value, _TRANSITION_STYLE["other"])
        table.add_row(
            f"[{color}]{glyph} {label}[/{color}]",
            r.benchmark_id,
            r.scenario,
            format_status(r.status_a, rich=True) if r.status_a else "[dim]-[/dim]",
            format_status(r.status_b, rich=True) if r.status_b else "[dim]-[/dim]",
            _fmt_value(r.runtime_a, "runtime"),
            _fmt_value(r.runtime_b, "runtime"),
            _fmt_runtime_delta(r.runtime_delta_pct),
            _fmt_value(r.max_error_a, "err"),
            _fmt_value(r.max_error_b, "err"),
        )
    console.print(table)


def _render_diff_summary(console: Any, summary: Any, label_a: str, label_b: str) -> None:
    from _console import _RichPanel, _RichText  # type: ignore[import-not-found]

    if not is_rich_console_safe(console):
        console.print("")
        console.print("Diff summary")
        console.print("------------")
        console.print(
            f"  regressed={summary.regressed}  fixed={summary.fixed}  "
            f"still_failing={summary.still_failing}  still_passing={summary.still_passing}  "
            f"new={summary.new}  removed={summary.removed}"
        )
        if summary.median_runtime_delta_pct is not None:
            console.print(f"  median Δruntime={summary.median_runtime_delta_pct:+.1f}%")
        console.print(f"  common cases between {label_a} and {label_b}: {summary.common}")
        return

    text = _RichText()
    text.append("✗→ regressed   ", style="bold")
    text.append(f"{summary.regressed:>3}", style="red" if summary.regressed else "dim")
    text.append("    →✓ fixed   ", style="bold")
    text.append(f"{summary.fixed:>3}", style="green" if summary.fixed else "dim")
    text.append("    == still failing   ", style="bold")
    text.append(f"{summary.still_failing:>3}", style="red" if summary.still_failing else "dim")
    text.append("    + new   ", style="bold")
    text.append(f"{summary.new:>3}", style="cyan" if summary.new else "dim")
    text.append("    − removed   ", style="bold")
    text.append(f"{summary.removed:>3}", style="magenta" if summary.removed else "dim")
    if summary.median_runtime_delta_pct is not None:
        pct = summary.median_runtime_delta_pct
        color = "red" if pct > 5 else ("green" if pct < -5 else "dim")
        text.append("\nmedian Δruntime  ", style="dim")
        text.append(f"{pct:+.1f}%", style=color)
        text.append(f"    common cases  {summary.common}", style="dim")
    else:
        text.append(f"\ncommon cases  {summary.common}", style="dim")
    console.print(_RichPanel(text, title="Diff Summary", title_align="left", border_style="cyan"))


def is_rich_console_safe(console: Any) -> bool:
    """Light wrapper so callers don't have to import is_rich_console
    from the legacy benchmarks/_console module before it's on path."""
    from _console import is_rich_console  # type: ignore[import-not-found]

    return is_rich_console(console)


@app.command("local-limit", help="Local fixed+variable limit suite (10 progressive circuits).")
def cmd_local_limit(
    only: Optional[List[str]] = typer.Option(
        None, "--only", "-o", help="Restrict to these benchmark ids."
    ),
    mode: str = typer.Option(
        "both", "--mode", "-m", help="fixed | variable | both",
    ),
    duration_scale: float = typer.Option(
        1.0, "--duration-scale", help="Multiply tstop by this factor."
    ),
    min_samples: int = typer.Option(8, "--min-samples"),
    min_completion: float = typer.Option(0.97, "--min-completion"),
    max_runtime_s: Optional[float] = typer.Option(None, "--max-runtime-s"),
    output_dir: Path = typer.Option(
        Path("benchmarks/out_local_limit"), "--output-dir", "-O"
    ),
    manifest: Optional[Path] = typer.Option(
        None, "--manifest", help="Custom local-limit manifest."
    ),
    list_circuits: bool = typer.Option(
        False, "--list-circuits", help="Print available benchmark ids and exit."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    if mode not in ("fixed", "variable", "both"):
        raise typer.BadParameter("--mode must be one of: fixed, variable, both")
    _ensure_pulsim_backend()
    import sys as _sys

    import local_limit_suite as ll  # type: ignore[import-not-found]

    # The legacy main() already does all the orchestration we want
    # (manifest materialization + progress wiring + evaluation + write).
    # Rather than duplicate the manifest-rewriting code, delegate to it
    # by patching sys.argv. Keeps DRY and inherits future changes.
    argv = [_sys.argv[0] if _sys.argv else "pulsim-bench-local-limit"]
    if manifest is not None:
        argv += ["--manifest", str(manifest)]
    argv += ["--output-dir", str(output_dir)]
    argv += ["--mode", mode]
    argv += ["--duration-scale", str(duration_scale)]
    argv += ["--min-samples", str(min_samples)]
    argv += ["--min-completion", str(min_completion)]
    if max_runtime_s is not None:
        argv += ["--max-runtime-s", str(max_runtime_s)]
    if list_circuits:
        argv += ["--list-circuits"]
    if only:
        argv += ["--only", *only]
    if quiet:
        argv += ["--quiet"]

    saved_argv = _sys.argv
    try:
        _sys.argv = argv
        code = ll.main()
    finally:
        _sys.argv = saved_argv
    raise typer.Exit(code=int(code or 0))


if __name__ == "__main__":  # pragma: no cover
    app()
