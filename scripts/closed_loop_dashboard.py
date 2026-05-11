#!/usr/bin/env python3
"""Pretty terminal dashboard for the regression-only bench
(closed-loop control blocks + sine / multilevel circuits).

Companion to ``scripts/parity_dashboard.py``: where that one wraps
benchmark_ngspice.py for SPICE-parity tests, this one wraps
benchmark_runner.py for tests that validate against a captured
Pulsim baseline.  The presentation is intentionally the same
style so the two dashboards feel like a pair.

Auto-discovers any benchmark in the manifest that has a
`validation: type: reference` block and no `ngspice_netlist`
mapping — i.e. anything that's regression-only against a
captured Pulsim baseline. This currently covers Phase 19
(closed-loop / control-block coverage) and Phase 20 (sine
input/output + multilevel converters).

Usage::

    # All closed-loop benches:
    python scripts/closed_loop_dashboard.py

    # Specific ones:
    python scripts/closed_loop_dashboard.py --only cl_buck_pi cl_buck_pid

    # Re-baseline (when you've intentionally changed a YAML):
    python scripts/closed_loop_dashboard.py --regenerate

The runner emits the usual JSON / CSV under
``benchmarks/closed_loop_dashboard_out/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "benchmarks.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "closed_loop_dashboard_out"
RUNNER = REPO_ROOT / "benchmarks" / "benchmark_runner.py"

# Default set: every benchmark in the manifest that has no
# ngspice netlist (i.e. validates against a captured Pulsim
# baseline). We probe each entry's circuit YAML for a
# `validation: type: reference` block to confirm.


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _discover_closed_loop_ids(manifest_path: Path) -> List[str]:
    """Return the list of benchmark IDs that validate against a
    captured Pulsim baseline (no ngspice reference). We look for:
      - manifest entries with no `ngspice_netlist` field, AND
      - circuit YAMLs with `validation: type: reference`.
    """
    manifest = _load_yaml(manifest_path)
    base_dir = manifest_path.parent
    ids: List[str] = []
    for entry in manifest.get("benchmarks", []) or []:
        path = entry.get("path", "")
        if not path:
            continue
        if entry.get("ngspice_netlist") or entry.get("ltspice_netlist"):
            continue  # has an external SPICE reference — handled by parity_dashboard
        # Open the circuit YAML and check the validation type
        circuit_path = (base_dir / path).resolve()
        if not circuit_path.exists():
            continue
        try:
            circuit = _load_yaml(circuit_path)
        except Exception:
            continue
        validation = circuit.get("benchmark", {}).get("validation", {}) or {}
        if validation.get("type") == "reference":
            ids.append(circuit_path.stem)
    return ids


def _ensure_pulsim_visible(env: Dict[str, str]) -> None:
    """Mirror the build_py-ABI guard from parity_dashboard.py so the
    spawned runner sees a compatible pulsim. Without it, a stale
    cross-version `.so` in build_py/python silently shadows the real
    install and every test fails with 'pulsim not available'."""
    if "PYTHONPATH" in env:
        return
    local_build = REPO_ROOT / "build_py" / "python"
    if not local_build.exists():
        return
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    compatible = list((local_build / "pulsim").glob(f"_pulsim*{ext_suffix}"))
    if compatible:
        env["PYTHONPATH"] = str(local_build)


def _run_benchmarks(
    bench_ids: List[str],
    output_dir: Path,
    regenerate: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        sys.executable,
        str(RUNNER),
        "--only", *bench_ids,
        "--output-dir", str(output_dir),
    ]
    if regenerate:
        cmd.append("--generate-baselines")
    env = os.environ.copy()
    _ensure_pulsim_visible(env)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    results_json = output_dir / "results.json"
    if not results_json.exists():
        return {"results": [], "summary": {"passed": 0, "failed": 0, "skipped": 0, "baseline": 0}}
    return json.loads(results_json.read_text())


# --- Pretty output (rich if available, plain otherwise) -----------------
_STATUS_ICONS = {
    "passed":   ("✓", "green"),
    "failed":   ("✗", "red"),
    "skipped":  ("○", "yellow"),
    "baseline": ("◐", "cyan"),
    "error":    ("!", "red"),
}


def _fmt_sci(value: Optional[float]) -> str:
    if value is None or value != value:
        return "—"
    try:
        return f"{float(value):.3e}"
    except (TypeError, ValueError):
        return "—"


def _fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 1000:.1f}"
    except (TypeError, ValueError):
        return "—"


def _collect_kpi_keys(items: List[Dict[str, Any]]) -> List[str]:
    """Return the union of KPI keys present in any item's `kpis` dict,
    sorted for stable column ordering."""
    keys = set()
    for it in items:
        kpis = it.get("kpis") or {}
        if isinstance(kpis, dict):
            keys.update(kpis.keys())
    return sorted(keys)


def _short_kpi_label(key: str) -> str:
    """Trim the long internal `kpi__<metric>__<label>` form into something
    that fits in a terminal column."""
    if key.startswith("kpi__"):
        rest = key[len("kpi__"):]
        parts = rest.split("__", 1)
        # `metric__label` → "label.metric" (more readable in a header)
        if len(parts) == 2:
            metric, label = parts
            return f"{label}/{metric}"[:18]
        return rest[:18]
    return key[:18]


def _print_plain(items: List[Dict[str, Any]], summary: Dict[str, Any], title: str, show_kpis: bool = True) -> None:
    print()
    print("=" * 110)
    print(title)
    print("-" * 110)
    header = f"  {'benchmark':30s} {'status':10s} {'max_err':>12s} {'thr_max':>10s} {'ss_err':>12s} {'thr_ss':>10s} {'runtime':>9s} note"
    print(header)
    print("-" * 110)
    for it in items:
        bench = str(it.get("benchmark_id", ""))[:30]
        status = str(it.get("status", "?"))
        icon, _ = _STATUS_ICONS.get(status, ("?", "white"))
        max_err = _fmt_sci(it.get("max_error"))
        ss_err = _fmt_sci(it.get("steady_state_max_error"))
        thr_max = _fmt_sci(it.get("max_error_threshold"))
        thr_ss = _fmt_sci(it.get("steady_state_max_error_threshold"))
        ms = _fmt_ms(it.get("runtime_s"))
        msg = (str(it.get("message", "")) or "")[:30]
        print(f"{icon} {bench:30s} {status:10s} {max_err:>12s} {thr_max:>10s} {ss_err:>12s} {thr_ss:>10s} {ms:>9s} {msg}")

        # KPIs (Phase 23): if this row carries any KPIs, render them on a
        # continuation line under the benchmark name.
        if show_kpis:
            kpis = it.get("kpis") or {}
            if isinstance(kpis, dict) and kpis:
                kpi_str = "  ".join(
                    f"{_short_kpi_label(k)}={v:.3g}"
                    for k, v in sorted(kpis.items())
                )
                print(f"    └─ KPIs: {kpi_str}")
    print("-" * 110)
    total = summary.get("passed", 0) + summary.get("failed", 0) + summary.get("skipped", 0) + summary.get("baseline", 0)
    rate = (100.0 * summary.get("passed", 0) / total) if total else 0.0
    print(
        f"Result: {summary.get('passed', 0)}/{total} passed   "
        f"failed={summary.get('failed', 0)}   skipped={summary.get('skipped', 0)}   "
        f"baseline={summary.get('baseline', 0)}   pass_rate={rate:.1f}%"
    )
    print()


def _print_rich(items: List[Dict[str, Any]], summary: Dict[str, Any], title: str, show_kpis: bool = True) -> None:
    try:
        from rich.console import Console  # type: ignore[import-not-found]
        from rich.table import Table       # type: ignore[import-not-found]
    except ImportError:
        _print_plain(items, summary, title)
        return

    console = Console()
    table = Table(title=title, header_style="bold cyan", show_lines=False, expand=True)
    table.add_column("benchmark", style="bold", no_wrap=True)
    table.add_column("status", justify="center")
    table.add_column("max_err", justify="right")
    table.add_column("thr_max", justify="right", style="dim")
    table.add_column("ss_err", justify="right")
    table.add_column("thr_ss", justify="right", style="dim")
    table.add_column("runtime (ms)", justify="right")
    table.add_column("note", style="dim")

    # KPI columns (Phase 23) — only added when at least one row carries a KPI
    kpi_keys = _collect_kpi_keys(items) if show_kpis else []
    for k in kpi_keys:
        table.add_column(_short_kpi_label(k), justify="right", style="green")

    for it in items:
        bench = str(it.get("benchmark_id", ""))
        status = str(it.get("status", "?"))
        icon, color = _STATUS_ICONS.get(status, ("?", "white"))
        row = [
            bench,
            f"[{color}]{icon} {status}[/{color}]",
            _fmt_sci(it.get("max_error")),
            _fmt_sci(it.get("max_error_threshold")),
            _fmt_sci(it.get("steady_state_max_error")),
            _fmt_sci(it.get("steady_state_max_error_threshold")),
            _fmt_ms(it.get("runtime_s")),
            (str(it.get("message", "")) or "")[:40],
        ]
        kpis = it.get("kpis") or {}
        for k in kpi_keys:
            v = kpis.get(k) if isinstance(kpis, dict) else None
            row.append("—" if v is None else f"{float(v):.3g}")
        table.add_row(*row)
    console.print(table)

    total = summary.get("passed", 0) + summary.get("failed", 0) + summary.get("skipped", 0) + summary.get("baseline", 0)
    rate = (100.0 * summary.get("passed", 0) / total) if total else 0.0
    console.print(
        f"\nResult: [bold green]{summary.get('passed', 0)}[/bold green]/{total} passed   "
        f"failed=[bold red]{summary.get('failed', 0)}[/bold red]   "
        f"skipped=[bold yellow]{summary.get('skipped', 0)}[/bold yellow]   "
        f"baseline=[bold cyan]{summary.get('baseline', 0)}[/bold cyan]   "
        f"pass_rate={rate:.1f}%"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help=f"Benchmark manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"Where the runner writes results (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Specific benchmark IDs (default: every cl_* in the manifest)",
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Re-capture baselines from the current Pulsim build",
    )
    parser.add_argument(
        "--plain", action="store_true",
        help="Force plain text output (no rich)",
    )
    args = parser.parse_args()

    bench_ids = args.only or _discover_closed_loop_ids(args.manifest)
    if not bench_ids:
        print("No closed-loop benchmarks found in manifest.")
        return 0

    title = "Pulsim — closed-loop / control-block dashboard"
    if args.regenerate:
        title += "   [regenerating baselines]"

    payload = _run_benchmarks(bench_ids, args.output_dir, args.regenerate)
    items = list(payload.get("results", []))
    summary = payload.get("summary", {}) or {
        "passed": sum(1 for it in items if it.get("status") == "passed"),
        "failed": sum(1 for it in items if it.get("status") == "failed"),
        "skipped": sum(1 for it in items if it.get("status") == "skipped"),
        "baseline": sum(1 for it in items if it.get("status") == "baseline"),
    }
    if args.plain:
        _print_plain(items, summary, title)
    else:
        _print_rich(items, summary, title)
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
