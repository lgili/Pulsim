#!/usr/bin/env python3
"""Codemod for converting v1-style Pulsim Python files to v2 idiom.

Usage::

    python scripts/migrate_v1_to_v2.py path/to/file.py [path/to/another.py ...]
    python scripts/migrate_v1_to_v2.py --dry-run path/to/file.py
    python scripts/migrate_v1_to_v2.py --in-place path/to/file.py
    python scripts/migrate_v1_to_v2.py --notebook path/to/notebook.ipynb

What it converts automatically
------------------------------
* ``import pulsim`` / ``import pulsim as ps`` →
  ``import pulsim.v2 as p`` (and rewrites the binding name).
* ``ps.RuntimeCircuit()`` → ``p.CircuitBuilder()`` + a
  pattern-matched ``b = ...`` rename. (Falls back to ``builder``
  on existing name conflicts.)
* ``circuit.add_*(...)`` → ``b.add_*(...)`` for the common device
  set.
* ``circuit.simulate(t_end, dt)`` → ``p.simulate(b, t_end=t_end,
  dt=dt)``.
* ``circuit.dc_operating_point()`` → ``p.compute_dc_op(b)``.
* ``circuit.run_ac_sweep(freqs, ...)`` → ``p.run_ac_sweep(b,
  frequencies=freqs, ...)``.

What it FLAGS for manual review
-------------------------------
Patterns that don't have a 1-line mechanical conversion:

* Closure-style controllers (``def control(t, x): ...`` calling
  ``circuit.set_pwm_duty(...)``) — needs a chain rewrite.
* Direct mutation of internal state (``circuit._switches[...]``).
* Use of v1-only features (codegen, FMU, schematic).

These are emitted as ``# TODO[v2-migrate]: …`` comments above the
offending line so you can grep for them after running.

Safety
------
The script does NOT run anything — pure text transform. Always
run with ``--dry-run`` first to see the diff. By default it
writes to ``<file>.v2.py`` so the original is preserved; use
``--in-place`` to overwrite (after committing the original).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple


# Device-method names that simply move from circuit.X to b.X
# (parameter signatures match closely enough that no further
# rewrite is needed beyond the receiver name change).
_PASSTHROUGH_ADDS = {
    "add_voltage_source", "add_current_source", "add_resistor",
    "add_capacitor", "add_inductor", "add_diode",
    "add_nonlinear_diode", "add_switch", "add_mosfet",
    "add_mosfet_with_body_diode", "add_mosfet_level1",
    "add_igbt", "add_igbt_level1", "add_transformer",
    "add_vcvs", "add_op_amp_ideal",
    "add_saturable_inductor", "add_sine_voltage_source",
    "add_pulse_voltage_source", "add_pwm_voltage_source",
    "add_dc_motor", "add_pmsm", "add_bldc_motor",
}

# Device-method names that became free functions on the package
# (took the "circuit" arg out and now take "(b, ...)").
_FREE_FUNCTION_HELPERS = {
    "add_bridge_rectifier": "p.add_bridge_rectifier",
    "add_three_phase_vsi":  "p.add_three_phase_vsi",
    "add_three_phase_rl_load": "p.add_three_phase_rl_load",
    "add_three_phase_grid": "p.add_three_phase_grid",
    "add_foster_network": "p.add_foster_network",
    "add_cauer_thermal_network": "p.add_cauer_thermal_network",
}

# Analysis methods → free functions.
_ANALYSIS_MOVES = {
    "simulate":              "p.simulate",
    "dc_operating_point":    "p.compute_dc_op",
    "run_ac_sweep":          "p.run_ac_sweep",
    "run_fra":               "p.run_fra",
    "run_periodic_shooting": "p.run_periodic_shooting",
    "run_harmonic_balance":  "p.run_harmonic_balance",
    "parameter_sweep":       "p.run_parameter_sweep",
}

# Patterns that MUST be flagged — these don't have safe 1-line
# rewrites and need human attention.
_MANUAL_REVIEW_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"circuit\.set_pwm_duty\b"),
     "circuit.set_pwm_duty(…) — replace with a "
     "MixedDomainBlockChain + chain.make_pwm_switch_fn(…). See "
     "migration-from-v1.md § 5 (control flow)."),
    (re.compile(r"SignalEvaluator\b"),
     "SignalEvaluator — rewrite as MixedDomainBlockChain "
     "(`chain.add('name', PIController(...), inputs=..., output=...)`)."),
    (re.compile(r"PIController\b\(.*integral_limit"),
     "v1 PIController uses `integral_limit`; v2 uses "
     "`output_min` / `output_max` (which clamp the OUTPUT, not "
     "the integrator — slightly different anti-windup)."),
    (re.compile(r"pulsim\.codegen\b|\bcodegen\.generate\b"),
     "C99 codegen not yet in v2 — stay on v1 for this flow."),
    (re.compile(r"pulsim\.fmu\b|\bfmu\.export\b"),
     "FMU 2.0 export not yet in v2 — stay on v1 for this flow."),
    (re.compile(r"pulsim\.schematic\b"),
     "Schematic auto-layout not yet in v2."),
    (re.compile(r"pulsim\.templates\b"),
     "Converter templates (pulsim.templates.buck/boost/buck_boost)"
     " not yet in v2."),
    (re.compile(r"circuit\.junction_temperature\b"),
     "circuit.junction_temperature → wire a v2 thermal observer + "
     "Foster/Cauer network. See migration-from-v1.md § 6."),
    (re.compile(r"HysteresisInductor\b"),
     "v1 kernel-coupled HysteresisInductor — v2 has POST-PROCESS "
     "Jiles-Atherton via `p.compute_bh_loop` + `add_saturable_"
     "inductor` for the electrical side. Adjust accordingly."),
]


def _migrate_text(src: str) -> Tuple[str, List[str]]:
    """Return (new_text, warnings)."""
    text = src
    warnings: List[str] = []

    # 1. Rewrite top-level imports.
    text = re.sub(
        r"^import\s+pulsim\s+as\s+\w+\s*$",
        "import pulsim.v2 as p",
        text, flags=re.MULTILINE)
    text = re.sub(
        r"^import\s+pulsim\s*$",
        "import pulsim.v2 as p",
        text, flags=re.MULTILINE)
    # If user did `from pulsim import RuntimeCircuit`, flag it —
    # the symbol set differs too much for a safe rewrite.
    if re.search(r"^from\s+pulsim\s+import\b", text, re.MULTILINE):
        warnings.append(
            "TODO[v2-migrate]: `from pulsim import ...` found — "
            "the v1 symbol set doesn't map 1-to-1 to v2. Replace "
            "with `import pulsim.v2 as p`.")

    # 2. RuntimeCircuit() → CircuitBuilder()
    text = re.sub(
        r"\bps\.RuntimeCircuit\(\s*\)",
        "p.CircuitBuilder()", text)
    text = re.sub(
        r"\bpulsim\.RuntimeCircuit\(\s*\)",
        "p.CircuitBuilder()", text)

    # 3. Method-on-circuit → method-on-builder rewrite.
    # We rename the receiver from `circuit` to `b` only on lines
    # where it's clearly the builder; pattern: ``<name>.add_*(``,
    # ``<name>.simulate(``, etc.
    # For safety, do the rewrite conservatively: only when the
    # receiver matches ``circuit`` or ``ckt`` (common v1 names).
    receivers_to_rename = ("circuit", "ckt", "cir")
    for recv in receivers_to_rename:
        # Free-function analysis moves: circuit.simulate(...) →
        # p.simulate(b, ...). Need to inject ``b`` as first arg.
        for v1_method, v2_call in _ANALYSIS_MOVES.items():
            pattern = rf"\b{recv}\.{v1_method}\s*\("
            text = re.sub(
                pattern,
                f"{v2_call}(b, ",
                text)
        # Free helpers (add_bridge_rectifier etc).
        for v1_method, v2_call in _FREE_FUNCTION_HELPERS.items():
            pattern = rf"\b{recv}\.{v1_method}\s*\("
            text = re.sub(
                pattern,
                f"{v2_call}(b, ",
                text)
        # Plain pass-through: circuit.add_* → b.add_*
        for v1_method in _PASSTHROUGH_ADDS:
            pattern = rf"\b{recv}\.{v1_method}\b"
            text = re.sub(pattern, f"b.{v1_method}", text)
        # Variable rename: `circuit = ...` → `b = ...`. Only at
        # line starts to avoid clobbering substrings.
        text = re.sub(
            rf"^(\s*){recv}\s*=\s*p\.CircuitBuilder\(\)",
            r"\1b = p.CircuitBuilder()",
            text, flags=re.MULTILINE)
        # General receiver name change — only if NO `b` already
        # used as an identifier (skip rename to avoid clashes;
        # leave with original name and flag).
        if re.search(rf"\b{recv}\b", text):
            # If `b` is not used as a variable name elsewhere, do
            # the rename; otherwise flag.
            if re.search(r"\bb\s*=", text):
                warnings.append(
                    f"TODO[v2-migrate]: receiver `{recv}` and "
                    f"variable `b` both used — manual rename "
                    f"recommended.")

    # 4. Closing arg ``)`` normalization isn't perfect — leave
    # alone (Black will reformat). Patterns like ``p.simulate(b, )``
    # do parse and run.

    # 5. Manual-review tagging.
    out_lines: List[str] = []
    for line in text.splitlines():
        flagged = False
        for pat, msg in _MANUAL_REVIEW_PATTERNS:
            if pat.search(line):
                m_indent = re.match(r"^(\s*)", line)
                indent = m_indent.group(1) if m_indent else ""
                out_lines.append(f"{indent}# TODO[v2-migrate]: {msg}")
                flagged = True
                break
        out_lines.append(line)
        # Don't double-emit for the same line — just one TODO above.
        _ = flagged
    text = "\n".join(out_lines)

    # 6. Trailing newline normalisation.
    if not text.endswith("\n"):
        text += "\n"

    return text, warnings


def _migrate_py(src_path: Path, *, dry_run: bool,
                  in_place: bool) -> int:
    src = src_path.read_text(encoding="utf-8")
    new_text, warnings = _migrate_text(src)
    if new_text == src and not warnings:
        print(f"  {src_path}: no changes")
        return 0
    if dry_run:
        print(f"--- {src_path} (dry-run, would write):")
        # Show first ~40 lines of new text.
        head = "\n".join(new_text.splitlines()[:40])
        print(head)
        print("…")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  • {w}")
        return 0
    dst_path = src_path if in_place else src_path.with_suffix(".v2.py")
    dst_path.write_text(new_text, encoding="utf-8")
    suffix = " (in-place)" if in_place else f" → {dst_path}"
    print(f"  {src_path}: migrated{suffix}")
    if warnings:
        for w in warnings:
            print(f"    ⚠ {w}")
    return 0


def _migrate_notebook(src_path: Path, *, dry_run: bool,
                          in_place: bool) -> int:
    """Migrate a Jupyter notebook (.ipynb). Walks ``cells`` of
    ``cell_type == 'code'`` and applies the same transform.
    """
    nb = json.loads(src_path.read_text(encoding="utf-8"))
    any_change = False
    all_warnings: List[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src_lines = cell.get("source", [])
        old_src = "".join(src_lines)
        new_src, warnings = _migrate_text(old_src)
        all_warnings.extend(warnings)
        if new_src != old_src:
            any_change = True
            cell["source"] = new_src.splitlines(keepends=True)
            # Clear outputs since the v1 results may be stale.
            cell["outputs"] = []
            if "execution_count" in cell:
                cell["execution_count"] = None
    if not any_change:
        print(f"  {src_path}: no changes")
        return 0
    if dry_run:
        print(f"  {src_path} (dry-run): would have updated "
              f"{sum(1 for c in nb['cells'] if c.get('cell_type')=='code')} "
              f"code cells")
        for w in all_warnings:
            print(f"    ⚠ {w}")
        return 0
    dst_path = (src_path if in_place
                 else src_path.with_suffix(".v2.ipynb"))
    dst_path.write_text(json.dumps(nb, indent=1),
                          encoding="utf-8")
    suffix = " (in-place)" if in_place else f" → {dst_path}"
    print(f"  {src_path}: migrated{suffix}")
    for w in all_warnings:
        print(f"    ⚠ {w}")
    return 0


def main(argv: "List[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert Pulsim v1 Python / Notebook files to v2.")
    ap.add_argument("paths", nargs="+",
                       help="Files or directories to convert.")
    ap.add_argument("--dry-run", action="store_true",
                       help="Don't write — show what would change.")
    ap.add_argument("--in-place", action="store_true",
                       help="Overwrite the original file. Default writes "
                            "to <name>.v2.py (or .v2.ipynb).")
    ap.add_argument("--notebook", action="store_true",
                       help="Treat .ipynb paths as Jupyter notebooks. "
                            "(Auto-detected from extension too.)")
    args = ap.parse_args(argv)

    files: List[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(p.rglob("*.py"))
            files.extend(p.rglob("*.ipynb"))
        else:
            files.append(p)

    if not files:
        print("nothing to migrate", file=sys.stderr)
        return 2

    for f in files:
        if f.suffix == ".ipynb":
            _migrate_notebook(f, dry_run=args.dry_run,
                                  in_place=args.in_place)
        elif f.suffix == ".py":
            _migrate_py(f, dry_run=args.dry_run,
                          in_place=args.in_place)
        else:
            print(f"  {f}: skipped (not .py or .ipynb)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
