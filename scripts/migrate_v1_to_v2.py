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

    # 1. Find the user's chosen alias (`as <name>`) so we can rewrite
    #    references throughout the file.
    aliases: List[str] = []
    for m in re.finditer(
            r"^import\s+pulsim\s+as\s+(\w+)\s*$",
            text, re.MULTILINE):
        aliases.append(m.group(1))
    # Plain `import pulsim` — the alias is literally `pulsim`.
    if re.search(r"^import\s+pulsim\s*$", text, re.MULTILINE):
        aliases.append("pulsim")
    # Deduplicate.
    aliases = sorted(set(aliases))

    # Process the file line-by-line so we can SKIP import lines when
    # doing the alias substitution (otherwise ``pulsim.`` in the
    # rewritten import line ``import pulsim.v2 as p`` would itself
    # match the substitution and break to ``import p.v2 as p``).
    new_lines: List[str] = []
    for line in text.splitlines():
        # Rewrite the line ITSELF if it's a pulsim import.
        rewritten_import = re.sub(
            r"^(\s*)import\s+pulsim\s+as\s+\w+\s*$",
            r"\1import pulsim.v2 as p", line)
        rewritten_import = re.sub(
            r"^(\s*)import\s+pulsim\s*$",
            r"\1import pulsim.v2 as p", rewritten_import)
        if rewritten_import != line:
            new_lines.append(rewritten_import)
            continue
        # NOT an import line — apply alias substitutions on the body.
        body_line = line
        for alias in aliases:
            body_line = re.sub(
                rf"\b{re.escape(alias)}\.",
                "p.", body_line)
        new_lines.append(body_line)
    text = "\n".join(new_lines)

    # ``from pulsim import X`` — flag, don't rewrite. v1's symbol
    # set isn't 1:1 with v2.
    if re.search(r"^from\s+pulsim\s+import\b", text, re.MULTILINE):
        warnings.append(
            "TODO[v2-migrate]: `from pulsim import ...` found — "
            "the v1 symbol set doesn't map 1-to-1 to v2. Replace "
            "with `import pulsim.v2 as p`.")

    # 2. RuntimeCircuit() / Circuit() → CircuitBuilder().
    text = re.sub(
        r"\bp\.RuntimeCircuit\(\s*\)",
        "p.CircuitBuilder()", text)
    text = re.sub(
        r"\bp\.Circuit\(\s*\)",
        "p.CircuitBuilder()", text)

    # 2a-prelude. EARLY receiver-normalisation pass: rename any
    # ``circuit`` / ``ckt`` / ``cir`` receiver in the body of the
    # file to ``b`` so subsequent method-rewrites can assume the
    # canonical name. Done here (not in step 3 below) so the
    # method translations in steps 2c+ match ``b.foo()`` cleanly.
    for recv in ("circuit", "ckt", "cir"):
        # Variable assignment.
        text = re.sub(
            rf"^(\s*){recv}\s*=\s*p\.CircuitBuilder\(\)",
            r"\1b = p.CircuitBuilder()",
            text, flags=re.MULTILINE)
        # Method references and bare-identifier usages.
        text = re.sub(rf"\b{recv}\.", "b.", text)
        text = re.sub(rf"\b{recv}\b(?=[,\s\)])", "b", text)

    # 2b. Node-handle pattern: v1 used ``in_ = ckt.add_node("in")``
    # to get an integer node id, then passed it to add_*. v2 just
    # takes node names as strings directly. Walk the file, find
    # ``<var> = <recv>.add_node("<name>")`` assignments, record
    # the (var → name) mapping, replace bare ``<var>`` usages with
    # the quoted name, and comment out the original add_node line.
    node_aliases: dict = {}
    for line in text.splitlines():
        m = re.match(
            r"^\s*(\w+)\s*=\s*[\w\.]+\.add_node\(\s*['\"]([^'\"]+)['\"]\s*\)\s*$",
            line)
        if m:
            node_aliases[m.group(1)] = m.group(2)
    if node_aliases:
        # Replace bare references with quoted string literals. Use
        # negative lookbehind/lookahead so we don't reach INTO an
        # existing string literal (which would yield "" "out" "")
        # or into a longer identifier (``output`` would catch
        # ``out``).
        for var, name in node_aliases.items():
            text = re.sub(
                rf"(?<![\w\"\']){re.escape(var)}(?![\w\"\'])",
                f'"{name}"', text)
        # Comment out the original add_node assignment lines (these
        # now look like ``"in" = b.add_node(...)`` after the
        # substitution above — invalid syntax, so we MUST replace
        # them with a comment).
        text = re.sub(
            r"^(\s*)(\"\w+\"|\w+)\s*=\s*\w+\.add_node\([^)]*\)\s*$",
            r"\1# [v2-migrate] removed add_node call  "
            r"(v2 uses node names directly)",
            text, flags=re.MULTILINE)

    # 2c. Common v1→v2 method translations on the builder.
    text = re.sub(r"\bb\.ground\(\s*\)", '"gnd"', text)
    text = re.sub(r"\bb\.num_nodes\(\s*\)", "b.graph.num_nodes", text)
    text = re.sub(r"\bb\.num_branches\(\s*\)",
                      "b.graph.num_branches", text)
    text = re.sub(r"\bb\.node_idx\(", "b.node_id_of(", text)

    # 3. Method-on-builder → v2 equivalent rewrites. By this point
    # the early receiver-normalisation pass converted every
    # ``ckt.``/``circuit.``/``cir.`` to ``b.`` — we just translate
    # the methods that became free functions in v2.
    # Free-function analysis moves: b.simulate(...) → p.simulate(b, ...)
    for v1_method, v2_call in _ANALYSIS_MOVES.items():
        text = re.sub(
            rf"\bb\.{v1_method}\s*\(",
            f"{v2_call}(b, ",
            text)
    # Free helpers (add_bridge_rectifier etc): b.add_…(...) →
    # p.add_…(b, ...).
    for v1_method, v2_call in _FREE_FUNCTION_HELPERS.items():
        text = re.sub(
            rf"\bb\.{v1_method}\s*\(",
            f"{v2_call}(b, ",
            text)

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
