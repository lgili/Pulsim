"""benchmarks/tools/circuit_lint.py: the benchmark circuits are free of the
two authoring errors it finds (an uncoupled winding; a switch whose
turn-off interrupts inductor current with nowhere to go), and it finds
both in the ll16 netlist as it was before 2026-11-18."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "tools"))
import circuit_lint  # noqa: E402

OLD_LL16 = """
components:
  - {type: voltage_source, name: Vin, nodes: [vin, 0], waveform: {type: dc, value: 48}}
  - {type: inductor, name: Lleak_prim, nodes: [vin, drain], value: 2u}
  - {type: inductor, name: Lm_core, nodes: [vin, drain], value: 160u}
  - {type: inductor, name: Lsec, nodes: [sec, 0], value: 10u}
  - {type: vcswitch, name: S_main, nodes: [0, drain, 0], ron: 0.05, roff: 1e7}
  - {type: diode, name: D_sec, nodes: [sec, vout]}
  - {type: capacitor, name: Cout, nodes: [vout, 0], value: 470u}
  - {type: resistor, name: Rload, nodes: [vout, 0], value: 5}
"""


def test_the_benchmark_circuits_are_lint_clean(capsys):
    assert circuit_lint.main([]) == 0, capsys.readouterr().out


def test_the_old_ll16_netlist_is_flagged_twice(tmp_path):
    p = tmp_path / "ll16_old.yaml"
    p.write_text(OLD_LL16)
    rules = {rule: msg for rule, msg in circuit_lint.lint(p)}
    assert "Lsec" in rules["source_free"]
    assert "S_main open" in rules["inductor_cutset"]
    assert "Lm_core" in rules["inductor_cutset"]


def test_a_coupled_flyback_is_clean(tmp_path):
    p = tmp_path / "flyback.yaml"
    p.write_text(OLD_LL16.replace(
        "  - {type: inductor, name: Lleak_prim, nodes: [vin, drain], value: 2u}\n", "").replace(
        "  - {type: inductor, name: Lsec, nodes: [sec, 0], value: 10u}\n",
        "  - {type: transformer, name: T1, nodes: [vin, drain, 0, sec], turns_ratio: 4}\n"))
    assert circuit_lint.lint(p) == []
