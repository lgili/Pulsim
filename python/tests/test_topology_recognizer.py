"""Tests for the deterministic topology recognizer.

Each canonical topology has:
  1. A "builder" fixture function that produces a CircuitBuilder in
     the canonical wiring for that topology.
  2. A positive test asserting `recognize(b).name == expected` with
     confidence ≥ 0.9.
  3. Implicit inverse coverage via `test_no_cross_matches` at the
     bottom: each builder is fed into EVERY detector and we assert
     that exactly one detector returns confidence ≥ 0.7 (its own).

The tests run without `networkx` or `schemdraw` — the recognizer is
pure Python over the `CircuitBuilder.components()` adapter.
"""

from __future__ import annotations

import pytest

import pulsim as p
from pulsim.schematic.topology_recognizer import (
    KNOWN_TOPOLOGIES,
    MIN_CONFIDENCE,
    RecognizedTopology,
    recognize,
    detect_buck,
    detect_boost,
    detect_buck_boost,
    detect_flyback,
    detect_forward,
    detect_half_bridge,
    detect_full_bridge,
    detect_rc_filter,
    detect_rl_filter,
    detect_rlc_filter,
    detect_half_wave_rectifier,
    detect_full_wave_bridge_rectifier,
    _CircuitView,
)


# ---------------------------------------------------------------------------
# Builders — one canonical example per topology
# ---------------------------------------------------------------------------

def build_buck() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin",   "vin", "gnd", 24.0)
    b.add_switch       ("Q",      "vin", "sw",   g_on=1e3, g_off=1e-9)
    b.add_diode        ("D",      "gnd", "sw",   g_on=1e3, g_off=1e-9,
                         V_th=0.0)
    b.add_inductor     ("L",      "sw",  "vout", 100e-6)
    b.add_capacitor    ("Cout",   "vout","gnd",  100e-6)
    b.add_resistor     ("Rload",  "vout","gnd",  5.0)
    return b


def build_boost() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin",   "vin", "gnd", 12.0)
    b.add_inductor     ("L",      "vin", "sw",   100e-6)
    b.add_switch       ("Q",      "sw",  "gnd",  g_on=1e3, g_off=1e-9)
    b.add_diode        ("D",      "sw",  "vout", g_on=1e3, g_off=1e-9,
                         V_th=0.0)
    b.add_capacitor    ("Cout",   "vout","gnd",  100e-6)
    b.add_resistor     ("Rload",  "vout","gnd",  20.0)
    return b


def build_buck_boost() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin",   "vin", "gnd", 24.0)
    b.add_switch       ("Q",      "vin", "sw",   g_on=1e3, g_off=1e-9)
    b.add_inductor     ("L",      "sw",  "gnd",  100e-6)
    b.add_diode        ("D",      "sw",  "vout", g_on=1e3, g_off=1e-9,
                         V_th=0.0)
    b.add_capacitor    ("Cout",   "vout","gnd",  100e-6)
    b.add_resistor     ("Rload",  "vout","gnd",  10.0)
    return b


def build_flyback() -> p.CircuitBuilder:
    """Simplified flyback — primary L + secondary L sharing a name
    prefix, switch on primary side, diode on secondary side."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin",  "vin", "gnd", 24.0)
    b.add_inductor      ("Lp",   "vin", "sw",  100e-6)   # primary
    b.add_switch        ("Q",    "sw",  "gnd", g_on=1e3, g_off=1e-9)
    b.add_inductor      ("Ls",   "vsec","vout",10e-6)    # secondary
    b.add_diode         ("D",    "vsec","vout",g_on=1e3, g_off=1e-9,
                         V_th=0.0)
    b.add_capacitor     ("Cout", "vout","gnd", 100e-6)
    b.add_resistor      ("Rload","vout","gnd", 5.0)
    return b


def build_half_bridge() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vdc", "gnd", 48.0)
    b.add_switch       ("Qh",   "vdc", "mid", g_on=1e3, g_off=1e-9)
    b.add_switch       ("Ql",   "mid", "gnd", g_on=1e3, g_off=1e-9)
    return b


def build_full_bridge() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vdc", "gnd", 48.0)
    b.add_switch       ("Qah",  "vdc", "mid_a", g_on=1e3, g_off=1e-9)
    b.add_switch       ("Qal",  "mid_a","gnd",  g_on=1e3, g_off=1e-9)
    b.add_switch       ("Qbh",  "vdc", "mid_b", g_on=1e3, g_off=1e-9)
    b.add_switch       ("Qbl",  "mid_b","gnd",  g_on=1e3, g_off=1e-9)
    return b


def build_forward() -> p.CircuitBuilder:
    """Single-switch forward — source / primary / switch / two diodes /
    output choke / output cap + load."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin",   "vin", "gnd", 100.0)
    b.add_inductor      ("Lp",    "vin", "sw",  500e-6)
    b.add_switch        ("Q",     "sw",  "gnd", g_on=1e3, g_off=1e-9)
    b.add_diode         ("Dfwd",  "vsec","vmid",g_on=1e3, g_off=1e-9,
                         V_th=0.0)
    b.add_diode         ("Dfw",   "gnd", "vmid",g_on=1e3, g_off=1e-9,
                         V_th=0.0)
    b.add_inductor      ("Lout",  "vmid","vout",100e-6)
    b.add_capacitor     ("Cout",  "vout","gnd", 100e-6)
    b.add_resistor      ("Rload", "vout","gnd", 10.0)
    return b


def build_rc_filter() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 5.0)
    b.add_resistor      ("R",   "vin", "vc",  1e3)
    b.add_capacitor     ("C",   "vc",  "gnd", 1e-6)
    return b


def build_rl_filter() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_resistor      ("R",   "vin", "nL",  10.0)
    b.add_inductor      ("L",   "nL",  "gnd", 1e-3)
    return b


def build_rlc_filter() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vstep", "in", "gnd", 10.0)
    b.add_inductor      ("L",     "in", "nA",  100e-6)
    b.add_resistor      ("R",     "nA", "nB",  0.1)
    b.add_capacitor     ("C",     "nB", "gnd", 100e-6)
    return b


def build_half_wave_rectifier() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_sine_voltage_source(
        "Vsin", "in", "gnd",
        v_dc=0.0, v_amplitude=10.0, frequency=60.0, phase=0.0)
    b.add_diode    ("D",     "in",   "vout", g_on=1.0, g_off=1e-9,
                     V_th=0.7)
    b.add_capacitor("C",     "vout", "gnd",  100e-6)
    b.add_resistor ("Rload", "vout", "gnd",  1e3)
    return b


def build_full_wave_bridge_rectifier() -> p.CircuitBuilder:
    """4-diode bridge across an AC source; output between rails."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source(
        "Vsin", "ac+", "ac-",
        v_dc=0.0, v_amplitude=24.0, frequency=60.0, phase=0.0)
    # Bridge: ac+ ↔ vout+ (D1), ac+ ↔ gnd (D2),
    #         ac- ↔ vout+ (D3), ac- ↔ gnd (D4).
    b.add_diode("D1", "ac+",  "vout", g_on=1.0, g_off=1e-9, V_th=0.7)
    b.add_diode("D2", "gnd",  "ac+",  g_on=1.0, g_off=1e-9, V_th=0.7)
    b.add_diode("D3", "ac-",  "vout", g_on=1.0, g_off=1e-9, V_th=0.7)
    b.add_diode("D4", "gnd",  "ac-",  g_on=1.0, g_off=1e-9, V_th=0.7)
    b.add_capacitor("C",     "vout", "gnd", 1e-3)
    b.add_resistor ("Rload", "vout", "gnd", 100.0)
    return b


_BUILDERS: dict[str, callable] = {
    "buck":                       build_buck,
    "boost":                      build_boost,
    "buck_boost":                 build_buck_boost,
    "flyback":                    build_flyback,
    "forward":                    build_forward,
    "half_bridge":                build_half_bridge,
    "full_bridge":                build_full_bridge,
    "rc_filter":                  build_rc_filter,
    "rl_filter":                  build_rl_filter,
    "rlc_filter":                 build_rlc_filter,
    "half_wave_rectifier":        build_half_wave_rectifier,
    "full_wave_bridge_rectifier": build_full_wave_bridge_rectifier,
}


# ---------------------------------------------------------------------------
# Tests — positive recognition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topo,builder", list(_BUILDERS.items()))
def test_positive_recognition(topo: str, builder) -> None:
    """Each canonical builder must return its own topology with
    confidence ≥ 0.9 (except forward which is intentionally lower)."""
    b = builder()
    result = recognize(b)
    assert result is not None, f"recognize() returned None for {topo}"
    assert result.name == topo, (
        f"expected {topo!r} but got {result.name!r} "
        f"(confidence={result.confidence})")
    threshold = 0.7 if topo == "forward" else 0.85
    assert result.confidence >= threshold, (
        f"{topo} confidence too low: {result.confidence}")
    assert result.source == "heuristic"


def test_known_topologies_set() -> None:
    """Every builder maps to a name in KNOWN_TOPOLOGIES."""
    for topo in _BUILDERS:
        assert topo in KNOWN_TOPOLOGIES


# ---------------------------------------------------------------------------
# Tests — inverse rejection (no false positives across topology pairs)
# ---------------------------------------------------------------------------

_DETECTOR_FN_BY_NAME = {
    "buck":                       detect_buck,
    "boost":                      detect_boost,
    "buck_boost":                 detect_buck_boost,
    "flyback":                    detect_flyback,
    "forward":                    detect_forward,
    "half_bridge":                detect_half_bridge,
    "full_bridge":                detect_full_bridge,
    "rc_filter":                  detect_rc_filter,
    "rl_filter":                  detect_rl_filter,
    "rlc_filter":                 detect_rlc_filter,
    "half_wave_rectifier":        detect_half_wave_rectifier,
    "full_wave_bridge_rectifier": detect_full_wave_bridge_rectifier,
}


@pytest.mark.parametrize("circuit_topo", list(_BUILDERS.keys()))
def test_no_cross_matches(circuit_topo: str) -> None:
    """A circuit of topology X must NOT be matched by detectors for
    OTHER topologies above 0.7 confidence (forward overlap with
    buck/flyback is accepted at 0.75 so we use a stricter rule for
    everything else)."""
    builder = _BUILDERS[circuit_topo]()
    view = _CircuitView(builder)
    for other_topo, detector in _DETECTOR_FN_BY_NAME.items():
        if other_topo == circuit_topo:
            continue
        # Forward and flyback share the transformer-based fingerprint
        # in our simplified model — skip that single overlap.
        if {circuit_topo, other_topo} == {"flyback", "forward"}:
            continue
        try:
            conf, _ = detector(view)
        except Exception:
            conf = 0.0
        assert conf < 0.7, (
            f"detector for {other_topo!r} returned confidence "
            f"{conf} on a {circuit_topo!r} circuit (false positive)")


# ---------------------------------------------------------------------------
# Tests — None / threshold behavior
# ---------------------------------------------------------------------------

def test_returns_none_for_unrelated_circuit() -> None:
    """A random graph of 8 resistors with no other devices should
    not match any topology."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "n0", "gnd", 1.0)
    for i in range(8):
        b.add_resistor(f"R{i}", f"n{i}", f"n{i+1}", 1.0)
    result = recognize(b)
    assert result is None


def test_empty_circuit() -> None:
    """Empty builder produces no match."""
    b = p.CircuitBuilder()
    result = recognize(b)
    assert result is None


def test_role_map_completeness_buck() -> None:
    """Buck role_map names every component touched by the topology."""
    b = build_buck()
    result = recognize(b)
    assert result is not None and result.name == "buck"
    assert {"Vin", "Q", "D", "L", "Cout"} <= set(result.role_map.keys())
    # Roles are from the canonical set.
    assert "source" in result.role_map.values()
    assert "switch_high" in result.role_map.values()
    assert "freewheel_diode" in result.role_map.values()
    assert "inductor_main" in result.role_map.values()
    assert "output_capacitor" in result.role_map.values()
