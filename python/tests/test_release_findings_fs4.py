"""Release finding F-S-4: the exact engine (engine='dsed') dropped an
inductor's initial current at a CONSISTENT rectifier start.

Minimal circuit (11 elements, from the PSFB's power-transfer start): 400 V ->
L_lk (10 uH, 0 A) -> ideal 4:1 primary (p, b); b to ground through 1 mOhm ||
a diode (gnd -> b) || 200 pF; the secondary into a full diode bridge feeding
L_o (20 uH, 24 A) to ground. The given state is consistent: the four bridge
diodes share L_o (12 A each) and short the secondary, so L_lk ramps from 0 A
under the whole bus.

Truths, both independent of pulsim:
* closed form of the first interval (all four bridge diodes ON, D4 OFF): the
  bridge shorts the secondary through R_s = 1 mOhm (two 2 mOhm paths of g_on
  = 1000 S diodes in parallel; L_o's current splits symmetrically and puts
  no voltage across the winding), reflected to the primary as R_s / n^2, in
  series with S4 (1 mOhm): i_Llk(t) = (V / R)(1 - exp(-R t / L_lk)),
  R = 1e-3 + 1e-3 / 0.25^2 = 17 mOhm (C4 = 200 pF across 1 mOhm: 0.2 ps,
  and D4's 1 nS off-conductance are below 1e-9 of it over 100 ns);
* benchmarks/tools/exact_truth.py (mpmath, 30 digits, as on
  feat/core-soft-switching-validation fa4b5c71) on the same netlist:
  i_Lo(0) = 24 A, D6 turns off at 150.018 ns, i_Llk(200 ns) =
  6.06050914046011 A.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p

I_LLK_200NS = 6.06050914046011   # exact_truth.py, dps 30 (module docstring)
T_D6_OFF = 150.018002831e-9      # exact_truth.py: the first commutation


def _builder(body_diode=True, cap=True, rail_diode=False):
    b = p.CircuitBuilder()
    b.add_voltage_source("Va", "a", "gnd", 400.0)
    b.add_resistor("S4", "b", "gnd", 1e-3)
    if body_diode:
        b.add_diode("D4", "gnd", "b", 1000.0, 1e-9)
    if cap:
        b.add_capacitor("C4", "b", "gnd", 200e-12)
    b.add_inductor("Llk", "a", "p", 10e-6, 0.0)
    b.add_ideal_transformer("T1", "p", "b", "s1", "s2", 0.25)
    for name, an, ca in (("D5", "s1", "r"), ("D6", "s2", "r"), ("D7", "gnd", "s1"), ("D8", "gnd", "s2")):
        b.add_diode(name, an, ca, 1000.0, 1e-9)
    b.add_inductor("Lo", "r", "gnd", 20e-6, 24.0)
    if rail_diode:
        b.add_diode("Dx", "b", "a", 1000.0, 1e-9)
    return b


def _closed_form_llk(t):
    R, L, V = 1e-3 + 1e-3 / 0.25**2, 10e-6, 400.0
    return V / R * -math.expm1(-R * t / L)


SELECTORS = [None, "lcp", "auto"]
VARIANTS = {"full": {}, "no_cap": {"cap": False}, "no_body_diode": {"body_diode": False},
            "rail_diode": {"rail_diode": True}}


def _run(selector, output_rtol=None, **variant):
    kw = {"engine": "dsed"}
    if selector is not None:
        kw["mode_selection"] = selector
    if output_rtol is not None:
        kw["output_rtol"] = output_rtol
    return p.simulate(_builder(**variant), t_end=2e-7, dt=1e-9, **kw)


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("selector", SELECTORS, ids=lambda s: s or "legacy")
def test_exact_engine_keeps_the_initial_inductor_current_at_a_rectifier_start(selector, variant):
    r = _run(selector, **VARIANTS[variant])
    t = np.asarray(r.times)
    assert r.i("Lo")[0] == pytest.approx(24.0, abs=1e-6)
    assert r.i("Llk")[0] == pytest.approx(0.0, abs=1e-6)
    # every bridge diode conducts at the start (the consistent mode)
    assert np.interp(100e-9, t, np.asarray(r.i("Llk"))) == pytest.approx(_closed_form_llk(100e-9), rel=1e-6)
    assert np.interp(2e-7, t, np.asarray(r.i("Llk"))) == pytest.approx(I_LLK_200NS, rel=1e-6)


@pytest.mark.parametrize("selector", SELECTORS, ids=lambda s: s or "legacy")
def test_exact_engine_rectifier_start_on_the_output_rule(selector):
    r = _run(selector, output_rtol=1e-5)
    t = np.asarray(r.times)
    assert r.i("Lo")[0] == pytest.approx(24.0, abs=1e-6)
    assert np.interp(2e-7, t, np.asarray(r.i("Llk"))) == pytest.approx(I_LLK_200NS, rel=1e-6)


def test_closed_form_matches_the_exact_tool_on_the_first_interval():
    # the two truths agree where both hold: 1.5e-7 s, just before D6 turns off
    # (exact_truth.py, dps 30: i_Llk(150 ns) = 5.99923506514082 A)
    assert _closed_form_llk(150e-9) == pytest.approx(5.99923506514082, rel=1e-7)
    assert T_D6_OFF > 150e-9

