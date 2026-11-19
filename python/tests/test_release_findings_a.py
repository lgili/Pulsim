"""Release findings, group A: impulsive events under the restart rule.

Each test reproduces a finding of the number-changing release on the PR line with the
release behaviour turned on EXPLICITLY (``discontinuity_restart=True`` on the fixed
engine; ``engine='dsed'`` for the exact engine) and checks it against an independent
truth (flux / charge conservation, the exact engine, a fine fixed step).
Notes and traces: openspec/changes/update-number-changing-release/fixes/group-a.md.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

RESTART = {"engine": "pwl", "discontinuity_restart": True}


# ---- F-5-1: a pathless inductor under the restart rule ---------------------------------------

def _freewheel(**kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vg", "g", "gnd", 15.0)
    b.add_inductor("L", "sw", "gnd", 1e-3, i0=100.0)
    b.add_igbt_level1("QL", "sw", "gnd", "g", 1.5, 0.05, 5.0, with_fwd=False)
    return p.simulate(b, t_end=1e-6, dt=1e-9, **kw)


def test_f51_pathless_inductor_never_gains_energy_and_is_named():
    """No conduction path: the 100 A is an impulse (the IGBT blocks reverse current). The
    restart must not turn it into -141 A (energy 5 J -> 10 J, no source in the loop), and the
    voltage-sanity guard must name the open path, as it does restart-less."""
    with pytest.warns(UserWarning, match="conduction path opens"):
        res = _freewheel(**RESTART)
    i = np.asarray(res.i("L"))
    # stored energy never exceeds the initial 5 J (flux cannot be created)
    assert np.max(np.abs(i)) <= 100.0 * (1 + 1e-12), np.max(np.abs(i))
    # the cut-set leaves no path: the flux is gone after the impulse
    assert np.max(np.abs(i[1:])) < 1e-3, i[:4]


# ---- F-3-3: the flyback's MOSFET turned off into its body diode ------------------------------

L_M = 111.4e-6
N_RATIO = 13 / 25


def _flyback(R_L=1.0, L_leak_p=5e-6):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_mosfet("Q1", "sw", "gnd", R_on=1e-2, R_off=1e9)
    L_p = L_leak_p + L_M
    L_s = N_RATIO * N_RATIO * L_M
    b.add_transformer("T1", "vin", "sw", "sec_neg", "sec_anode", L_p=L_p, L_s=L_s,
                      k=N_RATIO * L_M / np.sqrt(L_p * L_s))
    b.add_diode("D1", "sec_anode", "vout", 1e3, 1e-9, V_th=0.7)
    b.add_capacitor("Cout", "vout", "sec_neg", 100e-6)
    b.add_resistor("R_L", "vout", "sec_neg", R_L)
    b.add_resistor("Rgnd", "sec_neg", "gnd", 1e-6)
    return b


def _run_flyback(t_end=1e-3, **kw):
    b = _flyback()
    pwm = p.make_pwm_switch_fn(frequency=100e3, duty=0.5, switch_idx=0,
                               num_switches=b.graph.num_switches, phase=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=t_end, switch_fn=pwm, **kw)
    t = np.asarray(res.times)
    m = t > 0.8 * t_end
    return np.asarray(res.i("T1.p"))[m]


def test_f33_flyback_turn_off_does_not_enter_the_body_diode():
    """Truth: the exact engine (LCP mode selection) takes the primary to ~0 at turn-off (the
    leakage current has no path but R_off = 1 GOhm) with a 16.382 A peak. The restart's
    first stage takes it there too; the body diode must not pick up -sqrt(2) times it."""
    ip_exact = _run_flyback(engine="dsed", mode_selection="auto")
    ip = _run_flyback(dt=1e-8, **RESTART)
    assert ip_exact.min() > -1e-6 and ip_exact.max() == pytest.approx(16.382, abs=1e-3)
    assert ip.min() > -1e-3, ip.min()
    assert ip.max() == pytest.approx(ip_exact.max(), rel=2e-3)
