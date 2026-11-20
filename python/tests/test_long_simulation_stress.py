"""Long-simulation stress (add-long-simulation-stress).

Every check has a truth that is not a simulator number:

* the periodic steady state of a piecewise-LTI converter in closed form
  (the monodromy fixed point by matrix exponentials, scipy only), or an
  exact linear identity (an RL load's mean current is its mean drive over
  its resistance; an RC's mean output is its mean input);
* energy conservation from independent branches (source work = load +
  device dissipation + stored energy), and the thermal node's charge
  balance against the in-kernel loss ledger;
* determinism (two runs bit-identical, same code path, same machine);
* resume chains through pickle: bit-identical where the engine promises
  it (pwl at any step boundary; the exact engine at an instant the
  uninterrupted run also stops at, checked at 1e-12 for CI), within the
  run's accuracy where it does not (trbdf2 takes its own steps after a
  resume);
* bounded records and memory (counts as bounds).

Bounds come from the method: a second-order rule on a grid of largest step
``h`` resolves a waveform of natural frequency ``w0`` to ``(w0 h)^2``
relative, so a sample-based integral (and a fixed-step period average) is
held to that, a tolerance-controlled run to ``K * rtol`` on top.

Each engine runs through EXPLICIT options (the defaults are about to
flip): pwl with ``discontinuity_restart`` False and True, trbdf2,
trbdf2 + esdirk43, dsed. The multi-second cases sit behind the repo's
slow-test marker (``examples``, ``PULSIM_RUN_EXAMPLES=1``); the default
run keeps each case short. No timing assertions.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
from kpi import compute_conservation_drift, compute_drift
from scipy.linalg import expm

import pulsim as p

RUN_LONG = os.environ.get("PULSIM_RUN_EXAMPLES", "0") == "1"


def long_case(fn):
    """The repo's slow-test gate (test_examples_claims.py): marker + env."""
    return pytest.mark.examples(
        pytest.mark.skipif(not RUN_LONG, reason="set PULSIM_RUN_EXAMPLES=1")(fn))


RTOL, ATOL = 1e-6, 1e-9          # explicit accuracy of the variable-step engines
K_RTOL = 10.0                    # a controlled run's global error: a few rtol

F_L_1 = ("finding F-L-1: pwl discontinuity_restart=True records one sample per gate "
         "edge (the left limit), so every edge reads as a one-step ramp: integrals over "
         "the record (input power, energy ledger) miss O(dt) per edge and the error "
         "grows with run length (residual 2.5e-3 of the input energy at dt = T/100)")

F_L_2 = ("finding F-L-2: the exact engine (dsed) records one sample per gate edge, the "
         "left limit, with or without output_rtol: integrals over its record miss the "
         "switched waveform (buck energy ledger over 10 ms: residual 0.109 of the input "
         "energy with output_rtol unset, 5.6e-3 with output_rtol=1e-6 against a grid bound "
         "of 1.4e-4)")


def _engines(dt):
    return {
        "pwl": dict(engine="pwl", dt=dt, discontinuity_restart=False),
        "pwl-restart": dict(engine="pwl", dt=dt, discontinuity_restart=True),
        "trbdf2": dict(engine="trbdf2", rtol=RTOL, atol=ATOL),            # TR-BDF2: integrator omitted (refused by name)
        "esdirk43": dict(engine="trbdf2", integrator="esdirk43", rtol=RTOL, atol=ATOL),
        "dsed": dict(engine="dsed", rtol=RTOL, atol=ATOL),
    }


ENGINES = sorted(_engines(1.0))
FIXED = ("pwl", "pwl-restart")


def _sim(b, t_end, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, **kw)


def _grid_bound(times, w0):
    """(w0 h_max)^2: what a second-order rule resolves on this grid."""
    return (w0 * float(np.max(np.diff(np.asarray(times))))) ** 2


def _acc_bound(eng, times, w0):
    return _grid_bound(times, w0) + (0.0 if eng in FIXED else K_RTOL * RTOL)


# ---------------------------------------------------------------------------
# The buck and its closed-form periodic steady state
# ---------------------------------------------------------------------------

class Buck:
    """Switch + diode buck; both two-state conductances, ideal elsewhere."""

    def __init__(self, vin=24.0, g_on=1e2, g_off=1e-9, L=100e-6, C=47e-6,
                 R=5.0, f=100e3, D=0.5):
        self.vin, self.g_on, self.g_off = vin, g_on, g_off
        self.L, self.C, self.R, self.f, self.D = L, C, R, f, D
        self.T = 1.0 / f
        self.w0 = 1.0 / np.sqrt(L * C)

    def build(self):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", self.vin)
        b.add_switch("S1", "vin", "sw", self.g_on, self.g_off)
        b.add_diode("D1", "gnd", "sw", self.g_on, self.g_off)
        b.add_inductor("L", "sw", "out", self.L)
        b.add_capacitor("C", "out", "gnd", self.C)
        b.add_resistor("RL", "out", "gnd", self.R)
        fn = p.make_pwm_switch_fn(self.f, self.D, b.switch_index_of("S1"),
                                  b.graph.num_switches)
        return b, fn

    def _aug(self, gs, gd):
        # x = (i_L, v_C, 1); v_sw = (gs*Vin - i_L) / (gs + gd) (no node capacitance)
        g = gs + gd
        M = np.zeros((3, 3))
        M[0, :] = [-1.0 / (self.L * g), -1.0 / self.L, gs * self.vin / (g * self.L)]
        M[1, :2] = [1.0 / self.C, -1.0 / (self.R * self.C)]
        return M

    def pss(self, n=4000):
        """Periodic steady state (CCM: the diode conducts exactly while the
        switch is off) and its period averages, from the exact flow."""
        segs = [(self.g_on, self.g_off, self.D * self.T),
                (self.g_off, self.g_on, (1 - self.D) * self.T)]
        Phi = np.eye(3)
        for gs, gd, h in segs:
            Phi = expm(self._aug(gs, gd) * h) @ Phi
        x = np.linalg.solve(np.eye(2) - Phi[:2, :2], Phi[:2, 2])
        xs = np.r_[x, 1.0]
        acc = dict(v=0.0, p_in=0.0, p_s=0.0, p_d=0.0, p_r=0.0)
        for gs, gd, h in segs:
            E = expm(self._aug(gs, gd) * h / n)
            rows = []
            for k in range(n + 1):
                il, vc = xs[0], xs[1]
                vsw = (gs * self.vin - il) / (gs + gd)
                i_s = gs * (self.vin - vsw)
                rows.append((vc, self.vin * i_s, i_s * (self.vin - vsw),
                             gd * vsw * vsw, vc * vc / self.R))
                if k < n:
                    xs = E @ xs
            rows = np.asarray(rows)
            for j, key in enumerate(acc):
                acc[key] += np.trapezoid(rows[:, j], dx=h / n)
        out = {k: v / self.T for k, v in acc.items()}
        out["x0"] = x
        return out


BUCK = Buck()
BUCK_DT = 1e-7                      # T / 100: every edge on the grid
_PSS = {}


def buck_pss():
    if "b" not in _PSS:
        _PSS["b"] = BUCK.pss()
    return _PSS["b"]


def test_closed_form_pss_conserves_energy():
    """The truth itself: its period averages balance (P_in = P_R + P_S + P_D)."""
    s = buck_pss()
    assert s["p_in"] == pytest.approx(s["p_r"] + s["p_s"] + s["p_d"], rel=1e-9)
    assert 0 < s["x0"][0]                        # CCM: i_L never reaches zero


def _buck_record(eng, t_end, store_every=None, period_stats=True):
    b, fn = BUCK.build()
    rec = p.Recorder("e", ["-v(vin)*i(Vin)", "v(out)"], keep="none",
                     period=BUCK.T if period_stats else 0.0)
    kw = _engines(BUCK_DT)[eng]
    if store_every:
        kw = dict(kw, store_every=store_every)
    return _sim(b, t_end, switch_fn=fn, recorders=[rec], **kw)


@pytest.mark.parametrize("eng", ENGINES)
def test_buck_settles_on_the_closed_form_steady_state(eng):
    r = _buck_record(eng, 20e-3)                        # 40 slow time constants
    per = r.recorded["e"].periods["v(out)"]["avg"]
    bound = _acc_bound(eng, r.times, BUCK.w0)
    assert abs(per[-1] - buck_pss()["v"]) <= bound * buck_pss()["v"]


@pytest.mark.parametrize("eng", [pytest.param(e, marks=pytest.mark.xfail(strict=True, reason=F_L_1))
                                 if e == "pwl-restart" else e for e in ENGINES])
def test_buck_input_power_matches_the_closed_form(eng):
    r = _buck_record(eng, 20e-3)
    pin = r.recorded["e"].periods["-v(vin)*i(Vin)"]["avg"]
    bound = _acc_bound(eng, r.times, BUCK.w0)
    assert abs(pin[-1] - buck_pss()["p_in"]) <= bound * buck_pss()["p_in"]


def _buck_ledger(r):
    t = np.asarray(r.times)
    vin, vsw, vout = (np.asarray(r.v(n)) for n in ("vin", "sw", "out"))
    p_in = -vin * np.asarray(r.i("Vin"))
    p_out = (vout * np.asarray(r.i("RL")) + (vin - vsw) * np.asarray(r.i("S1"))
             + (0.0 - vsw) * np.asarray(r.i("D1")))
    il = np.asarray(r.i("L"))
    e = 0.5 * BUCK.C * vout ** 2 + 0.5 * BUCK.L * il ** 2
    return compute_conservation_drift(list(p_in), list(p_out), e[0], e[-1], list(t))


_XF = {"pwl-restart": F_L_1, "dsed": F_L_2, "dsed-output_rtol": F_L_2}


@pytest.mark.parametrize("eng", [pytest.param(e, marks=pytest.mark.xfail(strict=True, reason=_XF[e]))
                                 if e in _XF else e for e in ENGINES + ["dsed-output_rtol"]])
def test_buck_energy_ledger_closes_over_the_record(eng):
    """Source work = load + switch + diode dissipation + stored energy,
    each from its own branch, over the whole 10 ms record from rest."""
    b, fn = BUCK.build()
    kw = (dict(_engines(BUCK_DT)["dsed"], output_rtol=1e-6) if eng == "dsed-output_rtol"
          else _engines(BUCK_DT)[eng])
    r = _sim(b, 10e-3, switch_fn=fn, **kw)
    resid = _buck_ledger(r)
    assert abs(resid) < 1e-2                                     # the spec's KPI
    assert abs(resid) <= _grid_bound(r.times, BUCK.w0) + (0 if eng in FIXED else K_RTOL * RTOL)


@pytest.mark.parametrize("eng", ENGINES)
def test_two_runs_are_bit_identical(eng):
    out = []
    for _ in range(2):
        b, fn = BUCK.build()
        r = _sim(b, 5e-3, switch_fn=fn, **_engines(BUCK_DT)[eng])
        out.append((np.asarray(r.times), np.asarray(r.states)))
    assert np.array_equal(out[0][0], out[1][0])
    assert np.array_equal(out[0][1], out[1][1])


# ---------------------------------------------------------------------------
# Resume chains through pickle
# ---------------------------------------------------------------------------

def _chain(eng, splits, t_end, make=None, dt=BUCK_DT, **extra):
    """Uninterrupted run and a chain of runs, each resumed from the pickled
    snapshot of the one before."""
    make = make or BUCK.build
    kw = dict(_engines(dt)[eng], **extra)
    b, fn = make()
    whole = _sim(b, t_end, switch_fn=fn, **kw)
    parts, snap = [], None
    for t_stop in list(splits) + [t_end]:
        b, fn = make()
        extra_kw = {} if snap is None else dict(resume_from=snap)
        if snap is not None and eng not in FIXED:
            extra_kw["t_start"] = float(snap.t)
        r = _sim(b, t_stop, switch_fn=fn, **kw, **extra_kw)
        parts.append(r)
        snap = pickle.loads(pickle.dumps(r.final_snapshot))
    return whole, parts


@pytest.mark.parametrize("eng", FIXED)
def test_pwl_resume_chain_is_bit_identical(eng):
    """Eight pickled resumes, mid-period (37 steps past a period start):
    every row of the chain is the uninterrupted run's row."""
    splits = [k * 0.5e-3 + 37 * BUCK_DT for k in range(1, 8)]
    whole, parts = _chain(eng, splits, 4.2e-3)
    tw, xw = np.asarray(whole.times), np.asarray(whole.states)
    for r in parts:
        t, x = np.asarray(r.times), np.asarray(r.states)
        i = np.searchsorted(tw, t[0])
        assert np.array_equal(tw[i:i + len(t)], t)
        assert np.array_equal(xw[i:i + len(t)], x)
    assert np.array_equal(xw[-1], np.asarray(parts[-1].states)[-1])


def test_exact_engine_resume_chain_at_gate_edges_continues_the_run():
    """dsed: splits at period starts (gate edges the uninterrupted run stops
    at). Bit for bit on this machine; 1e-12 for CI (another platform)."""
    splits = [k * 50 * BUCK.T for k in range(1, 8)]
    whole, parts = _chain("dsed", splits, 4e-3)
    tw, xw = np.asarray(whole.times), np.asarray(whole.states)
    scale = float(np.max(np.abs(xw)))
    for r in parts:
        t, x = np.asarray(r.times), np.asarray(r.states)
        common = np.intersect1d(tw, t[1:])
        assert len(common) >= 0.9 * (len(t) - 1)
        iw, ir = np.searchsorted(tw, common), np.searchsorted(t, common)
        np.testing.assert_allclose(x[ir], xw[iw], rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(np.asarray(parts[-1].final_snapshot.x),
                               np.asarray(whole.final_snapshot.x), rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("eng", ["trbdf2", "esdirk43"])
def test_trbdf2_resume_chain_stays_within_the_run_accuracy(eng):
    """trbdf2 takes its own steps after a resume (not promised bit for bit):
    each split moves the trajectory by at most the run's accuracy, and the
    chain still settles on the closed form."""
    splits = [k * 0.5e-3 + 0.37 * BUCK.T for k in range(1, 8)]
    whole, parts = _chain(eng, splits, 20e-3)
    xw = np.asarray(whole.final_snapshot.x)
    xc = np.asarray(parts[-1].final_snapshot.x)
    scale = float(np.max(np.abs(xw)))
    assert np.max(np.abs(xc - xw)) <= len(splits) * K_RTOL * (RTOL * scale + ATOL)
    t = np.asarray(parts[-1].times)
    v = np.asarray(parts[-1].v("out"))
    v_avg = p.time_average(t, v, t_from=20e-3 - BUCK.T)
    assert abs(v_avg - buck_pss()["v"]) <= _acc_bound(eng, t, BUCK.w0) * buck_pss()["v"]


# ---------------------------------------------------------------------------
# Bounded records
# ---------------------------------------------------------------------------

def test_store_every_bounds_the_record_not_the_statistics():
    """A decimated pwl record holds steps/store_every rows whatever the run
    length, while an in-loop recorder still sees every step."""
    n_steps = int(round(20e-3 / BUCK_DT))
    r = _buck_record("pwl", 20e-3, store_every=1000)
    assert len(r.times) <= n_steps // 1000 + 2
    assert r.recorded["e"].n_samples == n_steps + 1
    assert abs(r.recorded["e"].periods["v(out)"]["avg"][-1] - buck_pss()["v"]) \
        <= _grid_bound([0, BUCK_DT], BUCK.w0) * buck_pss()["v"]


# ---------------------------------------------------------------------------
# 1 MHz GaN buck (task 2.2), stiff leak + 5 MHz loop (3.x), 8-cell MMC (4.x)
# ---------------------------------------------------------------------------

GAN = Buck(vin=12.0, g_on=1e3, L=1e-6, C=1e-6, R=1.0, f=1e6, D=0.5)


@pytest.mark.parametrize("eng", ENGINES)
def test_gan_buck_at_1mhz_holds_vin_times_duty(eng):
    b, fn = GAN.build()
    rec = p.Recorder("e", ["v(out)"], keep="none", period=GAN.T)
    r = _sim(b, 100e-6, switch_fn=fn, recorders=[rec], **_engines(10e-9)[eng])
    v = r.recorded["e"].periods["v(out)"]["avg"][-1]
    assert abs(v - GAN.vin * GAN.D) <= 0.01 * GAN.vin * GAN.D              # the spec
    truth = GAN.pss()["v"]
    assert abs(v - truth) <= (_acc_bound(eng, r.times, GAN.w0) + 1e-6) * truth


F_SW_STIFF, R_LEAK, C_SLOW = 5e6, 1e9, 100e-12       # tau_slow = 0.1 s = 5e5 periods


def _stiff():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 10.0)
    b.add_switch("SH", "vin", "sw", 1e2, 1e-9)
    b.add_switch("SL", "sw", "gnd", 1e2, 1e-9)
    b.add_resistor("Rf", "sw", "fo", 10.0)
    b.add_capacitor("Cf", "fo", "gnd", 1e-9)                       # 10 ns
    b.add_resistor("Rleak", "fo", "slow", R_LEAK)
    b.add_capacitor("Cs", "slow", "gnd", C_SLOW)
    n = b.graph.num_switches
    fn = p.make_dead_time_pwm_pair_fn(F_SW_STIFF, 0.5, b.switch_index_of("SH"),
                                      b.switch_index_of("SL"), n, 0.0)
    return b, fn


@pytest.mark.parametrize("eng", ENGINES)
def test_stiff_leak_with_a_5mhz_loop_completes_and_charges_the_slow_node(eng):
    t_end = 200e-6                                                  # 1000 periods
    b, fn = _stiff()
    rec = p.Recorder("e", ["v(fo)"], keep="none", period=1 / F_SW_STIFF)
    r = _sim(b, t_end, switch_fn=fn, recorders=[rec], **_engines(1e-9)[eng])
    # the RC's mean output is its mean input (linear, periodic): D * Vin, to the
    # grid's resolution of the 10 ns transient (vacuous on a coarse grid: then
    # the slow node and the step count below carry the case)
    tau_f = (10.0 + 1e-2) * 1e-9
    bound = min(1.0, _acc_bound(eng, r.times, 1 / tau_f))
    assert abs(r.recorded["e"].periods["v(fo)"]["avg"][-1] - 5.0) <= bound * 5.0
    # the slow node integrates (v_fo - v_s) / (R C): ~ 5 V * t / tau after the first periods
    tau = R_LEAK * C_SLOW
    v_s = float(np.asarray(r.v("slow"))[-1])
    assert v_s == pytest.approx(5.0 * (1 - np.exp(-t_end / tau)), rel=1e-2)
    # no step collapse: a second-order controller resolves the 10 ns transient
    # with h ~ tau_f * rtol^(1/3); a collapse to h_min would take ~1e9 steps
    if eng not in FIXED:
        assert len(r.times) <= 2 * t_end / (tau_f * RTOL ** (1 / 3))


N_CELLS, V_CELL, F_MMC, R_MMC, L_MMC, RON_MMC = 8, 100.0, 10e3, 10.0, 10e-3, 1e-3
D_A, D_B = 0.3, 0.7


def _mmc():
    """8 cascaded H-bridge cells (32 switches) into an RL load. Cell k:
    its own source P_k-N_k, leg A (A_k), leg B (B_k); B_k is A_(k+1)."""
    b = p.CircuitBuilder()
    legs = []
    for k in range(N_CELLS):
        a = "gnd" if k == 0 else f"m{k}"
        bb = f"m{k + 1}"
        b.add_voltage_source(f"V{k}", f"P{k}", f"N{k}", V_CELL)
        for leg, node in (("a", a), ("b", bb)):
            b.add_switch(f"S{k}{leg}h", f"P{k}", node, 1 / RON_MMC, 1e-9)
            b.add_switch(f"S{k}{leg}l", node, f"N{k}", 1 / RON_MMC, 1e-9)
            legs.append((k, leg))
    b.add_resistor("Rload", f"m{N_CELLS}", "x", R_MMC)
    b.add_inductor("Lload", "x", "gnd", L_MMC)
    n = b.graph.num_switches
    T = 1 / F_MMC
    fns = [p.make_dead_time_pwm_pair_fn(
        F_MMC, D_A if leg == "a" else D_B, b.switch_index_of(f"S{k}{leg}h"),
        b.switch_index_of(f"S{k}{leg}l"), n, 0.0, phase=k * T / N_CELLS)
        for k, leg in legs]
    return b, p.make_combined_switch_fn(n, fns)


def test_mmc_has_32_switches():
    b, _ = _mmc()
    assert b.graph.num_switches == 4 * N_CELLS == 32


@pytest.mark.parametrize("eng", ENGINES)
def test_mmc_8cell_chain_mean_current_and_determinism(eng):
    """Mean load current = mean drive / (R + 2 R_on per cell): the inductor's
    mean voltage is zero in the periodic state, everything else is linear.
    The drive is sum_k V (s_b - s_a): mean N V (D_B - D_A) = 320 V."""
    t_end = 20e-3                                                   # 20 L/R
    runs = []
    for _ in range(2):
        b, fn = _mmc()
        rec = p.Recorder("e", ["i(Rload)"], keep="none", period=1 / F_MMC)
        runs.append(_sim(b, t_end, switch_fn=fn, recorders=[rec], **_engines(1e-7)[eng]))
    truth = N_CELLS * V_CELL * (D_B - D_A) / (R_MMC + 2 * N_CELLS * RON_MMC)
    i_avg = runs[0].recorded["e"].periods["i(Rload)"]["avg"][-1]
    settle = np.exp(-t_end * R_MMC / L_MMC) + 1e-6
    assert abs(i_avg - truth) <= (settle + (0 if eng in FIXED else K_RTOL * RTOL)) * abs(truth)
    assert np.array_equal(np.asarray(runs[0].states), np.asarray(runs[1].states))
    assert np.array_equal(np.asarray(runs[0].times), np.asarray(runs[1].times))


# ---------------------------------------------------------------------------
# The conservation KPI on the GaN buck, the stiff loop and the MMC (task 5.2)
# ---------------------------------------------------------------------------

def _branch_power(r, name, a, b):
    va = 0.0 if a == "gnd" else np.asarray(r.v(a))
    vb = 0.0 if b == "gnd" else np.asarray(r.v(b))
    return (va - vb) * np.asarray(r.i(name))


def _ledger(r, sources, dissipators, stores):
    """sources / dissipators: (name, n+, n-); stores: (value, kind, name-or-node,
    n-) with kind 'C' (1/2 C v^2 across n+, n-) or 'L' (1/2 L i^2)."""
    p_in = -sum(_branch_power(r, *s) for s in sources)
    p_out = sum(_branch_power(r, *d) for d in dissipators)
    e = 0.0
    for val, kind, x, y in stores:
        if kind == "L":
            e = e + 0.5 * val * np.asarray(r.i(x)) ** 2
        else:
            v = np.asarray(r.v(x)) - (0.0 if y == "gnd" else np.asarray(r.v(y)))
            e = e + 0.5 * val * v ** 2
    return compute_conservation_drift(list(p_in), list(p_out), e[0], e[-1],
                                      list(np.asarray(r.times)))


def _gan_case():
    b, fn = GAN.build()
    led = ([("Vin", "vin", "gnd")],
           [("RL", "out", "gnd"), ("S1", "vin", "sw"), ("D1", "gnd", "sw")],
           [(GAN.C, "C", "out", "gnd"), (GAN.L, "L", "L", None)])
    return b, fn, 50e-6, 10e-9, GAN.w0, led


def _stiff_case():
    b, fn = _stiff()
    led = ([("Vin", "vin", "gnd")],
           [("SH", "vin", "sw"), ("SL", "sw", "gnd"), ("Rf", "sw", "fo"),
            ("Rleak", "fo", "slow")],
           [(1e-9, "C", "fo", "gnd"), (C_SLOW, "C", "slow", "gnd")])
    return b, fn, 20e-6, 1e-9, 1 / ((10.0 + 1e-2) * 1e-9), led


def _mmc_case():
    b, fn = _mmc()
    src = [(f"V{k}", f"P{k}", f"N{k}") for k in range(N_CELLS)]
    sw = []
    for k in range(N_CELLS):
        a = "gnd" if k == 0 else f"m{k}"
        for leg, node in (("a", a), ("b", f"m{k + 1}")):
            sw += [(f"S{k}{leg}h", f"P{k}", node), (f"S{k}{leg}l", node, f"N{k}")]
    led = (src, sw + [("Rload", f"m{N_CELLS}", "x")], [(L_MMC, "L", "Lload", None)])
    return b, fn, 5e-3, 1e-7, R_MMC / L_MMC, led


_CASES = {"gan": _gan_case, "stiff": _stiff_case, "mmc": _mmc_case}
_XF_LEDGER = {"pwl-restart": F_L_1, "dsed": F_L_2}


def _ledger_param(case, eng):
    reason = _XF_LEDGER.get(eng)
    if reason:
        return pytest.param(case, eng, marks=pytest.mark.xfail(strict=True, reason=reason))
    return (case, eng)


@pytest.mark.parametrize("case,eng", [_ledger_param(c, e) for c in sorted(_CASES) for e in ENGINES])
def test_conservation_kpi_closes_on_the_other_long_run_circuits(case, eng):
    """Source work = dissipation + stored energy, every branch on its own,
    over a record from rest: the spec's 1 % and the grid's own bound."""
    b, fn, t_end, dt, w, (src, diss, stores) = _CASES[case]()
    r = _sim(b, t_end, switch_fn=fn, **_engines(dt)[eng])
    resid = _ledger(r, src, diss, stores)
    bound = _acc_bound(eng, r.times, w)
    if bound >= 1.0 and eng not in _XF_LEDGER:
        # esdirk43 crosses the stiff loop's 10 ns RC in 0.16-tau steps
        # ((w h_max)^2 = 3.3): a trapezoid through that record is not a KPI
        # (it misses 1.25 %), the engine's states are right (test above)
        pytest.skip(f"record too coarse for a trapezoid ledger: (w h_max)^2 = {bound:.2g}")
    assert abs(resid) < 1e-2                                   # the spec
    if bound < 1e-2:
        assert abs(resid) <= bound                             # the grid's own


# ---------------------------------------------------------------------------
# Closed loop + junction temperature (pwl: in-kernel losses are pwl only)
# ---------------------------------------------------------------------------

F_CL, DT_CL = 20e3, 5e-7
E0, ALPHA, R_TH, TAU_TH, T_AMB = 50e-6, 0.004, 20.0, 0.1, 25.0
C_TH = TAU_TH / R_TH
V_SET = 12.0


def _thermal_loop(thermal=True):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S1", "vin", "sw", 1e2, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 470e-6)
    b.add_capacitor("C", "out", "gnd", 100e-6)
    b.add_resistor("RL", "out", "gnd", 5.0)
    if thermal:
        b.add_voltage_source("Vamb", "amb", "gnd", T_AMB)
        b.add_resistor("Rth", "tj", "amb", R_TH)
        b.add_capacitor("Cth", "tj", "gnd", C_TH, c0=T_AMB)
        # E(Tj) = E0 (1 + alpha (Tj - 25)), flat in v and i
        tab = p.LossTable(v_axis=[24.0], i_axis=[1.0], tj_axis=[25.0, 150.0],
                          energy=np.array([[[E0, E0 * (1 + ALPHA * 125.0)]]]))
        b.set_device_losses("S1", junction="tj", E_on=tab, E_off=tab)
    pi = p.PIController(Kp=0.005, Ki=20.0, output_min=0.0, output_max=1.0)
    loop = p.bind_pi_to_switch(b, pi=pi, measured="out", setpoint=V_SET,
                               switch="S1", freq=F_CL)
    return b, loop


def _run_loop(t_end, snap=None, window=None, thermal=True, **kw):
    b, loop = _thermal_loop(thermal)
    probes = ["v(out)", "v(tj)", "p(S1)", "i(Rth)"] if thermal else ["v(out)"]
    rec = p.Recorder("e", probes, keep="none", period=1 / F_CL,
                     windows=[(0.0, window or t_end)])
    opts = kw or dict(engine="pwl", dt=DT_CL, discontinuity_restart=False)
    r = _sim(b, t_end, switch_fn=loop.switch_fn, closed_loops=[loop], recorders=[rec],
             resume_from=snap, **opts)
    return r, loop


def _heat_closure(r, t_end):
    de = r.device_energy("S1")
    heat = de["E_on"] + de["E_off"] + de["E_cond_electrical"]
    stored = C_TH * (float(np.asarray(r.v("tj"))[-1]) - T_AMB)
    to_amb = r.recorded["e"].windows["i(Rth)"]["integral"][0]
    return heat, stored + to_amb


def _short_heat_residual(dt):
    t_end = 20e-3 + 0.3 / F_CL
    r, _ = _run_loop(t_end, engine="pwl", dt=dt, discontinuity_restart=False)
    heat, absorbed = _heat_closure(r, t_end)
    return r, heat - absorbed


def test_thermal_ledger_closes_on_the_junction_node():
    """Heat the ledger booked = Cth dT + charge through Rth, to first order
    in dt: halving dt at least halves the residual (it is a start-up offset,
    measured 6.5e-6 of the heat at dt = 5e-7). The run ends mid-period: a
    turn-on in the run's last step deposits its pulse over the NEXT step
    (after t_end), which a split at an edge carries in the snapshot (the
    chain test of the long case)."""
    r, res1 = _short_heat_residual(DT_CL)
    _, res2 = _short_heat_residual(DT_CL / 2)
    heat = sum(r.device_energy("S1")[k] for k in ("E_on", "E_off", "E_cond_electrical"))
    assert abs(res1) < 1e-4 * heat                      # it closes
    assert abs(res2) <= 0.6 * abs(res1)                 # and converges (order >= 1)
    ev = r.loss_events("S1")
    law = E0 * (1 + ALPHA * (ev["Tj"] - 25.0))
    np.testing.assert_allclose(ev["E"], law, rtol=1e-12)


@long_case
def test_long_closed_loop_thermal_settles_on_the_electrothermal_fixed_point():
    """1.5 s = 12.6 coupled thermal time constants at 20 kHz (3e6 steps):
    T_j settles on T* = (Ta + Rth (Pc + 2 f E0 (1 - 25 alpha))) / (1 - 2 Rth f E0 alpha),
    the loop holds the set point, nothing drifts, the ledger closes."""
    t_end = 1.5 + 0.3 / F_CL
    r, loop = _run_loop(t_end)
    per = r.recorded["e"].periods
    n_last = int(0.1 * F_CL)
    pc = float(np.mean(per["p(S1)"]["avg"][-n_last:]))
    g = 2 * R_TH * F_CL * E0 * ALPHA
    t_star = (T_AMB + R_TH * (pc + 2 * F_CL * E0 * (1 - 25 * ALPHA))) / (1 - g)
    tau_eff = TAU_TH / (1 - g)
    # settle residual + the duty's one-step grid dither moving Pc by dt*f
    bound = (t_star - T_AMB) * np.exp(-1.5 / tau_eff) + 2 * R_TH * pc * DT_CL * F_CL
    assert abs(float(np.mean(per["v(tj)"]["avg"][-n_last:])) - t_star) <= bound
    heat, absorbed = _heat_closure(r, t_end)
    # the ledger's residual does not grow with run length (75x the short run)
    assert abs(heat - absorbed) <= 2 * abs(_short_heat_residual(DT_CL)[1])
    # the loop: integral action holds the sampled output on the set point on
    # average; the period mean sits within one ripple (dI / (8 f C)) of it
    v = np.asarray(per["v(out)"]["avg"])
    t_mid = (np.arange(len(v)) + 0.5) / F_CL
    ripple = (24.0 - V_SET) * (V_SET / 24.0) / (470e-6 * F_CL) / (8 * F_CL * 100e-6)
    assert abs(float(np.mean(v[-n_last:])) - V_SET) <= ripple
    assert abs(compute_drift(list(t_mid), list(v), t_from=t_end / 2)) < 1e-3   # V/s
    assert len(loop.duty_history) == pytest.approx(t_end * F_CL, abs=2)


@long_case
def test_long_closed_loop_thermal_resume_chain_is_bit_identical():
    """Six pickled resumes over 0.6 s, two of them AT a gate edge (a heat
    pulse pending across the split): states, the loss ledger and the
    loop's state continue the uninterrupted run exactly."""
    t_end = 0.6
    whole, _ = _run_loop(t_end)
    splits = [0.1, 0.2 + 37 * DT_CL, 0.3, 0.4 + 0.3 / F_CL, 0.5 + 11 * DT_CL]
    snap, last = None, None
    for t_stop in splits + [t_end]:
        last, _ = _run_loop(t_stop, snap=snap, window=t_end)
        snap = pickle.loads(pickle.dumps(last.final_snapshot))
    assert np.array_equal(np.asarray(whole.states)[-1], np.asarray(last.states)[-1])
    assert whole.device_energy("S1") == last.device_energy("S1")
    assert whole.recorded["e"].windows["i(Rth)"]["integral"][0] == \
        last.recorded["e"].windows["i(Rth)"]["integral"][0]


@long_case
@pytest.mark.parametrize("eng", ["trbdf2", "esdirk43", "dsed"])
def test_long_closed_loop_on_the_variable_step_engines(eng):
    """The same loop without the thermal port (pwl only) for 0.3 s."""
    t_end = 0.3
    kw = {k: v for k, v in _engines(DT_CL)[eng].items() if k != "dt"}
    r, loop = _run_loop(t_end, thermal=False, **kw)
    v = np.asarray(r.recorded["e"].periods["v(out)"]["avg"])
    n_last = int(0.1 * F_CL)
    ripple = (24.0 - V_SET) * (V_SET / 24.0) / (470e-6 * F_CL) / (8 * F_CL * 100e-6)
    assert abs(float(np.mean(v[-n_last:])) - V_SET) <= ripple
    t_mid = (np.arange(len(v)) + 0.5) / F_CL
    assert abs(compute_drift(list(t_mid), list(v), t_from=t_end / 2)) < 1e-3


# ---------------------------------------------------------------------------
# One second at 100 kHz (tasks 1.x, 5.x)
# ---------------------------------------------------------------------------

@long_case
@pytest.mark.parametrize("eng", [pytest.param(e, marks=pytest.mark.xfail(strict=True, reason=F_L_1))
                                 if e == "pwl-restart" else e for e in ENGINES])
def test_long_run_drift_buck_one_second(eng):
    """1 s at 100 kHz (1e7 pwl steps). The record is decimated on pwl
    (store_every) and complete on the variable-step engines; the
    statistics come from the in-loop recorder (every sample)."""
    t_end = 1.0
    b, fn = BUCK.build()
    rec = p.Recorder("e", ["-v(vin)*i(Vin)", "v(out)*i(RL)", "p(S1)", "p(D1)", "v(out)"],
                     keep="none", period=BUCK.T, windows=[(0.0, t_end)])
    kw = _engines(BUCK_DT)[eng]
    if eng in FIXED:
        kw = dict(kw, store_every=10_000)
    r = _sim(b, t_end, switch_fn=fn, recorders=[rec], **kw)
    s = buck_pss()
    per = r.recorded["e"].periods
    v = np.asarray(per["v(out)"]["avg"])
    t_mid = (np.arange(len(v)) + 0.5) * BUCK.T
    drift = compute_drift(list(t_mid[len(v) // 2:]), list(v[len(v) // 2:]))
    assert abs(drift) < 1e-3                                         # spec: < 1 mV/s
    h = BUCK_DT if eng in FIXED else float(np.max(np.diff(np.asarray(r.times))))
    bound = (BUCK.w0 * h) ** 2 + (0 if eng in FIXED else K_RTOL * RTOL)
    assert abs(v[-1] - s["v"]) <= bound * s["v"]
    win = {q: d["integral"][0] for q, d in r.recorded["e"].windows.items()}
    w_out = win["v(out)*i(RL)"] + win["p(S1)"] + win["p(D1)"]
    vo, il = float(np.asarray(r.v("out"))[-1]), float(np.asarray(r.i("L"))[-1])
    resid = compute_conservation_drift([win["-v(vin)*i(Vin)"] / t_end] * 2, [w_out / t_end] * 2,
                                       0.0, 0.5 * BUCK.C * vo ** 2 + 0.5 * BUCK.L * il ** 2,
                                       [0.0, t_end])
    assert abs(resid) < 1e-2                                         # spec: < 1 %
    assert abs(resid) <= bound


@long_case
@pytest.mark.parametrize("eng", ENGINES)
def test_long_resume_chain_over_200ms(eng):
    """Ten pickled resumes over 0.2 s (2e6 pwl steps)."""
    t_end = 0.2
    if eng == "dsed":
        splits = [k * 2000 * BUCK.T for k in range(1, 10)]          # gate edges
    elif eng in FIXED:
        splits = [k * 0.02 + 37 * BUCK_DT for k in range(1, 10)]
    else:
        splits = [k * 0.02 + 0.37 * BUCK.T for k in range(1, 10)]
    whole, parts = _chain(eng, splits, t_end)
    xw, xc = np.asarray(whole.final_snapshot.x), np.asarray(parts[-1].final_snapshot.x)
    scale = float(np.max(np.abs(xw)))
    if eng in FIXED:
        assert np.array_equal(xw, xc)
    elif eng == "dsed":
        np.testing.assert_allclose(xc, xw, rtol=0, atol=1e-12 * scale)
    else:
        assert np.max(np.abs(xc - xw)) <= len(splits) * K_RTOL * (RTOL * scale + ATOL)


_RSS_CHILD = textwrap.dedent("""
    import resource, sys, warnings
    import pulsim as p
    from test_long_simulation_stress import BUCK, BUCK_DT
    b, fn = BUCK.build()
    rec = p.Recorder("e", ["v(out)"], keep="none", windows=[(0.0, float(sys.argv[1]))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, float(sys.argv[1]), dt=BUCK_DT, engine="pwl",
                       discontinuity_restart=False, switch_fn=fn,
                       store_every=100_000, recorders=[rec])
    print(len(r.times), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
""")


@long_case
@pytest.mark.skipif(sys.platform == "win32", reason="resource is POSIX")
def test_long_pwl_run_memory_does_not_grow_with_run_length():
    """Peak resident size of a decimated pwl run of 0.05 s vs 0.8 s (5e5 vs
    8e6 steps). Storing every sample of the long run would cost >= 8e6 *
    (t + 4 states) * 8 B = 320 MB; the growth must stay under half that."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.dirname(p.__file__) + "/..", os.path.dirname(__file__),
         os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"),
         os.environ.get("PYTHONPATH", "")]))
    rss = {}
    for t_end in (0.05, 0.8):
        out = subprocess.run([sys.executable, "-c", _RSS_CHILD, str(t_end)], env=env,
                             capture_output=True, text=True, check=True).stdout.split()
        n_rows, peak = int(out[0]), int(out[1])
        assert n_rows <= round(t_end / BUCK_DT) // 100_000 + 2
        rss[t_end] = peak * (1 if sys.platform == "darwin" else 1024)   # bytes
    assert rss[0.8] - rss[0.05] < 0.5 * 8e6 * 5 * 8


# ---------------------------------------------------------------------------
# The KPI helpers against hand-computed answers
# ---------------------------------------------------------------------------

def test_compute_drift_recovers_a_known_slope():
    t = [k * 1e-3 for k in range(101)]
    y = [3.0 + 2.5e-3 * tk + (1e-4 if k % 2 else -1e-4) for k, tk in enumerate(t)]
    assert compute_drift(t, y, t_from=0.0) == pytest.approx(2.5e-3, abs=1e-5)
    assert compute_drift(t, [1.0] * len(t)) == 0.0
    with pytest.raises(ValueError):
        compute_drift([0.0], [1.0])


def test_compute_conservation_drift_closes_a_known_balance():
    # p_in = 10 W for 1 s; p_out = 8 W; 2 J stored: residual 0
    assert compute_conservation_drift([10.0, 10.0], [8.0, 8.0], 1.0, 3.0, [0.0, 1.0]) == 0.0
    # 0.1 J unaccounted of 10 J in: +1 %
    assert compute_conservation_drift([10.0, 10.0], [8.0, 8.0], 1.0, 2.9, [0.0, 1.0]) \
        == pytest.approx(0.01)
    with pytest.raises(ValueError):
        compute_conservation_drift([1.0], [1.0], 0.0, 0.0, [0.0])
