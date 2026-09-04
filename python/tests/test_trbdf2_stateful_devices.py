"""The TR-BDF2 second stage for the three devices that carry state.

Until now the router simply REFUSED the variable-step engine for the
Lauritzen diode, the IGBT tail and the MNA-native PMSM. The reason
was not conservatism. TR-BDF2 takes a trapezoidal stage over
gamma*h and then a BDF2 stage over the rest, and the two stages
approximate dX/dt differently:

    trapezoidal   dX/dt ~ (2/h)*(X - X_n) - (dX/dt)_n
    BDF2 stage 2  dX/dt ~ (c1*X + c2*X_gamma + c3*X_n)/h

but c1/h == 2/(gamma*h) by construction, so the CONDUCTANCE is
identical between them. Only the history term differs. Stamp the
trapezoidal history inside a BDF2 stage and you get the right
matrix, the right sparsity, a perfectly quadratic Newton — and a
converged answer to the wrong problem. That is the failure these
tests exist to catch, and it is why the stage is an explicit
argument through every stamp rather than a default.

The coefficients themselves are pinned by exact order conditions in
core/tests/dsed/test_trbdf2_stage_coeffs.cpp. What is checked HERE
is the wiring: that each device, run end-to-end on the variable-step
engine, lands on the same answer as the fixed-step trapezoidal
engine. A history term stamped in the wrong stage does not fail
loudly; it lands somewhere else.
"""

import math

import numpy as np
import pytest

import pulsim as p


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


# ---------------------------------------------------------------
# The router used to name all three. It must not any more.
# ---------------------------------------------------------------
def _blockers(b):
    return p._trbdf2_blockers(
        b, dt=None, step_observer=None, closed_loops=[], on_cache=None,
        live_stream=None, progress=False, start_from_dc_op=False,
        strict_event_iterations=False, switch_fn=None,
        controller_period=None, max_dt_halvings=None, store_every=None,
        mmc_arms=None, enable_substep_state_correction=None,
        inductor_freeze_di_max=None, inductor_abs_clamp=None)


def test_the_three_devices_no_longer_block_the_router():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_lauritzen_diode("D1", "m", "gnd", tau=1e-7, T_M=1e-8)
    b.add_igbt_level1("Q1", "a", "e", "g", 1.5, 0.05, 5.0,
                      tau_tail=1e-6, k_tail=0.30)
    b.add_resistor("Re", "e", "gnd", 1.0)
    b.add_voltage_source("Vg", "g", "gnd", 0.0)
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th",
                   R_s=0.5, L_d=1e-3, L_q=3e-3, psi_pm=0.05,
                   pole_pairs=4, J=1e-3, B=1e-4)
    for name in ("Lauritzen", "tail", "PMSM"):
        assert not any(name in w for w in _blockers(b)), _blockers(b)


# ---------------------------------------------------------------
# 1. Lauritzen diode — reverse recovery is an INTEGRAL of the
#    charge state, so a mis-stamped history moves it directly.
# ---------------------------------------------------------------
def _lauritzen(engine, step):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 400.0)
    b.add_inductor("Lload", "dc", "sw", 200e-6, i0=20.0)
    b.add_inductor("Lstray", "d2", "dc", 4e-6, i0=20.0)
    b.add_lauritzen_diode("D1", "sw", "d2", tau=1e-7, T_M=1e-8)
    b.add_switch("S1", "sw", "gnd", 1e3, 1e-9)
    b.add_resistor("Rsnub", "sw", "gnd", 1e5)
    n, k = b.graph.num_switches, b.switch_index_of("S1")

    def sf(t):
        m = p.SwitchStateMask(n)
        if t >= 1e-6:
            m.set(k, True)
        return m

    kw = dict(t_end=2e-6, switch_fn=sf)
    if engine == "pwl":
        res = p.simulate(b, dt=step, engine="pwl", **kw)
    else:
        res = p.simulate(b, dt=step, engine="trbdf2",
                         rtol=1e-6, atol=1e-9, **kw)
    t = np.asarray(res.times)
    i = np.asarray(res.i("Lstray"))
    return float(-np.trapezoid(np.minimum(i, 0.0), t))


def test_lauritzen_reverse_recovery_agrees_across_engines():
    ref = _lauritzen("pwl", 1e-10)
    got = _lauritzen("trbdf2", 2e-9)
    # A real, sizeable Q_rr — otherwise "they agree" would be the
    # trivial agreement of two zeros, which is what a dead stamp
    # produces.
    assert ref > 0.3e-6, ref
    assert _rel(got, ref) < 0.02, (got, ref)


# ---------------------------------------------------------------
# 2. IGBT turn-off tail — the tail charge decays with tau_tail, so
#    E_off integrates the history term over ~5 time constants.
# ---------------------------------------------------------------
def _igbt_tail(engine, step, tail=True):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 600.0)
    b.add_pulse_voltage_source("Vg", "g", "gnd", 0.0, 15.0, 0.0,
                               2e-6, 0.0, 1e-8, 1e-8)
    b.add_inductor("Lload", "dc", "c", 500e-6, i0=100.0)
    b.add_diode("Dfw", "c", "dc", 1e3, 1e-9, 0.7)
    # k_tail=0.0 is refused by name (a tail needs both numbers or
    # neither), so "no tail" means omitting both.
    tail_kw = dict(tau_tail=1e-6, k_tail=0.30) if tail else {}
    b.add_igbt_level1("Q1", "c", "e", "g", 1.5, 0.05, 5.0, **tail_kw)
    b.add_resistor("Rsense", "e", "gnd", 1e-4)
    b.add_capacitor("Cs", "c", "gnd", 2e-9)
    n = b.graph.num_switches

    def sf(_t):
        return p.SwitchStateMask(n)

    kw = dict(t_end=6e-6, switch_fn=sf)
    if engine == "pwl":
        res = p.simulate(b, dt=step, engine="pwl", **kw)
    else:
        res = p.simulate(b, dt=step, engine="trbdf2",
                         rtol=1e-6, atol=1e-9, **kw)
    t = np.asarray(res.times)
    v = np.asarray(res.v("c"))
    i = np.asarray(res.i("Rsense"))
    m = t > 1.95e-6
    return float(np.trapezoid(np.maximum(v[m] * i[m], 0.0), t[m]))


def test_igbt_tail_turnoff_energy_agrees_across_engines():
    ref = _igbt_tail("pwl", 1e-9)
    got = _igbt_tail("trbdf2", 2e-8)
    assert ref > 1e-3, ref
    assert _rel(got, ref) < 0.02, (got, ref)


def test_igbt_tail_is_actually_stamped_on_the_variable_engine():
    """Guard against the agreement above being vacuous. If the tail
    history never ran under TR-BDF2, k_tail would not matter — and
    two engines that both ignore it would still 'agree'."""
    with_tail = _igbt_tail("trbdf2", 2e-8, tail=True)
    without = _igbt_tail("trbdf2", 2e-8, tail=False)
    assert with_tail > without * 1.05, (with_tail, without)


# ---------------------------------------------------------------
# 3. MNA-native PMSM — the only one of the three where the two
#    stages have different SHAPES: BDF2 is one-sided, so the
#    previous step's dlambda/dt is absent, not rescaled.
# ---------------------------------------------------------------
def _pmsm(engine, step, t_end=0.02):
    b = p.CircuitBuilder()
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(f"Vs_{'abc'[k]}", node, "gnd",
                                  v_dc=0.0, v_amplitude=12.0,
                                  frequency=30.0,
                                  phase=-2.0 * math.pi / 3.0 * k)
    b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th",
                   R_s=0.5, L_d=1e-3, L_q=3e-3, psi_pm=0.05,
                   pole_pairs=4, J=1e-3, B=1e-4)
    if engine == "pwl":
        res = p.simulate(b, t_end=t_end, dt=step, engine="pwl")
    else:
        res = p.simulate(b, t_end=t_end, dt=step, engine="trbdf2",
                         rtol=1e-6, atol=1e-9)
    return float(np.asarray(res.v("w"))[-1])


def test_pmsm_speed_agrees_across_engines():
    ref = _pmsm("pwl", 1e-7)
    got = _pmsm("trbdf2", 4e-6)
    # The machine has actually accelerated; comparing two numbers
    # near zero would pass with any history term at all.
    assert abs(ref) > 5.0, ref
    assert _rel(got, ref) < 1e-3, (got, ref)


def test_pmsm_reaches_the_same_answer_in_far_fewer_steps():
    """The point of routing these devices onto TR-BDF2 at all: the
    same answer, at a step the fixed engine cannot take. Measured,
    not asserted from the requested dt — the adaptive controller
    picks its own."""
    def run(engine, step):
        b = p.CircuitBuilder()
        for k, node in enumerate(("ua", "ub", "uc")):
            b.add_sine_voltage_source(
                f"Vs_{'abc'[k]}", node, "gnd", v_dc=0.0,
                v_amplitude=12.0, frequency=30.0,
                phase=-2.0 * math.pi / 3.0 * k)
        b.add_pmsm_mna("M1", "ua", "ub", "uc", "nn", "w", "th",
                       R_s=0.5, L_d=1e-3, L_q=3e-3, psi_pm=0.05,
                       pole_pairs=4, J=1e-3, B=1e-4)
        kw = (dict(engine="pwl") if engine == "pwl"
              else dict(engine="trbdf2", rtol=1e-6, atol=1e-9))
        res = p.simulate(b, t_end=0.02, dt=step, **kw)
        return (float(np.asarray(res.v("w"))[-1]), len(res.times))

    ref, n_ref = run("pwl", 1e-7)
    got, n_got = run("trbdf2", 4e-6)
    assert _rel(got, ref) < 1e-3, (got, ref)
    assert n_got * 10 < n_ref, (n_got, n_ref)


# ---------------------------------------------------------------
# The stage switch must not depend on the tolerance.
# ---------------------------------------------------------------
@pytest.mark.parametrize("rtol,atol", [(1e-4, 1e-7), (1e-6, 1e-9),
                                       (1e-8, 1e-11)])
def test_the_answer_converges_with_the_tolerance_not_drifts(rtol, atol):
    """Tightening the tolerance makes the controller reject and
    re-take steps. A rejected step must re-solve from the SAME
    committed history: if a stage-2 commit leaked into one, the
    answer would WANDER as the tolerance changed instead of settling
    onto the fixed-step reference."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 600.0)
    b.add_pulse_voltage_source("Vg", "g", "gnd", 0.0, 15.0, 0.0,
                               2e-6, 0.0, 1e-8, 1e-8)
    b.add_inductor("Lload", "dc", "c", 500e-6, i0=100.0)
    b.add_diode("Dfw", "c", "dc", 1e3, 1e-9, 0.7)
    b.add_igbt_level1("Q1", "c", "e", "g", 1.5, 0.05, 5.0,
                      tau_tail=1e-6, k_tail=0.30)
    b.add_resistor("Rsense", "e", "gnd", 1e-4)
    b.add_capacitor("Cs", "c", "gnd", 2e-9)
    n = b.graph.num_switches
    res = p.simulate(b, t_end=6e-6, dt=2e-8, engine="trbdf2",
                     rtol=rtol, atol=atol,
                     switch_fn=lambda _t: p.SwitchStateMask(n))
    t = np.asarray(res.times)
    v = np.asarray(res.v("c"))
    i = np.asarray(res.i("Rsense"))
    m = t > 1.95e-6
    got = float(np.trapezoid(np.maximum(v[m] * i[m], 0.0), t[m]))
    assert _rel(got, _igbt_tail("pwl", 1e-9)) < 0.03, got
