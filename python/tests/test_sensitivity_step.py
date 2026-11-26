"""add-parameter-sensitivities, group 5 (and 4.4): forward sensitivities on
the time-stepping engines, the stage recurrence of design D1.

``simulate(engine='pwl' | 'trbdf2', sensitivities=[...])`` differentiates the
discrete step the run took: J·S = a·(dψ/dp − ∂q/∂p) − ∂f/∂p per stage, on the
stage's own factor (`core/include/pulsim/solver/step/sensitivity.hpp`).

Truths, each independent of the code under test:
- the RC step's closed form (derived here);
- central differences of the SAME fixed-step scheme (pwl: the step grid does
  not move with the parameter, so the difference of two runs is the
  derivative of the same map);
- for a Newton law on TR-BDF2: central differences of a scipy Radau
  solution at rtol 1e-12 of the reduced ODE (an independent integrator).
"""

import math
import warnings

import numpy as np
import pytest

import pulsim as p

from . import _sens_reference as ref


def _run(b, t_end, engine, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, engine=engine, **kw)


# ---------------------------------------------------------------- RC step
V, R0, C0 = 10.0, 1e3, 1e-6
TAU = R0 * C0


def _rc(R=R0, C=C0):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", V)
    b.add_resistor("R", "in", "out", R)
    b.add_capacitor("C", "out", "gnd", C)
    return b


def _rc_closed_form(t, R=R0, C=C0):
    # v = V(1 − e^{−t/RC}): ∂v/∂R = −V·t/(R²C)·e^{−t/RC}, ∂v/∂C = −V·t/(RC²)·e^{−t/RC}
    e = math.exp(-t / (R * C))
    return -V * t / (R * R * C) * e, -V * t / (R * C * C) * e


# The fixed engine's release default opens a run with the restart rule's three
# SDIRK(2,2) steps (two stages each, one more than the trapezoid): a pwl run of
# n steps solves the sensitivity system n_p (n + 3) times (F-Y-1 closed by the
# sensitivity-events merge 65f09dc3: the split steps carry the recurrence).
SDIRK_EXTRA = 3


def test_rc_on_trbdf2_matches_the_closed_form():
    """5.4: at t = RC, ∂v/∂R = −V/(R·e), ∂v/∂C = −V/(C·e) within 1e-5
    relative on TR-BDF2 at rtol 1e-8."""
    b = _rc()
    r = _run(b, TAU, "trbdf2", rtol=1e-8, atol=1e-12,
             sensitivities=["R.R", "C.C"])
    s = r.sensitivity
    assert s.engine == "trbdf2" and s.params == ["R.R", "C.C"]
    assert s.times[-1] == pytest.approx(TAU, rel=1e-15)
    dR, dC = _rc_closed_form(TAU)
    assert dR == pytest.approx(-V / (R0 * math.e), rel=1e-15)
    assert dC == pytest.approx(-V / (C0 * math.e), rel=1e-15)
    out = b.node_id_of("out")
    got_R = r.sens("out", "R.R")[-1]
    got_C = s.states[-1, out, 1]
    assert abs(got_R / dR - 1) < 1e-5, got_R
    assert abs(got_C / dC - 1) < 1e-5, got_C
    # the whole trajectory, not only the end
    t = s.times
    ref_R = np.array([_rc_closed_form(tk)[0] for tk in t])
    assert np.max(np.abs(r.sens("out", "R.R") - ref_R)) < 1e-5 * abs(dR)


def test_rc_on_esdirk43_matches_the_closed_form():
    """The ESDIRK4(3) stages carry the same recurrence: at t = RC within
    1e-5 relative, over the whole trajectory."""
    b = _rc()
    r = _run(b, TAU, "trbdf2", integrator="esdirk43", rtol=1e-8, atol=1e-12,
             sensitivities=["R.R", "C.C"])
    s = r.sensitivity
    dR, dC = _rc_closed_form(TAU)
    assert abs(r.sens("out", "R.R")[-1] / dR - 1) < 1e-5
    assert abs(r.sens("out", "C.C")[-1] / dC - 1) < 1e-5
    ref_R = np.array([_rc_closed_form(tk)[0] for tk in s.times])
    assert np.max(np.abs(r.sens("out", "R.R") - ref_R)) < 1e-5 * abs(dR)
    # 5 implicit stages, one first solve per stage and parameter
    assert s.counters["n_sens_solves"] == 2 * 5 * r.stats.n_accept
    assert s.counters["n_sens_iterations"] == 0


def _pwl_rc_final(R=R0, C=C0, dt=1e-6, t_end=TAU):
    b = _rc(R, C)
    r = _run(b, t_end, "pwl", dt=dt)
    return float(r.states[-1][b.node_id_of("out")])


def test_rc_on_pwl_matches_central_differences_and_the_closed_form():
    b = _rc()
    r = _run(b, TAU, "pwl", dt=1e-6, sensitivities=["R.R", "C.C"])
    out = b.node_id_of("out")
    S = r.sensitivity.states[-1, out]
    fd_R = ref.central_difference(lambda v: _pwl_rc_final(R=v), R0,
                                  rel_steps=(1e-3, 1e-4))[0]
    fd_C = ref.central_difference(lambda v: _pwl_rc_final(C=v), C0,
                                  rel_steps=(1e-3, 1e-4))[0]
    assert abs(S[0] / fd_R - 1) < 1e-6
    assert abs(S[1] / fd_C - 1) < 1e-6
    # the trapezoid from a zero derivative slot is O(dt) off at the start:
    # the continuous derivative only to 1e-3 here
    dR, dC = _rc_closed_form(TAU)
    assert abs(S[0] / dR - 1) < 1e-3 and abs(S[1] / dC - 1) < 1e-3


def test_linear_counters_and_factorizations():
    """5.5: n_sens_solves = n_p × stages × accepted steps on a linear
    circuit, no refinement, and the factor traffic unchanged."""
    b = _rc()
    r = _run(b, 2e-4, "pwl", dt=1e-6, sensitivities=["R.R", "C.C"])
    c = r.sensitivity.counters
    n_steps = len(r.sensitivity.times) - 1
    assert n_steps == 200
    assert c == {"n_sens_solves": 2 * (n_steps + SDIRK_EXTRA), "n_sens_iterations": 0,
                 "n_sens_discarded": 0}

    b0, b1 = _rc(), _rc()
    r0 = _run(b0, 2e-3, "trbdf2", rtol=1e-6)
    r1 = _run(b1, 2e-3, "trbdf2", rtol=1e-6, sensitivities=["R.R", "C.C"])
    n_acc = int(r1.stats.n_accept)
    assert n_acc == len(r1.sensitivity.times) - 1
    c1 = r1.sensitivity.counters
    assert (c1["n_sens_solves"], c1["n_sens_iterations"]) == (2 * 2 * n_acc, 0)
    # a rejected attempt's stages were solved too: at most 2 per parameter
    assert c1["n_sens_discarded"] <= 2 * 2 * int(r1.stats.n_reject)
    for k in ("n_factorizations", "n_solves", "n_accept", "n_reject",
              "n_newton_iterations"):
        assert getattr(r0.stats, k) == getattr(r1.stats, k), k
    assert np.array_equal(np.asarray(r0.times), np.asarray(r1.times))
    assert np.array_equal(np.asarray(r0.states), np.asarray(r1.states))


# ------------------------------------------------- Shockley rectifier (4.4)
P0 = dict(R=1.0, C=100e-6, Rl=50.0, I_S=1e-9, n=2.0)
V_T = 0.025          # n·V_T = 0.05 V, the prototype's
NAMES = {"R": "R.R", "C": "C.C", "Rl": "Rl.R", "I_S": "D.I_S", "n": "D.n"}
NEWTON = dict(tol_newton_dx=1e-12, tol_newton_res=1e-12)


def _rect(**kw):
    v = dict(P0, **kw)
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "in", "gnd", 0.0, 20.0, 50.0)
    b.add_resistor("R", "in", "a", v["R"])
    b.add_shockley_diode("D", "a", "out", I_S=v["I_S"], n=v["n"], V_T=V_T)
    b.add_capacitor("C", "out", "gnd", v["C"])
    b.add_resistor("Rl", "out", "gnd", v["Rl"])
    return b


def _pwl_rect(t_end=40e-3, dt=1e-5, sens=None, **kw):
    b = _rect(**kw)
    r = _run(b, t_end, "pwl", dt=dt, sensitivities=sens, **NEWTON)
    return b, r


def test_shockley_rectifier_on_pwl_matches_central_differences():
    """5.4 / 4.4: the trapezoidal rectifier of the prototype (4000 steps over
    40 ms), through the Shockley law: forward sensitivities within 1e-6 of
    central differences of the same fixed-step scheme."""
    names = list(NAMES.values())
    b, r = _pwl_rect(sens=names)
    out = b.node_id_of("out")
    a = b.node_id_of("a")
    S = r.sensitivity.states[-1]
    worst = {}
    for j, (k, nm) in enumerate(NAMES.items()):
        def fin(v, k=k):
            bb, rr = _pwl_rect(**{k: v})
            x = np.asarray(rr.states[-1])
            return np.array([x[bb.node_id_of("out")], x[bb.node_id_of("a")]])
        fd, est, _h = ref.central_difference(fin, P0[k], rel_steps=(1e-4, 1e-5))
        got = np.array([S[out, j], S[a, j]])
        worst[nm] = ref.rel_err(got, fd)
        assert worst[nm] < 1e-6, (nm, got, fd, est)
    # Newton: refinement on the last iteration's factor, at most two per stage
    c = r.sensitivity.counters
    assert c["n_sens_solves"] == len(names) * (4000 + SDIRK_EXTRA)
    assert 0 < c["n_sens_iterations"] <= 2 * c["n_sens_solves"]


def test_states_and_factorizations_identical_with_sensitivities():
    """5.4 / 5.5: the run's states are the same bits with and without
    sensitivities, and the Newton factorizations are the same count."""
    for engine, kw in (("pwl", dict(dt=1e-5)), ("trbdf2", dict(rtol=1e-6))):
        b0, b1 = _rect(), _rect()
        r0 = _run(b0, 20e-3, engine, **kw, **NEWTON)
        r1 = _run(b1, 20e-3, engine, sensitivities=["D.I_S", "C.C"], **kw,
                  **NEWTON)
        assert np.array_equal(np.asarray(r0.times), np.asarray(r1.times)), engine
        assert np.array_equal(np.asarray(r0.states), np.asarray(r1.states)), engine
        assert r0.newton.factorizations == r1.newton.factorizations, engine
        assert r0.newton.iterations == r1.newton.iterations, engine
        assert r0.newton.factorizations > 0


def _radau_rect(t_end, **kw):
    """The reduced ODE (v1 algebraic, solved per call) by scipy Radau at
    rtol 1e-12: an integrator independent of the engines."""
    from scipy.integrate import solve_ivp
    v = dict(P0, **kw)
    R, C, Rl, Is, nvt = v["R"], v["C"], v["Rl"], v["I_S"], v["n"] * V_T
    gmin = 1e-12   # the law's default G_min

    def i_d(u):
        return Is * math.expm1(u / nvt) + gmin * u, Is / nvt * math.exp(u / nvt) + gmin

    v1_prev = [0.0]

    def rhs(t, y):
        vs = 20.0 * math.sin(2 * math.pi * 50.0 * t)
        v2 = y[0]
        v1 = v1_prev[0]
        for _ in range(100):   # (v1 − vs)/R + i_D(v1 − v2) = 0
            i, g = i_d(v1 - v2)
            f = (v1 - vs) / R + i
            dv = -f / (1.0 / R + g)
            v1 += dv
            if abs(dv) < 1e-15 * (1 + abs(v1)):
                break
        v1_prev[0] = v1
        return [(i_d(v1 - v2)[0] - v2 / Rl) / C]

    sol = solve_ivp(rhs, (0.0, t_end), [0.0], method="Radau", rtol=1e-12,
                    atol=1e-14)
    return float(sol.y[0, -1])


@pytest.mark.parametrize("k,integrator", [
    *[(k, None) for k in ("R", "C", "Rl", "I_S", "n")],
    ("C", "esdirk43"), ("I_S", "esdirk43")])
def test_shockley_rectifier_on_trbdf2_against_an_independent_integration(
        k, integrator):
    """4.4: a circuit gradient THROUGH a Newton law on TR-BDF2 (two stages,
    the BDF2 history of dq/dp) and on ESDIRK4(3) against central differences
    of a Radau solution at rtol 1e-12."""
    pytest.importorskip("scipy")
    t_end = 10e-3
    b = _rect()
    r = _run(b, t_end, "trbdf2", rtol=1e-9, atol=1e-12, integrator=integrator,
             sensitivities=[NAMES[k]], **NEWTON)
    got = float(r.sens("out", NAMES[k])[-1])
    fd = ref.central_difference(lambda v: _radau_rect(t_end, **{k: v}), P0[k],
                                rel_steps=(1e-4,))[0]
    assert abs(got / float(fd) - 1) < 1e-4, (got, fd)


# --------------------------------------------- Level-1 MOSFET (4.4)
def _mos(K=0.5, V_T=2.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdd", "vdd", "gnd", 12.0)
    b.add_sine_voltage_source("Vg", "g", "gnd", 4.0, 1.5, 1e3)
    b.add_resistor("Rd", "vdd", "d", 20.0)
    b.add_capacitor("Cd", "d", "gnd", 10e-6)
    b.add_mosfet_level1("M", "d", "gnd", "g", K=K, V_T=V_T,
                        with_body_diode=False)
    return b


@pytest.mark.parametrize("k", ["K", "V_T"])
def test_mosfet_level1_on_pwl_matches_central_differences(k):
    p0 = {"K": 0.5, "V_T": 2.0}[k]
    b = _mos()
    r = _run(b, 2e-3, "pwl", dt=1e-6, sensitivities=[f"M.{k}"], **NEWTON)
    got = float(r.sens("d", f"M.{k}")[-1])

    def fin(v):
        bb = _mos(**{k: v})
        rr = _run(bb, 2e-3, "pwl", dt=1e-6, **NEWTON)
        return float(rr.states[-1][bb.node_id_of("d")])

    fd = ref.central_difference(fin, p0, rel_steps=(1e-4, 1e-5))[0]
    assert abs(got / float(fd) - 1) < 1e-6, (got, fd)


# ------------------------------------------------------------- refusals
def _diode_rect():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "in", "gnd", 0.0, 20.0, 50.0)
    b.add_resistor("R", "in", "a", 1.0)
    b.add_diode("D", "a", "out", 1e3, 1e-9)
    b.add_capacitor("C", "out", "gnd", 100e-6)
    b.add_resistor("Rl", "out", "gnd", 50.0)
    return b


def test_an_unlocated_state_event_is_refused_by_name():
    """Group 6: a diode commutation the fixed engine did not locate in time
    (no event splitting) is refused, naming the diode; located events are
    carried (test_sensitivity_events.py)."""
    with pytest.raises(p.UnsupportedError) as ei:
        _run(_diode_rect(), 20e-3, "pwl", sensitivities=["C.C"], dt=1e-5)
    assert (ei.value.engine, ei.value.feature) == ("pwl", "sensitivities")
    assert "group 6" in str(ei.value) and "'D' turns" in str(ei.value)


def test_resume_and_other_refusals():
    b = _rc()
    r = _run(b, 1e-4, "trbdf2")
    for engine, kw in (("trbdf2", {}), ("pwl", dict(dt=1e-6))):
        with pytest.raises(p.UnsupportedError) as ei:
            _run(_rc(), 2e-4, engine, resume_from=r.final_snapshot,
                 sensitivities=["R.R"], **kw)
        assert ei.value.feature == "sensitivities"
        assert "snapshot" in str(ei.value)
    with pytest.raises(p.UnsupportedError) as ei:
        _run(_rc(), 1e-4, "auto", sensitivities=["R.R"])
    assert ei.value.feature == "sensitivities"
    with pytest.raises(p.UnsupportedError) as ei:
        _run(_rc(), 1e-4, "pwl", dt=1e-6, sensitivities=["V.V"])
    assert ei.value.feature == "sensitivity_parameter"
    with pytest.raises(ValueError, match="no component"):
        _run(_rc(), 1e-4, "pwl", dt=1e-6, sensitivities=["X.R"])
    b = _rc()
    b.add_saturable_inductor("Ls", "out", "gnd", 1e-3, 1.0)
    with pytest.raises(p.UnsupportedError) as ei:
        _run(b, 1e-4, "pwl", dt=1e-6, sensitivities=["R.R"])
    assert ei.value.feature == "sensitivities"
    assert "Ls" in str(ei.value)


def _loss_rc():
    b = _rc()
    b.add_switch("S1", "in", "o", 1e6, 1e-12)
    b.add_resistor("RL", "o", "gnd", 10.0)
    b.add_voltage_source("Vamb", "amb", "gnd", 25.0)
    b.add_resistor("Rth", "tj", "amb", 1.0)
    b.add_capacitor("Cth", "tj", "amb", 1e-3)
    t = p.LossTable(v_axis=[100.0], i_axis=[1.0], tj_axis=[25.0],
                    energy=np.full((1, 1, 1), 1e-3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b.set_device_losses("S1", junction="tj", E_on=t, E_off=t)
    return b


@pytest.mark.parametrize("engine,kw", [("pwl", dict(dt=1e-6)), ("trbdf2", {})])
def test_what_the_recurrence_does_not_carry_is_refused(engine, kw):
    """In-kernel losses (heat from earlier states), Levenberg–Marquardt
    (a factor J + λI) and an on_cache hook are refused by name, never a
    silently wrong derivative."""
    with pytest.raises(p.UnsupportedError) as ei:
        _run(_loss_rc(), 1e-4, engine, sensitivities=["R.R"],
             switch_fn=p.make_pwm_switch_fn(10e3, 0.5, 0, 1), **kw)
    assert ei.value.feature == "sensitivities" and "losses" in str(ei.value)
    with pytest.raises(p.UnsupportedError) as ei:
        _run(_rect(), 1e-4, engine, sensitivities=["C.C"],
             enable_newton_lm=True, **kw, **NEWTON)
    assert ei.value.feature == "sensitivities" and "lm" in str(ei.value)
    with pytest.raises(p.UnsupportedError) as ei:
        _run(_rc(), 1e-4, engine, sensitivities=["R.R"],
             on_cache=lambda c: None, **kw)
    # trbdf2 refuses on_cache on its own
    assert ei.value.feature in ("sensitivities", "on_cache")


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_a_transmission_line_is_refused_by_name(engine):
    """The Bergeron line injects its own past, which the stage recurrence
    does not differentiate: refused by name (merge 14t), never a silent
    derivative without the line's history."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "b", 50.0)
    b.add_transmission_line("T", "b", "gnd", "c", "gnd", Z0=50.0, delay=1e-7,
                            model="bergeron")
    b.add_resistor("RL", "c", "gnd", 50.0)
    with pytest.raises(p.UnsupportedError, match="TransmissionLine") as ei:
        p.simulate(b, t_end=1e-6, dt=1e-8, engine=engine, sensitivities=["R.R"])
    assert ei.value.feature == "sensitivities"
