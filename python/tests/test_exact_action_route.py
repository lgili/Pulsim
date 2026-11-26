"""add-exact-propagator-strategy 15.2 (D13): a mode routed to the action of the exponential.

An RLC ladder of 100 sections (200 states) visited for 400 steps: the dense setup
(dense_setup_n3 n^3, calibrated in 15.4) is not repaid, the action's plan is within m_max, so the mode
steps by the truncated-Taylor action on its sparse matrix and has no eigenbasis. Its
event predicates are evaluated on the action's states and located by the engine's
bracketing. Truth: scipy's dense expm of the ladder's own ODE, written here from the
element values (not from the engine), and a bisection on it for the diode's turn-on."""
import warnings

import numpy as np
import pytest

import pulsim as p

expm = pytest.importorskip("scipy.linalg").expm

M_SEC, L0, C0, RS, VIN, DT = 100, 10e-6, 1e-6, 10e-3, 10.0, 1e-6
RLOAD, GON, GOFF, VTH = 10.0, 1.0, 1e-9, 1.0


def _ladder(diode):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a0", "gnd", VIN)
    for k in range(1, M_SEC + 1):
        b.add_resistor(f"R{k}", f"a{k - 1}", f"r{k}", RS)
        b.add_inductor(f"L{k}", f"r{k}", f"a{k}", L0)
        b.add_capacitor(f"C{k}", f"a{k}", "gnd", C0)
    if diode:
        b.add_diode("D", f"a{M_SEC}", "o", GON, GOFF, VTH)
        b.add_resistor("RL", "o", "gnd", RLOAD)
    else:
        b.add_resistor("RL", f"a{M_SEC}", "gnd", RLOAD)
    return b


def _aug(g_load):
    """[[A, b], [0, 0]], states [i_1, v_1, ..., i_m, v_m, 1]; the last capacitor sees g_load."""
    n = 2 * M_SEC
    A = np.zeros((n + 1, n + 1))
    for k in range(M_SEC):
        i, v = 2 * k, 2 * k + 1
        A[i, i] = -RS / L0
        A[i, v] = -1 / L0
        if k == 0:
            A[i, n] = VIN / L0
        else:
            A[i, v - 2] = 1 / L0
        A[v, i] = 1 / C0
        if k + 1 < M_SEC:
            A[v, i + 2] = -1 / C0
    A[n - 1, n - 1] = -g_load / C0
    return A


def _x0():
    x = np.zeros(2 * M_SEC + 1)
    x[-1] = 1.0
    return x


def _states(r):
    return np.array([[np.asarray(r.i(f"L{k}"))[-1], np.asarray(r.v(f"a{k}"))[-1]]
                     for k in range(1, M_SEC + 1)]).ravel()


def _run(b, t_end):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=t_end, engine="exact", dt=DT, integrator="rk45")


def test_ladder_steps_by_the_action_and_meets_dense_expm():
    t_end = 400 * DT
    r = _run(_ladder(False), t_end)
    rt = r.exact["routes"]
    assert [q["kernel"] for q in rt] == ["action"]
    assert rt[0]["n"] == 201 and r.exact["n_eig"] == 0
    assert rt[0]["action_steps"] >= 400
    ref = expm(_aug(1 / RLOAD) * t_end) @ _x0()
    err = np.abs(_states(r) - ref[:-1]).max()
    assert np.abs(ref[:-1]).max() > 1.0          # the ladder moved
    assert err <= 1e-9, err


def test_diode_turn_on_located_on_the_action_trajectory():
    # the far end reaches V_th after the line's delay (~ m sqrt(LC) = 316 us); off, the
    # diode's v_D = v_m / (1 + g_off R); on, the last capacitor sees g_on / (1 + g_on R)
    t_end = 400 * DT
    r = _run(_ladder(True), t_end)
    rt = r.exact["routes"]
    assert all(q["kernel"] == "action" for q in rt) and len(rt) == 2
    A_off, A_on = _aug(GOFF / (1 + GOFF * RLOAD)), _aug(GON / (1 + GON * RLOAD))
    x0 = _x0()

    def g(t):
        return (expm(A_off * t) @ x0)[2 * M_SEC - 1] / (1 + GOFF * RLOAD) - VTH

    lo, hi = 0.0, t_end
    assert g(lo) < 0 < g(hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    t_on = 0.5 * (lo + hi)
    ev = np.asarray(r.event_times)
    assert len(ev) == 1
    assert abs(ev[0] - t_on) <= 1e-12, ev[0] - t_on
    ref = expm(A_on * (t_end - t_on)) @ (expm(A_off * t_on) @ x0)
    err = np.abs(_states(r) - ref[:-1]).max()
    assert err <= 1e-8, err


def test_small_circuits_stay_dense():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_capacitor("C", "b", "gnd", 1e-6)
    r = _run(b, 10e-6)
    assert [(q["kernel"], q["reason"]) for q in r.exact["routes"]] == [("dense", "n < route_min_n")]


def test_double_crossing_inside_one_step_is_found_on_an_action_mode():
    # An undamped LC tank (v_C = 1 - cos(w t)) beside a 32-section ladder (66 states: routed).
    # dt = one tank period, so v_C is back near 0 at every step end: the crossing of
    # V_th = 1.5 at w t = 2 pi / 3 and back at 4 pi / 3 is invisible to the step ends and
    # is found only by the scan (an action mode keeps its eigenvalues for it).
    lt, ct, vth, msec = 1e-6, 1e-6, 1.5, 32
    w = 1 / np.sqrt(lt * ct)
    dt = 2 * np.pi / w
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a0", "gnd", VIN)
    for k in range(1, msec + 1):
        b.add_resistor(f"R{k}", f"a{k - 1}", f"r{k}", RS)
        b.add_inductor(f"L{k}", f"r{k}", f"a{k}", L0)
        b.add_capacitor(f"C{k}", f"a{k}", "gnd", C0)
    b.add_resistor("RL", f"a{msec}", "gnd", RLOAD)
    b.add_voltage_source("VT", "t0", "gnd", 1.0)
    b.add_inductor("LT", "t0", "t1", lt)
    b.add_capacitor("CT", "t1", "gnd", ct)
    b.add_diode("D", "t1", "o", GON, GOFF, vth)
    b.add_resistor("RD", "o", "gnd", RLOAD)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=3 * dt, engine="exact", dt=dt, integrator="rk45")
    assert r.exact["routes"][0]["kernel"] == "action"
    # truth: the tank's own ODE with the off diode's leak g = g_off / (1 + g_off R_D), by expm
    g_l = GOFF / (1 + GOFF * RLOAD)
    A = np.array([[0.0, -1 / lt, 1 / lt], [1 / ct, -g_l / ct, 0.0], [0.0, 0.0, 0.0]])

    def gv(t):
        return (expm(A * t) @ np.array([0.0, 0.0, 1.0]))[1] / (1 + GOFF * RLOAD) - vth

    lo, hi = 0.5 * dt / np.pi * 2, 0.5 * dt          # brackets w t = 2 pi / 3
    assert gv(lo) < 0 < gv(hi) and gv(dt) < 0       # the step end sees no crossing
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gv(mid) < 0 else (lo, mid)
    ev = np.asarray(r.event_times)
    assert len(ev) >= 1
    assert abs(ev[0] - 0.5 * (lo + hi)) <= 1e-12, ev[0] - 0.5 * (lo + hi)
