"""add-exact-propagator-strategy group 10 (D8): the fixed-step form of the exact engine,
``integrator='fixed'``. Exact steps go from grid point to grid point (k*dt); a full cell is
x <- Phi x + Gamma from a (mask, dt) cache; a diode event between grid points is located on
the analytic trajectory and the rest of its cell is an exact sub-step to the next grid point.

Truth: the 50-digit references of tasks.md (``probes/references_50_digits.py``, the circuit
of ``numerics/p2_buck.py``)."""
import warnings

import numpy as np
import pytest

import pulsim as p

VIN, L0, C0, GON, GOFF, T = 24.0, 100e-6, 47e-6, 1e3, 1e-9, 1e-5
I_CCM, V_CCM = 11.8015131102418, 12.2288664641139     # state at 40 T, CCM (R = 1, D = 0.5)
V_DCM, TAU_DCM = 8.99624708728943, 0.799830698444468  # DCM orbit (R = 50, D = 0.3)


def _buck(R, D, vth=1e-6):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF, vth)
    b.add_inductor("L", "sw", "vout", L0)
    b.add_capacitor("C", "vout", "gnd", C0)
    b.add_resistor("RL", "vout", "gnd", R)
    fn = p.make_pwm_switch_fn(1 / T, D, b.switch_index_of("S"), b.graph.num_switches, 0.0)
    return b, fn


def _run(R, D, t_end, **kw):
    b, fn = _buck(R, D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=t_end, engine="exact", switch_fn=fn, **kw)


@pytest.mark.parametrize("div", [10, 50, 100])
def test_ccm_buck_on_the_grid_meets_the_50_digit_reference(div):
    # tasks 10.3: dt = T/10, T/50, T/100 within 1e-10 (prototype 7e-14, 1.3e-12, 2.5e-12)
    r = _run(1.0, 0.5, 40 * T, dt=T / div, integrator="fixed")
    assert abs(np.asarray(r.i("L"))[-1] - I_CCM) < 1e-10
    assert abs(np.asarray(r.v("vout"))[-1] - V_CCM) < 1e-10
    ex = r.exact
    # the grid is stepped by the pair: every full cell from a grid point (a cell that starts
    # at an edge an ulp off the grid is a modal step). Measured 352 / 400, 1999 / 2000,
    # 3999 / 4000; errors 3.5e-13, 1.8e-12, 7.6e-13 A
    assert ex["n_fixed_steps"] >= 0.85 * 40 * div
    # tasks 10.4: n^2 + n multiply-adds per pair step (n = 2), one pair and one
    # eigendecomposition per visited mask (no factorization after the first visit)
    assert ex["fixed_step_flops"] == ex["n_fixed_steps"] * (2 * 2 + 2)
    n_masks = len(ex["masks"])
    assert ex["n_eig"] == n_masks and ex["n_phi_gamma_builds"] <= n_masks
    assert ex["n_phi_gamma_refused"] == 0


@pytest.mark.parametrize("div", [10, 100])
def test_fixed_form_equals_the_event_driven_form(div):
    # tasks 10.3: identical to the event-driven form to 1e-11 (same dt as its step cap)
    a = _run(1.0, 0.5, 40 * T, dt=T / div, integrator="fixed")
    b = _run(1.0, 0.5, 40 * T, dt=T / div)
    assert b.exact["n_fixed_steps"] == 0
    for get in (lambda r: r.i("L"), lambda r: r.v("vout")):
        assert abs(np.asarray(get(a))[-1] - np.asarray(get(b))[-1]) < 1e-11


def _starts(r, ts):
    t, v = np.asarray(r.times), np.asarray(r.v("vout"))
    return np.array([v[np.where(np.abs(t - s) < 1e-13)[0][-1]] for s in ts])


def test_dcm_orbit_on_the_grid_with_the_turn_off_between_grid_points():
    # tasks 10.2 / 10.3: the diode turns off at 0.7998 T, between grid points of T/10; the
    # event is located on the analytic trajectory and the cell finished by a sub-step.
    # Orbit within 1e-9 of the 50-digit reference after 4000 periods.
    nper = 4000
    r = _run(50.0, 0.3, nper * T, dt=T / 10, integrator="fixed")
    t = np.asarray(r.times)
    k = int(np.argmin(abs(t - (nper - 1) * T)))
    assert abs(t[k] - (nper - 1) * T) < 1e-13
    assert abs(np.asarray(r.v("vout"))[k] - V_DCM) < 1e-9
    off = r.event_times[(r.event_types == 2) & (r.event_times > (nper - 1) * T)]
    assert abs((off[0] - (nper - 1) * T) / T - TAU_DCM) < 5e-12
    # every grid point is a sample (the sub-step lands on it)
    last = t[(t > (nper - 1) * T - 1e-13)]
    grid = (nper - 1) * T + np.arange(11) * T / 10
    assert all(np.min(np.abs(last - g)) < 1e-13 for g in grid)
    assert r.exact["n_fixed_steps"] > 0.7 * nper * 10
    # the same orbit as the event-driven form at the period starts
    q = _run(50.0, 0.3, nper * T, dt=T / 10)
    ts = np.arange(nper - 50, nper) * T
    assert np.abs(_starts(r, ts) - _starts(q, ts)).max() < 1e-11


def test_fixed_needs_dt():
    b, fn = _buck(1.0, 0.5)
    with pytest.raises(ValueError, match="pass dt"):
        p.simulate(b, t_end=T, engine="exact", switch_fn=fn, integrator="fixed")
