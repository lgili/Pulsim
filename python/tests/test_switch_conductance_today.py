"""add-branch-form-ideal-switch task 1.1: what the resistor-pair switch does with extreme values.

Port of ``probes/current_kernel_goff0.py``. ``g_off = 0`` is a legal conductance (the leak is
off): opened on an inductor carrying 9.93 A it leaves the inductor's companion with no path, and
the fixed engine rings at +-397 V while TR-BDF2 reaches ~1e9 V, with no warning. That is the
defect the branch form's re-initialisation removes (task 5); these tests pin it until then.
"""
import math
import warnings

import numpy as np
import pytest

import pulsim as p


def _rl(g_on, g_off, closed, engine, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    # The conductance form is the subject (branch is the release default): pinned.
    b.add_switch("S", "vin", "a", g_on, g_off, form="conductance")
    b.add_resistor("R", "a", "b", 1.0)
    b.add_inductor("L", "b", "gnd", 1e-3)
    n = b.graph.num_switches
    k = b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, bool(closed(t)))
        return m

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=10e-3, switch_fn=fn, engine=engine, **kw)


def test_closed_switch_control_conducts():
    r = _rl(1e3, 1e-9, lambda t: True, "pwl", dt=5e-5)
    # Closed form of the RL behind 1 mOhm. The restart-less trapezoidal start
    # was 1.06e-5 low (pinned 9.98955 +- 1e-5); the release start: 9.3e-7.
    exact = 10 / 1.001 * (1 - math.exp(-10e-3 * 1.001 / 1e-3))
    assert r.i("L")[-1] == pytest.approx(exact, abs=2e-6)


def test_goff_zero_opened_on_an_inductor_rings_on_pwl():
    # The restart-less trapezoid's +-397 V alternation is the subject (with the
    # release restart rule the same defect rings at 958 V): pinned legacy.
    r = _rl(1e3, 0.0, lambda t: t < 5e-3, "pwl", dt=5e-5, discontinuity_restart=False)
    v = np.asarray(r.v("b"))
    i = np.asarray(r.i("L"))
    assert np.all(np.isfinite(v))
    assert abs(i[-1]) < 1e-12
    assert 390.0 < np.abs(v).max() < 400.0           # +-396.8 V, sign alternating
    late = v[np.asarray(r.times) > 6e-3]
    assert np.all(np.abs(late) > 390.0) and np.all(np.sign(late[1:]) == -np.sign(late[:-1]))


def test_goff_zero_opened_on_an_inductor_explodes_on_trbdf2():
    r = _rl(1e3, 0.0, lambda t: t < 5e-3, "trbdf2")
    assert np.abs(np.asarray(r.v("b"))).max() > 1e9


def test_infinite_g_on_is_refused_at_build_time():
    # Before task 2.1 this built and aborted mid-run "numerically singular".
    with pytest.raises(ValueError, match=r'add_switch\("S"\): g_on = inf'):
        _rl(float("inf"), 1e-9, lambda t: True, "pwl", dt=5e-5)
