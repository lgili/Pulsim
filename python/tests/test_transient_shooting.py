"""Periodic steady state on the fixed and variable engines
(add-periodic-steady-state-shooting group 5, tasks 5.1-5.5).

`periodic_steady_state(engine='pwl' | 'trbdf2')` shoots on the true state
with the engine's own period runs (the fixed engine with its restart rule
and event splitting) and M by finite differences + Broyden. The oracle is
the engine itself: the orbit must be what that engine's long transient
reaches. Counters decide, never wall time.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import pulsim as p
from pulsim._result_views import states_as_array
from pulsim.harmonic import run_harmonic_balance
from pulsim.periodic import run_periodic_shooting

from . import _pss_reference as ref

T = ref.T
import pulsim._transient_shooting  # noqa: E402,F401

ts = sys.modules["pulsim._transient_shooting"]


def test_pulsim_is_imported_from_this_tree():
    assert Path(p.__file__).resolve().is_relative_to(
        Path(__file__).resolve().parents[2])


def _buck(R, D, phase=0.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", ref.VIN)
    b.add_switch("S", "vin", "sw", ref.GON, ref.GOFF)
    b.add_diode("D1", "gnd", "sw", ref.GON, ref.GOFF)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", R)
    return b, p.make_pwm_switch_fn(1 / T, D, b.switch_index_of("S"),
                                   b.graph.num_switches, phase)


def _boost(R):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_switch("S", "sw", "gnd", ref.GON, ref.GOFF)
    b.add_diode("D1", "sw", "vout", ref.GON, ref.GOFF)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", R)
    return b, p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                                   b.graph.num_switches)


CASES = {"buck CCM": lambda: _buck(1.0, 0.5),
         "buck DCM": lambda: _buck(50.0, 0.3),
         "boost CCM": lambda: _boost(20.0)}


def _engine_kw(engine, dt):
    if engine == "pwl":
        return dict(dt=dt, discontinuity_restart=True,
                    enable_substep_state_correction=True)
    return dict(dt=dt, rtol=1e-10, atol=1e-13)


def _v(r):
    return r.x0[r.state_labels.index("v(vout)")]


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
@pytest.mark.parametrize("name", list(CASES))
def test_the_orbit_is_the_engines_own_4000_period_transient(name, engine):
    """5.4: on each engine, the shooting orbit and the state its own
    4000-period transient reaches agree within 1e-8 of the state scale
    (measured 1.5e-14 .. 4.2e-9)."""
    b, sf = CASES[name]()
    r = p.periodic_steady_state(b, period=T, switch_fn=sf, engine=engine,
                                dt=T / 100)
    b2, sf2 = CASES[name]()
    long = p.simulate(b2, t_end=4000 * T, engine=engine, switch_fn=sf2,
                      **_engine_kw(engine, T / 100))
    t = np.asarray(long.times)
    X = states_as_array(long) @ r._ctx.Cm.T
    i = int(np.argmin(np.abs(t - 4000 * T)))
    j = int(np.argmin(np.abs(t - 3999 * T)))
    assert abs(t[i] - 4000 * T) <= 1e-9 * T
    scale = float(np.abs(X[j:]).max())
    assert np.abs(X[i] - X[j]).max() <= 1e-9 * scale   # the transient settled
    assert np.abs(X[i] - r.x0).max() <= 1e-8 * scale
    # and the engine orbit is the exact one to the engine's accuracy
    bx, sfx = CASES[name]()
    ex = p.periodic_steady_state(bx, period=T, switch_fn=sfx)
    assert _v(r) == pytest.approx(_v(ex), rel=1e-6)
    assert np.abs(np.sort(np.abs(r.multipliers))
                  - np.sort(np.abs(ex.multipliers))).max() <= 1e-4
    assert r.stable and r._ctx.loop == "fd-broyden"


def test_fixed_engine_orbit_is_converged_in_dt():
    """5.4: the fixed engine at T/100 and T/101 on the DCM buck: within
    1e-4 (measured 2.4e-9 relative)."""
    def solve(dt):
        b, sf = _buck(50.0, 0.3)
        return p.periodic_steady_state(b, period=T, switch_fn=sf,
                                       engine="pwl", dt=dt)
    a, b = solve(T / 100), solve(T / 101)
    assert _v(a) == pytest.approx(_v(b), rel=1e-4)
    assert _v(a) == pytest.approx(ref.DCM_ORBIT[1], rel=1e-6)


def test_dcm_buck_from_zero_in_at_most_12_period_runs_on_pwl(monkeypatch):
    """5.5: every period run is a simulate() call through the map; count
    them and hold the budget (12 measured: 1 + 3 Jacobians of 2 runs
    + 5 Broyden steps)."""
    calls = []
    orig = ts.TransientMap.simulate

    def counted(self, x, periods=1):
        calls.append(periods)
        return orig(self, x, periods)

    monkeypatch.setattr(ts.TransientMap, "simulate", counted)
    b, sf = _buck(50.0, 0.3)
    r = p.periodic_steady_state(b, period=T, switch_fn=sf, engine="pwl",
                                dt=T / 100)
    assert len(calls) == r.n_period_runs <= 12
    assert r._ctx.n_jacobians >= 2 and r._ctx.n_broyden >= 1
    # the full MNA vector of the section resumes the engine's transient
    res = p.simulate(b, t_end=T, engine="pwl", switch_fn=sf,
                     initial_state=r.full_state, **_engine_kw("pwl", T / 100))
    x1 = states_as_array(res)[-1] @ r._ctx.Cm.T
    assert np.abs(x1 - r.x0).max() <= 1e-9 * abs(_v(r))


def test_the_multipliers_are_the_fixed_engines():
    """The multipliers come from a difference Jacobian taken at the
    orbit (the final refresh): the DCM buck's of the 50-digit prototype
    within 1e-5."""
    b, sf = _buck(50.0, 0.3)
    r = p.periodic_steady_state(b, period=T, switch_fn=sf, engine="pwl",
                                dt=T / 100)
    mu = np.sort(np.abs(r.multipliers))
    assert mu[1] == pytest.approx(ref.DCM_MULTIPLIERS[1], abs=1e-5)
    assert mu[0] < 1e-6


def test_refusals_on_the_transient_engines():
    b, sf = _buck(50.0, 0.3, phase=0.25 * T)   # phase in seconds
    with pytest.raises(p.UnsupportedError) as ei:
        p.periodic_steady_state(b, period=T, switch_fn=sf, engine="pwl",
                                dt=T / 100)
    assert ei.value.feature == "section"
    ok, why = p.periodic_steady_state_supported(b, T, sf, engine="pwl",
                                                dt=T / 100)
    assert not ok and "gate edge at the section" in why
    assert p.periodic_steady_state_supported(b, T, sf, engine="trbdf2")[0]
    assert p.periodic_steady_state_supported(b, T, sf)[0]

    b, sf = _buck(50.0, 0.3)
    with pytest.raises(p.UnsupportedError) as ei:
        p.periodic_steady_state(b, period=T, switch_fn=sf, engine="pwl",
                                dt=T / 100, sensitivities=["RL.R"])
    assert ei.value.feature == "sensitivities"
    with pytest.raises(ValueError, match="not an integer"):
        p.periodic_steady_state(b, period=T, switch_fn=sf, engine="pwl",
                                dt=T / 99.5)
    with pytest.raises(ValueError, match="needs dt"):
        p.periodic_steady_state(b, period=T, switch_fn=sf, engine="pwl")
    with pytest.raises(ValueError, match="takes neither"):
        p.periodic_steady_state(b, period=T, switch_fn=sf, dt=T / 100)

    def opaque(t, _m=sf):
        return _m(t)
    with pytest.raises(p.UnsupportedError) as ei:
        p.periodic_steady_state(b, period=T, switch_fn=opaque,
                                engine="trbdf2")
    assert ei.value.feature == "switch_fn"
    for k in ("t_end", "initial_state", "step_observer"):
        with pytest.raises(p.UnsupportedError) as ei:
            p.periodic_steady_state(b, period=T, switch_fn=sf,
                                    engine="trbdf2", options={k: None})
        assert ei.value.feature == k

    s = p.CircuitBuilder()
    s.add_voltage_source("Vin", "vin", "gnd", 24.0)
    s.add_switch("S", "vin", "sw", 1e3, 1e-9)
    s.add_shockley_diode("D1", "gnd", "sw")
    s.add_inductor("L", "sw", "vout", 100e-6)
    s.add_capacitor("C", "vout", "gnd", 47e-6)
    s.add_resistor("RL", "vout", "gnd", 5.0)
    sf = p.make_pwm_switch_fn(1 / T, 0.5, s.switch_index_of("S"),
                              s.graph.num_switches)
    with pytest.raises(p.UnsupportedError) as ei:
        p.periodic_steady_state(s, period=T, switch_fn=sf, engine="pwl",
                                dt=T / 100)
    assert ei.value.device == "D1"


def test_run_periodic_shooting_delegates_and_refuses_callbacks():
    """5.3: the v1 front end is the delegate (the DCM flip is in
    test_periodic_steady_state); a non-convergence is converged=False
    with NaN, never the last iterate."""
    b, sf = _buck(50.0, 0.3)
    with pytest.warns(DeprecationWarning):
        r = run_periodic_shooting(b, t_period=T, dt=T / 100, switch_fn=sf)
    assert r.converged
    assert r.x_steady_state[b.node_id_of("vout")] == pytest.approx(
        ref.DCM_ORBIT[1], rel=1e-6)
    with pytest.warns(DeprecationWarning), \
            pytest.raises(p.UnsupportedError) as ei:
        run_periodic_shooting(b, t_period=T, dt=T / 100, switch_fn=sf,
                              b_extra_fn=lambda t: None)
    assert ei.value.feature == "b_extra_fn"
    with pytest.warns(DeprecationWarning):
        r = run_periodic_shooting(
            b, t_period=T, dt=T / 100, switch_fn=sf,
            x_initial=np.zeros(b.pool.state_size(b.graph)))
    assert r.converged and r.n_simulations <= 12


def test_run_periodic_shooting_non_convergence_is_nan(monkeypatch):
    from pulsim.periodic_steady_state import _conv

    def no(*a, **k):
        raise _conv("no periodic orbit found within max_period_runs=1")
    monkeypatch.setattr(p, "periodic_steady_state", no)
    b, sf = _buck(50.0, 0.3)
    with pytest.warns(DeprecationWarning):
        r = run_periodic_shooting(b, t_period=T, dt=T / 100, switch_fn=sf)
    assert not r.converged and np.isnan(r.x_steady_state).all()
    assert "max_period_runs" in r.message


def test_run_harmonic_balance_reads_the_orbit_period():
    b, sf = _buck(50.0, 0.3)
    with pytest.warns(DeprecationWarning):
        h = run_harmonic_balance(b, period=T, dt=T / 100, switch_fn=sf,
                                   n_harmonics=5)
    k = b.node_id_of("vout")
    # the DC bin is the orbit's (time-weighted) mean of v(vout)
    b2, sf2 = _buck(50.0, 0.3)
    r = p.periodic_steady_state(b2, period=T, switch_fn=sf2, engine="pwl",
                                dt=T / 100)
    run = r.orbit
    t = np.asarray(run.times)
    v = states_as_array(run)[:, k]
    assert h.amplitudes[k, 0] == pytest.approx(
        np.trapezoid(v, t) / (t[-1] - t[0]), rel=1e-5)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_a_lossless_loop_is_not_returned_as_an_orbit(engine):
    """An undamped LC beside a switched branch: its multipliers lie on
    the unit circle and its transient never settles. A difference
    Jacobian puts them at 1 - 1.8e-9 (pwl) and 1 + 5.8e-9 (trbdf2), so a
    1e-10 marginal test passed them (an orbit on pwl, an 'unstable' one
    on trbdf2); the band is the Jacobian's accuracy (1e-6, 1e-4)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_inductor("Lx", "a", "m", 1e-3)
    b.add_capacitor("Cx", "m", "gnd", 1e-6)
    b.add_voltage_source("V2", "q", "gnd", 1.0)
    b.add_resistor("Rq", "q", "r", 1.0)
    b.add_switch("S", "r", "gnd", ref.GON, ref.GOFF)
    sf = p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                              b.graph.num_switches)
    with pytest.raises(p.ConvergenceError, match=r"not isolated.*(Lx|Cx)"):
        p.periodic_steady_state(b, period=T, switch_fn=sf, engine=engine,
                                dt=T / 100)
