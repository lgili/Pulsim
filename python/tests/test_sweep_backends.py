"""Sweep backends and the step-loop callback report they route on.

`sweep` / `monte_carlo` with ``parallel=True`` run on threads when the
step loop is native (the kernel releases the GIL) and on processes when
it calls Python. Every check here is deterministic: results must be
identical to the serial run, never "faster than".
"""

from __future__ import annotations

import numpy as np
import pytest

import pulsim as p

T_END, DT = 2e-4, 1e-7  # 2000 steps: tiny, the point is equality


def build_buck(R=10.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", g_on=1e3, g_off=1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", R)
    return b


def build_rc(R=10.0):
    """No switch at all; picklable, so it can reach a process pool (the
    native schedules from make_pwm_switch_fn cannot)."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "in", "gnd", 0.0, 10.0, 5e3)
    b.add_resistor("R", "in", "vout", R)
    b.add_capacitor("C", "vout", "gnd", 1e-6)
    return b


def native_pwm(duty=0.5):
    b = build_buck()
    return p.make_pwm_switch_fn(100e3, duty, b.switch_index_of("Q1"),
                                b.graph.num_switches)


class PyPwm:
    """A picklable Python switch_fn (a lambda would not reach a process
    pool)."""

    def __init__(self, duty):
        b = build_buck()
        self.duty = duty
        self.n, self.q1 = b.graph.num_switches, b.switch_index_of("Q1")

    def __call__(self, t):
        m = p.SwitchStateMask(self.n)
        if (t * 100e3) % 1.0 < self.duty:
            m.set(self.q1, True)
        return m


def kpi(res, params):
    v = np.asarray(res.v("vout"))
    return {"v_end": float(v[-1]), "v_sum": float(v.sum())}


LOADS = [5.0 + k for k in range(8)]  # every job differs: order shows


def failing_build(R):
    if R == LOADS[3]:
        raise RuntimeError("job three refuses")
    return build_buck(R)


def failing_rc(R):
    if R == LOADS[3]:
        raise RuntimeError("job three refuses")
    return build_rc(R)


def retired_rc(R):
    b = build_rc(R)
    b.add_nonlinear_diode("Dn", "vout", "gnd", p.IdealDiodeParams())  # retired
    return b


def shockley_rc(R):
    b = build_rc(R)
    b.add_shockley_diode("D", "vout", "gnd")
    return b


def _sweep(switch_fn=None, **kw):
    return p.sweep(build_buck, params={"R": LOADS}, kpi_fn=kpi,
                   t_end=T_END, dt=DT, switch_fn=switch_fn or native_pwm(),
                   verbose=False, **kw)


def _sweep_rc(**kw):
    return p.sweep(build_rc, params={"R": LOADS}, kpi_fn=kpi,
                   t_end=T_END, dt=DT, verbose=False, **kw)


def _assert_same(a, b):
    assert a.kpis.keys() == b.kpis.keys()
    for k in a.kpis:
        assert np.array_equal(a.kpis[k], b.kpis[k]), k
    for k in a.params:
        assert np.array_equal(a.params[k], b.params[k]), k


# ---- step-loop callback report -------------------------------------------

@pytest.mark.parametrize("engine,dt", [("pwl", DT), ("trbdf2", None),
                                       ("dsed", None)])
def test_native_schedule_reports_no_python_callback(engine, dt):
    res = p.simulate(build_buck(), t_end=T_END, dt=dt, engine=engine,
                     switch_fn=native_pwm())
    assert res.step_loop_python_callbacks == ()


@pytest.mark.parametrize("engine,dt", [("pwl", DT), ("trbdf2", None),
                                       ("dsed", None)])
def test_python_switch_fn_is_reported(engine, dt):
    res = p.simulate(build_buck(), t_end=T_END, dt=dt, engine=engine,
                     switch_fn=PyPwm(0.5))
    assert res.step_loop_python_callbacks == ("switch_fn",)


def test_step_observer_is_reported():
    res = p.simulate(build_buck(), t_end=T_END, dt=DT,
                     switch_fn=native_pwm(), step_observer=lambda t, x: None)
    assert "step_observer" in res.step_loop_python_callbacks
    assert "switch_fn" not in res.step_loop_python_callbacks


def test_periodic_step_observer_is_reported_on_trbdf2():
    # After fix-control-tick-semantics the trbdf2 branch takes its
    # ticks from the control plan; the report must still name them.
    res = p.simulate(build_buck(), t_end=T_END, engine="trbdf2",
                     switch_fn=native_pwm(),
                     step_observer=lambda t, x: None,
                     controller_period=1e-5)
    assert res.step_loop_python_callbacks == ("step_observer",)


@pytest.mark.parametrize("parallel,backend", [(False, "auto"),
                                              (True, "thread")])
def test_a_refusal_is_not_a_failed_point(parallel, backend):
    # add-engine-conformance-matrix: an UnsupportedError holds for every
    # point, so it propagates instead of landing in `failed`.
    def factory(R):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "a", "gnd", 1.0)
        b.add_resistor("R", "a", "o", R)
        b.add_shockley_diode("D", "o", "gnd")
        return b

    with pytest.raises(p.UnsupportedError):
        p.sweep(factory, params={"R": [1e3, 2e3]},
                kpi_fn=lambda r, prm: {}, parallel=parallel,
                backend=backend, n_workers=2, verbose=False,
                t_end=1e-5, engine="dsed")


def test_combination_with_a_python_callable_is_reported():
    n = build_buck().graph.num_switches
    mixed = p.make_combined_switch_fn(n, [native_pwm(), PyPwm(0.1)])
    res = p.simulate(build_buck(), t_end=T_END, dt=DT, switch_fn=mixed)
    assert res.step_loop_python_callbacks == ("switch_fn",)


# ---- sweep backends --------------------------------------------------------

@pytest.fixture(scope="module")
def serial():
    return _sweep()


def test_native_sweep_runs_on_threads_with_serial_results(serial):
    assert serial.backend == "serial"
    par = _sweep(parallel=True, n_workers=4)
    assert par.backend == "thread"
    assert par.failed == []
    _assert_same(par, serial)


def test_python_callback_sweep_runs_on_processes():
    ser = _sweep(switch_fn=PyPwm(0.5))
    par = _sweep(switch_fn=PyPwm(0.5), parallel=True, n_workers=2)
    assert par.backend == "process"
    _assert_same(par, ser)


def test_forced_backends_are_honoured():
    ser_rc = _sweep_rc()
    assert len(set(ser_rc.kpis["v_end"])) == len(LOADS)
    proc = _sweep_rc(parallel=True, n_workers=2, backend="process")
    assert proc.backend == "process"
    _assert_same(proc, ser_rc)
    ser_py = _sweep(switch_fn=PyPwm(0.5))
    thr = _sweep(switch_fn=PyPwm(0.5), parallel=True, n_workers=2,
                 backend="thread")
    assert thr.backend == "thread"
    _assert_same(thr, ser_py)


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="backend"):
        _sweep(parallel=True, backend="dask")


def test_failing_job_is_isolated_on_threads():
    out = p.sweep(failing_build, params={"R": LOADS}, kpi_fn=kpi,
                  t_end=T_END, dt=DT, switch_fn=native_pwm(),
                  parallel=True, n_workers=4, backend="thread",
                  verbose=False)
    assert [i for i, _ in out.failed] == [3]
    assert "job three refuses" in out.failed[0][1]
    v = out.kpis["v_end"]
    assert np.isnan(v[3]) and np.isfinite(np.delete(v, 3)).all()


def test_unpicklable_builder_factory_runs_on_threads(serial):
    offset = 0.0

    def closure_build(R):  # a closure: pickle refuses it
        return build_buck(R + offset)

    out = p.sweep(closure_build, params={"R": LOADS}, kpi_fn=kpi,
                  t_end=T_END, dt=DT, switch_fn=native_pwm(),
                  parallel=True, n_workers=4, verbose=False)
    assert out.backend == "thread"
    _assert_same(out, serial)


def test_seeded_monte_carlo_is_backend_independent():
    def run(**kw):
        return p.monte_carlo(
            build_rc, distributions={"R": lambda r: r.uniform(5.0, 15.0)},
            kpi_fn=kpi, n_samples=4, seed=42, t_end=T_END, dt=DT,
            verbose=False, **kw)

    ser = run()
    # No switch_fn: the fixed engine's default mask is the native
    # make_constant_switch_fn (update-fixed-step-loop-bookkeeping), so
    # "auto" takes threads (it kept processes on a Python default mask).
    for backend, ran in (("auto", "thread"), ("thread", "thread"),
                         ("process", "process")):
        par = run(parallel=True, n_workers=2, backend=backend)
        assert par.backend == ran
        _assert_same(par, ser)


# One rule on every backend: a configuration refusal propagates, any
# other raising job is recorded in `failed`. Before, the process backend
# raised on a per-job failure, a RetiredDeviceError was recorded as a
# failed point (empty KPIs) on serial / thread, and backend="auto" did
# one or the other depending on job 0.
_ALL = [(False, "auto"), (True, "thread"), (True, "process")]


@pytest.mark.parametrize("parallel,backend", _ALL)
def test_failing_job_is_recorded_on_every_backend(parallel, backend):
    out = p.sweep(failing_rc, params={"R": LOADS}, kpi_fn=kpi,
                  t_end=T_END, dt=DT, parallel=parallel, n_workers=2,
                  backend=backend, verbose=False)
    assert [i for i, _ in out.failed] == [3]
    assert "job three refuses" in out.failed[0][1]
    v = out.kpis["v_end"]
    assert np.isnan(v[3]) and np.isfinite(np.delete(v, 3)).all()


@pytest.mark.parametrize("parallel,backend", _ALL)
def test_retired_device_propagates_on_every_backend(parallel, backend):
    with pytest.raises(p.RetiredDeviceError):
        p.sweep(retired_rc, params={"R": LOADS[:3]}, kpi_fn=kpi,
                t_end=T_END, dt=DT, parallel=parallel, n_workers=2,
                backend=backend, verbose=False)


def test_unsupported_error_propagates_from_a_process_pool():
    with pytest.raises(p.UnsupportedError):
        p.sweep(shockley_rc, params={"R": LOADS[:3]}, kpi_fn=kpi,
                t_end=1e-5, parallel=True, n_workers=2,
                backend="process", verbose=False, engine="dsed")


def test_retired_device_propagates_from_monte_carlo():
    with pytest.raises(p.RetiredDeviceError):
        p.monte_carlo(retired_rc,
                      distributions={"R": lambda r: r.uniform(9.0, 11.0)},
                      n_samples=3, kpi_fn=kpi, seed=1, t_end=T_END,
                      dt=DT, verbose=False)
