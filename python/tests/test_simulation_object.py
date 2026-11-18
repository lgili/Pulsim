"""add-kernel-simulation-object: the option x engine table, the one defaults
table, and the kernel `Simulation` object."""

import inspect
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim import _run_options as ro


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_resistor("R", "in", "n1", 1e3)
    b.add_capacitor("C", "n1", "gnd", 1e-6)
    return b


def test_the_table_covers_every_simulate_parameter():
    params = set(inspect.signature(p.simulate).parameters) - {"builder"}
    assert set(ro.TABLE) - {"c_blocks"} == params


def test_option_capabilities_is_the_table():
    caps = p.option_capabilities()
    assert caps["store_every"] == {"pwl": True, "trbdf2": False, "dsed": False}
    assert caps["rtol"] == {"pwl": False, "trbdf2": True, "dsed": True}
    assert all(set(v) == {"pwl", "trbdf2", "dsed"} for v in caps.values())


def test_every_default_comes_from_one_table():
    d = ro.DEFAULTS
    so = p.SimulationOptions()
    assert so.max_newton_iterations == d["max_newton_iterations"]
    assert so.tol_newton_dx == d["tol_newton_dx"] == so.tol_newton_res
    assert so.max_event_iterations == d["max_event_iterations"]
    assert so.max_dt_halvings == d["max_dt_halvings"]
    assert so.newton_roundoff_budget == d["newton_roundoff_budget"]
    assert so.store_every == d["store_every"]
    t = ro.trbdf2_options(engine="trbdf2", t_start=0.0, t_end=1.0)
    assert (t["rtol"], t["atol"]) == (d["rtol_variable"], d["atol_variable"])
    assert t["max_newton_iterations"] == d["max_newton_iterations"]
    from pulsim._dsed_dispatch import DSEDOptions, _resolve_dsed_options
    for o in (DSEDOptions(), _resolve_dsed_options()):
        assert (o.rtol, o.atol, o.dt_init, o.dt_max) == (
            d["rtol_exact"], d["atol_exact"], d["dt_init_exact"],
            d["dt_max_exact"])
        assert o.stiffness_threshold == d["stiffness_threshold"]
        assert o.h_bdf2 == d["h_bdf2"]
    # The signature's own literals say "not set" (None), never a value.
    sig = inspect.signature(p.simulate).parameters
    for n in ("rtol", "atol", "dt_init", "max_newton_iterations",
              "max_event_iterations", "tol_newton_dx", "max_dt_halvings",
              "store_every", "newton_roundoff_budget"):
        assert sig[n].default is None, n
    assert p.SolverOptions().max_newton_iterations is None
    # The YAML loader's `simulation:` block defaults.
    y = p.load_yaml_string("""
circuit:
  devices:
    - {type: resistor, name: R, from: a, to: gnd, R: 1.0}
simulation: {t_end: 1.0e-4, dt: 1.0e-6}
""").options
    assert (y.max_newton_iterations, y.max_event_iterations,
            y.tol_newton_dx, y.tol_newton_res, y.store_every) == (
        d["max_newton_iterations"], d["max_event_iterations"],
        d["tol_newton_dx"], d["tol_newton_res"], d["store_every"])


# A value per keyword that asks its engine for something.
_ASK = {
    "rtol": 1e-6, "atol": 1e-9, "dt_init": 1e-9, "integrator": "rk45",
    "stiffness_threshold": 5.0, "h_bdf2": 1e-7,
    "step_observer": lambda t, x: None, "start_from_dc_op": True,
    "enable_nonlinear_refresh": True, "max_newton_iterations": 30,
    "max_event_iterations": 8, "strict_event_iterations": True,
    "max_dt_halvings": 2, "store_every": 2, "tol_newton_dx": 1e-8,
    "tol_newton_res": 1e-8, "enable_newton_line_search": True,
    "enable_newton_lm": True, "newton_roundoff_budget": 50,
    "enable_substep_state_correction": True, "discontinuity_restart": True,
    "resolution_diagnostics": True, "inductor_freeze_di_max": 1e3,
    "inductor_abs_clamp": 1e3, "progress": True,
    "should_continue": lambda: True, "on_cache": lambda cache: None,
    "sensitivities": ["R.R"], "mode_selection": "lcp",
    "allow_implicit_segments": True, "control_tick": "state",
}
# Cells exercised elsewhere (they need a controller, a stream, an MMC arm or
# a snapshot to be built): test_closed_loop*, test_live*, test_mmc*,
# test_snapshot_resume*, test_control_tick*.
_ELSEWHERE = {"closed_loops", "live_stream", "mmc_arms", "controller_period",
              "resume_from"}


def _refused_cells():
    for name, row in ro.TABLE.items():
        if name == "c_blocks" or row["unset"] == "always":
            continue
        for eng in ("pwl", "trbdf2", "dsed"):
            if eng not in row["engines"]:
                yield name, eng


def test_every_refused_cell_is_tested_here_or_elsewhere():
    names = {n for n, _ in _refused_cells()}
    assert names <= set(_ASK) | _ELSEWHERE, names - set(_ASK) - _ELSEWHERE


@pytest.mark.parametrize("name,engine", [
    c for c in _refused_cells() if c[0] in _ASK])
def test_an_option_the_engine_cannot_honour_is_refused_by_name(name, engine):
    kw = {name: _ASK[name]}
    dt = 1e-6 if engine == "pwl" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(p.UnsupportedError) as e:
            p.simulate(_rc(), t_end=1e-4, dt=dt, engine=engine, **kw)
    assert e.value.feature == name, str(e.value)
    assert e.value.engine == engine


def test_zero_newton_iterations_is_deprecated_as_default():
    ref = p.simulate(_rc(), t_end=1e-4, dt=1e-6)
    with pytest.warns(DeprecationWarning, match="max_newton_iterations=0"):
        res = p.simulate(_rc(), t_end=1e-4, dt=1e-6, max_newton_iterations=0)
    assert np.array_equal(np.asarray(ref.states), np.asarray(res.states))


def test_max_event_iterations_zero_is_refused_on_trbdf2():
    with pytest.raises(p.UnsupportedError, match="max_event_iterations"):
        p.simulate(_rc(), t_end=1e-4, engine="trbdf2", max_event_iterations=0)


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("S", "in", "sw", 1e3, 1e-9)
    b.add_shockley_diode("D", "gnd", "sw")
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("R", "out", "gnd", 5.0)
    return b


def _gate():
    return p.make_pwm_switch_fn(frequency=50e3, duty=0.4, switch_idx=0,
                                num_switches=1)


@pytest.mark.parametrize("engine,dt", [("pwl", 2e-7), ("trbdf2", None)])
def test_simulation_object_equals_simulate_bit_for_bit(engine, dt):
    b = _buck()
    ref = p.simulate(b, t_end=2e-4, dt=dt, engine=engine, switch_fn=_gate())
    sim = p.Simulation(b.build(), p.RunOptions(t_end=2e-4, dt=dt,
                                               engine=engine))
    sim.set_switch_fn(_gate())
    res = sim.run()
    assert res.engine_used == engine == sim.plan["engine"]
    assert np.array_equal(np.asarray(ref.times), np.asarray(res.times))
    assert np.array_equal(np.asarray(ref.states), np.asarray(res.states))
    assert np.array_equal(ref.v("out"), res.v("out"))


def test_simulation_object_refuses_before_running():
    with pytest.raises(p.UnsupportedError) as e:
        p.Simulation(_rc().build(), p.RunOptions(t_end=1e-4, engine="exact"))
    assert e.value.engine == "dsed"
    o = p.RunOptions(t_end=1e-4, engine="trbdf2")
    o.store_every = 2
    with pytest.raises(p.UnsupportedError, match="store_every"):
        p.Simulation(_rc().build(), o)


def test_simulation_object_split_run_equals_continuous():
    dt = 2.0 ** -20
    c = _buck().build()
    whole = p.Simulation(c, p.RunOptions(t_end=512 * dt, dt=dt, engine="pwl"))
    whole.set_switch_fn(_gate())
    full = np.asarray(whole.run().states)
    a = p.Simulation(c, p.RunOptions(t_end=512 * dt, dt=dt, engine="pwl"))
    a.set_switch_fn(_gate())
    first = np.asarray(a.run_until(200 * dt).states)
    b = p.Simulation(c, p.RunOptions(t_end=512 * dt, dt=dt, engine="pwl"))
    b.set_switch_fn(_gate())
    b.restore(a.snapshot())
    second = np.asarray(b.run().states)
    assert np.array_equal(np.vstack([first, second[1:]]), full)
    # The same object continues where it stopped.
    rest = np.asarray(a.run().states)
    assert np.array_equal(np.vstack([first, rest[1:]]), full)


def test_simulation_keeps_its_python_callables_alive():
    import gc
    dt = 2.0 ** -20
    sim = p.Simulation(_rc().build(), p.RunOptions(t_end=20 * dt, dt=dt))
    hits = []
    sim.set_switch_fn(lambda t: p.SwitchStateMask(0))
    sim.add_tick(4 * dt, lambda t, x: hits.append(t))
    gc.collect()
    sim.run()
    assert hits == [k * 4 * dt for k in range(5)]


def test_the_docs_page_is_the_option_table():
    import pathlib
    doc = (pathlib.Path(__file__).resolve().parents[2] / "docs"
           / "known-limitations.md").read_text()
    block = doc.split("<!-- option-capabilities:begin -->\n")[1].split(
        "<!-- option-capabilities:end -->")[0]
    assert block.strip() == ro.markdown_table()


def test_every_pulsim_error_is_a_pulsim_error_and_keeps_its_std_base():
    for cls, base in ((p.UnsupportedError, ValueError),
                      (p.OptionsError, ValueError),
                      (p.ConvergenceError, RuntimeError),
                      (p.SimulationAborted, RuntimeError),
                      (p.Cancelled, RuntimeError),
                      (p.NameNotFoundError, KeyError)):
        assert issubclass(cls, p.PulsimError) and issubclass(cls, base), cls
    with pytest.raises(p.OptionsError) as e:  # pwl needs a step
        p.Simulation(_rc().build(), p.RunOptions(t_end=1e-4, engine="pwl"))
    assert e.value.option == "dt"


def _count(pattern):
    import pathlib
    import re
    pkg = pathlib.Path(p.__file__).resolve().parent
    rx = re.compile(pattern, re.M)
    return sum(len(rx.findall(f.read_text())) for f in pkg.rglob("*.py"))


def test_ratchet_untyped_raises_and_broad_handlers_do_not_rise():
    # Counts on 2026-11-18 (add-kernel-simulation-object 1.3). Lower them
    # when a raise becomes typed or a handler is narrowed; never raise them.
    assert _count(r"raise ValueError") <= 526
    assert _count(r"except Exception|except:\s*$|except BaseException") <= 89
