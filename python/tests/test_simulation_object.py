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


# simulate() keywords that are not a row of sim::kRunOptionTable, each with
# the reason. Everything else MUST be a row (honoured or refused by name per
# engine): this test is what keeps the table honest.
_NOT_A_ROW = {
    "builder": "the circuit itself, not a run option",
}


def test_the_table_covers_every_simulate_parameter():
    params = set(inspect.signature(p.simulate).parameters)
    rows = set(ro.TABLE) - {"c_blocks"}  # c_blocks: a circuit feature
    missing = params - rows - set(_NOT_A_ROW)
    assert not missing, f"simulate() keyword(s) with no table row: {missing}"
    assert rows <= params, f"table rows that are no keyword: {rows - params}"
    assert not set(_NOT_A_ROW) & rows


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
    assert sig["voltage_sanity_factor"].default is None
    assert d["voltage_sanity_factor"] == 100.0
    # The YAML loader's `simulation:` block defaults.
    y = p.load_yaml_string("""
circuit:
  devices:
    - {type: resistor, name: R, from: a, to: gnd, R: 1.0}
simulation: {t_end: 1.0e-4, dt: 1.0e-6}
""").options
    assert (y.max_newton_iterations, y.max_event_iterations,
            y.tol_newton_dx, y.tol_newton_res, y.store_every,
            y.rtol, y.atol) == (
        d["max_newton_iterations"], d["max_event_iterations"],
        d["tol_newton_dx"], d["tol_newton_res"], d["store_every"],
        d["rtol_variable"], d["atol_variable"])


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
    "allow_implicit_segments": True, "device_path": "interface",
    "output_rtol": 1e-4,
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


def test_zero_newton_iterations_still_defers_to_the_bundle():
    # 7 iterations are too few for this buck (it needs about 20): the run
    # aborts only if the bundle's value, not the default 50, is used.
    bundle = p.SolverOptions(max_newton_iterations=7)
    with pytest.warns(DeprecationWarning), pytest.raises(p.SimulationAborted):
        p.simulate(_buck(), t_end=2e-5, dt=2e-7, switch_fn=_gate(),
                   solver=bundle, max_newton_iterations=0)
    p.simulate(_buck(), t_end=2e-5, dt=2e-7, switch_fn=_gate())  # 50 runs


def test_option_values_are_type_checked_as_before():
    for bad in ({"enable_newton_lm": "False"}, {"max_newton_iterations": 30.7},
                {"tol_newton_dx": "1e-9"}):
        with pytest.raises(TypeError):
            p.simulate(_rc(), t_end=1e-5, dt=1e-6, **bad)
    with pytest.warns(DeprecationWarning):  # False was "not set" (False == 0)
        p.simulate(_rc(), t_end=1e-5, dt=1e-6, max_newton_iterations=False)


def test_a_negative_count_is_refused_by_name():
    with pytest.raises(p.OptionsError) as e:
        p.simulate(_rc(), t_end=1e-4, dt=1e-6, max_newton_iterations=-1)
    assert e.value.option == "max_newton_iterations"


def test_max_event_iterations_zero_is_refused_on_trbdf2():
    with pytest.raises(p.UnsupportedError, match="max_event_iterations"):
        p.simulate(_rc(), t_end=1e-4, engine="trbdf2", max_event_iterations=0)
    o = p.RunOptions(t_end=1e-4)  # 'auto', no dt: trbdf2, refused up front
    o.max_event_iterations = 0
    with pytest.raises(p.UnsupportedError, match="max_event_iterations"):
        p.Simulation(_rc().build(), o)


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
    # one result type: the object's run carries simulate()'s grid / stats
    assert (res.grid, res.stats.engine, res.stats.n_steps) == (
        ref.grid, ref.stats.engine, ref.stats.n_steps)


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
    b = _rc()
    b.add_switch("S1", "in", "h", 1e3, 1e-9)
    b.add_switch("S2", "h", "gnd", 1e3, 1e-9)
    b.add_resistor("Rh", "h", "gnd", 10.0)
    sim = p.Simulation(b.build(), p.RunOptions(t_end=20 * dt, dt=dt))
    hits = []
    # A native PWM is held by raw pointer: only keep_alive holds it.
    import weakref
    pwm = p.NativePwm2Switch(8 * dt, 0.5, 2)
    alive = weakref.ref(pwm)
    sim.set_switch_fn(pwm)
    del pwm
    sim.add_tick(4 * dt, lambda t, x: hits.append(t))
    gc.collect()
    assert alive() is not None
    sim.run()
    assert hits == [k * 4 * dt for k in range(5)]
    # A replaced gate source is released (the object holds one, not all).
    sim.set_switch_fn(p.NativePwm2Switch(8 * dt, 0.25, 2))
    gc.collect()
    assert alive() is None


def test_the_docs_page_is_the_option_table():
    import pathlib
    doc = (pathlib.Path(__file__).resolve().parents[2] / "docs"
           / "known-limitations.md").read_text(encoding="utf-8")
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
    return sum(len(rx.findall(f.read_text(encoding="utf-8")))
               for f in pkg.rglob("*.py"))


def test_ratchet_untyped_raises_and_broad_handlers_do_not_rise():
    # Counts on 2026-11-18 (add-kernel-simulation-object 1.3). Lower them
    # when a raise becomes typed or a handler is narrowed; never raise them.
    # Re-based at integration batch 14c on the merged tree: +36 ValueError
    # from tracks written before the ratchet (machines 7, passives 4,
    # topology 3; _recorders 22), handlers +4 (_control_state.pickle_payload:
    # any pickling failure becomes the refusal marker; _recorders 3: an
    # unknown device name, and the run's error winning over its warnings).
    assert _count(r"raise ValueError") <= 562
    assert _count(r"except Exception|except:\s*$|except BaseException") <= 90


# simulate()'s body since add-kernel-simulation-object Stage A: the planner,
# the three engine runners and finalize.
_SIMULATE_BODY = ("simulate", "_plan_run", "_apply_ties", "_compose", "_route",
                  "_compose_for", "_run_pwl", "_run_trbdf2", "_run_dsed",
                  "_finalize")


def _simulate_body_nodes():
    import ast
    import pathlib
    src = (pathlib.Path(p.__file__).resolve().parent / "__init__.py").read_text(
        encoding="utf-8")
    return [x for fn in ast.parse(src).body
            if isinstance(fn, ast.FunctionDef) and fn.name in _SIMULATE_BODY
            for x in ast.walk(fn)]


def test_simulate_snapshots_every_refusable_option():
    import ast
    nodes = _simulate_body_nodes()
    snap = next(n.value for n in nodes if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "_opt_values"
                and isinstance(n.value, ast.Dict))
    dsed = next(c.args[1] for c in nodes if isinstance(c, ast.Call)
                and getattr(c.func, "attr", "") == "refused"
                and isinstance(c.args[0], ast.Constant)
                and c.args[0].value == "dsed")
    for lit in (snap, dsed):
        pairs = [(k, v) for k, v in zip(lit.keys, lit.values) if k is not None]
        assert all(isinstance(v, ast.Name) and v.id == k.value
                   for k, v in pairs if k.value != "c_blocks")
    refusable = {n for n, r in ro.TABLE.items()
                 if n != "c_blocks" and r["unset"] != "always"}
    # The snapshot: every row an MNA engine may refuse; the exact engine's
    # branch adds the rows only it refuses (and c_blocks).
    assert {k.value for k in snap.keys} == refusable - _dsed_only()
    assert {k.value for k in dsed.keys if k is not None} == (
        _dsed_only() | {"c_blocks"})


def _dsed_only():
    return {n for n, r in ro.TABLE.items() if n != "c_blocks"
            and r["unset"] != "always" and {"pwl", "trbdf2"} <= set(r["engines"])
            and "dsed" not in r["engines"]}


def test_simulation_object_reads_device_path_monitors_and_the_tick_grid():
    """Integration batch 14a on the object: device_path reaches both
    drivers; a Monitor neither lands a step nor re-samples the gate (run
    identical without it); a resume keeps the snapshot's tick grid."""
    b = _buck()
    for engine, dt in (("pwl", 2e-7), ("trbdf2", None)):
        ref = p.simulate(b, t_end=1e-4, dt=dt, engine=engine,
                         switch_fn=_gate(), device_path="interface")
        o = p.RunOptions(t_end=1e-4, dt=dt, engine=engine)
        o.device_path = "interface"
        assert o.device_path == "interface"  # reaches the drivers: test_simulation.cpp
        sim = p.Simulation(b.build(), o)
        sim.set_switch_fn(_gate())
        assert np.array_equal(np.asarray(ref.states),
                              np.asarray(sim.run().states)), engine
    plain = p.Simulation(b.build(), p.RunOptions(t_end=1e-4, engine="trbdf2"))
    plain.set_switch_fn(_gate())
    seen = []
    mon = p.Simulation(b.build(), p.RunOptions(t_end=1e-4, engine="trbdf2"))
    mon.set_switch_fn(_gate())
    mon.add_tick(3e-6, lambda t, x: seen.append(t), 0.0, True)
    assert np.array_equal(np.asarray(plain.run().states),
                          np.asarray(mon.run().states))
    assert len(seen) >= 24  # 34 instants; a bound, not a count
    # Split at a tick instant: the continuation fires each tick once.
    dt = 2.0 ** -20
    T = 16 * dt
    hits = []
    s = p.Simulation(b.build(), p.RunOptions(t_end=128 * dt, dt=dt, engine="pwl"))
    s.set_switch_fn(_gate())
    s.add_tick(T, lambda t, x: hits.append(t))
    s.run_until(64 * dt)
    assert list(s.snapshot().tick_origin) == [0.0]
    s.run()
    assert hits == [k * T for k in range(8)]


def test_the_trbdf2_route_rows_are_the_blockers_own():
    # simulate() hands _trbdf2_blockers its _opt_values: the pseudo-engine
    # must judge only the 13 routing rows (h_bdf2 is refused on a dt-less
    # 'auto' run, not routed; integrator is a trbdf2 row since 'esdirk43').
    vals = {"progress": True, "mode_selection": "lcp", "rtol": 1e-6,
            "store_every": 2, "integrator": "rk45", "h_bdf2": 1e-7}
    assert set(ro.refused(ro.TRBDF2_ROUTE, vals)) == {"store_every", "progress"}
    assert set(ro.refused("trbdf2", vals)) > set(ro.refused(ro.TRBDF2_ROUTE, vals))


def _hook_rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "out", 1.0e3)
    b.add_capacitor("C", "out", "gnd", 1.0e-6)
    return b


def test_on_cache_hook_writes_reach_the_run_and_rebuilds_are_refused():
    # add-kernel-simulation-object 17: the hook gets the object's cache
    # (a checked view): its refactor reaches the solve; build/build_lazy
    # would re-key the factors on another dt and are refused by name.
    b = _hook_rc()
    rid = b.branch_id_of("R")
    def hook(cache):
        cache.refactor_parametric(rid, 1.0e2)  # tau 1 ms -> 0.1 ms

    def v_end(r):
        return float(np.asarray(r.v("out"))[-1])

    base = p.simulate(b, t_end=2e-4, dt=1e-6)
    assert v_end(p.simulate(b, t_end=2e-4, dt=1e-6, on_cache=hook)) \
        > v_end(base) + 0.1
    for name in ("build", "build_lazy"):
        with pytest.raises(ValueError, match=f"cache.{name}"):
            p.simulate(b, t_end=2e-5, dt=1e-6,
                       on_cache=lambda c, n=name: getattr(c, n)(1e-7))


def test_hook_numbering_check_refuses_a_renumbered_builder():
    from pulsim import _check_run_numbering
    a, other = _hook_rc(), p.CircuitBuilder()
    other.add_voltage_source("V", "in", "gnd", 1.0)
    other.add_capacitor("C", "out", "gnd", 1.0e-6)
    other.add_resistor("R", "in", "out", 1.0e3)
    _check_run_numbering(a, _hook_rc(), "mmc_arms")  # same numbering: ok
    with pytest.raises(ValueError, match="mmc_arms"):
        _check_run_numbering(a, other, "mmc_arms")


def test_a_hook_on_the_chain_fast_path_keeps_one_cache():
    # DECISIONS 17's trap: a hook that returns None leaves the C++ chain
    # observer in place; the chain then runs on the object's builder and
    # the cache the hook was handed (not a second one), bit for bit.
    def run(**kw):
        b = _hook_rc()
        ch = p.MixedDomainBlockChain(rate=p.Rate(1e-5))
        ch.add("g", p.Gain(k=0.5), inputs=dict(x="out"), output="y")
        obs = ch.make_step_observer(b)
        assert getattr(obs, "_cxx_chain", None) is not None
        return p.simulate(b, 1e-4, engine="pwl", dt=1e-6,
                          step_observer=obs, **kw)
    seen = []
    plain, hooked = run(), run(on_cache=seen.append)
    assert np.array_equal(np.asarray(plain.states), np.asarray(hooked.states))
    assert seen and seen[0].num_built_segments() >= 1  # the run used it


def test_simulate_swallows_no_kernel_error():
    """1.3: a broad handler in simulate() re-raises, except the listed
    post-run diagnostic (a finished run is not broken by its own check)."""
    import ast
    allowed = {"_volt = None"}  # find_implausible_voltage, after the run
    swallowing = []
    for h in _simulate_body_nodes():
        if not isinstance(h, ast.ExceptHandler):
            continue
        kind = ast.unparse(h.type) if h.type is not None else ""
        if kind and "Exception" not in kind:
            continue
        if not any(isinstance(x, ast.Raise) for x in ast.walk(h)) \
                and ast.unparse(h.body[0]) not in allowed:
            swallowing.append((h.lineno, kind or "bare"))
    assert swallowing == []
