"""The per-engine blocks of add-kernel-simulation-object DECISIONS 23, made
uniform for the number-changing release (DECISIONS 26). Each rule is pinned
twice: today's per-engine behaviour (the default, `_plan.UNIFORM_ENGINE_RULES`
False) and the release behaviour (True: the value `next_defaults.py` sets)."""

import inspect
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim import _plan

ENGINES = {"pwl": dict(engine="pwl", dt=1e-6), "trbdf2": dict(engine="trbdf2"),
           "exact": dict(engine="exact"), "auto": dict(engine="auto")}


@pytest.fixture
def release(monkeypatch):
    monkeypatch.setattr(_plan, "UNIFORM_ENGINE_RULES", True)


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "out", 1e3)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    return b


def _plan_of(*args, **kw):
    bound = inspect.signature(p.simulate).bind(*args, **kw)
    bound.apply_defaults()
    return p._plan_run(**bound.arguments)


# ---- count checks (DECISIONS 23 item 7) -----------------------------------

TRBDF2_TODAY = "(None = the default)"       # the kernel's own wording
PWL = "(0 disables event iteration)"


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "auto"])
def test_max_event_iterations_today_each_engine_words_its_own(engine):
    with pytest.raises(p.OptionsError) as ei:
        p.simulate(_rc(), 1e-5, **ENGINES[engine], max_event_iterations=-1)
    assert ei.value.option == "max_event_iterations"
    assert (PWL if engine == "pwl" else TRBDF2_TODAY) in str(ei.value)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "auto"])
def test_max_event_iterations_at_the_release_one_message(release, engine):
    with pytest.raises(p.OptionsError, match=r"\(0 disables event iteration\)") as ei:
        p.simulate(_rc(), 1e-5, **ENGINES[engine], max_event_iterations=-1)
    assert ei.value.option == "max_event_iterations"


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("option,value", [("store_every", 0), ("max_dt_halvings", -1),
                                          ("max_event_iterations", -1)])
def test_an_engine_that_does_not_read_a_count_refuses_it_by_name(monkeypatch, uniform,
                                                                 option, value):
    monkeypatch.setattr(_plan, "UNIFORM_ENGINE_RULES", uniform)
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(_rc(), 1e-5, engine="exact", **{option: value})
    assert (ei.value.engine, ei.value.feature) == ("dsed", option)
    if option != "max_event_iterations":   # trbdf2 does not read these two either
        with pytest.raises(p.UnsupportedError) as ei:
            p.simulate(_rc(), 1e-5, engine="trbdf2", **{option: value})
        assert (ei.value.engine, ei.value.feature) == ("trbdf2", option)


def _mmc():
    b = p.CircuitBuilder()
    arm = p.add_mmc_thevenin_arm(b, "arm", "a", "gnd", n_sm=2, c_sm=1e-3, r_on=1e-3,
                                 v_c_init=10.0, n_on=lambda t: 1)
    b.add_resistor("Rload", "a", "gnd", 1.0)
    return b, arm


def test_a_bad_count_with_mmc_arms_today_reads_as_the_mmc_combination():
    b, arm = _mmc()
    with pytest.raises(ValueError, match="decimation can drop"):
        p.simulate(b, t_end=1e-4, dt=1e-6, mmc_arms=[arm], store_every=0)


def test_a_bad_count_with_mmc_arms_at_the_release_is_the_count_check(release):
    b, arm = _mmc()
    with pytest.raises(p.OptionsError, match="must be >= 1") as ei:
        p.simulate(b, t_end=1e-4, dt=1e-6, mmc_arms=[arm], store_every=0)
    assert ei.value.option == "store_every"


# ---- on_impulse (DECISIONS 23 item 6) -------------------------------------

def _cap_loop():
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", 1.5e-6, 8.0)
    b.add_capacitor("C2", "b", "gnd", 1.5e-6, 0.0)
    b.add_voltage_source("V0", "a", "b", 0.0)
    b.add_resistor("R", "a", "gnd", 1e3)
    return b


def _ideal_cut():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_switch("S", "vin", "a", form="branch", R_on=1e-6, g_off=0.0)
    b.add_resistor("R", "a", "b", 10.0)
    b.add_inductor("L", "b", "gnd", 1e-3)
    return b, p.make_pwm_switch_fn(1e3, 0.5, b.switch_index_of("S"), b.graph.num_switches)


@pytest.mark.parametrize("on_impulse", ["warn", "error"])
def test_exact_bdf2_today_cannot_detect_an_impulse_and_runs(on_impulse):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = p.simulate(_cap_loop(), 1e-4, engine="exact", dt=1e-6, integrator="bdf2",
                       on_impulse=on_impulse)
    assert len(r.reinit_records) == 0
    assert not any("impulsive" in str(x.message) for x in w)


@pytest.mark.parametrize("on_impulse", ["warn", "error"])
def test_exact_bdf2_at_the_release_refuses_on_impulse_by_name(release, on_impulse):
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(_cap_loop(), 1e-4, engine="exact", dt=1e-6, integrator="bdf2",
                   on_impulse=on_impulse)
    assert (ei.value.engine, ei.value.feature) == ("dsed", "on_impulse")
    r = p.simulate(_cap_loop(), 1e-4, engine="exact", dt=1e-6, integrator="bdf2")
    assert len(r.reinit_records) == 0          # 'record' (the default) still runs


@pytest.mark.parametrize("uniform", [False, True])
def test_on_impulse_error_refuses_on_every_engine_that_records(monkeypatch, uniform):
    monkeypatch.setattr(_plan, "UNIFORM_ENGINE_RULES", uniform)
    kinds = {}
    for engine, kw in (("trbdf2", {}), ("exact", dict(dt=1e-6))):
        b, fn = _ideal_cut()
        with pytest.raises(p.UnsupportedError) as ei:
            p.simulate(b, 2e-3, engine=engine, switch_fn=fn, on_impulse="error", **kw)
        e = ei.value
        assert (e.feature, e.device) == ("impulse", "S")
        assert "an inductor cut-set" in str(e) and "0.000993262 V*s" in str(e)
        kinds[engine] = e.kind
    # the kernel names the kind; the exact engine's refusal does at the release
    assert kinds == {"trbdf2": "Switch", "exact": "Switch" if uniform else ""}


# ---- mode_selection='auto' (DECISIONS 23 item 5): resolved in the plan ----

def _rectifier():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_diode("D", "in", "out", 1e3, 1e-9)
    b.add_resistor("R", "out", "gnd", 1.0)
    return b


@pytest.mark.parametrize("engine,mode,reason", [
    ("pwl", "lcp", None), ("trbdf2", None, "engine 'trbdf2' (no 'auto' rule there yet)"),
    ("exact", "lcp", None)])
def test_mode_auto_is_resolved_in_the_plan_on_every_engine(engine, mode, reason):
    plan = _plan_of(_rectifier(), 1e-5, **ENGINES[engine], mode_selection="auto")
    assert plan.events.mode_selection == mode
    assert plan.events.mode_auto["reason"] == reason
    r = p.simulate(_rectifier(), 1e-5, **ENGINES[engine], mode_selection="auto")
    assert r.diagnostics["mode_selection_auto"] == plan.events.mode_auto


def test_the_exact_dispatcher_resolves_an_auto_that_reaches_it_unresolved():
    from pulsim._dsed_dispatch import run_dsed_from_builder
    r = run_dsed_from_builder(builder=_rectifier(), t_end=1e-5, mode_selection="auto")
    assert r.diagnostics["mode_selection_auto"]["chosen"] == "lcp"   # the plan's rule


def test_mode_auto_on_the_exact_engine_without_diodes_is_the_settle():
    plan = _plan_of(_rc(), 1e-5, engine="exact", mode_selection="auto")
    assert plan.events.mode_selection is None
    assert plan.events.mode_auto["reason"] == "no_diodes"


# ---- resume_from / t_start (DECISIONS 23 item 4): one clock rule ----------

@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_a_resume_continues_the_snapshots_clock_and_the_plan_says_so(engine):
    first = p.simulate(_rc(), 1e-5, **ENGINES[engine])
    snap = first.final_snapshot
    plan = _plan_of(_rc(), 2e-5, **ENGINES[engine], resume_from=snap)
    assert plan.t_start == pytest.approx(float(snap.t))
    assert plan.resolved_from["t_start"] == "resume_from"
    plan = _plan_of(_rc(), 2e-5, **ENGINES[engine], resume_from=snap, t_start=5e-6)
    assert plan.t_start == 5e-6 and plan.resolved_from["t_start"] == "explicit"


def test_the_exact_engine_refuses_a_resume_by_name():
    snap = p.simulate(_rc(), 1e-5, engine="pwl", dt=1e-6).final_snapshot
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(_rc(), 2e-5, engine="exact", resume_from=snap)
    assert (ei.value.engine, ei.value.feature) == ("dsed", "resume_from")


# ---- the voltage check (task 6.2): one function, the caller's line --------

@pytest.mark.parametrize("engine", ENGINES)
def test_one_voltage_check_names_the_callers_line_on_every_engine(engine):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = p.simulate(_rc(), 1e-5, **ENGINES[engine], voltage_sanity_factor=1e-9)
    (hit,) = [x for x in w if "node in reached" in str(x.message)]
    assert hit.filename == __file__
    assert r._implausible_voltage.node >= 0
    assert np.isfinite(np.asarray(r.v("out"))).all()
