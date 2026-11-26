"""add-kernel-simulation-object group 8: typed errors and run diagnostics.

Every error pulsim raises on purpose is a ``PulsimError`` that keeps the
standard base its raise site used before (so no ``except ValueError`` /
``except RuntimeError`` / ``except KeyError`` stops catching it), and the
warnings a run emits are kept on ``result.diagnostics["messages"]`` with
the category and text the user saw."""

import ast
import pathlib
import warnings
from unittest import mock

import pytest

import pulsim as p
from pulsim import errors

_PKG = pathlib.Path(p.__file__).resolve().parent


def _buck(diode="pwl"):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 12.0)
    b.add_switch("S", "in", "sw", 1e3, 1e-9)
    if diode == "shockley":
        b.add_shockley_diode("D", "gnd", "sw")
    else:
        b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("R", "out", "gnd", 5.0)
    return b


_SW = p.make_pwm_switch_fn(frequency=50e3, duty=0.4, switch_idx=0, num_switches=2)


def test_the_hierarchy_keeps_every_standard_base():
    for cls, bases in (
            (p.NetlistError, (ValueError,)),
            (p.OptionsError, (ValueError,)),
            (p.UnsupportedError, (ValueError,)),
            (p.ConvergenceError, (RuntimeError,)),
            (p.SimulationAborted, (p.ConvergenceError, RuntimeError)),
            (p.SingularError, (p.SimulationAborted, RuntimeError)),
            (p.NewtonRoundoffBudgetExceeded, (p.SimulationAborted,)),
            (p.Cancelled, (RuntimeError,)),
            (p.NameNotFoundError, (KeyError,)),
            (p.InternalError, (RuntimeError,))):
        assert issubclass(cls, p.PulsimError), cls
        for base in bases:
            assert issubclass(cls, base), (cls, base)
        assert getattr(errors, cls.__name__) is cls
    # a name lookup is not a ValueError: an `except ValueError` fallback
    # around a lookup must not start swallowing it (DECISIONS 28)
    assert not issubclass(p.NameNotFoundError, ValueError)
    assert p.NameNotFoundError.__module__ == "pulsim"


_TYPED_BODY = {
    "__init__.py": {"simulate", "_plan_run", "_apply_ties", "_compose", "_route",
                    "_compose_for", "_run_pwl", "_run_trbdf2", "_run_dsed",
                    "_finalize", "_resolve_loaded_circuit"},
    "_dsed_dispatch.py": {"_validate_engine_kwargs", "run_dsed_from_builder"},
    "_control_plan.py": {"_check_fixed_periods"},
}
_UNTYPED = {"ValueError", "RuntimeError", "NotImplementedError", "KeyError",
            "Exception"}
# a broken install, not a refusal of the run
_ALLOWED = {("_dsed_dispatch.py", "run_dsed_from_builder", "RuntimeError")}


def test_simulate_path_raises_are_typed():
    """8.2: simulate()'s own planning, validation and dispatch raise only
    pulsim's classes (TypeError for a missing argument stays Python's)."""
    found = []
    for fname, names in _TYPED_BODY.items():
        tree = ast.parse((_PKG / fname).read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not (isinstance(fn, ast.FunctionDef) and fn.name in names):
                continue
            for r in ast.walk(fn):
                if not isinstance(r, ast.Raise) or r.exc is None:
                    continue
                e = r.exc.func if isinstance(r.exc, ast.Call) else r.exc
                name = ast.unparse(e)
                if name in _UNTYPED and (fname, fn.name, name) not in _ALLOWED:
                    found.append((fname, fn.name, r.lineno, name))
    assert not found, found


def test_init_warns_only_through_the_run_diagnostics():
    """8.3: every warning of __init__.py goes through _plan.emit, so a run
    keeps what it told the user."""
    src = (_PKG / "__init__.py").read_text(encoding="utf-8")
    calls = [ast.unparse(c.func) for c in ast.walk(ast.parse(src))
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
             and c.func.attr == "warn"]
    assert calls == [], calls


def test_options_errors_name_the_option():
    b = _buck()
    with pytest.raises(p.OptionsError) as e:
        p.simulate(b, 1e-5, dt=-1e-7, engine="pwl", switch_fn=_SW)
    assert e.value.option == "dt" and isinstance(e.value, ValueError)
    with pytest.raises(ValueError) as e:     # the old catch still works
        p.simulate(b, 1e-5, dt=1e-7, switch_fn=_SW, control_tick="bogus")
    assert isinstance(e.value, p.OptionsError) and e.value.option == "control_tick"
    with pytest.raises(p.OptionsError) as e:
        p.simulate(b, 0.0, dt=1e-7, switch_fn=_SW)
    assert e.value.option == "t_end"
    with pytest.raises(p.OptionsError) as e:
        p.simulate(b, 1e-5, dt=1e-7, switch_fn=_SW, on_impulse="loud")
    assert e.value.option == "on_impulse"


def test_unsupported_error_names_engine_feature_and_device():
    """8.4: one refusal per field: a feature (the reserved integrator, still
    a NotImplementedError as before), a device (the exact engine and a
    Shockley diode, through simulate() and through the kernel object)."""
    with pytest.raises(NotImplementedError) as e:
        p.simulate(_buck(), 1e-5, dt=1e-7, switch_fn=_SW, engine="pwl",
                   integrator="dopri5")
    assert isinstance(e.value, p.UnsupportedError)
    assert (e.value.engine, e.value.feature) == ("pwl", "integrator")
    with pytest.raises(p.UnsupportedError) as e:
        p.simulate(_buck("shockley"), 1e-5, engine="exact", switch_fn=_SW)
    assert (e.value.engine, e.value.device, e.value.kind) == (
        "dsed", "D", "ShockleyDiode")
    with pytest.raises(p.UnsupportedError) as e:
        p.Simulation(_buck("shockley").build(),
                     p.RunOptions(t_end=1e-5, engine="exact"))
    assert (e.value.engine, e.value.device, e.value.kind) == (
        "dsed", "D", "ShockleyDiode")


def test_errors_module_builders():
    e = errors.options_error("rtol", "bad rtol")
    assert isinstance(e, p.OptionsError) and e.option == "rtol"
    n = errors.netlist_error("bad pin", device="D1", node="x")
    assert isinstance(n, ValueError) and (n.device, n.node) == ("D1", "x")
    assert (p.NetlistError.device, p.NetlistError.node) == ("", "")


def _magnetic_without_reference():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "a", 1.0)
    b.add_winding("W", "a", "gnd", "m1", "m2", 10)
    b.add_permeance("P", "m1", "m2", 1e-6)
    return b


def test_netlist_errors_come_from_the_kernel_typed():
    """NetlistError through the translator: the preflight's magnetic
    reference check (simulate()) and a tie on a missing node."""
    with pytest.raises(ValueError) as e:      # the old catch still works
        p.simulate(_magnetic_without_reference(), 1e-5, dt=1e-7)
    assert isinstance(e.value, p.NetlistError)
    assert (e.value.device, e.value.node) == ("", "m1")
    c = _buck().build()
    with pytest.raises(p.NetlistError) as e:
        c.with_ties([("Tbad", 999, 1.0)])
    assert e.value.device == "Tbad" and "999" in str(e.value)


@pytest.mark.parametrize("halvings", [0, None])
def test_a_singular_step_is_a_singular_error(halvings):
    """Two voltage sources in parallel: the step matrix is singular. The
    run aborts with SingularError (a SimulationAborted carrying the partial
    run), with or without the dt-halving retries in between."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "a", "gnd", 1.0)
    b.add_voltage_source("V2", "a", "gnd", 2.0)
    b.add_resistor("R", "a", "gnd", 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(p.SimulationAborted) as e:
            p.simulate(b, 1e-5, dt=1e-7, engine="pwl", max_dt_halvings=halvings)
    assert isinstance(e.value, p.SingularError), type(e.value)
    assert e.value.partial is not None
    assert "singular" in str(e.value)


def _line_of(fn, needle):
    import inspect
    src, first = inspect.getsourcelines(fn)
    return first + next(i for i, ln in enumerate(src) if needle in ln)


def _undriven_switch():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_switch("S", "in", "o", 1e3, 1e-9)
    b.add_resistor("R", "o", "gnd", 1.0)
    return b


def test_routed_warnings_name_the_callers_line():
    """Three sites at three stack depths (_plan_run, a helper of it, the
    undriven policy's own stacklevel argument): each warning is attributed
    to the line of the simulate() call, and kept on the result."""
    def run():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r1 = p.simulate(_buck(), 1e-5, dt=1e-7, switch_fn=_SW, enable_nonlinear_refresh=True)  # A
            r2 = p.simulate(_buck(), 1e-5, dt=1e-7, switch_fn=_SW, max_newton_iterations=0)  # B
            # release: undriven defaults to "refuse" (a refusal, not a warning); the
            # warning site is the per-engine policy of the legacy period
            with mock.patch.object(p._plan, "DEFAULT_UNDRIVEN", None):
                r3 = p.simulate(_undriven_switch(), 1e-6, dt=1e-7, engine="pwl")  # C
        return w, (r1, r2, r3)
    w, rs = run()
    for tag, r in zip("ABC", rs):
        line = _line_of(run, f"# {tag}")
        mine = [x for x in w if x.lineno == line]
        assert mine and all(x.filename == __file__ for x in mine), tag
        kept = r.diagnostics["messages"]
        assert [(d.category, d.message) for d in kept] == [
            (x.category, str(x.message)) for x in mine], tag


def test_a_run_keeps_the_warnings_it_emitted():
    """8.3: result.diagnostics['messages'] holds each warning the run
    emitted (category, text), and the warning still names the caller's
    line (the stacklevel is unchanged by the routing)."""
    b = _buck()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = p.simulate(b, 1e-5, dt=1e-7, switch_fn=_SW,
                         enable_nonlinear_refresh=True)
    mine = [x for x in w if issubclass(x.category, DeprecationWarning)
            and "enable_nonlinear_refresh" in str(x.message)]
    assert len(mine) == 1
    assert mine[0].filename == __file__
    msgs = res.diagnostics["messages"]
    hit = [d for d in msgs if d.code == "deprecated_option"]
    assert len(hit) == 1
    assert hit[0].category is DeprecationWarning
    assert hit[0].message == str(mine[0].message)
    # a quiet run keeps an empty record, on every engine
    for kw in ({"dt": 1e-7, "engine": "pwl"}, {"engine": "trbdf2"},
               {"engine": "exact"}):
        r = p.simulate(_buck(), 1e-5, switch_fn=_SW, **kw)
        assert r.diagnostics["messages"] == (), (kw, r.diagnostics["messages"])
    # outside a run, nothing is recorded and nothing leaks into the next run
    from pulsim import _plan
    assert _plan.RUN_MESSAGES.get() is None


def test_a_warning_is_recorded_once_after_an_auto_fallback():
    """'auto' took the exact engine, which refused the run by name: the call is
    planned and run again on trbdf2, and the planning warning both attempts
    emit is recorded once (it was recorded twice)."""
    b = p.CircuitBuilder()   # a PWL buck: 'auto' without dt routes it to the exact engine
    b.add_voltage_source("V", "in", "gnd", 12.0)
    b.add_switch("S", "in", "sw", 1e3, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("R", "out", "gnd", 5.0)
    sw = p.make_pwm_switch_fn(frequency=50e3, duty=0.4, switch_idx=0, num_switches=2)
    real, seen = p._ENGINES["dsed"], []

    def refuse(plan):
        seen.append(plan.engine)
        raise p._unsupported("dsed", "probe", "simulate(): refused for the test")

    with mock.patch.dict(p._ENGINES, {"dsed": refuse}), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 4e-5, switch_fn=sw, max_newton_iterations=0)   # a planning warning
    assert p._ENGINES["dsed"] is real and seen == ["dsed"]
    assert r.engine_used == "trbdf2"
    assert [d.code for d in r.diagnostics["messages"]].count("deprecated_option") == 1
