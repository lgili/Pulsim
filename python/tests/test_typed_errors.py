"""add-kernel-simulation-object group 8: typed errors and run diagnostics.

Every error pulsim raises on purpose is a ``PulsimError`` that keeps the
standard base its raise site used before (so no ``except ValueError`` /
``except RuntimeError`` / ``except KeyError`` stops catching it), and the
warnings a run emits are kept on ``result.diagnostics["messages"]`` with
the category and text the user saw."""

import ast
import pathlib
import warnings

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
