"""add-kernel-simulation-object Stage A (tasks 1.1-1.3): simulate() plans
the run once, above the engine dispatch; the engine runners read only the
plan; no broad handler on simulate()'s call path swallows a kernel error."""

import ast
import builtins
import inspect
import pathlib

import pytest

import pulsim as p

_PKG = pathlib.Path(p.__file__).resolve().parent
_RUNNERS = ("_run_pwl", "_run_trbdf2", "_run_dsed")
# simulate()'s own body since Stage A: the planner, the runners, finalize.
_SIMULATE_BODY = ("simulate", "_plan_run", *_RUNNERS, "_finalize")
# Ambient inputs a runner must take from the plan, never read itself.
_AMBIENT = {"_KERNEL_RUN", "_PWL_RECIPE"}


def _functions(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name: n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names}


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_resistor("R", "in", "n1", 1e3)
    b.add_capacitor("C", "n1", "gnd", 1e-6)
    return b


def _plan(*args, **kw):
    bound = inspect.signature(p.simulate).bind(*args, **kw)
    bound.apply_defaults()
    return p._plan_run(**bound.arguments)


@pytest.mark.xfail(strict=True, reason="Stage A group 2 plans the keywords")
def test_plan_resolves_every_kwarg():
    """1.1: a keyword cannot be added to simulate() without being planned."""
    plan = _plan(_rc(), 1e-4, 1e-6)
    params = set(inspect.signature(p.simulate).parameters) - {"builder"}
    assert set(plan.resolved_from) == params
    assert set(plan.resolved_from.values()) == {"explicit", "default"}
    assert plan.resolved_from["dt"] == "explicit"
    assert plan.resolved_from["rtol"] == "default"


@pytest.mark.xfail(strict=True, reason="Stage A group 2 plans the keywords")
def test_resolved_from_names_the_solver_bundle_and_the_yaml():
    plan = _plan(_rc(), 1e-4, 1e-6,
                 solver=p.SolverOptions(store_every=2, rtol=None))
    assert plan.resolved_from["store_every"] == "solver"
    assert plan.resolved_from["rtol"] == "default"
    assert plan.resolved_from["solver"] == "explicit"
    lc = p.load_yaml_string(
        "circuit:\n"
        "  devices:\n"
        "    - {type: voltage_source, name: V1, from: in, to: gnd, V: 1.0}\n"
        "    - {type: resistor, name: R1, from: in, to: gnd, R: 1.0e3}\n"
        "simulation: {t_end: 1.0e-4, dt: 1.0e-6}\n")
    plan = _plan(lc)
    assert plan.resolved_from["t_end"] == "yaml"
    assert plan.resolved_from["dt"] == "yaml"


@pytest.mark.xfail(strict=True, reason="Stage A group 7 builds the runners")
def test_engine_branches_read_only_the_plan():
    """1.2: each engine runner takes the plan and nothing else: its only
    parameter is `plan` and every free name it reads is a module-level
    definition or a builtin (no ambient context variable)."""
    src = _PKG / "__init__.py"
    fns = _functions(src, _RUNNERS)
    assert set(fns) == set(_RUNNERS)
    module_names = set(vars(p)) | set(dir(builtins))
    for name, fn in fns.items():
        a = fn.args
        assert [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs] == ["plan"], name
        assert a.vararg is None and a.kwarg is None, name
        bound = {"plan"}
        for node in ast.walk(fn):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                pytest.fail(f"{name} declares {type(node).__name__}")
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                bound.add(node.id)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.Lambda)) and node is not fn:
                if isinstance(node, ast.FunctionDef):
                    bound.add(node.name)
            elif isinstance(node, ast.alias):
                bound.add((node.asname or node.name).split(".")[0])
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
        free = {n.id for n in ast.walk(fn)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - bound
        assert not free & _AMBIENT, (name, free & _AMBIENT)
        assert free <= module_names, (name, sorted(free - module_names))


@pytest.mark.xfail(strict=True, reason="Stage A group 7 builds the table")
def test_simulate_is_plan_table_finalize():
    """7.1: simulate() does not branch on the engine; the table does."""
    fn = _functions(_PKG / "__init__.py", {"simulate"})["simulate"]
    compares = [ast.unparse(c) for c in ast.walk(fn) if isinstance(c, ast.Compare)
                and "engine" in ast.unparse(c)]
    assert compares == []
    assert set(p._ENGINES) == {"pwl", "trbdf2", "dsed"}


def _calls(fn):
    out = set()
    for c in ast.walk(fn):
        if isinstance(c, ast.Call):
            f = c.func
            if isinstance(f, ast.Name):
                out.add(("__init__.py", f.id))
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                    and f.value.id in ("_dsed", "_dsed_dispatch_module"):
                out.add(("_dsed_dispatch.py", f.attr))
    return out


# 1.3, one level down: helpers simulate() calls whose broad handler does not
# re-raise. Each is a guard around a NON-kernel call or a post-run report
# (never the run itself). Shrink this; never grow it.
_SWALLOWING_HELPERS = {
    ("_dsed_dispatch.py", "run_dsed_from_builder"),  # _switch_census, test doubles
    ("__init__.py", "_trbdf2_blockers"),             # _switch_census, test doubles
    ("__init__.py", "_warn_resolution"),             # state_var_names, test doubles
    ("__init__.py", "_auto_attach_motor_traces"),    # a bundle's attach_to_result
    ("__init__.py", "_auto_check_voltage_sanity"),   # the post-run check
}


def test_helpers_one_level_down_swallow_no_kernel_error():
    body = _functions(_PKG / "__init__.py", _SIMULATE_BODY)
    callees = set().union(*(_calls(fn) for fn in body.values()))
    found = set()
    for fname in ("__init__.py", "_dsed_dispatch.py"):
        names = {n for f, n in callees if f == fname} - set(_SIMULATE_BODY)
        for name, fn in _functions(_PKG / fname, names).items():
            for h in ast.walk(fn):
                if not isinstance(h, ast.ExceptHandler):
                    continue
                kind = ast.unparse(h.type) if h.type is not None else ""
                if kind and "Exception" not in kind:
                    continue
                if not any(isinstance(x, ast.Raise) for x in ast.walk(h)):
                    found.add((fname, name))
    assert found <= _SWALLOWING_HELPERS, sorted(found - _SWALLOWING_HELPERS)
