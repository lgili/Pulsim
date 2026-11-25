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
_SIMULATE_BODY = ("simulate", "_plan_run", "_apply_ties", "_compose", "_route",
                  "_compose_for", *_RUNNERS, "_finalize")
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


def test_plan_resolves_every_kwarg():
    """1.1: a keyword cannot be added to simulate() without being planned."""
    plan = _plan(_rc(), 1e-4, 1e-6)
    params = set(inspect.signature(p.simulate).parameters) - {"builder"}
    assert set(plan.resolved_from) == params
    assert set(plan.resolved_from.values()) == {"explicit", "default"}
    assert plan.resolved_from["dt"] == "explicit"
    assert plan.resolved_from["rtol"] == "default"


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
    ("__init__.py", "_undriven"),                    # _switch_census, test doubles
    ("__init__.py", "_trbdf2_blockers"),             # _switch_census, test doubles
    ("__init__.py", "_warn_resolution"),             # state_var_names, test doubles
    ("__init__.py", "_auto_attach_motor_traces"),    # a bundle's attach_to_result
    ("__init__.py", "_auto_check_voltage_sanity"),   # the post-run check
    ("__init__.py", "_check_right_limit_voltage"),   # the same check, right limits
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


def _star_names(call):
    """The names of a call `f(*((a, b, ...) + (c, ...)))`, in order."""
    assert not call.args[:-1] or all(isinstance(a, ast.Name) for a in call.args[:-1])
    names = [a.id for a in call.args if isinstance(a, ast.Name)]
    star = [a for a in call.args if isinstance(a, ast.Starred)]
    assert len(star) == 1 and not call.keywords, ast.unparse(call)
    expr = star[0].value
    parts = []
    while isinstance(expr, ast.BinOp):
        parts.insert(0, expr.right)
        expr = expr.left
    parts.insert(0, expr)
    for t in parts:
        assert isinstance(t, ast.Tuple) and len(t.elts) <= 30, ast.unparse(t)
        names += [e.id for e in t.elts]
    return names


def test_positional_plan_calls_follow_the_signatures():
    """7.2 (fixed overhead): simulate() -> _plan_run, _plan_run's captured
    values and _plan_run -> _route pass their values positionally (locals()
    and > 30-keyword calls cost per call); each name list must equal the
    callee's parameters, in order, in tuples of <= 30 names."""
    names = list(inspect.signature(p.simulate).parameters)
    assert list(inspect.signature(p._plan_run).parameters) == names
    assert list(p._SIMULATE_DEFAULTS) == names[1:]
    fns = _functions(_PKG / "__init__.py", {"simulate", "_plan_run"})

    def call_to(fn, callee):
        hits = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name) and c.func.id == callee]
        assert len(hits) == 1, callee
        return hits[0]

    assert _star_names(call_to(fns["simulate"], "_plan_run")) == names
    captured = call_to(fns["_plan_run"], "_resolved_from").args[0]
    assert _star_names(ast.Call(func=ast.Name("f"), keywords=[],
                                args=[ast.Starred(captured)])) == names[1:]
    assert _star_names(call_to(fns["_plan_run"], "_route")) == list(
        inspect.signature(p._route).parameters)


def test_plan_fields_and_immutability():
    """The plan's records are NamedTuples built positionally: each field
    holds what its name says, and none can be set from the outside."""
    b = _rc()
    plan = _plan(b, 1e-4, 1e-6, engine="pwl")
    assert plan.caller_builder is b and plan.builder is not b
    assert isinstance(plan.circuit, p.Circuit)
    assert isinstance(plan.circuit_plan, p.RunPlan)
    assert plan.circuit_plan.preflight is plan.preflight
    assert plan.kernel_run is None and plan.recipe is None
    assert (plan.engine, plan.engine_asked, plan.route_reasons) == ("pwl", "pwl", ())
    assert (plan.t_start, plan.t_end) == (0.0, 1e-4)
    assert type(plan.step).__name__ == "FixedStep" and plan.step.dt == 1e-6
    assert type(plan.newton).__name__ == "NewtonOptions"
    assert type(plan.events).__name__ == "EventOptions"
    assert plan.events.on_impulse == "record"
    assert type(plan.initial).__name__ == "Initial"
    assert type(plan.drive).__name__ == "Drive"
    assert plan.device_path == "legacy"
    assert plan.device_path_enum == p.DevicePath.Legacy
    assert plan.voltage_sanity_factor is None and plan.auto_regularize is True
    assert plan.resolved_from["dt"] == "explicit"
    assert plan.diagnostics == ()
    for rec, field in ((plan, "engine"), (plan.step, "dt"), (plan.drive, "switch_fn"),
                       (plan.newton, "tol_dx"), (plan.events, "strict"),
                       (plan.initial, "state")):
        with pytest.raises(AttributeError):
            setattr(rec, field, None)
        with pytest.raises(AttributeError):
            rec.new_attribute = 1
    routed = _plan(b, 1e-4, 1e-6)   # 'auto' with dt: routed to pwl, with a reason
    assert routed.engine == "pwl" and routed.engine_asked == "auto"
    assert [d.message for d in routed.diagnostics] == list(routed.route_reasons)


def test_every_positional_record_has_its_arity():
    """`_planning.new(_planning.X, (...))` (tuple.__new__, no check of its
    own) passes exactly X's fields."""
    import pulsim._plan as planning
    tree = ast.parse((_PKG / "__init__.py").read_text(encoding="utf-8"))
    seen = set()
    for c in ast.walk(tree):
        if isinstance(c, ast.Call) and ast.unparse(c.func) == "_planning.new":
            cls, values = c.args
            assert isinstance(values, ast.Tuple), ast.unparse(c)
            name = ast.unparse(cls).removeprefix("_planning.")
            assert len(values.elts) == len(getattr(planning, name)._fields), name
            seen.add(name)
    assert seen == {"SimPlan", "NewtonOptions", "EventOptions", "Initial", "Drive",
                    "FixedStep", "VariableStep", "ExactStep", "RunOutcome"}


def test_run_outcome_fields():
    """RunOutcome, built positionally by each runner: each field holds what
    its name says (pwl and trbdf2)."""
    obs = []
    plan = _plan(_rc(), 1e-5, 1e-6, engine="pwl",
                 step_observer=lambda t, x: obs.append(t))
    out = p._run_pwl(plan)
    assert len(out.result.times) > 1 and out.builder is not None
    assert callable(out.switch_fn) and out.b_extra_fn is None
    assert callable(out.step_observer) and out.should_continue is None
    assert out.stats is None and out.cxx_chain is None and out.mmc_finalize is None
    assert plan.drive.user_step_observer is not None
    plan = _plan(_rc(), 1e-5, engine="trbdf2")
    out = p._run_trbdf2(plan)
    assert isinstance(out.stats, dict) and "control_tick_counts" in out.stats
    assert out.cxx_chain is None and out.mmc_finalize is None
