"""Controller state per run, snapshot and sweep point (DECISIONS G26).

The final snapshot carries the state of the run's control graphs
(closed-loop helpers, chains passed as ``step_observer``, the circuit's
C blocks), and ``resume_from`` restores it, so a resumed run continues
the snapshot's run whatever ran in between. A plain run keeps G22: the
circuit's C blocks restart, the caller's graphs carry over.

What a graph's state is: the kernel blocks' states, signals, tick grid
and recordings (``CxxBlockChain.save_state``); a ``PythonBlock`` (or a
Python library block) wrapping an object: a deep copy of the object's
``__dict__`` (a block wrapping a function is taken as pure); a C block
with a Python step: its state dict and buffers; a native C block: the
optional ``pulsim_cblock_save`` / ``pulsim_cblock_restore`` pair (every
``export_c`` library has it). A state no copy reaches is refused by
name where it would be needed (resume, a multi-point sweep), never
shared silently.

Sweeps: a point is one run. The caller's graphs restart per point (the
state they had when the sweep began), the caller's Python controllers
are deep-copied per point, a closure over mutable state is refused by
name. On threads one graph would be stepped by several points at once:
refused by name.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import dis
import types
from functools import partial

from ._refusal import unsupported

_FEATURE = "controller_state"

# Inside a sweep: id(chain) -> (chain, GraphState at the sweep's start).
_POINTS: contextvars.ContextVar = contextvars.ContextVar(
    "pulsim_sweep_points", default=None)


@contextlib.contextmanager
def sweep_points():
    """Every run inside restarts the caller's graphs from the state they
    had when the block began (one sweep's points)."""
    tok = _POINTS.set({})
    try:
        yield
    finally:
        _POINTS.reset(tok)


def _cxx(obs):
    return getattr(obs, "_cxx_chain", None)


def run_graphs(plan) -> list:
    """The run's control-graph observers (one per C++ graph, plan order)."""
    seen, out = set(), []
    for obs in list(plan.inner_observers) + [plan.c_graph]:
        cxx = _cxx(obs)
        if cxx is not None and id(cxx) not in seen:
            seen.add(id(cxx))
            out.append(obs)
    return out


def _is_pure(fn) -> bool:
    return isinstance(fn, (types.FunctionType, types.BuiltinFunctionType,
                           types.BuiltinMethodType, partial))


def _block_parts(graph):
    """(kind, object) per Python-held state of the graph's blocks."""
    parts = []
    for spec in getattr(graph, "blocks", ()):
        b = spec.block
        kind = type(b).__name__
        if kind == "CBlock":
            parts.append(("cblock", spec.name, b))
            fn = getattr(b, "fn", None)
            if getattr(b, "_keep", None) is None and _closure_state(fn):
                parts.append(("closure", spec.name, fn))
            continue
        obj = getattr(b, "_wrapped", b) if kind == "PythonBlock" else (
            b if kind in ("SlidingModeObserver", "FluxMRASObserver") else None)
        if obj is None:
            continue
        if _is_pure(obj):
            if _closure_state(obj):     # a function over mutable cells
                parts.append(("closure", spec.name, obj))
            continue
        obj = getattr(obj, "__self__", obj)        # a bound method's object
        parts.append(("object", spec.name, obj))
    return parts


def _closure_state(fn, seen=None) -> bool:
    """A function (or partial) whose cells, defaults or bound arguments
    may hold mutable state (followed into the functions they hold)."""
    seen = set() if seen is None else seen
    if isinstance(fn, partial):
        return _mutable_cell(fn.func, seen) or any(
            _mutable_cell(v, seen) for v in (*fn.args, *fn.keywords.values()))
    if not isinstance(fn, types.FunctionType):
        return False
    seen.add(id(fn))
    cells = [c.cell_contents for c in (fn.__closure__ or ()) if _has(c)]
    cells += list(fn.__defaults__ or ())
    cells += list((fn.__kwdefaults__ or {}).values())
    g = fn.__globals__        # module-level names the body reads
    cells += [g[i.argval] for i in dis.get_instructions(fn)
              if i.opname in ("LOAD_GLOBAL", "LOAD_NAME") and i.argval in g]
    return any(_mutable_cell(v, seen) for v in cells)


def _save_part(kind, name, obj):
    """A restorable copy, or a string saying why there is none."""
    if kind == "closure":
        return (f"block '{name}' runs a function whose closure or defaults "
                "may hold mutable state (a list, dict, array or object); "
                "make it a callable object (its __dict__ is copied)")
    if kind == "object":
        d = getattr(obj, "__dict__", None)
        if d is None:
            return (f"block '{name}' wraps a {type(obj).__name__} without "
                    "a __dict__ (its state cannot be copied)")
        try:
            return copy.deepcopy(d)
        except Exception as exc:  # noqa: BLE001 - named in the refusal
            return f"block '{name}': its state does not deep-copy ({exc})"
    keep = getattr(obj, "_keep", None)
    if keep is not None:                       # native library
        saved = keep._save() if hasattr(keep, "_save") else None
        if saved is None:
            return (f"C block '{name}' (native library): its *state is "
                    "opaque; the library exports no pulsim_cblock_save / "
                    "pulsim_cblock_restore pair (export_c libraries do)")
        return (saved, obj.inputs.copy(), obj.outputs.copy())
    try:
        return (copy.deepcopy(obj.state), obj.inputs.copy(), obj.outputs.copy())
    except Exception as exc:  # noqa: BLE001
        return f"C block '{name}': its state dict does not deep-copy ({exc})"


def _restore_part(kind, obj, saved):
    if kind == "object":
        obj.__dict__.clear()
        obj.__dict__.update(copy.deepcopy(saved))
        return
    state, inputs, outputs = saved
    keep = getattr(obj, "_keep", None)
    if keep is not None:
        keep._restore(state)
    else:
        obj.state.clear()
        obj.state.update(copy.deepcopy(state))
    obj.inputs[:] = inputs
    obj.outputs[:] = outputs


class GraphState:
    """One graph's state: the kernel part and the Python-held parts."""

    def __init__(self, obs):
        graph = getattr(obs, "_graph", None)
        cxx = _cxx(obs)
        self.cxx_state = cxx.save_state()
        self.parts = []
        self.missing = [f"kernel block '{n}' (state not captured)"
                        for n in cxx.uncaptured_blocks()]
        for kind, name, obj in _block_parts(graph):
            saved = _save_part(kind, name, obj)
            if isinstance(saved, str):
                self.missing.append(saved)
            else:
                self.parts.append((kind, obj, saved))

    def restore(self, obs, why: str) -> None:
        if self.missing:
            raise unsupported(
                "resume_from" if why == "resume" else "sweep", _FEATURE,
                f"{why}: a controller holds state pulsim cannot copy — "
                + "; ".join(self.missing) + ". Run the points / segments "
                "in separate processes, or give the block a capturable "
                "state.")
        _cxx(obs).restore_state(self.cxx_state)
        for kind, obj, saved in self.parts:
            _restore_part(kind, obj, saved)


def start_run(plan, snapshot) -> list:
    """Put every graph of the run in its starting state: the snapshot's
    (``resume_from``), the sweep's start (inside ``sweep_points``), or
    as the last run left it (a plain run, G22). Returns the graphs."""
    graphs = run_graphs(plan)
    if not graphs:
        return graphs
    payload = None if snapshot is None else getattr(
        snapshot, "_control_state", None)
    if isinstance(payload, str):
        raise unsupported(
            "resume_from", _FEATURE,
            "simulate(resume_from=...): this snapshot was pickled (or "
            "copied through pickle) and its controllers' state does not "
            "pickle, so the closed loops / C blocks would continue "
            "whatever their last run left. Resume from the result's own "
            "final_snapshot.")
    saved = {} if payload is None else {id(c): (c, s) for c, s in payload}
    for obs in graphs:
        cxx = _cxx(obs)
        if snapshot is not None and saved:
            hit = saved.get(id(cxx))
            if hit is None or hit[0] is not cxx:
                raise unsupported(
                    "resume_from", _FEATURE,
                    "simulate(resume_from=...): the snapshot holds the "
                    "state of other controllers than this run's; resume "
                    "with the same closed_loops / step_observer / circuit "
                    "objects that produced it.")
            hit[1].restore(obs, "resume")
            continue
        if obs is plan.c_graph:
            # C blocks: a fresh run was reset by _c_graph_observer (natives
            # re-run init); a resume from a snapshot without them restarts
            if snapshot is not None:
                from .c_block import _reset_c_blocks
                _reset_c_blocks(obs._graph)
            continue
        if snapshot is not None and cxx.n_calls:
            raise unsupported(
                "resume_from", _FEATURE,
                "simulate(resume_from=...): the snapshot carries no state "
                "for a controller that has already run, so the resumed run "
                "would continue whatever its last run left. Resume from the "
                "final_snapshot of a run with this controller, or use a "
                "fresh one.")
        starts = _POINTS.get()
        if starts is None:
            continue                 # a plain run: the caller's graph carries over (G22)
        s0 = starts.get(id(cxx))
        if s0 is None:
            starts[id(cxx)] = (cxx, GraphState(obs))   # first point
        else:
            s0[1].restore(obs, "sweep")
    return graphs


def save_into(graphs, res) -> None:
    """Attach the graphs' state to the result's final snapshot."""
    if not graphs:
        return
    snap = getattr(res, "final_snapshot", None)
    if snap is None or not getattr(snap, "valid", False):
        return
    snap._control_state = [(_cxx(o), GraphState(o)) for o in graphs]


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

_MUTABLE = (list, dict, set, bytearray)


def _mutable_cell(v, seen=None) -> bool:
    """May ``v`` (a closure cell, default or bound argument) hold state?"""
    seen = set() if seen is None else seen
    if id(v) in seen:
        return False
    seen.add(id(v))
    if isinstance(v, _MUTABLE):
        return True
    mod = type(v).__module__ or ""
    if mod.startswith("numpy"):
        return getattr(v, "flags", None) is not None and v.flags.writeable
    if isinstance(v, (types.FunctionType, partial)):
        return _closure_state(v, seen)
    if isinstance(v, types.MethodType):
        return _mutable_cell(v.__self__, seen)
    if isinstance(v, (types.ModuleType, type, types.BuiltinFunctionType)):
        return False
    if mod.startswith("pulsim._pulsim"):
        return False        # native objects (builders, masks, schedules)
    return hasattr(v, "__dict__")


def _kind(obj):
    """'graph', 'object', 'closure' or None (pure / absent)."""
    if obj is None:
        return None
    from .periodic_steady_state import _closed_loop_source
    if _cxx(obj) is not None or getattr(obj, "_graph", None) is not None \
            or getattr(obj, "follows_controller", False) \
            or _closed_loop_source(obj):
        return "graph"
    if (type(obj).__module__ or "").startswith("pulsim._pulsim"):
        return None                                # a native schedule
    if isinstance(obj, (types.FunctionType, partial)):
        return "closure" if _closure_state(obj) else None
    if isinstance(obj, types.BuiltinFunctionType):
        return None
    return "object"


def _has(cell) -> bool:
    try:
        cell.cell_contents
    except ValueError:
        return False
    return True


def _controllers(kw):
    """(name, object) for each controller-like keyword argument."""
    out = []
    for name in ("step_observer", "b_extra_fn", "switch_fn"):
        if kw.get(name) is not None:
            out.append((name, kw[name]))
    for i, loop in enumerate(kw.get("closed_loops") or ()):
        out.append((f"closed_loops[{i}]", getattr(loop, "step_observer", loop)))
        if getattr(loop, "switch_fn", None) is not None:
            out.append((f"closed_loops[{i}].switch_fn", loop.switch_fn))
    return out


def point_kwargs(kw: dict, n: int, parallel: bool) -> list:
    """The ``simulate()`` keyword arguments of each of ``n`` sweep points:
    the caller's Python controller objects deep-copied per point; a
    closure over mutable state refused; a control graph (restarted per
    point under ``sweep_points``) refused on parallel points, which
    would step it at once."""
    if n < 2:
        return [kw] * n
    kinds = [(name, _kind(o)) for name, o in _controllers(kw)]
    bad = [name for name, k in kinds if k == "closure"]
    if bad:
        raise unsupported(
            "sweep", _FEATURE,
            f"sweep / monte_carlo: {', '.join(bad)} is a closure that may "
            "hold mutable state (a list, dict, array or object in its cells "
            "or defaults), so every point could continue the previous "
            "point's controller and it cannot be copied. Make the "
            "controller a callable object (deep-copied per point) or "
            "build it with bind_pi_to_switch / a MixedDomainBlockChain "
            "(restarted per run).")
    graphs = [name for name, k in kinds if k == "graph"]
    if parallel and graphs:
        raise unsupported(
            "sweep", _FEATURE,
            f"sweep(parallel=True) with {', '.join(graphs)}: the points "
            "would share one controller's state on threads (one control "
            "graph stepped by several runs at once). Run parallel=False "
            "(every run restarts the graph), or pass a builder factory "
            "that builds a fresh controller per point.")
    if not any(k == "object" for _, k in kinds):
        return [kw] * n
    return [_copy_point(kw) for _ in range(n)]


def _copy_point(kw: dict) -> dict:
    """``kw`` with each controller object deep-copied (one memo, so an
    observer and its gate stay paired); graphs are left to sweep_points."""
    memo: dict = {}
    d = dict(kw)

    def cp(name, obj):
        try:
            return copy.deepcopy(obj, memo)
        except Exception as exc:  # noqa: BLE001 - named in the refusal
            raise unsupported(
                "sweep", _FEATURE,
                f"sweep / monte_carlo: {name} holds state that does not "
                f"deep-copy ({exc}); every point would continue the "
                "previous point's controller. Make it copyable, or run one "
                "simulate() per point with a fresh controller.") from exc

    for k in ("step_observer", "b_extra_fn", "switch_fn"):
        if kw.get(k) is not None and _kind(kw[k]) == "object":
            d[k] = cp(k, kw[k])
    loops = kw.get("closed_loops")
    if loops:
        d["closed_loops"] = [
            cp(f"closed_loops[{i}]", lp)
            if "object" in (_kind(getattr(lp, "step_observer", lp)),
                            _kind(getattr(lp, "switch_fn", None)))
            else lp
            for i, lp in enumerate(loops)]
    return d


__all__ = ["GraphState", "point_kwargs", "run_graphs", "save_into",
           "start_run", "sweep_points"]
