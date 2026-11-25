"""Compiled control-graph support for :func:`pulsim.periodic_steady_state`.

Extracts and seeds the discrete state vector ``x_ctrl`` of compiled
control graphs (``MixedDomainBlockChain`` / ``ClosedLoop`` /
``CxxBlockChain``) via ``CxxBlockChain.dump_state`` / ``load_state`` so
Newton/Broyden shooting solves ``P([x_circuit; x_ctrl]) = [x_circuit;
x_ctrl]`` without mutating caller state on failure.

Refuses any plain Python closure, stateful user object, or opaque
``PythonBlock`` / ``CBlock`` carrying hidden uninspectable state with
``UnsupportedError("periodic_steady_state", "hidden_controller_state", ...)``.
"""

from __future__ import annotations

import copy
import dis
import math
import struct
import types
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np

from ._control_state import _cxx, _has, _mutable_cell
from ._refusal import unsupported

_ANALYSIS = "periodic_steady_state"

_STATELESS_KINDS = frozenset({
    "Gain",
    "Subtract",
    "Sum",
    "MathBlock",
    "Limiter",
    "OpAmp",
    "LookupTable1D",
    "SignalMux",
    "SignalDemux",
    "PwmGenerator",
    "SpaceVectorModulator",
    "ClarkeTransform",
    "InverseClarkeTransform",
    "ParkTransform",
    "InverseParkTransform",
})

_NON_SMOOTH_KINDS = frozenset({
    "Comparator",
    "StateMachine",
    "SampleHold",
})

_PY_LIBRARY_KINDS = frozenset({
    "SlidingModeObserver",
    "FluxMRASObserver",
})

_SENTINEL = 1.1234567890123456


def _unsup_hidden(where: str, reason: str, *, analysis: str = _ANALYSIS):
    return unsupported(
        analysis,
        "hidden_controller_state",
        f"{analysis}: {where} holds hidden or uninspectable controller "
        f"state ({reason}), which the periodic shooting solver cannot read "
        "or seed as part of x0. Use a compiled control graph "
        "(MixedDomainBlockChain or bind_pi_to_switch) for closed-loop "
        "periodic steady state, or a stateless schedule "
        "(make_pwm_switch_fn / SwitchSchedule) for open-loop shooting.",
    )


def _mutates_deref_or_global(fn: types.FunctionType) -> str:
    """Reason string when ``fn`` mutates closure, global, or attribute state."""
    for ins in dis.get_instructions(fn):
        if ins.opname in ("STORE_DEREF", "DELETE_DEREF"):
            return f"mutates closed-over variable {ins.argval!r}"
        if ins.opname in ("STORE_GLOBAL", "DELETE_GLOBAL"):
            return f"mutates global variable {ins.argval!r}"
        if ins.opname in ("STORE_ATTR", "DELETE_ATTR"):
            return f"mutates attribute {ins.argval!r}"
    return ""


def _hidden_state_reason(obj: Any, seen: set | None = None) -> str:
    """Return a non-empty reason if ``obj`` carries hidden mutable Python state."""
    if obj is None:
        return ""
    mod = type(obj).__module__ or ""
    if mod.startswith("pulsim._pulsim"):
        return ""
    if _cxx(obj) is not None or getattr(obj, "_graph", None) is not None:
        return ""
    if getattr(obj, "_pulsim_monitor", False):
        return _hidden_state_reason(getattr(obj, "fn", None), seen)

    seen = set() if seen is None else seen
    if id(obj) in seen:
        return ""
    seen.add(id(obj))

    if isinstance(obj, partial):
        r = _hidden_state_reason(obj.func, seen)
        if r:
            return r
        for v in (*obj.args, *obj.keywords.values()):
            if _mutable_cell(v, set(seen)):
                return f"partial binds mutable state ({type(v).__name__})"
        return ""

    if isinstance(obj, types.MethodType):
        return _hidden_state_reason(obj.__self__, seen)

    if isinstance(obj, types.BuiltinFunctionType):
        return ""

    if isinstance(obj, types.FunctionType):
        mut = _mutates_deref_or_global(obj)
        if mut:
            return mut
        fn_dict = getattr(obj, "__dict__", None)
        if fn_dict:
            attrs = [k for k in fn_dict if not str(k).startswith("_pulsim")]
            if attrs:
                return f"function holds attribute state ({attrs[0]!r})"
        for cell in obj.__closure__ or ():
            if not _has(cell):
                continue
            v = cell.cell_contents
            if isinstance(v, (types.FunctionType, partial, types.MethodType)):
                r = _hidden_state_reason(v, seen)
                if r:
                    return r
            elif _mutable_cell(v, set(seen)):
                return f"closure captures mutable {type(v).__name__}"
        for v in obj.__defaults__ or ():
            if _mutable_cell(v, set(seen)):
                return f"default argument holds mutable {type(v).__name__}"
        for v in (obj.__kwdefaults__ or {}).values():
            if _mutable_cell(v, set(seen)):
                return f"keyword default holds mutable {type(v).__name__}"
        g = obj.__globals__
        for ins in dis.get_instructions(obj):
            if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME") and ins.argval in g:
                v = g[ins.argval]
                if isinstance(v, (types.FunctionType, partial, types.MethodType)):
                    r = _hidden_state_reason(v, seen)
                    if r:
                        return r
                elif _mutable_cell(v, set(seen)):
                    return f"reads mutable global {ins.argval!r} ({type(v).__name__})"
        return ""

    if isinstance(obj, (types.ModuleType, type)):
        return ""

    # User callable object (class instance)
    d = getattr(obj, "__dict__", None)
    if d:
        return f"stateful {type(obj).__name__} instance"
    slots = getattr(type(obj), "__slots__", None)
    if slots:
        slot_names = (slots,) if isinstance(slots, str) else slots
        for s in slot_names:
            if s not in ("__dict__", "__weakref__") and hasattr(obj, s):
                return f"stateful {type(obj).__name__} instance"
    # Slotless instance: inspect mutable class attributes across MRO & __call__
    for cls in type(obj).__mro__:
        if cls is object:
            continue
        for k, v in cls.__dict__.items():
            if k.startswith("__") and k.endswith("__"):
                continue
            if isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    staticmethod,
                    classmethod,
                    property,
                    types.MemberDescriptorType,
                    types.GetSetDescriptorType,
                ),
            ):
                continue
            if _mutable_cell(v, set(seen)):
                return (
                    f"stateful {type(obj).__name__} class attribute "
                    f"{k!r} ({type(v).__name__})"
                )
    call_fn = getattr(type(obj), "__call__", None)
    if isinstance(call_fn, types.FunctionType):
        r = _hidden_state_reason(call_fn, seen)
        if r:
            return r
    return ""


class _ObserverLoop:
    """Loop-compatible adapter for a resolved observer / bare chain."""

    __slots__ = ("step_observer", "switch_fn", "period")

    def __init__(self, step_observer, switch_fn=None, period=None):
        self.step_observer = step_observer
        self.switch_fn = switch_fn
        self.period = period


def _resolve_observer(builder, obj, *, allow_compile: bool = True):
    """Turn a MixedDomainBlockChain or observer into a compiled observer."""
    from .blockchain import MixedDomainBlockChain

    if isinstance(obj, MixedDomainBlockChain):
        if obj._cxx_chain is not None:
            ref = getattr(obj, "_last_observer_ref", None)
            obs = ref() if ref is not None else None
            if obs is not None and getattr(obs, "_cxx_chain", None) is obj._cxx_chain:
                return obs
        if not allow_compile:
            from . import _options_error
            raise _options_error("step_observer",
                "periodic_steady_state: MixedDomainBlockChain has not been "
                "compiled; call chain.make_step_observer(builder) or pass "
                "the circuit builder."
            )
        return obj.make_step_observer(builder)
    return obj


def normalize_control_args(
    builder,
    *,
    switch_fn=None,
    closed_loops=None,
    step_observer=None,
    controller_period: float | None = None,
    control=None,
):
    """Normalize ``control=`` / ``closed_loops=`` / ``step_observer=``."""
    from .blockchain import MixedDomainBlockChain

    if control is not None:
        if closed_loops is not None or step_observer is not None:
            from . import _options_error
            raise _options_error("control",
                "periodic_steady_state: pass control= OR closed_loops= / "
                "step_observer=, not both."
            )
        if isinstance(control, MixedDomainBlockChain):
            step_observer = _resolve_observer(builder, control)
        elif hasattr(control, "step_observer") and hasattr(control, "switch_fn"):
            closed_loops = [control]
        elif isinstance(control, Sequence) and not isinstance(control, (str, bytes)):
            closed_loops = list(control)
        else:
            step_observer = control

    if isinstance(step_observer, MixedDomainBlockChain):
        step_observer = _resolve_observer(builder, step_observer)

    loops = list(closed_loops) if closed_loops is not None else None
    if loops is not None:
        norm_loops = []
        bare_observers = []
        for lp in loops:
            if isinstance(lp, MixedDomainBlockChain):
                bare_observers.append(_resolve_observer(builder, lp))
            elif hasattr(lp, "step_observer") and hasattr(lp, "switch_fn"):
                obs = lp.step_observer
                if isinstance(obs, MixedDomainBlockChain):
                    obs = _resolve_observer(builder, obs)
                    norm_loops.append(
                        _ObserverLoop(obs, lp.switch_fn, getattr(lp, "period", None))
                    )
                else:
                    norm_loops.append(lp)
            else:
                bare_observers.append(lp)

        if bare_observers:
            if step_observer is None:
                pick = 0
                for idx, obs in enumerate(bare_observers):
                    if (
                        getattr(obs, "_b_extra_fn", None) is not None
                        or getattr(obs, "_rate_offsets", None)
                        or len(getattr(obs, "_rate_ticks", ())) > 1
                    ):
                        pick = idx
                        break
                step_observer = bare_observers.pop(pick)
            for obs in bare_observers:
                p_obs = getattr(obs, "period", None)
                if p_obs is None and getattr(obs, "_rate_ticks", None):
                    p_obs = float(obs._rate_ticks[0][1])
                if p_obs is None:
                    p_obs = controller_period
                norm_loops.append(_ObserverLoop(obs, None, p_obs))

        loops = norm_loops or None

    return switch_fn, loops, step_observer, controller_period


def check_hidden_controller_state(
    *,
    switch_fn=None,
    closed_loops=None,
    step_observer=None,
    analysis: str = _ANALYSIS,
    refuse_closed_loop_fn=None,
    period: float | None = None,
) -> None:
    """Refuse hidden Python controller state across switch_fn / observers / loops.

    Note: ``refuse_closed_loop_fn(sf)`` is checked before the closure check
    on ``switch_fn`` so closed-loop gate wrappers like a closure around
    ``duty_get`` of ``bind_pi_to_duty_callable`` keep raising
    ``feature='closed_loop'`` when their controller is absent.
    """
    # 1. Check closed_loops entries
    for i, lp in enumerate(closed_loops or ()):
        obs = getattr(lp, "step_observer", lp)
        if _cxx(obs) is None:
            reason = _hidden_state_reason(obs) or (
                f"non-graph callable {type(obs).__name__}"
            )
            raise _unsup_hidden(f"closed_loops[{i}]", reason, analysis=analysis)
        _check_graph_blocks_inspectable(obs, analysis=analysis)
        sw = getattr(lp, "switch_fn", None)
        if sw is not None:
            r = _hidden_state_reason(sw)
            if r:
                raise _unsup_hidden(
                    f"closed_loops[{i}].switch_fn", r, analysis=analysis
                )

    # 2. Check step_observer
    if step_observer is not None:
        if _cxx(step_observer) is not None:
            _check_graph_blocks_inspectable(step_observer, analysis=analysis)
        else:
            reason = _hidden_state_reason(step_observer)
            if reason:
                raise _unsup_hidden("step_observer", reason, analysis=analysis)
            # Plain Python step_observer without compiled control graph
            raise _unsup_hidden(
                "step_observer",
                f"plain Python callable {type(step_observer).__name__} is not "
                "a compiled control graph",
                analysis=analysis,
            )

    # 3. Check switch_fn (allow closed-loop refusal to run first if provided)
    if switch_fn is not None:
        if refuse_closed_loop_fn is not None:
            refuse_closed_loop_fn(switch_fn)
        r = _hidden_state_reason(switch_fn)
        if r and period is not None:
            # state that exists is not refused, state that moves is
            r = _gate_state_moves(switch_fn, float(period))
        if r:
            raise _unsup_hidden("switch_fn", r, analysis=analysis)


# ---------------------------------------------------------------------------
# Measured gate state: snapshot what the gate can reach, run it over one
# period, refuse by name what changed.
# ---------------------------------------------------------------------------

_HEAPTYPE = 1 << 9          # Py_TPFLAGS_HEAPTYPE: a class defined in Python
_VALUE_TYPES = (type(None), bool, int, str, bytes, range, type(Ellipsis))


class _Opaque(Exception):
    pass


_IMMUTABLETYPE = 1 << 8     # set on C types (static and most heap ones)


def _pure_python(cls) -> bool:
    """Every class in the MRO is a class statement: no C-level state can
    hide behind the instance's ``__dict__`` / slots."""
    n_slots = 0
    for c in reversed(cls.__mro__):
        if c is object:
            continue
        if not c.__flags__ & _HEAPTYPE or c.__flags__ & _IMMUTABLETYPE:
            return False
        sl = c.__dict__.get("__slots__", ())
        n_slots += len((sl,) if isinstance(sl, str) else sl)
        # a C base (a mutable heap type such as _random.Random) adds a
        # layout no slot accounts for; 16 = dict + weakref pointers
        if c.__basicsize__ > object.__basicsize__ + 8 * n_slots + 16:
            return False
        meta = type(c)
        # shortcut: binding metaclasses named here; another binding
        # generator with mutable heap types would read as Python
        if (meta.__module__ or "").split(".")[0] in ("pybind11_builtins",
                                                     "nanobind", "sip",
                                                     "shiboken6", "Boost"):
            return False
    return True


def _library(v) -> bool:
    """Code (a function or class) of pulsim, the standard library or an
    installed package: taken as stateless, its data is not walked."""
    import sys
    mod = (getattr(v, "__module__", None) or "") if isinstance(
        v, (type, types.FunctionType)) else ""
    if mod.split(".")[0] in ("pulsim", "numpy", "scipy"):
        return True
    code = getattr(v, "__code__", None)
    fname = getattr(code, "co_filename", "") or ""
    if not fname and isinstance(v, type):
        m = sys.modules.get(mod)
        fname = getattr(m, "__file__", "") or ""
    return "site-packages" in fname or (
        bool(fname) and fname.startswith(sys.base_prefix))


def _code_names(code, gl: set, at: set) -> None:
    for ins in dis.get_instructions(code):
        if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME", "STORE_GLOBAL",
                          "DELETE_GLOBAL", "STORE_NAME"):
            gl.add(ins.argval)
        elif ins.opname in ("LOAD_ATTR", "STORE_ATTR", "DELETE_ATTR",
                            "LOAD_METHOD"):
            at.add(ins.argval)
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            _code_names(c, gl, at)


def _fn_roots(fn, path: str, out: dict, seen: set) -> None:
    """Snapshot what a Python function can reach: closure cells, function
    attributes, defaults, and the globals (and module attributes) its code
    names."""
    code = fn.__code__
    for name, cell in zip(code.co_freevars, fn.__closure__ or ()):
        p_ = f"{path}.<closure {name}>"
        if _has(cell):
            _snap(cell.cell_contents, p_, out, seen)
        else:
            out[p_] = ("empty",)
    for k, v in (getattr(fn, "__dict__", None) or {}).items():
        if not str(k).startswith("_pulsim"):
            _snap(v, f"{path}.{k}", out, seen)
    for i, v in enumerate(fn.__defaults__ or ()):
        _snap(v, f"{path}.<default {i}>", out, seen)
    for k, v in (fn.__kwdefaults__ or {}).items():
        _snap(v, f"{path}.<default {k}>", out, seen)
    gl: set = set()
    at: set = set()
    _code_names(code, gl, at)
    g = fn.__globals__
    for name in sorted(gl, key=str):
        p_ = f"{path}.<global {name}>"
        if name not in g:
            out[p_] = ("missing",)
            continue
        _snap(g[name], p_, out, seen)
        _mod_attrs(g[name], at, p_, out, seen, 0)


def _mod_attrs(mod, at: set, path: str, out: dict, seen: set, depth: int):
    # a module's attributes the code names (`cfg.duty`, `np.random.rand`)
    if not isinstance(mod, types.ModuleType) or depth > 3:
        return
    for a in sorted(at, key=str):
        if hasattr(mod, a):
            v = getattr(mod, a)
            _snap(v, f"{path}.{a}", out, seen)
            _mod_attrs(v, at, f"{path}.{a}", out, seen, depth + 1)


def _snap(v, path: str, out: dict, seen: set) -> None:
    """Flatten ``v`` into ``out[path] = comparable leaf``. Plain data and
    arrays compare by value, mutable containers and objects by identity and
    contents; a value with no visible state and no ``__eq__`` raises
    :class:`_Opaque` naming the path."""
    if isinstance(v, _VALUE_TYPES):
        out[path] = (type(v), v)
        return
    if isinstance(v, (float, complex)):
        out[path] = (type(v), repr(v))              # NaN == NaN by repr
        return
    if isinstance(v, np.generic):
        out[path] = ("np", v.dtype.str, v.tobytes())
        return
    if isinstance(v, np.ndarray):
        if v.dtype.hasobject:
            out[path] = ("nd", v.shape)
            for i, e in enumerate(v.ravel()):
                _snap(e, f"{path}[{i}]", out, seen)
        else:
            out[path] = ("nd", v.dtype.str, v.shape, v.tobytes())
        return
    if isinstance(v, types.ModuleType):
        out[path] = ("mod", id(v))
        return
    tmod = (v.__module__ if isinstance(v, type) else type(v).__module__) or ""
    if str(tmod).startswith("pulsim._pulsim"):
        out[path] = ("kernel", id(v))   # C++ objects: not Python state
        return
    if id(v) in seen:
        out[path] = ("seen", id(v))
        return
    if isinstance(v, (tuple, frozenset)) and type(v) in (tuple, frozenset):
        seen.add(id(v))
        items = sorted(v, key=repr) if isinstance(v, frozenset) else v
        out[path] = (type(v), len(v))
        for i, e in enumerate(items):
            _snap(e, f"{path}[{i}]", out, seen)
        return
    if isinstance(v, (set, frozenset)):
        seen.add(id(v))
        out[path] = ("set", id(v), frozenset(v))
        return
    if isinstance(v, (list, bytearray)) or type(v).__name__ == "deque":
        seen.add(id(v))
        out[path] = ("list", id(v), len(v))
        for i, e in enumerate(v):
            _snap(e, f"{path}[{i}]", out, seen)
        return
    if isinstance(v, dict):
        seen.add(id(v))
        out[path] = ("dict", id(v), tuple(repr(k) for k in v))
        for k, e in v.items():
            _snap(e, f"{path}[{k!r}]", out, seen)
        return
    if isinstance(v, types.FunctionType):
        seen.add(id(v))
        out[path] = ("fn", id(v))
        if not _library(v):
            _fn_roots(v, path, out, seen)
        return
    if isinstance(v, types.MethodType):
        out[path] = ("method", id(v.__func__))
        _snap(v.__self__, f"{path}.__self__", out, seen)
        _snap(v.__func__, f"{path}.__func__", out, seen)
        return
    if isinstance(v, partial):
        seen.add(id(v))
        out[path] = ("partial", id(v))
        _snap(v.func, f"{path}.func", out, seen)
        _snap(v.args, f"{path}.args", out, seen)
        _snap(dict(v.keywords), f"{path}.keywords", out, seen)
        return
    if isinstance(v, (types.BuiltinFunctionType, types.BuiltinMethodType,
                      types.MethodWrapperType)):
        s = getattr(v, "__self__", None)
        out[path] = ("builtin", id(v))
        if s is not None and not isinstance(s, (types.ModuleType, type)):
            _snap(s, f"{path}.__self__", out, seen)  # e.g. an RNG's method
        return
    if isinstance(v, type):
        seen.add(id(v))
        out[path] = ("class", id(v))
        if _library(v) or not v.__flags__ & _HEAPTYPE:
            return
        for c in v.__mro__:
            if c is object or not c.__flags__ & _HEAPTYPE:
                continue
            for k, a in c.__dict__.items():
                if k.startswith("__") and k.endswith("__") and k != "__call__":
                    continue
                if isinstance(a, (staticmethod, classmethod)):
                    a = a.__func__
                if isinstance(a, (property, types.MemberDescriptorType,
                                  types.GetSetDescriptorType)):
                    continue
                _snap(a, f"{path}.{k}", out, seen)
        return
    cls = type(v)
    if _pure_python(cls):
        seen.add(id(v))
        out[path] = ("obj", id(v), id(cls))
        for k, a in (getattr(v, "__dict__", None) or {}).items():
            _snap(a, f"{path}.{k}", out, seen)
        for c in cls.__mro__:
            sl = c.__dict__.get("__slots__", ())
            for s in (sl,) if isinstance(sl, str) else sl:
                if s not in ("__dict__", "__weakref__") and hasattr(v, s):
                    _snap(getattr(v, s), f"{path}.{s}", out, seen)
        _snap(cls, f"type({path})", out, seen)
        return
    if callable(v) and (cls.__module__ or "").split(".")[0] in (
            "numpy", "scipy", "builtins"):
        out[path] = ("callable", id(v))      # a ufunc, a C function object
        return
    if cls.__eq__ is not object.__eq__:
        try:
            out[path] = ("eq", copy.deepcopy(v))
            return
        except Exception:
            pass
    raise _Opaque(f"{path} ({cls.__name__}: no visible state and no __eq__)")


def _gate_snapshot(sf) -> dict:
    out: dict = {}
    _snap(sf, "switch_fn", out, set())
    return out


def _leaf_equal(a, b) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return False


def _gate_state_moves(sf, T: float) -> str:
    """Reason string when the gate's reachable state changes while it is
    evaluated over one period ``[0, T]`` (or cannot be compared), else ''."""
    try:
        before = _gate_snapshot(sf)
    except _Opaque as e:
        return f"uncomparable state at {e}"
    try:
        for k in range(257):
            sf(T * k / 256)
        nea = getattr(sf, "next_edge_after", None)
        if callable(nea):
            t, n = 0.0, 0
            while n < 10000:
                t2 = nea(t)
                if t2 is None or not (t2 > t) or t2 > T:
                    break
                sf(t2)
                t, n = t2, n + 1
    except Exception as e:  # a gate that cannot be probed is not measured
        return (f"its state could not be measured: evaluating it over one "
                f"period raised {type(e).__name__}: {e}")
    try:
        after = _gate_snapshot(sf)
    except _Opaque as e:
        return f"uncomparable state at {e}"
    moved = [k for k in after if k not in before
             or not _leaf_equal(before[k], after[k])]
    moved += [k for k in before if k not in after]
    if not moved:
        return ""
    moved.sort(key=lambda k: (k.count(".") + k.count("["), k))
    more = f" and {len(moved) - 1} more" if len(moved) > 1 else ""
    return (f"its state moved while it was evaluated over one period: "
            f"{moved[0]!r} changed{more}")


def _check_graph_blocks_inspectable(obs, *, analysis: str = _ANALYSIS) -> None:
    cxx = _cxx(obs)
    graph = getattr(obs, "_graph", None)
    uncaptured = list(cxx.uncaptured_blocks())
    if uncaptured:
        raise _unsup_hidden(
            f"control graph block {uncaptured[0]!r}",
            "state is not captured by CxxBlockChain.save_state()",
            analysis=analysis,
        )
    if graph is None:
        raise _unsup_hidden(
            "control graph",
            "observer carries a CxxBlockChain without Python block metadata",
            analysis=analysis,
        )
    for spec in getattr(graph, "blocks", ()):
        b = spec.block
        kind = type(b).__name__
        if kind == "CBlock":
            raise _unsup_hidden(
                f"block {spec.name!r} (CBlock)",
                "C block state is opaque to shooting",
                analysis=analysis,
            )
        if kind in _PY_LIBRARY_KINDS:
            raise _unsup_hidden(
                f"block {spec.name!r} ({kind})",
                "Python library block state is opaque to shooting",
                analysis=analysis,
            )
        if kind == "PythonBlock":
            wrapped = getattr(b, "_wrapped", b)
            fn = getattr(b, "fn", wrapped)
            reason = _hidden_state_reason(wrapped) or _hidden_state_reason(fn)
            if reason:
                raise _unsup_hidden(
                    f"PythonBlock {spec.name!r}",
                    reason,
                    analysis=analysis,
                )
            # A PythonBlock wrapping a callable object even with empty __dict__
            # or anything non-function is uninspectable unless pure function.
            if not isinstance(wrapped, (types.FunctionType, types.BuiltinFunctionType, partial)):
                raise _unsup_hidden(
                    f"PythonBlock {spec.name!r}",
                    f"callable object {type(wrapped).__name__} has no "
                    "inspectable state vector",
                    analysis=analysis,
                )
        elif kind in _NON_SMOOTH_KINDS:
            raise unsupported(
                analysis,
                "controller_state",
                f"{analysis}: control block {spec.name!r} ({kind}) holds "
                "non-differentiable discrete/mode state, which Newton "
                "shooting cannot solve.",
            )
        elif kind not in _STATELESS_KINDS and kind not in _STATE_SPECS:
            raise _unsup_hidden(
                f"block {spec.name!r} ({kind})",
                "unsupported block kind in periodic shooting",
                analysis=analysis,
            )


# ---------------------------------------------------------------------------
# State-slot probing and serialization layout
# ---------------------------------------------------------------------------

def _dummy_chain_for_spec(spec, overrides: dict[str, Any] | None = None):
    """Compile a minimal 1-block CxxBlockChain matching ``spec``."""
    from . import _pulsim as _k

    overrides = overrides or {}
    c = _k.CxxBlockChain()
    b = spec.block
    kind = type(b).__name__
    inputs = spec.inputs
    output = spec.output
    outs = [str(o) for o in output] if isinstance(output, (tuple, list)) else [str(output)]

    def cref(n=1):
        return [_k.CxxInputRef.const_value(0.0) for _ in range(n)]

    def add_ported(pkind, params, n_in, n_out=1):
        c.add_ported(
            pkind,
            {
                k: [float(e) for e in v]
                if isinstance(v, (list, tuple))
                else [float(v)]
                for k, v in params.items()
            },
            cref(n_in),
            [f"o{i}" for i in range(n_out)],
        )

    if kind == "Gain":
        c.add_gain(k=float(b.k), x=_k.CxxInputRef.const_value(0.0), output="o0")
    elif kind == "Subtract":
        c.add_subtract(
            a=_k.CxxInputRef.const_value(0.0),
            b=_k.CxxInputRef.const_value(0.0),
            output="o0",
        )
    elif kind == "Sum":
        srcs = (
            inputs["inputs"].value
            if "inputs" in inputs
            else [inputs[k] for k in ("a", "b", "c") if k in inputs]
        )
        add_ported("sum", {"weights": [float(v) for v in b.weights]}, len(srcs))
    elif kind == "PIController":
        c.add_pi_controller(
            Kp=float(b.Kp),
            Ki=float(b.Ki),
            output_min=float(b.output_min),
            output_max=float(b.output_max),
            setpoint=_k.CxxInputRef.const_value(0.0),
            measured=_k.CxxInputRef.const_value(0.0),
            dt=_k.CxxInputRef.const_value(0.0),
            output="o0",
            integrator_state=float(overrides.get("integrator_state", 0.0)),
            prev_error=float(overrides.get("_prev_error", 0.0)),
        )
    elif kind == "PIDController":
        add_ported(
            "pid",
            dict(
                Kp=b.Kp,
                Ki=b.Ki,
                Kd=b.Kd,
                alpha=b.derivative_alpha,
                output_min=b.output_min,
                output_max=b.output_max,
                integrator_state=overrides.get("integrator_state", 0.0),
                derivative_state=overrides.get("derivative_state", 0.0),
                prev_error=overrides.get("_prev_error", 0.0),
            ),
            3,
        )
    elif kind == "FirstOrderLowPass":
        add_ported(
            "first_order_low_pass",
            dict(tau=b.tau, state=overrides.get("state", 0.0)),
            2,
        )
    elif kind == "Integrator":
        add_ported(
            "integrator",
            dict(
                gain=b.gain,
                output_min=b.output_min,
                output_max=b.output_max,
                state=overrides.get("state", 0.0),
            ),
            2,
        )
    elif kind == "Differentiator":
        add_ported(
            "differentiator",
            dict(
                alpha=b.alpha,
                state=overrides.get("state", 0.0),
                prev_x=overrides.get("_prev_x", 0.0),
            ),
            2,
        )
    elif kind == "Limiter":
        c.add_limiter(
            min_v=float(b.output_min),
            max_v=float(b.output_max),
            x=_k.CxxInputRef.const_value(0.0),
            output="o0",
        )
    elif kind == "MovingAverageFilter":
        add_ported("ema", dict(alpha=b.alpha, state=overrides.get("state", 0.0)), 1)
    elif kind == "RateLimiter":
        add_ported(
            "rate_limiter",
            dict(
                slew_up=b.slew_up,
                slew_down=b.slew_down,
                state=overrides.get("state", 0.0),
            ),
            2,
        )
    elif kind == "TransferFunction":
        nx = len(b.num)
        ny = len(b.den)
        xh = list(overrides.get("xh", [0.0] * nx))
        yh = list(overrides.get("yh", [0.0] * ny))
        add_ported(
            "transfer_function",
            {"num": list(b.num), "den": list(b.den), "x_hist": xh, "y_hist": yh},
            1,
        )
    elif kind == "OpAmp":
        add_ported("op_amp", dict(gain=b.gain, rail_min=b.rail_min, rail_max=b.rail_max), 2)
    elif kind == "LookupTable1D":
        add_ported("lookup_table_1d", {"xs": list(b.xs), "ys": list(b.ys)}, 1)
    elif kind == "MathBlock":
        ops = ("add", "sub", "mul", "div", "abs", "neg", "sqrt", "pow2")
        add_ported("math", dict(op=ops.index(str(b.op).lower())), 2)
    elif kind == "SignalMux":
        n_in = 1 + len(inputs["inputs"].value)
        add_ported("signal_mux", {}, n_in)
    elif kind == "SignalDemux":
        add_ported("signal_demux", dict(n_outputs=int(b.n_outputs)), 1, n_out=int(b.n_outputs))
    elif kind == "PwmGenerator":
        add_ported("pwm", dict(frequency=b.frequency, phase=b.phase), 2)
    elif kind == "SpaceVectorModulator":
        add_ported("svm", dict(v_dc=b.v_dc), 2, n_out=3)
    elif kind == "ClarkeTransform":
        add_ported("clarke", {}, 3, n_out=3)
    elif kind == "InverseClarkeTransform":
        add_ported("inverse_clarke", {}, 3, n_out=3)
    elif kind == "ParkTransform":
        add_ported("park", {}, 3, n_out=2)
    elif kind == "InverseParkTransform":
        add_ported("inverse_park", {}, 3, n_out=2)
    elif kind == "PLL":
        add_ported(
            "pll",
            dict(
                f_nominal=b.f_nominal,
                Kp=b.Kp,
                Ki=b.Ki,
                theta_hat=overrides.get("theta_hat", 0.0),
                omega_hat=overrides.get("omega_hat", 0.0),
                integ=overrides.get("_integ", 0.0),
            ),
            3,
            n_out=len(outs),
        )
    elif kind == "DelayBlock":
        c.add_delay(
            samples=int(b.samples),
            initial_value=float(overrides.get("initial_value", b.initial_value)),
            x=_k.CxxInputRef.const_value(0.0),
            output="o0",
        )
        c.reset()
    elif kind == "PythonBlock":
        c.add_python_block(lambda *a: 0.0, [], ["o0"], with_state=False)
    else:
        from . import _options_error
        raise _options_error("kind", f"unsupported control block kind {kind!r}")
    c.name_last("b", kind)
    c.compile()
    return c


# The CxxBlockChain::dump_state layout this module parses (chain.hpp
# serialize()). The last 4 bytes of the magic are its layout version: a
# C++ change to the layout must bump it, and then every read here is
# refused by name instead of mis-seeding x_ctrl silently.
_STATE_MAGIC = 0x5053474300000001
_COUNTERS_BYTES = 24        # n_evals, n_block_evals, n_calls (3 x u64)
_GROUP_REC_BYTES = 33       # k, n_ticks (u64), pending_at (f64), pending
#                             (bool), jitter_max (f64)


def _layout(ok: bool, what: str) -> None:
    """A named refusal (not ``assert``, which ``python -O`` removes) when
    the bytes do not have the layout this module parses."""
    if not ok:
        raise unsupported(
            _ANALYSIS, "controller_state_layout",
            f"{_ANALYSIS}: the control graph's saved state does not have "
            f"the byte layout this pulsim's Python side parses ({what}); "
            "the C++ extension and the Python package are from different "
            "builds. Rebuild / reinstall them together.")


_STATE_SPECS: dict[str, tuple[str, ...]] = {
    "PIController": ("integrator_state", "_prev_error"),
    "PIDController": ("integrator_state", "derivative_state", "_prev_error"),
    "FirstOrderLowPass": ("state",),
    "Integrator": ("state",),
    "Differentiator": ("state", "_prev_x"),
    "MovingAverageFilter": ("state",),
    "RateLimiter": ("state",),
    "PLL": ("theta_hat", "omega_hat", "_integ"),
    "DelayBlock": ("buf",),
    "TransferFunction": ("xh", "yh"),
}

_EMPTY_CHAIN_LEN: int | None = None


def _empty_chain_len() -> int:
    global _EMPTY_CHAIN_LEN
    if _EMPTY_CHAIN_LEN is None:
        from . import _pulsim as _k

        c = _k.CxxBlockChain()
        c.compile()
        raw = c.dump_state(c.save_state())
        _check_magic(raw)
        _EMPTY_CHAIN_LEN = len(raw)
    return _EMPTY_CHAIN_LEN


def _check_magic(raw: bytes) -> None:
    _layout(len(raw) >= 16, f"{len(raw)} bytes, shorter than the header")
    magic = struct.unpack_from("<Q", raw, 0)[0]
    _layout(magic == _STATE_MAGIC,
            f"layout tag {magic:#018x}, expected {_STATE_MAGIC:#018x}")


def _dummy_dump_bytes(spec, overrides: dict[str, Any] | None = None) -> tuple[bytes, int]:
    """Return ``(block_dump_bytes, n_sig)`` for a 1-block dummy chain."""
    c = _dummy_chain_for_spec(spec, overrides)
    raw = c.dump_state(c.save_state())
    kind = type(spec.block).__name__
    outs = (
        [str(o) for o in spec.output]
        if isinstance(spec.output, (tuple, list))
        else [str(spec.output)]
    )
    if kind in ("SpaceVectorModulator", "ClarkeTransform", "InverseClarkeTransform"):
        n_sig = 3
    elif kind in ("ParkTransform", "InverseParkTransform"):
        n_sig = 2
    elif kind == "PLL":
        n_sig = len(outs)
    elif kind == "SignalDemux":
        n_sig = int(spec.block.n_outputs)
    else:
        n_sig = 1
    hdr = 16 + 8 + 1  # magic + n_blocks + name_len("b") + "b"
    tail_len = (_empty_chain_len() - 16) + 16 * n_sig
    return raw[hdr : len(raw) - tail_len], n_sig


def _find_double_offset(b0: bytes, b1: bytes) -> int:
    diff = [i for i in range(len(b0)) if b0[i] != b1[i]]
    if not diff:
        raise RuntimeError("failed to locate controller state byte offset")
    off = diff[0]
    if off + 8 > len(b0):
        raise RuntimeError("controller state byte offset out of range")
    return off


def _probe_block_schema(spec) -> tuple[int, list[tuple[str, int, float]]]:
    """Return ``(block_dump_len, [(field_name, offset_in_dump, init_val), ...])``."""
    b = spec.block
    kind = type(b).__name__
    d0, _ = _dummy_dump_bytes(spec)
    dump_len = len(d0)
    if kind in _STATELESS_KINDS or kind == "PythonBlock":
        return dump_len, []

    if kind == "DelayBlock":
        samples = int(b.samples)
        init_v = float(b.initial_value)
        # In DelayBlock, buf (k doubles) sits at the end of block_dump
        # (preceded by uint64 size == samples). Verify by sentinel probe.
        d1, _ = _dummy_dump_bytes(spec, {"initial_value": _SENTINEL})
        diff = [i for i in range(dump_len) if d0[i] != d1[i]]
        # Last samples*8 bytes are buf[0..samples-1]
        buf_start = dump_len - 8 * samples
        _layout(buf_start >= 8 and bool(diff) and diff[-1] < dump_len,
                f"DelayBlock dump of {dump_len} bytes for {samples} samples")
        count = struct.unpack_from("<Q", d1, buf_start - 8)[0]
        _layout(count == samples,
                f"DelayBlock buffer length {count}, expected {samples}")
        fields = [(f"buf[{k}]", buf_start + 8 * k, init_v) for k in range(samples)]
        return dump_len, fields

    if kind == "TransferFunction":
        nx = len(b.num)
        ny = len(b.den)
        fields: list[tuple[str, int, float]] = []
        base_xh = list(getattr(b, "_x_hist", []) or [])
        base_yh = list(getattr(b, "_y_hist", []) or [])
        for i in range(nx):
            xh = [0.0] * nx
            xh[i] = _SENTINEL
            d1, _ = _dummy_dump_bytes(spec, {"xh": xh, "yh": [0.0] * ny})
            off = _find_double_offset(d0, d1)
            val = float(base_xh[i]) if i < len(base_xh) else 0.0
            fields.append((f"x_hist[{i}]", off, val))
        for i in range(ny):
            yh = [0.0] * ny
            yh[i] = _SENTINEL
            d1, _ = _dummy_dump_bytes(spec, {"xh": [0.0] * nx, "yh": yh})
            off = _find_double_offset(d0, d1)
            val = float(base_yh[i]) if i < len(base_yh) else 0.0
            fields.append((f"y_hist[{i}]", off, val))
        return dump_len, fields

    attrs = _STATE_SPECS[kind]
    fields = []
    for attr in attrs:
        d1, _ = _dummy_dump_bytes(spec, {attr: _SENTINEL})
        off = _find_double_offset(d0, d1)
        fields.append((attr, off, float(getattr(b, attr))))
    return dump_len, fields


def _count(raw: bytes, pos: int) -> int:
    """A u64 element count at ``pos``, checked against the buffer (a wrong
    offset reads garbage; it must not size an allocation)."""
    _layout(pos + 8 <= len(raw), f"count at byte {pos} past the end ({len(raw)})")
    n = struct.unpack_from("<Q", raw, pos)[0]
    _layout(n <= len(raw), f"count {n} at byte {pos} exceeds the {len(raw)} bytes")
    return n


def _skip_tail_to_pub(raw: bytes, sig_count_pos: int) -> tuple[int, int, int]:
    """From ``sig_count_pos``, return ``(sig_data_pos, pub_data_pos, end_pos)``."""
    pos = sig_count_pos
    n_sig = _count(raw, pos)
    sig_data_pos = pos + 8
    pos = sig_data_pos + 8 * n_sig
    n_bk = _count(raw, pos)
    pos += 8 + 8 * n_bk
    n_bv = _count(raw, pos)
    pos += 8 + 8 * n_bv
    n_rec = _count(raw, pos)
    pos += 8
    for _ in range(n_rec):
        cnt = _count(raw, pos)
        pos += 8 + 8 * cnt
    n_rt = _count(raw, pos)
    pos += 8 + 8 * n_rt
    pos += _COUNTERS_BYTES
    n_grp = _count(raw, pos)
    pos += 8 + _GROUP_REC_BYTES * n_grp
    n_pub = _count(raw, pos)
    pub_data_pos = pos + 8
    pos = pub_data_pos + 8 * n_pub
    return sig_data_pos, pub_data_pos, pos


def _clean_template_bytes(cxx, chain) -> tuple[bytes, int, list[tuple[str, int, float]]]:
    """Build a clean ``dump_state`` template (zero ticks/calls/recordings)
    and return ``(template_bytes, sig_count_pos, block_state_slots)``."""
    try:
        return _parse_template(cxx, chain)
    except struct.error as e:
        _layout(False, f"the bytes end inside a field: {e}")
        raise


def _parse_template(cxx, chain):
    raw = cxx.dump_state(cxx.save_state())
    _check_magic(raw)
    n_blocks = struct.unpack_from("<Q", raw, 8)[0]
    _layout(n_blocks == len(chain.blocks),
            f"{n_blocks} blocks in the state, {len(chain.blocks)} in the graph")
    pos = 16  # magic(8) + n_blocks(8)
    out = bytearray(raw[:pos])
    slots: list[tuple[str, int, float]] = []

    for spec in chain.blocks:
        name_len = _count(raw, pos)
        name_bytes = raw[pos : pos + 8 + name_len]
        _layout(name_len < 4096 and name_bytes[8:].decode("utf-8", "replace")
                == str(spec.name),
                f"block name at byte {pos} is not {spec.name!r}")
        out.extend(name_bytes)
        pos += 8 + name_len

        kind = type(spec.block).__name__
        expected_len, block_fields = _probe_block_schema(spec)
        block_start_in_out = len(out)

        if kind == "DelayBlock":
            samples = int(spec.block.samples)
            init_v = float(spec.block.initial_value)
            # Check actual buf size in raw (may be 0 before first tick)
            # Format: size_t samples (8) + double initial_value (8) + uint64 buf_len (8) + buf
            buf_len = struct.unpack_from("<Q", raw, pos + 16)[0]
            raw_block_len = 24 + 8 * buf_len
            if buf_len == samples:
                out.extend(raw[pos : pos + raw_block_len])
            else:
                out.extend(raw[pos : pos + 16])
                out.extend(struct.pack("<Q", samples))
                out.extend(struct.pack(f"<{samples}d", *([init_v] * samples)))
            pos += raw_block_len
        elif kind == "TransferFunction":
            nx = len(spec.block.num)
            ny = len(spec.block.den)
            block_bytes = bytearray(raw[pos : pos + expected_len])
            # Ensure nx == len(num) and ny == len(den) in the template
            # Last 8 bytes of PyTransferFunction dump are int nx, int ny.
            struct.pack_into("<ii", block_bytes, expected_len - 8, nx, ny)
            out.extend(block_bytes)
            pos += expected_len
        else:
            out.extend(raw[pos : pos + expected_len])
            pos += expected_len

        for fname, rel_off, init_val in block_fields:
            slots.append((f"ctrl({spec.name}.{fname})", block_start_in_out + rel_off, init_val))

    sig_count_pos_in_raw = pos
    sig_count_pos_in_out = len(out)
    # Parse tail of raw to copy sig, b_extra, zeroed recordings, zeroed groups, pub, zeroed horizon
    n_sig = _count(raw, pos)
    out.extend(raw[pos : pos + 8 + 8 * n_sig])
    pos += 8 + 8 * n_sig

    n_bk = _count(raw, pos)
    out.extend(raw[pos : pos + 8 + 8 * n_bk])
    pos += 8 + 8 * n_bk
    n_bv = _count(raw, pos)
    out.extend(raw[pos : pos + 8 + 8 * n_bv])
    pos += 8 + 8 * n_bv

    # Clear recordings & recording_times while preserving recording channel count
    n_rec = _count(raw, pos)
    pos += 8
    out.extend(struct.pack("<Q", n_rec))
    for _ in range(n_rec):
        cnt = _count(raw, pos)
        pos += 8 + 8 * cnt
        out.extend(struct.pack("<Q", 0))
    n_rt = _count(raw, pos)
    pos += 8 + 8 * n_rt
    out.extend(struct.pack("<Q", 0))

    # Zero n_evals, n_block_evals, n_calls
    pos += _COUNTERS_BYTES
    out.extend(struct.pack("<QQQ", 0, 0, 0))

    # Zero each group's runtime counters (k, n_ticks, pending_at, pending, jitter_max)
    n_grp = _count(raw, pos)
    pos += 8
    out.extend(struct.pack("<Q", n_grp))
    for _ in range(n_grp):
        out.extend(struct.pack("<QQd?d", 0, 0, -math.inf, False, 0.0))
        pos += _GROUP_REC_BYTES

    # Copy pub_
    n_pub = _count(raw, pos)
    out.extend(raw[pos : pos + 8 + 8 * n_pub])
    pos += 8 + 8 * n_pub

    # last_t, t0, solver_dt, queued, horizon_t, has_horizon_x, hx
    last_t, t0, solver_dt = struct.unpack_from("<ddd", raw, pos)
    pos += 24
    out.extend(struct.pack("<ddd", 0.0, 0.0, solver_dt))
    n_q = _count(raw, pos)
    pos += 8 + n_q
    out.extend(struct.pack("<Q", n_q))
    out.extend(b"\x00" * n_q)
    pos += 8 + 1  # horizon_t, has_horizon_x
    n_hx = _count(raw, pos)
    pos += 8 + 8 * n_hx
    out.extend(struct.pack("<d?Q", 0.0, False, 0))
    _layout(pos == len(raw), f"parsed {pos} of {len(raw)} bytes")
    _ = sig_count_pos_in_raw, last_t, t0

    return bytes(out), sig_count_pos_in_out, slots


def _divides(P: float, T: float) -> bool:
    if not (P > 0.0 and math.isfinite(P)):
        return False
    k = round(T / P)
    return k >= 1 and abs(T / P - k) <= 1e-12 * max(1.0, T / P)


class _SingleGraphState:
    """State mapper for one compiled ``CxxBlockChain``."""

    def __init__(self, obs, period: float):
        self.obs = obs
        self.cxx = _cxx(obs)
        self.graph = obs._graph
        self.orig_state = self.cxx.save_state()

        # Periodicity check on all discrete rate groups
        n_groups = int(self.cxx.tick_counts().__len__())
        for gi in range(1, n_groups):
            p_grp = float(self.cxx.group_period(gi))
            if not _divides(p_grp, period):
                raise unsupported(
                    _ANALYSIS,
                    "period",
                    f"{_ANALYSIS}: control graph rate period {p_grp!r} s "
                    f"does not divide shooting period={period!r} s.",
                )

        template, sig_count_pos, block_slots = _clean_template_bytes(
            self.cxx, self.graph
        )
        self._template = template
        self._sig_count_pos = sig_count_pos
        sig_data_pos, pub_data_pos, _ = _skip_tail_to_pub(template, sig_count_pos)

        # If any rate group has compute_delay != 0 or offset > 0, its
        # published output channel values carry across the period boundary.
        delayed_slots = self._delayed_channel_slots(sig_data_pos, pub_data_pos)
        self.slots: list[tuple[str, int, float]] = [*block_slots, *delayed_slots]
        # Verify template round-trips cleanly through load_state
        self.cxx.load_state(self._template)

    def _delayed_channel_slots(
        self, sig_data_pos: int, pub_data_pos: int
    ) -> list[tuple[str, int, float]]:
        groups = getattr(self.graph, "_groups", {}) or {}
        if not groups:
            return []
        inv_group = {gid: rate for rate, gid in groups.items()}
        block_rates = getattr(self.graph, "_block_rates", None) or []
        slot_names = list(self.graph.blocks)
        _ = slot_names
        # Map channel name -> slot index via chain's sink/channel names
        # We can find the channel slot index by probing set_channel on template?
        # Even simpler: CxxBlockChain doesn't expose slot_names, so only add
        # delayed channel slots when offset > 0 or compute_delay != 0!
        delayed_channels: list[tuple[str, str]] = []
        for idx, spec in enumerate(self.graph.blocks):
            rate = block_rates[idx] if idx < len(block_rates) else spec.rate
            if rate is None:
                continue
            delay_kind, _ = rate._delay()
            if delay_kind == 2:  # one_sample: sig[s] at t=0- becomes pub[s] at t=0
                outs = (
                    list(spec.output)
                    if isinstance(spec.output, (tuple, list))
                    else [spec.output]
                )
                for ch in outs:
                    delayed_channels.append((ch, "sig"))
            elif float(rate.offset) > 0.0 or delay_kind == 1:
                outs = (
                    list(spec.output)
                    if isinstance(spec.output, (tuple, list))
                    else [spec.output]
                )
                for ch in outs:
                    delayed_channels.append((ch, "pub"))
        _ = inv_group
        if not delayed_channels:
            return []

        # Probe slot index of each delayed channel using save/set_channel/dump
        live = self.cxx.save_state()
        out_slots: list[tuple[str, int, float]] = []
        try:
            for ch, which in delayed_channels:
                self.cxx.restore_state(self.cxx.load_state(self._template))
                b0 = self.cxx.dump_state(self.cxx.save_state())
                self.cxx.set_output(ch, _SENTINEL)
                b1 = self.cxx.dump_state(self.cxx.save_state())
                s0, p0, _ = _skip_tail_to_pub(b0, self._sig_count_pos)
                n_sig = struct.unpack_from("<Q", b0, self._sig_count_pos)[0]
                slot_idx = None
                for k in range(n_sig):
                    if b0[s0 + 8 * k : s0 + 8 * (k + 1)] != b1[s0 + 8 * k : s0 + 8 * (k + 1)]:
                        slot_idx = k
                        break
                if slot_idx is None:
                    continue
                if which == "sig":
                    off = sig_data_pos + 8 * slot_idx
                    init_v = struct.unpack_from("<d", self._template, off)[0]
                    out_slots.append((f"ctrl(sig:{ch})", off, init_v))
                else:
                    off = pub_data_pos + 8 * slot_idx
                    init_v = struct.unpack_from("<d", self._template, off)[0]
                    out_slots.append((f"ctrl(pub:{ch})", off, init_v))
        finally:
            self.cxx.restore_state(live)
        return out_slots

    @property
    def n_ctrl(self) -> int:
        return len(self.slots)

    def initial_values(self) -> np.ndarray:
        return np.array([v for _, _, v in self.slots], dtype=float)

    def seed(self, values: np.ndarray) -> None:
        buf = bytearray(self._template)
        for (_, off, _), val in zip(self.slots, values):
            struct.pack_into("<d", buf, off, float(val))
        self.cxx.restore_state(self.cxx.load_state(bytes(buf)))

    def read(self) -> np.ndarray:
        raw = self.cxx.dump_state(self.cxx.save_state())
        _s_pos, p_pos, _ = _skip_tail_to_pub(raw, self._sig_count_pos)
        t_pub_start = _skip_tail_to_pub(self._template, self._sig_count_pos)[1]
        delta_pub = p_pos - t_pub_start
        vals = []
        for _, off, _ in self.slots:
            actual_off = off + delta_pub if off >= t_pub_start else off
            vals.append(struct.unpack_from("<d", raw, actual_off)[0])
        return np.array(vals, dtype=float)

    def restore_original(self) -> None:
        self.cxx.restore_state(self.orig_state)


class ControlGraphStateMapper:
    """Flat discrete state vector ``x_ctrl`` over all active control graphs."""

    def __init__(self, graphs: Sequence, period: float):
        self.graphs = [_SingleGraphState(g, float(period)) for g in graphs]
        self.labels = [lbl for g in self.graphs for lbl, _, _ in g.slots]
        self.devices = [
            [lbl[5:].split(".")[0]] for g in self.graphs for lbl, _, _ in g.slots
        ]
        self.n_ctrl = len(self.labels)
        self._slices: list[tuple[_SingleGraphState, slice]] = []
        i = 0
        for g in self.graphs:
            k = g.n_ctrl
            self._slices.append((g, slice(i, i + k)))
            i += k

    def initial_state(self) -> np.ndarray:
        if self.n_ctrl == 0:
            return np.empty(0, dtype=float)
        return np.concatenate([g.initial_values() for g in self.graphs])

    def seed(self, x_ctrl: np.ndarray) -> None:
        x_ctrl = np.asarray(x_ctrl, dtype=float)
        for g, sl in self._slices:
            g.seed(x_ctrl[sl])

    def read(self) -> np.ndarray:
        if self.n_ctrl == 0:
            return np.empty(0, dtype=float)
        return np.concatenate([g.read() for g in self.graphs])

    def restore_original(self) -> None:
        for g in self.graphs:
            g.restore_original()


def active_graph_ids(plan) -> set[int]:
    return {id(_cxx(g)) for g in getattr(plan, "graphs", ()) if _cxx(g) is not None}


def active_sink_sources(plan) -> set[str]:
    names: set[str] = set()
    for g in getattr(plan, "graphs", ()):
        gr = getattr(g, "_graph", None)
        for _ch, _wire, name in getattr(gr, "_sinks", ()) or ():
            names.add(name)
    return names


def gate_graph_ids(sf) -> set[int]:
    """Return ``id(cxx_chain)`` for graphs owning ``sf`` (if identifiable)."""
    g = getattr(sf, "_graph", None)
    cxx = getattr(g, "_cxx_chain", None) if g is not None else None
    if cxx is not None:
        return {id(cxx)}
    return set()
