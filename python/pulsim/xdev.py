"""Equations to C: external devices declared with SymPy (add-external-device-abi).

    import sympy as sp
    from pulsim import xdev

    a, k = xdev.pins("a", "k")
    I_S, n, V_T = xdev.params(I_S=(1e-12, "A", 1e-30, 1.0), n=(1.0, ""),
                              V_T=(0.025852, "V"))
    i = I_S * (sp.exp((a - k) / (n * V_T)) - 1)
    dev = xdev.compile("my_diode", pins=[a, k], f=[i, -i])
    dev.add_to(builder, "D1", "anode", "cathode", I_S=2e-12)

``compile`` differentiates the law with SymPy, takes the structurally
non-zero derivatives as the fixed Jacobian pattern, shares common
sub-expressions, writes plain C99 against ``pulsim_xdev.h`` (``exp(u) - 1``
and ``log(1 + u)`` become ``expm1`` / ``log1p``), compiles it through the
per-user native cache and loads it with ``CircuitBuilder.add_external_device``
(whose registration checks then run on the generated model like on any other).

The law follows the device interface's row convention: ``f`` and ``q`` have
one entry per pin (the current / charge leaving the node into the device)
and then one per internal unknown (its own equation); each row is
``f + d/dt q = 0``.

SMOOTHNESS LINT. ``Piecewise``, ``Abs``, ``sign``, ``Max``, ``Min``,
``Heaviside``, ``floor`` (and ``ceiling``, ``frac``, ``Mod``) of an unknown are
refused with the offending sub-expression, unless ``allow_nonsmooth=True``;
then the model does not claim SMOOTH and its Jacobian is the derivative of
the branch the value takes (one-sided, which the registration check accepts
for a model without the claim). A two-piece ``Piecewise`` whose condition is
one relational solvable for one unknown, and whose pieces agree in value and
gradient on that boundary (proved symbolically), is C1: it is accepted and
the model keeps SMOOTH. A condition on parameters / states only is no kink in
the unknowns and is always accepted. Anything else is refused (the proof is
conservative: a C1 law it cannot prove needs ``allow_nonsmooth=True``).

TRUST. The generated code is native code compiled and loaded into this
process. Names that reach the C (model, pins, parameters, states, outputs)
must match ``[A-Za-z][A-Za-z0-9_]*`` and not be C keywords; units are a short
plain-ASCII set; expressions reach the C only through SymPy's C99 printer
from SymPy objects. The compiler runs with an argument list (no shell) and
the cache keeps ``_native_cache``'s rules (per-user directory, 0700, a cached
library someone else owns or can write is rebuilt).

Needs SymPy (``pip install pulsim[codegen]``) and a C compiler for
``compile``; ``load``, ``scan_monotone`` and ``add_external_device`` do not
need SymPy.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import _native_cache
from ._pulsim import PulsimError  # type: ignore[import-not-found]


class XdevError(ValueError, PulsimError):
    """A law, a name or a model that pulsim.xdev refuses (a ValueError)."""


__all__ = ["XdevError", "pins", "internal", "params", "state", "compile", "load", "Device",
           "scan_monotone", "MonotoneScan", "add_external_device"]

_SMOOTH, _ABI = 1, 1

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,62}\Z")
_UNIT = re.compile(r"[A-Za-z0-9 _./*^()%+-]{0,32}\Z")
_C_KEYWORDS = frozenset("""
    auto break case char const continue default do double else enum extern
    float for goto if inline int long register restrict return short signed
    sizeof static struct switch typedef union unsigned void volatile while
    alignas alignof bool constexpr false nullptr static_assert thread_local
    true typeof typeof_unqual near far asm fortran""".split())


def _check_name(what: str, name: Any) -> str:
    if not isinstance(name, str) or not _NAME.match(name) or name in _C_KEYWORDS:
        raise XdevError(
            f"xdev: {what} name {name!r} is refused: names that reach the generated C "
            "must match [A-Za-z][A-Za-z0-9_]* (at most 63 characters) and not be a "
            "C keyword")
    return name


def _check_unit(name: str, unit: Any) -> str:
    if not isinstance(unit, str) or not _UNIT.match(unit):
        raise XdevError(
            f"xdev: unit {unit!r} of {name!r} is refused: units are at most 32 "
            "characters of letters, digits, space and _ . / * ^ ( ) % + -")
    return unit


# ---------------------------------------------------------------------------
# SymPy, imported only when a symbolic call needs it
# ---------------------------------------------------------------------------
_NS: dict = {}


def _sp():
    """SymPy and the handle class, built once."""
    if _NS:
        return _NS
    try:
        import sympy as sp
    except ImportError:
        raise ImportError(
            "pulsim.xdev needs SymPy to turn equations into C: "
            "pip install pulsim[codegen]") from None

    class Var(sp.Dummy):
        """A pin / internal unknown / parameter / state of an xdev model.
        A Dummy (unique per declaration), never printed into the C."""

        def __new__(cls, name, kind, meta=None, **assumptions):
            obj = super().__new__(cls, name, real=True, **assumptions)
            obj.xkind = kind
            obj.meta = meta
            return obj

    _NS.update(sp=sp, Var=Var, h=Var("h", "h", positive=True))
    return _NS


def _many(names, kind, what):
    Var = _sp()["Var"]
    out = tuple(Var(_check_name(what, n), kind) for n in names)
    return out[0] if len(out) == 1 else out


def pins(*names: str):
    """The pins' node voltages, in the order the device's nodes are given."""
    return _many(names, "x", "pin")


def internal(*names: str):
    """Internal unknowns (one extra row each in ``f`` and ``q``)."""
    return _many(names, "x", "internal unknown")


def _finite(v: float) -> float:
    # DBL_MAX for an infinite bound: no inf / HUGE_VAL in the C initialisers
    v = float(v)
    if math.isnan(v):
        raise XdevError("xdev: a parameter default / bound is NaN")
    return max(-sys.float_info.max, min(sys.float_info.max, v))


def params(**decl):
    """Parameters, ``name=default`` | ``(default, unit)`` |
    ``(default, unit, lo, hi)``; the host refuses values outside ``[lo, hi]``
    (default: unbounded). Field order is the declaration order."""
    Var = _sp()["Var"]
    out = []
    for name, d in decl.items():
        _check_name("parameter", name)
        d = d if isinstance(d, tuple) else (d,)
        if len(d) not in (1, 2, 4):
            raise XdevError(f"xdev: parameter {name!r}: give default, (default, unit) "
                             "or (default, unit, lo, hi)")
        default = _finite(d[0])
        unit = _check_unit(name, d[1] if len(d) > 1 else "")
        lo, hi = (_finite(d[2]), _finite(d[3])) if len(d) == 4 \
            else (-sys.float_info.max, sys.float_info.max)
        if not lo <= default <= hi:
            raise XdevError(f"xdev: parameter {name!r}: default {default} outside "
                             f"[{lo}, {hi}]")
        assume = {"positive": True} if lo > 0 else {}
        out.append(Var(name, "p", (default, unit, lo, hi), **assume))
    return out[0] if len(out) == 1 else tuple(out)


def state(name: str, init=0.0):
    """A committed model state (``ctx->state``): ``init`` (an expression of
    the unknowns and parameters) at the start, then the ``update`` expression
    given to ``compile`` after each accepted step (it may use ``xdev.h``, the
    step length, and the committed states). The law may read it; it is not
    an unknown (no Jacobian column)."""
    return _sp()["Var"](_check_name("state", name), "s", init)


def __getattr__(name):
    if name == "h":                 # the accepted step's length, for `update`
        return _sp()["h"]
    raise AttributeError(f"module 'pulsim.xdev' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# The smoothness lint
# ---------------------------------------------------------------------------
def _nonsmooth_types(sp):
    return (sp.Abs, sp.sign, sp.Max, sp.Min, sp.Heaviside, sp.floor, sp.ceiling,
            sp.frac, sp.Mod, sp.DiracDelta)


def _proves_c1(sp, pw, unknowns) -> bool:
    """A two-piece Piecewise((e1, rel), (e2, True)) is C1 across its
    boundary when rel's lhs - rhs solves for one unknown s and e1 - e2 and
    its gradient vanish there."""
    if len(pw.args) != 2 or pw.args[1].cond is not sp.true:
        return False
    (e1, cond), (e2, _) = pw.args
    if not isinstance(cond, (sp.StrictGreaterThan, sp.StrictLessThan,
                             sp.GreaterThan, sp.LessThan)):
        return False
    g = cond.lhs - cond.rhs
    d = e1 - e2
    for s in sorted(g.free_symbols & unknowns, key=lambda v: v.name):  # names are unique
        try:
            sols = sp.solve(g, s)
        except (NotImplementedError, ValueError, TypeError):
            continue
        # affine in s: exactly one boundary point per line (solve() returns
        # ONE root of tan(u) and would "prove" a jump at u = pi)
        slope = sp.diff(g, s)
        if slope == 0 or s in slope.free_symbols:
            continue
        if len(sols) != 1 or s in sols[0].free_symbols:
            continue
        checks = [d] + [sp.diff(d, y) for y in unknowns]
        try:
            return all(sp.simplify(c.subs(s, sols[0])) == 0 for c in checks)
        except (NotImplementedError, ValueError, TypeError):
            return False
    return False


def _lint(sp, expr, unknowns) -> list:
    """The sub-expressions that are not C1 in the unknowns."""
    bad = []
    nonsmooth = _nonsmooth_types(sp)

    def walk(e):
        if not e.free_symbols & unknowns:
            return                  # a function of parameters / states only
        if isinstance(e, sp.Piecewise):
            for piece, _ in e.args:   # conditions are not differentiated
                walk(piece)
            conds_x = any(c.free_symbols & unknowns for _, c in e.args)
            if conds_x and not _proves_c1(sp, e, unknowns):
                bad.append(e)
            return
        if isinstance(e, nonsmooth):
            bad.append(e)
            return
        for a in e.args:
            walk(a)

    walk(expr)
    return bad


def _to_piecewise(sp, expr):
    """Non-smooth functions as Piecewise, so their derivative is the derivative
    of the branch the value takes (a one-sided Jacobian, consistent)."""
    def two_piece(e):
        z = e.args[0]
        if isinstance(e, sp.sign):        # the right-hand value at 0 (SymPy's is 0)
            return sp.Piecewise((-1, z < 0), (1, True))
        if isinstance(e, sp.Heaviside):   # likewise (SymPy's H(0) is 1/2)
            return sp.Piecewise((0, z < 0), (1, True))
        return e.rewrite(sp.Piecewise)

    expr = expr.replace(lambda e: isinstance(e, sp.frac),
                        lambda e: e.args[0] - sp.floor(e.args[0]))
    expr = expr.replace(lambda e: isinstance(e, sp.Mod),
                        lambda e: e.args[0] - e.args[1] * sp.floor(e.args[0] / e.args[1]))
    kinds = (sp.Abs, sp.sign, sp.Max, sp.Min, sp.Heaviside)
    return expr.replace(lambda e: isinstance(e, kinds), two_piece)


def _diff(sp, expr, x):
    d = sp.diff(expr, x)
    # floor / ceiling are piecewise constant: derivative 0 away from the jump
    # (frac and Mod were rewritten through floor before this)
    d = d.replace(lambda e: isinstance(e, sp.Derivative)
                  and isinstance(e.expr, (sp.floor, sp.ceiling)), lambda e: sp.S.Zero)
    d = d.replace(lambda e: isinstance(e, sp.Subs), lambda e: e.doit())  # chain rule wrapper
    if d.has(sp.Derivative, sp.Subs, sp.DiracDelta):
        raise XdevError(f"xdev: cannot differentiate {expr} by {x} in closed form")
    return d


# ---------------------------------------------------------------------------
# C emission
# ---------------------------------------------------------------------------
def _printer(sp):
    from sympy.printing.c import C99CodePrinter

    class Printer(C99CodePrinter):
        # numbers only: no M_PI / M_E (not C99), no integer literal overflow
        def _print_NumberSymbol(self, e):
            return repr(float(e))

        _print_Pi = _print_Exp1 = _print_GoldenRatio = _print_EulerGamma = \
            _print_Catalan = _print_NumberSymbol

        def _print_Integer(self, e):
            n = int(e)
            return str(n) if abs(n) < 2 ** 31 else repr(float(n))

        def _print_Infinity(self, e):
            raise XdevError("xdev: the law evaluates to an infinity symbolically")

        _print_NegativeInfinity = _print_ComplexInfinity = _print_NaN = _print_Infinity

        def _print_Subs(self, e):       # StrPrinter would print them as text
            raise XdevError(f"xdev: unevaluated calculus cannot be emitted: {e}")

        _print_Derivative = _print_Integral = _print_Subs

        def _print_ImaginaryUnit(self, e):
            raise XdevError("xdev: the law is complex-valued (I appears); a device law is real")

        # SymPy rewrites coth as (e^x + e^-x)/(e^x - e^-x): a cancellation of
        # its own near 0. The reciprocals of the C functions are accurate.
        def _print_coth(self, e):
            return f"(1.0/tanh({self._print(e.args[0])}))"

        def _print_csch(self, e):
            return f"(1.0/sinh({self._print(e.args[0])}))"

        def _print_sech(self, e):
            return f"(1.0/cosh({self._print(e.args[0])}))"

        def _print_Piecewise(self, e):
            # one line of nested ternaries; the value and its derivative take
            # the same branch (the Jacobian is the branch's own derivative)
            if e.args[-1].cond is not sp.true:
                raise XdevError(f"xdev: {e} has no otherwise (True) branch")
            out = self._print(e.args[-1].expr)
            for ex, cond in reversed(e.args[:-1]):
                out = f"(({self._print(cond)}) ? ({self._print(ex)}) : ({out}))"
            return out

    return Printer({"strict": True, "precision": 17})


def _c_double(v: float) -> str:
    r = repr(float(v))
    if not math.isfinite(v):
        raise XdevError(f"xdev: non-finite constant {v}")
    return r if any(ch in r for ch in ".en") else r + ".0"


def _c_str(s: str) -> str:
    return '"' + s + '"'            # validated: no quote, backslash or '?'


def _c_array(ctype: str, name: str, values: Sequence[str]) -> tuple[str, str]:
    """('static const <ctype> NAME[n] = {...};', 'NAME') or ('', 'NULL')."""
    if not values:
        return "", "NULL"
    return f"static const {ctype} {name}[{len(values)}] = {{{', '.join(values)}}};\n", name


def _accurate(sp, e):
    """exp(z) - 1 -> expm1(z), log(1 + z) -> log1p(z). Plain pattern
    rewrites: SymPy's expm1_opt / log1p_opt try factor(), which expanded
    log(1 + (a - b)**2) into a cancelling polynomial."""
    from sympy.codegen.cfunctions import expm1, log1p

    def fix_add(add):
        num = next((t for t in add.args if t.is_Number), None)
        for t in add.args:
            c, rest = t.as_coeff_Mul()
            if num is not None and isinstance(rest, sp.exp) and c == -num:
                others = [u for u in add.args if u is not t and u is not num]
                return sp.Add(c * expm1(rest.args[0]), *others)
        return add

    e = e.replace(lambda a: a.is_Add, fix_add)
    return e.replace(lambda a: isinstance(a, sp.log) and a.args[0].is_Add
                     and sp.S.One in a.args[0].args,
                     lambda a: log1p(a.args[0] - 1))


def _emit_body(sp, printer, assigns) -> str:
    """CSE over the (target, expression) pairs, then C statements."""
    if not assigns:
        return "", ""
    exprs = [_accurate(sp, e) for _, e in assigns]
    temps, reduced = sp.cse(exprs, symbols=sp.numbered_symbols("t"), order="canonical")
    decls, stmts = [], []
    for sym, e in temps:
        ctype = "int" if isinstance(e, sp.logic.boolalg.Boolean) else "double"
        decls.append(f"    const {ctype} {printer.doprint(sym)} = {printer.doprint(e)};\n")
    for (target, _), e in zip(assigns, reduced):
        stmts.append(f"    {target} = {printer.doprint(e)};\n")
    return "".join(decls), "".join(stmts)


def _c_function(sp, printer, name, args, uses_p, uses_s, assigns) -> str:
    """One model function: declarations first (C90-clean), then the
    statements. ``args`` lists the parameters after (ctx, x)."""
    decls, stmts = _emit_body(sp, printer, assigns)
    names = ["c", "x"] + [a.split()[-1].lstrip("*") for a in args]
    head = "".join([("    const double *p = c->p;\n" if uses_p else ""),
                    ("    const double *s = c->state;\n" if uses_s else "")])
    void = "    " + " ".join(f"(void){n};" for n in names + ["p"] * uses_p
                             + ["s"] * uses_s) + "\n"
    return (f"\nstatic int32_t PULSIM_XDEV_CALL {name}(const pulsim_xdev_ctx *c, "
            f"const double *x, {', '.join(args)}) {{\n{head}{decls}{void}{stmts}"
            "    return 0;\n}\n")


@dataclass
class Device:
    """A compiled (or loaded) external device model."""
    name: str
    lib: str
    pins: list
    internal: list
    fields: list                   # [(name, default, unit, lo, hi)]
    smooth: bool
    source: str = ""               # the generated C ('' for a loaded library)
    f: list = field(default_factory=list)       # the declared SymPy law
    q: list = field(default_factory=list)
    unknowns: list = field(default_factory=list)  # handles, x order
    param_syms: list = field(default_factory=list)
    state_syms: list = field(default_factory=list)
    f_pattern: list = field(default_factory=list)  # [(row, col)]
    q_pattern: list = field(default_factory=list)
    _cdll: Any = field(default=None, repr=False)
    _desc: Any = field(default=None, repr=False)

    @property
    def rows(self) -> int:
        return len(self.pins) + len(self.internal)

    def add_to(self, builder, name: str, *nodes: str, record: bool = False,
               check_contract: bool = True, **params):
        """``builder.add_external_device(name, nodes, lib, model, params)``."""
        return builder.add_external_device(name, list(nodes), self.lib, self.name,
                                           params or None, check_contract=check_contract,
                                           record=record)

    def eval(self, x, params: dict | None = None, state=None):
        """The model's own ``eval`` (through ctypes) at ``x``: ``(f, q, J_f,
        J_q)`` with dense Jacobians. Parameters default to the field table's."""
        d = self._descriptor()
        R = self.rows
        x = np.ascontiguousarray(x, dtype=float)
        if x.shape != (R,):
            raise XdevError(f"xdev: {self.name}: x needs {R} values, got {x.shape}")
        given = dict(params or {})
        p = [float(given.pop(fn, df)) for fn, df, *_ in self.fields]
        if given:
            raise XdevError(f"xdev: {self.name}: unknown parameter(s) {sorted(given)}")
        P = (ctypes.c_double * max(len(p), 1))(*p)
        ns = d.n_state
        if state is not None and len(state) != ns:
            raise XdevError(f"xdev: {self.name}: state needs {ns} values, got {len(state)}")
        S = (ctypes.c_double * max(ns, 1))(*(list(state) if state is not None else [0.0] * ns))
        ctx = _Ctx(ctypes.sizeof(_Ctx), 0, 0.0, P, S if ns else None, None)
        f = np.zeros(R)
        q = np.zeros(R)
        df = np.zeros(max(d.nnz_f, 1))
        dq = np.zeros(max(d.nnz_q, 1))
        ptr = ctypes.POINTER(ctypes.c_double)
        rc = _EVAL(d.eval)(ctypes.byref(ctx), x.ctypes.data_as(ptr), f.ctypes.data_as(ptr),
                           q.ctypes.data_as(ptr), df.ctypes.data_as(ptr),
                           dq.ctypes.data_as(ptr))
        if rc != 0:
            raise RuntimeError(f"xdev: {self.name}: eval returned {rc}")
        Jf = np.zeros((R, R))
        Jq = np.zeros((R, R))
        for k in range(d.nnz_f):
            Jf[d.f_rows[k], d.f_cols[k]] += df[k]
        for k in range(d.nnz_q):
            Jq[d.q_rows[k], d.q_cols[k]] += dq[k]
        return f, q, Jf, Jq

    def _descriptor(self):
        if self._desc is None:
            self._cdll, self._desc = _find_model(self.lib, self.name)
        if self._desc.create:
            raise XdevError(f"xdev: {self.name} allocates per-instance data (create): "
                             "evaluate it through a circuit")
        return self._desc


def compile(name: str, pins: Sequence, f: Sequence, q: Sequence | None = None, *,
            internal: Sequence = (), update: dict | None = None,
            outputs: dict | None = None, probe: dict | None = None,
            allow_nonsmooth: bool = False) -> Device:
    """Generate, compile and load the C model of the law ``f``, ``q``.

    ``pins`` and ``internal`` give the unknowns' order; parameters and states
    are found in the expressions (declaration order). ``update`` maps each
    state to its value after an accepted step; ``outputs`` maps a name to
    ``expr`` or ``(expr, unit)`` (read with ``res.device_output`` for a device
    added with ``record=True``); ``probe`` maps an unknown to the ``(lo, hi)``
    box the registration checks sample (default ``(-1, 1)``).
    """
    ns = _sp()
    sp = ns["sp"]
    _check_name("model", name)
    unk = list(pins) + list(internal)
    for s in unk:
        if not (isinstance(s, ns["Var"]) and s.xkind == "x"):
            raise TypeError(f"xdev: {name}: pins / internal must come from xdev.pins / "
                            f"xdev.internal, got {s!r}")
    if len(set(unk)) != len(unk) or not pins:
        raise XdevError(f"xdev: {name}: at least one pin, and no unknown twice")
    R = len(unk)
    f = [sp.sympify(e, strict=True) for e in f]
    q = [sp.sympify(e, strict=True) for e in (q if q is not None else [0] * R)]
    if len(f) != R or len(q) != R:
        raise XdevError(f"xdev: {name}: f and q need {R} rows (pins, then internal "
                         f"unknowns); got {len(f)} and {len(q)}")
    update = dict(update or {})
    for s in update:
        if not (isinstance(s, ns["Var"]) and s.xkind == "s"):
            raise TypeError(f"xdev: {name}: update keys must come from xdev.state, got {s!r}")
    update = {s: sp.sympify(e, strict=True) for s, e in update.items()}
    outs = {}
    for k, v in (outputs or {}).items():
        e, unit = v if isinstance(v, tuple) else (v, "")
        outs[_check_name("output", k)] = (sp.sympify(e, strict=True), _check_unit(k, unit))

    # every symbol must be one of ours
    every = set()
    for e in [*f, *q, *update.values(), *(e for e, _ in outs.values())]:
        every |= e.free_symbols
    foreign = sorted(str(s) for s in every
                     if not isinstance(s, ns["Var"]) or (s.xkind == "x" and s not in unk))
    if foreign:
        raise XdevError(f"xdev: {name}: symbols not declared for this model: {foreign} "
                         "(use xdev.pins / internal / params / state and list the "
                         "unknowns in pins= / internal=)")
    law_syms = set().union(*(e.free_symbols for e in [*f, *q]))
    if ns["h"] in law_syms or any(ns["h"] in e.free_symbols for e, _ in outs.values()):
        raise XdevError(f"xdev: {name}: xdev.h (the step length) may appear only in "
                         "update expressions")
    sts = sorted({s for s in every if s.xkind == "s"} | set(update), key=lambda s: s.dummy_index)
    for s in sts:
        if s not in update:
            raise XdevError(f"xdev: {name}: state {s.name!r} needs an update expression")
        init = sp.sympify(s.meta, strict=True)
        if init.free_symbols - set(unk) - {v for v in init.free_symbols if getattr(v, "xkind", None) == "p"}:
            raise XdevError(f"xdev: {name}: the init of state {s.name!r} may use only the "
                             "unknowns and parameters")
        every |= init.free_symbols
    prm = sorted({s for s in every if s.xkind == "p"}, key=lambda s: s.dummy_index)
    names = [s.name for s in unk + prm + sts] + list(outs)
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise XdevError(f"xdev: {name}: names used twice: {dup}")

    # the lint
    unknowns = set(unk)
    bad = [b for e in [*f, *q] for b in _lint(sp, e, unknowns)]
    if bad and not allow_nonsmooth:
        raise XdevError(
            f"xdev: {name}: not C1 in the unknowns (a Newton device law must be smooth): "
            + "; ".join(str(b) for b in bad[:4])
            + ". Rewrite it smoothly, or pass allow_nonsmooth=True (the model then does "
            "not claim SMOOTH and its Jacobian is one-sided)")
    smooth = not bad
    if bad:
        f = [_to_piecewise(sp, e) for e in f]
        q = [_to_piecewise(sp, e) for e in q]

    # Jacobian pattern (structurally non-zero, row-major)
    df, dq = [], []
    for r in range(R):
        for c, xs in enumerate(unk):
            d = _diff(sp, f[r], xs)
            if d != 0:
                df.append((r, c, d))
            d = _diff(sp, q[r], xs)
            if d != 0:
                dq.append((r, c, d))

    # the C symbols: x[i], p[j], s[k], h (no user name is ever printed)
    sub = {s: sp.Symbol(f"x[{i}]", real=True) for i, s in enumerate(unk)}
    sub.update({s: sp.Symbol(f"p[{j}]", real=True) for j, s in enumerate(prm)})
    sub.update({s: sp.Symbol(f"s[{k}]", real=True) for k, s in enumerate(sts)})
    sub[ns["h"]] = sp.Symbol("h", positive=True)

    def c(e):
        return sp.sympify(e, strict=True).xreplace(sub)

    pr = _printer(sp)
    probe = dict(probe or {})
    box = [probe.pop(s, (-1.0, 1.0)) for s in unk]
    if probe:
        raise XdevError(f"xdev: {name}: probe keys must be the model's unknowns")
    for (lo, hi), s in zip(box, unk):
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise XdevError(f"xdev: {name}: probe box of {s.name!r} must be finite lo < hi")

    src = [f"/* {name}: generated by pulsim.xdev from SymPy expressions. Do not edit. */\n",
           "#define PULSIM_XDEV_DECLARE_ENTRY\n#include \"pulsim/xdev/pulsim_xdev.h\"\n",
           "#include <math.h>\n#include <stddef.h>\n#include <stdint.h>\n",
           # no FMA contraction: c*x0 - c*x1 must be 0 at x0 == x1 (clang contracts
           # by default; gcc's -std=c99 does not, and warns on the pragma)
           "#if defined(__clang__)\n#pragma STDC FP_CONTRACT OFF\n#endif\n\n"]
    arrays = {}
    for key, ctype, vals in (
            ("PINS", "char *const", [_c_str(s.name) for s in pins]),
            ("FIELDS", "pulsim_xdev_field",
             ["{" + ", ".join([_c_str(s.name), _c_str(s.meta[1]),
                               *(_c_double(v) for v in (s.meta[0], s.meta[2], s.meta[3]))])
              + "}" for s in prm]),
            ("F_ROWS", "uint32_t", [f"{r}u" for r, _, _ in df]),
            ("F_COLS", "uint32_t", [f"{k}u" for _, k, _ in df]),
            ("Q_ROWS", "uint32_t", [f"{r}u" for r, _, _ in dq]),
            ("Q_COLS", "uint32_t", [f"{k}u" for _, k, _ in dq]),
            ("PROBE_LO", "double", [_c_double(lo) for lo, _ in box]),
            ("PROBE_HI", "double", [_c_double(hi) for _, hi in box]),
            ("OUTPUTS", "pulsim_xdev_output",
             ["{" + _c_str(k) + ", " + _c_str(u) + "}" for k, (_, u) in outs.items()])):
        text, arrays[key] = _c_array(ctype, key, vals)
        src.append(text)
    rows = [(f"f[{r}]", c(f[r])) for r in range(R)] + [(f"q[{r}]", c(q[r])) for r in range(R)] \
        + [(f"df[{k}]", c(d)) for k, (_, _, d) in enumerate(df)] \
        + [(f"dq[{k}]", c(d)) for k, (_, _, d) in enumerate(dq)]
    src.append(_c_function(sp, pr, "eval_", ["double *f", "double *q", "double *df",
                                            "double *dq"], bool(prm), bool(sts), rows))
    init_fn = accept_fn = outputs_fn = "NULL"
    if sts:
        src.append(_c_function(sp, pr, "init_", ["double *state"], bool(prm), False,
                               [(f"state[{k}]", c(sp.sympify(s.meta, strict=True)))
                                for k, s in enumerate(sts)]))
        src.append(_c_function(sp, pr, "accept_", ["double h", "double *state"], bool(prm),
                               True, [(f"state[{k}]", c(update[s]))
                                      for k, s in enumerate(sts)]))
        init_fn, accept_fn = "init_", "accept_"
    if outs:
        src.append(_c_function(sp, pr, "outputs_", ["double *out"], bool(prm), bool(sts),
                               [(f"out[{k}]", c(e)) for k, (e, _) in enumerate(outs.values())]))
        outputs_fn = "outputs_"
    src.append(
        "\nstatic const pulsim_xdev_descriptor DESC = {\n"
        f"    PULSIM_XDEV_ABI_VERSION, (uint32_t)sizeof(pulsim_xdev_descriptor), {_c_str(name)},\n"
        f"    {len(pins)}u, {len(internal)}u, {len(prm)}u, {len(sts)}u,\n"
        f"    {len(df)}u, {len(dq)}u, 0u, {_SMOOTH if smooth else 0}u,\n"
        f"    {arrays['PINS']}, {arrays['FIELDS']},\n"
        f"    {arrays['F_ROWS']}, {arrays['F_COLS']}, {arrays['Q_ROWS']}, {arrays['Q_COLS']},\n"
        f"    {arrays['PROBE_LO']}, {arrays['PROBE_HI']}, NULL,\n"
        f"    eval_, NULL, NULL, NULL, {init_fn}, {accept_fn},\n"
        "    {NULL, NULL, NULL, NULL},\n"
        f"    {len(outs)}u, 0u, {arrays['OUTPUTS']}, {outputs_fn}}};\n\n"
        "PULSIM_XDEV_EXTERN_C PULSIM_XDEV_EXPORT const pulsim_xdev_descriptor *PULSIM_XDEV_CALL\n"
        "pulsim_xdev_get(uint32_t index) {\n    return index == 0 ? &DESC : NULL;\n}\n")
    source = "".join(src)
    lib = _compile_c(source, "c", who=f"xdev.compile({name!r})")
    return Device(name=name, lib=str(lib), pins=[s.name for s in pins],
                  internal=[s.name for s in internal],
                  fields=[(s.name, s.meta[0], s.meta[1], s.meta[2], s.meta[3]) for s in prm],
                  smooth=smooth, source=source, f=f, q=q, unknowns=unk, param_syms=prm,
                  state_syms=sts, f_pattern=[(r, k) for r, k, _ in df],
                  q_pattern=[(r, k) for r, k, _ in dq])


def _compile_c(source: str, lang: str, *, who: str) -> Path:
    """``source`` (a whole translation unit against pulsim_xdev.h) built
    through the per-user native cache."""
    from . import c_block
    from . import get_include
    compiler = c_block._find_compiler(lang)
    if compiler is None:
        raise RuntimeError(
            f"{who}: no C{'++' if lang == 'cpp' else ''} compiler found (looked for $CC / "
            f"cc / gcc / clang{' / their C++ drivers' if lang == 'cpp' else ''}). Install "
            "one, or build the model yourself and pass lib=.")
    inc = get_include()
    header = (Path(inc) / "pulsim" / "xdev" / "pulsim_xdev.h").read_bytes()
    flags = ["-std=c99" if lang == "c" else "-std=c++17", "-O2", "-shared", "-fPIC",
             f"-I{inc}"]
    link = ["-lm"] if lang == "c" and sys.platform not in ("darwin", "win32") else []
    return _native_cache.compiled(
        source, ".c" if lang == "c" else ".cpp", [compiler, *flags],
        [compiler, lang, *flags, *link, hashlib.sha256(header).hexdigest(), source],
        link=link, sub="xdev", prefix="xdev", who=who)


# ---------------------------------------------------------------------------
# Reading a library's descriptor (ctypes; the layout is frozen by the header)
# ---------------------------------------------------------------------------
_u32, _dbl = ctypes.c_uint32, ctypes.c_double


class _Field(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("unit", ctypes.c_char_p),
                ("default", _dbl), ("lo", _dbl), ("hi", _dbl)]


class _Ctx(ctypes.Structure):
    _fields_ = [("struct_size", _u32), ("host_features", _u32), ("t", _dbl),
                ("p", ctypes.POINTER(_dbl)), ("state", ctypes.POINTER(_dbl)),
                ("instance", ctypes.c_void_p)]


class _Desc(ctypes.Structure):     # the version-1 part (offsets asserted in the header)
    _fields_ = [("abi_version", _u32), ("struct_size", _u32), ("name", ctypes.c_char_p),
                ("n_pins", _u32), ("n_internal", _u32), ("n_fields", _u32),
                ("n_state", _u32), ("nnz_f", _u32), ("nnz_q", _u32),
                ("n_monotone", _u32), ("claims", _u32),
                ("pin_names", ctypes.POINTER(ctypes.c_char_p)),
                ("fields", ctypes.POINTER(_Field)),
                ("f_rows", ctypes.POINTER(_u32)), ("f_cols", ctypes.POINTER(_u32)),
                ("q_rows", ctypes.POINTER(_u32)), ("q_cols", ctypes.POINTER(_u32)),
                ("probe_lo", ctypes.POINTER(_dbl)), ("probe_hi", ctypes.POINTER(_dbl)),
                ("monotone", ctypes.c_void_p), ("eval", ctypes.c_void_p),
                ("validate", ctypes.c_void_p), ("create", ctypes.c_void_p),
                ("destroy", ctypes.c_void_p), ("init", ctypes.c_void_p),
                ("accept", ctypes.c_void_p), ("reserved", ctypes.c_void_p * 4)]


_PD = ctypes.POINTER(_dbl)
_EVAL = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(_Ctx), _PD, _PD, _PD, _PD, _PD)


def _models(lib: str):
    """(cdll, [descriptor, ...]) of a library; loading it runs its code."""
    cdll = ctypes.CDLL(str(lib))
    get = cdll.pulsim_xdev_get
    get.restype = ctypes.POINTER(_Desc)
    get.argtypes = [_u32]
    out = []
    for i in range(1024):
        d = get(i)
        if not d:
            break
        out.append(d.contents)
    return cdll, out


def _find_model(lib: str, model: str | None):
    cdll, ds = _models(lib)
    names = [d.name.decode() for d in ds]
    if model is None:
        if len(ds) != 1:
            raise XdevError(f"xdev: {lib} holds {len(ds)} models {names}: name one (model=)")
        d = ds[0]
    elif model not in names:
        raise XdevError(f"xdev: {lib} has no model {model!r} (it has {names})")
    else:
        d = ds[names.index(model)]
    if d.abi_version != _ABI:
        raise XdevError(f"xdev: {lib}: model {d.name.decode()!r} speaks ABI "
                         f"{d.abi_version}, not {_ABI}")
    rows = d.n_pins + d.n_internal
    missing = [nm for nm, bad in (
        ("struct_size", d.struct_size < ctypes.sizeof(_Desc)), ("eval", not d.eval),
        ("pin_names", d.n_pins and not d.pin_names), ("fields", d.n_fields and not d.fields),
        ("f pattern", d.nnz_f and not (d.f_rows and d.f_cols)),
        ("q pattern", d.nnz_q and not (d.q_rows and d.q_cols)),
        ("probe box", rows and not (d.probe_lo and d.probe_hi))) if bad]
    if missing:
        raise XdevError(f"xdev: {lib}: model {d.name.decode()!r} has a malformed "
                        f"descriptor ({', '.join(missing)})")
    return cdll, d


def load(lib: str, model: str | None = None) -> Device:
    """A model of an existing library (no SymPy), for ``eval`` and
    ``scan_monotone``. Loading a library runs its native code."""
    cdll, d = _find_model(lib, model)
    pin_names = [d.pin_names[k].decode() for k in range(d.n_pins)]
    fields = [(d.fields[k].name.decode(), d.fields[k].default, d.fields[k].unit.decode(),
               d.fields[k].lo, d.fields[k].hi) for k in range(d.n_fields)]
    return Device(name=d.name.decode(), lib=str(lib), pins=pin_names,
                  internal=[f"internal{k}" for k in range(d.n_internal)], fields=fields,
                  smooth=bool(d.claims & _SMOOTH), _cdll=cdll, _desc=d)


# ---------------------------------------------------------------------------
# Monotonicity scan
# ---------------------------------------------------------------------------
@dataclass
class MonotoneScan:
    """``f[row]`` along one unknown: where it DECREASES by more than a few
    ulps of f (``decreasing``, merged intervals) and where it is not finite
    (``nonfinite``)."""
    x: np.ndarray
    f: np.ndarray
    decreasing: list
    nonfinite: list

    @property
    def monotone(self) -> bool:
        return not self.decreasing and not self.nonfinite


def scan_monotone(dev: Device, pin, lo: float, hi: float, *, row: int | None = None,
                  at=None, params: dict | None = None, n: int = 4001) -> MonotoneScan:
    """Evaluate ``f[row]`` (default: the unknown's own row, i.e. the current
    into the device at that pin) on a fine grid of ``pin`` over ``[lo, hi]``,
    plus a geometric run down to 1e-15 of the width on each side of 0, the
    other unknowns at ``at`` (default 0). A Newton device law must be
    non-decreasing down to rounding; the registration check cannot see a
    cancellation at 1e-9 of its probe box, this scan does."""
    names = dev.pins + dev.internal
    col = names.index(pin) if isinstance(pin, str) else int(pin)
    row = col if row is None else row
    if not lo < hi:
        raise XdevError("xdev.scan_monotone: needs lo < hi")
    grid = np.linspace(lo, hi, n)
    if lo < 0 < hi:
        w = max(-lo, hi)
        g = w * 10.0 ** -np.linspace(0, 15, 241)
        grid = np.concatenate([grid, g[g <= hi], -g[-g >= lo], [0.0]])
    grid = np.unique(grid)
    # the scan resolves its grid, not 1-ulp neighbours (linspace and the
    # geometric run can land within rounding of each other)
    grid = grid[np.concatenate([[True], np.diff(grid) > 1e-9 * np.abs(grid[1:])])]
    base = np.zeros(dev.rows) if at is None else np.asarray(at, dtype=float).copy()
    fv = np.empty_like(grid)
    for k, v in enumerate(grid):
        base[col] = v
        fv[k] = dev.eval(base, params)[0][row]
    fin = np.isfinite(fv)
    dec, start = [], None
    idx = np.flatnonzero(fin)
    eps = np.finfo(float).eps
    for a, b in zip(idx[:-1], idx[1:]):
        # a decrease beyond f's own rounding (a few ulps of f)
        if fv[b] < fv[a] - 4 * eps * max(abs(fv[a]), abs(fv[b])):
            if start is None:
                start = grid[a]
            end = grid[b]
        elif start is not None:
            dec.append((float(start), float(end)))
            start = None
    if start is not None:
        dec.append((float(start), float(end)))
    return MonotoneScan(grid, fv, dec, [float(v) for v in grid[~fin]])


# ---------------------------------------------------------------------------
# The free function (task 1.6)
# ---------------------------------------------------------------------------
def add_external_device(builder, name: str, nodes: Sequence[str], *, lib: str | None = None,
                        code: str | None = None, lang: str = "c", model: str | None = None,
                        params: dict | None = None, check_jacobian: bool = True,
                        record: bool = False):
    """Add an external device from a library (``lib=``) or from source
    (``code=``: a whole C / C++ translation unit against ``pulsim_xdev.h``,
    compiled through the per-user native cache). ``model`` may be left out
    when the library holds one model. Either way the code is native code run
    in this process with your privileges. ``check_jacobian=False`` skips the
    registration checks (``check_contract``)."""
    if (lib is None) == (code is None):
        raise XdevError(f"add_external_device({name!r}): give exactly one of lib= or code=")
    if code is not None:
        lang = {"c++": "cpp", "cxx": "cpp"}.get(lang.lower(), lang.lower())
        if lang not in ("c", "cpp"):
            raise XdevError(f"add_external_device({name!r}): lang must be 'c' or 'cpp'")
        lib = str(_compile_c(code, lang, who=f"add_external_device({name!r})"))
    if model is None:
        model = _find_model(lib, None)[1].name.decode()
    return builder.add_external_device(name, list(nodes), str(lib), model, params,
                                       check_contract=check_jacobian, record=record)
