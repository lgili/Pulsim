"""The kernel batch path of :meth:`pulsim.design.Evaluator.evaluate_many`.

add-design-optimization-workflow DECISIONS 26. For the circuit form on
the exact steady-state route with declarative KPIs and constraints, a
whole design -- ``with_params`` / ``with_value`` of the shared base, the
periodic orbit, the period integrals, the losses, the KPIs, the energy
balance, the constraints and the status -- is one kernel call
(``_pulsim._design_batch``, ``core/include/pulsim/dsed/design_batch.hpp``)
on native threads with the GIL released.

The plan is built once per evaluator. Its arithmetic is a register
program written here operation by operation in the order of the Python
path (``design._kpis``, ``_period_kpis.loss_rows``,
``design._energy_balance``), so a kernel record equals the Python path's
record bit for bit. A design the kernel cannot finish comes back as a
``fallback`` with its reason and is re-run on the Python path, which
owns the wording of every refused / failed outcome; the reason is in the
record's ``counters['batch']``.

Also here: :class:`Formula`, the arithmetic of ``Kpi.formula``, compiled
once and run by the same interpreter on both paths.
"""

from __future__ import annotations

import ast
import math
import time

import numpy as np

OPS = dict(const=0, var=1, slot=2, stat=3, cross=4, ystart=5, yend=6, sw=7,
           period=8, load=9, store=10, add=11, sub=12, mul=13, div=14,
           neg=15, pow=16, max2=17, min2=18, abs=19, sqrt=20, exp=21,
           log=22, divpos=23, sumn=24)
_BIN = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div",
        ast.Pow: "pow"}
_FUN1 = ("abs", "sqrt", "exp", "log")
_FUNN = ("max", "min")


# ----------------------------------------------------------------------
# Kpi.formula
# ----------------------------------------------------------------------

def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return None if head is None else f"{head}.{node.attr}"
    return None


class Formula:
    """An arithmetic expression over KPI names (``i_peak.L``, earlier
    KPIs), design variables and ``loss.<device>`` (the device's total
    loss): numbers, ``+ - * / **``, unary minus, ``max``, ``min``,
    ``abs``, ``sqrt``, ``exp``, ``log`` and ``inf``. Compiled once to a
    stack program that the Python path and the kernel batch run with the
    same IEEE operations (a division by zero is ``inf`` / ``nan``, which
    the evaluator then records as a non-finite KPI)."""

    def __init__(self, expr: str):
        self.expr = str(expr)
        try:
            tree = ast.parse(self.expr, mode="eval").body
        except SyntaxError as e:
            raise ValueError(f"Kpi.formula: {self.expr!r} is not an "
                             f"expression ({e.msg})") from None
        self.code: list = []
        self.names: list = []
        self._emit(tree)

    def _emit(self, n):
        code = self.code
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            code.append(("const", float(n.value)))
        elif isinstance(n, ast.Name) and n.id == "inf":
            code.append(("const", math.inf))
        elif _dotted(n) is not None:
            name = _dotted(n)
            code.append(("name", name))
            self.names.append(name)
        elif isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            self._emit(n.left)
            self._emit(n.right)
            code.append((_BIN[type(n.op)], None))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub,
                                                             ast.UAdd)):
            self._emit(n.operand)
            if isinstance(n.op, ast.USub):
                code.append(("neg", None))
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and not n.keywords and n.args
              and (n.func.id in _FUNN or (n.func.id in _FUN1
                                          and len(n.args) == 1))):
            self._emit(n.args[0])
            for a in n.args[1:]:
                self._emit(a)
                code.append((n.func.id + "2", None))
            if n.func.id in _FUN1:
                code.append((n.func.id, None))
        else:
            raise ValueError(
                f"Kpi.formula: {ast.unparse(n)!r} in {self.expr!r} is not "
                "allowed (numbers, names, + - * / **, max, min, abs, sqrt, "
                "exp, log)")

    def __eq__(self, o):
        return isinstance(o, Formula) and o.expr == self.expr

    def __hash__(self):
        return hash(self.expr)

    def __repr__(self):
        return f"Formula({self.expr!r})"

    def __reduce__(self):
        return (Formula, (self.expr,))

    def evaluate(self, lookup) -> float:
        """Run the program; ``lookup(name) -> float``."""
        st: list = []
        for op, a in self.code:
            if op == "const":
                st.append(a)
            elif op == "name":
                st.append(float(lookup(a)))
            elif op == "neg":
                st[-1] = -st[-1]
            elif op in _UNARY:
                st[-1] = _UNARY[op](st[-1])
            else:
                b = st.pop()
                st[-1] = _BINARY[op](st[-1], b)
        return st[0]


def _div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        if a != a or a == 0:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)


def _pow(a, b):
    try:
        return math.pow(a, b)
    except OverflowError:
        return -math.inf if a < 0 and b == int(b) and int(b) % 2 else math.inf
    except ValueError:
        if a == 0 and b < 0:
            return -math.inf if (math.copysign(1.0, a) < 0 and b == int(b)
                                 and int(b) % 2) else math.inf
        return math.nan


def _guard(f, bad):
    def g(x):
        try:
            return f(x)
        except (ValueError, OverflowError):
            return bad(x)
    return g


_UNARY = {
    "abs": abs,
    "sqrt": _guard(math.sqrt, lambda x: math.nan),
    "exp": _guard(math.exp, lambda x: math.inf),
    "log": _guard(math.log, lambda x: -math.inf if x == 0 else math.nan),
}
_BINARY = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b, "div": _div, "pow": _pow,
    "max2": lambda a, b: b if b > a else a,
    "min2": lambda a, b: b if b < a else a,
}


def _psum(xs) -> float:
    """CPython >= 3.12's ``sum()`` of floats (Neumaier-compensated),
    on every Python version: the record arithmetic's sum, mirrored by
    the kernel program's ``SumN`` (3.10 / 3.11 ``sum`` is a plain
    loop, which would make a record depend on the Python version)."""
    f = c = 0.0
    for x in xs:
        x = float(x)
        t = f + x
        if abs(f) >= abs(x):
            c += (f - t) + x
        else:
            c += (x - t) + f
        f = t
    if c and math.isfinite(c):
        f += c
    return f


def formula_lookup(kpis, design, losses):
    """The names a :class:`Formula` reads on the Python path."""
    def look(name):
        if name in kpis:
            return kpis[name]
        if name.startswith("loss."):
            return _psum(losses.get(name[5:], {}).values())
        if name in design:
            return design[name]
        raise KeyError(f"Kpi.formula: {name!r} is neither a KPI computed "
                       f"before it, a design variable nor loss.<device>; "
                       f"KPIs: {sorted(kpis)}")
    return look


# ----------------------------------------------------------------------
# the plan
# ----------------------------------------------------------------------

class _Prog:
    def __init__(self):
        self.code: list = []
        self.n_regs = 0

    def op(self, name, a=0, b=0, c=0.0):
        self.code.append((OPS[name], int(a), int(b), float(c)))

    def store(self):
        r = self.n_regs
        self.n_regs += 1
        self.op("store", r)
        return r

    def load(self, r):
        self.op("load", r)

    def const(self, c):
        self.op("const", c=c)

    def sum_of(self, regs):
        """``design._psum`` of the registers, in order."""
        regs = list(regs)
        for r in regs:
            self.load(r)
        self.op("sumn", len(regs))


_FIELD = {"resistor": "R", "inductor": "L", "capacitor": "C",
          "voltage_source": "V"}
_ALLOWED = ("resistor", "inductor", "capacitor", "switch", "diode",
            "voltage_source")


class Plan:
    """What :func:`run` hands the kernel, built once per evaluator."""

    def __init__(self, ev):
        from . import _pulsim as _k
        from ._dsed_dispatch import H_E_REL, MODE_GUARD
        from .design import _KpiSpec, _SEMIS, _as_builder
        from .loss_tables import LossTable
        c = ev.circuit
        b = _as_builder(c)
        spec = ev._spec(c)
        self.reason = None
        if not isinstance(spec, type(_pk_spec())):
            self.reason = f"the period integrals do not cover it ({spec})"
            return
        comps = {x["name"]: x for x in b.components()}
        self.var_names = list(ev._targets)
        self.names = [t for _, t in ev._targets.values()]
        self.is_param = [k == "param" for k, _ in ev._targets.values()]
        self.topology = tuple((x["kind"], tuple(x.get("nodes", ())))
                              for x in b.components())
        P = _Prog()
        slots: list = []

        def slot(name):
            f = f"{name}.{_FIELD[comps[name]['kind']]}"
            if f not in slots:
                slots.append(f)
            P.op("slot", slots.index(f))

        def stat(key, col):
            P.op("stat", spec.idx[key], col)

        def mean(key):           # st[j, 0] / T
            stat(key, 0)
            P.op("period")
            P.op("div")

        def rms(key):            # sqrt(max(st[j, 1] / T, 0.0))
            stat(key, 1)
            P.op("period")
            P.op("div")
            P.const(0.0)
            P.op("max2")
            P.op("sqrt")

        def p2p(key):
            stat(key, 2)
            stat(key, 4)
            P.op("sub")

        def peak_abs(key):
            stat(key, 2)
            P.op("abs")
            stat(key, 4)
            P.op("abs")
            P.op("max2")

        # -- loss rows (_period_kpis.loss_rows) --------------------------
        specs = ev.loss_options.get("switch_specs") or {}
        rows: dict = {}
        for name, x in comps.items():
            if x["kind"] == "resistor":
                stat(("v", name), 1)
                slot(name)
                P.op("div")
                P.op("period")
                P.op("div")
                rows[name] = {"kind": "resistor", "P_avg": P.store()}
        switching = []
        for name, kind, bit in spec.semis:
            q = spec.pairs.index((spec.idx[("i", name)], spec.idx[("v", name)]))
            P.op("cross", q)
            P.op("period")
            P.op("div")
            row = {"kind": kind, "P_avg": P.store()}
            peak_abs(("i", name))
            row["i_peak"] = P.store()
            rms(("i", name))
            row["i_rms"] = P.store()
            sp = specs.get(name)
            if sp is not None:          # a switch's or a diode's tables
                on, off, tj = (sp.get("E_on_table"), sp.get("E_off_table"),
                               sp.get("Tj"))
                ok = ((on is not None or off is not None)
                      and type(tj) in (int, float) and math.isfinite(tj)
                      and all(t is None or isinstance(t, LossTable)
                              for t in (on, off)))
                for t, ref, curve in ((on, "E_on_ref", "E_on_curve"),
                                      (off, "E_off_ref", "E_off_curve")):
                    if t is None and (float(sp.get(ref, 0.0)) > 0
                                      or sp.get(curve) is not None):
                        ok = False
                if not ok:
                    self.reason = (f"switch_specs[{name!r}] is not a pair of "
                                   "loss tables at a numeric Tj")
                    return
                k = len(switching)
                switching.append((bit, spec.idx[("v", name)],
                                  spec.idx[("i", name)],
                                  None if on is None else on._kernel(),
                                  None if off is None else off._kernel(),
                                  float(tj)))
                P.op("sw", k, 0)
                P.op("sw", k, 1)
                P.op("add")
                P.op("period")
                P.op("div")
                row["P_sw_avg"] = P.store()
            rows[name] = row
        for n in ev.loads:
            if n not in rows:
                self.reason = f"load {n!r} has no dissipated power"
                return
        losses: dict = {}
        for n, r in rows.items():
            if n in ev.loads:
                continue
            cat = {("resistive" if r["kind"] == "resistor"
                    else "conduction"): r["P_avg"]}
            if "P_sw_avg" in r:
                cat["switching" if r["kind"] == "switch"
                    else "recovery"] = r["P_sw_avg"]
            losses[n] = cat
        # -- KPIs (design._kpis on a kernel Period) ----------------------
        kp: dict = {}
        src = ev.sources
        if src is None:
            src = [n for n, x in comps.items() if x["kind"] in (
                "voltage_source", "pwm_voltage_source",
                "pulse_voltage_source", "sine_voltage_source")]
        for n in src:
            if comps.get(n, {}).get("kind") != "voltage_source":
                self.reason = f"source {n!r} is not a DC voltage source"
                return
        p_in = None
        if src:
            P.const(0.0)
            for n in src:
                slot(n)
                mean(("i", n))
                P.op("mul")
                P.op("sub")
            p_in = kp["p_in"] = P.store()
        for c_ in losses.values():
            P.sum_of(c_.values())
        P.op("sumn", len(losses))
        kp["p_loss"] = P.store()
        energy = False
        if ev.loads:
            P.sum_of(rows[n]["P_avg"] for n in ev.loads)
            kp["p_out"] = P.store()
            P.load(kp["p_out"])
            P.load(kp["p_out"])
            P.load(kp["p_loss"])
            P.op("add")
            P.op("divpos")
            kp["efficiency"] = P.store()
            energy = p_in is not None and not (
                any(x["kind"] not in _ALLOWED for x in comps.values())
                or any(x["kind"] == "voltage_source" and n not in src
                       for n, x in comps.items()))
            if energy:
                # (p_in - p_dis - p_stored) / p_in; p_stored as
                # _stored_power_kernel: e += 0.5 k (y1^2 - y0^2), e / T
                P.const(0.0)
                for n, x in comps.items():
                    if x["kind"] not in ("inductor", "capacitor"):
                        continue

                    def at(end, n=n, x=x):
                        op = "yend" if end else "ystart"
                        if x["kind"] == "inductor":
                            P.op(op, spec.idx[("i", n)])
                            return
                        m = 0
                        for k_, sg in zip(x["nodes"], (1.0, -1.0)):
                            if int(k_) >= 0:
                                P.const(sg)
                                P.op(op, spec.idx[("node#", int(k_))])
                                P.op("mul")
                                m += 1
                        P.op("sumn", m)
                    P.const(0.5)
                    slot(n)
                    P.op("mul")
                    at(True)
                    at(True)
                    P.op("mul")
                    at(False)
                    at(False)
                    P.op("mul")
                    P.op("sub")
                    P.op("mul")
                    P.op("add")
                P.op("period")
                P.op("div")
                stored = P.store()
                P.load(p_in)
                P.sum_of(r["P_avg"] for r in rows.values())
                P.op("sub")
                P.load(stored)
                P.op("sub")
                P.load(p_in)
                P.op("div")
                res = P.store()
                P.sum_of(v for c_ in losses.values() for k_, v in c_.items()
                         if k_ in ("switching", "recovery", "core"))
                p_table = P.store()
        if ev.output is not None:
            mean(("node", ev.output))
            kp["v_out"] = P.store()
            p2p(("node", ev.output))
            kp["ripple_v"] = P.store()
        for n, x in comps.items():
            if x["kind"] == "inductor":
                key = ("i", n)
                p2p(key)
                kp[f"ripple_i.{n}"] = P.store()
                peak_abs(key)
                kp[f"i_peak.{n}"] = P.store()
                rms(key)
                kp[f"i_rms.{n}"] = P.store()
            elif x["kind"] in _SEMIS and n in rows:
                kp[f"i_peak.{n}"] = rows[n]["i_peak"]
                kp[f"i_rms.{n}"] = rows[n]["i_rms"]
                peak_abs(("v", n))
                kp[f"v_peak.{n}"] = P.store()
        for k in ev.kpis:
            s = k.fn
            if not isinstance(s, _KpiSpec):
                self.reason = f"KPI {k.name!r} is a Python callable"
                return
            if s.kind in ("efficiency", "i_peak", "i_rms", "ripple_i"):
                name = s.kind if s.target is None else f"{s.kind}.{s.target}"
                if name not in kp:
                    self.reason = f"Kpi {k.name!r} names no KPI"
                    return
                kp[k.name] = kp[name]
            elif s.kind in ("ripple_v", "v_avg"):
                try:
                    col = int(b.node_id_of(s.target))
                except Exception:   # noqa: BLE001 - the Python path names it
                    self.reason = f"Kpi {k.name!r} names no node"
                    return
                if col < 0:
                    P.const(0.0)
                elif s.kind == "ripple_v":
                    p2p(("node", s.target))
                else:
                    mean(("node", s.target))
                kp[k.name] = P.store()
            elif s.kind == "formula":
                for op, a in s.target.code:
                    if op == "const":
                        P.const(a)
                    elif op == "name":
                        if a in kp:
                            P.load(kp[a])
                        elif a.startswith("loss."):
                            P.sum_of(losses.get(a[5:], {}).values())
                        elif a in self.var_names:
                            P.op("var", self.var_names.index(a))
                        else:
                            self.reason = (f"Kpi {k.name!r} reads {a!r}, "
                                           "which the plan does not know")
                            return
                    else:
                        P.op(op)
                kp[k.name] = P.store()
            else:
                self.reason = f"Kpi kind {s.kind!r}"
                return
        viol = []
        for name, (lo, hi) in ev.constraints.items():
            if name not in kp:
                self.reason = f"constraint {name!r} names no KPI"
                return
            P.const(0.0)
            if lo is not None:
                P.const(float(lo))
                P.load(kp[name])
                P.op("sub")
            else:
                P.const(0.0)
            P.op("max2")
            if hi is not None:
                P.load(kp[name])
                P.const(float(hi))
                P.op("sub")
            else:
                P.const(0.0)
            P.op("max2")
            viol.append(P.store())
        _, nd, _ = _k._switch_census(b.graph, b.pool)
        self.kpi_names = list(kp)
        self.loss_keys = [(d, cat) for d, c_ in losses.items() for cat in c_]
        self.loss_groups, a = [], 0
        for d, c_ in losses.items():
            self.loss_groups.append((d, list(c_), a, a + len(c_)))
            a += len(c_)
        self.constraint_names = list(ev.constraints)
        self.energy = energy
        self.kernel = dict(
            names=self.names, is_param=self.is_param, obs=spec.obs,
            pairs=spec.pairs, slots=slots, switching=switching, code=P.code,
            n_regs=P.n_regs, kpi_regs=[kp[n] for n in self.kpi_names],
            loss_regs=[losses[d][c_] for d, c_ in self.loss_keys],
            viol_regs=viol, energy_static=energy,
            p_in_reg=p_in if energy else -1,
            res_reg=res if energy else -1,
            p_table_reg=p_table if energy else -1,
            energy_tol=_energy_tol(), node_obs0=spec.idx[("node#", 0)]
            if spec.n_nodes else -1, n_nodes=spec.n_nodes,
            n_full=spec.n_full, rtol=_pss_rtol(),
            max_period_runs=int(ev.steady.max_period_runs), n_relax=5,
            mode_guard=bool(MODE_GUARD and nd), h_e_rel=float(H_E_REL),
            run_rtol=1e-11, run_atol=1e-14)


def _pk_spec():
    from ._period_kpis import Spec
    return Spec()


def _energy_tol():
    from .design import _ENERGY_RTOL
    return _ENERGY_RTOL["exact"]


def _pss_rtol():
    from .periodic_steady_state import _RTOL
    return _RTOL


def plan_of(ev):
    """The evaluator's kernel plan, or a string naming why it has none
    (the batch then takes the Python path for every design)."""
    p = ev.__dict__.get("_kplan")
    if p is not None:
        ev._check_frozen()   # the plan froze kpis= / constraints=
    if p is None:
        p = _plan_or_reason(ev)
        ev._kplan = p
    return p


def _plan_or_reason(ev):
    if ev.circuit is None:
        return "the factory form (build=)"
    if ev.steady.method not in ("pss", "auto") or ev.engine not in (
            "auto", "dsed"):
        return f"method {ev.steady.method!r} on engine {ev.engine!r}"
    if ev.thermal is not None:
        return "a thermal loop"
    if ev.gradients:
        return "gradients="
    if ev.options:
        return "options= (refused on this route, per design)"
    if not ev._spec_static:
        return "a design variable moves a value the observables read"
    from . import _pulsim as _k
    if not hasattr(_k, "_design_batch"):
        return "this build has no kernel batch"
    try:
        pl = Plan(ev)
    except (KeyError, ValueError, IndexError) as e:
        return f"the plan cannot be built ({type(e).__name__}: {e})"
    return pl.reason or pl


# ----------------------------------------------------------------------
# the batch
# ----------------------------------------------------------------------

def _native(sf):
    from . import _pulsim as _k
    from .periodic_steady_state import _closed_loop_source
    return (isinstance(sf, (_k.SwitchSchedule, _k.NativePwm2Switch,
                            _k.NativeMultiMaskPwm))
            and not _closed_loop_source(sf)       # the one predicate
            and (not isinstance(sf, _k.SwitchSchedule) or sf.has_edges))


def run(ev, plan, designs, workers):
    """Records of ``designs`` (input order) and the batch counters."""
    from . import _pulsim as _k
    from .design import EvaluationRecord, FrozenDict, _fz
    n = len(designs)
    empty = FrozenDict()
    t0 = time.perf_counter()
    stats = {"designs": n, "kernel": 0, "python": 0, "python_reasons": {},
             "symbolic_analyses": 0, "python_calls_in_kernel": 0,
             "nogil": True, "threads": int(workers or 1),
             "periodicity_checks": 0}
    rows, periods, scheds, idx, py_idx = [], [], [], [], {}
    for i, d in enumerate(designs):
        missing = [v for v in plan.var_names if v not in d]
        if missing:
            py_idx[i] = "design variables missing"
            continue
        try:
            T = ev._period_of(d)
            sf = (ev.schedule(d) if ev.schedule is not None
                  else ev.switch_fn)
        except Exception:  # noqa: BLE001 - the Python path raises it
            py_idx[i] = "period or schedule raised"
            continue
        if sf is None or not _native(sf):
            py_idx[i] = "a schedule that is not a native schedule with edges"
            continue
        try:
            row = [float(d[v]) for v in plan.var_names]
        except (TypeError, ValueError):
            py_idx[i] = "a non-numeric design variable"
            continue
        rows.append(row)
        periods.append(T)
        scheds.append(sf)
        idx.append(i)
    out: list = [None] * n
    if idx:
        table = np.asarray(rows, dtype=float).reshape(len(idx),
                                                      len(plan.var_names))
        r = _k._design_batch(ev.circuit, plan.kernel, table, periods,
                             scheds, int(workers or 1))
        stats["symbolic_analyses"] = int(r["symbolic_analyses"])
        stats["python_calls_in_kernel"] = int(r["python_calls"])
        stats["nogil"] = bool(r["nogil"])
        _periodicity(ev, plan, r, idx, designs, scheds, periods, stats)
        prov = ev._provenance()
        K, L, V = r["kpis"], r["losses"], r["viol"]
        for j, i in enumerate(idx):
            st = int(r["status"][j])
            if st == 2:
                py_idx[i] = "kernel: " + r["reasons"][j]
                continue
            d = designs[i]
            runs = int(r["n_runs"][j])
            counters = {"simulations": runs, "steady_method": "pss",
                        "route": "pss/dsed", "pss_period_runs": runs,
                        "engine_steps": int(r["n_steps"][j]),
                        "python_callbacks": 0, "nogil_period_runs": runs,
                        "pss_loop": "kernel",
                        "pss_iterations": int(r["n_iterations"][j]),
                        "kpi_path": "kernel"}
            if r["energy_applied"][j]:
                counters["energy_residual"] = float(r["energy_res"][j])
                counters["p_table"] = float(r["p_table"][j])
            lrow = L[j].tolist()
            out[i] = EvaluationRecord(
                design=_fz(d), design_id=ev.design_id(d),
                status="ok" if st == 0 else "infeasible",
                kpis=FrozenDict(zip(plan.kpi_names, K[j].tolist())),
                losses=FrozenDict(
                    (dev, FrozenDict(zip(cats, lrow[a:z])))
                    for dev, cats, a, z in plan.loss_groups),
                t_j=empty, violations=FrozenDict(zip(
                    plan.constraint_names, V[j].tolist())),
                reason=None, refusal=None, counters=FrozenDict(counters),
                provenance=prov, wall_s=float(r["wall_s"][j]), grads=empty)
            stats["kernel"] += 1
    if py_idx:
        _python_path(ev, designs, py_idx, out, workers, stats)
    stats["wall_s"] = time.perf_counter() - t0
    ev.last_batch = _fz(stats)
    return out


def _periodicity(ev, plan, r, idx, designs, scheds, periods, stats):
    """The independent periodicity check (design._check_periodic) once
    per event pattern (first design in input order) and for every
    design with an event at the section; a mismatch raises."""
    from . import simulate as _simulate
    from .design import PeriodicityError
    todo = []
    for names, j, x0 in r["groups"]:
        key = (plan.topology, tuple(names))
        if key not in ev.periodicity_checks:
            todo.append((j, x0, key))
    todo += [(j, x0, None) for j, x0 in r["at_section"]]
    for j, x0, key in sorted(todo, key=lambda t: t[0]):
        i = idx[j]
        T, sf = periods[j], scheds[j]
        src = ev._circuit_of(designs[i])
        try:
            run_ = _simulate(src, t_end=T, dt=T, engine="dsed", switch_fn=sf,
                             initial_state=np.asarray(x0), rtol=1e-11,
                             atol=1e-14)
        except (RuntimeError, ValueError) as e:
            if type(e).__name__ == "_CxxCancelled":
                raise
            raise PeriodicityError(
                "periodicity: a plain simulate(engine='dsed') period from "
                f"the shooting's x* failed ({type(e).__name__}: {e}) where "
                "the shooting map ran it (kernel batch, design "
                f"{i})") from e
        stats["periodicity_checks"] += 1
        x0 = np.asarray(x0, dtype=float)
        dev = _deviation(np.asarray(run_.states_reduced, dtype=float), x0)
        if key is not None:
            ev.periodicity_checks[key] = dev
        from .design import _PERIODIC_RTOL
        if not dev <= _PERIODIC_RTOL:
            raise PeriodicityError(
                "periodicity: a plain simulate(engine='dsed') period from "
                f"the shooting's x* moved it by {dev:.3e} (relative, "
                f"tolerance {_PERIODIC_RTOL:.0e}); the shooting map and the "
                f"engine disagree (kernel batch, design {i})")


def _deviation(xs, x0):
    """design._check_periodic's relative deviation."""
    dev = np.abs(xs[-1] - x0)
    den = np.abs(xs).max(axis=0) + np.abs(x0)
    den = np.maximum(den, 1e-12 * float(den.max(initial=0.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(dev == 0.0, 0.0, dev / np.where(den > 0, den, 1e-300))
    return float(q.max())


def _python_path(ev, designs, py_idx, out, workers, stats):
    """The designs the kernel did not finish, on the Python path (thread
    jobs), each record marked with the reason."""
    from dataclasses import replace

    from .design import _fz
    from .sweep import _run_jobs
    from .design import _evaluate_job
    order = sorted(py_idx)
    jobs = [(ev, dict(designs[i])) for i in order]
    try:
        recs, _, _ = _run_jobs(jobs, parallel=(workers or 1) != 1 and
                               len(jobs) > 1, backend="thread",
                               n_workers=workers, verbose=False,
                               job=_evaluate_job, reraise=True)
    except Exception as e:  # noqa: BLE001 - re-raised with the design index
        if hasattr(e, "add_note"):
            e.add_note(f"evaluate_many: on the Python path of designs "
                       f"{order[:8]}{'...' if len(order) > 8 else ''}")
        raise
    for i, rec in zip(order, recs):
        why = py_idx[i]
        out[i] = replace(rec, counters=_fz({**rec.counters,
                                            "batch": "python: " + why}))
        stats["python"] += 1
        key = why.split(" (")[0]
        stats["python_reasons"][key] = stats["python_reasons"].get(key, 0) + 1
