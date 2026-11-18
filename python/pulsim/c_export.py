"""C export of a control graph (add-compiled-control-graph, group 8).

:func:`export_c` renders a rated :class:`MixedDomainBlockChain` to one
C99 translation unit: the kernel's law table (``block_laws.h``, the
functions the kernel graph itself calls) copied verbatim, a state
struct, an init function, one step function per rate, the rate table,
64-bit tick counters and, for a single-rate graph, the
``pulsim_cblock_step`` shim (:data:`pulsim.c_block.CBLOCK_ABI`) so the
export loads back through ``add_c_block(lib=...)``. No allocation: the
shim's state is a static struct.

Bit for bit with the kernel graph when compiled without FMA contraction
(``-ffp-contract=off``; MSVC's default ``/fp:precise`` does not
contract). Refused by name: a PythonBlock (or a Python library block),
a CBlock (its compiled object carries no source), an every-step graph
(no Rate: a continuous-time block has no sample time in C), and a rate
with a compute_delay (a circuit-sink property the export has no sinks
for).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CExport", "export_c"]


@dataclass
class CExport:
    """The generated C and its interface.

    ``inputs[k]`` is what the host passes in ``in[k]``: a node name, or
    ``"channel:<name>"`` for a channel no block writes. ``outputs[k]``
    is the signal the shim writes to ``out[k]``; ``signals[j]`` is the
    signal held in ``state.sig[j]`` (``<prefix>_get(s, j)``).
    ``periods`` / ``offsets`` are the rate table, one entry per rate in
    the order of ``<prefix>_step_rate<g>``.
    """

    source: str
    prefix: str
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    periods: list = field(default_factory=list)
    offsets: list = field(default_factory=list)

    def write(self, path) -> Path:
        path = Path(path)
        path.write_text(self.source)
        return path


def _lit(v) -> str:
    """An exact C99 double literal (hex float; HUGE_VAL / NAN)."""
    v = float(v)
    if math.isnan(v):
        return "NAN"
    if math.isinf(v):
        return "HUGE_VAL" if v > 0 else "(-HUGE_VAL)"
    return v.hex() if v != 0.0 else ("-0.0" if math.copysign(1, v) < 0 else "0.0")


def _arr(vals) -> str:
    return "{" + ", ".join(_lit(v) for v in vals) + "}"


def _refuse(msg):
    from ._pulsim import ControlGraphBuildError  # type: ignore[import-not-found]
    return ControlGraphBuildError("export_c: " + msg)


def export_c(chain, builder, outputs, *, prefix: str = "pulsim_graph"
             ) -> CExport:
    """Render ``chain`` (built against ``builder``) to C.

    ``outputs`` names the signals the ``pulsim_cblock_step`` shim writes,
    in order (``out[k]``). Raises ``ControlGraphBuildError`` naming the
    block it cannot export."""
    from .blockchain import _PY_LIBRARY
    if not prefix.isidentifier():
        raise ValueError(f"export_c: prefix {prefix!r} is not a C identifier")
    outputs = [outputs] if isinstance(outputs, str) else list(outputs)
    for spec in chain.blocks:
        cls = type(spec.block).__name__
        if cls == "PythonBlock" or cls in _PY_LIBRARY:
            raise _refuse(
                f"block '{spec.name}' ({cls}) runs Python; a C export holds "
                "only kernel blocks")
        if cls == "CBlock":
            raise _refuse(
                f"block '{spec.name}' is a C block: its compiled object has "
                "no source to inline; export the graph around it")
    if chain.rate is None and all(s.rate is None for s in chain.blocks):
        raise _refuse(
            "the graph has no Rate: its blocks run on every solver step "
            "(continuous time), which has no sample time in C; give the "
            "chain a Rate")
    chain.make_step_observer(builder)        # validates and orders
    cxx = chain._cxx_chain
    rates = sorted(chain._groups, key=chain._groups.get)
    for r in rates:
        if r._delay()[0] != 0:
            raise _refuse(
                f"Rate(period={r.period}) has compute_delay="
                f"{r.compute_delay!r}: a delay applies to circuit sinks, "
                "which an export has none of; export with compute_delay=0")
    gid = {r: g for g, r in enumerate(rates)}
    by_name = {s.name: i for i, s in enumerate(chain.blocks)}
    order = [by_name[n] for n in cxx.schedule()]

    signals: list = []
    for spec in chain.blocks:
        for o in (spec.output if isinstance(spec.output, (tuple, list))
                  else (spec.output,)):
            if o not in signals:
                signals.append(str(o))
    for o in outputs:
        if o not in signals:
            raise _refuse(f"output '{o}' is not written by any block")
    sig = {n: j for j, n in enumerate(signals)}
    inputs: list = []
    state: list = []      # (C declaration, init statement)
    steps = {g: [] for g in range(len(rates))}

    for i in order:
        spec = chain.blocks[i]
        b, cls, ins = spec.block, type(spec.block).__name__, spec.inputs
        g = gid[chain._block_rates[i]]
        period = _lit(rates[g].period)
        outs = list(spec.output) if isinstance(spec.output, (tuple, list)) \
            else [spec.output]
        o = [f"s->sig[{sig[str(n)]}]" for n in outs]
        p = f"s->b{i}_"

        def src(s_):
            if s_.kind == "const":
                return _lit(s_.value)
            if s_.kind == "time":
                return "t"
            if s_.kind == "dt":
                return period
            if s_.kind == "channel" and s_.value in sig:
                return f"s->sig[{sig[s_.value]}]"
            if s_.kind in ("node", "channel"):
                key = s_.value if s_.kind == "node" else "channel:" + s_.value
                if key not in inputs:
                    inputs.append(key)
                return f"in[{inputs.index(key)}]"
            raise _refuse(f"block '{spec.name}': input kind {s_.kind!r}")

        def a(name, default=None):
            if name not in ins:
                if default is None:
                    raise _refuse(f"block '{spec.name}': no input '{name}'")
                return _lit(default)
            return src(ins[name])

        def st(name, init, n=None):
            if n is None:
                state.append((f"double b{i}_{name};",
                              f"s->b{i}_{name} = {_lit(init)};"))
            else:
                state.append((f"double b{i}_{name}[{max(n, 1)}];",
                              "".join(f"s->b{i}_{name}[{k}] = {_lit(v)}; "
                                      for k, v in enumerate(init))))

        def tup(name):
            v = ins[name]
            return [src(e) for e in (v.value if v.kind == "tuple" else [v])]

        code = []
        if cls == "Gain":
            code.append(f"{o[0]} = pl_gain({_lit(b.k)}, {a('x')});")
        elif cls == "Subtract":
            code.append(f"{o[0]} = pl_subtract({a('a')}, {a('b')});")
        elif cls == "Sum":
            xs = tup("inputs") if "inputs" in ins else \
                [a(k) for k in ("a", "b", "c") if k in ins]
            n = len(xs)
            code.append(f"{{ static const double w[{n}] = {_arr(b.weights)};"
                        f" const double x[{n}] = {{{', '.join(xs)}}};"
                        f" {o[0]} = pl_sum(w, x, {n}); }}")
        elif cls == "PIController":
            st("i", b.integrator_state)
            st("e", b._prev_error)
            code.append(
                f"{o[0]} = pl_pi({_lit(b.Kp)}, {_lit(b.Ki)}, "
                f"{_lit(b.output_min)}, {_lit(b.output_max)}, &{p}i, &{p}e, "
                f"{a('setpoint')}, {a('measured')}, {a('dt')});")
        elif cls == "PIDController":
            st("i", b.integrator_state)
            st("d", b.derivative_state)
            st("e", b._prev_error)
            code.append(
                f"{o[0]} = pl_pid({_lit(b.Kp)}, {_lit(b.Ki)}, {_lit(b.Kd)}, "
                f"{_lit(b.derivative_alpha)}, {_lit(b.output_min)}, "
                f"{_lit(b.output_max)}, &{p}i, &{p}d, &{p}e, "
                f"{a('setpoint')}, {a('measured')}, {a('dt')});")
        elif cls == "FirstOrderLowPass":
            st("s", b.state)
            code.append(f"{o[0]} = pl_low_pass({_lit(b.tau)}, &{p}s, "
                        f"{a('input_value')}, {a('dt')});")
        elif cls == "Integrator":
            st("s", b.state)
            code.append(f"{o[0]} = pl_integrator({_lit(b.gain)}, "
                        f"{_lit(b.output_min)}, {_lit(b.output_max)}, &{p}s, "
                        f"{a('x')}, {a('dt')});")
        elif cls == "Differentiator":
            st("s", b.state)
            st("x", b._prev_x)
            code.append(f"{o[0]} = pl_differentiator({_lit(b.alpha)}, &{p}s, "
                        f"&{p}x, {a('x')}, {a('dt')});")
        elif cls == "Limiter":
            code.append(f"{o[0]} = pl_limiter({_lit(b.output_min)}, "
                        f"{_lit(b.output_max)}, {a('x')});")
        elif cls == "MovingAverageFilter":
            st("s", b.state)
            code.append(f"{o[0]} = pl_ema({_lit(b.alpha)}, &{p}s, "
                        f"{a('input_value')});")
        elif cls == "Comparator":
            st("s", float(b.state))
            code.append(
                f"{o[0]} = pl_comparator({_lit(b.threshold)}, "
                f"{_lit(b.hysteresis)}, {_lit(b.output_high)}, "
                f"{_lit(b.output_low)}, &{p}s, {a('input_value')});")
        elif cls == "RateLimiter":
            st("s", b.state)
            code.append(f"{o[0]} = pl_rate_limiter({_lit(b.slew_up)}, "
                        f"{_lit(b.slew_down)}, &{p}s, {a('target')}, "
                        f"{a('dt')});")
        elif cls == "SampleHold":
            st("s", b.state)
            st("t", b._last_sample_t)
            code.append(f"{o[0]} = pl_sample_hold({_lit(b.sample_period)}, "
                        f"&{p}s, &{p}t, {a('input_value')}, {a('t')});")
        elif cls == "TransferFunction":
            num, den = list(b.num), list(b.den)
            xh, yh = list(b._x_hist), list(b._y_hist)
            st("xh", xh, max(len(xh), len(num)))
            st("yh", yh, max(len(yh), len(den), 1))
            state.append((f"int b{i}_nx; int b{i}_ny;",
                          f"s->b{i}_nx = {len(xh)}; s->b{i}_ny = {len(yh)};"))
            code.append(
                f"{{ static const double num[{max(len(num), 1)}] = "
                f"{_arr(num or [0.0])}; static const double den["
                f"{max(len(den), 1)}] = {_arr(den or [0.0])}; {o[0]} = "
                f"pl_transfer_function(num, {len(num)}, den, {len(den)}, "
                f"{p}xh, &{p}nx, {p}yh, &{p}ny, {a('x')}); }}")
        elif cls == "StateMachine":
            mode = {"toggle": 0, "level": 1, "sr_latch": 2}[str(b.mode).lower()]
            st("s", b.state)
            st("p", b._trigger_prev)
            code.append(
                f"{o[0]} = pl_state_machine({mode}, {_lit(b.threshold)}, "
                f"{_lit(b.high_value)}, {_lit(b.low_value)}, &{p}s, &{p}p, "
                f"{a('trigger', 0.0)}, {a('set_', 0.0)}, {a('reset_', 0.0)});")
        elif cls == "OpAmp":
            code.append(f"{o[0]} = pl_op_amp({_lit(b.gain)}, "
                        f"{_lit(b.rail_min)}, {_lit(b.rail_max)}, "
                        f"{a('in_pos')}, {a('in_neg')});")
        elif cls == "LookupTable1D":
            n = len(b.xs)
            code.append(
                f"{{ static const double xs[{max(n, 1)}] = "
                f"{_arr(list(b.xs) or [0.0])}; static const double ys["
                f"{max(n, 1)}] = {_arr(list(b.ys) or [0.0])}; {o[0]} = "
                f"pl_lookup_1d(xs, ys, {n}, {a('x')}); }}")
        elif cls == "MathBlock":
            ops = ("add", "sub", "mul", "div", "abs", "neg", "sqrt", "pow2")
            code.append(f"{o[0]} = pl_math({ops.index(str(b.op).lower())}, "
                        f"{a('a')}, {a('b', 0.0)});")
        elif cls == "SignalMux":
            xs = tup("inputs")
            # a non-finite selector raises in the kernel; here it holds
            code.append(
                f"{{ const double x[{len(xs)}] = {{{', '.join(xs)}}}; "
                f"const double sel = {a('selector')}; if (isfinite(sel)) "
                f"{o[0]} = x[pl_mux_index({len(xs)}, sel)]; }}")
        elif cls == "SignalDemux":
            x = a("x")
            code += [f"{oo} = {x};" for oo in o]
        elif cls == "PwmGenerator":
            code.append(f"{o[0]} = pl_pwm({_lit(b.frequency)}, "
                        f"{_lit(b.phase)}, {a('duty')}, {a('t')});")
        elif cls == "SpaceVectorModulator":
            code.append(f"pl_svm({_lit(b.v_dc)}, {a('v_alpha')}, "
                        f"{a('v_beta')}, &{o[0]}, &{o[1]}, &{o[2]});")
        elif cls == "ClarkeTransform":
            code.append(f"pl_clarke({a('a')}, {a('b')}, {a('c')}, "
                        f"&{o[0]}, &{o[1]}, &{o[2]});")
        elif cls == "InverseClarkeTransform":
            code.append(f"pl_inverse_clarke({a('alpha')}, {a('beta')}, "
                        f"{a('zero', 0.0)}, &{o[0]}, &{o[1]}, &{o[2]});")
        elif cls == "ParkTransform":
            code.append(f"pl_park({a('alpha')}, {a('beta')}, {a('theta')}, "
                        f"&{o[0]}, &{o[1]});")
        elif cls == "InverseParkTransform":
            code.append(f"pl_inverse_park({a('d')}, {a('q')}, {a('theta')}, "
                        f"&{o[0]}, &{o[1]});")
        elif cls == "PLL":
            st("th", b.theta_hat)
            st("w", b.omega_hat)
            st("i", b._integ)
            code.append(
                f"pl_pll({_lit(b.f_nominal)}, {_lit(b.Kp)}, {_lit(b.Ki)}, "
                f"&{p}th, &{p}w, &{p}i, {a('v_alpha')}, {a('v_beta')}, "
                f"{a('dt')}); {o[0]} = {p}th; {o[1]} = {p}w;")
            if len(o) == 3:
                code.append(f"{o[2]} = {p}w / (2.0 * 3.141592653589793);")
        elif cls == "DelayBlock":
            n = int(b.samples)
            if n < 1:
                raise _refuse(f"block '{spec.name}': DelayBlock needs "
                              "samples >= 1")
            buf = list(b._buf) if len(b._buf) == n \
                else [b.initial_value] * n
            st("q", buf, n)
            state.append((f"int b{i}_h;", f"s->b{i}_h = 0;"))
            code.append(
                f"{{ const double x = {a('x')}; {o[0]} = {p}q[{p}h]; "
                f"{p}q[{p}h] = x; {p}h = ({p}h + 1) % {n}; }}")
        else:
            raise _refuse(f"block '{spec.name}': no C rendering for {cls}")
        steps[g].append(f"    /* {spec.name}: {cls} */\n    "
                        + "\n    ".join(code))

    ng = len(rates)
    P = prefix
    laws = _laws_source()
    body = [
        f"/* generated by pulsim.export_c: {len(chain.blocks)} blocks, "
        f"{ng} rate(s). Compile without FMA contraction "
        "(-ffp-contract=off) for kernel bit identity. */",
        "#include <stddef.h>", "#include <stdint.h>", laws, "",
        "/* in[k]: " + ", ".join(f"{k}={n}" for k, n in enumerate(inputs))
        + " */" if inputs else "/* no external inputs */",
        "/* sig[j]: " + ", ".join(f"{j}={n}" for j, n in enumerate(signals))
        + " */",
        "typedef struct {",
        f"    double sig[{max(len(signals), 1)}];",
        f"    uint64_t ticks[{ng}];",
        *("    " + d for d, _ in state),
        f"}} {P}_state;", "",
        f"static const double {P}_period[{ng}] = "
        f"{_arr(r.period for r in rates)};",
        f"static const double {P}_offset[{ng}] = "
        f"{_arr(r.offset for r in rates)};", "",
        f"void {P}_init({P}_state* s) {{",
        "    size_t j;",
        f"    for (j = 0; j < {max(len(signals), 1)}; ++j) s->sig[j] = 0.0;",
        f"    for (j = 0; j < {ng}; ++j) s->ticks[j] = 0;",
        *("    " + c for _, c in state), "}", "",
        f"size_t {P}_state_size(void) {{ return sizeof({P}_state); }}",
        f"double {P}_get(const {P}_state* s, int j) {{ return s->sig[j]; }}",
        f"uint64_t {P}_ticks(const {P}_state* s, int g) "
        "{ return s->ticks[g]; }",
        "/* instant of the next tick of rate g: offset + k * period, k a "
        "64-bit count (exact as a double up to 2^53) */",
        f"double {P}_next_tick(const {P}_state* s, int g) {{",
        f"    return {P}_offset[g] + (double)s->ticks[g] * {P}_period[g];",
        "}", "",
    ]
    for g in range(ng):
        body += [f"void {P}_step_rate{g}({P}_state* s, const double* in, "
                 "double t) {",
                 "    (void)in; (void)t;", *steps[g],
                 f"    s->ticks[{g}] += 1;", "}", ""]
    if ng == 1:
        n_in, n_out = len(inputs), len(outputs)
        body += [
            "/* pulsim_cblock_step shim (single-rate graph): add_c_block("
            f"lib=...) with {n_in} input(s) and {n_out} output(s) "
            f"[{', '.join(outputs)}] and dt = {rates[0].period!r}. */",
            f"static {P}_state {P}_static;",
            "void* pulsim_cblock_init(int n_in, int n_out) {",
            "    (void)n_in; (void)n_out;",
            f"    {P}_init(&{P}_static);",
            f"    return &{P}_static;", "}",
            "void pulsim_cblock_step(const double* in, int n_in, double* out,"
            " int n_out, double t, double dt, void** state) {",
            f"    {P}_state* s;",
            "    (void)dt;",
            f"    if (n_in != {n_in} || n_out != {n_out}) return;",
            "    if (*state == NULL) *state = pulsim_cblock_init(n_in, n_out);",
            f"    s = ({P}_state*)*state;",
            f"    {P}_step_rate0(s, in, t);",
            *(f"    out[{k}] = s->sig[{sig[o]}];" for k, o in
              enumerate(outputs)),
            "}", ""]
    return CExport(source="\n".join(body), prefix=P, inputs=inputs,
                   outputs=outputs, signals=signals,
                   periods=[r.period for r in rates],
                   offsets=[r.offset for r in rates])


def _laws_source() -> str:
    from ._pulsim import block_laws_source  # type: ignore[import-not-found]
    return block_laws_source()
