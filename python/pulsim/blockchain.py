"""Pulsim — mixed-domain block chain executor.

Lets users compose closed-loop controllers, signal-processing pipelines,
or modulators from the building blocks in `control.py` (PIController,
Clarke/Park, PLL, SVM, …) and run them inside the v2 simulator via a
single `step_observer(t, x)` callback.

Two authoring paths exist:

  1. Python — build the chain procedurally with `chain.add(...)`.
  2. YAML — parse a `controllers:` section from a YAML file via
     `parse_block_chain(yaml_dict, builder)`. The same chain object
     is produced either way.

The chain reads input signals from:
  * Constants                                  `from_value=12.0`
  * Node voltages in the state vector          `from_node="vout"`
  * Outputs of other blocks in the same chain  `from_channel="error"`
  * Simulation time                            `from_time=True`
  * Simulation dt                              `from_dt=True`

And writes its outputs to **channels** (named float buckets). Helper
factories convert channel values to:
  * `switch_fn(t)` masks via `chain.make_pwm_switch_fn(channel, ...)`
  * `b_extra_fn(t)` overlays via `chain.make_b_extra_fn(channel, ...)`

Example — closed-loop buck with software PI:

    chain = p.MixedDomainBlockChain()
    chain.add("sub",  p.Subtract(),
                inputs=dict(a=12.0, b="vout"),  # setpoint − measurement
                output="error")
    chain.add("pi",   p.PIController(Kp=0.05, Ki=200,
                                       output_min=0.05, output_max=0.95),
                inputs=dict(setpoint=12.0, measured="vout", dt="dt"),
                output="duty")
    chain.add("pwm",  p.PwmGenerator(frequency=100e3),
                inputs=dict(duty="duty", t="time"),
                output="gate")

    res = p.simulate(b, t_end=3e-3, dt=1e-7,
                       switch_fn=chain.make_pwm_switch_fn("gate", n_sw=3),
                       step_observer=chain.make_step_observer(b, dt=1e-7))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


__all__ = [
    "BlockSpec",
    "Rate",
    "PythonBlock",
    "MixedDomainBlockChain",
    "parse_block_chain",
]


# =============================================================================
# Input-source descriptors
# =============================================================================

@dataclass
class _InputSource:
    """How a block's `update()` kwarg is resolved at each step."""
    kind: str         # "const" | "node" | "channel" | "time" | "dt"
    value: Any = None

    @classmethod
    def from_spec(cls, spec):
        """Parse a compact spec written in the user-friendly syntax.

        Accepted forms:
          12.0                 — constant
          "vout"               — node voltage
          "channel:my_signal"  — output of another block
          "node:vout"          — explicit node ref
          "time"               — current simulation time
          "dt"                 — current step size
        """
        if isinstance(spec, (int, float)):
            return cls(kind="const", value=float(spec))
        if isinstance(spec, str):
            if spec == "time":
                return cls(kind="time")
            if spec == "dt":
                return cls(kind="dt")
            if spec.startswith("channel:"):
                return cls(kind="channel", value=spec[len("channel:"):])
            if spec.startswith("node:"):
                return cls(kind="node", value=spec[len("node:"):])
            # Bare string defaults to node name.
            return cls(kind="node", value=spec)
        if isinstance(spec, _InputSource):
            return spec
        if isinstance(spec, (tuple, list)):
            # An N-input block (Sum, SignalMux): one source per element.
            return cls(kind="tuple", value=[cls.from_spec(e) for e in spec])
        raise ValueError(f"Unrecognised input spec: {spec!r}")


@dataclass(frozen=True)
class Rate:
    """Sample rate of a group of control-graph blocks.

    The group ticks at ``t_start + offset + k*period`` (integer ``k``) on
    every engine; ``dt`` passed to its blocks is ``period``.
    ``compute_delay`` says when its outputs reach the circuit: ``0``
    (at the tick), a time ``0 < Td < period`` (at the tick + Td), or
    ``"one_sample"`` (at the next tick).
    """

    period: float
    offset: float = 0.0
    compute_delay: Any = 0.0

    def _delay(self):
        d = self.compute_delay
        if d == "one_sample":
            return 2, 0.0
        if isinstance(d, str):
            raise ValueError(
                f"Rate.compute_delay must be 0, a time or 'one_sample'; "
                f"got {d!r}")
        return (1, float(d)) if float(d) > 0.0 else (0, 0.0)


class PythonBlock:
    """A control-graph block that runs Python: ``fn(**inputs)`` (or
    ``obj.update(**inputs)`` for an object) returns a float, or a tuple
    with one value per output name.

    Each evaluation is one Python call under the GIL, so a graph holding
    one is no longer Python-free per step. Every input is direct
    feedthrough; the block counts as stateful (give it a rate when the
    graph has rates).
    """

    def __init__(self, fn) -> None:
        upd = getattr(fn, "update", None)
        fn = upd if callable(upd) else fn
        if not callable(fn):
            raise TypeError(f"PythonBlock needs a callable; got {fn!r}")
        self.fn = fn

    def _call_with(self, names):
        fn = self.fn

        def call(*values):
            return fn(**dict(zip(names, values)))
        return call


# pulsim blocks whose implementation is Python (PythonBlock adapter).
_PY_LIBRARY = frozenset({"SlidingModeObserver", "FluxMRASObserver"})

# Blocks with no state: they inherit the rate of their inputs.
_STATELESS = frozenset({
    "Gain", "Subtract", "Sum", "MathBlock", "Limiter", "OpAmp",
    "SignalMux", "SignalDemux", "LookupTable1D", "ClarkeTransform",
    "InverseClarkeTransform", "ParkTransform", "InverseParkTransform",
    "SpaceVectorModulator",
})


def _warn_reorders(cxx_chain) -> None:
    """Report, once per build, every consumer that the schedule now
    runs after its producer although it was inserted before it: in the
    insertion-order chain each such pair was a one-step delay."""
    pairs = list(cxx_chain.reorders())
    if pairs:
        import warnings
        warnings.warn(
            "control graph: evaluation order differs from insertion "
            "order (each was a one-step delay before the schedule): "
            + "; ".join(pairs) + ". Insert a DelayBlock to keep a delay.",
            UserWarning, stacklevel=4)


# =============================================================================
# BlockSpec — one entry in the chain
# =============================================================================

@dataclass
class BlockSpec:
    """One block in a `MixedDomainBlockChain`.

    `inputs`  maps the block's `update(...)` kwarg names to input
              sources (constants, nodes, channels, time/dt).
    `output`  is either:
      * A string         — the channel name to write the scalar output
      * A tuple of names — when the block returns a tuple (Clarke,
                            Park, SVM, PLL, Demux), each component is
                            written to the corresponding channel.
    """
    name: str
    block: Any
    inputs: dict       # kwarg → _InputSource
    output: Any        # str OR tuple[str, ...]
    rate: Any = None   # Rate, or None: inherited / the chain's

    # Internal: cached node→state_idx mapping after `bind(builder)`.
    _node_idx_cache: dict = field(default_factory=dict, repr=False)


# =============================================================================
# MixedDomainBlockChain
# =============================================================================

class MixedDomainBlockChain:
    """Topologically-evaluated list of mixed-domain blocks."""

    def __init__(self, rate: "Rate | None" = None) -> None:
        # Default rate of the blocks that declare none (a stateless
        # block inherits its inputs' rate first). No rate anywhere: the
        # chain runs on every solver step with make_step_observer's dt.
        self.rate = rate
        self._sinks: list = []
        self.blocks: list[BlockSpec] = []
        # node-name → state-vector index (populated by bind()).
        self._node_idx: dict[str, int] = {}
        # The Python integer time-step used by step_observer for "dt"
        # input resolution. The user passes it via `make_step_observer(dt=)`.
        self._dt: float = 0.0
        # C++ chain backend — built on demand by `make_step_observer`.
        self._cxx_chain = None

    # -------- authoring API ------------------------------------------------

    def add(self, name: str, block, *, inputs: dict,
              output, rate: "Rate | None" = None) -> "MixedDomainBlockChain":
        """Add a block to the chain. `inputs` is a kwargs dict; each
        value follows the `_InputSource.from_spec(...)` syntax.
        Returns self for chaining."""
        parsed_inputs = {kw: _InputSource.from_spec(v)
                          for kw, v in inputs.items()}
        self.blocks.append(
            BlockSpec(name=name, block=block,
                       inputs=parsed_inputs, output=output, rate=rate)
        )
        return self

    def reset(self) -> None:
        """Reset every block (and the compiled graph's ticks and state)."""
        for spec in self.blocks:
            r = getattr(spec.block, "reset", None)
            if callable(r):
                try:
                    r()
                except TypeError:
                    pass  # block.reset() requires args we don't know about
        if self._cxx_chain is not None:
            self._cxx_chain.reset()

    # -------- binding ------------------------------------------------------

    def bind(self, builder) -> "MixedDomainBlockChain":
        """Resolve every "node:" input to a state-vector index using
        `builder.node_id_of(...)`. Must be called once before using the
        chain in a simulation. Idempotent."""
        for spec in self.blocks:
            for kw, src in spec.inputs.items():
                if src.kind == "node":
                    name = src.value
                    if name not in self._node_idx:
                        try:
                            self._node_idx[name] = int(
                                builder.node_id_of(name))
                        except Exception as exc:
                            raise ValueError(
                                f"chain block '{spec.name}': input '{kw}' "
                                f"refers to unknown node '{name}'") from exc
        return self

    # -------- callback factories ------------------------------------------

    def make_step_observer(self, builder, *, dt: "float | None" = None,
                              use_kernel: bool = True
                              ) -> Callable[[float, Any], None]:
        """Build the `step_observer(t, x)` callback to pass to
        `simulate(...)`. Binds the chain to `builder` and stashes
        `dt` for blocks that need it.

        The chain is compiled to the kernel graph (`CxxBlockChain`):
        no Python call per step unless it holds a `PythonBlock`. A block
        with no kernel implementation, an algebraic loop or a signal
        with two writers raise `ControlGraphBuildError`.

        ``use_kernel=False`` (the Python evaluator) was removed in 2.1
        and raises `DeprecationWarning`.
        """
        if not use_kernel:
            raise DeprecationWarning(
                "make_step_observer(use_kernel=False): the Python evaluator "
                "was removed in Pulsim 2.1; every block runs in the kernel "
                "graph (bit-identical to its Python class). Drop the "
                "argument; record signals with chain.record_channel(name), "
                "and wrap custom Python logic in pulsim.PythonBlock(fn).")
        self.bind(builder)
        has_rate = self.rate is not None or any(
            spec.rate is not None for spec in self.blocks)
        if dt is None and not has_rate:
            raise ValueError(
                "make_step_observer: pass dt= (the chain has no Rate, so "
                "it runs on every solver step with this dt)")
        dt = 0.0 if dt is None else float(dt)
        self._dt = dt

        self._cxx_chain = None
        self._cxx_chain = self._compile_to_cxx(builder)
        # Use the C++ chain's step_observer.
        cxx_observer = self._cxx_chain.make_step_observer(dt)
        # Tag the returned observer so simulate() knows it
        # can use run_transient_with_chain() (the kernel
        # fast path that bypasses the std::function wrap).
        def observer(t, x):
            cxx_observer(t, x)
        observer._cxx_chain = self._cxx_chain  # type: ignore[attr-defined]
        observer._chain_dt = float(dt)  # type: ignore[attr-defined]
        cxx = self._cxx_chain
        if cxx.has_rates:
            # The variable-step engine ticks each group at its
            # period; any tick runs every group due then (C++).
            tick = cxx.make_tick_observer()
            observer._rate_ticks = [  # type: ignore[attr-defined]
                (tick, cxx.group_period(g))
                for g in sorted(self._groups.values())]
            observer._rate_offsets = [  # type: ignore[attr-defined]
                r.offset for r in self._groups]
            observer._rate_delays = [  # type: ignore[attr-defined]
                r._delay()[0] for r in self._groups]
        if self._sinks:
            state_size = int(builder.pool.state_size(builder.graph))
            observer._b_extra_fn = cxx.make_sink_b_extra_fn(state_size)  # type: ignore[attr-defined]
        return observer

    # -------- C++ backend compilation -------------------------------------

    def _compile_to_cxx(self, builder):
        """Translate this Python chain into a `CxxBlockChain`. Each
        block's parameters are read from the Python POD class via
        ``getattr``; each InputSource is mapped to ``CxxInputRef``.

        Returns the populated C++ chain. Raises ValueError if any
        block type isn't (yet) supported.
        """
        from . import _pulsim as _k  # type: ignore[import-not-found]
        ControlGraphBuildError = _k.ControlGraphBuildError
        cxx_chain = _k.CxxBlockChain()

        def _outs(output):
            return [str(o) for o in output] \
                if isinstance(output, (tuple, list)) else [str(output)]

        for spec in self.blocks:
            block = spec.block
            inputs = spec.inputs
            output = spec.output
            cls_name = type(block).__name__

            # Translate inputs. Node references are PRE-RESOLVED here
            # (the closure inside the C++ chain captures InputRef by
            # value, so we can't patch the index later via bind_nodes).
            def ref(name_kw, default=None):
                if isinstance(name_kw, _InputSource):
                    src = name_kw
                elif name_kw not in inputs and default is not None:
                    return _k.CxxInputRef.const_value(float(default))
                else:
                    src = inputs[name_kw]
                if src.kind == "tuple":
                    raise ControlGraphBuildError(
                        f"block '{spec.name}': input '{name_kw}' is a tuple; "
                        f"{cls_name} takes a single source there")
                if src.kind == "const":
                    return _k.CxxInputRef.const_value(float(src.value))
                if src.kind == "channel":
                    return _k.CxxInputRef.channel(str(src.value))
                if src.kind == "node":
                    idx = int(builder.node_id_of(str(src.value)))
                    return _k.CxxInputRef.node_idx(idx)
                if src.kind == "time":
                    return _k.CxxInputRef.time()
                if src.kind == "dt":
                    return _k.CxxInputRef.dt()
                raise ValueError(f"unknown source kind {src.kind}")

            def add_ported(kind, params, refs, _out=output):
                cxx_chain.add_ported(
                    kind, {k: [float(e) for e in v]
                           if isinstance(v, (list, tuple)) else [float(v)]
                           for k, v in params.items()}, refs, _outs(_out))

            if cls_name == "Gain":
                cxx_chain.add_gain(k=float(block.k),
                                      x=ref("x"), output=output)
            elif cls_name == "Subtract":
                cxx_chain.add_subtract(a=ref("a"), b=ref("b"),
                                          output=output)
            elif cls_name == "Sum":
                # Sum.weights, one per input: `inputs=(...)` as the
                # class takes them, or the a / b / c keys in that order.
                srcs = (inputs["inputs"].value if "inputs" in inputs
                        else [inputs[k] for k in ("a", "b", "c")
                              if k in inputs])
                w = [float(v) for v in block.weights]
                if len(w) != len(srcs):
                    raise ControlGraphBuildError(
                        f"block '{spec.name}': Sum has {len(w)} weights "
                        f"but {len(srcs)} inputs")
                add_ported("sum", {"weights": w}, [ref(s_) for s_ in srcs])
            elif cls_name == "PIController":
                cxx_chain.add_pi_controller(
                    Kp=float(block.Kp), Ki=float(block.Ki),
                    output_min=float(block.output_min),
                    output_max=float(block.output_max),
                    setpoint=ref("setpoint"),
                    measured=ref("measured"),
                    dt=ref("dt"),
                    output=output,
                    integrator_state=float(block.integrator_state),
                    prev_error=float(block._prev_error))
            elif cls_name == "PIDController":
                # Python-exact (trapezoidal I, filtered D on the error);
                # the kernel PIDController (D on measurement, tau_d) is
                # a different law and is no longer mapped here.
                add_ported("pid", dict(
                    Kp=block.Kp, Ki=block.Ki, Kd=block.Kd,
                    alpha=block.derivative_alpha,
                    output_min=block.output_min, output_max=block.output_max,
                    integrator_state=block.integrator_state,
                    derivative_state=block.derivative_state,
                    prev_error=block._prev_error),
                    [ref("setpoint"), ref("measured"), ref("dt")])
            elif cls_name == "FirstOrderLowPass":
                add_ported("first_order_low_pass",
                           dict(tau=block.tau, state=block.state),
                           [ref("input_value"), ref("dt")])
            elif cls_name == "Integrator":
                # Python-exact: the kernel Integrator (no anti-windup
                # hold, gain*x*dt) differed on 9559 of 10^4 inputs.
                add_ported("integrator", dict(
                    gain=block.gain, output_min=block.output_min,
                    output_max=block.output_max, state=block.state),
                    [ref("x"), ref("dt")])
            elif cls_name == "Differentiator":
                add_ported("differentiator", dict(
                    alpha=block.alpha, state=block.state,
                    prev_x=block._prev_x), [ref("x"), ref("dt")])
            elif cls_name == "Limiter":
                cxx_chain.add_limiter(
                    min_v=float(block.output_min),
                    max_v=float(block.output_max),
                    x=ref("x"), output=output)
            elif cls_name == "MovingAverageFilter":
                add_ported("ema", dict(alpha=block.alpha, state=block.state),
                           [ref("input_value")])
            elif cls_name == "Comparator":
                add_ported("comparator", dict(
                    threshold=block.threshold, hysteresis=block.hysteresis,
                    output_high=block.output_high,
                    output_low=block.output_low, state=float(block.state)),
                    [ref("input_value")])
            elif cls_name == "RateLimiter":
                add_ported("rate_limiter", dict(
                    slew_up=block.slew_up, slew_down=block.slew_down,
                    state=block.state), [ref("target"), ref("dt")])
            elif cls_name == "SampleHold":
                add_ported("sample_hold", dict(
                    sample_period=block.sample_period, state=block.state,
                    last_t=block._last_sample_t),
                    [ref("input_value"), ref("t")])
            elif cls_name == "TransferFunction":
                add_ported("transfer_function", {
                    "num": list(block.num), "den": list(block.den),
                    "x_hist": list(block._x_hist),
                    "y_hist": list(block._y_hist)}, [ref("x")])
            elif cls_name == "StateMachine":
                modes = {"toggle": 0, "level": 1, "sr_latch": 2}
                mode = modes.get(str(block.mode).lower())
                if mode is None:
                    raise ControlGraphBuildError(
                        f"block '{spec.name}': StateMachine mode "
                        f"{block.mode!r} is not one of {sorted(modes)}")
                add_ported("state_machine", dict(
                    mode=mode, threshold=block.threshold,
                    high_value=block.high_value, low_value=block.low_value,
                    state=block.state, trigger_prev=block._trigger_prev),
                    [ref("trigger", 0.0), ref("set_", 0.0),
                     ref("reset_", 0.0)])
            elif cls_name == "OpAmp":
                add_ported("op_amp", dict(
                    gain=block.gain, rail_min=block.rail_min,
                    rail_max=block.rail_max),
                    [ref("in_pos"), ref("in_neg")])
            elif cls_name == "LookupTable1D":
                add_ported("lookup_table_1d", {
                    "xs": list(block.xs), "ys": list(block.ys)}, [ref("x")])
            elif cls_name == "MathBlock":
                ops = ("add", "sub", "mul", "div", "abs", "neg", "sqrt",
                       "pow2")
                op = str(block.op).lower()
                if op not in ops:
                    raise ControlGraphBuildError(
                        f"block '{spec.name}': MathBlock op {block.op!r} "
                        f"is not one of {list(ops)}")
                add_ported("math", dict(op=ops.index(op)),
                           [ref("a"), ref("b", 0.0)])
            elif cls_name == "SignalMux":
                add_ported("signal_mux", {}, [ref("selector")] + [
                    ref(s_) for s_ in inputs["inputs"].value])
            elif cls_name == "SignalDemux":
                add_ported("signal_demux",
                           dict(n_outputs=int(block.n_outputs)), [ref("x")])
            elif cls_name == "PythonBlock" or cls_name in _PY_LIBRARY:
                # Library blocks implemented in Python run through the
                # same adapter (DECISIONS G11); user classes must be
                # wrapped explicitly.
                pyb = block if cls_name == "PythonBlock" else PythonBlock(block)
                names = list(inputs)
                cxx_chain.add_python_block(
                    pyb._call_with(names), [ref(n) for n in names],
                    _outs(output))
            elif cls_name == "PwmGenerator":
                # Python-exact: the kernel PwmGenerator reads phase as a
                # fraction of the period, the class as seconds.
                add_ported("pwm", dict(frequency=block.frequency,
                                       phase=block.phase),
                           [ref("duty"), ref("t")])
            elif cls_name == "SpaceVectorModulator":
                if not isinstance(output, (tuple, list)) or len(output) != 3:
                    raise ValueError("SVM output must be 3-tuple")
                cxx_chain.add_svm(
                    v_dc=float(block.v_dc),
                    v_alpha=ref("v_alpha"), v_beta=ref("v_beta"),
                    output_a=output[0], output_b=output[1],
                    output_c=output[2])
            elif cls_name == "ClarkeTransform":
                if not isinstance(output, (tuple, list)) or len(output) != 3:
                    raise ValueError("Clarke output must be 3-tuple")
                cxx_chain.add_clarke(
                    a=ref("a"), b=ref("b"), c=ref("c"),
                    output_alpha=output[0], output_beta=output[1],
                    output_zero=output[2])
            elif cls_name == "InverseClarkeTransform":
                if not isinstance(output, (tuple, list)) or len(output) != 3:
                    raise ValueError("InverseClarke output must be 3-tuple")
                zero_ref = ref("zero") if "zero" in inputs \
                            else _k.CxxInputRef.const_value(0.0)
                cxx_chain.add_inverse_clarke(
                    alpha=ref("alpha"), beta=ref("beta"),
                    zero=zero_ref,
                    output_a=output[0], output_b=output[1],
                    output_c=output[2])
            elif cls_name == "ParkTransform":
                if not isinstance(output, (tuple, list)) or len(output) != 2:
                    raise ValueError("Park output must be 2-tuple")
                cxx_chain.add_park(
                    alpha=ref("alpha"), beta=ref("beta"),
                    theta=ref("theta"),
                    output_d=output[0], output_q=output[1])
            elif cls_name == "InverseParkTransform":
                if not isinstance(output, (tuple, list)) or len(output) != 2:
                    raise ValueError("InverseParkTransform output must be 2-tuple")
                cxx_chain.add_inverse_park(
                    d=ref("d"), q=ref("q"), theta=ref("theta"),
                    output_alpha=output[0], output_beta=output[1])
            elif cls_name == "PLL":
                if not isinstance(output, (tuple, list)) or len(output) != 3:
                    raise ValueError("PLL output must be 3-tuple")
                cxx_chain.add_pll(
                    f_nominal=float(block.f_nominal),
                    Kp=float(block.Kp), Ki=float(block.Ki),
                    v_alpha=ref("v_alpha"), v_beta=ref("v_beta"),
                    dt=ref("dt"),
                    output_theta=output[0], output_omega=output[1],
                    output_freq=output[2])
            elif cls_name == "DelayBlock":
                cxx_chain.add_delay(
                    samples=int(block.samples),
                    initial_value=float(block.initial_value),
                    x=ref("x"), output=output)
            else:
                raise ControlGraphBuildError(
                    f"control graph: block '{spec.name}' is a {cls_name}, "
                    "which has no kernel implementation. Wrap custom "
                    "Python logic in pulsim.PythonBlock(fn) (one Python "
                    "call per evaluation).")
            cxx_chain.name_last(spec.name, cls_name)

        # Schedule: direct-feedthrough topological order; an algebraic
        # loop raises ControlGraphBuildError naming its blocks.
        cxx_chain.compile()
        _warn_reorders(cxx_chain)
        self._assign_groups(cxx_chain)
        from .c_block import _I_SRC_SIGN, _V_KINDS, _V_SRC_SIGN
        for channel, wire, name in self._sinks:
            if wire[0] in _V_KINDS:
                bid = int(builder.branch_id_of(name))
                row = int(builder.pool.branch_var_id_for_source(
                    bid, builder.graph))
                cxx_chain.add_source_sink(channel, row, _V_SRC_SIGN)
            else:
                cxx_chain.add_source_sink(
                    channel, int(builder.node_id_of(wire[1])), _I_SRC_SIGN)
                cxx_chain.add_source_sink(
                    channel, int(builder.node_id_of(wire[2])), -_I_SRC_SIGN)
        cxx_chain.compile()
        for name, reserve_n in getattr(self, "_recorded_names", []):
            cxx_chain.record_channel(name, reserve_n)
        return cxx_chain

    def _assign_groups(self, cxx_chain) -> None:
        """Rate of every block: its own; else, if stateless, the fastest
        rate among the blocks writing its inputs; else the chain's. A
        stateful block left without one is a build error."""
        if self.rate is None and all(s.rate is None for s in self.blocks):
            return                      # every-step chain (group 0)
        from ._pulsim import ControlGraphBuildError  # type: ignore[import-not-found]
        writer = {}
        for i, spec in enumerate(self.blocks):
            outs = spec.output if isinstance(spec.output, (tuple, list)) \
                else (spec.output,)
            for o in outs:
                writer[o] = i
        rate = [spec.rate for spec in self.blocks]
        for spec in self.blocks:
            if spec.rate is None and type(spec.block).__name__ == "PwmGenerator":
                # A carrier compared at the controller rate is a
                # sample-and-hold of the gate, not a PWM.
                raise ControlGraphBuildError(
                    f"control graph: PwmGenerator '{spec.name}' in a rated "
                    "chain would compare its carrier only at the controller "
                    "ticks; give it rate=Rate(dt) (the solver step, pwl), "
                    "or gate the switch from the duty outside the graph")
        for _ in range(len(self.blocks)):
            changed = False
            for i, spec in enumerate(self.blocks):
                if rate[i] is not None or \
                        type(spec.block).__name__ not in _STATELESS:
                    continue
                srcs = [e for src in spec.inputs.values()
                        for e in (src.value if src.kind == "tuple" else [src])]
                ups = [rate[writer[src.value]] for src in srcs
                       if src.kind == "channel" and src.value in writer
                       and rate[writer[src.value]] is not None]
                if ups:
                    rate[i] = min(ups, key=lambda r: (r.period, r.offset))
                    changed = True
            if not changed:
                break
        groups: dict = {}
        for i, spec in enumerate(self.blocks):
            r = rate[i] if rate[i] is not None else self.rate
            if r is None:
                raise ControlGraphBuildError(
                    f"control graph: block '{spec.name}' "
                    f"({type(spec.block).__name__}) has no rate: give it "
                    "rate=Rate(...), or give the chain one "
                    "(MixedDomainBlockChain(rate=...))")
            if r not in groups:
                kind, td = r._delay()
                groups[r] = cxx_chain.add_group(float(r.period),
                                                float(r.offset), kind, td)
            cxx_chain.set_block_group(i, groups[r])
        self._groups = groups

    def drive_source(self, builder, channel: str, wire) -> None:
        """Drive a controlled source from signal ``channel``: ``wire`` is
        ``("v", n_plus, n_minus)`` or ``("i", n_plus, n_minus)``, as for
        :func:`add_c_block`. The source is added to ``builder`` now; its
        value is the signal held between ticks (after compute_delay)."""
        from .c_block import _I_KINDS, _V_KINDS
        if not isinstance(wire, (tuple, list)) or len(wire) != 3 \
                or wire[0] not in _V_KINDS + _I_KINDS:
            raise ValueError(
                f"drive_source: wire must be ('v', n+, n-) or ('i', n+, n-);"
                f" got {wire!r}")
        name = f"graph_{channel}_{len(self._sinks)}"
        if wire[0] in _V_KINDS:
            builder.add_voltage_source(name, wire[1], wire[2], 0.0)
        else:
            builder.add_current_source(name, wire[1], wire[2], 0.0)
        # Rows are resolved when the graph is built: the state layout
        # moves while the circuit is still growing.
        self._sinks.append((channel, wire, name))

    @property
    def tick_counts(self) -> list:
        """Ticks delivered per rate group (in order of first use)."""
        if self._cxx_chain is None:
            return []
        return [int(c) for c in self._cxx_chain.tick_counts()[1:]]

    def make_pwm_switch_fn(self, channel: str, *,
                              num_switches: int,
                              switch_idx: int = 0,
                              threshold: float = 0.5):
        """Build a `switch_fn(t)` that toggles `switch_idx` based on the
        binary state of `channel` (treats >`threshold` as ON).

        The returned switch_fn runs entirely in C++ — no GIL acquire
        per kernel step — when the chain has been compiled to C++
        (which :meth:`make_step_observer` does automatically). For a
        100 kHz PWM at dt=100 ns this kills ~20 µs/step of GIL
        handshakes that the equivalent Python code costs.

        Parameters
        ----------
        channel
            Name of the chain channel whose value drives the switch.
        num_switches
            Total switch bits in the circuit
            (``builder.graph.num_switches``).
        switch_idx
            Which switch bit this PWM drives. Default 0.
        threshold
            Channel value comparison threshold. Default 0.5.

        Notes
        -----
        The fast C++ path kicks in only after the C++ chain backend
        exists. The normal flow ::

            observer = chain.make_step_observer(builder, dt=dt)
            sw_fn    = chain.make_pwm_switch_fn(channel, …)
            simulate(builder, …, step_observer=observer, switch_fn=sw_fn)

        does the right thing automatically. If you're driving the
        chain in Python-only mode (no ``make_step_observer`` call —
        rare and almost always slower than the C++ path), this
        method silently falls back to a Python closure.
        """
        # Import here to avoid a circular import at module load time.
        # NOTE: the C++ fast-path (`_pulsim.v2_kernel.make_chain_channel_switch_fn`)
        # added by origin/main relied on a `v2_kernel` submodule that no
        # longer exists in the flat-namespace kernel. The Python fallback
        # below stays correct; re-exposing the C++ helper in the flat
        # `_pulsim` is tracked as a follow-up.
        import pulsim as _v2_mod
        chain_self = self

        def switch_fn(t):  # noqa: ARG001
            del t
            m = _v2_mod.SwitchStateMask(num_switches)
            if chain_self._sink(channel) > threshold:
                m.set(switch_idx, True)
            return m
        return switch_fn

    def make_multi_pwm_switch_fn(self, channels,
                                       *, num_switches: int,
                                       switch_indices=None,
                                       threshold: float = 0.5):
        """Build a `switch_fn(t)` driving multiple switches from
        chain channels.

        Like :meth:`make_pwm_switch_fn` but reads N channels and ORs
        them into a single mask. Used for half-bridge / 3-phase
        topologies where each switch leg has its own gate signal.
        Runs entirely in C++ — no GIL per kernel step.

        Parameters
        ----------
        channels
            Sequence of channel names whose values (>`threshold`)
            drive the corresponding switch bit.
        num_switches
            Total switch count in the circuit
            (``builder.graph.num_switches``).
        switch_indices
            Optional sequence of integer switch indices, parallel to
            ``channels``. If omitted, ``channels[i]`` drives bit ``i``.
            Pass explicit indices when the circuit contains PWL diodes
            (each counted as a switch with internal event tracking)
            to avoid clobbering diode bits.
        threshold
            Channel value comparison threshold. Default 0.5.
        """
        # See note above re: the C++ fast-path removal.
        import pulsim as _v2_mod
        chan_list = [str(c) for c in channels]
        chain_self = self
        if switch_indices is None:
            idx_list = list(range(len(chan_list)))
        else:
            idx_list = [int(i) for i in switch_indices]
            if len(idx_list) != len(chan_list):
                raise ValueError(
                    "switch_indices must be parallel to channels")
        # Python-side switch_fn — the C++ fast-path from the v2 era
        # (`_pulsim.v2_kernel.make_chain_channels_switch_fn`) is parked
        # until we re-expose it in the flat `_pulsim`. The Python path
        # below stays correct.
        def switch_fn(t):  # noqa: ARG001
            del t
            m = _v2_mod.SwitchStateMask(num_switches)
            for ch, idx in zip(chan_list, idx_list):
                if chain_self._sink(ch) > threshold:
                    m.set(idx, True)
            return m
        return switch_fn

    def _sink(self, channel: str) -> float:
        # A gate sees a rate group's output after its compute_delay.
        if self._cxx_chain is None:
            return 0.0
        return self._cxx_chain.sink_value(channel)

    def get(self, channel: str, default: float = 0.0):
        """Read the latest value on a channel (``default`` before the
        graph is compiled). Useful for debugging."""
        if self._cxx_chain is not None:
            return self._cxx_chain.get_channel(channel)
        return default

    # -------- channel logging --------------------------------------------

    def record_channel(self, name: str, reserve_n: int = 0) -> None:
        """Register a channel name for per-step logging.

        After ``simulate(...)`` returns, fetch the trace via
        ``get_channel_history(name)``. Works with both C++ and
        Python backends.

        Parameters
        ----------
        name
            Channel name to record (must be written by some block).
        reserve_n
            Hint for the expected number of samples (the number of
            simulation steps). Pre-allocates internal buffers for
            efficient appending — defaults to 0 (grow as needed).
        """
        # Queued for the kernel graph built by make_step_observer (a
        # name registered before it was dropped: the 2.0 kernel path
        # never recorded it).
        if not hasattr(self, "_recorded_names"):
            self._recorded_names: list = []
        if name not in [n for n, _ in self._recorded_names]:
            self._recorded_names.append((name, int(reserve_n)))
        if self._cxx_chain is not None:
            self._cxx_chain.record_channel(name, int(reserve_n))

    def get_channel_history(self, name: str):
        """Return the recorded history of `name` as a numpy array.
        Empty if not registered via :meth:`record_channel`."""
        import numpy as np
        if self._cxx_chain is not None:
            return self._cxx_chain.get_channel_history(name)
        return np.zeros(0)

    def get_recording_times(self):
        """Return the parallel time vector for the recorded channel
        histories."""
        import numpy as np
        if self._cxx_chain is not None:
            return self._cxx_chain.get_recording_times()
        return np.zeros(0)

    def clear_recordings(self) -> None:
        """Clear all recorded histories (keeps registered names)."""
        if self._cxx_chain is not None:
            self._cxx_chain.clear_recordings()


# =============================================================================
# YAML parser — produces a chain from a parsed YAML dict
# =============================================================================

def parse_block_chain(yaml_obj, builder=None) -> MixedDomainBlockChain:
    """Build a chain from a parsed YAML dict.

    Expected YAML structure (top-level keys):

        controllers:
          - type: subtract
            name: error
            a: 12.0
            b: vout                  # node voltage
            output: error            # channel name
          - type: pi_controller
            name: vloop
            Kp: 0.05
            Ki: 200
            output_min: 0.05
            output_max: 0.95
            setpoint: 12.0
            measured: vout
            dt: dt                   # special: use simulation dt
            output: duty
          - type: pwm_generator
            frequency: 100e3
            duty: channel:duty       # output of the PI above
            t: time
            output: gate

    Returns
    -------
    MixedDomainBlockChain (NOT yet bound to a builder unless `builder`
    is supplied; in that case `chain.bind(builder)` is called).
    """
    chain = MixedDomainBlockChain()
    items = yaml_obj.get("controllers", []) if isinstance(yaml_obj, dict) else yaml_obj
    if not items:
        return chain

    for spec in items:
        if not isinstance(spec, dict):
            raise ValueError(f"controllers entry must be a dict, got {spec!r}")
        block_type = spec.get("type")
        if block_type is None:
            raise ValueError(f"controllers entry missing 'type': {spec!r}")
        name = spec.get("name", block_type)
        params, inputs, output = _split_yaml_spec(block_type, spec)
        block = _instantiate_block(block_type, params)
        chain.add(name=name, block=block, inputs=inputs, output=output)

    if builder is not None:
        chain.bind(builder)
    return chain


# Inputs (vs. params) per block type — kwargs that go into `update()`.
_BLOCK_INPUT_KWARGS: dict[str, tuple[str, ...]] = {
    "gain":                ("x",),
    "sum":                 ("inputs",),
    "subtract":            ("a", "b"),
    "math_block":          ("a", "b"),
    "integrator":          ("x", "dt"),
    "differentiator":      ("x", "dt"),
    "transfer_function":   ("x",),
    "state_machine":       ("trigger", "set_", "reset_"),
    "op_amp":              ("in_pos", "in_neg"),
    "limiter":             ("x",),
    "delay_block":         ("x",),
    "pwm_generator":       ("duty", "t"),
    "space_vector_modulator": ("v_alpha", "v_beta"),
    "clarke_transform":    ("a", "b", "c"),
    "inverse_clarke_transform": ("alpha", "beta", "zero"),
    "park_transform":      ("alpha", "beta", "theta"),
    "inverse_park_transform": ("d", "q", "theta"),
    "pll":                 ("v_alpha", "v_beta", "dt"),
    "signal_mux":          ("selector", "inputs"),
    "signal_demux":        ("x",),
    "pi_controller":       ("setpoint", "measured", "dt"),
    "pid_controller":      ("setpoint", "measured", "dt"),
    "comparator":          ("input_value",),
    "rate_limiter":        ("target", "dt"),
    "sample_hold":         ("input_value", "t"),
    "first_order_low_pass": ("input_value", "dt"),
    "lookup_table_1d":     ("x",),
    "moving_average_filter": ("input_value",),
}


def _split_yaml_spec(block_type: str, spec: dict):
    """Split a YAML block spec into (constructor params, input sources,
    output channel name)."""
    inputs_kws = _BLOCK_INPUT_KWARGS.get(block_type, ())
    # Reserved keys handled outside.
    reserved = {"type", "name", "output"} | set(inputs_kws)

    params = {k: v for k, v in spec.items() if k not in reserved}
    inputs = {kw: spec[kw] for kw in inputs_kws if kw in spec}
    output = spec.get("output", spec.get("name", block_type))
    return params, inputs, output


def _instantiate_block(block_type: str, params: dict):
    """Construct a block instance by type name."""
    # Lazy import to avoid cycles.
    from . import control

    type_to_class = {
        "gain": control.Gain,
        "sum": control.Sum,
        "subtract": control.Subtract,
        "math_block": control.MathBlock,
        "integrator": control.Integrator,
        "differentiator": control.Differentiator,
        "transfer_function": control.TransferFunction,
        "state_machine": control.StateMachine,
        "op_amp": control.OpAmp,
        "limiter": control.Limiter,
        "delay_block": control.DelayBlock,
        "pwm_generator": control.PwmGenerator,
        "space_vector_modulator": control.SpaceVectorModulator,
        "clarke_transform": control.ClarkeTransform,
        "inverse_clarke_transform": control.InverseClarkeTransform,
        "park_transform": control.ParkTransform,
        "inverse_park_transform": control.InverseParkTransform,
        "pll": control.PLL,
        "signal_mux": control.SignalMux,
        "signal_demux": control.SignalDemux,
        "pi_controller": control.PIController,
        "pid_controller": control.PIDController,
        "comparator": control.Comparator,
        "rate_limiter": control.RateLimiter,
        "sample_hold": control.SampleHold,
        "first_order_low_pass": control.FirstOrderLowPass,
        "lookup_table_1d": control.LookupTable1D,
        "moving_average_filter": control.MovingAverageFilter,
    }
    cls = type_to_class.get(block_type)
    if cls is None:
        raise ValueError(f"Unknown controllers type {block_type!r}; "
                          f"available: {sorted(type_to_class)}")
    return cls(**params)
