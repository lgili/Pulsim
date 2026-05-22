"""Pulsim v2 — mixed-domain block chain executor.

Lets users compose closed-loop controllers, signal-processing pipelines,
or modulators from the building blocks in `v2_control.py` (PIController,
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
        raise ValueError(f"Unrecognised input spec: {spec!r}")


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

    # Internal: cached node→state_idx mapping after `bind(builder)`.
    _node_idx_cache: dict = field(default_factory=dict, repr=False)


# =============================================================================
# MixedDomainBlockChain
# =============================================================================

class MixedDomainBlockChain:
    """Topologically-evaluated list of mixed-domain blocks."""

    def __init__(self) -> None:
        self.blocks: list[BlockSpec] = []
        self.channels: dict[str, Any] = {}
        # node-name → state-vector index (populated by bind()).
        self._node_idx: dict[str, int] = {}
        # The Python integer time-step used by step_observer for "dt"
        # input resolution. The user passes it via `make_step_observer(dt=)`.
        self._dt: float = 0.0

    # -------- authoring API ------------------------------------------------

    def add(self, name: str, block, *, inputs: dict,
              output) -> "MixedDomainBlockChain":
        """Add a block to the chain. `inputs` is a kwargs dict; each
        value follows the `_InputSource.from_spec(...)` syntax.
        Returns self for chaining."""
        parsed_inputs = {kw: _InputSource.from_spec(v)
                          for kw, v in inputs.items()}
        self.blocks.append(
            BlockSpec(name=name, block=block,
                       inputs=parsed_inputs, output=output)
        )
        return self

    def reset(self) -> None:
        """Reset every block + clear all channels."""
        for spec in self.blocks:
            r = getattr(spec.block, "reset", None)
            if callable(r):
                try:
                    r()
                except TypeError:
                    pass  # block.reset() requires args we don't know about
        self.channels = {}

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

    # -------- evaluation ---------------------------------------------------

    def update(self, t: float, x) -> None:
        """Evaluate every block in order. Outputs are stored in
        `self.channels` keyed by the block's `output` field."""
        for spec in self.blocks:
            kwargs = {}
            for kw, src in spec.inputs.items():
                if src.kind == "const":
                    kwargs[kw] = src.value
                elif src.kind == "time":
                    kwargs[kw] = float(t)
                elif src.kind == "dt":
                    kwargs[kw] = float(self._dt)
                elif src.kind == "node":
                    idx = self._node_idx[src.value]
                    kwargs[kw] = float(x[idx])
                elif src.kind == "channel":
                    kwargs[kw] = self.channels.get(src.value, 0.0)
            # Some blocks expect special kwarg "inputs" as a tuple of
            # already-resolved values — accept either pattern.
            try:
                result = spec.block.update(**kwargs)
            except TypeError:
                # Maybe the block's update() takes positional only or
                # a different convention. Re-raise with context.
                raise
            # Store result.
            if isinstance(spec.output, (list, tuple)):
                if not isinstance(result, tuple):
                    raise ValueError(
                        f"chain block '{spec.name}' output is a tuple of "
                        f"names but the block returned a scalar")
                if len(result) != len(spec.output):
                    raise ValueError(
                        f"chain block '{spec.name}' returned "
                        f"{len(result)} values but {len(spec.output)} "
                        f"output names were specified")
                for ch, v in zip(spec.output, result):
                    self.channels[ch] = v
            else:
                self.channels[spec.output] = result

    # -------- callback factories ------------------------------------------

    def make_step_observer(self, builder, *, dt: float
                              ) -> Callable[[float, Any], None]:
        """Build the `step_observer(t, x)` callback to pass to
        `simulate(...)`. Binds the chain to `builder` and stashes `dt`
        for blocks that need it."""
        self.bind(builder)
        self._dt = float(dt)

        def observer(t, x):
            self.update(t, x)

        return observer

    def make_pwm_switch_fn(self, channel: str, *,
                              num_switches: int,
                              switch_idx: int = 0):
        """Build a `switch_fn(t)` that toggles `switch_idx` based on the
        binary state of `channel` (treats >0.5 as ON). Use for chains
        whose final block is a PwmGenerator."""
        # Import here to avoid a circular import at module load time.
        from . import v2 as _v2_mod

        def switch_fn(t):  # noqa: ARG001 — t is part of the switch_fn contract
            del t
            m = _v2_mod.SwitchStateMask(num_switches)
            if self.channels.get(channel, 0.0) > 0.5:
                m.set(switch_idx, True)
            return m

        return switch_fn

    def make_multi_pwm_switch_fn(self, channels,
                                       *, num_switches: int):
        """Same as `make_pwm_switch_fn` but for N channels driving N
        switches. `channels` is a sequence of channel names mapped to
        switch indices 0..N-1.

        Returns a switch_fn that sets bit i high iff channels[i] > 0.5.
        """
        from . import v2 as _v2_mod
        chan_list = list(channels)

        def switch_fn(t):  # noqa: ARG001 — t is part of the switch_fn contract
            del t
            m = _v2_mod.SwitchStateMask(num_switches)
            for i, ch in enumerate(chan_list):
                if self.channels.get(ch, 0.0) > 0.5:
                    m.set(i, True)
            return m

        return switch_fn

    def get(self, channel: str, default: float = 0.0):
        """Read the latest value on a channel. Useful for debugging."""
        return self.channels.get(channel, default)


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
    from . import v2_control

    type_to_class = {
        "gain": v2_control.Gain,
        "sum": v2_control.Sum,
        "subtract": v2_control.Subtract,
        "math_block": v2_control.MathBlock,
        "integrator": v2_control.Integrator,
        "differentiator": v2_control.Differentiator,
        "transfer_function": v2_control.TransferFunction,
        "state_machine": v2_control.StateMachine,
        "op_amp": v2_control.OpAmp,
        "limiter": v2_control.Limiter,
        "delay_block": v2_control.DelayBlock,
        "pwm_generator": v2_control.PwmGenerator,
        "space_vector_modulator": v2_control.SpaceVectorModulator,
        "clarke_transform": v2_control.ClarkeTransform,
        "inverse_clarke_transform": v2_control.InverseClarkeTransform,
        "park_transform": v2_control.ParkTransform,
        "inverse_park_transform": v2_control.InverseParkTransform,
        "pll": v2_control.PLL,
        "signal_mux": v2_control.SignalMux,
        "signal_demux": v2_control.SignalDemux,
        "pi_controller": v2_control.PIController,
        "pid_controller": v2_control.PIDController,
        "comparator": v2_control.Comparator,
        "rate_limiter": v2_control.RateLimiter,
        "sample_hold": v2_control.SampleHold,
        "first_order_low_pass": v2_control.FirstOrderLowPass,
        "lookup_table_1d": v2_control.LookupTable1D,
        "moving_average_filter": v2_control.MovingAverageFilter,
    }
    cls = type_to_class.get(block_type)
    if cls is None:
        raise ValueError(f"Unknown controllers type {block_type!r}; "
                          f"available: {sorted(type_to_class)}")
    return cls(**params)
