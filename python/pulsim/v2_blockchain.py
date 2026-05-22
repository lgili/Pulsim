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
        # C++ chain backend — built on demand by `make_step_observer`.
        self._cxx_chain = None

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
        # Channel recording (Python fallback path).
        recorded = getattr(self, "_recorded_names", None)
        if recorded:
            self._py_recording_times.append(float(t))
            for name in recorded:
                self._py_history[name].append(
                    float(self.channels.get(name, 0.0)))

    # -------- callback factories ------------------------------------------

    def make_step_observer(self, builder, *, dt: float,
                              use_kernel: bool = True
                              ) -> Callable[[float, Any], None]:
        """Build the `step_observer(t, x)` callback to pass to
        `simulate(...)`. Binds the chain to `builder` and stashes
        `dt` for blocks that need it.

        When ``use_kernel=True`` (default), the entire chain is
        compiled to the C++ `CxxBlockChain` executor and the step
        observer is a pure-C++ lambda — no Python roundtrip per
        simulation step. The closures, channels, and state all live
        in C++. Falls back to the Python evaluator if any block
        type isn't (yet) supported by the C++ backend.

        Pass ``use_kernel=False`` to force the Python evaluator
        (mostly useful for debugging mismatches between the two
        backends).
        """
        self.bind(builder)
        self._dt = float(dt)

        if use_kernel:
            try:
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
                return observer
            except Exception as exc:  # pragma: no cover — graceful fallback
                import warnings
                warnings.warn(
                    f"BlockChain: C++ backend failed "
                    f"({type(exc).__name__}: {exc}); falling back to "
                    f"Python evaluator", stacklevel=2)
                self._cxx_chain = None

        # Python fallback path.
        def observer(t, x):
            self.update(t, x)
        return observer

    # -------- C++ backend compilation -------------------------------------

    def _compile_to_cxx(self, builder):
        """Translate this Python chain into a `CxxBlockChain`. Each
        block's parameters are read from the Python POD class via
        ``getattr``; each InputSource is mapped to ``CxxInputRef``.

        Returns the populated C++ chain. Raises ValueError if any
        block type isn't (yet) supported.
        """
        from ._pulsim import v2_kernel as _k  # type: ignore[import-not-found]
        cxx_chain = _k.CxxBlockChain()

        for spec in self.blocks:
            block = spec.block
            inputs = spec.inputs
            output = spec.output
            cls_name = type(block).__name__

            # Translate inputs. Node references are PRE-RESOLVED here
            # (the closure inside the C++ chain captures InputRef by
            # value, so we can't patch the index later via bind_nodes).
            def ref(name_kw: str):
                src = inputs[name_kw]
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

            if cls_name == "Gain":
                cxx_chain.add_gain(k=float(block.k),
                                      x=ref("x"), output=output)
            elif cls_name == "Subtract":
                cxx_chain.add_subtract(a=ref("a"), b=ref("b"),
                                          output=output)
            elif cls_name == "Sum":
                cxx_chain.add_sum(
                    w0=float(getattr(block, "w0", 1.0)),
                    w1=float(getattr(block, "w1", 1.0)),
                    w2=float(getattr(block, "w2", 0.0)),
                    a=ref("a") if "a" in inputs else
                       _k.CxxInputRef.const_value(0.0),
                    b=ref("b") if "b" in inputs else
                       _k.CxxInputRef.const_value(0.0),
                    c=ref("c") if "c" in inputs else
                       _k.CxxInputRef.const_value(0.0),
                    output=output)
            elif cls_name == "PIController":
                cxx_chain.add_pi_controller(
                    Kp=float(block.Kp), Ki=float(block.Ki),
                    output_min=float(block.output_min),
                    output_max=float(block.output_max),
                    setpoint=ref("setpoint"),
                    measured=ref("measured"),
                    dt=ref("dt"),
                    output=output)
            elif cls_name == "PIDController":
                cxx_chain.add_pid_controller(
                    Kp=float(block.Kp), Ki=float(block.Ki),
                    Kd=float(block.Kd),
                    tau_d=float(getattr(block, "tau_d", 0.0)),
                    output_min=float(block.output_min),
                    output_max=float(block.output_max),
                    setpoint=ref("setpoint"),
                    measured=ref("measured"),
                    dt=ref("dt"),
                    output=output)
            elif cls_name == "FirstOrderLowPass":
                cxx_chain.add_first_order_lpf(
                    tau=float(block.tau),
                    input_value=ref("input_value"),
                    dt=ref("dt"),
                    output=output)
            elif cls_name == "Integrator":
                cxx_chain.add_integrator(
                    gain=float(block.gain),
                    output_min=float(block.output_min),
                    output_max=float(block.output_max),
                    x=ref("x"), dt=ref("dt"),
                    output=output)
            elif cls_name == "Differentiator":
                cxx_chain.add_differentiator(
                    filter_alpha=float(block.filter_alpha),
                    x=ref("x"), dt=ref("dt"),
                    output=output)
            elif cls_name == "Limiter":
                cxx_chain.add_limiter(
                    min_v=float(block.min),
                    max_v=float(block.max),
                    x=ref("x"), output=output)
            elif cls_name == "MovingAverageFilter":
                cxx_chain.add_moving_average(
                    window=int(block.window),
                    x=ref("input_value"), output=output)
            elif cls_name == "PwmGenerator":
                cxx_chain.add_pwm_generator(
                    frequency=float(block.frequency),
                    phase=float(getattr(block, "phase", 0.0)),
                    duty=ref("duty"), t=ref("t"),
                    output=output)
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
            else:
                raise ValueError(
                    f"block type {cls_name!r} not yet supported by "
                    f"C++ backend")

        # Bind node references.
        cxx_chain.bind_nodes(builder.node_id_of)
        return cxx_chain

    def make_pwm_switch_fn(self, channel: str, *,
                              num_switches: int,
                              switch_idx: int = 0,
                              threshold: float = 0.5,
                              use_kernel: bool = False):
        """Build a `switch_fn(t)` that toggles `switch_idx` based on the
        binary state of `channel` (treats >`threshold` as ON).

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
        use_kernel
            When ``True`` (RECOMMENDED for high step-rate sims feeding
            a live scope), the returned switch_fn runs entirely in C++
            with no GIL acquire per kernel step — eliminating ~20 µs
            of Python overhead per kernel step compared to the default
            Python path. Requires that this chain was constructed in
            kernel mode (i.e. ``chain.make_step_observer(use_kernel=
            True)`` is also active so the underlying C++ chain exists
            and is the one being stepped by the kernel).
        """
        if use_kernel:
            # Resolve the underlying C++ chain. ``_cxx_chain`` is set
            # on this object lazily by ``make_step_observer(use_kernel=
            # True)``. If absent, build it now so the helper still
            # works when called before the observer.
            cxx_chain = getattr(self, "_cxx_chain", None)
            if cxx_chain is None:
                raise RuntimeError(
                    "make_pwm_switch_fn(use_kernel=True) requires the "
                    "chain's C++ backend to be initialised first. Call "
                    "`chain.make_step_observer(builder, dt=…, "
                    "use_kernel=True)` BEFORE this method so the "
                    "underlying CxxBlockChain exists.")
            from ._pulsim import v2_kernel as _k  # type: ignore[import-not-found]
            return _k.make_chain_channel_switch_fn(
                chain=cxx_chain,
                channel_name=channel,
                num_switches=num_switches,
                switch_idx=switch_idx,
                threshold=threshold,
            )

        # Default — Python switch_fn.
        from . import v2 as _v2_mod
        chain_self = self

        def switch_fn(t):  # noqa: ARG001 — t is part of the switch_fn contract
            del t
            m = _v2_mod.SwitchStateMask(num_switches)
            if chain_self.get(channel, 0.0) > threshold:
                m.set(switch_idx, True)
            return m

        return switch_fn

    def make_multi_pwm_switch_fn(self, channels,
                                       *, num_switches: int,
                                       switch_indices=None):
        """Build a `switch_fn(t)` that drives multiple switches from
        the chain's channels.

        Parameters
        ----------
        channels
            Sequence of channel names whose values (>0.5 = ON) drive
            the corresponding switch bit.
        num_switches
            Total switch count in the circuit (from
            ``builder.graph.num_switches``).
        switch_indices
            Optional sequence of integer switch indices, parallel to
            ``channels``. If omitted, channels[i] drives bit i (the
            default). When the circuit contains PWL diodes (each
            counted as a switch with internal event tracking), pass
            explicit indices to avoid setting diode bits.

        Returns
        -------
        Callable[[float], SwitchStateMask]
            A switch_fn suitable for ``simulate(switch_fn=...)``.
        """
        from . import v2 as _v2_mod
        chan_list = list(channels)
        chain_self = self
        if switch_indices is None:
            idx_list = list(range(len(chan_list)))
        else:
            idx_list = [int(i) for i in switch_indices]
            if len(idx_list) != len(chan_list):
                raise ValueError(
                    "switch_indices must be parallel to channels")

        def switch_fn(t):  # noqa: ARG001 — t is part of the switch_fn contract
            del t
            m = _v2_mod.SwitchStateMask(num_switches)
            for ch, idx in zip(chan_list, idx_list):
                if chain_self.get(ch, 0.0) > 0.5:
                    m.set(idx, True)
            return m

        return switch_fn

    def get(self, channel: str, default: float = 0.0):
        """Read the latest value on a channel. Useful for debugging.

        If the C++ backend is active, reads from there; otherwise
        from the Python-side `channels` dict.
        """
        if self._cxx_chain is not None:
            return self._cxx_chain.get_channel(channel)
        return self.channels.get(channel, default)

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
        # Both paths: queue the name + reservation. The actual recording
        # happens inside the C++ chain (when active) or via a Python
        # fallback observer (when not).
        if not hasattr(self, "_recorded_names"):
            self._recorded_names = []
            self._py_history: dict[str, list] = {}
            self._py_recording_times: list = []
        if name not in self._recorded_names:
            self._recorded_names.append(name)
            self._py_history[name] = []
        if self._cxx_chain is not None:
            self._cxx_chain.record_channel(name, int(reserve_n))

    def get_channel_history(self, name: str):
        """Return the recorded history of `name` as a numpy array.
        Empty if not registered via :meth:`record_channel`."""
        import numpy as np
        if self._cxx_chain is not None:
            return self._cxx_chain.get_channel_history(name)
        return np.asarray(getattr(self, "_py_history", {}).get(name, []),
                            dtype=float)

    def get_recording_times(self):
        """Return the parallel time vector for the recorded channel
        histories."""
        import numpy as np
        if self._cxx_chain is not None:
            return self._cxx_chain.get_recording_times()
        return np.asarray(getattr(self, "_py_recording_times", []),
                            dtype=float)

    def clear_recordings(self) -> None:
        """Clear all recorded histories (keeps registered names)."""
        if self._cxx_chain is not None:
            self._cxx_chain.clear_recordings()
        if hasattr(self, "_py_history"):
            for k in self._py_history:
                self._py_history[k] = []
            self._py_recording_times = []


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
