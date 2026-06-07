"""Custom-code block ("C block") — a PSIM-style sampled subsystem.

A C block reads a user-chosen set of circuit signals (its *inputs*), runs
user code at a user-chosen sample time, and drives a user-chosen set of
signals back into the circuit (its *outputs*), held zero-order between
block steps.

Phase 1 (this module) supports the **Python** step function on the
fixed-step (**PWL**) engine. The block rides Pulsim's existing per-step
hooks — a throttled ``step_observer`` reads inputs + runs the user code,
and a ``b_extra_fn`` injects the held outputs into controlled sources —
so no kernel change is required. C / C++ delivery (compiled shared
library + inline source) and the YAML/GUI surfaces land in later phases.

Wires
-----
Inputs (read from the circuit):
  * ``("v", node)``    — voltage at ``node``
  * ``("i", branch)``  — current through ``branch`` (inductor or source)

Outputs (drive the circuit), each creates one controlled source:
  * ``("v", n_pos, n_neg)`` — controlled **voltage** source
  * ``("i", n_pos, n_neg)`` — controlled **current** source

Usage
-----
::

    b = p.CircuitBuilder()
    # ... build the power stage ...
    def step(t, dt, inp, out, state):
        out[0] = 0.5 * inp[0]          # your control law
    p.add_c_block(b,
                  inputs=[("v", "vout")],
                  outputs=[("v", "ctrl", "gnd")],
                  dt=1e-4, fn=step)
    res = p.simulate(b, t_end=..., dt=2e-6)   # block runs automatically
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

__all__ = ["CBlockHandle", "add_c_block"]

# Sign of the b_extra injection into a source's augmented row, i.e. how
# the per-step setpoint maps to the imposed voltage/current. The kernel's
# source-row convention negates the RHS. Validated against the kernel by
# tests/test_c_block.py.
_V_SRC_SIGN = -1.0
_I_SRC_SIGN = -1.0

_V_KINDS = ("v", "V", "node", "voltage")
_I_KINDS = ("i", "I", "current")


@dataclass
class CBlockHandle:
    """Live handle for an added C block (returned by :func:`add_c_block`).

    ``outputs`` and ``state`` are updated in place each block step, so the
    caller can read them between / after simulation for diagnostics.
    """
    name: str
    n_in: int
    n_out: int
    dt_block: float
    state: dict
    inputs: np.ndarray            # last sampled inputs
    outputs: np.ndarray           # last computed outputs (held, ZOH)
    step_observer: Callable[[float, Any], None]
    b_extra_fn: Callable[[float], list]
    n_fires: int = 0


def _resolve_input(builder, wire) -> int:
    """Resolve an input wire to a state-vector index."""
    if not isinstance(wire, (tuple, list)) or len(wire) != 2:
        raise ValueError(
            f"c_block input wire must be ('v', node) or ('i', branch); "
            f"got {wire!r}")
    kind, name = wire[0], wire[1]
    if kind in _V_KINDS:
        return int(builder.node_id_of(name))
    if kind in _I_KINDS:
        bid = int(builder.branch_id_of(name))
        for fn_name in ("branch_var_id_for_inductor",
                        "branch_var_id_for_source"):
            try:
                idx = int(getattr(builder.pool, fn_name)(bid, builder.graph))
            except Exception:  # noqa: BLE001 — wrong family, try the next
                continue
            if idx >= 0:
                return idx
        raise ValueError(
            f"c_block input ('i', {name!r}): no current state variable — "
            f"only inductor and source currents are reconstructible.")
    raise ValueError(
        f"c_block input wire kind {kind!r} not in {_V_KINDS + _I_KINDS}")


def _add_output_source(builder, wire, src_name):
    """Create the controlled source for an output wire and return an
    injector ``(kind, *rows)`` describing how to write its value into
    ``b_extra``."""
    if not isinstance(wire, (tuple, list)) or len(wire) != 3:
        raise ValueError(
            f"c_block output wire must be ('v', n+, n-) or ('i', n+, n-); "
            f"got {wire!r}")
    kind, n_pos, n_neg = wire[0], wire[1], wire[2]
    if kind in _V_KINDS:
        builder.add_voltage_source(src_name, n_pos, n_neg, 0.0)
        bid = int(builder.branch_id_of(src_name))
        row = int(builder.pool.branch_var_id_for_source(bid, builder.graph))
        return ("v", row)
    if kind in _I_KINDS:
        builder.add_current_source(src_name, n_pos, n_neg, 0.0)
        # A current source has no augmented unknown; its value lands in
        # the two node KCL rows of the RHS. Ground (id < 0) is skipped.
        r_pos = int(builder.node_id_of(n_pos))
        r_neg = int(builder.node_id_of(n_neg))
        return ("i", r_pos, r_neg)
    raise ValueError(
        f"c_block output wire kind {kind!r} not in {_V_KINDS + _I_KINDS}")


def add_c_block(builder,
                inputs: Sequence,
                outputs: Sequence,
                *,
                dt: float,
                fn: Callable[..., Any],
                name: str = "CBLK",
                state: Optional[dict] = None,
                sim_dt: Optional[float] = None) -> CBlockHandle:
    """Attach a custom-code block to ``builder`` (Python step fn, PWL).

    Parameters
    ----------
    builder
        The :class:`CircuitBuilder` (call after the circuit topology is
        complete — input wires and output sources are resolved now).
    inputs
        Input wires: each ``("v", node)`` or ``("i", branch)``.
    outputs
        Output wires: each ``("v", n+, n-)`` or ``("i", n+, n-)``; each
        creates one controlled source.
    dt
        Block sample time (s). The step fn fires at ``t=0`` and every
        ``dt`` thereafter; outputs are held (ZOH) in between. Must be
        ``>= sim dt`` (warns + clamps otherwise; pass ``sim_dt`` so the
        clamp check can run at add time).
    fn
        Step function ``fn(t, dt, inp, out, state)`` where ``inp``/``out``
        are length-N/M numpy buffers and ``state`` is a persistent dict.
        It may write ``out`` in place or ``return`` a length-M sequence.
    name
        Block name; output sources are named ``{name}_out{k}``.
    state
        Optional initial state dict (persists across steps).

    Returns
    -------
    CBlockHandle
        Registered on ``builder._c_blocks``; :func:`simulate` picks it up
        automatically — just call ``simulate(builder, ...)``.
    """
    inputs = list(inputs)
    outputs = list(outputs)
    n_in, n_out = len(inputs), len(outputs)
    if dt <= 0.0:
        raise ValueError(f"c_block dt must be > 0; got {dt}")
    if sim_dt is not None and dt < sim_dt:
        warnings.warn(
            f"c_block {name!r}: dt={dt:g}s < simulation dt={sim_dt:g}s; "
            f"clamping the block to run every simulation step.")
        dt = float(sim_dt)

    in_locs = [_resolve_input(builder, w) for w in inputs]
    injectors = [_add_output_source(builder, w, f"{name}_out{k}")
                 for k, w in enumerate(outputs)]
    # state_size must be read AFTER the output sources are added.
    state_size = int(builder.pool.state_size(builder.graph))

    user_state: dict = dict(state) if state else {}
    inp_buf = np.zeros(n_in, dtype=float)
    out_buf = np.zeros(n_out, dtype=float)
    fire = {"last_t": None, "n": 0}

    def step_observer(t: float, x) -> None:
        last_t = fire["last_t"]
        if last_t is not None and (t - last_t) < dt - 1e-12:
            return  # ZOH — hold previous outputs
        for k, loc in enumerate(in_locs):
            inp_buf[k] = float(x[loc])
        ret = fn(float(t), float(dt), inp_buf, out_buf, user_state)
        if ret is not None:
            ret_arr = np.asarray(ret, dtype=float).ravel()
            if ret_arr.size != n_out:
                raise ValueError(
                    f"c_block {name!r}: step fn returned {ret_arr.size} "
                    f"values but the block has {n_out} outputs")
            out_buf[:] = ret_arr
        fire["last_t"] = t
        fire["n"] += 1

    def b_extra_fn(t: float) -> list:
        vec = [0.0] * state_size
        for k, inj in enumerate(injectors):
            val = float(out_buf[k])
            if inj[0] == "v":                       # voltage source: aug row
                vec[inj[1]] += _V_SRC_SIGN * val
            else:                                    # current source: node rows
                r_pos, r_neg = inj[1], inj[2]
                if 0 <= r_pos < state_size:
                    vec[r_pos] += _I_SRC_SIGN * val
                if 0 <= r_neg < state_size:
                    vec[r_neg] -= _I_SRC_SIGN * val
        return vec

    handle = CBlockHandle(
        name=name, n_in=n_in, n_out=n_out, dt_block=float(dt),
        state=user_state, inputs=inp_buf, outputs=out_buf,
        step_observer=step_observer, b_extra_fn=b_extra_fn)

    # Keep n_fires in sync for diagnostics.
    def _wrapped_observer(t, x, _orig=step_observer):
        _orig(t, x)
        handle.n_fires = fire["n"]
    handle.step_observer = _wrapped_observer

    if not hasattr(builder, "_c_blocks"):
        try:
            builder._c_blocks = []
        except Exception:  # noqa: BLE001 — builder may forbid attrs
            raise RuntimeError(
                "add_c_block: this builder does not accept dynamic "
                "attributes; cannot register the block.")
    builder._c_blocks.append(handle)
    return handle
