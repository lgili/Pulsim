"""Pulsim — switchgear & protection helpers (Phase C.4).

Composite device helpers built on top of v2's primitive R/C/switch
devices. Each one is a thin Python wrapper that wires the right
combination of primitives + (optionally) a switch_fn / step_observer
to model behaviour the kernel doesn't represent natively.

  * **RC snubber** (`add_rc_snubber`) — series R + C across a node
    pair. The most common protection device for switching converters.
    Pair it with `p.recommend_snubber(...)` (Phase E.8) to get the
    sized values.

  * **Thyristor** (`make_thyristor_switch_fn`) — gate-triggered
    latching switch with current-zero turn-off. The kernel doesn't
    have a native thyristor device, but with a `switch_fn` +
    `step_observer` pair we can implement the latching behaviour
    purely on the Python side:

        gate goes HIGH while V_AK ≥ V_BO  →  switch latches ON
        i_AK crosses zero (with hysteresis) →  switch latches OFF

    The user provides the gate-trigger signal as a function ``gate_fn
    (t) -> bool`` and the helper returns the matching `switch_fn` +
    `step_observer` pair.

  * **Fuse** (`make_fuse_switch_fn`) — opens irreversibly when the
    integral of i² · t (I²t energy) exceeds a threshold. Models a
    real fuse's thermal melting characteristic. Once blown it stays
    open until simulation end.

These helpers exercise the existing v2 event-detection / step-
observer mechanism — no kernel changes needed.
"""

from __future__ import annotations

from typing import Callable


__all__ = [
    "add_rc_snubber",
    "make_thyristor_switch_fn",
    "make_fuse_switch_fn",
]


# =============================================================================
# RC snubber
# =============================================================================

def add_rc_snubber(builder,
                      *,
                      R: float,
                      C: float,
                      from_node: str,
                      to_node: str,
                      name_prefix: str = "Snub",
                      ) -> None:
    """Add an R-in-series-with-C snubber across two nodes.

    Adds two primitives:
      * Resistor ``{prefix}_R`` from `from_node` to ``{prefix}_mid``.
      * Capacitor ``{prefix}_C`` from ``{prefix}_mid`` to `to_node`.

    Typical use:

        rec = p.recommend_snubber(L_loop_H=200e-6, C_oss_F=200e-12,
                                     V_bus_V=400, I_peak_A=8,
                                     f_sw_Hz=100e3)
        p.add_rc_snubber(builder, R=rec.R_snub_ohm,
                            C=rec.C_snub_F,
                            from_node="vdc",
                            to_node="sw")
    """
    mid = f"{name_prefix}_mid"
    builder.add_resistor(f"{name_prefix}_R", from_node, mid, float(R))
    builder.add_capacitor(f"{name_prefix}_C", mid, to_node, float(C))


# =============================================================================
# Thyristor — gate-triggered latching with current-zero turn-off
# =============================================================================

def make_thyristor_switch_fn(*,
                                 num_switches: int,
                                 switch_idx: int,
                                 gate_fn: Callable[[float], bool],
                                 current_probe: Callable[[float, "object"], float],
                                 v_breakover: float = 0.0,
                                 i_holding: float = 0.0,
                                 ):
    """Return a ``(switch_fn, step_observer)`` pair that models a
    thyristor / SCR.

    The thyristor sits on switch index ``switch_idx`` of the kernel's
    mask. It latches ON when:
      * The gate signal goes HIGH (``gate_fn(t)`` returns True), AND
      * The forward-bias condition is met (always true here; pass a
        non-zero ``v_breakover`` to require V_AK ≥ V_BO via a custom
        wrapper).

    It latches OFF when:
      * The conduction current ``|i_AK|`` falls below ``i_holding``.

    Implementation: a Python closure holds the latch state. The
    `step_observer` updates the latch each step; `switch_fn` then
    reads the latch when the kernel asks for the switch mask.

    Parameters
    ----------
    num_switches
        Total switch count in the circuit.
    switch_idx
        Index of THIS thyristor's switch.
    gate_fn
        ``t -> bool``. True while the gate trigger is asserted.
    current_probe
        ``(t, x) -> float``. The instantaneous anode-cathode current.
        Used to detect current-zero turn-off.
    v_breakover
        Reserved for future use — pass ``0`` (default).
    i_holding
        Current magnitude below which the thyristor turns off.
        Default 0 — turn off exactly at i = 0 (with no hysteresis).

    Returns
    -------
    (switch_fn, step_observer)
        Both should be passed to ``simulate(...)``. To combine with
        OTHER switches in the same circuit, compose via
        ``p.make_combined_switch_fn``.
    """
    # Lazy import to avoid circular dependency.
    import pulsim as _v2

    state = {"latched_on": False}

    def step_observer(t, x):
        if state["latched_on"]:
            # Off condition: current falls below holding.
            if abs(current_probe(t, x)) < i_holding + 1e-12:
                state["latched_on"] = False
        else:
            # On condition: gate trigger asserted.
            if gate_fn(t):
                state["latched_on"] = True

    def switch_fn(t):
        m = _v2.SwitchStateMask(num_switches)
        # Re-evaluate gate at the switch sample (the observer captured
        # the latest latch state already).
        if state["latched_on"] or gate_fn(t):
            m.set(switch_idx, True)
        return m

    return switch_fn, step_observer


# =============================================================================
# Fuse — irreversibly open on I²t > threshold
# =============================================================================

def make_fuse_switch_fn(*,
                            num_switches: int,
                            switch_idx: int,
                            current_probe: Callable[[float, "object"], float],
                            i2t_threshold: float,
                            initial_state: bool = True,
                            ):
    """Return a ``(switch_fn, step_observer)`` pair for a thermal fuse.

    The fuse starts closed (or open if `initial_state=False`) and
    integrates ``i² · dt`` over the simulation. When the integral
    exceeds `i2t_threshold` [A²·s], the fuse opens irreversibly.

    Implementation: a Python closure holds the running integral and
    blown/intact state. The `step_observer` updates them each step.

    Parameters
    ----------
    num_switches
        Total switch count.
    switch_idx
        Index of the fuse switch.
    current_probe
        ``(t, x) -> float`` — instantaneous fuse current.
    i2t_threshold
        Integral threshold (A²·s) above which the fuse blows.
        Typical values:
          * 5×20 mm 5 A glass fuse: ~5 A²·s
          * 10×38 mm 32 A NH cartridge: ~5000 A²·s
    initial_state
        True = fuse intact (closed), False = pre-blown (open).

    Returns
    -------
    (switch_fn, step_observer)
        Pass both to ``simulate(...)``.
    """
    import pulsim as _v2

    state = {
        "blown": (not initial_state),
        "i2t": 0.0,
        "t_last": None,
    }

    def step_observer(t, x):
        if state["blown"]:
            return
        if state["t_last"] is not None:
            dt = float(t) - state["t_last"]
            i = float(current_probe(t, x))
            state["i2t"] += (i * i) * dt
            if state["i2t"] >= i2t_threshold:
                state["blown"] = True
        state["t_last"] = float(t)

    def switch_fn(t):  # noqa: ARG001
        del t
        m = _v2.SwitchStateMask(num_switches)
        if not state["blown"]:
            m.set(switch_idx, True)
        return m

    return switch_fn, step_observer
