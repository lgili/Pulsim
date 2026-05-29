"""Pulsim v2 — composite-topology helpers.

These functions compose multiple ``CircuitBuilder.add_*`` calls into a
single one-shot construction of a common multi-device topology
(bridge rectifier, 3-phase VSI, 3-phase RL load, ...). They mirror
the v1 equivalents (``runtime_circuit.add_bridge_rectifier``,
``add_three_phase_vsi``, ``add_three_phase_rl_load``) but follow the
v2 philosophy of keeping **topology** and **control** separate:

* Topology helpers here build only the power-stage components.
* Switching/gate control is composed by the user via
  :class:`pulsim.v2.MixedDomainBlockChain` + the appropriate
  ``make_*_switch_fn`` builder.

The two layers are deliberately decoupled — that's what lets a
single VSI helper serve open-loop SPWM, closed-loop FOC, current
control, voltage control, and grid-forming use cases without a
parameter explosion on the topology side.

Each function returns a small dataclass with whatever switch /
device IDs the caller needs to wire control to.

Free functions (rather than methods on ``CircuitBuilder``) keep
these implementable in pure Python without re-touching the C++
pybind11 binding for every new helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


__all__ = [
    "BridgeRectifierResult",
    "ThreePhaseVsiResult",
    "ThreePhaseRLLoadResult",
    "BuckResult",
    "BoostResult",
    "FlybackResult",
    "VoltageDoublerBridgeResult",
    "add_bridge_rectifier",
    "add_voltage_doubler_bridge",
    "add_three_phase_vsi",
    "add_three_phase_rl_load",
    "add_buck",
    "add_boost",
    "add_flyback",
]


# =============================================================================
# Result types
# =============================================================================

@dataclass
class BridgeRectifierResult:
    """Output of :func:`add_bridge_rectifier`. Holds the branch IDs of
    the four diodes so the caller can probe individual diode currents
    or stamp losses."""
    diode_branch_ids: List[int] = field(default_factory=list)
    # Names of the auto-created devices, in (D1, D2, D3, D4) order.
    diode_names: List[str] = field(default_factory=list)


@dataclass
class VoltageDoublerBridgeResult:
    """Output of :func:`add_voltage_doubler_bridge`.

    Carries the deterministic device names + handles the caller needs
    to drive the doubler switch externally (via ``switch_fn``) and to
    probe individual diode/capacitor currents.

    Attributes
    ----------
    diode_names, diode_branch_ids
        The four bridge diodes, in ``(D1, D2, D3, D4)`` order — same
        ordering as :class:`BridgeRectifierResult`.
    cap_top_name, cap_top_branch_id
        The upper electrolytic capacitor (``dc_pos`` ↔ ``mid_node``).
    cap_bot_name, cap_bot_branch_id
        The lower electrolytic capacitor (``mid_node`` ↔ ``dc_neg``).
    mid_node
        Name of the auto-created midpoint node between the two caps.
        Useful if you want to monitor the midpoint voltage or wire a
        snubber.
    switch_name, switch_branch_id
        The controlled switch tying ``ac_n`` to ``mid_node``.
    switch_index
        Index into ``SwitchStateMask`` for this switch — the value the
        caller passes to ``mask.set(switch_index, True)`` to enable
        doubler mode. **Convention:** ``True``/``1`` = closed switch
        = doubler mode (low-line input); ``False``/``0`` = open
        switch = bridge mode (high-line input).
    """
    # Bridge diodes — same order as BridgeRectifierResult.
    diode_branch_ids: List[int] = field(default_factory=list)
    diode_names: List[str] = field(default_factory=list)
    # Bus capacitors (top: dc_pos↔mid, bot: mid↔dc_neg).
    cap_top_name: str = ""
    cap_top_branch_id: int = -1
    cap_bot_name: str = ""
    cap_bot_branch_id: int = -1
    mid_node: str = ""
    # Doubler switch (ac_n ↔ mid_node).
    switch_name: str = ""
    switch_branch_id: int = -1
    switch_index: int = -1


@dataclass
class ThreePhaseVsiResult:
    """Output of :func:`add_three_phase_vsi`.

    Attributes
    ----------
    switch_indices
        Six switch indices in the order
        ``[HSa, LSa, HSb, LSb, HSc, LSc]``. Use these as
        ``switch_indices=`` in
        :meth:`MixedDomainBlockChain.make_multi_pwm_switch_fn` to
        wire SPWM/FOC/etc. control to the bridge.
    high_side_switch_indices, low_side_switch_indices
        Convenience views — useful for half-bridge dead-time helpers.
    """
    switch_indices: List[int] = field(default_factory=list)

    @property
    def high_side_switch_indices(self) -> List[int]:
        return self.switch_indices[0::2]

    @property
    def low_side_switch_indices(self) -> List[int]:
        return self.switch_indices[1::2]


@dataclass
class ThreePhaseRLLoadResult:
    """Output of :func:`add_three_phase_rl_load`.

    Attributes
    ----------
    inductor_branch_ids
        Branch IDs of the three (or three line-to-line) inductors,
        in (A or AB, B or BC, C or CA) order. Use as input to
        :meth:`CircuitBuilder.pool.branch_var_id_for_inductor` to
        read per-leg current.
    intermediate_node_ids
        IDs of the auto-created mid nodes between R and L. Useful
        for probing the voltage across just R (loss calc).
    """
    inductor_branch_ids: List[int] = field(default_factory=list)
    intermediate_node_ids: List[int] = field(default_factory=list)


# =============================================================================
# Bridge rectifier
# =============================================================================

def add_bridge_rectifier(builder,
                            name: str,
                            *,
                            ac_a: str,
                            ac_b: str,
                            dc_pos: str,
                            dc_neg: str,
                            g_on: float = 1e3,
                            g_off: float = 1e-9,
                            V_th: float = 0.7,
                            ) -> BridgeRectifierResult:
    """Add a four-diode full-wave bridge rectifier to the builder.

    Topology
    --------
    ::

                   ┌─ D1 ─┐         ┌─ D4 ──┐
            ac_a ──┤       ├── dc_pos       │
                   └─ D3 ─┘         │       │
                                    │       │
                   ┌─ D2 ─┐         │       │
            ac_b ──┤       ├──┘   D3 ↑    D4 ↑
                   └─ D4 ─┘
                                  dc_neg ──┘

    Concretely:
      * D1: ac_a → dc_pos (forward when ac_a is the high-side)
      * D2: ac_b → dc_pos
      * D3: dc_neg → ac_b  (return path)
      * D4: dc_neg → ac_a

    Parameters
    ----------
    builder
        A populated :class:`CircuitBuilder`.
    name
        Prefix for the auto-created diode names. The four diodes are
        registered as ``f"{name}__D1"``, …, ``f"{name}__D4"``.
    ac_a, ac_b
        AC-side input node names.
    dc_pos, dc_neg
        DC-side output node names.
    g_on
        Forward conductance (S) — diode resistance when ON is 1/g_on.
        Default 1e3 (R_on = 1 mΩ).
    g_off
        Reverse leakage conductance (S). Default 1e-9.
    V_th
        Forward voltage threshold (V). Default 0.7.

    Returns
    -------
    BridgeRectifierResult
        Holds the four diode branch IDs (useful for probing per-diode
        current with :meth:`pool.branch_var_id_for_inductor`-style
        lookups via the kernel's branch-current accessors).
    """
    out = BridgeRectifierResult()
    # We compute the four branch IDs by reading the graph BEFORE and
    # AFTER each add_diode call (the builder doesn't return them).
    for tag, frm, to in (
        ("D1", ac_a,   dc_pos),
        ("D2", ac_b,   dc_pos),
        ("D3", dc_neg, ac_b),
        ("D4", dc_neg, ac_a),
    ):
        bid = builder.graph.num_branches
        diode_name = f"{name}__{tag}"
        builder.add_diode(diode_name, frm, to, g_on, g_off, V_th=V_th)
        out.diode_branch_ids.append(int(bid))
        out.diode_names.append(diode_name)
    return out


# =============================================================================
# Voltage-doubler / universal-input bridge rectifier
# =============================================================================

def add_voltage_doubler_bridge(builder,
                                name: str = "rect",
                                *,
                                ac_l: str,
                                ac_n: str,
                                dc_pos: str,
                                dc_neg: str,
                                mid_node: Optional[str] = None,
                                C_top: float = 470e-6,
                                C_bot: float = 470e-6,
                                g_on: float = 1e3,
                                g_off: float = 1e-9,
                                V_th: float = 0.7,
                                sw_g_on: float = 1e3,
                                sw_g_off: float = 1e-9,
                                c_top_v0: Optional[float] = None,
                                c_bot_v0: Optional[float] = None,
                                ) -> VoltageDoublerBridgeResult:
    """Add a universal-input (110/220 VAC) bridge rectifier with a
    controlled voltage-doubler switch.

    The same hardware acts as a full-wave bridge OR a voltage doubler,
    selected by one controlled switch. This is the classic AC-DC
    front-end of PC PSUs, industrial chargers, and universal-input
    adapters — the user code (or a comparator on the input RMS)
    flips the switch and the same DC bus voltage works on both 110 V
    and 220 V mains.

    Topology
    --------
    ::

                    ┌─── D1 ───┐                  ┌──── dc_pos ────┐
            ac_l ───┤          ├──── dc_pos       │                │
                    └─── D3 ───┘                  │              C_top
                                                  ├──── mid ───────┤
                    ┌─── D2 ───┐                  │              C_bot
            ac_n ───┤          ├──── dc_pos       │                │
                    └─── D4 ───┘                  └──── dc_neg ────┘
                    │
                    └──── SW ──── mid

    Diode directions:
      * ``D1``: ``ac_l`` → ``dc_pos``  (positive half-cycle, ac_l up)
      * ``D2``: ``ac_n`` → ``dc_pos``  (negative half-cycle, ac_n up)
      * ``D3``: ``dc_neg`` → ``ac_l``  (return path, ac_l low)
      * ``D4``: ``dc_neg`` → ``ac_n``  (return path, ac_n low)

    Operating modes
    ---------------
    * **SW open (mask bit = ``False``)** — full-wave bridge. C_top and
      C_bot sit in series across the DC bus. Output settles at
      ``V_DC ≈ √2·V_AC_rms`` (the bridge peak), minus 2 diode drops.
      Use this for high-line (e.g. 220 VAC).
    * **SW closed (mask bit = ``True``)** — voltage doubler. SW shorts
      ``ac_n`` to the cap midpoint. **D1 charges C_top on the
      positive half-cycle** via
      ``ac_l → D1 → dc_pos → C_top → mid → SW → ac_n``. **D4 charges
      C_bot on the negative half-cycle** via
      ``ac_n → SW → mid → C_bot → dc_neg → D4 → ac_l``. D2 and D3
      stay reverse-biased throughout. Each cap settles at
      ``≈ √2·V_AC_rms`` (minus one diode drop), so total
      ``V_DC = V_{C_{top}} + V_{C_{bot}} ≈ 2·√2·V_AC_rms``.
      Use this for low-line (e.g. 110 VAC) — the resulting V_DC
      matches what the bridge produces on 220 VAC.

    Parameters
    ----------
    builder
        A populated :class:`CircuitBuilder`.
    name
        Prefix for the auto-created device names. The devices land as
        ``{name}__D1`` … ``{name}__D4``, ``{name}__C_top``,
        ``{name}__C_bot``, ``{name}__SW_dbl``.
    ac_l, ac_n
        AC-side line and neutral nodes. ``ac_n`` is the side that
        connects to the cap midpoint via the doubler switch — pick
        whichever input pole you treat as neutral in your circuit.
    dc_pos, dc_neg
        DC-bus output node names.
    mid_node
        Optional explicit name for the cap-midpoint node. Defaults to
        ``f"{name}__mid"`` when ``None``.
    C_top, C_bot
        Top and bottom bus capacitor values, in farads. Defaults are
        470 µF — typical electrolytics for ≤200 W universal-input
        designs. For doubler mode the two caps see asymmetric currents
        during transients but recover to ≈ equal voltages at
        steady-state; equal C_top / C_bot is the standard choice.
    g_on, g_off, V_th
        Diode on/off conductances (S) and threshold (V). Defaults:
        ``g_on = 1e3`` (R_on ≈ 1 mΩ), ``g_off = 1e-9`` (1 GΩ leakage),
        ``V_th = 0.7 V``.
    sw_g_on, sw_g_off
        Doubler-switch on/off conductances (S). Defaults assume an
        ideal-ish controlled switch (1 mΩ / 1 GΩ).
    c_top_v0, c_bot_v0
        Optional initial cap voltages (V) seeded via
        :meth:`CircuitBuilder.set_initial`. Useful for warm-start
        simulations that skip the cap pre-charge transient. When
        ``None`` (default), the capacitor starts uncharged.

    Returns
    -------
    VoltageDoublerBridgeResult
        Diode + cap + switch metadata. Notable field:
        ``switch_index`` is what you pass to
        ``mask.set(switch_index, True/False)`` to flip modes.

    Examples
    --------
    Build a universal-input front-end and drive the doubler from a
    plain Python flag based on the measured RMS:

    ::

        import pulsim as p

        b = p.CircuitBuilder()
        b.add_sine_voltage_source(
            name="Vac", n_pos="ac_l", n_neg="ac_n",
            v_dc=0.0, v_amplitude=311.0,   # 220 VAC peak (high line)
            frequency=60.0,
        )
        rect = p.add_voltage_doubler_bridge(
            b, name="Rect",
            ac_l="ac_l", ac_n="ac_n",
            dc_pos="vdc", dc_neg="gnd",
        )
        b.add_resistor("R_load", "vdc", "gnd", 100.0)

        # Decide once before t=0 whether to use bridge or doubler:
        is_low_line = False  # set True for 110 VAC operation

        def sw(t):
            m = p.SwitchStateMask(b.graph.num_switches)
            m.set(rect.switch_index, is_low_line)
            return m

        res = p.simulate(b, t_end=0.2, dt=20e-6, switch_fn=sw)

    The switch can also be driven dynamically (e.g. flipped after a
    soft-start interval, or by a comparator on a measured RMS
    channel) — the same ``switch_fn`` pattern that drives every other
    Pulsim PWM helper.

    Notes
    -----
    * **The doubler switch carries the full doubler-mode return
      current** (it's in series with the C_bot charging loop and the
      ac_n side return path). Size ``sw_g_on`` based on the
      conduction-loss budget you'd accept; the default 1 mΩ matches
      typical relay / contactor steady-state R_on.
    * **Cap pre-charge** — both modes have the textbook capacitor
      in-rush at start-up. Add a series NTC or relay-bypass inrush
      limiter externally if you're modelling the AC-side transient.
    * **Mode transitions** — flipping the switch while the bus is
      energised causes a hard discontinuity (the two caps suddenly
      see a different effective topology); model with a small
      `dt_max` around the transition if you care about that
      transient.
    """
    # --- input validation ---
    if C_top <= 0:
        raise ValueError(
            f"add_voltage_doubler_bridge: C_top must be positive "
            f"(got {C_top}). Typical universal-input designs use "
            f"≥ 100 µF.")
    if C_bot <= 0:
        raise ValueError(
            f"add_voltage_doubler_bridge: C_bot must be positive "
            f"(got {C_bot}).")
    if g_on <= 0 or g_off <= 0:
        raise ValueError(
            f"add_voltage_doubler_bridge: diode g_on / g_off must be "
            f"positive (got g_on={g_on}, g_off={g_off}).")
    if sw_g_on <= 0 or sw_g_off <= 0:
        raise ValueError(
            f"add_voltage_doubler_bridge: switch sw_g_on / sw_g_off "
            f"must be positive (got sw_g_on={sw_g_on}, "
            f"sw_g_off={sw_g_off}).")

    if mid_node is None:
        mid_node = f"{name}__mid"

    out = VoltageDoublerBridgeResult()
    out.mid_node = mid_node

    # --- four bridge diodes (same topology + naming as
    #     add_bridge_rectifier — D1/D2 anodes on the AC side,
    #     D3/D4 cathodes on the AC side, dc_pos on top, dc_neg on
    #     the return path).
    for tag, frm, to in (
        ("D1", ac_l,   dc_pos),
        ("D2", ac_n,   dc_pos),
        ("D3", dc_neg, ac_l),
        ("D4", dc_neg, ac_n),
    ):
        bid = builder.graph.num_branches
        diode_name = f"{name}__{tag}"
        builder.add_diode(diode_name, frm, to, g_on, g_off, V_th=V_th)
        out.diode_branch_ids.append(int(bid))
        out.diode_names.append(diode_name)

    # --- series cap stack: dc_pos ─ C_top ─ mid ─ C_bot ─ dc_neg ---
    out.cap_top_name = f"{name}__C_top"
    out.cap_top_branch_id = int(builder.graph.num_branches)
    if c_top_v0 is not None:
        builder.add_capacitor(out.cap_top_name, dc_pos, mid_node,
                                 C_top, c0=float(c_top_v0))
    else:
        builder.add_capacitor(out.cap_top_name, dc_pos, mid_node, C_top)

    out.cap_bot_name = f"{name}__C_bot"
    out.cap_bot_branch_id = int(builder.graph.num_branches)
    if c_bot_v0 is not None:
        builder.add_capacitor(out.cap_bot_name, mid_node, dc_neg,
                                 C_bot, c0=float(c_bot_v0))
    else:
        builder.add_capacitor(out.cap_bot_name, mid_node, dc_neg, C_bot)

    # --- the doubler switch (ac_n ↔ mid) — closed = doubler mode ---
    out.switch_name = f"{name}__SW_dbl"
    out.switch_branch_id = int(builder.graph.num_branches)
    out.switch_index = int(builder.graph.num_switches)
    builder.add_switch(out.switch_name, ac_n, mid_node,
                          g_on=sw_g_on, g_off=sw_g_off)
    return out


# =============================================================================
# Three-phase Voltage Source Inverter
# =============================================================================

def add_three_phase_vsi(builder,
                            name: str,
                            *,
                            vdc_pos: str,
                            vdc_neg: str,
                            out_a: str,
                            out_b: str,
                            out_c: str,
                            R_on: float = 1e-3,
                            R_off: float = 1e9,
                            ) -> ThreePhaseVsiResult:
    """Add the power stage of a 3-phase voltage source inverter (6
    ideal switches arranged as three half-bridges).

    Topology
    --------
    Each phase leg is a complementary pair (HS/LS) between the two
    DC rails, with the leg output going to the AC node. Six switches
    total::

           vdc_pos ─┬───HSa───┬───HSb───┬───HSc───┐
                    │         │         │         │
                  out_a     out_b     out_c
                    │         │         │
                    ├───LSa───┼───LSb───┼───LSc───┘
                    │
           vdc_neg ─┘

    The helper builds **only the topology** — no PWM, no dead-time.
    Wire control via :class:`MixedDomainBlockChain` and one of:

      * :meth:`chain.make_multi_pwm_switch_fn(["ga","gb","gc"],
        num_switches=builder.graph.num_switches,
        switch_indices=result.high_side_switch_indices)`
        for complementary PWM with the low side derived as ¬HS,
      * or three independent gate channels for dead-time-aware SPWM.

    Parameters
    ----------
    builder
        Populated CircuitBuilder.
    name
        Prefix for auto-created switch names. Six switches are
        registered as ``{name}__HSa``, ``{name}__LSa``, …,
        ``{name}__LSc``.
    vdc_pos, vdc_neg
        DC-link positive and negative rail node names.
    out_a, out_b, out_c
        Three-phase AC output node names.
    R_on, R_off
        Switch on/off resistance (Ω). Default 1 mΩ / 1 GΩ.

    Returns
    -------
    ThreePhaseVsiResult
        With ``switch_indices`` = [HSa, LSa, HSb, LSb, HSc, LSc] —
        the order MakeMultiPwmSwitchFn expects when you map gates
        to legs.

    Examples
    --------
    Open-loop SPWM::

        vsi = p.add_three_phase_vsi(b, "INV",
                  vdc_pos="vdc", vdc_neg="gnd",
                  out_a="ua", out_b="ub", out_c="uc")
        chain.add("ga", p.PwmGenerator(frequency=10e3),
                  inputs=dict(duty=0.5, t="time"), output="ga")
        # ... gb, gc shifted 120°, 240° ...
        sw_fn = chain.make_multi_pwm_switch_fn(
            ["ga","gb","gc"],
            num_switches=b.graph.num_switches,
            switch_indices=vsi.high_side_switch_indices)
        # Low-side: composed via a separate complementary observer
        # or a dead-time half-bridge helper.
    """
    g_on = 1.0 / R_on if R_on > 0 else 1e3
    g_off = 1.0 / R_off if R_off > 0 else 1e-9
    out = ThreePhaseVsiResult()
    # Walk in (HS, LS) pairs so the resulting switch_indices list is
    # [HSa, LSa, HSb, LSb, HSc, LSc] — the order most natural for
    # half-bridge dead-time helpers.
    legs = [
        ("HSa", vdc_pos, out_a),  ("LSa", out_a, vdc_neg),
        ("HSb", vdc_pos, out_b),  ("LSb", out_b, vdc_neg),
        ("HSc", vdc_pos, out_c),  ("LSc", out_c, vdc_neg),
    ]
    for tag, frm, to in legs:
        sw_idx = builder.graph.num_switches
        builder.add_switch(f"{name}__{tag}", frm, to,
                              g_on=g_on, g_off=g_off)
        out.switch_indices.append(int(sw_idx))
    return out


# =============================================================================
# Three-phase RL load
# =============================================================================

def add_three_phase_rl_load(builder,
                                 name: str,
                                 *,
                                 node_a: str,
                                 node_b: str,
                                 node_c: str,
                                 node_neutral: str = "gnd",
                                 R: float,
                                 L: float,
                                 topology: str = "star",
                                 unbalance: float = 0.0,
                                 ) -> ThreePhaseRLLoadResult:
    """Add a 3-phase R-L load (Star or Delta connection).

    Each phase leg is an R in series with an L. Intermediate nodes
    between R and L are auto-created (named ``{name}__mid_<leg>``)
    so the inductor's branch row carries the per-leg current — read
    it via ``builder.pool.branch_var_id_for_inductor(branch_id,
    builder.graph)`` and use as a state-vector index.

    Parameters
    ----------
    builder
        Populated CircuitBuilder.
    name
        Prefix for auto-created device + node names.
    node_a, node_b, node_c
        Phase input node names.
    node_neutral
        Common return point in Star topology. Ignored in Delta.
        Default ``"gnd"``.
    R, L
        Per-phase resistance (Ω) and inductance (H).
    topology
        ``"star"`` (Y, default) or ``"delta"`` (Δ).
    unbalance
        Optional asymmetry factor in ``[0, 1)``. Phase B is scaled
        by ``(1-u)``, Phase C by ``(1+u)``. Default 0.0 (balanced).

    Returns
    -------
    ThreePhaseRLLoadResult
        Three inductor branch IDs in (A, B, C) order — or
        (AB, BC, CA) for Delta — plus the three intermediate node IDs
        for voltage probing.
    """
    if R <= 0.0:
        raise ValueError("add_three_phase_rl_load: R must be > 0")
    if L < 0.0:
        raise ValueError("add_three_phase_rl_load: L must be >= 0")
    if not (0.0 <= unbalance < 1.0):
        raise ValueError(
            "add_three_phase_rl_load: unbalance must be in [0, 1)")
    topology = topology.lower()
    if topology not in ("star", "delta"):
        raise ValueError(
            f"add_three_phase_rl_load: topology must be 'star' or "
            f"'delta', got {topology!r}")

    scale_b = 1.0 - unbalance
    scale_c = 1.0 + unbalance

    out = ThreePhaseRLLoadResult()

    def emit_leg(suffix: str, left: str, right: str,
                  r_scale: float, l_scale: float) -> None:
        mid_name = f"{name}__mid_{suffix}"
        # add_node returns the node id — but the v2 builder uses
        # named nodes everywhere, so we just reference it by name and
        # let the builder create-on-first-use.
        builder.add_resistor(
            f"{name}__R_{suffix}", left, mid_name, R * r_scale)
        ind_bid = builder.graph.num_branches
        builder.add_inductor(
            f"{name}__L_{suffix}", mid_name, right, L * l_scale)
        out.inductor_branch_ids.append(int(ind_bid))
        # node id is resolved lazily by node_id_of (created by the
        # add_resistor / add_inductor calls above).
        out.intermediate_node_ids.append(
            int(builder.node_id_of(mid_name)))

    if topology == "star":
        emit_leg("A", node_a, node_neutral, 1.0,     1.0)
        emit_leg("B", node_b, node_neutral, scale_b, scale_b)
        emit_leg("C", node_c, node_neutral, scale_c, scale_c)
    else:
        emit_leg("AB", node_a, node_b, 1.0,     1.0)
        emit_leg("BC", node_b, node_c, scale_b, scale_b)
        emit_leg("CA", node_c, node_a, scale_c, scale_c)
    return out


# =============================================================================
# SMPS topology factories
# =============================================================================
#
# These collapse the 5–8 ``add_*`` calls that a typical DC/DC plant
# needs into one call. They DON'T compose a switch_fn — the caller
# still drives the switch with their own PWM/PI/controller. That's a
# deliberate split: plant and control stay decoupled, the way every
# textbook keeps them.

@dataclass
class BuckResult:
    """Output of :func:`add_buck`. Carries the deterministic device
    names + branch ids the caller needs to wire control / probe
    waveforms."""
    switch_name: str = ""
    switch_branch_id: int = -1
    diode_name: str = ""
    inductor_name: str = ""
    inductor_branch_id: int = -1
    capacitor_name: str = ""
    load_name: str = ""
    suggested_dt: float = 1e-7  # 1/10 of a 1 MHz switching period


@dataclass
class BoostResult:
    """Output of :func:`add_boost`."""
    switch_name: str = ""
    switch_branch_id: int = -1
    diode_name: str = ""
    inductor_name: str = ""
    inductor_branch_id: int = -1
    capacitor_name: str = ""
    load_name: str = ""
    suggested_dt: float = 1e-7


@dataclass
class FlybackResult:
    """Output of :func:`add_flyback`. Holds the primary + secondary
    branch ids so the caller can probe both windings."""
    switch_name: str = ""
    switch_branch_id: int = -1
    primary_name: str = ""
    primary_branch_id: int = -1
    secondary_name: str = ""
    secondary_branch_id: int = -1
    diode_name: str = ""
    capacitor_name: str = ""
    load_name: str = ""
    suggested_dt: float = 1e-7


def _suggested_dt_for(f_sw: float) -> float:
    """Heuristic: aim for ≥ 100 samples per switching period so PWM
    edges and ringing on L/C aren't aliased."""
    if f_sw <= 0:
        return 1e-7
    return 1.0 / (100.0 * f_sw)


def add_buck(builder,
              name: str = "buck",
              *,
              vin_node: str = "vin",
              vout_node: str = "vout",
              gnd_node: str = "gnd",
              V_in: float = 48.0,
              L: float = 100e-6,
              C: float = 100e-6,
              R_load: float = 4.0,
              f_sw: float = 100e3,
              g_on: float = 1e3,
              g_off: float = 1e-9,
              diode_g_on: float = 1e3,
              diode_g_off: float = 1e-6,
              V_th: float = 0.0,
              ) -> BuckResult:
    """Add a synchronous-rectifier-style buck power stage.

    Topology::

        V_in ──HS────┬──── L ──── V_out ──┬── R_load ── gnd
                     │                     │
                     ⊽ D                  ═══ C
                     │                     │
                    gnd                   gnd

    Devices added (with deterministic names):

      * `{name}__Vin`   — DC source `vin_node` ↔ `gnd_node`, voltage `V_in`
      * `{name}__HS`    — ideal switch `vin_node` ↔ `sw_mid`
      * `{name}__D`     — freewheeling diode `gnd_node` ↔ `sw_mid`
      * `{name}__L`     — inductor `sw_mid` ↔ `vout_node`
      * `{name}__C`     — output capacitor `vout_node` ↔ `gnd_node`
      * `{name}__R`     — load resistor `vout_node` ↔ `gnd_node`

    The intermediate `sw_mid` node is auto-named `{name}__sw`.

    The function **does not** wire a PWM driver. The caller drives the
    switch via `simulate(switch_fn=...)` with their own PWM helper
    (see :func:`pulsim.make_pwm_switch_fn`) and may close the loop via
    :class:`pulsim.ClosedLoop` (`bind_pi_to_switch`).

    Parameters
    ----------
    builder
        A :class:`CircuitBuilder`. The function appends to it.
    name
        Prefix for the auto-created device + intermediate node names.
        Default ``"buck"``.
    vin_node, vout_node, gnd_node
        External node names for the source, output, and ground rails.
        Defaults wire the buck to a topology with `vin`/`vout`/`gnd`
        rails ready to connect to upstream / downstream devices.
    V_in, L, C, R_load
        Component values (V / H / F / Ω).
    f_sw
        Nominal switching frequency (Hz). Used only to compute
        `suggested_dt`; the caller picks the actual PWM frequency.
    g_on, g_off
        High-side switch on/off conductances. Default 1 kS / 1 nS
        (R_on = 1 mΩ, R_off = 1 GΩ).
    diode_g_on, diode_g_off, V_th
        Diode parameters. Default 1 kS / 1 µS / 0 V.

    Returns
    -------
    BuckResult
        Device names + branch ids for wiring loss summaries,
        thermal models, control. `suggested_dt` = 1 / (100·f_sw).
    """
    if V_in <= 0:
        raise ValueError("add_buck: V_in must be > 0")
    if L <= 0:
        raise ValueError("add_buck: L must be > 0")
    if C <= 0:
        raise ValueError("add_buck: C must be > 0")
    if R_load <= 0:
        raise ValueError("add_buck: R_load must be > 0")
    if f_sw <= 0:
        raise ValueError("add_buck: f_sw must be > 0")

    sw_mid = f"{name}__sw"
    out = BuckResult(
        switch_name=f"{name}__HS",
        diode_name=f"{name}__D",
        inductor_name=f"{name}__L",
        capacitor_name=f"{name}__C",
        load_name=f"{name}__R",
        suggested_dt=_suggested_dt_for(f_sw),
    )

    builder.add_voltage_source(f"{name}__Vin", vin_node, gnd_node, V_in)

    out.switch_branch_id = int(builder.graph.num_branches)
    builder.add_switch(out.switch_name, vin_node, sw_mid,
                          g_on=g_on, g_off=g_off)

    builder.add_diode(out.diode_name, gnd_node, sw_mid,
                          g_on=diode_g_on, g_off=diode_g_off, V_th=V_th)

    out.inductor_branch_id = int(builder.graph.num_branches)
    builder.add_inductor(out.inductor_name, sw_mid, vout_node, L)

    builder.add_capacitor(out.capacitor_name, vout_node, gnd_node, C)
    builder.add_resistor(out.load_name, vout_node, gnd_node, R_load)
    return out


def add_boost(builder,
               name: str = "boost",
               *,
               vin_node: str = "vin",
               vout_node: str = "vout",
               gnd_node: str = "gnd",
               V_in: float = 12.0,
               L: float = 100e-6,
               C: float = 100e-6,
               R_load: float = 50.0,
               f_sw: float = 100e3,
               g_on: float = 1e3,
               g_off: float = 1e-9,
               diode_g_on: float = 1e3,
               diode_g_off: float = 1e-6,
               V_th: float = 0.0,
               ) -> BoostResult:
    """Add a boost power stage.

    Topology::

                ┌── L ──┬── D ── V_out ──┬── R_load ── gnd
                │        │                │
        V_in ───┤        LS              ═══ C
                │        │                │
               gnd      gnd              gnd

    Devices added: `{name}__Vin`, `{name}__L`, `{name}__LS`,
    `{name}__D`, `{name}__C`, `{name}__R`. Intermediate switch /
    inductor node: `{name}__sw`.

    Same docstring + same caller contract as :func:`add_buck`.
    """
    if V_in <= 0:
        raise ValueError("add_boost: V_in must be > 0")
    if L <= 0 or C <= 0 or R_load <= 0:
        raise ValueError("add_boost: L, C, R_load must be > 0")
    if f_sw <= 0:
        raise ValueError("add_boost: f_sw must be > 0")

    sw_mid = f"{name}__sw"
    out = BoostResult(
        switch_name=f"{name}__LS",
        diode_name=f"{name}__D",
        inductor_name=f"{name}__L",
        capacitor_name=f"{name}__C",
        load_name=f"{name}__R",
        suggested_dt=_suggested_dt_for(f_sw),
    )

    builder.add_voltage_source(f"{name}__Vin", vin_node, gnd_node, V_in)

    out.inductor_branch_id = int(builder.graph.num_branches)
    builder.add_inductor(out.inductor_name, vin_node, sw_mid, L)

    out.switch_branch_id = int(builder.graph.num_branches)
    builder.add_switch(out.switch_name, sw_mid, gnd_node,
                          g_on=g_on, g_off=g_off)

    builder.add_diode(out.diode_name, sw_mid, vout_node,
                          g_on=diode_g_on, g_off=diode_g_off, V_th=V_th)

    builder.add_capacitor(out.capacitor_name, vout_node, gnd_node, C)
    builder.add_resistor(out.load_name, vout_node, gnd_node, R_load)
    return out


def add_flyback(builder,
                 name: str = "flyback",
                 *,
                 vin_node: str = "vin",
                 vout_node: str = "vout",
                 gnd_node: str = "gnd",
                 V_in: float = 48.0,
                 L_pri: float = 200e-6,
                 L_sec: float = 200e-6,
                 k: float = 0.99,
                 C: float = 100e-6,
                 R_load: float = 10.0,
                 f_sw: float = 100e3,
                 g_on: float = 1e3,
                 g_off: float = 1e-9,
                 diode_g_on: float = 1e3,
                 diode_g_off: float = 1e-6,
                 V_th: float = 0.0,
                 ) -> FlybackResult:
    """Add a flyback power stage.

    Topology::

        V_in ──┬── Lp(prim) ── LS ── gnd
               │
              gnd
               ┌── Ls(sec) ── D ── V_out ──┬── R_load ── gnd
               │                            │
              sec_ret                      ═══ C
               │                            │
              gnd                          gnd

    Coupled inductors `Lp` (primary) and `Ls` (secondary) with
    coupling coefficient `k`.

    Devices added: `{name}__Vin`, `{name}__Lp`, `{name}__Ls`,
    `{name}__LS`, `{name}__D`, `{name}__C`, `{name}__R`.
    Intermediate nodes: `{name}__sw` (primary-switch midpoint),
    `{name}__sec` (secondary-winding midpoint).
    """
    if V_in <= 0:
        raise ValueError("add_flyback: V_in must be > 0")
    if not (0 < k <= 1):
        raise ValueError(
            f"add_flyback: k must be in (0, 1], got {k}")
    if L_pri <= 0 or L_sec <= 0 or C <= 0 or R_load <= 0:
        raise ValueError(
            "add_flyback: L_pri, L_sec, C, R_load must be > 0")
    if f_sw <= 0:
        raise ValueError("add_flyback: f_sw must be > 0")

    sw_mid = f"{name}__sw"
    sec_mid = f"{name}__sec"
    out = FlybackResult(
        switch_name=f"{name}__LS",
        primary_name=f"{name}__Lp",
        secondary_name=f"{name}__Ls",
        diode_name=f"{name}__D",
        capacitor_name=f"{name}__C",
        load_name=f"{name}__R",
        suggested_dt=_suggested_dt_for(f_sw),
    )

    builder.add_voltage_source(f"{name}__Vin", vin_node, gnd_node, V_in)

    out.primary_branch_id = int(builder.graph.num_branches)
    builder.add_inductor(out.primary_name, vin_node, sw_mid, L_pri)

    out.switch_branch_id = int(builder.graph.num_branches)
    builder.add_switch(out.switch_name, sw_mid, gnd_node,
                          g_on=g_on, g_off=g_off)

    out.secondary_branch_id = int(builder.graph.num_branches)
    builder.add_inductor(out.secondary_name, sec_mid, gnd_node, L_sec)

    # Couple the two inductors. Builder API exposes
    # add_inductor_coupling(name_a, name_b, k).
    if hasattr(builder, "add_inductor_coupling"):
        builder.add_inductor_coupling(
            out.primary_name, out.secondary_name, k)

    builder.add_diode(out.diode_name, sec_mid, vout_node,
                          g_on=diode_g_on, g_off=diode_g_off, V_th=V_th)

    builder.add_capacitor(out.capacitor_name, vout_node, gnd_node, C)
    builder.add_resistor(out.load_name, vout_node, gnd_node, R_load)
    return out
