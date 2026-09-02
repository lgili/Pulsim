"""Bidirectional electro-thermal coupling — audit C.2.

Pulsim already closes half the loop. `make_electrothermal_heatsink_observer`
recomputes each device's injected power from its current junction
temperature every step, so **loss follows temperature**. But the
electrical side never sees the temperature: the ``R_on`` in the MNA
matrix, the diode's ``V_th``, the IGBT's ``V_CE_sat`` are whatever
they were when the circuit was built, forever.

So the loop runs

    electrical -> lumped power -> thermal -> lumped power     (closed)
    thermal -> device parameters in the matrix                (OPEN)

WHAT THAT COSTS, measured honestly. On a well-designed buck the
answer is: almost nothing. Doubling a 5 mOhm MOSFET's ``R_on`` from
25 C to 125 C moves ``v_out`` by **-0.22 %**, because the device
drop is tiny next to the load. That is not the motivation, and
pretending otherwise would oversell this module.

The motivation is the case where the answer IS the coupling. Two
identical MOSFETs in parallel sharing 100 A, one mounted worse
(R_th 2.0 vs 1.2 K/W), Rds_on doubling by 125 C:

                        frozen        coupled
    current imbalance     0.0 %         6.9 %
    hottest junction     65.0 C        72.1 C

The frozen model reports the imbalance as **exactly zero**. That is
not a small error, it is a structural blindness: current sharing
between paralleled devices is entirely a temperature effect, and
paralleling derating is a real design decision the simulator was
giving no information about. It also under-reads the hottest
junction by 7 C, which matters against a 125 C limit.

HOW THE COUPLING IS AFFORDABLE. Changing a device parameter
invalidates the PWL cache's factorisations for every switch state,
and refactorising on every step would be ruinous. It does not have
to be: thermal time constants are milliseconds to seconds while the
switching period is microseconds, so the parameters move slowly by
orders of magnitude. This module exploits that separation
explicitly — parameters are refreshed on their own cadence, and
`update_every_s` is a required argument rather than a default,
because the right value follows from the thermal network the user
built and guessing it silently is how a coupling becomes wrong.

The refactoring path itself is not new: `mmc_thevenin` already
drives `cache.refactor_parametric` mid-run from circuit state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass
class TempCoResistance:
    """A resistance in the circuit that follows a junction temperature.

    ``R(Tj) = R_ref * (1 + a * (Tj - T_ref))``

    Attributes
    ----------
    branch
        Name of a **resistor** branch in the circuit. It has to be a
        resistor: those are what `refactor_parametric` can update in
        place. Model a switch's conduction as an ideal switch in
        series with this resistor.
    junction_node
        Thermal node whose voltage is this device's Tj [°C] — the
        name from `SharedHeatsink.junction_nodes`.
    R_ref_ohms, a_per_C, T_ref_C
        The datasheet's ``R_ds(on)`` tempco. A Si MOSFET that roughly
        doubles by 150 °C is ``a ≈ +0.008``. Copper windings are
        ``+0.00393``. A negative ``a`` is allowed and is the
        destabilising direction — see `runaway_margin`.
    R_min_ohms
        Floor, applied after the tempco. A linear law taken far below
        ``T_ref`` eventually goes negative, and a negative resistance
        is a different circuit, not a cold one.
    """

    branch: str
    junction_node: str
    R_ref_ohms: float
    a_per_C: float
    T_ref_C: float = 25.0
    R_min_ohms: float = 1e-9

    def __post_init__(self) -> None:
        if not (self.R_ref_ohms > 0):
            raise ValueError(
                f"TempCoResistance({self.branch!r}): R_ref_ohms must "
                f"be > 0, got {self.R_ref_ohms!r}")
        if not (self.R_min_ohms > 0):
            raise ValueError(
                f"TempCoResistance({self.branch!r}): R_min_ohms must "
                f"be > 0 — it is the floor that keeps a linear tempco "
                "from producing a negative resistance far below "
                "T_ref, which would be a different circuit rather "
                "than a cold one")

    def at(self, tj_c: float) -> float:
        """Resistance [Ω] at junction temperature `tj_c` [°C]."""
        r = self.R_ref_ohms * (
            1.0 + self.a_per_C * (float(tj_c) - self.T_ref_C))
        return max(float(r), self.R_min_ohms)


def make_bidirectional_observer(
    builder,
    cache,
    heatsink,
    tempco: Sequence[TempCoResistance],
    *,
    update_every_s: float,
    extra_power_fns: Mapping[str, Callable] | None = None,
    on_update: Callable[[float, dict], None] | None = None,
):
    """Close the second half of the loop: temperature -> the matrix.

    Both directions are driven by the circuit itself:

    * **loss follows temperature** — each device's dissipation is
      read as ``i^2 * R`` from the state vector, with ``R`` whatever
      the last temperature update wrote into the matrix. There is no
      `TempCoLoss` in the middle. That matters: `TempCoLoss` is a
      hand-fitted linear proxy anchored on a reference power from a
      separate run, and once the resistance is genuinely in the
      matrix the proxy is not just unnecessary, it is a second,
      independent model of the same physics that can disagree with
      the first.
    * **temperature follows loss** — the thermal network is the one
      `add_shared_heatsink` built, integrated by the same solver.

    Parameters
    ----------
    cache
        The `PwlStateSpaceCache` the run will use. Parameters are
        written through `refactor_parametric`, so it must be that
        same object.
    tempco
        The resistances that follow a junction temperature. Each
        names a **resistor** branch (those are what
        `refactor_parametric` can update in place) and the thermal
        node carrying its Tj. Model a switch's conduction as an
        ideal switch in series with one of these.
    update_every_s
        How often to refresh the parameters [s]. **Required, with no
        default**: the right value follows from the thermal network
        the user built, and a silently chosen one is how a coupling
        becomes wrong. Refactorising is the expensive half of this
        loop, and it is affordable only because thermal time
        constants run milliseconds to seconds while the switching
        period runs microseconds. Pick it well below the fastest
        thermal time constant and well above the switching period —
        the separation is usually three decades or more.
    extra_power_fns
        Optional ``{device_name: (t, x) -> P_W}`` for dissipation the
        circuit does not carry — switching loss being the usual case,
        since an ideal switch dissipates nothing during its
        transition. Added to the conduction power computed above.
    on_update
        Optional ``(t, {branch: R_ohms}) -> None`` after each refresh.

    Returns
    -------
    (step_observer, b_extra_fn)
        Pass both to `simulate`.
    """
    from .thermal import make_heatsink_observer

    if not (update_every_s > 0):
        raise ValueError(
            "make_bidirectional_observer: update_every_s must be > 0. "
            "It is how often the junction temperature is written back "
            "into the electrical parameters; there is no sensible "
            "default because it follows from your thermal network's "
            "time constants.")

    j_nodes = dict(getattr(heatsink, "junction_nodes", {}) or {})
    node_to_device = {v: k for k, v in j_nodes.items()}
    for tc in tempco:
        if tc.junction_node not in node_to_device:
            raise KeyError(
                f"make_bidirectional_observer: {tc.branch!r} names "
                f"junction node {tc.junction_node!r}, which is not on "
                f"this heatsink. Available: "
                f"{sorted(node_to_device)}.")

    # Branch geometry, resolved once.
    by_id = {int(br["id"]): br for br in builder.graph.branches}
    info = {}
    for tc in tempco:
        b_id = builder.branch_id_of(tc.branch)
        if b_id < 0 or b_id not in by_id:
            raise KeyError(
                f"make_bidirectional_observer: no branch named "
                f"{tc.branch!r} in this circuit")
        br = by_id[b_id]
        info[tc.branch] = {
            "id": b_id,
            "from": int(br["from_"]),
            "to": int(br["to"]),
            "tj": builder.node_id_of(tc.junction_node),
            "device": node_to_device[tc.junction_node],
        }

    # Live resistance per branch, seeded at the reference value the
    # circuit was actually built with.
    r_now = {tc.branch: tc.R_ref_ohms for tc in tempco}

    def _v(x, idx):
        return 0.0 if idx < 0 else float(x[idx])

    # Conduction power per DEVICE — several branches may share one
    # junction, so they accumulate rather than overwrite.
    def _make_power_fn(device: str):
        mine = [tc for tc in tempco
                if info[tc.branch]["device"] == device]
        extra = (extra_power_fns or {}).get(device)

        def power_fn(t, x, _mine=mine, _extra=extra):
            p_w = 0.0
            for tc in _mine:
                d = info[tc.branch]
                dv = _v(x, d["from"]) - _v(x, d["to"])
                p_w += dv * dv / r_now[tc.branch]
            if _extra is not None:
                p_w += float(_extra(t, x))
            return p_w

        return power_fn

    devices = sorted({info[tc.branch]["device"] for tc in tempco})
    power_fns = {d: _make_power_fn(d) for d in devices}
    for d, fn in (extra_power_fns or {}).items():
        if d not in power_fns:
            power_fns[d] = fn

    base_obs, b_extra = make_heatsink_observer(
        builder, heatsink, power_fns)

    state = {"next_t": -np.inf}

    def step_observer(t, x):
        if t >= state["next_t"]:
            state["next_t"] = float(t) + update_every_s
            applied = {}
            for tc in tempco:
                d = info[tc.branch]
                r_new = tc.at(_v(x, d["tj"]))
                prev = r_now[tc.branch]
                # Skip a refactorisation that would not move the
                # matrix: refactorising is the expensive half of the
                # loop, and a settled thermal network stops changing
                # long before the run does.
                if abs(r_new - prev) > 1e-12 * max(abs(prev), 1e-30):
                    builder.update_resistor_R(tc.branch, r_new)
                    cache.refactor_parametric(d["id"], r_new)
                    r_now[tc.branch] = r_new
                applied[tc.branch] = r_now[tc.branch]
            if on_update is not None:
                on_update(float(t), applied)
        # Power injection last, so it uses the resistance just written.
        base_obs(t, x)

    return step_observer, b_extra


def runaway_margin(tempco: Sequence[TempCoResistance],
                    currents_a: Mapping[str, float],
                    r_th_j_to_amb: Mapping[str, float]) -> dict:
    """Is the loss-temperature loop stable?

    For a device dissipating ``P = I^2 * R(Tj)`` into a thermal
    resistance ``R_th`` to ambient, the loop gain is

        G = dP/dTj * R_th = I^2 * R_ref * a * R_th

    and the fixed point exists only for ``G < 1``. At ``G >= 1`` every
    extra degree produces more than a degree's worth of heating and
    there is no steady state at all — the transient run would climb
    until something else in the model stopped it, which is a
    simulation artefact rather than an answer.

    Returns ``{device: {"gain": G, "stable": bool}}`` plus a
    ``"worst"`` key. A negative tempco gives ``G < 0``, which is
    unconditionally stable in this sense (it is the *current sharing*
    of negative-tempco devices that misbehaves, not the single-device
    loop).
    """
    out: dict = {}
    worst = -np.inf
    for tc in tempco:
        i = float(currents_a.get(tc.branch, 0.0))
        rth = float(r_th_j_to_amb.get(tc.branch, 0.0))
        g = i * i * tc.R_ref_ohms * tc.a_per_C * rth
        out[tc.branch] = {"gain": g, "stable": bool(g < 1.0)}
        worst = max(worst, g)
    out["worst"] = {"gain": float(worst) if np.isfinite(worst) else 0.0,
                    "stable": bool(worst < 1.0)}
    return out
