"""Post-hoc conduction state of every PWL switched diode in a result.

One source of truth for :meth:`SimulationResult.i`,
:func:`pulsim.device_loss_summary` and :func:`pulsim.device_thermal_summary`
— they used to carry three copies of ``closed = v_D > V_th``, which is
wrong whenever ``V_th > 0`` and put zeros in loss tables.

What the kernel actually does
-----------------------------
``SwitchedDiode`` (``core/include/pulsim/models/switched_diode.hpp``) is
a pure conductance in both states — the stamp
(``pwl/assemble.hpp``, ``case BranchKind::Switch``) is ``g_on`` or
``g_off`` between anode and cathode with NO voltage offset, so a
conducting diode drops ``i / g_on`` whatever ``V_th`` is. ``V_th`` only
enters ``decide_next_state``::

    OFF -> ON   iff  v_D >= V_th
    ON  -> OFF  iff  i_D = g_on * v_D <= 0      (current reversal)

Consequences for a stored sample with diode voltage ``v``:

* ``v >= V_th`` (and ``v > 0``): only ON is self-consistent.
* ``v <= 0``: only OFF is self-consistent.
* ``0 < v < V_th``: BOTH states are self-consistent — this is the
  hysteresis band, and a conducting diode lives in it (48 A through
  ``g_on = 1e3`` is 0.048 V, far below ``V_th = 0.7``). The voltage
  alone cannot tell; the old ``v_D > V_th`` rule called every such
  sample OFF and reported a conducting bridge rectifier as idle.

The result does not record the diode bits per sample, so the band is
resolved here by Kirchhoff's current law: at a node whose every other
branch current is known (resistors, inductors, voltage sources,
current sources, switches with a known mask, diodes already decided —
exact to solver precision — and capacitors: exact where KCL at a
node with no other unknown gives them, e.g. in series with a resistor,
otherwise estimated from C·Δv/Δt under a stricter acceptance
threshold), only one hypothesis for the diode closes the current
balance. Ground is one of those nodes. That test is also applied OUTSIDE the band,
where it catches samples in which the engine stored a state that
contradicts its own commutation rule (``engine='auto'`` can carry a
diode OFF across a forward bias it never sampled a crossing of).

Where KCL cannot decide (a device whose current is not reconstructed
sits on every node of the diode, or the balance is not decisive), the
band sample
inherits the previous sample's state — the kernel's own hysteresis —
and a ``RuntimeWarning`` names the diode and bounds the error. On
``engine='trbdf2'`` a commutation record stamped with a sample's time
hands its new state to the samples after it, and an OFF→ON landing
(committed with the pre-flip OFF mask just past ``V_th``) is kept OFF
where KCL could not look.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import _result_views as _views

# Component kinds whose branch current is fully reconstructible from
# the stored state vector + params, and whose terminals are exactly the
# (from, to) pair `builder.components()` reports. Anything else (ideal
# transformers reflect current into primary nodes that are not listed,
# VCVS / PMSM / nonlinear devices are not reconstructed) disables the
# KCL resolver for the whole circuit — a node with a hidden current
# would make a confident wrong decision possible.
_KCL_KINDS = frozenset({
    "resistor", "capacitor", "inductor", "current_source",
    "voltage_source", "pwm_voltage_source", "sine_voltage_source",
    "pulse_voltage_source", "switch", "diode",
})
_SOURCE_KINDS = frozenset({
    "voltage_source", "pwm_voltage_source", "sine_voltage_source",
    "pulse_voltage_source",
})

# A KCL decision is accepted when the winning hypothesis leaves a
# residual below this fraction of the gap between the two hypotheses.
# Nodes whose balance includes an ESTIMATED capacitor current use the
# stricter value.
_KCL_DECISIVE = 0.05
_KCL_DECISIVE_APPROX = 0.02


@dataclass
class DiodeStateTrace:
    """Per-sample conduction state of one switched diode.

    ``closed[k]`` is True when the ON stamp (``g_on``) produced sample
    ``k``; ``current = v * (g_on if closed else g_off)`` is then the
    branch current the solver actually carried."""

    branch_id: int
    name: str
    g_on: float
    g_off: float
    V_th: float
    v: np.ndarray
    closed: np.ndarray
    n_band: int = 0
    n_kcl_decided: int = 0
    n_unresolved: int = 0
    max_unresolved_error_A: float = 0.0
    unresolved_reason: str = ""
    n_rule_violations: int = 0
    max_violation_V: float = 0.0
    violation_t_range: Tuple[float, float] = (float("nan"), float("nan"))
    _messages: List[str] = field(default_factory=list, repr=False)

    @property
    def current(self) -> np.ndarray:
        return self.v * np.where(self.closed, self.g_on, self.g_off)

    @property
    def conducting(self) -> np.ndarray:
        """Per-STEP conduction flag: element ``k`` describes the step
        ``(t[k-1], t[k]]`` that produced sample ``k`` (element 0 the
        initial point), for use with right-hold time weighting
        (``pulsim.losses._time_fraction``).

        A step conducts when its stamp is ON (``closed[k]``) AND the
        diode carries a material forward current at EITHER end of it
        (``g_on·v_D > 1e-6 · i_peak``):

        * either end, because a variable-step engine lands a natural
          turn-off ON the current zero — the last ON sample of a long
          conducting step sits at ``v_D ≈ -4e-9 V``; judging the step
          by that end alone dropped whole ON steps (DCM boost on
          ``engine='auto'``: duty 0.166 vs 0.223 on ``engine='pwl'``);
        * a material current, because a diode the engine leaves ON at
          the boundary is not conducting in any sense a duty cycle
          should count: ``engine='auto'`` rides a V_th = 0 diode ON at
          ``v_D ≈ -3e-10 V`` for milliseconds, and the fixed engine
          chatters a V_th = 0 diode ON/OFF at ``v_D ≈ 1e-10 V``
          (≈ 1e-7 A) on alternate samples.

        A step that ENDS on a current-zero landing (``g_on·|v_D|``
        below the same threshold, so ON and OFF stamps carry the same
        current there and the stored state is immaterial) keeps the
        state of the step before it: the adaptive engine commits a
        crossing landing with the pre-flip mask, and the step that
        brought the current down to zero conducted all the way."""
        closed = np.asarray(self.closed, dtype=bool)
        if closed.size == 0:
            return closed.copy()
        i_pk = float(np.max(np.abs(self.current)))
        thr = max(1e-12, 1e-6 * i_pk)
        i_fwd = self.g_on * np.asarray(self.v, dtype=float)
        fwd = i_fwd > thr
        either = fwd.copy()
        either[1:] |= fwd[:-1]
        stamp = closed.copy()
        stamp[1:] |= closed[:-1] & (np.abs(i_fwd[1:]) <= thr)
        return stamp & either

    def warn(self, stacklevel: int = 3) -> None:
        """Emit one ``RuntimeWarning`` per problem found."""
        for msg in self._messages:
            warnings.warn(msg, RuntimeWarning, stacklevel=stacklevel + 1)


def _ffill_state(decided: np.ndarray, carry: np.ndarray,
                 initial: int = 0) -> np.ndarray:
    """Fill undecided samples (``decided == -1``) with the carry value
    of the last decided sample before them (``initial`` if none)."""
    n = decided.size
    idx = np.where(decided >= 0, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    out = decided.copy()
    und = decided < 0
    src = idx[und]
    # The carry of an undecided sample is never read: `idx` points at
    # the last DECIDED sample, whose carry is the state it hands on.
    out[und] = np.where(src >= 0, carry[np.clip(src, 0, None)], initial)
    return out


def _switch_closed_traces(result, sf, times, builder,
                          switches: List[Tuple[dict, np.ndarray]],
                          ) -> Dict[int, Optional[np.ndarray]]:
    """Closed trace of every controlled switch, ``None`` where it is
    not reliably known (no switch_fn, a failing switch_fn, or the
    post-hoc mask guard fired). ``switches`` is ``[(desc, v_branch)]``.

    A plain switch_fn is called ONCE per sample for all switches (not
    once per switch): on a six-switch bridge that is the difference
    between N and 6·N Python calls."""
    out: Dict[int, Optional[np.ndarray]] = {
        int(d["branch_id"]): None for d, _ in switches}
    if sf is None or not switches:
        return out
    idx: Dict[int, int] = {}
    for d, _ in switches:
        try:
            idx[int(d["branch_id"])] = int(
                builder.switch_index_of(d.get("name", "")))
        except Exception:  # noqa: BLE001 — unnamed / unknown switch
            continue
    exact = (callable(getattr(sf, "recorded_mask_trace", None))
             or getattr(result, "_switch_masks", None) is not None)
    masks = None
    if not exact:
        q = _views.mask_sample_times(result, times)
        try:
            masks = [sf(float(t)) for t in q]
        except Exception:  # noqa: BLE001 — a broken switch_fn
            return out
    for d, v_sw in switches:
        bid = int(d["branch_id"])
        if bid not in idx:
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if exact:
                    closed = _views.resolve_switch_closed_trace(
                        result, sf, times, idx[bid], v_sw,
                        name=d.get("name"))
                else:
                    raw = np.fromiter(
                        (bool(m.get(idx[bid])) for m in masks),
                        dtype=bool, count=len(masks))
                    closed = _views._voltage_consistency_guard(
                        raw, v_sw, name=d.get("name"))
            except Exception:  # noqa: BLE001 — a broken mask object
                continue
        if caught:
            continue
        out[bid] = np.asarray(closed, dtype=bool)
    return out


def _cache_key(result, times, sf) -> tuple:
    return (int(times.size),
            float(times[0]) if times.size else 0.0,
            float(times[-1]) if times.size else 0.0,
            id(sf))


def diode_state_traces(builder, result, *,
                       switch_fn: Optional[Callable[[float], Any]] = None,
                       ) -> Dict[int, DiodeStateTrace]:
    """Reconstruct the conduction state of every switched diode.

    Returns ``{branch_id: DiodeStateTrace}``. Cached on ``result``."""
    sf = switch_fn if switch_fn is not None else getattr(
        result, "_switch_fn", None)
    states = _views.states_as_array(result)
    times = _views.times_as_array(result)
    key = _cache_key(result, times, sf)
    cached = getattr(result, "_diode_state_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    traces = _reconstruct(builder, result, states, times, sf)
    try:
        result._diode_state_cache = (key, traces)
    except AttributeError:  # pragma: no cover — slotted result type
        pass
    return traces


def diode_state_trace(builder, result, branch_id: int, *,
                      switch_fn: Optional[Callable[[float], Any]] = None,
                      warn: bool = True,
                      stacklevel: int = 3) -> DiodeStateTrace:
    """The :class:`DiodeStateTrace` of one diode; warns if asked."""
    tr = diode_state_traces(builder, result, switch_fn=switch_fn)[
        int(branch_id)]
    if warn:
        tr.warn(stacklevel=stacklevel)
    return tr


def _reconstruct(builder, result, states, times, sf) -> Dict[int, DiodeStateTrace]:
    comps = list(builder.components())
    n = int(times.size)
    node_v = _views.node_voltage_trace

    def _vb(desc) -> np.ndarray:
        f, t = desc["nodes"]
        return node_v(states, int(f)) - node_v(states, int(t))

    diodes: List[dict] = [c for c in comps if c.get("kind") == "diode"]
    if not diodes:
        return {}

    # ---- 1. voltage classification (the kernel's own rule) ----------
    V: Dict[int, np.ndarray] = {}
    G: Dict[int, Tuple[float, float, float]] = {}
    decided: Dict[int, np.ndarray] = {}   # -1 band, 0 OFF, 1 ON
    carry: Dict[int, np.ndarray] = {}     # state handed to the next sample
    evt: Dict[int, np.ndarray] = {}
    band: Dict[int, np.ndarray] = {}
    for d in diodes:
        bid = int(d["branch_id"])
        prm = d.get("params", {}) or {}
        g_on = float(prm.get("g_on", float("nan")))
        g_off = float(prm.get("g_off", float("nan")))
        V_th = float(prm.get("V_th", 0.0))
        G[bid] = (g_on, g_off, V_th)
        v = _vb(d)
        V[bid] = v
        on_forced = (v > 0.0) & (v >= V_th)
        off_forced = v <= 0.0
        st = np.full(n, -1, dtype=np.int8)
        st[on_forced] = 1
        st[off_forced] = 0
        band[bid] = st < 0
        decided[bid] = st
        carry[bid] = st.copy()
        evt[bid] = np.zeros(n, dtype=bool)

    # ---- 2. KCL resolution (exact where a node balance is known) ---
    kcl_reason = ""
    kinds = {str(c.get("kind")) for c in comps}
    hidden = sorted(kinds - _KCL_KINDS)
    kcl_decided = {bid: np.zeros(n, dtype=bool) for bid in decided}
    if hidden:
        kcl_reason = (f"the circuit contains {', '.join(hidden)} "
                      f"device(s), whose terminal currents are not "
                      f"reconstructed, so no node balance is trusted")
    elif n:
        kcl_reason = _kcl_resolve(builder, result, comps, states, times,
                                  sf, V, G, decided, carry, kcl_decided)

    # ---- 3. trbdf2 commutation records ------------------------------
    # The engine flips a diode AFTER the sample stamped with the
    # event time, so the samples that follow inherit the record's
    # new state. An OFF→ON landing is committed with the PRE-flip
    # (OFF) mask just past v_D = V_th: where KCL could not look, the
    # voltage rule would call it ON and report g_on·V_th (700 A at
    # the defaults) on a diode that carried nothing. The record is
    # NOT trusted the other way: some ON→OFF records sit on a sample
    # the solver had already stamped OFF (measured on a buck: an
    # "ON" pin there read −48 kA).
    if getattr(result, "engine_used", None) == "trbdf2" and n:
        for ev in getattr(result, "commutation_events", []) or []:
            bid = int(ev.branch_id)
            if bid not in decided:
                continue
            te = float(ev.t_estimated)
            k = int(np.searchsorted(times, te))
            # Several samples may share a time stamp (zero-time
            # cascades); the record follows the LAST one stored at te.
            while k + 1 < n and times[k + 1] == te:
                k += 1
            if k >= n or times[k] != te:
                continue
            new = 1 if bool(ev.new_state) else 0
            evt[bid][k] = True
            if not kcl_decided[bid][k]:
                carry[bid][k] = new
                if decided[bid][k] < 0:
                    decided[bid][k] = new
                elif new == 1 and decided[bid][k] == 1:
                    decided[bid][k] = 0

    # ---- 4. breach samples: the solver itself flagged them ----------
    breach_mask = np.zeros(n, dtype=bool)
    for br in getattr(result, "event_iteration_breaches", []) or []:
        k = int(np.searchsorted(times, float(br.t)))
        if k < n and times[k] == float(br.t):
            breach_mask[k] = True

    # ---- 5. fill + bookkeeping -------------------------------------
    i_ref = 0.0
    for c in comps:
        if c.get("kind") in _SOURCE_KINDS or c.get("kind") == "inductor":
            col = _state_native_current(builder, states, c)
            if col is not None and col.size:
                i_ref = max(i_ref, float(np.nanmax(np.abs(col))))

    out: Dict[int, DiodeStateTrace] = {}
    for d in diodes:
        bid = int(d["branch_id"])
        g_on, g_off, V_th = G[bid]
        v = V[bid]
        st = decided[bid]
        filled = _ffill_state(st, carry[bid], initial=0)
        closed = filled.astype(bool)
        tr = DiodeStateTrace(branch_id=bid, name=str(d.get("name", "")),
                             g_on=g_on, g_off=g_off, V_th=V_th, v=v,
                             closed=closed)
        tr.n_band = int(band[bid].sum())
        tr.n_kcl_decided = int(kcl_decided[bid].sum())
        unres = st < 0
        gap = np.abs((g_on - g_off) * v)
        i_on = np.abs(v[closed & ~unres]) * g_on
        i_scale = max(i_ref, float(i_on.max()) if i_on.size else 0.0)
        tol = max(1e-12, 1e-3 * i_scale)
        material = unres & (gap > tol)
        tr.n_unresolved = int(material.sum())
        if tr.n_unresolved:
            tr.max_unresolved_error_A = float(gap[material].max())
            tr.unresolved_reason = kcl_reason
            tr._messages.append(
                f"diode {tr.name!r}: {tr.n_unresolved} of {n} samples sit "
                f"inside the 0 < v_D < V_th = {V_th:g} V band, where the "
                f"PWL SwitchedDiode is self-consistent both ON and OFF, "
                f"and the conduction state could not be recovered from "
                f"the result: {tr.unresolved_reason}. Those samples "
                f"inherit the previous sample's state (the kernel's "
                f"hysteresis); the reported current there may be off by "
                f"up to {tr.max_unresolved_error_A:.3g} A. For exact "
                f"reporting give the diode V_th=0 and model its forward "
                f"drop with conduction_specs={{'V_f0': ..., 'r_on': ...}}, "
                f"or keep a known branch (resistor / inductor / source) "
                f"on one of its nodes.")
        # Rule violations: a KCL-decided state that the kernel's own
        # commutation rule forbids, beyond a 1 mV tolerance. Samples
        # carrying a commutation record and breach steps are expected
        # exceptions.
        on = closed
        viol = kcl_decided[bid] & ~evt[bid] & ~breach_mask & (
            (~on & (v - V_th > 1e-3) & (v > 1e-3))
            | (on & (v < -1e-3)))
        if viol.any():
            tr.n_rule_violations = int(viol.sum())
            dv = np.where(on, -v, v - V_th)
            tr.max_violation_V = float(dv[viol].max())
            tv = times[viol]
            tr.violation_t_range = (float(tv.min()), float(tv.max()))
            engine = getattr(result, "engine_used", "?")
            tr._messages.append(
                f"diode {tr.name!r}: on {tr.n_rule_violations} samples "
                f"(t = {tv.min():.6g} .. {tv.max():.6g} s) the simulation "
                f"held the diode in a state its commutation rule forbids "
                f"(by up to {tr.max_violation_V:.3g} V; e.g. OFF while "
                f"forward-biased above V_th = {V_th:g} V). The reported "
                f"current follows what the solver actually stamped "
                f"(checked by KCL), but the engine ({engine!r}) missed a "
                f"commutation there, so the waveforms themselves are "
                f"suspect in that window; cross-check with "
                f"engine='pwl' and a fixed dt.")
        out[bid] = tr
    return out


def _state_native_current(builder, states, desc) -> Optional[np.ndarray]:
    bid = int(desc["branch_id"])
    kind = desc.get("kind")
    try:
        if kind == "inductor":
            col = builder.pool.branch_var_id_for_inductor(bid, builder.graph)
        elif kind in _SOURCE_KINDS:
            col = builder.pool.branch_var_id_for_source(bid, builder.graph)
        else:
            return None
    except Exception:  # noqa: BLE001 — wrong kind / unbound branch
        return None
    return np.asarray(states[:, int(col)], dtype=float)


def _kcl_resolve(builder, result, comps, states, times, sf, V, G,
                 decided, carry, kcl_decided) -> str:
    """Decide diode states by node current balance. Mutates
    ``decided`` / ``carry`` / ``kcl_decided``; returns a reason string
    describing why some nodes are unusable (empty if all usable)."""
    n = int(times.size)
    node_v = _views.node_voltage_trace
    known: Dict[int, np.ndarray] = {}
    approx_bids: set = set()
    unknown_bids: set = set()
    pending_switches: List[Tuple[dict, np.ndarray]] = []
    for c in comps:
        bid = int(c["branch_id"])
        kind = c.get("kind")
        prm = c.get("params", {}) or {}
        f, t = c["nodes"]
        if kind == "diode":
            continue
        if kind == "resistor":
            R = float(prm.get("R_ohms", float("nan")))
            if np.isfinite(R) and R > 0:
                known[bid] = (node_v(states, int(f))
                              - node_v(states, int(t))) / R
                continue
        elif kind == "inductor" or kind in _SOURCE_KINDS:
            col = _state_native_current(builder, states, c)
            if col is not None:
                known[bid] = col
                continue
        elif kind == "current_source":
            # `from` is the + terminal: the source pushes I OUT of it
            # into the external circuit (stamp_current_source.hpp), so
            # the from→to current THROUGH the element is −I.
            known[bid] = np.full(n, -float(prm.get("I", 0.0)))
            continue
        elif kind == "capacitor":
            # Backward difference C·Δv/Δt. For the trapezoidal and
            # backward-Euler companions this is EXACTLY the current
            # averaged over the step, so it misses the end-point
            # current by half the step's change: negligible where the
            # current is smooth, about half the jump on the sample a
            # diode commutes — where the stricter decision threshold
            # below then declines to decide rather than guess.
            C = float(prm.get("C_farads", float("nan")))
            if np.isfinite(C) and C > 0 and n > 1:
                vc = node_v(states, int(f)) - node_v(states, int(t))
                dt = np.diff(times)
                ic = np.full(n, np.nan)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ic[1:] = np.where(dt > 0, C * np.diff(vc) / dt, np.nan)
                known[bid] = ic
                approx_bids.add(bid)
                continue
        elif kind == "switch":
            v_sw = node_v(states, int(f)) - node_v(states, int(t))
            pending_switches.append((c, v_sw))
            continue
        unknown_bids.add(bid)

    closed_by_bid = _switch_closed_traces(result, sf, times, builder,
                                          pending_switches)
    for c, v_sw in pending_switches:
        bid = int(c["branch_id"])
        prm = c.get("params", {}) or {}
        closed = closed_by_bid.get(bid)
        if closed is None or closed.size != n:
            unknown_bids.add(bid)
            continue
        known[bid] = v_sw * np.where(closed, float(prm.get("g_on", 0.0)),
                                     float(prm.get("g_off", 0.0)))

    # node -> list of (bid, sign); sign +1 = current leaves the node.
    # Ground (every negative id, -1 in practice) is a node like any
    # other: with every branch listed (guaranteed above — any unlisted
    # current disables KCL), its balance is exact too, and it is the
    # only one that decides a diode whose other node is shared with an
    # undecided partner (a bridge whose AC side is grounded and DC
    # side floats: D3/D4 meet the source only at ground).
    _GND = -1
    incid: Dict[int, List[Tuple[int, int]]] = {}
    for c in comps:
        bid = int(c["branch_id"])
        f, t = (int(x) for x in c["nodes"])
        incid.setdefault(f if f >= 0 else _GND, []).append((bid, +1))
        incid.setdefault(t if t >= 0 else _GND, []).append((bid, -1))

    # Exact currents for estimated capacitors: a node whose every
    # other branch is exactly known gives the capacitor's current by
    # KCL, with no C·Δv/Δt error (a capacitor in series with a
    # resistor or a source — a Villard doubler's pump capacitor).
    # Iterated so the exactness propagates along series chains.
    for _pass in range(len(approx_bids) + 1):
        upgraded = False
        for lst in incid.values():
            pending = [(b, sg) for b, sg in lst
                       if b in approx_bids or b not in known]
            if len(pending) != 1 or pending[0][0] not in approx_bids:
                continue
            b_c, s_c = pending[0]
            acc = np.zeros(n)
            for b, sg in lst:
                if b != b_c:
                    acc += sg * known[b]
            known[b_c] = -s_c * acc
            approx_bids.discard(b_c)
            upgraded = True
        if not upgraded:
            break

    usable: Dict[int, Tuple[np.ndarray, np.ndarray,
                            List[Tuple[int, int]], float]] = {}
    for node, lst in incid.items():
        if any(b in unknown_bids for b, _ in lst):
            continue
        base = np.zeros(n)
        scale = np.zeros(n)
        dl = []
        for b, s in lst:
            if b in known:
                base += s * known[b]
                scale += np.abs(known[b])
            else:
                dl.append((b, s))
        if dl:
            approx = any(b in approx_bids for b, _ in lst)
            usable[node] = (base, scale, dl,
                            _KCL_DECISIVE_APPROX if approx
                            else _KCL_DECISIVE)
    if not usable:
        return ("every node of every diode touches a switch without a "
                "known mask or another device whose current is not "
                "reconstructed")

    def _cur(b: int, st: np.ndarray) -> np.ndarray:
        g_on, g_off, _ = G[b]
        return V[b] * np.where(st > 0, g_on, g_off)

    trusted = {b: decided[b] >= 0 for b in decided}
    for _pass in range(2 * len(decided) + 2):
        changed = False
        for node, (base, scale, dl, thresh) in usable.items():
            for b, s in dl:
                others = [(o, so) for o, so in dl if o != b]
                ok = np.isfinite(base)
                acc = base.copy()
                sc = scale.copy()
                for o, so in others:
                    ok &= trusted[o]
                    io = _cur(o, decided[o])
                    acc += so * io
                    sc += np.abs(io)
                if not ok.any():
                    continue
                g_on, g_off, _ = G[b]
                r_on = acc + s * g_on * V[b]
                r_off = acc + s * g_off * V[b]
                gap = np.abs(r_on - r_off)
                best_on = np.abs(r_on) < np.abs(r_off)
                best = np.where(best_on, np.abs(r_on), np.abs(r_off))
                tiny = 1e-12 + 1e-9 * (sc + gap)
                with np.errstate(invalid="ignore"):
                    dec = ok & (gap > tiny) & (best <= thresh * gap)
                new_kcl = dec & ~kcl_decided[b]
                if not new_kcl.any():
                    continue
                val = best_on[new_kcl].astype(np.int8)
                decided[b][new_kcl] = val
                carry[b][new_kcl] = val
                kcl_decided[b][new_kcl] = True
                trusted[b] = trusted[b] | new_kcl
                changed = True
        if not changed:
            break
    return ("no node of this diode closed its current balance decisively "
            "at those samples (an undecided neighbour diode, a device "
            "whose current is not reconstructed, or a capacitor whose "
            "current changes too fast between samples to estimate)")
