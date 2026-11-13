"""add-exact-propagator-strategy D11, report-only slice: is each segment of a run exact-eligible?

A segment is the interval between two events; its mask decides its dynamics, so the record is
one row per mask visited, with the number of segments the run spent in it:

    {"mask": [bool, ...], "exact": bool, "reason": None | 'device' | 'source' | 'extraction'
     | 'cost', "device": str, "kind": str, "detail": str, "n_segments": int}

Reasons (D11 rules 1-4): 'device' -- a device with no linear / PWL law (named, with its kind);
'source' -- a source the exact kernel has no closed form for (PWM / pulse sources, a user
b_extra_fn); 'extraction' -- the per-mask state extraction fails (coupled inductors, a
capacitor loop); 'cost' -- the mask's propagator is outside its accuracy tiers (a
near-defective eigenbasis with a stiff ||A*h||, tier 3; D11 names this rule 'cost': size and
stiffness are one cost question, D13). The default route of engine='auto' does not read this.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ._refusal import UnsupportedError


def _reason_of_tier3(detail: str) -> str:
    return "source" if ("source" in detail or "b_extra_fn" in detail) else "cost"


def segments_from_exact(exact: dict, switch_states: Any) -> list[dict]:
    """The route record of a finished dsed run: one row per mask of ``exact['masks']``."""
    rows = []
    S = None if switch_states is None else np.asarray(switch_states, dtype=bool)
    runs: dict[tuple, int] = {}
    if S is not None and S.ndim == 2 and len(S):
        # a row holds the mask of the interval ending at it; a segment is a maximal run of rows
        starts = np.flatnonzero(np.r_[True, np.any(S[1:] != S[:-1], axis=1)])
        for i in starts:
            key = tuple(bool(v) for v in S[i])
            runs[key] = runs.get(key, 0) + 1
    for m in exact.get("masks", []):
        tier, detail = int(m["tier"]), str(m.get("reason", ""))
        ok = tier in (1, 2)
        rows.append({
            "mask": list(m["mask"]),
            "exact": ok,
            "reason": None if ok else _reason_of_tier3(detail),
            "device": "",
            "kind": "",
            "detail": "" if ok else detail,
            "tier": tier,
            "n_segments": runs.get(tuple(bool(v) for v in m["mask"]), 0),
        })
    return rows


def exact_route(builder, t_end: float, switch_fn=None, **simulate_kwargs) -> dict:
    """Classify every segment of the run ``simulate(builder, t_end, engine='exact', ...)`` as
    exact-eligible or not, with the reason (D11). Report only: nothing is refused here and the
    route of ``engine='auto'`` is unchanged.

    Returns ``{"exact": bool, "reason": str | None, "segments": [row, ...]}`` -- ``exact`` is
    True when every segment is eligible; ``reason`` is the first failing rule. A circuit that
    fails before any segment (a non-PWL device, an extraction refusal) gets one row with an
    empty mask. The run uses the event integrator (``integrator='rk45'``), the one that steps
    masks exactly."""
    from . import _pulsim as _kern
    from . import simulate

    def one_row(reason: str, device: str, kind: str, detail: str) -> dict:
        return {"exact": False, "reason": reason, "segments": [{
            "mask": [], "exact": False, "reason": reason, "device": device, "kind": kind,
            "detail": detail, "tier": 3, "n_segments": 0}]}

    def from_err(reason: str, err: UnsupportedError) -> dict:
        return one_row(reason, getattr(err, "device", "") or "", getattr(err, "kind", "") or "",
                       str(err).split("\n")[0])

    try:
        _kern._require_supported(builder.graph, builder.pool, "exact")
    except UnsupportedError as e:
        return from_err("device", e)
    # Rule 2 is structural: a PWM / pulse source has no closed form between its corners yet
    # (every mask is tier 3), and a user function never has one. Decided without a run (such a
    # run is numeric and can fail on its own account).
    if simulate_kwargs.get("b_extra_fn") is not None:
        return one_row("source", "", "", "b_extra_fn: a user source function has no closed form")
    g, pool = builder.graph, builder.pool
    for i in range(g.num_branches):
        try:
            kind = pool.kind_of(i).name
        except Exception:  # noqa: BLE001 -- an unregistered branch
            continue
        if kind in ("PWMVoltageSource", "PulseVoltageSource"):
            return one_row("source", g.branch_name(i), kind,
                           "PWM / pulse source: not piecewise-constant events yet")
    kw = dict(simulate_kwargs)
    kw.setdefault("integrator", "rk45")
    if switch_fn is not None:
        kw["switch_fn"] = switch_fn
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = simulate(builder, t_end=t_end, engine="exact", **kw)
    except UnsupportedError as e:
        if getattr(e, "feature", "") == "extraction":
            return from_err("extraction", e)
        raise
    segs = (r.exact or {}).get("segments", [])
    bad = [s for s in segs if not s["exact"]]
    return {"exact": not bad, "reason": bad[0]["reason"] if bad else None, "segments": segs}
