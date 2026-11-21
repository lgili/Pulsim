"""Datasheet fitters for the power MOSFET (add-power-mosfet-datasheet-device).

Siblings of :func:`pulsim.fit_foster_from_zth` and
:func:`pulsim.fit_ja_from_bh_curve`. Every fitter returns its parameters
together with its largest residual and RAISES when that residual exceeds
the documented limit, instead of handing back a poor fit:

==============================  =========================================  ==========
fitter                          residual                                   limit
==============================  =========================================  ==========
``fit_charge_table``            max |C_table(v_i) − C_i| / C_i              ``tol`` (1 %)
``fit_mosfet_channel``          max |I_model − I_i| / I_i (fitted points)   ``max_residual`` (5 %)
``fit_rdson_temperature``       max |r_model − r_i|                         ``max_residual`` (0.02)
``fit_body_diode_recovery``     charge-control model, see its docstring     solvable or raises
``power_mosfet_from_datasheet`` the worst of the above, per curve           each fitter's own
==============================  =========================================  ==========

A fit is not a validation (owner note *motor-inductance-is-the-free-
parameter*): a part built from these fitters records the curves it was
fitted on in ``fitted_on`` and is checked only against held-out data
(``validated_on``), see :mod:`pulsim.lib.parts`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

__all__ = [
    "ChargeTableFit",
    "fit_charge_table",
    "fit_mosfet_channel",
    "fit_rdson_temperature",
    "fit_body_diode_recovery",
    "power_mosfet_from_datasheet",
]


@dataclass
class ChargeTableFit:
    table: Any            #: pulsim.ChargeTable
    residual: float       #: max relative error of C at the input points
    knots: int
    clamped_tangents: int


def _log_pchip(v: np.ndarray, c: np.ndarray):
    """Monotone (Fritsch–Carlson) cubic through (v, ln C); returns f(x) -> C."""
    y = np.log(c)
    h = np.diff(v)
    d = np.diff(y) / h
    n = v.size
    s = np.zeros(n)
    for k in range(1, n - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            s[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    s[0], s[-1] = d[0], d[-1]

    def f(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        j = np.clip(np.searchsorted(v, x) - 1, 0, n - 2)
        t = (x - v[j]) / h[j]
        out = ((2 * t**3 - 3 * t**2 + 1) * y[j] + (t**3 - 2 * t**2 + t) * h[j] * s[j]
               + (-2 * t**3 + 3 * t**2) * y[j + 1] + (t**3 - t**2) * h[j] * s[j + 1])
        lo, hi = x < v[0], x > v[-1]
        out[lo] = y[0]
        out[hi] = y[-1]
        return np.exp(out)
    return f


def fit_charge_table(v: Sequence[float], C: Sequence[float], *, tol: float = 0.01,
                     max_refinements: int = 12) -> ChargeTableFit:
    """A :class:`pulsim.ChargeTable` from a capacitance curve ``C(v)``.

    ``v`` is the law's own variable (``v_gd = −V_DS`` for C_gd, ``v_ds``
    for C_ds) and must be strictly increasing; ``C`` positive [F]. The
    knots start as the input points (a knot at ``v = 0`` is added with
    C from the log-monotone interpolant of the data, held constant beyond
    its ends). Where the Fritsch–Carlson clamp moves a tangent far
    enough that the table misses an input point by more than ``tol``, the
    neighbouring segments are split (new knot values from the same
    log-monotone interpolant) and the table rebuilt, up to
    ``max_refinements`` times; then it raises.
    """
    import pulsim as _p

    v = np.asarray(v, dtype=float)
    c = np.asarray(C, dtype=float)
    if v.ndim != 1 or v.shape != c.shape or v.size < 2:
        raise ValueError("fit_charge_table: v and C must be 1-D of equal length >= 2.")
    if np.any(np.diff(v) <= 0):
        raise ValueError("fit_charge_table: v must be strictly increasing.")
    if np.any(c <= 0) or not np.all(np.isfinite(c)):
        raise ValueError("fit_charge_table: every C must be positive and finite.")
    interp = _log_pchip(v, c)
    knots = set(v.tolist()) | {0.0}
    for _ in range(max_refinements + 1):
        kv = np.array(sorted(knots))
        if kv.size < 3:
            kv = np.array(sorted(set(kv.tolist()) | {kv[0] + (kv[-1] - kv[0]) / 2}))
        kc = interp(kv)
        table = _p.ChargeTable.from_capacitance(kv.tolist(), kc.tolist())
        model = np.array([table.capacitance(float(x)) for x in v])
        err = np.abs(model - c) / c
        resid = float(err.max())
        if resid <= tol:
            return ChargeTableFit(table, resid, int(kv.size), int(table.clamped_tangents))
        for x in v[err > tol]:
            j = int(np.searchsorted(kv, x))
            for a, b in ((j - 1, j), (j, j + 1)):
                if 0 <= a and b < kv.size:
                    knots.add(float((kv[a] + kv[b]) / 2))
    raise ValueError(
        f"fit_charge_table: residual {resid:.3g} > tol {tol:g} after "
        f"{max_refinements} refinements ({kv.size} knots); the curve is not "
        "representable as a monotone charge table at this tolerance.")


def fit_mosfet_channel(v_gs: Sequence[float], i_d: Sequence[float], *, v_ds: float,
                       lambda_: float = 0.0, min_fraction: float = 0.05,
                       max_residual: float = 0.05) -> Dict[str, float]:
    """Level-1 ``K`` [A/V²] and ``V_T`` [V] from a saturation transfer curve.

    Fits ``sqrt(I_D / (1 + λ·V_DS)) = sqrt(K)·(V_GS − V_T)`` by linear least
    squares over the points whose current is at least ``min_fraction`` of
    the largest (sub-threshold points are not in the Level-1 law). Returns
    ``{"K", "V_T", "residual", "points"}``; the residual is the largest
    relative current error over the fitted points.
    """
    vg = np.asarray(v_gs, dtype=float)
    i = np.asarray(i_d, dtype=float)
    if vg.shape != i.shape or vg.size < 2:
        raise ValueError("fit_mosfet_channel: v_gs and i_d must be equal-length arrays.")
    i_n = i / (1.0 + lambda_ * v_ds)
    m = i_n >= min_fraction * i_n.max()
    if m.sum() < 2:
        raise ValueError("fit_mosfet_channel: fewer than two points above min_fraction.")
    A = np.column_stack([vg[m], np.ones(m.sum())])
    (a, b), *_ = np.linalg.lstsq(A, np.sqrt(i_n[m]), rcond=None)
    K, V_T = float(a * a), float(-b / a)
    model = K * np.maximum(vg[m] - V_T, 0.0) ** 2 * (1.0 + lambda_ * v_ds)
    resid = float(np.max(np.abs(model - i[m]) / i[m]))
    if resid > max_residual:
        raise ValueError(
            f"fit_mosfet_channel: residual {resid:.3g} > {max_residual:g} — the transfer "
            "curve is not square-law over the fitted range (transconductance roll-off); "
            "a constant-K channel will misplace the Miller plateau.")
    return {"K": K, "V_T": V_T, "residual": resid, "points": int(m.sum())}


def fit_rdson_temperature(T_j: Sequence[float], r_norm: Sequence[float], *,
                          max_residual: float = 0.02) -> Dict[str, float]:
    """``rds_tc1`` [1/K] and ``rds_tc2`` [1/K²] from a normalised
    R_DS(on)(T_j) curve (1.0 at 25 °C): ``r = 1 + tc1·ΔT + tc2·ΔT²``.
    Returns ``{"rds_tc1", "rds_tc2", "residual"}`` (largest |Δr|)."""
    T = np.asarray(T_j, dtype=float)
    r = np.asarray(r_norm, dtype=float)
    if T.shape != r.shape or T.size < 2:
        raise ValueError("fit_rdson_temperature: T_j and r_norm must be equal-length arrays.")
    dT = T - 25.0
    A = np.column_stack([dT, dT * dT])
    (tc1, tc2), *_ = np.linalg.lstsq(A, r - 1.0, rcond=None)
    resid = float(np.max(np.abs(1.0 + tc1 * dT + tc2 * dT * dT - r)))
    if resid > max_residual:
        raise ValueError(
            f"fit_rdson_temperature: residual {resid:.3g} > {max_residual:g}; the curve "
            "is not quadratic in T_j (or is not normalised to 1.0 at 25 °C).")
    return {"rds_tc1": float(tc1), "rds_tc2": float(tc2), "residual": resid}


def fit_body_diode_recovery(*, Q_rr: float, I_rrm: float, I_F: float, di_dt: float,
                            V_R: float) -> Dict[str, float]:
    """Lauritzen ``tau`` and ``T_M`` [s] from a datasheet recovery point.

    Charge-control estimate (Lauritzen & Ma) for a constant ramp
    ``di/dt`` from ``I_F``: the stored charge ``q_M`` follows
    ``dq_M/dt = i − q_M/τ``; the junction stops conducting when
    ``q_M + T_M·i = 0``, at ``i = −I_rrm``; after that the current is
    ``−q_M/T_M`` and decays with ``τ_e = τ·T_M/(τ + T_M)``. Hence

        Q_rr = I_rrm² / (2·di/dt) + I_rrm·τ_e
        T_M·I_rrm = di/dt·τ²·(1 − exp(−t0/τ)) − τ·I_rrm,  t0 = (I_F + I_rrm)/(di/dt)

    solved for ``τ`` by bisection. ``V_R`` does not enter this model
    (recorded for the part file). The estimate assumes an abrupt
    junction and a stiff circuit; it is NOT checked by simulation here —
    a part's double-pulse acceptance check is what tests it. Raises when
    the data admit no solution (``Q_rr ≤ I_rrm²/(2·di/dt)``: the
    recovery is faster than the ramp alone allows).
    """
    a = float(di_dt)
    if min(Q_rr, I_rrm, I_F, a) <= 0:
        raise ValueError("fit_body_diode_recovery: Q_rr, I_rrm, I_F and di_dt must be positive.")
    tau_e = (Q_rr - I_rrm * I_rrm / (2 * a)) / I_rrm
    if tau_e <= 0:
        raise ValueError(
            f"fit_body_diode_recovery: Q_rr = {Q_rr:.3g} C is not above I_rrm²/(2·di/dt) "
            f"= {I_rrm * I_rrm / (2 * a):.3g} C — no charge-control solution (check units: "
            "di_dt in A/s).")
    t0 = (I_F + I_rrm) / a

    def g(tau):
        T_M = (a * tau * tau * (-math.expm1(-t0 / tau)) - tau * I_rrm) / I_rrm
        if T_M <= 0:
            return -tau_e
        return tau * T_M / (tau + T_M) - tau_e

    # g changes sign once on a log grid of lifetimes; bisect the first change.
    grid = np.geomspace(1e-12, 1.0, 241)
    vals = [g(x) for x in grid]
    k = next((j for j in range(len(grid) - 1) if vals[j] <= 0 < vals[j + 1]), None)
    if k is None:
        raise ValueError("fit_body_diode_recovery: no lifetime in [1 ps, 1 s] fits the data.")
    lo, hi = float(grid[k]), float(grid[k + 1])
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
    tau = hi
    T_M = (a * tau * tau * (-math.expm1(-t0 / tau)) - tau * I_rrm) / I_rrm
    resid = abs(g(tau)) / tau_e
    return {"tau": float(tau), "T_M": float(T_M), "residual": float(resid), "V_R": float(V_R)}


def power_mosfet_from_datasheet(
        *, transfer: Dict[str, Any], R_ds_on: float, V_gs_spec: float,
        C_iss: Sequence[Sequence[float]], C_oss: Sequence[Sequence[float]],
        C_rss: Sequence[Sequence[float]], R_g: float = 0.0,
        rds_temperature: Optional[Dict[str, Sequence[float]]] = None,
        recovery: Optional[Dict[str, float]] = None,
        diode_static: Optional[Dict[str, float]] = None,
        tol: float = 0.01) -> Dict[str, Any]:
    """``add_power_mosfet`` keyword arguments from datasheet curves.

    * ``transfer``: ``{"v_gs": [...], "i_d": [...], "v_ds": V}`` → K, V_T
    * ``R_ds_on`` at ``V_gs_spec`` → ``R_d = R_ds_on − 1/(2K(V_gs_spec − V_T))``,
      refused if negative (the transfer curve and R_DS(on) disagree)
    * ``C_iss``, ``C_oss``, ``C_rss``: ``[(V_DS, C), ...]`` at V_GS = 0 →
      ``C_gs = C_iss − C_rss`` at the highest common V_DS; ``C_gd`` a table
      law of ``v_gd = −V_DS`` from C_rss (held at C_rss(0) for
      ``v_gd > 0``, which a V_GS = 0 curve does not cover); ``C_ds`` a
      table law from ``C_oss − C_rss``
    * ``rds_temperature``: ``{"T_j": [...], "r_norm": [...]}`` → tc1, tc2
    * ``recovery``: keyword arguments of :func:`fit_body_diode_recovery`;
      ``diode_static``: ``{"I_S": ..., "n": ...}`` (defaults 1e-12, 1)

    Returns ``{"kwargs": {...}, "residuals": {...}}``.
    """
    import pulsim as _p

    ch = fit_mosfet_channel(transfer["v_gs"], transfer["i_d"], v_ds=transfer["v_ds"])
    K, V_T = ch["K"], ch["V_T"]
    r_ch = 1.0 / (2.0 * K * (V_gs_spec - V_T))
    R_d = R_ds_on - r_ch
    if R_d < 0:
        raise ValueError(
            f"power_mosfet_from_datasheet: R_ds_on = {R_ds_on:.4g} Ω is below the channel's "
            f"own 1/(2K(V_GS − V_T)) = {r_ch:.4g} Ω at V_GS = {V_gs_spec:g} V — the transfer "
            "curve and R_DS(on) disagree.")
    iss, oss, rss = (np.asarray(x, dtype=float) for x in (C_iss, C_oss, C_rss))
    v_hi = min(iss[:, 0].max(), rss[:, 0].max())
    C_gs = float(np.interp(v_hi, iss[:, 0], iss[:, 1]) - np.interp(v_hi, rss[:, 0], rss[:, 1]))
    if C_gs <= 0:
        raise ValueError(f"power_mosfet_from_datasheet: C_iss − C_rss = {C_gs:.3g} F at {v_hi:g} V.")
    gd = fit_charge_table((-rss[::-1, 0]).tolist(), rss[::-1, 1].tolist(), tol=tol)
    c_ds = oss[:, 1] - np.interp(oss[:, 0], rss[:, 0], rss[:, 1])
    if np.any(c_ds <= 0):
        raise ValueError("power_mosfet_from_datasheet: C_oss − C_rss is not positive everywhere.")
    ds = fit_charge_table(oss[:, 0].tolist(), c_ds.tolist(), tol=tol)
    kw: Dict[str, Any] = dict(K=K, V_T=V_T, R_d=R_d, C_gs=C_gs,
                              C_gd=_p.ChargeLaw.table(gd.table),
                              C_ds=_p.ChargeLaw.table(ds.table), R_g=R_g)
    res: Dict[str, float] = {"transfer": ch["residual"], "C_rss": gd.residual,
                             "C_oss": ds.residual}
    if rds_temperature is not None:
        rt = fit_rdson_temperature(rds_temperature["T_j"], rds_temperature["r_norm"])
        kw.update(rds_tc1=rt["rds_tc1"], rds_tc2=rt["rds_tc2"])
        res["rds_temperature"] = rt["residual"]
    if recovery is not None:
        rr = fit_body_diode_recovery(**recovery)
        st = dict(diode_static or {})
        kw["body_diode"] = {"model": "lauritzen", "tau": rr["tau"], "T_M": rr["T_M"],
                            **{k: st[k] for k in ("I_S", "n") if k in st}}
        res["recovery"] = rr["residual"]
    elif diode_static is not None:
        kw["body_diode"] = {"model": "shockley", **diode_static}
    return {"kwargs": kw, "residuals": res}

