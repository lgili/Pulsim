"""Time-weighted comparison metrics for the reference harness.

Every waveform statistic here is weighted by TIME, never by sample: the
variable-step engines cluster samples around events, and an unweighted
``y.mean()`` once reported 3.04 V where the time-weighted value was
6.33 V (and was filed as an engine defect). Integrals are trapezoidal
on the piecewise-linear waveform through the samples.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np

Reference = Union[Callable[[np.ndarray], np.ndarray], tuple]


def first_kept(t, exclude_t0: bool) -> int:
    """Index of the first sample a comparison keeps. ``exclude_t0`` drops the
    initial sample(s) (on pwl / trbdf2 the stored initial state of an algebraic
    node, not the t = 0+ solution: design.md, Context) -- unless that leaves fewer
    than two samples: a short trace (the exact engine stores a DC circuit as its
    two ends under the output rule) is then compared whole, not called
    undersampled."""
    t = np.asarray(t, dtype=float)
    if not exclude_t0:
        return 0
    # A right limit one ulp after t[0] (with_right_limits on a restart-rule
    # run) belongs to the initial instant too.
    k = int(np.searchsorted(t, np.nextafter(t[0], np.inf), side="right"))
    return k if t.size - k >= 2 else 0


def _window(t, y, exclude_t0: bool):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.shape != y.shape or t.size < 2 or not t[-1] > t[0]:
        raise ValueError(f"a trace needs >= 2 samples spanning a time window to be compared, "
                         f"got {t.shape} / {y.shape}")
    k = first_kept(t, exclude_t0)
    return t[k:], y[k:]


def _ref_window(t, ref, exclude_t0: bool):
    """A short trace kept whole (first_kept) is compared from the reference's own
    first kept sample on: the reference's initial state is not t = 0+ either."""
    if not exclude_t0 or callable(ref) or first_kept(t, True) > 0:
        return ref
    tr, yr = (np.asarray(a, dtype=float) for a in ref)
    k = first_kept(tr, True)
    return tr[k:], yr[k:]


def time_mean(t, y) -> float:
    """Time-weighted mean over the trace's own span."""
    t = np.asarray(t, dtype=float)
    return float(np.trapezoid(np.asarray(y, dtype=float), t) / (t[-1] - t[0]))


def _on_grid(t, y, ref):
    """Return (grid, y, ref) on a common grid.

    ``ref`` is a callable (analytic, evaluated on the trace's own grid) or a
    ``(t_ref, y_ref)`` trace (union of both grids, linear interpolation).
    """
    if callable(ref):
        return t, y, np.asarray(ref(t), dtype=float)
    tr, yr = (np.asarray(a, dtype=float) for a in ref)
    lo, hi = max(t[0], tr[0]), min(t[-1], tr[-1])
    g = np.union1d(t, tr)
    g = g[(g >= lo) & (g <= hi)]
    return g, np.interp(g, t, y), np.interp(g, tr, yr)


def _inside(g, exclude):
    """Mask of grid points strictly inside one of the ``(lo, hi)`` intervals."""
    iv = np.asarray(exclude, dtype=float).reshape(-1, 2)
    iv = iv[np.argsort(iv[:, 0])]
    reach = np.maximum.accumulate(iv[:, 1])        # furthest end among intervals starting earlier
    k = np.searchsorted(iv[:, 0], g, side="left") - 1  # last interval with lo < g
    return (k >= 0) & (g < reach[np.maximum(k, 0)])


def err_rms(t, y, ref: Reference, *, abs_scale: float = 0.0, exclude_t0: bool = False,
            exclude=()) -> float:
    """``sqrt(∫(y − ref)² dt / window) / span``, span = max(ref) − min(ref)
    floored at ``abs_scale`` (for a flat reference).

    ``exclude``: ``(lo, hi)`` time intervals left out of the integral and the
    window (the edge DIAGNOSTIC of the differential runner, never its gate)."""
    ref = _ref_window(t, ref, exclude_t0)
    t, y = _window(t, y, exclude_t0)
    g, a, r = _on_grid(t, y, ref)
    span = max(float(r.max() - r.min()), abs_scale)
    if span <= 0.0:
        raise ValueError("flat reference: give abs_scale > 0")
    return _rms(g, (a - r) ** 2, exclude) / span


def _rms(g, d2, exclude):
    """sqrt of the time mean of ``d2`` on ``g``, ``exclude`` intervals left out."""
    if not len(exclude):
        return float(np.sqrt(np.trapezoid(d2, g) / (g[-1] - g[0])))
    seg = np.diff(g) * 0.5 * (d2[1:] + d2[:-1])
    keep = ~_inside(0.5 * (g[1:] + g[:-1]), exclude)
    window = float(np.diff(g)[keep].sum())
    if window <= 0.0:
        raise ValueError("every interval excluded")
    return float(np.sqrt(seg[keep].sum() / window))


def err_rms_symmetric(a: tuple, b: tuple, *, abs_scale: float = 0.0, exclude=()) -> float:
    """:func:`err_rms` with neither trace the reference, for a DIAGNOSTIC:
    ``err_rms_symmetric(a, b) == err_rms_symmetric(b, a)`` bit for bit.

    ``a``, ``b`` are ``(t, y)`` traces. Both lose their initial sample(s) (the
    stored initial state of pwl / trbdf2, not the t = 0+ solution: kept on one
    side only, it was interpolated into the other's first step) and the span is
    the larger of the two traces' spans on the common window (floored at
    ``abs_scale``). ``err_rms`` keeps the reference's first sample and divides by
    the reference's span: on an ideal boost whose ``V(out)`` holds 24 V (6 mV of
    ripple) two converged traces read 0.030 one way and 0.0023 the other."""
    ta, ya = _window(*a, exclude_t0=True)
    tb, yb = _window(*b, exclude_t0=True)
    g, va, vb = _on_grid(ta, ya, (tb, yb))
    if len(g) < 2 or g[-1] <= g[0]:
        raise ValueError("the two traces share no time window")
    span = max(float(np.ptp(va)), float(np.ptp(vb)), abs_scale)
    if span <= 0.0:
        raise ValueError("flat traces: give abs_scale > 0")
    return _rms(g, (va - vb) ** 2, exclude) / span


def rule_ratio(t, y, ref: Reference, *, rtol: float, atol: float, exclude=()) -> float:
    """What the exact engine's output rule promises, checked: between events the
    linear interpolation of the stored samples is within ``max(rtol * range, atol)``
    of the solution (``simulate(output_rtol=)``). Returns the worst
    ``|interp(m) - ref(m)| / (rtol * range + atol)`` over the trace's interval
    midpoints ``m`` (``range``: the reference's range over the run, at least 1e-3 of
    its peak; a sum, not a max, of the two tolerances, so the reference's own error
    near atol is absorbed). Intervals of zero length or overlapping an ``exclude``
    interval (a recorded switch-state change: the rule promises nothing across an
    event) are skipped. <= 1 is within the promise; nan when no interval is left.
    A callable ``ref`` (a closed form) is read at eleven interior instants of every line,
    against ``max(rtol * range, atol)`` exactly (the promise, no slack)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.diff(t) > 0
    lo, hi = t[:-1][keep], t[1:][keep]
    m = 0.5 * (lo + hi)
    ym = 0.5 * (y[:-1][keep] + y[1:][keep])
    if len(exclude):
        iv = np.asarray(exclude, dtype=float).reshape(-1, 2)
        bad = np.zeros(m.shape, bool)
        for a, b in iv:
            bad |= (lo < b) & (hi > a)
        if not callable(ref):
            m, ym = m[~bad], ym[~bad]
    if callable(ref):   # a closed form: every line at several interior instants (session 7)
        lo_, hi_ = t[:-1][keep], t[1:][keep]
        y0, y1 = y[:-1][keep], y[1:][keep]
        if len(exclude):
            lo_, hi_, y0, y1 = lo_[~bad], hi_[~bad], y0[~bad], y1[~bad]
        f = np.array([1e-3, 1e-2, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 0.999])
        m = (lo_[:, None] + (hi_ - lo_)[:, None] * f[None, :]).ravel()
        ym = (y0[:, None] + (y1 - y0)[:, None] * f[None, :]).ravel()
        rm = np.asarray(ref(m), dtype=float)
        rr = np.asarray(ref(t), dtype=float)
    else:
        tr, yr = (np.asarray(a, dtype=float) for a in ref)
        inside = (m >= tr[0]) & (m <= tr[-1])
        m, ym = m[inside], ym[inside]
        rm, rr = np.interp(m, tr, yr), yr
    if not m.size:
        return float("nan")
    rng = max(float(np.ptp(rr)), 1e-3 * float(np.abs(rr).max()))
    if callable(ref):   # the promise itself: max(rtol·range, atol), no added slack
        return float(np.abs(ym - rm).max() / max(rtol * rng, atol))
    return float(np.abs(ym - rm).max() / (rtol * rng + atol))


def rule_ratio_flow(t, y, td, yd, *, rtol: float, atol: float, exclude=()) -> float:
    """:func:`rule_ratio` against a DENSE trace of the exact flow (the exact engine at a
    much smaller ``output_rtol``, the same events): at each chord's midpoint the flow is
    the quadratic through three samples bracketing it among the dense samples inside
    the chord and the chord's own two ends (exact samples too) -- one smooth piece: a
    rule chord holds no event and no source corner. The rule's promise is about the
    exact trajectory: trbdf2 at rtol 1e-8 read linearly between its steps put 48 of the
    54 cells past 1 that sit at 0.96..0.9997 against the flow (group F, session 4)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    td = np.asarray(td, dtype=float)
    yd = np.asarray(yd, dtype=float)
    keep = np.flatnonzero(np.diff(t) > 0)
    lo, hi = t[keep], t[keep + 1]
    if len(exclude):
        bad = np.zeros(lo.shape, bool)
        for a, b in np.asarray(exclude, dtype=float).reshape(-1, 2):
            bad |= (lo < b) & (hi > a)
        keep, lo, hi = keep[~bad], lo[~bad], hi[~bad]
    if not keep.size:
        return float("nan")
    i0 = np.searchsorted(td, lo, side="right")
    i1 = np.searchsorted(td, hi, side="left")
    worst = 0.0
    for q, k in enumerate(keep):
        m = 0.5 * (lo[q] + hi[q])
        ts = np.r_[lo[q], td[i0[q]:i1[q]], hi[q]]
        ys = np.r_[y[k], yd[i0[q]:i1[q]], y[k + 1]]
        if ts.size >= 3:   # bracket m (never extrapolate): the nearest on each side + the next nearest
            j = int(np.searchsorted(ts, m))
            c = [j - 1, j]
            c.append(j - 2 if j + 1 >= ts.size or (j >= 2 and m - ts[j - 2] < ts[j + 1] - m) else j + 1)
            c = sorted(c)
            h1, h2 = ts[c[1]] - ts[c[0]], ts[c[2]] - ts[c[1]]
            # a lopsided stencil extrapolates a short piece's curvature over a long one:
            # there the dense run's own chord (its promise: within its smaller rtol) is read
            # (also with ONE dense sample inside: ll09's 1.8 ns + 985 ns stencil read 5.8x)
            c = c if max(h1, h2) <= 4.0 * min(h1, h2) else [j - 1, j]
            ts, ys = ts[c], ys[c]
        if ts.size < 3:
            f = ys[0] + (ys[-1] - ys[0]) * (m - ts[0]) / (ts[-1] - ts[0])
        else:
            f = (ys[0] * (m - ts[1]) * (m - ts[2]) / ((ts[0] - ts[1]) * (ts[0] - ts[2]))
                 + ys[1] * (m - ts[0]) * (m - ts[2]) / ((ts[1] - ts[0]) * (ts[1] - ts[2]))
                 + ys[2] * (m - ts[0]) * (m - ts[1]) / ((ts[2] - ts[0]) * (ts[2] - ts[1])))
        worst = max(worst, abs(0.5 * (y[k] + y[k + 1]) - f))
    rng = max(float(np.ptp(yd)), 1e-3 * float(np.abs(yd).max()))
    return float(worst / (rtol * rng + atol))


def err_max(t, y, ref: Reference, *, abs_scale: float = 0.0, exclude_t0: bool = False,
            exclude=()) -> float:
    """``max|y − ref| / span`` at the same points as :func:`err_rms` (grid
    points strictly inside an ``exclude`` interval are skipped)."""
    ref = _ref_window(t, ref, exclude_t0)
    t, y = _window(t, y, exclude_t0)
    g, a, r = _on_grid(t, y, ref)
    span = max(float(r.max() - r.min()), abs_scale)
    if span <= 0.0:
        raise ValueError("flat reference: give abs_scale > 0")
    d = np.abs(a - r)
    if len(exclude):
        d = d[~_inside(g, exclude)]
    return float(d.max() / span)



def exact_dev(y, cand) -> np.ndarray:
    """|y_i − nearest candidate truth value at t_i| (``cand``: one row per
    sample, NaN-padded; see ``exact_truth.Solver.at_float``). A non-finite
    sample is an infinite deviation."""
    y = np.asarray(y, dtype=float)
    d = np.abs(np.asarray(cand, dtype=float) - y[:, None])
    e = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    return np.where(np.isfinite(y), e, np.inf)


def err_exact(t, y, cand, *, span: float, gap=None, exclude_t0: bool = True) -> float:
    """The OLD ``exact`` measure, kept only to print the previous definition
    beside :func:`exact_parts` for one release (``exact_old``). It misprices a
    displaced event (charged its row's weight, not jump² × δ, once δ exceeds
    the snap window) and grades the right limit at t = 0⁺; do not use it.

    ``sqrt((Σ w_i e_i² + Σ c_i) / Σ w_i) / span``
    over the trace's OWN samples, e_i = :func:`exact_dev`, w_i the trapezoid
    time weight of sample i (half of each adjacent interval). No
    interpolation of the trace, so a perfect trace scores the truth's
    evaluation error on any grid (no floor); what a trace does between its
    samples is not graded.

    ``gap`` (from ``at_float(..., gaps=True)``): a sample that matches a
    truth branch across an event (gap g_i > 0, the event displaced by g_i)
    is charged the L2 of that displacement, c_i = (jump at t_i)² g_i, so the
    snap window hides no timing error (two samples inside one displaced
    interval are both charged: conservative). Gaps within 4 ulp of t_i are
    not charged: a float time cannot place an event closer."""
    t = np.asarray(t, dtype=float)
    cand = np.asarray(cand, dtype=float)
    if np.any(np.diff(t) < 0):
        raise ValueError("times must be non-decreasing")
    e = exact_dev(y, cand)
    c = np.zeros(t.size)
    if gap is not None:
        rows = np.arange(t.size)
        d = np.abs(cand - np.asarray(y, dtype=float)[:, None])
        k = np.nanargmin(np.where(np.isnan(d), np.inf, d), axis=1)
        g = np.where(np.isnan(gap), np.inf, gap)
        g = np.where(g <= 4 * np.spacing(np.abs(t))[:, None], 0.0, g)   # below the float time's own resolution
        # the segment holding t_i (several at an exact event time: the one nearest the match)
        near = np.abs(cand - cand[rows, k][:, None])
        kc = np.argmin(np.where(g == g.min(axis=1, keepdims=True), np.nan_to_num(near, nan=np.inf), np.inf), axis=1)
        jump = cand[rows, kc] - cand[rows, k]
        c = np.where((k != kc) & np.isfinite(e), np.nan_to_num(jump * jump * g[rows, k], posinf=np.inf), 0.0)
    if exclude_t0:
        k0 = int(np.searchsorted(t, t[0], side="right"))
        t, e, c = t[k0:], e[k0:], c[k0:]
    if t.size < 2 or span <= 0.0:
        raise ValueError("need >= 2 graded samples and span > 0")
    if not np.all(np.isfinite(e)):
        return float("inf")
    h = np.diff(t)
    w = np.zeros(t.size)
    w[:-1] += h / 2
    w[1:] += h / 2
    return float(np.sqrt((np.sum(w * e * e) + np.sum(c)) / w.sum()) / span)


# ---------------------------------------------------------------- exact --
# The `exact` measure (three parts). One implementation for every tool that
# grades an engine against the exact truth (exact_scoreboard.grade and what
# calls it: the accuracy summary, the rehearsal, the tests).

EVENT_WINDOW = 2.0          # matching window, in units of h
PROJECTION_SETTLE = 1e-9    # rule (1): settling tolerance, of span (or 1e-6 of the jump, the larger)
PROJECTION_STEPS = 60       # rule (1): dyadic offsets R·2^-k, k < this, ...
PROJECTION_SUB = 8          # ... each interval cut in this many at irrational fractions (no aliasing on 2^-k)
PROJECTION_REACH = 32.0     # rule (1): the truth is followed to this many windows (W bounds τ; ~21 τ settle)


def collapsed_transient(right, truth_right, truth_after, t_truth, window, atol, settle, horizon=np.inf):
    """Rule (1), the projected transient. ``right``: the engine's recorded
    right value at an event where it reports a projection; ``truth_right``:
    the truth's right limit at its instant ``t_truth``; ``truth_after(t, s)``:
    the truth's right-side values at ``t + s``; ``window``: the engine's
    declared parasitic window W (s: the longest time constant it projects);
    ``atol``: the limits tolerance; ``settle``: the settling tolerance (both
    absolute); ``horizon``: the time to the next truth event (the truth is
    one transient only up to there).

    The truth is evaluated on R = min(PROJECTION_REACH·W, horizon) times
    2^-k, each interval cut at irrational fractions. Returns ``(tau_eff,
    cost, residual)`` when the truth itself goes to the engine's value: from
    the dyadic offset tau_eff on, one doubling moves it by at most
    ``settle`` and EVERY evaluated value up to R is within ``atol`` of
    ``right``. ``cost`` = ∫_0^tau_eff (truth − right)² ds (trapezoid on the
    evaluated offsets: an overshoot or a ringing is priced by its size, not
    by the jump), ``residual`` = |right − truth(tau_eff)|, what is left of the
    limit once the transient is over. None otherwise: the truth does not
    settle there, or settles elsewhere (a wrong limit), or its own right
    limit is already within ``atol`` (nothing collapsed: an impulse the
    truth makes too)."""
    reach = min(PROJECTION_REACH * window, horizon)
    if abs(right - truth_right) <= atol or not reach > 0.0:
        return None
    nodes = reach * 2.0 ** -np.arange(PROJECTION_STEPS - 1, -1, -1, dtype=float)   # ascending, ends at reach
    frac = np.sort((np.arange(1, PROJECTION_SUB) * 0.6180339887498949) % 1.0)
    lo = np.r_[0.0, nodes[:-1]]
    s = np.sort(np.r_[nodes, (lo[:, None] + (nodes - lo)[:, None] * frac[None, :]).ravel()])
    s = s[s > 0.0]
    v = np.asarray(truth_after(t_truth, s), dtype=float)
    vn = v[np.searchsorted(s, nodes)]
    ok = np.abs(np.diff(vn)) <= settle
    k = 0 if ok.all() else ok.size - int(np.argmin(ok[::-1]))      # vn[k:] settled
    if k >= vn.size - 1:
        return None
    i = int(np.searchsorted(s, nodes[k]))
    if not np.all(np.abs(v[i:] - right) <= atol):
        return None
    ss, e2 = np.r_[0.0, s[:i + 1]], np.r_[truth_right - right, v[:i + 1] - right] ** 2
    return float(nodes[k]), float(np.sum(np.diff(ss) * (e2[1:] + e2[:-1]) / 2)), float(abs(right - v[i]))


def match_events(truth, engine, window: float):
    """Match engine events to truth events by key and order.

    ``truth``, ``engine``: sequences of ``(t, key)``; a key names one device
    and the state it enters (``"D1+"``). Per key, in time order, each truth
    event takes the engine event of the same key nearest to it within
    ``window`` among those after the key's previous match (order is kept: a
    chattering engine cannot match one truth event twice, and the extra
    flips are spurious); a truth event with none is missing. A cluster of
    truth events closer than h is matched one by one the same way.
    Returns ``(pairs, missing, spurious)``: ``[(i_truth, j_engine)]`` and two
    index lists."""
    pairs, missing, spurious = [], [], []
    for key in {k for _, k in truth} | {k for _, k in engine}:
        T = sorted((i for i, (_, k) in enumerate(truth) if k == key), key=lambda i: truth[i][0])
        E = sorted((j for j, (_, k) in enumerate(engine) if k == key), key=lambda j: engine[j][0])
        q0 = 0
        for i in T:
            ti = truth[i][0]
            while q0 < len(E) and engine[E[q0]][0] < ti - window:   # too early for this or any later one
                spurious.append(E[q0])
                q0 += 1
            best = None
            for q in range(q0, len(E)):
                d = engine[E[q]][0] - ti
                if d > window:
                    break
                if best is None or abs(d) < abs(engine[E[best]][0] - ti):
                    best = q
            if best is None:
                missing.append(i)
                continue
            spurious += E[q0:best]
            pairs.append((i, E[best]))
            q0 = best + 1
        spurious += E[q0:]
    return sorted(pairs), sorted(missing), sorted(spurious)


def exact_parts(t, y, cand, *, span: float, h: float, truth_events=(), engine_events=(),
                right_rows=(), events_available: bool = True, window: float = EVENT_WINDOW,
                truth_after=None, parasitic: float = 0.0, limits_tol: float = 1e-3) -> dict:
    """The ``exact`` measure of one column, in three parts (fractions of ``span``).

    ``t``, ``y``: the engine's STORED rows (left limits); ``cand``: the truth
    at those times, one column per one-sided value (``at_float(..., 0.0)``;
    two only where t_i is a truth event instant, where either is right).
    ``truth_events``: ``(t_e, key, left, right)``, the truth's one-sided
    values of this column at its device events. ``engine_events``:
    ``(t_e, key, left, right, truth_left, truth_right)``, the engine's own
    instant and one-sided values (``right`` None when the engine records no
    right limit) and the truth's at that same instant (used when unmatched).
    ``right_rows``: ``(t, value, truth_right)``, every other right limit the
    engine records. ``h``: the unit of the reported instant error.

    smooth: ``sqrt(Σ w_i e_i² / Σ w_i)``, e_i = :func:`exact_dev`, over the
      stored rows except the initial instant and the rows inside a matched
      event's displacement, ``[min(t_engine, t_truth), max(...)]`` (where the
      engine is on the other branch: priced by ``instants``; the row at the
      engine's instant is its left limit, graded by ``limits``); w_i the
      trapezoid weights of the grid; no snap window, no timing charge. Every
      other row is graded, the first one solved after an event included.
      None if no row is left.
    instants: ``sqrt((Σ_e jump_e² |δ_e| + Σ_m jump_m² d_m) / T)``: δ_e =
      engine − truth instant of each matched truth instant (the largest over
      its devices; 0 within 4 ulp, a float time's own resolution), jump the
      truth's right − left: the L2 cost of the displacement; a MISSING truth
      event is charged as displaced by d_m = the time to that device's next
      truth event, at most the window (beyond it the rows carry the error).
      Also max |δ|/h over the matched. None (not available) when the engine
      exposes no event record: then the rows carry every timing error.
    limits: ``max |engine − truth|`` over every one-sided value the engine
      records at an instant after the initial one: at a matched event its left
      (and right) value against the truth's limits at the TRUTH's instant; at
      an unmatched engine event and at every other right-limit row, against
      the truth's limits at the engine's instant.
    rule (1), a projected transient: at a matched event where the engine
      reports a projection and records a right value the truth does not
      have at its instant but SETTLES to, followed to PROJECTION_REACH ×
      ``parasitic`` and not past the next truth event
      (:func:`collapsed_transient`: within ``limits_tol`` of span from
      tau_eff on, settled to PROJECTION_SETTLE of span or 1e-6 of the jump),
      that right value is a
      transient collapsed in time, not a wrong limit: it is charged in
      ``instants`` by its true L2 cost ∫(truth − engine right)² over
      [t, t + tau_eff] (once per truth instant), and in ``limits`` by its
      distance to the settled truth at t + tau_eff (the part of it that IS a
      wrong limit). Counted in
      ``n_projected`` (``projected``: the truth instants). Either condition
      failing leaves the right value graded as a limit.

    Events at or after the last stored row are not graded (the record ends
    there). Unmatched events are listed (``missing`` / ``spurious``); a
    spurious one is not charged a duration (no rows are left out around it:
    its effect is in ``smooth`` and its one-sided values in ``limits``). An
    event displaced by more than the window is a missing + spurious pair.
    Engine events at the
    initial instant without a match are the engine's initial mode selection
    (``n_initial``), not spurious. ``summary`` = the largest part, ``part``
    its name; a non-finite value anywhere, the engine's or the truth's,
    scores inf (part ``non-finite``)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    cand = np.asarray(cand, dtype=float).reshape(t.size, -1)
    if np.any(np.diff(t) < 0):
        raise ValueError("times must be non-decreasing")
    if t.size < 2 or span <= 0.0 or h <= 0.0:
        raise ValueError("need >= 2 stored rows, span > 0 and h > 0")
    t0 = t[0]
    t0p = np.nextafter(t0, np.inf)
    T = float(t[-1] - t0)
    # the record ends at the last stored row: an event there or later is not graded on either side
    truth_events = [e for e in truth_events if e[0] < t[-1]]
    engine_events = [e for e in engine_events if e[0] < t[-1]]
    pairs, missing, spurious = match_events([(e[0], e[1]) for e in truth_events],
                                            [(e[0], e[1]) for e in engine_events], window * h)
    initial = [j for j in spurious if engine_events[j][0] <= t0p]
    spurious = [j for j in spurious if engine_events[j][0] > t0p]
    out = dict(missing=[(float(truth_events[i][0]), truth_events[i][1]) for i in missing],
               spurious=[(float(engine_events[j][0]), engine_events[j][1]) for j in spurious],
               n_initial=len(initial), n_matched=len(pairs), instants_available=bool(events_available))
    recorded = [y, np.nanmin(np.where(np.isnan(cand), np.inf, cand), axis=1)]
    recorded += [np.array([v for e in engine_events for v in e[2:4] if v is not None], dtype=float),
                 np.array([v for e in engine_events for v in e[4:6]], dtype=float),
                 np.array([v for e in truth_events for v in e[2:4]], dtype=float),
                 np.array([v for r in right_rows for v in r[1:3]], dtype=float)]
    if not all(np.all(np.isfinite(a)) for a in recorded):
        out.update(summary=float("inf"), part="non-finite", smooth=float("inf"), instants=None,
                   limits=float("inf"), smooth_rows=0, kept=np.zeros(t.size, dtype=bool),
                   n_projected=0, projected=[])
        return out

    # (a) smooth: every row but those inside a matched event's displacement
    near = np.zeros(t.size, dtype=bool)
    for i, j in pairs:
        lo, hi = sorted((truth_events[i][0], engine_events[j][0]))
        near[np.searchsorted(t, lo, side="left"):np.searchsorted(t, hi, side="right")] = True
    graded = t > t0
    w = np.zeros(t.size)
    g = t[graded]
    w_g = np.zeros(g.size)
    w_g[:-1] += np.diff(g) / 2
    w_g[1:] += np.diff(g) / 2
    w[graded] = w_g
    kept = graded & ~near
    e = exact_dev(y, cand)
    out["smooth_rows"] = int(kept.sum())
    out["kept"] = kept
    out["smooth"] = (float(np.sqrt(np.sum(w[kept] * e[kept] ** 2) / w[kept].sum()) / span)
                     if kept.any() and w[kept].sum() > 0 else None)

    # rule (1): matched events whose right value is a collapsed transient
    collapsed = {}                     # engine index -> (tau_eff, cost, residual, truth instant)
    for i, j in pairs:
        ev = engine_events[j]
        if len(ev) > 6 and ev[6] and ev[3] is not None and truth_after is not None and ev[0] > t0p:
            ti, tr_ = truth_events[i][0], truth_events[i][3]
            # one transient only up to the next truth event (any device)
            nxt = [e[0] for e in truth_events if e[0] > ti + 4 * np.spacing(abs(ti))]
            horizon = min(nxt, default=np.inf) - ti
            c = collapsed_transient(ev[3], tr_, truth_after, ti, parasitic, limits_tol * span,
                                    max(PROJECTION_SETTLE * span, 1e-6 * abs(tr_ - ev[3])), horizon)
            if c is not None:
                collapsed[j] = (c[0], c[1], c[2], ti)
    out["projected"] = sorted({float(c[3]) for c in collapsed.values()})
    out["n_projected"] = len(out["projected"])

    # (b) instants: per matched truth instant, its largest displacement
    worst = {}
    for i, j in pairs:
        d = float(engine_events[j][0] - truth_events[i][0])
        te = float(truth_events[i][0])
        if abs(d) <= 4 * np.spacing(abs(te)):
            d = 0.0                    # below the float time's own resolution
        if te not in worst or abs(d) > abs(worst[te][0]):
            worst[te] = (d, truth_events[i][3] - truth_events[i][2])
    if events_available:
        cost = sum(jmp * jmp * abs(d) for d, jmp in worst.values())
        miss = {}
        for i in missing:
            te, key = truth_events[i][:2]
            dev = key[:-1]
            nxt = [e[0] for e in truth_events if e[1][:-1] == dev and e[0] > te]
            d = min(min(nxt, default=t[-1]) - te, window * h)
            jmp = truth_events[i][3] - truth_events[i][2]
            miss[te] = max(miss.get(te, 0.0), jmp * jmp * d)
        cost += sum(miss.values())
        coll = {}                      # one transient per truth instant: its devices share it
        for _, c, _, ti in collapsed.values():
            coll[ti] = max(coll.get(ti, 0.0), c)
        cost += sum(coll.values())
        out["instants"] = float(np.sqrt(cost / T) / span) if T > 0 else 0.0
        dmax = max(worst.values(), key=lambda x: abs(x[0]))[0] if worst else 0.0
        out["max_dt_s"] = dmax
        out["max_dt_h"] = dmax / h
    else:
        out["instants"] = None

    # (c) one-sided limits, after the initial instant
    errs = []
    for i, j in pairs:
        te, _, left, right = engine_events[j][:4]
        if te > t0p:
            errs.append((te, abs(left - truth_events[i][2])))
            if j in collapsed:
                errs.append((te, collapsed[j][2]))
            elif right is not None:
                errs.append((te, abs(right - truth_events[i][3])))
    for j in spurious:
        te, _, left, right, tl, tr = engine_events[j][:6]
        errs.append((te, abs(left - tl)))
        if right is not None:
            errs.append((te, abs(right - tr)))
    errs += [(tr, abs(v - tv)) for tr, v, tv in right_rows if tr > t0p]
    if errs:
        a = np.array([x for _, x in errs])
        k = int(np.argmax(a))
        out.update(limits=float(a[k] / span), limits_rms=float(np.sqrt(np.mean(a * a)) / span),
                   n_limits=len(errs), limits_worst_t=float(errs[k][0]))
    else:
        out.update(limits=None, n_limits=0)

    parts = {p: out[p] for p in ("smooth", "instants", "limits") if out[p] is not None}
    if not parts:
        raise ValueError("nothing graded: no smooth row, no event, no recorded limit")
    out["part"] = max(parts, key=parts.get)
    out["summary"] = parts[out["part"]]
    return out
