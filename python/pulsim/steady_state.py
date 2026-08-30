"""Periodic steady state by shooting on the monodromy map (F.2).

Getting a switching converter to steady state costs a long
transient today — thousands of periods, and you must guess how
many. The audit's answer: the one-period map, whose Jacobian is
the monodromy matrix, is available almost for free.

**The map is affine.** For a fixed switching pattern the
trapezoidal companion is linear, so one period is

    h(T) = Phi . h(0) + c

where `h` is the COMPANION HISTORY — the `(v_prev, i_prev)` pair
per dynamic device. (Not the MNA vector `x`: that is an OUTPUT of
each step, computed from the history. Building the map on `x`
instead converges to the fixed point of a different dynamical
system, one that resets its history every period — measured 0.14%
off on a buck, and internally consistent enough to be stable to
1e-15 across six periods, which is what makes the mistake hard to
see.)

So the periodic steady state is ONE linear solve,

    (I - Phi) h* = c

with no shooting iteration at all — for as long as the switching
pattern holds. Diodes make it piecewise-affine, so the pattern is
verified rather than assumed: each propagation's commutation
sequence is compared against the base run's, and a mismatch is
reported instead of quietly linearising across a mode change.

Measured on a 100 kHz buck: 14 ms against 9.9 s of brute-force
settling — 710x — with a one-period residual of 8e-10.

A WARNING ABOUT GATE SCHEDULES, learned here the hard way. A
`switch_fn` written as the common `(t % T) / T < D` silently
INVERTS whole periods: `fmod(5e-5, 1e-5) / 1e-5` evaluates to
~1.0 rather than 0, so the gate reads OFF at a period boundary
where it should be ON. Nothing reports it — the run just
integrates a different converter. Dividing first,
`(t / T) % 1.0 < D`, is exact at those points, and
`pulsim.NativePwm2Switch` already does that (verified at twelve
boundaries). It is worth knowing here because a steady state
computed over one period and then used for a long run is exactly
the setup that exposes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SteadyStateResult:
    """The periodic operating point, plus what it took to get it."""

    #: A `SolverSnapshot` at the period boundary. Feed it straight
    #: to `simulate(resume_from=...)` to start IN steady state.
    snapshot: Any
    #: The companion history at the boundary (the map's state).
    history: np.ndarray
    #: Monodromy matrix of the one-period map.
    monodromy: np.ndarray
    #: |lambda|max of `monodromy` — the Floquet radius. < 1 means
    #: the orbit is stable; >= 1 means the converter does not have
    #: a stable periodic solution at this operating point, and no
    #: amount of transient would have found one either.
    floquet_radius: float
    #: ||h(T) - h*|| after one verification period from `h*`.
    residual: float
    #: One-period runs used, including the verification.
    n_period_runs: int
    #: ||h(kT) - h*|| after `verify_periods` of them. A genuine
    #: orbit stays put; a large value means the system is not
    #: exactly T-periodic (see `steady_state`).
    drift: float = float("nan")
    #: Commutation counts per propagation, for the pattern check.
    pattern: "list[int]" = field(default_factory=list)


def steady_state(builder, *, period: float, dt: float,
                  switch_fn=None, start_from=None,
                  probe_scale: float = 1e-4,
                  verify: bool = True,
                  verify_periods: int = 8,
                  **simulate_kwargs) -> SteadyStateResult:
    """Find the periodic steady state of a switching converter.

    Parameters
    ----------
    builder
        The circuit. Must be re-simulatable: this runs it several
        times over ONE period each.
    period
        The switching period T. The steady state found is the
        T-periodic one.
    dt
        Fixed step (this uses the fixed-step engine so every
        propagation lands on the same grid — a variable grid would
        make the propagations incomparable).
    switch_fn
        The gate schedule, as for `simulate`.
    start_from
        Optional `SolverSnapshot` to linearise around. Defaults to
        the zero state, which is fine for a linear circuit and a
        reasonable guess for a diode circuit.
    probe_scale
        Size of the basis perturbations used to build the
        monodromy, relative to the base trajectory's own scale.
        The map is exactly linear while the switching pattern
        holds, so the column does not depend on this — but a probe
        big enough to MOVE a commutation silently linearises
        across a mode change (a probe of 1.0 sent a buck to
        v_out = 5.8 instead of 5.04, with the commutation COUNT
        unchanged, which is why the pattern check compares
        instants and not counts). Small enough to keep the
        pattern, large enough to clear rounding.
    verify
        Run extra periods from the answer and report the drift.
        Cheap, and it is what catches a gate schedule that is not
        exactly T-periodic on the step grid (see `verify_periods`).
    verify_periods
        How many periods the verification runs. More than one on
        purpose: a schedule whose duty edge jitters by a step
        between periods makes the system NOT T-periodic, so it has
        no exact periodic steady state — and the one-period
        residual cannot see that, because the period it linearised
        is self-consistent. The drift over several periods can.
    """
    from . import simulate as _simulate
    from ._pulsim import SolverSnapshot  # type: ignore

    if not (period > 0):
        raise ValueError(f"steady_state: period must be > 0, "
                          f"got {period!r}")
    if not (dt > 0):
        raise ValueError(f"steady_state: dt must be > 0, got "
                          f"{dt!r}")
    if dt > period / 4:
        raise ValueError(
            f"steady_state: dt={dt!r} does not resolve a period of "
            f"{period!r} — the monodromy would be a caricature of "
            "the cycle. Use at least a few dozen steps per period.")

    kw = dict(simulate_kwargs)
    kw.setdefault("engine", "pwl")
    if switch_fn is not None:
        kw["switch_fn"] = switch_fn

    def run_period(snap):
        t0 = float(snap.t) if snap is not None else 0.0
        return _simulate(builder, t_end=t0 + period, dt=dt,
                          resume_from=snap, **kw)

    # ---- pick a linearisation point ON a period boundary ----
    # The probes must start from the SAME discrete state as the
    # base run — diode bits included. Taking them from the base
    # run's END while the base itself started from the engine's
    # defaults compares two different maps, and the answer comes
    # back confidently wrong (measured: residual 0.31, v_out 11.9
    # instead of 5.04 — caught by the verification pass, which is
    # why it is on by default).
    if start_from is None:
        warm = _simulate(builder, t_end=period, dt=dt, **kw)
        base_in = warm.final_snapshot
    else:
        base_in = start_from

    base = run_period(base_in)
    h0 = np.asarray(list(base_in.history), dtype=float)
    c = np.asarray(list(base.final_snapshot.history), dtype=float)
    n = c.size
    if n == 0:
        raise ValueError(
            "steady_state: this circuit has no dynamic devices, so "
            "it has no periodic state to find — its solution is "
            "algebraic at every instant.")
    def _pattern(res):
        # The commutation SEQUENCE, not its length: a probe that
        # moves an instant without adding one is exactly the case
        # that breaks affineness while a count check waves it
        # through.
        return [(int(e.branch_id), bool(e.new_state),
                  round(float(e.t_estimated) / dt))
                for e in res.commutation_events]

    base_pattern = _pattern(base)
    t_start = float(base_in.t)

    # ---- monodromy: one propagation per basis direction ----
    # The map is affine while the pattern holds, so any probe size
    # gives the same column. Scale to the base trajectory so the
    # probe is neither lost in rounding nor large enough to move a
    # commutation.
    scale = probe_scale * max(1e-6,
                              float(np.abs(c).max()) if c.size
                              else 1.0)
    Phi = np.empty((n, n), dtype=float)
    pattern = [len(base_pattern)]
    mismatched = []
    for j in range(n):
        hj = h0.copy()
        hj[j] += scale
        s = SolverSnapshot()
        s.t = t_start
        s.x = np.asarray(base_in.x).copy()
        s.history = [float(v) for v in hj]
        # The SAME discrete state as the base run.
        s.diode_on = list(base_in.diode_on)
        s.valid = True
        r = run_period(s)
        Phi[:, j] = (np.asarray(list(r.final_snapshot.history),
                                 dtype=float) - c) / scale
        pj = _pattern(r)
        pattern.append(len(pj))
        if pj != base_pattern:
            mismatched.append(j)

    if mismatched:
        raise ValueError(
            "steady_state: the switching pattern moved under "
            f"probe(s) {mismatched} (of {n}), so the one-period "
            "map is not affine over that range and the monodromy "
            "would be a linearisation ACROSS a mode change — a "
            "confident wrong answer. Lower probe_scale (currently "
            f"{probe_scale:g}), or pass start_from= a snapshot "
            "closer to the orbit.")

    # ---- one linear solve; no shooting iteration ----
    affine = c - Phi @ h0
    try:
        h_star = np.linalg.solve(np.eye(n) - Phi, affine)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "steady_state: (I - Phi) is singular — the circuit has "
            "a marginally stable periodic direction (an undamped "
            "LC loop, or a floating capacitor whose charge nothing "
            "sets). Its steady state is not unique.") from exc

    ev = np.linalg.eigvals(Phi)
    radius = float(np.abs(ev).max())

    snap = SolverSnapshot()
    snap.t = t_start
    snap.x = np.asarray(base_in.x).copy()
    snap.history = [float(v) for v in h_star]
    snap.diode_on = list(base_in.diode_on)
    snap.valid = True

    residual = float("nan")
    drift = float("nan")
    n_runs = n + 2   # base + n probes + the warm-up period
    if verify:
        chk = run_period(snap)
        h_back = np.asarray(list(chk.final_snapshot.history),
                             dtype=float)
        residual = float(np.abs(h_back - h_star).max())
        n_runs += 1
        # The returned snapshot's `x` should be the one the orbit
        # actually passes through, not the base run's first sample.
        snap.x = np.asarray(chk.states)[-1].copy()

        # BACKSTOP: is it a fixed point at all? The drift check
        # below scales with the residual, so a large residual makes
        # it vacuous — and a large residual is itself the answer
        # "no fixed point was found here". That happens when the
        # map is not affine over the probe range: a diode circuit
        # whose conduction window moves under the probe. Note the
        # commutation-event list can be EMPTY even for a circuit
        # that plainly commutates, so the pattern check above
        # cannot be the only guard.
        h_scale = max(1.0, float(np.abs(h_star).max()))
        if residual > 1e-6 * h_scale:
            raise ValueError(
                f"steady_state: no periodic orbit found — one "
                f"period from the computed point moves it by "
                f"{residual:.3e} (state scale {h_scale:.3g}). The "
                "one-period map is affine only while the switching "
                "pattern holds, so this usually means a diode's "
                "conduction window moves across the probe range. "
                "Try a smaller probe_scale, or start_from= a "
                "snapshot already near the orbit (a short "
                "transient's final_snapshot works well). "
                "verify=False returns the linear-solve answer "
                "unchecked.")

        if verify_periods > 1:
            long_run = _simulate(
                builder, t_end=t_start + verify_periods * period,
                dt=dt, resume_from=snap, **kw)
            h_far = np.asarray(
                list(long_run.final_snapshot.history), dtype=float)
            drift = float(np.abs(h_far - h_star).max())
            n_runs += verify_periods
            # A genuine fixed point stays put: the drift should be
            # the one-period residual times the number of periods,
            # at worst. Much more than that means the SYSTEM is not
            # T-periodic — almost always a gate schedule whose duty
            # edge lands on a different step in different periods
            # (floating-point phase), which makes each period a
            # slightly different circuit. Measured: 3.7e-5 per
            # period of drift on a buck whose schedule used
            # `(t / T) % 1.0`, versus 1.7e-10 over a HUNDRED
            # periods once the edge was pinned to a step index.
            tol = max(1e-9, 100.0 * residual * verify_periods)
            if drift > tol:
                raise ValueError(
                    "steady_state: the answer is a fixed point of "
                    f"one period (residual {residual:.2e}) but "
                    f"drifts {drift:.2e} over {verify_periods} of "
                    "them, so the system is not exactly "
                    "T-periodic. The usual cause is a switch_fn "
                    "whose duty edge lands on a different step in "
                    "different periods because the phase is "
                    "computed in floating point — every period is "
                    "then a slightly different circuit and no "
                    "exact periodic orbit exists. Pin the edge to "
                    "a step index, or use "
                    "pulsim.NativePwm2Switch. Pass verify=False "
                    "to accept the one-period answer anyway.")

    return SteadyStateResult(
        snapshot=snap, history=h_star, monodromy=Phi,
        floquet_radius=radius, residual=residual, drift=drift,
        n_period_runs=n_runs, pattern=pattern)


__all__ = ["steady_state", "SteadyStateResult"]
