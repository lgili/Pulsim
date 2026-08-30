"""Sampled-data frequency response of a switching converter (F.2).

A Bode plot of a converter costs one transient per frequency
today: `run_fra` needs 2.9 s per point on a 100 kHz buck, so a
20-point sweep is 58 s and a real 100-point one is five minutes.
It is the control engineer's daily bottleneck, and it is
avoidable.

Around the periodic orbit the SAMPLED dynamics are linear:

    h[k+1] = Phi.h[k] + B.u[k]
    y[k]   = C.h[k]   + D.u[k]

with `Phi` the monodromy matrix `steady_state` already computes.
`B`, `C` and `D` are one one-period run each. So the entire Bode
is

    H(z) = C (zI - Phi)^-1 B + D,   z = exp(j.2.pi.f.T)

— a handful of period runs TOTAL, and then one small linear solve
per frequency point. The cost stops depending on how many points
you ask for.

Measured on the 100 kHz buck, control-to-output:

    run_fra, 20 points      57.7 s      (2.9 s / point)
    this,    60 points      33 ms       (0.55 ms / point)

and it agrees with the analytic `Vin / (1 + sL/R + s^2 LC)` to
**0.77%** everywhere below 5 kHz, straight through a Q = 8
resonance. Above about a fifth of the switching frequency the two
part company — correctly: the sampled model carries the sampling
and hold that the continuous formula omits, and near Nyquist that
is the real behaviour, not an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FrequencyResponse:
    """`H(f)` plus the state-space it came from."""

    frequencies: np.ndarray
    #: Complex response at each frequency.
    H: np.ndarray
    magnitude_db: np.ndarray
    phase_deg: np.ndarray
    #: The sampled state-space around the orbit.
    Phi: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: float
    #: The steady state it was linearised about.
    steady: Any
    #: One-period runs spent building it.
    n_period_runs: int


def frequency_response(builder, *, period: float, dt: float,
                        switch_fn_of, output: str,
                        frequencies,
                        u0: float = 0.0,
                        du: float | None = None,
                        steady=None,
                        **simulate_kwargs) -> FrequencyResponse:
    """Sampled-data Bode of `output` with respect to an input `u`.

    Parameters
    ----------
    switch_fn_of
        `u -> switch_fn`. The input is whatever this parameterises
        — duty for control-to-output, phase shift for a DAB, and
        so on. It must be a genuine family: `switch_fn_of(u0)`
        defines the operating point and `switch_fn_of(u0 + du)`
        the perturbation.
    output
        Node or device name, resolved the way `result.v` /
        `result.i` resolve it. Its value at the PERIOD BOUNDARY is
        the sampled output — which is what a digital controller
        actually sees.
    u0, du
        Operating point and perturbation size. `du` defaults to
        one step of the period (`dt / period`), the finest
        perturbation a fixed grid can actually represent: asking
        for less does not move the gate at all and returns a
        column of zeros.
    steady
        A `SteadyStateResult` to reuse. Recomputed if omitted.
    """
    from . import simulate as _simulate
    from .steady_state import steady_state as _steady_state
    from ._pulsim import SolverSnapshot  # type: ignore

    freqs = np.asarray(frequencies, dtype=float)
    if freqs.size == 0:
        raise ValueError("frequency_response: no frequencies")
    if np.any(freqs <= 0):
        raise ValueError(
            "frequency_response: frequencies must be positive")
    f_nyq = 0.5 / period
    if float(freqs.max()) >= f_nyq:
        raise ValueError(
            f"frequency_response: {float(freqs.max()):g} Hz is at "
            f"or above the sampling Nyquist {f_nyq:g} Hz "
            f"(= 1/2T). A sampled model says nothing there — the "
            "response aliases. Sweep below it.")
    if du is None:
        du = dt / period
    if not (du > 0):
        raise ValueError(
            f"frequency_response: du must be > 0, got {du!r}")

    kw = dict(simulate_kwargs)
    kw.setdefault("engine", "pwl")

    if steady is None:
        steady = _steady_state(builder, period=period, dt=dt,
                                switch_fn=switch_fn_of(u0), **kw)
    Phi = np.asarray(steady.monodromy, dtype=float)
    h0 = np.asarray(steady.history, dtype=float)
    snap = steady.snapshot
    n = h0.size
    n_runs = int(steady.n_period_runs)

    def one_period(hist, sfn):
        s = SolverSnapshot()
        s.t = float(snap.t)
        s.x = np.asarray(snap.x).copy()
        s.history = [float(v) for v in hist]
        s.diode_on = list(snap.diode_on)
        s.valid = True
        return _simulate(builder, t_end=s.t + period, dt=dt,
                          switch_fn=sfn, resume_from=s, **kw)

    def sampled_output(res):
        # The output at the PERIOD BOUNDARY — what a controller
        # sampling once per cycle actually measures.
        try:
            return float(np.asarray(res.v(output))[-1])
        except Exception:  # noqa: BLE001 — try it as a device
            return float(np.asarray(res.i(output))[-1])

    sf0 = switch_fn_of(u0)
    base = one_period(h0, sf0)
    y0 = sampled_output(base)
    n_runs += 1

    # ---- C: how the sampled output moves with the state ----
    scale = max(1e-9, float(np.abs(h0).max()) * 1e-4)
    C = np.empty(n, dtype=float)
    for j in range(n):
        hj = h0.copy()
        hj[j] += scale
        C[j] = (sampled_output(one_period(hj, sf0)) - y0) / scale
        n_runs += 1

    # ---- B and D: how state and output move with the input ----
    sf1 = switch_fn_of(u0 + du)
    # Does the perturbation move the GATE at all? On a fixed grid
    # it can only move in whole steps, so a du below dt/period is
    # invisible — and then B is not zero but WORSE: it is the
    # steady state's own residual divided by a tiny du, i.e. noise
    # amplified into a confident-looking response (measured
    # |B| = 0.76 from a 7.6e-10 residual at du = 1e-9). Comparing
    # the two schedules over the period settles it exactly, with
    # no threshold to tune.
    t0_ = float(snap.t)
    moved = False
    for k in range(int(round(period / dt)) + 1):
        tk = t0_ + k * dt
        if not (sf0(tk) == sf1(tk)):
            moved = True
            break
    if not moved:
        raise ValueError(
            f"frequency_response: du={du:g} does not move the gate "
            "anywhere in the period — on a fixed grid the edge "
            "only lands on whole steps, so this perturbation is "
            "invisible and B would be the steady state's residual "
            "divided by du (noise, not a response). Use du >= "
            f"dt/period = {dt / period:g}, or a finer dt.")

    pert = one_period(h0, sf1)
    B = (np.asarray(list(pert.final_snapshot.history), dtype=float)
         - h0) / du
    D = (sampled_output(pert) - y0) / du
    n_runs += 1

    # ---- H(z) = C (zI - Phi)^-1 B + D ----
    eye = np.eye(n)
    H = np.empty(freqs.size, dtype=complex)
    for k, f in enumerate(freqs):
        z = np.exp(2j * np.pi * f * period)
        H[k] = C @ np.linalg.solve(z * eye - Phi, B) + D

    with np.errstate(divide="ignore"):
        mag_db = 20.0 * np.log10(np.abs(H))
    return FrequencyResponse(
        frequencies=freqs, H=H, magnitude_db=mag_db,
        phase_deg=np.degrees(np.angle(H)),
        Phi=Phi, B=B, C=C, D=float(D), steady=steady,
        n_period_runs=n_runs)


__all__ = ["frequency_response", "FrequencyResponse"]
