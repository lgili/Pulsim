"""Pulsim benchmark KPI extraction primitives.

These helpers operate on already-captured simulation traces (time + sample
arrays) and produce the kind of quantitative metrics that engineers compare
across simulators: total harmonic distortion, power factor, efficiency,
transient-response figures, and per-cycle ripple.

The functions are intentionally pure — no Pulsim runtime dependency — so the
same module can score traces captured from anywhere (a `pulsim.csv`,
a captured PSIM/PLECS run for cross-validation, a measured oscilloscope CSV).

See `openspec/changes/add-kpi-measurement-suite/` for the spec scenarios that
drive the API.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import math


# =============================================================================
# Discrete-time Fourier / Goertzel (used by THD and PF)
# =============================================================================


def _goertzel_complex(samples: Sequence[float], freq_hz: float, sample_rate_hz: float) -> Tuple[float, float]:
    """Single-bin Goertzel — returns (real, imag) of the DFT at `freq_hz`.

    Pure-Python; not optimized for huge N but fine for benchmark traces (≤ 10⁶
    samples). The phase convention matches `numpy.fft.fft` (positive ω in
    e^{-jωt}).
    """
    if sample_rate_hz <= 0 or freq_hz <= 0:
        return (0.0, 0.0)
    n = len(samples)
    if n == 0:
        return (0.0, 0.0)
    k = freq_hz / sample_rate_hz  # fractional bin
    omega = 2.0 * math.pi * k
    cos_omega = math.cos(omega)
    coeff = 2.0 * cos_omega
    s_prev = 0.0
    s_prev2 = 0.0
    for x in samples:
        s = x + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    real = s_prev - s_prev2 * cos_omega
    imag = s_prev2 * math.sin(omega)
    return (real, imag)


def _bin_magnitude(samples: Sequence[float], freq_hz: float, sample_rate_hz: float) -> float:
    re, im = _goertzel_complex(samples, freq_hz, sample_rate_hz)
    # Goertzel returns the un-normalized DFT bin; divide by N/2 to get the
    # amplitude of a real-valued sinusoid at this frequency (standard
    # one-sided spectrum convention).
    n = max(1, len(samples))
    return 2.0 * math.hypot(re, im) / n


# =============================================================================
# THD
# =============================================================================


def compute_thd(
    samples: Sequence[float],
    sample_rate_hz: float,
    fundamental_hz: float,
    num_harmonics: int = 20,
) -> float:
    """Total Harmonic Distortion as a percentage.

    `THD% = 100 · √(Σ_{k=2..N} A_k²) / A_1`

    where `A_k` is the amplitude of the k-th harmonic at `k · fundamental_hz`.

    The samples should span an integer number of fundamental periods for a
    clean reading; in practice we just window over a long enough span that
    leakage is below the harmonic floor we care about.
    """
    if fundamental_hz <= 0 or len(samples) < 4:
        return 0.0
    a1 = _bin_magnitude(samples, fundamental_hz, sample_rate_hz)
    if a1 <= 0.0:
        return 0.0
    sum_sq = 0.0
    for k in range(2, num_harmonics + 1):
        a_k = _bin_magnitude(samples, k * fundamental_hz, sample_rate_hz)
        sum_sq += a_k * a_k
    return 100.0 * math.sqrt(sum_sq) / a1


# =============================================================================
# Power factor
# =============================================================================


def compute_power_factor(
    v_samples: Sequence[float],
    i_samples: Sequence[float],
) -> float:
    """Power factor as P_real / |S|, computed over the whole sample window.

    Both sequences must be the same length and sampled on a common time base.
    Returns a value in [-1, 1]; sign follows the direction of real power.
    """
    n = len(v_samples)
    if n == 0 or n != len(i_samples):
        return 0.0
    sum_p = 0.0
    sum_v2 = 0.0
    sum_i2 = 0.0
    for v, i in zip(v_samples, i_samples):
        sum_p += v * i
        sum_v2 += v * v
        sum_i2 += i * i
    real = sum_p / n
    apparent = math.sqrt(sum_v2 / n) * math.sqrt(sum_i2 / n)
    if apparent <= 1e-18:
        return 0.0
    return real / apparent


# =============================================================================
# Efficiency
# =============================================================================


def compute_efficiency(
    p_in_samples: Sequence[float],
    p_out_samples: Sequence[float],
    steady_state_fraction: float = 0.2,
) -> float:
    """Efficiency η = ⟨P_out⟩ / ⟨P_in⟩ over the last `steady_state_fraction`
    of the trace. Result is in percent (so a perfect converter returns 100.0).

    `steady_state_fraction` defaults to the last 20 % of the trace, matching
    the convention used by `compute_thd` and the existing `steady_state_*`
    tolerance bookkeeping in the manifest.
    """
    n = min(len(p_in_samples), len(p_out_samples))
    if n == 0 or not (0.0 < steady_state_fraction <= 1.0):
        return 0.0
    start = max(0, int(n * (1.0 - steady_state_fraction)))
    window_in = p_in_samples[start:n]
    window_out = p_out_samples[start:n]
    p_in_avg = sum(window_in) / max(1, len(window_in))
    p_out_avg = sum(window_out) / max(1, len(window_out))
    if abs(p_in_avg) < 1e-18:
        return 0.0
    return 100.0 * p_out_avg / p_in_avg


# =============================================================================
# Transient response: rise / settling / overshoot
# =============================================================================


def compute_transient_response(
    times: Sequence[float],
    samples: Sequence[float],
    target: float,
    tolerance_pct: float = 2.0,
    rise_low_pct: float = 10.0,
    rise_high_pct: float = 90.0,
) -> Dict[str, Optional[float]]:
    """Returns rise time, settling time, overshoot %, undershoot % of the
    response toward `target`.

    - rise_time: time from `rise_low_pct%` to `rise_high_pct%` of the
      final value (or `None` if the response never reaches `rise_high_pct%`).
    - settling_time: the earliest time after which the trace stays within
      `tolerance_pct%` of `target` for the rest of the simulation.
    - overshoot_pct: 100 · (max − target) / target (only positive lobes;
      0 if the response never exceeds the target).
    - undershoot_pct: 100 · (target − min) / target after the first crossing
      of `target` (0 if the response never undershoots after settling).
    """
    n = len(samples)
    if n == 0 or n != len(times) or target == 0:
        return {
            "rise_time": None,
            "settling_time": None,
            "overshoot_pct": 0.0,
            "undershoot_pct": 0.0,
        }
    target_abs = abs(target)
    tol = (tolerance_pct / 100.0) * target_abs
    rise_low = (rise_low_pct / 100.0) * target
    rise_high = (rise_high_pct / 100.0) * target

    # rise time: first index where sample crosses rise_low_pct, then crosses rise_high_pct
    t_low: Optional[float] = None
    t_high: Optional[float] = None
    sign = 1 if target >= 0 else -1
    for t, s in zip(times, samples):
        if t_low is None and sign * s >= sign * rise_low:
            t_low = t
        if t_high is None and sign * s >= sign * rise_high:
            t_high = t
            break
    rise_time = (t_high - t_low) if (t_low is not None and t_high is not None) else None

    # settling time: last index where the response is OUTSIDE tolerance
    settling_time: Optional[float] = None
    for idx in range(n - 1, -1, -1):
        if abs(samples[idx] - target) > tol:
            settling_time = times[idx] if idx + 1 < n else None
            if idx + 1 < n:
                settling_time = times[idx + 1]
            break
    if settling_time is None:
        settling_time = times[0]  # always within tolerance from the start
    # If the response never settled (still out of band at the last sample),
    # return None for the settling time.
    if abs(samples[-1] - target) > tol:
        settling_time = None

    # overshoot: max excursion above target (in target's direction)
    if sign >= 0:
        peak = max(samples)
        overshoot = 100.0 * max(0.0, (peak - target)) / target_abs
        trough = min(samples)
        undershoot = 100.0 * max(0.0, (target - trough)) / target_abs
    else:
        trough = min(samples)
        overshoot = 100.0 * max(0.0, (target - trough)) / target_abs
        peak = max(samples)
        undershoot = 100.0 * max(0.0, (peak - target)) / target_abs

    return {
        "rise_time": rise_time,
        "settling_time": settling_time,
        "overshoot_pct": overshoot,
        "undershoot_pct": undershoot,
    }


# =============================================================================
# Output ripple (peak-to-peak)
# =============================================================================


def compute_ripple_pkpk(
    samples: Sequence[float],
    steady_state_fraction: float = 0.2,
) -> float:
    """Peak-to-peak ripple over the last `steady_state_fraction` of the
    trace. Returns 0 if the window is empty.
    """
    n = len(samples)
    if n == 0 or not (0.0 < steady_state_fraction <= 1.0):
        return 0.0
    start = max(0, int(n * (1.0 - steady_state_fraction)))
    window = samples[start:n]
    if not window:
        return 0.0
    return max(window) - min(window)


def compute_time_average(times: Sequence[float], samples: Sequence[float]) -> float:
    """Time-weighted mean (trapezoid) of a trace on any grid, uniform or not."""
    if len(times) < 2 or not times[-1] - times[0] > 0.0:
        raise ValueError("a time average needs at least two samples over a positive span")
    area = sum(0.5 * (samples[k] + samples[k + 1]) * (times[k + 1] - times[k])
               for k in range(len(times) - 1))
    return area / (times[-1] - times[0])


def compute_rms(times: Sequence[float], samples: Sequence[float]) -> float:
    """Time-weighted RMS: the square is integrated exactly for a trace that is linear
    between samples (Simpson on each interval), so a coarse grid does not bias it."""
    if len(times) < 2 or not times[-1] - times[0] > 0.0:
        raise ValueError("an RMS needs at least two samples over a positive span")
    area = sum((samples[k] ** 2 + samples[k] * samples[k + 1] + samples[k + 1] ** 2) / 3.0
               * (times[k + 1] - times[k]) for k in range(len(times) - 1))
    return math.sqrt(area / (times[-1] - times[0]))


# =============================================================================
# Conduction + switching loss breakdown
# =============================================================================


def compute_loss_breakdown(
    switch_states: Sequence[bool],
    branch_currents: Sequence[float],
    branch_voltages: Sequence[float],
    r_on: float,
    times: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Approximate conduction + switching loss for a single PWL switch.

    - conduction_w_avg ≈ R_on · ⟨I²⟩_{ON} averaged over the trace
    - switching_w_avg ≈ ½ · |V·I| at each ON↔OFF transition, divided by
      the total time

    `branch_voltages` should be the voltage across the switch (V_DS for a
    MOSFET, V_AK for an IGBT). For the cleanest reading, sample at every
    timestep — this function doesn't interpolate.
    """
    n = min(len(switch_states), len(branch_currents), len(branch_voltages))
    if n < 2:
        return {"conduction_w_avg": 0.0, "switching_w_avg": 0.0}

    if times is None or len(times) < n:
        dt = 1.0  # unit-time; loss numbers are in W per sample
    else:
        dt = max(1e-18, times[n - 1] - times[0])

    # Conduction loss: integrate I²R while ON
    cond_energy = 0.0
    sample_dt = (times[n - 1] - times[0]) / (n - 1) if (times is not None and n > 1) else 1.0
    for k in range(n):
        if switch_states[k]:
            i_k = branch_currents[k]
            cond_energy += r_on * i_k * i_k * sample_dt

    # Switching loss: ½·V·I at each transition (using the sample just
    # before the flip to capture pre-transition state).
    sw_energy = 0.0
    for k in range(1, n):
        if switch_states[k] != switch_states[k - 1]:
            v = branch_voltages[k - 1]
            i = branch_currents[k - 1]
            sw_energy += 0.5 * abs(v * i)

    return {
        "conduction_w_avg": cond_energy / dt,
        "switching_w_avg": sw_energy / dt,
    }


# =============================================================================
# Soft-switching detection: ZVS / ZCS fraction + switching loss
# =============================================================================


def compute_zvs_fraction(
    switch_states: Sequence[bool],
    v_ds_samples: Sequence[float],
    threshold_v: float = 1.0,
    lookback_samples: int = 1,
) -> Dict[str, float]:
    """Fraction of turn-ON events that occur at near-zero V_DS (ZVS).

    Walks the switch_state sequence; at each False→True transition (turn-ON),
    looks at the V_DS sample `lookback_samples` indices earlier. If |V_DS| <
    `threshold_v` at that moment, the event counts as ZVS.

    Returns:
        {"zvs_fraction": <0..1>, "total_turn_on_events": <int>}
    """
    n = min(len(switch_states), len(v_ds_samples))
    if n < 2:
        return {"zvs_fraction": 0.0, "total_turn_on_events": 0.0}

    total = 0
    zvs = 0
    for k in range(1, n):
        if switch_states[k] and not switch_states[k - 1]:  # OFF→ON
            total += 1
            j = max(0, k - lookback_samples)
            if abs(v_ds_samples[j]) < threshold_v:
                zvs += 1
    if total == 0:
        return {"zvs_fraction": 0.0, "total_turn_on_events": 0.0}
    return {
        "zvs_fraction": float(zvs) / float(total),
        "total_turn_on_events": float(total),
    }


def compute_zcs_fraction(
    switch_states: Sequence[bool],
    i_d_samples: Sequence[float],
    threshold_a: float = 0.1,
    lookback_samples: int = 1,
) -> Dict[str, float]:
    """Fraction of turn-OFF events that occur at near-zero I_D (ZCS).

    Walks the switch_state sequence; at each True→False transition
    (turn-OFF), looks at the I_D sample `lookback_samples` indices earlier.
    If |I_D| < `threshold_a` at that moment, the event counts as ZCS.

    Returns:
        {"zcs_fraction": <0..1>, "total_turn_off_events": <int>}
    """
    n = min(len(switch_states), len(i_d_samples))
    if n < 2:
        return {"zcs_fraction": 0.0, "total_turn_off_events": 0.0}

    total = 0
    zcs = 0
    for k in range(1, n):
        if not switch_states[k] and switch_states[k - 1]:  # ON→OFF
            total += 1
            j = max(0, k - lookback_samples)
            if abs(i_d_samples[j]) < threshold_a:
                zcs += 1
    if total == 0:
        return {"zcs_fraction": 0.0, "total_turn_off_events": 0.0}
    return {
        "zcs_fraction": float(zcs) / float(total),
        "total_turn_off_events": float(total),
    }


def compute_switching_loss_per_event(
    switch_states: Sequence[bool],
    v_ds_samples: Sequence[float],
    i_d_samples: Sequence[float],
    times: Sequence[float],
    window_s: float,
) -> Dict[str, float]:
    """Switching energy E = int v_DS * i_D dt over window_s after each
    state transition (trapezoidal over the samples, the window end
    interpolated), summed and divided by the span: the average switching
    loss in W.

    The window starts at the first sample that shows the new state (the
    edge itself when the engine lands its gate edges or they sit on the
    grid) and must stop before the next transition; it includes the
    conduction of that window. (Until 2026-11 it started one sample
    earlier: on a variable grid that sample can be far before the edge,
    and the conduction up to the edge was booked as switching energy --
    on the PSFB reference's S3 under trbdf2, 2.55e-10 J against the exact
    3.18e-11 J; from the edge, 3.66e-11 J.) An edge the solver does not
    resolve (a capacitor collapsing through milliohms inside one step) is
    not in the samples, so it is not in this integral either: book it from
    the stored-energy change instead.

    (Before 2026-11 this returned sum(0.5*|v*i|) / span over the
    pre-transition samples: watts divided by seconds, not an energy.)
    """
    if not window_s > 0:
        raise ValueError(f"window_s must be > 0 s, got {window_s!r}")
    n = min(len(switch_states), len(v_ds_samples), len(i_d_samples), len(times))
    if n < 2:
        return {"switching_loss_w_avg": 0.0, "switching_event_count": 0.0,
                "switching_energy_j": 0.0}
    total_time = max(1e-18, times[n - 1] - times[0])
    p = [v_ds_samples[k] * i_d_samples[k] for k in range(n)]
    energy = 0.0
    events = 0
    for k in range(1, n):
        if switch_states[k] == switch_states[k - 1]:
            continue
        events += 1
        t0 = times[k]               # the first sample that shows the new state: the edge
        t1 = t0 + window_s
        j = k
        while j + 1 < n and times[j + 1] <= t1:
            energy += 0.5 * (p[j] + p[j + 1]) * (times[j + 1] - times[j])
            j += 1
        if j + 1 < n and times[j + 1] > times[j] and t1 > times[j]:
            f = (t1 - times[j]) / (times[j + 1] - times[j])
            p1 = p[j] + f * (p[j + 1] - p[j])
            energy += 0.5 * (p[j] + p1) * (t1 - times[j])
    return {
        "switching_loss_w_avg": energy / total_time,
        "switching_event_count": float(events),
        "switching_energy_j": energy,
    }


# =============================================================================
# Magnetic core loss (Steinmetz)
# =============================================================================


def compute_core_loss_steinmetz(
    flux_density_samples: Sequence[float],
    sample_rate_hz: float,
    fundamental_hz: float,
    k: float,
    alpha: float,
    beta: float,
) -> Dict[str, float]:
    """Average core loss density (W/kg or W/m³ depending on `k` units)
    from a B(t) waveform via Steinmetz: P = k · f^α · B^β.

    `flux_density_samples` is B(t) in Tesla. The peak amplitude at
    `fundamental_hz` is recovered via Goertzel, then plugged into
    Steinmetz with the given material coefficients.
    """
    n = len(flux_density_samples)
    if n < 4 or fundamental_hz <= 0:
        return {"core_loss_w_per_kg": 0.0, "b_peak_tesla": 0.0}
    b_peak = _bin_magnitude(flux_density_samples, fundamental_hz, sample_rate_hz)
    p = k * (fundamental_hz ** alpha) * (b_peak ** beta)
    return {"core_loss_w_per_kg": p, "b_peak_tesla": b_peak}


def compute_inductor_flux_density(
    current_samples: Sequence[float],
    inductance_h: float,
    turns: int,
    area_m2: float,
) -> List[float]:
    """Convert an inductor current trace to a B(t) trace in Tesla.

    Uses B ≈ (L · I) / (N · A) — exact for an ideal inductor, an
    approximation for real cores. Good enough for relative
    Steinmetz comparisons.
    """
    if turns <= 0 or area_m2 <= 0:
        return [0.0] * len(current_samples)
    factor = inductance_h / (float(turns) * area_m2)
    return [factor * i for i in current_samples]


# =============================================================================
# Electrothermal: junction temperature from electrical loss + thermal RC
# =============================================================================


def compute_power_dissipation_resistor(
    v_a_samples: Sequence[float],
    v_b_samples: Sequence[float],
    resistance: float,
) -> List[float]:
    """Instantaneous power dissipation P(t) = (V_a − V_b)² / R."""
    if resistance <= 0:
        return [0.0] * len(v_a_samples)
    n = min(len(v_a_samples), len(v_b_samples))
    return [((v_a_samples[k] - v_b_samples[k]) ** 2) / resistance for k in range(n)]


def compute_junction_temperature(
    power_dissipation_samples: Sequence[float],
    times: Sequence[float],
    r_th_jc: float,
    c_th_jc: float,
    t_ambient_c: float = 25.0,
    r_th_ca: float = 0.0,
) -> Dict[str, float]:
    """Solve a first-order thermal RC network forward in time to predict
    junction temperature T_j(t) from a P_diss(t) trace.

    Model: C_th · dT_j/dt + (T_j − T_a)/R_th = P(t)
    Steady-state: T_j − T_a = R_th · P_avg.

    P(t) is the linear interpolation of the samples and each interval is
    solved exactly (θ = T_j − T_a, τ = R·C, e = exp(−h/τ)):
    θ(h) = θ0·e + R·[P0·(1 − e) + s·(h − τ·(1 − e))], s the interval's slope.
    Exact for any sample spacing: the explicit Euler step it replaces read a
    tolerance-chosen grid (two samples over a DC run, F-X-4) as its time
    step, and was unstable past h = 2τ.

    Returns:
        {"t_j_max_c", "t_j_final_c", "delta_t_j_c"}.
    """
    n = min(len(power_dissipation_samples), len(times))
    if n < 2:
        return {"t_j_max_c": t_ambient_c, "t_j_final_c": t_ambient_c, "delta_t_j_c": 0.0}

    R_total = r_th_jc + r_th_ca
    if R_total <= 0:
        return {"t_j_max_c": t_ambient_c, "t_j_final_c": t_ambient_c, "delta_t_j_c": 0.0}

    t_j = t_ambient_c
    t_max = t_j
    use_c = c_th_jc if c_th_jc > 0 else 1.0
    tau = R_total * use_c
    for k in range(1, n):
        h = times[k] - times[k - 1]
        p0 = power_dissipation_samples[k - 1]
        s = (power_dissipation_samples[k] - p0) / h if h > 0 else 0.0
        one_m_e = -math.expm1(-h / tau)
        theta = (t_j - t_ambient_c) * (1.0 - one_m_e) + R_total * (p0 * one_m_e + s * (h - tau * one_m_e))
        t_j = t_ambient_c + theta
        if t_j > t_max:
            t_max = t_j

    return {
        "t_j_max_c": t_max,
        "t_j_final_c": t_j,
        "delta_t_j_c": t_j - t_ambient_c,
    }


def compute_drift(
    times: Sequence[float],
    samples: Sequence[float],
    t_from: Optional[float] = None,
    t_to: Optional[float] = None,
) -> float:
    """Least-squares slope (units per second) of ``samples`` over
    ``[t_from, t_to]`` (add-long-simulation-stress). A settled waveform's
    slope is its secular drift; feed it per-period averages, not raw
    switching samples, or the ripple's phase leaks into the slope."""
    pts = [(t, y) for t, y in zip(times, samples)
           if (t_from is None or t >= t_from) and (t_to is None or t <= t_to)]
    if len(pts) < 2:
        raise ValueError("compute_drift needs at least two samples in the window")
    n = len(pts)
    tm = sum(t for t, _ in pts) / n
    ym = sum(y for _, y in pts) / n
    stt = sum((t - tm) ** 2 for t, _ in pts)
    if stt == 0.0:
        raise ValueError("compute_drift: all samples share one instant")
    return sum((t - tm) * (y - ym) for t, y in pts) / stt


def compute_conservation_drift(
    p_in_samples: Sequence[float],
    p_out_samples: Sequence[float],
    e_stored_initial: float,
    e_stored_final: float,
    time: Sequence[float],
) -> float:
    """Relative energy residual ``(∫(p_in − p_out) dt − ΔE_stored) / ∫p_in dt``
    (add-long-simulation-stress). ``p_out`` is everything that leaves the
    stored energy (load plus losses); the integrals are trapezoidal over the
    sample instants ``time``, exact for the piecewise-linear waveform through
    the samples. Signed: positive means energy went in and is unaccounted."""
    n = len(time)
    if n < 2 or len(p_in_samples) != n or len(p_out_samples) != n:
        raise ValueError("compute_conservation_drift: p_in, p_out and time need one sample per instant (>= 2)")
    w_in = w_out = 0.0
    for k in range(1, n):
        h = time[k] - time[k - 1]
        w_in += 0.5 * h * (p_in_samples[k] + p_in_samples[k - 1])
        w_out += 0.5 * h * (p_out_samples[k] + p_out_samples[k - 1])
    if w_in == 0.0:
        raise ValueError("compute_conservation_drift: no input energy")
    return (w_in - w_out - (e_stored_final - e_stored_initial)) / w_in


__all__ = [
    "compute_thd",
    "compute_power_factor",
    "compute_efficiency",
    "compute_transient_response",
    "compute_ripple_pkpk",
    "compute_time_average",
    "compute_rms",
    "compute_loss_breakdown",
    "compute_zvs_fraction",
    "compute_zcs_fraction",
    "compute_switching_loss_per_event",
    "compute_core_loss_steinmetz",
    "compute_inductor_flux_density",
    "compute_power_dissipation_resistor",
    "compute_junction_temperature",
    "compute_drift",
    "compute_conservation_drift",
]
