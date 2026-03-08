"""Waveform post-processing pipeline for pulsim.

Provides deterministic, backend-owned metric computation for transient simulation
results. Supports time-domain metrics, spectral (FFT/harmonic/THD) analysis, and
power/efficiency measurements.

Public API:
    PostProcessingWindowMode    -- Window specification mode enum
    PostProcessingJobKind       -- Job type enum
    PostProcessingDiagnosticCode -- Diagnostic reason codes
    WindowFunction              -- Spectral window function enum
    PostProcessingWindowSpec    -- Window configuration dataclass
    PostProcessingJob           -- Single job configuration dataclass
    PostProcessingOptions       -- Collection of jobs to execute
    ScalarMetric                -- Named scalar metric result
    SpectralBin                 -- FFT frequency bin result
    HarmonicEntry               -- Harmonic component result
    UndefinedMetricEntry        -- Undefined metric with reason
    PostProcessingJobResult     -- Per-job result dataclass
    PostProcessingResult        -- Aggregated result dataclass
    run_post_processing         -- Execute jobs on a SimulationResult
    parse_post_processing_yaml  -- Parse simulation.post_processing YAML node
"""

from __future__ import annotations

import math
import time as _time_module
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PostProcessingWindowMode(str, Enum):
    """Window specification mode for post-processing jobs."""
    Time = "time"    # bounds given as [t_start, t_end] in seconds
    Index = "index"  # bounds given as [i_start, i_end] sample indices
    Cycle = "cycle"  # bounds given as [cycle_start, cycle_end] with period


class PostProcessingJobKind(str, Enum):
    """Type of post-processing job."""
    TimeDomain = "time_domain"
    Spectral = "spectral"
    PowerEfficiency = "power_efficiency"


class PostProcessingDiagnosticCode(str, Enum):
    """Stable machine-readable reason codes for post-processing diagnostics."""
    Ok = "ok"
    InvalidConfiguration = "invalid_configuration"
    SignalNotFound = "signal_not_found"
    InvalidWindow = "invalid_window"
    InsufficientSamples = "insufficient_samples"
    SamplingMismatch = "sampling_mismatch"
    UndefinedMetric = "undefined_metric"
    NumericalFailure = "numerical_failure"


class WindowFunction(str, Enum):
    """Spectral window functions for FFT analysis."""
    Rectangular = "rectangular"
    Hann = "hann"
    Hamming = "hamming"
    Blackman = "blackman"
    FlatTop = "flat_top"


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PostProcessingWindowSpec:
    """Window specification for a post-processing job.

    The active fields depend on *mode*:
      - Time:  t_start / t_end  (seconds)
      - Index: i_start / i_end  (sample indices, -1 means last)
      - Cycle: cycle_start / cycle_end / period  (cycle count + period seconds)
    """
    mode: PostProcessingWindowMode = PostProcessingWindowMode.Time
    t_start: float = 0.0
    t_end: float = float("inf")
    i_start: int = 0
    i_end: int = -1            # -1 → last sample
    cycle_start: int = 0
    cycle_end: int = -1        # -1 → cycle_start + 1
    period: float = 0.0        # required for Cycle mode
    min_samples: int = 4       # minimum samples required in window


@dataclass
class PostProcessingJob:
    """Configuration for a single post-processing job.

    Fields used depend on *kind*:
      - TimeDomain:      signals, window, metrics
      - Spectral:        signals[0], window, n_harmonics, fundamental_hz, window_function
      - PowerEfficiency: input/output voltage/current signals, window
    """
    job_id: str
    kind: PostProcessingJobKind
    signals: List[str] = field(default_factory=list)
    window: PostProcessingWindowSpec = field(default_factory=PostProcessingWindowSpec)

    # TimeDomain options
    metrics: List[str] = field(
        default_factory=lambda: ["rms", "mean", "min", "max", "p2p", "crest", "ripple"]
    )

    # Spectral options
    n_harmonics: int = 10
    fundamental_hz: float = 0.0   # 0 = auto-detect peak
    window_function: WindowFunction = WindowFunction.Rectangular

    # PowerEfficiency signal bindings
    input_voltage_signal: str = ""
    input_current_signal: str = ""
    output_voltage_signal: str = ""
    output_current_signal: str = ""


@dataclass
class PostProcessingOptions:
    """Collection of post-processing jobs to execute after simulation."""
    jobs: List[PostProcessingJob] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScalarMetric:
    """A single named scalar metric value."""
    name: str
    value: float
    unit: str = ""
    domain: str = "time"
    source_signal: str = ""


@dataclass
class SpectralBin:
    """One FFT frequency bin."""
    frequency_hz: float
    magnitude: float
    phase_deg: float


@dataclass
class HarmonicEntry:
    """One harmonic component."""
    harmonic_number: int
    frequency_hz: float
    magnitude: float
    phase_deg: float
    magnitude_pct_fundamental: float


@dataclass
class UndefinedMetricEntry:
    """A metric that could not be computed, with stable reason code."""
    name: str
    reason: PostProcessingDiagnosticCode
    reason_message: str


@dataclass
class PostProcessingJobResult:
    """Result from a single post-processing job."""
    job_id: str
    kind: PostProcessingJobKind
    success: bool
    diagnostic: PostProcessingDiagnosticCode
    diagnostic_message: str = ""

    # TimeDomain results
    scalar_metrics: Dict[str, ScalarMetric] = field(default_factory=dict)

    # Spectral results
    spectrum_bins: List[SpectralBin] = field(default_factory=list)
    harmonics: List[HarmonicEntry] = field(default_factory=list)
    thd_pct: float = float("nan")
    fundamental_hz: float = float("nan")

    # PowerEfficiency results
    average_input_power: float = float("nan")
    average_output_power: float = float("nan")
    efficiency: float = float("nan")
    power_factor: float = float("nan")

    # Undefined metrics with reason codes
    undefined_metrics: List[UndefinedMetricEntry] = field(default_factory=list)

    # Metadata
    signal_names: List[str] = field(default_factory=list)
    window_i_start: int = 0
    window_i_end: int = 0
    sample_count: int = 0
    runtime_seconds: float = 0.0


@dataclass
class PostProcessingResult:
    """Aggregated result from all post-processing jobs."""
    success: bool
    jobs: List[PostProcessingJobResult] = field(default_factory=list)
    total_runtime_seconds: float = 0.0
    run_count: int = 1


# ---------------------------------------------------------------------------
# Internal helpers: window functions
# ---------------------------------------------------------------------------

def _make_window(kind: WindowFunction, n: int) -> np.ndarray:
    """Return a deterministic spectral window array of length *n*."""
    if n <= 0:
        return np.ones(0, dtype=np.float64)
    if kind == WindowFunction.Rectangular:
        return np.ones(n, dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)
    m = max(n - 1, 1)
    if kind == WindowFunction.Hann:
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * idx / m))
    if kind == WindowFunction.Hamming:
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * idx / m)
    if kind == WindowFunction.Blackman:
        return (
            0.42
            - 0.5 * np.cos(2.0 * np.pi * idx / m)
            + 0.08 * np.cos(4.0 * np.pi * idx / m)
        )
    if kind == WindowFunction.FlatTop:
        a0, a1, a2, a3, a4 = (
            0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368
        )
        x = 2.0 * np.pi * idx / m
        return (
            a0
            - a1 * np.cos(x)
            + a2 * np.cos(2 * x)
            - a3 * np.cos(3 * x)
            + a4 * np.cos(4 * x)
        )
    return np.ones(n, dtype=np.float64)


def _window_coherent_gain(w: np.ndarray) -> float:
    """Return the coherent gain (mean) of a window array."""
    g = float(np.mean(w))
    return g if g != 0.0 else 1.0


# ---------------------------------------------------------------------------
# Internal helpers: window planning
# ---------------------------------------------------------------------------

def _resolve_window_indices(
    time_arr: np.ndarray,
    spec: PostProcessingWindowSpec,
) -> tuple[int, int, PostProcessingDiagnosticCode, str]:
    """Convert a PostProcessingWindowSpec to (i_start, i_end) sample bounds.

    Returns (i_start, i_end, code, message). i_end is exclusive.
    On error, returns (0, 0, error_code, description).
    """
    n = len(time_arr)
    if n == 0:
        return 0, 0, PostProcessingDiagnosticCode.InvalidWindow, "Empty time array"

    if spec.mode == PostProcessingWindowMode.Index:
        i0 = spec.i_start
        i1 = spec.i_end
        if i0 < 0:
            i0 = n + i0
        if i1 < 0:
            i1 = n + i1 + 1
        i0 = max(0, min(i0, n))
        i1 = max(i0, min(i1, n))
        if i1 <= i0:
            return 0, 0, PostProcessingDiagnosticCode.InvalidWindow, (
                f"Index window [{spec.i_start}, {spec.i_end}] is empty "
                f"after clamping to [{i0}, {i1}]"
            )
        return i0, i1, PostProcessingDiagnosticCode.Ok, ""

    if spec.mode == PostProcessingWindowMode.Time:
        t0 = spec.t_start
        t1 = spec.t_end if not math.isinf(spec.t_end) else float(time_arr[-1])
        if t1 < t0:
            return 0, 0, PostProcessingDiagnosticCode.InvalidWindow, (
                f"Time window t_start={t0} > t_end={t1}"
            )
        i0 = int(np.searchsorted(time_arr, t0, side="left"))
        i1 = int(np.searchsorted(time_arr, t1, side="right"))
        i0 = max(0, min(i0, n))
        i1 = max(i0, min(i1, n))
        if i1 <= i0:
            return 0, 0, PostProcessingDiagnosticCode.InvalidWindow, (
                f"Time window [{t0}, {t1}] contains no simulation samples"
            )
        return i0, i1, PostProcessingDiagnosticCode.Ok, ""

    if spec.mode == PostProcessingWindowMode.Cycle:
        if spec.period <= 0.0:
            return 0, 0, PostProcessingDiagnosticCode.InvalidWindow, (
                "Cycle window requires period > 0"
            )
        c_end = spec.cycle_end if spec.cycle_end >= 0 else spec.cycle_start
        t0 = spec.cycle_start * spec.period
        t1 = (c_end + 1) * spec.period
        t1 = min(t1, float(time_arr[-1]))
        if t1 <= t0:
            return 0, 0, PostProcessingDiagnosticCode.InvalidWindow, (
                f"Cycle window [{spec.cycle_start}, {c_end}] with "
                f"period={spec.period} yields empty time range"
            )
        i0 = int(np.searchsorted(time_arr, t0, side="left"))
        i1 = int(np.searchsorted(time_arr, t1, side="right"))
        i0 = max(0, min(i0, n))
        i1 = max(i0, min(i1, n))
        if i1 <= i0:
            return 0, 0, PostProcessingDiagnosticCode.InvalidWindow, (
                f"Cycle window [{spec.cycle_start}, {c_end}] contains no samples"
            )
        return i0, i1, PostProcessingDiagnosticCode.Ok, ""

    return 0, 0, PostProcessingDiagnosticCode.InvalidConfiguration, (
        f"Unknown window mode: {spec.mode}"
    )


# ---------------------------------------------------------------------------
# Internal helpers: metric engines
# ---------------------------------------------------------------------------

def _compute_time_domain_metrics(
    x: np.ndarray,
    requested: List[str],
    signal_name: str,
    unit: str = "",
) -> tuple[Dict[str, ScalarMetric], List[UndefinedMetricEntry]]:
    """Compute requested time-domain scalar metrics.

    Supported metric names (case-insensitive):
      rms, mean, min, max, p2p, crest, ripple, std
    """
    metrics: Dict[str, ScalarMetric] = {}
    undefined: List[UndefinedMetricEntry] = []

    if len(x) == 0:
        for name in requested:
            undefined.append(UndefinedMetricEntry(
                name=name,
                reason=PostProcessingDiagnosticCode.InsufficientSamples,
                reason_message="Empty signal window",
            ))
        return metrics, undefined

    mean_val = float(np.mean(x))
    rms_val = float(np.sqrt(np.mean(x ** 2)))
    min_val = float(np.min(x))
    max_val = float(np.max(x))
    p2p_val = max_val - min_val
    abs_max = max(abs(min_val), abs(max_val))

    for raw_name in requested:
        k = raw_name.lower().strip()
        if k == "rms":
            metrics["rms"] = ScalarMetric("rms", rms_val, unit, "time", signal_name)
        elif k == "mean":
            metrics["mean"] = ScalarMetric("mean", mean_val, unit, "time", signal_name)
        elif k in ("min", "minimum"):
            metrics["min"] = ScalarMetric("min", min_val, unit, "time", signal_name)
        elif k in ("max", "maximum"):
            metrics["max"] = ScalarMetric("max", max_val, unit, "time", signal_name)
        elif k in ("p2p", "peak_to_peak", "peak-to-peak", "pkpk"):
            metrics["p2p"] = ScalarMetric("p2p", p2p_val, unit, "time", signal_name)
        elif k in ("crest", "crest_factor"):
            if rms_val == 0.0:
                undefined.append(UndefinedMetricEntry(
                    "crest",
                    PostProcessingDiagnosticCode.UndefinedMetric,
                    "Crest factor undefined when RMS is zero",
                ))
            else:
                metrics["crest"] = ScalarMetric(
                    "crest", abs_max / rms_val, "", "time", signal_name
                )
        elif k == "ripple":
            # Ripple = peak-to-peak / |mean| (power-electronics convention).
            # Treat ripple as undefined when mean is negligible relative to RMS.
            # Threshold: |mean| < 0.1% of RMS (or RMS==0 edge case).
            rms_ref = max(rms_val, 1e-100)
            if abs(mean_val) / rms_ref < 1e-3:
                undefined.append(UndefinedMetricEntry(
                    "ripple",
                    PostProcessingDiagnosticCode.UndefinedMetric,
                    "Ripple undefined when mean is zero",
                ))
            else:
                metrics["ripple"] = ScalarMetric(
                    "ripple", p2p_val / abs(mean_val), "", "time", signal_name
                )
        elif k in ("std", "stddev", "standard_deviation"):
            metrics["std"] = ScalarMetric(
                "std", float(np.std(x)), unit, "time", signal_name
            )
        else:
            undefined.append(UndefinedMetricEntry(
                raw_name,
                PostProcessingDiagnosticCode.InvalidConfiguration,
                f"Unknown time-domain metric: '{raw_name}' "
                f"(accepted: rms, mean, min, max, p2p, crest, ripple, std)",
            ))

    return metrics, undefined


def _compute_spectral_metrics(
    x: np.ndarray,
    time_arr: np.ndarray,
    fundamental_hz: float,
    n_harmonics: int,
    window_kind: WindowFunction,
) -> tuple[
    List[SpectralBin],
    List[HarmonicEntry],
    float,
    float,
    List[UndefinedMetricEntry],
]:
    """Compute FFT spectrum, harmonics table, and THD.

    Returns (bins, harmonics, thd_pct, detected_fundamental_hz, undefined_metrics).
    """
    undefined: List[UndefinedMetricEntry] = []
    n = len(x)

    if n < 4:
        undefined.append(UndefinedMetricEntry(
            "spectrum",
            PostProcessingDiagnosticCode.InsufficientSamples,
            f"Spectral analysis requires >= 4 samples, got {n}",
        ))
        return [], [], float("nan"), float("nan"), undefined

    dt_arr = np.diff(time_arr)
    if len(dt_arr) == 0:
        undefined.append(UndefinedMetricEntry(
            "spectrum",
            PostProcessingDiagnosticCode.InsufficientSamples,
            "Need at least 2 time samples to determine sample rate",
        ))
        return [], [], float("nan"), float("nan"), undefined

    dt_mean = float(np.mean(dt_arr))
    if dt_mean <= 0.0:
        undefined.append(UndefinedMetricEntry(
            "spectrum",
            PostProcessingDiagnosticCode.SamplingMismatch,
            "Non-positive mean time step in window",
        ))
        return [], [], float("nan"), float("nan"), undefined

    # Apply spectral window
    w = _make_window(window_kind, n)
    cg = _window_coherent_gain(w)
    x_windowed = x * w

    # Compute FFT and one-sided amplitude spectrum
    X = np.fft.rfft(x_windowed)
    freqs = np.fft.rfftfreq(n, d=dt_mean)
    mag = np.abs(X) / (cg * n)

    # One-sided correction: double all bins except DC (index 0) and Nyquist
    mag_os = mag.copy()
    nyquist_idx = len(mag_os) - 1
    mag_os[1:nyquist_idx] *= 2.0

    phase_deg = np.degrees(np.angle(X))

    # Build spectrum bin list (exclude DC)
    bins: List[SpectralBin] = [
        SpectralBin(
            frequency_hz=float(freqs[i]),
            magnitude=float(mag_os[i]),
            phase_deg=float(phase_deg[i]),
        )
        for i in range(1, len(freqs))
    ]

    if not bins:
        undefined.append(UndefinedMetricEntry(
            "harmonics",
            PostProcessingDiagnosticCode.InsufficientSamples,
            "No spectral bins available",
        ))
        return bins, [], float("nan"), float("nan"), undefined

    freqs_arr = np.array([b.frequency_hz for b in bins], dtype=np.float64)
    mags_arr = np.array([b.magnitude for b in bins], dtype=np.float64)

    # Detect or locate fundamental
    if fundamental_hz <= 0.0:
        peak_idx = int(np.argmax(mags_arr))
        detected_f = float(freqs_arr[peak_idx])
    else:
        fundamental_bin_idx = int(np.argmin(np.abs(freqs_arr - fundamental_hz)))
        detected_f = float(freqs_arr[fundamental_bin_idx])

    f0_bin_idx = int(np.argmin(np.abs(freqs_arr - detected_f)))
    h1_mag = float(mags_arr[f0_bin_idx])

    if h1_mag == 0.0:
        undefined.append(UndefinedMetricEntry(
            "thd",
            PostProcessingDiagnosticCode.UndefinedMetric,
            "THD undefined: fundamental component magnitude is zero",
        ))
        return bins, [], float("nan"), detected_f, undefined

    # Build harmonic table
    harmonics: List[HarmonicEntry] = []
    harmonic_sum_sq = 0.0
    for h_num in range(1, n_harmonics + 1):
        target_f = detected_f * h_num
        h_idx = int(np.argmin(np.abs(freqs_arr - target_f)))
        h_mag = float(mags_arr[h_idx])
        h_phase = float(bins[h_idx].phase_deg)
        pct = h_mag / h1_mag * 100.0
        harmonics.append(HarmonicEntry(
            harmonic_number=h_num,
            frequency_hz=float(freqs_arr[h_idx]),
            magnitude=h_mag,
            phase_deg=h_phase,
            magnitude_pct_fundamental=pct,
        ))
        if h_num > 1:
            harmonic_sum_sq += h_mag ** 2

    thd_pct = math.sqrt(harmonic_sum_sq) / h1_mag * 100.0

    return bins, harmonics, thd_pct, detected_f, undefined


def _compute_power_efficiency_metrics(
    v_in: np.ndarray,
    i_in: np.ndarray,
    v_out: np.ndarray,
    i_out: np.ndarray,
) -> tuple[float, float, float, float, List[UndefinedMetricEntry]]:
    """Compute average power, efficiency, and power factor.

    Returns (p_in_W, p_out_W, efficiency_pct, power_factor, undefined_metrics).
    """
    undefined: List[UndefinedMetricEntry] = []

    def _avg_power(
        v: np.ndarray, i: np.ndarray, label: str
    ) -> tuple[float, Optional[str]]:
        if len(v) == 0 or len(i) == 0:
            return float("nan"), f"{label}: empty array"
        if len(v) != len(i):
            return float("nan"), (
                f"{label}: voltage/current length mismatch "
                f"({len(v)} vs {len(i)})"
            )
        return float(np.mean(v * i)), None

    p_in = float("nan")
    p_out = float("nan")
    efficiency = float("nan")
    pf = float("nan")

    if len(v_in) > 0 and len(i_in) > 0:
        p_in, err = _avg_power(v_in, i_in, "input")
        if err:
            undefined.append(UndefinedMetricEntry(
                "input_power", PostProcessingDiagnosticCode.InvalidConfiguration, err
            ))
            p_in = float("nan")

    if len(v_out) > 0 and len(i_out) > 0:
        p_out, err = _avg_power(v_out, i_out, "output")
        if err:
            undefined.append(UndefinedMetricEntry(
                "output_power", PostProcessingDiagnosticCode.InvalidConfiguration, err
            ))
            p_out = float("nan")

    if not math.isnan(p_in) and not math.isnan(p_out):
        if p_in > 0.0:
            efficiency = p_out / p_in * 100.0
        elif p_in == 0.0 and p_out == 0.0:
            efficiency = 100.0
        else:
            undefined.append(UndefinedMetricEntry(
                "efficiency",
                PostProcessingDiagnosticCode.UndefinedMetric,
                f"Efficiency undefined for P_in={p_in:.6g} W",
            ))

    # Power factor from input: P / S = P / (V_rms * I_rms)
    if len(v_in) > 0 and len(i_in) > 0 and not math.isnan(p_in):
        v_rms = float(np.sqrt(np.mean(v_in ** 2)))
        i_rms = float(np.sqrt(np.mean(i_in ** 2)))
        s = v_rms * i_rms
        if s > 0.0:
            pf = p_in / s
        else:
            undefined.append(UndefinedMetricEntry(
                "power_factor",
                PostProcessingDiagnosticCode.UndefinedMetric,
                "Power factor undefined: apparent power S = V_rms * I_rms is zero",
            ))

    return p_in, p_out, efficiency, pf, undefined


def _get_signal_data(
    channels: Dict[str, Any],
    signal_name: str,
) -> tuple[Optional[np.ndarray], PostProcessingDiagnosticCode, str]:
    """Resolve a named signal from the virtual_channels dict.

    Performs case-insensitive fallback lookup if exact name not found.
    Returns (array, code, message). On failure, array is None.
    """
    if not signal_name:
        return None, PostProcessingDiagnosticCode.InvalidConfiguration, (
            "Empty signal name"
        )
    if signal_name in channels:
        data = channels[signal_name]
        return np.asarray(data, dtype=np.float64), PostProcessingDiagnosticCode.Ok, ""
    # Case-insensitive fallback
    lower = signal_name.lower()
    for k, v in channels.items():
        if k.lower() == lower:
            return np.asarray(v, dtype=np.float64), PostProcessingDiagnosticCode.Ok, ""
    return None, PostProcessingDiagnosticCode.SignalNotFound, (
        f"Signal '{signal_name}' not found in result channels "
        f"(available: {list(channels.keys())})"
    )


# ---------------------------------------------------------------------------
# Single-job executor
# ---------------------------------------------------------------------------

def _run_single_job(
    job: PostProcessingJob,
    time_arr: np.ndarray,
    channels: Dict[str, Any],
) -> PostProcessingJobResult:
    """Execute one post-processing job and return its result."""
    t0 = _time_module.monotonic()

    def _fail(
        code: PostProcessingDiagnosticCode, msg: str
    ) -> PostProcessingJobResult:
        return PostProcessingJobResult(
            job_id=job.job_id,
            kind=job.kind,
            success=False,
            diagnostic=code,
            diagnostic_message=msg,
            runtime_seconds=_time_module.monotonic() - t0,
        )

    # --- Signal resolution (for jobs that need signal data) ---
    signal_data: Dict[str, np.ndarray] = {}
    if job.kind in (PostProcessingJobKind.TimeDomain, PostProcessingJobKind.Spectral):
        if not job.signals:
            return _fail(
                PostProcessingDiagnosticCode.InvalidConfiguration,
                f"Job '{job.job_id}': no signals specified",
            )
        for sig_name in job.signals:
            arr, code, msg = _get_signal_data(channels, sig_name)
            if arr is None:
                return _fail(code, f"Job '{job.job_id}': {msg}")
            signal_data[sig_name] = arr

    # --- Window planning ---
    i_start, i_end, code, msg = _resolve_window_indices(time_arr, job.window)
    if code != PostProcessingDiagnosticCode.Ok:
        return _fail(code, f"Job '{job.job_id}': {msg}")

    sample_count = i_end - i_start
    if sample_count < job.window.min_samples:
        return _fail(
            PostProcessingDiagnosticCode.InsufficientSamples,
            f"Job '{job.job_id}': window has {sample_count} samples, "
            f"minimum is {job.window.min_samples}",
        )

    time_window = time_arr[i_start:i_end]

    # --- Time-domain job ---
    if job.kind == PostProcessingJobKind.TimeDomain:
        all_metrics: Dict[str, ScalarMetric] = {}
        all_undefined: List[UndefinedMetricEntry] = []
        multi = len(signal_data) > 1

        for sig_name, sig_arr in signal_data.items():
            if len(sig_arr) != len(time_arr):
                all_undefined.append(UndefinedMetricEntry(
                    sig_name,
                    PostProcessingDiagnosticCode.SamplingMismatch,
                    f"Signal '{sig_name}' length {len(sig_arr)} "
                    f"!= time array length {len(time_arr)}",
                ))
                continue
            x = sig_arr[i_start:i_end]
            m, u = _compute_time_domain_metrics(x, job.metrics, sig_name)
            if multi:
                for k, v in m.items():
                    all_metrics[f"{sig_name}.{k}"] = v
            else:
                all_metrics.update(m)
            all_undefined.extend(u)

        return PostProcessingJobResult(
            job_id=job.job_id,
            kind=job.kind,
            success=True,
            diagnostic=PostProcessingDiagnosticCode.Ok,
            scalar_metrics=all_metrics,
            undefined_metrics=all_undefined,
            signal_names=list(signal_data.keys()),
            window_i_start=i_start,
            window_i_end=i_end,
            sample_count=sample_count,
            runtime_seconds=_time_module.monotonic() - t0,
        )

    # --- Spectral job ---
    if job.kind == PostProcessingJobKind.Spectral:
        sig_name = job.signals[0]
        sig_arr = signal_data[sig_name]

        if len(sig_arr) != len(time_arr):
            return _fail(
                PostProcessingDiagnosticCode.SamplingMismatch,
                f"Job '{job.job_id}': signal '{sig_name}' length {len(sig_arr)} "
                f"!= time array length {len(time_arr)}",
            )

        x = sig_arr[i_start:i_end]
        bins, harmonics, thd, f0, undef = _compute_spectral_metrics(
            x, time_window, job.fundamental_hz, job.n_harmonics, job.window_function
        )

        return PostProcessingJobResult(
            job_id=job.job_id,
            kind=job.kind,
            success=True,
            diagnostic=PostProcessingDiagnosticCode.Ok,
            spectrum_bins=bins,
            harmonics=harmonics,
            thd_pct=thd,
            fundamental_hz=f0,
            undefined_metrics=undef,
            signal_names=[sig_name],
            window_i_start=i_start,
            window_i_end=i_end,
            sample_count=sample_count,
            runtime_seconds=_time_module.monotonic() - t0,
        )

    # --- Power/efficiency job ---
    if job.kind == PostProcessingJobKind.PowerEfficiency:

        def _get_windowed(name: str) -> np.ndarray:
            if not name:
                return np.array([], dtype=np.float64)
            arr, c, m = _get_signal_data(channels, name)
            if arr is None:
                return np.array([], dtype=np.float64)
            if len(arr) != len(time_arr):
                return np.array([], dtype=np.float64)
            return arr[i_start:i_end]

        v_in = _get_windowed(job.input_voltage_signal)
        i_in = _get_windowed(job.input_current_signal)
        v_out = _get_windowed(job.output_voltage_signal)
        i_out = _get_windowed(job.output_current_signal)

        if len(v_in) > 0 and len(i_in) > 0 and len(v_in) != len(i_in):
            return _fail(
                PostProcessingDiagnosticCode.SamplingMismatch,
                f"Job '{job.job_id}': input voltage/current arrays "
                f"have different lengths ({len(v_in)} vs {len(i_in)})",
            )
        if len(v_out) > 0 and len(i_out) > 0 and len(v_out) != len(i_out):
            return _fail(
                PostProcessingDiagnosticCode.SamplingMismatch,
                f"Job '{job.job_id}': output voltage/current arrays "
                f"have different lengths ({len(v_out)} vs {len(i_out)})",
            )

        p_in, p_out, eff, pf, undef = _compute_power_efficiency_metrics(
            v_in, i_in, v_out, i_out
        )

        sig_names = [
            s
            for s in [
                job.input_voltage_signal,
                job.input_current_signal,
                job.output_voltage_signal,
                job.output_current_signal,
            ]
            if s
        ]

        return PostProcessingJobResult(
            job_id=job.job_id,
            kind=job.kind,
            success=True,
            diagnostic=PostProcessingDiagnosticCode.Ok,
            average_input_power=p_in,
            average_output_power=p_out,
            efficiency=eff,
            power_factor=pf,
            undefined_metrics=undef,
            signal_names=sig_names,
            window_i_start=i_start,
            window_i_end=i_end,
            sample_count=sample_count,
            runtime_seconds=_time_module.monotonic() - t0,
        )

    return _fail(
        PostProcessingDiagnosticCode.InvalidConfiguration,
        f"Unknown job kind: {job.kind}",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_post_processing(
    result: Any,
    config: PostProcessingOptions,
) -> PostProcessingResult:
    """Execute waveform post-processing jobs on a simulation result.

    Args:
        result: A SimulationResult object with ``.time`` and
                ``.virtual_channels`` attributes (as returned by
                :class:`~pulsim.Simulator`).
        config: :class:`PostProcessingOptions` specifying which jobs to run.

    Returns:
        :class:`PostProcessingResult` containing per-job outputs.

    The function is deterministic: calling it twice with identical inputs
    produces identical outputs (modulo ``runtime_seconds`` telemetry).
    """
    wall_start = _time_module.monotonic()

    time_arr = np.asarray(result.time, dtype=np.float64)
    channels: Dict[str, Any] = dict(result.virtual_channels)

    job_results: List[PostProcessingJobResult] = []
    overall_success = True

    for job in config.jobs:
        jr = _run_single_job(job, time_arr, channels)
        job_results.append(jr)
        if not jr.success:
            overall_success = False

    return PostProcessingResult(
        success=overall_success,
        jobs=job_results,
        total_runtime_seconds=_time_module.monotonic() - wall_start,
        run_count=1,
    )


# ---------------------------------------------------------------------------
# YAML parsing helpers (for simulation.post_processing block)
# ---------------------------------------------------------------------------

_WINDOW_MODE_MAP: Dict[str, PostProcessingWindowMode] = {
    "time": PostProcessingWindowMode.Time,
    "index": PostProcessingWindowMode.Index,
    "cycle": PostProcessingWindowMode.Cycle,
}

_JOB_KIND_MAP: Dict[str, PostProcessingJobKind] = {
    "time_domain": PostProcessingJobKind.TimeDomain,
    "spectral": PostProcessingJobKind.Spectral,
    "power_efficiency": PostProcessingJobKind.PowerEfficiency,
    "power": PostProcessingJobKind.PowerEfficiency,
    "efficiency": PostProcessingJobKind.PowerEfficiency,
}

_WINDOW_FUNC_MAP: Dict[str, WindowFunction] = {
    "rectangular": WindowFunction.Rectangular,
    "rect": WindowFunction.Rectangular,
    "hann": WindowFunction.Hann,
    "hanning": WindowFunction.Hann,
    "hamming": WindowFunction.Hamming,
    "blackman": WindowFunction.Blackman,
    "flattop": WindowFunction.FlatTop,
    "flat_top": WindowFunction.FlatTop,
}


def _parse_window_spec(
    node: Dict[str, Any],
    path: str,
    errors: List[str],
) -> PostProcessingWindowSpec:
    spec = PostProcessingWindowSpec()
    if not isinstance(node, dict):
        errors.append(
            f"[PULSIM_PP_E_WINDOW_INVALID] {path}: window must be a mapping"
        )
        return spec

    mode_raw = str(node.get("mode", "time")).lower()
    mode = _WINDOW_MODE_MAP.get(mode_raw)
    if mode is None:
        accepted = list(_WINDOW_MODE_MAP.keys())
        errors.append(
            f"[PULSIM_PP_E_WINDOW_MODE] {path}.mode: unsupported mode "
            f"'{mode_raw}' (accepted: {accepted})"
        )
    else:
        spec.mode = mode

    if "t_start" in node:
        spec.t_start = float(node["t_start"])
    if "t_end" in node:
        spec.t_end = float(node["t_end"])
    if "i_start" in node:
        spec.i_start = int(node["i_start"])
    if "i_end" in node:
        spec.i_end = int(node["i_end"])
    if "cycle_start" in node:
        spec.cycle_start = int(node["cycle_start"])
    if "cycle_end" in node:
        spec.cycle_end = int(node["cycle_end"])
    if "period" in node:
        spec.period = float(node["period"])
    if "min_samples" in node:
        spec.min_samples = int(node["min_samples"])

    return spec


def _parse_post_processing_job(
    node: Any,
    idx: int,
    path: str,
    errors: List[str],
) -> Optional[PostProcessingJob]:
    if not isinstance(node, dict):
        errors.append(
            f"[PULSIM_PP_E_JOB_INVALID] {path}[{idx}]: job must be a mapping"
        )
        return None

    job_id = str(node.get("id", f"job_{idx}"))
    kind_raw = str(node.get("kind", "")).lower()

    if not kind_raw:
        accepted = list(_JOB_KIND_MAP.keys())
        errors.append(
            f"[PULSIM_PP_E_JOB_KIND] {path}[{idx}].kind: "
            f"missing required field 'kind' (accepted: {accepted})"
        )
        return None

    kind = _JOB_KIND_MAP.get(kind_raw)
    if kind is None:
        accepted = list(_JOB_KIND_MAP.keys())
        errors.append(
            f"[PULSIM_PP_E_JOB_KIND] {path}[{idx}].kind: "
            f"unsupported kind '{kind_raw}' (accepted: {accepted})"
        )
        return None

    job = PostProcessingJob(job_id=job_id, kind=kind)

    # Signals
    signals_raw = node.get("signals", [])
    if isinstance(signals_raw, str):
        signals_raw = [signals_raw]
    job.signals = [str(s) for s in signals_raw]

    # Window
    if "window" in node:
        job.window = _parse_window_spec(
            node["window"], f"{path}[{idx}].window", errors
        )

    # Time-domain metric list
    if "metrics" in node:
        m = node["metrics"]
        if isinstance(m, str):
            job.metrics = [x.strip() for x in m.split(",")]
        elif isinstance(m, list):
            job.metrics = [str(x) for x in m]

    # Spectral options
    if "n_harmonics" in node:
        job.n_harmonics = int(node["n_harmonics"])
    if "fundamental_hz" in node:
        job.fundamental_hz = float(node["fundamental_hz"])
    if "window_function" in node:
        wf_raw = str(node["window_function"]).lower()
        wf = _WINDOW_FUNC_MAP.get(wf_raw)
        if wf is None:
            accepted = list(_WINDOW_FUNC_MAP.keys())
            errors.append(
                f"[PULSIM_PP_E_WINDOW_FUNC] {path}[{idx}].window_function: "
                f"unsupported function '{wf_raw}' (accepted: {accepted})"
            )
        else:
            job.window_function = wf

    # Power/efficiency signal bindings
    for attr, keys in [
        ("input_voltage_signal", ["input_voltage", "v_in", "vin"]),
        ("input_current_signal", ["input_current", "i_in", "iin"]),
        ("output_voltage_signal", ["output_voltage", "v_out", "vout"]),
        ("output_current_signal", ["output_current", "i_out", "iout"]),
    ]:
        for key in keys:
            if key in node:
                setattr(job, attr, str(node[key]))
                break

    return job


def parse_post_processing_yaml(
    node: Any,
    errors: Optional[List[str]] = None,
) -> PostProcessingOptions:
    """Parse a ``simulation.post_processing`` YAML node into PostProcessingOptions.

    Args:
        node: The raw Python dict from a parsed YAML ``simulation.post_processing``
              block.  Pass ``None`` for no post-processing (returns empty options).
        errors: Optional list; parse errors are appended here using stable
                machine-readable ``[PULSIM_PP_E_*]`` prefixes.

    Returns:
        :class:`PostProcessingOptions` (empty if *node* is ``None`` or invalid).

    Example YAML block::

        simulation:
          post_processing:
            jobs:
              - id: output_metrics
                kind: time_domain
                signals: ["V(out)"]
                window: {mode: time, t_start: 1.5e-3, t_end: 2.0e-3}
                metrics: [rms, mean, min, max, p2p, ripple]
              - id: input_fft
                kind: spectral
                signals: ["I(L1)"]
                window: {mode: time, t_start: 1.5e-3, t_end: 2.0e-3}
                n_harmonics: 5
                window_function: hann
              - id: efficiency
                kind: power_efficiency
                window: {mode: time, t_start: 1.5e-3, t_end: 2.0e-3}
                input_voltage: "V(Vdc)"
                input_current: "I(Vdc)"
                output_voltage: "V(out)"
                output_current: "I(Rload)"
    """
    if errors is None:
        errors = []

    opts = PostProcessingOptions()

    if node is None:
        return opts

    if not isinstance(node, dict):
        errors.append(
            "[PULSIM_PP_E_BLOCK_INVALID] simulation.post_processing: "
            "must be a mapping (dict)"
        )
        return opts

    jobs_raw = node.get("jobs", [])
    if not isinstance(jobs_raw, list):
        errors.append(
            "[PULSIM_PP_E_JOBS_INVALID] simulation.post_processing.jobs: "
            "must be a list"
        )
        return opts

    path = "simulation.post_processing.jobs"
    for idx, job_node in enumerate(jobs_raw):
        job = _parse_post_processing_job(job_node, idx, path, errors)
        if job is not None:
            opts.jobs.append(job)

    return opts
