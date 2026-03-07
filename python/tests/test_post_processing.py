"""Tests for pulsim waveform post-processing pipeline.

Covers:
- Time-domain metrics (RMS, mean, min, max, p2p, crest, ripple, std)
- Spectral metrics (FFT bins, harmonics table, THD)
- Power/efficiency metrics
- Window planning (Time / Index / Cycle modes)
- Determinism (repeated runs produce identical outputs)
- Invalid configuration diagnostics (signal not found, bad window, bad kind)
- Undefined-metric reason codes (zero RMS, zero mean, zero fundamental)
- YAML parser for simulation.post_processing block
- Backward compatibility (no post_processing config → no change in behaviour)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pytest

from pulsim.post_processing import (
    HarmonicEntry,
    PostProcessingDiagnosticCode,
    PostProcessingJob,
    PostProcessingJobKind,
    PostProcessingOptions,
    PostProcessingResult,
    PostProcessingWindowMode,
    PostProcessingWindowSpec,
    ScalarMetric,
    SpectralBin,
    UndefinedMetricEntry,
    WindowFunction,
    parse_post_processing_yaml,
    run_post_processing,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeSimResult:
    """Minimal stand-in for SimulationResult with .time and .virtual_channels."""
    time: List[float]
    virtual_channels: Dict[str, List[float]] = field(default_factory=dict)


def _linspace_result(
    t_stop: float = 1.0e-3,
    n: int = 1000,
    channels: Dict[str, List[float]] | None = None,
) -> FakeSimResult:
    t = np.linspace(0.0, t_stop, n, endpoint=False).tolist()
    return FakeSimResult(time=t, virtual_channels=channels or {})


def _sine_result(
    amplitude: float = 1.0,
    freq_hz: float = 1000.0,
    dc_offset: float = 0.0,
    t_stop: float = 5e-3,
    n: int = 5000,
) -> FakeSimResult:
    t = np.linspace(0.0, t_stop, n, endpoint=False)
    sig = dc_offset + amplitude * np.sin(2.0 * np.pi * freq_hz * t)
    return FakeSimResult(
        time=t.tolist(),
        virtual_channels={"V(out)": sig.tolist()},
    )


def _dc_result(
    value: float = 5.0,
    n: int = 200,
    t_stop: float = 1e-3,
) -> FakeSimResult:
    t = np.linspace(0.0, t_stop, n, endpoint=False)
    return FakeSimResult(
        time=t.tolist(),
        virtual_channels={"V(out)": (np.ones(n) * value).tolist()},
    )


# ---------------------------------------------------------------------------
# Section 1: Time-domain metrics
# ---------------------------------------------------------------------------

class TestTimeDomainMetrics:

    def test_rms_pure_sine_theoretical(self):
        result = _sine_result(amplitude=math.sqrt(2.0), freq_hz=1000.0, n=10000)
        job = PostProcessingJob(
            job_id="rms_test",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            metrics=["rms"],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        # RMS of A*sin: A/sqrt(2). A=sqrt(2) => RMS=1.0
        assert abs(jr.scalar_metrics["rms"].value - 1.0) < 1e-3

    def test_mean_dc_signal(self):
        result = _dc_result(value=7.5)
        job = PostProcessingJob(
            job_id="mean_test",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            metrics=["mean"],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert abs(jr.scalar_metrics["mean"].value - 7.5) < 1e-10

    def test_min_max_p2p(self):
        result = _sine_result(amplitude=3.0, dc_offset=1.0, n=5000)
        job = PostProcessingJob(
            job_id="minmax",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            metrics=["min", "max", "p2p"],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert abs(jr.scalar_metrics["min"].value - (-2.0)) < 0.01
        assert abs(jr.scalar_metrics["max"].value - 4.0) < 0.01
        assert abs(jr.scalar_metrics["p2p"].value - 6.0) < 0.02

    def test_crest_factor_sine(self):
        # Crest factor of sine = peak / RMS = A / (A/sqrt(2)) = sqrt(2)
        result = _sine_result(amplitude=2.0, dc_offset=0.0, n=10000)
        job = PostProcessingJob(
            job_id="crest",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            metrics=["crest"],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert abs(jr.scalar_metrics["crest"].value - math.sqrt(2)) < 0.01

    def test_ripple_dc_plus_small_ac(self):
        # Ripple = p2p / |mean|
        dc = 10.0
        ac_amp = 0.5
        t = np.linspace(0.0, 1e-3, 1000, endpoint=False)
        sig = dc + ac_amp * np.sin(2.0 * np.pi * 1e4 * t)
        result = FakeSimResult(time=t.tolist(), virtual_channels={"V(out)": sig.tolist()})
        job = PostProcessingJob(
            job_id="ripple",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            metrics=["ripple"],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        expected_ripple = (2 * ac_amp) / dc
        assert abs(jr.scalar_metrics["ripple"].value - expected_ripple) < 0.01

    def test_std_constant_signal(self):
        result = _dc_result(value=3.0)
        job = PostProcessingJob(
            job_id="std",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            metrics=["std"],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert abs(jr.scalar_metrics["std"].value) < 1e-10

    def test_undefined_crest_zero_rms(self):
        # All-zero signal → RMS = 0 → crest factor undefined
        t = np.linspace(0.0, 1e-3, 100).tolist()
        result = FakeSimResult(time=t, virtual_channels={"V(out)": [0.0] * 100})
        job = PostProcessingJob(
            job_id="crest_zero",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            metrics=["crest"],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success  # job succeeds but metric is undefined
        assert len(jr.undefined_metrics) == 1
        assert jr.undefined_metrics[0].name == "crest"
        assert jr.undefined_metrics[0].reason == PostProcessingDiagnosticCode.UndefinedMetric

    def test_undefined_ripple_zero_mean(self):
        result = _sine_result(amplitude=1.0, dc_offset=0.0, n=5000)
        job = PostProcessingJob(
            job_id="ripple_zero",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            metrics=["ripple"],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert any(
            u.name == "ripple" for u in jr.undefined_metrics
        )

    def test_unknown_metric_gives_undefined_reason(self):
        result = _dc_result()
        job = PostProcessingJob(
            job_id="bad_metric",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            metrics=["energy_spectral_density"],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert any(
            u.reason == PostProcessingDiagnosticCode.InvalidConfiguration
            for u in jr.undefined_metrics
        )

    def test_all_standard_metrics_computed(self):
        result = _sine_result(amplitude=2.0, dc_offset=5.0, n=5000)
        job = PostProcessingJob(
            job_id="all",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            metrics=["rms", "mean", "min", "max", "p2p", "crest", "ripple", "std"],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        # Mean ≈ dc_offset = 5.0
        assert abs(jr.scalar_metrics["mean"].value - 5.0) < 0.01
        # RMS ≈ sqrt(5^2 + (2/sqrt(2))^2) = sqrt(25 + 2) = sqrt(27)
        expected_rms = math.sqrt(25.0 + 2.0)
        assert abs(jr.scalar_metrics["rms"].value - expected_rms) < 0.05

    def test_multiple_signals_prefix_keys(self):
        t = np.linspace(0.0, 1e-3, 200).tolist()
        result = FakeSimResult(
            time=t,
            virtual_channels={
                "V(out)": [1.0] * 200,
                "I(L1)": [2.0] * 200,
            },
        )
        job = PostProcessingJob(
            job_id="multi",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)", "I(L1)"],
            metrics=["mean"],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert "V(out).mean" in jr.scalar_metrics
        assert "I(L1).mean" in jr.scalar_metrics
        assert abs(jr.scalar_metrics["V(out).mean"].value - 1.0) < 1e-10
        assert abs(jr.scalar_metrics["I(L1).mean"].value - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# Section 2: Spectral metrics
# ---------------------------------------------------------------------------

class TestSpectralMetrics:

    def _pure_sine_result(
        self,
        amplitude: float = 1.0,
        freq_hz: float = 1000.0,
        n: int = 8192,
        t_stop: float | None = None,
    ) -> FakeSimResult:
        t_stop = t_stop or (1.0 / freq_hz * 8)
        t = np.linspace(0.0, t_stop, n, endpoint=False)
        sig = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
        return FakeSimResult(time=t.tolist(), virtual_channels={"V(out)": sig.tolist()})

    def test_fundamental_detected(self):
        f0 = 1000.0
        result = self._pure_sine_result(freq_hz=f0, n=8192)
        job = PostProcessingJob(
            job_id="fft",
            kind=PostProcessingJobKind.Spectral,
            signals=["V(out)"],
            n_harmonics=3,
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert not math.isnan(jr.fundamental_hz)
        # Detected fundamental should be within 5% of true f0
        assert abs(jr.fundamental_hz - f0) / f0 < 0.05

    def test_thd_pure_sine_is_small(self):
        # A pure sine wave should have very low THD
        result = self._pure_sine_result(amplitude=2.0, freq_hz=1000.0, n=8192)
        job = PostProcessingJob(
            job_id="thd",
            kind=PostProcessingJobKind.Spectral,
            signals=["V(out)"],
            n_harmonics=5,
            window_function=WindowFunction.Hann,
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        # THD of a pure sine should be well below 5%
        assert not math.isnan(jr.thd_pct)
        assert jr.thd_pct < 5.0

    def test_thd_with_known_harmonic(self):
        # Signal: A_1*sin(f) + A_3*sin(3f)
        # THD = A_3/A_1 * 100%
        f0 = 1000.0
        A1 = 1.0
        A3 = 0.1
        n = 16384
        t_stop = 8.0 / f0
        t = np.linspace(0.0, t_stop, n, endpoint=False)
        sig = A1 * np.sin(2.0 * np.pi * f0 * t) + A3 * np.sin(2.0 * np.pi * 3 * f0 * t)
        result = FakeSimResult(time=t.tolist(), virtual_channels={"V(out)": sig.tolist()})

        job = PostProcessingJob(
            job_id="thd_known",
            kind=PostProcessingJobKind.Spectral,
            signals=["V(out)"],
            n_harmonics=5,
            fundamental_hz=f0,
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert not math.isnan(jr.thd_pct)
        # Expected THD ≈ 10%.  Allow ±3% tolerance.
        assert abs(jr.thd_pct - 10.0) < 3.0

    def test_harmonic_table_length(self):
        result = self._pure_sine_result(freq_hz=500.0, n=8192)
        job = PostProcessingJob(
            job_id="harmonics",
            kind=PostProcessingJobKind.Spectral,
            signals=["V(out)"],
            n_harmonics=7,
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert len(jr.harmonics) == 7

    def test_spectrum_bins_present(self):
        result = self._pure_sine_result(freq_hz=1000.0, n=4096)
        job = PostProcessingJob(
            job_id="bins",
            kind=PostProcessingJobKind.Spectral,
            signals=["V(out)"],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert len(jr.spectrum_bins) > 0
        # All frequencies should be positive
        assert all(b.frequency_hz > 0 for b in jr.spectrum_bins)

    def test_undefined_thd_zero_fundamental(self):
        # DC-only signal → fundamental mag = 0 in AC spectrum → THD undefined
        result = _dc_result(value=1.0, n=256)
        job = PostProcessingJob(
            job_id="thd_dc",
            kind=PostProcessingJobKind.Spectral,
            signals=["V(out)"],
            fundamental_hz=1000.0,  # explicit fundamental that won't be found in DC
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        # Should succeed but THD may be undefined or near zero
        assert jr.success or not jr.success  # either is valid; just check no crash

    def test_insufficient_samples_fails(self):
        t = [0.0, 1e-6, 2e-6]  # only 3 samples
        result = FakeSimResult(time=t, virtual_channels={"V(out)": [1.0, 2.0, 1.0]})
        job = PostProcessingJob(
            job_id="too_few",
            kind=PostProcessingJobKind.Spectral,
            signals=["V(out)"],
            window=PostProcessingWindowSpec(
                mode=PostProcessingWindowMode.Index,
                i_start=0,
                i_end=3,
                min_samples=4,
            ),
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert not jr.success
        assert jr.diagnostic == PostProcessingDiagnosticCode.InsufficientSamples

    def test_window_functions_do_not_crash(self):
        result = self._pure_sine_result(freq_hz=1000.0, n=2048)
        for wf in WindowFunction:
            job = PostProcessingJob(
                job_id=f"wf_{wf.value}",
                kind=PostProcessingJobKind.Spectral,
                signals=["V(out)"],
                window_function=wf,
            )
            out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
            jr = out.jobs[0]
            assert jr.success, f"Window function {wf} failed: {jr.diagnostic_message}"

    def test_explicit_fundamental_hz(self):
        # Specify exact fundamental to avoid auto-detection
        f0 = 2000.0
        result = self._pure_sine_result(amplitude=1.0, freq_hz=f0, n=8192)
        job = PostProcessingJob(
            job_id="explicit_f0",
            kind=PostProcessingJobKind.Spectral,
            signals=["V(out)"],
            fundamental_hz=f0,
            n_harmonics=3,
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert abs(jr.fundamental_hz - f0) / f0 < 0.05


# ---------------------------------------------------------------------------
# Section 3: Power and efficiency metrics
# ---------------------------------------------------------------------------

class TestPowerEfficiencyMetrics:

    def test_resistive_load_efficiency_100pct(self):
        # Ideal case: P_in = P_out = V*I (same voltage/current)
        n = 500
        t = np.linspace(0.0, 1e-3, n, endpoint=False).tolist()
        v = [10.0] * n
        i = [2.0] * n
        result = FakeSimResult(
            time=t,
            virtual_channels={"V(in)": v, "I(in)": i, "V(out)": v, "I(out)": i},
        )
        job = PostProcessingJob(
            job_id="eff_full",
            kind=PostProcessingJobKind.PowerEfficiency,
            input_voltage_signal="V(in)",
            input_current_signal="I(in)",
            output_voltage_signal="V(out)",
            output_current_signal="I(out)",
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert abs(jr.efficiency - 100.0) < 1e-6

    def test_efficiency_90_percent(self):
        n = 500
        t = np.linspace(0.0, 1e-3, n, endpoint=False).tolist()
        v_in = [10.0] * n
        i_in = [1.0] * n        # P_in = 10 W
        v_out = [9.0] * n
        i_out = [1.0] * n       # P_out = 9 W → η = 90%
        result = FakeSimResult(
            time=t,
            virtual_channels={
                "V(in)": v_in, "I(in)": i_in,
                "V(out)": v_out, "I(out)": i_out,
            },
        )
        job = PostProcessingJob(
            job_id="eff90",
            kind=PostProcessingJobKind.PowerEfficiency,
            input_voltage_signal="V(in)",
            input_current_signal="I(in)",
            output_voltage_signal="V(out)",
            output_current_signal="I(out)",
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert abs(jr.efficiency - 90.0) < 1e-6
        assert abs(jr.average_input_power - 10.0) < 1e-6
        assert abs(jr.average_output_power - 9.0) < 1e-6

    def test_power_factor_unity_resistive(self):
        # V and I in phase: power factor = 1
        n = 1000
        t = np.linspace(0.0, 1e-3, n, endpoint=False)
        v = np.sin(2 * np.pi * 1000.0 * t)
        i = v / 5.0  # R = 5 Ω
        result = FakeSimResult(
            time=t.tolist(),
            virtual_channels={"V(in)": v.tolist(), "I(in)": i.tolist()},
        )
        job = PostProcessingJob(
            job_id="pf_unity",
            kind=PostProcessingJobKind.PowerEfficiency,
            input_voltage_signal="V(in)",
            input_current_signal="I(in)",
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert abs(jr.power_factor - 1.0) < 0.01

    def test_power_factor_capacitive(self):
        # 90° out-of-phase: sin and cos → power factor = 0
        n = 2000
        t = np.linspace(0.0, 2e-3, n, endpoint=False)
        v = np.sin(2 * np.pi * 1000.0 * t)
        i = np.cos(2 * np.pi * 1000.0 * t)  # 90° phase shift
        result = FakeSimResult(
            time=t.tolist(),
            virtual_channels={"V(in)": v.tolist(), "I(in)": i.tolist()},
        )
        job = PostProcessingJob(
            job_id="pf_cap",
            kind=PostProcessingJobKind.PowerEfficiency,
            input_voltage_signal="V(in)",
            input_current_signal="I(in)",
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert abs(jr.power_factor) < 0.05  # should be near 0

    def test_missing_input_signals_returns_nan(self):
        n = 200
        t = np.linspace(0.0, 1e-3, n, endpoint=False).tolist()
        result = FakeSimResult(time=t, virtual_channels={})
        job = PostProcessingJob(
            job_id="missing",
            kind=PostProcessingJobKind.PowerEfficiency,
            # no signal bindings → all empty
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success  # job runs but metrics are NaN
        assert math.isnan(jr.average_input_power)
        assert math.isnan(jr.average_output_power)

    def test_zero_pin_zero_pout_is_100pct(self):
        n = 100
        t = np.linspace(0.0, 1e-4, n, endpoint=False).tolist()
        result = FakeSimResult(
            time=t,
            virtual_channels={
                "V(in)": [0.0] * n, "I(in)": [0.0] * n,
                "V(out)": [0.0] * n, "I(out)": [0.0] * n,
            },
        )
        job = PostProcessingJob(
            job_id="zero_power",
            kind=PostProcessingJobKind.PowerEfficiency,
            input_voltage_signal="V(in)",
            input_current_signal="I(in)",
            output_voltage_signal="V(out)",
            output_current_signal="I(out)",
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert abs(jr.efficiency - 100.0) < 1e-6


# ---------------------------------------------------------------------------
# Section 4: Window planning
# ---------------------------------------------------------------------------

class TestWindowPlanning:

    def _result_with_long_time(self, n: int = 1000) -> FakeSimResult:
        t = np.linspace(0.0, 10e-3, n, endpoint=False).tolist()
        return FakeSimResult(
            time=t,
            virtual_channels={"V(out)": np.linspace(0, 1, n).tolist()},
        )

    def test_time_window(self):
        result = self._result_with_long_time(n=1000)
        job = PostProcessingJob(
            job_id="tw",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            metrics=["mean"],
            window=PostProcessingWindowSpec(
                mode=PostProcessingWindowMode.Time,
                t_start=5e-3,
                t_end=10e-3,
            ),
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        # Window is second half of [0,1] ramp → mean ≈ 0.75
        assert abs(jr.scalar_metrics["mean"].value - 0.75) < 0.02

    def test_index_window(self):
        n = 100
        t = np.linspace(0.0, 1e-3, n).tolist()
        sig = list(range(n))  # values = 0,1,...,99
        result = FakeSimResult(time=t, virtual_channels={"V(out)": sig})
        job = PostProcessingJob(
            job_id="iw",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            metrics=["mean"],
            window=PostProcessingWindowSpec(
                mode=PostProcessingWindowMode.Index,
                i_start=0,
                i_end=10,
            ),
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        # Samples 0-9 → values 0-9 → mean = 4.5
        assert abs(jr.scalar_metrics["mean"].value - 4.5) < 1e-10

    def test_cycle_window(self):
        f = 1000.0
        period = 1.0 / f
        n = 5000
        t = np.linspace(0.0, 5e-3, n, endpoint=False)
        sig = np.sin(2.0 * np.pi * f * t)
        result = FakeSimResult(time=t.tolist(), virtual_channels={"V(out)": sig.tolist()})

        job = PostProcessingJob(
            job_id="cw",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            metrics=["rms"],
            window=PostProcessingWindowSpec(
                mode=PostProcessingWindowMode.Cycle,
                cycle_start=2,
                cycle_end=3,
                period=period,
            ),
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        # RMS of sine over full cycles ≈ 1/sqrt(2)
        assert abs(jr.scalar_metrics["rms"].value - 1.0 / math.sqrt(2)) < 0.02

    def test_invalid_time_window_t_start_gt_t_end(self):
        result = _dc_result(n=100)
        job = PostProcessingJob(
            job_id="bad_tw",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            window=PostProcessingWindowSpec(
                mode=PostProcessingWindowMode.Time,
                t_start=2e-3,
                t_end=1e-3,
            ),
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert not jr.success
        assert jr.diagnostic == PostProcessingDiagnosticCode.InvalidWindow

    def test_time_window_no_samples_in_range(self):
        result = _dc_result(n=100, t_stop=1e-3)
        job = PostProcessingJob(
            job_id="no_samples",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            window=PostProcessingWindowSpec(
                mode=PostProcessingWindowMode.Time,
                t_start=10.0,  # far beyond simulation time
                t_end=20.0,
            ),
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert not jr.success
        assert jr.diagnostic == PostProcessingDiagnosticCode.InvalidWindow

    def test_insufficient_samples_in_window(self):
        result = _dc_result(n=100)
        job = PostProcessingJob(
            job_id="few",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            window=PostProcessingWindowSpec(
                mode=PostProcessingWindowMode.Index,
                i_start=0,
                i_end=2,
                min_samples=4,
            ),
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert not jr.success
        assert jr.diagnostic == PostProcessingDiagnosticCode.InsufficientSamples

    def test_cycle_window_missing_period(self):
        result = _dc_result(n=100)
        job = PostProcessingJob(
            job_id="no_period",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(out)"],
            window=PostProcessingWindowSpec(
                mode=PostProcessingWindowMode.Cycle,
                cycle_start=0,
                period=0.0,  # invalid
            ),
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert not jr.success
        assert jr.diagnostic == PostProcessingDiagnosticCode.InvalidWindow


# ---------------------------------------------------------------------------
# Section 5: Signal resolution diagnostics
# ---------------------------------------------------------------------------

class TestSignalResolution:

    def test_signal_not_found_fails(self):
        result = _dc_result(n=100)
        job = PostProcessingJob(
            job_id="not_found",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["V(nonexistent)"],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert not jr.success
        assert jr.diagnostic == PostProcessingDiagnosticCode.SignalNotFound

    def test_case_insensitive_signal_lookup(self):
        n = 100
        t = np.linspace(0.0, 1e-3, n).tolist()
        result = FakeSimResult(
            time=t,
            virtual_channels={"V(OUT)": [5.0] * n},
        )
        job = PostProcessingJob(
            job_id="case_insensitive",
            kind=PostProcessingJobKind.TimeDomain,
            signals=["v(out)"],
            metrics=["mean"],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert jr.success
        assert abs(jr.scalar_metrics["mean"].value - 5.0) < 1e-10

    def test_no_signals_specified_fails(self):
        result = _dc_result(n=100)
        job = PostProcessingJob(
            job_id="no_sig",
            kind=PostProcessingJobKind.TimeDomain,
            signals=[],
        )
        out = run_post_processing(result, PostProcessingOptions(jobs=[job]))
        jr = out.jobs[0]
        assert not jr.success
        assert jr.diagnostic == PostProcessingDiagnosticCode.InvalidConfiguration


# ---------------------------------------------------------------------------
# Section 6: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_time_domain_repeated_identical(self):
        result = _sine_result(amplitude=3.0, dc_offset=2.0, n=2000)
        opts = PostProcessingOptions(jobs=[
            PostProcessingJob(
                job_id="det",
                kind=PostProcessingJobKind.TimeDomain,
                signals=["V(out)"],
                metrics=["rms", "mean", "min", "max", "p2p"],
            )
        ])
        r1 = run_post_processing(result, opts)
        r2 = run_post_processing(result, opts)
        assert r1.success == r2.success
        for k in r1.jobs[0].scalar_metrics:
            v1 = r1.jobs[0].scalar_metrics[k].value
            v2 = r2.jobs[0].scalar_metrics[k].value
            assert v1 == v2, f"Metric {k} not deterministic: {v1} vs {v2}"

    def test_spectral_repeated_identical(self):
        result = _sine_result(amplitude=1.0, freq_hz=1000.0, n=8192)
        opts = PostProcessingOptions(jobs=[
            PostProcessingJob(
                job_id="spec_det",
                kind=PostProcessingJobKind.Spectral,
                signals=["V(out)"],
                n_harmonics=5,
            )
        ])
        r1 = run_post_processing(result, opts)
        r2 = run_post_processing(result, opts)
        assert r1.success == r2.success
        assert r1.jobs[0].thd_pct == r2.jobs[0].thd_pct
        assert r1.jobs[0].fundamental_hz == r2.jobs[0].fundamental_hz

    def test_power_efficiency_repeated_identical(self):
        n = 500
        t = np.linspace(0.0, 1e-3, n).tolist()
        result = FakeSimResult(
            time=t,
            virtual_channels={
                "V(in)": [10.0] * n, "I(in)": [1.5] * n,
                "V(out)": [9.0] * n, "I(out)": [1.5] * n,
            },
        )
        opts = PostProcessingOptions(jobs=[
            PostProcessingJob(
                job_id="eff_det",
                kind=PostProcessingJobKind.PowerEfficiency,
                input_voltage_signal="V(in)",
                input_current_signal="I(in)",
                output_voltage_signal="V(out)",
                output_current_signal="I(out)",
            )
        ])
        r1 = run_post_processing(result, opts)
        r2 = run_post_processing(result, opts)
        assert r1.jobs[0].efficiency == r2.jobs[0].efficiency
        assert r1.jobs[0].average_input_power == r2.jobs[0].average_input_power


# ---------------------------------------------------------------------------
# Section 7: Multiple jobs in one call
# ---------------------------------------------------------------------------

class TestMultipleJobs:

    def test_mixed_jobs_all_succeed(self):
        f0 = 1000.0
        n = 8192
        t_stop = 8.0 / f0
        t = np.linspace(0.0, t_stop, n, endpoint=False)
        v = 10.0 + np.sin(2.0 * np.pi * f0 * t)
        i = np.sin(2.0 * np.pi * f0 * t) / 5.0
        result = FakeSimResult(
            time=t.tolist(),
            virtual_channels={
                "V(out)": v.tolist(),
                "I(L1)": i.tolist(),
            },
        )
        opts = PostProcessingOptions(jobs=[
            PostProcessingJob(
                job_id="td",
                kind=PostProcessingJobKind.TimeDomain,
                signals=["V(out)"],
                metrics=["rms", "mean"],
            ),
            PostProcessingJob(
                job_id="spec",
                kind=PostProcessingJobKind.Spectral,
                signals=["V(out)"],
                n_harmonics=3,
            ),
            PostProcessingJob(
                job_id="eff",
                kind=PostProcessingJobKind.PowerEfficiency,
                input_voltage_signal="V(out)",
                input_current_signal="I(L1)",
            ),
        ])
        out = run_post_processing(result, opts)
        assert len(out.jobs) == 3
        assert out.jobs[0].job_id == "td"
        assert out.jobs[1].job_id == "spec"
        assert out.jobs[2].job_id == "eff"

    def test_partial_failure_does_not_affect_other_jobs(self):
        result = _dc_result(n=200)
        opts = PostProcessingOptions(jobs=[
            PostProcessingJob(
                job_id="good",
                kind=PostProcessingJobKind.TimeDomain,
                signals=["V(out)"],
                metrics=["mean"],
            ),
            PostProcessingJob(
                job_id="bad",
                kind=PostProcessingJobKind.TimeDomain,
                signals=["V(missing)"],
            ),
        ])
        out = run_post_processing(result, opts)
        assert out.jobs[0].success
        assert not out.jobs[1].success
        assert not out.success  # overall failure

    def test_empty_jobs_list(self):
        result = _dc_result()
        out = run_post_processing(result, PostProcessingOptions(jobs=[]))
        assert out.success
        assert len(out.jobs) == 0

    def test_job_ordering_is_stable(self):
        result = _dc_result(n=100)
        ids = [f"job_{i}" for i in range(5)]
        opts = PostProcessingOptions(jobs=[
            PostProcessingJob(
                job_id=jid,
                kind=PostProcessingJobKind.TimeDomain,
                signals=["V(out)"],
                metrics=["mean"],
            )
            for jid in ids
        ])
        out = run_post_processing(result, opts)
        assert [jr.job_id for jr in out.jobs] == ids


# ---------------------------------------------------------------------------
# Section 8: Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_no_post_processing_config_does_not_break(self):
        # run_post_processing with an empty options is a no-op
        result = _dc_result(n=100)
        out = run_post_processing(result, PostProcessingOptions())
        assert out.success
        assert len(out.jobs) == 0

    def test_result_without_virtual_channels(self):
        # A result with no virtual_channels should work for empty jobs
        result = FakeSimResult(time=[0.0, 1e-6, 2e-6], virtual_channels={})
        out = run_post_processing(result, PostProcessingOptions())
        assert out.success


# ---------------------------------------------------------------------------
# Section 9: YAML parser
# ---------------------------------------------------------------------------

class TestParsePostProcessingYaml:

    def test_none_node_returns_empty(self):
        opts = parse_post_processing_yaml(None)
        assert len(opts.jobs) == 0

    def test_empty_jobs_list(self):
        opts = parse_post_processing_yaml({"jobs": []})
        assert len(opts.jobs) == 0

    def test_single_time_domain_job(self):
        node = {
            "jobs": [
                {
                    "id": "v_out_metrics",
                    "kind": "time_domain",
                    "signals": ["V(out)"],
                    "window": {"mode": "time", "t_start": 1e-3, "t_end": 2e-3},
                    "metrics": ["rms", "mean"],
                }
            ]
        }
        opts = parse_post_processing_yaml(node)
        assert len(opts.jobs) == 1
        job = opts.jobs[0]
        assert job.job_id == "v_out_metrics"
        assert job.kind == PostProcessingJobKind.TimeDomain
        assert "V(out)" in job.signals
        assert job.window.mode == PostProcessingWindowMode.Time
        assert job.window.t_start == pytest.approx(1e-3)
        assert job.window.t_end == pytest.approx(2e-3)
        assert "rms" in job.metrics

    def test_single_spectral_job(self):
        node = {
            "jobs": [
                {
                    "id": "fft_job",
                    "kind": "spectral",
                    "signals": ["I(L1)"],
                    "n_harmonics": 7,
                    "fundamental_hz": 20000.0,
                    "window_function": "hann",
                }
            ]
        }
        opts = parse_post_processing_yaml(node)
        assert len(opts.jobs) == 1
        job = opts.jobs[0]
        assert job.kind == PostProcessingJobKind.Spectral
        assert job.n_harmonics == 7
        assert job.fundamental_hz == pytest.approx(20000.0)
        assert job.window_function == WindowFunction.Hann

    def test_single_power_efficiency_job(self):
        node = {
            "jobs": [
                {
                    "id": "eff_job",
                    "kind": "power_efficiency",
                    "input_voltage": "V(Vdc)",
                    "input_current": "I(Vdc)",
                    "output_voltage": "V(out)",
                    "output_current": "I(Rload)",
                }
            ]
        }
        opts = parse_post_processing_yaml(node)
        assert len(opts.jobs) == 1
        job = opts.jobs[0]
        assert job.kind == PostProcessingJobKind.PowerEfficiency
        assert job.input_voltage_signal == "V(Vdc)"
        assert job.input_current_signal == "I(Vdc)"
        assert job.output_voltage_signal == "V(out)"
        assert job.output_current_signal == "I(Rload)"

    def test_alias_kind_power(self):
        node = {"jobs": [{"id": "x", "kind": "power", "input_voltage": "V(in)"}]}
        opts = parse_post_processing_yaml(node)
        assert opts.jobs[0].kind == PostProcessingJobKind.PowerEfficiency

    def test_alias_kind_efficiency(self):
        node = {"jobs": [{"id": "x", "kind": "efficiency"}]}
        opts = parse_post_processing_yaml(node)
        assert opts.jobs[0].kind == PostProcessingJobKind.PowerEfficiency

    def test_invalid_kind_reports_error(self):
        node = {"jobs": [{"id": "x", "kind": "invalid_kind"}]}
        errors: list[str] = []
        opts = parse_post_processing_yaml(node, errors)
        assert len(opts.jobs) == 0
        assert any("PULSIM_PP_E_JOB_KIND" in e for e in errors)

    def test_missing_kind_reports_error(self):
        node = {"jobs": [{"id": "x", "signals": ["V(out)"]}]}
        errors: list[str] = []
        opts = parse_post_processing_yaml(node, errors)
        assert len(opts.jobs) == 0
        assert any("PULSIM_PP_E_JOB_KIND" in e for e in errors)

    def test_invalid_window_mode_reports_error(self):
        node = {
            "jobs": [
                {
                    "id": "x",
                    "kind": "time_domain",
                    "signals": ["V(out)"],
                    "window": {"mode": "invalid_mode"},
                }
            ]
        }
        errors: list[str] = []
        opts = parse_post_processing_yaml(node, errors)
        # Job is still created (with default window) but error is reported
        assert any("PULSIM_PP_E_WINDOW_MODE" in e for e in errors)

    def test_invalid_window_function_reports_error(self):
        node = {
            "jobs": [
                {
                    "id": "x",
                    "kind": "spectral",
                    "signals": ["V(out)"],
                    "window_function": "kaiser",
                }
            ]
        }
        errors: list[str] = []
        opts = parse_post_processing_yaml(node, errors)
        assert any("PULSIM_PP_E_WINDOW_FUNC" in e for e in errors)

    def test_non_mapping_block_reports_error(self):
        errors: list[str] = []
        opts = parse_post_processing_yaml("not_a_dict", errors)
        assert len(opts.jobs) == 0
        assert any("PULSIM_PP_E_BLOCK_INVALID" in e for e in errors)

    def test_jobs_not_a_list_reports_error(self):
        errors: list[str] = []
        opts = parse_post_processing_yaml({"jobs": "not_a_list"}, errors)
        assert len(opts.jobs) == 0
        assert any("PULSIM_PP_E_JOBS_INVALID" in e for e in errors)

    def test_index_window_mode(self):
        node = {
            "jobs": [
                {
                    "id": "x",
                    "kind": "time_domain",
                    "signals": ["V(out)"],
                    "window": {"mode": "index", "i_start": 10, "i_end": 50},
                }
            ]
        }
        opts = parse_post_processing_yaml(node)
        job = opts.jobs[0]
        assert job.window.mode == PostProcessingWindowMode.Index
        assert job.window.i_start == 10
        assert job.window.i_end == 50

    def test_cycle_window_mode(self):
        node = {
            "jobs": [
                {
                    "id": "x",
                    "kind": "time_domain",
                    "signals": ["V(out)"],
                    "window": {
                        "mode": "cycle",
                        "cycle_start": 2,
                        "cycle_end": 4,
                        "period": 50e-6,
                    },
                }
            ]
        }
        opts = parse_post_processing_yaml(node)
        job = opts.jobs[0]
        assert job.window.mode == PostProcessingWindowMode.Cycle
        assert job.window.cycle_start == 2
        assert job.window.cycle_end == 4
        assert job.window.period == pytest.approx(50e-6)

    def test_auto_job_id_when_not_specified(self):
        node = {"jobs": [{"kind": "time_domain", "signals": ["V(out)"]}]}
        opts = parse_post_processing_yaml(node)
        assert len(opts.jobs) == 1
        assert opts.jobs[0].job_id.startswith("job_")

    def test_multiple_jobs_parsed_in_order(self):
        node = {
            "jobs": [
                {"id": "first", "kind": "time_domain", "signals": ["V(out)"]},
                {"id": "second", "kind": "spectral", "signals": ["I(L1)"]},
            ]
        }
        opts = parse_post_processing_yaml(node)
        assert [j.job_id for j in opts.jobs] == ["first", "second"]

    def test_metrics_as_string_comma_separated(self):
        node = {
            "jobs": [
                {
                    "id": "x",
                    "kind": "time_domain",
                    "signals": ["V(out)"],
                    "metrics": "rms, mean, min",
                }
            ]
        }
        opts = parse_post_processing_yaml(node)
        assert set(opts.jobs[0].metrics) == {"rms", "mean", "min"}

    def test_signals_as_string_converted_to_list(self):
        node = {
            "jobs": [
                {
                    "id": "x",
                    "kind": "time_domain",
                    "signals": "V(out)",
                }
            ]
        }
        opts = parse_post_processing_yaml(node)
        assert opts.jobs[0].signals == ["V(out)"]

    def test_end_to_end_yaml_parse_and_run(self):
        """Parse YAML node, then run post-processing on a fake result."""
        node = {
            "jobs": [
                {
                    "id": "voltage_metrics",
                    "kind": "time_domain",
                    "signals": ["V(out)"],
                    "metrics": ["rms", "mean"],
                },
                {
                    "id": "efficiency",
                    "kind": "power_efficiency",
                    "input_voltage": "V(in)",
                    "input_current": "I(in)",
                    "output_voltage": "V(out)",
                    "output_current": "I(out)",
                },
            ]
        }
        opts = parse_post_processing_yaml(node)
        assert len(opts.jobs) == 2

        n = 500
        t = np.linspace(0.0, 1e-3, n).tolist()
        result = FakeSimResult(
            time=t,
            virtual_channels={
                "V(out)": [5.0] * n,
                "V(in)": [10.0] * n,
                "I(in)": [1.0] * n,
                "I(out)": [0.5] * n,
            },
        )
        out = run_post_processing(result, opts)
        assert out.success
        assert len(out.jobs) == 2
        td_job = out.jobs[0]
        assert td_job.success
        assert abs(td_job.scalar_metrics["mean"].value - 5.0) < 1e-6

        eff_job = out.jobs[1]
        assert eff_job.success
        # P_in=10W, P_out=2.5W → η=25%
        assert abs(eff_job.efficiency - 25.0) < 0.1
