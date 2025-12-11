"""
Result comparison utilities for validation tests.

This module provides tools for comparing simulation results with various
error metrics and interpolation support.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy import interpolate
from scipy import stats


class ResultComparator:
    """
    Compares simulation results from different sources.

    Handles interpolation for different time bases and calculates
    various error metrics.
    """

    def __init__(
        self,
        interpolation_kind: str = "linear",
        skip_initial_transient: float = 0.0,
    ):
        """
        Initialize the comparator.

        Args:
            interpolation_kind: Interpolation method ('linear', 'cubic', etc.)
            skip_initial_transient: Time to skip at start (for settling)
        """
        self.interpolation_kind = interpolation_kind
        self.skip_initial_transient = skip_initial_transient

    def compare(
        self,
        time1: np.ndarray,
        values1: np.ndarray,
        time2: np.ndarray,
        values2: np.ndarray,
        reference_is_second: bool = True,
    ) -> Dict[str, float]:
        """
        Compare two waveforms and compute error metrics.

        Args:
            time1: Time array for first waveform (e.g., Pulsim)
            values1: Values for first waveform
            time2: Time array for second waveform (e.g., reference)
            values2: Values for second waveform
            reference_is_second: If True, errors are computed relative to waveform2

        Returns:
            Dictionary containing:
                - max_error: Maximum absolute error
                - rms_error: Root mean square error
                - mean_error: Mean absolute error
                - max_relative_error: Maximum relative error
                - mean_relative_error: Mean relative error
                - correlation: Pearson correlation coefficient
                - r_squared: R-squared (coefficient of determination)
        """
        # Determine common time range
        t_min = max(time1.min(), time2.min(), self.skip_initial_transient)
        t_max = min(time1.max(), time2.max())

        if t_min >= t_max:
            raise ValueError(
                f"No overlapping time range: "
                f"[{time1.min():.2e}, {time1.max():.2e}] vs "
                f"[{time2.min():.2e}, {time2.max():.2e}]"
            )

        # Use the denser time array for comparison
        n_points1 = np.sum((time1 >= t_min) & (time1 <= t_max))
        n_points2 = np.sum((time2 >= t_min) & (time2 <= t_max))

        if n_points1 >= n_points2:
            # Use time1 as base, interpolate values2
            mask = (time1 >= t_min) & (time1 <= t_max)
            t_compare = time1[mask]
            v1 = values1[mask]
            v2 = self._interpolate(time2, values2, t_compare)
        else:
            # Use time2 as base, interpolate values1
            mask = (time2 >= t_min) & (time2 <= t_max)
            t_compare = time2[mask]
            v1 = self._interpolate(time1, values1, t_compare)
            v2 = values2[mask]

        # Compute errors
        error = v1 - v2
        abs_error = np.abs(error)

        # Reference values for relative error
        ref = v2 if reference_is_second else v1
        ref_max = np.max(np.abs(ref))
        ref_abs = np.abs(ref)

        # Avoid division by zero for relative errors
        nonzero_mask = ref_abs > (ref_max * 1e-10)

        max_error = np.max(abs_error)
        rms_error = np.sqrt(np.mean(error**2))
        mean_error = np.mean(abs_error)

        # Max relative error (as fraction of reference maximum)
        max_relative_error = max_error / ref_max if ref_max > 0 else 0.0

        # Mean relative error (only where reference is significant)
        if np.any(nonzero_mask):
            mean_relative_error = np.mean(abs_error[nonzero_mask] / ref_abs[nonzero_mask])
        else:
            mean_relative_error = 0.0

        # Correlation coefficient
        if np.std(v1) > 0 and np.std(v2) > 0:
            correlation, _ = stats.pearsonr(v1, v2)
        else:
            correlation = 1.0 if np.allclose(v1, v2) else 0.0

        # R-squared
        ss_res = np.sum(error**2)
        ss_tot = np.sum((v2 - np.mean(v2))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        return {
            "max_error": max_error,
            "rms_error": rms_error,
            "mean_error": mean_error,
            "max_relative_error": max_relative_error,
            "mean_relative_error": mean_relative_error,
            "correlation": correlation,
            "r_squared": r_squared,
        }

    def _interpolate(
        self,
        t_source: np.ndarray,
        v_source: np.ndarray,
        t_target: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate values from source time base to target time base.

        Args:
            t_source: Source time array
            v_source: Source values
            t_target: Target time array

        Returns:
            Interpolated values at target times
        """
        f = interpolate.interp1d(
            t_source,
            v_source,
            kind=self.interpolation_kind,
            fill_value="extrapolate",
            bounds_error=False,
        )
        return f(t_target)

    def compare_steady_state(
        self,
        time1: np.ndarray,
        values1: np.ndarray,
        time2: np.ndarray,
        values2: np.ndarray,
        steady_state_fraction: float = 0.2,
    ) -> Dict[str, float]:
        """
        Compare steady-state behavior (last portion of waveforms).

        Useful for converters where initial transient should be ignored.

        Args:
            time1, values1: First waveform
            time2, values2: Second waveform
            steady_state_fraction: Fraction of end time to use (0.2 = last 20%)

        Returns:
            Same metrics as compare(), but only for steady-state region
        """
        # Find steady-state start time
        t_max = min(time1.max(), time2.max())
        t_ss_start = t_max * (1 - steady_state_fraction)

        # Create comparator that skips to steady state
        ss_comparator = ResultComparator(
            interpolation_kind=self.interpolation_kind,
            skip_initial_transient=t_ss_start,
        )

        return ss_comparator.compare(time1, values1, time2, values2)

    def compute_frequency_response(
        self,
        time: np.ndarray,
        values: np.ndarray,
        fundamental_freq: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute frequency response metrics via FFT.

        Args:
            time: Time array
            values: Value array
            fundamental_freq: Expected fundamental frequency (for THD calculation)

        Returns:
            Dictionary with frequency metrics
        """
        dt = np.mean(np.diff(time))
        n = len(values)
        fs = 1.0 / dt

        # FFT
        fft_vals = np.fft.rfft(values)
        freqs = np.fft.rfftfreq(n, dt)
        magnitudes = np.abs(fft_vals) * 2 / n

        # Find dominant frequency
        idx_max = np.argmax(magnitudes[1:]) + 1  # Skip DC
        dominant_freq = freqs[idx_max]
        dominant_magnitude = magnitudes[idx_max]

        # DC component
        dc_component = magnitudes[0] / 2

        # THD calculation (if fundamental known)
        thd = None
        if fundamental_freq is not None:
            # Find fundamental index
            fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
            if fund_idx > 0:
                fundamental_magnitude = magnitudes[fund_idx]
                # Sum harmonic powers
                harmonic_power = 0
                for h in range(2, 11):  # Up to 10th harmonic
                    h_freq = fundamental_freq * h
                    if h_freq < fs / 2:
                        h_idx = np.argmin(np.abs(freqs - h_freq))
                        harmonic_power += magnitudes[h_idx]**2
                if fundamental_magnitude > 0:
                    thd = np.sqrt(harmonic_power) / fundamental_magnitude

        return {
            "dominant_freq": dominant_freq,
            "dominant_magnitude": dominant_magnitude,
            "dc_component": dc_component,
            "thd": thd,
        }


def quick_compare(
    pulsim_time: np.ndarray,
    pulsim_values: np.ndarray,
    ref_time: np.ndarray,
    ref_values: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Quick comparison returning just max error, RMS error, and correlation.

    Convenience function for simple tests.

    Returns:
        Tuple of (max_error, rms_error, correlation)
    """
    comparator = ResultComparator()
    metrics = comparator.compare(pulsim_time, pulsim_values, ref_time, ref_values)
    return metrics["max_error"], metrics["rms_error"], metrics["correlation"]
