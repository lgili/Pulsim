"""Comparison utilities for validation framework."""

import numpy as np
from .base import ValidationResult


def interpolate_to_common_times(
    ref_times: np.ndarray,
    ref_values: np.ndarray,
    target_times: np.ndarray
) -> np.ndarray:
    """Interpolate reference values to target time points.

    Args:
        ref_times: Reference time array
        ref_values: Reference value array
        target_times: Target time points for interpolation

    Returns:
        Interpolated values at target times
    """
    return np.interp(target_times, ref_times, ref_values)


def compare_results(
    test_name: str,
    pulsim_times: np.ndarray,
    pulsim_values: np.ndarray,
    ref_times: np.ndarray,
    ref_values: np.ndarray,
    tolerance: float = 0.01
) -> ValidationResult:
    """Compare Pulsim results against reference.

    Args:
        test_name: Name of the test
        pulsim_times: Pulsim time array
        pulsim_values: Pulsim value array
        ref_times: Reference time array
        ref_values: Reference value array
        tolerance: Maximum relative error tolerance

    Returns:
        ValidationResult with comparison metrics
    """
    # Interpolate reference to Pulsim times if needed
    if len(ref_times) != len(pulsim_times) or not np.allclose(ref_times, pulsim_times):
        ref_interpolated = interpolate_to_common_times(ref_times, ref_values, pulsim_times)
    else:
        ref_interpolated = ref_values

    # Calculate errors
    errors = np.abs(pulsim_values - ref_interpolated)
    max_error = np.max(errors)
    rms_error = np.sqrt(np.mean(errors**2))

    # Calculate relative error
    max_ref = np.max(np.abs(ref_interpolated))
    if max_ref > 1e-12:
        max_rel_error = max_error / max_ref
    else:
        max_rel_error = max_error

    passed = max_rel_error <= tolerance

    return ValidationResult(
        test_name=test_name,
        passed=passed,
        pulsim_times=pulsim_times,
        pulsim_values=pulsim_values,
        reference_times=ref_times,
        reference_values=ref_interpolated,
        max_error=max_error,
        rms_error=rms_error,
        max_relative_error=max_rel_error,
        tolerance=tolerance,
        execution_time_ms=0.0
    )


def calculate_steady_state_error(
    times: np.ndarray,
    values: np.ndarray,
    expected_final: float,
    settling_fraction: float = 0.9
) -> float:
    """Calculate steady-state error.

    Args:
        times: Time array
        values: Value array
        expected_final: Expected final value
        settling_fraction: Fraction of time to consider settled (default 90%)

    Returns:
        Relative steady-state error
    """
    # Use last portion of simulation
    settle_idx = int(len(times) * settling_fraction)
    final_value = np.mean(values[settle_idx:])

    if abs(expected_final) > 1e-12:
        return abs(final_value - expected_final) / abs(expected_final)
    return abs(final_value - expected_final)


def calculate_rise_time(
    times: np.ndarray,
    values: np.ndarray,
    initial: float,
    final: float,
    low_pct: float = 0.1,
    high_pct: float = 0.9
) -> float:
    """Calculate rise time (10% to 90% by default).

    Args:
        times: Time array
        values: Value array
        initial: Initial value
        final: Final value
        low_pct: Lower threshold percentage (default 10%)
        high_pct: Upper threshold percentage (default 90%)

    Returns:
        Rise time in same units as times
    """
    delta = final - initial
    low_thresh = initial + low_pct * delta
    high_thresh = initial + high_pct * delta

    # Find crossing points
    low_cross_idx = np.argmax(values >= low_thresh)
    high_cross_idx = np.argmax(values >= high_thresh)

    if low_cross_idx < high_cross_idx:
        return times[high_cross_idx] - times[low_cross_idx]
    return float('nan')
