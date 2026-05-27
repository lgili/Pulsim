"""Regression coverage for kernel-deep cancellation
(add-python-builder-ergonomics tasks 3.1-3.4, 3.6).

Each test maps to a Scenario in the spec's "Cancellation on All
Top-Level Analyses" requirement.
"""
from __future__ import annotations

import numpy as np
import pytest

import pulsim as ps


# ---------------------------------------------------------------------------
# compute_temperature — Python-side checkpoint every ~1% of trace
# ---------------------------------------------------------------------------
class TestComputeTemperature:
    def test_cancel_mid_convolution(self):
        stages = [ps.thermal.FosterStage(R_th_K_per_W=1.0, tau_s=1e-3)]
        t = np.linspace(0, 1.0, 100_000)
        p = np.full_like(t, 50.0)

        calls = [0]

        def stop_after_10():
            calls[0] += 1
            return calls[0] < 10

        with pytest.raises(ps.Cancelled) as excinfo:
            ps.compute_temperature(t, p, stages,
                                    should_continue=stop_after_10)
        assert excinfo.value.where == "compute_temperature"
        assert excinfo.value.point_index is not None
        # Should cancel within ~10 checkpoints * 1000 samples each.
        assert excinfo.value.point_index < 15_000

    def test_no_cancel_when_callback_none(self):
        # Default behaviour (callback=None) preserves existing API.
        stages = [ps.thermal.FosterStage(R_th_K_per_W=1.0, tau_s=1e-3)]
        t = np.linspace(0, 1e-3, 100)
        p = np.full_like(t, 50.0)
        T = ps.compute_temperature(t, p, stages)
        assert T.shape == t.shape
        assert T[-1] > 25.0  # heated up from ambient


# ---------------------------------------------------------------------------
# compute_dc_op — Python wrapper checks before each strategy attempt
# ---------------------------------------------------------------------------
class TestComputeDcOp:
    def test_cancel_at_entry(self):
        b = ps.CircuitBuilder()
        b.add_voltage_source("V1", "a", "gnd", 12.0)
        b.add_resistor("R1", "a", "gnd", 10.0)
        with pytest.raises(ps.Cancelled) as excinfo:
            ps.compute_dc_op(b, should_continue=lambda: False)
        assert excinfo.value.where == "compute_dc_op"

    def test_default_callback_preserves_result(self):
        b = ps.CircuitBuilder()
        b.add_voltage_source("V1", "a", "gnd", 12.0)
        b.add_resistor("R1", "a", "gnd", 10.0)
        x = ps.compute_dc_op(b)
        # Verify the basic operating point: v_a=12, i_V1=-1.2A.
        assert abs(x[b.node_id_of("a")] - 12.0) < 1e-9


# ---------------------------------------------------------------------------
# C++-level cancellation on run_mna_sweep_kernel
# ---------------------------------------------------------------------------
class TestMnaSweepCancellation:
    def test_cancel_between_freq_points(self):
        from pulsim import _pulsim as _k

        b = ps.CircuitBuilder()
        b.add_voltage_source("V1", "a", "gnd", 12.0)
        b.add_resistor("R1", "a", "gnd", 10.0)
        mask = _k.SwitchStateMask(b.graph.num_switches)
        freqs = list(range(1, 51))

        calls = [0]

        def stop_after_5():
            calls[0] += 1
            return calls[0] < 5

        # The C++ binding throws `_CxxCancelled` (RuntimeError
        # subclass); we accept either that or `pulsim.Cancelled`
        # depending on how the Python wrapper handles it.
        with pytest.raises(RuntimeError) as excinfo:
            _k.run_mna_sweep_kernel(b.graph, b.pool, mask, freqs,
                                      0, 0, should_continue=stop_after_5)
        assert "cancelled" in str(excinfo.value).lower()
