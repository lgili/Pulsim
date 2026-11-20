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


# ---------------------------------------------------------------------------
# run_ac_sweep / run_mna_sweep (Python entry points): tasks 3.3, 3.6, 4.3
# ---------------------------------------------------------------------------
def _rc_sweep_circuit():
    import math
    b = ps.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 1.0)
    b.add_resistor("R1", "in", "out", 1e3)
    b.add_capacitor("C1", "out", "gnd", 1e-6)
    n = b.pool.state_size(b.graph)
    k = b.pool.branch_var_id_for_source(b.branch_id_of("Vin"), b.graph)

    def excite(t, eps, f):
        out = [0.0] * n
        out[k] = -eps * math.sin(2 * math.pi * f * t)
        return out
    return b, excite


def _ac(should_continue=None, freqs=(100.0, 159.15, 1e3)):
    b, excite = _rc_sweep_circuit()
    return ps.run_ac_sweep(b, freqs=list(freqs), excite_fn=excite,
                           output_idx=b.node_id_of("out"), cycles_per_point=4,
                           settling_cycles=4, verbose=False,
                           should_continue=should_continue)


class _StopAfter:
    def __init__(self, n):
        self.n, self.calls = n, 0

    def __call__(self):
        self.calls += 1
        return self.calls <= self.n


class TestAcSweepCancellation:
    def test_cancel_before_the_first_point(self):
        with pytest.raises(ps.Cancelled) as e:
            _ac(lambda: False)
        assert e.value.where == "run_ac_sweep"
        assert e.value.point_index == 0

    def test_cancel_inside_a_point_raises_that_point(self):
        """The callback says False inside point 1's transient. simulate()
        would return that transient cut short; the sweep must not keep it
        as the point's |H|."""
        first = _StopAfter(10**9)
        _ac(first, freqs=(100.0,))
        per_point = first.calls          # polls for one whole point
        stop = _StopAfter(per_point + 5)
        with pytest.raises(ps.Cancelled) as e:
            _ac(stop)
        assert e.value.point_index == 1

    def test_a_raising_callback_propagates(self):
        class Boom(Exception):
            pass

        def boom():
            raise Boom("from the callback")
        with pytest.raises(Boom):
            _ac(boom)

    def test_a_true_callback_changes_nothing(self):
        """|H| of the RC low-pass against 1/sqrt(1 + (w R C)^2), and the
        same bits with and without an always-True callback."""
        import math
        a, c = _ac(), _ac(lambda: True)
        np.testing.assert_array_equal(a.H, c.H)
        for f, h in zip(a.freqs, a.H):
            w = 2 * math.pi * f * 1e-3
            assert abs(h) == pytest.approx(1 / math.sqrt(1 + w * w), rel=2e-2)


class TestRunMnaSweepCancellation:
    def _kernel(self, should_continue):
        b, _ = _rc_sweep_circuit()
        return ps.run_mna_sweep(b, output_idx=b.node_id_of("out"),
                                freqs=list(np.logspace(1, 4, 20)),
                                v_in_branch_id=b.branch_id_of("Vin"),
                                should_continue=should_continue)

    def test_kernel_path_raises_pulsim_cancelled(self):
        with pytest.raises(ps.Cancelled) as e:
            self._kernel(_StopAfter(4))
        assert e.value.where == "run_mna_sweep"

    def test_kernel_path_true_callback_and_truth(self):
        r = self._kernel(lambda: True)
        w = 2 * np.pi * r.freqs * 1e-3
        np.testing.assert_allclose(np.abs(r.H), 1 / np.sqrt(1 + w * w),
                                   rtol=1e-9)

    def test_impulse_path_raises_instead_of_a_short_trace(self):
        b, _ = _rc_sweep_circuit()
        with pytest.raises(ps.Cancelled) as e:
            ps.run_mna_sweep(b, output_idx=b.node_id_of("out"), dt=1e-6,
                             t_end=2e-3, use_kernel=False,
                             should_continue=_StopAfter(50))
        assert e.value.where == "run_mna_sweep"


def test_settle_strategy_cancelled_mid_transient_returns_no_point():
    """strategy='settle' runs a transient: cancelled inside it, simulate()
    returns the trace so far, and the steady-state check used to judge
    that partial trace (returning it as the operating point when it read
    flat, or a RuntimeError 'too short'). It raises Cancelled now."""
    b = ps.CircuitBuilder()
    b.add_voltage_source("V1", "a", "gnd", 12.0)
    b.add_resistor("R1", "a", "b", 10.0)
    b.add_capacitor("C1", "b", "gnd", 1e-9)
    with pytest.raises(ps.Cancelled) as e:
        ps.compute_dc_op(b, strategy="settle", should_continue=_StopAfter(200))
    assert e.value.where == "compute_dc_op"
