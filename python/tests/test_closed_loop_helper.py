"""Regression coverage for ``add-python-closed-loop-helper`` (v1.5).

Each test maps to a Scenario block in
``openspec/changes/add-python-closed-loop-helper/specs/python-bindings/spec.md``.
"""
from __future__ import annotations

import numpy as np
import pytest

import pulsim as ps


# ---------------------------------------------------------------------------
# Buck factory — reused across tests
# ---------------------------------------------------------------------------
def _build_buck(*, vin: float = 12.0, R_L: float = 8.0):
    b = ps.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", vin)
    b.add_mosfet_with_body_diode(
        "Q1", "vin", "sw", R_on=1e-3, R_off=1e9, V_F=0.7,
    )
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "vout", 220e-6)
    b.add_capacitor("Cout", "vout", "gnd", 220e-6)
    b.add_resistor("R_L", "vout", "gnd", R_L)
    return b


# ---------------------------------------------------------------------------
# Requirement: PI-PWM Closed-Loop Binding Helper
# ---------------------------------------------------------------------------
class TestBindPiToSwitch:
    def test_buck_converges_to_setpoint(self):
        # Spec scenario: "Wrap a buck closed-loop in one call" —
        # V_in=12, V_ref=5 → ideal duty ≈ 0.4167.
        b = _build_buck()
        pi = ps.PIController(
            Kp=0.08, Ki=40.0, output_min=0.05, output_max=0.95,
        )
        loop = ps.bind_pi_to_switch(
            b, pi=pi,
            measured=lambda x: x[b.node_id_of("vout")],
            setpoint=5.0,
            switch="Q1",
            freq=10e3,
        )
        res = ps.simulate(
            b, t_end=20e-3, dt=2e-6,
            switch_fn=loop.switch_fn,
            step_observer=loop.step_observer,
        )
        # Steady-state within ±5 % of 5 V.
        v_out_ss = np.mean(res.v("vout")[-1000:])
        assert abs(v_out_ss - 5.0) / 5.0 < 0.05, (
            f"V_out steady-state {v_out_ss:.3f} V off target 5 V")
        # Duty trace lands near 5/12 ≈ 0.4167 (±10 %).
        final_duty = loop.duty_history[-1][1]
        assert 0.375 < final_duty < 0.46, (
            f"final duty {final_duty:.4f} off 0.4167")

    def test_histories_track_per_cycle_updates(self):
        # Spec scenario: "Histories track per-cycle updates" — 10 ms
        # × 10 kHz = 100 PWM cycles.
        b = _build_buck()
        pi = ps.PIController(Kp=0.08, Ki=40.0)
        loop = ps.bind_pi_to_switch(
            b, pi=pi,
            measured=lambda x: x[b.node_id_of("vout")],
            setpoint=5.0,
            switch="Q1",
            freq=10e3,
        )
        res = ps.simulate(
            b, t_end=10e-3, dt=2e-6,
            switch_fn=loop.switch_fn,
            step_observer=loop.step_observer,
        )
        assert 99 <= len(loop.duty_history) <= 101, (
            f"expected ~100 PWM ticks, got {len(loop.duty_history)}")
        # Timestamps monotonically non-decreasing.
        ts = [t for t, _ in loop.duty_history]
        assert ts == sorted(ts)
        # Span covers at least 9.5 ms.
        assert ts[-1] - ts[0] >= 9.5e-3
        # Error history is the same length.
        assert len(loop.error_history) == len(loop.duty_history)
        _ = res  # silence unused warning

    def test_switch_by_int_index(self):
        # bind_pi_to_switch accepts an integer index too — fallback
        # for callers that haven't migrated to named lookups.
        b = _build_buck()
        pi = ps.PIController(Kp=0.08, Ki=40.0)
        loop = ps.bind_pi_to_switch(
            b, pi=pi,
            measured=lambda x: x[b.node_id_of("vout")],
            setpoint=5.0,
            switch=b.switch_index_of("Q1"),  # int, not str
            freq=10e3,
        )
        assert callable(loop.switch_fn)
        # Quick sanity: invoke the switch_fn at t=0 to confirm it
        # returns a valid mask (no crash).
        mask = loop.switch_fn(0.0)
        assert mask is not None

    def test_freq_validation(self):
        b = _build_buck()
        pi = ps.PIController(Kp=0.08, Ki=40.0)
        with pytest.raises(ValueError, match="freq must be"):
            ps.bind_pi_to_switch(
                b, pi=pi,
                measured=lambda x: 0.0,
                setpoint=5.0, switch="Q1", freq=0.0,
            )

    def test_unknown_switch_raises(self):
        b = _build_buck()
        pi = ps.PIController(Kp=0.08, Ki=40.0)
        with pytest.raises(IndexError):
            ps.bind_pi_to_switch(
                b, pi=pi,
                measured=lambda x: 0.0,
                setpoint=5.0, switch="Q_unknown", freq=10e3,
            )


# ---------------------------------------------------------------------------
# Requirement: closed_loops= composition + conflict detection
# ---------------------------------------------------------------------------
class TestComposition:
    def test_composes_closed_loops_with_extra_switch_fn(self):
        # v1.6.5 (T1.1 fix): closed_loops + an extra user switch_fn
        # must compose via mask-OR so a closed-loop stage can run
        # alongside another openly-switched stage in the same call.
        # Pre-fix this raised ``ValueError("pass closed_loops OR
        # switch_fn/step_observer, not both")``. See
        # `test_closed_loops_compose_switch_fn.py` for the full
        # composition contract; this case is the spec scenario
        # "Composing closed_loops with an extra callback".
        b = _build_buck()
        pi = ps.PIController(Kp=0.08, Ki=40.0)
        loop = ps.bind_pi_to_switch(
            b, pi=pi,
            measured=lambda x: x[b.node_id_of("vout")],
            setpoint=5.0, switch="Q1", freq=10e3,
        )

        def custom_fn(_t):
            # Empty mask contribution — the loop still owns Q1; the
            # user contributes nothing extra. The point is that the
            # call itself must SUCCEED rather than rejecting.
            return ps.SwitchStateMask(b.graph.num_switches)

        res = ps.simulate(
            b, t_end=1e-3, dt=1e-6,
            closed_loops=[loop],
            switch_fn=custom_fn,
        )
        assert res.num_steps() > 0

    def test_closed_loops_single(self):
        # Single-loop composition path should give the same result as
        # the explicit switch_fn + step_observer pair.
        b = _build_buck()
        pi = ps.PIController(
            Kp=0.08, Ki=40.0, output_min=0.05, output_max=0.95,
        )
        loop = ps.bind_pi_to_switch(
            b, pi=pi,
            measured=lambda x: x[b.node_id_of("vout")],
            setpoint=5.0, switch="Q1", freq=10e3,
        )
        res = ps.simulate(
            b, t_end=20e-3, dt=2e-6,
            closed_loops=[loop],
        )
        v_out_ss = np.mean(res.v("vout")[-1000:])
        assert abs(v_out_ss - 5.0) / 5.0 < 0.05


# ---------------------------------------------------------------------------
# Requirement: bind_pi_to_duty_callable
# ---------------------------------------------------------------------------
class TestDutyCallable:
    def test_duty_get_evolves(self):
        # bind_pi_to_duty_callable returns the duty as a 0-arg getter
        # that callers can compose into custom switch_fns.
        b = _build_buck()
        pi = ps.PIController(
            Kp=0.08, Ki=40.0, output_min=0.05, output_max=0.95,
        )
        duty_get, observer, history = ps.bind_pi_to_duty_callable(
            b, pi=pi,
            measured=lambda x: x[b.node_id_of("vout")],
            setpoint=5.0, freq=10e3,
        )

        T_PWM = 100e-6
        Q1_idx = b.switch_index_of("Q1")
        n = b.graph.num_switches

        def switch_fn(t):
            mask = ps.SwitchStateMask(n)
            phase = (t % T_PWM) / T_PWM
            if phase < duty_get():
                mask.set(Q1_idx, True)
            return mask

        res = ps.simulate(
            b, t_end=20e-3, dt=2e-6,
            switch_fn=switch_fn,
            step_observer=observer,
        )
        v_out_ss = np.mean(res.v("vout")[-1000:])
        assert abs(v_out_ss - 5.0) / 5.0 < 0.05
        assert len(history) >= 100
        # Final duty close to ideal.
        assert 0.35 < history[-1][1] < 0.5
