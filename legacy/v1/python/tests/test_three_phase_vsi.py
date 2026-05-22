"""Smoke tests for ``Circuit.add_three_phase_vsi``.

Verifies the pybind boundary for the 3-phase 2-level VSI helper added in
Pulsim 0.10.0a5. Structural checks only — the slow per-event PWL
state-space rebuild makes long-transient SPWM magnitude validation
impractical at the unit-test level. Magnitude verification is left to
the benchmark suite.
"""
from __future__ import annotations

import math

import pytest

import pulsim as ps


def _run_short_transient(circuit, tstop=3e-3, dt=25e-6):
    opts = ps.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = tstop
    opts.dt = dt
    opts.dt_min = 1e-9
    opts.dt_max = dt
    opts.adaptive_timestep = False
    opts.enable_bdf_order_control = False
    opts.newton_options.num_nodes = circuit.num_nodes()
    opts.newton_options.num_branches = circuit.num_branches()
    return ps.Simulator(circuit, opts).run_transient()


class TestThreePhaseVsiBindings:
    """Verify the pybind11 surface area for ``add_three_phase_vsi``."""

    def test_params_struct_exposed(self):
        params = ps.ThreePhaseVsiParams()
        params.switching_frequency_hz = 10e3
        params.modulation_index = 0.85
        params.modulation_frequency_hz = 60.0
        params.phase_a_deg = 30.0
        params.positive_sequence = False
        params.mosfet_r_on_ohm = 0.02
        params.mosfet_vth = 2.0
        assert params.switching_frequency_hz == pytest.approx(10e3)
        assert params.modulation_index == pytest.approx(0.85)
        assert params.modulation_frequency_hz == pytest.approx(60.0)
        assert params.phase_a_deg == pytest.approx(30.0)
        assert params.positive_sequence is False
        assert params.mosfet_r_on_ohm == pytest.approx(0.02)
        assert params.mosfet_vth == pytest.approx(2.0)

    def test_add_three_phase_vsi_method_present(self):
        circuit = ps.Circuit()
        assert hasattr(circuit, "add_three_phase_vsi")


class TestThreePhaseVsiComposition:
    """Verify that the helper builds the expected switching topology."""

    def test_composes_12_devices_and_6_internal_nodes(self):
        circuit = ps.Circuit()
        vdc_pos = circuit.add_node("VDC+")
        na, nb, nc = (circuit.add_node(x) for x in ("A", "B", "C"))
        nodes_before = circuit.num_nodes()

        params = ps.ThreePhaseVsiParams()
        params.switching_frequency_hz = 1e3
        params.modulation_index = 0.8
        params.modulation_frequency_hz = 50.0
        circuit.add_three_phase_vsi(
            "VSI", vdc_pos, ps.Circuit.ground(),
            na, nb, nc, params,
        )

        # 6 MOSFETs + 6 PWM gate sources.
        assert circuit.num_devices() == 12
        # 6 internal gate nodes added.
        assert circuit.num_nodes() - nodes_before == 6

    def test_convenience_overload_matches(self):
        circuit = ps.Circuit()
        vdc_pos = circuit.add_node("VDC+")
        na, nb, nc = (circuit.add_node(x) for x in ("A", "B", "C"))
        circuit.add_three_phase_vsi(
            "VSI", vdc_pos, ps.Circuit.ground(),
            na, nb, nc,
            1e3, 0.8, 50.0,
        )
        assert circuit.num_devices() == 12


class TestThreePhaseVsiTransient:
    """End-to-end smoke check: helper composes a runnable inverter."""

    def test_runs_3ms_transient_with_RL_load(self):
        circuit = ps.Circuit()
        vdc_pos = circuit.add_node("VDC+")
        na, nb, nc = (circuit.add_node(x) for x in ("A", "B", "C"))
        circuit.add_voltage_source("Vbus", vdc_pos, ps.Circuit.ground(), 100.0)

        vsi = ps.ThreePhaseVsiParams()
        vsi.switching_frequency_hz = 1e3
        vsi.modulation_index = 0.5
        vsi.modulation_frequency_hz = 50.0
        circuit.add_three_phase_vsi(
            "VSI", vdc_pos, ps.Circuit.ground(),
            na, nb, nc, vsi,
        )

        load = ps.ThreePhaseRLLoadParams()
        load.resistance_per_phase = 10.0
        load.inductance_per_phase = 1e-3
        circuit.add_three_phase_rl_load(
            "Load", na, nb, nc, ps.Circuit.ground(), load,
        )

        result = _run_short_transient(circuit)
        assert result.success
        assert len(result.time) > 10

        # Every sample of every phase voltage must be finite.
        for state in result.states:
            assert math.isfinite(state[na])
            assert math.isfinite(state[nb])
            assert math.isfinite(state[nc])

    def test_drives_pmsm_dynamic_end_to_end(self):
        """Compose the new VSI helper with the dynamic PMSM device."""
        circuit = ps.Circuit()
        vdc_pos = circuit.add_node("VDC+")
        na, nb, nc = (circuit.add_node(x) for x in ("A", "B", "C"))
        circuit.add_voltage_source("Vbus", vdc_pos, ps.Circuit.ground(), 50.0)

        vsi = ps.ThreePhaseVsiParams()
        vsi.switching_frequency_hz = 1e3
        vsi.modulation_index = 0.6
        vsi.modulation_frequency_hz = 50.0
        circuit.add_three_phase_vsi(
            "VSI", vdc_pos, ps.Circuit.ground(),
            na, nb, nc, vsi,
        )

        pmsm = ps.PmsmParams()
        pmsm.Rs = 0.5
        pmsm.Ld = pmsm.Lq = 2e-3
        pmsm.psi_pm = 0.1
        pmsm.pole_pairs = 2
        pmsm.J = 5e-3
        pmsm.b_friction = 1e-3
        # Match electrical sync to source frequency.
        pmsm.omega_init = 2 * math.pi * 50.0 / pmsm.pole_pairs
        circuit.add_pmsm("M1", na, nb, nc, ps.Circuit.ground(), pmsm)

        result = _run_short_transient(circuit, tstop=2e-3)
        assert result.success
        # Motor accessors return finite values after the transient.
        for accessor_name in (
            "pmsm_i_d", "pmsm_i_q", "pmsm_tau_em", "pmsm_omega",
            "pmsm_i_a", "pmsm_i_b", "pmsm_i_c",
        ):
            value = getattr(circuit, accessor_name)("M1")
            assert math.isfinite(value), f"{accessor_name} → {value}"
