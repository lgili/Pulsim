"""Phase 7 of `add-converter-templates`: Python builder API tests."""

from __future__ import annotations

import math

import pytest

pulsim = pytest.importorskip("pulsim")


def test_buck_auto_designs_L_and_C():
    exp = pulsim.templates.buck(Vin=24, Vout=5, Iout=2, fsw=100e3)
    assert exp.topology == "buck"
    assert math.isclose(exp.parameters["D"], 5.0 / 24.0, rel_tol=1e-9)
    assert exp.parameters["L"] > 0
    assert exp.parameters["C"] > 0
    assert exp.parameters["Rload"] == pytest.approx(2.5, rel=1e-9)
    assert "L" in exp.notes
    assert "C" in exp.notes


def test_buck_user_overrides_take_precedence():
    exp = pulsim.templates.buck(
        Vin=24, Vout=5, Iout=2, fsw=100e3,
        L=100e-6, C=200e-6, Rload=2.0,
    )
    assert exp.parameters["L"] == pytest.approx(100e-6)
    assert exp.parameters["C"] == pytest.approx(200e-6)
    assert exp.parameters["Rload"] == pytest.approx(2.0)
    # User-supplied parameters don't get auto-design notes.
    assert "L" not in exp.notes
    assert "C" not in exp.notes


def test_boost_validates_Vout_above_Vin():
    with pytest.raises(ValueError):
        pulsim.templates.boost(Vin=24, Vout=12, Iout=1, fsw=100e3)


def test_buck_boost_handles_negative_or_positive_Vout():
    exp_neg = pulsim.templates.buck_boost(
        Vin=12, Vout=-24, Iout=1, fsw=100e3)
    exp_pos = pulsim.templates.buck_boost(
        Vin=12, Vout=24, Iout=1, fsw=100e3)
    # Both should converge to D = 2/3 (since |Vout|/(Vin+|Vout|) = 24/36).
    assert exp_neg.parameters["D"] == pytest.approx(2.0 / 3.0, rel=1e-9)
    assert exp_pos.parameters["D"] == pytest.approx(2.0 / 3.0, rel=1e-9)
    # Output is conventionally negative.
    assert exp_neg.parameters["Vout"] < 0
    assert exp_pos.parameters["Vout"] < 0


def test_buck_circuit_runs_through_simulator():
    """End-to-end: build + simulate. Output is finite and bounded."""
    exp = pulsim.templates.buck(Vin=24, Vout=5, Iout=2, fsw=100e3)

    # Switch to PWL Ideal so segment-primary engine resolves switching.
    exp.circuit.set_switching_mode_for_all(pulsim.SwitchingMode.Ideal)
    exp.circuit.set_pwl_state("Q1", False)
    exp.circuit.set_pwl_state("D1", False)

    opts = pulsim.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 1e-3
    opts.dt = 1e-7
    opts.dt_min = 1e-12
    opts.dt_max = 1e-7
    opts.adaptive_timestep = False
    opts.integrator = pulsim.Integrator.BDF1
    opts.switching_mode = pulsim.SwitchingMode.Ideal
    opts.newton_options.num_nodes = exp.circuit.num_nodes()
    opts.newton_options.num_branches = exp.circuit.num_branches()

    sim = pulsim.Simulator(exp.circuit, opts)
    dc = sim.dc_operating_point()
    assert dc.success
    run = sim.run_transient(dc.newton_result.solution)
    assert run.success
    assert len(run.states) > 0

    out_idx = exp.circuit.get_node("out")
    V_final = run.states[-1][out_idx]
    assert math.isfinite(V_final)
    assert abs(V_final) < 30.0  # bounded by Vin
