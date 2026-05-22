"""Extra mixed-domain behavioral tests."""

from __future__ import annotations

import pytest

import pulsim as ps


def test_lookup_table_hold_and_nearest_modes() -> None:
    hold = ps.Circuit()
    n_in_h = hold.add_node("in")
    n_out_h = hold.add_node("out")
    hold.add_virtual_component(
        "lookup_table",
        "LUT_HOLD",
        [n_in_h, n_out_h],
        {},
        {"x": "[0, 1, 2]", "y": "[0, 10, 20]", "mode": "hold"},
    )

    x_hold = [0.0] * hold.system_size()
    x_hold[n_in_h] = 1.4
    step_hold = hold.execute_mixed_domain_step(x_hold, 1e-6)
    assert step_hold.channel_values["LUT_HOLD"] == pytest.approx(10.0, rel=0.0, abs=1e-12)

    near = ps.Circuit()
    n_in_n = near.add_node("in")
    n_out_n = near.add_node("out")
    near.add_virtual_component(
        "lookup_table",
        "LUT_NEAR",
        [n_in_n, n_out_n],
        {},
        {"x": "[0, 1, 2]", "y": "[0, 10, 20]", "mode": "nearest"},
    )

    x_near = [0.0] * near.system_size()
    x_near[n_in_n] = 1.6
    step_near = near.execute_mixed_domain_step(x_near, 1e-6)
    assert step_near.channel_values["LUT_NEAR"] == pytest.approx(20.0, rel=0.0, abs=1e-12)


def test_transfer_function_alpha_fallback_behaves_like_filter() -> None:
    circuit = ps.Circuit()
    n_in = circuit.add_node("in")
    n_out = circuit.add_node("out")
    circuit.add_virtual_component(
        "transfer_function",
        "TF_ALPHA",
        [n_in, n_out],
        {"alpha": 0.5},
        {},
    )

    x = [0.0] * circuit.system_size()
    x[n_in] = 1.0
    step_0 = circuit.execute_mixed_domain_step(x, 0.0)
    step_1 = circuit.execute_mixed_domain_step(x, 1e-6)

    assert step_0.channel_values["TF_ALPHA"] == pytest.approx(0.5, rel=0.0, abs=1e-12)
    assert step_1.channel_values["TF_ALPHA"] == pytest.approx(0.75, rel=0.0, abs=1e-12)


def test_state_machine_set_reset_mode() -> None:
    circuit = ps.Circuit()
    n_set = circuit.add_node("set")
    n_reset = circuit.add_node("reset")
    circuit.add_virtual_component(
        "state_machine",
        "SM1",
        [n_set, n_reset],
        {"threshold": 0.5, "high": 1.0, "low": 0.0},
        {"mode": "set_reset"},
    )

    x = [0.0] * circuit.system_size()
    x[n_set] = 1.0
    x[n_reset] = 0.0
    step_set = circuit.execute_mixed_domain_step(x, 0.0)
    assert step_set.channel_values["SM1"] == pytest.approx(1.0, rel=0.0, abs=1e-12)

    x[n_set] = 1.0
    x[n_reset] = 1.0
    step_reset_priority = circuit.execute_mixed_domain_step(x, 1e-6)
    assert step_reset_priority.channel_values["SM1"] == pytest.approx(0.0, rel=0.0, abs=1e-12)

    x[n_set] = 0.0
    x[n_reset] = 0.0
    step_hold = circuit.execute_mixed_domain_step(x, 2e-6)
    assert step_hold.channel_values["SM1"] == pytest.approx(0.0, rel=0.0, abs=1e-12)


def test_signal_mux_and_demux_channels() -> None:
    circuit = ps.Circuit()
    n_a = circuit.add_node("a")
    n_b = circuit.add_node("b")
    n_c = circuit.add_node("c")
    circuit.add_virtual_component("signal_mux", "MUX1", [n_a, n_b, n_c], {"select_index": 2.0}, {})
    circuit.add_virtual_component("signal_demux", "DMX1", [n_a, n_b, n_c], {}, {})

    x = [0.0] * circuit.system_size()
    x[n_a] = 1.0
    x[n_b] = 2.0
    x[n_c] = 3.0
    step = circuit.execute_mixed_domain_step(x, 1e-6)

    assert step.channel_values["MUX1"] == pytest.approx(3.0, rel=0.0, abs=1e-12)
    assert step.channel_values["DMX1"] == pytest.approx(1.0, rel=0.0, abs=1e-12)
