"""End-to-end test for `wire_chain_from_yaml` — Phase 2 closure.

Demonstrates that a user can describe the entire IM simulation (topology
in YAML circuit:, chain wiring in YAML chain:) and run it through
`pulsim.simulate` without any hand-coded indices.
"""
from __future__ import annotations

import math

import pytest

pulsim = pytest.importorskip("pulsim")


_IM_E2E_YAML = """
circuit:
  devices:
    - {type: voltage_source, name: Va, from: a, to: gnd, V: 0.0}
    - {type: voltage_source, name: Vb, from: b, to: gnd, V: 0.0}
    - {type: voltage_source, name: Vc, from: c, to: gnd, V: 0.0}
    - type: induction_motor
      name: IM
      phase_nodes: [a, b, c]
      neutral_node: n
      R_s: 2.4565
      L_s: 0.09475
      R_r: 1.22825
      L_r: 0.09475
      L_m: 0.087392
      pole_pairs: 2
    - {type: resistor, name: R_leak, from: n, to: gnd, R: 1.0e6}
"""

_IM_CHAIN_YAML = """
- type: induction_motor
  device: IM
  J: 0.01
  B: 0.05
  T_load: 0.0
  R_s: 2.4565
  L_s: 0.09475
  R_r: 1.22825
  L_r: 0.09475
  L_m: 0.087392
  pole_pairs: 2
  omega_channel: omega
  theta_channel: theta
  torque_channel: torque
"""


def test_yaml_chain_wires_induction_motor_end_to_end():
    """YAML → wire_chain_from_yaml → simulate. Verify the chain is
    actually exercised (mechanical state non-zero by the end of a
    sine-driven sim)."""
    loaded = pulsim.load_yaml_string(_IM_E2E_YAML)
    builder = loaded.builder

    chain = pulsim.wire_chain_from_yaml(loaded, _IM_CHAIN_YAML)
    state_size = builder.pool.state_size(builder.graph)

    # Inject sinusoidal 3-φ voltage at 25 Hz, 200 V phase peak.
    V_AMP = 200.0
    F_HZ = 25.0
    omega = 2.0 * math.pi * F_HZ
    src_a_idx = builder.pool.branch_var_id_for_source(
        builder.branch_id_of("Va"), builder.graph)
    src_b_idx = builder.pool.branch_var_id_for_source(
        builder.branch_id_of("Vb"), builder.graph)
    src_c_idx = builder.pool.branch_var_id_for_source(
        builder.branch_id_of("Vc"), builder.graph)

    chain_obs = chain.make_step_observer(50e-6)
    chain_b_extra = chain.make_b_extra_fn(state_size)

    def combined_b_extra(t):
        out = list(chain_b_extra(t))
        out[src_a_idx] += V_AMP * math.sin(omega * t)
        out[src_b_idx] += V_AMP * math.sin(omega * t - 2*math.pi/3)
        out[src_c_idx] += V_AMP * math.sin(omega * t + 2*math.pi/3)
        return out

    res = pulsim.simulate(
        builder, t_end=0.05, dt=50e-6,
        switch_fn=lambda t: pulsim.SwitchStateMask(0),
        step_observer=chain_obs,
        b_extra_fn=combined_b_extra)
    assert res.num_steps() > 0

    # The IM should have accelerated above 0 by 50 ms under 25 Hz
    # excitation (synchronous mech ω = 2π·25/pp = 78.5 rad/s).
    omega_final = chain.get_channel("omega")
    assert omega_final > 1.0, (
        f"IM didn't accelerate via YAML-wired chain: ω={omega_final:.3f}")
    # Torque channel should also be exercised (non-zero).
    torque_final = chain.get_channel("torque")
    assert abs(torque_final) > 0.01, (
        f"IM torque channel empty: T={torque_final:.3e}")


def test_yaml_chain_rejects_unknown_block_type():
    """An unknown chain block type should error cleanly."""
    loaded = pulsim.load_yaml_string(_IM_E2E_YAML)
    with pytest.raises(KeyError, match="unknown type"):
        pulsim.wire_chain_from_yaml(loaded, [
            {"type": "not_a_real_block_type", "device": "IM"}
        ])


def test_yaml_chain_rejects_missing_device_name():
    """A chain block must reference a device that the loader created."""
    loaded = pulsim.load_yaml_string(_IM_E2E_YAML)
    with pytest.raises(Exception):
        pulsim.wire_chain_from_yaml(loaded, [
            {
                "type": "induction_motor",
                "device": "NONEXISTENT",
                "R_s": 1.0, "L_s": 0.1, "R_r": 1.0,
                "L_r": 0.1, "L_m": 0.05, "pole_pairs": 2,
            }
        ])


def test_yaml_chain_empty_spec_yields_empty_chain():
    """No-op chain spec should produce an empty (valid) chain."""
    loaded = pulsim.load_yaml_string(_IM_E2E_YAML)
    chain = pulsim.wire_chain_from_yaml(loaded, [])
    assert chain.size() == 0


def test_yaml_chain_hysteretic_inductor_e2e():
    """End-to-end test with the JA hysteretic inductor."""
    yaml_topology = """
circuit:
  devices:
    - type: sine_voltage_source
      name: Vs
      from: a
      to: gnd
      v_amplitude: 50.0
      frequency: 50.0
    - {type: resistor, name: Rs, from: a, to: n1, R: 1.0}
    - type: hysteretic_inductor
      name: L_core
      from: n1
      to: gnd
      Ms: 4.0e5
      a: 50.0
      alpha: 5e-5
      c: 0.20
      k: 30.0
      N_turns: 100
      l_m: 0.05
      A_core: 1e-4
"""
    chain_spec = [{
        "type": "hysteretic_inductor",
        "device": "L_core",
        "Ms": 4.0e5,
        "a": 50.0,
        "alpha": 5e-5,
        "c": 0.20,
        "k": 30.0,
        "N_turns": 100,
        "l_m": 0.05,
        "A_core": 1e-4,
        "M_channel": "M",
        "B_channel": "B",
    }]
    loaded = pulsim.load_yaml_string(yaml_topology)
    builder = loaded.builder
    chain = pulsim.wire_chain_from_yaml(loaded, chain_spec)
    state_size = builder.pool.state_size(builder.graph)

    res = pulsim.simulate(
        builder, t_end=0.05, dt=100e-6,
        switch_fn=lambda t: pulsim.SwitchStateMask(0),
        step_observer=chain.make_step_observer(100e-6),
        b_extra_fn=chain.make_b_extra_fn(state_size))
    assert res.num_steps() > 0

    # Magnetisation should build up under 50 V 50 Hz excitation.
    M_final = chain.get_channel("M")
    assert abs(M_final) > 1.0, (
        f"JA core magnetisation didn't build up: M={M_final:.3e}")
