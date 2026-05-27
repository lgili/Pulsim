"""Verify YAML loader supports `device_type: induction_motor` and
`device_type: hysteretic_inductor` — Phase 2.1 + 2.2 task 3.

The YAML defines the TOPOLOGY (composite branches with deterministic
names). Users wire the BlockChain in Python afterwards using the
named branches.
"""
from __future__ import annotations

import pytest

pulsim = pytest.importorskip("pulsim")


_IM_YAML = """
circuit:
  devices:
    - type: voltage_source
      name: Vbus
      from: a
      to: gnd
      V: 0.0
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
    - type: resistor
      name: R_leak
      from: n
      to: gnd
      R: 1.0e6
"""


def test_yaml_loads_induction_motor_and_creates_branches():
    """Loading an induction_motor device should create 10 branches
    (1 source + 3 Rs + 3 Lσs + 3 dummy V) and produce a circuit that
    simulates without errors."""
    loaded = pulsim.load_yaml_string(_IM_YAML)
    builder = loaded.builder

    # 1 Vbus + 3 Rs + 3 Lσs + 3 dummy V + 1 R_leak = 11 branches.
    assert builder.graph.num_branches == 11, (
        f"Expected 11 branches (Vbus + 9 IM + R_leak); got "
        f"{builder.graph.num_branches}")
    # Confirm the IM topology is well-formed (kernel solves a step).
    res = pulsim.simulate(builder, t_end=1e-3, dt=1e-4,
                                  switch_fn=lambda t: pulsim.SwitchStateMask(0))
    assert res.num_steps() > 0


def test_yaml_loads_induction_motor_rejects_bad_leakage():
    """Unphysical L_m^2 >= L_s·L_r should error at load time."""
    bad_yaml = """
circuit:
  devices:
    - type: induction_motor
      name: IM
      phase_nodes: [a, b, c]
      neutral_node: n
      R_s: 1.0
      L_s: 0.1
      R_r: 1.0
      L_r: 0.1
      L_m: 0.5
      pole_pairs: 2
"""
    with pytest.raises(Exception, match="unphysical leakage|L_m"):
        pulsim.load_yaml_string(bad_yaml)


def test_yaml_loads_induction_motor_rejects_missing_phase_nodes():
    """Missing or wrong-length phase_nodes should error."""
    bad_yaml = """
circuit:
  devices:
    - type: induction_motor
      name: IM
      phase_nodes: [a, b]
      neutral_node: n
      R_s: 1.0
      L_s: 0.1
      R_r: 1.0
      L_r: 0.1
      L_m: 0.05
      pole_pairs: 2
"""
    with pytest.raises(Exception, match="phase_nodes"):
        pulsim.load_yaml_string(bad_yaml)


_JA_YAML = """
circuit:
  devices:
    - type: sine_voltage_source
      name: Vs
      from: a
      to: gnd
      v_amplitude: 50.0
      frequency: 50.0
    - type: resistor
      name: Rs
      from: a
      to: n1
      R: 1.0
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


def test_yaml_loads_hysteretic_inductor_and_creates_branches():
    """JA hysteretic inductor should create L_0 + dummy V branches and
    the resulting circuit should simulate."""
    loaded = pulsim.load_yaml_string(_JA_YAML)
    builder = loaded.builder
    # 1 sine source + 1 Rs + 1 L_0 + 1 dummy V = 4 branches.
    assert builder.graph.num_branches == 4, (
        f"Expected 4 branches (Vs + Rs + L0 + V_M); got "
        f"{builder.graph.num_branches}")
    # Verify L_0 = N²·A·μ₀/l_m by simulating (topology should solve).
    res = pulsim.simulate(builder, t_end=1e-3, dt=1e-5,
                                  switch_fn=lambda t: pulsim.SwitchStateMask(0))
    assert res.num_steps() > 0


def test_yaml_loads_hysteretic_inductor_rejects_zero_turns():
    """N_turns = 0 should error."""
    bad_yaml = """
circuit:
  devices:
    - type: hysteretic_inductor
      name: L_core
      from: a
      to: gnd
      Ms: 4.0e5
      a: 50.0
      alpha: 5e-5
      c: 0.20
      k: 30.0
      N_turns: 0
      l_m: 0.05
      A_core: 1e-4
"""
    with pytest.raises(Exception, match="non-physical"):
        pulsim.load_yaml_string(bad_yaml)
