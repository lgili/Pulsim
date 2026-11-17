"""YAML spellings of the Python composites build the same circuit.

The C++ loader expands `dc_machine` and the capacitor/inductor parasitic
keys itself (openspec add-mna-native-machines, add-realistic-passives);
these tests pin that the expansion is the Python one, element for
element, so the two cannot drift apart.
"""
import pytest

import pulsim as p


def _components(b):
    return [(c["kind"], c["name"], c["nodes"], c["params"])
            for c in b.components()]


def _yaml(devices: str):
    return p.load_yaml_string("circuit:\n  devices:\n" + devices).builder


def test_dc_machine_yaml_is_the_python_composite():
    y = _yaml("    - {type: voltage_source, name: V, from: a, to: gnd, V: 10.0}\n"
              "    - {type: dc_machine, name: M, a_pos: a, a_neg: gnd, "
              "omega_node: w, R_a: 0.5, L_a: 1.0e-3, K: 0.1, J: 1.0e-5, "
              "B: 1.0e-4, T_load: 0.2}\n")
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_dc_machine("M", "a", "gnd", "w", R_a=0.5, L_a=1e-3, K=0.1,
                     J=1e-5, B=1e-4, T_load=0.2)
    assert _components(y) == _components(b)
    x = p.compute_dc_op(y, gmin=0.0)
    w_ss = (10.0 - 0.5 * 0.2 / 0.1) / (0.1 + 0.5 * 1e-4 / 0.1)
    assert x[y.node_id_of("w")] == pytest.approx(w_ss, rel=1e-9)


def test_passive_parasitics_yaml_is_the_python_composite():
    y = _yaml("    - {type: capacitor, name: C, from: a, to: gnd, C: 1.0e-6, "
              "esr: 0.05, esl: 1.0e-9}\n"
              "    - {type: inductor, name: L, from: a, to: b, L: 1.0e-4, "
              "esr: 0.1, c_par: 1.0e-11}\n")
    b = p.CircuitBuilder()
    b.add_capacitor("C", "a", "gnd", 1e-6, esr=0.05, esl=1e-9)
    b.add_inductor("L", "a", "b", 1e-4, esr=0.1, c_par=1e-11)
    assert _components(y) == _components(b)


def test_dc_machine_speed_node_needs_no_bleeder():
    # The preflight now counts the ideal transformer primary as a DC
    # path: with B = 0 the speed node is not tied to ground.
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_dc_machine("M", "a", "gnd", "w", R_a=0.5, L_a=1e-3, K=0.1,
                     J=1e-5, T_load=0.2)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x = p.compute_dc_op(b, gmin=0.0)
    assert x[b.node_id_of("w")] == pytest.approx(90.0, rel=1e-9)
