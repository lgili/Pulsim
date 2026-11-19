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


def test_rac_ladder_yaml_is_the_python_composite():
    # add-realistic-passives 3.2: the ladder's cells in YAML expand into
    # the elements add_inductor(rac=) builds, element for element.
    from pulsim.passives import RacLadder

    lad = RacLadder(R_dc=0.01, L_hf=0.7e-4, R_k=(0.09, 0.2), L_k=(2e-5, 1e-5))
    y = _yaml("    - {type: inductor, name: L, from: a, to: b, L: 1.0e-4, c_par: 1.0e-11,\n"
              "       rac: {R_dc: 0.01, L_hf: 0.7e-4, R: [0.09, 0.2], L: [2.0e-5, 1.0e-5]}}\n")
    b = p.CircuitBuilder()
    b.add_inductor("L", "a", "b", 1e-4, c_par=1e-11, rac=lad)
    assert _components(y) == _components(b)
    # coupling the main inductor alone would leave the cells uncoupled
    y.add_inductor("L2", "b", "gnd", 1e-4)
    with pytest.raises(p.UnsupportedError, match="R_ac ladder"):
        y.add_inductor_coupling("L", "L2", 0.9)


@pytest.mark.parametrize("rac, match", [
    ("{f: [1.0e3, 1.0e5], R_ac: [0.01, 0.05]}", "fit_rac_ladder"),
    ("{R_dc: 0.01, L_hf: 0.7e-4, R: [0.09], L: [2.0e-5]}", "differs from L"),
    ("{R_dc: 0.01, L_hf: 0.7e-4, R: [0.09, -1.0], L: [2.0e-5, 1.0e-5]}", "> 0"),
    ("{R_dc: 0.01, L_hf: 0.7e-4, R: [0.09, 0.2], L: [2.0e-5, 1.0e-5], x: 1}",
     "unknown key 'x'"),
])
def test_rac_ladder_yaml_refusals_name_the_inductor(rac, match):
    with pytest.raises(RuntimeError, match=match):
        _yaml("    - {type: inductor, name: Lw, from: a, to: b, L: 1.0e-4, "
              f"rac: {rac}}}\n")


def test_nonlinear_capacitor_yaml_law_and_dc_bias():
    # add-realistic-passives 4.5. `law:` is the power_mosfet map; `dc_bias:`
    # is an MLCC curve mirrored into an odd table.
    import numpy as np

    y = _yaml("    - {type: nonlinear_capacitor, name: C1, from: a, to: gnd,\n"
              "       law: {law: junction, C0: 2.0e-9, V0: 25.0, m: 0.5}}\n"
              "    - {type: nonlinear_capacitor, name: C2, from: a, to: gnd,\n"
              "       dc_bias: {v: [0.0, 5.0, 10.0, 20.0, 40.0],\n"
              "                 C: [1.0e-5, 8.0e-6, 5.0e-6, 2.0e-6, 6.0e-7]}}\n")
    b = p.CircuitBuilder()
    b.add_nonlinear_capacitor("C1", "a", "gnd", p.ChargeLaw.junction(2e-9, 25.0, 0.5))
    v = [-40.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 40.0]
    c = [6e-7, 2e-6, 5e-6, 8e-6, 1e-5, 8e-6, 5e-6, 2e-6, 6e-7]
    b.add_nonlinear_capacitor("C2", "a", "gnd",
                              p.ChargeLaw.table(p.ChargeTable.from_capacitance(v, c)))
    assert _components(y) == _components(b)
    # a mirrored from_capacitance table is odd and passes through its points
    t = p.ChargeTable.from_capacitance([-20.0, -10.0, 0.0, 10.0, 20.0],
                                       [2e-6, 5e-6, 1e-5, 5e-6, 2e-6])
    for x in (3.0, 15.0, 30.0):
        assert t.charge(-x) == pytest.approx(-t.charge(x), rel=1e-12)
    assert np.isclose(t.capacitance(10.0), 5e-6, rtol=1e-12)


@pytest.mark.parametrize("extra, match", [
    ("", "exactly one"),
    (", law: {law: junction, C0: 1.0e-9, V0: 1.0}, dc_bias: {v: [0.0, 1.0], C: [1.0, 1.0]}",
     "exactly one"),
    (", law: 1.0e-9", "must be a map"),
    (", dc_bias: {v: [1.0, 2.0], C: [1.0e-6, 1.0e-6]}", r"v\[0\] = 0"),
    (", dc_bias: [0.0, 1.0]", "'dc_bias' must be a map"),
    (", dc_bias: {v: [0.0, 1.0], C: [1.0e-6, 1.0e-6], Q: [0.0, 1.0]}", "unknown key 'Q'"),
    (", law: {law: table, v: [0.0, 1.0, 2.0], C: [1.0e-6, -1.0e-6, 1.0e-6]}", "C2"),
])
def test_nonlinear_capacitor_yaml_refusals(extra, match):
    with pytest.raises((RuntimeError, ValueError), match=match):
        _yaml("    - {type: nonlinear_capacitor, name: C2, from: a, to: gnd"
              f"{extra}}}\n")


_IM_KEYS = ("phase_nodes: [ua, ub, uc], neutral_node: nn, omega_node: w, "
            "R_s: 0.5, R_r: 0.4, L_m: 0.06, pole_pairs: 2, J: 0.01, B: 0.001")


def _im_builder():
    b = p.CircuitBuilder()
    b.add_induction_machine_mna("M", "ua", "ub", "uc", "nn", "w", R_s=0.5,
                                L_ls=2e-3, R_r=0.4, L_lr=2e-3, L_m=0.06,
                                pole_pairs=2, J=0.01, B=0.001)
    return b


def test_induction_machine_yaml_is_the_builder_device():
    y = _yaml("    - {type: induction_machine, name: M, " + _IM_KEYS
              + ", L_ls: 2.0e-3, L_lr: 2.0e-3}\n")
    assert _components(y) == _components(_im_builder())
    assert y.pool.induction_machines() == _im_builder().pool.induction_machines()


def test_induction_motor_model_mna_converts_self_inductances():
    y = _yaml("    - {type: induction_motor, model: mna, name: M, " + _IM_KEYS
              + ", L_s: 0.062, L_r: 0.062}\n")
    got = y.pool.induction_machines()[0]["params"]
    want = _im_builder().pool.induction_machines()[0]["params"]
    assert got == pytest.approx(want, rel=1e-12)


def test_induction_motor_unknown_model_is_refused():
    with pytest.raises(Exception, match="model must be 'observer' or 'mna'"):
        _yaml("    - {type: induction_motor, model: fast, name: M, "
              "phase_nodes: [a, b, c], neutral_node: n, R_s: 1, L_s: 0.1, "
              "R_r: 1, L_r: 0.1, L_m: 0.09}\n")


def test_induction_machine_observer_model_is_refused():
    with pytest.raises(Exception, match="induction_machine 'M': model must be 'mna'"):
        _yaml("    - {type: induction_machine, model: observer, name: M, "
              + _IM_KEYS + ", L_ls: 2.0e-3, L_lr: 2.0e-3}\n")


def test_induction_machine_misspelled_key_is_refused():
    with pytest.raises(Exception, match="T_Load"):
        _yaml("    - {type: induction_machine, name: M, " + _IM_KEYS
              + ", L_ls: 2.0e-3, L_lr: 2.0e-3, T_Load: 3}\n")


_PM_KEYS = ("phase_nodes: [ua, ub, uc], neutral_node: nn, omega_node: w, "
            "theta_node: th, R_s: 0.5, pole_pairs: 4, J: 1.0e-3, B: 1.0e-4")


def test_pmsm_yaml_is_the_builder_device():
    y = _yaml("    - {type: pmsm, name: M, " + _PM_KEYS
              + ", L_d: 1.0e-3, L_q: 3.0e-3, psi_pm: 0.05, omega0: 2.0, "
              "theta0: 0.1}\n")
    b = p.CircuitBuilder()
    b.add_pmsm_mna("M", "ua", "ub", "uc", "nn", "w", "th", R_s=0.5, L_d=1e-3,
                   L_q=3e-3, psi_pm=0.05, pole_pairs=4, J=1e-3, B=1e-4,
                   omega0=2.0, theta0=0.1)
    assert _components(y) == _components(b)
    assert y.pool.pmsm_mna_machines() == b.pool.pmsm_mna_machines()


def test_bldc_yaml_is_the_python_composite():
    y = _yaml("    - {type: bldc, name: M, " + _PM_KEYS
              + ", L_s: 1.0e-3, k_e: 0.05, flat_top_deg: 110, "
              "n_harmonics: 15}\n")
    b = p.CircuitBuilder()
    b.add_bldc_mna("M", "ua", "ub", "uc", "nn", "w", "th", R_s=0.5, L_s=1e-3,
                   k_e=0.05, pole_pairs=4, J=1e-3, B=1e-4, flat_top_deg=110.0,
                   n_harmonics=15)
    assert _components(y) == _components(b)
    got, want = y.pool.pmsm_mna_machines(), b.pool.pmsm_mna_machines()
    assert len(got[0]["params"]["harmonics"]) == 8   # odd h <= 15
    assert got == want


@pytest.mark.parametrize("typ,extra,frag", [
    ("pmsm", "L_d: 1.0e-3, L_q: 1.0e-3, psi_pm: 0.05, Psi_pm: 1", "Psi_pm"),
    ("bldc", "L_s: 1.0e-3, k_e: 0.05, flat_top_deg: 180", "flat_top_deg"),
])
def test_pmsm_bldc_bad_keys_are_refused_by_name(typ, extra, frag):
    with pytest.raises(Exception, match=frag):
        _yaml(f"    - {{type: {typ}, name: M, " + _PM_KEYS + ", " + extra + "}\n")
