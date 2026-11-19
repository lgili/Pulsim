"""Node domains (add-immutable-circuit-ir group 9).

A node is electrical unless it is declared otherwise
(``builder.node(name, "thermal")``, YAML ``{name, domain}``) or a typed
pin implies it (a PMSM's speed and angle pins are mechanical). The tag
never reaches the solve: it drives the connection check at ``build()``,
the preflight's per-domain findings and the unit a probe reports.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p


def _rc_thermal(tag):
    """A loss current into a two-stage thermal RC ladder (nodes in degC)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Ta", "ta", "gnd", 25.0)
    b.add_current_source("P", "gnd", "tj", 10.0)
    b.add_resistor("Rjc", "tj", "tc", 0.5)
    b.add_capacitor("Cj", "tj", "ta", 1e-3)
    b.add_resistor("Rca", "tc", "ta", 1.0)
    b.add_capacitor("Cc", "tc", "ta", 1e-2)
    if tag:   # after the devices: the node order (and so every index) is the twin's
        for n in ("tj", "tc", "ta"):
            b.node(n, "thermal")
    return b


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_a_tag_changes_no_number(engine):
    tagged, plain = _rc_thermal(True).build(), _rc_thermal(False).build()
    assert tagged.devices() == plain.devices() and tagged.nodes == plain.nodes
    assert tagged.structure_hash() != plain.structure_hash()   # the tag is structure
    ra = p.simulate(tagged, 1e-2, dt=1e-4, engine=engine)
    rb = p.simulate(plain, 1e-2, dt=1e-4, engine=engine)
    np.testing.assert_array_equal(np.asarray(ra.times), np.asarray(rb.times))
    np.testing.assert_array_equal(np.asarray(ra.states), np.asarray(rb.states))


def test_an_untagged_circuit_keeps_its_hashes():
    # declaring a node electrical is the default: no hash moves
    a = _rc_thermal(False)
    b = _rc_thermal(False)
    b.node("tj", "electrical")
    assert a.build().structure_hash() == b.build().structure_hash()
    assert a.build().value_hash() == b.build().value_hash()


def test_domains_and_units_are_reported():
    c = _rc_thermal(True).build()
    assert c.node_domain("tj") == "thermal" and c.unit_of("tj") == "degC"
    assert c.node_domain("gnd") == "electrical"
    plain = _rc_thermal(False).build()
    assert plain.node_domain("tj") == "electrical" and plain.unit_of("tj") == "V"


def test_pmsm_speed_and_angle_are_mechanical():
    b = p.CircuitBuilder()
    for i, n in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(f"V{i}", n, "gnd", 0.0, 12.0, 30.0, -2.094 * i)
    b.add_pmsm_mna("M", "ua", "ub", "uc", "nn", "w", "th", R_s=0.5, L_d=2e-3,
                   L_q=2e-3, psi_pm=0.05, pole_pairs=4, J=1e-3, B=1e-4)
    c = b.build()
    assert (c.node_domain("w"), c.unit_of("w")) == ("mechanical", "rad/s")
    assert (c.node_domain("th"), c.unit_of("th")) == ("mechanical", "rad")
    assert (c.node_domain("ua"), c.unit_of("ua")) == ("electrical", "V")
    # a load on the shaft is a mechanical branch: a resistor there is fine
    b.add_resistor("Bload", "w", "gnd", 100.0)
    b.build()
    # the speed node declared thermal contradicts the typed pin
    b2 = p.CircuitBuilder()
    b2.node("w", "thermal")
    b2.add_pmsm_mna("M", "ua", "ub", "uc", "nn", "w", "th", R_s=0.5, L_d=2e-3,
                    L_q=2e-3, psi_pm=0.05, pole_pairs=4, J=1e-3)
    with pytest.raises(ValueError, match='device "M_a": pin omega .* node "w", which is thermal'):
        b2.build()


def test_a_cross_domain_connection_is_refused_by_name():
    b = _rc_thermal(True)
    b.add_diode("D1", "tj", "gnd", 1e3, 1e-9)        # a semiconductor on a temperature
    with pytest.raises(ValueError, match=r'device "D1": pin from \(electrical\) is '
                                         r'connected to node "tj", which is thermal'):
        b.build()
    b = _rc_thermal(True)
    b.add_voltage_source("Vx", "x", "gnd", 1.0)
    b.add_diode("Dx", "x", "gnd", 1e3, 1e-9)         # x is electrical by the diode
    b.add_resistor("Rx", "x", "tj", 1.0)             # a pair across two domains
    with pytest.raises(ValueError, match=r'device "Rx": pin to .*"tj", which is thermal'):
        b.build()


def test_a_domain_spreads_along_generic_devices():
    # a user's case node behind a helper's junction is thermal, not refused
    b = p.CircuitBuilder()
    p.add_foster_network(b, [p.FosterStage(R_th_K_per_W=0.1, tau_s=1e-3)])
    b.add_resistor("Rcs", "T_j", "T_case", 0.2)
    b.add_capacitor("Ccs", "T_case", "gnd", 1e-2)
    c = b.build()
    assert (c.node_domain("T_case"), c.unit_of("T_case")) == ("thermal", "degC")
    # a two-mass shaft behind a PMSM's speed node is mechanical
    b = p.CircuitBuilder()
    for i, n in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(f"V{i}", n, "gnd", 0.0, 12.0, 30.0, -2.094 * i)
    b.add_pmsm_mna("M", "ua", "ub", "uc", "nn", "w", "th", R_s=0.5, L_d=2e-3,
                   L_q=2e-3, psi_pm=0.05, pole_pairs=4, J=1e-3, B=1e-4)
    b.add_inductor("Kshaft", "w", "w2", 1e-3)
    b.add_capacitor("J2", "w2", "gnd", 1e-3)
    assert b.build().node_domain("w2") == "mechanical"


def test_a_tagged_dc_machine_shaft_changes_no_number():
    def mk(tag):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "a", "gnd", 12.0)
        b.add_dc_machine("DM", "a", "gnd", "w", R_a=1.0, L_a=1e-3, K=0.1, J=1e-3, B=1e-4)
        if tag:
            b.node("w", "mechanical")
        return b.build()
    c = mk(True)
    assert c.node_domain("w") == "mechanical"
    ra, rb = p.simulate(c, 1e-3, dt=1e-6), p.simulate(mk(False), 1e-3, dt=1e-6)
    np.testing.assert_array_equal(np.asarray(ra.states), np.asarray(rb.states))


def test_builder_refusals():
    b = p.CircuitBuilder()
    b.node("t", "thermal")
    with pytest.raises(ValueError, match='already declared thermal, now mechanical'):
        b.node("t", "mechanical")
    b.node("m", "magnetic")   # the permeance domain: a flux node
    with pytest.raises(ValueError, match="already magnetic"):
        b.node("m", "thermal")
    with pytest.raises(ValueError, match="already declared thermal, now magnetic"):
        b.node("t", "magnetic")
    with pytest.raises(ValueError, match="ground is the reference"):
        b.node("gnd", "thermal")
    with pytest.raises(ValueError, match='unknown domain "fluid"'):
        b.node("q", "fluid")


def test_a_floating_thermal_network_is_a_thermal_finding():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("R", "a", "gnd", 1.0)
    for n in ("t1", "t2"):
        b.node(n, "thermal")
    b.add_resistor("Rt", "t1", "t2", 1.0)
    b.add_capacitor("Ct", "t1", "t2", 1e-3)
    _, report, _ = b.build().with_preflight(p.PreflightOptions(auto_regularize=False))
    found = [f for f in report.findings if f.domain == "thermal"]
    assert found and found[0].detail.startswith("thermal-domain: ")
    assert all(f.domain == "electrical" for f in report.findings if f not in found)


_DOC = """
circuit:
  nodes: [a, {name: tj, domain: thermal}]
  devices:
    - {type: voltage_source, name: V, from: a, to: gnd, V: 1.0}
    - {type: resistor, name: R, from: a, to: gnd, R: 1.0}
    - {type: current_source, name: P, from: gnd, to: tj, I: 1.0}
    - {type: resistor, name: Rth, from: tj, to: gnd, R: 1.0}
"""


def test_yaml_domains_load_and_round_trip():
    c = p.load_yaml_string(_DOC).builder.build()
    assert c.node_domain("tj") == "thermal" and c.node_domain("a") == "electrical"
    text = c.to_yaml()
    assert "{name: tj, domain: thermal}" in text.replace('"', "")
    c2 = p.Circuit.from_yaml(text)
    assert c2.node_domain("tj") == "thermal"
    assert (c2.structure_hash(), c2.value_hash()) == (c.structure_hash(), c.value_hash())
    # a derivation keeps the tag
    assert c.with_value("R.R", 2.0).node_domain("tj") == "thermal"


def test_yaml_unknown_domain_is_refused_by_name():
    with pytest.raises(RuntimeError, match="node 'tj': unknown domain 'fluid'"):
        p.load_yaml_string(_DOC.replace("domain: thermal", "domain: fluid"))
    bad = p.load_yaml_string(_DOC).builder.build().to_yaml().replace(
        "domain: thermal", "domain: fluid")
    with pytest.raises(Exception, match='unknown domain "fluid"'):
        p.Circuit.from_yaml(bad)


def test_thermal_helpers_tag_their_nodes():
    b = p.CircuitBuilder()
    p.add_foster_network(b, [p.FosterStage(R_th_K_per_W=0.1, tau_s=1e-3),
                             p.FosterStage(R_th_K_per_W=0.2, tau_s=1e-2)])
    c = b.build()
    assert {c.node_domain(n) for n in ("T_j", "T_amb", "Th_stage_0")} == {"thermal"}
    assert c.nodes == ["T_amb", "Th_stage_0", "T_j"]   # the order of the untagged helper
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b.add_diode("D", "T_j", "gnd", 1e3, 1e-9)
    with pytest.raises(ValueError, match='node "T_j", which is thermal'):
        b.build()


def test_an_infinite_parameter_bound_is_no_bound():
    import pickle
    b = p.CircuitBuilder()
    q = b.param("x", 1.0, "", 0.0, float("inf"))
    assert q.hi is None and q.lo == 0.0
    b.add_resistor("R", "a", "gnd", q)
    c = b.build()
    assert pickle.loads(pickle.dumps(c)).value_hash() == c.value_hash()
    with pytest.raises(ValueError, match="hi must be a number or none"):
        b.param("y", 1.0, "", None, float("nan"))


def _magnetic(diode_on_core=False, sat=False):
    # A winding (gyrator) joins an electrical pair and a magnetic pair by
    # design; the core's nodes are magnetic (MMF, A) from its pins.
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "e", 10.0)
    b.add_winding("W", "e", "gnd", "m1", "gnd", 25)
    if sat:
        b.add_saturable_permeance("Ps", "m1", "m2", area=76e-6, length=72e-3,
                                  mu_r=2000.0, B_sat=0.35)
    else:
        b.add_permeance("Pc", "m1", "m2", 2.6e-6)
    b.add_air_gap("G", "m2", "gnd", 76e-6, 0.5e-3)
    if diode_on_core:
        b.add_diode("D", "m1", "gnd", 1e3, 1e-9)
    return b


@pytest.mark.parametrize("sat", [False, True])
def test_magnetic_nodes_are_the_magnetic_domain(sat):
    c = _magnetic(sat=sat).build()
    for n in ("m1", "m2"):
        assert (c.node_domain(n), c.unit_of(n)) == ("magnetic", "A")
    assert (c.node_domain("e"), c.unit_of("e")) == ("electrical", "V")
    # a declared magnetic node is the same flux node (YAML round trip)
    b = _magnetic(sat=sat)
    b.node("m2", "magnetic")
    c2 = b.build()
    assert c2.structure_hash() == c.structure_hash()
    assert p.Circuit.from_yaml(c.to_yaml()).node_domain("m1") == "magnetic"


def test_a_semiconductor_pin_on_a_magnetic_node_is_refused_by_name():
    with pytest.raises(ValueError, match=r'device "D": pin from \(electrical\) is '
                                         r'connected to node "m1", which is magnetic'):
        _magnetic(diode_on_core=True).build()


def test_the_induction_machine_speed_pin_is_mechanical():
    b = p.CircuitBuilder()
    for i, n in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(f"V{i}", n, "gnd", 0.0, 100.0, 60.0, -2.0944 * i)
    b.add_induction_machine_mna("M", "ua", "ub", "uc", "nn", "w", R_s=0.5, L_ls=2e-3,
                                R_r=0.4, L_lr=2e-3, L_m=60e-3, pole_pairs=2, J=0.01)
    c = b.build()
    assert (c.node_domain("w"), c.unit_of("w")) == ("mechanical", "rad/s")
    assert c.node_domain("ua") == "electrical"
