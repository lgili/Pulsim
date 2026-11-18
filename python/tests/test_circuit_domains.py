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
    b.add_resistor("Rx", "x", "tj", 1.0)             # a pair across two domains
    with pytest.raises(ValueError, match=r'device "Rx": pin to .*"tj", which is thermal'):
        b.build()


def test_builder_refusals():
    b = p.CircuitBuilder()
    b.node("t", "thermal")
    with pytest.raises(ValueError, match='already declared thermal, now mechanical'):
        b.node("t", "mechanical")
    with pytest.raises(ValueError, match="magnetic domain is reserved"):
        b.node("m", "magnetic")
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b.add_diode("D", "T_j", "gnd", 1e3, 1e-9)
    with pytest.raises(ValueError, match='node "T_j", which is thermal'):
        b.build()
