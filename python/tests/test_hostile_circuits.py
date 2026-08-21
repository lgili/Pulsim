"""v2.0 Phase 2 gate: hostile circuits must converge with NO manual fix.

The audit set the Phase-2 bar as "a suite of hostile circuits (floating
nodes, isolated secondaries, rectifiers in DCM, bridges with no driver)
converges without any manual intervention; every remaining failure
names the node or device".

This file is that suite. It starts with the class B.1 closes — nodes
with no voltage reference — and is meant to GROW as the rest of Phase 2
lands (gmin stepping, dt-halving, chatter resolution). Each case is a
circuit that, before auto-regularization, died with a singular-matrix
error and could only be fixed by the user knowing to type
`add_resistor("R_iso", ..., 1e9)`.
"""

import warnings

import numpy as np
import pytest

import pulsim as p


def _run(builder, **kw):
    """Simulate, swallowing the (expected) preflight warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(builder, t_end=2e-4, dt=1e-6, **kw)


# ---------------------------------------------------------------------
# The circuits
# ---------------------------------------------------------------------

def isolated_transformer_secondary():
    """A flyback/forward secondary referenced to its own ground. The
    single most common real-world instance of this failure."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_resistor("Rp", "vin", "p1", 0.1)
    b.add_transformer("T1", "p1", "gnd", "s1", "s_gnd", 1e-3, 4e-3, 0.98)
    b.add_resistor("Rs", "s1", "s_gnd", 10.0)
    return b


def capacitor_only_node():
    """A divider tap hanging off nothing but a capacitor: galvanically
    connected, but no DC path, so the operating point is singular."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_resistor("R1", "vin", "gnd", 10.0)
    b.add_capacitor("Cfloat", "vin", "vfloat", 1e-6)
    return b


def two_floating_islands():
    """Two independent unreferenced sub-circuits — one tie each, not
    one per node."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_resistor("R1", "vin", "gnd", 10.0)
    b.add_resistor("Ra", "a1", "a2", 1.0)
    b.add_capacitor("Ca", "a1", "a2", 1e-6)
    b.add_resistor("Rb", "b1", "b2", 2.0)
    return b


def floating_rc_ladder():
    """A longer isolated chain, to prove the pass ties the SUBNET once
    rather than walking node by node."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_resistor("R1", "vin", "gnd", 10.0)
    b.add_resistor("Rx1", "x1", "x2", 1.0)
    b.add_resistor("Rx2", "x2", "x3", 1.0)
    b.add_capacitor("Cx", "x3", "x1", 1e-7)
    return b


HOSTILE = [
    pytest.param(isolated_transformer_secondary, 1, id="isolated-secondary"),
    pytest.param(capacitor_only_node, 1, id="capacitor-only-node"),
    pytest.param(two_floating_islands, 2, id="two-islands"),
    pytest.param(floating_rc_ladder, 1, id="floating-rc-ladder"),
]

# Which of them are singular in a plain TRANSIENT run. A node hanging
# off a capacitor is NOT: at dt > 0 the trap companion stamps a
# conductance, so the matrix has rank. It breaks only where capacitors
# are open — the DC operating point and the dt = 0 static build — so
# its opt-out case is tested separately below rather than lumped in.
BREAKS_IN_TRANSIENT = [
    pytest.param(isolated_transformer_secondary, id="isolated-secondary"),
    pytest.param(two_floating_islands, id="two-islands"),
    pytest.param(floating_rc_ladder, id="floating-rc-ladder"),
]


# ---------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------

@pytest.mark.parametrize("make,expected_ties", HOSTILE)
def test_hostile_circuit_simulates_with_no_manual_intervention(
        make, expected_ties):
    res = _run(make())
    assert res.num_steps() > 0
    states = np.asarray(res.states)
    assert np.isfinite(states).all()
    # And it says what it did.
    assert res._preflight is not None
    assert res._preflight.num_fixed() == expected_ties


@pytest.mark.parametrize("make", BREAKS_IN_TRANSIENT)
def test_opting_out_restores_the_named_error(make):
    """auto_regularize=False must give back the Phase-1 diagnostic —
    which names the node — not a bare mask bitstring."""
    with pytest.raises(RuntimeError) as exc:
        _run(make(), auto_regularize=False)
    msg = str(exc.value)
    assert "singular" in msg
    # Phase 1's contribution: the message localises the failure.
    assert "node" in msg


def test_capacitor_only_node_breaks_at_dc_and_is_fixed_there():
    """The cap-only node is fine in a transient (the companion stamps
    a conductance at dt > 0) and singular where capacitors are open.
    Pin both halves, so nobody 'simplifies' the DC pass away on the
    grounds that the transient works."""
    # Without the tie: the static (dt = 0) build has an empty column.
    b = capacitor_only_node()
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    with pytest.raises(RuntimeError, match="vfloat"):
        cache.build(0.0)

    # With it: solvable, and the pass says which node it was.
    b2 = capacitor_only_node()
    report = b2.run_preflight()
    assert report.num_fixed() == 1
    assert "vfloat" in report.findings[0].detail
    cache2 = p.PwlStateSpaceCache(b2.graph, b2.pool)
    cache2.build(0.0)          # no longer singular
    assert cache2.num_built_segments() >= 1


@pytest.mark.parametrize("make,expected_ties", HOSTILE)
def test_report_names_the_node_and_the_inserted_device(
        make, expected_ties):
    del expected_ties
    b = make()
    report = b.run_preflight()
    assert not report.empty()
    for f in report.findings:
        assert f.was_fixed()
        assert f.inserted_resistance == 1e9
        assert "Pulsim inserted" in f.detail
        assert "R_auto_iso_" in f.detail
        assert f.anchor_node in f.component


def test_preflight_warns_once_and_is_greppable():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p.simulate(isolated_transformer_secondary(), t_end=1e-4, dt=1e-6)
    hits = [w for w in caught if "Pulsim preflight" in str(w.message)]
    assert len(hits) == 1
    msg = str(hits[0].message)
    # Names the node, the fix, and the way to opt out.
    assert "s1" in msg
    assert "auto_regularize=False" in msg
    assert "result._preflight" in msg


def test_auto_tie_matches_the_hand_written_one():
    """The tie must be electrically invisible: the whole reason it is
    1 GΩ and not the 1 µΩ an older tutorial suggested."""
    auto = _run(isolated_transformer_secondary())

    manual = p.CircuitBuilder()
    manual.add_voltage_source("Vin", "vin", "gnd", 12.0)
    manual.add_resistor("Rp", "vin", "p1", 0.1)
    manual.add_transformer("T1", "p1", "gnd", "s1", "s_gnd",
                           1e-3, 4e-3, 0.98)
    manual.add_resistor("Rs", "s1", "s_gnd", 10.0)
    manual.add_resistor("R_iso", "s1", "gnd", 1e9)
    hand = _run(manual)

    np.testing.assert_allclose(np.asarray(auto.states),
                               np.asarray(hand.states),
                               rtol=0, atol=1e-9)


def test_well_posed_circuit_is_untouched_and_unwarned():
    """No false positives, no noise, no inserted devices."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_resistor("R1", "vin", "vout", 1.0)
    b.add_capacitor("C1", "vout", "gnd", 1e-6)
    b.add_inductor("L1", "vout", "gnd", 1e-3)
    n_before = b.graph.num_branches

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = p.simulate(b, t_end=1e-4, dt=1e-6)
    assert [w for w in caught if "preflight" in str(w.message)] == []
    assert b.graph.num_branches == n_before
    assert res._preflight.empty()


def test_preflight_also_runs_for_the_dsed_engine():
    """The pass mutates the BUILDER before the engine dispatch, so it
    is not a PWL-only feature that DSED silently ignores."""
    pytest.importorskip("scipy")
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 5.0)
    b.add_resistor("R", "vin", "vout", 10.0)
    b.add_capacitor("C", "vout", "gnd", 1e-6)
    b.add_resistor("Rfloat", "iso1", "iso2", 1.0)   # unreferenced island

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=1e-4, engine="dsed")
    assert res.num_steps() > 0
    # The tie really was inserted into the builder both engines share.
    assert any(b.graph.branch_name(i).startswith("R_auto_iso_")
               for i in range(b.graph.num_branches))
