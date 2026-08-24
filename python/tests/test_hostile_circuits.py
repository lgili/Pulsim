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


def _run(builder, t_end=2e-4, dt=1e-6, **kw):
    """Simulate, swallowing the (expected) preflight warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(builder, t_end=t_end, dt=dt, **kw)


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


# The node each hostile circuit's error must name when the user opts
# out. Asserting the bare word "node" would be vacuous — it appears in
# every singular-matrix message the kernel emits.
OPT_OUT_NAMES = {
    "isolated-secondary": ("s1", "s_gnd"),
    "two-islands": ("a1", "a2", "b1", "b2"),
    "floating-rc-ladder": ("x1", "x2", "x3"),
}


@pytest.mark.parametrize("make", BREAKS_IN_TRANSIENT)
def test_opting_out_restores_the_named_error(make, request):
    """auto_regularize=False must give back the Phase-1 diagnostic —
    which names the OFFENDING node — not a bare mask bitstring."""
    with pytest.raises(RuntimeError) as exc:
        _run(make(), auto_regularize=False)
    msg = str(exc.value)
    assert "singular" in msg
    # Phase 1's contribution: the message localises the failure to a
    # node of the actually-floating subnet.
    case = request.node.callspec.id
    assert any(n in msg for n in OPT_OUT_NAMES[case]), msg


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
    # ...and the report reaches the DSED result too. It used to not:
    # that branch returns ~400 lines before the PWL tail that attaches
    # it, so `result._preflight` — which the warning tells every user
    # to read — was a PWL-only attribute.
    assert res._preflight is not None
    assert res._preflight.num_fixed() == 1


def test_dc_floating_block_nested_inside_an_isolated_island():
    """Adversarial-review finding (CRITICAL): a galvanic finding covers
    a whole island but earns it ONE tie, so a DC-floating sub-block
    INSIDE that island is still floating afterwards. The first version
    filtered DC findings against galvanic ones by component
    MEMBERSHIP, so it reported those sub-blocks as fixed and then threw
    the very error the feature exists to remove — with a spurious
    resistor already in the user's circuit.

    A current source is the clean repro: it connects na-nb
    galvanically but contributes no DC conductance at all.
    """
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_resistor("R1", "vin", "gnd", 10.0)
    b.add_current_source("Ibias", "na", "nb", 1e-3)
    b.add_resistor("Rb", "nb", "nc", 100.0)

    report = b.run_preflight()
    # TWO ties: one for the island, one for the DC-floating sub-block.
    assert report.num_fixed() == 2
    kinds = {f.issue for f in report.findings}
    assert p.PreflightIssue.IsolatedSubnet in kinds
    assert p.PreflightIssue.NoDcPathToGround in kinds

    # ...and the circuit actually runs.
    b2 = p.CircuitBuilder()
    b2.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b2.add_resistor("R1", "vin", "gnd", 10.0)
    b2.add_current_source("Ibias", "na", "nb", 1e-3)
    b2.add_resistor("Rb", "nb", "nc", 100.0)
    res = _run(b2, t_end=1e-5)
    assert np.isfinite(np.asarray(res.states)).all()


def test_node_behind_a_nonlinear_device_is_not_mistaken_for_grounded():
    """Adversarial-review finding (HIGH): `conducts_at_dc` claimed
    nonlinear branches "stamp their linearization" and returned True.
    They do not — `dc_assemble` skips them as OPEN CIRCUITS — so a node
    touching only a nonlinear device and a capacitor sailed through
    preflight and then failed at DC."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_nonlinear_diode("D1", "vin", "out",
                          p.IdealDiodeParams())
    b.add_capacitor("Cout", "out", "gnd", 10e-6)

    report = b.run_preflight()
    assert report.num_fixed() == 1
    f = report.findings[0]
    assert f.issue == p.PreflightIssue.NoDcPathToGround
    assert "out" in f.detail


def test_tie_resistance_must_be_positive_and_finite():
    """It becomes a conductance 1/R, so 0 would stamp an infinity and
    turn the solution into NaN without a word."""
    b = isolated_transformer_secondary()
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="tie_resistance"):
            b.run_preflight(p.PreflightOptions(tie_resistance=bad))


def test_preflight_options_accepts_keyword_arguments():
    """The docs (and this file) use PreflightOptions(auto_regularize=
    False); the first binding exposed only a no-arg constructor."""
    o = p.PreflightOptions(auto_regularize=False, tie_resistance=1e7)
    assert o.auto_regularize is False
    assert o.tie_resistance == 1e7
    b = isolated_transformer_secondary()
    n_before = b.graph.num_branches
    report = b.run_preflight(o)
    assert not report.empty()
    assert report.num_fixed() == 0          # reported, not applied
    assert b.graph.num_branches == n_before  # untouched


# ---------------------------------------------------------------------
# Class 2 (Phase 2, B.2) — circuits that are well-posed but that the
# direct solve cannot converge on. B.1 gave every node a reference;
# these need the operating point itself to be walked in.
# ---------------------------------------------------------------------

def blocking_diode_chain(n=10, g_off=1e-9):
    """Ten REVERSE-biased junctions in series. Topologically fine and
    not singular — but each interior node is held by a nanosiemens,
    so its pivot carries almost no significant digits and Newton
    wanders instead of converging."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_resistor("R", "vin", "n0", 1e3)
    for i in range(n):
        q = p.IdealDiodeParams()
        q.G_off = g_off
        anode = "gnd" if i == n - 1 else f"n{i+1}"
        b.add_nonlinear_diode(f"D{i}", anode, f"n{i}", q)
    return b


def mains_rectifier(vpk=170.0, kappa=20.0):
    """A 170 V peak half-wave rectifier into an RC load — about as
    ordinary as a power circuit gets. It could not be simulated at
    all before the logistic overflow fix: past kappa*|v| = 709 the
    sigmoid's AD derivative was inf/inf, one NaN reached the
    Jacobian, and Levenberg-Marquardt failed at every lambda."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, vpk, 60.0)
    q = p.IdealDiodeParams()
    q.kappa = kappa
    b.add_nonlinear_diode("D", "ac", "vout", q)
    b.add_resistor("R", "vout", "gnd", 50.0)
    b.add_capacitor("C", "vout", "gnd", 100e-6)
    return b


def test_a_mains_rectifier_runs_at_all():
    """The circuit every power-electronics course opens with. The
    peak must land one diode drop below the source peak."""
    res = _run(mains_rectifier(), t_end=3.4e-2, dt=1e-5)
    v = np.asarray(res.v("vout"))
    assert np.isfinite(v).all()
    assert v.max() == pytest.approx(169.3, abs=0.5)


@pytest.mark.parametrize("vpk,kappa", [
    (24.0, 40.0), (170.0, 20.0), (170.0, 60.0), (400.0, 100.0),
])
def test_no_sharpness_or_voltage_poisons_the_jacobian(vpk, kappa):
    """The old failure boundary tracked kappa*|v| = 709 — the double
    -precision exp limit — which is a property of how the formula was
    WRITTEN, not of the circuit. Sweep well past it."""
    res = _run(mains_rectifier(vpk, kappa), t_end=1.7e-2, dt=1e-5)
    v = np.asarray(res.v("vout"))
    assert np.isfinite(v).all()
    assert v.max() == pytest.approx(vpk - 0.7, rel=0.02)


def test_blocking_chain_finds_its_operating_point_unaided():
    """The direct solve wanders here. Nothing should be asked of the
    user for that: the cascade clamps every node to ground and
    relaxes the clamp by decades until the answer appears."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError):
            p.compute_dc_op(blocking_diode_chain(), strategy="naive")
        rep = []
        x = p.compute_dc_op(blocking_diode_chain(), report=rep)
    assert np.isfinite(x).all()
    assert x[0] == pytest.approx(10.0, abs=1e-6)
    assert rep[0].strategy != "naive"
    # And the point it returns solves the circuit the user described.
    assert rep[0].residual < 1e-6


# ---------------------------------------------------------------------
# Class 3 (Phase 2, B.4) — circuits that are well-posed and
# well-conditioned but whose STEP is too big. B.1 gave every node a
# reference; B.2 gave the operating point a cascade; this class needs
# the step itself walked in.
# ---------------------------------------------------------------------

def test_a_mains_rectifier_runs_at_a_coarse_dt():
    """dt = 1e-4 on a 60 Hz rectifier is ~170 samples per cycle —
    a perfectly reasonable thing for a user to type. One step of it
    will not converge, and that used to end the run."""
    res = _run(mains_rectifier(), t_end=1.7e-2, dt=1e-4)
    v = np.asarray(res.v("vout"))
    assert np.isfinite(v).all()
    assert v.max() == pytest.approx(169.3, abs=0.5)
    assert len(res.dt_retries) >= 1
    assert max(d.halvings for d in res.dt_retries) == 1


def test_the_retry_does_not_move_a_single_sample():
    """Sub-steps are internal. `times[k]` must stay exactly k·dt or
    an FFT of the result is silently wrong — which is the whole
    reason store_every was made a pure stride in Phase 1."""
    dt = 1e-4
    res = _run(mains_rectifier(), t_end=1.7e-2, dt=dt)
    t = np.asarray(res.times)
    assert len(res.dt_retries) >= 1
    np.testing.assert_allclose(t, np.arange(len(t)) * dt,
                               rtol=0, atol=1e-12)
    spacing = np.diff(t)
    assert np.allclose(spacing, dt, rtol=0, atol=1e-15)


def test_opting_out_restores_the_hard_failure():
    with pytest.raises(RuntimeError, match="converge"):
        _run(mains_rectifier(), t_end=1.7e-2, dt=1e-4,
             max_dt_halvings=0)
