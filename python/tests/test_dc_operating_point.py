"""v2.0 Phase 2 (B.2): the DC operating point tells the truth.

Two audit findings meet here.

`no-gmin-infrastructure` — a stiff operating point had no recovery
path at all: Newton either converged or the run died.

And the one that motivated rebuilding the whole entry point: before
this change `pulsim.compute_dc_op(builder)` answered **5.000 V** for
the anode of a diode fed from 5 V through 1 kΩ. The truth is 0.700 V,
and `simulate(start_from_dc_op=True)` had known it since Phase 0 —
the standalone entry point simply solved a different circuit, with no
warning, because `dc_assemble` skips `BranchKind::Nonlinear` as an
open circuit.
"""

import warnings

import numpy as np
import pytest

import pulsim as p


# ---------------------------------------------------------------------
# Circuits
# ---------------------------------------------------------------------

def diode_divider():
    """5 V through 1 kΩ into one smooth diode. The whole silent-wrong
    -answer regression fits in three components."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 5.0)
    b.add_resistor("R", "vin", "na", 1000.0)
    b.add_nonlinear_diode("D", "na", "gnd", p.IdealDiodeParams())
    return b


def stiff_diode_chain(n=10, g_off=1e-9):
    """A chain of REVERSE-biased junctions: nothing is singular, but
    every interior node is held by a nanosiemens, so its pivot has
    almost no significant digits and Newton wanders. This is the
    textbook case gmin exists for.

    (It used to be a chain of sharp FORWARD-biased diodes. That was
    not stiffness — it was the logistic overflow fixed in
    `numeric/logistic.hpp`; once the NaN was gone the forward chain
    converged directly at every sharpness tried, up to kappa=20000.)
    """
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_resistor("R", "vin", "n0", 1e3)
    for i in range(n):
        q = p.IdealDiodeParams()
        q.G_off = g_off
        anode = "gnd" if i == n - 1 else f"n{i+1}"
        b.add_nonlinear_diode(f"D{i}", anode, f"n{i}", q)
    return b


def _dc(builder, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.compute_dc_op(builder, **kw)


# ---------------------------------------------------------------------
# The regression that motivated the rewrite
# ---------------------------------------------------------------------

def test_compute_dc_op_stamps_nonlinear_devices():
    x = _dc(diode_divider())
    # v(na): 0.700 V across the diode, NOT the 5.0 V you get by
    # treating it as an open circuit.
    assert x[1] == pytest.approx(0.70, abs=0.02)
    assert x[1] < 1.0, "the diode is being solved as an open circuit"


def test_it_agrees_with_the_transient_seed():
    """The two entry points must answer the same question. They
    disagreed by 700% until this change, and nothing said so."""
    dc = _dc(diode_divider())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(diode_divider(), t_end=1e-6, dt=1e-8,
                          start_from_dc_op=True)
    seed = np.asarray(res.states)[0]
    np.testing.assert_allclose(dc, seed, rtol=1e-9, atol=1e-12)


def test_opening_the_diode_is_reachable_but_never_the_default():
    """The old answer is still available — as an explicit request,
    which is the only honest way to offer it."""
    opened = _dc(diode_divider(), enable_nonlinear_refresh=False)
    assert opened[1] == pytest.approx(5.0, abs=1e-6)
    assert _dc(diode_divider())[1] != pytest.approx(5.0, abs=1e-3)


# ---------------------------------------------------------------------
# gmin: the floor
# ---------------------------------------------------------------------

def test_the_gmin_floor_is_electrically_invisible():
    b_off = p.CircuitBuilder()
    b_off.add_voltage_source("V", "vin", "gnd", 12.0)
    b_off.add_resistor("R1", "vin", "vout", 4.0)
    b_off.add_resistor("R2", "vout", "gnd", 6.0)
    b_on = p.CircuitBuilder()
    b_on.add_voltage_source("V", "vin", "gnd", 12.0)
    b_on.add_resistor("R1", "vin", "vout", 4.0)
    b_on.add_resistor("R2", "vout", "gnd", 6.0)

    x_off = _dc(b_off, gmin=0.0)
    x_on = _dc(b_on)                       # default floor, 1e-12 S
    np.testing.assert_allclose(x_on, x_off, rtol=1e-9, atol=1e-12)
    assert x_on[1] == pytest.approx(7.2, abs=1e-6)   # 12·6/10


def test_the_floor_does_not_hide_a_floating_node():
    """A conductance to ground on every node would make an
    unreferenced node solvable and report a confident 0 V for it.
    The structural probe runs first, so the named error still wins —
    the floor conditions the matrix, it does not invent equations."""
    def floating():
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "vin", "gnd", 12.0)
        b.add_resistor("R1", "vin", "gnd", 10.0)
        b.add_capacitor("Cfloat", "vin", "vfloat", 1e-6)
        return b

    # auto_regularize=False so preflight does not repair it first;
    # this is about what the SOLVER does when handed the defect.
    for gmin in (0.0, 1e-12, 1e-6):
        with pytest.raises(RuntimeError, match="vfloat"):
            _dc(floating(), gmin=gmin, auto_regularize=False)


# ---------------------------------------------------------------------
# gmin: the ramp
# ---------------------------------------------------------------------

def test_gmin_stepping_solves_what_the_direct_solve_cannot():
    with pytest.raises(RuntimeError):
        _dc(stiff_diode_chain(), strategy="naive")

    rep = []
    x = _dc(stiff_diode_chain(), strategy="gmin_step", report=rep)
    assert np.isfinite(x).all()
    assert x[0] == pytest.approx(10.0, abs=1e-6)   # the source rail
    assert rep[0].rungs_attempted >= 11
    # The property that matters, and the only one worth asserting:
    # the point it returns solves the ORIGINAL circuit, not the
    # regularized one it walked through.
    assert rep[0].residual < 1e-6


def test_two_independent_homotopies_land_in_the_same_place():
    """Convergence alone proves nothing — a homotopy can converge to
    the wrong branch. Two unrelated continuations agreeing is the
    evidence that matters, on a circuit whose answer is unique."""
    xg = _dc(diode_divider(), strategy="gmin_step")
    xs = _dc(diode_divider(), strategy="source_step")
    np.testing.assert_allclose(xg, xs, rtol=0, atol=1e-6)


def test_a_multi_valued_circuit_is_allowed_to_disagree():
    """The reverse-biased chain is NOT such a circuit, and pretending
    otherwise would assert something false about the model:
    `IdealDiode`'s smooth blend is non-monotone in reverse
    (i_on = alpha*delta/R_d stays slightly negative while i_off
    grows), so the chain has more than one operating point. Both
    routes return one, each satisfying the original equations."""
    rg, rs = [], []
    _dc(stiff_diode_chain(), strategy="gmin_step", report=rg)
    _dc(stiff_diode_chain(), strategy="source_step", report=rs)
    assert rg[0].residual < 1e-6
    assert rs[0].residual < 1e-6


def test_auto_falls_through_and_says_which_rung_answered():
    rep = []
    x = _dc(stiff_diode_chain(), strategy="auto", report=rep)
    assert np.isfinite(x).all()
    assert x[0] == pytest.approx(10.0, abs=1e-6)
    # Rung 1 fails on this circuit, so the report must not claim it.
    assert rep[0].strategy != "naive"
    assert "solved by" in rep[0].summary()


def test_the_easy_case_never_leaves_rung_one():
    """Robustness must not cost the common case anything."""
    rep = []
    _dc(diode_divider(), strategy="auto", report=rep)
    assert rep[0].strategy == "naive"
    assert rep[0].rungs_attempted == 1


def test_every_rung_agrees_on_an_easy_circuit():
    ref = _dc(diode_divider(), strategy="naive")
    for strategy in ("gmin_step", "source_step", "auto"):
        x = _dc(diode_divider(), strategy=strategy)
        np.testing.assert_allclose(
            x, ref, rtol=0, atol=1e-6,
            err_msg=f"strategy={strategy} disagrees with naive")


def test_unknown_strategy_is_rejected_with_the_valid_list():
    with pytest.raises(ValueError, match="gmin_step"):
        p.compute_dc_op(diode_divider(), strategy="gmin-stepping")


# ---------------------------------------------------------------------
# Consistency with simulate()
# ---------------------------------------------------------------------

def test_compute_dc_op_runs_the_same_preflight_simulate_does():
    """The two entry points must not disagree about what "floating"
    means. Both repair an unreferenced subnet and both say so."""
    def isolated_secondary():
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", 12.0)
        b.add_resistor("Rp", "vin", "p1", 0.1)
        b.add_transformer("T1", "p1", "gnd", "s1", "s_gnd",
                           1e-3, 4e-3, 0.98)
        b.add_resistor("Rs", "s1", "s_gnd", 10.0)
        return b

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = p.compute_dc_op(isolated_secondary())
    hits = [w for w in caught if "preflight" in str(w.message)]
    assert len(hits) == 1
    assert "s1" in str(hits[0].message)
    assert np.isfinite(x).all()

    # Opting out restores the named error, exactly as in simulate().
    with pytest.raises(RuntimeError, match="s1|s_gnd"):
        p.compute_dc_op(isolated_secondary(), auto_regularize=False)


def test_bdf1_refuses_a_circuit_it_would_silently_open():
    """The BDF1 driver has no Newton loop, so a nonlinear device
    would be an open circuit for the whole run — not just at DC.

    Reached through the kernel binding: `simulate()` has no name for
    this driver, which is why the hole went unnoticed."""
    from pulsim import _pulsim as k

    b = diode_divider()
    opts = k.SimulationOptions(t_start=0.0, t_end=1e-5, dt=1e-6)

    def sw(_t):
        return k.SwitchStateMask(b.graph.num_switches)

    with pytest.raises((ValueError, RuntimeError),
                        match="Newton|trapezoidal"):
        k.run_transient_bdf1(b, opts, sw)


def test_settle_is_reachable_and_agrees_where_a_fixed_point_exists():
    """`"settle"` runs an actual transient, so it is the only strategy
    that can answer for a switching steady state. On a circuit that
    HAS a fixed point it must still land on it — otherwise the escape
    hatch the failure message points at is broken."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 12.0)
    b.add_resistor("R1", "vin", "vout", 4.0)
    b.add_resistor("R2", "vout", "gnd", 6.0)
    b.add_capacitor("C", "vout", "gnd", 1e-6)

    cfg = p.SettleConfig(t_settle=2e-3, dt=1e-6, tol_steady=1e-3,
                          t_check=1e-4)
    settled = _dc(b, strategy="settle", pseudo_trans=cfg)
    assert settled[1] == pytest.approx(7.2, abs=1e-4)   # 12·6/10
    np.testing.assert_allclose(settled, _dc(b), rtol=0, atol=1e-4)


def test_cancellation_reaches_inside_the_cascade():
    """`should_continue` has to be honoured DURING the walk, not only
    before it starts. A cascade can spend seconds on a hostile
    circuit, and a Cancel button that only responds beforehand is not
    a Cancel button."""
    calls = {"n": 0}

    def keep_going():
        calls["n"] += 1
        return calls["n"] < 2       # stop on the second check

    with pytest.raises(p.Cancelled) as exc:
        _dc(stiff_diode_chain(), strategy="auto",
            should_continue=keep_going)
    assert calls["n"] >= 2
    # The typed exception must survive the cascade's error wrapping —
    # a user pressing Cancel is not a convergence failure.
    assert "compute_dc_op" in str(exc.value)


def test_cancellation_is_not_reported_as_a_convergence_failure():
    with pytest.raises(p.Cancelled):
        _dc(diode_divider(), strategy="auto",
            should_continue=lambda: False)


# ---------------------------------------------------------------------
# Regression guards for the adversarial review of this change
# ---------------------------------------------------------------------

def coupled_floating_subnet():
    """GMIN-COVERS-DC-FLOATING-SUBNET. `na`/`nb` reach ground only
    through a capacitor, so the DC block is rank-deficient — but the
    subnet is galvanically connected AND every column is populated by
    its own resistor, so neither a reachability probe over all
    branches nor an emptiness probe can see it."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_resistor("R1", "vin", "gnd", 10.0)
    b.add_capacitor("Cc", "vin", "na", 1e-6)
    b.add_resistor("Rload", "na", "nb", 1000.0)
    return b


def current_source_island():
    """The same hole reached the other way: an ideal current source
    is an edge in the graph but contributes no conductance, so the
    nodes behind it have no DC equation."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_resistor("Rk", "vin", "gnd", 10.0)
    b.add_current_source("I1", "gnd", "n1", 1.0)
    b.add_resistor("R1", "n1", "n2", 1000.0)
    b.add_capacitor("C1", "n2", "gnd", 1e-6)
    return b


@pytest.mark.parametrize("make,node", [
    (coupled_floating_subnet, "na"),
    (current_source_island, "n1"),
])
def test_the_floor_does_not_supply_missing_rank(make, node):
    """A conductance floor made these factorizable and the solver
    reported 0 V — or I/(2·gmin) volts — as an operating point. The
    structural check has to see rank, not just emptiness."""
    with pytest.raises(RuntimeError) as exc:
        _dc(make(), auto_regularize=False)
    msg = str(exc.value)
    assert node in msg, msg
    assert "no DC path to ground" in msg, msg

    # And with the preflight on it is repaired rather than refused.
    x = _dc(make())
    assert np.isfinite(x).all()


def test_named_rungs_resolve_pwl_diode_states_too():
    """PY-RUNGS-SKIP-DIODE-ITERATION. Iterating a PWL diode's on/off
    bit is part of what "the DC operating point" MEANS, not a feature
    of one rung — so every strategy must do it, or the same function
    answers a different circuit depending on the name you passed."""
    def rectified():
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "vin", "gnd", 5.0)
        b.add_resistor("R", "vin", "na", 1000.0)
        b.add_diode("D", "na", "gnd", 1.0, 1e-9)   # PWL switch diode
        return b

    ref = _dc(rectified(), strategy="naive")
    # The diode conducts: v(na) is a few mV, not the 5 V you get with
    # it frozen open.
    assert ref[1] < 0.1, ref
    for strategy in ("gmin_step", "source_step", "auto"):
        x = _dc(rectified(), strategy=strategy)
        np.testing.assert_allclose(
            x, ref, rtol=0, atol=1e-6,
            err_msg=f"strategy={strategy} froze the diode open")


def test_the_report_measures_the_residual_it_advertises():
    """REPORT-RESIDUAL-FABRICATED-ON-NAIVE. The default path never
    measured it and reported a default-constructed 0 — worse than
    reporting nothing, for the one field that can reveal a
    load-bearing floor."""
    rep = []
    _dc(diode_divider(), strategy="auto", report=rep)
    assert rep[0].strategy == "naive"
    # A real measurement of a converged Newton: small but not the
    # tell-tale exact zero of an unset field.
    assert 0.0 <= rep[0].residual < 1e-6
    assert rep[0].final_gmin == pytest.approx(1e-12)


def test_the_dc_vector_is_writable_and_owned():
    """DC-OP-RETURNS-READONLY — the same leak Phase 1's review caught
    in res.v()/res.i(). An operating point is a plain vector users
    subtract and scale."""
    from pulsim import _pulsim as k

    b = diode_divider()
    m = k.SwitchStateMask(b.graph.num_switches)
    op = k.compute_dc_operating_point(b.graph, b.pool, m)

    x = np.asarray(op.x)
    assert x.flags.writeable
    x -= x.mean()                    # must not raise
    # And it was a COPY, not a view onto the kernel object: mutating
    # it must not be visible through a second read.
    np.testing.assert_allclose(np.asarray(op.x),
                               _dc(diode_divider()), rtol=0, atol=1e-9)

    y = _dc(diode_divider())
    assert y.flags.writeable
    y *= 2.0


def test_settle_refuses_what_it_cannot_honour():
    """SETTLE-IGNORES-KWARGS. Silently dropping `gmin=0` or a `mask`
    would leave the caller believing something else was computed."""
    with pytest.raises(ValueError, match="gmin"):
        p.compute_dc_op(diode_divider(), strategy="settle", gmin=0.0)
    with pytest.raises(ValueError, match="max_event_iterations"):
        p.compute_dc_op(diode_divider(), strategy="settle",
                         max_event_iterations=4)


def test_bdf1_refuses_pwl_diodes_as_well():
    """BDF1-GUARD-MISSES-PWL-DIODES. A PWL diode is a Switch branch,
    so `has_nonlinear_devices()` never saw it — and BDF1 has no event
    iteration, so a rectifier would run with no rectification."""
    from pulsim import _pulsim as k

    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 5.0)
    b.add_resistor("R", "vin", "na", 1000.0)
    b.add_diode("D", "na", "gnd", 1.0, 1e-9)
    opts = k.SimulationOptions(t_start=0.0, t_end=1e-5, dt=1e-6)

    with pytest.raises((ValueError, RuntimeError),
                        match="PWL diode|frozen"):
        k.run_transient_bdf1(b, opts,
                              lambda _t: k.SwitchStateMask(
                                  b.graph.num_switches))


def test_a_settle_config_is_never_silently_dropped():
    """PSEUDO-TRANS-CONFIG-DROPPED. `"pseudo_trans"` changed meaning
    in v2.0, so passing the settling config alongside it configures
    nothing. Say so rather than discarding it."""
    cfg = p.SettleConfig(t_settle=1e-3)
    with pytest.raises(ValueError, match="strategy='settle'"):
        p.compute_dc_op(diode_divider(), strategy="pseudo_trans",
                         settle=cfg)
    with pytest.raises(ValueError, match="not both"):
        p.compute_dc_op(diode_divider(), strategy="settle",
                         settle=cfg, pseudo_trans=cfg)
    # The old keyword still works under the new strategy name.
    rc = p.CircuitBuilder()
    rc.add_voltage_source("V", "vin", "gnd", 12.0)
    rc.add_resistor("R1", "vin", "vout", 4.0)
    rc.add_resistor("R2", "vout", "gnd", 6.0)
    rc.add_capacitor("C", "vout", "gnd", 1e-6)
    x = _dc(rc, strategy="settle",
            pseudo_trans=p.SettleConfig(t_settle=2e-3, dt=1e-6,
                                         t_check=1e-4))
    assert x[1] == pytest.approx(7.2, abs=1e-4)
