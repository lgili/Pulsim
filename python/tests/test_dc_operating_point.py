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


def stiff_diode_chain(n=12, kappa=50.0):
    """A chain sharp enough that Newton from the warm start diverges.
    This is the circuit gmin stepping exists for."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 20.0)
    b.add_resistor("R", "vin", "n0", 100.0)
    for i in range(n):
        q = p.IdealDiodeParams()
        q.kappa = kappa
        to = "gnd" if i == n - 1 else f"n{i+1}"
        b.add_nonlinear_diode(f"D{i}", f"n{i}", to, q)
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
    # 12 forward drops of ~0.70 V.
    assert x[1] == pytest.approx(8.4, abs=0.2)
    assert rep[0].rungs_attempted >= 11
    assert rep[0].residual < 1e-6


def test_two_independent_homotopies_land_in_the_same_place():
    """Convergence alone proves nothing — a homotopy can converge to
    the wrong branch. Two unrelated continuations agreeing is the
    evidence that matters."""
    xg = _dc(stiff_diode_chain(), strategy="gmin_step")
    xs = _dc(stiff_diode_chain(), strategy="source_step")
    np.testing.assert_allclose(xg, xs, rtol=0, atol=1e-6)


def test_auto_falls_through_and_says_which_rung_answered():
    rep = []
    x = _dc(stiff_diode_chain(), strategy="auto", report=rep)
    assert np.isfinite(x).all()
    assert x[1] == pytest.approx(8.4, abs=0.2)
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


def test_a_hopeless_circuit_names_every_rung_it_tried():
    """When nothing works the user gets a map, not a shrug."""
    with pytest.raises(RuntimeError) as exc:
        _dc(stiff_diode_chain(kappa=5000.0), strategy="auto")
    msg = str(exc.value)
    for rung in ("naive", "gmin stepping", "source stepping",
                  "pseudo-transient"):
        assert rung in msg, f"{rung!r} missing from:\n{msg}"


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
