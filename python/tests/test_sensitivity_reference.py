"""add-parameter-sensitivities, group 1: characterisation.

* 1.1 — today's finite-difference gradients, kept as the record of why
  the change exists: on the fixed engine a central difference of
  ``v_C(40 T)`` of the DCM buck with respect to ``L`` is wrong by tens
  of percent to a factor of ten, with either sign, because the diode
  commutation sits on the step grid; on ``trbdf2`` it is right to
  about 1e-4 at the cost of two runs per parameter.
* 1.2 — the finite-difference harness (step selection, Richardson).
* 1.3 — the prototypes as reference implementations: they reproduce the
  table of ``tasks.md`` and their sensitivities equal central
  differences of their own maps.
"""

import warnings

import numpy as np
import pytest

import pulsim as p

from . import _sens_reference as ref


# ---------------------------------------------------------------------
# 1.1 today's finite differences
# ---------------------------------------------------------------------

def _vend(L, engine, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", ref.VIN)
    b.add_switch("S", "vin", "sw", ref.GON, ref.GOFF)
    b.add_diode("D1", "gnd", "sw", ref.GON, ref.GOFF, 1e-6)
    b.add_inductor("L", "sw", "vout", L)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 50.0)
    fn = p.make_pwm_switch_fn(1 / ref.T, 0.3, b.switch_index_of("S"),
                              b.graph.num_switches)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end=40 * ref.T, engine=engine, switch_fn=fn,
                       **kw)
    return float(np.asarray(r.v("vout"))[-1])


def _fd_errors(engine, rels, **kw):
    L0 = 100e-6
    out = []
    for rel in rels:
        d = rel * L0
        fd = (_vend(L0 + d, engine, **kw) - _vend(L0 - d, engine, **kw)) \
            / (2 * d)
        out.append(fd / ref.DCM_TRANSIENT["L"] - 1)
    return np.array(out)


def test_fixed_engine_finite_difference_is_wrong():
    """Measured (proposal): -9.3, -9.5, -12.9, +2.0, -0.46 at relative
    steps 1e-6 .. 1e-2. Pinned loosely: some step is off by more than
    100 %, and the errors change sign with the step."""
    err = _fd_errors("pwl", (1e-5, 1e-4, 1e-3, 1e-2), dt=ref.T / 100)
    assert np.abs(err).max() > 1.0
    assert err.min() < 0 < err.max() or np.abs(err).min() > 0.1


def test_trbdf2_finite_difference_is_close_but_not_exact():
    """Measured: about -1.1e-4 at default tolerances."""
    err = _fd_errors("trbdf2", (1e-4,))
    assert np.abs(err).max() < 1e-2


# ---------------------------------------------------------------------
# 1.2 harness
# ---------------------------------------------------------------------

def test_harness_selects_a_step_and_estimates_its_error():
    d, err, h = ref.central_difference(np.exp, 1.0)
    assert abs(d - np.e) < 1e-9
    assert err < 1e-8
    assert h > 0
    # vector-valued, and a function whose truncation error dominates
    # the large steps: the ladder must not pick the largest one
    d, err, h = ref.central_difference(
        lambda x: np.array([np.sin(x), x ** 5]), 2.0)
    assert np.allclose(d, [np.cos(2.0), 5 * 2.0 ** 4], rtol=1e-9)
    assert h < 2e-2


# ---------------------------------------------------------------------
# 1.3 prototypes as references
# ---------------------------------------------------------------------

def test_reference_rc_closed_form():
    assert ref.RC_DV_DR == pytest.approx(-3.678794e-3, rel=1e-6)
    assert ref.RC_DV_DC == pytest.approx(-3.678794e6, rel=1e-6)


@pytest.mark.parametrize("name", ["L", "R", "D"])
def test_reference_dcm_transient_table_and_fd(name):
    p0 = ref.DCM_P0
    _x, s = ref.dcm_transient(p0, name)
    assert s[1] == pytest.approx(ref.DCM_TRANSIENT[name], rel=1e-7)

    def vc(v):
        return ref.dcm_transient(dict(p0, **{name: v}), name)[0]

    fd, _err, _h = ref.central_difference(vc, p0[name],
                                          rel_steps=(1e-4, 1e-5))
    assert ref.rel_err(s, fd) < 1e-7      # prototype: <= 1.0e-8


def test_reference_dcm_duty_needs_the_gate_edge_jump():
    _x, s = ref.dcm_transient(ref.DCM_P0, "D", jump=False)
    assert np.abs(s).max() == 0.0


@pytest.mark.parametrize("name", ["L", "R", "D"])
def test_reference_dcm_orbit_table_and_fd(name):
    from . import _pss_reference as pss
    p0 = ref.DCM_P0
    xs, g = ref.dcm_orbit_gradient(p0, name)
    assert g[1] == pytest.approx(ref.DCM_ORBIT[name], rel=1e-7)
    h = 1e-5 * p0[name]
    xp = pss.shoot(pss.buck_period, dict(p0, **{name: p0[name] + h}), xs)[0]
    xm = pss.shoot(pss.buck_period, dict(p0, **{name: p0[name] - h}), xs)[0]
    assert ref.rel_err(g, (xp - xm) / (2 * h)) < 1e-7   # prototype 5.5e-9


def test_reference_shockley_forward_equals_central_differences():
    """The capacitor voltage, as in the table (the other unknown sits
    at 1e-9 V behind the blocking diode, where a difference of two
    runs is Newton noise: 1e-13 V / 2e-6)."""
    p0 = ref.SHOCKLEY_P0
    _x, S = ref.shockley_run(p0)
    for j, n in enumerate(ref.SHOCKLEY_NAMES):
        d = 1e-6 * p0[n]
        xp = ref.shockley_run(dict(p0, **{n: p0[n] + d}), sens=False)[0]
        xm = ref.shockley_run(dict(p0, **{n: p0[n] - d}), sens=False)[0]
        fd = (xp[1] - xm[1]) / (2 * d)
        assert abs(S[1, j] - fd) / abs(fd) < 1e-6, n
        # the table's worst values, with a margin for libm differences
        assert abs(S[1, j] - fd) / abs(fd) < 10 * ref.SHOCKLEY_TOL[n], n
