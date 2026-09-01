"""An IGBT cannot conduct in reverse (audit C.1).

It is a minority-carrier device: there is no channel to run
backwards the way a MOSFET's does, and the collector junction
blocks. The level-1 law did not know that — below the knee it
went negative with the full on-state slope. Measured before the
fix, gate at 15 V, V_CE_sat = 1.5 V, R_CE_sat = 50 mOhm:

    V_CE (V)    I_C (A)
      -10        -230      <- 230 A backwards through a device
       -5        -130         that physically blocks
        0         -30      <- both terminals shorted, still
      1.5           0         sourcing 30 A from nothing
        5         +70      <- the only correct row

The model's own header waved the region off as "in normal
operation V_CE >> V_CE_sat during conduction". But during
freewheeling V_CE is NEGATIVE, and freewheeling an inductive load
is what the low-side device does every switching cycle in every
voltage-source inverter. The current belongs to the anti-parallel
FWD, and the transistor was taking it.
"""

import numpy as np
import pytest

import pulsim as p

V_CE_SAT, R_CE_SAT, V_T = 1.5, 0.05, 5.0


def _ic(vce, vge=15.0, **kw):
    """Collector current at a forced V_CE. The source's own
    current is the negative of the device's."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vc", "c", "gnd", vce)
    b.add_voltage_source("Vg", "g", "gnd", vge)
    b.add_igbt_level1("Q1", "c", "gnd", "g", V_CE_SAT, R_CE_SAT,
                       V_T, **kw)
    res = p.simulate(b, t_end=1e-8, dt=1e-8, engine="pwl")
    return -float(np.asarray(res.i("Vc"))[-1])


def test_it_never_conducts_backwards():
    """The property. Every one of these was negative before."""
    for vce in (-100, -10, -5, -1, 0, 0.5, 1.0, 1.5):
        assert _ic(vce) >= 0.0, (vce, _ic(vce))


def test_shorted_terminals_source_nothing():
    """V_CE = 0 used to give -30 A out of nowhere."""
    assert abs(_ic(0.0)) < 1.0


def test_ordinary_conduction_is_untouched():
    """The clamp must not cost accuracy where the device
    actually works. 3.5 V above the knee, the smooth max is
    within 0.02% of the raw slope."""
    for vce in (5.0, 10.0, 50.0):
        ideal = (vce - V_CE_SAT) / R_CE_SAT
        assert _ic(vce) == pytest.approx(ideal, rel=1e-3)


def test_the_knee_is_monotone_and_smooth():
    vce = np.linspace(-5.0, 10.0, 400)
    i = np.array([_ic(v) for v in vce])
    assert np.all(np.diff(i) >= -1e-9)           # monotone
    # No kink: the second difference stays bounded through the
    # knee, which is what keeps Newton's Jacobian honest.
    assert np.max(np.abs(np.diff(i, 2))) < 5.0


def test_the_gate_still_gates():
    assert _ic(10.0, vge=0.0) < 1e-3             # cutoff
    assert _ic(10.0, vge=15.0) > 100.0           # on


def test_v_knee_must_be_positive():
    """Zero would restore reverse conduction, so it is refused
    by name rather than silently accepted."""
    b = p.CircuitBuilder()
    with pytest.raises(Exception, match="v_knee"):
        b.add_igbt_level1("Q1", "c", "gnd", "g", v_knee=0.0)


def _freewheel(with_fwd):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vg", "g", "gnd", 15.0)
    b.add_inductor("L", "sw", "gnd", 1e-3, i0=100.0)
    b.add_igbt_level1("QL", "sw", "gnd", "g", V_CE_SAT,
                       R_CE_SAT, V_T, with_fwd=with_fwd)
    return p.simulate(b, t_end=1e-6, dt=1e-9, engine="pwl")


def test_the_fwd_carries_the_freewheeling_current():
    """100 A of inductor current, low-side device gated on. The
    co-packaged diode takes it and clamps the node just below
    ground — instead of the transistor running it backwards."""
    res = _freewheel(with_fwd=True)
    v = float(np.asarray(res.v("sw"))[-1])
    i = float(np.asarray(res.i("L"))[-1])
    assert -2.0 < v < 0.0, v
    assert i == pytest.approx(100.0, rel=1e-3)


def test_without_the_fwd_the_solver_says_so():
    """The trade this model makes on purpose. A device that
    correctly refuses reverse current leaves an inductive load
    with NO path, and the voltage-sanity guard names that —
    2e8 V on a 15 V circuit — instead of the silent wrong answer
    the old reverse conduction gave."""
    with pytest.warns(UserWarning, match="conduction path opens"):
        _freewheel(with_fwd=False)
