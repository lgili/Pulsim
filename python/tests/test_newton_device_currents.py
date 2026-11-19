"""``res.i`` of the Newton kinds against a 1 mOhm sense resistor
(add-unified-device-interface 9.2 / 9.4).

The static kinds (smooth diode, level-1 MOSFET, level-1 IGBT without a
tail) read the kernel's device law at each stored state
(``static_device_readings``, the reading the in-loop recorders take); the
Shockley diode keeps its Python form of the same law; the saturable inductor
is its own state column. Each current, and the energy ∫v·i dt (trapezoid on
the run's own grid: time-weighted), must match the sense resistor's on
``pwl`` and ``trbdf2``. Kinds whose current needs a stored charge the states
do not hold (Lauritzen diode, IGBT tail, nonlinear capacitor) still raise
``NotImplementedError`` naming the sense-resistor recipe.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p

RS = 1e-3


def _base(drive_hz=1e3, amp=10.0):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, amp, drive_hz)
    b.add_resistor("R", "in", "s", 5.0)
    b.add_resistor("Rs", "s", "a", RS)   # the sense resistor
    return b


def _mosfet():
    b = _base()
    b.add_voltage_source("Vg", "g", "gnd", 6.0)
    b.add_mosfet_level1("M", "a", "gnd", "g", K=0.2, V_T=2.0, with_body_diode=False)
    return b, "M", ("a", "gnd")


def _igbt():
    b = _base()
    b.add_voltage_source("Vg", "g", "gnd", 15.0)
    b.add_igbt_level1("Q", "a", "gnd", "g")
    return b, "Q", ("a", "gnd")


def _smooth_diode():
    b = _base()
    b.add_nonlinear_diode("D", "a", "gnd", p.IdealDiodeParams(0.7, 0.05, 1e-6, 20.0),
                          allow_non_passive=True)
    return b, "D", ("a", "gnd")


def _shockley():
    b = _base()
    b.add_shockley_diode("D", "a", "gnd")
    return b, "D", ("a", "gnd")


def _saturable():
    b = _base(amp=1.0)
    b.add_saturable_inductor("L", "a", "gnd", 1e-3, 0.05, 1e-4)
    return b, "L", ("a", "gnd")


CASES = {"mosfet_level1": _mosfet, "igbt_level1": _igbt,
         "nonlinear_diode": _smooth_diode, "shockley_diode": _shockley,
         "saturable_inductor": _saturable}


def _run(make, engine):
    b, name, (na, nb) = make()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if engine == "pwl":
            r = p.simulate(b, 2e-3, dt=1e-6, engine="pwl")
        else:
            r = p.simulate(b, 2e-3, engine="trbdf2")
    return r, name, na, nb, b


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
@pytest.mark.parametrize("kind", list(CASES))
def test_newton_current_and_energy_against_a_sense_resistor(kind, engine):
    r, name, na, nb, b = _run(CASES[kind], engine)
    t = np.asarray(r.times, float)
    i_dev = np.asarray(r.i(name), float)
    i_ref = np.asarray(r.i("Rs"), float)
    scale = float(np.max(np.abs(i_ref)))
    assert scale > 1e-2, "the device must conduct (control)"
    v = np.asarray(r.v(na), float) - (0.0 if nb == "gnd" else np.asarray(r.v(nb), float))
    # the device current at each stored state: KCL through the sense
    # resistor, which reads (v_s − v_a)/1 mΩ (a 1e-12 V difference is 1e-9 A)
    # Row 0 is the zero start, not a solve: the smooth diode's non-passive
    # law reads -1.2e-5 A at v = 0 there while the resistor reads 0.
    np.testing.assert_allclose(i_dev[1:], i_ref[1:], rtol=0, atol=1e-6 * scale)
    e_dev = np.trapezoid(v * i_dev, t)
    e_ref = np.trapezoid(v * i_ref, t)
    e_abs = np.trapezoid(np.abs(v * i_ref), t)
    assert abs(e_dev - e_ref) <= 1e-4 * e_abs, (e_dev, e_ref, e_abs)
    if kind in ("mosfet_level1", "igbt_level1", "nonlinear_diode"):
        # 9.3: the loss and thermal summaries list the static Newton kinds
        e = [d for d in p.device_loss_summary(b, r) if d["name"] == name]
        assert len(e) == 1 and e[0]["kind"] == kind
        # the summary integrates the product of the piecewise-linear v and i
        # (not the trapezoid of v·i): the same quadrature on the resistor
        v0, v1, i0, i1 = v[:-1], v[1:], i_ref[:-1], i_ref[1:]
        e_pl = np.sum(np.diff(t) * (2 * v0 * i0 + 2 * v1 * i1 + v0 * i1 + v1 * i0) / 6)
        assert abs(e[0]["E_total"] - e_pl) <= 1e-4 * e_abs, (e[0], e_pl)
        stages = [p.FosterStage(R_th_K_per_W=0.5, tau_s=1e-4)]
        th = p.device_thermal_summary(
            b, r, thermal_specs={name: {"stages": stages}})
        assert len(th) == 1 and th[0]["kind"] == kind
        P_avg = e_pl / (t[-1] - t[0])
        assert th[0]["P_cond_avg"] == pytest.approx(P_avg, rel=1e-4, abs=1e-4 * e_abs / t[-1])


@pytest.mark.parametrize("add", [
    lambda b: b.add_lauritzen_diode("X", "a", "gnd"),
    lambda b: b.add_nonlinear_capacitor("X", "a", "gnd", 1e-6, 1.0),
    lambda b: (b.add_voltage_source("Vg", "g", "gnd", 15.0),
               b.add_igbt_level1("X", "a", "gnd", "g", tau_tail=1e-6, k_tail=0.3)),
])
def test_kinds_with_a_stored_charge_still_name_the_recipe(add):
    b = _base()
    add(b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 1e-4, dt=1e-6, engine="pwl")
    with pytest.raises(NotImplementedError, match="1 mOhm"):
        r.i("X")
