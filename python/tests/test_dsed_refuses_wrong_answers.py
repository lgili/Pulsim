"""v2.0 Phase 3: engine='dsed' refuses what it cannot yet simulate.

Both refusals below replace a measured SILENT WRONG ANSWER:

* A PWL diode's bit was frozen at whatever ``switch_fn`` returned —
  a reverse-biased series diode conducted BACKWARDS (vout = −10.909 V
  where the pwl engine correctly blocks at −1e-06 V). RESOLVED by
  Phase-3 item 1: diodes are now commutated by auto-derived event
  predicates; the refusal survives only for integrator='bdf2'.
* A Nonlinear device was skipped by the assembler entirely — a 5 V
  source through 1 kΩ into a nonlinear diode charged the cap toward
  5·(1−e^{-t/τ}) as if the diode were absent.

The refusals are structural (a switch census, not an exception
handler), because the extraction SUCCEEDS on the wrong circuit — the
old fallback message claimed the extractor rejects these and it never
did. Diode commutation on dsed is the Phase-3 feature these tests
will be relaxed for, one covered path at a time.
"""

import warnings

import numpy as np
import pytest

import pulsim as p


def _pwl_diode_rectifier():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", -12.0)   # reverse bias
    b.add_resistor("R1", "vin", "na", 10.0)
    b.add_diode("D", "na", "vout", 1e3, 1e-9, 0.7)
    b.add_capacitor("C", "vout", "gnd", 10e-6)
    b.add_resistor("Rl", "vout", "gnd", 100.0)
    return b


def _nonlinear_diode_circuit():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 5.0)
    b.add_resistor("R", "vin", "na", 1000.0)
    b.add_nonlinear_diode("D", "na", "gnd", p.IdealDiodeParams())
    b.add_capacitor("C", "na", "gnd", 1e-6)
    return b


def test_dsed_now_commutates_pwl_diodes():
    """This test REFUSED diode circuits when the stop-gap landed.
    Phase-3 item 1 relaxed it: the reverse-biased rectifier that
    conducted backwards must now block, matching the pwl engine.
    (The full behaviour suite lives in test_dsed_diode_events.py.)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(_pwl_diode_rectifier(), t_end=1e-3,
                        engine="dsed")
    assert abs(np.asarray(r.states)[-1][0]) < 1e-3   # blocks


def test_dsed_refuses_nonlinear_devices():
    with pytest.raises(ValueError, match="nonlinear"):
        p.simulate(_nonlinear_diode_circuit(), t_end=1e-3,
                    engine="dsed")


def test_the_pwl_engine_still_takes_both():
    """The refusal must push users somewhere that works."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = p.simulate(_pwl_diode_rectifier(), t_end=1e-3, dt=1e-7)
        # Reverse-biased: the diode blocks, vout ~ 0 — the answer
        # dsed was getting wrong by 10.9 V.
        assert abs(np.asarray(r1.v("vout"))[-1]) < 1e-3

        r2 = p.simulate(_nonlinear_diode_circuit(), t_end=5e-3,
                         dt=1e-7)
        # Forward-biased smooth diode: one diode drop, not 5 V.
        assert np.asarray(r2.v("na"))[-1] == pytest.approx(0.7, abs=0.1)


def test_a_diode_free_circuit_still_runs_on_dsed():
    """No false positives: the census must not catch controlled
    switches or plain RLC."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 12.0)
    b.add_resistor("R", "vin", "vout", 10.0)
    b.add_capacitor("C", "vout", "gnd", 1e-6)
    r = p.simulate(b, t_end=1e-3, engine="dsed")
    assert len(r.times) > 0
    assert np.isfinite(np.asarray(r.states)).all()
