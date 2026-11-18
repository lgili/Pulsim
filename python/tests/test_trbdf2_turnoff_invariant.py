"""TR-BDF2: no committed sample turns a diode OFF while it carries reverse
current beyond the run's tolerance.

An ON diode turned OFF past its current zero leaves the overshoot in the
inductor with only g_off to go through (v = i/g_off): the boost of the
shooting suite turned off at -0.705 mA (sw at -3.5e5 V, 9577 floor-accepted
steps) and the DCM buck at -9e-7 A (sw at 446 V on a 24 V circuit).
Measured after the fix: boost 0 A, DCM buck -1.2e-16 A, CCM buck 0 A.
"""
import numpy as np
import pytest

import pulsim as p

from . import test_transient_shooting as ts


def _bridge():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 325.0, 50.0)
    b.add_resistor("Rs", "s", "a", 0.5)
    b.add_inductor("Ls", "a", "ac1", 1e-3)
    for nm, a, c in (("D1", "ac1", "dp"), ("D2", "gnd", "dp"),
                     ("D3", "dn", "ac1"), ("D4", "dn", "gnd")):
        b.add_diode(nm, a, c, 1e3, 1e-9)
    b.add_capacitor("C", "dp", "dn", 470e-6)
    b.add_resistor("RL", "dp", "dn", 100.0)
    b.add_resistor("Rtie", "dn", "gnd", 1e6)
    return b, None


CASES = dict(ts.CASES, bridge=_bridge)


@pytest.mark.parametrize("name", list(CASES))
def test_no_turn_off_carries_reverse_current(name):
    b, sf = CASES[name]()
    if sf is None:
        rtol, atol, t_end, kw = 2e-5, 2e-8, 40e-3, {}
    else:
        rtol, atol, t_end, kw = 1e-10, 1e-13, 1e-3, dict(dt=ts.T / 100)
    r = p.simulate(b, t_end=t_end, engine="trbdf2", switch_fn=sf, rtol=rtol,
                   atol=atol, **kw)
    t = np.asarray(r.times)
    S = np.asarray(r.switch_states)
    diodes = [c["name"] for c in b.components() if c["name"].startswith("D")]
    i_max = max(float(np.abs(np.asarray(r.i(n))).max()) for n in diodes)
    tol = atol + rtol * i_max
    n_off = 0
    for nm in diodes:
        k = b.switch_index_of(nm)
        iD = np.asarray(r.i(nm))
        for j in np.nonzero(S[:-1, k] & ~S[1:, k])[0]:
            n_off += 1
            assert iD[j] >= -tol, (nm, t[j + 1], iD[j], tol)
    assert n_off > 0
    if name.startswith("buck"):
        # no L*g_off spike: a buck's nodes stay within [0, Vin]
        assert np.abs(np.asarray(r.v("sw"))).max() <= ts.ref.VIN * (1 + 1e-9)
