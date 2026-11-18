"""add-branch-form-ideal-switch 6.1-6.2: the ideal DCM buck on `pwl`.

Ideal switch and ideal diode (R_on = 0, g_off = 0), SDIRK x3 restart, the
diode turn-off located on its solved current i_s (the crossing signal of an
ON branch-form diode) and the step split there. Started on the orbit
(v_C = 8.9963970815477 V, i_L = 0 at the rising edge), 300 periods. The
prototype (`probes/sdirk_restart_ideal.out` (3)) errs by -2.442e-6,
-6.591e-7, -1.709e-7, -4.345e-8 V at N = 50, 100, 200, 400.
"""
import math
import warnings

import numpy as np
import pytest

import pulsim as p
from tests.oracles import ideal_switch_reference as ref

PROTO = {50: -2.442e-6, 100: -6.591e-7, 200: -1.709e-7, 400: -4.345e-8}
T = ref.T_BUCK


def _buck(N, periods=300):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_ideal_switch("S", "vin", "sw", 0.0, 0.0)
    b.add_diode("D1", "gnd", "sw", form="branch", R_on=0.0, g_off=0.0)
    b.add_inductor("L", "sw", "vout", 100e-6, 0.0)
    b.add_capacitor("C", "vout", "gnd", 47e-6, ref.VC_REF)
    b.add_resistor("R", "vout", "gnd", 50.0)
    k, n = b.switch_index_of("S"), b.graph.num_switches
    on_steps = round(0.3 * N)

    fn_obj = p.make_pwm_switch_fn(1 / T, on_steps / N, k, n, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=periods * T + 1e-3 * T / N, dt=T / N, engine="pwl",
                          switch_fn=fn_obj, discontinuity_restart=True,
                          enable_substep_state_correction=True)


@pytest.fixture(scope="module")
def runs():
    return {N: _buck(N) for N in PROTO}


def test_ideal_dcm_buck_error_and_order(runs):
    errs = {N: float(np.asarray(r.v("vout"))[-1]) - ref.VC_REF for N, r in runs.items()}
    for N, e in errs.items():
        assert abs(e) <= 1.5 * abs(PROTO[N]), (N, e, PROTO[N])
    assert math.log2(errs[200] / errs[400]) >= 1.85, errs


def test_ideal_dcm_buck_idle_current_is_exactly_zero(runs):
    r = runs[100]
    iL = np.asarray(r.i("L"))
    sw = np.asarray(r.switch_states)
    idle = ~sw.any(axis=1)                 # switch and diode both off
    assert idle.sum() > 1000
    assert np.all(iL[idle] == 0.0)
    assert len(r.reinit_records) == 0
