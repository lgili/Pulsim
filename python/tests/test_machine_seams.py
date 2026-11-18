"""Integration batch 12: the MNA-native machines (add-mna-native-machines) at
the seams with the device interface (flat snapshot state, unified refresh)
and the circuit IR (YAML round trip beside scopes and gated devices)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p

F = 60.0


def _im(b, J=0.01):
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(f"V{'abc'[k]}", node, "gnd", 0.0,
                                  100.0 * math.sqrt(2), F, -2 * math.pi / 3 * k)
    b.add_induction_machine_mna("M", "ua", "ub", "uc", "nn", "w", R_s=0.5,
                                L_ls=2e-3, R_r=0.4, L_lr=2e-3, L_m=60e-3,
                                pole_pairs=2, J=J)
    return b


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_an_induction_machine_resumed_mid_start_equals_the_uninterrupted_run(engine):
    kw = {"dt": 5e-5} if engine == "pwl" else {}
    t_mid, t_end = 0.03, 0.06
    full = p.simulate(_im(p.CircuitBuilder()), t_end=t_end, engine=engine, **kw)
    first = p.simulate(_im(p.CircuitBuilder()), t_end=t_mid, engine=engine, **kw)
    snap = first.final_snapshot
    assert "pmsm_mna" in list(snap.dev_layout_names)   # the machine's block
    second = p.simulate(_im(p.CircuitBuilder()), t_end=t_end, t_start=t_mid,
                        engine=engine, resume_from=snap, **kw)
    w_full = float(np.interp(t_end, full.times, full.v("w")))
    w_res = float(np.asarray(second.v("w"))[-1])
    assert float(np.asarray(first.v("w"))[-1]) > 20.0   # control: accelerating
    if engine == "pwl":
        # same state; the resumed grid is t_mid + n*dt, so equal to rounding
        assert w_res == pytest.approx(float(np.asarray(full.v("w"))[-1]), rel=1e-9)
        assert float(np.asarray(second.i("M_a"))[-1]) == pytest.approx(
            float(np.asarray(full.i("M_a"))[-1]), rel=1e-9)
    else:
        assert w_res == pytest.approx(w_full, rel=1e-4)
    if engine == "pwl":   # the assembled carrier at every dt survives resume
        assert not second.dt_retries and not full.dt_retries


def test_bldc_harmonics_im_gated_device_and_scope_round_trip_yaml():
    # the three emitter changes merged here: machines (rotor, harmonics),
    # gated diodes (gate, I_H), IR scopes / subsystems
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "p", "gnd", 100.0)
    b.add_thyristor("T", "p", "k", 1e-3, 1e9, I_H=0.05)
    b.add_resistor("Rk", "k", "gnd", 10.0)
    _im(b)
    b.add_bldc_mna("B", "ua", "ub", "uc", "bn", "wb", "thb", R_s=0.3,
                   L_s=1e-3, k_e=0.05, pole_pairs=4, J=1e-4, n_harmonics=7)
    f = p.define_subsystem("RC", ports=("a",), params={"R": 1.0})

    @f.body
    def _(s):
        s.add_resistor("R", "a", "m", s.p.R)
        s.add_capacitor("C", "m", "gnd", 1e-6)

    f.instantiate(b, "filt", a="ua")
    c = b.build()
    y = c.to_yaml()
    assert "harmonics: [" in y and "rotor:" in y and "gate: 2" in y
    c2 = p.Circuit.from_yaml(y)
    assert c2.value_hash() == c.value_hash()
    assert c2.structure_hash() == c.structure_hash()
    assert c2.to_yaml() == y
    assert c2.scopes() == c.scopes()
    gates = {d["name"]: d["params"].get("gate") for d in c2.components()}
    assert gates["T"] == 2 and gates.get("Rk") is None
