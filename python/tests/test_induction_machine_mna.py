"""MNA-native squirrel-cage induction machine (add-mna-native-machines, group 2).

Every reference here is independent of the engine:

* held speed (locked rotor, synchronous, slip 0.05): the per-phase
  equivalent circuit in closed form (`_machine_reference/
  induction_reference.py::equivalent_circuit`);
* free acceleration: the same equations integrated as a 5-state ODE by
  SciPy Radau and DOP853 at rtol 1e-12 (they agree to 1e-11); the numbers
  are pinned below so the test needs no SciPy.

Declared test machine (design.md; not a real machine, nothing is tuned):
R_s = 0.5 Ω, L_ls = 2 mH, R_r = 0.4 Ω, L_lr = 2 mH, L_m = 60 mH, pp = 2,
100 V rms per phase, 60 Hz.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p

RS, LLS, RR, LLR, LM, PP = 0.5, 2e-3, 0.4, 2e-3, 60e-3, 2
V_RMS, F = 100.0, 60.0
W_SYNC = 2 * math.pi * F / PP           # 188.4956 rad/s
KR = LM / (LLR + LM)

# induction_reference.py, closed form.
I_LOCKED = 57.87987167      # A rms, slip 1
I_SYNC = 4.277380114        # A rms, slip 0
I_S05, T_S05 = 12.28908815, 16.11973513   # slip 0.05

# induction_reference.py, free acceleration (J = 0.01, B = 0, no load).
J_DOL = 0.01
T_REF = (0.02, 0.05, 0.1, 0.15, 0.2)
OMEGA_REF = (47.567768929, 109.024125628, 187.233840055, 187.941137155,
             188.357018519)
IA_REF = (28.4250712409, -49.2154404944, -9.0343375932, -6.05400189765,
          -6.0073579194)


def _machine(b, *, J, held_speed=None):
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(
            f"V{'abc'[k]}", node, "gnd", 0.0, V_RMS * math.sqrt(2), F,
            -2 * math.pi / 3 * k)
    b.add_induction_machine_mna("M", "ua", "ub", "uc", "nn", "w",
                                R_s=RS, L_ls=LLS, R_r=RR, L_lr=LLR, L_m=LM,
                                pole_pairs=PP, J=J,
                                omega0=held_speed or 0.0)
    if held_speed is not None:
        b.add_voltage_source("Vw", "w", "gnd", held_speed)
    return b


def _torque(res):
    ia, ib, ic = (np.asarray(res.i(f"M_{x}")) for x in "abc")
    isa = (2 * ia - ib - ic) / 3
    isb = (ib - ic) / math.sqrt(3)
    lra, lrb = np.asarray(res.v("M_psi_ra")), np.asarray(res.v("M_psi_rb"))
    return 1.5 * PP * KR * (lra * isb - lrb * isa)


def _held(speed, t_end=1.5, dt=5e-5):
    b = _machine(p.CircuitBuilder(), J=1.0, held_speed=speed)
    res = p.simulate(b, t_end=t_end, dt=dt, engine="pwl")
    t = np.asarray(res.times)
    t0 = t_end - 30 / F             # last 30 cycles, exactly clipped
    ia = np.asarray(res.i("M_a"))
    rms = math.sqrt(p.time_average(t, ia**2, t0, t_end))
    torque = p.time_average(t, _torque(res), t0, t_end)
    # The holding source carries the torque the stamp put into the node:
    # its mean current equals T_em (an independent read of the stamp).
    i_src = p.time_average(t, np.asarray(res.i("Vw")), t0, t_end)
    return rms, torque, i_src


def test_locked_rotor_current_matches_the_equivalent_circuit():
    rms, _t, _ = _held(0.0)
    assert rms == pytest.approx(I_LOCKED, rel=5e-3)


def test_synchronous_speed_draws_magnetising_current_and_no_torque():
    rms, torque, _ = _held(W_SYNC)
    assert rms == pytest.approx(I_SYNC, rel=5e-3)
    assert abs(torque) < 2e-3 * T_S05     # measured 0.011 N·m at 50 µs


def test_torque_and_current_at_slip_0_05():
    rms, torque, i_src = _held((1 - 0.05) * W_SYNC)
    assert rms == pytest.approx(I_S05, rel=5e-3)
    assert torque == pytest.approx(T_S05, rel=5e-3)
    assert i_src == pytest.approx(torque, rel=1e-6)   # the source sinks T_em


def _dol(engine, dt=None, **kw):
    b = _machine(p.CircuitBuilder(), J=J_DOL)
    if engine == "pwl":
        res = p.simulate(b, t_end=T_REF[-1], dt=dt, engine="pwl", **kw)
    else:
        res = p.simulate(b, t_end=T_REF[-1], engine=engine, **kw)
    t = np.asarray(res.times)
    return (t, np.asarray(res.v("w")), np.asarray(res.i("M_a")))


def _dol_error(t, w, ia):
    ew = np.max(np.abs(np.interp(T_REF, t, w) - OMEGA_REF)) / W_SYNC
    ei = np.max(np.abs(np.interp(T_REF, t, ia) - IA_REF)) / I_LOCKED
    return ew, ei


def test_free_acceleration_on_pwl_is_second_order():
    # The fixed engine is first order from a start whose derivative is
    # not consistent (phases b, c are at ±86.6 V at t = 0 with zero
    # current; plain RL loads show the same 2.0 ratio): the start window's
    # restart is what makes the run second order. Measured 1.60e-4,
    # 4.01e-5, 1.00e-5 (ratios 3.99, 4.00).
    errs = [_dol_error(*_dol("pwl", dt, discontinuity_restart=True))[0]
            for dt in (8e-5, 4e-5, 2e-5)]
    assert errs[-1] < 2e-5
    for a, b in zip(errs, errs[1:]):
        assert a / b > 3.5, errs


def test_free_acceleration_on_trbdf2_matches_the_reference():
    ew, ei = _dol_error(*_dol("trbdf2", rtol=1e-8, atol=1e-9))
    assert ew < 1e-4
    assert ei < 1e-3


def test_pwl_and_trbdf2_agree_on_the_start():
    t1, w1, _ = _dol("pwl", 2e-5)
    t2, w2, _ = _dol("trbdf2")
    ts = np.linspace(0.01, T_REF[-1], 40)
    diff = np.max(np.abs(np.interp(ts, t1, w1) - np.interp(ts, t2, w2)))
    assert diff < 1e-3 * W_SYNC


def test_dsed_refuses_the_machine_by_name():
    b = _machine(p.CircuitBuilder(), J=J_DOL)
    with pytest.raises(p.UnsupportedError) as e:
        p.simulate(b, t_end=1e-3, engine="dsed")
    assert e.value.engine == "dsed"
    assert e.value.device.startswith("M_")


def test_stored_kind_and_pool_record():
    b = _machine(p.CircuitBuilder(), J=J_DOL)
    m = b.pool.induction_machines()
    assert len(m) == 1
    assert m[0]["params"]["L_m"] == LM
    assert str(b.pool.kind_of(m[0]["branch_ids"][0])) == (
        "StoredKind.InductionMachineMna")


def _energy_residual(res):
    """(∫p_in − ∫copper − ∫T_em·ω − ΔW) / ∫p_in over the run, every term
    by the trapezoidal rule in time; W = (3/4)(i_s·λ_s + i_r·λ_r)."""
    t = np.asarray(res.times)
    vn = np.asarray(res.v("nn"))
    v = [np.asarray(res.v(n)) - vn for n in ("ua", "ub", "uc")]
    ia, ib, ic = (np.asarray(res.i(f"M_{x}")) for x in "abc")
    isa, isb = (2 * ia - ib - ic) / 3, (ib - ic) / math.sqrt(3)
    lra, lrb = np.asarray(res.v("M_psi_ra")), np.asarray(res.v("M_psi_rb"))
    lr_ = LLR + LM
    ira, irb = (lra - LM * isa) / lr_, (lrb - LM * isb) / lr_
    sls = LLS + LM * LLR / lr_
    w_mag = 0.75 * (isa * (sls * isa + KR * lra) + isb * (sls * isb + KR * lrb)
                    + ira * lra + irb * lrb)
    p_in = v[0] * ia + v[1] * ib + v[2] * ic
    cu = 1.5 * (RS * (isa**2 + isb**2) + RR * (ira**2 + irb**2))
    e_in = np.trapezoid(p_in, t)
    rest = (np.trapezoid(cu, t) + np.trapezoid(_torque(res) * res.v("w"), t)
            + w_mag[-1] - w_mag[0])
    return (e_in - rest) / e_in


def test_energy_balance_on_a_start_converges_at_second_order():
    # The model identity holds to 1e-9 at random states (C++ test); on a
    # run the residual is the integration error: measured 5.17e-7 at
    # 20 µs and 1.29e-7 at 10 µs (690.6 J delivered).
    runs = [p.simulate(
        _machine(p.CircuitBuilder(), J=J_DOL), t_end=T_REF[-1], dt=dt,
        engine="pwl", discontinuity_restart=True) for dt in (2e-5, 1e-5)]
    # Every step at the dt asked for (each used to fail into a dt/2 retry).
    assert not any(res.dt_retries for res in runs)
    r = [_energy_residual(res) for res in runs]
    assert abs(r[1]) < 2e-7
    assert abs(r[0] / r[1]) > 3.5, r


def test_locked_rotor_torque_matches_the_equivalent_circuit():
    _rms, torque, _ = _held(0.0)
    assert torque == pytest.approx(19.96766081, rel=5e-3)


def test_ac_sweep_refuses_the_machine_by_name():
    from pulsim import _pulsim as k

    b = _machine(p.CircuitBuilder(), J=J_DOL)
    with pytest.raises(p.UnsupportedError) as e:
        k.run_mna_sweep_kernel(
            b.graph, b.pool, p.SwitchStateMask(b.graph.num_switches),
            [1e2, 1e4], b.node_id_of("ua"), b.node_id_of("w"))
    assert e.value.engine == "ac_sweep"
    assert e.value.device.startswith("M_")


def test_signals_carry_torque_and_rotor_flux():
    res = p.simulate(_machine(p.CircuitBuilder(), J=J_DOL), t_end=0.02,
                     dt=5e-5, engine="pwl")
    assert np.allclose(res.signal("M.T_em"), _torque(res), rtol=0, atol=1e-9)
    assert np.array_equal(res.signal("M.omega"), np.asarray(res.v("w")))


class _SixStep:
    """Six-step (180°) gating of a three-leg bridge at 60 Hz, with the edge
    oracle the variable-step engine needs. Switch order: (a+, a-, b+, b-,
    c+, c-), the order `_inverter` adds them."""

    def __init__(self, f=F):
        self.T = 1.0 / f

    def _sector(self, t):
        return int(math.floor(t / (self.T / 6) + 1e-9)) % 6

    def __call__(self, t):
        s = self._sector(t)
        m = p.SwitchStateMask(6)
        for leg, shift in enumerate((0, 4, 2)):   # b lags a by 120°
            up = ((s + shift) % 6) < 3
            m.set(2 * leg, up)
            m.set(2 * leg + 1, not up)
        return m

    def next_edge_after(self, t):
        k = math.floor(t / (self.T / 6) + 1e-9) + 1
        return k * self.T / 6


def _inverter():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 300.0)
    for x in ("a", "b", "c"):
        b.add_switch(f"S{x}p", "dc", f"u{x}", 1e3, 1e-6)
        b.add_switch(f"S{x}n", f"u{x}", "gnd", 1e3, 1e-6)
    b.add_induction_machine_mna("M", "ua", "ub", "uc", "nn", "w",
                                R_s=RS, L_ls=LLS, R_r=RR, L_lr=LLR, L_m=LM,
                                pole_pairs=PP, J=J_DOL)
    return b


def test_inverter_fed_start_agrees_on_pwl_and_trbdf2():
    sw = _SixStep()
    r1 = p.simulate(_inverter(), t_end=0.15, dt=1e-5, engine="pwl",
                    switch_fn=sw)
    r2 = p.simulate(_inverter(), t_end=0.15, engine="trbdf2", switch_fn=sw,
                    rtol=1e-7, atol=1e-8)
    ts = np.linspace(0.01, 0.15, 40)
    w1 = np.interp(ts, r1.times, r1.v("w"))
    w2 = np.interp(ts, r2.times, r2.v("w"))
    assert w1[-1] > 0.9 * W_SYNC          # it accelerates to near sync
    assert np.max(np.abs(w1 - w2)) < 1e-3 * W_SYNC
