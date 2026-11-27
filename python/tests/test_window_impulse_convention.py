"""One convention for a window integral across an impulse, on every engine
(release groups A x B2): a window integral is the REGULAR part of the
reading; an impulse is a Dirac and is reported by its record (the reinit
record: the moved charge and its energy), never folded into a line.

The circuit: 10 V source, a switch closing at 3 us onto 1 uF || 10 Ohm
(the capacitor at ~0 V). Ideal switch: the capacitor jumps to V; a 1 mOhm
switch under the restart rule: it jumps to the divider's V*R/(R+R_on)
(tau = 1 ns << dt). Both jumps are closed forms, so is the regular part
after the jump (0: the capacitor sits at its final value).

Identities, per window (a, b] with the impulse counting in the window its
instant starts (right-continuous):
  (i)  charge:  int i_C dt + sum C*J = C (v(b) - v(a))
  (ii) energy:  dE_stored + sum int p + (E_impulse - W_source,impulse) = 0,
       W_source,impulse = V * C * J (the source's Dirac current times V)
  (iii) dsed reports the same split (regular part and record).
"""
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim.losses import device_loss_summary

V, C, R, R_ON, T_C, DT = 10.0, 1e-6, 10.0, 1e-3, 3e-6, 1e-6
WINS = [(1e-6, 2e-6), (2e-6, 3e-6), (3e-6, 4e-6), (2e-6, 4e-6), (2e-6, 6e-6)]
PROBES = ["i(C)", "p(R)", "p(S)", "p(V)"]


class _Gate:
    def __call__(self, t):
        m = p.SwitchStateMask(1)
        m.set(0, t >= T_C - 1e-15)
        return m

    def next_edge_after(self, t):
        return T_C if t < T_C else float("inf")


def _circuit(ideal):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", V)
    if ideal:
        b.add_ideal_switch("S", "vin", "c", 0.0, 1e-9)
    else:
        b.add_switch("S", "vin", "c", 1.0 / R_ON, 1e-9)
    b.add_capacitor("C", "c", "gnd", C)
    b.add_resistor("R", "c", "gnd", R)
    return b


def _run(ideal, engine="pwl", probes=PROBES):
    b = _circuit(ideal)
    kw = dict(dt=DT)
    if engine == "pwl":
        kw.update(discontinuity_restart=True, enable_substep_state_correction=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 2e-5, engine=engine, switch_fn=_Gate(), auto_regularize=False,
                       recorders=[p.Recorder("r", probes, keep="none", windows=WINS)], **kw)
    return b, r


def _v_left(r, t):
    # the sample AT t (a left limit: the stored row of the step ending there)
    tt = np.asarray(r.times)
    k = int(np.argmin(np.abs(tt - t)))
    assert abs(tt[k] - t) < 1e-15
    return float(np.asarray(r.v("c"))[k])


def _jump(ideal, v_minus):
    v_plus = V if ideal else V * R / (R + R_ON)
    return v_plus - v_minus


@pytest.mark.parametrize("ideal", [True, False], ids=["ideal", "r_on"])
def test_window_charge_is_the_regular_part_and_the_record_carries_the_jump(ideal):
    b, r = _run(ideal)
    assert r.recorder_feed["mode"] == "in-loop"
    (rec,) = r.reinit_records
    assert rec.t == pytest.approx(T_C, abs=1e-15) and rec.kind == "capacitor_loop"
    J = _jump(ideal, _v_left(r, T_C))
    assert abs(rec.lambda_) == pytest.approx(C * J, rel=1e-9)
    q = r.recorded["r"].windows["i(C)"]["integral"]
    for k, (a, z) in enumerate(WINS):
        imp = C * J if a <= T_C < z else 0.0
        total = C * (_v_left(r, z) - _v_left(r, a))
        # (i) the identity, to rounding of the 1e-5 C charge
        assert q[k] + imp == pytest.approx(total, rel=1e-9, abs=1e-17), (a, z)
    # the window starting AT the impulse reads the regular part only (0 here:
    # the capacitor sits at its post-jump value), not the 1e-5 C jump
    k3 = WINS.index((3e-6, 4e-6))
    assert abs(q[k3]) < 1e-12 * C * V


def _ledger(ideal):
    b, r = _run(ideal)
    (rec,) = r.reinit_records
    W = r.recorded["r"].windows
    e_imp = {e["name"]: e for e in device_loss_summary(b, r)}["S"]["E_impulse_total"]
    assert e_imp == pytest.approx(rec.dW, rel=1e-15)
    v_m = _v_left(r, T_C)
    J = _jump(ideal, v_m)
    # the record's dW is the jump's own loss, (1/2) C J^2 (whole jump); with
    # R_on the source also works across the R_on drop during the jump,
    # C J (V - v+), which the regular lines cannot see
    assert rec.dW == pytest.approx(0.5 * C * J * J, rel=1e-9)
    w_src = V * C * J
    e_net = e_imp + C * J * (V - (v_m + J)) - w_src
    for k, (a, z) in enumerate(WINS):
        de = 0.5 * C * (_v_left(r, z) ** 2 - _v_left(r, a) ** 2)
        p_sum = sum(W[s]["integral"][k] for s in ("p(R)", "p(S)", "p(V)"))
        inside = a <= T_C < z
        ledger = de + p_sum + (e_net if inside else 0.0)
        assert abs(ledger) < 1e-9 * (abs(de) + w_src * inside) + 1e-18, (a, z, ledger)


def test_energy_ledger_closes_with_the_record_after_a_restart_jump():
    _ledger(False)


# F-W-1 closed on the restart line (item C): stage 2 of an ideal element's jump starts from x0+
def test_energy_ledger_closes_with_the_records_on_the_ideal_switch():
    _ledger(True)


def test_dsed_reports_the_same_split():
    _, rp = _run(True)
    _, rd = _run(True, engine="dsed", probes=["p(R)", "p(S)", "p(V)"])
    (a,), (d,) = rp.reinit_records, rd.reinit_records
    assert d.t == pytest.approx(T_C, abs=1e-15)
    assert abs(d.lambda_) == pytest.approx(abs(a.lambda_), rel=1e-9)
    assert d.dW == pytest.approx(a.dW, rel=1e-9)
    # the regular charge: pwl's window integral, dsed's C dv minus its record
    q = rp.recorded["r"].windows["i(C)"]["integral"]
    for k, (lo, hi) in enumerate(WINS):
        imp = abs(d.lambda_) if lo <= T_C < hi else 0.0
        qd = C * (_v_left(rd, hi) - _v_left(rd, lo)) - imp
        assert q[k] == pytest.approx(qd, abs=1e-9 * C * V), (lo, hi)
    np.testing.assert_allclose(rp.recorded["r"].windows["p(R)"]["integral"],
                               rd.recorded["r"].windows["p(R)"]["integral"],
                               rtol=1e-6, atol=1e-12)


def test_dsed_and_pwl_source_power_is_the_regular_part():
    _, rp = _run(True)
    _, rd = _run(True, engine="dsed", probes=["p(R)", "p(S)", "p(V)"])
    np.testing.assert_allclose(rp.recorded["r"].windows["p(V)"]["integral"],
                               rd.recorded["r"].windows["p(V)"]["integral"],
                               rtol=1e-6, atol=1e-12)


def test_dsed_source_power_is_the_regular_part():
    _, rd = _run(True, engine="dsed", probes=["p(R)", "p(S)", "p(V)"])
    k3 = WINS.index((3e-6, 4e-6))
    assert rd.recorded["r"].windows["p(V)"]["integral"][k3] == pytest.approx(
        -V * V / R * DT, rel=1e-6)
