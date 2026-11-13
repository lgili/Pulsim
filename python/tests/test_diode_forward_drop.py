"""PSIM-style forward drop ``V_f`` of the PWL ``SwitchedDiode``.

While ON the diode is ``g_on`` in series with ``V_f``
(``i = g_on·(v_D − V_f)``); OFF is a plain ``g_off``. It turns on at
``V_on = max(V_th, V_f)`` (``V_th`` alone at ``V_f = 0``) and off at
zero current (``v_D = V_f``). Every engine decides through the one
``SwitchedDiode`` law, and ``res.i`` / the loss and thermal summaries
use the same current. A decision site that missed the law would let an
ON diode conduct BACKWARDS up to ``g_on·V_f`` with no warning, so most
tests below also assert ``min(i_D) >= -(1e-7 + 1e-4·max|i_D|)``
(``_no_reverse``: OFF leakage plus a variable-step landing just past
the current zero, far above ``-g_on·V_f``).

Numbers marked "analytic" are closed-form; the others were measured on
this branch and are pinned with the reason next to them.
"""

from __future__ import annotations

import functools
import math
import warnings

import numpy as np
import pytest

import pulsim as p

G_ON = 20.0
V_F = 0.7


def _sim(b, t_end, engine, dt=None, **kw):
    if engine == "pwl":
        kw.setdefault("dt", dt if dt is not None else 1e-6)
        kw["engine"] = "pwl"
    else:
        kw["engine"] = engine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, **kw)


def _no_reverse(i):
    """No reverse current beyond OFF leakage (g_off·|v| ~ 1e-8 A) and a
    variable-step landing just past the current zero (1.6e-6 A on the
    bridge below; trbdf2's held tolerance is 1e-8 of the node voltages,
    i.e. up to g_on·4e-7 V). A decision site that ignored V_f would let an ON diode
    carry up to g_on·V_f backwards (14 A here, 700 A at g_on = 1e3)."""
    i = np.asarray(i, dtype=float)
    return float(i.min()) >= -(1e-7 + 1e-4 * float(np.abs(i).max()))


def _entry(summary, name):
    return next(e for e in summary if e["name"] == name)


def _loss(b, r, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.device_loss_summary(b, r, **kw)


# ---------------------------------------------------------------------
# 6. Forced current: 10 A into g_on = 20 S with V_f = 0.7 -> 1.2 V, 12 W
# ---------------------------------------------------------------------

def _forced(cap: bool):
    b = p.CircuitBuilder()
    b.add_current_source("I1", "a", "gnd", 10.0)   # 10 A out of "a"
    # dsed needs a state variable (the C) and, for the zero-current
    # turn-on the C charges into, a hysteresis band (V_th > V_f).
    b.add_diode("D1", "a", "gnd", G_ON, 1e-9,
                V_th=V_F + 1e-3 if cap else 0.0, V_f=V_F)
    if cap:
        b.add_capacitor("C", "a", "gnd", 1e-6)
    return b


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "dsed"])
def test_forced_current_drops_v_f_plus_i_over_g_on(engine):
    """Analytic: v = 0.7 + 10/20 = 1.2 V, P = 0.7·10 + 10²/20 = 12 W."""
    b = _forced(cap=(engine == "dsed"))
    r = _sim(b, 2e-3, engine, dt=1e-6)
    v = np.asarray(r.v("a"))
    i = np.asarray(r.i("D1"))
    assert v[-1] == pytest.approx(1.2, rel=1e-9)
    assert i[-1] == pytest.approx(10.0, rel=1e-9)
    e = _entry(_loss(b, r, t_from=1e-3), "D1")
    assert e["V_f"] == V_F
    assert e["P_avg"] == pytest.approx(12.0, rel=1e-9)
    assert e["i_avg"] == pytest.approx(10.0, rel=1e-9)


def test_forced_current_dc_operating_point():
    b = _forced(cap=False)
    r = _sim(b, 1e-5, "pwl", dt=1e-6, start_from_dc_op=True)
    assert np.asarray(r.v("a"))[0] == pytest.approx(1.2, rel=1e-12)
    assert np.asarray(r.i("D1"))[0] == pytest.approx(10.0, rel=1e-12)


# ---------------------------------------------------------------------
# 7. Resistive half-wave: 10 V peak, 50 Hz, R = 10 ohm
# ---------------------------------------------------------------------

V_PK, F_AC, R_HW = 10.0, 50.0, 10.0
W = 2 * math.pi * F_AC
T_ON = math.asin(V_F / V_PK) / W                       # 222.999 us
T_OFF = (math.pi - math.asin(V_F / V_PK)) / W          # 9.7770 ms
K_DIV = R_HW / (R_HW + 1.0 / G_ON)
V_PEAK = (V_PK - V_F) * K_DIV                          # 9.25373 V
V_AVG = K_DIV * (2 * V_PK * math.cos(math.asin(V_F / V_PK))
                 - V_F * (math.pi - 2 * math.asin(V_F / V_PK))) / (2 * math.pi)


def test_half_wave_analytic_constants():
    assert T_ON == pytest.approx(222.999e-6, abs=1e-9)
    assert T_OFF == pytest.approx(9.7770e-3, abs=1e-7)
    assert V_PEAK == pytest.approx(9.25373, abs=1e-5)
    assert V_AVG == pytest.approx(2.82677, abs=1e-5)


def _half_wave(V_th=0.0, cap=False):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "s", "gnd", 0.0, V_PK, F_AC)
    b.add_diode("D1", "s", "k", G_ON, 1e-9, V_th=V_th, V_f=V_F)
    b.add_resistor("R", "k", "gnd", R_HW)
    if cap:
        b.add_capacitor("Ck", "k", "gnd", 1e-9)
    return b


@functools.lru_cache(maxsize=None)
def _run_half_wave(engine, V_th=0.0):
    b = _half_wave(V_th, cap=(engine == "dsed"))
    return b, _sim(b, 0.02, engine, dt=1e-6)


def _on_interval(t, i):
    on = np.flatnonzero(i > 1e-6)
    return float(t[on[0]]), float(t[on[-1]])


def test_half_wave_pwl_is_the_drop_law_on_every_sample():
    b, r = _run_half_wave("pwl")
    t = np.asarray(r.times)
    vs, vk = np.asarray(r.v("s")), np.asarray(r.v("k"))
    i = np.asarray(r.i("D1"))
    on = i > 1e-6
    assert np.allclose(vk[on], (vs[on] - V_F) * K_DIV, rtol=1e-12,
                       atol=1e-12)
    t_on, t_off = _on_interval(t, i)
    # First / last conducting sample on the 1 us grid (the event
    # iteration turns the diode on in the step after the crossing).
    assert abs(t_on - T_ON) <= 2e-6 and abs(t_off - T_OFF) <= 2e-6
    # The sub-step watcher of the no-L/C (static) path signals on
    # v - V_on: on v - V_th, v stays above 0 across both flip steps and
    # the events vanish. Measured 0.7 ns / 2.7 ns off.
    ev = sorted(e.t_estimated for e in r.commutation_events)
    assert ev == pytest.approx([T_ON, T_OFF], abs=1e-6)
    assert float(vk.max()) == pytest.approx(V_PEAK, abs=1e-6)
    assert p.time_average(t, vk) == pytest.approx(V_AVG, abs=1e-5)
    assert _no_reverse(i)


@pytest.mark.parametrize("engine", ["trbdf2", "auto"])
def test_half_wave_variable_step(engine):
    b, r = _run_half_wave(engine)
    t = np.asarray(r.times)
    vk = np.asarray(r.v("k"))
    i = np.asarray(r.i("D1"))
    assert _no_reverse(i)
    assert float(vk.max()) == pytest.approx(V_PEAK, rel=1e-4)
    assert p.time_average(t, vk) == pytest.approx(V_AVG, rel=1e-4)
    # The turn-on (at v = V_f) and the current-zero turn-off (also at
    # v = V_f) are localised events: measured 0.4 ns / 0.5 ns off.
    ev = sorted(e.t_estimated for e in r.commutation_events)
    assert ev == pytest.approx([T_ON, T_OFF], abs=1e-8)


def test_half_wave_dsed_with_a_band_matches_pwl():
    """DSED settles a zero-current turn-on only with a hysteresis band
    (V_th > V_f); with it the drop law holds and it matches pwl."""
    b, r = _run_half_wave("dsed", V_th=V_F + 1e-3)
    t = np.asarray(r.times)
    vk = np.asarray(r.v("k"))
    i = np.asarray(r.i("D1"))
    assert _no_reverse(i)
    assert float(vk.max()) == pytest.approx(V_PEAK, rel=1e-4)
    # The band delays turn-on by 1 mV of a 10 V sine: 0.1 % of V_avg.
    assert p.time_average(t, vk) == pytest.approx(V_AVG, rel=2e-3)


# ---------------------------------------------------------------------
# 8. RL half-wave: extinction at i = 0, against the analytic solution
# ---------------------------------------------------------------------

L_RL = 20e-3


def _rl_analytic():
    """i(t) = (V/Z) sin(wt − phi) − V_f/R' + A e^{−(t − t_on)/tau},
    i(t_on) = 0; extinction at the next zero (bisection)."""
    Rp = R_HW + 1.0 / G_ON
    Z = math.hypot(Rp, W * L_RL)
    phi = math.atan2(W * L_RL, Rp)
    tau = L_RL / Rp

    def ss(t):
        return V_PK / Z * math.sin(W * t - phi) - V_F / Rp
    A = -ss(T_ON)

    def i(t):
        return ss(t) + A * math.exp(-(t - T_ON) / tau)
    lo, hi = T_ON + 5e-3, T_ON + 19e-3
    assert i(lo) > 0 > i(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if i(mid) > 0 else (lo, mid)
    return i, 0.5 * (lo + hi)


def _rl(V_th=0.0):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "s", "gnd", 0.0, V_PK, F_AC)
    b.add_diode("D1", "s", "k", G_ON, 1e-9, V_th=V_th, V_f=V_F)
    b.add_resistor("R", "k", "m", R_HW)
    b.add_inductor("L", "m", "gnd", L_RL)
    return b


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "auto", "dsed"])
def test_rl_half_wave_extinguishes_at_zero_current(engine):
    i_ref, t_ext = _rl_analytic()
    V_th = V_F + 1e-3 if engine == "dsed" else 0.0
    b = _rl(V_th)
    r = _sim(b, 0.02, engine, dt=1e-6)
    t = np.asarray(r.times)
    i = np.asarray(r.i("D1"))
    i_L = np.asarray(r.i("L"))
    assert _no_reverse(i)
    assert float(i_L.min()) >= -1e-6
    if engine in ("trbdf2", "auto"):
        # Localised events: extinction measured 2.5 ns from analytic.
        ev = sorted(e.t_estimated for e in r.commutation_events)
        assert ev == pytest.approx([T_ON, t_ext], abs=1e-8)
    else:
        # Last conducting sample (pwl: 1 us grid; dsed: 1 mV band at
        # turn-on shifts the whole conduction slightly).
        _, t_off = _on_interval(t, i)
        tol = 2e-6 if engine == "pwl" else 2e-4
        assert t_off == pytest.approx(t_ext, abs=tol)
    if engine == "pwl":
        # The dynamic path's sub-step watcher (v - V_on, linear in the
        # step): measured 0.7 ns / 0.47 us off on the 1 us grid.
        ev = sorted(e.t_estimated for e in r.commutation_events)
        assert ev == pytest.approx([T_ON, t_ext], abs=1e-6)
    mid = (t > T_ON + 1e-3) & (t < t_ext - 1e-3)
    ref = np.array([i_ref(x) for x in t[mid]])
    assert np.max(np.abs(i[mid] - ref)) <= (1e-4 if engine != "dsed"
                                            else 2e-3) * ref.max()


# ---------------------------------------------------------------------
# 9. Bridge: charge and power balance with the drop
# ---------------------------------------------------------------------

def _bridge(V_th=0.0, V_f=V_F):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "bn", 0.0, 20.0, 50.0)
    b.add_resistor("Rs", "s", "a", 0.1)
    p.add_bridge_rectifier(b, "BR", ac_a="a", ac_b="bn", dc_pos="p",
                           dc_neg="gnd", V_th=V_th, V_f=V_f)
    b.add_capacitor("C", "p", "gnd", 100e-6)
    b.add_resistor("RL", "p", "gnd", 100.0)
    return b


@functools.lru_cache(maxsize=None)
def _run_bridge(engine, V_th=0.0):
    b = _bridge(V_th)
    return b, _sim(b, 0.1, engine, dt=1e-6)


def _trap(t, y, a, z):
    m = (t >= a) & (t <= z)
    return float(np.trapezoid(y[m], t[m]))


def test_bridge_charge_and_power_balance_pwl():
    b, r = _run_bridge("pwl")
    t = np.asarray(r.times)
    a, z = 0.08, 0.1
    D = [f"BR__D{k}" for k in (1, 2, 3, 4)]
    i = {d: np.asarray(r.i(d)) for d in D}
    for d in D:
        assert _no_reverse(i[d]), d
    vp = np.asarray(r.v("p"))
    k = np.flatnonzero((t >= a) & (t <= z))
    q_d = _trap(t, i["BR__D1"] + i["BR__D2"], a, z)
    q_dc = _trap(t, vp / 100.0, a, z) + 100e-6 * (vp[k[-1]] - vp[k[0]])
    assert q_d == pytest.approx(q_dc, rel=1e-6)
    # Tellegen per sample: every branch power by the SAME trapezoid
    # closes; the diode powers come from the drop law v·i.
    vs, va, vbn = (np.asarray(r.v(n)) for n in ("s", "a", "bn"))
    i_s = np.asarray(r.i("Rs"))
    P_src = _trap(t, (vs - vbn) * i_s, a, z)
    P_rs = _trap(t, (vs - va) * i_s, a, z)
    P_rl = _trap(t, vp * vp / 100.0, a, z)
    E_c = 0.5 * 100e-6 * (vp[k[-1]] ** 2 - vp[k[0]] ** 2)
    nodes = {"BR__D1": ("a", "p"), "BR__D2": ("bn", "p"),
             "BR__D3": ("gnd", "bn"), "BR__D4": ("gnd", "a")}
    vn = {n: (np.zeros_like(t) if n == "gnd" else np.asarray(r.v(n)))
          for n in ("a", "p", "bn", "gnd")}
    P_d = sum(_trap(t, (vn[x] - vn[y]) * i[d], a, z)
              for d, (x, y) in nodes.items())
    assert P_src == pytest.approx(P_rs + P_rl + E_c + P_d, rel=1e-6)
    # The loss summary's diode energies agree with v·i (its same-stamp
    # steps are exact, the trapezoid above is O(dt²)).
    s = _loss(b, r, t_from=a, t_to=z)
    E_sum = sum(_entry(s, d)["E_total"] for d in D)
    assert E_sum == pytest.approx(P_d, rel=1e-4)
    # A naive g·v² would miss the -V_f·i term by far more than that.
    naive = sum(_trap(t, (vn[x] - vn[y]) ** 2
                      * np.where(i[d] > 1e-6, 1e3, 1e-9), a, z)
                for d, (x, y) in nodes.items())
    assert abs(naive - P_d) > 1e-2 * P_d


@pytest.mark.parametrize("engine", ["trbdf2", "auto", "dsed"])
def test_bridge_banded_engines_agree_with_pwl(engine):
    _, rp = _run_bridge("pwl", V_F + 1e-3)
    _, r = _run_bridge(engine, V_F + 1e-3)
    tp, t = np.asarray(rp.times), np.asarray(r.times)
    avg_p = p.time_average(tp, rp.v("p"), t_from=0.08)
    avg = p.time_average(t, r.v("p"), t_from=0.08)
    assert avg == pytest.approx(avg_p, rel=1e-3)
    for d in ("BR__D1", "BR__D3"):
        assert _no_reverse(np.asarray(r.i(d))), d


# ---------------------------------------------------------------------
# 10. Half-bridge with dead time: the diode conducts only in the dead
#     time (R_on·|i| < V_f), at about 2·t_d·f_sw
# ---------------------------------------------------------------------

def _half_bridge(V_f):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "bp", "gnd", 300.0)
    vsi = p.add_three_phase_vsi(b, "INV", vdc_pos="bp", vdc_neg="gnd",
                                out_a="ua", out_b="ub", out_c="uc",
                                V_f=V_f)
    b.add_inductor("L", "ua", "m", 2e-3)
    b.add_resistor("R", "m", "gnd", 5.0)
    b.add_resistor("Rb", "ub", "gnd", 1e6)
    b.add_resistor("Rc", "uc", "gnd", 1e6)
    hs, ls = vsi.switch_indices[0], vsi.switch_indices[1]
    fn = p.make_dead_time_pwm_pair_fn(5e3, 0.5, hs, ls,
                                      b.graph.num_switches, 1e-6)
    return b, fn


@functools.lru_cache(maxsize=None)
def _run_half_bridge(V_f):
    b, fn = _half_bridge(V_f)
    r = _sim(b, 4e-3, "pwl", dt=1e-7, switch_fn=fn)
    return b, fn, r


def test_half_bridge_dead_time_duty_and_no_sharing():
    b, fn, r = _run_half_bridge(V_F)
    s = _loss(b, r, switch_fn=fn, t_from=2e-3)
    e = _entry(s, "INV__Dla")
    # 2 dead times of 1 us per 200 us period: 0.010 (+1 step each).
    assert 0.0095 < e["duty_conducting"] < 0.0115, e
    # Each dead time carries the inductor current I for t_d:
    # P = Σ t_d·(V_f·I + R_d·I²) / T. Measured 0.6542 W against 0.6515 W:
    # the step that flips the diode on is the trapezoid of its end
    # powers (an ideal transfer would be a step), 1 of ~11 steps per
    # dead time. V_f·i_avg + R_d·i_rms² reads 0.6393 W here because
    # i_avg / i_rms integrate that same step as a current ramp.
    t = np.asarray(r.times)
    i_d = np.asarray(r.i("INV__Dla"))
    i_l = np.asarray(r.i("L"))
    w = t >= 2e-3
    starts = np.flatnonzero(w[1:] & (i_d[1:] > 1e-3) & (i_d[:-1] <= 1e-3)) + 1
    i_dt = i_l[starts]
    P_ref = float(np.sum(1e-6 * (V_F * i_dt + 0.05 * i_dt * i_dt))) / (t[-1] - 2e-3)
    assert e["P_avg"] == pytest.approx(P_ref, rel=1e-2)
    # While LS is gated on, R_on·|i| (~1 mOhm · 30 A) < V_f: no share.
    ls = np.array([bool(fn(x).get(1)) for x in t])
    m = (t >= 2e-3) & ls
    m[1:] &= ls[:-1]
    assert np.abs(i_d[m]).max() < 1e-6
    # The old threshold semantics (V_f as V_th, no drop) shared the
    # current for the whole LS on-time.
    b0, fn0, r0 = _run_half_bridge(0.0)
    e0 = _entry(_loss(b0, r0, switch_fn=fn0, t_from=2e-3), "INV__Dla")
    assert e0["duty_conducting"] > 0.4, e0


# ---------------------------------------------------------------------
# 11. V_f = 0 is bit-identical to omitting it
# ---------------------------------------------------------------------

def _buck(**kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", g_on=1e3, g_off=1e-9)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9, **kw)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 5.0)
    return b


def _boost_dcm(**kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 10e-6)
    b.add_switch("S", "sw", "gnd", g_on=1e3, g_off=1e-9)
    b.add_diode("D", "sw", "vout", 1e3, 1e-9, **kw)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 200.0)
    return b


def _bridge_id(**kw):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "bn", 0.0, 20.0, 50.0)
    b.add_resistor("Rs", "s", "a", 0.1)
    p.add_bridge_rectifier(b, "BR", ac_a="a", ac_b="bn", dc_pos="p",
                           dc_neg="gnd", V_th=0.7, **kw)
    b.add_capacitor("C", "p", "gnd", 100e-6)
    b.add_resistor("RL", "p", "gnd", 100.0)
    return b


def _pwm(b, f=100e3, d=0.4):
    return p.make_pwm_switch_fn(f, d, b.switch_index_of("S"),
                                b.graph.num_switches)


_CIRCUITS = {
    "buck": (_buck, 1e-3, True),
    "boost_dcm": (_boost_dcm, 1e-3, True),
    "bridge": (_bridge_id, 0.03, False),
}


# P_avg of the switch and the diodes on the parent commit (8935b2a5,
# before V_f existed), with the diode built WITHOUT V_f. There, the
# whole record (times, states, switch states, every summary field) was
# bit-identical to V_f = 0.0 on this branch; boost_dcm on dsed refuses on
# both. Pinned as numbers, not hashes, so that other platforms pass:
# rel 1e-9 on the fixed grid, 1e-6 where the step sequence can move, and
# 1e-3 on dsed — its event times go through libm, and Windows CI moved
# these ~0.1 mW leakage losses by up to 9e-5. A lost V_f = 0 identity
# would move them by orders of magnitude, not parts per thousand.
_V_F0_REF = {
    ("boost_dcm", "auto"): {"S": 0.0452680383528, "D": 0.0649591561202},
    ("boost_dcm", "pwl"): {"S": 0.0452766957162, "D": 0.0649599299511},
    ("boost_dcm", "trbdf2"): {"S": 0.0452680383528, "D": 0.0649591561202},
    ("bridge", "auto"): {"BR__D1": 0.000138585164756, "BR__D2": 5.6813204064e-05,
                         "BR__D3": 0.000138585164754, "BR__D4": 5.68132040543e-05},
    ("bridge", "dsed"): {"BR__D1": 0.000138571769601, "BR__D2": 5.67576669568e-05,
                         "BR__D3": 0.000138582323524, "BR__D4": 5.67838168133e-05},
    ("bridge", "pwl"): {"BR__D1": 0.000138205386016, "BR__D2": 5.66352667669e-05,
                        "BR__D3": 0.000138205386014, "BR__D4": 5.66352667572e-05},
    ("bridge", "trbdf2"): {"BR__D1": 0.000138585164756, "BR__D2": 5.6813204064e-05,
                           "BR__D3": 0.000138585164754, "BR__D4": 5.68132040543e-05},
    ("buck", "auto"): {"S": 0.0035509287608, "D": 0.00533941662173},
    ("buck", "dsed"): {"S": 0.00354982022022, "D": 0.00533604800703},
    ("buck", "pwl"): {"S": 0.00354999944119, "D": 0.00533789852559},
    ("buck", "trbdf2"): {"S": 0.0035509287608, "D": 0.00533941662173},
}


@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "dsed", "auto"])
@pytest.mark.parametrize("circuit", sorted(_CIRCUITS))
def test_v_f_zero_matches_the_pre_v_f_engine(circuit, engine):
    make, t_end, switched = _CIRCUITS[circuit]
    b = make(V_f=0.0)
    skw = {"switch_fn": _pwm(b)} if switched else {}
    if engine == "pwl":
        skw["dt"] = 1e-7 if switched else 1e-6
    ref = _V_F0_REF.get((circuit, engine))
    if ref is None:  # refused before V_f too
        with pytest.raises(RuntimeError, match="did not settle"):
            _sim(b, t_end, engine, **skw)
        return
    r = _sim(b, t_end, engine, **skw)
    s = _loss(b, r, **({"switch_fn": skw["switch_fn"]} if switched else {}))
    got = {e["name"]: e["P_avg"] for e in s if e["name"] in ref}
    rel = {"pwl": 1e-9, "dsed": 1e-3}.get(engine, 1e-6)
    assert got == pytest.approx(ref, rel=rel)


# ---------------------------------------------------------------------
# 13. DSED band behaviour (a pinned limitation)
# ---------------------------------------------------------------------

def test_dsed_without_a_band_refuses_or_matches_and_never_reverses():
    """``V_th <= V_f`` leaves the Jacobi settle no band. Either the run
    refuses with the band hint, or it ran and then it must still obey
    the law (no reverse current) and agree with pwl."""
    b = _bridge(V_th=0.0)
    try:
        r = _sim(b, 0.1, "dsed")
    except Exception as exc:  # noqa: BLE001
        assert "V_th = V_f + 1e-3" in str(exc), exc
        return
    _, rp = _run_bridge("pwl")
    for d in ("BR__D1", "BR__D3"):
        assert _no_reverse(np.asarray(r.i(d))), d
    assert p.time_average(np.asarray(r.times), r.v("p"), t_from=0.08) \
        == pytest.approx(p.time_average(np.asarray(rp.times), rp.v("p"),
                                         t_from=0.08), rel=1e-3)


def test_dsed_forced_turn_on_without_a_band_names_the_band():
    """Pinned limitation: a capacitor charged by a current source into a
    V_th = 0, V_f = 0.7 diode reaches V_f with zero ON current, and the
    Jacobi settle cycles. Since add-engine-conformance-matrix dsed
    refuses V_f > V_th up front (it held the diode ON through reverse
    bias, 10.4 V off); the refusal must still name the band advice."""
    b = p.CircuitBuilder()
    b.add_current_source("I1", "a", "gnd", 10.0)
    b.add_diode("D1", "a", "gnd", G_ON, 1e-9, V_f=V_F)
    b.add_capacitor("C", "a", "gnd", 1e-6)
    with pytest.raises(p.UnsupportedError, match=r"V_th = V_f \+ 1e-3") as e:
        _sim(b, 2e-3, "dsed")
    assert (e.value.engine, e.value.feature, e.value.device) == \
        ("dsed", "V_f", "D1")


def test_dsed_t0_settle_with_a_drop():
    """A zero IC is strictly inside OFF when V_f > 0: the t = 0 settle
    of a forward-driven diode passes. dsed needs the band V_th > V_f
    (V_th = 0 is refused, see above); the drop is unchanged by it."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "s", "gnd", 12.0)
    b.add_resistor("R1", "s", "a", 10.0)
    b.add_diode("D", "a", "k", 1e3, 1e-9, V_th=V_F + 1e-3, V_f=V_F)
    b.add_capacitor("C", "k", "gnd", 10e-6)
    b.add_resistor("RL", "k", "gnd", 100.0)
    r = _sim(b, 5e-3, "dsed")
    # Analytic steady state: (12 − 0.7)·100/(100 + 10 + 1e-3).
    assert np.asarray(r.v("k"))[-1] == pytest.approx(
        11.3 * 100 / 110.001, rel=1e-4)
    assert _no_reverse(r.i("D"))


# ---------------------------------------------------------------------
# 14. Losses vs thermal, and windows that add up to the whole run
# ---------------------------------------------------------------------

def test_thermal_conduction_energy_equals_loss_energy():
    b, r = _run_bridge("pwl")
    stages = [p.FosterStage(R_th_K_per_W=1.0, tau_s=1e-3)]
    names = ("BR__D1", "BR__D3")
    s = _loss(b, r, t_from=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        th = p.device_thermal_summary(
            b, r, thermal_specs={n: {"stages": stages} for n in names},
            t_from=0.05)
    for n in names:
        e = _entry(s, n)
        assert e["P_avg"] > 0.01
        assert _entry(th, n)["P_cond_avg"] == pytest.approx(
            e["P_avg"], rel=1e-12)


@pytest.mark.parametrize("v_d, n_gaps", [(0.5, 0), (0.75, 1)])
def test_chatter_detector_measures_from_v_f(v_d, n_gaps):
    """A 2-step gap between two 1-step ON blips is chatter only if its
    step average does not block: with V_f = 0.7, V_th = 0 the diode at
    0.5 V blocks (a real gap), at 0.75 V it does not. Measured from
    v_D = 0 instead, 0.5 V would also read as chatter."""
    from types import SimpleNamespace

    from pulsim.losses import _chatter_gaps
    cond = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
    v = np.full(cond.size, v_d)
    tr = SimpleNamespace(conducting=cond, v=v, v_start=v, V_th=0.0,
                         V_f=V_F)
    assert _chatter_gaps(tr)[1] == n_gaps


def test_loss_windows_add_up_to_the_whole_run():
    b, r = _run_bridge("pwl")
    t = np.asarray(r.times)
    i = np.asarray(r.i("BR__D1"))
    # A bound inside the step that turns D1 on (a flip step, cut at the
    # interpolated power) and one inside a conducting step.
    k_on = int(np.flatnonzero((i[1:] > 1e-6) & (i[:-1] <= 1e-6))[2]) + 1
    bounds = [0.0, 0.5 * (t[k_on - 1] + t[k_on]),
              float(t[k_on + 500]) + 0.3e-6, float(t[-1])]
    whole = _entry(_loss(b, r), "BR__D1")
    parts = [_entry(_loss(b, r, t_from=x, t_to=y), "BR__D1")
             for x, y in zip(bounds[:-1], bounds[1:])]
    assert sum(e["E_total"] for e in parts) == pytest.approx(
        whole["E_total"], rel=1e-12)


def test_window_offset_branch_matches_exact_integral():
    """Same-stamp ON step: E = h·[V_f·i_mean + (ia²+ia·ic+ic²)/(3g)],
    exact for a linear current (analytic)."""
    from pulsim.losses import _branch_energy
    t = np.array([0.0, 1.0])
    v = np.array([1.2, 1.7])          # i: 10 -> 20 A at g = 20, V_f 0.7
    g = np.array([20.0, 20.0])
    off = np.array([0.7, 0.7])
    E = _branch_energy(t, v, g, v, g, off, off)
    # ∫0^1 (0.7 + i/20)·i dt, i = 10 + 10 t
    exact = 0.7 * 15.0 + (100 + 200 + 400) / 3.0 / 20.0
    assert E == pytest.approx(exact, rel=1e-14)


# ---------------------------------------------------------------------
# 15. Plumbing: components(), YAML, refusals
# ---------------------------------------------------------------------

def test_components_carry_v_f():
    b = p.CircuitBuilder()
    b.add_diode("D1", "a", "gnd", 1e3, 1e-9, V_th=0.75, V_f=0.7)
    prm = next(c for c in b.components() if c["name"] == "D1")["params"]
    assert prm["V_f"] == 0.7 and prm["V_th"] == 0.75


_YAML = """\
circuit:
  devices:
    - {{type: current_source, name: I1, from: a, to: gnd, I: 10.0}}
    - {{type: diode, name: D1, anode: a, cathode: gnd,
       g_on: 20.0, g_off: 1.0e-9{extra}}}
simulation:
  t_start: 0.0
  t_end: 1.0e-5
  dt: 1.0e-6
"""


def test_yaml_v_f_key_changes_the_result():
    with_drop = p.load_yaml_string(_YAML.format(extra=", V_f: 0.7"))
    without = p.load_yaml_string(_YAML.format(extra=""))
    prm = next(c for c in with_drop.builder.components()
               if c["name"] == "D1")["params"]
    assert prm["V_f"] == 0.7
    r1 = _sim(with_drop.builder, 1e-5, "pwl", dt=1e-6)
    r0 = _sim(without.builder, 1e-5, "pwl", dt=1e-6)
    assert np.asarray(r1.v("a"))[-1] == pytest.approx(1.2, rel=1e-12)
    assert np.asarray(r0.v("a"))[-1] == pytest.approx(0.5, rel=1e-12)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_invalid_v_f_is_refused_by_name(bad):
    b = p.CircuitBuilder()
    with pytest.raises(Exception, match=r"D_bad.*V_f|V_f.*D_bad"):
        b.add_diode("D_bad", "a", "gnd", 1e3, 1e-9, V_f=bad)


def test_invalid_v_f_is_refused_from_yaml():
    with pytest.raises(Exception, match=r"D1"):
        p.load_yaml_string(_YAML.format(extra=", V_f: -1.0"))
