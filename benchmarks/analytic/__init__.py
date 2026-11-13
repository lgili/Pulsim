"""Analytic reference set: 20 circuits with a closed-form (or exact
piecewise-LTI) solution, one per entry of :data:`CASES`.

Each :class:`Case` has ``build(k)`` -> ``(builder, switch_fn)`` where ``k``
scales ONE component value (``k = 1.01`` is the non-vacuity check),
``observe(res)`` -> samples, ``exact(t)`` and the per-engine run options.
Tolerance ceilings live in ``benchmarks/manifests/analytic.yaml``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import lambertw

import pulsim

TRBDF2 = {"rtol": 1e-6}
DSED: dict = {}


@dataclass
class Case:
    id: str
    title: str
    build: Callable
    observe: Callable
    exact: Callable
    t_end: float
    dt: float
    engines: Dict[str, dict]
    abs_scale: float = 0.0         # floor of the span, for flat references
    exclude_t0: bool = False       # algebraic observable (design.md, Context)
    checks: Dict[str, float] = field(default_factory=dict)  # self-checks of the closed form

    def run(self, engine: str, k: float = 1.0):
        b, sw = self.build(k)
        key = "dsed" if engine == "exact" else engine   # 'exact' = 'dsed'
        kw = dict(self.engines.get(key, ALL[key]))
        if engine == "pwl":
            kw["dt"] = self.dt
        if sw is not None:
            kw["switch_fn"] = sw
        return pulsim.simulate(b, self.t_end, engine=engine, **kw)


def _v(node):
    return lambda r: np.asarray(r.v(node), dtype=float)


def _i(name):
    return lambda r: np.asarray(r.i(name), dtype=float)


ALL = {"pwl": {}, "trbdf2": TRBDF2, "dsed": DSED}
NO_DSED = {"pwl": {}, "trbdf2": TRBDF2}


def _b():
    return pulsim.CircuitBuilder()


# ------------------------------------------------------------ 1-8: R, L, C --

def _rc(k):
    b = _b()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "o", 1e3 * k)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    return b, None


def _rc_dis(k):
    b = _b()
    b.add_capacitor("C", "o", "gnd", 1e-6, 5.0)
    b.add_resistor("R", "o", "gnd", 1e3 * k)
    return b, None


def _rl(k):
    b = _b()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "a", 10.0 * k)
    b.add_inductor("L", "a", "gnd", 1e-3)
    return b, None


def _rlc(R):
    def build(k):
        b = _b()
        b.add_voltage_source("V", "in", "gnd", 10.0)
        b.add_resistor("R", "in", "a", R)
        b.add_inductor("L", "a", "o", 1e-3)
        b.add_capacitor("C", "o", "gnd", 10e-6 * k)
        return b, None
    return build


_A4, _W4 = 500.0, math.sqrt(1 / (1e-3 * 10e-6) - 500.0 ** 2)
_A5, _W0 = 150000.0, 1e4
_S1, _S2 = -_A5 + math.sqrt(_A5 ** 2 - _W0 ** 2), -_A5 - math.sqrt(_A5 ** 2 - _W0 ** 2)


def _lc(k):
    b = _b()
    b.add_capacitor("C", "o", "gnd", 10e-6, 10.0)
    b.add_inductor("L", "o", "gnd", 1e-3 * k)
    return b, None


_W7, _T7 = 2 * math.pi * 1e3, 1e-3
_P7 = math.atan(_W7 * _T7)


def _sine_rc(k):
    b = _b()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
    b.add_resistor("R", "in", "o", 1e3 * k)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    return b, None


_Z8 = math.hypot(10.0, _W7 * 10e-3)
_P8 = math.atan(_W7 * 10e-3 / 10.0)


def _sine_rl(k):
    b = _b()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
    b.add_resistor("R", "in", "a", 10.0 * k)
    b.add_inductor("L", "a", "gnd", 10e-3)
    return b, None


# ------------------------------------------------- 9-13: resistive, sources --

def _bridge(k):
    b = _b()
    b.add_voltage_source("V", "vcc", "gnd", 10.0)
    b.add_resistor("R1", "vcc", "a", 100.0 * k)
    b.add_resistor("R2", "a", "gnd", 200.0)
    b.add_resistor("R3", "vcc", "b", 150.0)
    b.add_resistor("R4", "b", "gnd", 100.0)
    return b, None


# pulse: 0 -> 10 V, delay 0.5 ms, rise/fall 10 us, width 2 ms, period 5 ms
_PULSE = (0.0, 10.0, 0.5e-3, 2e-3, 5e-3, 10e-6, 10e-6)


def _pulse_rc(k):
    b = _b()
    b.add_pulse_voltage_source("V", "in", "gnd", *_PULSE)
    b.add_resistor("R", "in", "o", 1e3 * k)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    return b, None


def _pulse_rc_exact(t, tau=1e-3, t_end=10e-3):
    """RC response to a piecewise-linear input: superposition of ramps."""
    v0, v1, td, pw, per, tr, tf = _PULSE
    kinks = []  # (time, slope change)
    s = (v1 - v0)
    for n in range(int(t_end / per) + 1):
        t0 = td + n * per
        kinks += [(t0, s / tr), (t0 + tr, -s / tr), (t0 + tr + pw, -s / tf), (t0 + tr + pw + tf, s / tf)]
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t)
    for tk, m in kinks:
        x = np.clip(t - tk, 0.0, None)
        y += m * (x - tau * (1.0 - np.exp(-x / tau)))
    return y


def _coupled(k):
    b = _b()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("Rs", "in", "p", 10.0 * k)
    b.add_transformer("T", "p", "gnd", "s", "gnd", 1e-3, 1e-3, 0.99)
    b.add_resistor("RL", "s", "gnd", 10.0)
    return b, None


def _coupled_exact(t):
    Lp = Ls = 1e-3
    M = 0.99 * 1e-3
    Lm = np.array([[Lp, M], [M, Ls]])
    A = np.linalg.solve(Lm, -np.diag([10.0, 10.0]))
    B = np.linalg.solve(Lm, np.array([10.0, 0.0]))
    xs = np.linalg.solve(A, -B)  # steady state
    out = np.array([(expm(A * ti) @ (-xs) + xs)[1] for ti in np.atleast_1d(t)])
    return -10.0 * out  # v(s) = -RL * i_s (winding current s_from -> s_to)


def _ideal_tr(k):
    b = _b()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
    b.add_resistor("Rs", "in", "p", 1.0 * k)
    b.add_ideal_transformer("T", "p", "gnd", "s", "gnd", 2.0)
    b.add_resistor("RL", "s", "gnd", 8.0)
    return b, None


def _vcvs(k):
    b = _b()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 1.0, 1e3)
    b.add_resistor("Ri", "in", "gnd", 1e3)
    b.add_vcvs("E", "in", "gnd", "o", "gnd", 2.0 * k)
    b.add_resistor("R", "o", "gnd", 10.0)
    return b, None


# ---------------------------------------------------------- 14-17: diodes --

_VT = 0.025852


def _shockley(k):
    b = _b()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_resistor("R", "in", "a", 1e3 * k)
    b.add_shockley_diode("D", "a", "gnd", I_S=1e-12, n=1.0, V_T=_VT)
    return b, None


def _shockley_vd(V=5.0, R=1e3, Is=1e-12, nvt=_VT):
    i = nvt / R * lambertw(Is * R / nvt * math.exp((V + Is * R) / nvt)).real - Is
    return V - i * R


def _pwl_diode(k):
    b = _b()
    b.add_current_source("I", "a", "gnd", 10.0)  # 10 A out of `a` into the diode
    b.add_diode("D", "a", "gnd", 20.0 * k, 1e-9, V_f=0.7)
    return b, None


def _hw_r(k):
    b = _b()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
    b.add_diode("D", "in", "o", 100.0, 1e-9, V_f=0.7)
    b.add_resistor("R", "o", "gnd", 10.0 * k)
    return b, None


def _hw_r_exact(t):
    vin = 10.0 * np.sin(2 * np.pi * 1e3 * np.asarray(t))
    return np.maximum(0.0, (vin - 0.7) * 10.0 / (10.0 + 0.01))


_R17, _L17 = 10.0 + 1e-3, 1e-3   # load + diode on-resistance (g_on = 1e3)
_Z17 = math.hypot(_R17, _W7 * _L17)
_P17 = math.atan(_W7 * _L17 / _R17)


def _hw_rl(k):
    b = _b()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
    b.add_diode("D", "in", "a", 1e3, 1e-9)
    b.add_resistor("R", "a", "b", 10.0 * k)
    b.add_inductor("L", "b", "gnd", _L17)
    return b, None


def _hw_rl_i(th):
    return 10.0 / _Z17 * (np.sin(th - _P17) + math.sin(_P17) * np.exp(-th / math.tan(_P17)))


_BETA17 = brentq(_hw_rl_i, math.pi, 2 * math.pi - 1e-9)  # extinction angle


def _hw_rl_exact(t):
    th = np.mod(2 * np.pi * 1e3 * np.asarray(t), 2 * np.pi)
    return np.where(th <= _BETA17, _hw_rl_i(th), 0.0)


# ------------------------------------------------- 18-20: switched exact ---

_F18 = 100e3
_T18 = 1 / _F18


def _caploop(k):
    b = _b()
    b.add_voltage_source("V", "in", "gnd", 100.0 * k)
    b.add_resistor("R", "in", "sw", 10.0)
    b.add_switch("S", "sw", "gnd", 1e3, 1e-9)
    b.add_capacitor("C", "sw", "gnd", 1e-9)
    n = b.graph.num_switches
    # open for the first half period, closed for the second
    return b, pulsim.make_pwm_switch_fn(_F18, 0.5, b.switch_index_of("S"), n, _T18 / 2)


def _caploop_exact(t, R=10.0, C=1e-9, V=100.0, gon=1e3, goff=1e-9):
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    v = 0.0
    j = 0
    edges = np.arange(0.0, t[-1] + _T18, _T18 / 2)
    for n, ta in enumerate(edges):
        g = gon if n % 2 else goff
        tau = C / (1 / R + g)
        vinf = V / R / (1 / R + g)
        tb = ta + _T18 / 2
        while j < len(t) and t[j] < tb:
            out[j] = vinf + (v - vinf) * math.exp(-(t[j] - ta) / tau)
            j += 1
        v = vinf + (v - vinf) * math.exp(-(tb - ta) / tau)
    return out


_VIN, _LB, _CB, _GON, _GOFF, _FSW = 24.0, 100e-6, 47e-6, 1e3, 1e-9, 100e3
_TB = 1 / _FSW


def _buck(R, D):
    def build(k):
        b = _b()
        b.add_voltage_source("Vin", "vin", "gnd", _VIN)
        b.add_switch("S", "vin", "sw", _GON, _GOFF)
        b.add_diode("D1", "gnd", "sw", _GON, _GOFF)
        b.add_inductor("L", "sw", "vout", _LB * k)
        b.add_capacitor("C", "vout", "gnd", _CB)
        b.add_resistor("RL", "vout", "gnd", R)
        return b, pulsim.make_pwm_switch_fn(_FSW, D, b.switch_index_of("S"), b.graph.num_switches)
    return build


class _PiecewiseLTI:
    """Exact buck trajectory from rest: x = [i_L, v_C], one 3x3 matrix
    exponential per segment, diode events located with brentq (the method of
    the 2026-11 numerics audit, core_review_2026-11/numerics/p2_buck.py)."""

    def __init__(self, R, D, t_end):
        self.R = R
        segs = []  # (t_start, M) with x(t) = prop(M, x_start, t - t_start)
        x, t = np.zeros(2), 0.0
        n = 0
        while t < t_end:
            for gs, tb in ((_GON, n * _TB + D * _TB), (_GOFF, (n + 1) * _TB)):
                while t < tb - 1e-18:
                    on = self._vsw(x, gs, _GOFF) <= 0.0  # diode forward-biased
                    gd = _GON if on else _GOFF
                    M = self._aug(gs, gd)
                    def f(h, M=M, x=x, gd=gd, gs=gs):  # diode voltage after h
                        return -self._vsw(self._prop(M, x, h), gs, gd)
                    hs = np.linspace(0.0, tb - t, 65)[1:]
                    fv = np.array([f(h) for h in hs])
                    cross = np.nonzero((fv < 0) if on else (fv > 0))[0]
                    h = tb - t
                    if cross.size:
                        lo = hs[cross[0] - 1] if cross[0] else 1e-15
                        h = brentq(f, lo, hs[cross[0]], xtol=1e-18, rtol=1e-15)
                    segs.append((t, x.copy(), M))
                    x, t = self._prop(M, x, h), t + h
                    if cross.size:
                        # past the event the other diode state holds until tb
                        M2 = self._aug(gs, _GOFF if on else _GON)
                        segs.append((t, x.copy(), M2))
                        x, t = self._prop(M2, x, tb - t), tb
                t = tb
            n += 1
        self.segs = segs
        self.starts = np.array([s[0] for s in segs])

    @staticmethod
    def _vsw(x, gs, gd):
        return (gs * _VIN - x[0]) / (gs + gd)

    def _aug(self, gs, gd):
        k = 1 / (gs + gd)
        M = np.zeros((3, 3))
        M[:2, :2] = [[-k / _LB, -1 / _LB], [1 / _CB, -1 / (self.R * _CB)]]
        M[:2, 2] = [gs * _VIN * k / _LB, 0.0]
        return M

    @staticmethod
    def _prop(M, x, h):
        E = expm(M * h)
        return E[:2, :2] @ x + E[:2, 2]

    def __call__(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        j = np.clip(np.searchsorted(self.starts, t, side="right") - 1, 0, len(self.segs) - 1)
        return np.array([self._prop(self.segs[k][2], self.segs[k][1], ti - self.segs[k][0])[1]
                         for k, ti in zip(j, t)])


_EXACT_CACHE: dict = {}


def _buck_exact(R, D, t_end):
    def f(t):
        key = (R, D)
        if key not in _EXACT_CACHE:
            _EXACT_CACHE[key] = _PiecewiseLTI(R, D, t_end)
        return _EXACT_CACHE[key](t)
    return f


# ------------------------------------------------------------- the table ---

CASES = [
    Case("rc_charge", "RC charge, 10 V, 1 kOhm, 1 uF", _rc, _v("o"),
         lambda t: 10 * (1 - np.exp(-np.asarray(t) / 1e-3)), 5e-3, 1e-5, ALL),
    Case("rc_discharge", "RC discharge from 5 V", _rc_dis, _v("o"),
         lambda t: 5 * np.exp(-np.asarray(t) / 1e-3), 5e-3, 1e-5, ALL),
    Case("rl_step", "RL step, 10 V, 10 Ohm, 1 mH", _rl, _i("L"),
         lambda t: 1.0 * (1 - np.exp(-np.asarray(t) / 1e-4)), 1e-3, 1e-6, ALL),
    Case("rlc_underdamped", "series RLC, 1 Ohm, 1 mH, 10 uF", _rlc(1.0), _v("o"),
         lambda t: 10 * (1 - np.exp(-_A4 * np.asarray(t)) * (np.cos(_W4 * np.asarray(t))
                                                              + _A4 / _W4 * np.sin(_W4 * np.asarray(t)))),
         10e-3, 1e-6, ALL,
         checks={"peak": 10 * (1 + math.exp(-_A4 * math.pi / _W4))}),
    Case("rlc_overdamped", "series RLC, 300 Ohm", _rlc(300.0), _v("o"),
         lambda t: 10 * (1 - (_S2 * np.exp(_S1 * np.asarray(t)) - _S1 * np.exp(_S2 * np.asarray(t)))
                         / (_S2 - _S1)), 15e-3, 1e-6, ALL),
    Case("lc_lossless", "LC tank from 10 V", _lc, _v("o"),
         lambda t: 10 * np.cos(_W0 * np.asarray(t)), 2e-3, 1e-6, ALL),
    Case("sine_rc", "sine-driven RC from rest", _sine_rc, _v("o"),
         lambda t: 10 / math.sqrt(1 + (_W7 * _T7) ** 2) * (np.sin(_W7 * np.asarray(t) - _P7)
                                                           + math.sin(_P7) * np.exp(-np.asarray(t) / _T7)),
         5e-3, 1e-6, ALL),
    Case("sine_rl", "sine-driven series RL from rest", _sine_rl, _i("L"),
         lambda t: 10 / _Z8 * (np.sin(_W7 * np.asarray(t) - _P8)
                               + math.sin(_P8) * np.exp(-10.0 / 10e-3 * np.asarray(t))),
         5e-3, 1e-6, ALL),
    Case("wheatstone", "Wheatstone bridge (bridge_resistive_pure)", _bridge, _v("a"),
         lambda t: np.full_like(np.asarray(t, dtype=float), 10 * 200 / 300), 1e-3, 5e-6, NO_DSED,
         abs_scale=10.0, exclude_t0=True, checks={"V(a)": 10 * 200 / 300, "V(b)": 10 * 100 / 250}),
    Case("pulse_rc", "pulse-driven RC", _pulse_rc, _v("o"), _pulse_rc_exact, 10e-3, 1e-5, ALL),
    Case("coupled_k099", "coupled inductors k=0.99, step", _coupled, _v("s"), _coupled_exact,
         2e-3, 1e-7, NO_DSED, exclude_t0=True),
    Case("ideal_transformer", "ideal transformer n=2, Rs, RL", _ideal_tr, _v("s"),
         lambda t: 2 * 10 * np.sin(_W7 * np.asarray(t)) * 2.0 / (1.0 + 2.0), 2e-3, 1e-6, NO_DSED,
         exclude_t0=True),
    Case("vcvs_gain2", "VCVS gain 2, sine input", _vcvs, _v("o"),
         lambda t: 2 * np.sin(_W7 * np.asarray(t)), 2e-3, 1e-6, NO_DSED, exclude_t0=True),
    Case("shockley_op", "Shockley diode + 1 kOhm from 5 V (Lambert W)", _shockley, _v("a"),
         lambda t: np.full_like(np.asarray(t, dtype=float), _shockley_vd()), 1e-4, 1e-6, NO_DSED,
         abs_scale=5.0, exclude_t0=True),
    Case("pwl_diode_drop", "PWL diode, 10 A forced, g_on 20 S, V_f 0.7 V", _pwl_diode, _v("a"),
         lambda t: np.full_like(np.asarray(t, dtype=float), 0.7 + 10 / 20), 1e-4, 1e-6, NO_DSED,
         abs_scale=1.2, exclude_t0=True, checks={"V": 1.2, "P": 12.0}),
    Case("halfwave_r", "half-wave rectifier, R load, V_f 0.7 V", _hw_r, _v("o"), _hw_r_exact,
         3e-3, 1e-6, NO_DSED, exclude_t0=True),
    Case("halfwave_rl", "half-wave rectifier, RL load", _hw_rl, _i("L"), _hw_rl_exact,
         3e-3, 1e-6, ALL),
    Case("hard_switched_rc", "hard-switched RC (p8_caploop)", _caploop, _v("sw"), _caploop_exact,
         5 * _T18, 1e-9, ALL),
    Case("buck_ccm", "buck CCM, R 1 Ohm, D 0.5, from rest", _buck(1.0, 0.5), _v("vout"),
         _buck_exact(1.0, 0.5, 40 * _TB), 40 * _TB, _TB / 200, ALL),
    Case("buck_dcm", "buck DCM, R 50 Ohm, D 0.3, from rest", _buck(50.0, 0.3), _v("vout"),
         _buck_exact(50.0, 0.3, 40 * _TB), 40 * _TB, _TB / 200, ALL),
]

BY_ID = {c.id: c for c in CASES}
