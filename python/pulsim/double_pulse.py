"""Clamped inductive double-pulse test (add-power-mosfet-datasheet-device).

    r = pulsim.double_pulse_test(
        dict(K=1.43, V_T=3.0, C_gs=1e-9, C_gd=100e-12, C_ds=0.0),
        V_dd=400.0, I_load=10.0, R_g_ext=10.0, V_drive=12.0, L_load=500e-6)
    r.E_on, r.E_off, r.t_plateau

The fixture is the one of the review probe ``p4_dpt.py``:

    Vdc ─┬─ L_load (i0 = I_load) ─ sw ─ R_sense 1 mΩ ─ [L_stray] ─ D (DUT)
         └──────── freewheel diode (sw → Vdc) ───┘          S ─ gnd
    drv ─ R_g_ext ─ G (DUT);  drv: 0 → V_drive at t_on, back at
    t_on + t_edge + t_pulse (5 ns edges)

The run starts from the off-state bias (``initial_state``): the load
inductor at ``I_load`` through the conducting freewheel, the drain at
V_dd + V_f, the gate at 0 and every device charge seeded from it. So
the first pulse is the SECOND pulse of a bench test: the DUT turns on
into the full current and turns it off again. Energies are ``np.trapezoid`` of v_DS·i_D on the
result's own grid (time-weighted, so a uniform ``pwl`` grid and an
adaptive ``trbdf2`` grid give the same number), with IEC 60747-8 style
windows:

* E_on: from the gate command ``t_on`` until v_DS first falls to 2 % of
  V_dd;
* E_off: from the turn-off command until i_D first falls to 2 % of
  I_load (if it never does, to the end of the run);
* plateau: time the DUT's internal gate spends within ±5 % of
  V_plateau, the gate voltage when v_DS crosses V_dd/2 during turn-on;
* dv/dt and di/dt: 90 %→10 % and 10 %→90 % crossings (mean slopes);
* I_rr: peak i_D during turn-on minus I_load (the freewheel diode's
  reverse-recovery peak).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

__all__ = ["DoublePulseResult", "double_pulse_metrics", "double_pulse_test"]

_R_SENSE = 1e-3


@dataclass
class DoublePulseResult:
    E_on: float
    E_off: float
    t_plateau: float
    V_plateau: float
    dv_dt_on: float
    dv_dt_off: float
    di_dt_on: float
    di_dt_off: float
    I_rr: float
    engine: str
    n_points: int
    waveforms: dict = field(repr=False, default_factory=dict)


def _first_crossing(t, y, level, start, falling):
    """Linear-interpolated first time >= start where y crosses level."""
    k0 = int(np.searchsorted(t, start))
    seg = y[k0:]
    hit = np.nonzero(seg <= level if falling else seg >= level)[0]
    if hit.size == 0:
        return None
    k = k0 + int(hit[0])
    if k == 0 or k == k0:
        return float(t[k])
    y0, y1 = y[k - 1], y[k]
    return float(t[k - 1] + (level - y0) * (t[k] - t[k - 1]) / (y1 - y0))


def _energy(t, p, t0, t1):
    if t1 is None or t1 <= t0:
        return 0.0
    m = (t >= t0) & (t <= t1)
    tt = np.concatenate(([t0], t[m], [t1]))
    pp = np.interp(tt, t, p)
    return float(np.trapezoid(pp, tt))


def double_pulse_test(device: Any, *, V_dd: float, I_load: float, R_g_ext: float,
                      V_drive: float, L_load: float, L_stray: float = 0.0,
                      freewheel: Optional[Mapping[str, Any]] = None, T_j: float = 25.0,
                      engine: str = "trbdf2", t_on: float = 100e-9,
                      t_pulse: float = 700e-9, t_tail: float = 700e-9,
                      t_edge: float = 5e-9, dt: float = 2e-10) -> DoublePulseResult:
    """Run the clamped inductive double-pulse test on ``device``.

    ``device`` is a mapping of :meth:`CircuitBuilder.add_power_mosfet`
    keyword arguments, or an object with ``add_to(builder, name, drain,
    source, gate, T_j=...)`` (a :class:`pulsim.lib.Part` with a
    ``power_mosfet`` block). ``freewheel`` is None (Shockley junction,
    defaults) or ``{"model": "shockley"|"lauritzen", ...}`` with the
    keyword arguments of ``add_shockley_diode`` / ``add_lauritzen_diode``.
    ``dt`` is used on ``engine="pwl"`` only; it must resolve the gate
    transient (0.2 ns for a 60 ns plateau). See the module docstring for
    the measurement windows.
    """
    import pulsim as _p

    if V_dd <= 0 or I_load <= 0 or L_load <= 0 or V_drive <= 0:
        raise ValueError("double_pulse_test: V_dd, I_load, L_load and V_drive must be positive.")
    if R_g_ext < 0 or L_stray < 0:
        raise ValueError("double_pulse_test: R_g_ext and L_stray must be >= 0.")
    b = _p.CircuitBuilder()
    b.add_voltage_source("Vdc", "vdc", "gnd", V_dd)
    b.add_inductor("Lload", "vdc", "sw", L_load, i0=I_load)
    fw = dict(freewheel or {"model": "shockley"})
    model = fw.pop("model", None)
    if model == "shockley":
        b.add_shockley_diode("Dfw", "sw", "vdc", **fw)
    elif model == "lauritzen":
        b.add_lauritzen_diode("Dfw", "sw", "vdc", **fw)
    else:
        raise ValueError(
            f"double_pulse_test: freewheel model {model!r}; expected 'shockley' or 'lauritzen'.")
    b.add_resistor("Rsense", "sw", "x" if L_stray > 0 else "d", _R_SENSE)
    if L_stray > 0:
        b.add_inductor("Lstray", "x", "d", L_stray)
    b.add_pulse_voltage_source("Vg", "drv", "gnd", 0.0, V_drive, t_on, t_pulse,
                               0.0, t_edge, t_edge)
    gate = "g" if R_g_ext > 0 else "drv"
    if R_g_ext > 0:
        b.add_resistor("Rg_ext", "drv", "g", R_g_ext)
    if isinstance(device, Mapping):
        kw = dict(device)
        kw["T_j"] = T_j
        b.add_power_mosfet("DUT", "d", "gnd", gate, **kw)
        R_g_int = float(kw.get("R_g", 0.0))
    elif hasattr(device, "add_to"):
        device.add_to(b, "DUT", "d", "gnd", gate, T_j=T_j)
        R_g_int = float(device.power_mosfet_kwargs().get("R_g", 0.0))
    else:
        raise TypeError(
            "double_pulse_test: device must be a mapping of add_power_mosfet "
            "keyword arguments or a part with add_to().")
    gate_int = "DUT.g" if R_g_int > 0 else gate

    # Consistent start: the freewheel carries I_load, the drain sits at
    # V_dd + V_f and every device charge is seeded from that bias. From an
    # all-zero state the drain charges through C_gd and turns the DUT on
    # before t_on (a 5.6 V Miller false turn-on, measured 2026-11-12).
    V_f = fw.get("n", 1.0) * fw.get("V_T", 0.025852) * np.log1p(I_load / fw.get("I_S", 1e-12))
    x0 = np.array(b.initial_state(), dtype=float).ravel()
    names = b.state_var_names()
    for node, v in (("vdc", V_dd), ("sw", V_dd + V_f), ("x", V_dd + V_f),
                    ("d", V_dd + V_f), ("DUT.d", V_dd + V_f)):
        if f"V({node})" in names:
            x0[names.index(f"V({node})")] = v

    t_end = t_on + t_edge + t_pulse + t_tail
    kw_sim = dict(dt=dt) if engine == "pwl" else {}
    r = _p.simulate(b, t_end=t_end, engine=engine, initial_state=x0, **kw_sim)
    t = np.asarray(r.times, dtype=float)
    vd = np.asarray(r.v("d"), dtype=float)
    vg = np.asarray(r.v(gate_int), dtype=float)
    vs = np.asarray(r.v("x" if L_stray > 0 else "d"), dtype=float)
    i = (np.asarray(r.v("sw"), dtype=float) - vs) / _R_SENSE
    m = double_pulse_metrics(t, vd, i, vg, V_dd=V_dd, I_load=I_load, t_on=t_on,
                             t_off=t_on + t_edge + t_pulse)
    return DoublePulseResult(**m, engine=engine, n_points=int(t.size),
                             waveforms={"t": t, "v_ds": vd, "i_d": i, "v_gs": vg})


def double_pulse_metrics(t, v_ds, i_d, v_gs, *, V_dd: float, I_load: float, t_on: float,
                         t_off: float) -> dict:
    """The module docstring's measurements on any waveform set (Pulsim's or
    an external simulator's), with ``t_on`` / ``t_off`` the gate commands."""
    t, vd, i, vg = (np.asarray(x, dtype=float) for x in (t, v_ds, i_d, v_gs))
    p = vd * i
    t_on_end = _first_crossing(t, vd, 0.02 * V_dd, t_on, falling=True)
    t_off_end = _first_crossing(t, i, 0.02 * I_load, t_off, falling=True)
    E_on = _energy(t, p, t_on, t_on_end)
    E_off = _energy(t, p, t_off, t_off_end if t_off_end is not None else float(t[-1]))

    def slope(y, a, b_, start, falling, stop):
        t1 = _first_crossing(t, y, a, start, falling)
        t2 = _first_crossing(t, y, b_, start, falling)
        if t1 is None or t2 is None or t2 <= t1 or t2 > stop:
            return float("nan")
        return abs(b_ - a) / (t2 - t1)

    dv_on = slope(vd, 0.9 * V_dd, 0.1 * V_dd, t_on, True, t_off)
    dv_off = slope(vd, 0.1 * V_dd, 0.9 * V_dd, t_off, False, t[-1])
    di_on = slope(i, 0.1 * I_load, 0.9 * I_load, t_on, False, t_off)
    di_off = slope(i, 0.9 * I_load, 0.1 * I_load, t_off, True, t[-1])

    on = (t >= t_on) & (t < t_off)
    t_half = _first_crossing(t, vd, 0.5 * V_dd, t_on, falling=True)
    if t_half is None:
        V_pl, t_pl = float("nan"), float("nan")
    else:
        V_pl = float(np.interp(t_half, t, vg))
        band = on & (np.abs(vg - V_pl) <= 0.05 * V_pl)
        t_pl = float(np.sum(np.diff(t)[band[:-1]]))
    I_rr = float(max(np.max(i[on]) - I_load, 0.0)) if np.any(on) else 0.0
    return dict(E_on=E_on, E_off=E_off, t_plateau=t_pl, V_plateau=V_pl,
                dv_dt_on=dv_on, dv_dt_off=dv_off, di_dt_on=di_on, di_dt_off=di_off,
                I_rr=I_rr)
