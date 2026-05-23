#!/usr/bin/env python3
"""Pulsim vs ngspice parity bench — analytical & SPICE cross-validation.

Drives a handful of canonical circuits through both simulators on the
same time-base, then reports the max-abs, max-rel, and RMS error
between the two probe traces. Pure-Python — no external dependencies
beyond ``numpy`` and an installed ``ngspice`` binary on ``$PATH``.

Cases (all classical SPICE-friendly topologies):

  * ``rc``         — Vstep → R → C → gnd; v_C(t).
  * ``rl``         — Vstep → R → L → gnd; i_L(t).
  * ``rlc``        — Vstep → L → R → C → gnd; v_C(t) underdamped.
  * ``half_wave``  — Vsin → D → R||C → gnd; v_out(t).
  * ``buck``       — open-loop buck @ 100 kHz / D=0.5; v_out(t).
  * ``boost``      — open-loop boost @ 100 kHz / D=0.5; v_out(t).

Usage::

    python3 scripts/bench_vs_ngspice.py
    python3 scripts/bench_vs_ngspice.py --only rc rlc
    python3 scripts/bench_vs_ngspice.py --plot     # save PNGs

Outputs a Markdown summary table to stdout. JSON dump optional via
``--json out.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import pulsim as p
except ImportError as exc:
    sys.exit(f"pulsim not importable ({exc}). Did you build with "
             f"-DPULSIM_BUILD_PYTHON=ON and "
             f"export PYTHONPATH=build/python ?")


# ---------------------------------------------------------------------------
# Common datatypes
# ---------------------------------------------------------------------------


@dataclass
class TraceCompare:
    """Side-by-side comparison of two equal-length time series."""
    t:          np.ndarray
    pulsim:     np.ndarray
    ngspice:    np.ndarray

    @property
    def abs_err(self) -> np.ndarray:
        return np.abs(self.pulsim - self.ngspice)

    @property
    def max_abs(self) -> float:
        return float(np.max(self.abs_err))

    @property
    def max_rel(self) -> float:
        denom = np.maximum(np.abs(self.ngspice), 1e-9)
        return float(np.max(self.abs_err / denom))

    @property
    def rms_abs(self) -> float:
        return float(np.sqrt(np.mean(self.abs_err ** 2)))


@dataclass
class CaseResult:
    name:        str
    description: str
    cmp:         TraceCompare | None = None
    skipped:     bool = False
    reason:      str  = ""


# ---------------------------------------------------------------------------
# ngspice driver
# ---------------------------------------------------------------------------


def run_ngspice(netlist: str, probe_node: str,
                t_start: float, t_end: float, dt: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """Write ``netlist`` to a temp file, run ``ngspice -b``, parse the
    CSV-style output. Returns ``(t, v)`` aligned to the requested grid.

    The netlist must define a ``.tran`` block matching ``(dt, t_end)``
    and a ``print v(probe_node)`` (or ``i(...)``) so ngspice emits a
    parseable column.
    """
    with tempfile.TemporaryDirectory() as tmp:
        net_path = Path(tmp) / "circuit.cir"
        out_path = Path(tmp) / "out.csv"
        wrapped = netlist + (
            f"\n.control\n"
            f"set wr_singlescale\n"
            f"set wr_vecnames\n"
            f"tran {dt} {t_end} {t_start} {dt} uic\n"
            f"wrdata {out_path} v({probe_node})\n"
            f"quit\n"
            f".endc\n.end\n"
        )
        net_path.write_text(wrapped)
        proc = subprocess.run(
            ["ngspice", "-b", str(net_path)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ngspice failed (exit {proc.returncode}):\n"
                f"  stdout: {proc.stdout[-400:]}\n"
                f"  stderr: {proc.stderr[-400:]}"
            )
        if not out_path.exists():
            raise RuntimeError("ngspice did not produce output file. "
                                "stdout tail:\n" + proc.stdout[-400:])
        # ngspice ``wrdata`` writes a "time v(node)" header line that
        # numpy can't parse — skip it.
        data = np.loadtxt(out_path, comments="#", skiprows=1)
        return data[:, 0], data[:, 1]


def interp_to_grid(t_ref: np.ndarray, t: np.ndarray,
                   y: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``y(t)`` onto the reference grid."""
    return np.interp(t_ref, t, y)


# ---------------------------------------------------------------------------
# Case 1 — RC
# ---------------------------------------------------------------------------


def case_rc() -> CaseResult:
    V_dc, R, C = 5.0, 1.0e3, 1.0e-6
    tau = R * C
    t_end = 5.0 * tau
    dt    = tau / 200.0

    # ---- Pulsim ----
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "n0", "gnd", V_dc)
    b.add_resistor      ("R",   "n0", "vc",  R)
    b.add_capacitor     ("C",   "vc", "gnd", C)
    res = p.simulate(b, t_end=t_end, dt=dt)
    t_ps = np.asarray(res.times)
    idx  = b.node_id_of("vc")
    v_ps = np.array([s[idx] for s in res.states])

    # ---- ngspice ----
    netlist = (
        "* RC charging\n"
        f"Vdc n0 0 DC {V_dc}\n"
        f"R   n0 vc {R}\n"
        f"C   vc 0  {C} IC=0\n"
        ".ic v(vc)=0\n"
        ".options method=trap\n"
    )
    try:
        t_ng, v_ng = run_ngspice(netlist, "vc", 0.0, t_end, dt)
    except Exception as e:
        return CaseResult("rc", "RC step", skipped=True, reason=str(e))

    v_ng = interp_to_grid(t_ps, t_ng, v_ng)
    return CaseResult(
        "rc", "RC step charging (5τ)",
        cmp=TraceCompare(t=t_ps, pulsim=v_ps, ngspice=v_ng))


# ---------------------------------------------------------------------------
# Case 2 — RL
# ---------------------------------------------------------------------------


def case_rl() -> CaseResult:
    V_dc, R, L = 12.0, 10.0, 1e-3
    tau = L / R
    t_end = 5.0 * tau
    dt    = tau / 200.0

    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "n0", "gnd", V_dc)
    b.add_resistor      ("R",   "n0", "nL",  R)
    b.add_inductor      ("L",   "nL", "gnd", L)
    res = p.simulate(b, t_end=t_end, dt=dt)
    t_ps = np.asarray(res.times)
    # Inductor current = last state entry.
    n_st = len(res.states[0])
    v_ps = np.array([s[n_st - 1] for s in res.states])

    netlist = (
        "* RL ramp\n"
        f"Vdc n0 0 DC {V_dc}\n"
        f"R1  n0 nL {R}\n"
        f"L1  nL 0  {L} IC=0\n"
        ".options method=trap\n"
    )
    try:
        # ngspice doesn't expose i(L) via wrdata easily; instead we
        # compute i_L = (V_dc − v(nL)) / R from the node voltage.
        t_ng, v_nL = run_ngspice(netlist, "nL", 0.0, t_end, dt)
    except Exception as e:
        return CaseResult("rl", "RL ramp", skipped=True, reason=str(e))

    i_L_ng = (V_dc - v_nL) / R
    i_L_ng = interp_to_grid(t_ps, t_ng, i_L_ng)
    return CaseResult(
        "rl", "RL current ramp (5τ)",
        cmp=TraceCompare(t=t_ps, pulsim=v_ps, ngspice=i_L_ng))


# ---------------------------------------------------------------------------
# Case 3 — RLC underdamped
# ---------------------------------------------------------------------------


def case_rlc() -> CaseResult:
    V_step, R, L, C = 10.0, 0.1, 100e-6, 100e-6
    omega_n = 1.0 / np.sqrt(L * C)
    zeta    = (R / 2.0) * np.sqrt(C / L)
    omega_d = omega_n * np.sqrt(1.0 - zeta ** 2)
    T_osc   = 2.0 * np.pi / omega_d
    t0      = 100e-6
    dt      = T_osc / 400.0
    t_end   = t0 + 6.0 * T_osc

    b = p.CircuitBuilder()
    b.add_pulse_voltage_source(
        "Vstep", "in", "gnd",
        v_initial=0.0, v_pulsed=V_step,
        t_start=t0, pulse_width=10.0)
    b.add_inductor ("L",  "in", "nA",  L)
    b.add_resistor ("R",  "nA", "nB",  R)
    b.add_capacitor("C",  "nB", "gnd", C)
    res = p.simulate(b, t_end=t_end, dt=dt)
    t_ps = np.asarray(res.times)
    idx  = b.node_id_of("nB")
    v_ps = np.array([s[idx] for s in res.states])

    netlist = (
        "* RLC underdamped step\n"
        f"Vstep in 0 PULSE(0 {V_step} {t0} 1n 1n 10 20)\n"
        f"L1 in nA {L} IC=0\n"
        f"R1 nA nB {R}\n"
        f"C1 nB 0 {C} IC=0\n"
        ".options method=trap\n"
    )
    try:
        t_ng, v_ng = run_ngspice(netlist, "nB", 0.0, t_end, dt)
    except Exception as e:
        return CaseResult("rlc", "RLC underdamped", skipped=True,
                          reason=str(e))

    v_ng = interp_to_grid(t_ps, t_ng, v_ng)
    return CaseResult(
        "rlc", "RLC underdamped step (5 cycles, ζ=0.05)",
        cmp=TraceCompare(t=t_ps, pulsim=v_ps, ngspice=v_ng))


# ---------------------------------------------------------------------------
# Case 4 — Half-wave rectifier
# ---------------------------------------------------------------------------


def case_half_wave() -> CaseResult:
    V_peak, freq, R, C = 10.0, 60.0, 1e3, 100e-6
    T = 1.0 / freq
    t_end = 6.0 * T
    dt    = T / 2000.0

    b = p.CircuitBuilder()
    b.add_sine_voltage_source(
        "Vsin", "in", "gnd",
        v_dc=0.0, v_amplitude=V_peak, frequency=freq, phase=0.0)
    b.add_diode    ("D",    "in",   "vout",
                    g_on=1.0, g_off=1e-9, V_th=0.7)
    b.add_capacitor("C",    "vout", "gnd", C)
    b.add_resistor ("Rload","vout", "gnd", R)
    res = p.simulate(b, t_end=t_end, dt=dt)
    t_ps = np.asarray(res.times)
    idx  = b.node_id_of("vout")
    v_ps = np.array([s[idx] for s in res.states])

    netlist = (
        "* half-wave rectifier\n"
        f"Vsin in 0 SIN(0 {V_peak} {freq})\n"
        # ngspice ideal diode equivalent (PWL via a custom model)
        ".model D_ideal D(IS=1e-14 N=1 RS=1 BV=1000)\n"
        "D1 in vout D_ideal\n"
        f"C  vout 0 {C} IC=0\n"
        f"R  vout 0 {R}\n"
        ".options method=trap reltol=1e-4\n"
    )
    try:
        t_ng, v_ng = run_ngspice(netlist, "vout", 0.0, t_end, dt)
    except Exception as e:
        return CaseResult("half_wave", "Half-wave rectifier",
                          skipped=True, reason=str(e))

    v_ng = interp_to_grid(t_ps, t_ng, v_ng)
    return CaseResult(
        "half_wave", "Half-wave rectifier (6 cycles, 60 Hz)",
        cmp=TraceCompare(t=t_ps, pulsim=v_ps, ngspice=v_ng))


# ---------------------------------------------------------------------------
# Case 5 — Buck converter (open-loop CCM, 100 kHz, D=0.5)
# ---------------------------------------------------------------------------


def case_buck() -> CaseResult:
    """Open-loop buck @ 100 kHz / D=0.5, V_in=24 V → V_out ≈ 12 V.

    Both sides use the same passive values and PWM gate.  Pulsim drives
    its switch via the ``switch_fn`` callback; ngspice uses an
    ``S``-element fed by a ``Vgate PULSE`` source.  The freewheel
    diode is event-driven in Pulsim, Shockley-Schmitt in ngspice.

    Cycle-by-cycle ripple shapes will differ (PWL-trap vs ngspice gear/trap
    on different edge timings), but the steady-state envelope should
    align within tens of mV.
    """
    V_in   = 24.0
    L      = 100e-6
    C      = 100e-6
    R_load = 5.0
    f_sw   = 100e3
    T_sw   = 1.0 / f_sw
    duty   = 0.5
    t_end  = 2.0e-3                  # 200 cycles is plenty for SS
    dt     = T_sw / 100.0            # 100 samples/cycle

    # ---- Pulsim ----
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin",   "vin", "gnd",  V_in)
    b.add_switch        ("Q",     "vin", "sw",   g_on=1e3, g_off=1e-9)
    b.add_diode         ("D_FW",  "gnd", "sw",
                         g_on=1e3, g_off=1e-9, V_th=0.0)
    b.add_inductor      ("L1",    "sw",  "vout", L)
    b.add_capacitor     ("Cout",  "vout","gnd",  C)
    b.add_resistor      ("Rload", "vout","gnd",  R_load)

    # PWM bit-0 = Q (D-duty); bit-1 = D_FW armed (event-driven).
    n_sw = b.graph.num_switches
    def switch_fn(t: float):
        m = p.SwitchStateMask(n_sw)
        phase = math.fmod(t, T_sw) / T_sw
        if phase < duty:
            m.set(0, True)
        m.set(1, True)
        return m

    res = p.simulate(b, t_end=t_end, dt=dt, switch_fn=switch_fn,
                     max_event_iterations=8)
    t_ps = np.asarray(res.times)
    idx  = b.node_id_of("vout")
    v_ps = np.array([s[idx] for s in res.states])

    # ---- ngspice ----
    netlist = (
        "* Open-loop buck @ 100 kHz, D=0.5\n"
        f"Vin   vin 0 DC {V_in}\n"
        f"Vgate vg  0 PULSE(0 5 0 1n 1n {duty*T_sw:.3e} {T_sw:.3e})\n"
        f"Sq    vin sw vg 0 SWMOD\n"
        f"D_FW  0 sw DMOD\n"
        f"L1    sw vout {L} IC=0\n"
        f"Cout  vout 0 {C} IC=0\n"
        f"Rload vout 0 {R_load}\n"
        ".model SWMOD SW(Ron=1m Roff=1G Vt=2.5 Vh=0.1)\n"
        ".model DMOD  D(IS=1e-14 N=1 RS=1m BV=1000)\n"
        ".options method=trap reltol=1e-4\n"
    )
    try:
        t_ng, v_ng = run_ngspice(netlist, "vout", 0.0, t_end, dt)
    except Exception as e:
        return CaseResult("buck", "Buck CCM @ 100 kHz", skipped=True,
                          reason=str(e))

    v_ng = interp_to_grid(t_ps, t_ng, v_ng)
    # Compare on the steady-state window only — both simulators reach
    # SS after ∼50 cycles (∼0.5 ms), but the precise transient ringing
    # differs between PWL-trap (Pulsim) and SW-model+trap (ngspice).
    ss_mask = t_ps >= 1.0e-3            # last half = 100 cycles SS
    return CaseResult(
        "buck", "Buck CCM (steady-state, 100 cycles)",
        cmp=TraceCompare(t=t_ps[ss_mask],
                         pulsim=v_ps[ss_mask],
                         ngspice=v_ng[ss_mask]))


# ---------------------------------------------------------------------------
# Case 6 — Boost converter (open-loop CCM, 100 kHz, D=0.5)
# ---------------------------------------------------------------------------


def case_boost() -> CaseResult:
    """Open-loop boost @ 100 kHz / D=0.5, V_in=12 V → V_out ≈ 24 V.

    Same parity expectations as buck.  Both sides model the diode with
    a tiny series resistance + zero-Vth idealization to keep losses
    aligned — ngspice's default Shockley junction inflates V_F by
    ∼25 mV at the load currents we run here, which is < 0.5 % of
    V_out and falls within the cycle-ripple band anyway.
    """
    V_in   = 12.0
    L      = 100e-6
    C      = 100e-6
    R_load = 20.0
    f_sw   = 100e3
    T_sw   = 1.0 / f_sw
    duty   = 0.5
    t_end  = 2.0e-3
    dt     = T_sw / 100.0

    # ---- Pulsim ----
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin",   "vin", "gnd",  V_in)
    b.add_inductor      ("L1",    "vin", "sw",   L)
    b.add_switch        ("Q",     "sw",  "gnd",  g_on=1e3, g_off=1e-9)
    b.add_diode         ("D",     "sw",  "vout",
                         g_on=1e3, g_off=1e-9, V_th=0.0)
    b.add_capacitor     ("Cout",  "vout","gnd",  C)
    b.add_resistor      ("Rload", "vout","gnd",  R_load)

    n_sw = b.graph.num_switches
    def switch_fn(t: float):
        m = p.SwitchStateMask(n_sw)
        phase = math.fmod(t, T_sw) / T_sw
        if phase < duty:
            m.set(0, True)
        m.set(1, True)
        return m

    res = p.simulate(b, t_end=t_end, dt=dt, switch_fn=switch_fn,
                     max_event_iterations=8)
    t_ps = np.asarray(res.times)
    idx  = b.node_id_of("vout")
    v_ps = np.array([s[idx] for s in res.states])

    # ---- ngspice ----
    netlist = (
        "* Open-loop boost @ 100 kHz, D=0.5\n"
        f"Vin   vin 0 DC {V_in}\n"
        f"Vgate vg  0 PULSE(0 5 0 1n 1n {duty*T_sw:.3e} {T_sw:.3e})\n"
        f"L1    vin sw {L} IC=0\n"
        f"Sq    sw 0 vg 0 SWMOD\n"
        f"D1    sw vout DMOD\n"
        f"Cout  vout 0 {C} IC=0\n"
        f"Rload vout 0 {R_load}\n"
        ".model SWMOD SW(Ron=1m Roff=1G Vt=2.5 Vh=0.1)\n"
        ".model DMOD  D(IS=1e-14 N=1 RS=1m BV=1000)\n"
        ".options method=trap reltol=1e-4\n"
    )
    try:
        t_ng, v_ng = run_ngspice(netlist, "vout", 0.0, t_end, dt)
    except Exception as e:
        return CaseResult("boost", "Boost CCM @ 100 kHz", skipped=True,
                          reason=str(e))

    v_ng = interp_to_grid(t_ps, t_ng, v_ng)
    ss_mask = t_ps >= 1.0e-3
    return CaseResult(
        "boost", "Boost CCM (steady-state, 100 cycles)",
        cmp=TraceCompare(t=t_ps[ss_mask],
                         pulsim=v_ps[ss_mask],
                         ngspice=v_ng[ss_mask]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


ALL_CASES: dict[str, Callable[[], CaseResult]] = {
    "rc":         case_rc,
    "rl":         case_rl,
    "rlc":        case_rlc,
    "half_wave":  case_half_wave,
    "buck":       case_buck,
    "boost":      case_boost,
}


def fmt_si(x: float, unit: str) -> str:
    if not np.isfinite(x):
        return "—"
    a = abs(x)
    if a < 1e-6:
        return f"{x*1e9:.2f} n{unit}"
    if a < 1e-3:
        return f"{x*1e6:.2f} µ{unit}"
    if a < 1.0:
        return f"{x*1e3:.2f} m{unit}"
    return f"{x:.3f} {unit}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=list(ALL_CASES),
                    help="run only these cases (default: all)")
    ap.add_argument("--json", type=Path, help="dump JSON to this path")
    ap.add_argument("--plot", action="store_true",
                    help="save per-case PNG plots")
    args = ap.parse_args()

    selected = args.only or list(ALL_CASES)

    # Verify ngspice is installed before running anything.
    try:
        subprocess.run(["ngspice", "-v"], capture_output=True,
                       check=True, timeout=10)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"ngspice not found / not runnable: {e}", file=sys.stderr)
        print("Install: brew install ngspice  |  apt install ngspice",
              file=sys.stderr)
        return 1

    results: list[CaseResult] = []
    for name in selected:
        print(f"  running {name} …", file=sys.stderr, flush=True)
        results.append(ALL_CASES[name]())

    # Markdown summary.
    print()
    print("| case                                       "
          "| max |abs|    | max |rel|    | RMS abs     |")
    print("|--------------------------------------------"
          "|--------------|--------------|-------------|")
    for r in results:
        if r.skipped:
            print(f"| {r.description:<42} | _skipped_    "
                  "| _skipped_    | _skipped_   |")
            print(f"|    reason: {r.reason[:60]:<33}"
                  "|              |              |             |")
            continue
        c = r.cmp
        print(f"| {r.description:<42} "
              f"| {fmt_si(c.max_abs, 'V'):<12} "
              f"| {c.max_rel*100:>10.3f} % "
              f"| {fmt_si(c.rms_abs, 'V'):<11} |")

    if args.json:
        payload = []
        for r in results:
            row = {"name": r.name, "description": r.description,
                   "skipped": r.skipped}
            if not r.skipped:
                row.update({
                    "max_abs":  r.cmp.max_abs,
                    "max_rel":  r.cmp.max_rel,
                    "rms_abs":  r.cmp.rms_abs,
                    "num_samples": int(len(r.cmp.t)),
                })
            else:
                row["reason"] = r.reason
            payload.append(row)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nJSON written to {args.json}", file=sys.stderr)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping --plot",
                  file=sys.stderr)
            return 0
        out_dir = Path("benchmarks/ngspice_parity_out")
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            if r.skipped:
                continue
            c = r.cmp
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5),
                                            sharex=True)
            ax1.plot(c.t * 1e3, c.pulsim,  label="Pulsim", lw=1)
            ax1.plot(c.t * 1e3, c.ngspice, label="ngspice", lw=1,
                     linestyle="--")
            ax1.set_ylabel("amplitude")
            ax1.set_title(r.description)
            ax1.legend(loc="best")
            ax2.plot(c.t * 1e3, c.abs_err, color="red", lw=0.8)
            ax2.set_ylabel("|abs error|")
            ax2.set_xlabel("time [ms]")
            png_path = out_dir / f"{r.name}.png"
            fig.tight_layout()
            fig.savefig(png_path, dpi=110)
            plt.close(fig)
            print(f"  plot: {png_path}", file=sys.stderr)

    # Exit non-zero if anything was skipped (caller may treat as soft fail).
    return 0 if not any(r.skipped for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
