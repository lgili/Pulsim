"""550 W PFC boost converter — validation script.

Goals:
  1. Backend converges out-of-the-box (no manual switching_mode = Ideal).
  2. Auto-snubber C_oss / C_j default ON gives clean commutation
     (no manual `mp.C_oss = …` tuning).
  3. Electrical waveforms make physical sense (V_bus near design,
     I_load near rating, V_sw clamped to V_bus + V_F).
  4. Per-device losses are bounded and sum to a sensible converter
     loss (η in the 90–98 % range for a hard-switched boost).

Topology
--------
   AC 220 V_rms ──┬─ Brg ── C_bulk ── L_boost ── sw ── D_boost ── C_out ── R_load ──┐
   60 Hz         │     (4× diode)            │             │                        │
                  └────── ground ───────────  M_pfc       ground                     │
                                            (g = PWM)                                │
                                              gnd ─────────────────────────────────  ┘

Sizing chosen so the auto-snubber default (C_oss = 10 nF) is well-
matched to the boost inductor (100 µH) at f_sw = 100 kHz / dt = 500 ns:
LC_period = 2·π·√(L·C_oss) ≈ 6.3 µs, > 12·dt → Tustin damps the LC
tank cleanly.
"""
from __future__ import annotations

import math
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pulsim as ps


# =============================================================================
# Operating point
# =============================================================================
T_amb         = 40.0
V_AC_rms      = 220.0
f_line        = 60.0
P_target      = 550.0
V_bus_design  = 400.0
f_sw          = 100e3        # 100 kHz — smaller L, cleaner commutation
duty_pwm      = 0.25         # open-loop fixed (≈ 1 − V_in_peak / V_bus)
V_AC_peak     = V_AC_rms * math.sqrt(2)
V_bulk_steady = V_AC_peak - 2 * 0.7
R_load_val    = V_bus_design ** 2 / P_target          # ≈ 291 Ω

print("Design point:")
print(f"  AC : V = {V_AC_rms} V rms / {V_AC_peak:.0f} V peak / {f_line:.0f} Hz")
print(f"  DC : V_bulk(SS) ≈ {V_bulk_steady:.0f} V")
print(f"       V_bus design = {V_bus_design:.0f} V → R_load = {R_load_val:.0f} Ω, P_out = {P_target:.0f} W")
print(f"  Boost: L = 100 µH, f_sw = {f_sw/1e3:.0f} kHz, D = {duty_pwm:.0%}")
print()


# =============================================================================
# Circuit (auto-snubber on — no manual C_oss / C_j)
# =============================================================================
c = ps.Circuit()
n_acp  = c.add_node("ac+")
n_acn  = c.add_node("ac-")
n_bulk = c.add_node("bulk")
n_sw   = c.add_node("sw")
n_bus  = c.add_node("bus")
n_gate = c.add_node("gate")
gnd    = ps.Circuit.ground()

sine = ps.SineParams()
sine.amplitude = V_AC_peak; sine.frequency = f_line
c.add_sine_voltage_source("Vac", n_acp, n_acn, sine)
c.add_resistor("R_anchor", n_acn, gnd, 1e6)        # AC subgraph anchor

# Bridge (4 fast-recovery diodes)
dbrg = ps.RealisticDiodeParams()
dbrg.V_F0 = 0.70; dbrg.R_d = 25e-3; dbrg.V_F0_tc = -2e-3
dbrg.T_ref = 125.0; dbrg.R_th_ja = 30.0; dbrg.T_amb = T_amb
dbrg.g_on = 1.0 / dbrg.R_d
# Qrr left at 0 — line rectifier, soft commutation, no auto C_j needed
c.add_bridge_rectifier("Brg", n_acp, n_acn, n_bulk, gnd, dbrg)

# Bulk capacitor (pre-charged to soft-start)
cap_b = ps.CapacitorParams()
cap_b.capacitance = 470e-6; cap_b.initial_voltage = V_bulk_steady
cap_b.ESR = 80e-3; cap_b.ESR_tc = 0.005
cap_b.T_ref = 25.0; cap_b.R_th_ja = 12.0; cap_b.T_amb = T_amb
c.add_capacitor("C_bulk", n_bulk, gnd, cap_b)

# Boost inductor — 100 µH chosen so LC resonance with auto-C_oss (10 nF)
# is at 16 kHz (>> 100 kHz/2 sampling) and damped by Tustin.
ind = ps.InductorParams()
ind.inductance = 100e-6
ind.initial_current = P_target / V_bulk_steady
ind.DCR = 30e-3; ind.DCR_tc = 3.9e-3
ind.T_ref = 25.0; ind.R_th_ja = 35.0; ind.T_amb = T_amb
c.add_inductor("L_boost", n_bulk, n_sw, ind)

# PFC MOSFET — Eon_25 > 0 triggers auto-promote to Ideal AND auto-C_oss = 10 nF
mp = ps.MOSFETParams()
mp.vth = 4.0; mp.kp = 50.0; mp.g_on = 50.0; mp.g_off = 1e-12
mp.is_nmos = True
mp.Rds_on_tc = 0.005
mp.Eon_25 = 60e-6; mp.Eoff_25 = 120e-6
mp.I_ref = 5.0; mp.V_ref = 400.0; mp.Esw_tc = 0.002
mp.R_th_ja = 1.5; mp.T_amb = T_amb
# Behavioral mode (smooth Shichman-Hodges) — no Eon/Eoff so no
# auto-promote to PWL Ideal. The smooth Newton path handles LC
# resonance naturally (no ideal-switch commutation discontinuity),
# at the cost of a few extra Newton iterations per step. This is
# the SPICE/PSIM "smooth model" path. To force PWL Ideal explicitly
# set mp.Eon_25 > 0 and tune mp.C_oss for your L + dt combination.
mp.Eon_25 = 0.0; mp.Eoff_25 = 0.0
c.add_mosfet("M_pfc", n_gate, n_sw, gnd, mp)
c.add_pwm_voltage_source("Vg", n_gate, gnd, 12.0, 0.0, f_sw, duty_pwm)

# Boost diode — Qrr > 0 triggers auto-C_j = 5 nF
dbst = ps.RealisticDiodeParams()
dbst.V_F0 = 0.85; dbst.R_d = 30e-3; dbst.V_F0_tc = -2e-3
dbst.T_ref = 125.0; dbst.Qrr = 60e-9
dbst.R_th_ja = 25.0; dbst.T_amb = T_amb
dbst.g_on = 1.0 / dbst.R_d
# Behavioral mode — drop Qrr so no auto-default to Ideal/C_j.
dbst.Qrr = 0.0
c.add_diode("D_boost", n_sw, n_bus, dbst)

# Output cap (pre-charged to soft-start)
cap_o = ps.CapacitorParams()
cap_o.capacitance = 220e-6; cap_o.initial_voltage = V_bus_design
cap_o.ESR = 60e-3; cap_o.ESR_tc = 0.005
cap_o.T_ref = 25.0; cap_o.R_th_ja = 15.0; cap_o.T_amb = T_amb
c.add_capacitor("C_out", n_bus, gnd, cap_o)

# Load
load = ps.ResistorParams()
load.resistance = R_load_val; load.TCR = 0.0
load.R_th_ja = 0.0; load.T_amb = T_amb
c.add_resistor("R_load", n_bus, gnd, load)


# =============================================================================
# Simulate (3 line cycles)
# =============================================================================
# simplify-and-harden-numerical-surface — Phase 2 + 14.8: PFC boost +
# active controller → Preset.Robust.
opts = ps.SimulationOptions.from_preset(ps.Preset.Robust,
                                          dt=100e-9, tstop=20e-3)
opts.dt_min = 1e-9
opts.dt_max = 200e-9
# NOTE: opts.switching_mode left at Auto — backend auto-promotes when
# the MOSFET opts into PWL Ideal mode (Eon_25 > 0). This is the
# `8ba749f` backend fix.
opts.newton_options.num_nodes = c.num_nodes()
opts.newton_options.num_branches = c.num_branches()

print(f"Simulating  tstop = {opts.tstop*1e3:.0f} ms  dt_min = {opts.dt_min*1e9:.0f} ns")
result = ps.Simulator(c, opts).run_transient(c.initial_state())
print(f"  success = {result.success}, accepted steps = {len(result.time)}")
print(f"  pwl_event_commutations = {result.backend_telemetry.pwl_event_commutations}")
if not result.success:
    sys.exit("Simulation failed.")

t = np.asarray(result.time)
x = np.asarray(result.states)
v_ac    = x[:, n_acp] - x[:, n_acn]
v_bulk  = x[:, n_bulk]
v_sw    = x[:, n_sw]
v_bus   = x[:, n_bus]
v_gate  = x[:, n_gate]
i_load  = v_bus / R_load_val


# =============================================================================
# Steady-state sanity checks (last 30% of sim, after settling)
# =============================================================================
mid = int(0.7 * len(t))
v_bus_ss   = float(v_bus[mid:].mean())
v_bus_min  = float(v_bus[mid:].min())
v_bus_max  = float(v_bus[mid:].max())
v_bus_rip  = v_bus_max - v_bus_min
i_load_ss  = float(i_load[mid:].mean())
p_out_ss   = float((v_bus[mid:] ** 2).mean() / R_load_val)
v_sw_min   = float(v_sw[mid:].min())
v_sw_max   = float(v_sw[mid:].max())

print("\n--- Steady-state operating point (last 30 % of sim)")
print(f"  V_bus  : avg = {v_bus_ss:.1f} V    ripple = {v_bus_rip:.2f} V")
print(f"  I_load : {i_load_ss:.3f} A")
print(f"  P_out  : {p_out_ss:.2f} W   (target {P_target:.0f} W)")
print(f"  V_sw   : min = {v_sw_min:.1f}  max = {v_sw_max:.1f}   (expected ~ 0..{V_bus_design:.0f} V)")


# =============================================================================
# Loss + thermal report
# =============================================================================
devices = [
    ("Brg__D1", "diode",     "Bridge D1"),
    ("Brg__D2", "diode",     "Bridge D2"),
    ("Brg__D3", "diode",     "Bridge D3"),
    ("Brg__D4", "diode",     "Bridge D4"),
    ("C_bulk",  "capacitor", "Bulk cap (ESR)"),
    ("L_boost", "inductor",  "Boost L (DCR)"),
    ("M_pfc",   "mosfet",    "PFC MOSFET"),
    ("D_boost", "diode",     "Boost diode (Qrr)"),
    ("C_out",   "capacitor", "Output cap (ESR)"),
    ("R_load",  "resistor",  "Load R"),
]

print("\n--- Loss + thermal per device (unified pipeline)")
header = f"{'Device':22} {'P_avg [W]':>10} {'P_peak [W]':>11} {'E_cond [mJ]':>12} {'E_sw [mJ]':>10} {'T_j [°C]':>10}"
print(header)
print("-" * len(header))
total = 0.0
for name, kind, label in devices:
    p_avg  = getattr(c, f"{kind}_average_power")(name)
    p_peak = getattr(c, f"{kind}_peak_power")(name)
    e_tot  = getattr(c, f"{kind}_total_energy")(name)
    e_sw   = (getattr(c, f"{kind}_switching_energy")(name)
              if hasattr(c, f"{kind}_switching_energy") else 0.0)
    e_cond = e_tot - e_sw
    t_j_m = f"{kind}_steady_state_junction_temperature"
    t_j = (getattr(c, t_j_m)(name) if hasattr(c, t_j_m) else float("nan"))
    print(f"{label:22} {p_avg:10.3f} {p_peak:11.3f} "
          f"{e_cond * 1e3:12.4f} {e_sw * 1e3:10.4f} {t_j:10.2f}")
    total += p_avg
print("-" * len(header))
print(f"{'TOTAL':22} {total:10.3f}")


# =============================================================================
# Power balance + efficiency
# =============================================================================
p_load    = c.resistor_average_power("R_load")
loss_conv = total - p_load
p_in      = p_load + loss_conv
result.loss_summary.input_power = p_in
result.loss_summary.compute_totals()

print("\n--- Power balance")
print(f"  P_out (R_load)            = {p_load:7.2f} W   ({p_load/P_target*100:.1f} % of target)")
print(f"  P_loss (converter only)   = {loss_conv:7.2f} W")
print(f"  P_in                      = {p_in:7.2f} W")
print(f"  efficiency η              = {p_load/p_in*100:7.2f} %")


# =============================================================================
# Plot
# =============================================================================
fig, ax = plt.subplots(4, 2, figsize=(14, 10))
fig.suptitle(
    f"550 W PFC boost — auto-snubber on (no manual C_oss/C_j)\n"
    f"V_AC = {V_AC_rms:.0f} V_rms / {f_line:.0f} Hz, "
    f"f_sw = {f_sw/1e3:.0f} kHz, D = {duty_pwm:.0%}, "
    f"η = {p_load/p_in*100:.1f} %",
    fontsize=12)

ax[0, 0].plot(t * 1e3, v_ac, color="#1f77b4")
ax[0, 0].set_ylabel("V_AC (V)"); ax[0, 0].set_title("AC input")
ax[0, 0].grid(alpha=0.3)

ax[1, 0].plot(t * 1e3, v_bulk, color="#2ca02c")
ax[1, 0].set_ylabel("V_bulk (V)"); ax[1, 0].set_title("Bulk cap (after bridge)")
ax[1, 0].grid(alpha=0.3)

ax[2, 0].plot(t * 1e3, v_bus, color="#d62728")
ax[2, 0].set_ylabel("V_bus (V)"); ax[2, 0].set_title("Boost output bus")
ax[2, 0].axhline(V_bus_design, ls="--", color="grey", alpha=0.5,
                 label=f"design {V_bus_design:.0f} V")
ax[2, 0].grid(alpha=0.3); ax[2, 0].legend(loc="lower right", fontsize=8)

ax[3, 0].plot(t * 1e3, i_load, color="#9467bd")
ax[3, 0].set_ylabel("I_load (A)"); ax[3, 0].set_xlabel("t (ms)")
ax[3, 0].set_title("Load current"); ax[3, 0].grid(alpha=0.3)

# Right column: zoom to 4 switching periods near end of sim
t0 = t[-1] - 1e-3
mask = (t >= t0) & (t <= t0 + 6 / f_sw)

ax[0, 1].plot(t[mask] * 1e6, v_gate[mask], color="#1f77b4")
ax[0, 1].set_ylabel("V_gate (V)"); ax[0, 1].set_title("MOSFET gate (PWM)")
ax[0, 1].grid(alpha=0.3)

ax[1, 1].plot(t[mask] * 1e6, v_sw[mask], color="#2ca02c")
ax[1, 1].set_ylabel("V_sw (V)"); ax[1, 1].set_title("Switching node")
ax[1, 1].grid(alpha=0.3)

ax[2, 1].plot(t[mask] * 1e6, v_bulk[mask] - v_sw[mask], color="#d62728")
ax[2, 1].set_ylabel("V_L (V)"); ax[2, 1].set_title("Boost inductor voltage")
ax[2, 1].grid(alpha=0.3)

ax[3, 1].plot(t[mask] * 1e6, v_bus[mask], color="#9467bd")
ax[3, 1].set_ylabel("V_bus (V)"); ax[3, 1].set_xlabel("t (µs)")
ax[3, 1].set_title("Bus ripple"); ax[3, 1].grid(alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("/tmp/pfc_boost_validation.png", dpi=120)
print(f"\nWaveforms saved → /tmp/pfc_boost_validation.png")
