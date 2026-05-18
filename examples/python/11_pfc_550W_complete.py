"""550 W single-phase AC-DC + boost PFC converter — full electrical
and loss/thermal exercise of Pulsim's unified pipeline.

Topology
--------
                                                     L_boost    sw      D_boost
   AC 220 V_rms ──┬─ Brg ── C_bulk ──┬── L_filter ──┬─/\/\/\─┬──►|──┬─ C_out ── R_load ──┐
   60 Hz         │     (4× diode)   │   (filter)   │          M_pfc                       │
                  └──── ground ──────┴──── ground ──┴── gnd ──┴──── ground ────────────────┘
                                                              (g = PWM 65 kHz)

Stage 1 — Front-end (AC 220 V → bridge → bulk cap → output filter)
Stage 2 — Boost chopper (V_bulk → boost L → MOSFET + boost diode → V_bus → R_load)

Backend fix this script exercises:
  Setting `MOSFETParams.Eon_25 > 0` auto-promotes the MOSFET's switching
  mode to `Ideal`. The simulator now propagates that intent up to the
  global `options.switching_mode = Ideal` (via `apply_auto_transient_profile`)
  so the PWL state-space path is selected end-to-end. Before this fix,
  the DAE/Newton path was used and Newton couldn't honour the PWM step
  discontinuity in an L+MOSFET topology (V_gate stayed frozen).

Devices instrumented (every loss/thermal number reads from the same per-
device accumulator the system-level `result.loss_summary` is built from):
  * 4× bridge diodes  (V_F0, R_d, R_th_ja)
  * Bulk cap          (ESR, R_th_ja)
  * Filter L          (DCR, R_th_ja)
  * Boost L           (DCR, R_th_ja)
  * PFC MOSFET        (Rds_on_tc, Eon_25, Eoff_25, R_th_ja)  → auto-promote
  * Boost diode       (V_F0, R_d, Qrr, R_th_ja)
  * Output cap        (ESR, R_th_ja)
  * Load resistor
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
f_sw          = 65e3
duty_pwm      = 0.30        # open-loop fixed (~ 75 % of theoretical D for 400 V)
V_AC_peak     = V_AC_rms * math.sqrt(2)
V_bulk_steady = V_AC_peak - 2 * 0.7
R_load_val    = V_bus_design ** 2 / P_target

print("Design point:")
print(f"  V_AC = {V_AC_rms:.0f} V_rms ({V_AC_peak:.0f} V peak), f = {f_line:.0f} Hz")
print(f"  V_bulk (SS) ≈ {V_bulk_steady:.0f} V (after bridge)")
print(f"  V_bus design = {V_bus_design:.0f} V, R_load = {R_load_val:.0f} Ω → P_target = {P_target:.0f} W")
print(f"  f_sw = {f_sw/1e3:.0f} kHz, D = {duty_pwm:.0%}")
print()


# =============================================================================
# Circuit
# =============================================================================
c = ps.Circuit()
n_acp  = c.add_node("ac+")
n_acn  = c.add_node("ac-")
n_bulk = c.add_node("bulk")
n_filt = c.add_node("filt")
n_sw   = c.add_node("sw")
n_bus  = c.add_node("bus")
n_gate = c.add_node("gate")
gnd    = ps.Circuit.ground()

# AC source
sine = ps.SineParams()
sine.amplitude = V_AC_peak; sine.frequency = f_line
c.add_sine_voltage_source("Vac", n_acp, n_acn, sine)
c.add_resistor("R_anchor", n_acn, gnd, 1e6)

# Bridge rectifier
dbrg = ps.RealisticDiodeParams()
dbrg.V_F0 = 0.70; dbrg.R_d = 25e-3; dbrg.V_F0_tc = -2e-3
dbrg.T_ref = 125.0; dbrg.R_th_ja = 30.0; dbrg.T_amb = T_amb
dbrg.g_on = 1.0 / dbrg.R_d
c.add_bridge_rectifier("Brg", n_acp, n_acn, n_bulk, gnd, dbrg)

# Bulk capacitor (pre-charged)
cap_b = ps.CapacitorParams()
cap_b.capacitance = 470e-6; cap_b.initial_voltage = V_bulk_steady
cap_b.ESR = 80e-3; cap_b.ESR_tc = 0.005
cap_b.T_ref = 25.0; cap_b.R_th_ja = 12.0; cap_b.T_amb = T_amb
c.add_capacitor("C_bulk", n_bulk, gnd, cap_b)

# Filter inductor (between bulk and boost input)
ind_f = ps.InductorParams()
ind_f.inductance = 100e-6
ind_f.initial_current = P_target / V_bulk_steady
ind_f.DCR = 30e-3; ind_f.DCR_tc = 3.9e-3
ind_f.T_ref = 25.0; ind_f.R_th_ja = 35.0; ind_f.T_amb = T_amb
c.add_inductor("L_filter", n_bulk, n_filt, ind_f)

# Boost inductor
ind_b = ps.InductorParams()
ind_b.inductance = 200e-6
ind_b.initial_current = P_target / V_bulk_steady
ind_b.DCR = 50e-3; ind_b.DCR_tc = 3.9e-3
ind_b.T_ref = 25.0; ind_b.R_th_ja = 35.0; ind_b.T_amb = T_amb
c.add_inductor("L_boost", n_filt, n_sw, ind_b)

# PFC MOSFET — Eon_25 > 0 triggers auto-promote to Ideal (backend fix)
mp = ps.MOSFETParams()
mp.vth = 4.0; mp.kp = 50.0; mp.g_on = 50.0; mp.g_off = 1e-12
mp.is_nmos = True
mp.Rds_on_tc = 0.005; mp.T_ref = 25.0
mp.Eon_25 = 60e-6; mp.Eoff_25 = 120e-6
mp.I_ref = 5.0; mp.V_ref = 400.0; mp.Esw_tc = 0.002
mp.R_th_ja = 1.5; mp.T_amb = T_amb
c.add_mosfet("M_pfc", n_gate, n_sw, gnd, mp)

# PWM gate drive
c.add_pwm_voltage_source("Vg", n_gate, gnd, 12.0, 0.0, f_sw, duty_pwm)

# Boost diode (fast)
dbst = ps.RealisticDiodeParams()
dbst.V_F0 = 0.85; dbst.R_d = 30e-3; dbst.V_F0_tc = -2e-3
dbst.T_ref = 125.0; dbst.Qrr = 60e-9
dbst.R_th_ja = 25.0; dbst.T_amb = T_amb
dbst.g_on = 1.0 / dbst.R_d
c.add_diode("D_boost", n_sw, n_bus, dbst)

# Output cap (pre-charged near design)
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
# Simulate (2 line cycles — enough to settle and capture switching detail)
# =============================================================================
opts = ps.SimulationOptions()
opts.tstart = 0.0; opts.tstop = 33e-3
opts.dt = 2e-6
opts.dt_min = 1e-9; opts.dt_max = 5e-6
opts.adaptive_timestep = True
opts.enable_bdf_order_control = True
opts.enable_events = True
opts.enable_losses = True
# NOTE: opts.switching_mode left at Auto — Pulsim 0.10.0a12+ auto-promotes
# to Ideal because the MOSFET opted in (Eon_25 > 0). This is the backend
# fix landed in `apply_auto_transient_profile`.
opts.newton_options.num_nodes = c.num_nodes()
opts.newton_options.num_branches = c.num_branches()

print("Simulating…")
r = ps.Simulator(c, opts).run_transient(c.initial_state())
print(f"  success = {r.success}, steps = {len(r.time)}")
print(f"  pwl_event_commutations = {r.backend_telemetry.pwl_event_commutations}")
if not r.success:
    sys.exit("Simulation failed.")

t = np.asarray(r.time)
x = np.asarray(r.states)
v_acp, v_acn = x[:, n_acp], x[:, n_acn]
v_bulk = x[:, n_bulk]
v_filt = x[:, n_filt]
v_sw   = x[:, n_sw]
v_bus  = x[:, n_bus]
v_gate = x[:, n_gate]
v_ac   = v_acp - v_acn
i_load = v_bus / R_load_val


# =============================================================================
# Plot all electrical waveforms
# =============================================================================
fig, ax = plt.subplots(4, 2, figsize=(14, 10))
fig.suptitle(
    f"550 W AC-DC + PFC Boost — full electrical waveforms\n"
    f"V_AC={V_AC_rms:.0f} V rms / {f_line:.0f} Hz, f_sw={f_sw/1e3:.0f} kHz, D={duty_pwm:.0%}, "
    f"PWL commutations={r.backend_telemetry.pwl_event_commutations}",
    fontsize=12,
)

# Left column: line-cycle scale
ax[0, 0].plot(t * 1e3, v_ac, color="#1f77b4")
ax[0, 0].set_ylabel("V_AC (V)"); ax[0, 0].set_title("AC input")
ax[0, 0].grid(alpha=0.3)

ax[1, 0].plot(t * 1e3, v_bulk, color="#2ca02c")
ax[1, 0].set_ylabel("V_bulk (V)"); ax[1, 0].set_title("Bulk capacitor (after bridge)")
ax[1, 0].grid(alpha=0.3)

ax[2, 0].plot(t * 1e3, v_bus, color="#d62728")
ax[2, 0].set_ylabel("V_bus (V)"); ax[2, 0].set_title("Boost output bus")
ax[2, 0].axhline(V_bus_design, ls="--", color="grey", alpha=0.5,
                 label=f"design = {V_bus_design:.0f} V")
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

ax[2, 1].plot(t[mask] * 1e6, v_filt[mask] - v_sw[mask], color="#d62728")
ax[2, 1].set_ylabel("V_L_boost (V)"); ax[2, 1].set_title("Boost inductor voltage")
ax[2, 1].grid(alpha=0.3)

ax[3, 1].plot(t[mask] * 1e6, v_bus[mask], color="#9467bd")
ax[3, 1].set_ylabel("V_bus (V)"); ax[3, 1].set_xlabel("t (µs)")
ax[3, 1].set_title("Bus ripple"); ax[3, 1].grid(alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.94])
elec_path = "/tmp/pfc_550W_waveforms.png"
fig.savefig(elec_path, dpi=120)
print(f"  saved {elec_path}")


# =============================================================================
# Loss + thermal report
# =============================================================================
print("\n" + "=" * 92)
print("  Loss + Thermal report — every number reads from the *same* per-device")
print("  accumulator the system summary is built from (unified pipeline)")
print("=" * 92 + "\n")

devices = [
    ("Brg__D1", "diode",     "Bridge D1"),
    ("Brg__D2", "diode",     "Bridge D2"),
    ("Brg__D3", "diode",     "Bridge D3"),
    ("Brg__D4", "diode",     "Bridge D4"),
    ("C_bulk",  "capacitor", "Bulk cap (ESR)"),
    ("L_filter","inductor",  "Filter L (DCR)"),
    ("L_boost", "inductor",  "Boost L (DCR)"),
    ("M_pfc",   "mosfet",    "PFC MOSFET"),
    ("D_boost", "diode",     "Boost D (Qrr)"),
    ("C_out",   "capacitor", "Output cap (ESR)"),
    ("R_load",  "resistor",  "Load R"),
]

header = (
    f"{'Device':22} {'P_avg [W]':>10} {'P_peak [W]':>11} "
    f"{'E_cond [mJ]':>12} {'E_sw [mJ]':>10} {'T_j [°C]':>10}"
)
print(header)
print("-" * len(header))

total_loss = 0.0
for name, kind, label in devices:
    p_avg  = getattr(c, f"{kind}_average_power")(name)
    p_peak = getattr(c, f"{kind}_peak_power")(name)
    e_tot  = getattr(c, f"{kind}_total_energy")(name)
    e_sw   = (getattr(c, f"{kind}_switching_energy")(name)
              if hasattr(c, f"{kind}_switching_energy") else 0.0)
    e_cond = e_tot - e_sw
    t_j_m = f"{kind}_steady_state_junction_temperature"
    t_j = (getattr(c, t_j_m)(name) if hasattr(c, t_j_m) else float("nan"))
    print(
        f"{label:22} {p_avg:10.3f} {p_peak:11.3f} "
        f"{e_cond * 1e3:12.4f} {e_sw * 1e3:10.4f} {t_j:10.2f}"
    )
    total_loss += p_avg

print("-" * len(header))
print(f"{'TOTAL':22} {total_loss:10.3f}")


# =============================================================================
# System summary + efficiency
# =============================================================================
p_out  = float(np.mean(v_bus ** 2) / R_load_val)
losses = r.loss_summary.total_loss - c.resistor_average_power("R_load")
p_in   = p_out + losses
r.loss_summary.input_power = p_in
r.loss_summary.compute_totals()

print("\n--- SystemLossSummary")
print(f"  total_loss       = {r.loss_summary.total_loss:7.3f} W")
print(f"  total_conduction = {r.loss_summary.total_conduction:7.3f} W")
print(f"  total_switching  = {r.loss_summary.total_switching:7.3f} W")
print(f"\n--- Power flow")
print(f"  P_out (R_load)            = {p_out:7.2f} W")
print(f"  P_loss (converter only)   = {losses:7.2f} W")
print(f"  P_in                      = {p_in:7.2f} W")
print(f"  efficiency η              = {p_out/p_in*100:7.2f} %")


# =============================================================================
# Loss + thermal bar charts
# =============================================================================
fig2, (axL, axT) = plt.subplots(1, 2, figsize=(14, 6))
labels = [lab for _, _, lab in devices]
p_vals = [getattr(c, f"{k}_average_power")(n) for n, k, _ in devices]
t_vals = [getattr(c, f"{k}_steady_state_junction_temperature")(n)
          if hasattr(c, f"{k}_steady_state_junction_temperature") else float("nan")
          for n, k, _ in devices]

colors = (["#1f77b4"] * 4 + ["#8c564b", "#ff7f0e", "#ffbb78", "#d62728",
                              "#2ca02c", "#9467bd", "#7f7f7f"])

axL.barh(labels, p_vals, color=colors)
axL.set_xlabel("P_avg (W)")
axL.set_title(f"Loss breakdown — converter losses = {losses:.2f} W, η = {p_out/p_in*100:.1f} %")
for i, v in enumerate(p_vals):
    axL.text(v, i, f" {v:.2f}", va="center", fontsize=9)
axL.grid(axis="x", alpha=0.3)

axT.barh(labels, t_vals, color=colors)
axT.axvline(T_amb, ls="--", color="grey", alpha=0.5, label=f"T_amb = {T_amb:.0f} °C")
axT.set_xlabel("Steady-state T_j (°C)")
axT.set_title("Junction temperatures (T_amb + P · R_th_ja)")
for i, v in enumerate(t_vals):
    if math.isfinite(v):
        axT.text(v, i, f" {v:.0f}", va="center", fontsize=9)
axT.grid(axis="x", alpha=0.3)
axT.legend(loc="lower right")

fig2.tight_layout()
loss_path = "/tmp/pfc_550W_losses.png"
fig2.savefig(loss_path, dpi=120)
print(f"\n  saved {loss_path}")
print("\n" + "=" * 92)
print(f"  Waveforms  → {elec_path}")
print(f"  Losses+T_j → {loss_path}")
print("=" * 92)
