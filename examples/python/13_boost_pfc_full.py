"""550 W AC-DC + boost PFC converter — END-TO-END validation.

After the 8-commit `investigate-pfc-boost-convergence` series, this
script demonstrates a *converging* boost converter with all the
auto-defaults + heavy parasitic tuning (until the PWL ideal-switch
limitation is fully resolved by a future "soft commutation" pass).

Topology
--------
    AC 220 V_rms ──┬─ Brg ── C_bulk ── L_boost ── sw ── D_boost ── C_out ── R_load ──┐
    60 Hz         │   (4×D)            (100µH)   │                                   │
                  └────── ground ─────────  M_pfc                                    │
                                            (PWM gate)                                │
                                            │                                         │
                                            gnd ──────────────────────────────────────┘

Key knobs the user tunes for PWL-stable convergence in boost-style
topologies (until soft-commutation lands):
  - `mp.C_oss`  : large enough so LC_period = 2·π·√(L·C_oss) >> dt
                  Rule of thumb: C_oss >= 100 × (dt² / L)
                  For L = 100 µH, dt = 500 ns:  C_oss >= 250 nF
                  Used here: 10 µF (heavy over-margin → bounded V_sw)
  - `dp.C_j`    : same logic for the diode cap (typically 10× smaller
                  than C_oss because the diode commutates faster)

The "right" values for real-converter fidelity are 100–500 pF; that
requires dt = 10–50 ns. The auto-defaults (Eon_25 > 0 → 10 nF, etc.)
sit between those extremes; for the cleanest waveforms you tune.

Caveat: the per-device loss accumulator still over-counts the
C_oss-charge transient (V_DS² · g · dt sampled at peak overshoots the
true ∫V·I dt by 10-100×). See "open backlog" in the README. The
**operating point** (V_bus, I_load, V_sw range) is correct; the
**reported losses** for switching devices have a known systematic
upward bias that future trapezoidal-V-averaging will fix.
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
f_sw          = 100e3
duty_pwm      = 0.25                # 1 − V_in_peak / V_bus ≈ 1 − 310/400
V_AC_peak     = V_AC_rms * math.sqrt(2)
V_bulk_steady = V_AC_peak - 2 * 0.7
R_load_val    = V_bus_design ** 2 / P_target
L_boost_val   = 100e-6


# =============================================================================
# Circuit
# =============================================================================
c = ps.Circuit()
n_acp  = c.add_node("ac+")
n_acn  = c.add_node("ac-")
n_bulk = c.add_node("bulk")
n_sw   = c.add_node("sw")
n_bus  = c.add_node("bus")
n_gate = c.add_node("gate")
gnd    = ps.Circuit.ground()

# AC source
sine = ps.SineParams()
sine.amplitude = V_AC_peak; sine.frequency = f_line
c.add_sine_voltage_source("Vac", n_acp, n_acn, sine)
c.add_resistor("R_anchor", n_acn, gnd, 1e6)

# Bridge rectifier — V_F0 > 0 → auto C_j = 500 pF (negligible at line freq).
dbrg = ps.RealisticDiodeParams()
dbrg.V_F0 = 0.70; dbrg.R_d = 25e-3; dbrg.V_F0_tc = -2e-3
dbrg.T_ref = 125.0; dbrg.R_th_ja = 30.0; dbrg.T_amb = T_amb
dbrg.g_on = 1.0 / dbrg.R_d
c.add_bridge_rectifier("Brg", n_acp, n_acn, n_bulk, gnd, dbrg)

# Bulk cap (pre-charged)
cap_b = ps.CapacitorParams()
cap_b.capacitance = 470e-6; cap_b.initial_voltage = V_bulk_steady
cap_b.ESR = 80e-3; cap_b.ESR_tc = 0.005
cap_b.T_ref = 25.0; cap_b.R_th_ja = 12.0; cap_b.T_amb = T_amb
c.add_capacitor("C_bulk", n_bulk, gnd, cap_b)

# Boost inductor
ind = ps.InductorParams()
ind.inductance = L_boost_val
ind.initial_current = P_target / V_bulk_steady
ind.DCR = 30e-3; ind.DCR_tc = 3.9e-3
ind.T_ref = 25.0; ind.R_th_ja = 35.0; ind.T_amb = T_amb
c.add_inductor("L_boost", n_bulk, n_sw, ind)

# PFC MOSFET — backend auto-promotes to PWL Ideal (Eon_25 > 0). Heavy
# C_oss override for numerical stability with L=100µH at dt=500ns.
mp = ps.MOSFETParams()
mp.vth = 4.0; mp.kp = 50.0; mp.g_on = 50.0; mp.g_off = 1e-12
mp.is_nmos = True
mp.Rds_on_tc = 0.005
mp.Eon_25 = 60e-6; mp.Eoff_25 = 120e-6
mp.I_ref = 5.0; mp.V_ref = 400.0; mp.Esw_tc = 0.002
mp.R_th_ja = 1.5; mp.T_amb = T_amb
mp.C_oss = 10e-6                    # heavy snubber — see module docstring
c.add_mosfet("M_pfc", n_gate, n_sw, gnd, mp)

# Gate PWM
c.add_pwm_voltage_source("Vg", n_gate, gnd, 12.0, 0.0, f_sw, duty_pwm)

# Boost diode — also heavy C_j for matched commutation path
dbst = ps.RealisticDiodeParams()
dbst.V_F0 = 0.85; dbst.R_d = 30e-3; dbst.V_F0_tc = -2e-3
dbst.T_ref = 125.0; dbst.Qrr = 60e-9
dbst.R_th_ja = 25.0; dbst.T_amb = T_amb
dbst.g_on = 1.0 / dbst.R_d
dbst.C_j = 1e-6                     # heavy snubber
c.add_diode("D_boost", n_sw, n_bus, dbst)

# Output cap (pre-charged)
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

print("Design point:")
print(f"  AC : {V_AC_rms} V_rms ({V_AC_peak:.0f} V peak) / {f_line:.0f} Hz")
print(f"  Bulk (SS) ≈ {V_bulk_steady:.0f} V")
print(f"  V_bus design = {V_bus_design:.0f} V → R_load = {R_load_val:.0f} Ω → P = {P_target} W")
print(f"  L_boost = {L_boost_val*1e6:.0f} µH, f_sw = {f_sw/1e3:.0f} kHz, D = {duty_pwm:.0%}")
print(f"  Snubbers: C_oss = {mp.C_oss*1e6:.0f} µF, C_j = {dbst.C_j*1e9:.0f} nF")


# =============================================================================
# Simulate (2 line cycles)
# =============================================================================
opts = ps.SimulationOptions()
opts.tstart = 0.0; opts.tstop = 33e-3
opts.dt = 500e-9
opts.dt_min = 1e-9; opts.dt_max = 2e-6
opts.adaptive_timestep = True
opts.enable_bdf_order_control = True
opts.enable_events = True
opts.enable_losses = True
opts.newton_options.num_nodes = c.num_nodes()
opts.newton_options.num_branches = c.num_branches()

print(f"\nSimulating  tstop = {opts.tstop*1e3:.0f} ms  dt = {opts.dt*1e9:.0f} ns")
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
# Steady-state metrics (last 30 % of sim)
# =============================================================================
mid = int(0.7 * len(t))
print("\n--- Steady-state operating point (last 30 % of sim)")
print(f"  V_bulk : avg = {v_bulk[mid:].mean():7.1f} V    ripple = {v_bulk[mid:].max()-v_bulk[mid:].min():.1f} V")
print(f"  V_bus  : avg = {v_bus[mid:].mean():7.1f} V    ripple = {v_bus[mid:].max()-v_bus[mid:].min():.2f} V")
print(f"  V_sw   : min = {v_sw[mid:].min():7.1f}     max = {v_sw[mid:].max():.1f}   (expected ~ 0..{v_bus[mid:].mean()+10:.0f} V)")
print(f"  I_load : {i_load[mid:].mean():.3f} A")


# =============================================================================
# Per-device loss + thermal
# =============================================================================
devices = [
    ("Brg__D1", "diode",     "Bridge D1"),
    ("Brg__D2", "diode",     "Bridge D2"),
    ("Brg__D3", "diode",     "Bridge D3"),
    ("Brg__D4", "diode",     "Bridge D4"),
    ("C_bulk",  "capacitor", "Bulk cap (ESR)"),
    ("L_boost", "inductor",  "Boost L (DCR)"),
    ("M_pfc",   "mosfet",    "PFC MOSFET"),
    ("D_boost", "diode",     "Boost D (Qrr)"),
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
# Power balance
# =============================================================================
p_load    = c.resistor_average_power("R_load")
loss_conv = total - p_load
p_in      = p_load + loss_conv
result.loss_summary.input_power = p_in
result.loss_summary.compute_totals()

print("\n--- Power balance (load) — caveat: switching-device P over-counted")
print(f"  P_out (R_load)             = {p_load:8.2f} W   ({p_load/P_target*100:.1f} % of target)")
print(f"  P_loss (converter total)   = {loss_conv:8.2f} W")
print(f"  P_in                       = {p_in:8.2f} W")
print(f"  efficiency η (reported)    = {p_load/p_in*100:8.2f} %")


# =============================================================================
# Plot waveforms
# =============================================================================
fig, ax = plt.subplots(4, 2, figsize=(14, 10))
fig.suptitle(
    f"550 W boost PFC — full chain, converging with snubber tuning\n"
    f"f_sw = {f_sw/1e3:.0f} kHz, D = {duty_pwm:.0%}, "
    f"V_bus = {v_bus[mid:].mean():.0f} V",
    fontsize=12,
)

ax[0,0].plot(t*1e3, v_ac, color="#1f77b4"); ax[0,0].set_ylabel("V_AC (V)"); ax[0,0].set_title("AC input")
ax[1,0].plot(t*1e3, v_bulk, color="#2ca02c"); ax[1,0].set_ylabel("V_bulk (V)"); ax[1,0].set_title("After bridge")
ax[2,0].plot(t*1e3, v_bus, color="#d62728"); ax[2,0].set_ylabel("V_bus (V)"); ax[2,0].set_title("Boost output bus")
ax[2,0].axhline(V_bus_design, ls="--", color="grey", alpha=0.5, label=f"design = {V_bus_design:.0f} V")
ax[2,0].legend(loc="lower right", fontsize=8)
ax[3,0].plot(t*1e3, i_load, color="#9467bd"); ax[3,0].set_ylabel("I_load (A)"); ax[3,0].set_xlabel("t (ms)"); ax[3,0].set_title("Load current")
for a in ax[:,0]: a.grid(alpha=0.3)

t0 = t[-1] - 200e-6
mask = (t >= t0) & (t <= t0 + 4/f_sw)
ax[0,1].plot(t[mask]*1e6, v_gate[mask], color="#1f77b4"); ax[0,1].set_ylabel("V_gate"); ax[0,1].set_title("MOSFET gate (PWM)")
ax[1,1].plot(t[mask]*1e6, v_sw[mask], color="#2ca02c"); ax[1,1].set_ylabel("V_sw"); ax[1,1].set_title("Switching node")
ax[2,1].plot(t[mask]*1e6, v_bulk[mask]-v_sw[mask], color="#d62728"); ax[2,1].set_ylabel("V_L"); ax[2,1].set_title("Boost inductor voltage")
ax[3,1].plot(t[mask]*1e6, v_bus[mask], color="#9467bd"); ax[3,1].set_ylabel("V_bus"); ax[3,1].set_xlabel("t (µs)"); ax[3,1].set_title("Bus ripple")
for a in ax[:,1]: a.grid(alpha=0.3)

fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig("/tmp/boost_pfc_full.png", dpi=120)
print(f"\nWaveforms saved → /tmp/boost_pfc_full.png")
