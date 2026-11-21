"""device_loss_summary: a RESISTOR's P_avg in a hard-switched branch reads low by
~h/(3·t_on) (edges joined by a one-step ramp, AI_GUIDE trap 14), while the
SWITCH's P_avg in the same branch is exact. 48 V, R = 1 Ω, switch R_on = 10 mΩ,
50 % PWM at 20 kHz.

Reference (examples audit, 2026-11-13): R_line 1114.24 / 1121.77 / 1125.54 W
at 50 / 100 / 200 steps per period against the exact 1129.30 W (first
order in dt); M1 exact at every dt. Owner: the loss post-processing
(python/pulsim/losses — device_loss_summary integrates R·i² over the
stored samples, so each switching edge is a one-step ramp)."""
import pulsim as p
for n in (50, 100, 200):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_resistor("R_line", "vin", "ds", 1.0)
    b.add_switch("M1", "ds", "gnd", g_on=100.0, g_off=1e-9)
    T = 5e-5
    def pwm(t):
        m = p.SwitchStateMask(1); m.set(0, (t % T) < 0.5 * T); return m
    r = p.simulate(b, t_end=200 * T, dt=T / n, switch_fn=pwm)
    L = {e["name"]: e["P_avg"] for e in p.device_loss_summary(b, r, switch_fn=pwm)}
    i_pk = 48 / 1.01
    print(f"steps/period {n}: R_line {L['R_line']:.2f} W (exact {0.5*i_pk*i_pk:.2f}), M1 {L['M1']:.4f} W (exact {0.5*i_pk*i_pk*0.01:.4f})")
