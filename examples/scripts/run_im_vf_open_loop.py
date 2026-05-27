#!/usr/bin/env python3
"""IM V/f open-loop ramp from standstill — accelerate the
4-pole 4 kW machine by ramping the source frequency from 0 → 50 Hz
with proportional voltage (constant V/f ratio).

This is the CANONICAL open-loop start strategy for industrial IM
drives: keep V/f ratio constant so the magnetising flux stays
roughly nominal as the motor accelerates, then hand off to a
closed-loop FOC (sensorless or sensored) above ~10 % rated speed.

Topology
--------
3 sine voltage sources (Va, Vb, Vc) feeding the IM through a short
series R per phase. The sine-source amplitude AND frequency are
modulated each simulation step by a Python step_observer wrapping
the IM observer.

Comparison to ``run_im_direct_online_start.py``
-----------------------------------------------
That demo starts the motor "direct on line" — full nominal V at full
nominal f from t=0. The rotor experiences a violent inrush current
and large torque pulsation before settling. THIS demo shows the
softer industrial start: the source ramps over 1 s, the motor
follows synchronously through the whole speed range, and the peak
current stays close to rated.

Honest caveats
--------------
* Pulsim ``add_sine_voltage_source`` takes amplitude+frequency at
  build time and the kernel uses them as constants per simulation.
  To ramp them we either (a) rebuild the cache mid-sim (expensive)
  or (b) override the source via ``b_extra_fn`` — we go with (b)
  here. The 3 underlying sine sources are zeroed at build time
  and the observer injects the time-varying values via b_extra.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


# ---------- 4 kW 400 V 50 Hz 4-pole reference ------------------------------

P_RATED = 4_000.0
V_LL    = 400.0
F_END   = 50.0       # final frequency
PP      = 2
RAMP_S  = 1.0        # ramp 0 → F_END over this duration
HOLD_S  = 0.5        # hold at F_END for this long

DT      = 50e-6
T_END   = RAMP_S + HOLD_S


def main():
    # Phase peak voltage (line-to-line / sqrt(3) * sqrt(2)).
    V_AMP_NOMINAL = V_LL * math.sqrt(2.0 / 3.0)
    # V/f ratio: keep amplitude proportional to frequency so the flux
    # stays nominal.  Below the "boost frequency" (~5 Hz) we usually
    # add a small voltage boost to overcome Rs drop — included here.
    V_BOOST = 0.05 * V_AMP_NOMINAL
    F_BOOST = 5.0

    def vf(t: float) -> tuple[float, float]:
        """Return (V_amp_now, f_now) at simulation time ``t``."""
        if t < RAMP_S:
            f_now = F_END * (t / RAMP_S)
        else:
            f_now = F_END
        # V/f profile with low-frequency boost.
        if f_now > F_BOOST:
            v_now = V_AMP_NOMINAL * (f_now / F_END)
        else:
            # Linear interp between V_BOOST at f=0 and the V/f line
            # at f = F_BOOST.
            v_at_boost = V_AMP_NOMINAL * (F_BOOST / F_END)
            v_now = V_BOOST + (v_at_boost - V_BOOST) * (f_now / F_BOOST)
        return v_now, f_now

    # Build plant. The 3 sine sources start at 0 amplitude / 0 frequency
    # — the b_extra_fn injects the ramping voltage each step.
    b = p.CircuitBuilder()
    src_a = b.graph.num_branches
    b.add_voltage_source("Va", "a", "gnd", 0.0)
    src_b = b.graph.num_branches
    b.add_voltage_source("Vb", "b", "gnd", 0.0)
    src_c = b.graph.num_branches
    b.add_voltage_source("Vc", "c", "gnd", 0.0)
    # Small series R to limit the inrush during the very-low-frequency
    # bootstrap (essentially the cable resistance).
    b.add_resistor("Ra", "a", "a_int", 0.1)
    b.add_resistor("Rb", "b", "b_int", 0.1)
    b.add_resistor("Rc", "c", "c_int", 0.1)
    # 4 kW IM nameplate.
    kw = p.im_parameters_from_nameplate(
        P_rated_W=P_RATED, V_LL_V=V_LL, f_Hz=F_END, pole_pairs=PP)
    diag = kw.pop("_diagnostics")
    motor = p.add_induction_motor(
        b, name="IM",
        phase_nodes=("a_int", "b_int", "c_int"),
        neutral_node="n",
        T_load=0.0,
        **kw,
    )
    b.add_resistor("R_leak", "n", "gnd", 1e6)

    im_obs, im_b_extra_motor = p.make_induction_motor_observer(
        b, motor, dt=DT)

    # Resolve source-row indices so the b_extra_fn can target them.
    src_a_idx = b.pool.branch_var_id_for_source(src_a, b.graph)
    src_b_idx = b.pool.branch_var_id_for_source(src_b, b.graph)
    src_c_idx = b.pool.branch_var_id_for_source(src_c, b.graph)

    # Phase log every 5 ms.
    log_t = []
    log_omega_m = []
    log_v_amp = []
    log_f = []

    def step_observer(t, x):
        im_obs(t, x)
        if not log_t or t - log_t[-1] >= 5e-3:
            v_amp_now, f_now = vf(t)
            log_t.append(t)
            log_omega_m.append(motor.mech.omega_rad_s)
            log_v_amp.append(v_amp_now)
            log_f.append(f_now)

    def combined_b_extra(t):
        out = list(im_b_extra_motor(t))
        v_amp_now, f_now = vf(t)
        omega = 2.0 * math.pi * f_now
        # The voltage-source row of the MNA enforces V_source = b[src_idx].
        # We add the time-varying value here (overrides the 0 baked into
        # add_voltage_source).
        out[src_a_idx] += v_amp_now * math.sin(omega * t + 0.0)
        out[src_b_idx] += v_amp_now * math.sin(
            omega * t - 2.0 * math.pi / 3.0)
        out[src_c_idx] += v_amp_now * math.sin(
            omega * t + 2.0 * math.pi / 3.0)
        return out

    print(f"  IM: P={P_RATED}W, V_LL={V_LL}V, f={F_END}Hz, pp={PP}")
    print(f"  ω_sync (final) = {diag['omega_sync_rad_s']:.2f} rad/s "
          f"({diag['omega_sync_rad_s']*60/(2*math.pi):.0f} rpm)")
    print(f"  V/f profile: ramp 0→{F_END}Hz over {RAMP_S}s + "
          f"{HOLD_S}s hold; boost {V_BOOST:.1f}V below {F_BOOST}Hz")
    print(f"  Sim: {T_END}s @ dt={DT*1e6:.0f}µs = "
          f"{int(T_END/DT):,} steps")

    res = p.simulate(
        b, t_end=T_END, dt=DT,
        switch_fn=lambda t: p.SwitchStateMask(0),
        step_observer=step_observer,
        b_extra_fn=combined_b_extra,
        progress=True,
    )

    omega_final = motor.mech.omega_rad_s
    print(f"\n  Result:")
    print(f"    ω_final  = {omega_final:.2f} rad/s "
          f"({omega_final*60/(2*math.pi):.1f} rpm)")
    print(f"    %sync    = {omega_final/diag['omega_sync_rad_s']*100:.1f} %")
    print(f"    T_em     = {motor.last_T_em_Nm:.2f} Nm")
    print(f"    |ψ_r|    = "
          f"{math.hypot(motor.psi_r_alpha_Wb, motor.psi_r_beta_Wb):.4f} Wb")
    print(f"    steps    = {res.num_steps():,}")

    # CSV trace
    try:
        import csv
        out_path = Path(__file__).with_name("im_vf_ramp_trace.csv")
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "f_source_hz", "v_amp_V", "omega_m_rad_s"])
            for i in range(len(log_t)):
                w.writerow([log_t[i], log_f[i], log_v_amp[i],
                                log_omega_m[i]])
        print(f"    trace    → {out_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"    (CSV export skipped: {exc})")


if __name__ == "__main__":
    main()
