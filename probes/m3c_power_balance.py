"""Probe: M3C (notebook 05) power balance and fundamental voltages.

The modules are ideal voltage sources whose capacitor voltages are a
bookkeeping state in the controller, so P_in - P_out is the power the
sources deliver; with the bookkeeping balanced it should be ~0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects/inverters/m3c_3phase"))
from m3c_3phase_model import (  # noqa: E402
    M3cParams, build_l1_plant, predict_i_out_peak,
    run_l1_dq_full_closed_loop_with_cap_loop,
)

T_END = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
DT = 25e-6
prm = M3cParams()
r, ctrl, *_ = run_l1_dq_full_closed_loop_with_cap_loop(
    build_l1_plant(prm), prm, i_d_out_ref=predict_i_out_peak(prm), t_end=T_END, dt=DT)
n = int(round(0.2 / DT))  # 10 input / 9 output periods
t = r.t[-n:]
ph = np.pi / 2
VA = [prm.V_in_phase_peak * np.sin(prm.omega_in * t + ph - k * 2 * np.pi / 3) for k in range(3)]
IN = [r.i_a_in[-n:], r.i_b_in[-n:], r.i_c_in[-n:]]
OUT = [r.i_a_out[-n:], r.i_b_out[-n:], r.i_c_out[-n:]]
p_in = np.mean(sum(v * i for v, i in zip(VA, IN)))
p_out = prm.R_load * np.mean(sum(i * i for i in OUT))


def fund(x, w):
    return float(np.hypot(2 * np.mean(x * np.cos(w * t)), 2 * np.mean(x * np.sin(w * t))))


i1 = fund(OUT[0], prm.omega_out)
z = abs(prm.R_load + 1j * prm.omega_out * prm.L_load)
print(f"P_in = {p_in/1e6:.4f} MW, P_out (R i^2) = {p_out/1e6:.4f} MW, "
      f"modules deliver {(p_out-p_in)/1e3:+.1f} kW")
print(f"I_a_out fundamental = {i1:.2f} A, V_ab load fundamental = {np.sqrt(3)*z*i1/1e3:.3f} kV "
      f"(Tab 16 {prm.V_out_LL_peak/1e3:.2f} kV)")
print(f"I_A_in fundamental = {fund(IN[0], prm.omega_in):.2f} A")
h = np.array(ctrl.v_caps_module_history)[-int(0.5 / prm.T_s):]
tgt = prm.v_cap_total_per_module
print(f"cap max |dev| over last 0.5 s = {np.max(np.abs(h - tgt))/tgt*100:.2f} %")
