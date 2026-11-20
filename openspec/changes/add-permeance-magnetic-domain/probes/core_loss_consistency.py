"""Task 4.5 — a CONSISTENCY report, not a validation.

Sine loss data (synthetic three-term data: k_h*f*B^beta + k_e*f^2*B^2 +
k_x*f^1.5*B^1.5; an illustration, not a material) are fitted twice:
by `fit_core_loss_elements` (eddy R_m + excess c in the loop, the
hysteresis term k_h*f*B^beta as the fit reports it — it belongs to a JA
segment, rate-independent per cycle) and by a Steinmetz law k*f^a*B^b
(log-linear least squares) for iGSE. Both then predict the loss of a
TRIANGULAR flux (a +/-V square voltage on the winding) at the same peak
B and frequency. The in-loop eddy and excess shares are simulated; the
difference to iGSE is printed. Nothing here is gated: two models built
from the same sine data disagree on a triangle by construction.

Run: PYTHONPATH=<worktree>/python python core_loss_consistency.py
"""
import math

import numpy as np

import pulsim as p

K_H, BETA, K_E, K_X = 40.0, 2.3, 0.6, 1.8
A, SEG_L, N = 76e-6, 72e-3, 10


def sine_data():
    f = np.repeat([25e3, 50e3, 100e3, 200e3], 3)
    B = np.tile([0.05, 0.1, 0.2], 4)
    return f, B, K_H * f * B ** BETA + K_E * f ** 2 * B ** 2 + K_X * f ** 1.5 * B ** 1.5


def triangle_in_loop(fit, f, B_pk):
    """Eddy + excess loss [W] of a triangular flux of peak B_pk at f."""
    T = 1 / f
    U = 4 * B_pk * A * f                     # |dPhi/dt| per turn on each slope
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "e", "gnd", -N * U, N * U, 0.0, T / 2, T,
                               T * 1e-4, T * 1e-4)
    b.add_winding("W", "e", "gnd", "m1", "gnd", N)
    b.add_permeance("P", "m1", "m2", 2e-6)
    b.add_magnetic_resistance("Rm", "m2", "m3", fit.R_m)
    b.add_excess_loss("X", "m3", "gnd", fit.c, 1e-4 * U)
    r = p.simulate(b, t_end=3 * T, dt=T / 20000, engine="pwl")
    t = np.asarray(r.times)
    w = dict(t_from=t[-1] - 2 * T, t_to=t[-1])
    P_exc = p.time_average(t, r.v("X.e") * r.i("X"), **w)
    P_edd = p.time_average(t, (r.v("m2") - r.v("m3")) ** 2 / fit.R_m, **w)
    return P_edd, P_exc, U


def report():
    f, B, P = sine_data()
    fit = p.fit_core_loss_elements(f, B, P, geometry=(A, SEG_L))
    X = np.column_stack([np.ones_like(f), np.log(f), np.log(B)])
    lk, alpha, beta = np.linalg.lstsq(X, np.log(P), rcond=None)[0]
    mat = p.CoreMaterial(name="sine-fit", K=float(math.exp(lk)), alpha=float(alpha),
                         beta=float(beta))
    rows = []
    vol = A * SEG_L
    for fk in (50e3, 100e3):
        for Bk in (0.1, 0.2):
            P_edd, P_exc, U = triangle_in_loop(fit, fk, Bk)
            P_hys = fit.k_h * fk * Bk ** fit.beta * vol
            T = 1 / fk
            t = np.linspace(0, T, 4001)
            Bt = np.interp(t % T, [0, T / 2, T], [-Bk, Bk, -Bk])
            P_igse = p.igse_loss_density(Bt, t, mat) * vol
            rows.append((fk, Bk, P_hys, P_edd, P_exc, P_igse,
                         K_X * (U / A) ** 1.5 / (fit.c * U ** 1.5 / vol)))
    return fit, (math.exp(lk), alpha, beta), rows


if __name__ == "__main__":
    fit, st, rows = report()
    print(f"fit: k_h={fit.k_h:.6g} beta={fit.beta:.6g} R_m={fit.R_m:.6g} S c={fit.c:.6g}")
    print(f"Steinmetz on the same sine data: k={st[0]:.4g} alpha={st[1]:.4f} beta={st[2]:.4f}")
    print("f [kHz]  B [T]  hyst [W]  eddy [W]  excess [W]  in-loop sum [W]  iGSE [W]  diff")
    for fk, Bk, h, e, x, g, _ in rows:
        s = h + e + x
        print(f"{fk/1e3:7.0f}  {Bk:5.2f}  {h:8.4f}  {e:8.4f}  {x:10.4f}  {s:15.4f}  {g:8.4f}  {(s-g)/g:+.3f}")
