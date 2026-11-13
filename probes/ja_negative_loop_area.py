"""compute_bh_loop: a closed, symmetric sine H loop gives a NEGATIVE area
∮H dB for the `permalloy` J-A set, and the `ferrite_n87` area falls as the
drive rises into saturation. A passive material's closed loop has positive
area that grows (then saturates) with the drive.

Reference (examples audit, 2026-11-13), J/m³ per cycle, 100 kHz, 400 steps
per period, last of 5 periods (B closes to 1e-15 T):
  permalloy   H_pk 100 / 300 / 1000 A/m:  29.1 / -12.4 / -14.8
  ferrite_n87 H_pk 300 / 1000 / 3000 A/m: 42.8 / 28.8 / 13.9
Owner: python/pulsim/hysteresis.py, JilesAthertonModel.update — the H-form
irreversible slope (M_an − M)/(k·δ − α(M_an − M)) at line 336. Neither the
δ_M guard (Jiles 1992) nor clamping that slope at ≥ 0 removes the negative
permalloy area, so the sign flip of k·δ − α(M_an − M) is not the whole
cause; not resolved."""
import numpy as np
import pulsim as p
f = 100e3
t = np.linspace(0, 5 / f, 5 * 400 + 1)
for mat, drives in (("permalloy", (100, 300, 1000)), ("ferrite_n87", (300, 1000, 3000))):
    for H_pk in drives:
        r = p.compute_bh_loop(H_pk * np.sin(2 * np.pi * f * t), t,
                              params=p.reference_material(mat),
                              N_turns=1, l_m=1.0, A_core=1.0, period=1 / f)
        print(f"{mat:12s} H_pk {H_pk:5d} A/m: ∮H dB = {r.energy_per_cycle_per_m3:8.2f} J/m³")
