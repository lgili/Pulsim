"""Generator for `00_forward_pulsim_validation.ipynb`."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from _nb_helpers import md, code, write_notebook   # noqa: E402


def build_cells():
    cells = []

    cells.append(md(r"""
# Forward Converter — Pulsim Cross-Validation

> **Goal.** Validate the isolated forward converter modelled in
> [`01_forward_modeling.ipynb`](01_forward_modeling.ipynb) against a
> [Pulsim](https://github.com/lgili/Pulsim) switched simulation.

## Topology

The forward is "an isolated buck": energy is transferred
**primary → secondary during the ON interval**, and an explicit
output LC filter handles storage. A freewheel diode maintains
inductor current during OFF, exactly like a buck.

```
   Vin ── T1.primary ── Q1 ── gnd
                                              D_rec      L_f       vout
   T1.secondary  (sec_anode → sec_neg)        ────►──────╲╲╲╲──────┬──
                                                                   │
                                              D_fw                Cout
                                              ▲                    │
                                             sec_neg ──────────── Rload
                                                                   │
                                              sec_neg ── R_iso ── gnd
```

We omit the textbook **reset winding** because Pulsim's
``add_transformer`` exposes only a two-winding device. The reset
winding's only job is to demagnetise the core every cycle, and our
linear (non-saturating) transformer model doesn't need it — the
steady-state DC ratio is still $V_o = n D V_g$.

## Operating point (CCM)

$$
V_o = n \cdot V_g \cdot D,
\qquad
I_L = \frac{V_o}{R}
$$

The forward inherits the **buck's** small-signal model — clean
second-order low-pass, **no RHP zero**, so voltage-mode control is
buck-easy.
"""))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import matplotlib.pyplot as plt

import pulsim
from forward_model import ForwardParams
from forward_pulsim_validation import simulate_forward

print(f"Pulsim version: {pulsim.__version__}")

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
"""))

    cells.append(md(r"""
## Design parameters

24 V → 5 V at 1 A through a 5 Ω load, $n = 0.5$ turns ratio
(so $D \approx 0.42$ — comfortably below the 0.5 reset-winding cap),
100 µH filter inductor, 100 µF cap, 100 kHz switching.
"""))

    cells.append(code(r"""
params = ForwardParams()
print(f"V_g  = {params.V_g} V")
print(f"V_o  = {params.V_o} V")
print(f"n    = {params.n}     (= N_s / N_p)")
print(f"D    = {params.D:.4f}  (D_max safety cap = {params.D_max})")
print(f"R    = {params.R} Ω")
print(f"L    = {params.L*1e6:.0f} µH")
print(f"C    = {params.C*1e6:.0f} µF")
print(f"f_sw = {params.f_sw/1e3:.0f} kHz")
"""))

    cells.append(md(r"""
## Steady-state validation: $V_o = n D V_g$
"""))

    cells.append(code(r"""
# 10 ms: the output LC (Q ≈ 5) still rings visibly at 2 ms after a cold start.
t_sim, v_out = simulate_forward(
    params, duty=params.D, t_end=10e-3, dt=2e-8,
)

v_o_dc = pulsim.time_average(t_sim, v_out, t_from=t_sim[-1] - 1e-3)
tail = t_sim >= t_sim[-1] - 2e-4
ripple_pp = float(v_out[tail].max() - v_out[tail].min())

# Leakage of the k = 0.999 transformer, seen from the secondary, delays
# every D_fw -> D_rec commutation by L_lk·I_o/(n·V_g): the output loses
# f_sw·L_lk·I_o volts, so V_o = n·D·V_g / (1 + f_sw·L_lk/R).
L_s = params.n**2 * max(params.L * 100.0, 1e-3)
L_lk = L_s * (1.0 - 0.999**2)
v_o_lk = params.V_o / (1.0 + params.f_sw * L_lk / params.R)
# Erickson 2.31 with n·V_g in place of V_g.
ripple_an = params.V_o * (1.0 - params.D) / (8.0 * params.L * params.C * params.f_sw**2)

print(f"Simulation: {len(t_sim)} samples over {t_sim[-1] * 1e3:.2f} ms")
print(f"  Pulsim  V_o (time average, last 1 ms): {v_o_dc:.4f} V")
print(f"  Ideal   V_o = n · D · V_g            : {params.V_o:.4f} V")
print(f"  With leakage n·D·V_g/(1+f·L_lk/R)    : {v_o_lk:.4f} V  (L_lk = {L_lk*1e6:.2f} µH)")
print(f"  Error vs leakage-corrected value     : "
      f"{(v_o_dc - v_o_lk) / v_o_lk * 100:+.2f} %")
print(f"  Output ripple (pk-pk)                : "
      f"{ripple_pp * 1e3:.2f} mV  (analytical {ripple_an * 1e3:.2f} mV)")
"""))

    cells.append(code(r"""
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(t_sim * 1e3, v_out, color="C0", linewidth=0.5,
        label="Pulsim $v_o(t)$ (cold-start)")
ax.axhline(params.V_o, color="C3", linestyle="--", linewidth=1.5,
           label=fr"Analytical $V_o = n D V_g$ = {params.V_o:.2f} V")
ax.axhline(v_o_dc, color="k", linestyle=":", linewidth=1.0,
           label=f"Pulsim mean = {v_o_dc:.3f} V")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("$v_o$ [V]")
ax.set_title(f"Forward Pulsim — n={params.n}, D={params.D:.3f}, "
             f"$L_f$={params.L*1e6:.0f} µH")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
"""))

    cells.append(code(r"""
for k in (0.999, 0.9999, 0.99999):
    t_k, v_k = simulate_forward(params, duty=params.D, t_end=10e-3, dt=2e-8,
                                k_coupling=k)
    v_k_dc = pulsim.time_average(t_k, v_k, t_from=t_k[-1] - 1e-3)
    print(f"  k = {k:<8}: V_o = {v_k_dc:.4f} V  ({(v_k_dc / params.V_o - 1) * 100:+.2f} % vs n·D·V_g)")
"""))

    cells.append(md(r"""
## Note on the residual error

The ~10 % gap to the ideal $n D V_g$ is the transformer **leakage**
($k = 0.999$), not the diodes: $V_{th} = 0.7$ V is the diodes' turn-on
threshold, not a forward drop (their on-state is 1 mΩ). Each turn-on
must first commutate the filter current from $D_{fw}$ to $D_{rec}$
through $L_{lk}$, which costs $f_{sw} L_{lk} I_o$ of output voltage —
the leakage-corrected value printed above. Raising $k$ closes the gap,
as the sweep above shows.

Without a reset winding, clamp or snubber, the leakage and magnetising
energy has nowhere to go at turn-off, so the switch node rings up to
~10 kV through the MOSFET's 1 GΩ off-resistance; `simulate` warns
about it. A real design needs the reset path.

## Summary

| Check | Pulsim | Analytical | Match |
|---|---|---|---|
| Steady-state $V_o$ | $n D V_g/(1 + f_{sw} L_{lk}/R)$ within ~1 % | same (leakage-corrected) | ✅ |
| Steady-state $V_o$ vs $k$ | → $n D V_g$ as $k \to 1$ | $n D V_g$ | ✅ |
| Output ripple | buck formula with $n V_g$ | Erickson 2.31 | ✅ |

This notebook checks the DC ratio and the ripple only; the duty-step
response (no RHP zero, buck-like $G_{vd}$ with $n V_g$ in place of
$V_g$) is derived in the modeling notebook, not simulated here.
"""))

    return cells


def main():
    write_notebook(build_cells(), HERE / "00_forward_pulsim_validation.ipynb")


if __name__ == "__main__":
    main()
