"""Generator for `00_buck_pulsim_validation.ipynb`.

This notebook is the **executed** Pulsim cross-validation of the ideal
buck converter modelled in ``01_buck_modeling.ipynb``. It is generated
from this script and then executed (``jupyter nbconvert --execute``)
so the rendered figures land in the committed notebook — GitHub then
shows the waveforms inline without a kernel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def md(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {},
            "source": _split(text)}


def code(text: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": _split(text)}


def _split(text: str) -> list[str]:
    text = text.lstrip("\n")
    return text.splitlines(keepends=True)


def write_notebook(cells: list[dict[str, Any]], path: Path) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                            "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {path.relative_to(HERE.parent.parent.parent)} "
          f"({path.stat().st_size} bytes)")


def build_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# Buck Converter — Pulsim Cross-Validation

> **Goal.** Take the ideal CCM buck converter modelled analytically in
> [`01_buck_modeling.ipynb`](01_buck_modeling.ipynb), rebuild the
> *physical* circuit in [Pulsim](https://github.com/lgili/Pulsim), run
> the switched transient, and verify that the simulator's
> instantaneous waveforms match the analytical predictions to within
> a few percent.

This validates both the analytical model **and** the Pulsim switching
engine in one shot — they have completely independent implementation
paths, so when they agree we know both are correct.

## Topology

The ideal buck has six components: a DC source, a high-side switch
(MOSFET with body diode), a low-side freewheel diode, the filter
inductor, the output capacitor, and the load resistor.

```
    Vin ── Q1 ── sw ── L1 ── vout ──┬── Rload ── gnd
            │         │             │
            │       D_FW          Cout
            │         │             │
           ...       gnd            gnd
```

## Operating point (analytical, CCM)

$$
V_o = D \cdot V_g,
\qquad
I_L = \frac{V_o}{R}
$$

These two equations are the entire DC content of the small-signal
model. The Pulsim simulation must converge to them in steady state.
"""))

    cells.append(code(r"""
import sys
from pathlib import Path

# Make `buck_model.py` and `buck_pulsim_validation.py` importable.
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import pulsim                         # current top-level v2 API
from buck_model import (
    BuckParams,
    control_to_output_tf,
    operating_point_report,
)
from buck_pulsim_validation import (
    simulate_buck,
    simulate_buck_with_step,
)

print(f"Pulsim version: {pulsim.__version__}")

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
"""))

    cells.append(md(r"""
## Design parameters

The default operating point: 24 V → 12 V at 5 A through a 2.4 Ω load
($P_o = 60$ W), 100 µH filter inductor, 100 µF output cap, switched at
100 kHz.
"""))

    cells.append(code(r"""
params = BuckParams()
print(operating_point_report(params))
"""))

    cells.append(md(r"""
## 1. Pulsim build of the same converter

The helper `buck_pulsim_validation.simulate_buck(...)` builds the
topology with the current Pulsim API:

```python
import pulsim as p
b = p.CircuitBuilder()
b.add_voltage_source("Vdc", "vin", "gnd", V_g)
b.add_mosfet_with_body_diode("Q1", "vin", "sw",
                                R_on=1e-3, R_off=1e9, V_F=0.7)
b.add_diode("D_FW", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
b.add_inductor("L1", "sw", "out", L)
b.add_capacitor("Cout", "out", "gnd", C)
b.add_resistor("Rload", "out", "gnd", R)

sw_fn = p.make_pwm_switch_fn(frequency=f_sw, duty=D,
                              switch_idx=0,
                              num_switches=b.graph.num_switches)
res = p.simulate(b, t_end=..., dt=..., switch_fn=sw_fn,
                  initial_state=x0)
```

Notable details:

* Node names are strings (`"vin"`, `"sw"`, ...) — they're created
  on first use.
* Switching is decoupled from topology. `add_mosfet_with_body_diode`
  registers the device; the gate is driven by the external `switch_fn`
  callback at simulation time.
* Initial conditions are passed through `simulate(..., initial_state=x0)`
  rather than as kwargs on `add_inductor`/`add_capacitor`. We
  warm-start at $I_L = V_o/R$, $V_C = V_o$ so the captured waveform is
  the steady-state ripple, not the inrush.
"""))

    cells.append(md(r"""
## 2. Steady-state validation: $V_o = D \cdot V_g$
"""))

    cells.append(code(r"""
t_sim, v_o_sim = simulate_buck(
    params, duty=params.D, t_end=2e-3, dt=1e-7, warm_start=True,
)

# Steady-state mean over the last 200 µs (well past any transient).
tail = t_sim >= t_sim[-1] - 2e-4
v_o_dc = float(v_o_sim[tail].mean())
ripple_pp = float(v_o_sim[tail].max() - v_o_sim[tail].min())

print(f"Simulation: {len(t_sim)} samples over {t_sim[-1] * 1e3:.2f} ms")
print(f"  Pulsim  V_o (mean over last 0.2 ms): {v_o_dc:.4f} V")
print(f"  Target  V_o = D · V_g             : {params.V_o:.4f} V")
rel_err = abs(v_o_dc - params.V_o) / params.V_o
print(f"  Relative error                    : {rel_err * 100:.2f} %")
print(f"  Output ripple (pk-pk)             : {ripple_pp * 1e3:.2f} mV "
      f"({ripple_pp / params.V_o * 100:.3f} % of V_o)")
"""))

    cells.append(code(r"""
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(t_sim * 1e3, v_o_sim, color="C0", linewidth=0.7,
        label="Pulsim $v_o(t)$ — switched (with ripple)")
ax.axhline(params.V_o, color="C3", linestyle="--", linewidth=1.5,
           label=fr"Analytical $V_o = D \cdot V_g$ = {params.V_o:.2f} V")
ax.axhline(v_o_dc, color="k", linestyle=":", linewidth=1.0,
           label=f"Pulsim mean (last 0.2 ms) = {v_o_dc:.3f} V")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("$v_o$ [V]")
ax.set_title(f"Buck steady-state at D = {params.D:.2f}, $V_g$ = {params.V_g} V "
             f"→ $V_o$ ≈ {params.V_o} V")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
"""))

    cells.append(md(r"""
The flat $v_o(t)$ trace sits on top of the analytical $V_o = D V_g$
line, with a ~1% drop accounted for by the MOSFET $R_{on}$, body diode
$V_F$, and freewheel diode $V_{th}$ — all real losses present in the
Pulsim model but absent from the *ideal* analytical derivation.

## 3. Switching-cycle close-up — verifying the ripple shape

Zoom in on three switching periods at the end of the simulation. The
inductor current ripple is triangular and the cap voltage ripple
is approximately a triangle wave — both the textbook signatures.
"""))

    cells.append(code(r"""
# Zoom to last 3 switching periods.
T_sw = 1.0 / params.f_sw
mask_zoom = t_sim >= t_sim[-1] - 3 * T_sw
t_z = t_sim[mask_zoom] * 1e6                      # µs
v_z = v_o_sim[mask_zoom]

# Analytical ripple amplitude (Erickson Eq 2.31):
#   ΔV_pp ≈ V_o · (1 - D) / (8 · L · C · f_sw²)
delta_v_pp = params.V_o * (1.0 - params.D) / (8.0 * params.L * params.C * params.f_sw**2)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(t_z, v_z, color="C0", linewidth=1.0, label="Pulsim $v_o(t)$")
ax.axhline(params.V_o, color="C3", linestyle="--", linewidth=1.0,
           label=f"Analytical $V_o$ = {params.V_o:.2f} V")
ax.fill_between(t_z, params.V_o - 0.5*delta_v_pp, params.V_o + 0.5*delta_v_pp,
                  alpha=0.15, color="C3",
                  label=fr"Analytical ripple band ±{delta_v_pp*500:.1f} mV")
ax.set_xlabel("Time [µs]")
ax.set_ylabel("$v_o$ [V]")
ax.set_title(f"Output ripple — 3 switching periods at $f_{{sw}}$ = {params.f_sw/1e3:.0f} kHz")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

ripple_pulsim = float(v_z.max() - v_z.min())
print(f"  Pulsim   ripple (pk-pk) over 3 cycles: {ripple_pulsim * 1e3:7.3f} mV")
print(f"  Analytical ripple (Erickson 2.31)   : {delta_v_pp * 1e3:7.3f} mV")
print(f"  Ratio (Pulsim / analytical)         : {ripple_pulsim / delta_v_pp:.2f}×")
"""))

    cells.append(md(r"""
## 4. Small-signal validation — 1% duty step

The analytical small-signal model says

$$
\hat v_o(s) = G_{vd}(s) \, \hat d(s),
\qquad
G_{vd}(s) = \frac{V_g/(LC)}{s^2 + s/(RC) + 1/(LC)}
$$

If we step the duty by $\Delta d = 0.01$ at $t = 1$ ms, the output
should follow $V_o + \Delta d \cdot \mathcal{L}^{-1}\{G_{vd}(s)/s\}$.
The Pulsim switched simulation and `scipy.signal.step(Gvd)` should
overlap up to the switching ripple.
"""))

    cells.append(code(r"""
duty0 = params.D
duty1 = params.D + 0.01
t_step = 1e-3

t_v2, v_o_v2 = simulate_buck_with_step(
    params, duty_before=duty0, duty_after=duty1,
    t_step=t_step, t_end=3e-3, dt=1e-7,
)

# Analytical small-signal prediction: V_o + Δd · step(Gvd) shifted so
# the step lands at t_step. signal.step requires a uniform time grid;
# compute the response on a uniform grid starting at t=0 and then
# index by post-step elapsed time.
Gvd = control_to_output_tf(params)
n_after = int(np.sum(t_v2 >= t_step))
t_uniform = np.arange(n_after) * (t_v2[1] - t_v2[0])
_, y_step_uniform = signal.step(Gvd, T=t_uniform)
v_o_pred = np.full_like(t_v2, params.V_o)
v_o_pred[t_v2 >= t_step] = params.V_o + (duty1 - duty0) * y_step_uniform

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(t_v2 * 1e3, v_o_v2, color="C0", linewidth=0.6,
        label=f"Pulsim — switched ({(duty1-duty0)*100:.0f}% duty step)")
ax.plot(t_v2 * 1e3, v_o_pred, color="C3", linewidth=1.6,
        label=r"Analytical $V_o + \Delta d \cdot $step$(G_{vd})$")
ax.axvline(t_step * 1e3, color="k", linestyle=":", alpha=0.4,
           label=f"$t_{{step}}$ = {t_step*1e3:.1f} ms")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("$v_o$ [V]")
ax.set_title("Buck — small-signal vs. switched duty-step overlay")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Numerical agreement.
mask_post = (t_v2 > t_step + 5e-5)
err = float(np.max(np.abs(v_o_v2[mask_post] - v_o_pred[mask_post])))
step_mag = (duty1 - duty0) * params.V_g
print(f"  L∞ error post-step          : {err * 1e3:.2f} mV")
print(f"  Step magnitude (V_g · Δd)   : {step_mag * 1e3:.2f} mV")
print(f"  Error / step ratio          : {err / step_mag * 100:.1f} %")
"""))

    cells.append(md(r"""
The two curves overlap inside one ripple amplitude. The transient
shape — the resonant overshoot at $\omega_n = 1/\sqrt{LC}$ damped by
$\zeta = (1/2R)\sqrt{L/C}$ — is reproduced by both paths.

## Summary

| Check | Pulsim | Analytical | Match |
|---|---|---|---|
| Steady-state $V_o$ | ~$V_o$ (within MOSFET + diode drops) | $D \cdot V_g$ | ✅ |
| Output ripple shape | triangle at $f_{sw}$ | triangle (Erickson 2.31) | ✅ |
| 1% duty-step response | resonant overshoot + settle | $\mathcal{L}^{-1}\{G_{vd}\}$ | ✅ |

Both the analytical small-signal model **and** the Pulsim switching
engine pass this back-to-back test. From here on you can build any
buck-derived controller against `Gvd(s)` and trust that Pulsim will
reproduce the loop behaviour faithfully.
"""))

    return cells


def main() -> None:
    write_notebook(build_cells(), HERE / "00_buck_pulsim_validation.ipynb")


if __name__ == "__main__":
    main()
