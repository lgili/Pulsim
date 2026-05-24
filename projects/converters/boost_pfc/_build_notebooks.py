"""Generator for the boost PFC teaching notebooks (4 notebooks).

Run once after editing to regenerate the four ipynbs. Same patterns
as the other converters under projects/converters/, scaled up because
PFC has two control objectives (regulate V_o AND shape i_in) and two
distinct conduction modes (DCM and CCM) that deserve separate
treatment.

The 4 notebooks:
  01_boost_pfc_basics       — what is PFC, definitions, conduction modes
  02_boost_pfc_dcm          — DCM analysis + single-loop voltage controller
  03_boost_pfc_ccm          — CCM analysis + two-loop (current + voltage)
  04_boost_pfc_simulation   — DCM vs CCM comparison + line/load steps
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def md(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _split_lines(text)}


def code(text: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _split_lines(text)}


def _split_lines(text: str) -> list[str]:
    text = text.lstrip("\n")
    return text.splitlines(keepends=True)


def write_notebook(cells, path: Path) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {path.relative_to(HERE.parent.parent)} ({path.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# Notebook 1 — Basics
# ---------------------------------------------------------------------------


def build_basics_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 1 — Boost PFC Basics

> **Goal.** Understand *why* power-factor correction exists, what
> PF and THD mean, and how the boost stage between a diode bridge
> and the output cap turns a "bad" rectifier (PF ≈ 0.6, THD > 100%)
> into a "good" one (PF > 0.99, THD < 5%). Preview the two strategies
> that the next notebooks dive into: **DCM** (natural PFC, single
> voltage loop) and **CCM** (active current shaping, two loops).

**Prerequisites**

- Boost converter modeling notebook (`projects/converters/boost/`).
  The PFC is a boost stage with rectified AC at the input, plus a
  control architecture that has to do TWO things at once (regulate
  $V_o$ AND shape $i_{in}$).

**What you'll know at the end**

1. Why a cheap rectifier-with-cap fails IEC 61000-3-2.
2. What "power factor" and "THD" actually mean.
3. How the boost topology after a bridge can fix both.
4. The DCM vs CCM trade-off: $L$ small or large, single loop or two
   loops, natural shaping or active shaping.
5. Where each mode lives in the design space.
"""))

    cells.append(md("## Setup"))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from boost_pfc_model import (
    BoostPFCParams,
    dcm_steady_state_duty_exact,
    dcm_input_current_instantaneous,
    operating_point_report,
    power_factor, thd_current, line_band_filter,
)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
"""))

    cells.append(md(r"""
## 1. The "bad" rectifier

Imagine the simplest AC-to-DC front end: a diode bridge feeding a
large cap. The cap holds the output near $V_{g,pk}$, so the bridge
only conducts during the short interval each line cycle where
$|v_{ac}(t)|$ exceeds the cap voltage. The result is **narrow tall
current pulses** at twice the line frequency.

The line sees a current that's nothing like its voltage — full of
harmonics, mostly reactive. Power factor crashes.
"""))

    cells.append(code(r"""
# Simulate a cheap bridge + bulk-cap "rectifier" to motivate PFC
f_line = 50.0
V_ac_pk = np.sqrt(2) * 230.0
t = np.linspace(0, 3.0/f_line, 6000)
v_ac = V_ac_pk * np.sin(2*np.pi*f_line*t)
v_g = np.abs(v_ac)

# Model: cap holds V_cap; diode conducts only when v_g > V_cap.
C_bulk = 470e-6
R_load = 1600.0  # 100 W at ~400 V
V_cap = V_ac_pk * 0.95  # start near peak
i_line = np.zeros_like(t)
v_cap_hist = np.zeros_like(t)
for k, tk in enumerate(t):
    dt = t[1] - t[0]
    if v_g[k] > V_cap:
        # Diode conducts; cap charged from v_g through a tiny series R
        I_load = V_cap / R_load
        # Crude pulsed-conduction model: enough to draw enough charge
        i_pulse = (v_g[k] - V_cap) / 1.0  # 1Ω equivalent series R
        i_pulse = max(i_pulse, 0.0)
        V_cap = V_cap + (i_pulse - I_load) * dt / C_bulk
        i_line[k] = i_pulse * np.sign(v_ac[k])
    else:
        I_load = V_cap / R_load
        V_cap = V_cap - I_load * dt / C_bulk
        i_line[k] = 0.0
    v_cap_hist[k] = V_cap

fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax_top = axs[0]
ax_top.plot(t*1e3, v_ac, "C0", label="$v_{ac}(t)$")
ax_top.plot(t*1e3, v_cap_hist, "C2", label="$V_{cap}$ (bulk)")
ax_top.set_ylabel("Voltage [V]")
ax_top.legend(loc="lower right")
ax_top.set_title("Cheap diode-bridge + bulk-cap rectifier — the\\\"bad\\\" reference")

axs[1].plot(t*1e3, i_line, "C3", linewidth=1.0, label="$i_{line}(t)$")
axs[1].plot(t*1e3, V_ac_pk/np.max(np.abs(i_line))*i_line*0 + V_ac_pk/np.max(np.abs(v_ac))*v_ac, "C0:", alpha=0.4, label="$v_{ac}$ shape (scaled)")
axs[1].set_ylabel("Line current [A]")
axs[1].set_xlabel("Time [ms]")
axs[1].legend(loc="lower right")
plt.tight_layout(); plt.show()

# Measure PF on the steady-state portion
mask = t > 1.0/f_line
fs = 1.0/(t[1]-t[0])
pf_bad = power_factor(v_ac[mask], i_line[mask], f_s=fs, f_line=f_line)
i_filt = line_band_filter(i_line[mask], fs, f_line)
thd_bad = thd_current(i_filt, fs, f_line)
print(f"Bad rectifier — PF = {pf_bad:.3f}, THD = {thd_bad*100:.1f} %")
print(f"  (typical real designs measure PF ≈ 0.55-0.65, THD > 100 %)")
"""))

    cells.append(md(r"""
## 2. PF and THD — definitions

For periodic line-frequency signals:

$$
\text{PF} = \frac{\bar P}{V_{rms} I_{rms}}
= \underbrace{\cos(\varphi_1)}_{\text{disp. factor}}
\cdot
\underbrace{\frac{I_{1,rms}}{I_{rms}}}_{\text{distortion factor}}
$$

The displacement factor is the cosine of the fundamental phase
shift; the distortion factor is the ratio of fundamental RMS to
total RMS. For PFC stages we usually have $\varphi_1 \approx 0$
(current in phase with voltage) so:

$$
\text{PF} \approx \frac{1}{\sqrt{1 + \text{THD}^2}}
$$

$$
\text{THD} = \frac{\sqrt{\sum_{n \ge 2} I_n^2}}{I_1}
$$

So PF = 0.99 corresponds to THD ≈ 14%. PF = 0.95 ↔ THD ≈ 33%.

**Standard limits** (IEC 61000-3-2 Class D, applicable to PC and
consumer equipment > 75 W): 3rd harmonic ≤ 3.4 mA/W, 5th ≤ 1.9 mA/W,
etc. A boost PFC with PF > 0.95 passes these comfortably.
"""))

    cells.append(md(r"""
## 3. The boost PFC topology

Insert a boost converter between the diode bridge and the bulk cap.
The boost inductor forces a controlled current shape, decoupling
the **line-side** current from the **cap-side** charging behavior:

```
   v_ac ───┤                                            ┌─── V_o (400 V DC)
           │   bridge       ┌────┐    ┌── D ────┬───────┤
   v_ac ───┤              ──┤  L ├────┤         │       │
           │                └────┘    │        +C       R_load
           │                          S         │       │
           └──────────────────────────┴─────────┴───────┘
              |v_ac(t)| = v_g(t)        boost stage
```

The boost stage works exactly like a regular boost converter, **but
its input is no longer DC** — $v_g(t) = |v_{ac}(t)|$ is a 100-Hz
half-sine. The output bulk cap stores enough energy to ride out the
line zero-crossings (with a 2·$f_{line}$ ripple).

If we **control the inductor current to follow $|v_{ac}(t)|$**, the
line sees a resistive load — PF → 1, THD → 0.
"""))

    cells.append(md(r"""
## 4. Two strategies — DCM vs CCM

The boost stage can run in either conduction mode. The choice
fundamentally changes the controller:

| | DCM (small L) | CCM (large L) |
|---|---|---|
| $i_L$ per $T_{sw}$ | triangle → 0 | trapezoid above 0 |
| Avg $i_L$ shape | **automatic** $\propto v_g$ (if $D$ const) | needs an active current loop |
| Controller | **1 voltage loop** | **2 loops** (current + voltage) + multiplier |
| Plant for voltage loop | first-order (1/(1+sRC/2)) | first-order (after current loop closes) |
| Plant for current loop | — | integrator $V_o/(sL)$ |
| Peak $i_L$ | ~2× CCM | moderate |
| Has RHP zero? | **no** | yes (boost RHP zero) |

The headline: **DCM gives you PFC almost for free** — set $D$ ≈ const
over a line half-cycle, the natural physics shapes the current. Pay
in higher peak current and worse EMI.

**CCM gives lower stress and cleaner waveforms** at the cost of a
significantly more complex controller (and bringing back the boost's
RHP zero in the current loop dynamics).
"""))

    cells.append(code(r"""
# Show the two operating points side by side
p_dcm = BoostPFCParams.dcm_design()
p_ccm = BoostPFCParams.ccm_design()
print("DCM design:")
print(operating_point_report(p_dcm, mode="DCM"))
print()
print("="*70)
print("CCM design:")
print(operating_point_report(p_ccm, mode="CCM"))
"""))

    cells.append(md(r"""
## 5. The inductor current — shape preview

Here's what $i_L$ actually looks like over a line cycle in each
mode. This is purely an analytical sketch (not yet a switched sim);
just the **envelope** of $i_L$ — the per-$T_{sw}$ peak or average.

- **DCM**: $i_L$ is a triangle each $T_{sw}$. Peak $i_{pk}(t) =
  v_g(t) \cdot D \cdot T_{sw}/L$ — proportional to $v_g(t)$.
  Returns to 0 each switching period (the "dis-continuous" in DCM).

- **CCM**: $i_L$ ripples around a non-zero average. With active
  current shaping, that average is forced to track $v_g(t)$:
  $\langle i_L \rangle(t) \propto v_g(t)$. The ripple is at $f_{sw}$,
  the envelope is at $f_{line}$.
"""))

    cells.append(code(r"""
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

# DCM envelope at V_ac_nom = 230V
p = p_dcm
t_line = np.linspace(0, 1/p.f_line, 1000)
v_g = p.V_g_pk_nom * np.abs(np.sin(p.omega_line * t_line))
D = dcm_steady_state_duty_exact(p, V_ac=p.V_ac_nom)
# DCM peak: v_g · D · T_sw / L
i_pk = v_g * D * p.T_sw / p.L
# DCM average (cusp-distorted): v_g/Re · 1/(1-v_g/V_o)
Re = 2*p.L*p.f_sw/D**2
i_avg = v_g/Re / np.maximum(1.0 - v_g/p.V_o, 1e-3)
axs[0].plot(t_line*1000, i_pk, "C1", label="$i_{pk}(t)$ (DCM triangle peak)")
axs[0].plot(t_line*1000, i_avg, "C3", linewidth=2, label="$\\langle i_g \\rangle$ avg-per-$T_{sw}$")
axs[0].plot(t_line*1000, v_g / v_g.max() * i_pk.max() * 0.5, "C0:", alpha=0.5, label="$v_g$ shape")
axs[0].set_ylabel("Current [A]"); axs[0].set_xlabel("Time [ms]")
axs[0].set_title(f"DCM @ V_ac=230V, D={D:.3f}")
axs[0].legend(loc="lower right")

# CCM envelope: average i_L should be P_in/v_g instantaneously, but
# realistically it's P_in/V_ac_rms times a sin shape. Show the desired track.
p = p_ccm
v_g_c = p.V_g_pk_nom * np.abs(np.sin(p.omega_line * t_line))
# In CCM, controller forces <i_L>(t) = (P_in/V_ac_rms²) · v_g(t)
# scaling so that <v_g · i_L>_T = P_o
k_shape = p.P_o / (p.V_ac_nom ** 2)
i_avg_ccm = k_shape * v_g_c * 2  # ×2 because <sin²>=1/2
# Ripple amplitude (peak-to-peak)
delta_i_pp = v_g_c * (1.0 - v_g_c/p.V_o) * p.T_sw / p.L
i_top = i_avg_ccm + delta_i_pp/2
i_bot = np.maximum(i_avg_ccm - delta_i_pp/2, 0.0)
axs[1].fill_between(t_line*1000, i_bot, i_top, color="C1", alpha=0.3,
                    label="$i_L$ ripple envelope")
axs[1].plot(t_line*1000, i_avg_ccm, "C3", linewidth=2,
            label="$\\langle i_L \\rangle$ (current loop forces this)")
axs[1].plot(t_line*1000, v_g_c/v_g_c.max()*i_avg_ccm.max(), "C0:", alpha=0.5,
            label="$v_g$ shape")
axs[1].set_ylabel("Current [A]"); axs[1].set_xlabel("Time [ms]")
axs[1].set_title("CCM (with active current shaping)")
axs[1].legend(loc="lower right")

plt.tight_layout(); plt.show()
"""))

    cells.append(md(r"""
## 6. The voltage-loop bandwidth ceiling

Both modes have the **same** outer-loop constraint: the voltage
compensator must be MUCH slower than the line. Why?

The output voltage has a $2 f_{line}$ ripple (100 Hz at 50 Hz line),
because the boost stage delivers power in a pulsed pattern with
peaks near $|v_{ac}|$ maxima. If the voltage loop is fast enough to
"see" that ripple, it will respond by chopping the current reference
at 2·$f_{line}$ — which **distorts the input current shape** and
crashes PF.

**Rule of thumb**: $f_{c,v} \le 2 f_{line} / 10$.

At 50 Hz: $f_{c,v}$ ≤ 10 Hz. The voltage loop is **slow** —
settling times measured in 100s of ms, not in µs like the DC-DC
converters we've seen.

This makes PFC compensators feel different from DC-DC compensators:
- the plant is **easy** (first-order in both DCM and CCM)
- the bandwidth is **constrained** to be slow (not by stability,
  but by signal integrity)
- a simple **PI** is usually enough — Type-III is overkill.
"""))

    cells.append(md(r"""
## 7. Summary and what's next

| | DCM PFC | CCM PFC |
|---|---|---|
| $L$ | 100 µH (small) | 1 mH (large) |
| Voltage-loop plant | $K/(1+sRC/2)$ | $K/(1+sRC/2)$ |
| Current loop | **none** | $V_o/(sL)$ — integrator |
| Multiplier | no | yes |
| RHP zero | **no** | yes (in current-loop transient) |
| Best for | $\le$ 150 W | $\ge$ 200 W |

**Next notebooks**:

- `02_boost_pfc_dcm.ipynb` — analyze DCM in detail. Derive the
  equivalent resistance $R_e$, the cusp-distortion factor
  $F(m_{pk})$, the first-order voltage-loop plant. Design a PI.
  Run the switched closed-loop sim and verify PF/THD.

- `03_boost_pfc_ccm.ipynb` — analyze CCM. Two loops: inner current
  loop (PI on an integrator plant) + outer voltage loop (PI on the
  same first-order plant as DCM). Multiplier between them.
  Switched closed-loop sim.

- `04_boost_pfc_simulation.ipynb` — DCM vs CCM side by side.
  Line-step (90 → 230 V), load-step (full → half load), PF/THD
  across the line range.
"""))

    return cells


# ---------------------------------------------------------------------------
# Notebook 2 — DCM
# ---------------------------------------------------------------------------


def build_dcm_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 2 — Boost PFC in DCM (Discontinuous Conduction Mode)

> **Goal.** Derive the DCM analytical model, show that the input
> current is *naturally* proportional to $v_g(t)$ if $D$ is held
> approximately constant over a line half-cycle, design a single
> voltage-loop PI compensator, and verify with a switched closed-loop
> simulation that we get PF > 0.99 with regulated $V_o$.

**Prerequisites**

- Boost modeling notebook (`projects/converters/boost/`)
- PFC basics notebook (`01_boost_pfc_basics.ipynb`)

**What you'll be able to do at the end**

1. Sketch the 3-phase $i_L$ waveform in DCM (charge → discharge → zero)
   and derive the average input current $\langle i_g \rangle_{T_{sw}}$.
2. Compute the equivalent resistance $R_e = 2 L f_{sw}/D^2$ and use it
   to derive the steady-state duty $D(V_{ac})$.
3. Apply the cusp-distortion correction $F(m_{pk})$ to get the *exact*
   duty for a target $V_o$.
4. Derive the line-cycle averaged voltage-loop plant and design a
   PI compensator with $f_c \le 2 f_{line}/10$.
5. Discretize and run the switched closed-loop sim, then measure PF
   and THD and check they meet the IEC 61000-3-2 target.
"""))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from boost_pfc_model import (
    BoostPFCParams,
    dcm_critical_inductance, dcm_equivalent_resistance,
    dcm_steady_state_duty, dcm_steady_state_duty_exact,
    dcm_power_correction_factor,
    dcm_input_current_instantaneous,
    dcm_voltage_loop_plant,
    simulate_open_loop_dcm, simulate_closed_loop_dcm,
    operating_point_report,
    power_factor, thd_current, line_band_filter,
)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

p = BoostPFCParams.dcm_design()
print(operating_point_report(p, mode="DCM"))
"""))

    cells.append(md(r"""
## 1. The 3-phase DCM cycle

Within one $T_{sw}$, three phases:

```
phase 1 (S on):     L · di_L/dt = v_g                  i_L rises from 0 to i_pk
phase 2 (S off, D on): L · di_L/dt = v_g - V_o         i_L falls from i_pk to 0
phase 3 (S off, D off): L · di_L/dt = 0                i_L = 0  (DCM "dead" phase)
```

With on-duty $D$ and off-duty $D_2 = D \cdot v_g/(V_o - v_g)$ (from
volt-second balance on the inductor):

$$
i_{pk}(t) = v_g(t) \cdot D \cdot T_{sw} / L
$$

The **average** $i_g$ over one $T_{sw}$ is half the triangle area
over the period:

$$
\langle i_g \rangle_{T_{sw}}
= \frac{1}{2} \cdot i_{pk} \cdot (D + D_2)
= \frac{v_g}{R_e} \cdot \frac{1}{1 - v_g/V_o}
$$

where

$$
\boxed{R_e = \frac{2 L f_{sw}}{D^2}}
$$

is the **equivalent input resistance** of the DCM boost — Erickson's
key insight. With $D$ constant over a line cycle, the input looks
like a resistor $R_e$ in parallel with a small cusp-distortion term.
"""))

    cells.append(md(r"""
## 2. The cusp distortion

The factor $1/(1 - v_g/V_o)$ in the average input current grows as
$v_g$ approaches $V_o$. At low line ($V_{g,pk} \ll V_o$), the
distortion is small and $\langle i_g \rangle \approx v_g/R_e$ is
nearly resistive — natural PFC works beautifully.

At high line ($V_{g,pk}$ close to $V_o$), the distortion grows
rapidly and PF degrades. This is the **fundamental limitation** of
fixed-$D$ DCM PFC for universal-input designs.
"""))

    cells.append(code(r"""
# Plot the input current shape over a line cycle at several V_ac values
fig, ax = plt.subplots(figsize=(10, 4.5))
t_line = np.linspace(0, 1/p.f_line, 2000)
for V_ac, color in [(90, "C0"), (120, "C1"), (180, "C2"), (230, "C3"), (265, "C4")]:
    D = dcm_steady_state_duty_exact(p, V_ac=V_ac)
    i_g = dcm_input_current_instantaneous(p, t_line, D, V_ac=V_ac)
    ax.plot(t_line*1000, i_g, color=color, label=f"V_ac = {V_ac} V (D={D:.3f})")

ax.set_xlabel("Time [ms]"); ax.set_ylabel("$\\langle i_g \\rangle_{T_{sw}}$ [A]")
ax.set_title("DCM PFC input current envelope — cusp distortion grows with V_ac")
ax.legend()
plt.tight_layout(); plt.show()
"""))

    cells.append(code(r"""
# The cusp-distortion correction factor F(m_pk)
m_range = np.linspace(0.0, 0.98, 200)
F_vals = np.array([dcm_power_correction_factor(m) for m in m_range])
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(m_range, F_vals, "C0", linewidth=2)
ax.axhline(1.0, color="k", linestyle=":", alpha=0.4, label="resistive limit (F=1)")
for V_ac, color in [(90, "C1"), (120, "C2"), (180, "C3"), (230, "C4"), (265, "C5")]:
    m = np.sqrt(2)*V_ac/p.V_o
    F = dcm_power_correction_factor(m)
    ax.plot(m, F, "o", color=color, label=f"V_ac={V_ac}V (m={m:.2f}, F={F:.2f})")
ax.set_xlabel("$m_{pk} = V_{g,pk}/V_o$")
ax.set_ylabel("Correction factor $F(m_{pk})$")
ax.set_title("DCM PFC cusp-distortion correction factor (Erickson)")
ax.legend(fontsize=8, loc="upper left"); plt.tight_layout(); plt.show()
"""))

    cells.append(md(r"""
## 3. Steady-state duty — linearized vs exact

The Erickson-textbook "linearized" duty assumes $m_{pk} \to 0$ and
the converter is a pure resistor at $R_e$:

$$
D_{lin} = \frac{\sqrt{2 L f_{sw} P_o}}{V_{ac,rms}}
$$

The **exact** duty, including cusp distortion, solves

$$
P_o = \frac{V_{ac,rms}^2 \, D^2}{2 L f_{sw}} \cdot F(m_{pk})
$$

giving

$$
\boxed{D_{exact} = \sqrt{\frac{2 L f_{sw} P_o}{F(m_{pk}) \, V_{ac,rms}^2}}}
$$

For universal mains, the two differ by 5-40%. Use $D_{exact}$ when
setting up the open-loop verification; in closed-loop the controller
finds the right duty automatically.
"""))

    cells.append(code(r"""
print(f"{'V_ac':>6} {'D_lin':>8} {'D_exact':>8} {'F(m_pk)':>10}")
for V_ac in [90, 120, 180, 230, 265]:
    D_lin = dcm_steady_state_duty(p, V_ac=V_ac)
    D_ex = dcm_steady_state_duty_exact(p, V_ac=V_ac)
    m = np.sqrt(2)*V_ac/p.V_o
    F = dcm_power_correction_factor(m)
    print(f"{V_ac:6.0f} {D_lin:8.4f} {D_ex:8.4f} {F:10.3f}")
"""))

    cells.append(md(r"""
## 4. Open-loop natural PFC — proof of concept

Run the switched sim with a **fixed** $D$ (no controller) and watch
the input current naturally take the shape of $v_{ac}(t)$. PF should
be > 0.99 at low line.
"""))

    cells.append(code(r"""
sim_ol = simulate_open_loop_dcm(p, V_ac=120.0, n_line_cycles=8, samples_per_period=80)

fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
axs[0].plot(sim_ol['t']*1000, sim_ol['v_o'], "C2", linewidth=1.0)
axs[0].axhline(p.V_o, color="k", linestyle=":", alpha=0.4, label=f"target {p.V_o} V")
axs[0].set_ylabel("$v_o$ [V]")
axs[0].set_title("Open-loop DCM PFC at V_ac = 120 V, fixed D (exact)")
axs[0].legend()

axs[1].plot(sim_ol['t']*1000, sim_ol['v_ac'], "C0", linewidth=1.0, label="$v_{ac}$")
ax2 = axs[1].twinx()
ax2.plot(sim_ol['t']*1000, sim_ol['i_in'], "C3", linewidth=0.6, alpha=0.7, label="$i_{in}$ (raw)")
axs[1].set_ylabel("$v_{ac}$ [V]"); ax2.set_ylabel("$i_{in}$ [A]")
axs[1].set_title("Line voltage and input current (raw, with switching ripple)")
axs[1].legend(loc="upper left"); ax2.legend(loc="upper right")

# Filter the input current to the line band (anti-alias PQ measurement)
fs = 1.0/(sim_ol['t'][1] - sim_ol['t'][0])
i_filt = line_band_filter(sim_ol['i_in'], fs, p.f_line, n_harmonics=20)
axs[2].plot(sim_ol['t']*1000, sim_ol['v_ac']/sim_ol['v_ac'].max(), "C0:", alpha=0.5, label="$v_{ac}$ shape")
axs[2].plot(sim_ol['t']*1000, i_filt/np.max(np.abs(i_filt)), "C3", linewidth=1.5, label="$i_{in}$ filtered (normalized)")
axs[2].set_xlabel("Time [ms]"); axs[2].set_ylabel("Normalized")
axs[2].set_title("Filtered input current vs line voltage — natural PFC")
axs[2].legend()
plt.tight_layout(); plt.show()

# Metrics on the last 4 cycles (after transient)
mask = sim_ol['t'] >= 4.0/p.f_line
pf = power_factor(sim_ol['v_ac'][mask], sim_ol['i_in'][mask], f_s=fs, f_line=p.f_line)
i_filt_m = line_band_filter(sim_ol['i_in'][mask], fs, p.f_line, 20)
thd = thd_current(i_filt_m, fs, p.f_line)
print(f"Open-loop DCM @ V_ac=120V (steady-state): PF = {pf:.4f}, THD = {thd*100:.2f}%")
"""))

    cells.append(md(r"""
## 5. Line-cycle averaged voltage-loop plant

Now that we believe in DCM PFC, design the voltage compensator.
Average the converter's power flow over a *line cycle* (slow
dynamics, ignore the switching) — the bulk cap balance becomes:

$$
C V_o \frac{d V_o}{dt} = P_{in}(D, V_{ac}) - \frac{V_o^2}{R_{load}}
$$

With $P_{in} = V_{ac}^2 D^2 / (2 L f_{sw})$ (resistive-emulator
approximation), linearize around $(V_o, D)$:

$$
C V_o \frac{d \hat v_o}{dt} = \frac{V_{ac}^2 D}{L f_{sw}} \hat d
                              - \frac{2 V_o}{R_{load}} \hat v_o
$$

Rearrange:

$$
\boxed{
G_{vd}(s) = \frac{\hat v_o}{\hat d}
= \frac{K_{dc}}{1 + s \tau}, \quad
K_{dc} = \frac{V_{ac}^2 D}{L f_{sw}} \cdot \frac{R_{load}}{2 V_o}, \quad
\tau = \frac{R_{load} C}{2}
}
$$

A clean **first-order** plant! Compare to the boost converter's
second-order-plus-RHP-zero misery — DCM PFC is dramatically easier.
"""))

    cells.append(code(r"""
plant = dcm_voltage_loop_plant(p)
print(f"DCM voltage-loop plant:")
print(f"  num = {plant.num}")
print(f"  den = {plant.den}")
pole_freq = (plant.den[1]/plant.den[0]) / (2*np.pi)
print(f"  pole at f = {pole_freq:.3f} Hz")
print(f"  DC gain   = {plant.num[0]/plant.den[1]:.2f} V/duty")

# Bode plot
f = np.logspace(-1, 3, 500)
w = 2*np.pi*f
_, mag, ph = signal.bode(plant, w=w)
fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax_mag.semilogx(f, mag, "C0", linewidth=2)
ax_mag.axhline(0, color="k", linestyle=":", alpha=0.3)
ax_mag.axvline(2*p.f_line, color="C3", linestyle=":", alpha=0.5, label=f"2·f_line = {2*p.f_line} Hz")
ax_mag.axvline(2*p.f_line/10, color="C2", linestyle=":", alpha=0.5, label=f"BW ceiling ≈ {2*p.f_line/10} Hz")
ax_ph.semilogx(f, ph, "C0", linewidth=2)
ax_ph.axhline(-90, color="k", linestyle=":", alpha=0.3)
ax_mag.set_ylabel("Magnitude [dB]"); ax_ph.set_ylabel("Phase [deg]")
ax_ph.set_xlabel("Frequency [Hz]")
ax_mag.set_title("DCM voltage-loop plant — first-order low-pass")
ax_mag.legend(); plt.tight_layout(); plt.show()
"""))

    cells.append(md(r"""
## 6. PI compensator

For a first-order plant, a simple **PI** with the zero placed at the
plant pole and the gain set for the desired crossover is the
textbook recipe. No Type-III needed.

$$
G_c(s) = \frac{K}{s} (1 + s \tau_z)
$$

Place $\tau_z = R_{load} C / 2$ → the zero exactly cancels the plant
pole, giving an open-loop integrator. Then $K$ sets the crossover.

Target: $f_c = 5$ Hz, PM ≈ 90° (achievable because the plant is
single-pole and we've cancelled it).
"""))

    cells.append(code(r"""
V_ramp = 5.0
plant_pwm = signal.TransferFunction([x/V_ramp for x in plant.num], plant.den)

f_c = 10.0
omega_c = 2*np.pi*f_c
tau_z = plant.den[0]/plant.den[1]
# Compensator: K/s · (1 + s·tau_z). At omega_c (after canceling pole):
# open-loop = K/s · plant_dc_gain → magnitude at omega_c is K · plant_dc_gain / omega_c
plant_dc_gain = plant_pwm.num[0]/plant_pwm.den[1]
K = omega_c / plant_dc_gain
print(f"PI: K = {K:.4g}, tau_z = {tau_z:.4f} s ({1/(2*np.pi*tau_z):.2f} Hz zero)")

Gc = signal.TransferFunction([K*tau_z, K], [1.0, 0.0])
T_open = signal.TransferFunction(np.polymul(Gc.num, plant_pwm.num),
                                   np.polymul(Gc.den, plant_pwm.den))
_, mag_T, ph_T = signal.bode(T_open, w=2*np.pi*f)
idx_cross = np.argmin(np.abs(mag_T))
f_cross = f[idx_cross]
pm = 180 + ph_T[idx_cross]

fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax_mag.semilogx(f, mag_T, "C3", linewidth=2)
ax_mag.axhline(0, color="k", linestyle=":")
ax_mag.axvline(f_cross, color="g", linestyle=":", alpha=0.5, label=f"$f_c$ = {f_cross:.2f} Hz")
ax_mag.axvline(2*p.f_line, color="C3", linestyle=":", alpha=0.5, label=f"2·f_line = {2*p.f_line} Hz")
ax_ph.semilogx(f, ph_T, "C3", linewidth=2)
ax_ph.axhline(-180, color="k", linestyle=":")
ax_ph.axhline(-90, color="C2", linestyle=":", alpha=0.4, label="-90° (integrator)")
ax_mag.set_ylabel("Magnitude [dB]"); ax_ph.set_ylabel("Phase [deg]")
ax_ph.set_xlabel("Frequency [Hz]")
ax_mag.set_title(f"DCM voltage loop: f_c = {f_cross:.2f} Hz, PM = {pm:.1f}°")
ax_mag.legend(); ax_ph.legend(); plt.tight_layout(); plt.show()
print(f"f_c achieved = {f_cross:.2f} Hz, PM = {pm:.1f}°")
"""))

    cells.append(md(r"""
## 7. Discretization

Tustin (bilinear) at $T_s = 1/f_{sw}$:
"""))

    cells.append(code(r"""
T_s = p.T_sw
bv_raw, av_raw, _ = signal.cont2discrete((Gc.num, Gc.den), dt=T_s, method='bilinear')
bv = np.asarray(bv_raw).flatten() / av_raw[0]
av = np.asarray(av_raw) / av_raw[0]
print(f"Sample period T_s = {T_s*1e6:.3f} µs")
print(f"  b = {bv}")
print(f"  a = {av}")
"""))

    cells.append(md(r"""
## 8. Switched closed-loop simulation

Run with $V_{ac}$ = 120 V (nominal-ish in the universal range)
across 10 line cycles. Expect $V_o$ to regulate to 400 V with the
characteristic 100 Hz ripple. PF should be > 0.99, THD modest.
"""))

    cells.append(code(r"""
sim_cl = simulate_closed_loop_dcm(p, bv, av, V_ac=120.0, n_line_cycles=10,
                                    samples_per_period=80, v_ref=400.0,
                                    V_ramp=V_ramp, warm_start=True)

fig, axs = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
axs[0].plot(sim_cl['t']*1000, sim_cl['v_o'], "C2", linewidth=1.0)
axs[0].axhline(400.0, color="k", linestyle=":", alpha=0.4, label="$V_{ref}$ = 400 V")
axs[0].set_ylabel("$v_o$ [V]")
axs[0].set_title(f"Closed-loop DCM PFC @ V_ac = 120 V (V_g_pk = {np.sqrt(2)*120:.0f} V)")
axs[0].legend()

axs[1].plot(sim_cl['t']*1000, sim_cl['v_ac'], "C0", linewidth=1.0)
axs[1].set_ylabel("$v_{ac}$ [V]")

# Filtered input current
fs = 1.0/(sim_cl['t'][1] - sim_cl['t'][0])
i_filt = line_band_filter(sim_cl['i_in'], fs, p.f_line, 20)
axs[2].plot(sim_cl['t']*1000, sim_cl['i_in'], "C3", linewidth=0.4, alpha=0.4, label="raw $i_{in}$")
axs[2].plot(sim_cl['t']*1000, i_filt, "C1", linewidth=1.5, label="$i_{in}$ filtered")
axs[2].set_ylabel("$i_{in}$ [A]"); axs[2].legend(loc="lower right")

axs[3].plot(sim_cl['t']*1000, sim_cl['duty'], "C4", linewidth=0.6)
axs[3].set_ylabel("Duty"); axs[3].set_xlabel("Time [ms]")
plt.tight_layout(); plt.show()

# Steady-state metrics (drop first 5 cycles for transient)
mask = sim_cl['t'] >= 5.0/p.f_line
pf = power_factor(sim_cl['v_ac'][mask], sim_cl['i_in'][mask], f_s=fs, f_line=p.f_line)
i_filt_m = line_band_filter(sim_cl['i_in'][mask], fs, p.f_line, 20)
thd = thd_current(i_filt_m, fs, p.f_line)
v_o_mean = sim_cl['v_o'][mask].mean()
v_o_pp = sim_cl['v_o'][mask].max() - sim_cl['v_o'][mask].min()
print()
print("Closed-loop DCM PFC steady-state metrics:")
print(f"  V_o mean        = {v_o_mean:.2f} V  (target 400.0 V, error {(v_o_mean-400)/400*100:.2f}%)")
print(f"  V_o ripple pp   = {v_o_pp:.2f} V")
print(f"  PF              = {pf:.4f}")
print(f"  THD             = {thd*100:.2f}%")
print(f"  Duty range      = {sim_cl['duty'][mask].min():.4f} .. {sim_cl['duty'][mask].max():.4f}")

if pf > 0.95 and abs(v_o_mean - 400) < 5 and thd < 0.20:
    print()
    print("✅  Closed-loop DCM PFC PROVEN: PF > 0.95, V_o regulated, THD < 20%")
else:
    print()
    print("⚠️   Performance off-target — revisit compensator design.")
"""))

    cells.append(md(r"""
## 9. Summary

DCM boost PFC delivers:

- **Natural input-current shaping** by physics — the inductor current
  envelope naturally tracks $v_g$ if $D$ is held roughly constant.
- **First-order voltage-loop plant** $G_{vd}(s) = K/(1 + sRC/2)$ — a
  simple PI with $\tau_z = RC/2$ cancels the pole, leaving a pure
  integrator at the desired crossover.
- **No RHP zero, no current loop, no multiplier** — the simplest PFC
  architecture.
- **PF > 0.99 across low/mid line**, degrading to ~0.97 at high line
  due to cusp distortion.

The price: large peak inductor current (~5-6 A at 100 W) and the
sharp triangle waveform creates EMI that the input filter has to
absorb.

**Next**: `03_boost_pfc_ccm.ipynb` does the same exercise with a
CCM design — larger $L$, lower peak current, but a **two-loop**
controller with multiplier between the loops.

**Suggested exercises**

1. Vary the load (change `p.P_o` to 50 W). Re-simulate; does the
   loop hold $V_o$? Compute PF / THD at half load.
2. Step the line voltage during the simulation. The slow voltage
   loop will take ~200 ms to re-regulate. Plot the transient.
3. Reduce $C$ to 47 µF. The output ripple should grow to ~30 V
   pk-pk (predicted by $I_o / (\pi f_{line} C)$).
4. Increase the compensator crossover to 30 Hz. Watch the input
   current shape get distorted at 100 Hz — the loop is now
   "correcting" the 2·f_line ripple instead of leaving it alone.
"""))

    return cells


# ---------------------------------------------------------------------------
# Notebook 3 — CCM
# ---------------------------------------------------------------------------


def build_ccm_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 3 — Boost PFC in CCM (Continuous Conduction Mode)

> **Goal.** Larger $L$ keeps the inductor in continuous conduction
> throughout the line cycle, but now PFC isn't free — we need an
> **active current loop** to force $i_L$ to track a sinusoidal
> reference. The full architecture is **two loops**: fast inner
> current loop + slow outer voltage loop, joined by a **multiplier**
> that converts the voltage loop's amplitude command into a shape
> command via $|v_{ac}(t)|$.

**Prerequisites**

- Boost modeling notebook
- Notebooks 01 and 02 of this project (especially DCM)

**What you'll be able to do at the end**

1. Identify CCM operation on a switching waveform (inductor doesn't
   return to 0 each $T_{sw}$).
2. Linearize the CCM boost current dynamics into the integrator plant
   $G_{id}(s) = V_o/(sL)$ within the current-loop bandwidth.
3. Design a fast inner PI for the current loop ($f_{c,i} \approx
   f_{sw}/10$).
4. Wire the multiplier $i_{ref}(t) = K \cdot i_{ref,amp} \cdot |v_{ac}(t)|$.
5. Design a slow outer PI for the voltage loop (same plant shape as
   DCM, $f_{c,v} \le 2 f_{line}/10$).
6. Run the two-loop switched simulation and verify PF, THD, and
   regulation.
"""))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from boost_pfc_model import (
    BoostPFCParams,
    ccm_current_loop_plant, ccm_voltage_loop_plant,
    simulate_closed_loop_ccm,
    operating_point_report,
    power_factor, thd_current, line_band_filter,
)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

p = BoostPFCParams.ccm_design()
print(operating_point_report(p, mode="CCM"))
"""))

    cells.append(md(r"""
## 1. Why two loops?

In CCM, the inductor current doesn't naturally shape itself. With a
fixed $D$, the average $i_L$ over $T_{sw}$ is:

$$
\langle i_L \rangle = \frac{P_{in}}{V_g}
\;\;\text{at instantaneous operating point}
$$

That's $\propto 1/v_g$, **not** $\propto v_g$. So the current would
be highest near the line zero-crossings and lowest at the peak —
exactly backwards from what PFC needs.

The fix: an **inner current loop** that drives $i_L$ to track a
reference $i_{ref}(t)$, which is *constructed* to be $\propto v_g(t)$:

```
                   ┌────────── multiplier ──────────┐
                   │                                │
  v_ref ─→ (-) → [voltage PI] ──→ i_ref_amp ── × ──→ i_ref(t) → (-) → [current PI] → duty
              │                                 ↑                  │
              │                                 │                  │
              │                              |v_ac(t)|             │
              │                                                    │
              └──────── v_o feedback ───── boost stage ←── duty ───┘
              ←──── i_L feedback ───────────┘
```

- **Outer voltage loop** (slow, ~5-10 Hz): regulates $V_o$, outputs
  the *amplitude* of the desired current.
- **Multiplier**: scales the amplitude by $|v_{ac}(t)|$ → gives the
  shape.
- **Inner current loop** (fast, ~5-10 kHz): forces $i_L$ to track
  the reference instantaneously.
"""))

    cells.append(md(r"""
## 2. Current-loop plant

Within the current-loop bandwidth (kHz range), we can treat $V_o$ as
constant (the bulk cap is "stiff" at this timescale). The inductor
dynamics:

$$
L \frac{d i_L}{dt} = v_g - (1 - d) V_o
$$

Linearize around the instantaneous operating point:

$$
L \frac{d \hat i_L}{dt} = V_o \, \hat d
$$

(the $v_g$ contribution and the $(1-D)$ DC term vanish in the
perturbation equation when $V_o$ is treated as constant).

$$
\boxed{
G_{id}(s) = \frac{\hat i_L}{\hat d} = \frac{V_o}{s L}
}
$$

A pure **integrator**. The current loop is trivially stable — a
single PI with a gain matched to the unity-gain frequency.

Note: this hides the boost's RHP zero, which only emerges when the
bulk cap is allowed to move (i.e. at the slow timescale of the
voltage loop). The current loop running at $\gg 2 f_{line}$ doesn't
see it.
"""))

    cells.append(code(r"""
G_id = ccm_current_loop_plant(p)
print(f"CCM current-loop plant Gid(s) = V_o/(sL):")
print(f"  num = {G_id.num} (= V_o = {p.V_o} V)")
print(f"  den = {G_id.den} (= L = {p.L*1e3:.3f} mH integrator)")
unity_freq = p.V_o / (2*np.pi*p.L)
print(f"  unity-gain frequency = V_o / (2π L) = {unity_freq:.0f} Hz")

# Bode
f = np.logspace(1, 5, 500)
w = 2*np.pi*f
_, mag, ph = signal.bode(G_id, w=w)
fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax_mag.semilogx(f, mag, "C0", linewidth=2)
ax_mag.axhline(0, color="k", linestyle=":")
ax_mag.axvline(p.f_sw/10, color="C2", linestyle=":", alpha=0.5,
               label=f"$f_{{sw}}/10$ = {p.f_sw/10/1000:.0f} kHz target")
ax_ph.semilogx(f, ph, "C0", linewidth=2)
ax_ph.axhline(-90, color="C3", linestyle=":", alpha=0.4)
ax_mag.set_ylabel("Magnitude [dB]"); ax_ph.set_ylabel("Phase [deg]")
ax_ph.set_xlabel("Frequency [Hz]")
ax_mag.set_title("Current-loop plant: pure integrator (-20 dB/dec, -90° forever)")
ax_mag.legend(); plt.tight_layout(); plt.show()
"""))

    cells.append(md(r"""
## 3. Inner current-loop PI

For an integrator plant, a PI with zero placed below the crossover
gives 90° + (less than 90° approach) phase margin. Target $f_{c,i}
= 10$ kHz, PM ≈ 60°.

$$
G_{c,i}(s) = K_i \cdot \frac{1 + s \tau_{z,i}}{s}
$$

Place $\tau_{z,i}$ such that the zero is at $f_{c,i}/\tan(90° - PM)$.
With PM = 60°, $\tan(30°) = 0.577$ → zero at $f_{c,i}/0.577$ ≈ 17 kHz?
No wait, we want phase boost AT the crossover; the zero should be
*below* it. Let's place zero at $f_{c,i}/3$ ≈ 3.3 kHz (3× below
crossover) for ~70° phase contribution from the integrator-plus-zero
combo.
"""))

    cells.append(code(r"""
V_ramp = 5.0
plant_i_pwm = signal.TransferFunction(np.array(G_id.num)/V_ramp, np.array(G_id.den))

f_c_i = 10e3
omega_c_i = 2*np.pi*f_c_i
tau_z_i = 1.0 / (omega_c_i / 3.0)  # zero at f_c/3
# At f_c, plant magnitude (after pwm): |V_o/(jw_c · L)|/V_ramp
plant_mag_at_fc = p.V_o / (omega_c_i * p.L) / V_ramp
# Compensator: K (1+jw·tau)/(jw). At omega_c_i: |K · sqrt(1+(w·tau)²) / w|
# Want loop gain = 1: K · sqrt(1+(omega_c·tau_z)²)/omega_c · plant_mag_at_fc = 1
K_i = omega_c_i / (np.sqrt(1+(omega_c_i*tau_z_i)**2) * plant_mag_at_fc)
print(f"Inner current PI: K_i = {K_i:.4g}, tau_z = {tau_z_i*1e6:.2f} µs ({1/(2*np.pi*tau_z_i)/1000:.2f} kHz zero)")

Gc_i = signal.TransferFunction([K_i*tau_z_i, K_i], [1.0, 0.0])
T_i = signal.TransferFunction(np.polymul(Gc_i.num, plant_i_pwm.num),
                                np.polymul(Gc_i.den, plant_i_pwm.den))
_, mag_T, ph_T = signal.bode(T_i, w=2*np.pi*f)
idx_cross = np.argmin(np.abs(mag_T))
f_cross_i = f[idx_cross]
pm_i = 180 + ph_T[idx_cross]
print(f"  Achieved: f_c = {f_cross_i/1000:.2f} kHz, PM = {pm_i:.1f}°")

fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax_mag.semilogx(f, mag_T, "C3", linewidth=2)
ax_mag.axhline(0, color="k", linestyle=":")
ax_mag.axvline(f_cross_i, color="g", linestyle=":", alpha=0.5,
               label=f"f_c = {f_cross_i/1000:.1f} kHz")
ax_ph.semilogx(f, ph_T, "C3", linewidth=2)
ax_ph.axhline(-180, color="k", linestyle=":")
ax_mag.set_ylabel("Magnitude [dB]"); ax_ph.set_ylabel("Phase [deg]")
ax_ph.set_xlabel("Frequency [Hz]")
ax_mag.set_title(f"Inner current loop: f_c = {f_cross_i/1000:.2f} kHz, PM = {pm_i:.1f}°")
ax_mag.legend(); plt.tight_layout(); plt.show()
"""))

    cells.append(md(r"""
## 4. Outer voltage-loop plant

With the inner current loop closed and fast, the voltage loop sees
the same **line-cycle averaged** plant as DCM:

$$
G_{vc}(s) = \frac{\hat v_o}{\hat i_{ref,amp}}
= \frac{V_{ac,rms}/\sqrt{2}}{2 V_o/R_{load}} \cdot
  \frac{1}{1 + s R_{load} C / 2}
$$

First-order, same shape as DCM. Use the same PI recipe ($\tau_z =
R_{load} C / 2$, $f_{c,v} = 5$ Hz).
"""))

    cells.append(code(r"""
G_vc = ccm_voltage_loop_plant(p)
print(f"Outer voltage-loop plant Gvc(s):")
print(f"  num = {G_vc.num}")
print(f"  den = {G_vc.den}")
pole_freq = (G_vc.den[1]/G_vc.den[0]) / (2*np.pi)
dc_gain = G_vc.num[0]/G_vc.den[1]
print(f"  pole at f = {pole_freq:.3f} Hz")
print(f"  DC gain   = {dc_gain:.4f} V per A (of i_ref_amp)")

# Outer PI
f_c_v = 10.0
omega_c_v = 2*np.pi*f_c_v
tau_z_v = G_vc.den[0]/G_vc.den[1]
K_v = omega_c_v / dc_gain
print(f"Outer voltage PI: K_v = {K_v:.4g}, tau_z = {tau_z_v:.4f} s ({1/(2*np.pi*tau_z_v):.2f} Hz zero)")
Gc_v = signal.TransferFunction([K_v*tau_z_v, K_v], [1.0, 0.0])
"""))

    cells.append(md(r"""
## 5. The multiplier

The voltage loop outputs `i_ref_amp` (a slow DC-ish signal representing
the amplitude of the desired current). To turn it into a shape
command, multiply by $|v_{ac}(t)|$ and scale by a constant $K_{mult}$:

$$
i_{ref}(t) = K_{mult} \cdot i_{ref,amp} \cdot |v_{ac}(t)|
$$

$K_{mult}$ is a hardware constant — chosen so that the steady-state
$i_{ref,amp}$ sits in a reasonable numerical range. We pick
$K_{mult} = 0.01$ A/V here so that at full load $i_{ref,amp}$ ≈ a
few amps — easy to see on plots and easy to clamp/protect.

**Important wrinkle**: when the line voltage is at a zero-crossing,
$|v_{ac}|$ → 0 → $i_{ref}$ → 0. The current loop then commands
zero duty → boost stage shuts off near cusps. This is normal and
desired.
"""))

    cells.append(code(r"""
# Discretize both compensators
T_s = p.T_sw
bi_raw, ai_raw, _ = signal.cont2discrete((Gc_i.num, Gc_i.den), dt=T_s, method='bilinear')
bv_raw, av_raw, _ = signal.cont2discrete((Gc_v.num, Gc_v.den), dt=T_s, method='bilinear')
bi = np.asarray(bi_raw).flatten() / ai_raw[0]; ai = np.asarray(ai_raw)/ai_raw[0]
bv = np.asarray(bv_raw).flatten() / av_raw[0]; av = np.asarray(av_raw)/av_raw[0]
print(f"Inner current compensator (b, a): {bi}, {ai}")
print(f"Outer voltage compensator (b, a): {bv}, {av}")

# K_mult chosen so that <P_in> = P_o at V_ac_nom:
# <v_g · i_L> = <v_g · K_mult · i_ref_amp · v_g> = K_mult · i_ref_amp · <v_g²>
# = K_mult · i_ref_amp · V_g_pk²/2 = K_mult · i_ref_amp · V_ac²
# Want P_in = P_o → i_ref_amp = P_o / (K_mult · V_ac²)
# For pleasant range, pick K_mult such that i_ref_amp ≈ 1-5 A:
K_mult = 1.0 / (p.V_ac_nom ** 2) * 100  # i_ref_amp ≈ 1.89A at full load nominal
# Equivalent: K_mult · V_ac² = 100/V_ac² · V_ac² = 100... wait that's not dimensionally clean
# Let me redo: we want i_ref_amp · K_mult · <v_g²> = P_o → i_ref_amp = P_o/(K_mult · <v_g²>)
# At V_ac=230: <v_g²> = V_ac² = 52900. i_ref_amp = 100/(K_mult · 52900)
# Pick K_mult = 0.001 → i_ref_amp = 1.89 A
K_mult = 0.001
i_ref_amp_predicted = p.P_o / (K_mult * p.V_ac_nom**2)
print(f"K_mult = {K_mult} A/V, predicted i_ref_amp at full load = {i_ref_amp_predicted:.3f} A")
"""))

    cells.append(md(r"""
## 6. Switched closed-loop simulation — two loops + multiplier

Run at $V_{ac}$ = 120 V across 12 line cycles. Both loops should
settle. Expect:

- $V_o$ regulated to 400 V (slow voltage loop)
- $i_L$ tracking $i_{ref}(t) = K_{mult} \cdot i_{ref,amp} \cdot |v_{ac}|$
- PF > 0.99 (the current loop *enforces* the shape, not natural)
- THD low (much better than DCM because no cusp distortion)
"""))

    cells.append(code(r"""
sim_cl = simulate_closed_loop_ccm(p, bi, ai, bv, av, V_ac=120.0,
                                    n_line_cycles=40, samples_per_period=40,
                                    v_ref=400.0, V_ramp=V_ramp, warm_start=True,
                                    K_mult=K_mult)
print(f"Simulated {len(sim_cl['t'])} samples over {sim_cl['t'][-1]*1000:.2f} ms")

fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
axs[0].plot(sim_cl['t']*1000, sim_cl['v_o'], "C2", linewidth=1.0)
axs[0].axhline(400.0, color="k", linestyle=":", alpha=0.4, label="$V_{ref}$ = 400 V")
axs[0].set_ylabel("$v_o$ [V]")
axs[0].set_title(f"Closed-loop CCM PFC @ V_ac = 120 V (two-loop + multiplier)")
axs[0].legend()

axs[1].plot(sim_cl['t']*1000, sim_cl['v_ac'], "C0", linewidth=1.0, label="$v_{ac}$")
axs[1].set_ylabel("$v_{ac}$ [V]"); axs[1].legend()

axs[2].plot(sim_cl['t']*1000, sim_cl['i_L'], "C3", linewidth=0.6, alpha=0.7, label="$i_L$ (actual)")
axs[2].plot(sim_cl['t']*1000, sim_cl['i_ref'], "C1", linewidth=1.2, alpha=0.9, label="$i_{ref}$ (commanded)")
axs[2].set_ylabel("Current [A]"); axs[2].legend(loc="lower right")

axs[3].plot(sim_cl['t']*1000, sim_cl['duty'], "C4", linewidth=0.5)
axs[3].set_ylabel("Duty"); axs[3].set_xlabel("Time [ms]")
plt.tight_layout(); plt.show()

# Metrics (skip first 30 cycles for the slow 5Hz voltage loop to settle)
mask = sim_cl['t'] >= 30.0/p.f_line
fs = 1.0/(sim_cl['t'][1] - sim_cl['t'][0])
pf = power_factor(sim_cl['v_ac'][mask], sim_cl['i_in'][mask], f_s=fs, f_line=p.f_line)
i_filt_m = line_band_filter(sim_cl['i_in'][mask], fs, p.f_line, 20)
thd = thd_current(i_filt_m, fs, p.f_line)
v_o_mean = sim_cl['v_o'][mask].mean()
v_o_pp = sim_cl['v_o'][mask].max() - sim_cl['v_o'][mask].min()
print()
print("Closed-loop CCM PFC steady-state metrics (V_ac=120V):")
print(f"  V_o mean        = {v_o_mean:.2f} V  (target 400.0 V, error {(v_o_mean-400)/400*100:.2f}%)")
print(f"  V_o ripple pp   = {v_o_pp:.2f} V")
print(f"  PF              = {pf:.4f}")
print(f"  THD             = {thd*100:.2f}%")
print(f"  i_L max         = {sim_cl['i_L'][mask].max():.2f} A  (vs DCM ~5A)")

if pf > 0.95 and abs(v_o_mean - 400) < 10 and thd < 0.20:
    print()
    print("✅  Closed-loop CCM PFC PROVEN: two loops + multiplier work as designed")
else:
    print()
    print("⚠️   Performance off-target — revisit f_c_i or K_mult.")
"""))

    cells.append(md(r"""
## 7. Comparison with DCM

| | DCM (notebook 02) | CCM (this notebook) |
|---|---|---|
| Inductor | 100 µH | 1 mH (10× larger) |
| Peak $i_L$ at full load | ~5 A | ~2 A |
| Current waveform | sharp triangle pulses | smoothly-varying envelope |
| Compensator complexity | 1 PI | **2 PIs + 1 multiplier** |
| Lines of controller code | ~5 | ~15 |
| PF | 0.99 (low line) → 0.97 (high) | 0.99 across full range |
| THD | 6% (low) → 26% (high) | < 10% across full range |
| EMI | high (sharp transitions) | low (continuous current) |

CCM is the right answer for higher power, tighter PF/THD specs, and
designs where EMI matters. DCM is the right answer for power < 150 W
where the simplicity and "natural PFC" payoff are worth the higher
peak current and EMI.

## Summary

CCM boost PFC is the textbook "active PFC" — the architecture you'll
see in every modern off-line power supply above ~150 W. The two-loop
structure with multiplier is universal:

- **Inner current loop**: pure integrator plant $V_o/(sL)$, fast PI
  at $f_{sw}/10$.
- **Multiplier**: $i_{ref} = K_{mult} \cdot i_{ref,amp} \cdot |v_{ac}|$.
  Translates outer-loop amplitude into instantaneous shape.
- **Outer voltage loop**: same first-order plant as DCM, slow PI at
  $\le 2 f_{line}/10$.

**Next**: `04_boost_pfc_simulation.ipynb` runs both DCM and CCM
side by side, sweeps the line voltage, adds load steps, and shows
the full waveform comparison.

**Suggested exercises**

1. Push $f_{c,i}$ to 30 kHz and watch the current loop become
   marginally stable — the integrator plant doesn't gain phase
   margin from the zero anymore.
2. Run with $V_{ac} = 230$ V and $V_{ac} = 90$ V; the CCM design
   should give similar PF/THD at both, unlike DCM.
3. Trip the voltage loop's $i_{ref,amp}$ output to test the duty
   saturation — what happens if you ask for more power than the
   line can deliver?
4. Compare $i_L$ peak between CCM and DCM at the same operating
   point — CCM should be 2-3× lower because the current is
   continuous (not pulsed).
"""))

    return cells


# ---------------------------------------------------------------------------
# Notebook 4 — Simulation / Comparison
# ---------------------------------------------------------------------------


def build_simulation_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 4 — DCM vs CCM Side-by-Side Simulation

> **Goal.** With both designs validated independently, now run them
> on the same axes: steady-state at multiple line voltages, line-step
> response, load-step response, full-range PF/THD sweep. Synthesize
> the case for picking one or the other.

**Prerequisites**

- Notebooks 01, 02, 03 of this project.

This notebook reuses the compensators designed in the previous two
notebooks (briefly re-derived here for self-containedness) and runs
focused experiments.
"""))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from dataclasses import replace

from boost_pfc_model import (
    BoostPFCParams,
    dcm_voltage_loop_plant, ccm_current_loop_plant, ccm_voltage_loop_plant,
    simulate_closed_loop_dcm, simulate_closed_loop_ccm,
    power_factor, thd_current, line_band_filter,
)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

p_dcm = BoostPFCParams.dcm_design()
p_ccm = BoostPFCParams.ccm_design()
V_ramp = 5.0
T_s = p_dcm.T_sw  # same f_sw for both
print(f"DCM: L = {p_dcm.L*1e6:.1f} µH")
print(f"CCM: L = {p_ccm.L*1e6:.1f} µH")
"""))

    cells.append(md(r"""
## 1. Re-derive the compensators

Quick reproduction of notebooks 02 and 03 compensators.
"""))

    cells.append(code(r"""
def design_pi(plant, f_c, V_ramp=5.0, zero_freq=None):
    '''PI for a first-order plant K/(1+s·tau). Cancels the plant pole.'''
    plant_pwm = signal.TransferFunction(np.array(plant.num)/V_ramp, np.array(plant.den))
    omega_c = 2*np.pi*f_c
    if zero_freq is None:
        tau_z = plant.den[0]/plant.den[1]  # plant pole reciprocal
    else:
        tau_z = 1.0/(2*np.pi*zero_freq)
    dc_gain = plant_pwm.num[0]/plant_pwm.den[1]
    K = omega_c / dc_gain
    Gc = signal.TransferFunction([K*tau_z, K], [1.0, 0.0])
    bd, ad, _ = signal.cont2discrete((Gc.num, Gc.den), dt=T_s, method='bilinear')
    return np.asarray(bd).flatten()/ad[0], np.asarray(ad)/ad[0]

def design_pi_integrator(plant, f_c, V_ramp=5.0, zero_below_xover=3.0):
    '''PI for an integrator plant V_o/(sL). Zero placed below crossover.'''
    plant_pwm = signal.TransferFunction(np.array(plant.num)/V_ramp, np.array(plant.den))
    omega_c = 2*np.pi*f_c
    tau_z = zero_below_xover / omega_c  # zero at f_c/zero_below_xover
    plant_mag_at_fc = plant_pwm.num[0] / (omega_c * plant_pwm.den[0])
    K = omega_c / (np.sqrt(1+(omega_c*tau_z)**2) * plant_mag_at_fc)
    Gc = signal.TransferFunction([K*tau_z, K], [1.0, 0.0])
    bd, ad, _ = signal.cont2discrete((Gc.num, Gc.den), dt=T_s, method='bilinear')
    return np.asarray(bd).flatten()/ad[0], np.asarray(ad)/ad[0]

# DCM: 1 PI for voltage loop
bv_dcm, av_dcm = design_pi(dcm_voltage_loop_plant(p_dcm), f_c=10.0, V_ramp=V_ramp)
# CCM: 2 PIs (current + voltage) + multiplier
bi_ccm, ai_ccm = design_pi_integrator(ccm_current_loop_plant(p_ccm),
                                       f_c=10e3, V_ramp=V_ramp, zero_below_xover=3.0)
bv_ccm, av_ccm = design_pi(ccm_voltage_loop_plant(p_ccm), f_c=10.0, V_ramp=V_ramp)
K_mult = 0.001
print("Compensators designed.")
"""))

    cells.append(md(r"""
## 2. Steady-state at $V_{ac}$ = 120 V — both modes
"""))

    cells.append(code(r"""
sim_dcm = simulate_closed_loop_dcm(p_dcm, bv_dcm, av_dcm, V_ac=120.0,
                                    n_line_cycles=15, samples_per_period=60,
                                    v_ref=400.0, V_ramp=V_ramp)
sim_ccm = simulate_closed_loop_ccm(p_ccm, bi_ccm, ai_ccm, bv_ccm, av_ccm,
                                    V_ac=120.0, n_line_cycles=40, samples_per_period=40,
                                    v_ref=400.0, V_ramp=V_ramp, K_mult=K_mult)

fig, axs = plt.subplots(3, 2, figsize=(15, 9), sharex=True)
for col, (sim, title, p_) in enumerate([(sim_dcm, "DCM", p_dcm), (sim_ccm, "CCM", p_ccm)]):
    axs[0, col].plot(sim['t']*1000, sim['v_o'], "C2", linewidth=1.0)
    axs[0, col].axhline(400.0, color="k", linestyle=":", alpha=0.4)
    axs[0, col].set_ylabel("$v_o$ [V]")
    axs[0, col].set_title(f"{title} — V_ac = 120 V, P_o = {p_.P_o:.0f} W")
    fs = 1.0/(sim['t'][1] - sim['t'][0])
    i_filt = line_band_filter(sim['i_in'], fs, p_.f_line, 20)
    axs[1, col].plot(sim['t']*1000, sim['i_in'], "C3", linewidth=0.3, alpha=0.4)
    axs[1, col].plot(sim['t']*1000, i_filt, "C1", linewidth=1.5, label="filtered")
    axs[1, col].set_ylabel("$i_{in}$ [A]")
    axs[1, col].legend()
    axs[2, col].plot(sim['t']*1000, sim['duty'], "C4", linewidth=0.5)
    axs[2, col].set_ylabel("Duty"); axs[2, col].set_xlabel("Time [ms]")

plt.tight_layout(); plt.show()

# Metrics
for sim, name, p_ in [(sim_dcm, "DCM", p_dcm), (sim_ccm, "CCM", p_ccm)]:
    mask = sim['t'] >= (30.0 if "CCM" in name else 6.0)/p_.f_line
    fs = 1.0/(sim['t'][1] - sim['t'][0])
    pf = power_factor(sim['v_ac'][mask], sim['i_in'][mask], f_s=fs, f_line=p_.f_line)
    i_filt_m = line_band_filter(sim['i_in'][mask], fs, p_.f_line, 20)
    thd = thd_current(i_filt_m, fs, p_.f_line)
    v_o_mean = sim['v_o'][mask].mean()
    iL_max = sim['i_L'][mask].max()
    print(f"{name}: V_o = {v_o_mean:.2f}V, PF = {pf:.4f}, THD = {thd*100:.2f}%, "
          f"i_L peak = {iL_max:.2f}A")
"""))

    cells.append(md(r"""
## 3. PF and THD across the universal-mains range
"""))

    cells.append(code(r"""
V_ac_sweep = [90, 110, 130, 160, 200, 230]
results = {"DCM": {"PF": [], "THD": [], "iL_pk": []},
           "CCM": {"PF": [], "THD": [], "iL_pk": []}}
for V_ac in V_ac_sweep:
    s_d = simulate_closed_loop_dcm(p_dcm, bv_dcm, av_dcm, V_ac=V_ac,
                                    n_line_cycles=10, samples_per_period=60,
                                    v_ref=400.0, V_ramp=V_ramp)
    s_c = simulate_closed_loop_ccm(p_ccm, bi_ccm, ai_ccm, bv_ccm, av_ccm,
                                    V_ac=V_ac, n_line_cycles=40, samples_per_period=40,
                                    v_ref=400.0, V_ramp=V_ramp, K_mult=K_mult)
    for sim, name, p_ in [(s_d, "DCM", p_dcm), (s_c, "CCM", p_ccm)]:
        mask = sim['t'] >= (30.0 if name == "CCM" else 5.0)/p_.f_line
        fs = 1.0/(sim['t'][1] - sim['t'][0])
        pf = power_factor(sim['v_ac'][mask], sim['i_in'][mask], f_s=fs, f_line=p_.f_line)
        i_filt_m = line_band_filter(sim['i_in'][mask], fs, p_.f_line, 20)
        thd = thd_current(i_filt_m, fs, p_.f_line)
        results[name]["PF"].append(pf)
        results[name]["THD"].append(thd*100)
        results[name]["iL_pk"].append(sim['i_L'][mask].max())
    print(f"V_ac={V_ac}V done")

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
for col, key, ylabel, ylim in [
    (0, "PF", "Power factor", (0.85, 1.01)),
    (1, "THD", "THD [%]", (0, 35)),
    (2, "iL_pk", "Peak $i_L$ [A]", None),
]:
    axs[col].plot(V_ac_sweep, results["DCM"][key], "o-", color="C3", label="DCM")
    axs[col].plot(V_ac_sweep, results["CCM"][key], "s-", color="C0", label="CCM")
    axs[col].set_xlabel("$V_{ac,rms}$ [V]"); axs[col].set_ylabel(ylabel)
    if ylim: axs[col].set_ylim(ylim)
    axs[col].legend()
axs[0].axhline(0.95, color="C2", linestyle=":", alpha=0.5, label="IEC target 0.95")
axs[1].axhline(20, color="C2", linestyle=":", alpha=0.5)
plt.suptitle("PF, THD, and peak inductor current across universal mains")
plt.tight_layout(); plt.show()
"""))

    cells.append(md(r"""
## 4. Line step

What happens when the line voltage steps from 90 → 180 V mid-run?
The slow voltage loop has to re-regulate the output. Expect 100s of
ms of transient — far slower than the DC-DC converters we've seen,
because the BW is constrained to be << $f_{line}$.

We approximate this with a piecewise input: $V_{ac}$ = 90 V for the
first 200 ms, then 180 V for the next 200 ms. We use the simulator
with two consecutive runs and stitch.
"""))

    cells.append(code(r"""
# Run DCM with a manually-implemented line step via two segments
# We need a stitched simulator; simplest is to run for 20 line cycles
# at 90V, snapshot final state, restart at 180V. The shipped simulator
# doesn't expose state IO, so we use a quick custom inline run.

# For now, just run each line voltage independently and show the SS
# performance at each — full line-step transient is a future extension.
print("Line-step transient: see the future-work item in the project README.")
print("Here we just report steady-state performance at the two endpoints:")
for V_ac in [90.0, 180.0]:
    s_d = simulate_closed_loop_dcm(p_dcm, bv_dcm, av_dcm, V_ac=V_ac,
                                    n_line_cycles=15, samples_per_period=60,
                                    v_ref=400.0, V_ramp=V_ramp)
    mask = s_d['t'] >= 10.0/p_dcm.f_line
    fs = 1.0/(s_d['t'][1]-s_d['t'][0])
    pf = power_factor(s_d['v_ac'][mask], s_d['i_in'][mask], f_s=fs, f_line=p_dcm.f_line)
    i_filt = line_band_filter(s_d['i_in'][mask], fs, p_dcm.f_line, 20)
    thd = thd_current(i_filt, fs, p_dcm.f_line)
    print(f"  DCM @ V_ac={V_ac:.0f}V: V_o={s_d['v_o'][mask].mean():.1f}V, "
          f"PF={pf:.4f}, THD={thd*100:.2f}%")
"""))

    cells.append(md(r"""
## 5. Load step

Halve the load mid-run: $P_o$ goes from 100 W to 50 W. Both modes
should re-regulate; the transient duration depends on the voltage-
loop bandwidth (same for both modes by design).
"""))

    cells.append(code(r"""
# Approximation: run at half load (P_o=50W → R_load=3200Ω) and compare
p_dcm_half = replace(p_dcm, P_o=50.0)
p_ccm_half = replace(p_ccm, P_o=50.0)

# Re-design controllers for half-load plant (lower bandwidth penalty)
bv_dcm_h, av_dcm_h = design_pi(dcm_voltage_loop_plant(p_dcm_half), f_c=10.0, V_ramp=V_ramp)
# (current-loop plant is independent of P_o for CCM, voltage-loop changes)
bv_ccm_h, av_ccm_h = design_pi(ccm_voltage_loop_plant(p_ccm_half), f_c=10.0, V_ramp=V_ramp)

for label, p_, sim_fn, args in [
    ("DCM full load", p_dcm,    simulate_closed_loop_dcm,
        (bv_dcm, av_dcm)),
    ("DCM half load", p_dcm_half, simulate_closed_loop_dcm,
        (bv_dcm_h, av_dcm_h)),
    ("CCM full load", p_ccm,    simulate_closed_loop_ccm,
        (bi_ccm, ai_ccm, bv_ccm, av_ccm)),
    ("CCM half load", p_ccm_half, simulate_closed_loop_ccm,
        (bi_ccm, ai_ccm, bv_ccm_h, av_ccm_h)),
]:
    if "DCM" in label:
        s = sim_fn(p_, *args, V_ac=120.0, n_line_cycles=15, samples_per_period=40,
                   v_ref=400.0, V_ramp=V_ramp)
        mask = s['t'] >= 8.0/p_.f_line
    else:
        s = sim_fn(p_, *args, V_ac=120.0, n_line_cycles=40, samples_per_period=40,
                   v_ref=400.0, V_ramp=V_ramp, K_mult=K_mult)
        mask = s['t'] >= 30.0/p_.f_line
    fs = 1.0/(s['t'][1]-s['t'][0])
    pf = power_factor(s['v_ac'][mask], s['i_in'][mask], f_s=fs, f_line=p_.f_line)
    i_filt = line_band_filter(s['i_in'][mask], fs, p_.f_line, 20)
    thd = thd_current(i_filt, fs, p_.f_line)
    print(f"  {label}: V_o={s['v_o'][mask].mean():.1f}V, PF={pf:.4f}, "
          f"THD={thd*100:.2f}%, i_L_pk={s['i_L'][mask].max():.2f}A")
"""))

    cells.append(md(r"""
## 6. Pick a mode — design guide

After all this, the design choice for a real product:

**Pick DCM if**:
- Power ≤ 150 W
- BOM cost is tight (one less PI + multiplier in firmware)
- High switching frequency (>200 kHz) is OK (DCM benefits from
  it — keeps the inductor small)
- Some THD margin available (passes IEC class D easily at low line,
  marginal at high line)

**Pick CCM if**:
- Power ≥ 200 W
- Tight THD spec across full line range
- EMI is critical (CCM has continuous current → softer EMI signature)
- Mid-range $f_{sw}$ (50-100 kHz) is preferred for efficiency

**Transition-mode (CrM) PFC**, used in chips like L6562, is a third
option not covered here: variable $T_{sw}$ keeps $i_L$ on the
DCM/CCM boundary. Combines the natural-PFC property of DCM with the
lower peak current of CCM. The math is more involved and the
controller is mostly analog hardware logic; we skip it in this
notebook.

## Summary

Boost PFC is the gateway converter into the world of AC-input power
electronics. DCM and CCM offer two different philosophies — physics
vs active control — for solving the same problem (shape the input
current to look like a resistor at the line).

**Cross-validation against Pulsim**: open
[`00_boost_pfc_pulsim_validation.ipynb`](00_boost_pfc_pulsim_validation.ipynb)
for an executed notebook that builds the full AC-input power stage
in Pulsim (``add_sine_voltage_source`` → ``add_bridge_rectifier`` →
boost stage) and overlays the rectified line voltage against the
analytical $|V_{pk} \\sin(\\omega_{line} t)|$ envelope.

Both modes share:
- First-order voltage-loop plant
- 2·$f_{line}$ output ripple
- Voltage-loop bandwidth constrained to $\le 2 f_{line}/10$

They differ in:
- Inductor sizing (factor of 10×)
- Controller complexity (1 loop vs 2 loops + multiplier)
- PF/THD at high line (DCM degrades, CCM holds)
- Peak inductor current (DCM is 2-3× higher)

**Library now covers**:

| Converter | Settling | Input | Loops | Notes |
|---|---|---|---|---|
| buck → half-bridge (6 converters) | µs–ms | DC | 1 | |
| **boost PFC DCM** | **100 ms** | **AC** | **1** | **first-order plant!** |
| **boost PFC CCM** | **100 ms** | **AC** | **2** | **multiplier-based** |

**Suggested exercises**

1. Implement the missing line-step transient (run for 200 ms at
   90 V, snapshot state, continue at 230 V). Compare DCM vs CCM
   settling time.
2. Replace the PI compensators with a Type-II (add a high-frequency
   pole at 1 kHz). Does it help with switching ripple injection
   into the loop?
3. Push the voltage-loop bandwidth to 30 Hz on both modes and watch
   PF degrade as the loop "corrects" the 2·f_line ripple.
4. Add input filter (L + C between line and bridge) and watch how
   the filter resonance interacts with the PFC stage. (Hint: a
   2.2 µF cap and 0.5 mH inductor put a resonance at ~5 kHz — well
   below the current-loop bandwidth.)
"""))

    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    write_notebook(build_basics_notebook(),     HERE / "01_boost_pfc_basics.ipynb")
    write_notebook(build_dcm_notebook(),         HERE / "02_boost_pfc_dcm.ipynb")
    write_notebook(build_ccm_notebook(),         HERE / "03_boost_pfc_ccm.ipynb")
    write_notebook(build_simulation_notebook(),  HERE / "04_boost_pfc_simulation.ipynb")


if __name__ == "__main__":
    main()
