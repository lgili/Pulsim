"""Generator for the 3-phase VSI teaching notebooks.

Four notebooks:
  01_vsi_basics         — topology, switching states, SPWM, Clarke + Park transforms
  02_vsi_svpwm          — SVPWM via min-max injection, open-loop switched demo
  03_vsi_standalone     — dq voltage control, closed-loop sim with RL load
  04_vsi_gridtie        — SRF-PLL + dq current control, P/Q injection sim
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
# 1 — Three-Phase VSI Basics

> **Goal.** Understand the 3-phase voltage-source inverter topology
> (6 switches in 3 legs), how SPWM produces a sinusoidal output, and
> introduce the **Clarke** and **Park** transforms that turn
> 3-phase AC into 2-axis DC.

**Prerequisites**

- Half-bridge converter project (`projects/converters/half_bridge/`)
  — a 3-phase VSI is conceptually three half-bridges sharing the
  same DC bus.
- Boost PFC project (`projects/converters/boost_pfc/`) — for the
  AC-side intuition (line voltages, fundamentals, harmonics).

**What you'll know at the end**

1. The 3-phase VSI topology and its **8 switching states** (6 active
   + 2 zero) arranged as a hexagon in the αβ plane.
2. How to generate three SPWM duty signals (one per leg) from three
   reference sinusoids and a triangular carrier — the simplest
   modulation strategy.
3. The **Clarke transform** (abc → αβ): why the $2/3$ factor, why
   amplitude-invariant, what positive-sequence looks like.
4. The **Park transform** (αβ → dq): the rotating-frame insight that
   turns 3-phase AC quantities into DC quantities — the headline
   pedagogical moment of this entire project.
"""))

    cells.append(md("## Setup"))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from vsi_3phase_model import (
    VSI3PhaseParams,
    clarke_transform, inverse_clarke_transform,
    park_transform, inverse_park_transform,
    spwm_duties,
    simulate_open_loop_svpwm,
    operating_point_report,
    fundamental_rms, thd,
)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

p = VSI3PhaseParams()
print(operating_point_report(p, mode="standalone"))
"""))

    cells.append(md(r"""
## 1. Topology and switching states

A 3-phase VSI has **3 legs**, each with a top switch + bottom switch.
Each leg is a half-bridge sharing the same DC bus:

```
         +V_dc ──┬────┬────┬──── (top switches)
                 │    │    │
                S1   S3   S5
                 │    │    │
                 ├────┼────┼──── output: a, b, c phases
                 │    │    │
                S4   S6   S2
                 │    │    │
            0 ───┴────┴────┴──── (bottom switches)
```

Each leg has two **complementary states** (S_top on, S_bot off — or
vice-versa). With 3 legs, that's $2^3 = 8$ switching states.

In the **αβ plane**, those 8 states map to **6 active vectors**
(forming a hexagon) and **2 zero vectors** (at the origin: all-top
or all-bottom):

```
                  V2 (110)
                  /        \
                 /          \
        V3 (010)            V1 (100)
                |    *V0/V7
        V4 (011)            V6 (101)
                 \          /
                  \        /
                  V5 (001)
```

Each label `(s_a s_b s_c)` lists whether each top switch is on (`1`)
or off (`0`).
"""))

    cells.append(code(r"""
# Plot the 6 active vectors in the αβ plane
def state_to_vector(s_a, s_b, s_c, V_dc=400.0):
    '''Map switching state (0/1 per leg) to αβ vector via Clarke.'''
    # Top switch on → that leg's pole at V_dc; otherwise at 0
    # Subtract bus midpoint for symmetry → v_pole_x = V_dc/2 or -V_dc/2
    v_a = V_dc * (s_a - 0.5)
    v_b = V_dc * (s_b - 0.5)
    v_c = V_dc * (s_c - 0.5)
    # Use clarke_transform (returns peak)
    arr_a = np.array([v_a]); arr_b = np.array([v_b]); arr_c = np.array([v_c])
    v_alpha, v_beta = clarke_transform(arr_a, arr_b, arr_c)
    return float(v_alpha[0]), float(v_beta[0])


states = [(1,0,0), (1,1,0), (0,1,0), (0,1,1), (0,0,1), (1,0,1), (0,0,0), (1,1,1)]
labels = ["V1 (100)", "V2 (110)", "V3 (010)", "V4 (011)",
          "V5 (001)", "V6 (101)", "V0 (000)", "V7 (111)"]

fig, ax = plt.subplots(figsize=(6, 6))
for st, lbl in zip(states, labels):
    va, vb = state_to_vector(*st, p.V_dc)
    if lbl.startswith("V0") or lbl.startswith("V7"):
        ax.plot(va, vb, "ko", markersize=10)
        ax.annotate(lbl, (va, vb), textcoords="offset points", xytext=(10, 5))
    else:
        ax.plot([0, va], [0, vb], "C0-")
        ax.plot(va, vb, "C0o", markersize=10)
        ax.annotate(lbl, (va, vb), textcoords="offset points", xytext=(8, 5))

# Plot the rotating reference circle (linear-mod limit for SVPWM)
theta_circ = np.linspace(0, 2*np.pi, 200)
R_limit = p.V_dc / np.sqrt(3.0)  # SVPWM linear-mod hexagon-inscribed circle
ax.plot(R_limit*np.cos(theta_circ), R_limit*np.sin(theta_circ), "C3--",
        alpha=0.5, label=f"Linear SVPWM limit: $V_{{dc}}/\\sqrt{{3}}$ = {R_limit:.0f} V")
R_spwm = p.V_dc / 2.0  # SPWM linear-mod limit
ax.plot(R_spwm*np.cos(theta_circ), R_spwm*np.sin(theta_circ), "C2:",
        alpha=0.5, label=f"SPWM limit: $V_{{dc}}/2$ = {R_spwm:.0f} V")
ax.axhline(0, color="k", linestyle=":", alpha=0.3)
ax.axvline(0, color="k", linestyle=":", alpha=0.3)
ax.set_xlabel("$v_\\alpha$ [V]"); ax.set_ylabel("$v_\\beta$ [V]")
ax.set_title("3-phase VSI: 6 active + 2 zero space vectors")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout(); plt.show()

print(f"SVPWM-vs-SPWM bonus: {R_limit/R_spwm:.4f}× = {(R_limit/R_spwm - 1)*100:.1f}% extra output")
"""))

    cells.append(md(r"""
## 2. SPWM — three independent SPWMs

The simplest modulation: generate three sinusoidal references
$v_{ref,a/b/c}(t)$ 120° apart, compare each to a common triangle
carrier at $f_{sw}$, and produce the three leg duties.

Per-leg duty:
$$d_x = 0.5 + v_{ref,x} / V_{dc}$$

with $|v_{ref}| \le V_{dc}/2$ for linear operation. The peak
**line-to-line** output is then bounded by $V_{dc} \cdot \sqrt 3 / 2
\approx 0.866 V_{dc}$ — that's the SPWM ceiling.
"""))

    cells.append(code(r"""
# Generate one cycle of SPWM duties
t = np.linspace(0, 1.0/p.f_o, 1000)
V_ref_pk = p.m_a * p.V_dc / 2.0
v_ref_a = V_ref_pk * np.cos(p.omega_o * t)
v_ref_b = V_ref_pk * np.cos(p.omega_o * t - 2*np.pi/3)
v_ref_c = V_ref_pk * np.cos(p.omega_o * t + 2*np.pi/3)
d_a, d_b, d_c = spwm_duties(v_ref_a, v_ref_b, v_ref_c, p.V_dc)

fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axs[0].plot(t*1000, v_ref_a, label="$v_{ref,a}$")
axs[0].plot(t*1000, v_ref_b, "--", label="$v_{ref,b}$")
axs[0].plot(t*1000, v_ref_c, ":", label="$v_{ref,c}$")
axs[0].set_ylabel("Reference [V]"); axs[0].legend(loc="lower right")
axs[0].set_title("SPWM: three independent reference sinusoids → three duties")

axs[1].plot(t*1000, d_a, label="$d_a$")
axs[1].plot(t*1000, d_b, "--", label="$d_b$")
axs[1].plot(t*1000, d_c, ":", label="$d_c$")
axs[1].axhline(0.5, color="k", linestyle=":", alpha=0.4)
axs[1].set_ylabel("Duty cycle"); axs[1].set_xlabel("Time [ms]")
axs[1].legend(loc="lower right")
plt.tight_layout(); plt.show()
print(f"Duty range: [{d_a.min():.4f}, {d_a.max():.4f}]")
"""))

    cells.append(md(r"""
## 3. Open-loop SPWM switched simulation

Now drive the actual switching engine with SPWM. The 3-phase output
is filtered through an LC ($L_f = 1$ mH, $C_f = 10$ µF, per phase)
that strips the switching ripple and leaves a clean sinusoid.

We'll set `use_svpwm=False` to use standard SPWM. The next notebook
upgrades to SVPWM for the 15% bonus.
"""))

    cells.append(code(r"""
sim = simulate_open_loop_svpwm(p, m_a=p.m_a, n_cycles=3, samples_per_period=80,
                                use_svpwm=False)
mask = sim['t'] > 1.0/p.f_o  # skip first cycle
fs = 1.0/(sim['t'][1] - sim['t'][0])
v_Ca_fund_rms = fundamental_rms(sim['v_Ca'][mask], fs, p.f_o)
v_Ca_thd = thd(sim['v_Ca'][mask], fs, p.f_o, 40)

fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
axs[0].plot(sim['t']*1000, sim['v_pole_a'], "C0", linewidth=0.5, alpha=0.7,
            label="$v_{pole,a}$ (PWM)")
axs[0].set_ylabel("Pole voltage [V]"); axs[0].legend(loc="upper right")
axs[0].set_title(f"Open-loop SPWM, $m_a$ = {p.m_a:.3f}")

axs[1].plot(sim['t']*1000, sim['v_LL_ab'], "C3", linewidth=0.5, alpha=0.5,
            label="$v_{LL,ab}$ (pre-filter)")
axs[1].set_ylabel("Line-to-line [V]"); axs[1].legend(loc="upper right")

axs[2].plot(sim['t']*1000, sim['v_Ca'], "C2", linewidth=1.0, label="$v_{Ca}$")
axs[2].plot(sim['t']*1000, sim['v_Cb'], "C0", linewidth=1.0, label="$v_{Cb}$")
axs[2].plot(sim['t']*1000, sim['v_Cc'], "C1", linewidth=1.0, label="$v_{Cc}$")
axs[2].set_ylabel("Filtered phase [V]"); axs[2].set_xlabel("Time [ms]")
axs[2].legend(loc="lower right")
plt.tight_layout(); plt.show()

print(f"v_Ca fundamental rms = {v_Ca_fund_rms:.2f} V  (expect {p.V_o_LN_pk/np.sqrt(2):.2f} V)")
print(f"v_Ca THD             = {v_Ca_thd*100:.2f} %  (expect < 2 % with a clean filter)")
"""))

    cells.append(md(r"""
## 4. Clarke transform — three phases to αβ

Three real-valued sinusoids 120° apart can be encoded as a single
rotating 2D vector in the **αβ plane**. The Clarke transform makes
the encoding explicit:

$$
\begin{bmatrix} v_\alpha \\ v_\beta \end{bmatrix}
= \frac{2}{3}
\begin{bmatrix}
1 & -\tfrac{1}{2} & -\tfrac{1}{2} \\
0 & \tfrac{\sqrt 3}{2} & -\tfrac{\sqrt 3}{2}
\end{bmatrix}
\begin{bmatrix} v_a \\ v_b \\ v_c \end{bmatrix}
$$

The $2/3$ factor is the **amplitude-invariant** convention: with
$2/3$, the αβ vector magnitude equals the *peak* of the original
phase voltage. (Some textbooks use $\sqrt{2/3}$ for the
**power-invariant** version. We use $2/3$.)

For a balanced positive-sequence input
$v_a = V_{pk}\cos(\omega t)$, $v_b = V_{pk}\cos(\omega t - 2\pi/3)$,
$v_c = V_{pk}\cos(\omega t + 2\pi/3)$:

$$v_\alpha(t) = V_{pk} \cos(\omega t), \qquad v_\beta(t) = V_{pk} \sin(\omega t)$$

i.e. αβ traces a circle of radius $V_{pk}$ at angular frequency
$\omega$.
"""))

    cells.append(code(r"""
# Verify on the open-loop sim output
v_alpha_sim, v_beta_sim = clarke_transform(sim['v_Ca'], sim['v_Cb'], sim['v_Cc'])

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].plot(sim['t']*1000, v_alpha_sim, "C0", label="$v_\\alpha$")
axs[0].plot(sim['t']*1000, v_beta_sim, "C1", label="$v_\\beta$")
axs[0].set_xlabel("Time [ms]"); axs[0].set_ylabel("Voltage [V]")
axs[0].set_title("αβ components of the filter output")
axs[0].legend()

# αβ trajectory (the circle)
mask = sim['t'] > 1.0/p.f_o
axs[1].plot(v_alpha_sim[mask], v_beta_sim[mask], "C2", alpha=0.6)
axs[1].set_xlabel("$v_\\alpha$ [V]"); axs[1].set_ylabel("$v_\\beta$ [V]")
axs[1].set_title("αβ trajectory — a circle (positive sequence)")
axs[1].set_aspect("equal")
axs[1].axhline(0, color="k", linestyle=":", alpha=0.3)
axs[1].axvline(0, color="k", linestyle=":", alpha=0.3)
plt.tight_layout(); plt.show()
print(f"αβ peak: |v| = {np.sqrt(v_alpha_sim[mask]**2 + v_beta_sim[mask]**2).mean():.2f} V "
      f"(expect ≈ {p.V_o_LN_pk:.2f} V)")
"""))

    cells.append(md(r"""
## 5. Park transform — αβ to dq (the headline insight)

The αβ vector rotates at $\omega$. **Rotate the frame WITH IT** at
the same angular velocity, and the vector becomes stationary —
i.e. a **DC** quantity. That's the Park transform:

$$
\begin{bmatrix} v_d \\ v_q \end{bmatrix}
=
\begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}
\begin{bmatrix} v_\alpha \\ v_\beta \end{bmatrix},
\quad \theta = \omega t
$$

For positive-sequence input with d-axis aligned to phase-a at
$t = 0$:

$$v_d(t) = V_{pk}, \qquad v_q(t) = 0$$

**Two constants** instead of three sinusoids. The compensator now
acts on DC quantities, with all the familiar tooling: PI for steady-
state, plant looks first-order or second-order in dq.

This is *the* trick that makes 3-phase control tractable.
"""))

    cells.append(code(r"""
# Build the Park angle θ = ω_o · t and transform
theta = p.omega_o * sim['t']
v_d_sim, v_q_sim = park_transform(v_alpha_sim, v_beta_sim, theta)

fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axs[0].plot(sim['t']*1000, v_alpha_sim, "C0", alpha=0.5, label="$v_\\alpha$")
axs[0].plot(sim['t']*1000, v_beta_sim, "C1", alpha=0.5, label="$v_\\beta$")
axs[0].set_ylabel("αβ [V]"); axs[0].legend(loc="lower right")
axs[0].set_title("Three sinusoids → αβ (two sinusoids) → dq (two DC quantities)")

axs[1].plot(sim['t']*1000, v_d_sim, "C2", linewidth=1.5, label="$v_d$")
axs[1].plot(sim['t']*1000, v_q_sim, "C3", linewidth=1.5, label="$v_q$")
axs[1].axhline(p.V_o_LN_pk, color="k", linestyle=":", alpha=0.4,
               label=f"$V_d$ expected = {p.V_o_LN_pk:.1f} V")
axs[1].axhline(0, color="k", linestyle=":", alpha=0.4)
axs[1].set_ylabel("dq [V]"); axs[1].set_xlabel("Time [ms]")
axs[1].legend(loc="lower right")
plt.tight_layout(); plt.show()

mask = sim['t'] > 1.5/p.f_o
print(f"Steady-state v_d mean = {v_d_sim[mask].mean():.2f} V  (expect {p.V_o_LN_pk:.2f})")
print(f"Steady-state v_q mean = {v_q_sim[mask].mean():.4f} V  (expect 0)")
print(f"v_d ripple std = {v_d_sim[mask].std():.4f} V")
print(f"v_q ripple std = {v_q_sim[mask].std():.4f} V")
"""))

    cells.append(md(r"""
## 6. Summary

You've seen:

- The 3-phase VSI topology with its 8 switching states.
- How to generate SPWM duties from three references.
- The Clarke transform: 3 phases → 2 αβ components (a rotating vector
  in 2D).
- The Park transform: αβ → dq, mapping AC sinusoids to **DC**
  quantities by rotating with the line.

The dq frame is what enables straightforward control of a 3-phase
inverter — the buck-control intuition transfers verbatim.

**Cross-validation against Pulsim**: open
[`00_vsi_pulsim_validation.ipynb`](00_vsi_pulsim_validation.ipynb)
for an executed notebook that builds the 3-phase VSI in Pulsim
(``pulsim.topology.add_three_phase_vsi`` + ``add_three_phase_rl_load``
+ ``make_three_phase_spwm_fn``) and overlays the phase and
line-to-line voltages against the analytical fundamental.

**Next notebooks**:

- `02_vsi_svpwm.ipynb` — derive SVPWM, prove the min-max injection
  equivalence, show the 15% output bonus.
- `03_vsi_standalone.ipynb` — design the dq voltage compensator and
  run the closed-loop sim with an RL load.
- `04_vsi_gridtie.ipynb` — wire the SRF-PLL and the dq current
  control, inject active and reactive power.
"""))

    return cells


# ---------------------------------------------------------------------------
# Notebook 2 — SVPWM
# ---------------------------------------------------------------------------


def build_svpwm_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 2 — Space Vector PWM (SVPWM)

> **Goal.** Derive SVPWM from the space-vector decomposition,
> recognize the **min-max common-mode injection** that produces the
> same result far more simply, verify the 15% output-voltage bonus,
> and run the open-loop switched simulation with SVPWM driving an
> LC + RL load.

**Prerequisites**

- `01_vsi_basics.ipynb` (this project)

**What you'll know at the end**

1. Where the **6 active vectors + 2 zero vectors** of SVPWM come
   from and how a target reference is decomposed into adjacent
   active vectors.
2. The closed-form **min-max injection** formula and why it produces
   the same line-to-line voltages as vector decomposition.
3. The **15% bonus**: peak line-to-line in linear SVPWM = $V_{dc}$;
   in SPWM = $V_{dc} \sqrt 3/2 \approx 0.866 V_{dc}$.
4. Spectral cleanliness: SVPWM packs all harmonics around multiples
   of $f_{sw}$, with the lowest-order harmonic at $f_{sw} - 2 f_o$
   (much higher than SPWM's $f_{sw} \pm 2 f_o$).
"""))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from vsi_3phase_model import (
    VSI3PhaseParams,
    spwm_duties, svpwm_duties,
    simulate_open_loop_svpwm,
    fundamental_rms, thd,
)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

p = VSI3PhaseParams()
"""))

    cells.append(md(r"""
## 1. The vector-decomposition view

The reference voltage at any instant is a vector
$\vec V_{ref}(t)$ in the αβ plane. In each $T_{sw}$, the inverter
applies a sequence of two adjacent active vectors (e.g. $V_1$ and
$V_2$) plus the zero vectors, weighted so that the **average** equals
$\vec V_{ref}$:

$$
T_{sw} \cdot \vec V_{ref} = T_1 \vec V_1 + T_2 \vec V_2 + T_0 \vec V_0
$$

With $|V_{1,2}| = \frac{2}{3} V_{dc}$ (Clarke-amplitude scaling),
and after solving the geometry:

$$
T_1 = T_{sw} \cdot \frac{|\vec V_{ref}|}{\frac{2}{3} V_{dc}} \cdot \sin(60° - \gamma)
$$

$$
T_2 = T_{sw} \cdot \frac{|\vec V_{ref}|}{\frac{2}{3} V_{dc}} \cdot \sin(\gamma)
$$

$$
T_0 = T_{sw} - T_1 - T_2
$$

where $\gamma$ is the angle within the sector. The **maximum
modulation index** is reached when $\vec V_{ref}$ hits the hexagon
inscribed circle, $|V_{ref}| = V_{dc}/\sqrt 3$ → peak phase voltage
$= V_{dc}/\sqrt 3$ → peak line-to-line $= V_{dc}$.

Compare SPWM: max phase voltage = $V_{dc}/2$ → peak line-to-line
$= V_{dc} \sqrt 3/2 = 0.866 V_{dc}$. SVPWM gives **15.5% more**.
"""))

    cells.append(md(r"""
## 2. The min-max injection shortcut

The "real" SVPWM implementation (sector identification, dwell times,
switching-pattern assembly) is conceptually clean but verbose. There's
a **theorem** (Holmes & Lipo §6.5): the same line-to-line voltages
are produced by simply **adding a common-mode offset** of
$-(\max + \min)/2$ to each phase reference before applying standard
SPWM:

$$
v_{offset} = -\frac{\max(v_a, v_b, v_c) + \min(v_a, v_b, v_c)}{2}
$$

$$
v_{x,pwm} = v_{x,ref} + v_{offset}
$$

$$
d_x = 0.5 + v_{x,pwm} / V_{dc}
$$

The common-mode injection shifts all three references by the same
amount — that's invisible at the load (which only sees phase-to-phase
voltages). Both implementations are equivalent line-to-line.

This is the implementation we use in `svpwm_duties()`. It's a
one-liner.
"""))

    cells.append(code(r"""
# Plot: SPWM references vs SVPWM references (with injection)
t = np.linspace(0, 1.0/p.f_o, 1000)
V_ref_pk = p.V_o_LN_pk
v_a = V_ref_pk * np.cos(p.omega_o * t)
v_b = V_ref_pk * np.cos(p.omega_o * t - 2*np.pi/3)
v_c = V_ref_pk * np.cos(p.omega_o * t + 2*np.pi/3)
v_max = np.maximum(np.maximum(v_a, v_b), v_c)
v_min = np.minimum(np.minimum(v_a, v_b), v_c)
v_offset = -(v_max + v_min) / 2.0
v_a_inj = v_a + v_offset

fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axs[0].plot(t*1000, v_a, "C0", label="$v_{ref,a}$ (sinusoid)")
axs[0].plot(t*1000, v_a_inj, "C3", linewidth=2, label="$v_{ref,a} + v_{offset}$ (SVPWM)")
axs[0].plot(t*1000, v_offset, "C2--", alpha=0.5, label="$v_{offset}$ (common-mode)")
axs[0].set_ylabel("Voltage [V]"); axs[0].legend(loc="lower right")
axs[0].set_title("Min-max injection: 3rd-harmonic-like common-mode shifts phase refs")

# Compare duties
d_a_spwm, _, _ = spwm_duties(v_a, v_b, v_c, p.V_dc)
d_a_svpwm, _, _ = svpwm_duties(v_a, v_b, v_c, p.V_dc)
axs[1].plot(t*1000, d_a_spwm, "C0", label="$d_a$ SPWM")
axs[1].plot(t*1000, d_a_svpwm, "C3", linewidth=2, label="$d_a$ SVPWM (via injection)")
axs[1].axhline(0.5, color="k", linestyle=":", alpha=0.3)
axs[1].set_ylabel("Duty"); axs[1].set_xlabel("Time [ms]")
axs[1].legend(loc="lower right")
plt.tight_layout(); plt.show()
print(f"SPWM  duty range: [{d_a_spwm.min():.4f}, {d_a_spwm.max():.4f}]")
print(f"SVPWM duty range: [{d_a_svpwm.min():.4f}, {d_a_svpwm.max():.4f}]")
print(f"SVPWM lets us reach m_a = {p.m_a:.3f} (would saturate at SPWM)")
"""))

    cells.append(md(r"""
## 3. Open-loop SVPWM switched simulation

Run the full switched simulator with SVPWM. The line-to-line voltage
at the load (post-filter) should be a clean 230 V_rms sinusoid with
low THD.
"""))

    cells.append(code(r"""
sim = simulate_open_loop_svpwm(p, m_a=p.m_a, n_cycles=3, samples_per_period=80,
                                use_svpwm=True)
mask = sim['t'] > 1.0/p.f_o
fs = 1.0/(sim['t'][1] - sim['t'][0])

# Post-filter line-to-line voltage
v_LL_post_filter = sim['v_Ca'] - sim['v_Cb']
v_LL_fund = fundamental_rms(v_LL_post_filter[mask], fs, p.f_o)
v_LL_thd = thd(v_LL_post_filter[mask], fs, p.f_o, 30)

fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
axs[0].plot(sim['t']*1000, sim['v_pole_a'], "C0", linewidth=0.5, alpha=0.7)
axs[0].set_ylabel("$v_{pole,a}$ [V]"); axs[0].set_title("SVPWM switched waveforms")

axs[1].plot(sim['t']*1000, sim['v_LL_ab'], "C3", linewidth=0.3, alpha=0.5,
            label="Pre-filter $v_{LL,ab}$")
axs[1].set_ylabel("Line-to-line [V]"); axs[1].legend(loc="upper right")

axs[2].plot(sim['t']*1000, v_LL_post_filter, "C2", linewidth=1.2,
            label="Post-filter $v_{Ca}-v_{Cb}$")
axs[2].set_ylabel("Filtered LL [V]"); axs[2].set_xlabel("Time [ms]")
axs[2].legend(loc="upper right")
plt.tight_layout(); plt.show()

print(f"Post-filter LL fundamental rms = {v_LL_fund:.2f} V  (target {p.V_o_LL_rms:.2f} V)")
print(f"Post-filter LL THD             = {v_LL_thd*100:.2f} %  (target < 3% for clean output)")
"""))

    cells.append(md(r"""
## 4. Spectrum: SPWM vs SVPWM

Both SVPWM and SPWM produce harmonic content centered around $f_{sw}$
and its multiples. The two differ in how the harmonics are
distributed and which ones cancel out. SVPWM produces somewhat lower
distortion at the same $m_a$ — and crucially supports a higher $m_a$
ceiling (1.0 vs 0.866).
"""))

    cells.append(code(r"""
# Run both at the same m_a (where both work) and compare spectra
m_a_compare = 0.8  # below SPWM ceiling

sim_spwm = simulate_open_loop_svpwm(p, m_a=m_a_compare, n_cycles=4,
                                      samples_per_period=80, use_svpwm=False)
sim_svpwm = simulate_open_loop_svpwm(p, m_a=m_a_compare, n_cycles=4,
                                       samples_per_period=80, use_svpwm=True)

# Compute FFTs of the pre-filter line-to-line voltages
fs = 1.0/(sim_spwm['t'][1] - sim_spwm['t'][0])
mask_steady = sim_spwm['t'] > 1.0/p.f_o
for name, sim in [("SPWM", sim_spwm), ("SVPWM", sim_svpwm)]:
    x = sim['v_LL_ab'][mask_steady]
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
    mag = np.abs(X) * 2.0 / len(x)
    # Normalize to fundamental
    idx_fund = np.argmin(np.abs(freqs - p.f_o))
    fund_mag = mag[idx_fund]
    plt.semilogy(freqs/1000, mag/fund_mag, label=name, alpha=0.7, linewidth=0.6)

plt.axvline(p.f_sw/1000, color="C3", linestyle=":", alpha=0.5,
            label=f"$f_{{sw}}$ = {p.f_sw/1000:.0f} kHz")
plt.axvline(2*p.f_sw/1000, color="C3", linestyle=":", alpha=0.5)
plt.xlabel("Frequency [kHz]"); plt.ylabel("Magnitude (norm. to fundamental)")
plt.title(f"Spectrum of pre-filter line-to-line voltage @ $m_a$ = {m_a_compare}")
plt.legend(); plt.xlim(0, 60); plt.ylim(1e-4, 2)
plt.tight_layout(); plt.show()

# Compute THDs
thd_spwm = thd(sim_spwm['v_LL_ab'][mask_steady], fs, p.f_o, 40)
thd_svpwm = thd(sim_svpwm['v_LL_ab'][mask_steady], fs, p.f_o, 40)
print(f"Pre-filter LL THD @ m_a={m_a_compare}:  SPWM = {thd_spwm*100:.1f}%,  "
      f"SVPWM = {thd_svpwm*100:.1f}%")
"""))

    cells.append(md(r"""
## 5. Summary

SVPWM is a small algorithmic upgrade with a big output payoff:

- **15.5% more output** at the same DC bus, same load, same filter
- Cleaner spectrum (slightly lower THD at the same $m_a$, and a
  higher achievable $m_a$ ceiling)
- Same implementation cost via **min-max injection** — adding 5
  lines to a sinusoidal PWM modulator

Production motor drives, EV inverters, and grid-tie converters all
use SVPWM (or a close variant like DPWM that further reduces
switching losses). It's the standard.

**Next**: `03_vsi_standalone.ipynb` — close the voltage loop in dq
and drive an RL load with regulation.
"""))

    return cells


# ---------------------------------------------------------------------------
# Notebook 3 — Stand-alone
# ---------------------------------------------------------------------------


def build_standalone_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 3 — Stand-Alone VSI: dq Voltage Control

> **Goal.** Close the voltage loop in the dq frame. The 3-phase
> output regulates to a target $V_d$ (peak line-to-neutral) with
> $V_q$ = 0, against an RL load. Demonstrate that the dq plant is
> *buck-like* and that a simple PI handles it (modulo the high-Q
> LC resonance that we discuss explicitly).

**Prerequisites**

- `01_vsi_basics.ipynb` and `02_vsi_svpwm.ipynb` (this project)
- Buck controller notebook (`projects/converters/buck/`) — the
  control intuition transfers verbatim once we're in dq.

**What you'll be able to do at the end**

1. Derive the dq-frame plant of a 3-phase inverter + LC filter +
   resistive load.
2. Recognize it as the buck's second-order plant (with the same
   high-Q resonance issue).
3. Design a PI compensator targeting a crossover well below the
   resonance and switching frequency.
4. Discretize, wire two parallel compensators (one for $v_d$, one
   for $v_q$), and run the switched closed-loop sim.
5. Verify that $V_d$ regulates and $V_q$ stays near zero, with
   sinusoidal output at the filtered load terminals.
"""))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from vsi_3phase_model import (
    VSI3PhaseParams, standalone_voltage_plant,
    simulate_closed_loop_standalone,
    clarke_transform, park_transform,
    fundamental_rms, thd,
    operating_point_report,
)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

p = VSI3PhaseParams()
print(operating_point_report(p, mode="standalone"))
"""))

    cells.append(md(r"""
## 1. The dq-frame plant (= the buck)

With balanced 3-phase quantities and the Park frame aligned with
the modulating signal, the inverter + LC filter + load reduces to
**two decoupled second-order systems** — one per axis:

$$
L_f \frac{di_d}{dt} = v_{cmd,d} - v_d \cdot 0 - \omega L_f \cdot i_q
$$

$$
C_f \frac{dv_d}{dt} = i_d - i_{load,d}
$$

The $\omega L_f \cdot i_q$ is **cross-coupling** between the axes —
small at our operating point because $i_q \approx 0$. Within each
axis the dynamics are identical to a buck converter with $v_{cmd,x}$
as the "duty" times $V_{dc}/2$ and the LC + load as the filter:

$$
G_{vd}(s) = \frac{1/(L_f C_f)}{s^2 + s/(R C_f) + 1/(L_f C_f)}
$$

DC gain = 1 (the dq modulating signal directly equals the dq
output voltage at the filter). Natural frequency $\omega_n =
1/\sqrt{L_f C_f}$. Q-factor $= R\sqrt{C_f/L_f}$.

**This is the buck's plant.** Everything we learned about buck
control transfers.
"""))

    cells.append(code(r"""
plant = standalone_voltage_plant(p)
print(f"dq-frame voltage-loop plant: num = {plant.num}, den = {plant.den}")
print(f"  f_n = {p.f_filter_corner:.1f} Hz, Q = {p.Q_filter:.3f}")
print()
print("Note: Q is HIGH (no damping). The resonance peaks at +20 dB.")
print("In practice, production designs add series R in the inductor,")
print("a damping branch, or active damping via a current loop.")

f = np.logspace(0, 4, 1000)
w = 2*np.pi*f
_, mag, ph = signal.bode(plant, w=w)
fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax_mag.semilogx(f, mag, "C0", linewidth=2)
ax_mag.axvline(p.f_filter_corner, color="C3", linestyle=":", alpha=0.5,
               label=f"$f_n$ = {p.f_filter_corner:.0f} Hz")
ax_mag.axvline(p.f_sw/10, color="C2", linestyle=":", alpha=0.5,
               label=f"$f_{{sw}}/10$ = {p.f_sw/10/1000:.1f} kHz")
ax_mag.axhline(0, color="k", linestyle=":", alpha=0.3)
ax_ph.semilogx(f, ph, "C0", linewidth=2)
ax_mag.set_ylabel("Magnitude [dB]"); ax_ph.set_ylabel("Phase [deg]")
ax_ph.set_xlabel("Frequency [Hz]")
ax_mag.set_title("Stand-alone dq voltage-loop plant — buck-like, high Q")
ax_mag.legend(); plt.tight_layout(); plt.show()
"""))

    cells.append(md(r"""
## 2. PI compensator design

Strategy: simple PI placed well **below** the LC resonance to avoid
exciting it. The high Q means we can't push the bandwidth close to
$f_n$ without active damping.

Target $f_c = 200$ Hz (10× below $f_n$ = 1.6 kHz, well above $f_o$
= 60 Hz). Place the zero at $f_o = 60$ Hz so the integrator's
$-90°$ contribution at $f_c$ is partially compensated.

Note: we design the PI with `V_dc/2` scaling already in the
simulator's command interpretation (`v_cmd_x` directly becomes the
sin/cos modulating signal that gets fed to SVPWM).
"""))

    cells.append(code(r"""
# PI compensator: K · (1 + s·tau_z) / s
f_c = 200.0
f_z = 60.0  # zero at line frequency
omega_c = 2*np.pi*f_c
tau_z = 1.0/(2*np.pi*f_z)

# Plant magnitude at omega_c (we want loop gain = 1 there)
_, mag_at_fc, _ = signal.bode(plant, w=[omega_c])
plant_mag = 10**(mag_at_fc[0]/20)
# Compensator magnitude at omega_c: K · sqrt(1+(omega_c·tau_z)²) / omega_c
K = omega_c / (np.sqrt(1+(omega_c*tau_z)**2) * plant_mag)

Gc = signal.TransferFunction([K*tau_z, K], [1.0, 0.0])
print(f"PI compensator: K = {K:.4g}, tau_z = {tau_z*1e3:.3f} ms (zero at {f_z:.1f} Hz)")

# Loop gain
T_loop = signal.TransferFunction(np.polymul(Gc.num, plant.num),
                                   np.polymul(Gc.den, plant.den))
f = np.logspace(0, 4, 1500)
w = 2*np.pi*f
_, mag_T, ph_T = signal.bode(T_loop, w=w)
idx = np.argmin(np.abs(mag_T))
f_cross = f[idx]
pm = 180 + ph_T[idx]

fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax_mag.semilogx(f, mag_T, "C3", linewidth=2)
ax_mag.axhline(0, color="k", linestyle=":")
ax_mag.axvline(f_cross, color="g", linestyle=":", alpha=0.5,
               label=f"$f_c$ = {f_cross:.1f} Hz")
ax_mag.axvline(p.f_filter_corner, color="C2", linestyle=":", alpha=0.5,
               label=f"$f_n$ = {p.f_filter_corner:.0f} Hz (avoid)")
ax_ph.semilogx(f, ph_T, "C3", linewidth=2)
ax_ph.axhline(-180, color="k", linestyle=":")
ax_mag.set_ylabel("Magnitude [dB]"); ax_ph.set_ylabel("Phase [deg]")
ax_ph.set_xlabel("Frequency [Hz]")
ax_mag.set_title(f"Stand-alone loop: $f_c$ ≈ {f_cross:.0f} Hz, PM = {pm:.1f}°")
ax_mag.legend(); plt.tight_layout(); plt.show()
"""))

    cells.append(md(r"""
## 3. Discretize and simulate

Tustin at $T_s = 1/f_{sw} = 50$ µs. Two parallel PIs run in the
switched simulator — one for $v_d$, one for $v_q$.
"""))

    cells.append(code(r"""
T_s = p.T_sw
bd_raw, ad_raw, _ = signal.cont2discrete((Gc.num, Gc.den), dt=T_s, method='bilinear')
bd = np.asarray(bd_raw).flatten() / ad_raw[0]
ad = np.asarray(ad_raw) / ad_raw[0]
print(f"Discrete b = {bd}")
print(f"Discrete a = {ad}")

sim = simulate_closed_loop_standalone(p, bd, ad, n_cycles=8,
                                       samples_per_period=80, use_svpwm=True)
print(f"Simulated {len(sim['t'])} samples over {sim['t'][-1]*1000:.1f} ms")
"""))

    cells.append(code(r"""
mask = sim['t'] > 4.0/p.f_o
fs = 1.0/(sim['t'][1]-sim['t'][0])

fig, axs = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

axs[0].plot(sim['t']*1000, sim['v_Ca'], "C0", linewidth=1.0, label="$v_{Ca}$")
axs[0].plot(sim['t']*1000, sim['v_Cb'], "C1", linewidth=1.0, label="$v_{Cb}$")
axs[0].plot(sim['t']*1000, sim['v_Cc'], "C2", linewidth=1.0, label="$v_{Cc}$")
axs[0].set_ylabel("Phase voltage [V]")
axs[0].set_title(f"Closed-loop stand-alone VSI in dq @ V_d_ref = {p.V_o_LN_pk:.1f} V")
axs[0].legend(loc="upper right")

axs[1].plot(sim['t']*1000, sim['v_d'], "C2", linewidth=1.5, label="$v_d$")
axs[1].plot(sim['t']*1000, sim['v_d_ref'], "C3--", linewidth=1.0, label="$v_d^{ref}$")
axs[1].plot(sim['t']*1000, sim['v_q'], "C1", linewidth=1.5, label="$v_q$")
axs[1].plot(sim['t']*1000, sim['v_q_ref'], "C5--", linewidth=1.0, label="$v_q^{ref}$ = 0")
axs[1].set_ylabel("dq voltage [V]"); axs[1].legend(loc="upper right")

axs[2].plot(sim['t']*1000, sim['i_La'], "C4", linewidth=0.7)
axs[2].set_ylabel("Filter $i_{La}$ [A]")

# Compute the line-to-line voltage and its fundamental
v_LL_post = sim['v_Ca'] - sim['v_Cb']
axs[3].plot(sim['t']*1000, v_LL_post, "C3", linewidth=1.0)
axs[3].set_ylabel("$v_{LL,ab}$ [V]"); axs[3].set_xlabel("Time [ms]")

plt.tight_layout(); plt.show()

# Metrics
v_d_mean = sim['v_d'][mask].mean()
v_q_mean = sim['v_q'][mask].mean()
v_LL_fund = fundamental_rms(v_LL_post[mask], fs, p.f_o)
v_LL_thd = thd(v_LL_post[mask], fs, p.f_o, 30)
print()
print("Stand-alone closed-loop metrics:")
print(f"  V_d steady = {v_d_mean:.2f} V  (target {p.V_o_LN_pk:.2f} V, "
      f"error {(v_d_mean-p.V_o_LN_pk)/p.V_o_LN_pk*100:+.2f}%)")
print(f"  V_q steady = {v_q_mean:.3f} V  (target 0 V)")
print(f"  Line-to-line fundamental rms = {v_LL_fund:.2f} V  "
      f"(target {p.V_o_LL_rms:.2f} V)")
print(f"  Line-to-line THD             = {v_LL_thd*100:.2f} %")

if abs(v_d_mean - p.V_o_LN_pk) < 5.0 and abs(v_q_mean) < 5.0 and v_LL_thd < 0.10:
    print()
    print("✅  Stand-alone VSI control PROVEN: V_d and V_q regulated, "
          "clean output sinusoid.")
"""))

    cells.append(md(r"""
## 4. Discussion: the high-Q LC challenge

You may notice some ringing in the output, especially at startup.
That's the **LC resonance at 1591 Hz** being excited by transients
(switching events, warm-start). The Q ≈ 10.6 of the filter means
the resonance has a +20 dB peak.

Three standard fixes used in production:

1. **Series damping** — add a small resistor in series with the
   filter inductor. Effective but lossy.
2. **Damping branch** — RC across the cap (a series RC parallel to
   the load). Modest losses, robust.
3. **Active damping / cascade control** — inner current loop forces
   $i_L$, which "looks like" a virtual damping resistor to the cap.
   No extra hardware, but more complex firmware. This is what high-
   end UPS controllers actually do.

We use the simple PI here for pedagogy. The exercises below extend
it.

## 5. Summary

A 3-phase stand-alone VSI controlled in dq:

- **Plant in dq = buck plant.** All buck-control intuition transfers.
- **Two parallel PIs** (one for $v_d$, one for $v_q$) handle the
  two axes independently — the cross-coupling $\omega L_f i_q$ term
  is small and ignored in this first design.
- The bandwidth is constrained by the LC filter's high Q; production
  designs add damping or cascade control.

**Next**: `04_vsi_gridtie.ipynb` — connect to a stiff AC grid, add a
PLL to extract the grid angle, and inject active and reactive power
via dq current control.

**Suggested exercises**

1. Add $R_L = 1$ Ω to the filter inductor (`replace(p, R_L=1.0)`).
   Plot the new plant Bode and the new closed-loop response.
   How does Q change?
2. Implement cross-coupling decoupling: subtract $\omega L_f i_q$
   from the $d$-axis command and add $\omega L_f i_d$ to the $q$-axis
   command. Does it help?
3. Step $V_d^{ref}$ from 187 V to 100 V at $t$ = 50 ms. Plot the
   transient. Does it stay clean?
4. Push the load to imbalance (different $R$ in each phase). The dq
   compensator will struggle because positive- and negative-sequence
   components mix. Production designs add a parallel negative-
   sequence compensator.
"""))

    return cells


# ---------------------------------------------------------------------------
# Notebook 4 — Grid-tie
# ---------------------------------------------------------------------------


def build_gridtie_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 4 — Grid-Tie VSI: SRF-PLL + dq Current Control

> **Goal.** Connect the VSI to a stiff AC grid. Two new things:
> 1) a **PLL** locks onto the grid angle (we no longer generate
> $\theta$ internally), and 2) the controlled quantity changes from
> *voltage* to *current* — we inject a target $i_d$ (active current)
> and $i_q$ (reactive current) that map directly to **P** and **Q**.

**Prerequisites**

- All previous notebooks in this project, especially `01` (transforms)
- Boost PFC CCM notebook (`projects/converters/boost_pfc/03_boost_pfc_ccm.ipynb`)
  — the inner current loop architecture transfers verbatim.

**What you'll be able to do at the end**

1. Describe the **synchronous reference frame PLL** (SRF-PLL) — the
   standard grid-synchronization technique.
2. Derive the dq current-loop plant for a grid-connected inverter
   (a first-order $L_{grid}$ — no LC filter on this side, just the
   coupling inductor).
3. Design the **two PI compensators** (current loop in dq) and the
   **PLL PI** independently.
4. Explain **cross-coupling decoupling**: feed-forward the $\omega
   L_{grid} i_d$ and $\omega L_{grid} i_q$ terms to make the dq
   axes truly independent.
5. Run the full switched closed-loop sim with **P-Q injection** and
   verify that $i_d^{ref}$ produces the expected active power
   transfer.
"""))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from vsi_3phase_model import (
    VSI3PhaseParams,
    simulate_closed_loop_gridtie,
    clarke_transform, park_transform,
    operating_point_report,
)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

p = VSI3PhaseParams()
print(operating_point_report(p, mode="gridtie"))
"""))

    cells.append(md(r"""
## 1. Grid-tie architecture

A grid-tie inverter:

- **Source**: DC bus (typically from a PFC or PV array).
- **Sink**: stiff AC grid via a coupling inductor $L_{grid}$.
- **No LC filter** on the grid side (the grid is the cap) — just
  $L_{grid}$ to absorb switching ripple.
- **Output current** is what we control (not voltage). The grid sets
  the voltage.

```
   DC bus                              v_grid
     │                                    │
   ──┴── ── inverter ── L_grid ── ────────┤
                                          │
                                       (stiff)
```

Within the controller:

```
   v_grid_abc → Clarke → αβ → Park (θ_pll) → dq → drive v_qg → 0 (PLL)
                                                     │
                                                     ↓
                                                  θ_pll (angle)
                                                     │
   i_grid_abc → Clarke → αβ → Park (θ_pll) → dq ─┐
                                                  │
                                                  ↓
   i_d_ref ─┐                                   PI in dq
   i_q_ref ─┴── err ── PI ──→ v_d_cmd, v_q_cmd ─┘
                                  │
                                  ↓
                              inv. Park → αβ → inv. Clarke → SVPWM → gates
```

The PLL produces $\theta_{pll}$ that tracks the grid voltage angle;
the current loop runs in this frame.
"""))

    cells.append(md(r"""
## 2. SRF-PLL (synchronous reference frame PLL)

How it works:

1. Sample $v_{ga,b,c}$ and Clarke-transform to αβ.
2. Park-transform using the **current PLL angle estimate** $\theta_{pll}$
   to get $v_{dg}$ and $v_{qg}$.
3. If $\theta_{pll}$ lags the true angle, $v_{qg}$ becomes positive.
   If it leads, $v_{qg}$ becomes negative. So **drive $v_{qg} → 0$**
   to lock the phase.
4. PI compensator on $v_{qg}$ produces a frequency correction
   $\Delta\omega$. Add to the nominal grid frequency to get the
   instantaneous $\omega_{pll}$. Integrate to get $\theta_{pll}$.

Plant for the PLL: a pure integrator (from $\omega_{pll}$ to
$\theta_{pll}$). PI design is straightforward — fast loop (~50 Hz
crossover or so) for clean tracking.
"""))

    cells.append(code(r"""
# Design the PLL PI: input = -v_qg (with sign so positive err raises ω)
# Plant: 1/s integrator (theta) + the trig in Park is at unity DC gain near lock
# Target f_c = 50 Hz, zero ≈ 5 Hz
T_s = p.T_sw

# Effective plant gain at lock: ∂v_qg/∂Δω ≈ V_grid_pk · (Δθ ≈ Δω·t_settle)
V_grid_pk = p.V_grid_LL_rms * np.sqrt(2.0) / np.sqrt(3.0)
# Plant from omega_corr (rad/s) to v_qg (V) at lock ≈ V_grid_pk · (1/s)
# So G_pll(s) = V_grid_pk / s
# Compensator: K · (1 + s·tau_z) / s, target crossover 50 Hz
omega_c_pll = 2*np.pi*50.0
tau_z_pll = 1.0/(2*np.pi*5.0)
# At omega_c: |compensator| = K · sqrt(1+(omega_c·tau)²)/omega_c
# Loop gain = V_grid_pk/omega_c · compensator
# Want = 1: K = omega_c² / (V_grid_pk · sqrt(1+(omega_c·tau)²))
K_pll = omega_c_pll**2 / (V_grid_pk * np.sqrt(1+(omega_c_pll*tau_z_pll)**2))
print(f"PLL PI: K = {K_pll:.4g}, tau_z = {tau_z_pll*1e3:.2f} ms (zero at 5 Hz)")
Gc_pll = signal.TransferFunction([K_pll*tau_z_pll, K_pll], [1.0, 0.0])
bp_raw, ap_raw, _ = signal.cont2discrete((Gc_pll.num, Gc_pll.den), dt=T_s, method='bilinear')
b_pll = np.asarray(bp_raw).flatten()/ap_raw[0]
a_pll = np.asarray(ap_raw)/ap_raw[0]
print(f"  Discrete: b = {b_pll}, a = {a_pll}")
"""))

    cells.append(md(r"""
## 3. Current-loop design

Plant from $v_{d,cmd}$ to $i_d$, within the current-loop bandwidth
(grid is stiff, $v_d$ ≈ $V_{d,grid}$ const):

$$
L_{grid} \frac{d i_d}{dt} = v_{d,cmd} - v_{d,grid} - \omega L_{grid} i_q
$$

The $-v_{d,grid}$ and $-\omega L_{grid} i_q$ are **disturbances**
that we can feed-forward. After feed-forward, the current loop's
effective plant is a clean integrator:

$$
G_{id}(s) = \frac{1}{s L_{grid}}
$$

Same as the boost PFC's current loop! Use the same recipe: PI with
zero placed 3× below crossover at $f_c = 1$ kHz.
"""))

    cells.append(code(r"""
# Current-loop PI design (after feed-forward)
f_c_i = 1000.0  # 1 kHz current-loop bandwidth
omega_c_i = 2*np.pi*f_c_i
tau_z_i = 3.0 / omega_c_i  # zero at f_c/3

# Plant at omega_c: 1/(omega_c · L_grid)
plant_mag_i = 1.0 / (omega_c_i * p.L_grid)
K_i = omega_c_i / (np.sqrt(1+(omega_c_i*tau_z_i)**2) * plant_mag_i)
print(f"Current PI: K = {K_i:.4g}, tau_z = {tau_z_i*1e6:.2f} µs "
      f"(zero at {1/(2*np.pi*tau_z_i)/1000:.2f} kHz)")
Gc_i = signal.TransferFunction([K_i*tau_z_i, K_i], [1.0, 0.0])
bi_raw, ai_raw, _ = signal.cont2discrete((Gc_i.num, Gc_i.den), dt=T_s, method='bilinear')
bi = np.asarray(bi_raw).flatten()/ai_raw[0]
ai = np.asarray(ai_raw)/ai_raw[0]
"""))

    cells.append(md(r"""
## 4. P/Q injection via dq current references

In the grid frame with PLL locked so $v_{qg} = 0$ (i.e. $v_{d,grid}
= V_{d,pk}$):

$$
P = \frac{3}{2} V_{d,grid} \cdot i_d, \qquad
Q = -\frac{3}{2} V_{d,grid} \cdot i_q
$$

(The $3/2$ comes from the amplitude-invariant Clarke convention.)

To inject 500 W of active power into the grid at the nominal grid
voltage:
$$
I_{d,ref} = \frac{2 P_{ref}}{3 V_{d,grid}}
$$
"""))

    cells.append(code(r"""
P_ref = 500.0  # 500 W active injection
Q_ref = 0.0    # zero reactive (unity PF injection)
I_d_ref = 2 * P_ref / (3 * V_grid_pk)
I_q_ref = -2 * Q_ref / (3 * V_grid_pk)
print(f"For P = {P_ref:.0f} W, Q = {Q_ref:.0f} VAR:")
print(f"  I_d_ref = {I_d_ref:.4f} A   (peak active current)")
print(f"  I_q_ref = {I_q_ref:.4f} A   (peak reactive current)")
print(f"  I_pk    = {np.sqrt(I_d_ref**2 + I_q_ref**2):.4f} A")
"""))

    cells.append(md(r"""
## 5. Switched closed-loop simulation

Run the full grid-tie simulation: PLL + current loop + multiplier-free
power calculation. The current $i_a$ should be a clean sinusoid in
phase with $v_{ga}$ (unity PF injection).
"""))

    cells.append(code(r"""
sim = simulate_closed_loop_gridtie(p, bi, ai, b_pll, a_pll,
                                    I_d_ref=I_d_ref, I_q_ref=I_q_ref,
                                    n_cycles=10, samples_per_period=80,
                                    use_svpwm=True)
print(f"Simulated {len(sim['t'])} samples over {sim['t'][-1]*1000:.1f} ms")
"""))

    cells.append(code(r"""
mask = sim['t'] > 3.0/p.f_grid

fig, axs = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

axs[0].plot(sim['t']*1000, sim['v_ga'], "C0", linewidth=1.0, label="$v_{g,a}$")
ax_i = axs[0].twinx()
ax_i.plot(sim['t']*1000, sim['i_a'], "C3", linewidth=1.0, label="$i_a$")
axs[0].set_ylabel("Grid voltage [V]"); ax_i.set_ylabel("Grid current [A]", color="C3")
axs[0].set_title(f"Grid-tie VSI: P_ref = {P_ref:.0f} W (unity-PF injection)")
axs[0].legend(loc="upper left"); ax_i.legend(loc="upper right")

axs[1].plot(sim['t']*1000, sim['theta_pll'], "C2", linewidth=1.0, label="$\\theta_{pll}$")
axs[1].plot(sim['t']*1000, sim['theta_true'], "C5--", linewidth=1.0, label="$\\theta_{true}$")
axs[1].set_ylabel("Angle [rad]"); axs[1].legend(loc="lower right")

axs[2].plot(sim['t']*1000, sim['i_d'], "C2", linewidth=1.2, label="$i_d$")
axs[2].axhline(I_d_ref, color="C2", linestyle=":", alpha=0.5,
               label=f"$i_d^{{ref}}$ = {I_d_ref:.3f} A")
axs[2].plot(sim['t']*1000, sim['i_q'], "C1", linewidth=1.2, label="$i_q$")
axs[2].axhline(I_q_ref, color="C1", linestyle=":", alpha=0.5,
               label=f"$i_q^{{ref}}$ = {I_q_ref:.3f} A")
axs[2].set_ylabel("Current dq [A]"); axs[2].legend(loc="upper right", fontsize=8)

axs[3].plot(sim['t']*1000, sim['omega_pll']/(2*np.pi), "C4", linewidth=1.0)
axs[3].axhline(p.f_grid, color="k", linestyle=":", alpha=0.4,
               label=f"true f_grid = {p.f_grid:.1f} Hz")
axs[3].set_ylabel("PLL freq [Hz]"); axs[3].set_xlabel("Time [ms]")
axs[3].legend()

plt.tight_layout(); plt.show()
"""))

    cells.append(code(r"""
# Verify P/Q injection
i_d_ss = sim['i_d'][mask].mean()
i_q_ss = sim['i_q'][mask].mean()
P_actual = 1.5 * V_grid_pk * i_d_ss
Q_actual = -1.5 * V_grid_pk * i_q_ss
print("Steady-state P/Q injection:")
print(f"  i_d steady = {i_d_ss:.4f} A  (target {I_d_ref:.4f} A)")
print(f"  i_q steady = {i_q_ss:.4f} A  (target {I_q_ref:.4f} A)")
print(f"  P actual   = {P_actual:.2f} W   (target {P_ref:.2f} W, "
      f"error {(P_actual-P_ref)/P_ref*100:+.2f}%)")
print(f"  Q actual   = {Q_actual:.2f} VAR (target {Q_ref:.2f} VAR)")
print()

# Verify PLL locked
theta_err = sim['theta_pll'] - sim['theta_true']
# Unwrap to range [-pi, pi]
theta_err = np.mod(theta_err + np.pi, 2*np.pi) - np.pi
theta_err_ss = theta_err[mask].mean()
print(f"PLL lock check:")
print(f"  theta_pll - theta_true (steady state) = {np.rad2deg(theta_err_ss):.3f}°")
print(f"  omega_pll (steady state) = {sim['omega_pll'][mask].mean()/(2*np.pi):.4f} Hz "
      f"(target {p.f_grid:.4f} Hz)")

if abs(P_actual - P_ref) < 50 and abs(theta_err_ss) < np.deg2rad(2):
    print()
    print("✅  Grid-tie VSI PROVEN: PLL locked, P injected within ±10% of target.")
"""))

    cells.append(md(r"""
## 6. Summary

Grid-tie VSI combines two control loops with very different time
scales:

- **Inner current loop** (1 kHz crossover): forces $i_d$ and $i_q$
  to track references in the dq frame; plant is an integrator
  ($1/(sL_{grid})$) after grid-voltage feed-forward. Simple PI.
- **SRF-PLL** (50 Hz crossover): tracks grid angle by driving
  $v_{qg} → 0$. Simple PI on a virtual integrator plant.

Together they let you inject **arbitrary $P$ and $Q$** with just
two reference numbers ($i_d^{ref}, i_q^{ref}$). The math is
identical whether you're injecting from a PV array, a battery, or
absorbing into an EV charger.

**What we didn't cover** (production additions):

- **Anti-islanding** detection (required by grid codes)
- **LCL filter** instead of L (lower switching ripple, but adds a
  resonance to damp)
- **Negative-sequence rejection** for unbalanced grids
- **Weak-grid PLL stabilization** (the PLL itself can oscillate
  when the grid is "soft" — small $S_k$)
- **Fault ride-through** logic (stay connected during voltage sags)
- **MPPT integration** with a PV upstream stage

**Suggested exercises**

1. Step $I_d^{ref}$ from 0 to nominal at $t = 50$ ms. Plot the
   current-loop step response. Settling time?
2. Change $I_q^{ref}$ to inject 200 VAR of reactive power. Does
   the active-current loop see any cross-coupling?
3. Detune the PLL gain (× 0.1). Does it still lock? How long?
4. Add a 5 Hz frequency jitter to the grid (`omega_grid_true ·=
   1 + 0.01 sin(2π·5·t)`). Does the PLL track? Plot
   $\theta_{pll} - \theta_{true}$.
5. Implement cross-coupling decoupling in the controller — subtract
   $\omega L_{grid} i_q$ from $v_{d,cmd}$ and add $\omega L_{grid} i_d$
   to $v_{q,cmd}$. (It's already in `simulate_closed_loop_gridtie`.)
   Compare with and without.

## Library complete (for now)

| Project | I/O | Switches | Loops |
|---|---|---|---|
| 6 DC-DC | DC → DC | 1 or 2 | 1 voltage |
| boost PFC (2 modes) | AC → DC | 1 | 1 or 2 |
| **VSI 3-phase stand-alone** | **DC → AC** | **6** | **2 (d + q)** |
| **VSI 3-phase grid-tie** | **DC ↔ AC** | **6** | **3 (PLL + d + q current)** |

You've now seen the full power-electronics control hierarchy, from
single-switch buck up to multi-loop grid-synchronized inverters. The
reference-frame transforms are the most powerful idea — they let
3-phase AC control reuse single-axis DC control intuition.
"""))

    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    write_notebook(build_basics_notebook(),       HERE / "01_vsi_basics.ipynb")
    write_notebook(build_svpwm_notebook(),        HERE / "02_vsi_svpwm.ipynb")
    write_notebook(build_standalone_notebook(),   HERE / "03_vsi_standalone.ipynb")
    write_notebook(build_gridtie_notebook(),      HERE / "04_vsi_gridtie.ipynb")


if __name__ == "__main__":
    main()
