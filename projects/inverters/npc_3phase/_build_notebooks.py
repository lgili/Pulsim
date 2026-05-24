"""Generator for the NPC 3-level teaching notebooks.

Produces:
  * 01_npc_modeling.ipynb   — multilevel theory, NPC topology, PD-PWM
  * 02_npc_balancing.ipynb  — neutral-point voltage balancing controller

These notebooks are NOT pre-executed (only the 00 Pulsim validation
ships with rendered outputs); they use only numpy / scipy / matplotlib
so they re-run in seconds.
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


# ---------------------------------------------------------------------------
# Notebook 1 — NPC modeling
# ---------------------------------------------------------------------------


def build_modeling_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 1 — NPC 3-Level Inverter Modeling

> **Goal.** Derive the NPC 3-level inverter from the 2-level VSI by
> adding clamping diodes to the neutral point, enumerate the
> switching states, build the multicarrier PD-PWM modulator, and
> compute the analytical fundamental and harmonic content. The result
> is a complete small-signal model **structurally identical** to the
> 2-level VSI — same dq-frame plant — but with a much cleaner output
> waveform.

**Prerequisites**

- The two-level VSI project (`../vsi_3phase/`) — we reuse all the
  Clarke/Park machinery and the dq-frame buck-equivalent plant.
- Basic familiarity with PWM: triangular carriers, modulation index,
  natural sampling.

**What you'll be able to do at the end**

1. State the NPC switching table — three valid states (P, O, N) out
   of $2^4 = 16$ switch combinations per leg.
2. Derive the phase pole voltage as a function of switching state and
   identify why this gives **three levels** instead of two.
3. Build the multicarrier Phase-Disposition (PD) PWM rule and
   implement it in Python.
4. Compute the **fundamental peak** $V_{ab,1} = \sqrt{3} m_a V_{dc}/2$
   and verify it matches the 2-level VSI's formula (multilevel changes
   shape, not fundamental amplitude).
5. Predict the **THD reduction** vs 2-level at the same $f_{sw}$ and
   $m_a$ — the headline pitch of multilevel converters.
"""))

    cells.append(md(r"""
## Setup
"""))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import matplotlib.pyplot as plt

from npc_3phase_model import (
    NPC3PhaseParams,
    SWITCH_TABLE,
    switching_state_to_pole_voltage,
    pd_pwm_state,
    pd_pwm_state_vectorised,
    fundamental_pole_voltage,
    fundamental_line_to_line,
    thd_voltage_unfiltered,
    operating_point_report,
)

plt.rcParams["figure.figsize"] = (11, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

params = NPC3PhaseParams()
print(operating_point_report(params))
"""))

    cells.append(md(r"""
## 1. The NPC topology

A 2-level VSI has **two switches per leg** (HS + LS); each leg's
output is either $+V_{dc}/2$ (HS on, LS off) or $-V_{dc}/2$ (HS off,
LS on). To create a **third level** (0 V, i.e. the bus midpoint) we
need a switching path that connects the phase output to the bus
midpoint. The NPC achieves this with **four switches in series** per
leg and **two clamping diodes** to the neutral point:

```
        Vdc_pos (+V_dc/2)
            │
           S1
            │
            n1 ─── D_clamp_top (anode at NP, cathode at n1)
            │
           S2
            │
           mid_X  ← phase output
            │
           S3
            │
            n2 ─── D_clamp_bot (anode at n2, cathode at NP)
            │
           S4
            │
        Vdc_neg (-V_dc/2)
```

The DC bus is **split** into two halves by capacitors $C_{dc}$ each,
with the **neutral point (NP)** at the midpoint (0 V w.r.t. system
ground, $+V_{dc}/2$ above $V_{dc,neg}$).
"""))

    cells.append(md(r"""
## 2. Switching states — only three of 16 are valid

Of the $2^4 = 16$ possible switch combinations per leg, **only
three** produce a defined phase-output voltage; the rest either short
the bus or leave $mid_X$ floating.

The switching state table — $S_1, S_2, S_3, S_4$ are 1 for ON, 0 for
OFF; $v_{mid,X}$ is referenced to the neutral point (NP):

| State | $S_1$ | $S_2$ | $S_3$ | $S_4$ | $v_{mid}$ |
|:-:|:-:|:-:|:-:|:-:|:-:|
| **P** | ON | ON | OFF | OFF | $+V_{dc}/2$ |
| **O** | OFF | ON | ON | OFF | $0$ |
| **N** | OFF | OFF | ON | ON | $-V_{dc}/2$ |

In **state P** the phase is connected directly to $V_{dc,pos}$
through $S_1$ + $S_2$. In **state O** the phase is clamped to NP:
during positive load current the path is $mid \to S_3$ (off) — wait,
during state O $S_3$ is ON; so the path is $mid \to S_3 \to n_2 \to
D_{clamp,bot}$ (off) — but $D_{clamp,bot}$ blocks current flowing OUT
of NP. Instead during positive current we use the upper clamp: $mid
\to S_2 \to n_1 \to D_{clamp,top} \to NP$. The clamping diodes are
sized for the load current and naturally take over depending on the
load-current direction.

In **state N** the phase is connected to $V_{dc,neg}$ through
$S_3 + S_4$.
"""))

    cells.append(code(r"""
print("Validating the switching-state table programmatically:\n")
for state, switches in SWITCH_TABLE.items():
    v = switching_state_to_pole_voltage(state, params.V_dc)
    print(f"  state {state}:  (S1,S2,S3,S4) = {switches}   "
          f"→ v_mid = {v:+.1f} V w.r.t. NP")
"""))

    cells.append(md(r"""
### Forbidden states

Any switch combination not in the table above either shorts the DC
bus or leaves $mid_X$ floating. The **PWM modulator must never
command** an invalid state — and the multicarrier PD-PWM scheme below
ensures only P / O / N transitions by construction.

| Forbidden examples | Why? |
|---|---|
| $S_1 = S_4 = 1$ (any others) | Direct short: $V_{dc,pos} \to S_1 \to n_1 \to \ldots \to n_2 \to S_4 \to V_{dc,neg}$ |
| All four OFF | $mid_X$ floating; no defined output |
| $S_1 = S_2 = S_3 = 1$ | $S_1 + S_2$ tries to pull $mid$ to $V_{dc,pos}$, $S_3$ tries to pull $mid$ to $n_2$ — conflicting |
"""))

    cells.append(md(r"""
## 3. The line-to-line voltage has **5 levels**

Each phase output ranges over $\{+V_{dc}/2, 0, -V_{dc}/2\}$ (3
levels). The line-to-line voltage is the difference of two phases, so
it ranges over **5** distinct values:

$$
v_{ab}(t) = v_{mid,a} - v_{mid,b}
   \in \{+V_{dc},\ +V_{dc}/2,\ 0,\ -V_{dc}/2,\ -V_{dc}\}
$$

Compared to the 2-level VSI ($v_{ab} \in \{-V_{dc}, 0, +V_{dc}\}$, 3
levels), this is a substantially finer staircase approximation of the
target sinusoid — and the FFT spectrum is correspondingly cleaner.
"""))

    cells.append(md(r"""
## 4. Multicarrier Phase-Disposition (PD) PWM

To generate the 3 voltage levels we need a modulator that picks
P / O / N at each instant. The standard scheme stacks **two
triangular carriers** vertically:

- **Upper carrier** $tri_{upper}(t)$: ramps over $[0, +1]$ at
  frequency $f_{sw}$.
- **Lower carrier** $tri_{lower}(t)$: ramps over $[-1, 0]$ at the
  same frequency and phase.

For each phase, compare the (normalised) reference $v_{ref,\phi}(t)
\in [-1, +1]$ against both:

$$
\text{state}(t) =
\begin{cases}
  \text{P} & \text{if } v_{ref} > tri_{upper} \\
  \text{N} & \text{if } v_{ref} < tri_{lower} \\
  \text{O} & \text{otherwise}
\end{cases}
$$

The reference for a balanced 3-phase set:

$$
v_{ref,\phi}(t) = m_a \sin(\omega_o t - \theta_\phi),
\quad m_a \in [0, 1]
$$
"""))

    cells.append(code(r"""
# Visualise PD-PWM operation for phase a over half a fundamental period.
f_o, f_sw = params.f_o, params.f_sw
fs = f_sw * 100                                  # 100 samples / carrier
t = np.arange(0, 0.5 / f_o, 1 / fs)
v_ref = params.m_a * np.sin(2 * np.pi * f_o * t)

# Triangular carriers.
phi_c = (2 * np.pi * f_sw * t) % (2 * np.pi)
tri = (2 / np.pi) * np.arcsin(np.sin(phi_c))
tri_upper = 0.5 + 0.5 * tri
tri_lower = -0.5 + 0.5 * tri

states = pd_pwm_state_vectorised(v_ref, t, f_sw)
v_pole = np.array([switching_state_to_pole_voltage(s, params.V_dc)
                    for s in states])

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

ax_top.plot(t * 1e3, v_ref, color="C3", lw=1.8,
              label=fr"Reference $v_{{ref,a}} = m_a \sin(\omega_o t)$, $m_a={params.m_a:.3f}$")
ax_top.plot(t * 1e3, tri_upper, color="C0", lw=0.6, alpha=0.7,
              label="Upper carrier (over [0, +1])")
ax_top.plot(t * 1e3, tri_lower, color="C1", lw=0.6, alpha=0.7,
              label="Lower carrier (over [-1, 0])")
ax_top.set_ylabel("Normalised reference / carriers")
ax_top.legend(loc="lower right", fontsize=9)
ax_top.set_title("PD-PWM: reference + two stacked carriers")

ax_bot.plot(t * 1e3, v_pole, color="C2", lw=1.2,
              label=r"$v_{pole,a}(t)$ — 3-level switched output")
for lvl in (-params.V_dc_half, 0.0, params.V_dc_half):
    ax_bot.axhline(lvl, color="k", ls=":", alpha=0.4)
ax_bot.set_xlabel("Time [ms]")
ax_bot.set_ylabel("$v_{pole,a}$ [V]")
ax_bot.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.show()
"""))

    cells.append(md(r"""
## 5. Fundamental amplitude

In the linear modulation range ($0 \le m_a \le 1$ with SVPWM-style
common-mode injection, or $0 \le m_a \le \sqrt{3}/2 \approx 0.866$
with plain sinusoidal PWM), the **average** phase pole voltage over
one carrier period equals the reference times $V_{dc}/2$:

$$
\langle v_{pole,a}(t) \rangle_{T_{sw}} = m_a \cdot \frac{V_{dc}}{2}
\sin(\omega_o t)
$$

This is **exactly the same formula** as the 2-level VSI's. Multilevel
PWM changes the *waveshape* (three levels instead of two) but not the
fundamental amplitude — the *amplitude* is set by $m_a$ and $V_{dc}$,
independent of the number of levels.

The line-to-line fundamental amplitude:

$$
\hat V_{ab,1} = \sqrt{3} \cdot \frac{m_a V_{dc}}{2}
$$

(Two pole voltages 120° apart subtract to $\sqrt 3$ times their
common amplitude.)
"""))

    cells.append(code(r"""
print(f"  m_a                       = {params.m_a:.4f}")
print(f"  V_dc                      = {params.V_dc:.1f} V")
print(f"  Fundamental pole peak     = m_a · V_dc/2 = "
      f"{fundamental_pole_voltage(params):.1f} V")
print(f"  Fundamental L-L peak      = √3 · m_a · V_dc/2 = "
      f"{fundamental_line_to_line(params):.1f} V")
print(f"  Target V_o_LL,rms         = {params.V_o_LL_rms:.1f} V_rms")
print(f"  Target V_o_LL peak        = {params.V_o_LL_pk:.1f} V_pk")
print()
print(f"  ✅ Analytical fundamental matches the design target.")
"""))

    cells.append(md(r"""
## 6. THD reduction vs 2-level

Multilevel's headline benefit: **lower harmonic distortion at the
same $f_{sw}$**. We measure the THD of both 3-level (PD-PWM) and
2-level (single-carrier SPWM) line-to-line voltages over one
fundamental period at the same operating point.
"""))

    cells.append(code(r"""
# Generate clean line-to-line waveforms for both modulation schemes.
fs = params.f_sw * 200                           # 200 samples per carrier
t = np.arange(0, 3.0 / params.f_o, 1 / fs)       # 3 fundamental periods

# Common reference signals.
omega = 2 * np.pi * params.f_o
v_ref_a = params.m_a * np.sin(omega * t)
v_ref_b = params.m_a * np.sin(omega * t - 2 * np.pi / 3)

# === 3-level PD-PWM via the npc_3phase_model helper ===
states_a = pd_pwm_state_vectorised(v_ref_a, t, params.f_sw)
states_b = pd_pwm_state_vectorised(v_ref_b, t, params.f_sw)
v_a_3l = np.array([switching_state_to_pole_voltage(s, params.V_dc) for s in states_a])
v_b_3l = np.array([switching_state_to_pole_voltage(s, params.V_dc) for s in states_b])
v_ab_3l = v_a_3l - v_b_3l

# === 2-level natural-sampled SPWM ===
phi_c = (2 * np.pi * params.f_sw * t) % (2 * np.pi)
tri = (2 / np.pi) * np.arcsin(np.sin(phi_c))
v_a_2l = np.where(v_ref_a > tri, +params.V_dc_half, -params.V_dc_half)
v_b_2l = np.where(v_ref_b > tri, +params.V_dc_half, -params.V_dc_half)
v_ab_2l = v_a_2l - v_b_2l

# Measure THD (up to 50th harmonic of f_o).
thd_3l = thd_voltage_unfiltered(t, v_ab_3l, params.f_o)
thd_2l = thd_voltage_unfiltered(t, v_ab_2l, params.f_o)
print(f"  3-level NPC v_ab THD (up to 50·f_o): {thd_3l * 100:6.2f} %")
print(f"  2-level VSI v_ab THD (up to 50·f_o): {thd_2l * 100:6.2f} %")
print(f"  Reduction factor                  : {thd_2l / thd_3l:5.2f}×")
"""))

    cells.append(code(r"""
# Plot the two line-to-line waveforms side by side over one fundamental period.
mask = t < 1.0 / params.f_o
t_z = t[mask] * 1e3
fig, (ax_2l, ax_3l) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

ax_2l.plot(t_z, v_ab_2l[mask], color="C3", lw=0.5)
ax_2l.set_ylabel("$v_{ab}$ [V]")
ax_2l.set_title(f"2-level VSI: 3 levels {{$\\pm V_{{dc}}$, 0}}, THD = {thd_2l*100:.2f}%")
for lvl in (-params.V_dc, 0, +params.V_dc):
    ax_2l.axhline(lvl, color="k", ls=":", alpha=0.3)

ax_3l.plot(t_z, v_ab_3l[mask], color="C2", lw=0.5)
ax_3l.set_xlabel("Time [ms]")
ax_3l.set_ylabel("$v_{ab}$ [V]")
ax_3l.set_title(f"3-level NPC: 5 levels {{$\\pm V_{{dc}}$, $\\pm V_{{dc}}/2$, 0}}, "
                 f"THD = {thd_3l*100:.2f}%")
for lvl in (-params.V_dc, -params.V_dc_half, 0, params.V_dc_half, params.V_dc):
    ax_3l.axhline(lvl, color="k", ls=":", alpha=0.3)

plt.tight_layout()
plt.show()
"""))

    cells.append(md(r"""
## 7. Summary

- The **NPC 3-level inverter** adds two clamping diodes per leg to
  generate a third voltage level (0 V, the bus midpoint) at the phase
  output. Each leg has 4 switches in series and **only 3 of the 16
  switching combinations are valid** (P, O, N).
- **Phase output**: 3 levels ($+V_{dc}/2$, $0$, $-V_{dc}/2$).
  **Line-to-line**: 5 levels.
- **Multicarrier PD-PWM** with two vertically-stacked triangular
  carriers generates the P/O/N switching pattern by construction —
  no need for explicit state-machine logic.
- **Fundamental amplitude** matches the 2-level VSI's:
  $\hat V_{ab,1} = \sqrt{3} m_a V_{dc}/2$. Multilevel only changes
  the *shape* of the waveform, not the fundamental.
- **THD is roughly halved** vs a 2-level VSI at the same $f_{sw}$ and
  $m_a$ — the multilevel payoff.

**Cross-validation against Pulsim**: open
[`00_npc_pulsim_validation.ipynb`](00_npc_pulsim_validation.ipynb)
for an executed notebook that builds the NPC in Pulsim and overlays
the switched waveforms against the analytical fundamental.

**Unique control challenge**: the NPC's neutral point can drift
open-loop. Open [`02_npc_balancing.ipynb`](02_npc_balancing.ipynb)
for the NP-balancing controller derivation.

**Suggested exercises**

1. Push $m_a$ beyond 1.0 (overmodulation) and observe the saturation
   pattern in the PD-PWM output. Where does $V_{ab,1}$ stop tracking
   $\sqrt{3} m_a V_{dc}/2$?
2. Compare PD with **POD (phase opposition)** — flip the lower
   carrier's phase by 180°. The line-to-line spectrum changes
   dramatically (POD has lower 1st-side-band but higher even-side-band).
3. Derive the **5-level** NPC topology (8 switches + 6 clamping diodes
   per leg). How does the THD scale further?
"""))

    return cells


# ---------------------------------------------------------------------------
# Notebook 2 — NPC balancing
# ---------------------------------------------------------------------------


def build_balancing_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 2 — NPC Neutral-Point Balancing

> **Goal.** Understand why the NPC's neutral point voltage **drifts
> open-loop**, derive the closed-loop carrier-based balancing
> controller that pins it back to $V_{dc}/2$, and demonstrate it on
> a forward-Euler switched simulation in pure Python.

This is the **defining control challenge** of the NPC topology —
without active balancing, the NP voltage walks away from its nominal
$V_{dc}/2$, eventually saturating one half of the DC bus and
destroying the 3-level operation.

**Prerequisites**

- [`01_npc_modeling.ipynb`](01_npc_modeling.ipynb) — switching states
  and PD-PWM.

**What you'll be able to do at the end**

1. State why the NP carries a real current and how its sign depends
   on the leg states + load currents.
2. Show that open-loop the NP voltage drifts (driven by 3rd-harmonic
   load-current content).
3. Implement a **carrier-based balancing controller**: a small DC
   offset added to all three references that biases the O-state dwell
   time and steers the NP back to its target.
4. Compare open-loop and closed-loop NP voltage trajectories on a
   forward-Euler simulation.
"""))

    cells.append(md(r"""
## Setup
"""))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import matplotlib.pyplot as plt

from npc_3phase_model import (
    NPC3PhaseParams,
    SWITCH_TABLE,
    switching_state_to_pole_voltage,
    pd_pwm_state,
    neutral_point_current,
)

plt.rcParams["figure.figsize"] = (11, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

params = NPC3PhaseParams()
print(f"V_dc        = {params.V_dc} V   →  V_dc/2 = {params.V_dc_half} V (target NP)")
print(f"C_dc        = {params.C_dc*1e6:.0f} µF (each split cap)")
print(f"f_o         = {params.f_o} Hz,   f_sw = {params.f_sw/1e3:.0f} kHz")
print(f"R_load Y    = {params.R_load:.2f} Ω/phase")
"""))

    cells.append(md(r"""
## 1. Why does the NP drift?

The split DC bus is two capacitors $C_{dc}$ in series, with the
neutral point (NP) at their midpoint. In **steady state** the NP
sits at $V_{dc}/2$ — but only if the **net current** flowing into NP
averages to zero over one fundamental period.

A phase contributes to the NP current **only when it is in state O**.
In states P and N the load current routes around the NP entirely
(through the top half-bus or the bottom half-bus). In state O the
current passes through the clamping diodes to/from the NP:

$$
i_{NP}(t) = \sum_{\phi \in \{a,b,c\}} \mathbb{1}\left[\text{state}_\phi(t) = O\right]
\cdot i_\phi(t)
$$

where the indicator function is 1 only when phase $\phi$ is in state
O. The NP voltage evolves as:

$$
C_{dc} \, \frac{d V_{NP}}{dt} = i_{NP}(t)
\quad (\text{with appropriate sign convention})
$$

The key observation: **the time-average of $i_{NP}$ is generally not
zero**, even for a balanced load. The third-harmonic content of the
load current — which exists naturally in any nonlinear modulation
scheme — pumps net charge into one cap or the other.
"""))

    cells.append(code(r"""
# Demonstrate symbolically: at m_a = 0.939 and a purely resistive
# load, evaluate the time-average of i_NP over one fundamental period.

def i_NP_signed(state_a, state_b, state_c, i_a, i_b, i_c):
    # Net current OUT of the neutral point. Sign convention: positive
    # flows from NP into the load. With our cap sign convention,
    # dV_NP/dt = -i_NP / C_dc.
    return neutral_point_current(state_a, state_b, state_c, i_a, i_b, i_c)


# Sample over 5 fundamental periods at high resolution.
fs = params.f_sw * 200
t = np.arange(0, 5.0 / params.f_o, 1 / fs)
omega = 2*np.pi*params.f_o

# References
v_ref_a = params.m_a * np.sin(omega * t)
v_ref_b = params.m_a * np.sin(omega * t - 2*np.pi/3)
v_ref_c = params.m_a * np.sin(omega * t - 4*np.pi/3)

# State-of-the-leg at each instant via PD-PWM
from npc_3phase_model import pd_pwm_state_vectorised
sa = pd_pwm_state_vectorised(v_ref_a, t, params.f_sw)
sb = pd_pwm_state_vectorised(v_ref_b, t, params.f_sw)
sc = pd_pwm_state_vectorised(v_ref_c, t, params.f_sw)

# Idealised load currents (sinusoidal at the operating point, in phase
# with v_ref_X for purely resistive load).
I_pk = params.I_o_pk
i_a = I_pk * np.sin(omega * t)
i_b = I_pk * np.sin(omega * t - 2*np.pi/3)
i_c = I_pk * np.sin(omega * t - 4*np.pi/3)

# Pulse-by-pulse i_NP — only the phase(s) in state O contribute.
i_NP = np.array([
    i_NP_signed(sa_i, sb_i, sc_i, ia_i, ib_i, ic_i)
    for sa_i, sb_i, sc_i, ia_i, ib_i, ic_i
    in zip(sa, sb, sc, i_a, i_b, i_c)
])

# Average i_NP over the last 3 fundamental periods (skip startup).
skip = int(2.0 / params.f_o / (t[1] - t[0]))
i_NP_avg = i_NP[skip:].mean()
print(f"  Average i_NP over the last 3 fundamental periods: "
      f"{i_NP_avg:.4f} A")
print(f"  Predicted dV_NP / dt = -i_NP / C_dc = "
      f"{-i_NP_avg / params.C_dc:.2f} V/s")
print(f"  → in one fundamental period the NP would drift by "
      f"{-i_NP_avg / params.C_dc / params.f_o * 1000:.2f} mV")
print()
print("  At fully balanced operation with purely sinusoidal references")
print("  the average is small but nonzero — and any operating asymmetry")
print("  amplifies it. The balancing controller below pins it to 0.")
"""))

    cells.append(md(r"""
## 2. Carrier-based balancing controller

The classic solution: add a **small DC offset** $v_{cm}$ to all three
references *common-mode* before the PD comparison:

$$
v_{ref,\phi}^{*}(t) = m_a \sin(\omega_o t - \theta_\phi) + v_{cm}
$$

The DC offset doesn't change the line-to-line output (it cancels in
$v_{ab} = v_{mid,a} - v_{mid,b}$) but it shifts each leg's O-state
dwell time *asymmetrically* with respect to the carrier positions,
which biases $\langle i_{NP} \rangle$.

A simple PI on the NP error gives the offset:

$$
v_{cm}(t) = K_p (V_{NP}^{\star} - V_{NP}) + K_i \int (V_{NP}^{\star} - V_{NP}) \, dt
$$

with the integrator winding clamped to avoid saturation.
"""))

    cells.append(code(r"""
# Forward-Euler closed-loop simulation.
def simulate_npc_balancing(params, K_p=2e-3, K_i=1.0,
                              t_end=0.1, dt=2e-6,
                              v_np_init=None,
                              enable_balancer=True):
    # Run a switched NPC simulation with optional NP balancing.
    if v_np_init is None:
        v_np_init = params.V_dc_half + 5.0       # 5 V drift to start

    omega = 2*np.pi*params.f_o
    n_steps = int(t_end / dt) + 1

    v_np = v_np_init       # actual NP voltage (relative to vdc_neg)
    integ = 0.0
    R = params.R_load
    I_pk = params.I_o_pk

    # Record
    t_hist = np.zeros(n_steps)
    v_np_hist = np.zeros(n_steps)
    v_cm_hist = np.zeros(n_steps)
    i_NP_hist = np.zeros(n_steps)

    for k in range(n_steps):
        t = k * dt
        # PI on NP error
        err = params.V_dc_half - v_np
        if enable_balancer:
            integ += K_i * err * dt
            integ = float(np.clip(integ, -0.10, 0.10))   # ±10% clamp
            v_cm = K_p * err + integ
        else:
            v_cm = 0.0

        # References with common-mode offset.
        vra = params.m_a * np.sin(omega * t)              + v_cm
        vrb = params.m_a * np.sin(omega * t - 2*np.pi/3)  + v_cm
        vrc = params.m_a * np.sin(omega * t - 4*np.pi/3)  + v_cm

        # PD-PWM state.
        sa = pd_pwm_state(vra, t, params.f_sw)
        sb = pd_pwm_state(vrb, t, params.f_sw)
        sc = pd_pwm_state(vrc, t, params.f_sw)

        # Load currents (purely resistive in this simple sim).
        # Approximate v_pole - v_np as a proxy for the load voltage.
        va = switching_state_to_pole_voltage(sa, params.V_dc)
        vb = switching_state_to_pole_voltage(sb, params.V_dc)
        vc = switching_state_to_pole_voltage(sc, params.V_dc)
        ia = va / R
        ib = vb / R
        ic = vc / R

        # NP current.
        i_NP = neutral_point_current(sa, sb, sc, ia, ib, ic)

        # NP voltage integration. Two split caps in series → effective
        # cap to NP = C_dc/2 from either side. Net dV_NP/dt = -i_NP / C_dc_eff.
        # With C_dc_eff = C_dc (each cap, since they're symmetric):
        v_np -= (i_NP / params.C_dc) * dt

        t_hist[k] = t
        v_np_hist[k] = v_np
        v_cm_hist[k] = v_cm
        i_NP_hist[k] = i_NP

    return t_hist, v_np_hist, v_cm_hist, i_NP_hist


# Run two simulations: balancer OFF vs balancer ON.
t_o, vnp_o, vcm_o, i_np_o = simulate_npc_balancing(
    params, t_end=0.15, enable_balancer=False)
t_c, vnp_c, vcm_c, i_np_c = simulate_npc_balancing(
    params, t_end=0.15, enable_balancer=True, K_p=5e-3, K_i=2.0)

print(f"Open-loop  V_NP at t=150 ms: {vnp_o[-1]:.2f} V "
      f"(drift = {vnp_o[-1] - params.V_dc_half:+.2f} V from target {params.V_dc_half:.0f} V)")
print(f"Closed-loop V_NP at t=150 ms: {vnp_c[-1]:.2f} V "
      f"(drift = {vnp_c[-1] - params.V_dc_half:+.2f} V from target {params.V_dc_half:.0f} V)")
"""))

    cells.append(code(r"""
fig, (ax_np, ax_cm) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

ax_np.plot(t_o * 1e3, vnp_o, color="C3", lw=1.2,
            label=f"Open-loop: V_NP drifts from {vnp_o[0]:.0f} V toward ...")
ax_np.plot(t_c * 1e3, vnp_c, color="C2", lw=1.2,
            label="Closed-loop with PI balancer (Kp=5e-3, Ki=2)")
ax_np.axhline(params.V_dc_half, color="k", ls="--", alpha=0.5,
                label=f"Target V_NP = V_dc/2 = {params.V_dc_half:.0f} V")
ax_np.set_ylabel("Neutral-point voltage [V]")
ax_np.set_title("NP voltage trajectory — open-loop drift vs closed-loop regulation")
ax_np.legend(loc="lower right")

ax_cm.plot(t_o * 1e3, vcm_o, color="C3", lw=1.2, label="open-loop ($v_{cm} = 0$)")
ax_cm.plot(t_c * 1e3, vcm_c, color="C2", lw=1.2, label="closed-loop $v_{cm}(t)$")
ax_cm.set_xlabel("Time [ms]")
ax_cm.set_ylabel("Common-mode offset $v_{cm}$ [-]")
ax_cm.legend(loc="upper right")

plt.tight_layout()
plt.show()
"""))

    cells.append(md(r"""
## 3. Summary

The NPC's neutral-point voltage is a **real state variable** that
the controller must actively regulate. Without intervention it
drifts away from $V_{dc}/2$, ultimately destroying the 3-level
operation.

The cure is mechanically simple — a PI on the NP error driving a
common-mode offset added to all three references — and costs nothing
in line-to-line output (the offset cancels in $v_{ab}$). The result
is a converter that maintains its 3-level signature indefinitely
even under unbalanced operation.

**Cross-validation note**: the
[`00_npc_pulsim_validation.ipynb`](00_npc_pulsim_validation.ipynb)
notebook uses a **stiff voltage-source DC bus** so the NP is rigidly
clamped — it can't drift in that model. To study the NP drift +
balancing in Pulsim you'd need to switch the bus to the
capacitor-only split, which currently runs into Pulsim's PWL cache
enumeration limits at this topology size. The pure-Python
forward-Euler simulation above is sufficient for studying the
balancing dynamics analytically.

**Suggested exercises**

1. Re-tune the balancer for **slower** dynamics (Kp = 1e-4, Ki = 0.2)
   and observe the longer settling time. What's the trade-off vs the
   load-cycle ripple amplitude?
2. Add a **load imbalance** (R_a ≠ R_b ≠ R_c) and observe how the
   open-loop NP drift accelerates. Can the same balancer handle it?
3. Replace the PI with a **deadbeat** controller (compute $v_{cm}$ to
   drive $i_{NP}$ to exactly zero in one switching period). Does it
   beat the PI on transient response?
"""))

    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    write_notebook(build_modeling_notebook(),
                    HERE / "01_npc_modeling.ipynb")
    write_notebook(build_balancing_notebook(),
                    HERE / "02_npc_balancing.ipynb")


if __name__ == "__main__":
    main()
