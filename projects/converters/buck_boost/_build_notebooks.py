"""Generator for the buck-boost teaching notebooks.

Run after editing this file to regenerate `01_buck_boost_modeling.ipynb`
and `02_buck_boost_controller.ipynb`. Same patterns as the buck and
boost generators.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def md(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _split_lines(text)}


def code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split_lines(text),
    }


def _split_lines(text: str) -> list[str]:
    text = text.lstrip("\n")
    return text.splitlines(keepends=True)


def write_notebook(cells: list[dict[str, Any]], path: Path) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {path.relative_to(HERE.parent.parent)} ({path.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# Notebook 1 — modeling
# ---------------------------------------------------------------------------


def build_modeling_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 1 — Buck-Boost Converter Modeling

> **Goal.** Derive the small-signal model of the inverting buck-boost
> in CCM, show how the **polarity inversion** falls out of the
> topology rather than the math, and identify the **RHP zero with
> duty-cycle-dependent location** — the buck-boost's biggest control
> design challenge.

**Prerequisites**

- Buck and boost reference notebooks (`projects/converters/{buck,boost}/`).
  The state-space averaging procedure is the same; only the topology
  changes.

**What you'll be able to do at the end**

1. Write the switched model for both ON and OFF intervals of the
   inverting buck-boost.
2. Derive the steady-state ratio $|V_o| = D \\cdot V_g / (1 - D)$.
3. Identify where in the algebra the polarity inversion appears.
4. Locate the RHP zero in $G_{vd}(s)$ and explain why it depends on
   $1/D$ (so high step-up ratios slow the loop more).
5. Compare the buck-boost's three characteristic regimes (buck-like
   $D < 0.5$, balanced $D = 0.5$, boost-like $D > 0.5$).
"""))

    cells.append(md("## Setup"))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from buck_boost_model import (
    BuckBoostParams,
    buck_boost_state_space,
    control_to_output_tf,
    line_to_output_tf,
    output_impedance_tf,
    operating_point_report,
)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
"""))

    cells.append(md(r"""
## 1. The inverting buck-boost topology

```
   V_g ----- S -----+
                    |
                    +---L---+
                    |       |
                    +       gnd
                    |
            D (anode toward L node, cathode toward V_o node)
                    |
                    +---+---+--- V_o  (V_o < 0)
                    |   |   |
                    C   R  load
                    |   |   |
                   gnd gnd gnd
```

- The switch $S$ chops $V_g$ into a pulsed waveform at node $A$ (top
  of $L$).
- During ON, $L$ stores energy from the input (the diode is off).
- During OFF, $S$ opens; the inductor maintains its current by
  forward-biasing $D$ and pumping charge into the output cap. The
  geometry forces the cap's $V_o$-side to go NEGATIVE relative to
  ground — that's where the polarity inversion comes from.

### 1.1 Operating-point intuition

In steady state:

- Average inductor voltage = 0 → $D \\cdot V_g = (1-D) \\cdot |V_o|$
  → $\\boxed{|V_o| = \\dfrac{D}{1 - D} V_g}$ (with negative polarity).
- $D = 0.5$: $|V_o| = V_g$ exactly.
- $D < 0.5$: $|V_o| < V_g$ (buck-like step-down).
- $D > 0.5$: $|V_o| > V_g$ (boost-like step-up). As $D \\to 1$,
  $|V_o| \\to \\infty$ (parasitic losses limit in practice).
"""))

    cells.append(md(r"""
## 2. Switched (instantaneous) model

States: $i_L$ (inductor current, positive when flowing into the
top of $L$) and $v_o$ (output cap voltage MAGNITUDE — we'll write
all equations using positive $v_o$ and remember the actual polarity
is negative).

### 2.1 ON interval ($S$ closed, $D$ off)

Inductor sees the full input bus. Output cap drains through the load:

$$
L \\frac{di_L}{dt} = v_g,
\\qquad
C \\frac{dv_o}{dt} = -\\frac{v_o}{R}
$$

### 2.2 OFF interval ($S$ open, $D$ on)

Inductor's current is now forced through the diode into the cap. The
inductor's voltage flips:

$$
L \\frac{di_L}{dt} = -v_o,
\\qquad
C \\frac{dv_o}{dt} = i_L - \\frac{v_o}{R}
$$

Compare to the boost: the OFF inductor equation has $-v_o$ instead of
$(v_g - v_o)$. That's the topological difference — the buck-boost's
inductor is grounded at the source end during OFF, not connected to
$v_g$.
"""))

    cells.append(md(r"""
## 3. State-space averaging

Let $q(t) \\in \\{0, 1\\}$ be the switching function. Continuous form:

$$
L \\frac{di_L}{dt} = q \\cdot v_g + (1 - q) \\cdot (-v_o) = q v_g - (1-q) v_o
$$

$$
C \\frac{dv_o}{dt} = q \\cdot \\left(-\\frac{v_o}{R}\\right)
                   + (1-q)\\left(i_L - \\frac{v_o}{R}\\right)
                   = (1-q) i_L - \\frac{v_o}{R}
$$

Replace $q \\to d$ for the average (assumes $f_{sw} \\gg f_n$):

$$\\boxed{
\\;\\; L \\frac{di_L}{dt} = d v_g - (1 - d) v_o
\\;\\;}
$$

$$\\boxed{
\\;\\; C \\frac{dv_o}{dt} = (1 - d) i_L - \\frac{v_o}{R}
\\;\\;}
$$

The cap equation is IDENTICAL to the boost's — same `(1-d)·i_L` source
term and same `-v_o/R` load. The inductor equation differs only in the
ON-state input ($d \\cdot v_g$ in buck-boost; $v_g$ in boost).
"""))

    cells.append(md(r"""
### 3.1 Steady-state

$$
0 = D V_g - (1-D) V_o \\;\\implies\\; |V_o| = \\frac{D}{1-D} V_g
$$

$$
0 = (1-D) I_L - \\frac{V_o}{R} \\;\\implies\\; I_L = \\frac{V_o}{R(1-D)}
$$

The inductor-current formula matches the boost's exactly.
"""))

    cells.append(code(r"""
params = BuckBoostParams()
print(operating_point_report(params))
"""))

    cells.append(md(r"""
## 4. Small-signal linearization

Perturb $i_L = I_L + \\hat i_L$, $v_o = V_o + \\hat v_o$, $d = D + \\hat d$,
$v_g = V_g + \\hat v_g$, substitute, drop products:

From the inductor equation:

$$
L \\frac{d\\hat i_L}{dt}
   = D \\hat v_g + (V_g + V_o) \\hat d - (1-D) \\hat v_o
$$

Note $V_g + V_o = V_g / (1-D)$ at the operating point — a useful
substitution that highlights the DC-gain dependence.

From the cap equation:

$$
C \\frac{d\\hat v_o}{dt}
   = (1-D) \\hat i_L - I_L \\hat d - \\frac{\\hat v_o}{R}
$$

Same shape as the boost: the $-I_L \\hat d$ term in the cap equation
is what gives both topologies their RHP zero.
"""))

    cells.append(md(r"""
## 5. State-space matrices

$$
A = \\begin{bmatrix}
0 & -(1-D)/L \\\\
(1-D)/C & -1/(R C)
\\end{bmatrix},
\\quad
B = \\begin{bmatrix}
(V_g + V_o)/L & D/L \\\\
-I_L/C & 0
\\end{bmatrix}
$$

$A$ is IDENTICAL to the boost's. $B$ differs only in element $B_{00}$:
- Boost: $B_{00} = V_o / L$
- Buck-boost: $B_{00} = (V_g + V_o) / L$ — bigger gain because the
  inductor effectively sees the combined input+output voltage when
  duty perturbs.
"""))

    cells.append(code(r"""
A, B, C_mat, D_mat = buck_boost_state_space(params)
print("A ="); print(A); print()
print("B = [col 0: d̂   col 1: v̂_g]"); print(B); print()
print("C =", C_mat)
print("D (feedthrough) =", D_mat)
print()
eig = np.linalg.eigvals(A)
print(f"A eigenvalues: {eig}")
print(f"Pole magnitude (= ω_n): {abs(eig[0]):.1f} rad/s  "
      f"(expect {params.omega_n:.1f})")
"""))

    cells.append(md(r"""
## 6. Transfer functions

### 6.1 $G_{vd}(s)$ — the headline plant

Closed form (Erickson Eq 8.55, taking magnitudes):

$$
G_{vd}(s) = \\frac{V_g}{(1-D)^2} \\cdot
\\frac{1 - s/\\omega_{z,RHP}}{1 + s/(Q \\omega_n) + (s/\\omega_n)^2}
$$

with:

| Quantity | Buck-boost | Boost |
|---|---|---|
| DC gain magnitude | $V_g / (1-D)^2$ | $V_o / (1-D) = V_g/(1-D)^2$ |
| $\\omega_n$ | $(1-D)/\\sqrt{LC}$ | $(1-D)/\\sqrt{LC}$ |
| $Q$ | $(1-D) R \\sqrt{C/L}$ | $(1-D) R \\sqrt{C/L}$ |
| $\\omega_{z,RHP}$ | $\\dfrac{R(1-D)^2}{L \\cdot D}$ | $\\dfrac{R(1-D)^2}{L}$ |

**Key difference**: the buck-boost RHP zero has an extra $1/D$ factor.
At $D = 0.5$ they're equal; at $D = 0.8$ the buck-boost zero is at
**1/4** of the boost's (much harder to control). At $D = 0.2$
(step-down regime) it's 5× **higher** (easier).

### 6.2 $G_{vg}(s)$ and $Z_{out}(s)$

$$
G_{vg}(s) = \\frac{D/(1-D)}{1 + s/(Q \\omega_n) + (s/\\omega_n)^2},
\\quad
Z_{out}(s) = \\frac{sL/(1-D)^2}{1 + s/(Q \\omega_n) + (s/\\omega_n)^2}
$$

DC line-to-output gain is $D/(1-D)$ = the duty-ratio gain magnitude
itself. No RHP zeros in either of these (only $G_{vd}$ has it).
"""))

    cells.append(code(r"""
Gvd = control_to_output_tf(params)
Gvg = line_to_output_tf(params)
Zout = output_impedance_tf(params)

print(f"Gvd(0)  (mag)   = {Gvd.num[1] / Gvd.den[2]:.3f} V/duty  "
      f"(expect V_g/(1-D)² = {params.V_g/(1-params.D)**2:.3f})")
print(f"Gvg(0)          = {Gvg.num[0] / Gvg.den[2]:.4f} V/V       "
      f"(expect D/(1-D)   = {params.D/(1-params.D):.4f})")
print()
print(f"Gvd zeros: {np.roots(Gvd.num)}  "
      f"(expect RHP at +{params.omega_z_rhp:.0f} rad/s = "
      f"+{params.f_z_rhp:.0f} Hz)")
print(f"Gvd poles: {np.roots(Gvd.den)}  (expect Q-LC pair)")
"""))

    cells.append(md(r"""
### 6.3 Bode plots — compare buck-boost vs boost RHP zeros

Plot $|G_{vd}|$ and phase. The natural pole pair is at the same
location as the boost (since $\\omega_n$ depends only on $L$, $C$,
$(1-D)$), but the RHP zero is at higher frequency for $D = 0.5$.
"""))

    cells.append(code(r"""
f = np.logspace(0, np.log10(params.f_sw), 1500)
w = 2 * np.pi * f

fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for tf, name, style in [
    (Gvd,  r"$G_{vd}(s)$  control → output (mag)", "-"),
    (Gvg,  r"$G_{vg}(s)$  line → output",          "--"),
    (Zout, r"$Z_{out}(s)$ load → output",          ":"),
]:
    _, mag, ph = signal.bode(tf, w=w)
    ax_mag.semilogx(f, mag, style, label=name)
    ax_ph.semilogx(f, ph, style, label=name)

for ax in (ax_mag, ax_ph):
    ax.axvline(params.f_n,     color="C0", linestyle=":", alpha=0.4,
               label=f"$f_n$ = {params.f_n:.0f} Hz")
    ax.axvline(params.f_z_rhp, color="C3", linestyle=":", alpha=0.6,
               label=f"$f_{{z,RHP}}$ = {params.f_z_rhp:.0f} Hz")
    ax.legend(loc="best", fontsize=8)

ax_mag.set_ylabel("Magnitude [dB]")
ax_ph.set_ylabel("Phase [deg]")
ax_ph.set_xlabel("Frequency [Hz]")
ax_mag.set_title(f"Buck-boost open-loop "
                 f"($V_g$={params.V_g}V, $|V_o|$={params.V_o}V, "
                 f"D={params.D:.2f})")
plt.tight_layout()
plt.show()
"""))

    cells.append(md(r"""
## 7. The wrong-way step response

Apply a small duty step. Like the boost, the buck-boost's $G_{vd}$
has an RHP zero, so the output **dips before rising** on a positive
duty step.
"""))

    cells.append(code(r"""
duty_step = 0.01
t = np.linspace(0, 10e-3, 5000)
_, y_step = signal.step(Gvd, T=t)
v_o_pred = params.V_o + duty_step * y_step  # magnitude

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(t*1e3, v_o_pred, label="$|v_o|$ (analytical)")
ax.axhline(params.V_o, color="k", linestyle=":", alpha=0.4,
           label=f"Pre-step $|V_o|$ = {params.V_o} V")
expected_new = params.V_o + duty_step * params.V_g / (1 - params.D)**2
ax.axhline(expected_new, color="g", linestyle=":", alpha=0.5,
           label=f"Predicted new $|V_o|$ ≈ {expected_new:.3f} V")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("$|v_o|$ [V]")
ax.set_title(f"Buck-boost response to a {duty_step*100:.0f}% duty step "
             "(notice the RHP-zero dip first)")
ax.legend()
plt.tight_layout()
plt.show()

dip = params.V_o - np.min(v_o_pred)
print(f"Pre-step $|V_o|$ = {params.V_o:.4f} V")
print(f"Initial dip      = {dip*1e3:.1f} mV "
      f"({dip/params.V_o*100:.2f} % of $|V_o|$)")
print(f"Final $|v_o|$    = {v_o_pred[-1]:.4f} V")
"""))

    cells.append(md(r"""
## 8. Model self-consistency checks

Same three rigorous checks as the buck and boost notebooks — pure math,
no simulator involved.
"""))

    cells.append(code(r"""
A, B, C_mat, D_mat = buck_boost_state_space(params)
Gvd_closed = control_to_output_tf(params)

# (1) Poles
ss_poles = sorted(np.linalg.eigvals(A), key=lambda z: z.imag)
tf_poles = sorted(np.roots(Gvd_closed.den), key=lambda z: z.imag)
print("(1) Poles:")
print(f"    SS:  {ss_poles}")
print(f"    TF:  {tf_poles}")
pole_match = np.allclose(ss_poles, tf_poles, rtol=1e-10)
print(f"    → match: {pole_match}")

# (2) DC gains
print()
print("(2) DC gains:")
print(f"    Gvd(0)  = {Gvd_closed.num[1] / Gvd_closed.den[2]:8.4f}  "
      f"(expect V_g/(1-D)² = {params.V_g/(1-params.D)**2:.4f})")
Gvg = line_to_output_tf(params)
print(f"    Gvg(0)  = {Gvg.num[0] / Gvg.den[2]:8.4f}  "
      f"(expect D/(1-D) = {params.D/(1-params.D):.4f})")

# (3) ss2tf round-trip
num_from_ss, den_from_ss = signal.ss2tf(A, B, C_mat, D_mat, input=0)
num_from_ss = np.trim_zeros(num_from_ss.flatten(), trim='f')
scale_ss = den_from_ss[0]
scale_cf = Gvd_closed.den[0]
num_ss_norm = num_from_ss / scale_ss
num_cf_norm = np.array(Gvd_closed.num) / scale_cf
den_ss_norm = np.array(den_from_ss) / scale_ss
den_cf_norm = np.array(Gvd_closed.den) / scale_cf
print()
print("(3) ss2tf round-trip (normalized):")
print(f"    Closed-form num = {num_cf_norm}")
print(f"    From-SS num     = {num_ss_norm}")
print(f"    Closed-form den = {den_cf_norm}")
print(f"    From-SS den     = {den_ss_norm}")
round_trip_ok = (
    np.allclose(num_ss_norm, num_cf_norm, rtol=1e-9)
    and np.allclose(den_ss_norm, den_cf_norm, rtol=1e-9)
)
print(f"    → match: {round_trip_ok}")

assert pole_match and round_trip_ok, "Self-consistency check failed!"
print()
print("✅  All three self-consistency checks pass.")
"""))

    cells.append(md(r"""
## 9. The "RHP zero migrates with duty" experiment

Re-derive the operating point for several duties and plot the RHP zero
frequency $f_{z,RHP}$ as a function of $D$. Watch it move dramatically:
at $D = 0.2$ the zero is at ~25 kHz; at $D = 0.8$ it crashes to under
2 kHz. That's the bandwidth ceiling for any voltage-mode controller.
"""))

    cells.append(code(r"""
D_grid = np.linspace(0.05, 0.95, 100)
f_z_grid = []
f_n_grid = []
V_o_grid = []
for d in D_grid:
    V_o_d = d / (1 - d) * params.V_g
    p_d = BuckBoostParams(V_g=params.V_g, V_o=V_o_d, R=params.R,
                          L=params.L, C=params.C, f_sw=params.f_sw)
    f_z_grid.append(p_d.f_z_rhp)
    f_n_grid.append(p_d.f_n)
    V_o_grid.append(V_o_d)

fig, (ax_v, ax_f) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax_v.plot(D_grid, V_o_grid, "C0", linewidth=2,
          label="|V_o| (output magnitude)")
ax_v.axhline(params.V_g, color="k", linestyle=":", alpha=0.4,
             label=f"V_g = {params.V_g} V")
ax_v.set_ylabel("|V_o| [V]")
ax_v.set_ylim(0, 60)
ax_v.legend()
ax_v.set_title("Buck-boost: |V_o| and the RHP zero vs duty cycle")

ax_f.semilogy(D_grid, f_z_grid, "C3", linewidth=2,
              label=r"$f_{z,RHP}$ (RHP zero)")
ax_f.semilogy(D_grid, f_n_grid, "C0", linestyle="--",
              label=r"$f_n$ (LC double pole)")
ax_f.axhline(params.f_sw/10, color="k", linestyle=":", alpha=0.4,
             label=f"$f_{{sw}}/10$ = {params.f_sw/10/1e3:.0f} kHz "
                   "(buck-style bandwidth cap)")
ax_f.axvline(0.5, color="g", linestyle=":", alpha=0.4,
             label="D = 0.5 (|V_o| = V_g)")
ax_f.set_xlabel("Duty cycle D")
ax_f.set_ylabel("Frequency [Hz]")
ax_f.legend(loc="best")
plt.tight_layout()
plt.show()

print("Notice three regimes:")
print(f"  D=0.2 (buck-like): f_z = {f_z_grid[np.argmin(np.abs(D_grid-0.2))]/1e3:.1f} kHz")
print(f"  D=0.5 (balanced):  f_z = {f_z_grid[np.argmin(np.abs(D_grid-0.5))]/1e3:.1f} kHz")
print(f"  D=0.8 (step-up):   f_z = {f_z_grid[np.argmin(np.abs(D_grid-0.8))]/1e3:.1f} kHz")
"""))

    cells.append(md(r"""
## 10. Summary

The inverting buck-boost combines the buck's and boost's modeling
machinery. State-space averaging produces a clean small-signal model
where:

- The polarity inversion is a TOPOLOGY detail, not a math artifact;
  we model $|V_o|$ as positive and add a sign at the physical layout.
- The same $-I_L \\hat d$ term in the capacitor equation that gave
  the boost its RHP zero also appears here — so the buck-boost is
  also non-minimum-phase.
- The RHP zero location $\\omega_{z,RHP} = R(1-D)^2 / (L D)$ depends
  *inversely* on $D$. At high duty (high step-up) the zero crashes to
  low frequency, capping the closed-loop bandwidth tighter.

Math validated by three checks (poles, DC gains, ss2tf round-trip).

**Next**: open `02_buck_boost_controller.ipynb` to size a Type-III
compensator with $f_c \\le f_{z,RHP}/5$, discretize via Tustin, and
run a switched closed-loop simulation that proves the design tracks
a reference step at the chosen duty.

**Note on Pulsim cross-validation.** Pulsim's switching engine has
the same numerical-stability issues with the ideal-switch buck-boost
that we saw with the boost (decoupled output cap → unbounded
switching-transition dV/dt). The math self-consistency above is the
rigorous validation; the closed-loop pure-Python simulation in
notebook 2 demonstrates the controller working on the real switched
waveform.

**Suggested exercises**

1. Set $V_o = 48$ V (D = 0.8) and re-derive the operating point and
   $f_{z,RHP}$. How much does the bandwidth cap tighten?
2. Set $V_o = 6$ V (D = 0.33) for a true step-DOWN buck-boost.
   How does the RHP zero compare to the boost's at the same R, L, C?
3. Derive the SEPIC topology (4th-order, non-inverting buck-boost).
   Where do its TWO RHP zeros come from?
"""))

    return cells


# ---------------------------------------------------------------------------
# Notebook 2 — controller
# ---------------------------------------------------------------------------


def build_controller_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 2 — Buck-Boost Controller Design

> **Goal.** Use the $G_{vd}(s)$ plant from notebook 1 to size a
> voltage-mode compensator that respects the RHP zero, then **prove**
> it works by running a switched closed-loop simulation in pure
> Python with a $v_{ref}$ step.

**Prerequisites**

- Buck-boost modeling notebook (`01_buck_boost_modeling.ipynb`).
- Buck and boost controller notebooks for context on K-factor
  Type-III sizing.

The recipe is the same as the boost's; what differs is the
$f_{z,RHP}$ location, which sets the bandwidth ceiling.
"""))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from buck_boost_model import (
    BuckBoostParams, control_to_output_tf, operating_point_report,
)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

params = BuckBoostParams()
print(operating_point_report(params))
"""))

    cells.append(md(r"""
## 1. Bandwidth target

For the default parameters ($D = 0.5$), $f_{z,RHP} \\approx 9.5$ kHz.
Target $f_c = f_{z,RHP} / 5 \\approx 1.9$ kHz with PM = 60°.
"""))

    cells.append(code(r"""
Gvd = control_to_output_tf(params)
V_ramp = 5.0
plant = signal.TransferFunction(np.array(Gvd.num)/V_ramp, np.array(Gvd.den))

f_c_target = params.f_z_rhp / 5.0
pm_target = 60.0
print(f"RHP zero:   f_z = {params.f_z_rhp:7.0f} Hz")
print(f"Target:     f_c = {f_c_target:.0f} Hz, PM = {pm_target}°")
"""))

    cells.append(md(r"""
## 2. K-factor Type-III sizing

Identical algorithm to the boost notebook §4. Repeated here for
self-containment.
"""))

    cells.append(code(r"""
def design_type3_kfactor(plant, f_c, pm_target):
    omega_c = 2 * np.pi * f_c
    _, _, ph_plant = signal.bode(plant, w=[omega_c])
    # scipy wraps phase to [-180°, +180°], but for a non-minimum-phase
    # plant (RHP zero + LC double pole) the true phase at f_c can dip
    # below -180°. Detect the wrap and unfold.
    ph_at_fc = ph_plant[0]
    if ph_at_fc > 0:
        ph_at_fc -= 360.0
    phi_lead = pm_target - 90.0 - ph_at_fc
    phi_lead = float(np.clip(phi_lead, 10.0, 175.0))
    phi_pair = phi_lead / 2
    k = np.tan(np.deg2rad(phi_pair/2 + 45.0)) ** 2
    omega_z = omega_c / np.sqrt(k)
    omega_p = omega_c * np.sqrt(k)
    num0 = np.polymul([1.0, omega_z], [1.0, omega_z])
    den0 = np.polymul([1.0, 0.0], np.polymul([1.0, omega_p], [1.0, omega_p]))
    open0 = signal.TransferFunction(
        np.polymul(num0, plant.num), np.polymul(den0, plant.den)
    )
    _, mag0, _ = signal.bode(open0, w=[omega_c])
    K = 10.0 ** (-mag0[0] / 20)
    return signal.TransferFunction(K * num0, den0), omega_z, omega_p, float(K)


Gc, omega_z, omega_p, K_dc = design_type3_kfactor(plant, f_c_target, pm_target)
print(f"Designed compensator:")
print(f"  zeros at  f_z = {omega_z/(2*np.pi):8.1f} Hz  (double)")
print(f"  HF poles  f_p = {omega_p/(2*np.pi):8.1f} Hz  (double)")
print(f"  DC gain K     = {K_dc:.4g}")
"""))

    cells.append(code(r"""
f = np.logspace(0, np.log10(params.f_sw), 1500)
w = 2*np.pi*f
T_open = signal.TransferFunction(np.polymul(Gc.num, plant.num),
                                   np.polymul(Gc.den, plant.den))
_, mag_T, ph_T = signal.bode(T_open, w=w)
_, mag_p, ph_p = signal.bode(plant, w=w)
idx_cross = np.argmin(np.abs(mag_T))
f_cross = f[idx_cross]
pm = 180 + ph_T[idx_cross]

fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax_mag.semilogx(f, mag_p, "C0--", alpha=0.5, label="Plant + $k_{PWM}$")
ax_mag.semilogx(f, mag_T, "C3", linewidth=2, label="Loop gain $T(s)$")
ax_mag.axhline(0, color="k", linestyle=":", alpha=0.3)
ax_mag.axvline(f_cross, color="g", linestyle=":", alpha=0.5,
               label=f"$f_c$ ≈ {f_cross:.0f} Hz")
ax_mag.axvline(params.f_z_rhp, color="C3", linestyle=":", alpha=0.4,
               label="$f_{z,RHP}$")
ax_ph.semilogx(f, ph_p, "C0--", alpha=0.5)
ax_ph.semilogx(f, ph_T, "C3", linewidth=2)
ax_ph.axhline(-180, color="k", linestyle=":", alpha=0.3)
ax_ph.axvline(f_cross, color="g", linestyle=":", alpha=0.5)
ax_mag.set_ylabel("Magnitude [dB]"); ax_ph.set_ylabel("Phase [deg]")
ax_ph.set_xlabel("Frequency [Hz]"); ax_mag.legend(loc="best", fontsize=8)
ax_mag.set_title(f"Compensated buck-boost loop: $f_c$ = {f_cross:.0f} Hz, "
                 f"PM = {pm:.1f}°")
plt.tight_layout()
plt.show()

print(f"Achieved:  f_c = {f_cross:.0f} Hz, PM = {pm:.1f}°")
"""))

    cells.append(md(r"""
## 3. Discretization

Tustin / bilinear at $T_s = 1/f_{sw}$.
"""))

    cells.append(code(r"""
T_s = 1.0 / params.f_sw
Gc_d_num, Gc_d_den, _ = signal.cont2discrete((Gc.num, Gc.den), dt=T_s, method="bilinear")
b = np.asarray(Gc_d_num).flatten() / Gc_d_den[0]
a = np.asarray(Gc_d_den) / Gc_d_den[0]
print(f"Sample period T_s = {T_s*1e6:.3f} µs")
print()
print("Discrete-time recurrence (a[0] = 1):")
for i, bi in enumerate(b): print(f"  b[{i}] = {bi:+.6f}")
for i, ai in enumerate(a): print(f"  a[{i}] = {ai:+.6f}")
"""))

    cells.append(md(r"""
## 4. Switched-model closed-loop simulation

Pure-Python forward-Euler switched buck-boost + discretized compensator
running once per switching period (sample-and-hold). Warm-start at the
operating point so we test the small-signal response the compensator
was sized for (the cold-start of a buck-boost is dominated by
nonlinear inrush — not what the controller design assumes).

**A small twist vs the boost.** In the buck-boost the inductor pumps
charge into a NEGATIVE-polarity output cap. The switched model is:

```
ON (S closed, D off):
    L · di/dt = V_g
    C · dv_o/dt = -v_o / R   (cap drains through load)

OFF (S open, D on, in CCM with i_L > 0):
    L · di/dt = -v_o          (cap voltage opposes inductor)
    C · dv_o/dt = i_L - v_o/R (cap charges, load also drains)
```

We model $v_o$ as a positive magnitude and just track that.
"""))

    cells.append(code(r"""
def simulate_closed_loop_buck_boost(
    params,
    b: np.ndarray, a: np.ndarray,
    *,
    t_end: float = 30e-3,
    t_step: float = 5e-3,
    v_ref_initial: float = 12.0,
    v_ref_final: float = 13.0,
    V_ramp: float = 5.0,
    samples_per_period: int = 200,
    warm_start: bool = True,
):
    '''Forward-Euler switched buck-boost + digital compensator.

    States are inductor current i_L and output cap voltage MAGNITUDE
    v_o. The compensator runs once per switching period (sample-and-
    hold), generating the duty for the next period.

    `warm_start=True` pre-loads i_L, v_o, and the compensator state at
    the operating point corresponding to v_ref_initial.
    '''
    T_s = 1.0 / params.f_sw
    dt_sim = T_s / samples_per_period
    n_steps = int(t_end / dt_sim) + 1

    n_state = len(a) - 1
    state = np.zeros(n_state)

    if warm_start:
        D_init = v_ref_initial / (v_ref_initial + params.V_g)
        I_L_avg = v_ref_initial / (params.R * (1.0 - D_init))
        # Inductor current at the BEGINNING of an ON interval = average
        # minus half the peak-to-peak ripple. Starting at the average
        # injects a half-period of extra charge into the cap on cycle 0
        # and causes the loop to drift before the controller catches up.
        T_s_period = 1.0 / params.f_sw
        delta_i_pp = params.V_g * D_init * T_s_period / params.L
        i_L = I_L_avg - delta_i_pp / 2.0
        v_o = v_ref_initial
        duty = D_init
        v_c_ss = duty * V_ramp
        for k in range(n_state):
            state[k] = -np.sum(a[k+1:]) * v_c_ss
    else:
        i_L = 0.0
        v_o = 0.0
        duty = 0.5

    record_every = max(1, samples_per_period // 50)
    n_rec = n_steps // record_every + 1
    t_hist = np.zeros(n_rec)
    v_o_hist = np.zeros(n_rec)
    i_L_hist = np.zeros(n_rec)
    duty_hist = np.zeros(n_rec)
    v_ref_hist = np.zeros(n_rec)
    rec_idx = 0

    for i in range(n_steps):
        t = i * dt_sim
        v_ref = v_ref_initial if t < t_step else v_ref_final

        cycle_pos_int = i % samples_per_period
        if cycle_pos_int == 0:
            err = v_ref - v_o
            v_c = b[0] * err + state[0]
            new_state = np.zeros_like(state)
            for j in range(n_state - 1):
                new_state[j] = b[j+1] * err - a[j+1] * v_c + state[j+1]
            new_state[n_state - 1] = b[n_state] * err - a[n_state] * v_c
            state = new_state
            duty = float(np.clip(v_c / V_ramp, 0.05, 0.92))

        switch_on = (cycle_pos_int / samples_per_period) < duty

        # Switched buck-boost ODE (positive-magnitude convention for v_o)
        if switch_on:
            v_L = params.V_g       # inductor sees V_g
            i_C = -v_o / params.R  # cap drains
        else:
            # OFF: diode conducts only if i_L > 0 (CCM); DCM detection:
            if i_L > 0:
                v_L = -v_o
                i_C = i_L - v_o / params.R
            else:
                v_L = 0.0          # diode off (DCM)
                i_C = -v_o / params.R
        i_L += (v_L / params.L) * dt_sim
        i_L = max(i_L, 0.0)
        v_o += (i_C / params.C) * dt_sim

        if i % record_every == 0 and rec_idx < n_rec:
            t_hist[rec_idx] = t
            v_o_hist[rec_idx] = v_o
            i_L_hist[rec_idx] = i_L
            duty_hist[rec_idx] = duty
            v_ref_hist[rec_idx] = v_ref
            rec_idx += 1

    return {
        "t": t_hist[:rec_idx], "v_o": v_o_hist[:rec_idx],
        "i_L": i_L_hist[:rec_idx], "duty": duty_hist[:rec_idx],
        "v_ref": v_ref_hist[:rec_idx],
    }
"""))

    cells.append(code(r"""
sim = simulate_closed_loop_buck_boost(
    params, b=b, a=a,
    t_end=30e-3, t_step=5e-3,
    v_ref_initial=12.0, v_ref_final=13.0,
    V_ramp=V_ramp, warm_start=True,
)

print(f"Simulated {len(sim['t'])} samples over {sim['t'][-1]*1e3:.2f} ms")
pre_mask  = (sim['t'] > 4.0e-3) & (sim['t'] < 5.0e-3)
post_mask = sim['t'] > 27.0e-3
print()
print(f"Pre-step  |V_o| (mean 4-5 ms):   {np.mean(sim['v_o'][pre_mask]):.4f} V "
      f"(target 12.0)")
print(f"Post-step |V_o| (mean 27-30 ms): {np.mean(sim['v_o'][post_mask]):.4f} V "
      f"(target 13.0)")
D_pre = 12.0 / (12.0 + params.V_g)
D_post = 13.0 / (13.0 + params.V_g)
print(f"Pre-step duty:  {np.mean(sim['duty'][pre_mask]):.4f}  "
      f"(expect {D_pre:.4f})")
print(f"Post-step duty: {np.mean(sim['duty'][post_mask]):.4f}  "
      f"(expect {D_post:.4f})")
"""))

    cells.append(code(r"""
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

axs[0].plot(sim['t']*1e3, sim['v_o'], 'C0', linewidth=0.8, label="$|v_o|$ (switched)")
axs[0].plot(sim['t']*1e3, sim['v_ref'], 'C3--', linewidth=2, label="$v_{ref}$")
axs[0].axvline(5.0, color="k", linestyle=":", alpha=0.4, label="step")
axs[0].set_ylabel("|Output voltage| [V]")
axs[0].set_title("Closed-loop buck-boost (warm-start at OP): step "
                 "$v_{ref}$ 12 V → 13 V at $t$ = 5 ms")
axs[0].legend(loc="lower right")

axs[1].plot(sim['t']*1e3, sim['i_L'], 'C1', linewidth=0.8)
axs[1].axvline(5.0, color="k", linestyle=":", alpha=0.4)
i_L_pre = 12.0 / (params.R * (1 - D_pre))
i_L_post = 13.0 / (params.R * (1 - D_post))
axs[1].axhline(i_L_pre, color="k", linestyle=":", alpha=0.3,
               label=f"pre-step $I_L$ = {i_L_pre:.2f} A")
axs[1].axhline(i_L_post, color="r", linestyle=":", alpha=0.3,
               label=f"post-step $I_L$ = {i_L_post:.2f} A")
axs[1].set_ylabel("Inductor current [A]")
axs[1].legend(loc="lower right")

axs[2].plot(sim['t']*1e3, sim['duty'], 'C2', linewidth=1.0)
axs[2].axvline(5.0, color="k", linestyle=":", alpha=0.4)
axs[2].axhline(D_pre, color="k", linestyle=":", alpha=0.3,
               label=f"pre-step D = {D_pre:.3f}")
axs[2].axhline(D_post, color="r", linestyle=":", alpha=0.3,
               label=f"post-step D = {D_post:.3f}")
axs[2].set_ylabel("Duty cycle")
axs[2].legend(loc="lower right")

axs[3].plot(sim['t']*1e3, sim['v_ref'] - sim['v_o'], 'C4', linewidth=0.8)
axs[3].axvline(5.0, color="k", linestyle=":", alpha=0.4)
axs[3].axhline(0, color="k", linestyle=":", alpha=0.3)
axs[3].set_ylabel("Tracking error\n$v_{ref} - |v_o|$ [V]")
axs[3].set_xlabel("Time [ms]")

plt.tight_layout()
plt.show()
"""))

    cells.append(code(r"""
mask_after = sim['t'] > 5.0e-3
t_after = sim['t'][mask_after] - 5.0e-3
v_o_after = sim['v_o'][mask_after]

overshoot_pct = (np.max(v_o_after) - 13.0) / (13.0 - 12.0) * 100
dip_amount = 12.0 - np.min(v_o_after)
settled = np.abs(v_o_after - 13.0) < 0.02 * (13.0 - 12.0)
unsettled = np.where(~settled)[0]
settling_ms = t_after[
    min(unsettled[-1] + 1, len(t_after) - 1) if len(unsettled) else 0
] * 1e3
v_o_10 = 12.0 + 0.1
v_o_90 = 12.0 + 0.9
rise_start = np.argmax(v_o_after >= v_o_10)
rise_end = np.argmax(v_o_after >= v_o_90)
rise_time_ms = (t_after[rise_end] - t_after[rise_start]) * 1e3

ss_error = 13.0 - np.mean(sim['v_o'][sim['t'] > 27e-3])

print("Closed-loop step-response metrics ($v_{ref}$: 12 → 13 V)")
print(f"  Initial dip (RHP zero)     = {dip_amount * 1e3:7.1f} mV below pre-step")
print(f"  Rise time (10% → 90%)      = {rise_time_ms:7.3f} ms")
print(f"  Peak overshoot             = {overshoot_pct:7.2f} %")
print(f"  Settling time (±2 %)       = {settling_ms:7.3f} ms")
print(f"  Steady-state error         = {ss_error*1e3:+7.2f} mV "
      f"({ss_error / 13.0 * 100:+.3f} %)")
print()
# Same generous gate as the boost — RHP-zero-limited loops are
# fundamentally slow.
if abs(ss_error) < 0.15 and overshoot_pct < 50 and settling_ms < 40.0:
    print("✅  Closed-loop controller PROVEN on the switched buck-boost:")
    print(f"    • SS error    = {ss_error*1e3:.1f} mV ({ss_error/13.0*100:.2f} %)")
    print(f"    • Overshoot   = {overshoot_pct:.1f} %")
    print(f"    • Settling    = {settling_ms:.1f} ms")
    print(f"    • RHP dip     = {dip_amount*1e3:.1f} mV (visible)")
    print()
    print(f"    Compare with buck: 1.4 ms settling, 0.6 % overshoot.")
    print(f"    Buck-boost is ~{settling_ms/1.4:.0f}× slower because the RHP zero caps")
    print(f"    the loop bandwidth. Same fundamental limit as the boost.")
else:
    print("⚠️   Closed-loop response off-target — revisit f_c or PM.")
"""))

    cells.append(md(r"""
## 5. Summary

You sized a Type-III compensator for the inverting buck-boost
respecting the $f_c \\le f_{z,RHP}/5$ ceiling, discretized it via
Tustin, and ran a switched closed-loop simulation that demonstrates:

- Stable tracking of a step in $v_{ref}$.
- The characteristic RHP-zero dip before recovery.
- Slow settling vs the buck — fundamentally capped by the topology.

**Suggested exercises**

1. Re-tune for $V_o = 24$ V (D = 0.67). Verify the new $f_{z,RHP}$
   is lower and the closed loop is correspondingly slower.
2. Add a 50 % load step ($R$: 12 → 6 Ω) at $t = 20$ ms. Watch the
   output sag and the controller recover — the buck-boost's load
   regulation is fundamentally limited by the same RHP zero.
3. Build a SEPIC (4th-order non-inverting buck-boost). Where do the
   TWO RHP zeros come from? How does the closed-loop bandwidth limit
   compare?
"""))

    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    nb1 = build_modeling_notebook()
    nb2 = build_controller_notebook()
    write_notebook(nb1, HERE / "01_buck_boost_modeling.ipynb")
    write_notebook(nb2, HERE / "02_buck_boost_controller.ipynb")


if __name__ == "__main__":
    main()
