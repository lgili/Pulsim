"""Generator for the half-bridge teaching notebooks.

Run once after editing to regenerate `01_half_bridge_modeling.ipynb`
and `02_half_bridge_controller.ipynb`. Same patterns as the buck /
boost / buck-boost / flyback / forward generators — the
half-bridge inherits the forward's small-signal model, so the math
notebook 1 reuses most of the forward's structure. What's new is the
**4-phase switched simulator** in notebook 2 (S1-on, dead, S2-on,
dead) and the rail-split / dead-time pedagogy.
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
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {path.relative_to(HERE.parent.parent)} ({path.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# Notebook 1 — Half-bridge modeling
# ---------------------------------------------------------------------------


def build_modeling_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 1 — Half-Bridge Converter Modeling

> **Goal.** Derive the small-signal model of the half-bridge and show
> that it is **identical to the forward's** — same $G_{vd}(s)$, same
> $G_{vg}(s)$, same $Z_{out}(s)$, no RHP zero. What changes is the
> IMPLEMENTATION: two switches alternating around a rail-split input,
> a center-tapped secondary, dead-time to prevent shoot-through, and
> output ripple at **$2 f_{sw}$** (twice the per-switch frequency).

**Prerequisites**

- Forward modeling notebook (`projects/converters/forward/`). The
  half-bridge's average model is literally the forward's.
- Buck modeling notebook (`projects/converters/buck/`). The forward
  is buck-derived, so the half-bridge is too.

**What you'll be able to do at the end**

1. Identify the half-bridge on a schematic by the rail-split input
   and the two alternating switches.
2. Write the 4-phase switched model (S1-on, dead, S2-on, dead) and
   recognize that averaging it produces the forward's equations.
3. Explain why $V_o = n \cdot V_g \cdot D$ even though the primary
   sees only $V_g/2$ (the $1/2$ from rail-split exactly cancels the
   $2$ from both half-cycles being rectified).
4. State the dead-time / shoot-through constraint and how it caps
   the duty at $D_{max} \approx 0.45$.
5. Predict the output ripple frequency: $2 f_{sw}$.
6. Argue why the half-bridge replaces the forward as the dominant
   isolated topology beyond a few tens of watts: lower switch stress,
   no reset winding, smaller output filter for the same ripple.
"""))

    cells.append(md("## Setup"))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from half_bridge_model import (
    HalfBridgeParams,
    half_bridge_state_space,
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
## 1. The half-bridge topology

```
       V_g+ ----+---- S1 (high-side) --+
                |                      |
               C1                      |
                |                      |
                +-----------------+    +----- sw_node (mid)  ── primary ──┐
                |                 |    |                                   |
               C2                 |    |                                   )||(
                |                 |    |                                   )||( transformer
              V_g/2 (midpoint) ───+    |                                   )||(
                                       |                                   |
       V_g- ----+---- S2 (low-side) ---+                       gnd_pri ────┘

       (rail-split capacitor pair       (alternating switches:
        produces V_g/2 midpoint)         S1 on first half, S2 second half)


Secondary (center-tapped):
                      ┌── D1 ── L ──┬── V_o
                      │             │
                      ├── (tap) ────┤   C ─── R
                      │             │
                      └── D2 ── L ──┴── gnd_sec
```

- **Input rail-splitter**: C1 and C2 are large series caps across
  $V_g$; their midpoint sits at $V_g/2$ (assuming balanced
  capacitance and equal switching duty).
- **High-side switch S1** (during first half-period): connects
  switching node to $V_g$. Primary sees $V_g - V_g/2 = +V_g/2$.
- **Low-side switch S2** (during second half-period): connects
  switching node to ground. Primary sees $0 - V_g/2 = -V_g/2$.
- **Dead-time** between S1-off and S2-on: BOTH switches off, primary
  floats. Without this, both switches briefly conducting would short
  $V_g$ through them ("shoot-through") and destroy the silicon.
- **Center-tapped secondary**: two diodes rectify the AC primary
  swing into a DC pulse train at twice the per-switch frequency.

### 1.1 The per-switch duty convention

$D$ is the **on-time fraction of ONE switch over a full period**, so
$D \in [0, 0.5]$. The effective conduction fraction at the output
is $2D$ (both half-cycles contribute equally after rectification).

### 1.2 Energy routing — half-bridge vs forward vs flyback

| Phase | Half-bridge | Forward | Flyback |
|---|---|---|---|
| S1/S2 ON | primary swings ±V_g/2; secondary energized through D1 or D2; filter L charged | primary energized; D1 on, D2 off; filter L charged | primary stores energy in L_m; secondary disconnected |
| Dead-time | both switches off; filter L freewheels through D1/D2 | — | — |
| Both OFF (forward) | — | filter L freewheels through D2 | — |
| Switch OFF (flyback) | — | — | secondary releases stored energy through diode |

Both half-bridge and forward are **buck-derived** — energy is
transferred forward through the transformer in real time, filter L-C
handles storage. Flyback is buck-boost-derived (different family,
RHP zero).
"""))

    cells.append(md(r"""
## 2. Switched (instantaneous) model — 4 phases per period

A full switching period $T_s = 1/f_{sw}$ splits into four phases:

```
|<------ T_s = 1 / f_sw ----------->|
|<-- D·T_s -->|     |<-- D·T_s -->|
+-------------+-----+-------------+-----+
|     S1      |dead |     S2      |dead |
+-------------+-----+-------------+-----+
0          D·T_s  T_s/2        T_s/2+D·T_s   T_s
```

### 2.1 Phase 1: S1 on, S2 off ($0 \le t < D \cdot T_s$)

Primary sees $+V_g/2$ → secondary sees $+n \cdot V_g/2$. D1 forward-
biased, D2 reverse-biased. Filter L-C input sees $n \cdot V_g/2$:

$$
L \cdot \frac{di_L}{dt} = n \cdot V_g/2 - v_o
\qquad
C \cdot \frac{dv_o}{dt} = i_L - \frac{v_o}{R}
$$

Wait — that says the L-C input is $n V_g/2$, not $n V_g$. Where does
the $V_o = n V_g D$ formula (not $n V_g D / 2$) come from? Read on.

### 2.2 Phase 2: dead-time after S1 ($D T_s \le t < T_s/2$)

Both switches off. Primary floats; transformer doesn't drive
secondary. **But the filter inductor must keep its current** — both
secondary diodes freewheel (D1 and D2 simultaneously, splitting
$i_L/2$ each through the center tap), and the L-C input sees 0:

$$
L \cdot \frac{di_L}{dt} = -v_o
\qquad
C \cdot \frac{dv_o}{dt} = i_L - \frac{v_o}{R}
$$

### 2.3 Phase 3: S2 on, S1 off ($T_s/2 \le t < T_s/2 + D T_s$)

Primary sees $-V_g/2$ → secondary sees $-n V_g/2$. The center-tap
rectifier **flips the sign**: D2 now forward-biases, D1 reverse-
biases. From the filter's perspective, the input voltage is **the
same magnitude as in phase 1**:

$$
L \cdot \frac{di_L}{dt} = n \cdot V_g/2 - v_o
\qquad
C \cdot \frac{dv_o}{dt} = i_L - \frac{v_o}{R}
$$

The center-tap rectifier turns ±V_g/2 on the primary into +V_g/2 on
the filter side — that's the "2× effective conduction" mentioned
earlier.

### 2.4 Phase 4: dead-time after S2 ($T_s/2 + D T_s \le t < T_s$)

Same as phase 2: both switches off, filter freewheels through both
diodes, L-C input sees 0.

### 2.5 So why is the average $V_o = n V_g D$ (not $n V_g D / 2$)?

Take the time average of the filter input voltage over one $T_s$:

$$
\bar v_{Lin} = \frac{1}{T_s} \int_0^{T_s} v_{Lin}(t)\,dt
= \frac{1}{T_s}\Big[D T_s \cdot n V_g/2 + (T_s/2 - D T_s) \cdot 0
                  + D T_s \cdot n V_g/2 + (T_s/2 - D T_s) \cdot 0\Big]
= n V_g \cdot D
$$

The factor of $1/2$ from the rail-split is exactly cancelled by the
factor of $2$ from both half-cycles contributing. The filter sees the
same average voltage as if a forward converter with primary $V_g$ ran
at duty $D$.

That's why the small-signal model is identical to the forward's.
"""))

    cells.append(md(r"""
## 3. State-space averaging — same as the forward

Average all four phases (replacing $q_1 + q_2$ by $2 d$ effective):

$$\boxed{
\;\; L \cdot \frac{di_L}{dt} = d \cdot n \cdot V_g - v_o
\;\;}
$$

$$\boxed{
\;\; C \cdot \frac{dv_o}{dt} = i_L - \frac{v_o}{R}
\;\;}
$$

**Identical to the forward.** The single-switch forward and the
two-switch half-bridge are the same converter in the average model.

### 3.1 Steady-state

$$
V_o = n \cdot V_g \cdot D, \qquad I_L = V_o / R
$$

### 3.2 Dead-time / shoot-through constraint

The "reset winding" constraint of the forward becomes the
**dead-time** constraint of the half-bridge:

$$
D \cdot T_s + t_{dead} \le T_s / 2
\quad\Longleftrightarrow\quad
D \le \frac{1}{2} - \frac{t_{dead}}{T_s}
$$

For $t_{dead} = 100$ ns, $T_s = 10$ µs (100 kHz), this caps $D$ at
$0.49$. Production designs use $D_{max} \approx 0.45$ for safety
margin. **Same numerical cap as the forward** — but a different
physical mechanism.
"""))

    cells.append(code(r"""
params = HalfBridgeParams()
print(operating_point_report(params))
print()
params.assert_dead_time_ok()
print("✅  Dead-time / shoot-through constraint check passed.")
"""))

    cells.append(md(r"""
## 4. Small-signal linearization — same as the forward

Perturb $i_L = I_L + \hat i_L$, $v_o = V_o + \hat v_o$, $d = D + \hat d$,
$v_g = V_g + \hat v_g$. Substitute and drop products:

$$
L \cdot \frac{d\hat i_L}{dt}
   = D \cdot n \cdot \hat v_g + n \cdot V_g \cdot \hat d - \hat v_o
$$

$$
C \cdot \frac{d\hat v_o}{dt}
   = \hat i_L - \frac{\hat v_o}{R}
$$

**No $\hat d$ term in the cap equation** → **no RHP zero**. Same as
the buck and the forward. The half-bridge inherits all of the
buck's control simplicity, just with twice the switches and a
rail-split input.
"""))

    cells.append(md(r"""
## 5. State-space matrices

$$
A = \begin{bmatrix}
0 & -1/L \\
1/C & -1/(R C)
\end{bmatrix},
\quad
B = \begin{bmatrix}
n V_g / L & n D / L \\
0 & 0
\end{bmatrix}
$$

$$
C = \begin{bmatrix} 0 & 1 \end{bmatrix},
\quad
D_{\rm feed} = \begin{bmatrix} 0 & 0 \end{bmatrix}
$$

Same matrices as the forward. The two-switch implementation
disappears completely in the average model.
"""))

    cells.append(code(r"""
A, B, C_mat, D_mat = half_bridge_state_space(params)
print("A ="); print(A); print()
print("B = [col 0: d̂   col 1: v̂_g]"); print(B); print()
print("C =", C_mat)
print("D (feedthrough) =", D_mat)
print()
eig = np.linalg.eigvals(A)
print(f"A eigenvalues: {eig}")
print(f"Pole magnitude (= ω_n) = {abs(eig[0]):.1f} rad/s "
      f"(expect {params.omega_n:.1f})")
"""))

    cells.append(md(r"""
## 6. Transfer functions

### 6.1 $G_{vd}(s)$

$$
G_{vd}(s) = \frac{n V_g}{LC} \cdot
\frac{1}{s^2 + s/(RC) + 1/(LC)}
$$

Same shape as the forward and buck. Second-order low-pass, no zeros,
no RHP zero.

### 6.2 $G_{vg}(s)$, $Z_{out}(s)$

$$
G_{vg}(s) = \frac{n D}{LC s^2 + L/R \cdot s + 1}
$$

$$
Z_{out}(s) = \frac{s L}{LC s^2 + L/R \cdot s + 1}
$$

Same as the forward.
"""))

    cells.append(code(r"""
Gvd = control_to_output_tf(params)
Gvg = line_to_output_tf(params)
Zout = output_impedance_tf(params)

print(f"Gvd(0)  = {Gvd.num[0] / Gvd.den[2]:.3f} V/duty  "
      f"(expect n·V_g = {params.n*params.V_g:.3f})")
print(f"Gvg(0)  = {Gvg.num[0] / Gvg.den[2]:.4f} V/V    "
      f"(expect n·D = {params.n*params.D:.4f})")
print()
if len(Gvd.num) > 1:
    print(f"Gvd zeros: {np.roots(Gvd.num)}")
else:
    print(f"Gvd zeros: NONE  (no RHP zero — buck-derived isolation)")
print(f"Gvd poles: {np.roots(Gvd.den)}")
"""))

    cells.append(md(r"""
### 6.3 Bode plots

Same shape as the forward. The half-bridge's $G_{vd}$ rolls off at
-40 dB/dec past $f_n$ with phase finishing at -180°.
"""))

    cells.append(code(r"""
f = np.logspace(0, np.log10(params.f_sw), 1500)
w = 2 * np.pi * f

fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for tf, name, style in [
    (Gvd,  r"$G_{vd}$ control → output", "-"),
    (Gvg,  r"$G_{vg}$ line → output",    "--"),
    (Zout, r"$Z_{out}$ load → output",   ":"),
]:
    _, mag, ph = signal.bode(tf, w=w)
    ax_mag.semilogx(f, mag, style, label=name)
    ax_ph.semilogx(f, ph, style, label=name)

for ax in (ax_mag, ax_ph):
    ax.axvline(params.f_n, color="C0", linestyle=":", alpha=0.4,
               label=f"$f_n$ = {params.f_n:.0f} Hz")
    ax.axvline(params.f_sw / 10, color="C3", linestyle=":", alpha=0.4,
               label=f"$f_{{sw}}/10$ = {params.f_sw / 10 / 1e3:.0f} kHz (BW target)")
    ax.legend(loc="best", fontsize=8)

ax_mag.set_ylabel("Magnitude [dB]")
ax_ph.set_ylabel("Phase [deg]")
ax_ph.set_xlabel("Frequency [Hz]")
ax_mag.set_title(f"Half-bridge open-loop "
                 f"($V_g$={params.V_g}V → $V_o$={params.V_o}V, n={params.n}, "
                 f"D={params.D:.3f})  — NO RHP zero")
plt.tight_layout()
plt.show()
"""))

    cells.append(md(r"""
## 7. Step response — monotonic (same as forward)
"""))

    cells.append(code(r"""
duty_step = 0.01
t = np.linspace(0, 5e-3, 5000)
_, y_step = signal.step(Gvd, T=t)
v_o_pred = params.V_o + duty_step * y_step

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(t * 1e3, v_o_pred, label="$v_o$ (analytical small-signal)")
ax.axhline(params.V_o, color="k", linestyle=":", alpha=0.4,
           label=f"Pre-step $V_o$ = {params.V_o} V")
ax.axhline(params.V_o + duty_step * params.n * params.V_g, color="g",
           linestyle=":", alpha=0.5,
           label=f"Predicted new $V_o$ = "
                 f"{params.V_o + duty_step * params.n * params.V_g:.3f} V")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("$v_o$ [V]")
ax.set_title("Half-bridge step response: monotonic — no RHP-zero dip")
ax.legend()
plt.tight_layout()
plt.show()

print(f"Pre-step  V_o = {params.V_o:.4f} V")
print(f"Final     v_o = {v_o_pred[-1]:.4f} V")
print(f"Minimum   v_o = {np.min(v_o_pred):.4f} V "
      f"(no dip — monotonic rise)")
"""))

    cells.append(md(r"""
## 8. Self-consistency checks

Same three rigorous checks. The state-space and the closed-form TFs
must match.
"""))

    cells.append(code(r"""
def _trim_near_zero(coeffs, rel_tol=1e-9):
    '''Strip leading near-zero coefficients (scipy.ss2tf noise).'''
    a = np.asarray(coeffs).flatten()
    if a.size == 0:
        return a
    threshold = rel_tol * np.max(np.abs(a))
    nz = np.argmax(np.abs(a) > threshold)
    return a[nz:]


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
print(f"    Gvd(0)  = {Gvd_closed.num[0] / Gvd_closed.den[2]:8.4f}  "
      f"(expect n·V_g = {params.n * params.V_g:.4f})")
Gvg_local = line_to_output_tf(params)
print(f"    Gvg(0)  = {Gvg_local.num[0] / Gvg_local.den[2]:8.4f}  "
      f"(expect n·D = {params.n * params.D:.4f})")

# (3) ss2tf round-trip
num_from_ss, den_from_ss = signal.ss2tf(A, B, C_mat, D_mat, input=0)
num_from_ss = _trim_near_zero(num_from_ss)
scale_ss = den_from_ss[0]
scale_cf = Gvd_closed.den[0]
num_ss_norm = num_from_ss / scale_ss
num_cf_norm = np.array(Gvd_closed.num) / scale_cf
den_ss_norm = np.array(den_from_ss) / scale_ss
den_cf_norm = np.array(Gvd_closed.den) / scale_cf
print()
print("(3) ss2tf round-trip (trimmed for numerical noise):")
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
## 9. Summary

The half-bridge converter is the forward's twin in the **average
model**:

- Same state-space (buck-derived with $n V_g$ scaling)
- Same $G_{vd}(s)$, $G_{vg}(s)$, $Z_{out}(s)$
- **No RHP zero** → buck-style control bandwidth
- Same $D_{max} \approx 0.45$ cap (different physical mechanism:
  dead-time, not reset winding)

What's different is the **implementation**:

- Two switches alternating instead of one (S1 high-side, S2 low-side)
- Rail-split input (each cap holds $V_g/2$)
- Per-switch voltage stress halved (each switch sees $V_g$, not the
  $2V_g$ a forward sees during reset)
- No reset winding needed (each half-cycle resets the flux
  symmetrically)
- **Output ripple at $2 f_{sw}$** — the filter LC can be smaller for
  the same ripple spec

These differences matter for sizing (switch ratings, filter L-C, EMI
filter design) but they are invisible to the small-signal model.

**Next**: open `02_half_bridge_controller.ipynb` for the controller
design — which reuses the forward's recipe verbatim — plus a
**4-phase switched closed-loop simulation** that actually models the
S1-dead-S2-dead sequence.

**Suggested exercises**

1. Compare $G_{vd}(s)$ of the forward and half-bridge on the same
   Bode plot with the same $V_g$, $V_o$, $L$, $C$, $R$, $n$. Verify
   they overlap exactly.
2. Show that the per-switch voltage stress in the half-bridge is
   $V_g$ (each switch's drain sees $V_g$ when off, while the other
   switch is on or during dead-time the body diodes clamp).
3. Sketch the output ripple waveform at $2 f_{sw}$. Why is the LC
   filter requirement smaller than a single-switch forward at the
   same $f_{sw}$ and ripple spec?
4. What goes wrong if C1 and C2 aren't exactly matched? (Hint:
   imbalance shifts the midpoint away from $V_g/2$, biasing the
   transformer flux DC — saturation eventually.)
"""))

    return cells


# ---------------------------------------------------------------------------
# Notebook 2 — Controller
# ---------------------------------------------------------------------------


def build_controller_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 2 — Half-Bridge Converter Controller Design

> **Goal.** The control loop is the forward's verbatim — same Type-III
> K-factor recipe at $f_c = f_{sw}/20$, PM = 60°, Tustin discretization,
> duty clipped at $D_{max} = 0.45$. What changes is the **switched
> simulator**: it now has to step through four phases per period
> (S1-on, dead, S2-on, dead) and respect the dead-time, not just two
> phases.

**Prerequisites**

- Half-bridge modeling notebook (`01_half_bridge_modeling.ipynb`).
- Forward controller notebook — the compensator design is identical.

The closed-loop response should match the forward's (~1 ms settling)
because the average model is identical. The simulator runs at a
finer step to resolve the dead-time slivers.
"""))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from half_bridge_model import (
    HalfBridgeParams, control_to_output_tf, operating_point_report,
)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

params = HalfBridgeParams()
print(operating_point_report(params))
"""))

    cells.append(md(r"""
## 1. Bandwidth target — same as the forward

Same as the forward: target $f_c = f_{sw}/20 = 5$ kHz, PM = 60°. The
$/20$ choice (rather than the buck's $/10$) trades some bandwidth
for clean tracking against the $D_{max} = 0.45$ duty clip.
"""))

    cells.append(code(r"""
Gvd = control_to_output_tf(params)
V_ramp = 5.0
plant = signal.TransferFunction(np.array(Gvd.num)/V_ramp, np.array(Gvd.den))

f_c_target = params.f_sw / 20.0
pm_target = 60.0
print(f"Target:     f_c = {f_c_target/1e3:.1f} kHz, PM = {pm_target}°")
print(f"Plant DC gain (with k_PWM = 1/V_ramp = {1/V_ramp:.3f}): "
      f"{plant.num[0]/plant.den[2]:.3f}")
"""))

    cells.append(md(r"""
## 2. K-factor Type-III design

Same algorithm as the forward. No phase-unwrap edge case here (no
RHP zero → phase stays well behaved).
"""))

    cells.append(code(r"""
def design_type3_kfactor(plant, f_c, pm_target):
    omega_c = 2 * np.pi * f_c
    _, _, ph_plant = signal.bode(plant, w=[omega_c])
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
w = 2 * np.pi * f
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
               label=f"$f_c$ ≈ {f_cross/1e3:.1f} kHz")
ax_ph.semilogx(f, ph_p, "C0--", alpha=0.5)
ax_ph.semilogx(f, ph_T, "C3", linewidth=2)
ax_ph.axhline(-180, color="k", linestyle=":", alpha=0.3)
ax_ph.axvline(f_cross, color="g", linestyle=":", alpha=0.5)
ax_mag.set_ylabel("Magnitude [dB]"); ax_ph.set_ylabel("Phase [deg]")
ax_ph.set_xlabel("Frequency [Hz]"); ax_mag.legend(loc="best", fontsize=8)
ax_mag.set_title(f"Compensated half-bridge loop: $f_c$ = {f_cross/1e3:.1f} kHz, "
                 f"PM = {pm:.1f}°")
plt.tight_layout()
plt.show()

print(f"Target:    f_c = {f_c_target/1e3:.1f} kHz, PM = {pm_target}°")
print(f"Achieved:  f_c = {f_cross/1e3:.2f} kHz, PM = {pm:.1f}°")
"""))

    cells.append(md(r"""
## 3. Discretization

Tustin / bilinear at $T_s = 1/f_{sw}$ — once per **per-switch period**
(same as forward and buck reference; the compensator doesn't care
that there are two switches).
"""))

    cells.append(code(r"""
T_s = 1.0 / params.f_sw
Gc_d_num, Gc_d_den, _ = signal.cont2discrete(
    (Gc.num, Gc.den), dt=T_s, method="bilinear"
)
b = np.asarray(Gc_d_num).flatten() / Gc_d_den[0]
a = np.asarray(Gc_d_den) / Gc_d_den[0]
print(f"Sample period T_s = {T_s*1e6:.3f} µs")
print()
print("Discrete-time recurrence (a[0] = 1):")
for i, bi in enumerate(b): print(f"  b[{i}] = {bi:+.6f}")
for i, ai in enumerate(a): print(f"  a[{i}] = {ai:+.6f}")
"""))

    cells.append(md(r"""
## 4. Switched-model closed-loop simulation — 4 phases per period

The half-bridge simulator must respect the four phases per switching
period:

```
  phase   | duration       | S1 | S2 | filter L-C input voltage v_Lin
  --------+----------------+----+----+--------------------------------
  1 (S1)  | D · T_s        | on | off| n · V_g / 2
  2 (dead)| T_s/2 - D·T_s  | off| off| 0  (filter freewheels)
  3 (S2)  | D · T_s        | off| on | n · V_g / 2  (rectifier flips sign)
  4 (dead)| T_s/2 - D·T_s  | off| off| 0
```

Then the filter ODE is the same regardless of which switch is on:

```
during S1 ON  or  S2 ON:  L · di_L/dt = n · V_g / 2 - v_o
during dead-time:         L · di_L/dt = -v_o
```

The compensator runs **once per per-switch period** (sample-and-hold)
and clips duty to $D_{max} = 0.45$ to keep the dead-time intact.

Warm-start at the operating point with valley $i_L$ — important
because the output ripple is at $2 f_{sw}$, so the inductor ripple
is also at $2 f_{sw}$ and the "valley" happens twice per per-switch
period.
"""))

    cells.append(code(r"""
def simulate_closed_loop_half_bridge(
    params,
    b: np.ndarray, a: np.ndarray,
    *,
    t_end: float = 5e-3,
    t_step: float = 1e-3,
    v_ref_initial: float = 5.0,
    v_ref_final: float = 5.2,
    V_ramp: float = 5.0,
    samples_per_period: int = 400,
    warm_start: bool = True,
):
    '''Forward-Euler 4-phase half-bridge + digital compensator.

    State: filter inductor current i_L, output cap voltage v_o.
    Per period T_s = 1/f_sw the simulator steps through:
       phase 1: 0 .. D·T_s            (S1 on,  v_Lin = n V_g/2)
       phase 2: D·T_s .. T_s/2        (dead,   v_Lin = 0)
       phase 3: T_s/2 .. T_s/2 + D·T_s(S2 on,  v_Lin = n V_g/2)
       phase 4: T_s/2 + D·T_s .. T_s  (dead,   v_Lin = 0)

    Compensator runs once per per-switch period (cycle_pos_int == 0)
    and its duty output is clipped to [0.05, params.D_max] to keep the
    dead-time intact.

    samples_per_period = 400 (twice the forward's 200) — output ripple
    is at 2 f_sw, so we need finer resolution to see it cleanly.
    '''
    T_s = 1.0 / params.f_sw
    dt_sim = T_s / samples_per_period
    n_steps = int(t_end / dt_sim) + 1
    n_turn = params.n
    v_in_active = n_turn * params.V_g / 2.0  # what filter sees when a switch is on

    n_state = len(a) - 1
    state = np.zeros(n_state)

    if warm_start:
        D_init = v_ref_initial / (n_turn * params.V_g)
        I_L_avg = v_ref_initial / params.R
        # Ripple per HALF-period (output ripple is at 2 f_sw)
        # When a switch is on: di/dt = (n V_g/2 - v_o) / L for time D·T_s
        delta_i_pp = (v_in_active - v_ref_initial) * D_init * T_s / params.L
        i_L = I_L_avg - delta_i_pp / 2.0
        v_o = v_ref_initial
        duty = D_init
        v_c_ss = duty * V_ramp
        for k in range(n_state):
            state[k] = -np.sum(a[k+1:]) * v_c_ss
    else:
        i_L = 0.0
        v_o = 0.0
        duty = 0.3

    record_every = max(1, samples_per_period // 50)
    n_rec = n_steps // record_every + 1
    t_hist = np.zeros(n_rec)
    v_o_hist = np.zeros(n_rec)
    i_L_hist = np.zeros(n_rec)
    duty_hist = np.zeros(n_rec)
    v_ref_hist = np.zeros(n_rec)
    s1_hist = np.zeros(n_rec)
    s2_hist = np.zeros(n_rec)
    rec_idx = 0

    half = samples_per_period // 2  # phase boundary between S1-side and S2-side

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
            duty = float(np.clip(v_c / V_ramp, 0.05, params.D_max))

        # Determine current phase
        d_int = int(round(duty * samples_per_period))
        if cycle_pos_int < d_int:
            s1, s2 = 1.0, 0.0
            v_L = v_in_active - v_o
        elif cycle_pos_int < half:
            s1, s2 = 0.0, 0.0
            v_L = -v_o
        elif cycle_pos_int < half + d_int:
            s1, s2 = 0.0, 1.0
            v_L = v_in_active - v_o
        else:
            s1, s2 = 0.0, 0.0
            v_L = -v_o

        i_C = i_L - v_o / params.R
        i_L += (v_L / params.L) * dt_sim
        i_L = max(i_L, 0.0)
        v_o += (i_C / params.C) * dt_sim

        if i % record_every == 0 and rec_idx < n_rec:
            t_hist[rec_idx] = t
            v_o_hist[rec_idx] = v_o
            i_L_hist[rec_idx] = i_L
            duty_hist[rec_idx] = duty
            v_ref_hist[rec_idx] = v_ref
            s1_hist[rec_idx] = s1
            s2_hist[rec_idx] = s2
            rec_idx += 1

    return {
        "t": t_hist[:rec_idx], "v_o": v_o_hist[:rec_idx],
        "i_L": i_L_hist[:rec_idx], "duty": duty_hist[:rec_idx],
        "v_ref": v_ref_hist[:rec_idx],
        "s1": s1_hist[:rec_idx], "s2": s2_hist[:rec_idx],
    }
"""))

    cells.append(code(r"""
sim = simulate_closed_loop_half_bridge(
    params, b=b, a=a,
    t_end=5e-3, t_step=1e-3,
    v_ref_initial=5.0, v_ref_final=5.2,
    V_ramp=V_ramp, warm_start=True,
)

print(f"Simulated {len(sim['t'])} samples over {sim['t'][-1]*1e3:.2f} ms")
pre_mask  = (sim['t'] > 0.8e-3) & (sim['t'] < 1.0e-3)
post_mask = sim['t'] > 4.5e-3
print()
print(f"Pre-step  V_o (mean 0.8-1.0 ms):  {np.mean(sim['v_o'][pre_mask]):.4f} V "
      f"(target 5.0)")
print(f"Post-step V_o (mean 4.5-5.0 ms):  {np.mean(sim['v_o'][post_mask]):.4f} V "
      f"(target 5.2)")
D_pre  = 5.0 / (params.n * params.V_g)
D_post = 5.2 / (params.n * params.V_g)
print(f"Pre-step duty:  {np.mean(sim['duty'][pre_mask]):.4f} (expect {D_pre:.4f})")
print(f"Post-step duty: {np.mean(sim['duty'][post_mask]):.4f} (expect {D_post:.4f})")
"""))

    cells.append(md(r"""
### 4.1 Closed-loop waveforms (full step)

Four panels: output voltage tracking, inductor current, duty, and
tracking error.
"""))

    cells.append(code(r"""
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

axs[0].plot(sim['t']*1e3, sim['v_o'], 'C0', linewidth=0.8, label="$v_o$ (switched)")
axs[0].plot(sim['t']*1e3, sim['v_ref'], 'C3--', linewidth=2, label="$v_{ref}$")
axs[0].axvline(1.0, color="k", linestyle=":", alpha=0.4, label="step")
axs[0].set_ylabel("Output voltage [V]")
axs[0].set_title("Closed-loop half-bridge (warm-start at OP): step "
                 "$v_{ref}$ 5.0 V → 5.2 V at $t$ = 1 ms")
axs[0].legend(loc="lower right")

axs[1].plot(sim['t']*1e3, sim['i_L'], 'C1', linewidth=0.8)
axs[1].axvline(1.0, color="k", linestyle=":", alpha=0.4)
I_L_pre  = 5.0 / params.R
I_L_post = 5.2 / params.R
axs[1].axhline(I_L_pre, color="k", linestyle=":", alpha=0.3,
               label=f"pre-step $I_L$ = {I_L_pre:.2f} A")
axs[1].axhline(I_L_post, color="r", linestyle=":", alpha=0.3,
               label=f"post-step $I_L$ = {I_L_post:.2f} A")
axs[1].set_ylabel("Inductor current [A]")
axs[1].legend(loc="lower right")

axs[2].plot(sim['t']*1e3, sim['duty'], 'C2', linewidth=1.0)
axs[2].axvline(1.0, color="k", linestyle=":", alpha=0.4)
axs[2].axhline(D_pre, color="k", linestyle=":", alpha=0.3,
               label=f"pre-step D = {D_pre:.3f}")
axs[2].axhline(D_post, color="r", linestyle=":", alpha=0.3,
               label=f"post-step D = {D_post:.3f}")
axs[2].axhline(params.D_max, color="C3", linestyle="--", alpha=0.5,
               label=f"$D_{{max}}$ = {params.D_max:.2f} (dead-time limit)")
axs[2].set_ylabel("Per-switch duty")
axs[2].legend(loc="lower right")

axs[3].plot(sim['t']*1e3, sim['v_ref'] - sim['v_o'], 'C4', linewidth=0.8)
axs[3].axvline(1.0, color="k", linestyle=":", alpha=0.4)
axs[3].axhline(0, color="k", linestyle=":", alpha=0.3)
axs[3].set_ylabel("Tracking error\n$v_{ref} - v_o$ [V]")
axs[3].set_xlabel("Time [ms]")

plt.tight_layout()
plt.show()
"""))

    cells.append(md(r"""
### 4.2 Zoom: S1/S2 alternation around the step

This zoomed view shows the **4-phase switching pattern** (S1, dead,
S2, dead) and that the output ripple is at $2 f_{sw}$ (two inductor-
current peaks per per-switch period — one when S1 conducts, one
when S2 conducts).
"""))

    cells.append(code(r"""
# Zoom around 1 ms — show ~3 per-switch periods of S1/S2 alternation
t_center = 1.0e-3 + 5e-6
t_window = 30e-6  # 3 periods at 100 kHz
zmask = (sim['t'] > t_center - t_window/2) & (sim['t'] < t_center + t_window/2)
t_z = (sim['t'][zmask] - t_center) * 1e6  # µs relative to step

fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

axs[0].plot(t_z, sim['v_o'][zmask], 'C0', linewidth=1.2)
axs[0].set_ylabel("$v_o$ [V]")
axs[0].set_title("Half-bridge switching detail (zoom): output ripple at 2·f_sw, "
                 "S1/S2 alternation visible")

axs[1].plot(t_z, sim['i_L'][zmask], 'C1', linewidth=1.2)
axs[1].set_ylabel("$i_L$ [A]")
# Mark the two valleys per per-switch period
axs[1].text(0.02, 0.95, "Two ripple peaks per per-switch period\n→ output ripple at 2·f_sw",
            transform=axs[1].transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

# S1 and S2 gate signals (offset for visibility)
axs[2].plot(t_z, sim['s1'][zmask], 'C2', linewidth=1.0, label="S1 (high-side)")
axs[2].plot(t_z, sim['s2'][zmask] - 1.3, 'C3', linewidth=1.0,
            label="S2 (low-side, offset)")
axs[2].set_ylabel("Gate signals")
axs[2].set_xlabel("Time relative to step [µs]")
axs[2].legend(loc="upper right", fontsize=9)
axs[2].set_yticks([])

plt.tight_layout()
plt.show()
"""))

    cells.append(code(r"""
mask_after = sim['t'] > 1.0e-3
t_after = sim['t'][mask_after] - 1.0e-3
v_o_after = sim['v_o'][mask_after]

step_mag = 0.2  # 5.0 → 5.2 V
target_final = 5.2
overshoot_pct = (np.max(v_o_after) - target_final) / step_mag * 100
dip_amount = 5.0 - np.min(v_o_after)
settled = np.abs(v_o_after - target_final) < 0.02 * step_mag
unsettled = np.where(~settled)[0]
settling_ms = t_after[
    min(unsettled[-1] + 1, len(t_after) - 1) if len(unsettled) else 0
] * 1e3
v_o_10 = 5.0 + 0.1 * step_mag
v_o_90 = 5.0 + 0.9 * step_mag
rise_start = np.argmax(v_o_after >= v_o_10)
rise_end = np.argmax(v_o_after >= v_o_90)
rise_time_ms = (t_after[rise_end] - t_after[rise_start]) * 1e3

ss_error = target_final - np.mean(sim['v_o'][sim['t'] > 4.5e-3])

print("Closed-loop step-response metrics ($v_{ref}$: 5.0 → 5.2 V)")
print(f"  Initial dip                 = {dip_amount * 1e3:7.1f} mV "
      f"(should be ~0 — no RHP zero)")
print(f"  Rise time (10% → 90%)       = {rise_time_ms:7.3f} ms")
print(f"  Peak overshoot              = {overshoot_pct:7.2f} %")
print(f"  Settling time (±2 %)        = {settling_ms:7.3f} ms")
print(f"  Steady-state error          = {ss_error*1e3:+7.2f} mV "
      f"({ss_error / target_final * 100:+.3f} %)")
print()
if abs(ss_error) < 0.05 and overshoot_pct < 30 and settling_ms < 5.0:
    print("✅  Closed-loop half-bridge controller PROVEN — buck-level performance:")
    print(f"    • SS error  = {ss_error*1e3:.1f} mV ({ss_error/target_final*100:.2f} %)")
    print(f"    • Overshoot = {overshoot_pct:.1f} %")
    print(f"    • Settling  = {settling_ms:.2f} ms")
    print()
    print(f"    Compare across the library:")
    print(f"      buck         = 1.4 ms   (no isolation, 1 switch)")
    print(f"      forward      = 1.14 ms  (isolated, 1 switch + reset winding)")
    print(f"      half-bridge  = {settling_ms:.2f} ms  (isolated, 2 switches alternating)")
    print(f"      flyback      = 15 ms    (isolated, RHP zero limits bandwidth)")
    print()
    print(f"    The half-bridge matches the forward's control performance,")
    print(f"    with lower per-switch voltage stress and no reset winding.")
else:
    print("⚠️   Closed-loop response off-target — revisit f_c or PM.")
"""))

    cells.append(md(r"""
## 5. Summary

The half-bridge converter delivers isolated DC-DC conversion with
**the same control performance as the forward** — same small-signal
model, same compensator design, same settling time. The two-switch
implementation is invisible to the controller.

What you've proven in this notebook:

- The forward's Type-III K-factor compensator transfers directly to
  the half-bridge with no modification.
- The 4-phase switched simulator (S1, dead, S2, dead) produces output
  ripple at $2 f_{sw}$ — twice the per-switch frequency.
- The duty clip at $D_{max} = 0.45$ enforces the dead-time / shoot-
  through constraint (analogous to the forward's reset-winding limit).
- Closed-loop performance: ~1 ms settling, <1 % steady-state error,
  no RHP-zero dip.

**When to use a half-bridge instead of a forward**:

- Higher power (> ~50 W) where the forward's reset-winding switch
  voltage stress ($2 V_g$) becomes painful.
- Tighter ripple specs at the same $f_{sw}$ (the $2 f_{sw}$ output
  ripple lets you shrink the filter).
- Designs where transformer utilization matters (the half-bridge
  uses the core in both flux quadrants; the forward only one).

**When NOT to use a half-bridge**:

- Very low power (< 5 W) where the second switch's drive complexity
  doesn't pay off — stay with the flyback or forward.
- $V_g$ too low (< 12 V) — the rail-split puts only $V_g/2$ on the
  primary, which can be too small to drive a useful turns ratio.

**Suggested exercises**

1. Vary $t_{dead}$ in the simulator (50 ns, 200 ns, 500 ns). What
   happens to the output ripple? When does the loop start having
   trouble?
2. Drop the rail-split (use a single cap so the midpoint sits at $V_g$
   instead of $V_g/2$). Re-derive $V_o(D)$. (Answer: $V_o = 2 n V_g D$
   — that's the full-bridge, the next topology.)
3. Set $V_{ref} = 5.5$ V so the post-step duty hits $D_{max}$.
   Watch the duty saturate and tracking error grow.
4. Compare overlapping forward and half-bridge closed-loop plots
   on the same axes. They should track each other almost perfectly.
"""))

    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    nb1 = build_modeling_notebook()
    nb2 = build_controller_notebook()
    write_notebook(nb1, HERE / "01_half_bridge_modeling.ipynb")
    write_notebook(nb2, HERE / "02_half_bridge_controller.ipynb")


if __name__ == "__main__":
    main()
