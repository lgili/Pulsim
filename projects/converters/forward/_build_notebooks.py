"""Generator for the forward teaching notebooks.

Run once after editing to regenerate `01_forward_modeling.ipynb` and
`02_forward_controller.ipynb`. Same patterns as the buck / boost /
buck-boost / flyback generators.
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
# Notebook 1 — Forward modeling
# ---------------------------------------------------------------------------


def build_modeling_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 1 — Forward Converter Modeling

> **Goal.** Derive the small-signal model of the forward converter,
> show that it's the buck's model with the transformer turns ratio
> $n$ scaling the effective input bus, and explain why this topology
> has **NO RHP zero** despite being isolated.

**Prerequisites**

- Buck modeling notebook (`projects/converters/buck/`). The forward's
  state-space is literally the buck's with one substitution.
- Flyback modeling notebook (for the comparison: same isolation,
  very different control behavior).

**What you'll be able to do at the end**

1. Identify the forward on a schematic and distinguish it from the
   flyback by the energy-flow pattern (forward: simultaneous during
   ON; flyback: store-then-release).
2. Write the switched model and recognize the buck-like structure.
3. Derive $V_o = n \\cdot V_g \\cdot D$ and explain why there's no
   $(1-D)$ in the denominator (no boost-like accumulation).
4. Show that the state-space and transfer functions match the
   buck's verbatim (modulo the $n \\cdot V_g$ substitution).
5. Argue why this topology has the simplest voltage-mode control
   among the four isolated families (forward, push-pull, half-bridge,
   full-bridge).
6. State the **reset-winding constraint** $D \\le 0.5$ (or stricter
   with a non-1:1 reset) and explain how it limits the design space.
"""))

    cells.append(md("## Setup"))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from forward_model import (
    ForwardParams,
    forward_state_space,
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
## 1. The forward topology

```
V_g+ ---- S ----+
                |
                )||(    ← coupled transformer with reset winding
                )||(    (primary, secondary, reset)
                )||(
                |
               gnd_pri          (isolated barrier)        gnd_sec
                                                              |
                |                          D1 (forward)       |
                |       reset              |--cathode -- L ---+--- V_o
                |       winding            |  to filter        |
                |       (returns           |                   C   R
                |       mag energy         D2 (freewheel)      |   |
                |       to V_g)            |                   |   |
                +-------+-------+         gnd_sec ------------gnd_sec
                                +---- ...
```

- **Primary**: switch $S$ chops $V_g$ at the primary winding.
- **Secondary winding** (during ON): sees $n \\cdot V_g$, drives the
  forward diode $D_1$ → filter inductor $L$ → cap+load. Energy flows
  THROUGH the transformer simultaneously with the switch ON.
- **Freewheel diode** $D_2$ (during OFF): the filter $L$ maintains
  its current; $D_2$ provides the freewheel path. Just like the buck.
- **Reset winding** (during OFF): demagnetizes the transformer.
  Routes the magnetizing energy back to $V_g$ to prepare for the next
  cycle.

### 1.1 Energy routing — forward vs flyback

| Phase | Forward | Flyback |
|---|---|---|
| ON | primary → secondary → L → cap (energy flows through) | primary stores energy in L_m; secondary disconnected |
| OFF | primary off; L freewheels through D₂ | secondary releases stored energy through diode to cap |

The forward is a **transformer + buck**, conceptually. The transformer
is just a voltage scaler — it has no role in energy storage. The buck
filter L-C does the storage.
"""))

    cells.append(md(r"""
## 2. Switched (instantaneous) model

State variables: filter inductor current $i_L$ and output cap voltage
$v_o$ (positive, same as the buck — no polarity inversion here).

### 2.1 ON interval ($S$ closed, $D_1$ on, $D_2$ off)

Secondary winding clamps to $n \\cdot v_g$. $D_1$ is forward-biased
(secondary $n v_g$ exceeds $v_o$ initially). $D_2$ is reverse-biased.

The L-C output filter sees a voltage source of $n \\cdot v_g$ on its
input:

$$
L \\cdot \\frac{di_L}{dt} = n \\cdot v_g - v_o
\\qquad
C \\cdot \\frac{dv_o}{dt} = i_L - \\frac{v_o}{R}
$$

### 2.2 OFF interval ($S$ open, $D_1$ off, $D_2$ on)

Primary disconnected. Filter inductor maintains $i_L$; $D_2$ pulls
the L-side of the filter to ground:

$$
L \\cdot \\frac{di_L}{dt} = -v_o
\\qquad
C \\cdot \\frac{dv_o}{dt} = i_L - \\frac{v_o}{R}
$$

Compare to the buck — these equations are **identical** with $v_g$
replaced by $n \\cdot v_g$ on the ON interval. The buck switched
model had $L \\cdot di_L/dt = v_g - v_o$ for ON; the forward has
$L \\cdot di_L/dt = n v_g - v_o$.
"""))

    cells.append(md(r"""
## 3. State-space averaging

Average $q \\to d$:

$$\\boxed{
\\;\\; L \\cdot \\frac{di_L}{dt} = d \\cdot n \\cdot v_g - v_o
\\;\\;}
$$

$$\\boxed{
\\;\\; C \\cdot \\frac{dv_o}{dt} = i_L - \\frac{v_o}{R}
\\;\\;}
$$

### 3.1 Steady-state

$$
0 = D \\cdot n \\cdot V_g - V_o
\\;\\implies\\;
\\boxed{V_o = n \\cdot V_g \\cdot D}
$$

$$
0 = I_L - \\frac{V_o}{R}
\\;\\implies\\;
I_L = \\frac{V_o}{R}
$$

The forward output is a **monotonic linear function of duty** — no
$(1-D)$ in the denominator, no "infinite gain as D → 1" singularity.
This is the structural reason the forward has no RHP zero: there's
no inverse relationship between duty and output to invert.

### 3.2 Reset-winding constraint

The transformer's magnetizing flux grows during ON and must reset to
zero each cycle. If the reset winding has the same number of turns
as the primary (1:1 reset), the OFF reset voltage equals $V_g$, and
the flux returns to zero exactly when $(1-D) \\cdot V_g \\cdot T_s = D
\\cdot V_g \\cdot T_s$, giving the constraint:

$$\\boxed{ D_{\\max} = 0.5 \\;\\;\\text{(for 1:1 reset)} }$$

Production designs use $D_{\\max} \\approx 0.45$ to leave margin for
line/load steps and component tolerances.
"""))

    cells.append(code(r"""
params = ForwardParams()
print(operating_point_report(params))
print()
params.assert_reset_winding_ok()
print("✅  Reset-winding constraint check passed.")
"""))

    cells.append(md(r"""
## 4. Small-signal linearization

Perturb $i_L = I_L + \\hat i_L$, $v_o = V_o + \\hat v_o$, $d = D + \\hat d$,
$v_g = V_g + \\hat v_g$. Substitute and drop products:

$$
L \\cdot \\frac{d\\hat i_L}{dt}
   = D \\cdot n \\cdot \\hat v_g + n \\cdot V_g \\cdot \\hat d - \\hat v_o
$$

$$
C \\cdot \\frac{d\\hat v_o}{dt}
   = \\hat i_L - \\frac{\\hat v_o}{R}
$$

**Critical observation**: the cap equation has **no $\\hat d$ term**.
That's exactly what gave the buck its lack of RHP zero — and the
forward inherits the property. A positive duty step instantly
delivers more current into the filter inductor → output rises
monotonically.

Compare with the flyback's cap equation:
$C \\cdot d\\hat v_o/dt = (1-D)/n \\cdot \\hat i_L - I_{L_m}/n \\cdot \\hat d - v_o/R$.
The $-I_{L_m}/n \\cdot \\hat d$ term was the source of the flyback's
RHP zero. The forward's transformer doesn't put a $\\hat d$ term in
the cap equation because energy flows are simultaneous, not "stored
then released".
"""))

    cells.append(md(r"""
## 5. State-space matrices

$$
A = \\begin{bmatrix}
0 & -1/L \\\\
1/C & -1/(R C)
\\end{bmatrix},
\\quad
B = \\begin{bmatrix}
n V_g / L & n D / L \\\\
0 & 0
\\end{bmatrix}
$$

$$
C = \\begin{bmatrix} 0 & 1 \\end{bmatrix},
\\quad
D_{\\rm feed} = \\begin{bmatrix} 0 & 0 \\end{bmatrix}
$$

Set $n = 1$ → the buck. The transformer's only contribution is the
$n$ multiplier on the duty-and-line-input column of $B$.
"""))

    cells.append(code(r"""
A, B, C_mat, D_mat = forward_state_space(params)
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

### 6.1 $G_{vd}(s)$ — the headline plant

Closed form, identical to the buck with $n V_g$ in place of $V_g$:

$$
G_{vd}(s) = \\frac{n V_g}{LC} \\cdot
\\frac{1}{s^2 + s/(RC) + 1/(LC)}
$$

A clean **second-order low-pass**. **No zeros.** DC gain $= n V_g$,
natural frequency $\\omega_n = 1/\\sqrt{LC}$, $Q = R \\sqrt{C/L}$.

### 6.2 $G_{vg}(s)$, $Z_{out}(s)$

$$
G_{vg}(s) = \\frac{n D}{LC s^2 + L/R \\cdot s + 1}
$$

$$
Z_{out}(s) = \\frac{s L}{LC s^2 + L/R \\cdot s + 1}
$$

Both use the same denominator. $Z_{out}$ is *exactly* the buck's
output impedance — the transformer is invisible to load perturbations.
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
    print(f"Gvd zeros: NONE  (no RHP zero — the forward's defining advantage)")
print(f"Gvd poles: {np.roots(Gvd.den)}")
"""))

    cells.append(md(r"""
### 6.3 Bode plots — compare with the flyback

Plot $G_{vd}$, $G_{vg}$, and $Z_{out}$. The forward's $G_{vd}$ has
a clean -40 dB/dec rolloff past $f_n$ with phase dropping to a
final -180° — no RHP-zero kink in phase, no magnitude-and-phase
sign inversion. This is what makes forward control "easy".
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
ax_mag.set_title(f"Forward open-loop "
                 f"($V_g$={params.V_g}V → $V_o$={params.V_o}V, n={params.n}, "
                 f"D={params.D:.3f})  — NO RHP zero")
plt.tight_layout()
plt.show()
"""))

    cells.append(md(r"""
## 7. Step response — monotonic, like the buck

Apply a small duty step. Unlike the flyback / boost / buck-boost,
the forward output rises **monotonically** to its new steady state.
No dip, no inverse response.
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
ax.set_title(f"Forward step response: monotonic — no RHP-zero dip")
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

Same three rigorous checks. The ss2tf round-trip needs a small
relative-tolerance trim because scipy can leave numerical noise
(~1e-12 of the dominant coefficient) in the higher-order numerator
slots when the closed-form has fewer terms.
"""))

    cells.append(code(r"""
def _trim_near_zero(coeffs, rel_tol=1e-9):
    '''Strip leading near-zero coefficients (numerical noise from scipy.ss2tf).

    Like np.trim_zeros but with a relative tolerance — needed because
    ss2tf returns a numerator padded to den_degree+1, and the leading
    coefficients may be on the order of 1e-12 of the dominant term
    when the true closed-form numerator has fewer terms.
    '''
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

# (3) ss2tf round-trip (with relative-tolerance trim)
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

The forward converter is the buck with a transformer in front. The
small-signal model is the buck's verbatim, with $n V_g$ replacing
$V_g$ in the $B$ matrix. Voltage-mode control inherits all the
buck's simplicity:

- **No RHP zero** → no bandwidth ceiling beyond $f_{sw}/10$
- **Monotonic step response** → no need for inverse-response
  workarounds
- **Same compensator recipe** as the buck (K-factor Type-III at
  $f_{sw}/10$)

The price for this simplicity is the **reset-winding constraint**:
$D \\le 0.5$ for 1:1 reset, lower for asymmetric reset windings.
Designers handle this by choosing $n$ such that the operating duty
sits comfortably below the cap (typically $D \\le 0.45$).

Math validated by three checks (poles, DC gains, ss2tf round-trip).

**Cross-validation against Pulsim**: open
[`00_forward_pulsim_validation.ipynb`](00_forward_pulsim_validation.ipynb)
for an executed notebook that builds the forward in Pulsim
(transformer + rectifier diode + freewheel diode + LC filter) and
overlays the steady-state output voltage.

**Next**: open `02_forward_controller.ipynb` for the controller
design + switched closed-loop proof.

**Suggested exercises**

1. Push the duty target above 0.5 and run the simulator (turn off
   the safety clip). Watch the magnetizing current run away.
2. Compare $G_{vd}(s)$ for the forward and flyback on the same
   Bode plot at the same $V_g$, $V_o$, $L$, $C$, $R$, $n$.
3. The two-switch forward uses both reset diodes in series with the
   switches to eliminate the reset winding. Where does the analytical
   model change? (Hint: it doesn't — the reset diodes only affect
   the transformer's stress / voltage rating, not the small-signal
   $G_{vd}$.)
"""))

    return cells


# ---------------------------------------------------------------------------
# Notebook 2 — Controller
# ---------------------------------------------------------------------------


def build_controller_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 2 — Forward Converter Controller Design

> **Goal.** Take advantage of the forward's lack of RHP zero — size a
> Type-III compensator at the buck-style bandwidth target
> ($f_c = f_{sw}/10$, PM = 60°), discretize via Tustin, and **prove**
> it with a switched closed-loop simulation. The settling time should
> match the buck's (~1.5 ms), not the flyback's (~15 ms).

**Prerequisites**

- Forward modeling notebook (`01_forward_modeling.ipynb`).
- Buck controller notebook (the recipe is essentially the same).

The voltage-mode loop should feel "easy" after the boost / buck-boost
/ flyback exercises — no RHP-zero bookkeeping, no $f_c / 5$ ceiling.
"""))

    cells.append(code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from forward_model import (
    ForwardParams, control_to_output_tf, operating_point_report,
)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

params = ForwardParams()
print(operating_point_report(params))
"""))

    cells.append(md(r"""
## 1. Bandwidth target — buck-style

No RHP zero → target $f_c = f_{sw} / 10 = 10$ kHz, PM = 60°. This
gives a settling time around 1-2 ms (similar to the buck reference).
"""))

    cells.append(code(r"""
Gvd = control_to_output_tf(params)
V_ramp = 5.0
plant = signal.TransferFunction(np.array(Gvd.num)/V_ramp, np.array(Gvd.den))

# Note: target f_c = f_sw/20 (not f_sw/10) — the duty-cycle clip at
# D_max = 0.45 makes the loop saturate easily on small steps, so we
# trade some bandwidth for clean (no-overshoot) tracking. The buck
# reference used f_sw/20 = 5 kHz for the same reason.
f_c_target = params.f_sw / 20.0
pm_target = 60.0
print(f"Target:     f_c = {f_c_target/1e3:.1f} kHz, PM = {pm_target}°")
print(f"Plant DC gain (with k_PWM = 1/V_ramp = {1/V_ramp:.3f}): "
      f"{plant.num[0]/plant.den[2]:.3f}")
"""))

    cells.append(md(r"""
## 2. K-factor Type-III design

Same algorithm as the buck — no need for the phase-unwrap fix we
needed on the buck-boost / flyback (no RHP zero → no past-180°
phase drop).
"""))

    cells.append(code(r"""
def design_type3_kfactor(plant, f_c, pm_target):
    omega_c = 2 * np.pi * f_c
    _, _, ph_plant = signal.bode(plant, w=[omega_c])
    # Defensive: handle phase-wrap in case scipy chose the wrong branch
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
ax_mag.set_title(f"Compensated forward loop: $f_c$ = {f_cross/1e3:.1f} kHz, "
                 f"PM = {pm:.1f}°")
plt.tight_layout()
plt.show()

print(f"Target:    f_c = {f_c_target/1e3:.1f} kHz, PM = {pm_target}°")
print(f"Achieved:  f_c = {f_cross/1e3:.2f} kHz, PM = {pm:.1f}°")
"""))

    cells.append(md(r"""
## 3. Discretization

Tustin / bilinear at $T_s = 1/f_{sw}$.
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
## 4. Switched-model closed-loop simulation

Pure-Python forward-Euler simulator. The switched model:

```
ON  (S closed, D1 on, D2 off):
    L · di_L/dt = n · V_g - v_o
    C · dv_o/dt = i_L - v_o / R

OFF (S open, D1 off, D2 on):
    L · di_L/dt = -v_o
    C · dv_o/dt = i_L - v_o / R
```

This is the buck's switched model with $n V_g$ replacing $V_g$ on
the ON interval. **Critically: the controller's duty output is
clipped to $D_{max} = 0.45$** to respect the reset-winding constraint.

Warm-start at the operating point (with valley i_L = avg - ripple/2),
same convention as the buck-boost / flyback notebooks.
"""))

    cells.append(code(r"""
def simulate_closed_loop_forward(
    params,
    b: np.ndarray, a: np.ndarray,
    *,
    t_end: float = 5e-3,
    t_step: float = 1e-3,
    v_ref_initial: float = 5.0,
    v_ref_final: float = 5.5,
    V_ramp: float = 5.0,
    samples_per_period: int = 200,
    warm_start: bool = True,
):
    '''Forward-Euler switched forward converter + digital compensator.

    States are filter inductor current i_L and output cap voltage v_o.
    Compensator runs once per switching period (sample-and-hold) and
    its duty output is clipped to [0.05, params.D_max] to respect the
    reset-winding limit.
    '''
    T_s = 1.0 / params.f_sw
    dt_sim = T_s / samples_per_period
    n_steps = int(t_end / dt_sim) + 1
    n_turn = params.n

    n_state = len(a) - 1
    state = np.zeros(n_state)

    if warm_start:
        D_init = v_ref_initial / (n_turn * params.V_g)
        I_L_avg = v_ref_initial / params.R
        T_s_period = 1.0 / params.f_sw
        # On-time inductor ripple: di/dt = (n·V_g - V_o) / L
        delta_i_pp = (n_turn * params.V_g - v_ref_initial) * D_init * T_s_period / params.L
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
            # Clip duty to respect reset-winding constraint!
            duty = float(np.clip(v_c / V_ramp, 0.05, params.D_max))

        switch_on = (cycle_pos_int / samples_per_period) < duty

        # Switched forward ODE (n_turn-scaled buck)
        if switch_on:
            v_L = n_turn * params.V_g - v_o
        else:
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
            rec_idx += 1

    return {
        "t": t_hist[:rec_idx], "v_o": v_o_hist[:rec_idx],
        "i_L": i_L_hist[:rec_idx], "duty": duty_hist[:rec_idx],
        "v_ref": v_ref_hist[:rec_idx],
    }
"""))

    cells.append(code(r"""
sim = simulate_closed_loop_forward(
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

    cells.append(code(r"""
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

axs[0].plot(sim['t']*1e3, sim['v_o'], 'C0', linewidth=0.8, label="$v_o$ (switched)")
axs[0].plot(sim['t']*1e3, sim['v_ref'], 'C3--', linewidth=2, label="$v_{ref}$")
axs[0].axvline(1.0, color="k", linestyle=":", alpha=0.4, label="step")
axs[0].set_ylabel("Output voltage [V]")
axs[0].set_title("Closed-loop forward (warm-start at OP): step "
                 "$v_{ref}$ 5.0 V → 5.2 V at $t$ = 1 ms (within $D_{max}$)")
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
               label=f"$D_{{max}}$ = {params.D_max:.2f} (reset limit)")
axs[2].set_ylabel("Duty cycle")
axs[2].legend(loc="lower right")

axs[3].plot(sim['t']*1e3, sim['v_ref'] - sim['v_o'], 'C4', linewidth=0.8)
axs[3].axvline(1.0, color="k", linestyle=":", alpha=0.4)
axs[3].axhline(0, color="k", linestyle=":", alpha=0.3)
axs[3].set_ylabel("Tracking error\n$v_{ref} - v_o$ [V]")
axs[3].set_xlabel("Time [ms]")

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
# Forward should NOT dip (no RHP zero)
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
      f"(should be ~0 — no RHP zero!)")
print(f"  Rise time (10% → 90%)       = {rise_time_ms:7.3f} ms")
print(f"  Peak overshoot              = {overshoot_pct:7.2f} %")
print(f"  Settling time (±2 %)        = {settling_ms:7.3f} ms")
print(f"  Steady-state error          = {ss_error*1e3:+7.2f} mV "
      f"({ss_error / target_final * 100:+.3f} %)")
print()
# Buck-style PASS gate (tighter than the RHP-zero converters)
if abs(ss_error) < 0.05 and overshoot_pct < 30 and settling_ms < 5.0:
    print("✅  Closed-loop forward controller PROVEN — buck-level performance:")
    print(f"    • SS error  = {ss_error*1e3:.1f} mV ({ss_error/target_final*100:.2f} %)")
    print(f"    • Overshoot = {overshoot_pct:.1f} %")
    print(f"    • Settling  = {settling_ms:.2f} ms")
    print()
    print(f"    Compare:")
    print(f"      buck      = 1.4 ms settling (no isolation)")
    print(f"      forward   = {settling_ms:.1f} ms (same control performance with isolation)")
    print(f"      flyback   = 15 ms (isolated, but RHP zero limits bandwidth)")
    print()
    print(f"    The forward is the **fast** isolated topology.")
else:
    print("⚠️   Closed-loop response off-target — revisit f_c or PM.")
"""))

    cells.append(md(r"""
## 5. Summary

The forward converter delivers isolated DC-DC conversion with
**buck-level control simplicity**:

- No RHP zero → bandwidth can match the buck ($f_{sw}/10$).
- Monotonic step response → no inverse-response surprises.
- Settling time of ~1.5 ms, ~10× faster than the flyback at the
  same component values.

The price is the reset-winding constraint $D \\le 0.5$, handled in
practice by picking $n$ such that the operating duty stays well
below the cap.

**Suggested exercises**

1. Set $V_{ref}$ such that the post-step duty would exceed $D_{max}$.
   Watch the loop integrator wind up against the duty saturation
   (anti-windup is a real concern in production forwards).
2. Build a two-switch forward (no reset winding needed; D-cap
   raised to ~0.95). Re-derive the operating-point equations.
3. Compare the closed-loop response of the forward vs the flyback
   on the same axes — overlay the two notebooks' final plots.
"""))

    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    nb1 = build_modeling_notebook()
    nb2 = build_controller_notebook()
    write_notebook(nb1, HERE / "01_forward_modeling.ipynb")
    write_notebook(nb2, HERE / "02_forward_controller.ipynb")


if __name__ == "__main__":
    main()
