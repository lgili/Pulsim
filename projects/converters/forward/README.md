# Forward Converter — Reference Project

Fifth entry under `projects/converters/`. The **isolated buck**.

## Why the forward

The forward converter pairs naturally with the flyback:

| Topology | Non-isolated origin | Has RHP zero? | Notebook bandwidth |
|---|---|---|---|
| Flyback | buck-boost | **yes** | RHP-zero-limited |
| **Forward** | **buck** | **NO** | $f_{sw}/10$ (buck-like) |

Both deliver isolation via a transformer, but they route the energy
differently:

- **Flyback**: stores energy in the transformer's magnetizing field
  during ON, releases through the secondary during OFF. The
  transformer doubles as the energy storage element. Has a coupled
  RHP zero (same mechanism as the boost / buck-boost).

- **Forward**: transfers energy from primary to secondary
  *simultaneously* during ON; the secondary winding's voltage drives
  an explicit output filter L-C. During OFF, a freewheel diode lets
  the filter inductor maintain current — exactly like a buck. No RHP
  zero.

The forward inherits all of the buck's control simplicity, just with
$n \cdot V_g$ as the effective input voltage. Voltage-mode is easy.

## The price of "buck simplicity": the reset winding

The forward transformer would saturate after a few cycles if you did
nothing to demagnetize it. Each ON interval pushes flux up; the OFF
interval has to push it back down. Without explicit help, the OFF
flux doesn't reset, magnetizing current grows unbounded, switch sees
huge voltage spikes, and the converter destroys itself.

The standard fix: a **reset winding** (a third winding on the
transformer that conducts during OFF and routes the magnetizing
current back to the input bus). With a 1:1 reset winding, the duty
cycle is capped at $D \le 0.5$ — the design constraint that
distinguishes the forward from a "free" buck.

The notebooks call this out and clip the controller's duty output
at $D_{max} = 0.45$ for safety margin.

## Files

| File | Role |
|---|---|
| `01_forward_modeling.ipynb` | Derivation, transformer reset, state-space matching the buck, self-consistency. |
| `02_forward_controller.ipynb` | K-factor Type-III + switched closed-loop simulation. |
| `forward_model.py` | `ForwardParams` + state-space + 3 reference TFs + reset-winding check. |
| `_build_notebooks.py` | Generator. |

## Reference parameters

`ForwardParams()` default: 24 V → 5 V at 1 A (5 W), 1:0.5 transformer
($n = 0.5$), 100 µH filter inductor, 100 µF cap, 100 kHz switching.

- $D \approx 0.417$ — comfortable margin below the 0.45 cap
- $f_n \approx 1.6$ kHz, $Q \approx 5$ (same as the buck reference)
- **No RHP zero** — bandwidth target $f_{sw}/10 = 10$ kHz

## What you'll learn

- The state-space of the forward converter is *identical to the buck's*
  with $n \cdot V_g$ substituted for $V_g$ — the transformer is a pure
  voltage scaler in the average model.
- Why production "buck-derived" isolated topologies (forward,
  half-bridge, push-pull, full-bridge) are far easier to control than
  "buck-boost-derived" ones (flyback, single-ended forward) — they
  share the buck's freedom from RHP zeros.
- The reset-winding constraint and how the design D-cap interacts
  with line / load variation margins.

## Comparison across the library

| Converter | Settling | RHP zero | Isolation? | Topology family |
|---|---|---|---|---|
| buck | 1.4 ms | none | no | buck |
| boost | 25 ms | 4.6 kHz | no | boost |
| buck-boost | 25 ms | 9.5 kHz | no | buck-boost |
| flyback | 15 ms | 12.7 kHz | yes | buck-boost (isolated) |
| **forward** | **~1.4 ms (predicted)** | **none** | **yes** | **buck (isolated)** |

The forward should land near the buck's settling time — the transformer
is invisible to control.

## How to run

```bash
pip install numpy scipy matplotlib
jupyter lab projects/converters/forward/01_forward_modeling.ipynb
```

## Bibliography

- Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd
  ed., **Section 6.2** (forward converter derivation) and **8.2**
  (control of buck-derived isolated topologies).
- AN-3370 / TI / OnSemi forward design guides — practical considerations
  (leakage clamp, peak voltage rating, reset techniques) beyond the
  idealized analytical model.
