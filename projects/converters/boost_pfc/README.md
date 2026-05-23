# Boost PFC — Reference Project (DCM **and** CCM)

Seventh entry under `projects/converters/`. **First AC-input converter**,
**first multi-loop controller**, and **first time the control objective
is "shape the input current"** (in addition to "regulate the output").

## Why PFC

A garden-variety rectifier + bulk cap front end pulls input current
**only at the line peaks** — narrow tall pulses. The result is PF ≈ 0.6
and THD > 100%, which fails every modern regulation (IEC 61000-3-2,
ENERGY STAR, 80 PLUS) for any AC-powered equipment above ~75 W.

The boost PFC fixes that by inserting a boost stage **after** the diode
bridge and forcing the inductor current to track the rectified line:

```
  v_ac(t) ──┤                                          + ──── V_o (400 V DC)
            │ diode-bridge   ┌────┐    ┌──── D ────┬───
  v_ac(t) ──┤                │  L │    │           │
            │  v_g = |v_ac|  └────┘    S          C   R_load
            │                            │           │
            └──────────────────────────────┴──────────┴── ─
```

Done right, $i_{in}(t)$ becomes a near-perfect copy of $v_{ac}(t)$ →
PF > 0.99, THD < 5%, and the converter looks "resistive" to the line.

## Two strategies — DCM and CCM

The boost PFC stage can run in either conduction mode, and the
implications for the controller are huge:

| Property | **DCM** (natural PFC) | **CCM** (active PFC) |
|---|---|---|
| Inductor $L$ | small (~100 µH at 100 W) | large (~1 mH at 100 W) |
| $i_L$ shape per $T_{sw}$ | triangle, returns to 0 | trapezoidal, ripples around an average > 0 |
| Input current shaping | **automatic** if $D$ ≈ const | needs active current loop |
| Current loop? | **none** — just a voltage loop | yes — fast inner loop |
| Multiplier? | no | yes ($i_{ref} = K \cdot i_{ref,amp} \cdot |v_{ac}|$) |
| Peak $i_L$ | high (2× CCM at same power) | moderate |
| EMI noise | high (sharp triangle peaks) | low (continuous current) |
| Has RHP zero? | **no** (first-order plant!) | yes (full boost RHP zero) |
| Typical use | up to ~150 W | mid- to high-power (200 W +) |

The didactic punchline: **DCM PFC has a single first-order plant —
the easiest small-signal model in the library. CCM PFC inherits all
of the (non-isolated) boost converter's small-signal pain (RHP zero,
duty-dependent gain) plus the line-frequency disturbance.**

## The universal-input puzzle

With 90–265 V AC range, the boost ratio $V_o/V_{g,pk}$ swings from
3.14 (at 90 V) to 1.07 (at 265 V). The duty cycle in DCM swings
from ~0.4 to ~0.1, and the cusp-distortion correction factor
$F(m_{pk})$ grows from 1.4 to 16 across the range. This is what
makes universal-input PFC a real design challenge.

The notebooks cover this in detail: at low line the converter has
clean DCM; at high line it slips into CCM near the peak. Production
DCM PFC chips (L6562 family etc.) handle this with **transition-mode
control** (CrM) — variable $T_{sw}$ that keeps $i_L$ on the DCM/CCM
boundary throughout the line cycle. We don't model CrM here; the
two notebooks show what pure-DCM and pure-CCM look like.

## Files

| File | Role |
|---|---|
| `01_boost_pfc_basics.ipynb` | What is PFC, why we need it, rectification, PF and THD definitions, DCM vs CCM overview. |
| `02_boost_pfc_dcm.ipynb` | DCM analysis (equivalent resistance $R_e$, cusp distortion, single-loop voltage compensator). Switched closed-loop simulation. |
| `03_boost_pfc_ccm.ipynb` | CCM analysis (current loop + voltage loop + multiplier). Switched closed-loop simulation. |
| `04_boost_pfc_simulation.ipynb` | Full DCM ⇄ CCM comparison: PF, THD, output ripple, line-step and load-step responses, side-by-side waveforms. |
| `boost_pfc_model.py` | `BoostPFCParams` (with `.dcm_design()` and `.ccm_design()` factories) + DCM and CCM analytical models + power-quality helpers + three switched simulators. |
| `_build_notebooks.py` | Generator. |

## Reference parameters

`BoostPFCParams.dcm_design()` (DCM-friendly) and `.ccm_design()`
(CCM-friendly) defaults: 90–265 V AC, 50 Hz, 400 V DC output, 100 W
load, 100 kHz switching, 220 µF bulk cap.

- **DCM** uses $L = 100$ µH → DCM at low/medium line, slips into
  boundary CCM near the peak at high line (>180 V).
- **CCM** uses $L = 1$ mH → CCM throughout the line cycle at all
  voltages.

Hold-up: 220 µF holds $V_o$ above 340 V for one line cycle at 100 W
(IEC requirement for non-LED PFC). 2·f_line ripple: ~7 V pk-pk at
full load (well below the 20 V budget).

## What you'll learn

- Why a "passive PFC" (bulk cap behind a bridge) fails IEC limits
  and how the boost stage fixes it.
- How DCM achieves natural PFC: with constant $D$ over a line
  half-cycle, the average input current $\langle i_g \rangle$ is
  proportional to $v_g$ (with a small cusp correction).
- The exact duty formula for DCM PFC (linearized resistive-emulator
  vs the numerical solve including cusp distortion).
- Why DCM PFC has a **first-order** small-signal plant while CCM
  PFC has the boost's RHP zero — and why that makes DCM's voltage
  loop trivially easy.
- The two-loop CCM architecture: fast current loop (integrator
  plant) + slow voltage loop + multiplier that converts an "amplitude
  command" into a "shape command".
- The hard constraint on **outer voltage loop bandwidth**: must be
  $\ll 2 f_{line}$ (typically 5–20 Hz at 50 Hz line) so the loop
  doesn't try to "correct" the 2·$f_{line}$ output ripple by chopping
  the input current.
- Standard PQ metrics (PF, THD) and the role of the **line-band
  filter** that PQ meters apply before measurement (without it,
  switching-frequency ripple makes PF look artificially low).

## Comparison across the library

| Converter | Settling | RHP zero | Input | Loops |
|---|---|---|---|---|
| buck | 1.4 ms | none | DC | 1 voltage |
| boost | 25 ms | 4.6 kHz | DC | 1 voltage |
| buck-boost | 25 ms | 9.5 kHz | DC | 1 voltage |
| flyback | 15 ms | 12.7 kHz | DC | 1 voltage |
| forward | 1.14 ms | none | DC | 1 voltage |
| half-bridge | 1.19 ms | none | DC | 1 voltage |
| **boost PFC (DCM)** | **~100 ms** | **none** | **AC** | **1 voltage (slow)** |
| **boost PFC (CCM)** | **~100 ms** | **yes (4 kHz)** | **AC** | **2 (current fast + voltage slow)** |

PFC settling is "slow" by DC-DC standards because the voltage loop
must be slower than the 100/120 Hz output ripple — not because the
plant is slow.

## How to run

```bash
pip install numpy scipy matplotlib
jupyter lab projects/converters/boost_pfc/01_boost_pfc_basics.ipynb
```

## Bibliography

- Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd
  ed., **Chapter 18** (low-harmonic rectifiers): Sec. 18.1
  (PFC overview), Sec. 18.2 (DCM averaged model + equivalent
  resistance), Sec. 18.4 (CCM average current-mode + multiplier).
- Mohan, Undeland, Robbins, *Power Electronics*, 3rd ed., Chapter 18
  (worked examples with universal-mains designs).
- ON Semiconductor / Infineon / TI PFC application notes (L6562,
  NCP1607, UCC28019, FAN6982) — practical considerations beyond the
  idealized model: leakage clamp, NTC inrush limiter, current sense
  burden, brownout detection.
- IEC 61000-3-2 — harmonic-current limits that PFC stages are
  required to meet.
