# Three-Phase VSI — Reference Project

First entry under `projects/inverters/` — the canonical
**DC → AC** converter. 6 switches in 3 legs producing a balanced
three-phase output from a DC bus.

## Why the 3-phase VSI

Power-electronics catalogues converge on the 3-phase VSI for any
application above a few kW: motor drives, grid-tie PV inverters,
industrial UPS, EV traction, wind-turbine converters. It's the
"workhorse" of medium-power conversion.

It's also where students meet **three new ideas simultaneously**:

1. **Reference-frame transforms** (Clarke and Park) — three time-varying
   AC quantities become two DC quantities in the rotating dq frame.
   Suddenly the inverter looks like a *buck* on each axis. This is
   probably the single most important didactic moment in this whole
   library.

2. **Space Vector PWM (SVPWM)** — the 3-leg interleaving achieves
   ~15% more output voltage for the same DC bus than running three
   independent SPWMs. We use the **min-max injection** formula
   (mathematically equivalent to vector-based SVPWM but a one-liner).

3. **Grid synchronization** — when the inverter is connected to a
   stiff AC grid (not a passive load), you need a **PLL** to lock
   onto the grid angle. The PLL is itself a feedback loop running
   inside the controller.

## The dq-frame insight

In the rotating dq frame, a balanced 3-phase positive-sequence
signal becomes a **constant** (with $V_d$ = peak and $V_q$ = 0 if
the d-axis aligns with phase-a at $t = 0$). The inverter's LC
filter, expressed in dq, has exactly the buck converter's
small-signal model. Years of buck-control intuition transfer
verbatim.

This means: with the abc → αβ → dq pipeline, three-phase AC control
collapses into two parallel **DC regulation problems** with familiar
plants. The transform pipeline is the workhorse, not new control
math.

## Two operating modes

| Mode | Stand-alone | Grid-tie |
|---|---|---|
| Source | DC bus | DC bus |
| Sink | Isolated RL load | Stiff AC grid |
| Controlled quantity | Output **voltage** | Output **current** |
| Frequency comes from | Internal oscillator | **PLL** locked to grid |
| Angle estimation | None (we generate $\theta$) | SRF-PLL on grid voltage |
| Power direction | DC → AC always | Bidirectional ($P, Q$) |
| Plant in dq | Buck-like LC ($V/D$) | First-order $L_{grid}$ ($I/V$) |

## Files

| File | Role |
|---|---|
| `vsi_3phase_model.py` | `VSI3PhaseParams` + Clarke/Park transforms + SVPWM via min-max + three switched simulators + THD/fundamental helpers |
| `01_vsi_basics.ipynb` | Topology, 8 switching states, SPWM (independent legs), introduction to Clarke + Park transforms |
| `02_vsi_svpwm.ipynb` | SVPWM via vector decomposition AND via min-max injection (proved equivalent), open-loop switched demo, spectrum analysis |
| `03_vsi_standalone.ipynb` | dq-frame voltage-loop plant (= buck), PI design, closed-loop switched sim with RL load |
| `04_vsi_gridtie.ipynb` | SRF-PLL for grid synchronization, dq current loop with feed-forward, closed-loop sim injecting active + reactive power |
| `_build_notebooks.py` | Generator |

## Reference parameters

`VSI3PhaseParams()` defaults: 400 V DC → 230 V line-to-line rms at
60 Hz (Brazilian / EUA-3φ), 500 W, 1 mH + 10 µF LC output filter
per phase, 20 kHz switching. The 60-Hz operating point lets the
PFC's 400 V output feed the inverter directly — same library, full
power chain in one walk-through.

- $m_a \approx 0.939$ — comfortable margin below SVPWM limit of 1.0
- LC corner: $f_c \approx 1.59$ kHz, $Q \approx 10.6$
- $f_{sw}/f_c \approx 12.6$× — adequate switching-ripple attenuation
- Grid parameters: $L_{grid} = 2$ mH, $R_{grid} = 0.05$ Ω

## What you'll learn

- The 3-phase VSI **topology** and its 8 switching states (6 active
  + 2 zero), corresponding to the 6 outer points and 2 origin points
  of the space vector hexagon.
- The **Clarke and Park transforms** in derivation, code, and
  intuition: what's `2/3` doing in front, why power-invariant vs
  amplitude-invariant variants exist, what positive vs negative
  sequence look like in dq.
- **SVPWM via min-max injection** — derive the formula from sector
  identification and dwell-time decomposition; show it's equivalent
  to subtracting the common-mode of `(max+min)/2` from each phase
  reference; verify the 15% output bonus.
- **Stand-alone voltage control** in dq: PI design on the buck-like
  plant, the role of the LC filter's high Q, why R-L filter damping
  matters, how to add active or passive damping.
- **Grid-tie current control** in dq: the SRF-PLL architecture
  (project to dq with PLL angle, PI on $v_q$ → frequency correction
  → integrate → angle), why cross-coupling between $i_d$ and $i_q$
  exists (the $\omega L$ term) and how to decouple it with feed-
  forward, how dq references map to (P, Q) injection.
- Why **PLL stability** is its own non-trivial design problem
  (especially with weak grids).

## Comparison across the library

| Project | I/O | Switches | Loops | Control quantity |
|---|---|---|---|---|
| 6 DC-DC converters | DC → DC | 1 or 2 | 1 voltage | output voltage |
| boost PFC DCM | AC → DC | 1 | 1 (slow) | output voltage |
| boost PFC CCM | AC → DC | 1 | 2 (current + voltage) | output voltage |
| **VSI stand-alone** | **DC → AC** | **6** | **1 voltage in dq (or cascade)** | **3-phase output voltage** |
| **VSI grid-tie** | **DC ↔ AC** | **6** | **PLL + current loop in dq** | **active and reactive power** |

## How to run

```bash
pip install numpy scipy matplotlib
jupyter lab projects/inverters/vsi_3phase/01_vsi_basics.ipynb
```

## Bibliography

- Mohan, Undeland & Robbins, *Power Electronics*, 3rd ed., **Chapter 8**
  (VSI + SVPWM + dq frame). The classroom reference for this material.
- Yazdani & Iravani, *Voltage-Sourced Converters in Power Systems*,
  Wiley 2010 — the comprehensive grid-tie reference, including PLL
  design and weak-grid considerations.
- Holmes & Lipo, *Pulse Width Modulation for Power Converters*, IEEE
  Press 2003 — exhaustive spectral analysis of every modulation
  scheme (SPWM, SVPWM, DPWM variants).
- Teodorescu, Liserre & Rodríguez, *Grid Converters for Photovoltaic
  and Wind Power Systems*, Wiley 2011 — application-driven treatment
  of grid-tie inverters with practical code.
