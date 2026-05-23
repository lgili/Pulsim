# Three-Phase NPC 3-Level Inverter — Reference Project

Second entry under `projects/inverters/` — the canonical
**multilevel** converter. The Neutral-Point Clamped (NPC) inverter
generates **three** voltage levels at each phase-to-neutral output
($+V_{dc}/2$, $0$, $-V_{dc}/2$) instead of the two of a standard VSI
($+V_{dc}/2$, $-V_{dc}/2$).

## Why multilevel — why NPC

Multilevel inverters dominate medium-voltage drives, large PV
inverters (1500 V class), HVDC light, and EV traction at the high end
because more levels mean:

1. **Lower output harmonic distortion** at the same switching
   frequency — the THD scales roughly as $1/(N-1)$ where $N$ is the
   number of levels. Three-level → halves the THD of a two-level
   inverter at the same $f_{sw}$.
2. **Lower switch voltage stress** — each switch sees only $V_{dc}/2$
   instead of $V_{dc}$, halving its $V_{DS}$ rating. Enables
   medium-voltage operation with standard low-voltage silicon.
3. **Lower $dv/dt$ on the load** — three smaller steps instead of
   one full $V_{dc}$ step. Friendly to motor winding insulation and
   common-mode currents in long cables.
4. **The same $f_{sw}$ buys ~2× output bandwidth** (or alternately:
   ~½ the $f_{sw}$ for the same output quality, halving switching
   losses).

The **Neutral-Point Clamped (NPC)** topology of Nabae, Takahashi &
Akagi (1981) is the textbook three-level converter. Each leg has
**four switches** in series and **two clamping diodes** that pin the
midpoint to the bus neutral.

## NPC leg topology

```
        Vdc_pos (+V_dc/2)
            │
           S1
            │
            n1 ─── D_clamp_top (→ NP)
            │
           S2
            │
           mid_X  (phase output)
            │
           S3
            │
            n2 ─── D_clamp_bot (← NP)
            │
           S4
            │
        Vdc_neg (-V_dc/2)
```

The DC bus is split by two equal capacitors into a positive half
($+V_{dc}/2$) and a negative half ($-V_{dc}/2$), with the
**neutral point (NP)** at the midpoint (0 V w.r.t. system ground).

## Switching states (per leg)

Only three of the $2^4 = 16$ switch combinations are valid:

| State | $S_1$ | $S_2$ | $S_3$ | $S_4$ | $v_{mid}$ vs gnd |
|:-:|:-:|:-:|:-:|:-:|:-:|
| **P** | ON | ON | OFF | OFF | $+V_{dc}/2$ |
| **O** | OFF | ON | ON | OFF | $0$ (neutral) |
| **N** | OFF | OFF | ON | ON | $-V_{dc}/2$ |

Any other combination either short-circuits the DC bus or violates
the diode clamping. The control law enforces only these three states.

## Three new ideas (vs the 2-level VSI)

1. **Multicarrier modulation (PD / POD / APOD)** — to generate the
   three voltage levels we need *two* triangular carriers stacked
   vertically. The reference signal is compared against both; whichever
   side it crosses determines the state P / O / N.
2. **Neutral-point voltage balancing** — the NP carries a real current
   whenever any phase is in the O state. Without active balancing, the
   NP voltage drifts away from $V_{dc}/2$, ultimately saturating one
   half of the DC bus. This is the **defining control challenge** of
   the NPC.
3. **Line-to-line voltage has 5 levels** — $\{+V_{dc}, +V_{dc}/2, 0,
   -V_{dc}/2, -V_{dc}\}$. The FFT spectrum is dramatically cleaner
   than the 2-level's three levels — visible directly in the time
   waveform.

## Files

| File | Role |
|---|---|
| `npc_3phase_model.py` | `NPC3PhaseParams` + switching-state table + multicarrier PD-PWM math + fundamental amplitude + NP balancing dynamics + THD comparison helpers |
| `npc_3phase_pulsim_validation.py` | Pulsim builder — 12 switches + 6 clamping diodes + split DC bus caps + custom multicarrier PD-PWM `switch_fn` |
| `00_npc_pulsim_validation.ipynb` | **Executed** Pulsim cross-validation showing 3-level phase voltage and 5-level line-to-line voltage |
| `01_npc_modeling.ipynb` | Topology, switching states, PD-PWM derivation, fundamental amplitude vs $m_a$, THD comparison vs 2-level VSI |
| `02_npc_balancing.ipynb` | NP voltage drift mechanism, redundant-state balancing, carrier-based balancing controller |
| `_build_notebooks.py` | Generator for 01 + 02 |
| `_build_pulsim_validation.py` | Generator for 00 |

## Reference parameters

`NPC3PhaseParams()` defaults: 400 V DC bus → 230 V line-to-line rms at
60 Hz, 500 W per phase, 1 mH + 10 µF LC output filter per phase,
**5 kHz** switching frequency.

Note: the NPC uses a lower $f_{sw}$ than the 2-level VSI (which uses
20 kHz) yet produces comparable output quality because the effective
ripple frequency at the load is $2 f_{sw} = 10$ kHz (PD scheme), and
the 3-level steps already cut the fundamental ripple amplitude in
half. This is the "switching-frequency dividend" of multilevel — half
the switching losses for the same output quality.

- Each switch: $R_{on} = 1$ mΩ, $V_{rating} = V_{dc}/2 = 200$ V
- DC bus split caps: $C_{dc} = 470$ µF each (typical NP ripple < 5 V)
- Modulation index $m_a \approx 0.939$ (same as 2-level VSI for
  apples-to-apples comparison)

## Comparison: 2-level VSI vs NPC 3-level

| Metric | 2-level VSI | NPC 3-level | Winner |
|---|---|---|---|
| Switches per leg | 2 | 4 | VSI (simplicity) |
| Clamping diodes | 0 | 2 | VSI |
| DC bus caps | 1 | 2 (split) | VSI |
| Switch $V_{rating}$ | $V_{dc}$ | $V_{dc}/2$ | **NPC** (uses smaller switches) |
| Output levels (phase) | 2 | 3 | **NPC** |
| Output levels (line-to-line) | 3 | 5 | **NPC** |
| THD at same $f_{sw}$, $m_a = 0.9$ | ~40% | ~20% | **NPC** |
| Required $f_{sw}$ for same THD | 1× | ~½× | **NPC** (lower losses) |
| Unique control challenge | none | NP balancing | VSI |
| Industrial sweet spot | low voltage (<690 V) | medium voltage (690 V to 6.6 kV) | both |

## What you'll learn

- The **multilevel concept** and why more levels means lower THD.
  Why this matters more at higher voltages (silicon switch ratings
  cap individual cell voltage).
- The **NPC topology**: four switches, two clamping diodes, the
  neutral-point split, why only three of the 16 switch combinations
  are valid.
- **Multicarrier PD-PWM**: stack two carriers, compare reference
  against both, derive the switching-state-to-voltage mapping.
- **Neutral-point voltage balancing**: how the NP carries net current
  during O-state intervals, why this drifts the NP voltage open-loop,
  and the standard closed-loop fixes (redundant-state selection or
  carrier offset injection).
- Reading **multilevel waveforms** in the time domain — counting
  steps directly identifies the converter topology.

## How to run

```bash
pip install numpy scipy matplotlib jupyter
pip install -e .             # for Pulsim cross-validation
jupyter lab projects/inverters/npc_3phase/01_npc_modeling.ipynb
```

The `00_npc_pulsim_validation.ipynb` ships **executed** — open it
on GitHub to see the 3-level and 5-level waveforms without running
anything.

## Bibliography

- Nabae, A., Takahashi, I. & Akagi, H. *A New Neutral-Point-Clamped
  PWM Inverter*. IEEE Trans. Ind. Appl. IA-17, 518–523 (1981). The
  original NPC paper.
- Rodríguez, J., Lai, J.-S. & Peng, F. Z. *Multilevel Inverters: A
  Survey of Topologies, Controls, and Applications*. IEEE Trans.
  Ind. Electron. 49, 724–738 (2002). The standard multilevel
  reference.
- Holmes, D. G. & Lipo, T. A. *Pulse Width Modulation for Power
  Converters*. IEEE Press 2003. Chapter 11 covers all multicarrier
  PWM variants (PD / POD / APOD) with exhaustive spectral analysis.
- Wu, B. *High-Power Converters and AC Drives*. IEEE Press 2006.
  Chapter 8 on NPC inverters with practical design examples.
