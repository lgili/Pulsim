# Modular Multilevel Converter (MMC) — Reference Project

Third entry under `projects/inverters/` — the **state-of-the-art**
multilevel converter and the de-facto standard for HVDC transmission
and medium-voltage drives. Where the
[`npc_3phase`](../npc_3phase/) gives you 3 voltage levels by adding
clamping diodes, the MMC scales seamlessly to **N+1 levels per arm**
just by stacking more sub-modules in series.

## Why the MMC

The MMC was invented by Marquardt and Lesnicar (2003) for HVDC light
applications where:

1. **Switch voltage rating** is the hard limit. With $N$ sub-modules
   per arm, each switch sees only $V_{dc}/N$ — so you can build a
   500 kV converter from off-the-shelf 1700 V IGBTs by using
   $N \approx 300$.
2. **Output harmonic content** must be very low (HVDC grids cannot
   tolerate the THD of a 2-level VSI). The $N+1$-level output is
   nearly sinusoidal at $N = 10$ and indistinguishable from a
   sinusoid at $N = 100+$ — often requiring **no output filter at all**.
3. **Modularity** matters. SMs are identical, hot-swappable units;
   manufacturing scales linearly with $N$.

## Topology

```
                 +V_dc/2
                    │
                  L_arm
                    │
        arm_up_in ──┼── SM_u1 ── SM_u2 ── ··· ── SM_uN ──┐
                                                          │
                                                       ac_out (= phase output)
                                                          │
        arm_lo_in ──┼── SM_l1 ── SM_l2 ── ··· ── SM_lN ──┘
                    │
                  L_arm
                    │
                 -V_dc/2
```

Each half-bridge sub-module (HB-SM) has two IGBTs and one capacitor:

```
       Terminal A (= midpoint)
            │
       ┌────┤
       │    │
       │   S1            INSERT  (S1 ON, S2 OFF):  V_SM = +V_C
       │    │            BYPASS  (S1 OFF, S2 ON):  V_SM =  0
     C_SM   midpoint     INVALID: both ON (short) or both OFF (float)
       │    │
       │   S2
       │    │
       └────┤
            │
       Terminal B
```

The arm voltage is $v_{arm} = (\text{# inserted SMs}) \cdot V_C$ in
steady state where $V_C = V_{dc} / N$. The phase pole voltage is

$$
v_{ac}(t) = \tfrac{V_{dc}}{2} - v_{arm,upper}(t)
$$

which takes $N+1$ discrete values in $[-V_{dc}/2, +V_{dc}/2]$.

## Three unique control challenges (vs simpler topologies)

1. **Capacitor voltage balancing.** Each SM has its own floating
   cap. Without active balancing, the cap voltages drift due to the
   asymmetric current each SM sees. The standard cure is
   **sort-and-select**: at every switching instant, sort the SMs by
   their cap voltage and insert the lowest (for charging) or highest
   (for discharging) according to the arm-current sign.

2. **Circulating current control.** Even at zero load current,
   internal current circulates between the upper and lower arms at
   DC + $2 f_o$. The DC component is just the average input
   current; the $2 f_o$ component is parasitic and must be suppressed
   with a resonant controller — otherwise it pumps the cap voltages.

3. **Output current control.** Standard dq-frame PI like a VSI,
   but on the **common-mode** arm current $(i_{arm,up} - i_{arm,lo})/2$
   instead of a single phase current.

## Project scope

We model a **single-phase MMC** with **$N=3$ sub-modules per arm**
(4-level pole voltage) — large enough to demonstrate every MMC
phenomenon, small enough to fit in Pulsim's PWL cache (12 switches
in the switch mask = $2^{12} = 4096$ combinations, matching the
proven-working
[`vsi_3phase`](../vsi_3phase/) project).

Three-phase MMCs (the production form) are covered in the modeling
notebook as a natural extension; running a Pulsim sim of a full
three-phase MMC requires either GPU-scale solvers or a different
simulator architecture (lazy / topology-aware cache), out of scope
here.

## Files

| File | Role |
|---|---|
| `mmc_model.py` | `MMCParams` + HB-SM math + arm voltage / current dynamics + PSC-PWM + cap-voltage state equations + circulating-current model |
| `mmc_pulsim_validation.py` | Single-phase Pulsim builder (12 switches + 6 caps + 2 arm inductors + AC load) + PSC-PWM `switch_fn` with optional sort-and-select |
| `00_mmc_pulsim_validation.ipynb` | **Executed** — single-phase N=3 simulation showing 4-level arm voltage, 4-level pole AC voltage, cap voltage drift, circulating current at $2 f_o$ |
| `01_mmc_modeling.ipynb` | HB-SM topology + states, arm KVL/KCL, PSC-PWM derivation, fundamental amplitude, cap-voltage dynamics, circulating-current decomposition, 3-phase extension |
| `02_mmc_control.ipynb` | Sort-and-select cap balancing, circulating-current resonant suppression at $2 f_o$, closed-loop forward-Euler simulation |
| `_build_notebooks.py` | Generator for 01 + 02 |
| `_build_pulsim_validation.py` | Generator for 00 |

## Reference parameters

`MMCParams()` defaults: 400 V DC bus → 230 V_rms AC output at 60 Hz,
**$N = 3$ SMs per arm** ($V_C = V_{dc}/N \approx 133$ V), arm
inductors 1 mH, sub-module caps 470 µF each. Carrier frequency
**1 kHz per carrier**, with $N$ phase-shifted carriers per arm
(effective ripple frequency = $N \cdot f_{carrier} = 3$ kHz on the
load).

The MMC's **switching-frequency dividend** is even larger than NPC:
each SM switches at $f_{carrier}$ but the load sees the effective
ripple at $N \cdot f_{carrier}$. So the AC-side LC filter (if any)
can be much smaller.

## Comparison across the library

| Topology | Levels per phase | Switches per phase | Unique control |
|---|---|---|---|
| 2-level VSI | 2 | 2 | (none — just dq) |
| NPC 3-level | 3 | 4 + 2 clamping diodes | NP voltage balancing |
| **MMC (N=3)** | **4** | **2N = 6** | **Cap balancing + circulating current + arm energy** |
| MMC (N=20) | 21 | 40 | (same) |
| MMC (N=300, HVDC) | 301 | 600 | (same) |

The MMC's defining feature is **scalability**: doubling $N$ doubles
the voltage rating and halves the per-switch stress, but the control
machinery (sort-and-select + circulating-current PI + dq output PI)
stays the same.

## What you'll learn

- The **half-bridge sub-module**: 2 switches + 1 cap, two states
  (insert / bypass), and why the cap voltage is a state variable
  that must be actively regulated.
- The **arm energy balance**: why $\langle v_{arm,upper} \rangle +
  \langle v_{arm,lower} \rangle = V_{dc}$ in steady state and how
  KVL through the bus + two arm inductors gives the circulating
  current dynamics.
- **Phase-Shifted Carrier PWM (PSC-PWM)**: $N$ carriers per arm at
  the same frequency but offset by $2\pi / N$. The effective load
  ripple frequency is $N \cdot f_{carrier}$.
- **Sort-and-select capacitor balancing**: simple, robust, and the
  industry standard for MMC. Why it's $O(N \log N)$ and how it makes
  the per-cap-voltage control loop "passive".
- **2nd-harmonic resonant control** of the circulating current —
  textbook example of a resonant controller for a non-DC reference.

## How to run

```bash
pip install numpy scipy matplotlib jupyter
pip install -e .             # for Pulsim cross-validation
jupyter lab projects/inverters/mmc/01_mmc_modeling.ipynb
```

The `00_mmc_pulsim_validation.ipynb` ships **executed** — open it
on GitHub to see the multilevel waveforms inline.

## Bibliography

- Lesnicar, A. & Marquardt, R. *An innovative modular multilevel
  converter topology suitable for a wide power range*. IEEE Bologna
  PowerTech (2003). The original MMC paper.
- Glinka, M. & Marquardt, R. *A new AC/AC multilevel converter
  family*. IEEE Trans. Ind. Electron. 52, 662–669 (2005). Early
  control-architecture paper.
- Rohner, S., Bernet, S., Hiller, M. & Sommer, R. *Modulation,
  losses, and semiconductor requirements of modular multilevel
  converters*. IEEE Trans. Ind. Electron. 57, 2633–2642 (2010).
- Sharifabadi, K., Harnefors, L., Nee, H.-P., Norrga, S., Teodorescu,
  R. *Design, Control, and Application of Modular Multilevel
  Converters for HVDC Transmission Systems*. Wiley/IEEE Press, 2016.
  The comprehensive textbook reference.
