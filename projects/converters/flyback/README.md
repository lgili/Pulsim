# Flyback Converter — Reference Project

Fourth entry under `projects/converters/`. **First isolated topology.**

## Why the flyback

The flyback is the buck-boost's production-grade descendant:

| Feature | Buck-boost | Flyback |
|---|---|---|
| Inductor | single magnetizing coil | coupled primary + secondary winding |
| Isolation between input and output | none (shared ground) | **galvanic** via transformer |
| Turns ratio degree of freedom | n/a | $n = N_s / N_p$ — picks the operating $D$ |
| Steady-state ratio | $V_o = V_g \cdot D/(1-D)$ | $V_o = n \cdot V_g \cdot D/(1-D)$ |
| RHP zero | $R(1-D)^2 / (L D)$ | $R(1-D)^2 / (n^2 \, L_m \, D)$ |
| Real-world relevance | low | **HUGE** — most low-power offline supplies |

The flyback is the textbook isolated topology because it's the simplest
one that produces a galvanically-isolated DC output from a switched
input using a single switch.

## What's NEW vs the previous three references

1. **The transformer** — a coupled inductor with magnetizing inductance
   $L_m$ (referred to primary side) and a turns ratio $n = N_s/N_p$.
   We idealize it as a pure $L_m$ in shunt with an ideal turns
   transformation; real flybacks have a leakage inductance and a
   clamp/snubber to handle the energy stored in it. The notebooks
   stay in the idealized regime — the small-signal control model
   doesn't depend on the snubber.

2. **Reflected impedance** — the load $R$ on the secondary appears as
   $R / n^2$ from the primary side. Capacitor $C$ appears as $C \cdot n^2$.
   Voltages and currents scale by $n$ and $1/n$ respectively. This
   "Norton equivalent through a transformer" trick is the key to
   reducing the flyback's analysis to the buck-boost's.

3. **Design freedom: $n$ picks $D$.** Instead of being forced to
   operate at extreme duty (close to 0 or close to 1) for big
   step-up or step-down ratios, the engineer picks $n$ so that $D$
   sits in a comfortable range — typically 0.3 to 0.6.

## Files

| File | Role |
|---|---|
| `01_flyback_modeling.ipynb` | Derivation, transformer reflection, RHP zero, self-consistency checks. |
| `02_flyback_controller.ipynb` | K-factor type-III + switched closed-loop simulation with $v_{ref}$ step. |
| `flyback_model.py` | Reusable `FlybackParams` + state-space + 3 reference TFs. |
| `_build_notebooks.py` | Generator. |

## Reference parameters

`FlybackParams()` default: 24 V → 12 V isolated step-down at 12 W,
$n = 0.5$ (1:2 turns), 300 µH primary magnetizing, 100 µF output
cap, 100 kHz switching.

At $D = 0.5$ (the natural symmetric design point):
- $f_n \approx 920$ Hz, $Q \approx 7$
- $f_{z,RHP} \approx 12.7$ kHz (33 % higher than buck-boost at same
  $D$, thanks to the $1/n^2$ reflection)
- Bandwidth cap ≈ $f_z / 5 = 2.5$ kHz — faster than the buck-boost's
  1.9 kHz

## What you'll learn

- How a transformer's turns ratio shows up in the average model as a
  simple $n$ scaling on $v_o$ and $i_{Lm}$ (no new differential
  equation needed).
- Why production isolated supplies use the flyback: it's the
  cheapest, simplest isolated topology that handles any
  step-up-or-step-down ratio with a single switch + diode + cap.
- The reflected-impedance trick — analyzing the converter on the
  primary side as if it were a buck-boost, then multiplying back by
  $n$ for measured quantities.
- That the flyback inherits all the buck-boost's pain points (RHP
  zero, lightly-damped LC pole, slow voltage-mode loop) and offers
  the same workarounds.

## Pulsim cross-validation: same limitation

The flyback's ideal-switch + ideal-transformer topology shares the
boost / buck-boost's numerical stability issues in Pulsim's
switching engine. The math self-consistency checks (poles, DC gains,
ss2tf round-trip) are the rigorous validation; the closed-loop
pure-Python simulation in notebook 2 is the proof on the actual
switched waveform.

## How to run

```bash
pip install numpy scipy matplotlib
jupyter lab projects/converters/flyback/01_flyback_modeling.ipynb
```

## Bibliography

- Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd
  ed., Chapter 6 (transformer-isolated converters) and the
  buck-boost reference in 7.5.
- AN-7515 / ON Semiconductor / TI flyback design guides — practical
  considerations (leakage clamp, EMI, regulation) absent from the
  ideal analytical model.
