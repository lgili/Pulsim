# Boost Converter — Reference Project

Same template as the buck reference, but the boost has **one
fundamental difference** that makes it pedagogically richer: a
**right-half-plane (RHP) zero** in the control-to-output transfer
function $G_{vd}(s)$. This zero is the canonical example of
non-minimum-phase dynamics in power electronics, and the reason boost
controllers are bandwidth-limited regardless of how aggressive the
compensator is.

## Audience

Undergraduate / first-year grad — same as the buck. Assumes the
student has worked through `projects/converters/buck/` first (the
state-space averaging and small-signal procedure are identical;
this notebook builds on that).

## Files

| File | Role |
|---|---|
| `01_boost_modeling.ipynb` | Average + small-signal derivation, highlights the RHP zero, validates against Pulsim. |
| `02_boost_controller.ipynb` | Type-III compensator design with crossover capped at $f_{z,RHP}/5$. Closed-loop switched-buck simulation proves the design works on the real waveform. |
| `boost_model.py` | Reusable `BoostParams` + state-space + three reference transfer functions. |

## Reference parameters

The default `BoostParams()` is a 12 V → 24 V boost delivering 50 W
into an 11.52 Ω load with 100 µH input inductor, 100 µF output cap,
100 kHz switching. At $D = 0.5$ that gives an LC pole at ~800 Hz, Q ≈
5.8, and a **RHP zero at 4584 Hz**. Practical control bandwidth caps
at ~900 Hz — about $f_{z}/5$ — which is what notebook 02 targets.

## Why the RHP zero matters (the central lesson)

In a buck, asking for more duty immediately means more energy flowing
to the output → output rises monotonically. Easy.

In a boost, asking for more duty means more time with the switch
SHORTING the inductor to ground — that's the OFF interval for the
output. So MORE duty initially means LESS energy delivered to $V_o$,
and only later, once the inductor current has built up, does the
output recover and rise. The output **dips first, then rises** on a
positive duty step. This shows up in the math as a zero of $G_{vd}(s)$
at $s = +R(1-D)^2/L$ — positive real → right-half-plane.

In Bode terms: the RHP zero **drops the magnitude AND drops the
phase** at the same time, so it cannot be cancelled by a regular
left-half-plane compensator pole. The only design knob is to stay
well below $f_z$ so the RHP zero's phase loss doesn't reach the
crossover. Hence the "$f_c \le f_z / 5$" rule of thumb.

The notebooks make this concrete: the step response of $G_{vd}(s)$
visibly dips before recovering. Students see the non-minimum-phase
behavior directly, and the closed-loop simulation confirms that
trying to push $f_c$ above $f_z$ destabilizes the loop.

## How to run

```bash
pip install numpy scipy matplotlib
pip install -e ".[schematic]"      # optional: Pulsim cross-validation

jupyter lab projects/converters/boost/01_boost_modeling.ipynb
```

Each notebook is self-contained. The Pulsim cross-validation cell
skips gracefully if the wheel isn't built.

## What you'll learn

- **Notebook 1**: derive $V_o = V_g/(1-D)$ from state-space averaging,
  identify the source of the RHP zero in the algebra, plot the
  characteristic "wrong-way" step response, validate the steady-state
  ratio against a Pulsim transient.

- **Notebook 2**: size a type-III compensator that respects the
  $f_c \le f_z/5$ ceiling, discretize via Tustin, run a switched
  closed-loop simulation with a $V_{ref}$ step that shows the dip + 
  recovery + final tracking.

## Bibliography

Same as the buck reference; especially:

- Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd
  ed., **Chapter 7.6** (boost average model) and **Chapter 8.2**
  (RHP zero, control consequences).
