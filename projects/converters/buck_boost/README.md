# Buck-Boost Converter — Reference Project

Third entry under `projects/converters/`. Builds directly on the
buck and boost projects — the student should work through those first.
This converter combines features of both:

| Feature | Buck | Boost | Buck-boost |
|---|---|---|---|
| Output magnitude vs input | $V_o < V_g$ | $V_o > V_g$ | **either** |
| Output polarity | + | + | **inverted (−)** |
| RHP zero in $G_{vd}(s)$ | no | yes @ $R(1-D)^2 / L$ | yes @ $R(1-D)^2 / (L \cdot D)$ |
| Bandwidth cap | $f_{sw}/10$ | $\sim f_{z,RHP}/5$ | $\sim f_{z,RHP}/5$ |

## Why it matters pedagogically

Three things students see for the first time here:

1. **Polarity inversion is a topology feature, not a math result.** The
   state-space averaging produces a clean positive-magnitude equation;
   the negative sign is just how the components are wired.

2. **RHP zero location is duty-cycle dependent.** Unlike the boost
   (where $\omega_z = R(1-D)^2/L$ scales as $(1-D)^2$), the buck-boost
   has $\omega_z \propto (1-D)^2 / D$, which DROPS as duty increases.
   So a high step-up ratio means a SLOWER allowed loop — not just
   harder steady-state operating point.

3. **Bidirectional operating point.** At $D = 0.5$ the magnitude is
   identical to input ($|V_o| = V_g$). $D < 0.5$ gives a buck-like
   step-down; $D > 0.5$ gives a boost-like step-up. One converter,
   two regimes.

## Files

| File | Role |
|---|---|
| `01_buck_boost_modeling.ipynb` | Derives the model, identifies the RHP zero, plots Bode + wrong-way step response, validates internal math consistency. |
| `02_buck_boost_controller.ipynb` | K-factor type-III with $f_c \le f_{z,RHP}/5$, Tustin discretization, **switched-model closed-loop simulation** that proves the design tracks a $v_{ref}$ step. |
| `buck_boost_model.py` | Reusable `BuckBoostParams` + state-space + 3 reference TFs. |
| `_build_notebooks.py` | Generator script. |

## Reference parameters

Default `BuckBoostParams()`: 12 V → ±12 V at 12 W, 100 µH / 100 µF /
100 kHz, $D = 0.5$ (the "neither-step-up-nor-step-down" balanced case).

Operating point: $|V_o| = V_g$, $I_L = 2$ A, LC pole at 796 Hz, Q ≈ 6.

RHP zero: **9.5 kHz** — at this duty cycle the buck-boost zero is
higher than the boost's (4.6 kHz), so the loop can run faster. Set
$D = 0.8$ in `BuckBoostParams(V_o=48)` to see the zero collapse to
~600 Hz — the loop has to slow down dramatically.

## Pulsim cross-validation: known limitation

Same caveat as the boost notebook: Pulsim's switching engine has
numerical-stability issues with the ideal-switch buck-boost (decoupled
output cap → no natural damping for switch-transition dV/dt spikes).
The math self-consistency checks in notebook 1 (poles, DC gains, ss2tf
round-trip) are the rigorous validation; the closed-loop pure-Python
simulation in notebook 2 is the proof-of-life on the actual switched
waveform.

## How to run

```bash
pip install numpy scipy matplotlib
jupyter lab projects/converters/buck_boost/01_buck_boost_modeling.ipynb
```

## What you'll learn

By the end of these two notebooks you should be able to:

- Recognize the buck-boost on a schematic and predict its output
  polarity from the topology alone.
- Derive the average model, state-space, and three reference
  transfer functions from scratch.
- Explain WHY the RHP zero location depends on duty cycle (look at
  the algebra after the average is taken).
- Size a Type-III compensator that respects the $f_z/5$ ceiling.
- Argue convincingly why production buck-boost designs at high duty
  often use current-mode control or fixed-frequency hysteretic
  schemes — voltage-mode at high step-up ratios is fundamentally
  bandwidth-limited.

## Bibliography

- Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd ed.,
  Section 7.5 (buck-boost average model) and Eq. 8.55 (RHP zero
  closed form).
- Buck and boost reference notebooks in `projects/converters/`.
