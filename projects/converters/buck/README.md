# Buck Converter — Reference Project

Step-by-step derivation of the small-signal model of an ideal buck
converter in continuous conduction mode (CCM), with both the math and
the Python code that implements it, validated by side-by-side
comparison with a Pulsim transient simulation.

## Audience

Undergraduate / early-grad students taking a first course in power
electronics. Familiarity with KVL/KCL, Laplace transforms, and basic
linear systems is assumed; everything else is derived from scratch.

## Files

| File | Role |
|---|---|
| `01_buck_modeling.ipynb` | Derives the average + small-signal model. Builds state-space `(A, B, C, D)`, transfer functions, Bode/step plots, then validates against a Pulsim transient. |
| `02_buck_controller.ipynb` | Uses the model from notebook 1 to design a voltage-loop compensator. Verifies the closed loop in two ways: an analytical step response (scipy.signal) and a closed-loop Pulsim transient with the digital PID applied. |
| `buck_model.py` | Pure-Python module with `BuckParams`, `buck_state_space`, and the three reference transfer functions (`control_to_output_tf`, `line_to_output_tf`, `output_impedance_tf`). Both notebooks import this. |

## Reference parameters

The default `BuckParams()` uses a 24 V → 12 V buck with a 100 µH inductor,
100 µF cap, 2.4 Ω load (~5 A), 100 kHz switching. The LC corner sits at
~1.6 kHz and the switching frequency is 60× above the natural — that's
the standard "averaging assumption is comfortably valid" regime where
the analytical model and the time-domain simulation should match to
within ~1%.

## How to run

```bash
# from this folder
pip install numpy scipy matplotlib

# Optional but recommended — for the Pulsim cross-validation cells.
# Run from the repo root:
#   pip install -e ".[schematic]"

jupyter lab 01_buck_modeling.ipynb
```

Validation cells gracefully skip if the Pulsim Python module isn't
importable, so the math portions run on any clean numpy / scipy install.

## What you'll learn

By the end of `01_buck_modeling.ipynb` you should be able to:

- Write the switched model of a buck (KVL/KCL for each topology state).
- Apply state-space averaging to collapse the two topologies into one.
- Linearize the average model around a steady operating point and read
  off `(A, B, C, D)` directly.
- Derive the three classical buck transfer functions:
  - `G_vd(s)` — duty cycle to output voltage (the control-design plant).
  - `G_vg(s)` — input voltage to output voltage (audio susceptibility).
  - `Z_out(s)` — open-loop output impedance.
- Verify the model is correct by overlaying its step response with the
  Pulsim transient response of the same converter.

By the end of `02_buck_controller.ipynb` you should be able to:

- Identify the loop-shaping requirements (target crossover, phase
  margin, DC error).
- Design a type-II or type-III continuous-time compensator that meets
  those specs.
- Discretize the compensator (Tustin) for digital implementation.
- Confirm the closed-loop behavior with a step in reference voltage
  and a load step.
- Drop the same compensator into a Pulsim closed-loop simulation and
  confirm it tracks the analytical prediction.

## Bibliography

The derivations follow standard sources:

- **Erickson & Maksimović**, *Fundamentals of Power Electronics*, 3rd
  ed., 2020. Chapters 7–8 (state-space averaging) and 12 (compensator
  design).
- **Middlebrook**, "Modeling current-programmed buck and boost
  regulators", *IEEE Trans. PE*, 1989.
- **Krein**, *Elements of Power Electronics*, 2nd ed., 2014. Chapter 11.
