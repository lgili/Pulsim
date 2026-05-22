# PulsimCore Reference Projects

This folder holds **didactic reference projects** that pair worked-out
mathematical models with Pulsim simulations.

The goal is twofold:

1. **Educational.** Each project walks through the derivation step by
   step — equations are written out in LaTeX inside the notebook, the
   Python code mirrors the math one-for-one, and assumptions are stated
   explicitly. The audience is undergraduate / first-year graduate
   students learning power electronics for the first time.

2. **Cross-validation.** Every analytical model is validated against a
   Pulsim transient or AC sweep run side-by-side. When the analytical
   step response and the Pulsim transient overlay within tolerance, we
   have confidence in *both* the model AND the simulator.

## Layout

```
projects/
  converters/                         # DC-DC + DC-AC converters
    buck/                             # ← start here (simplest topology)
      01_buck_modeling.ipynb          # average model + state-space + validation
      02_buck_controller.ipynb        # PI/type-III compensator + closed-loop
      buck_model.py                   # shared model functions
      README.md
    boost/                            # (planned)
    buck-boost/                       # (planned)
    flyback/                          # (planned)
    full-bridge/                      # (planned)
    three-phase-vsi/                  # (planned)
  motors/                             # (planned: BLDC, PMSM, induction)
  filters/                            # (planned: LCL, EMI)
```

## How to run

The notebooks expect a Python environment with:

```bash
pip install numpy scipy matplotlib
pip install -e ".[schematic]"          # for the Pulsim cross-validation cells
```

Each notebook is self-contained — open it, run all cells top to bottom.
Cells that require Pulsim are gated with a try/except so the math
portions still work without a built kernel.

## Conventions

- **Symbols.** Lowercase = instantaneous (`v_o`, `i_L`). Uppercase = DC
  steady-state (`V_o`, `I_L`). Hat = small-signal perturbation
  (`v̂_o`, `î_L`, `d̂`).
- **State vector** ordering for every converter: passive currents first
  (inductor currents), then passive voltages (capacitor voltages).
- **Input vector** ordering: control inputs first (duty cycle d̂),
  then disturbance inputs (line voltage v̂_g, load step î_load).
- **State-space matrices**: `(A, B, C, D)` — `D` is the direct
  feed-through matrix (not the duty cycle).
- **Units**: SI throughout (V, A, Ω, H, F, s, Hz).

## What "validated" means here

Every modeling notebook ends with a validation cell that:

1. Builds the converter in Pulsim using the same parameters.
2. Applies a small duty-cycle step (or input perturbation) around the
   operating point.
3. Captures the output transient.
4. Overlays it with the analytical step response derived from the
   state-space model.
5. Reports the L∞ error and the rise/settling time agreement.

If the two curves don't agree, the converter parameters or the
linearization assumptions need to be revisited.
