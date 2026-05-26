## Why

Modern AC drives (servo / fan / pump / EV traction) almost universally use **sensorless control**: rotor position and speed are estimated from electrical measurements only, eliminating the encoder, its cabling, and a class of reliability problems. PSIM and PLECS ship reference implementations of sliding-mode (SMO), model-reference adaptive system (MRAS), and high-frequency-injection observers as drag-and-drop blocks.

Pulsim v1.4.2 only supports **sensored** FOC — the existing PMSM/BLDC examples feed the true rotor angle straight from the device's mechanical state into the Park transform. There is no mechanism for "close the loop using only `i_a`, `i_b`, `i_c`, and the inverter voltage command" — which is the only realistic configuration for shipped drives.

Adding two complementary observers (a sliding-mode observer for mid-to-high speed and a flux-MRAS observer that degrades gracefully near zero speed) is the minimum competitive surface. They live in the existing `MixedDomainBlockChain` control framework — no kernel changes, no new device types.

## What Changes

- **New Python module** `python/pulsim/motors/observers.py` (or extend `motors.py`) with:
  - `class SlidingModeObserver`: PMSM rotor-position observer using the back-EMF / sliding-surface algorithm with adaptive gain and a configurable low-pass filter on the equivalent control. Inputs: stator voltages `Vα`, `Vβ`, stator currents `iα`, `iβ`, motor parameters (Rs, Ls). Outputs: estimated back-EMF `eα_hat`, `eβ_hat`, estimated electrical angle `theta_hat`, estimated speed `omega_hat`.
  - `class FluxMRASObserver`: induction-motor rotor-speed observer using the reference / adaptive flux-magnitude error. Inputs: stator voltages `Vα`, `Vβ`, stator currents `iα`, `iβ`, motor parameters. Outputs: estimated rotor speed `omega_hat`, rotor flux estimate.
  - Both classes expose `.step(...)` returning the observer outputs and a `.reset()` method, conforming to the existing `BlockChain` block interface so they wire in with `chain.add("smo", ..., inputs=..., output=...)`.
- **Two showcase scripts**:
  - `examples/scripts/run_pmsm_foc_sensorless_smo.py` — PMSM-FOC closed loop where the Park transform uses `theta_hat` from the SMO instead of the true mechanical angle. Demonstrates startup ramp + load step + speed-reversal under sensorless operation.
  - `examples/scripts/run_im_ifoc_sensorless_mras.py` — Indirect-FOC induction-motor drive (requires `add-induction-motor-squirrel-cage` to land first) using MRAS for slip-frequency estimation. Step load.
- **Cross-validation** — pytest comparing the observer-estimated angle/speed to the true mechanical state from the device:
  - SMO: angle error ≤ 5 ° at 25–100 % rated speed, speed error ≤ 1 % once locked (allow a configurable startup window).
  - MRAS: speed error ≤ 2 % above 10 % rated speed.
- **Docs** — page `docs/v2/motors-sensorless.md` documenting both algorithms with equations, typical gains, and a tuning recipe. Honest section about each method's failure modes (SMO degrades < 10 % rated speed; MRAS depends on stator-flux integration drift).

## Impact

- **Affected specs**: NEW capability `motor-observers` (separate spec — observers are a control-layer abstraction distinct from device models). The existing `motor-models` spec stays untouched.
- **Affected code**: ~400 LOC Python in `motors/observers.py`, ~150 LOC of showcase scripts, ~80 LOC of pytest. Zero C++ kernel changes — observers are pure Python `BlockChain` blocks.
- **Backward compatibility**: PURE ADDITION.
- **Dependency on `add-induction-motor-squirrel-cage`**: MRAS validation scenarios depend on a real IM device existing. The SMO half is fully testable today against PMSM. We will ship the SMO first if the IM proposal lands later; the spec is structured so the two observers are independent requirements.
- **Risk**: observer gain tuning is sensitive to motor parameters. Mitigated by defaulting to gains derived from the standard pole-placement formulas and documenting how to retune.
