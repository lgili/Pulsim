## 1. Failing tests first
- [ ] 1.1 C++ law test: on 550 001 points in [−50 V, +5 V] with default parameters, `i` strictly increasing and `i·v ≥ 0`; `i(0) == 0` exactly. Fails today (minimum −1.392 A at 0.636 V)
- [ ] 1.2 C++ value table at the defaults, relative 1e-5: `i(0.5) = 0.090746`, `i(0.636) = 1.22662`, `i(0.7) = 3.46573`, `i(0.75) = 6.5663`, `i(1.0) = 30.0124`, `i(−10) = −4.168e-6` A
- [ ] 1.3 C++ AD check: Jacobian equals `logistic(κ(v − V_F0))/R_d + G_off` to 1e-12 at v ∈ {−40, 0, 0.7, 40}; finite at `κ·|v − V_F0| = 1000`
- [ ] 1.4 Python forced voltage: source across the diode at 0.636 V → diode current +1.2266 A ± 1e-3 on `pwl` and through `compute_dc_op` (today −1.3923 A)
- [ ] 1.5 Python clamp (20 V, 1 kHz sine, 10 Ω, diode to ground, 2 ms): `engine='pwl'`, `dt = 1 µs` completes (today `SimulationAborted`); time-weighted RMS of `v(a)` = 10.010 V ± 0.02 V on `pwl` and `trbdf2` (today 9.634 V on `trbdf2`); peak `v(a)` = 0.662 V ± 0.01 V
- [ ] 1.6 Python importer: `.model DX D(Ron=0.05 Roff=1Meg Vfwd=0.7)` gives `StoredKind.Diode` with `g_on = 20`, `g_off = 1e-6`, `V_f = 0.7`, `V_th = 0`; 10 A forced → 1.2 V, `device_loss_summary` 12.0 W; `area=2` doubles both conductances. Fails today (`StoredKind.NonlinearDiode`)

## 2. Law
- [ ] 2.1 `numeric/softplus.hpp`: scalar and `ADRealN` overloads, unit test beside `core/tests/layer0/test_logistic.cpp`
- [ ] 2.2 `IdealDiode::current` in softplus form; header comment states the law, `i(0) = 0` and the reverse floor
- [ ] 2.3 Update the C++ tests that pin the old law (`layer4_v3/test_diode_load_line.cpp`, `layer2/test_device_model_concept.cpp`, the six `layer5_v4` rectifier files, `layer5/test_dt_retry.cpp`): each new expected number derived from the formula, not copied from the run
- [ ] 2.4 Continuation gate: the κ- and `V_F0`-homotopy tests converge; if not, apply the fallback in design.md

## 3. Python
- [ ] 3.1 `spice_import.py`: PWL diode branch calls `add_diode(..., V_th=0.0, V_f=Vfwd)`; the import report lists it as a PWL diode
- [ ] 3.2 Grep `python/pulsim` for a re-implementation of the smooth-diode law (`res.i`, losses) and update or refuse it
- [ ] 3.3 Update importer tests that assert `add_nonlinear_diode` for PWL models

## 4. Docs
- [ ] 4.1 `CHANGELOG.md` [Unreleased]: both BREAKING notes with the numbers of proposal.md
- [ ] 4.2 `docs/known-limitations.md`, `KNOWN_LIMITATIONS.md`: soft knee, reverse floor formula
- [ ] 4.3 `python/pulsim/AI_GUIDE.md`: which diode to use; importer mapping
- [ ] 4.4 Docstring of `IdealDiodeParams` and `add_nonlinear_diode`; regenerate stubs if a docstring is part of them

## 5. Verification
- [ ] 5.1 `ruff check`
- [ ] 5.2 `pytest python/tests`
- [ ] 5.3 `ctest -j2`
- [ ] 5.4 `openspec validate --all --strict`
- [ ] 5.5 Stub check
- [ ] 5.6 Re-run `p2_ideal_diode_iv.py` and the smooth-diode row of `p1_engine_matrix.py`; record the new lines in the PR
