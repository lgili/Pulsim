## 1. Failing tests first
- [x] 1.1 Python: `add_nonlinear_diode` without the opt-in raises `RetiredDeviceError` (a `ValueError`), stamps nothing, and the message names the device, −1.39 A at +0.636 V, `add_diode(..., V_f=)`, `add_shockley_diode` and `allow_non_passive=True`
- [x] 1.2 Python: with `allow_non_passive=True` the legacy law still gives −1.3923 A at a forced 0.636 V on `pwl` and through `compute_dc_op`
- [x] 1.3 Python and C++: YAML `type: nonlinear_diode` is refused without `allow_non_passive: true` and builds with it
- [x] 1.4 Python importer: `.model DX D(Ron=0.05 Roff=1Meg Vfwd=0.7)` gives kind `diode` with `g_on = 20`, `g_off = 1e-6`, `V_f = 0.7`, `V_th = 0`; 10 A forced → 1.2 V, `device_loss_summary` 12.0 W; `area=2` doubles both conductances. Failed before (`nonlinear_diode`)
- [x] 1.5 Python: no module under `python/pulsim` other than the wrapper calls `add_nonlinear_diode`

## 2. Retire
- [x] 2.1 `_builder_ergonomics.py`: `RetiredDeviceError`, the wrapping `add_nonlinear_diode(..., *, allow_non_passive=False)`; export from `pulsim`
- [x] 2.2 `yaml/loader.hpp`: refuse `nonlinear_diode` without `allow_non_passive: true`
- [x] 2.3 `spice_import.py`: PWL diode branch calls `add_diode(..., V_th=0.0, V_f=Vfwd)`
- [x] 2.4 Python tests that build the device on purpose pass `allow_non_passive=True` (not rewritten); the C++ YAML round-trip test sets the key
- [x] 2.5 Library users: `discovery.py` lists `add_shockley_diode` instead; notebooks 05 and 07 use `add_shockley_diode`; notebook 15 sets the YAML opt-in

## 3. Docs
- [x] 3.1 `CHANGELOG.md` [Unreleased]: BREAKING notes (refusal, YAML, importer)
- [x] 3.2 `docs/known-limitations.md`: the defect and the retirement
- [x] 3.3 `python/pulsim/AI_GUIDE.md`: the device is retired
- [x] 3.4 `docs/api-reference.md`, `docs/migration-from-v1.md`, `docs/internals/layer8-yaml-parser.md`

## 4. Verification
- [x] 4.1 `ruff check`
- [x] 4.2 `pytest python/tests`
- [x] 4.3 `ctest -j2`
- [x] 4.4 `openspec validate --all --strict`
- [x] 4.5 Stub check
