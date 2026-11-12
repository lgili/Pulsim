## 1. C ABI and kernel kind (after `add-unified-device-interface`)
- [ ] 1.1 `python/pulsim/pulsim_xdev.h`: descriptor, context, field and output structs, `PULSIM_XDEV_ABI_VERSION`, the exported `pulsim_xdev_get`; a commented reference device (Shockley) in `examples/`
- [ ] 1.2 `core/include/pulsim/device/external_device.hpp`: `ExternalDevice` registry entry with run-time arity; span-based `stamp_fq` overload sharing the time-formula coefficients with the static overload; slot reservation from the fixed patterns
- [ ] 1.3 Loader and registration checks (version, size, pattern bounds and duplicates, parameter bounds, `validate`, finite outputs, Jacobian against central differences at 16 points); each refusal names the device and the reason
- [ ] 1.4 Non-finite `eval` output during a run → Newton failure naming the device
- [ ] 1.5 `python/pulsim/_native_cache.py`: the C block's compile cache moved to a per-user directory (mode 0700, ownership check); `c_block.py` uses it; test that a foreign-owned cached file is rebuilt, and that the existing C block tests pass
- [ ] 1.6 `pulsim.add_external_device(builder, name, nodes, *, lib=None, code=None, lang="c", params=None, check_jacobian=True)`; binding, stubs; `components()` entry with library path and hash
- [ ] 1.7 YAML type `external_device` (`lib` or `code`, `nodes`, `params`); refused without `allow_native_code=True`; strict keys from the field table

## 2. Acceptance: stateless devices
- [ ] 2.1 External Shockley diode (same expression as `models/shockley_diode.hpp`, including the linear extension above `i_lim` and `G_min`) in a 20 V half-wave rectifier: node voltages equal the built-in's within 1e-12 on `pwl` and in the DC operating point, within 1e-9 on `trbdf2`
- [ ] 2.2 External junction capacitor (`q` only, `C0 = 2 nF`, `V0 = 25 V`, `m = 0.5`): equals `add_nonlinear_capacitor` within 1e-12 on `pwl`; charge at 400 V = 312.31 nC
- [ ] 2.3 AC: external Shockley at 1 A → small-signal resistance `n·V_T/(I + I_S)` within 1e-9
- [ ] 2.4 Negative tests: wrong `abi_version`; pattern index out of range; a Jacobian entry scaled by 1.01; a parameter outside its bounds; `eval` returning NaN — each refused or reported with the device name
- [ ] 2.5 Sweep on 8 threads with an external device is bit-identical to the serial run

## 3. State, events, outputs
- [ ] 3.1 Internal unknowns: external Lauritzen diode (one internal charge) against the built-in within 1e-10
- [ ] 3.2 `init` / `accept` with kernel-owned state in the flat device-state vector; snapshot and restore round-trip (steady-state solver, rejected steps)
- [ ] 3.3 `events`: a current-limit latch whose opening instant matches the analytic crossing within the engine's event tolerance, on `pwl` and `trbdf2`
- [ ] 3.4 `outputs`: named outputs available as `res.device_output(name, key)`; current and power through the generic outputs of the device interface
- [ ] 3.5 Conformance rows (`add-engine-conformance-matrix`): `external_device` × {pwl ok, trbdf2 ok, dsed refused, DC ok, AC ok}

## 4. Equations to C (`pulsim.xdev`)
- [ ] 4.1 `pins`, `internal`, `params`, `state`; `compile(name, pins, f, q, ...)` → SymPy differentiation, pattern, CSE, C99 with `expm1` / `log1p`, descriptor, compile, load
- [ ] 4.2 Smoothness lint: `Piecewise`, `Abs`, `sign`, `Max`, `Min`, `Heaviside`, `floor` refused with the sub-expression unless `allow_nonsmooth=True`
- [ ] 4.3 Test: generated C against `lambdify` at 1000 random points within 1e-13; SymPy-declared Shockley against the built-in within 1e-12
- [ ] 4.4 `xdev.scan_monotone(dev, pin, lo, hi)`; test on a law with a deliberate cancellation (`coth x − 1/x` without a series) reports the non-monotone region
- [ ] 4.5 Optional extra `pulsim[codegen]`; a clear error when SymPy or the compiler is missing

## 5. OSDI host
- [ ] 5.1 Confirm the OSDI descriptor layout and its licence against the project rule; record both in `design.md`
- [ ] 5.2 Adapter: OSDI residuals and Jacobians (resistive, reactive) → `f`, `q`; nodes → pins and internal unknowns; parameters → field table; unsupported features refused by name
- [ ] 5.3 Test: a Verilog-A diode compiled by the user's tool chain, same circuit in ngspice and Pulsim, within 1e-6 (skipped when the compiler is absent)

## 6. Documentation
- [ ] 6.1 `docs/external-devices.md` (the row convention, purity of `eval`, the 1e-12 smoothness requirement, trust boundary), `docs/c-block.md` (cache location; C block versus external device), `AI_GUIDE.md`, `docs/known-limitations.md` and `KNOWN_LIMITATIONS.md` (no sandbox, no external PWL devices, OSDI coverage), `CHANGELOG.md` (cache move)

## 7. Verification
- [ ] 7.1 `ruff check`, `pytest`, `ctest`, `openspec validate --all --strict`, stub check
