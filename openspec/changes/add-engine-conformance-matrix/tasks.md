## 1. Failing tests first
- [ ] 1.1 `python/tests/test_engine_conformance.py`: device cell for `NonlinearCapacitor` (10 V, 1 kΩ, 100 nF ∥ `add_nonlinear_capacitor(1e-6, 1.0, 0.5)`, 1 MΩ load). Expected `v(100 µs)`: 1.02 V ± 0.1 V on `pwl` and `trbdf2`; presence control 6.30 V without the device; `dsed` and `run_transient_bdf1` must raise. Fails today (6.3185 V and 6.3002 V returned)
- [ ] 1.2 Shockley clamp (10 V, 1 kΩ, `add_shockley_diode`, 100 nF): `simulate(engine='pwl', enable_nonlinear_refresh=False)` must raise; raw `_pulsim.run_transient` with default arguments must read 0.594 V ± 0.01 V. Fails today (10.000 V)
- [ ] 1.3 `NativeMultiMaskPwm(T=1 ms, [0.5, 0.501, 1.0], [hi, lo, hi])` on a half bridge with a 1 H / 10 Ω load, 20 periods, `engine='trbdf2'`: `n_gate_events == 40` and every edge within 1e-12 s of `k·T + {0.5, 0.501}·T`. Fails today (0 events)
- [ ] 1.4 C block registered + `engine='dsed'` must raise. Fails today (warning, open-loop run)
- [ ] 1.5 C++: `kKindCaps` has one row per `StoredKind`, names equal the bound enum names, `require_supported` throws `UnsupportedError` with the branch name for `NonlinearCapacitor` on `Dsed` and `Bdf1`

## 2. Table and exception
- [ ] 2.1 `Engine`, `SolveClass`, `KindCaps`, `kKindCaps` and the `static_assert` in `pwl/device_pool.hpp`; `requires_newton()`, `requires_static_refresh()`; `has_nonlinear_devices()` becomes an alias
- [ ] 2.2 `solver/unsupported.hpp`: `UnsupportedError` and `require_supported(graph, pool, Engine)`
- [ ] 2.3 Binding: `UnsupportedError` (base `ValueError`), translator setting `engine`, `feature`, `device`, `kind`; `engine_capabilities()`; export both from `pulsim`
- [ ] 2.4 Private bridge subclass for the two `NotImplementedError` sites in `_dsed_dispatch.py`

## 3. Kernel-side checks
- [ ] 3.1 `run_transient_bdf1` and the DSED builder adapter call `require_supported`; `_dsed_dispatch.py:578` and `bdf1.hpp:297` lose their hand-written guards
- [ ] 3.2 `run_transient_trbdf2`: replace the `BranchKind::Nonlinear` scan with the table queries
- [ ] 3.3 Coupled inductors on `dsed`, PWL diodes under `integrator='bdf2'`, builder initial conditions on `dsed`: raise `UnsupportedError` with the existing text
- [ ] 3.4 `simulate()`: the `_dsed_unsupported` block and the `_trbdf2_blockers` refusals raise `UnsupportedError(engine, feature)`; the C-block warning becomes a refusal

## 4. Refresh ownership
- [ ] 4.1 `run_transient`, `run_transient_trbdf2`, `compute_dc_op_with_strategy` build the static chain when `nl_refresh` is empty and `requires_static_refresh()`; check every stateful wrapper still captures the effective chain (`test_stateful_device_stamps_compose`)
- [ ] 4.2 Bindings: `enable_nonlinear_refresh: Optional[bool] = None` on the three transient entry points; `False` with Newton devices raises on transient drivers; DC operating point keeps the opt-out
- [ ] 4.3 `simulate()` stops calling `has_nonlinear_devices()` at lines 2286 and 2677 and passes the user's value through
- [ ] 4.4 Gate: Coss-only circuit on `pwl` and `trbdf2` bit-identical before and after (states compared with `np.array_equal`)

## 5. Switch-function adapter
- [ ] 5.1 One helper returning `{SwitchScheduleFn, next_edge_fn}`; use it at `bindings.cpp:2598-2636`, `bindings.cpp:4016-4047`, `dsed_bindings.cpp:749-815`
- [ ] 5.2 Gate: `NativePwm2Switch` on `trbdf2` unchanged (1104 steps, 2254 solves, 20 gate events on the `probes/d10_multimask.py` circuit)

## 6. Conformance test
- [ ] 6.1 Device cells for all 20 kinds plus coupled inductors, four engines, with the presence control and the tolerance of design.md decision 5
- [ ] 6.2 Resolve every "n/c" cell of the two tables by running it; update the kernel table or add the refusal
- [ ] 6.3 Feature cells from design.md decision 3; strict xfail for the three cells owned by `fix-control-tick-semantics`
- [ ] 6.4 Non-vacuity: with `require_supported` patched out in memory, tasks 1.1 fails; with the refresh fallback patched out, 1.2 fails

## 7. Docs
- [ ] 7.1 `CHANGELOG.md` [Unreleased]: the new exception, the three cases that now refuse and what to use instead, the raw `run_transient` default
- [ ] 7.2 `KNOWN_LIMITATIONS.md` and `docs/known-limitations.md`: the two tables; test that the doc table equals `engine_capabilities()`
- [ ] 7.3 `python/pulsim/AI_GUIDE.md`: `UnsupportedError`, "catch it, do not catch `Exception`"; `docs/gotchas.md:63` wording
- [ ] 7.4 Docstrings of `simulate`, `run_transient`, `run_transient_trbdf2`; stubs via `scripts/generate_stubs.py`
- [ ] 7.5 Turn `probes/d10_multimask.py` and `probes/d10b_narrow.py` into the tests of 1.3 and 5.2, then delete the `probes/` folder

## 8. Verification
- [ ] 8.1 `ruff check`
- [ ] 8.2 `pytest python/tests` (throttled: `-n 2`, `nice`)
- [ ] 8.3 `ctest -j2`
- [ ] 8.4 `openspec validate --all --strict`
- [ ] 8.5 Stub check (`scripts/generate_stubs.py --check`)
