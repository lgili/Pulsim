# add-engine-conformance-matrix — decisions (2026-11-13)

## Open Questions of design.md

| # | Question | Option taken | Why | How to reverse |
|---|---|---|---|---|
| 1 | `UnsupportedError` base | `ValueError`, plus the private `_UnsupportedNotImplemented(UnsupportedError, NotImplementedError)` at the two DSED extractor sites (`python/pulsim/_refusal.py`) | Proposed | Delete the subclass in the next major; the two sites raise `UnsupportedError` |
| 2 | `enable_nonlinear_refresh=False` on the DC operating point | Kept as the documented "linear network" request | Proposed. Consequence: the kernel DC routines (`compute_dc_op_with_strategy`, `compute_dc_operating_point`) do **not** get the "empty callback → build the static chain" fallback of decision 6, since an empty callback is how the binding expresses that opt-out; the binding's `None` still resolves to the chain. The DC solve inside `run_transient(start_from_dc_op=True)` does get it (it receives the kernel's resolved chain) | Add the fallback in `pwl/dc_strategy.hpp` and make the binding pass a sentinel for the opt-out |
| 3 | `run_transient_bdf1` column | Public column `bdf1` in `engine_capabilities()` | Proposed (it is bound) | Drop the key from the binding's dict |
| 4 | Device-cell tolerance | 1 % of the reference span, 10 % presence control, reference converged to ¼ of the tolerance under dt halving | Proposed | `TOL` / `PRESENCE` in `test_engine_conformance.py` |
| 5 | `engine_capabilities()` public in 2.x | Public, documented as "the kernel's table, replaced by the device interface" | Proposed by tasks 2.3/7.2 (the docs test reads it) | Rename to `_engine_capabilities` |

## Choices the design left open

| Point | Choice | Why | How to reverse |
|---|---|---|---|
| Fifth column | `ac_sweep` (the kernel AC sweep, `run_mna_sweep(use_kernel=True)`) | The brief asks the matrix to record that the AC sweep refuses Newton devices; with the column, `mna_sweep.hpp` reads the table instead of `BranchKind::Nonlinear` (one rule, not two) | Drop `Engine::AcSweep`; restore the `BranchKind` scan |
| Row-count check | `kKindCaps = std::to_array<KindCaps>({...})` + `static_assert(size == DevicePool::kNumKinds)` (`kNumKinds = variant_size_v<Entry>`) | A `std::array<KindCaps, 20>{...}` with 19 initialisers compiles (the 20th row is zero); `to_array` takes its size from the rows | — |
| Refusal of `enable_nonlinear_refresh=False` | Only when the pool `requires_static_refresh()` (smooth / Shockley diode, MOSFET, IGBT), not for every Newton kind | The stateful kinds (junction capacitor, Lauritzen, PMSM, saturable inductor) are stamped by kernel wrappers whatever the flag says, so `False` changes nothing for them; refusing would refuse a run that is correct. Raised by the binding (`transient_refresh` in `bindings.cpp`), so the raw `run_transient`, `run_transient_trbdf2`, `run_transient_with_chain` and `simulate()` share it | Test `requires_newton()` in `transient_refresh` |
| Kernel fallback condition | `nl_refresh` empty and `pool.requires_static_refresh()` | Design decision 6; a Coss-only circuit keeps the empty chain, so its pwl/trbdf2 states are bit-identical (gate 4.4) | — |
| Message texts | `bdf1` and the AC sweep keep their texts (they now name the branch and kind); the DSED dispatch's hand guard is replaced by the kernel's generic text (`require_supported`), since its old text listed the kinds by hand and missed one | Design: "messages keep their current text" where a text exists that is still true | — |
| DSED kind check location | Kernel adapter constructor (`NativeCircuitBuilderAdapter`) **and** `_pulsim._require_supported(..., "dsed")` at the top of `run_dsed_from_builder` | The Python adapter (Bridge.10) runs `A_matrix()` before the native adapter is built and silently drops Newton devices; the refusal has to come first. The Bridge.11 `except (RuntimeError, ValueError): pass` now re-raises `UnsupportedError` first, so the fallback cannot swallow a refusal | — |
| Feature names | The `simulate()` keyword (`store_every`, `c_blocks`, `closed_loops`, ...); builder ICs on dsed = `initial_conditions`; extractor refusals = `extraction`; PWL diodes under `integrator='bdf2'` = `integrator` with `kind='Diode'` | Spec: "`feature` is the keyword name" | — |
| `_trbdf2_blockers` | Still returns a list of `str`; each is a `_Blocker(str)` carrying `.feature` | Three tests and `engine_route_reason` consume it as strings | — |
| Switch-function adapter | `python/switch_fn_adapter.hpp`, shared by `bindings.cpp` and `dsed_bindings.cpp` (two TUs). It captures raw pointers to the native schedules, never a `py::object` (the kernels run with the GIL released). Generic callables with a Python `next_edge_after` keep their DSED-only oracle path | Python callables on `trbdf2` keep today's sampling (no behaviour change outside the native types) | — |

## Silent cells found by the conformance test (not in the proposal)

| Cell | Before | Now |
|---|---|---|
| Coupled inductors (`add_transformer`, k = 0.9, 1 kHz) on `run_transient_bdf1` | ran with the windings uncoupled: 9.38 V max error on an 18 V span | `UnsupportedError(engine='bdf1', feature='coupled_inductors')` from `bdf1.hpp` |
| PWL diode with `V_f > V_th` (`add_diode(..., V_th=0, V_f=0.7)`) on `dsed`, sine clamp | diode held ON through the negative half-cycle: 0.70 V instead of −9.67 V (10.37 V error), no warning | `UnsupportedError(engine='dsed', feature='V_f', device=<branch>, kind='Diode')` from `require_supported(..., Dsed)`; workaround `V_th = V_f + 1e-3` (0.0016 V error). `V_th == V_f == 0` (the default diode) is not refused: it runs or raises the scheduler's "did not settle" error. The DSED event rule itself is not changed (non-goal). The refusal is per circuit, not per waveform: a diode that is only ever forward-driven ran correctly with `V_th = 0` (`test_dsed_t0_settle_with_a_drop`) and is now refused too, because whether reverse bias occurs is unknown before the run. That test now passes `V_th = V_f + 1e-3` (same analytic 10.2726 V); `test_dsed_forced_turn_on_without_a_band_names_the_band` now expects the `UnsupportedError` (a `ValueError`, was the scheduler's `RuntimeError`) with the same band hint. Reverse: drop the `Dsed` V_f block in `solver/unsupported.hpp` |
| `inductor_freeze_di_max`, `inductor_abs_clamp`, `enable_substep_state_correction` on `dsed` | silently ignored | added to the `_dsed_unsupported` keyword refusals |
| `NativeMultiMaskPwm.next_edge_after` | reported the period wrap of `[hi, lo, hi]` as an edge, so once trbdf2 used it the 1 µs-window run counted 60 gate events (20 non-events) | the oracle skips boundaries between equal masks (`sources/native_pwm_switch_fn.hpp`); 40 events |
| `enable_nonlinear_refresh=True` on a circuit without static Newton devices | routed the linear solve through Newton (states differ at ULP level from `None`) | the binding passes no chain, so `None`, `True` and `False` are bit-identical on such a circuit (spec scenario) |

## Proposal numbers

All reproduced on the old extension: dsed 6.3186 V (proposal 6.3185, rounding), bdf1 6.3002 V, pwl refresh off 10.0000 V vs 0.5937 V, multimask 0/40 at 1 µs and 1 µs·10 = 40/40 at 10 µs, `NativePwm2Switch` 1104 steps / 2254 solves / 20 events.
