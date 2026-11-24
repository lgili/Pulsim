## Why

Whether an engine can simulate a device is decided today by lists kept
by hand in several places, and the lists have diverged. Each divergence
is a run that finishes without a warning and reports the waveforms of a
different circuit. Measured on `feat/diode-forward-drop` (probe scripts
in `/Users/lgili/pulsim_wt/core_review_2026-11/arch/`):

- **`engine='dsed'` deletes a nonlinear capacitor.** RC (10 V, 1 kΩ,
  100 nF) with a 1 µF junction `Coss` across the capacitor:
  `v(100 µs)` = 1.0216 V on `pwl`, 1.0272 V on `trbdf2`, **6.3185 V on
  `dsed`** — the value of the circuit with no `Coss` (6.3185 V), zero
  warnings (`probe3.py`). The guard `DevicePool::has_nonlinear_devices()`
  (`core/include/pulsim/pwl/device_pool.hpp:460-473`, used at
  `python/pulsim/_dsed_dispatch.py:578`) is an enum list that omits
  `StoredKind::NonlinearCapacitor`.
- **`run_transient_bdf1` has the same hole**
  (`core/include/pulsim/solver/bdf1.hpp:297`): 6.3002 V with and without
  `Coss` (`probe5.py`).
- **The fixed engine simulates Newton devices as open circuits when the
  refresh is off.** `simulate(engine='pwl',
  enable_nonlinear_refresh=False)` and the raw bound `run_transient`
  (whose default is `False`, `python/bindings.cpp:2710`) give
  **10.000 V** on a Shockley clamp that reads 0.5937 V, zero warnings;
  `trbdf2` refuses the same call (`probe4.py`). The static device stamps
  are an optional callback built in the binding layer
  (`pwl::make_combined_diode_mosfet_refresh()`, four call sites in
  `python/bindings.cpp`), so a C++ caller of `run_transient` gets the
  open circuit by default.
- **`NativeMultiMaskPwm` is an opaque callable on `trbdf2`.** The
  binding looks for `SwitchSchedule` and `NativePwm2Switch` only
  (`python/bindings.cpp:4016-4047`); the DSED binding handles all three
  (`python/dsed_bindings.cpp:749-815`). Edges are then found by sampling
  every `h/4`: with a 1 µs window in a 1 ms period the run sees **0 of
  40** gate events, zero warnings; with a 10 µs window it sees 40 of 40
  (`probes/d10b_narrow.py` in this folder).
- Only 7 of 127 Python test files iterate over engines, so nothing
  catches the next hole.

## What Changes

- **One capability table in the kernel.** A `constexpr` table indexed by
  `DevicePool::StoredKind` states, for each device kind, its solve class
  (linear, source, switch, PWL diode, static Newton, stateful Newton)
  and the engines that accept it. A `static_assert` ties its length to
  the device variant, so adding a kind without a row does not compile.
  `has_nonlinear_devices()` is derived from the table and gains the
  missing kind.
- **One typed refusal.** `pulsim::UnsupportedError` (C++, derives from
  `std::invalid_argument`) carrying `engine`, `feature`, `device` (the
  branch name) and `kind`; surfaced in Python as
  `pulsim.UnsupportedError`, a subclass of `ValueError`, with the same
  four attributes. Every existing device-kind and feature refusal in
  `simulate()`, the DSED dispatch, `run_transient_bdf1` and
  `run_transient_trbdf2` raises it; messages keep their current text.
- **The check runs inside each kernel driver** (`run_transient`,
  `run_transient_trbdf2`, `run_transient_bdf1`, the DSED builder
  adapter), so the raw bindings and C++ callers get the same refusal as
  `simulate()`.
- **The kernel owns the static Newton refresh.** When the pool holds a
  static Newton kind and the caller supplied no callback, `run_transient`,
  `run_transient_trbdf2` and the DC operating point build the standard
  chain themselves. `enable_nonlinear_refresh=False` on a circuit with
  Newton devices is refused on every transient engine.
- **`NativeMultiMaskPwm` supplies exact edges on `trbdf2`** through one
  shared switch-function adapter used by the three binding sites.
- **C blocks on `engine='dsed'` are refused** (today: a warning, then an
  open-loop run, `python/pulsim/__init__.py:1996-2003`).
- **Conformance test.** For every (device kind, engine) and every
  (feature keyword, engine) the run either agrees with the reference
  engine within a stated tolerance or raises `UnsupportedError` naming
  the engine and the device or feature. Invariant: no silent cell.
- `pulsim.engine_capabilities()` returns the kernel table; a test checks
  the matrix in `docs/known-limitations.md` against it.
- **BREAKING (numbers):** circuits that were silently wrong now refuse:
  `NonlinearCapacitor` on `dsed` / `bdf1`; `enable_nonlinear_refresh=False`
  with Newton devices on `pwl`; C blocks on `dsed`. The raw
  `run_transient` binding now stamps Newton devices by default (10.000 V
  → 0.5937 V on the clamp above). `NativeMultiMaskPwm` runs on `trbdf2`
  change at the level of the edge-location error (7.4e-15 s measured)
  and gain the edges that sampling missed.

## Impact

- Affected specs: `engine-conformance` (new), `transient-timestep`.
- Affected code: `core/include/pulsim/pwl/device_pool.hpp` (table),
  a new `core/include/pulsim/solver/unsupported.hpp`,
  `solver/{run_transient,trbdf2_transient,bdf1}.hpp`,
  `dsed/builder_adapter.hpp`, `pwl/dc_operating_point.hpp`,
  `python/bindings.cpp`, `python/dsed_bindings.cpp`,
  `python/pulsim/{__init__,_dsed_dispatch}.py`, stubs,
  `python/tests/test_engine_conformance.py` (new),
  `core/tests/` (table and refusal unit tests), `KNOWN_LIMITATIONS.md`,
  `docs/known-limitations.md`, `python/pulsim/AI_GUIDE.md`,
  `CHANGELOG.md`.
- Dependencies: none. `fix-control-tick-semantics` owns three feature
  cells (two closed loops, multi-rate, `controller_period` on `pwl`);
  the conformance test carries them as strict expected failures naming
  that change until it lands. `add-unified-device-interface`,
  `add-unified-event-queue`, `add-kernel-simulation-object`,
  `add-pwl-igbt-and-thyristor` and `add-branch-form-ideal-switch` need
  this change.
- Users affected: anyone running a junction capacitor on `dsed`, calling
  the raw `run_transient` binding with Newton devices, or relying on
  `enable_nonlinear_refresh=False` to open them. Code that catches
  `ValueError` keeps working; see design.md for the two sites that raise
  `NotImplementedError` today.
