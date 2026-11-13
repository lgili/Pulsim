## Why

A device kind is not one thing in the kernel: it is a `StoredKind` value
that every consumer switches on. Counted on this branch
(`core/…` = `core/include/pulsim/…`):

- **123 `StoredKind::` dispatch sites in 19 files**
  (`grep -rn "StoredKind::" core/include python/*.cpp | wc -l`); the top
  five are `pwl/device_pool.hpp` 32, `python/bindings.cpp` 20,
  `pwl/assemble.hpp` 12, `pwl/dc_assemble.hpp` 11,
  `solver/trbdf2_transient.hpp` 8.
- **Five stateful-device history classes with the same de-facto API**
  (`init`, `empty`, `entries`, `update_from_state`, `seed_from_dc_op`,
  `snapshot`, `restore`, `to_flat`, `from_flat`; four add
  `capture_gamma`): `pwl/{nonlinear_capacitor,lauritzen_diode,igbt_tail,pmsm_mna,saturable_inductor}_history.hpp`.
  The two drivers call them by name **99 times**: 56 in
  `solver/run_transient.hpp`, 43 in `solver/trbdf2_transient.hpp`
  (`grep -oE "(coss|laur|pmsm|tail|sat)(_history)?\.[a-z_]+\("`).
- **The Newton refresh is a chain of five `std::function` wrappers,
  written twice**: `solver/run_transient.hpp:471-600` and
  `solver/trbdf2_transient.hpp:588-697`. The base of the chain is an
  optional callback chosen by the binding
  (`make_combined_diode_mosfet_refresh()`, four call sites:
  `python/bindings.cpp:2640, 2968, 4006, 4139`);
  `add-engine-conformance-matrix` makes the kernel build that base when
  no callback is given, and leaves the chain.
- **Each stateful device carries its own time integrator** (a hand
  derivation of the trapezoidal and the BDF2 stage formula inside its
  history class) and its own named field in `SolverSnapshot`
  (`solver/result.hpp:169-177`).
- **`DeviceModel` is algebraic and voltage-controlled only**
  (`models/device_model.hpp:44-52`): it has `current(v, p)` and nothing
  for charge, flux, internal unknowns or state, so five of the eight
  Newton devices bypass it (audit `raw/models.md` §c).
- **Adding one stateful device costs about 60 edits in 12–15 files**
  (audit, Lauritzen as the specimen; `grep -rci lauritzen` finds 105
  hits in 15 files of `core/include`, the two binding sources and
  `python/pulsim`).

Every one of these has already produced a wrong answer with no error:

| Symptom | Structural cause | Stop-gap change |
|---|---|---|
| D3: `dsed` and `bdf1` delete a `NonlinearCapacitor` (6.3185 V with and without Coss; pwl 1.02 V) | `has_nonlinear_devices()` is a hand-kept kind list (`pwl/device_pool.hpp:460-473`) | `add-engine-conformance-matrix` |
| D4: pwl with the refresh off simulates Newton devices as open circuits (10.000 V instead of 0.594 V) | device stamps are an optional callback | `add-engine-conformance-matrix` |
| D2: kernel AC sweep returns 0 for a coupled transformer, treats Shockley / Coss as open and a saturable L as a short | `E` is assembled by hand from two kinds (`analysis/mna_sweep.hpp:118-134`) | `fix-mna-sweep-correctness` |
| `res.i()` raises `NotImplementedError` for five Newton kinds; `device_loss_summary` lists none of them | currents are reconstructed per kind in Python (`python/pulsim/__init__.py:1129, 1158, 1177`) | — |
| A disconnected saturable inductor dropped an IGBT's turn-off energy from 16.87 mJ to 0.045 mJ | one wrapper captured the raw callback (owner note *nonlinear-refresh-wrapper-chain*) | fixed in place |
| SIGSEGV in the TR-BDF2 diode settle beside a saturable inductor or PMSM | the stamps read a captured `(dev_h, dev_stage)` that probe solves did not set (owner note *trbdf2-probe-solves-set-the-device-step*) | fixed in place |
| A resumed run lost 8 mWb of flux, a Coss fell from 109.85 V to 0.28 V, a 17.7 A tail vanished | stateful histories were not in the snapshot (`solver/result.hpp:153-162`) | fixed in place, one field per kind |

Probes: `core_review_2026-11/arch/{matrix,probe2,probe3,probe4,probe5}.py`,
`models/{p1_engine_matrix,p3_ac,p6_misc}.py`.

## What Changes

- **One residual contract.** A Newton device contributes
  `f(x, t) + d/dt q(x) = 0` to its rows through a single templated
  `eval(params, x, ctx, out)` that fills `out.f[r]` and `out.q[r]` over
  an AD scalar. The Jacobian (`∂f/∂x`, `∂q/∂x`) comes from AD; table
  laws supply their exact tangent through `ad::chain`.
- **The integrator owns the time formula.** One `stamp_fq<D>` turns
  `(f, q)` into the Newton row for the formula the solve asks for
  (`Dc`, `BackwardEuler`, `Trapezoidal`, `Bdf2Stage2`). The step context
  is an argument of every solve; no stamp reads a captured step size.
  Rows stay in conductance form (`a·∂q/∂x` with `a = 2/h`, `c1/h`): the
  flux row is never rescaled by `h`.
- **Optional members, detected at compile time:** `State` + `init` /
  `accept` (hysteresis, latches), `events` (zero-crossing functions),
  `outputs` (internal quantities), `fields` (name, member pointer, unit,
  bounds: the single source for binding keywords, YAML keys,
  `components()` and docs).
- **PWL two-state devices stay a second category** (`ModeDevice`): a
  per-mode constant stamp into the cached `(G, C, b)` split plus the one
  shared `decide` / `crossing_signal` law that `SwitchedDiode` already
  has. Linear devices and sources are a third (`StaticDevice`). All
  three sit in one compile-time registry; `StoredKind` is derived from
  it.
- **One flat device-state vector owned by the run.** Per instance:
  `(q_n, q̇_n)` per charge row, then the reals of `State`. Rollback is
  one copy. The linear companion history keeps its arithmetic and moves
  into the same vector.
- **BREAKING (snapshot format):** `SolverSnapshot` becomes
  `{t, x, dev_state, mask, stamped_mask, layout_signature}`. The five
  named history fields (`saturable_history`, `coss_history`,
  `lauritzen_history`, `igbt_tail_history`, `pmsm_history`), `history`
  and `diode_on` remain as read-only views for one release and are then
  removed. Pickles written by 2.0 (10- and 11-field tuples) keep
  loading.
- **BREAKING (binding):** `run_transient(..., enable_nonlinear_refresh=)`
  and the C++ `nl_refresh` parameter are deprecated for one release and
  then removed; the kernel builds the device stamps from the registry.
- **Generic outputs.** Pin currents `i_k = f_k + d/dt q_k` and power
  `Σ v_k·i_k` exist for every device with no per-kind code, so `res.i`,
  `res.p` and `device_loss_summary` cover `mosfet_level1`,
  `igbt_level1`, `lauritzen_diode`, `nonlinear_diode`,
  `nonlinear_capacitor` and the saturable inductor.
- **AC and DC derived.** DC is `f` alone (`a = 0`). The small-signal
  system at an operating point is
  `(G_lin(mask) + ∂f/∂x) + jω·(C_lin + ∂q/∂x)`. `run_mna_sweep` uses it
  and stops refusing Newton kinds.
- **Deleted at the end:** the five `pwl/*_history.hpp`, the eight
  `pwl/nonlinear_refresh_*.hpp`, both wrapper chains,
  `NonlinearRefreshFn`, `has_nonlinear_devices()`,
  `models/device_model.hpp` (subsumed), the kind ladder in
  `python/pulsim/__init__.py:1004-1188` and the Python side table
  `_shockley_params`.

## Impact

- Affected specs: `device-interface` (new), `device-models`,
  `ac-analysis`.
- Affected code: new `core/include/pulsim/device/`; `ad/ad_scalar.hpp`
  (`atan`, `atan2`, `pow`, `sin`, `cos`, `chain`); `models/*.hpp`;
  `pwl/{device_pool,assemble,dc_assemble,history_state,nonlinear_solve}.hpp`;
  `solver/{run_transient,trbdf2_transient,result}.hpp`;
  `analysis/mna_sweep.hpp`; `builder/circuit_builder.hpp`;
  `yaml/loader.hpp`; `python/bindings.cpp`, `python/dsed_bindings.cpp`;
  `python/pulsim/{__init__,_builder_ergonomics,losses,thermal,steady_state,mna_sweep}.py`;
  stubs; docs.
- Depends on: `add-engine-conformance-matrix` (the device × engine
  matrix is the safety net and its capability table becomes a function
  of the concept traits), `restore-reference-benchmark-harness` (golden
  hashes of the fixed engine, differential runner).
- Builds on, when present: `update-newton-persistent-solver` and
  `update-linear-time-assembly` (their stamp-slot map is the
  `Instance::jac_slot` array here).
- Unblocks: `refactor-unified-transient-stepper`,
  `add-probes-and-recorders`, `add-power-mosfet-datasheet-device`,
  `add-mna-native-machines`, `add-external-device-abi`,
  `add-parameter-sensitivities`.
- Performance constraints: no virtual call and no `std::function` per
  device per Newton iteration (static dispatch over the type list, one
  compiled loop per kind); zero heap allocations in the step loop
  (`core/tests/perf/test_zero_alloc.cpp` extended to every Newton kind);
  workspaces stay thread-local; per-iteration stamp cost for the ported
  kinds not above today's (counter gate from the harness plus a
  before/after timing of audit rows 10–13). Compile time: the registry
  instantiates one stamp loop per kind per formula; measured before and
  after, with a 15 % budget on the extension build.
- Numerical results: linear and PWL-only circuits stay bit-identical at
  every step of the migration. A circuit containing a ported Newton
  kind may move by floating-point reassociation; each port states its
  measured maximum deviation (budget 1e-10 relative) and re-baselines
  only the goldens that contain that kind.

## Re-centred scope, 2026-11-13

> Appended; the sections above stand except where named.

The PWL category (`ModeDevice`, `SelfCommutated`, D4) must carry the
complementarity form that `add-complementarity-mode-selection` solves:
for each self-commutated port the ideal element's orientation, `V_f`,
`R_on`, `g_off` and the hysteresis threshold `V_th`, and the crossing
signals in branch form (current ON, gap OFF) from
`add-branch-form-ideal-switch` D5. `decide` stays as the legacy law
until the selection service is the default, then becomes a fallback
used only by `mode_selection='legacy'`. The sentence of D4 "its
`decide` / `crossing_signal` are the functions that exist today,
unchanged" holds until that flip.
