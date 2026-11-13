# fix-nonlinear-diode-passivity — decisions

## 2026-11-12, owner decision: retire the device

The softplus law below was stopped. The owner then decided to **retire**
`add_nonlinear_diode` instead of changing its law; proposal.md,
design.md, tasks.md and the spec deltas now describe the retirement.
`implementation-attempt.patch` stays as the record of the law attempt
and is not applied.

| Question | Option taken | Why | How to reverse |
|---|---|---|---|
| Where to refuse | Python wrapper installed by `_builder_ergonomics.install` (like `add_shockley_diode`'s), not the binding | The binding and C++ builder stay unchanged, so the C++ tests and stubs do not move; Python is the public surface | Delete the wrapper in `install` |
| Error type | New `pulsim.RetiredDeviceError(ValueError)` | A named error; `ValueError` as for `ProvenanceError` (the `allow_synthetic` refusal) | — |
| Opt-in name | `allow_non_passive=True` (Python), `allow_non_passive: true` (YAML) | Names the hazard the caller accepts, like `allow_synthetic` | rename in both places |
| YAML refusal type | the loader's `std::runtime_error` (Python `RuntimeError`) | Every other YAML refusal uses it; same message text as the Python refusal | — |
| Existing fixtures | The 11 Python test files that build the device on purpose pass the opt-in; C++ tests call the C++ builder (unchanged) and need nothing; the YAML round-trip test sets the key | Owner: keep their premise, do not rewrite them | — |
| Library users | `discovery.catalog()` lists `add_shockley_diode`; notebooks 05 and 07 use `add_shockley_diode` (their outputs cleared, the 07 operating point moves to the Shockley drop); notebook 15 (one snippet per YAML kind) sets the opt-in | Shockley is the documented smooth replacement | — |
| Kernel users | `IdealDiode` stays for the DC continuation, `nonlinear_refresh_diode.hpp` and the MOSFET Level-1 body diode (`nonlinear_diode_params`); none goes through the public builder method | Not public entry points | — |

## Earlier: the passive-law attempt (stopped)

**Status: not implemented.** The law was implemented exactly as proposed
and every number of proposal.md reproduced, but the change breaks 27
existing tests (12 ctest, 15 pytest), most of them in a way the proposal
did not foresee, and one claim of design.md is wrong in a way that moves
user-visible numbers by up to 0.4 V. Per the Phase 0 brief, the change is
left unimplemented for the owner to decide. The full attempt is kept in
`implementation-attempt.patch` in this folder
(`git apply openspec/changes/fix-nonlinear-diode-passivity/implementation-attempt.patch`
from the repository root applies cleanly on `feat/core-phase0` at the
commit that adds this file; it includes the law, `numeric/softplus.hpp`,
the importer mapping, the new tests, the updated C++ tests and the docs).

## What reproduced (measured with the attempt built)

| Quantity | Proposal | Measured |
|---|---|---|
| `i(0.5)`, `i(0.636)`, `i(0.7)`, `i(0.75)`, `i(1.0)`, `i(−10)` | 0.090746, 1.22662, 3.46573, 6.5663, 30.0124 A, −4.168e-6 A | 0.0907455, 1.2266236, 3.4657317, 6.566304, 30.012374 A, −4.16764e-6 A |
| strictly increasing, `i·v ≥ 0`, `i(0) = 0` on 550 001 points | yes | yes |
| Forced 0.636 V, pwl and DC op (old) | −1.3923 A | −1.3923 A old, +1.2266 A new |
| 20 V / 1 kHz clamp, pwl dt = 1 µs (old) | `SimulationAborted` | `SimulationAborted` at t = 1.616 ms old |
| Clamp RMS / peak, trbdf2 (old → new) | 9.634 → 10.010 V / 0.662 V | 9.6341 → 10.0096 V, peak 0.6625 V (pwl and trbdf2) |
| Importer `D(Ron=0.05 Roff=1Meg Vfwd=0.7)`, 10 A | `Diode`, 20 S, 1e-6 S, 1.2 V, 12.0 W; area 2 → 40 S | as stated |

## Why it was stopped

1. **design.md is wrong about the knee at low current.** It says the knee
   "is rounded outward by at most `ln 2/κ` = 34.7 mV in voltage". That
   bound holds in *current* (`sp(x) − max(x, 0) ≤ ln 2`, i.e. at most
   `ln 2/(κ·R_d)` = 3.47 A), not in voltage. Below the current scale
   `1/(κ·R_d)` = 5 A the law is an exponential tail with slope `κ`
   (115 mV/decade), so the voltage falls far below `V_F0`:

   | Load line | old law `v_d` | new law `v_d` |
   |---|---|---|
   | 2 V, 1 kΩ (1.7 mA) | ≈ 0.70 V (I = 1.300 mA) | **0.30078 V** (I = 1.6992 mA) |
   | 5 V, 1 kΩ (4.6 mA) | ≈ 0.70 V | **0.35104 V** |
   | 1 A | — | 0.6246 V (as design.md states) |

   Every small-signal / DC use of `add_nonlinear_diode` at mA currents
   moves by ~0.35–0.4 V. With `V_F0` overridden to 0.3 V the reverse floor
   `sp(−κ·V_F0)/(κ·R_d)` is 12 mA, so a 2 V / 1 kΩ load line reads
   `v_d` = 7.5 mV. Open Question 3 (raise the default `κ`) matters more
   than design.md assumes: at κ = 100 the tail scale is 1 A.

2. **Many fixtures rely on the defect itself.** The negative-conductance
   region is what made Newton fail on these circuits; with the passive law
   they converge, so the test premise ("this really does fail") is gone.
   Only `layer5/test_dt_retry.cpp` is in tasks.md 2.3.

   | Test | Premise that disappears |
   |---|---|
   | `layer4/test_gmin.cpp` "gmin stepping solves a chain the direct solve cannot", "Auto falls through to the rung that can answer" | 20-diode reverse chain: the direct solve now converges (unique operating point; the test's own comment attributes the multiple roots to the non-monotone law). Its node then reads 9.99584 V (the 4.16 µA reverse floor), not 10.000 V |
   | `layer5/test_dt_retry.cpp` (4 cases) | 400 V mains rectifier fails at dt = 2e-4 |
   | `python/tests/test_dc_operating_point.py::test_gmin_stepping_solves_what_the_direct_solve_cannot`, `::test_auto_falls_through_and_says_which_rung_answered` | same chain |
   | `test_hostile_circuits.py::test_blocking_chain_finds_its_operating_point_unaided`, `::test_a_mains_rectifier_runs_at_a_coarse_dt`, `::test_the_retry_does_not_move_a_single_sample`, `::test_opting_out_restores_the_hard_failure` | blocking chain / mains rectifier fail without the ladder |
   | `test_partial_results.py` (4 cases), `test_phase2_review_fixes.py::test_the_retry_says_it_happened`, `::test_the_partial_trace_can_resolve_names` | a run that aborts mid-way (they need a failing step) |
   | `test_globalization_promotion.py::test_the_ladder_composes_over_the_whole_sweep` | a step that needs line-search / LM promotion |

   Probed replacements: no circuit made of the passive smooth diode or of
   Shockley diodes (reverse chains with `G_off` 0…1e-9, 10–1000 V; forward
   chains; mains rectifiers 400/4000 V, κ 100/2000, `R_d` 1e-2/1e-4,
   C 100 µF/10 mF, dt 2e-4/1e-3) failed the direct solve or a fixed step.
   An **iteration budget** reproduces the dt-retry premise: the 400 V
   rectifier with `max_newton_iterations = 6` fails at dt = 2e-4 (step
   ending t = 5.4 ms needs 7 iterations) and succeeds after exactly one
   halving (5 → six halvings, 7 → no retry). The attempt patch uses that
   for `test_dt_retry.cpp`; it is a one-iteration-wide window and may move
   across platforms. No budget-based stand-in was found for the gmin and
   Auto fall-through premise, because those rungs run with their own
   internal limits.

3. Three further numeric expectations move (each re-derived in the patch
   from a closed-form reference, `core/tests/support/smooth_diode_reference.hpp`):
   `test_diode_load_line`, `test_integration_smooth_rectifier`,
   `test_circuit_builder` "Nonlinear diode builder", `test_continuation_rectifier`,
   `test_vf0_continuation_rectifier`, `test_gmin` "honours nonlinear devices",
   `test_dc_operating_point::test_compute_dc_op_stamps_nonlinear_devices`,
   `test_dsed_refuses_wrong_answers::test_the_pwl_engine_still_takes_both`
   (0.70 V → 0.351 V).

## What the owner needs to decide

- Accept the low-current tail of the softplus law (and update the docs to
  state it), or change the law/defaults (e.g. raise `κ`, Open Question 3;
  or a law whose knee is sharp in voltage, such as a softplus in current
  space).
- How the "this circuit really fails" fixtures are rebuilt: an iteration
  budget (works for dt-retry), a different device, or a dedicated
  test-only failing device.
- Whether the importer half (Ron/Roff/Vfwd → `add_diode(V_f=Vfwd)`, which
  is independent and broke no test) should land on its own first.

## Open Questions (as they would have been answered)

| # | Question | Option taken in the attempt | Why | How to reverse |
|---|---|---|---|---|
| 1 | Refuse / warn on a large reverse floor? | No | Not proposed; DC continuation uses such sets | add a check in `CircuitBuilder::add_nonlinear_diode` |
| 2 | Importer `V_th = 0` or `Vfwd + 1 mV`? | `V_th = 0` | Proposed; LTspice semantics | `_spice_diode`: `V_th=Vfwd + 1e-3` |
| 3 | Raise the default `κ`? | Keep 20 /V | Not proposed — but see finding 1 | `IdealDiode::Params::kappa`, binding and YAML defaults |

The attempt also refused `kappa <= 0` (the softplus law divides by κ; the
old law accepted κ = 0).

## Found in passing, not fixed (out of scope)

- `spice_to_builder` maps `I1 n+ n- DC I` to
  `add_current_source(n+, n-, I)`, but the builder's source drives its
  current out of its first node into the circuit, the opposite of SPICE:
  `I1 0 a DC 1` + `R1 a 0 1` reads `v(a) = −1.0 V` (SPICE: +1.0 V).
- `python/tests/conftest.py` imports the installed wheel (a stale 1.8.0 in
  the framework Python) unless `<repo>/build/python/pulsim` holds the
  extension, even with `PYTHONPATH` set. The phase-0 worktree needed a
  gitignored `build/python/pulsim -> ../../python/pulsim` symlink (the
  other worktrees have one); `.scratch/verify.sh` alone tests the wrong
  package.
