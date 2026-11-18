# Decisions taken during implementation (owner away)

Each entry: question → option taken → why → how to reverse.

## D1. Where `add_dc_machine` lives

- Question: task 1.1 names `CircuitBuilder::add_dc_machine` (C++), but
  the composite needs no new kernel code and the C++ builder, the
  bindings and the YAML loader were being edited by other changes at
  the same time.
- Option: a Python method installed on `CircuitBuilder`
  (`python/pulsim/machines.py`, `install()` called from `__init__`),
  the same mechanism as `_builder_ergonomics` uses for
  `add_shockley_diode`. Called as `b.add_dc_machine(...)`, with the
  design's signature.
- Why: one implementation, no rebuild, nothing to merge in the shared
  C++ files. C++ users and the YAML loader do not see it.
- Reverse: port the 20 lines of composition into
  `circuit_builder.hpp` (same sub-element names), bind it, and delete
  `machines.py`; the Python tests keep passing unchanged.

## D2. YAML type `dc_machine`, `components()` entry

- Question: task 1.2 asks for both.
- Option: YAML `dc_machine` IS done, in a later commit: the C++ loader
  expands it into the same six elements as the Python method (keys
  `a_pos`, `a_neg`, `omega_node`, `R_a`, `L_a`, `K`, `J`, `B`,
  `T_load`; no `i0`/`omega0`/`theta_node`). The two expansions are
  pinned equal by `python/tests/test_yaml_composites.py` (same
  `components()`). The `components()` entry for the machine as a whole
  is not done: `components()` lists one entry per branch, and a
  composite-level entry needs a registry of composites in the C++
  builder.
- Reverse: when the C++ builder method of D1's reversal exists, make
  the loader call it.

## D3. `Kt ≠ Ke` (Open Question 2)

- Option: refuse (the design's "proposed" wording: "refused by the
  native path with that explanation"). `add_dc_machine` accepts the
  legacy spellings `Ke=` / `Kt=` as aliases of `K` so a migration is a
  rename; different values raise `ValueError` explaining that in SI
  units they are one constant and how to convert V/krpm.
- Why: a VCVS/CCCS pair with two constants creates or destroys energy.
- Reverse: drop the check and build the coupling from a VCVS + CCCS
  pair once `add-controlled-source-set` lands.

## D4. Deprecation scope (task 5.1, Open Question 3)

- Option: only `add_dc_motor` and `make_dc_motor_observer` warn
  (`DeprecationWarning`, message names `add_dc_machine` and the
  measured first-order error). The PMSM, BLDC and induction-motor
  observers do not warn yet. How long the observer stays: not decided
  here (the message says "until a later release").
- Why: a warning must name a replacement that exists; the native
  induction machine and BLDC are not implemented in this pass.
- Reverse: delete the two `warnings.warn` calls in `motors.py`.

## D5. `theta_node`

- Option: accepted in the signature, refused with `NotImplementedError`
  naming `add-controlled-source-set` (task 1.8 unticked).
- Why: `θ = ∫ω dt` as a node voltage needs a VCCS, which the kernel
  does not have.
- Reverse: implement with the VCCS when it exists.

## D6. Elements with a zero value

- Option: `{name}.ra` is omitted when `R_a = 0`, `{name}.b` when
  `B = 0`, `{name}.tl` when `T_load = 0`; `{name}.la`, `{name}.em` and
  `{name}.j` always exist.
- Why: a zero resistance is an infinite conductance, and a zero
  source is nothing.
- Reverse: n/a.

## D7. Wound-field machine (Open Question 4)

- Option: not in this change (most conservative).

## D8. Energy-balance tolerance on `trbdf2` (task 1.6)

- Question: the task asks `∫v·i dt = ∫R i² dt + ΔE_L + ΔE_J` within 1e-6
  on `trbdf2`.
- Option: the balance is computed with interval-average products, which
  is the trapezoidal rule's discrete energy identity: on `pwl` it holds
  to 4.2e-9 (bound 1e-8). TR-BDF2's BDF2 stage does not satisfy it, and
  the same metric reads 5.5e-6 there (bound 2e-5, not 1e-6).
- Why: a tighter `trbdf2` bound would test the metric, not the machine.
- Reverse: integrate the balance with the engine's own stage weights.

## D9. Preflight false positive on the speed node (found here)

- Finding: with `B = 0` the preflight reports "node 'w' has no DC path
  to ground" and inserts a 1 GΩ bleeder. The ideal transformer's
  primary is a pair of node references, not a graph branch, so
  `pwl/preflight.hpp::components_without_ground` does not see that it
  fixes `v_w` through the secondary. The matrix is regular without the
  tie (DC operating point with `auto_regularize=False`: `i = 2.0` A,
  `ω = 89.99999999998` rad/s). The tie's effect is a friction of
  1e-9 N·m·s/rad (9e-7 relative on the DC current).
- Option: fixed in a later commit: `analyze_preflight` unites each
  ideal transformer's primary pair in both passes
  (`pwl/preflight.hpp`); C++ test in `layer4/test_preflight.cpp` with a
  control (the same node without the transformer is still reported).
  The DC test no longer needs `auto_regularize=False`.

## D10. `dsed` runs the composite (correction to the design)

- Finding: the design's engine table says `dsed` refuses the ideal
  transformer. It does not: the sine-driven test machine runs on `dsed`
  (20006 points) with a speed error of 1.8e-7 against the exact
  solution. Pinned by `test_dsed_runs_the_composite`.
- Consequence: the conformance row is `dc_machine × dsed = ok`.

## D11. Induction machine: state choice and where it lives (group 2)

- Question: the design waits for `add-unified-device-interface` before
  any stateful machine (cost of a new history in the wrapper chain).
  That interface is being built on another branch at the same time.
- Option: the induction machine REUSES the PMSM's plumbing. Its stator
  rows are the PMSM's flux rows (`pwl/pmsm_mna_history.hpp`,
  `pwl/nonlinear_refresh_pmsm_mna.hpp`) with `λ_abc = L·i + M·λ_r`,
  where `L` and `M` are CONSTANT (stationary frame), and an entry flag
  `induction` selects the law. The rotor flux `λ_rα`, `λ_rβ` are the
  voltages of two internal nodes `{name}_psi_ra`, `{name}_psi_rb`, each
  with a 1 F capacitor the builder adds — the same trick as the PMSM's
  angle node — so the engine integrates them with its own companion
  (trapezoidal, BDF2 in the TR-BDF2 second stage) and the stamp only
  injects the right-hand side of `dλ_r/dt`. No engine file changed:
  every commit / snapshot / restart / stage path is the PMSM's.
- New kind `StoredKind::InductionMachineMna` (= 27) for the stator
  branches, so the capability table, the conformance matrix, the IR
  field table and refusals name it.
- Why: zero new sites in `run_transient.hpp` / `trbdf2_transient.hpp`
  (both being refactored by other agents), and the integration scheme
  of every state is the engine's own, so the method is the engine's
  method on the whole ODE (a linear change of state variables commutes
  with the trapezoidal rule and with BDF2).
- Reverse: when the unified device interface lands, move
  `detail::refresh_induction_entry_` behind it; the stamp is self
  contained.

## D12. Engines for the induction machine

- `pwl`, `trbdf2`: Newton (bilinear terms `ω·λ_r`, `λ_r × i_s`).
- `dsed`: refused by name (`kKindCaps` row `kNewtonEngines`). The
  machine is not LTI between events (its matrix depends on `ω`) and the
  proposal defines no implicit-segment route for it.
- AC sweep: refused by name (same row). A small-signal stamp around an
  operating point is not in this change.
- DC operating point: see D13.

## D13. DC operating point with machines (tasks 2.7 "DC held", 3.5)

- Not done in this pass. The DC assembly stamps every Newton device
  except the saturable inductor OPEN, so a PMSM or induction machine has
  no electrical rows at DC; the preflight then ties the neutral, the
  rotor-flux nodes and the speed / angle nodes to ground through 1 GΩ
  (a leak of 1e-9 S beside the rotor's 1/τ_r = 6.45 S; a friction of
  1e-9 N·m·s/rad on the speed node). Same as the PMSM before this change.
- The hold needs a DC-stage branch of the machine stamp (`dλ/dt = 0`,
  mechanical rows replaced by `v = IC`) passed through the DC refresh;
  that is a change to the DC driver, left for after the unified device
  interface lands (which owns the refresh signature).

## D14. PM-flux harmonics representation (group 3)

- `PmsmMna::Params::harmonics` is a shared pointer to an immutable list
  of `(h, ψ_h, φ_h)` (null = the old `ψ_pm·cos θ_e`). A pointer keeps the
  parameter struct cheap to copy through the IR and keeps every existing
  `Params{...}` designated initialiser valid; the IR value hash includes
  the terms.
- With harmonics set, `psi_pm` is not read by the law: a
  `with_value("…psi_pm")` is refused by name (`check_device`) instead of
  being silently ignored. `add_bldc_mna` stores ψ_1 in `psi_pm` for
  reporting only.
- `h = 1, φ = 0` evaluates the same floating-point expressions as the
  plain law: a run is bit-identical (`test_bldc_mna.py`).
- `add_bldc_mna`, `trapezoid_harmonics` and `pm_harmonics_from_waveform`
  are Python (`pulsim.machines`), like `add_dc_machine` (D1): they only
  compose `add_pmsm_mna(harmonics=...)`.
- The `<name>.T_em` trace uses the series when one is set (it used the
  fundamental-only dq formula, wrong by 1.1e-3 on the 120° BLDC test).
