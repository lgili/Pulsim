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
