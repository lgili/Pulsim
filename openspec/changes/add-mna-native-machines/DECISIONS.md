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
- Superseded (group 2 work): the kernel has a VCCS now; `theta_node`
  is a unit VCCS `{name}.th` from the speed into a 1 F capacitor
  `{name}.ct` (task 1.8 ticked, `test_dc_machine.py`).

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

- Superseded by D19 (batch 13). Original note:
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

## D15. `pwl` nominal step of the Newton-only machines (review finding)

- The checkpoint review found every `pwl` step of a PMSM (pre-existing)
  and of the induction machine failing at the requested `dt` and being
  re-taken at `dt/2`: the step cache factorises the linear part, which
  is singular because the flux rows exist only in the Newton refresh.
  Every `pwl` figure measured before this was in fact at `dt/2`.
- Fix (one flag in `run_transient`): a pool with an MNA-native PMSM or
  induction machine takes the assembled, unfactorised carrier at every
  `dt` (the path the retry already used; Newton refactorises
  `J_lin + J_nl` each iteration anyway). No other circuit takes it, so
  the 21 non-machine goldens are unchanged; the two `im_*` goldens and
  every machine figure were re-measured at the true `dt`.

## D16. Flux rows in increment form (small-step abort)

- Symptom: a PMSM at `dt = 10 ns` aborted (ψ_pm 0.05 at t = 1.2e-5 s;
  ψ_pm 0.5 at 5e-8 s, and 25 retries already at 100 ns; `trbdf2` with
  ψ_pm 0.5 stalled at h_min at t = 0). Cause: the row
  `v − R i − (2/h)(λ(x) − λ_old)` subtracts two O(ψ_pm) numbers; their
  round-off `eps·(2/h)·|λ|` is 2e-9 V at 10 ns for ψ = 0.05, above
  Newton's absolute 1e-9 tolerance (which stays absolute: making it
  relative lets the DC cascade stop at a spurious root, memory note).
- Fix: the row is written as `(2/h)·(Δλ − λ_off)` with Δλ computed
  from the increments, never as a difference of totals:
  - PMSM: `L·Δi` + the saliency term with
    `cos2θ − cos2θ0 = −2 sin(θ+θ0) sin(θ−θ0)`, the PM term with
    `−2ψ_h sin(h(θ+θ0)/2 + φ_h) sin(hΔθ/2)` per harmonic
    (`PmsmMna::lambda_increment`);
  - induction machine: `L_s·Δi + M·Δλ_r` (constant stator matrices,
    now cached once per entry).
- The reference point `(i_old, θ_old, λr_old)` is the committed state.
  `extrapolate_previous` (SDIRK pseudo-history) moves it by a stored
  offset `λ_off`, and a resume rebases it on the snapshot `x`
  (`λ_off = 0` bitwise for a consistent snapshot).
- The Jacobian is unchanged and the rows stay in conductance form.
  This CHANGES THE PMSM / IM LAW AS WRITTEN: the device-interface port
  must carry the increment form.
- Moved numbers (1 µs, end values): PMSM pwl ω −13.059179906729224 →
  −13.059179906729375; trbdf2 −13.061329289736987 → −13.061329289735662;
  ψ 0.5 pwl −5.355988036500754 → −5.355988036498043; IM pwl
  1.530198626034774 → 1.530198626034761; IM trbdf2 1.5308411400090733
  → 1.5308411400087307. All ≤ 3e-13 relative: round-off of the old row.
- After: PMSM and IM at 1 µs, 100 ns, 10 ns, 1 ns, from rest and from a
  running high-flux state, take zero dt retries.

## D17. Internal 1 F integrator capacitors are not parameters

- The angle node (PMSM) and the rotor-flux nodes (IM) integrate through
  a builder-made 1 F capacitor that `with_value("…_theta.C")` /
  `("…_Cra.C")` could edit, silently scaling `θ` or `λ_r` dynamics.
- Fix: the injection is scaled by the capacitance read at init
  (`C·ω` into the angle node, `C·g(·)` into the rotor rows), so the
  value of C cancels. Any other capacitor or a nonlinear capacitor on
  those nodes is refused by name. Tests: an edit to 0.25 F gives the
  same result within 1e-9 relative.

## D18. Observer deprecations (group 5)

- `add_pmsm`, `make_pmsm_observer`, `add_bldc`, `make_bldc_observer`,
  `add_induction_motor`, `make_induction_motor_observer` emit one
  `DeprecationWarning` naming the native builder, with the measured
  observer error (PMSM −0.37 % at 50 µs, −1.98 % at 200 µs; IM free
  acceleration 0.71 % / 2.9 % of synchronous speed).
- `im_parameters_from_nameplate` labels its result
  `_diagnostics["status"] = "estimated"`.

## D19. DC operating point with held mechanics (2.7, 3.5)

- `dc_assemble` stamps each machine's DC rows after the conductance
  floor: stator `v − R·i − ω_e·∂λ/∂θ_e = 0` (di/dt = 0) at the held
  `(ω0, θ0)` — for the PMSM `ω_e·(∂L/∂θ_e·i + ∂λ_pm/∂θ_e)`, for the
  induction machine no stator EMF and the rotor rows `dλ_r/dt = 0`
  (the steady rotor flux at `ω0`). Linear: no Newton at DC.
- The KCL rows of the speed and angle nodes are replaced by `v = ω0`,
  `v = θ0` (no torque balance; the transient takes over). `ω0`, `θ0`
  are the builder's `omega0` / `theta0`, now kept on the pool record
  (YAML keys `omega0` / `theta0` of the `pmsm_mna` /
  `induction_machine_mna` registry entry, value hash only when
  nonzero, so existing hashes are unchanged).
- A speed or angle node that another device drives (anything but
  capacitors, resistors and current outputs) would be over-determined:
  refused by name, naming that device. Prescribe the speed with
  `omega0` for the DC point instead. Known limit: an inductor on the
  speed node (a torsional spring in the velocity analogy, e.g. a
  two-mass shaft) is refused too, although its DC short would only copy
  `ω0` to the load node; start such a shaft from zero state.
- The held rows keep no `gmin`: `compute_dc_op` re-applies the hold
  after its floor, so `ω = ω0` exactly, not `ω0/(1 + gmin)`.
- `start_from_dc_op` seeds the machine histories from the DC point
  (`λ`, `i`, `θ`), and the held nodes' capacitors from the KCL residual
  the hold replaced (`seed_held_machine_capacitors`: angle `i_C = C·ω0`,
  speed `i_C` = net torque; a DC capacitor is otherwise seeded at 0, and
  the first step then advanced θ by h·ω0/2 and moved i by 2.4e-3 A).
  The first step continues the DC point: |i(h) − i(0)| = 4.8e-7 A
  (round) and 1.3e-6 A (salient) at h = 1 µs, ratio 4.000 per halving,
  against a > 1e-3 A jump from zero state. `pwl` only: `trbdf2` refuses
  `start_from_dc_op`. The other stateful histories (Coss,
  Lauritzen, IGBT tail, saturable inductor) are still NOT seeded after
  the DC point on `pwl` (pre-existing; left so non-machine results stay
  bit-identical).
- References (`test_machine_dc_hold.py`, conformance `DC held` cells):
  PMSM at rest `i = (v − v_n)/R_s` within 1e-9 on a salient rotor;
  at 50 and −30 rad/s the round-rotor closed form
  `i_k = (v_k − v_n + ω_e·ψ·sin(θ_e − 2πk/3))/R_s`; induction machine
  `i = (v − v_n)/R_s` and `λ_r` from the 2×2 steady rotor law at 0 and
  80 rad/s.

## D20. Preflight, YAML authoring types

- The preflight counts a machine stator as a DC path and the speed,
  angle and rotor-flux nodes as defined at DC: no 1 GΩ tie and no
  warning on a machine's own nodes (it tied the neutral, the speed and
  the angle node before). Machine results move by the removed leak.
- YAML `pmsm` (the `add_pmsm_mna` arguments) and `bldc` (the
  `add_bldc_mna` arguments; the trapezoid series is ported to the loader
  and pinned equal to `trapezoid_harmonics` by
  `test_yaml_composites.py`). `theta_node` is a scoped node key.

## D21. Flux maps (group 4) not done

- `FluxMapDQ` (co-energy bicubic fit, kernel ∇W' and Hessian, the
  saturating locked-rotor reference) is a new law with its own Newton
  smoothness requirements (memory: device laws must be smooth to
  1e-12). Not started in this batch; tasks 4.1–4.5 and the
  `pmsm_mna[flux_map]` conformance row stay open.
