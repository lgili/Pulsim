## Why

Three of Pulsim's four machine models live outside the matrix. The DC
motor, the BLDC and the induction motor are Python observers: the
back-EMF is computed from the previous step's state and injected through
`b_extra_fn`, and the mechanics and the rotor flux are advanced by
forward Euler (`python/pulsim/motors.py:240-249`, `:1047`, `:1110`).
They are first order in the step, run on `pwl` only (they need a
`step_observer`), and are invisible to `trbdf2`, `steady_state` and the
DC operating point.

Measured on this branch (review probe `models/p5b_dcmotor.py` under
`core_review_2026-11/`, re-run 2026-11-12): a DC motor (R = 0.5 Ω,
L = 1 mH, K = 0.1 V·s/rad, J = 1e-5 kg·m²) driven
by a 10 V, 200 Hz sine from a consistent start, speed error against the
exact solution, for `make_dc_motor_observer` and for the same machine
built from one `add_ideal_transformer` and a capacitor of value J:

| dt | Observer | MNA-native composite |
|---|---|---|
| 100 µs | 13.6 % | 0.476 % |
| 50 µs | 6.56 % | 0.119 % |
| 25 µs | 3.22 % | 0.0298 % |
| 12.5 µs | 1.59 % | 7.45e-5 |
| 1 µs | 0.126 % | 4.7e-7 |

The observer's error halves with the step (first order); the
composite's quarters (second order). The composite also runs on
`trbdf2` unchanged (9.2e-5 in 1061 points). It needs no new kernel
code: the ideal transformer is the electromechanical coupling
(`v_emf = K·ω`, torque `K·i` into the speed node) and the inertia is a
capacitor on a node whose voltage is the speed.

The PMSM is already done this way: `PmsmMna`
(`core/include/pulsim/models/pmsm_mna.hpp`) puts the flux-linkage
equations in the matrix with the mechanics as nodes, and its header
records the observer's cost (speed drift −0.37 % at dt = 50 µs, −8 % at
500 µs; τ_d wrong by +100 % on a salient machine). What it lacks is a
non-sinusoidal PM flux (BLDC), saturation, a DC operating point, and
YAML.

Other tools (general knowledge, not checked against current
documentation): PLECS and PSIM solve their machine models together with
the circuit, offer saturation for the induction and synchronous
machines, and accept flux-linkage maps for PM machines.

## What Changes

- **`add_dc_machine(name, a_pos, a_neg, omega_node, *, R_a, L_a, K, J,
  B=0.0, T_load=0.0, theta_node=None, i0=None, omega0=0.0)`**: a linear
  composite — `R_a`, `L_a`, an ideal transformer of ratio `K` between
  the speed node and the armature, a capacitor `J` and a resistor `1/B`
  on the speed node, a current source for the load torque. No Newton,
  no new device kind.
- **`InductionMachineMna`** kernel device and
  `add_induction_machine(...)`: squirrel-cage machine in the stationary
  frame, three stator branch currents and two rotor-flux unknowns in
  the matrix, torque into the speed node. Parameters `R_s`, `L_ls`,
  `R_r`, `L_lr`, `L_m` [Ω, H, referred to the stator], `pole_pairs`,
  `J`, `B`.
- **`PmsmMna` gains a PM-flux harmonic series**
  `λ_pm,a(θ_e) = Σ_h ψ_h·cos(h·θ_e + φ_h)`; `add_bldc_mna(...)` fills
  it for a trapezoidal back-EMF.
- **`PmsmMna` gains flux maps** `λ_d(i_d, i_q)`, `λ_q(i_d, i_q)` for
  saturation and cross-coupling, represented through a co-energy spline
  so the map is reciprocal and the torque is consistent with it.
- **DC operating point with machines**: mechanical nodes are held at
  their initial conditions and the electrical rows are solved.
- **YAML**: types `dc_machine`, `induction_machine`, `pmsm`, `bldc`;
  the existing `induction_motor` type gains `model: mna | observer`.
- **Deprecation path** for the observers: `add_dc_motor`, `add_pmsm`,
  `add_bldc`, `add_induction_motor` and their `make_*_observer`
  functions emit a `DeprecationWarning` naming the replacement.
  **BREAKING** (staged): the YAML `induction_motor` default moves from
  `observer` to `mna` one release after the warning; results change by
  the observer's first-order error.

## Impact

- Affected specs: `motor-models`.
- Affected code: `builder/circuit_builder.hpp`,
  `models/{pmsm_mna,induction_machine_mna}.hpp`,
  `pwl/{pmsm_mna_history,nonlinear_refresh_pmsm_mna,device_pool}.hpp`,
  `solver/{run_transient,trbdf2_transient}.hpp`, `pwl/dc_assemble.hpp`,
  `yaml/loader.hpp`, `python/bindings.cpp`,
  `python/pulsim/{motors,motor_helpers,steady_state}.py`, stubs,
  `docs/motors-induction.md`.
- Dependencies:
  - `add-unified-device-interface` (Phase 3, the roadmap's
    prerequisite) — needed for the induction machine, the harmonic PM
    flux and the flux maps. Each would otherwise be another
    hand-derived trapezoidal and BDF2 history in the wrapper chain
    (about 60 edit sites for a stateful kind, `raw/models.md` §c).
  - The DC machine composite (step 1) needs nothing and can ship on
    today's structure.
  - `add-controlled-source-set` — the DC machine's optional angle node
    (`dθ/dt = ω`) uses a VCCS.
  - `add-engine-conformance-matrix` — rows for the new kinds.
  - `add-periodic-steady-state-shooting` benefits (machines become
    visible to steady-state solvers); not required.
- Relation to `add-induction-motor-squirrel-cage` (open, 20 of 23
  tasks done): that change delivered the observer-based machine and its
  helpers; this one supersedes its integration method and keeps its
  parameter helpers and tutorials.
