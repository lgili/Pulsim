## Why

A switch is two conductances, `g_on` when closed and `g_off` when open,
stamped in nodal form (`core/include/pulsim/stamping/stamp_switch.hpp:41-77`).
The usual pair `1e3 / 1e-9 S` is a ratio of 1e12, and it puts into every
circuit time constants that have nothing to do with it (paths relative
to `core/include/pulsim/` unless they start with `python/`):

- **Picosecond modes.** An open switch in series with an inductor gives
  `τ = L·g_off`; a closed one across a capacitor gives `τ = R_on·C`.
  The idle mode of the audit's DCM buck has eigenvalues −5.0e12 and
  −425.5 s⁻¹ (`probes/branch_form.py`); with `g_off = 1e-12` the
  first becomes −5.0e15. Audit `raw/numerics.md` item 13: at
  `h = 1e-13 s` with 10 mH a node voltage jitters by 6e-5 V; item 2:
  TR-BDF2 commits steps of 8.26e-13 s for a whole run and takes twelve
  consecutive steps of 4e-14 … 1e-12 s in DCM.
- **The modes cost accuracy elsewhere.** A Padé matrix exponential
  loses 8.2e-11 on the slow mode of that idle matrix
  (`../add-exact-propagator-strategy/probes/phi_gamma_fixed_step.py`),
  and the leak leaves 6.0e-9 A in an inductor that should be at rest.
- **The ideal values are not available.** `add_switch` stores whatever
  it is given (`pwl/device_pool.hpp:102-104`,
  `builder/circuit_builder.hpp:1129-1139`; no validation). Measured on
  an RL (10 V, 1 Ω, 1 mH), `probes/current_kernel_goff0.py`:
  - `g_on = inf`: the run aborts, "factorize failed (numerically
    singular)".
  - `g_off = 0`, switch opened at 5 ms with 9.93 A in the inductor:
    `pwl` reports a node voltage alternating between ±397 V for the rest
    of the run, `trbdf2` reports 7.0e10 V. Neither warns.
  - `g_on = 1e3`: the current settles at 9.98955 A, 0.1 % below the
    ideal 9.99955 A — the 1 mΩ the model adds.
- The connectivity pre-check assumes the opposite of an ideal switch:
  "the assembler always stamps a conductance" (`pwl/preflight.hpp:45-48`).

PSIM and PLECS model the switch as ideal (general knowledge, not
measured here), so a user who ports a schematic meets a leak and an
on-resistance that the reference tool does not have.

Prototype (`probes/branch_form.py`, `branch_form_rl_order.py`,
`ideal_dcm_buck.py`; numpy and mpmath):

| Experiment | Result |
|---|---|
| RL through a closed switch, branch form, `R_on = 0`, trapezoidal with a backward-Euler start | max error 2.19e-2, 5.85e-3, 1.51e-3 A at `h = τ/20, τ/40, τ/80`; condition number of the step matrix 95 against 3930 for `g_on = 1e3` |
| Inductor cut-set (the switch opens on 9.93 A), `g_off = 0`, trapezoidal only | `i_L = 0` at once, node voltage ±397.3 V for ever, first-step volt-seconds twice `L·i` |
| Same, two backward-Euler steps after the event | one pulse of −198.65 V for one step, `h·v = −9.9325e-3 V·s = −L·i(t⁻)` exactly, then 0 |
| Capacitor loop (1 µF at 10 V paralleled with 3 µF at 2 V through `R_on = 0`), trapezoidal only | 4.0 V (correct) but ±120 A for ever and 12 µC in the first step |
| Same, two backward-Euler steps | 60 A for one step = 6.0 µC exactly, then 0 |
| DCM buck with ideal switch and diode, 40-digit | orbit `v_C = 8.9963970815477 V` (resistor pair: 8.99624708728943 V), diode off at `0.79983739563545·T`, idle `i_L = 0`, one state |

The stamp is small. The consequence — states that become dependent at
a commutation — is the real content of this change.

## What Changes

- **Branch-form stamp.** A switch or PWL diode in branch form owns one
  current unknown `i_s` and one row:
  ON `v_a − v_b − R_on·i_s = V_f`, OFF `i_s − g_off·(v_a − v_b) = 0`.
  `R_on = 0` and `g_off = 0` are legal. The five structural entries are
  present in both states, so the sparsity pattern does not depend on
  the mask.
- **Opt-in**: `add_ideal_switch(…, R_on=0.0, g_off=0.0)`,
  `add_diode(…, form='branch', R_on=0.0)`, YAML `form: branch`. The
  conductance form stays the default and is bit-identical.
- **Validation** of the conductance form: `g_on` finite and positive,
  `g_off` finite and non-negative, by name, at build time.
- **Floating-node detection per mask** when a branch-form switch with
  `g_off = 0` is open: a node with no path to ground through resistors,
  inductors, sources, closed switches or capacitors is reported with
  the mask and the node names before factorization.
- **Consistency at commutation.** When a mask makes inductor currents
  or capacitor voltages dependent (an inductor cut-set, a capacitor
  loop), the state is re-initialised by the flux- and charge-conserving
  projection, realised on the implicit engines by the backward-Euler
  restart of `add-discontinuity-restart-rule`. The transferred flux or
  charge and the energy lost are recorded per event; a policy
  (`on_impulse = 'warn' | 'error' | 'allow'`) decides what the run does.
- **Diode law in branch form**: the ON test reads the current unknown
  (`i_s > 0`) instead of `g_on·(v − V_f)`; the OFF test is unchanged.
- **Outputs**: `res.i("S1")` is the unknown itself; conduction loss is
  `R_on·i² + V_f·i`; the impulse energy is reported as an event energy.
- **Factor cache**: one symbolic analysis for all masks (the pattern is
  invariant), numeric factors with partial pivoting per mask, and a
  rank-one row update for a single flip, with a counted fallback to a
  full refactor when the pivot order changes.
- **BREAKING:** `add_switch(g_on=inf)` and non-finite or negative
  conductances, which abort or misbehave at run time today, are refused
  at build time.
- **BREAKING:** a circuit that passes `g_off = 0` in conductance form
  and opens a switch onto a floating node now gets the floating-node
  error at the first visit of that mask instead of a singular-matrix
  abort.

## Impact

- Affected specs: `device-models`, `linear-solver`.
- Affected code: `stamping/stamp_switch.hpp` (a second stamper),
  `pwl/{assemble,dc_assemble,device_pool,diode_event_state,preflight,
  cache}.hpp`, `models/switched_diode.hpp` (current-based ON signal),
  `builder/circuit_builder.hpp`, `yaml/loader.hpp`,
  `solver/{run_transient,trbdf2_transient}.hpp` (event record),
  `analysis/mna_sweep.hpp`, `python/bindings.cpp`,
  `python/pulsim/{_diode_state,losses,topology,__init__}.py`, docs,
  stubs.
- Depends on:
  - `add-engine-conformance-matrix` (Phase 0): the branch form is a new
    solve class in the capability table; engines that cannot take it
    refuse by name.
  - `add-discontinuity-restart-rule` (Phase 2): required for
    `g_off = 0` and for `R_on = 0` across capacitors. The roadmap row
    lists only the conformance matrix; the prototype shows that without
    the restart the trapezoidal rule rings for ever on the algebraic
    unknown (±397 V, ±120 A).
  - `update-trbdf2-post-event-restart` (Phase 2): the same for TR-BDF2.
  - `update-linear-time-assembly` (Phase 1): shared symbolic analysis
    and single-flip updates, which this change must not break.
  - `add-unified-device-interface` (Phase 3) is not required: the
    switch stays a stamp, as today.
- Ships first (tasks 1–4, after Phase 0): the stamp, validation and
  the floating-node check with `R_on = 0` and `g_off > 0`, which
  creates no dependent states in circuits without capacitor loops.
- Must wait for Phase 2 (tasks 5–7): `g_off = 0`, capacitor loops, the
  impulse record.
- Later (task 8): the exact strategy with ideal switches, where the
  reduced state changes dimension with the mask.
- Cost: one more unknown per branch-form switch. By the audit's
  per-unknown rates (26–70 ns below 50 unknowns, `raw/perf.md` row 4)
  a 12-switch inverter step grows by roughly 0.4–0.8 µs on 0.64–0.72 µs
  — an estimate, gated by counters in tasks.md. This is why the form is
  opt-in.
