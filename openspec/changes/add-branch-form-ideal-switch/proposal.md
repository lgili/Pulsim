## Why

> **Revised 2026-11-13.** The core roadmap now has two pillars: the
> ideal switch with complementarity mode selection, and exact
> propagation (`openspec/CORE_ROADMAP.md`, "The centre of the
> roadmap"). This change is the first pillar's device half; its
> decision half is `add-complementarity-mode-selection`. Superseded
> statements of the first version: the form stays opt-in (now: the
> default at the end of the migration); the restart after an inconsistent
> commutation is the backward-Euler pair (now: three consecutive
> Alexander SDIRK(2,2) steps); a floating node is always an error (now:
> an island bounded by OFF diodes is a gauge); an LCP solver is a
> Non-Goal (now: the mode is chosen by the complementarity service);
> the exact strategy with ideal switches is a later task (now: it is
> part of this change, task group 8).

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
  (`../add-exact-propagator-strategy/probes/phi_gamma_fixed_step.py`);
  the exact propagator's eigenvector condition number reaches 2.3e9 on
  the resistor-pair buck and its relative error 7.9e-9
  (`add-exact-propagator-strategy`), and the leak leaves 6.0e-9 A in an
  inductor that should be at rest.
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
- **The resistor pair is why diodes need a band.** The value law
  `g_on·(v − V_f)` decides a diode from a rounding-level voltage times
  1e3; with `V_th = 0` the kernel fails at rest and chatters in DCM
  (`../add-complementarity-mode-selection/probes/today_dsed.out`).
- The connectivity pre-check assumes the opposite of an ideal switch:
  "the assembler always stamps a conductance" (`pwl/preflight.hpp:45-48`).

PSIM and PLECS model the switch as ideal (general knowledge, not
measured here), so a user who ports a schematic meets a leak and an
on-resistance that the reference tool does not have.

Prototypes (numpy, scipy, mpmath):

| Experiment | Result |
|---|---|
| RL through a closed switch, branch form, `R_on = 0` (`probes/branch_form_rl_order.py`) | max error 2.19e-2, 5.85e-3, 1.51e-3 A at `h = τ/20, τ/40, τ/80`; condition number of the step matrix 95 against 3930 for `g_on = 1e3` |
| Inductor cut-set, switch opens on 9.93 A, `g_off = 0`, trapezoid only (`probes/sdirk_restart_ideal.out` (1)) | `i_L = 0` at once, node voltage ±397.307628 V for ever |
| Same, three SDIRK(2,2) steps after the event | one first-step sample of 959.018799 V, then 0; flux removed `L·Δi` = −9.93096483e-3 V·s equals the SDIRK quadrature of the node voltage over the step to all printed digits; record ½·L·i² = 49.3120 mJ |
| Capacitor loop, 1 µF at 10 V and 3 µF at 2 V closed by `R_on = 0`, trapezoid only ((2)) | 4.0 V (correct) but ±120 A for ever |
| Same, three SDIRK(2,2) steps | −289.7 A on the first step, then below 1e-14 A; 6.0 µC moved, 24.0 µJ lost, exact |
| Ideal DCM buck, fixed step `T/N`, 300 periods from the 40-digit orbit, modes by the LCP ((3)) | `v_C` error at N = 50/100/200/400: no restart 1.25e-3 … 1.57e-4 (order 1.00); BE pair 1.09e-3 … 1.71e-5 (order 2.00); SDIRK×3 −2.44e-6 … −4.35e-8 (order 1.89 → 1.98), about 400 times smaller than the BE pair; `i_L = 0` exactly in the idle interval |
| DCM buck with ideal switch and diode, 40-digit (`probes/ideal_dcm_buck.py`) | orbit `v_C = 8.9963970815477 V` (resistor pair: 8.99624708728943 V), diode off at `0.79983739563545·T` |
| Same circuit on the exact route with LCP mode selection, 4000 periods from rest (`../add-complementarity-mode-selection/probes/lcp_cases.out` (iii)) | `v_C` error +4.5e-12 V, turn-off +2.0e-12·T, 11 979 events, ≤ 2 pivots each |

The stamp is small. The consequence — states that become dependent at
a commutation — is the real content of this change.

## What Changes

- **Branch-form stamp.** A switch or PWL diode in branch form owns one
  current unknown `i_s` and one row:
  ON `v_a − v_b − R_on·i_s = V_f`, OFF `i_s − g_off·(v_a − v_b) = 0`.
  `R_on = 0` and `g_off = 0` are legal. The five structural entries are
  present in both states, so the sparsity pattern does not depend on
  the mask.
- **The ideal switch becomes the default** at the end of the migration,
  in the single number-changing release of the roadmap: `add_switch`,
  `add_diode`, the topology helpers and YAML build the branch form, ideal
  unless `R_on` or `g_off` is given. Before that it is opt-in
  (`add_ideal_switch`, `form='branch'`).
- **Resistor-pair behaviour remains available**, explicitly: `R_on` is
  a series resistance and `g_off` a parallel leak of the branch form;
  `add_switch(…, g_on, g_off)` maps to `R_on = 1/g_on` and agrees with
  today's conductance form within 1e-10 (task 3.4). What changes is
  that nobody gets a resistor pair without asking for one.
- **Validation** of the conductance form: `g_on` finite and positive,
  `g_off` finite and non-negative, by name, at build time.
- **Per-mode structural checks with named refusals**: a node with no
  path in a mode, bounded only by open gated switches, and a loop of
  closed ideal switches and voltage sources raise
  `pulsim.UnsupportedError(engine, feature, device, kind)` with
  `feature = 'floating_node'` or `'ideal_voltage_loop'`, the mode, the
  nodes and the switches, before factorization. An island bounded by
  OFF diodes is a gauge (`add-complementarity-mode-selection`), not an
  error.
- **Consistent re-initialisation, one service.** When a mode makes
  inductor currents or capacitor voltages dependent, the state is
  re-initialised by the flux- and charge-conserving projection, and
  the transferred flux or charge and the energy lost are recorded per
  event; `on_impulse = 'warn' | 'error' | 'allow'` decides what the run
  does. Per strategy: the exact propagator projects explicitly and
  continues on the reduced state; the implicit strategies keep the full
  MNA state and realise the projection with the restart rule, three
  consecutive Alexander SDIRK(2,2) steps, whose quadrature transfers
  the exact flux and charge.
- **Diode in branch form with complementarity**: the mode comes from
  `add-complementarity-mode-selection`; the crossing signals are the
  current unknown `i_s` (ON) and the gap `w = V_f + R_on·i_s − v` (OFF).
- **Losses**: conduction `R_on·i² + V_f·i` from the solved current;
  switching energies `E_on(i, v, T_j)` and `E_off(i, v, T_j)` read from
  the device's tables at each event with the commutated current and the
  blocked voltage; impulse energies reported separately.
- **Factor cache**: one symbolic analysis for all masks (the pattern is
  invariant), numeric factors with partial pivoting per mask, a
  rank-one row update for a single flip, a counted fallback to a full
  refactor when the pivot order changes.
- **BREAKING:** `add_switch(g_on=inf)` and non-finite or negative
  conductances are refused at build time.
- **BREAKING (at the default flip):** circuits built without explicit
  conductances become ideal; their numbers change by the removed
  resistor pair (0.1 % in the RL above, 1.5e-4 V in the DCM buck's
  `v_C`); circuits that relied on the leak for a path now get a named
  refusal.

## Impact

- Affected specs: `device-models`, `linear-solver`.
- Affected code: `stamping/stamp_switch.hpp` (a second stamper),
  `pwl/{assemble,dc_assemble,device_pool,diode_event_state,preflight,
  cache}.hpp`, `models/switched_diode.hpp`,
  `builder/circuit_builder.hpp`, `yaml/loader.hpp`,
  `solver/{run_transient,trbdf2_transient}.hpp` (restart, event
  record), `dsed/{exact_lti,event_projection,scheduler}.hpp` (reduced
  state per mode), `analysis/mna_sweep.hpp`, `python/bindings.cpp`,
  `python/pulsim/{_diode_state,losses,topology,__init__}.py`, docs,
  stubs.
- Depends on:
  - `add-engine-conformance-matrix` (done on `feat/core-integration`):
    the branch form is a row of the capability table.
  - `add-complementarity-mode-selection`: the diode decision.
  - `add-discontinuity-restart-rule` (on `feat/core-phase0`, being
    reworked to three SDIRK(2,2) steps) and
    `update-trbdf2-post-event-restart` (in progress): required for
    `g_off = 0` and for `R_on = 0` across capacitors on the implicit
    strategies.
  - `update-linear-time-assembly` (tier 1 on a side branch): shared
    symbolic analysis and single-flip updates.
  - `add-exact-propagator-strategy`: the reduced-state extraction the
    exact route uses (task group 8).
- Ships first (tasks 1–4): validation, the stamp and the structural
  checks with `R_on = 0` and `g_off > 0`, which creates no dependent
  states in circuits without capacitor loops.
- Then (tasks 5–8): `g_off = 0`, capacitor loops, the impulse record,
  the diode, losses, the exact route.
- Last (task 9): the default flip, with the other number-changing
  defaults, re-baselined through the reference harness.
- Cost: one more unknown per branch-form switch. By the audit's
  per-unknown rates (26–70 ns below 50 unknowns, `raw/perf.md` row 4)
  a 12-switch inverter step grows by roughly 0.4–0.8 µs on 0.64–0.72 µs
  on the implicit route — an estimate, gated by counters before the
  flip (task 7). On the exact route the ideal switch removes states
  (the idle DCM mode has one state instead of two) and the stiff
  eigenvalues, so the cost goes the other way.
