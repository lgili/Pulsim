## Why

A diode commutates by itself. Given the gate states and the state of
the energy-storage elements at an event, the set of conducting diodes
is not a guess to be iterated: it is the solution of a linear
complementarity problem (LCP) on the diode ports, and for a passive
network that problem has a solution that a pivoting method finds in a
finite number of steps. The kernel does not pose it. Every engine
settles the diodes by re-solving the network and re-applying the value
law of `SwitchedDiode::decide` (ON stays ON while `g_on·(v − V_f) > 0`,
OFF turns ON at `v ≥ V_on`, `V_on = V_th` when `V_f = 0`, else
`max(V_th, V_f)`), round after round, with a
64-round cap, a 64-entry mask-cycle detector, a chatter guard that
aborts after 32 commutations in a zero-length interval, and the advice
to add a hysteresis band `V_th > V_f` when it fails (owner notes
*pwl-diode-vth-is-not-a-drop*, *newton-absolute-dx-is-load-bearing*:
the 10 ns PSFB failure was this chatter inside the Newton loop).

Measured on today's kernel (`probes/today_dsed.py`, build of
`feat/core-integration-wip` 8b26db32, 2026-11-13; resistor-pair diodes
`g_on = 1e3`, `g_off = 1e-9`):

| Circuit | `V_th` | `dsed` today |
|---|---|---|
| buck at rest, every switch open | 0 | "diode configuration did not settle after 64 rounds at t = 0.000000" |
| DCM buck, 4000 periods | 0 | "D1 is chattering at t = 0.000996 — it commutated 32 times inside 0.000000 s" |
| boost, R = 50 Ω (CCM) | 0 / 1e-3 / 1e-2 | did not settle at t = 0 / at t = 0.000497 / runs |
| boost, R = 500 Ω (DCM) | 0 / 1e-3 | did not settle at t = 0 / runs |
| single-phase bridge, C filter | 0 / 1e-6 | did not settle at t = 0 / runs (602 ms for 0.2 s) |
| three-phase bridge, C filter | 0 / 1e-6 | did not settle at t = 0 / runs |

Earlier, on the same structure: the DCM buck with `V_th = 1e-6` aborts
"is chattering" when L changes by a relative 1e-5, 1e-4 or 1e-2 (three
of six runs, `feat/core-p4-exact-propagator` e1232005); the
periodic-steady-state solver cannot settle a boost with `V_th ≤ 1e-3`
(integration review, 2026-11-13). The first slice of
`add-exact-propagator-strategy` (on `feat/core-p4-exact-propagator`,
not yet integrated) replaces the value law at a zero crossing by a
look-ahead on the exact OFF trajectory and makes the rest-state and
DCM cases run on `dsed`; it does not change the fixed or the TR-BDF2
engine, and its band is still a tolerance (`1e-12 × max node voltage`).

The two engines that do run these circuits disagree on what they
cannot know: with the diodes off, the DC side of a bridge has no
defined potential, and `v(p)` at 0.2 s reads 0.000000 V on `trbdf2`
against 47.585852 V on `pwl` and 47.585559 V on `dsed`
(`probes/today_dsed.out`). The leak `g_off` and each engine's
numerics decide it, not physics.

Prototype (`probes/lcp_core.py`, `probes/lcp_cases.py`, output in
`probes/lcp_cases.out`; numpy / scipy / mpmath): ideal switches and
ideal diodes (`R_on = 0`, `g_off = 0`), one LCP per event posed on the
backward-Euler companion network of a step `h_e = 1e-6·T` (the
rational complementarity problem of linear complementarity systems at
`σ = 1/h_e`), Lemke's method with the lexicographic ratio test, exact
propagation between events:

| Case | Events | Lemke pivots per event (max / mean) | Wrong mode, zero-time loops | Oracle |
|---|---|---|---|---|
| buck at rest, `V_f = 0` | 1 LCP | 2 / 2 | 0 | stays OFF, state 0 |
| DCM buck, 4000 periods from rest | 11 979 | 2 / 0.66 (0 on 8001) | 0 | `v_C` 4.5e-12 V and turn-off 2.0e-12·T from the 40-digit orbit |
| DCM buck, L·(1 ± δ), δ = 1e-5, 1e-4, 1e-2 | 6 runs | 2 | 0 | all complete |
| boost CCM, 20 000 periods | 40 179 | 2 / 0.01 | 0 | 2.5e-12 V, 6.2e-13 A from the mpmath orbit |
| boost DCM (500 Ω), 4000 periods | 11 958 | 2 / 0.66 | 0 | completes |
| single-phase bridge, C filter, `V_f = 0.7` / `V_f = 0` | 40 / 39 | 6 / 3.61, 6 / 3.90 | 0 | 39 event times within 1.1e-14 s of the closed form |
| three-phase bridge, `L_s`, 10 A load (overlap) | 36 | 4 / 2.8 | 0 | 36 commutation times within 4.0e-15 s of `1 − cos μ = 2ωL_sI_d/(√3·V_m)` |
| three-phase bridge, C filter, C from 0 V | 60 | 5 / 3.0 | 0 | 56 event times within 8.1e-15 s; inrush impulse 14.758 J recorded at t = 0 |

No band, no round limit, no chatter guard; a tie (`i = w = 0` within
the resolution) is decided by the same LCP at `100·h_e` and `1e4·h_e`
and otherwise taken OFF, and that happened only where both modes are
consistent (series-pair turn-offs, the rest state).

## What Changes

- **One mode-selection service** (`ModeSelector`), called by every
  strategy at every event (gate edge, located diode crossing, source
  breakpoint, `t = 0`, resume) and by the DC operating point: given the
  gate mask, the held state and the previous mode, it returns the
  diode mode as the solution of one LCP on the diode ports. It replaces
  the four diode decision loops (fixed-step event iteration, TR-BDF2
  settle, DSED settle, DC seeding) and their caps and guards.
- **The PWL diode as a complementarity element**: an ideal element in
  series with `V_f` and `R_on`, in parallel with `g_off`;
  `0 ≤ i ⟂ V_f + R_on·i − v ≥ 0` on the ideal element. `V_th > V_f` keeps
  its meaning as a turn-on threshold (a diode that is OFF stays OFF
  until `v ≥ V_th`) and loses its role as an anti-chatter band; `V_th <
  V_f` stops meaning anything (it is `V_f`).
- **The LCP is posed on the companion network** of the event step
  `h_e`: `M = R_on + Eᵀ·K⁻¹·E` (the Schur complement of the diode ports
  with the states held through their companion sources), written in
  the hybrid coordinates of the previous mode so that the cached factor
  of that mode supplies it, one column per Lemke pivot.
- **Solver**: Lemke with the lexicographic rule (finite for every LCP;
  for a passive network the matrix is positive semidefinite, so a
  secondary ray proves that no consistent mode exists). Exhaustive
  enumeration (k ≤ 12) is the test oracle and the fallback when the
  network is not passive.
- **Floating islands are a gauge**, not an error: the potential of a
  sub-network that only diodes and open switches connect to the rest is
  reported at the centre of the interval those diodes allow, and the
  event predicates use gauge-free combinations.
- **Impulses** are part of the answer: an LCP solution that requires a
  jump (an inductor current into blocking diodes, a capacitor closed on
  a voltage) is followed by the re-initialisation service of
  `add-branch-form-ideal-switch` and recorded.
- **Shadow mode first**: `mode_selection='shadow'` runs the LCP beside
  today's settle on the same states and reports disagreements in the
  result; then `'lcp'` opt-in; then the default, engine by engine.
- **BREAKING (at the default flip only):** `V_th < V_f` with `V_f = 0`
  (a negative threshold) is no longer a distinct law; the ON branch of
  the conductance-form diode gains `g_off·v` (1e-9 S × v at the
  default leak) so that the law is monotone; runs that aborted with
  "did not settle" or "is chattering" complete.

## Impact

- Affected specs: `mode-selection` (new), `device-models`.
- Affected code: new `core/include/pulsim/events/mode_selector.hpp`
  (LCP, Lemke, enumeration, port model); `pwl/diode_event_state.hpp`,
  `solver/run_transient.hpp`, `solver/trbdf2_transient.hpp`,
  `dsed/scheduler.hpp`, `pwl/dc_operating_point.hpp` (callers);
  `models/switched_diode.hpp` (complementarity data);
  `python/pulsim/__init__.py` (`mode_selection=` option, diagnostics);
  docs.
- Depends on: `add-engine-conformance-matrix` (done; refusals),
  `restore-reference-benchmark-harness` (differential runner for the
  shadow comparison), `update-linear-time-assembly` (shared symbolic
  analysis: the event-step factor per mode is one more numeric factor).
  Works with the conductance form today and with the branch form of
  `add-branch-form-ideal-switch`; does not need Phase 3 — it gives
  Phase 3's `ModeSet` (`add-unified-event-queue` D6) its decision rule.
- Every strategy uses it: `pwl`, `trbdf2`, `dsed` / the exact
  propagator, the DC operating point and `periodic_steady_state`
  (saltation at a diode event uses the mode the LCP returns).
- Cost per event: one solve with a cached factor when the previous
  mode is still consistent (zero pivots — 67 % of the DCM buck's events
  and 99.6 % of the CCM boost's), plus one solve and O(k) dense work
  per pivot otherwise; at most 6 pivots per event over the five
  families (15 when the tie re-solves are counted).
  Estimates for C++ are gated in tasks.md, not claimed.
- Competitor behaviour (general knowledge, not measured): PLECS and
  PSIM model ideal switches and diodes and resolve commutations without
  user-visible bands; the complementarity formulation for switched
  circuits is the standard one in the non-smooth-dynamics literature
  (linear complementarity systems; Moreau–Jean time stepping).
