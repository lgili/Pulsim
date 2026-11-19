Reference values (`probes/`; ideal LC: 100 µH, 100 nF, 10 V,
`T₀ = 2π·√(LC)`).

- Fixed step, error of the complex amplitude after 10 periods:
  5.16e-4 at `T₀/20`, 3.24e-5 at `T₀/40`, 2.02e-6 at `T₀/80`
  (ratios 16.0: order 4).
- Stability function: `R(−10) = 0.13657`, `R(−100) = 0.075733`,
  `R(−1e4) = 9.3136e-4`, `R(−∞) = 0`; embedded `R̂(−∞) = −3/20`.
- Adaptive, 1000 periods, prototype controller: 14.4 steps and 72
  solves per period at rtol 1e-5 (energy −0.50 %, max error 1.9 V);
  25.9 and 130 at 1e-6 (−0.028 %, 0.18 V); 83.7 and 418 at 1e-8
  (−8.2e-5 %, 1.7e-3 V). TR-BDF2 with the same controller: 336, 701 and
  3164 solves per period.

## 1. Tableau
- [x] 1.1 Coefficients as exact rationals in one header; compare with the SUNDIALS table `ARKODE_ARK436L2SA_DIRK_6_3_4` and with the 2003 paper (`solver/step/esdirk43.hpp`; SUNDIALS identical; the paper itself not consulted, DECISIONS 1)
- [x] 1.2 Test in rational arithmetic (compile-time where possible): row sums equal `c`; the eight order conditions of `b` up to order 4 are exactly 0; `b̂` satisfies orders 1–3; `A·c = c²/2` for stages 2–6; `b = a₆` (exact in `probes/` and the reviewer's script; the C++ test checks the doubles to 1e-14)
- [x] 1.3 Test: `R(−10)`, `R(−100)`, `R(−1e4)` equal the values above within 1e-9; `|R(iy)| ≤ 1 + 1e-12` on 4000 log-spaced `y ∈ [1e-3, 1e6]`

## 2. Linear kernel (no events, not exposed in `simulate()`)
- [x] 2.1 Stage loop on the `(G, C, b)` split with `solve_at(mask, h/2)`; five stage derivatives stored; stiffly accurate update (on the step skeleton with a test body, `tests/step/mna_stage_body.hpp`, dt-keyed factor; `solve_at` itself is the engine-switch step)
- [x] 2.2 Embedded error, stiff filter, RMS norm on capacitor voltages and inductor currents, I-controller with exponent −1/4
- [x] 2.3 Tests, fixed step: LC amplitude error after 10 periods 5.16e-4, 3.24e-5, 2.02e-6 at `T₀/20, /40, /80` within 5 %, observed order ≥ 3.9; RC charged in steps of `h = 100·τ`: remaining error 7.57 % of the step after one step, positive (no overshoot), below 0.05 % after three
- [x] 2.4 Tests, adaptive on the LC over 1000 periods: at rtol 1e-6 at most 32 steps per period, energy loss at most 0.035 %, max error at most 0.25 V; at rtol 1e-8 at most 100 steps per period and max error at most 2.5e-3 V (all met on 1000 periods: 27.5 steps, 0.027 %, 0.178 V; 84 steps, 1.68e-3 V; DECISIONS 7, 13; the ctest runs 20 periods: energy loss 5.4e-6; energy at equal solves against TR-BDF2 and trapezoidal, DECISIONS 13)
- [x] 2.5 Test: Prothero–Robinson `y' = λ(y − sin t) + cos t`, `λ = −1e6`: record the observed order (expected between 2 and 3) and assert it is at least 2
- [x] 2.6 Measurement gate (counters, LC, 1000 periods, equal rtol ∈ {1e-5, 1e-6, 1e-8}): linear solves per period at most one quarter of TR-BDF2's (prototype: 4.7, 5.4, 7.6 times fewer); factorizations per accepted step at most 1

## 3. Integrator in the unified stepper (needs `refactor-unified-transient-stepper`, `add-unified-device-interface`)
- [x] 3.1 `Esdirk43` behind the integrator interface: the engine's take / error / after_solve / commit phases instantiated per integrator (`step::run<Esdirk43>`), stages through the linear companion histories with an arbitrary ψ (DECISIONS 15). the stage form `(β·h, ψ)` for the nonlinear capacitor, the saturable inductor (atan, table) and external devices (DECISIONS 16). NOT done: Lauritzen diode, IGBT tail, machines, the Jiles-Atherton law (refused by name), carry_q, stage predictors
- [x] 3.2 Newton per stage with the existing acceptance rules (the engine's own Newton per stage solve); failure of a stage rejects the step (h/4). NOT done: one Jacobian per attempt with the refresh rule of DECISIONS 8
- [x] 3.3 `integrator='esdirk43'` in `simulate()` on `engine='trbdf2'` (default off); refused by name on pwl and on the exact engine, on the legacy loop, and per device for stateful Newton histories (`feature='integrator=esdirk43'`); the `integrator` row of `kRunOptionTable` honoured on trbdf2
- [ ] 3.4 (partly: the Shockley clamp against scipy Radau and an LC of order 4 through the engine, DECISIONS 15; the saturable-inductor LC against an ODE truth 4.8e-8 at rtol 1e-9 and order 4 at a fixed ceiling, a Level-1 MOSFET + Coss + saturable BOOST 6.3e-6 from a fine TR-BDF2, DECISIONS 16; not done: the MOSFET buck itself, `engine='auto'` not re-checked) Tests: Shockley rectifier and a Level-1 MOSFET buck agree with TR-BDF2 at rtol 1e-9 within 1e-5; a saturable-inductor LC agrees within 1e-5; `engine='auto'` results unchanged

## 4. Events (needs `update-trbdf2-post-event-restart`)
- [x] 4.1 First stage from the consistent post-event solve; never reused across an event (a step whose slots belong to another mode is the shared SDIRK(2,2) step, its estimate filtered; DECISIONS 15)
- [ ] 4.2 (partly: bracket on the stage values (t, t + h/2, t + h), Illinois on ESDIRK re-steps, ESDIRK landing re-step; no dense-output bracket yet) Hermite bracket, Illinois on the interpolant, landing re-step
- [ ] 4.3 Tests: CCM and DCM bucks of `numerics/p2_buck.py` within 1e-6 V and 1e-8 A of the 50-digit references of `add-exact-propagator-strategy` at rtol 1e-8; gate edges landed within 1e-12 s; the TR-BDF2 event tests pass on this integrator

## 5. Resonant converters with Newton devices
- [ ] 5.1 Reference set: LLC half-bridge with a nonlinear `Coss`, phase-shifted full bridge with a saturable inductor, series-resonant LC with a Shockley rectifier (from the harness)
- [ ] 5.2 Measurement gate (go / no-go for exposing the method in the docs as recommended): at equal measured error against a rtol 1e-10 reference, at least 2 times fewer linear solves than TR-BDF2 on at least two of the three circuits; report all three with Newton iterations per stage
- [ ] 5.3 Measure the proposed `auto` rule (share of accuracy-limited steps) on the full reference set and record it in design.md; no change to `auto` in this change

## 5b. Added by the track brief (2026-11-18)
- [x] 5b.1 Observed orders separately on differential, index-1 algebraic and index-2 unknowns against closed forms (DECISIONS 5)
- [x] 5b.2 Stiff ladder (spread 1e9): L-stable damping against trapezoidal ringing; work-precision tables against TR-BDF2 on the skeleton (DECISIONS 7)
- [x] 5b.3 Dense output of order 4 on q (stage-value form), algebraic unknowns by re-solve; its limits measured (DECISIONS 3)
- [x] 5b.4 Events located on re-steps seeded by the dense output, on the skeleton's cut path (DECISIONS 6)
- [x] 5b.5 Newton: one Jacobian per attempt, stage predictor, counts as bounds (DECISIONS 8)
- [x] 5b.6 Restart: no history; consistent algebraic unknowns after an event (DECISIONS 9)

- [x] 5b.7 Nonlinear q on the test body: saturable inductor and junction Coss, stages in q, x by Newton with dq/dx; orders 4 on states and on a nonlinear algebraic unknown; dense output combined in q (local order 5 vs 3 in x); flux balance and `carry_q`; the x formulation compared (DECISIONS 10, 11; reviewed)
- [x] 5b.8 Newton through a 1e3 slope collapse and edges landing in saturation: refresh on slow contraction, backtracking, counts against TR-BDF2 (DECISIONS 8, 12)
- [x] 5b.9 Event-dominated buck at 100 kHz against TR-BDF2; selection guidance in design.md D5b (DECISIONS 14)

## 6. Docs and verification
- [ ] 6.1 Integrator guide: when fourth order pays (accuracy-limited steps), when it does not (event-limited steps, PWL circuits that the exact strategy covers), order reduction on index-2 unknowns (the measured content is in design.md D5b; the user-facing page waits for `integrator='esdirk43'` in `simulate()`, task 3.3)
- [ ] 6.2 ruff, pytest, ctest (`-j2`), `openspec validate --all --strict`, stub check
