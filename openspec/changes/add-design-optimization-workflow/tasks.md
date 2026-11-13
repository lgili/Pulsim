## 1. Records and the evaluator (first slice)
- [ ] 1.1 `python/pulsim/design.py`: `EvaluationRecord`, `Evaluator(build, period, kpis, constraints, steady, engine, options)`, `evaluate`; design hash; provenance
- [ ] 1.2 Default KPIs on one steady period, time-weighted (`np.trapezoid`): efficiency, losses by device and category, ripple, stresses
- [ ] 1.3 Oracle: a buck with ideal switches and a 10 mΩ inductor resistance as the only loss, KPIs against a numpy reference (matrix exponential per mode on the periodic orbit, in the test): efficiency, loss and ripple within 1e-9 relative; the ripple also within 2 % of the small-ripple formula `ΔV = (1 − D)·V_out/(8·L·C·f²)` as a sanity check

## 2. Failures as records
- [ ] 2.1 Status mapping of design D3; never a default value; other exceptions re-raised
- [ ] 2.2 Penalty and constraint adaptors
- [ ] 2.3 Tests: a design that builds a Shockley rectifier with `engine='exact'` gives `refused` with the four fields; a design with a non-settling steady state gives `failed` with `'not_settled'`; a ripple bound exceeded gives `infeasible` with the excess in volts; `as_objectives` and `as_constraints` return `penalty` for `refused`/`failed` and finite values otherwise

## 3. Thermal fixed point
- [ ] 3.1 Loop of design D4 with damping, 0.1 K tolerance, 20-iteration cap; re-simulation only when the circuit depends on `T_j`
- [ ] 3.2 Tests: linear loss model `P = P₀·(1 + α·(T − 25))` with `R_th` — the fixed point equals `(T_amb + R_th·P₀·(1 − 25α)) / (1 − R_th·P₀·α)` within 0.01 K; `R_th·P₀·α = 1.2` gives `infeasible` with `thermal_runaway`; a circuit without temperature dependence runs one simulation (counter)

## 4. Steady state
- [ ] 4.1 `SteadySpec` `pss`, `transient`, `auto`; the route in the counters; no silent fallback after a failure
- [ ] 4.2 Tests: DCM buck with `method='pss'` reaches `v_C` within 1e-9 of 8.99624708728943 V; `method='transient'` reaches it within the settle tolerance; `auto` picks `pss` for it and `transient` for a circuit `pss` refuses

## 5. Batch, determinism, thread safety
- [ ] 5.1 `evaluate_many` on the sweep job runner, thread backend, order kept, stop rule for three identical refusals
- [ ] 5.2 Tests: 64 designs on 1 and on 8 workers give equal records (all fields except `wall_s`); two calls of `evaluate` on the same design give equal records; a batch with one refusing design completes with one `refused` record

## 6. Worked example
- [ ] 6.1 `examples/optimization/buck_pareto.py` and a notebook (design D8); optional extra `pulsim[opt]` for pymoo; numpy fallback
- [ ] 6.2 Test (CI, small): 32 Latin-hypercube designs run, status counts printed, the non-dominated set is non-empty and every member satisfies the constraints
- [ ] 6.3 Docs: "Using Pulsim inside an optimiser" (contract, statuses, determinism, why derivative-free, the finite-difference table, synthetic parts)

## 7. Integration with later changes
- [ ] 7.1 With `add-in-kernel-event-losses`: `ThermalSpec(mode='kernel')` default; the fixed point kept as `mode='fixed_point'`; test that both agree within 0.1 K on the example's best design
- [ ] 7.2 With `add-immutable-circuit-ir`: `Evaluator(circuit=…)` with `with_params`; counter test that a batch on one topology performs one symbolic analysis
- [ ] 7.3 With `add-parameter-sensitivities`: `gradients=True`; test against the sensitivities change's oracle

## 8. Verification
- [ ] 8.1 ruff, pytest, `openspec validate --all --strict`
