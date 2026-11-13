## 1. Characterisation (committed first)
- [ ] 1.1 `python/tests/test_fixed_step_resolution.py`: pin `p13.py` — RC at `dt = 10τ`: error 1.87 V, peak 11.11 V, zero warnings; strict xfail of "one `ResolutionWarning`"
- [ ] 1.2 Record the perf harness's fixed-step counters and wall time as the "before"

## 2. Kernel
- [ ] 2.1 `SimulationOptions::resolution_diagnostics` (default true); `SimulationResult::resolution` (`lte` per stored sample, `lte_max`, its branch and time, number of branches covered, ringing records)
- [ ] 2.2 Truncation estimate in the dynamic path of `run_transient`: two retained vectors of derivative slots, running peaks, the stencil validity rule (same stamped mask, no restart step, no source jump, not the first two steps); `store_every` keeps the maximum since the previous stored sample
- [ ] 2.3 Ringing detector over the state vector: previous difference, counters, the 1 % local-magnitude and 1e-9 peak tests, one record per entry per run of four or more
- [ ] 2.4 C++ tests: RC at `dt = 10τ`, `τ` and `τ/100` — `lte_max` within a factor 2 of the prototype (0.35, 2.7e-2, 3.3e-6); a run of at least 4 alternating steps at `10τ` (prototype 6), none at `τ`
- [ ] 2.5 Bit-identity: states equal with the option on and off, on the linear path and on the Newton path

## 3. Python
- [ ] 3.1 Binding of the result struct; `res.resolution` view with branch and node names; `None` for `engine='trbdf2'` and `'dsed'`
- [ ] 3.2 `pulsim.ResolutionWarning(UserWarning)`; one per `simulate` call when `lte_max > 1e-2` or a ringing record exists; message with the variable, the time, the numbers and the two actions
- [ ] 3.3 `simulate(resolution_diagnostics=...)` and the `SolverOptions` bundle; stubs; un-xfail 1.1

## 4. Acceptance
- [ ] 4.1 RC: warning at `dt = 10τ` and `dt = τ`; none at `τ/10` and `τ/100` (`τ/2` sits at 7.5e-3 in the prototype, too close to the threshold to pin)
- [ ] 4.2 Buck CCM at T/20, T/100, T/400: no warning, `lte_max` below 1e-3 (prototype ≤ 2.7e-4)
- [ ] 4.3 DCM buck at T/100 without the restart rule: a ringing record on node `sw` (prototype: a run of 49); with `discontinuity_restart=True`, when that change is present: none
- [ ] 4.4 Hard-switched 1 nF / 1 mΩ at dt = 1e-9: a ringing record on node `sw` with and without the restart rule
- [ ] 4.5 Run the 25 v2 examples and the reference harness circuits; attach the list of those that warn, with the variable named, to the pull request; adjust nothing without the owner's answer to Open Question 1
- [ ] 4.6 Measurement gate: fixed-step wall time and counters of the perf harness within 3 % of 1.2 with the option on; identical with it off

## 5. Documentation
- [ ] 5.1 `CHANGELOG.md`: the diagnostics, the warning category and the filter line
- [ ] 5.2 `AI_GUIDE.md` §6.3: read `res.resolution` first, then halve `dt`; a trap row for "a fixed-step run with no warning is resolved" (the estimate is not a bound and does not cover stateful devices); `docs/gotchas.md`, `docs/solvers.md`
- [ ] 5.3 Docstrings of `simulate` and of the result view

## 6. Verification
- [ ] 6.1 `ruff check`, `pytest` (throttled), `ctest`, stub check
- [ ] 6.2 `openspec validate add-fixed-step-resolution-diagnostics --strict` and `openspec validate --all --strict`
