## Why

Newton's convergence test is absolute (`tol_newton_dx` =
`tol_newton_res` = 1e-9, `solver/options.hpp:88-92`; tested at
`pwl/nonlinear_solve.hpp:394`). At a small step an inductor row is
scaled by `2L/h`, and one unit in the last place of `(2L/h)·i` is
larger than 1e-9. The iteration then walks on a lattice and never
passes the test, although the iterate is exact to working precision
(core review 2026-11, `raw/numerics.md` item 4; `p7b_roundoff.py`,
re-run on this branch):

- 10 V, 10 Ω, two series inductors and a Shockley diode at 0.9 A fail
  once `dt < 2·L·ε·|i|/tol`. With 100 µH: predicted floor 4e-11 s, runs
  at 3e-11 s, aborts at 1e-11, 1e-12 and 1e-13 s. With 10 mH: predicted
  4e-9 s, runs at 1e-8 s, aborts at 1e-9 … 1e-11 s. (At 1e-12 and
  1e-13 s the 10 mH case completes again, with the divider node
  jittering by 6e-5 V.)
- At the abort `‖dx‖` at the divider node is 2^-29, 2^-26, 2^-23 and
  the residual of the inductor row is 2^-28, 2^-25, 2^-22: exact
  powers of two, quantisation and not a spurious root.
- The default floor of the variable-step engine, `h_min = span·1e-11`
  (`solver/trbdf2_transient.hpp:400`), is below this bound for
  ordinary inductors.

The documented stall ("TR-BDF2 stalls at `h_min`",
`docs/known-limitations.md`; pinned by
`test_saturable_transformer.py::test_engines_agree_below_saturation`)
aborts with `‖dx‖` = 8.787e-08 and a residual of 7.629e-06 on a linear
inductor row. 7.629e-06 is 2^-17 exactly, one unit in the last place of
a term between 3.4e10 and 6.9e10. That is the same signature. **The
audit did not run that reproducer, so the link is an inference**, and
this proposal treats it as one.

The owner's notes record three attempts to make the step test
scale-aware (by the right-hand-side norm, by the solution norm, per
unknown) and why each was wrong; the last one let the DC cascade stop
at a root 10.67 V from the homotopy's answer. The same note names what
would be right: a controller that does not go to `h_min`, or "a
row-equilibrated Jacobian so the `2L/h` rows do not set the scale".
This change is the second of those, restricted to transient solves.

## What Changes

- **Transient-only roundoff acceptance.** A transient Newton solve
  that exhausts its iteration budget returns its last iterate instead
  of raising when `dx` has stopped contracting, `dx` is small against
  the solution, and the row-wise backward error
  `ω = max_i |f_i| / (|J|·|x| + |b|)_i` (Oettli–Prager) is at most
  `8·ε`: every equation is satisfied to the roundoff of its own terms.
  This is the residual test on the row-equilibrated system.
- **The DC cascade is not touched.** The acceptance is a parameter of
  `solve_with_newton_b_extra` that defaults to off; only the transient
  callers (`run_transient.hpp:989`, `:1008`,
  `trbdf2_transient.hpp:716`) pass it. The DC callers
  (`pwl/continuation.hpp:88`, `pwl/dc_strategy.hpp:200`,
  `pwl/dc_assemble.hpp:565`) and the absolute tolerances keep their
  values, bit for bit.
- **A step floor at the roundoff bound.** Below
  `h_noise = 2·ε·max_L(L·|i_L|) / (1e-6·max(1, ‖x‖∞))` the last bit of
  an inductor row is worth more than 1e-6 of the solution, and the
  acceptance above no longer applies. The variable-step engine does
  not commit a step of a Newton circuit below it (it accepts at the
  floor and counts it); the fixed engine's dt-retry does not halve
  below it and a user `dt` below it is refused by name.
- **Every acceptance is recorded**: count, first time, worst `‖dx‖`
  and the row it came from, in the result.
- The abort message stops calling the stall "a step-controller defect"
  only if task 5.1 shows the reproducer passes.

## Impact

- Affected specs: `transient-timestep`.
- Affected code: `pwl/nonlinear_solve.hpp` (acceptance, stagnation
  tracking, row scale), `solver/run_transient.hpp`,
  `solver/trbdf2_transient.hpp` (callers, floor),
  `solver/options.hpp`, `solver/result.hpp`, `python/bindings.cpp`,
  `python/pulsim/__init__.py` (warning, result view), stubs, docs,
  `CHANGELOG.md`.
- Depends on: nothing. `update-trbdf2-post-event-restart` removes the
  cause of most steps at the floor; the two are independent and
  complementary.
- Golden baselines: a run that converged before is unchanged bit for
  bit, because the new branch replaces only the point where the solve
  raised. Runs that aborted may now complete; each such run reports
  its roundoff acceptances. The five
  DC fall-through tests must stay unchanged.
- Not a tolerance change: `tol_newton_dx` and `tol_newton_res` keep
  their meaning and their defaults, and the guide's advice not to
  loosen them stands.
