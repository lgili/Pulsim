## Phase 1 — Smart warm-start helper (~0.3 days)

- [x] 1.1 New header `pwl/initial_guess.hpp`.
- [x] 1.2 Function `make_diode_aware_initial_guess(graph,
      pool, b_extra) → Vector`.
- [x] 1.3 Iterates graph branches; for each
      `VoltageSource`, writes effective voltage onto
      `branch.from`.

## Phase 2 — PTC primitive (research-grade) (~0.3 days)

- [x] 2.1 New header `pwl/pseudo_transient.hpp`.
- [x] 2.2 `pseudo_transient_solve` with trust-region dt
      adaptation.
- [x] 2.3 Document the MNA limitation in the header.

## Phase 3 — Tests (~0.3 days)

- [x] 3.1 Unit: `make_diode_aware_initial_guess` writes
      source value onto from-node.
- [x] 3.2 Unit: `b_extra` modulation folds into the
      effective voltage.
- [x] 3.3 THE deferred test: κ=20 sinusoidal rectifier
      solves from auto warm-start + plain Newton + line
      search. > 95 % half-wave tracking, mean power within
      15 % of analytical.

## Phase 4 — Regression + docs (~0.2 days)

- [x] 4.1 All previous tests stay green (12 binaries, 3329
      assertions / 261 cases).
- [x] 4.2 `openspec validate pulsim-v2-pseudo-transient
      --strict` passes.
- [x] 4.3 `docs/pulsim-v2/layer4-v10-warm-start.md`.

## What was tried and what shipped

V10 was originally scoped to ship pure pseudo-transient
continuation as the robust fallback. Empirical exploration
during implementation found that PTC requires `J` to have
positive-real-part eigenvalues for stability. Pulsim's MNA
matrices have mixed-sign eigenvalues (constraint rows
contribute negative components). On those rows, PTC's
artificial dynamics is unstable and the iterate is
repelled from the solution.

The actual fix was discovered to be much simpler: a
STRUCTURAL warm-start helper that reads source values from
the device pool and places them onto source nodes. Combined
with plain Newton + line search, this solves the κ=20 stiff
rectifier (the V4 → V9 deferred test) at every time step.

PTC ships as research-grade for circuits with well-behaved
Jacobians; the smart warm-start is the recommended path for
canonical Pulsim circuits.
