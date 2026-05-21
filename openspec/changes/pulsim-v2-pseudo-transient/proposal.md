## Why

V4 (line search), V5 (LM), V8 (κ-homotopy), and V9 (V_F0-
homotopy) all failed to solve the κ=20 stiff sinusoidal
rectifier from `x = 0`. V10 was scoped to ship
pseudo-transient continuation (PTC) as the "robust fallback".

**Empirical finding during V10 implementation**: pure PTC is
fundamentally unsuitable for Pulsim MNA systems with voltage-
source constraints. The artificial dynamics `dx/dt = -F`
requires `J = ∂F/∂x` to have positive-real-part eigenvalues
for stability near the solution. MNA matrices have MIXED-
sign eigenvalues (the constraint rows contribute negative-
real-part components). On those rows, PTC's artificial
dynamics is UNSTABLE — the iterate is repelled from the
solution regardless of `dt` schedule.

**The actual fix that works**: a STRUCTURAL warm-start
helper, `make_diode_aware_initial_guess(graph, pool,
b_extra)`, that walks the `DevicePool`, reads voltage-source
effective voltages from `pool.V + b_extra` contribution,
and writes them onto the source's "from" node. For
canonical source → diode → load circuits, this puts Newton
inside the correct basin of attraction at every time step.

Combined with plain Newton + line search, this helper
cracks the κ=20 stiff sinusoidal rectifier deferred from
V4 → V9.

## What Changes

**Scope decision — Layer 4 V10** (smart warm-start +
research-grade PTC):

- New header `pwl/initial_guess.hpp`:
  ```cpp
  [[nodiscard]] Vector make_diode_aware_initial_guess(
      const Graph& graph,
      const DevicePool& pool,
      const Vector& b_extra);
  ```
  Walks the pool's voltage sources. For each, computes the
  effective voltage `V_eff = pool.V − b_extra[source_var]`
  and writes it onto the source's "from" node in the
  returned x_init.

- New header `pwl/pseudo_transient.hpp`:
  ```cpp
  [[nodiscard]] Vector pseudo_transient_solve(
      const PwlSegment& seg,
      const NonlinearRefreshFn& refresh,
      const topology::Graph& graph,
      const DevicePool& pool,
      const Vector& x_init,
      const Vector& b_extra,
      Real dt_init = 1.0,
      Real dt_max  = 1e10,
      Size max_iters = 500,
      Real tol_res = 1e-7);
  ```
  Implements PTC with trust-region-style dt adaptation.
  Ships as a RESEARCH-GRADE primitive for circuits with
  well-behaved Jacobians (no MNA constraint rows). NOT
  validated on canonical Pulsim circuits — see the header
  comment + docs for the limitation.

- **Tests** (3 cases):
  - `make_diode_aware_initial_guess` writes source values
    onto from-nodes correctly (with and without b_extra
    contribution).
  - κ=20 stiff sinusoidal rectifier solves at every time
    step using the auto warm-start + plain Newton + line
    search. > 95 % pos-half + neg-half tracking, mean
    power within 15 % of analytical.

## Impact

- **Affected specs**: ADDED requirement on
  `kernel-v2-solver` for `make_diode_aware_initial_guess`.
- **Affected code** (~200 LOC):
  - NEW `pwl/initial_guess.hpp`
  - NEW `pwl/pseudo_transient.hpp` (research-grade)
  - NEW `tests/v2/layer5_v4/test_pseudo_transient_rectifier.cpp`
- **Migration**: zero. All additive.
- **Risk**: low for the warm-start (purely structural; no
  numerical risk). Medium for PTC (algorithm is correct
  but its applicability to Pulsim MNA is limited;
  documented as such).
