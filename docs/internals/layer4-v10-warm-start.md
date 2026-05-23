# Layer 4 V10 — Smart warm-start + PTC (research)

V4 (line search), V5 (LM), V8 (κ-homotopy), and V9 (V_F0-
homotopy) all failed to solve the κ=20 stiff sinusoidal
rectifier from `x = 0`. V10 was scoped to ship
pseudo-transient continuation (PTC) as the robust fallback.

**Empirical finding during V10**: pure PTC is fundamentally
unsuitable for Pulsim MNA systems with voltage-source
constraints. The actual fix is much simpler — a STRUCTURAL
warm-start helper that places source values onto the
corresponding source nodes. Combined with plain Newton +
line search, this finally cracks the κ=20 stiff rectifier
that's been deferred for 6 successive OpenSpecs.

## API

### Primary: smart warm-start

```cpp
#include "pulsim/pwl/initial_guess.hpp"

[[nodiscard]] Vector make_diode_aware_initial_guess(
    const Graph& graph,
    const DevicePool& pool,
    const Vector& b_extra);
```

Walks the graph's branches. For each branch stored in the
DevicePool as `VoltageSource`, computes the effective
voltage `V_eff = pool.V − b_extra[source_branch_var]` and
writes it onto the source's "from" node in the returned
x_init.

For canonical source → diode → load circuits, this places
Newton inside the correct basin of attraction:
- Diode anode at `V_source`, cathode at 0 → `v_diode = V_source`.
- If `V_source > V_F0`, Newton pulls the cathode up toward
  `V_source − V_F0` (forward conducting).
- If `V_source < V_F0`, Newton finds `v_cathode ≈ 0`
  (diode off).

### Research-grade: pseudo-transient continuation

```cpp
#include "pulsim/pwl/pseudo_transient.hpp"

[[nodiscard]] Vector pseudo_transient_solve(
    const PwlSegment& seg,
    const NonlinearRefreshFn& refresh,
    const Graph& graph,
    const DevicePool& pool,
    const Vector& x_init,
    const Vector& b_extra,
    Real dt_init = 1.0,
    Real dt_max  = 1e10,
    Size max_iters = 500,
    Real tol_res = 1e-7);
```

Converts `F(x) = 0` into `dx/dt = -F(x)`, integrates with
implicit Euler + trust-region dt adaptation. Ships as
research-grade for circuits with positive-real-part `J`
eigenvalues. NOT recommended for canonical Pulsim MNA
circuits (see "PTC limitation" below).

## Recipe: cracking the κ=20 stiff sinusoidal rectifier from x=0

```cpp
for (Size k = 0; k < n_steps; ++k) {
    const Real t = k * dt_sim;
    const Vector b_extra = make_sinusoidal_b_extra(t);

    // STEP 1: smart warm-start.
    const Vector x_init =
        make_diode_aware_initial_guess(graph, pool, b_extra);

    // STEP 2: plain Newton + line search.
    Vector x = solve_with_newton_b_extra(
        seg, &refresh_smooth_diodes, graph, pool,
        x_init, b_extra,
        /*max_iters=*/100,
        /*tol_dx=*/1e-7, /*tol_res=*/1e-5,
        /*enable_line_search=*/true);

    // Use x ...
}
```

Verified across 2 cycles at dt=100µs (334 steps): every step
converges, > 95 % pos-half + neg-half tracking, mean power
within 15 % of analytical `(V_amp − V_F0)² / (4·R_load)`.

## PTC limitation (why it doesn't work for MNA)

PTC requires the artificial dynamics `dx/dt = -F(x)` to be
STABLE near the solution. This holds when `J = ∂F/∂x` has
positive-real-part eigenvalues.

Pulsim's MNA Jacobian for a voltage-source-driven circuit
has a constraint row like:
```
J[constraint_row] = [..., 1 (on v_node), ..., 0 (on i_src), ...]
```
The constraint coupling creates eigenvalue pairs whose real
parts can be EITHER sign depending on topology.

Concrete example — linear V_dc → R → GND:
- State: [v_n0, i_src]
- `J = [[1, 1], [1, 0]]` (MNA stamp)
- Eigenvalues = `(1 ± √5) / 2` ≈ 1.618 and **−0.618**

The negative eigenvalue means PTC's `-J` has eigenvalue
`+0.618` — UNSTABLE. The iterate is repelled from the
solution along that eigenvector. NO dt schedule recovers.

Multiple PTC variants were attempted (SER, trust-region,
exponential growth, rollback-on-rebound) — all failed.
Documented as a research note in
`pwl/pseudo_transient.hpp`.

## Test coverage

Three tests in
`tests/v2/layer5_v4/test_pseudo_transient_rectifier.cpp`:

1. **Helper writes source value onto from-node** —
   `pool.V = 3.5`, `b_extra = 0` → `x_init[n0] = 3.5`.
2. **Helper folds b_extra into the effective voltage** —
   `pool.V = 0`, `b_extra[src_var] = −7.5` → `x_init[n0] = +7.5`.
3. **κ=20 stiff sinusoidal rectifier from auto warm-start +
   plain Newton + line search** — the deferred test from
   V4 → V9, finally cracked.

## What V10 deliberately does NOT do

- **Diode load-line correction beyond v_to = 0**. The
  V0 helper sets only source nodes; diode cathodes are left
  at 0 (a "diode-off" assumption that works for the canonical
  rectifier topology). Future work could compute a proper
  load-line correction for multi-diode networks.
- **Cap/inductor warm-start**. Trap-companion history is
  handled separately by `HistoryState`; the warm-start
  helper doesn't touch those branches.
- **Make PTC work for MNA**. The PTC primitive ships as
  research-grade; a future OpenSpec could explore
  sign-corrected PTC, hybrid PTC+warm-start, or
  affine-invariant Newton as alternatives.

## Files

- NEW `core/include/pulsim/pwl/initial_guess.hpp`
- NEW `core/include/pulsim/pwl/pseudo_transient.hpp`
  (research-grade)
- NEW `core/tests/v2/layer5_v4/test_pseudo_transient_rectifier.cpp`
- NEW `openspec/changes/pulsim-v2-pseudo-transient/`
- MODIFIED `core/CMakeLists.txt` (test wired)
