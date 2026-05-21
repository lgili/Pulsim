# Design — `pulsim-v2-pseudo-transient` (Layer 4 V10)

## Two deliverables

V10 ships TWO complementary tools:

1. **`make_diode_aware_initial_guess`** — the recommended
   path for canonical Pulsim circuits. A structural helper
   that walks the DevicePool and places source values onto
   the corresponding "from" nodes. Pairs with plain Newton +
   line search to solve the κ=20 stiff sinusoidal rectifier
   that was deferred from V4 → V9.

2. **`pseudo_transient_solve`** — research-grade
   pseudo-transient continuation primitive. Ships for
   circuits with well-behaved Jacobians where the artificial
   dynamics `dx/dt = -F(x)` is stable. NOT recommended for
   canonical Pulsim MNA circuits (see "PTC limitation"
   below).

## The smart-warm-start algorithm

```
function make_diode_aware_initial_guess(graph, pool, b_extra):
    x_init = zero(state_size)
    for branch in graph.branches:
        if pool.kind_of(branch.id) == VoltageSource:
            V_pool = pool.voltage_source_params(branch.id).V
            src_var = pool.branch_var_id_for_source(branch.id, graph)
            V_eff = V_pool - b_extra[src_var]
            if branch.from is not ground:
                x_init[branch.from] = V_eff
    return x_init
```

Two cases the helper handles:
- **Constant DC source**: `pool.V = 5.0`, `b_extra = 0`.
  Returns `x_init[source_from] = 5.0`.
- **Sinusoidal source via b_extra**: `pool.V = 0`,
  `b_extra[src_var] = -V_sine(t)`. Returns
  `x_init[source_from] = +V_sine(t)`.

For canonical source → diode → load circuits, setting
`v_from = V_source` and leaving `v_to = 0` means:
- Diode anode at `V_source`, cathode at 0 → `v_diode = V_source`.
- If `V_source > V_F0`, Newton pulls `v_to` up toward
  `V_source − V_F0` (forward conducting).
- If `V_source < V_F0`, Newton finds `v_to ≈ 0` (diode off).

Either way, the starting point is INSIDE the correct basin.

## Why PTC alone fails for Pulsim MNA

PTC converts `F(x) = 0` into `dx/dt = -F(x)`. The artificial
dynamics is stable near a solution `x*` if the eigenvalues of
`-J(x*)` have negative real parts — equivalently, `J(x*)`
has positive-real-part eigenvalues.

Pulsim's MNA Jacobian for a voltage-source-driven circuit
includes the constraint row:
```
J[constraint_row] = [..., 1 (on v_node), ..., 0 (on i_src), ...]
```
and the corresponding column has the source's KCL
contribution. The constraint coupling creates eigenvalue
PAIRS whose real parts can be EITHER sign depending on the
circuit topology.

For our linear test (V_dc → R → GND), `J = [[1, 1], [1, 0]]`
(MNA stamp). Eigenvalues = `(1 ± √5) / 2 ≈ 1.618` and
`-0.618`. The NEGATIVE eigenvalue means PTC's artificial
dynamics has a corresponding eigenvalue of `+0.618` — UNSTABLE.

The iterate is repelled from the solution along that
eigenvector, leading to divergence regardless of `dt`
schedule.

This was discovered empirically during V10 implementation.
Multiple PTC variants (SER, trust-region, exponential
growth, multi-iter implicit Euler) were tried; all failed
with the same root cause.

## What the PTC primitive IS good for

Pure resistive nonlinear circuits with no MNA constraint
rows (e.g., a current-source-driven nonlinear load): `J`
can be positive-definite and PTC works.

Pulsim's typical circuits all have voltage sources →
constraint rows → mixed-sign `J`. For those, use the smart
warm-start.

PTC ships for completeness and as a building block for
future work (e.g., damped PTC, sign-corrected PTC, or
hybrid methods that combine warm-start + PTC for the inner
iterations).

## API

```cpp
namespace pulsim::v2::pwl {

// PRIMARY: smart warm-start.
[[nodiscard]] inline Vector make_diode_aware_initial_guess(
    const topology::Graph& graph,
    const DevicePool& pool,
    const Vector& b_extra);

// RESEARCH-GRADE: pseudo-transient continuation.
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

}  // namespace pulsim::v2::pwl
```

## Test plan

Three tests in
`tests/v2/layer5_v4/test_pseudo_transient_rectifier.cpp`:

1. **Helper writes source value onto from-node** (DC source).
2. **Helper folds b_extra into the effective voltage**
   (sinusoidal source).
3. **κ=20 stiff sinusoidal rectifier solves from auto
   warm-start + plain Newton + line search** (THE deferred
   integration test from V4 → V9, finally cracked).

## Cost

The helper is O(num_branches) per call — same as one MNA
assemble pass. Negligible compared to a Newton solve.

PTC primitive has Newton-comparable per-iteration cost; the
limitation is convergence, not throughput.

## Files

- NEW `core/include/pulsim/v2/pwl/initial_guess.hpp` (~70 LOC)
- NEW `core/include/pulsim/v2/pwl/pseudo_transient.hpp` (~150 LOC)
- NEW `core/tests/v2/layer5_v4/test_pseudo_transient_rectifier.cpp` (~250 LOC)
- NEW `docs/pulsim-v2/layer4-v10-warm-start.md`
