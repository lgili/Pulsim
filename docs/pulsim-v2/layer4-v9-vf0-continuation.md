# Layer 4 V9 — V_F0 + combined κ/V_F0 override helpers

V8 shipped `make_kappa_override_refresh` (a factory for
`NonlinearRefreshFn` that overrides the smooth-blend
`IdealDiode`'s κ). V9 completes the override set:

| Factory | Overrides | Pool-sourced |
|---------|-----------|--------------|
| `make_kappa_override_refresh` (V8) | κ | V_F0, R_d, G_off |
| `make_vf0_override_refresh` (V9) | V_F0 | R_d, G_off, κ |
| `make_kappa_vf0_override_refresh` (V9) | κ, V_F0 | R_d, G_off |

All three return a `NonlinearRefreshFn` usable directly by
`solve_with_newton_b_extra` or `continuation_solve`.

## API

```cpp
#include "pulsim/v2/pwl/nonlinear_refresh_diode.hpp"

namespace pulsim::v2::pwl {

[[nodiscard]] inline NonlinearRefreshFn
make_vf0_override_refresh(Real V_F0_override);

[[nodiscard]] inline NonlinearRefreshFn
make_kappa_vf0_override_refresh(Real kappa_override,
                                  Real V_F0_override);

}
```

Both are `[[nodiscard]] inline` — pure factory functions that
return a lambda capturing the override parameter(s) by value.
They walk `BranchKind::Nonlinear` branches with
`StoredKind::NonlinearDiode`, construct a temporary
`IdealDiode::Params` with the override(s) applied (other
fields taken from `pool.nonlinear_diode_params(branch.id)`),
and stamp via the standard 2-terminal pattern from V3
(`evaluate_current_and_jacobian<IdealDiode>`).

## Use cases

### Parameter sweep

```cpp
for (Real vf0 : {0.3, 0.5, 0.7, 1.0}) {
    auto refresh = make_vf0_override_refresh(vf0);
    Vector x = solve_with_newton_b_extra(
        seg, refresh, g, pool,
        Vector::Zero(seg.state_size),
        Vector::Zero(seg.state_size));
    log("V_F0=" << vf0 << ": v_out=" << x[output_node]);
}
```

The pool's diode is constructed once; each iteration overrides
just V_F0 without touching the pool. The user can
parametrically sweep V_F0 (or κ, or both) and study the
converged operating point.

### Combined homotopy

```cpp
const std::vector<std::pair<Real, Real>> chain{
    {2.0,  -5.0},
    {3.0,  -1.0},
    {5.0,   0.0},
    {10.0,  0.7}    // target
};
std::vector<NonlinearRefreshFn> refreshes;
for (auto [k, vf0] : chain) {
    refreshes.push_back(
        make_kappa_vf0_override_refresh(k, vf0));
}
Vector x = continuation_solve(seg, refreshes, g, pool,
                                x_init, b_extra);
```

Combined chains ramp `κ` and `V_F0` simultaneously, which is
more robust than ramping either alone for problems where the
two parameters jointly modulate stiffness.

## Honest scope: V9 does NOT solve the κ=20 stiff rectifier from x=0

V8 motivated the κ-override helper with the κ=20 sinusoidal
rectifier. V8 ended up requiring a per-step load-line
warm-start to converge (documented in
`layer4-v8-continuation.md`).

V9 explored whether V_F0 (or combined κ+V_F0) homotopies could
solve the κ=20 case from `x = 0`. Empirical findings:

- **Pure V_F0 chain at κ=20** (`{−10, −5, −2, −0.5, 0, 0.3,
  0.5, 0.7}` and finer variants): fails. Newton in the inner
  solve overshoots through the sigmoid's narrow knee, and
  the resulting Jacobian becomes singular at one of the
  intermediate steps.
- **Combined κ+V_F0 chain with κ_target=20**: same failure
  mode. Once κ exceeds ~10, the sigmoid is too sharp for any
  reasonable warm-start chain.
- **LM (Levenberg-Marquardt) damping**: stalls at local
  minima of `‖f‖²` where the residual gradient is non-zero
  but no descent direction satisfies the LM step rule.

**Conclusion**: for κ ≥ ~8 sinusoidal rectifiers from `x = 0`,
no continuation chain (with the currently available overrides
plus plain Newton + line search / LM) converges at every time
step. The V8 load-line warm-start remains the recommended
path for those problems.

V9 ships the override factories as PARAMETER-SWEEP and
GENERIC-HOMOTOPY building blocks, validated on the DC load-
line from V3 (V_dc=2V, R_load=1kΩ, κ=20). The factories
themselves are correct; the limitation is upstream in
Newton's behaviour on steep sigmoid problems.

## Test coverage

Four tests in
`tests/v2/layer5_v4/test_vf0_continuation_rectifier.cpp`:

1. **V_F0 override actually overrides**. Residual at a fixed
   x differs from the pool-default refresh, confirming the
   sigmoid centre has shifted.
2. **Combined override differs from each single override**.
   Residuals at a fixed x for `make_kappa_override_refresh(5)`,
   `make_vf0_override_refresh(0)`, and the combined
   `make_kappa_vf0_override_refresh(5, 0)` are all distinct.
3. **Single-element V_F0 continuation == direct Newton**.
   `continuation_solve` with a 1-element sequence containing
   the override refresh produces bit-identical output to
   `solve_with_newton_b_extra` directly.
4. **V_F0 sweep on DC load-line**. For V_F0 ∈ {0.3, 0.5,
   0.7}, the converged v_n1 matches the analytical
   `V_dc − V_F0` within 50 mV.

## What V9 deliberately does NOT do

- **Solve the κ=20 stiff sinusoidal rectifier from x=0**
  (empirically not solvable with current overrides).
- **R_d / G_off overrides**. Could be added trivially; not
  needed for any motivating use case identified to date.
- **Auto-tune V_F0 sequences**.
- **Adaptive continuation step refinement** on inner-solver
  failure.

## Files added / modified

- MODIFIED `core/include/pulsim/v2/pwl/nonlinear_refresh_diode.hpp`
  (+ `make_vf0_override_refresh`, + `make_kappa_vf0_override_refresh`)
- NEW `core/tests/v2/layer5_v4/test_vf0_continuation_rectifier.cpp`
  (4 cases)
- NEW `openspec/changes/pulsim-v2-vf0-continuation/`
- MODIFIED `core/CMakeLists.txt` (test wired into
  `pulsim_v2_layer5_v4_tests`)
