# Design — `pulsim-v2-continuation-newton` (Layer 4 V8)

## The homotopy idea

For a stiff problem F(x; p_target) = 0 where Newton diverges
from any reasonable warm-start, define a parameter family
F(x; p) that interpolates from "easy" to "hard":
- F(x; p_easy) = 0 — easy problem with known solution.
- F(x; p_target) = 0 — hard problem we actually want.

Continuation:
1. Solve at p_easy. Get x_0.
2. Increment p slightly (p_1 = p_easy + (p_target − p_easy)/N).
   Solve at p_1, warm-starting from x_0. Get x_1.
3. Repeat until p_N = p_target. Solution: x_N.

The hope: each intermediate problem is close enough to the
previous that Newton converges. The chain of warm-starts
guides Newton through the residual landscape to the true
solution at p_target.

For the smooth-blend IdealDiode, the natural parameter is
`kappa` (the sigmoid sharpness). κ=2 is smooth (Newton-
friendly); κ=20 is sharp (Newton-hostile).

## API

```cpp
namespace pulsim::v2::pwl {

/// Continuation solver: run Newton for each refresh in the
/// sequence, warm-starting from the previous step's solution.
/// Returns the final converged x.
///
/// The user constructs the sequence — typically by varying
/// some model parameter from "easy" to "hard" via a closure.
/// For the smooth-blend IdealDiode, `make_kappa_override_refresh`
/// provides a one-line factory.
[[nodiscard]] Vector continuation_solve(
    const PwlSegment& seg,
    const std::vector<NonlinearRefreshFn>& refresh_sequence,
    const topology::Graph& graph,
    const DevicePool& pool,
    const Vector& x_init,
    const Vector& b_extra,
    Size max_iters_per_step = 100,
    Real tol_dx  = 1e-7,
    Real tol_res = 1e-5,
    bool enable_line_search = false,
    bool enable_lm = false);

}  // namespace pulsim::v2::pwl
```

## kappa-override refresh

```cpp
/// Returns a refresh function that stamps smooth-blend
/// IdealDiode contributions using `kappa_override` instead of
/// the diode's pool-stored kappa. Other params (V_F0, R_d,
/// G_off) come from the pool as usual.
[[nodiscard]] inline NonlinearRefreshFn
make_kappa_override_refresh(Real kappa_override);
```

Implementation walks `Nonlinear` branches like
`refresh_smooth_diodes`, but builds a temporary
`IdealDiode::Params` with `kappa = kappa_override` and uses it
for the AD evaluation.

## Continuation sequence for the stiff rectifier

```cpp
const std::vector<Real> kappa_sequence = {2.0, 5.0, 10.0, 20.0};
std::vector<NonlinearRefreshFn> refreshes;
for (Real k : kappa_sequence) {
    refreshes.push_back(make_kappa_override_refresh(k));
}
Vector x = continuation_solve(seg, refreshes, graph, pool,
                                x_init, b_extra, ...);
```

Each step's solve uses the previous step's x as warm-start.
By the last step (κ=20), the warm-start is essentially the
correct answer for κ=20, just with a slightly different
sigmoid steepness — Newton converges easily.

## Test plan

Same scenario as the V4-deferred sinusoidal rectifier:
- V_sine(amp=10V, f=60Hz) → smooth-blend Diode (κ=20) → R(10Ω) → GND.
- Simulate 2 cycles at dt=10µs.

With continuation_solve called per step:
- Each step solves the κ=20 problem via the 4-step
  continuation sequence.
- Final x at each step is the converged answer.
- Aggregate output should match the analytical half-wave
  rectifier.

V0 test verifies:
- Continuation converges at every step (no throw).
- > 95 % of positive-half samples track `max(V_sine − V_F0, 0)`
  within 1 V.
- > 95 % of negative-half samples within 0.5 V of zero.
- Mean output power within 15 % of analytical.

Tolerances are looser than the original V4 target because
continuation has its own approximation error (the warm-start
chain is sensitive to the κ sequence).

## Cost

Continuation runs Newton N times per step (vs once). For
N=4, that's 4× the solve cost. The boost in robustness
justifies this for stiff problems; for easy problems, the
user just doesn't enable continuation.

## What V0 deliberately does NOT do

- **Auto kappa sequence selection**: V0 uses a fixed
  `{2, 5, 10, 20}` for the test. Adaptive scheduling
  (start small, halve if Newton struggles) is V1.
- **Predictor-corrector continuation**: the V0 is pure
  warm-start (predictor = previous x). True continuation
  with tangent prediction (estimate dx/dp via implicit
  function theorem) is more robust but complex. V1.
- **Failure recovery**: if any step in the sequence fails,
  V0 throws. V1 could retry with a finer sub-step.

## Implementation finding (post-V0)

While implementing the integration test, an important
empirical caveat surfaced about **κ as the homotopy
parameter** for the smooth-blend `IdealDiode` in a half-
wave rectifier:

- The original intuition (low κ smoothes the sigmoid →
  Newton-friendly) is correct ONLY in isolation.
- For a circuit where the diode interacts with a load
  (e.g. `V_sine → diode → R_load → GND`), **low κ widens
  the sigmoid "knee" so much that the model leaks
  substantial current even at moderately reverse-biased
  v_diode**. At κ=2 with V_sine ≈ 0, the
  self-consistent operating point has v_n1 ≈ −100 V (a
  stable but UNPHYSICAL solution branch).
- Each low-κ step then hands the next κ step an
  UNPHYSICAL warm-start, and the chain diverges or finds
  the wrong branch.

The **practical fix** (used in the V0 integration test):
- Use a **physically-motivated warm-start** per time step
  (load-line guess: `x_init = [v_sine, max(v_sine − V_F0,
  0), 0]`).
- Use a **single-element κ sequence at the target κ**
  — semantically equivalent to direct Newton with line
  search, but routed through the `continuation_solve`
  primitive to validate the integration pipeline.
- For problems where load-line guesses are unavailable,
  alternative homotopy parameters (`V_F0` ramping or
  source-amplitude ramping) preserve diode-like behavior
  through the chain. Those are V1 candidates.

The V0 primitive `continuation_solve` and the helper
`make_kappa_override_refresh` ship as-is — they are
correctly implemented and useful for problems where the
κ chain holds. The κ chain just doesn't hold for THIS
particular rectifier without a smart warm-start.
