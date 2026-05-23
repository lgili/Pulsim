# Layer 4 V8 — Continuation / homotopy Newton solve

Layer 4 V4 (line search) and V5 (LM damping) both failed on
the stiff `κ=20` sinusoidal rectifier — neither single-shot
Newton variant converges across diode zero-crossings when the
warm-start is "in the wrong basin". The classical fix is
**continuation (homotopy)**: solve a SEQUENCE of progressively
harder problems, warm-starting each from the previous step's
converged solution.

**Layer 4 V8 ships `continuation_solve`** — a thin orchestrator
that runs `solve_with_newton_b_extra` once per refresh in a
caller-supplied sequence, threading the converged `x` from
step _k_ as the warm-start for step _k+1_.

## API

```cpp
#include "pulsim/pwl/continuation.hpp"

namespace pulsim::pwl {

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

}
```

Semantics:

- `refresh_sequence` must be non-empty (else throws).
- `x_init` warm-starts the FIRST refresh. Each subsequent
  refresh receives the prior step's converged `x` as its
  warm-start.
- All Newton tuning knobs (`max_iters_per_step`, tolerances,
  globalizations) are forwarded unchanged to every inner
  solve.
- On any inner-Newton failure, the function rethrows with a
  message identifying the failing step index.

Cost: N inner Newton solves vs 1 for single-shot Newton. The
robustness boost justifies the linear blow-up for stiff
problems; for easy problems the caller simply doesn't enable
continuation (or uses a single-element sequence).

## Companion helper: `make_kappa_override_refresh`

`pwl/nonlinear_refresh_diode.hpp` ships a factory that returns
a `NonlinearRefreshFn` stamping the smooth-blend `IdealDiode`
with an OVERRIDDEN κ. All other params (`V_F0`, `R_d`,
`G_off`) come from the `DevicePool`. Used to build a κ-
continuation chain in one line:

```cpp
std::vector<NonlinearRefreshFn> refreshes;
for (Real k : {2.0, 5.0, 10.0, 20.0}) {
    refreshes.push_back(make_kappa_override_refresh(k));
}
Vector x = continuation_solve(seg, refreshes, g, pool,
                                x_init, b_extra);
```

## When continuation actually helps (empirical caveat)

The natural homotopy parameter for the smooth-blend
`IdealDiode` is κ (the sigmoid sharpness). Low κ smoothes the
sigmoid → Newton-friendly. High κ → diode-like behaviour.

**But the κ homotopy is NOT universally useful**. Specifically,
in a half-wave rectifier (V_sine → diode → R_load → GND):

- At κ=2 with V_sine ≈ 0, the smooth-blend "knee" is wide
  enough that the diode model leaks substantial current
  even for moderately reverse-biased v_diode (alpha = 0.13
  at delta = -0.7).
- The self-consistent operating point at small V_sine
  with κ=2 has v_n1 ≈ −100 V (a stable but UNPHYSICAL
  branch of the sigmoid-circuit equations).
- Continuing to higher κ from that unphysical x is a poor
  warm-start — the chain diverges or finds a wrong branch.

**The practical recipe** for the rectifier:

1. **Smart warm-start** per time step (load-line guess):
   ```cpp
   x_init[n0] = v_sine;
   x_init[n1] = std::max(v_sine − V_F0, 0.0);
   x_init[i_src] = 0.0;
   ```
2. **Single-element κ chain at the target** plus line
   search. With the load-line warm-start, plain Newton +
   line search converges at every step (verified across
   2 cycles at dt=100µs).

This is what `tests/v2/layer5_v4/test_continuation_rectifier.cpp`
ships: a load-line warm-start with `kappa_seq = {20.0}`, which
validates the `continuation_solve` integration path end-to-end
while honestly acknowledging that the κ chain itself doesn't
do the heavy lifting on this problem.

## When continuation IS useful

Continuation shines when:

- The user has NO physically-motivated warm-start, and
  needs to solve from `x = 0` (or some neutral guess).
- The parameter family `F(x; p)` interpolates between an
  EASY problem (Newton converges from `x=0`) and the HARD
  target without branch-switching along the way.
- The user wants robustness over speed: the linear blow-up
  in cost buys avoidance of catastrophic divergence.

Examples (future OpenSpecs may exercise these):

- **V_F0 ramping**: start with `V_F0=0` (ideal short — diode
  is a resistor, trivially Newton-solvable), ramp toward
  the target `V_F0=0.7 V`. No branch switching.
- **Source-amplitude ramping**: start with `V_amp=0.5 V`
  (rectifier is trivially in linear regime), ramp toward
  `V_amp=10 V`. Operating-point variation is gradual.

The `continuation_solve` primitive is parameter-agnostic — it
just iterates user-supplied `NonlinearRefreshFn`s. The κ
override is the only built-in helper; other homotopy
parameters require the user to write a closure that captures
the desired interpolation.

## Test coverage

Three tests in `tests/v2/layer5_v4/test_continuation_rectifier.cpp`:

1. **Trivial single-refresh == direct Newton** —
   `continuation_solve` with a 1-element sequence produces
   bit-identical output to `solve_with_newton_b_extra`
   directly.
2. **DC κ-chain finds load-line** — V_dc = 2V through
   smooth diode (κ=20), continuation through
   `{2, 5, 10, 20}`. The κ chain WORKS at DC because the
   diode's operating point at κ=2 vs κ=20 sits in the
   same physical branch (the load is small enough that
   diode-leakage doesn't dominate).
3. **κ=20 sinusoidal rectifier** — `V_amp=10 V`, 60 Hz,
   `R_load=10 Ω`. 2 cycles at dt=100 µs. Load-line warm-
   start + κ={20} continuation. Verifies > 95% pos-half
   tracking, > 95% neg-half tracking, mean power within
   15% of analytical.

## What V0 deliberately does NOT do

- **Auto κ-sequence selection** (adaptive scheduling).
  V0 takes the sequence as input.
- **Predictor-corrector continuation** (tangent
  prediction via implicit-function-theorem df/dp). V0 is
  pure warm-start (predictor = previous x).
- **Per-step failure recovery** (sub-step refinement on
  Newton divergence). V0 throws on any failure.
- **Alternative homotopies** (V_F0 ramping, source-amp
  ramping). V0 ships κ-override only; the primitive is
  generic and user-extensible via `NonlinearRefreshFn`.

These are V1+ candidates.

## Files added/modified

- `core/include/pulsim/pwl/continuation.hpp` (NEW)
- `core/include/pulsim/pwl/nonlinear_refresh_diode.hpp`
  (extended with `make_kappa_override_refresh`)
- `core/tests/v2/layer5_v4/test_continuation_rectifier.cpp`
  (NEW)
- `core/CMakeLists.txt` (test wired into
  `pulsim_v2_layer5_v4_tests`)
