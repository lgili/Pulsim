# Automatic Differentiation (AD) for Nonlinear Device Stamps

> Status: opt-in (build flag). Default builds use the legacy hand-coded
> stamps. Flip to AD with one CMake option to validate or stress-test the
> derived path.

Pulsim's nonlinear device library — `IdealDiode`, `MOSFET`, `IGBT`,
`VoltageControlledSwitch` — historically carried two parallel pieces of
code per device: a **residual** function describing what the device does,
and a **manually derived Jacobian** stamping the ∂residual/∂x partials
into the MNA matrix. The two have to stay in lock-step; a sign mistake in
the manual derivation manifests as silent convergence-rate degradation
that is hard to attribute back to its source.

The `add-automatic-differentiation` change introduces a parallel AD path
that derives the Jacobian by passing the residual through Eigen's
`AutoDiffScalar`. Both paths are present in the source tree; a build flag
selects which one is wired into the simulator's hot path. The two paths
agree on every J entry to within 1e-12 absolute on every operating point
the cross-validation suite exercises.

## Why two paths?

| Path | Strength | Weakness |
|------|----------|----------|
| Manual stamp | Fast (≈220 ns / MOSFET stamp on AppleClang 17 / Release+LTO). | Needs a re-derivation every time the residual changes; subtle sign bugs hide for years. |
| AD stamp | One residual = one source of truth. New devices are easier to author and audit. | Currently ≈3.8× slower per stamp because Eigen's `AutoDiffScalar` heap-allocates its derivative vector per call. Optimization is logged as Phase 6.2 of `add-automatic-differentiation`. |

## How to enable AD

```bash
cmake -S . -B build_ad -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPULSIM_USE_AD_STAMP=ON
cmake --build build_ad
```

The flag emits `PULSIM_USE_AD_STAMP=1` as a compile definition. Each of the
four nonlinear devices' `stamp_jacobian_impl` selects between manual and AD
via `#ifdef`, so the choice is fixed at build time — there is no runtime
toggle. Device-level `SwitchingMode::Ideal` PWL stamps remain on the
constant-conductance fast path regardless of the flag (AD adds no value
there because the Jacobian is structurally fixed per topology).

## When to flip the flag

- **You authored a new nonlinear device** and want a correctness signal
  before signing off on the manual stamp.
- **Convergence regressed** on a benchmark and you suspect a manual-stamp
  drift. Build with `PULSIM_USE_AD_STAMP=ON` and re-run; if convergence
  recovers, the manual path is the culprit.
- **You changed a residual** (for example, tuned the diode `tanh`
  smoothing width or the MOSFET region selection) and want to confirm the
  manual derivation still matches the new expression.

For routine production runs, the default (manual) path is faster.

## Validation layer

Inside `pulsim::v1::ad`:

```cpp
#include "pulsim/v1/ad/validate.hpp"

std::vector<Vector> operating_points = { x1, x2, x3 };
auto mismatches = pulsim::v1::ad::validate_nonlinear_jacobians(
    circuit, operating_points, /*abs_tol=*/1e-6);
if (!mismatches.empty()) {
    for (const auto& m : mismatches) {
        std::cerr << m.device_type << " " << m.device_name
                  << " op=" << m.op_point_index
                  << " J(" << m.local_row << "," << m.local_col << ")"
                  << " stamp=" << m.stamp_value
                  << " fd="    << m.fd_value
                  << " |Δ|="   << m.abs_delta << "\n";
    }
}
```

Walks every nonlinear device in the circuit, stamps the build-selected
Jacobian (manual or AD depending on the flag), and compares the canonical
"current-out" J row against centered finite differences on the device's
templated residual helper. Empty result means the build agrees with FD on
every entry of every device at every supplied operating point. Use it as
a CI gate when adding new devices or modifying residuals.

## Performance budget

Measured on AppleClang 17 / Release+LTO / MOSFET in saturation, 100k
iterations of `MOSFET::stamp_jacobian`:

| Build mode | ns / stamp | Ratio |
|------------|-----------:|------:|
| Manual (default) | 222 | 1.00× |
| AD (`PULSIM_USE_AD_STAMP=ON`) | 848 | 3.81× |

Per-stamp overhead is currently dominated by heap allocations inside
`Eigen::AutoDiffScalar`. Phase 6.2 of `add-automatic-differentiation`
covers the optimization (stack-sized derivative vector, pre-allocated
arena). Total simulation-runtime overhead at the buck-converter level is
much smaller — stamping is only a fraction of the per-step work — and
ranges from 1.5× to 3× depending on circuit size and step count (see
[`benchmarks/`](../benchmarks)).

## Linear devices opt out

`Resistor`, `Capacitor`, `Inductor`, `VoltageSource`, `CurrentSource` keep
direct stamping in both build modes. Their Jacobians are constant per
topology — AD evaluation buys nothing while paying ADReal arithmetic
cost. The opt-out is enforced by a compile-time SFINAE check in
`test_ad_linear_opt_out.cpp` (asserts these devices do *not* expose
`stamp_jacobian_via_ad`).

## Authoring a new device with the AD path

If your device's Behavioral stamp is best authored as
`i = f(terminal voltages, params)`, follow the existing pattern:

1. Implement a templated residual helper:

   ```cpp
   template <typename S>
   [[nodiscard]] S forward_current_behavioral(S v_a, S v_c) const {
       // ... use Real for parameters, S for state-dependent quantities ...
   }
   ```

   Coefficients (`v_smooth_`, `g_on_`, etc.) stay as `Real`. Constants
   inside the formula likewise stay as `Real{0.5}` etc. — constructing
   `S{constant}` zeroes the derivative chain (Eigen quirk).

2. Implement `stamp_jacobian_via_ad(J, f, x, nodes)` that:
   - Pulls terminal voltages from `x`.
   - Calls `ad::seed_from_values({...})` to build `ADReal` inputs.
   - Evaluates the templated helper on the seeded inputs.
   - Reads `i_ad.value()` and `i_ad.derivatives()[k]` for each terminal.
   - Stamps J following the device's standard or Norton form.

3. Wire the build-flag dispatch in `stamp_jacobian_impl`:

   ```cpp
   #ifdef PULSIM_USE_AD_STAMP
       stamp_jacobian_via_ad(J, f, x, nodes);
   #else
       stamp_jacobian_behavioral(J, f, x, nodes);
   #endif
   ```

4. Add a cross-validation test (`test_ad_<device>_stamp.cpp`) covering
   each region of the device's operating space — manual and AD must agree
   to within 1e-12 absolute on every J entry and `f` row.

5. Update `validate.hpp` to dispatch the new device variant in
   `validate_nonlinear_jacobians`.

## Future direction

`PULSIM_USE_AD_STAMP=ON` is currently opt-in. The default flips to AD
once Phase 6.2 closes the per-stamp performance gap (≤30 % overhead
target). The legacy manual-stamp branches will then move behind a
`PULSIM_LEGACY_MANUAL_JACOBIAN=ON` opt-out, and a follow-up change
(`remove-legacy-manual-jacobians`) will retire them entirely after one
release window.
