# Layer 2 — Forward-mode AD + AD-driven device models

The layer that kills the v1 four-stamp duplication problem
permanently. Each device exposes ONE templated `current<S>(...)`
function. The same function instantiates for `Real` (forward eval)
AND for `ADRealN<N>` (Jacobian via AD). Manual derivatives become
a type error.

## Public surface

```cpp
// pulsim/ad/ad_scalar.hpp
namespace pulsim::ad {
    template <Size N> class ADRealN;
    template <Size N> exp(ADRealN), log, sqrt, tanh, sinh, cosh, abs;
    auto seed2(Real v0, Real v1) -> std::array<ADRealN<2>, 2>;
    auto seed3(Real v0, Real v1, Real v2) -> std::array<ADRealN<3>, 3>;
    auto seed4(Real v0, Real v1, Real v2, Real v3) -> std::array<ADRealN<4>, 4>;
}

// pulsim/models/device_model.hpp
namespace pulsim::models {
    template <typename T>
    concept DeviceModel = /* see header */;

    template <DeviceModel T>
    using ModelInputs = std::array<Real, T::num_terminals>;

    template <DeviceModel T>
    auto evaluate_current_and_jacobian(const ModelInputs<T>&, const T::Params&)
        -> std::tuple<Real, std::array<Real, T::num_terminals>>;

    template <DeviceModel T>
    Real evaluate_current(const ModelInputs<T>&, const T::Params&);
}

// pulsim/models/resistor.hpp        — linear passive (Ohm's law)
// pulsim/models/voltage_source.hpp  — constraint (kind = Source)
// pulsim/models/ideal_diode.hpp     — nonlinear (smooth-blend)
```

## The AD killer pattern

```cpp
// One templated function. ONE source of truth for the math.
struct Resistor {
    struct Params { Real G; };
    static constexpr Size num_terminals = 2;
    // ...
    template <numeric::FloatingPoint S>
    static S current(const S* v, const Params& p) noexcept {
        return p.G * (v[0] - v[1]);
    }
};

// Forward evaluation (Real instantiation)
Real i = Resistor::current<Real>(v, p);

// Jacobian extraction (ADRealN<2> instantiation — SAME function)
auto seeded = ad::seed2(v[0], v[1]);
auto i_ad = Resistor::current<ad::ADRealN<2>>(seeded.data(), p);
// i_ad.value()      = same Real i from above
// i_ad.deriv(0)     = ∂i/∂v[0] = G (analytically)
// i_ad.deriv(1)     = ∂i/∂v[1] = -G
```

Layer 3's generic stamper will call `evaluate_current_and_jacobian`
once per device per Newton iteration, get back (current, ∂i/∂v[k]),
and stamp the residual + Jacobian rows. No device-specific stamping
code. No 4-place duplication. **The v1 diode-fails-after-reverse-
bias bug becomes structurally impossible** because the math lives in
exactly ONE function.

## Why forward mode (and not reverse mode)

For Pulsim's device shape — few inputs (≤ 8 terminals), one output
(the current) — forward mode is the right answer:

| Aspect           | Forward mode (chosen)         | Reverse mode (rejected)          |
|------------------|-------------------------------|----------------------------------|
| Memory           | `sizeof(ADRealN<8>) = 72 B`  | Heap-allocated tape, dynamic     |
| Cost per call    | O(N) per arithmetic op       | O(1) per arithmetic op, but tape allocation overhead |
| Implementation   | Operator overloads, stack    | Tape recording + back-propagation |
| Inlinable?       | Yes, fully                   | Hard — tape is dynamic           |
| Best for         | Few inputs, one output       | Many inputs, one output (ML)     |

For a device with N > 32 (none in Pulsim's catalogue today) we'd
revisit. Until then, forward mode wins on simplicity, speed, AND
zero heap traffic.

## How to add a new device (three steps)

1. **Write the `Params` struct** — user-facing parameters.
2. **Write the `current<S>` template** — pure math, AD-compatible.
3. **Declare the static constants** — `kind`, `num_terminals`,
   `is_linear`.

That's it. The `DeviceModel` concept verifies the contract at
compile time; Layer 3's stamper consumes it generically. No
boilerplate, no factory, no virtual dispatch.

Example: adding a `Capacitor` (in a future OpenSpec):

```cpp
struct Capacitor {
    struct Params { Real C; Real V_init; };
    static constexpr topology::BranchKind kind =
        topology::BranchKind::PassiveLinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear = true;

    // For trapezoidal companion, the stamp depends on dt and the
    // previous step's voltage — Layer 3 handles the dt scaling
    // and the history term separately. The device-model side just
    // provides the conductance-equivalent and history-current at a
    // given state.
    //   ...
};
```

The same pattern extends to MOSFET (N=3), IGBT (N=3), transformer
(N=4), motors (N=4-5). Each is one new header. Layer 3 doesn't
change.

## What this layer does NOT do

- No matrix stamping. Layer 3 takes `evaluate_current_and_jacobian`
  output and stamps it.
- No state-space cache. Layer 4 builds the cache by walking the
  graph + device models.
- No Newton iteration, no events, no integrator. Layer 5.
- No reverse-mode AD. Not needed for ≤ 8 terminals.
- No symbolic DSL (Modelica-style equation description). Would be
  a v3 conversation if Layer 2's compile-time pattern hits its
  limits.

## Validation

`pulsim_v2_layer2_tests` covers each header in isolation:

- **AD scalar** (14 cases): default + constant construction, seed,
  arithmetic (chain rule for + − × /), math functions (`exp`,
  `tanh`, `sqrt`), composition `exp(x) + y²`, comparison
  value-only, compound assignment.
- **DeviceModel concept** (6 cases): three reference models satisfy
  the concept; a broken stub fails; `evaluate_current_and_jacobian`
  returns matching (value, partials); `ModelInputs<T>` has correct
  size.
- **Resistor** (5 cases): Ohm's law, AD partials ±G analytically,
  zero-voltage edge case, partials-sum-to-zero invariant,
  large-G scaling.
- **VoltageSource** (4 cases): current is always 0; static_voltage
  returns the configured V; AD partials are all zero.
- **IdealDiode** (the AD killer, 7 cases): reverse-biased ≈ 0,
  forward-biased = (v_diode - V_F0)/R_d, threshold midpoint,
  AD-vs-FD partials at three op-points (sub-threshold,
  threshold, forward-biased), partials-sum-to-zero invariant
  across multiple op-points.

Current: **93 assertions / 36 test cases, all green.**
Layer 0 + 1 regression: 80 + 126 assertions, all green.
Total v2 surface: **299 assertions / 91 test cases**.
