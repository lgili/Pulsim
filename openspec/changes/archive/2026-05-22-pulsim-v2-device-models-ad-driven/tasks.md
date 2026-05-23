## Phase 1 — Forward-mode AD scalar (~1 day)

### 1.1 `ad/ad_scalar.hpp` — `ADRealN<N>` template
- [x] 1.1.1 Templated class `ADRealN<Size N>` carrying `Real value_`
      + `std::array<Real, N> derivs_`. Default-constructed to
      `(0, {0...})`.
- [x] 1.1.2 Constructor from `Real` — sets value, derivs all zero
      (a "constant" with no input dependence).
- [x] 1.1.3 Accessors: `Real value() const`, `std::array<Real, N>
      derivatives() const`, `Real deriv(Size i) const`.
- [x] 1.1.4 Arithmetic operators (member + free):
      - `operator+` (ad+ad, ad+real, real+ad)
      - `operator-` (ad-ad, ad-real, real-ad, unary -)
      - `operator*` (ad*ad, ad*real, real*ad)
      - `operator/` (ad/ad, ad/real, real/ad)
      - `+=`, `-=`, `*=`, `/=` for ad+ad and ad+real cases
- [x] 1.1.5 Math functions via ADL (`using std::exp;` etc. inside
      device template):
      - `exp(ADRealN)`
      - `log(ADRealN)`
      - `sqrt(ADRealN)`
      - `tanh(ADRealN)`
      - `sinh(ADRealN)`
      - `cosh(ADRealN)`
      - `abs(ADRealN)` — chain rule uses `sign(x)`; at x=0
        returns the right-derivative.
- [x] 1.1.6 Comparison operators (`<`, `<=`, `>`, `>=`, `==`, `!=`)
      compare only the values — derivatives carry no order info.
      ad vs ad and ad vs real both supported.
- [x] 1.1.7 `ad::seed<N>(Real v0, Real v1, ...)` helper — returns a
      `std::array<ADRealN<N>, N>` where element `i` has `value = vi`
      and `derivs[i] = 1`, all other derivs 0. The canonical
      pattern for seeding device terminal voltages.

### 1.2 Tests `tests/v2/layer2/test_ad_scalar.cpp`
- [x] 1.2.1 Default-constructed `ADRealN<3>` has value 0 + zeros.
- [x] 1.2.2 Construct from `Real(2.5)` → value=2.5, derivs=zeros.
- [x] 1.2.3 Seed: `auto v = ad::seed<2>(3.0, 4.0)` gives
      `v[0].value()==3, v[0].deriv(0)==1, v[0].deriv(1)==0,
       v[1].value()==4, v[1].deriv(0)==0, v[1].deriv(1)==1`.
- [x] 1.2.4 Arithmetic chain rule:
      - `(x + y).value()` == `x.value() + y.value()`,
        derivs add elementwise.
      - `(x * y).deriv(i)` == `x.deriv(i)·y.value() +
        x.value()·y.deriv(i)`.
      - `(x / y).deriv(i)` == quotient rule.
- [x] 1.2.5 Math functions match analytical derivatives:
      - `d/dx exp(x) = exp(x)` → `exp(x).deriv(0) ==
        exp(x.value())` when x is seeded.
      - `d/dx tanh(x) = 1 - tanh²(x)`.
      - `d/dx sqrt(x) = 1/(2·sqrt(x))`.
- [x] 1.2.6 Composition: `f(x, y) = exp(x) + y²` at (1, 2) →
      value = e + 4 ≈ 6.7182, ∂f/∂x = e ≈ 2.7182, ∂f/∂y = 4.
- [x] 1.2.7 Comparison operators compare values only.

## Phase 2 — Device model concept (~0.5 days)

### 2.1 `models/device_model.hpp` — the concept + helpers
- [x] 2.1.1 Concept `pulsim::v2::models::DeviceModel<T>` requires:
      - Nested `typename T::Params`
      - Static constant `T::kind` of type `topology::BranchKind`
      - Static constant `T::num_terminals` of type `Size`
      - Static constant `T::is_linear` of type `bool`
      - Templated static method `T::template current<S>(const S* v,
        const Params& p)` accepting any `numeric::FloatingPoint S`,
        returning `S`. The `v` pointer points to N terminal voltages
        in device-defined order; the function returns the current
        flowing from terminal 0 to terminal 1 (convention).
- [x] 2.1.2 Helper `evaluate_current_and_jacobian` — given a device
      model T + terminal voltages, returns a `Tuple<Real,
      std::array<Real, N>>` of (current, ∂current/∂terminal_i). This
      is the bridge Layer 3 will use: ONE call into the device model
      yields both the value and the partial derivatives without
      duplicated code.
- [x] 2.1.3 Convenience type alias `ModelInputs<T>` =
      `std::array<Real, T::num_terminals>` so device tests can
      declaratively name their voltage arrays.

### 2.2 Tests `tests/v2/layer2/test_device_model_concept.cpp`
- [x] 2.2.1 Verify the three reference models satisfy
      `DeviceModel`:
      `static_assert(DeviceModel<Resistor>);`
      `static_assert(DeviceModel<VoltageSource>);`
      `static_assert(DeviceModel<IdealDiode>);`
- [x] 2.2.2 A struct missing one of the required members FAILS the
      concept check (negative test via `static_assert(!DeviceModel<
      BrokenStub>)`).
- [x] 2.2.3 `evaluate_current_and_jacobian` returns matching value
      + derivatives for the Resistor (analytical Jacobian = G·[1,
      -1]).

## Phase 3 — Resistor model (~0.25 days)

### 3.1 `models/resistor.hpp`
- [x] 3.1.1 `struct Resistor::Params { Real G; };` — conductance in
      siemens (not resistance in ohms — keeps the math simple and
      avoids divisions in hot paths).
- [x] 3.1.2 `static constexpr Size num_terminals = 2;`
- [x] 3.1.3 `static constexpr topology::BranchKind kind =
      topology::BranchKind::PassiveLinear;`
- [x] 3.1.4 `static constexpr bool is_linear = true;`
- [x] 3.1.5 `template <numeric::FloatingPoint S> static S
      current(const S* v, const Params& p) noexcept` —
      `return p.G * (v[0] - v[1]);`. Terminal 0 = positive, terminal
      1 = negative. Current flows from 0 to 1 (positive when
      v[0] > v[1]).

### 3.2 Tests `tests/v2/layer2/test_resistor.cpp`
- [x] 3.2.1 Forward eval: G=2, v=[3, 1] → i=4.
- [x] 3.2.2 AD derivatives: ∂i/∂v[0] = G, ∂i/∂v[1] = -G. Verify
      with `evaluate_current_and_jacobian`.
- [x] 3.2.3 Edge case: v[0] == v[1] → i=0, derivatives still ±G.
- [x] 3.2.4 The concept is satisfied (compile-time
      `static_assert(DeviceModel<Resistor>)`).

## Phase 4 — VoltageSource model (~0.25 days)

### 4.1 `models/voltage_source.hpp`
- [x] 4.1.1 `struct VoltageSource::Params { Real V; };` — DC source
      voltage (time-varying sources are a Layer 6 concern).
- [x] 4.1.2 `kind = Source`, `num_terminals = 2`, `is_linear = true`.
- [x] 4.1.3 `current<S>` for a voltage source returns ZERO from the
      device-model contract — the source's contribution is a
      constraint equation `v[0] - v[1] = V`, not a stamped
      current. Layer 3 detects `kind == Source` and adds the
      constraint row instead of stamping a current. The `V` is
      exposed via `static_voltage(p)` for Layer 3's use.

### 4.2 Tests `tests/v2/layer2/test_voltage_source.cpp`
- [x] 4.2.1 `current<Real>` returns 0 regardless of terminal
      voltages (the source is a constraint, not a current).
- [x] 4.2.2 `static_voltage(p)` returns the configured V.
- [x] 4.2.3 The concept is satisfied
      (`static_assert(DeviceModel<VoltageSource>)`).

## Phase 5 — IdealDiode model (the AD killer test, ~0.75 days)

### 5.1 `models/ideal_diode.hpp` — smooth-blend behavioral form
- [x] 5.1.1 `struct IdealDiode::Params { Real V_F0; Real R_d;
      Real G_off; Real kappa; };` — forward drop, slope resistance,
      off-state conductance, sigmoid sharpness.
- [x] 5.1.2 `kind = Nonlinear`, `num_terminals = 2`,
      `is_linear = false`.
- [x] 5.1.3 `template <numeric::FloatingPoint S> static S
      current(const S* v, const Params& p) noexcept`:
      - Compute `v_diode = v[0] - v[1]` (anode minus cathode)
      - Smooth-blend on-state factor:
        `alpha = 1 / (1 + exp(-kappa·(v_diode - V_F0)))`
      - Norton-shifted on current: `i_on = alpha·(v_diode - V_F0) /
        R_d` (current scales linearly above V_F0)
      - Off leakage: `i_off = (1 - alpha)·v_diode·G_off`
      - Return: `i_on + i_off`
      - The whole function templated on S — instantiates for
        `double` (forward eval) and `ADRealN<2>` (Jacobian).

### 5.2 Tests `tests/v2/layer2/test_ideal_diode.cpp`
- [x] 5.2.1 Forward eval reverse-biased: V_F0=0.7, v=[0, 5] →
      i ≈ 0 (G_off leak only, small negative).
- [x] 5.2.2 Forward eval forward-biased: V_F0=0.7, R_d=0.01,
      v=[1.0, 0] → i ≈ (1.0 - 0.7)/0.01 = 30 A.
- [x] 5.2.3 Forward eval at threshold: V_F0=0.7, v=[0.7, 0] →
      i ≈ G_off · 0.7 / 2 (alpha ≈ 0.5).
- [x] 5.2.4 AD vs finite-difference: at v=[0.5, 0] (sub-threshold),
      v=[0.7, 0] (at-threshold), v=[1.0, 0] (forward-biased),
      compute ∂i/∂v[0] and ∂i/∂v[1] via
      `evaluate_current_and_jacobian` AND via central-difference
      `(i(v[0] + h) - i(v[0] - h)) / (2h)` with `h=1e-6`. Assert
      agreement within `1e-6` absolute.
- [x] 5.2.5 ∂i/∂v[0] + ∂i/∂v[1] == 0 always (the current depends
      only on `v[0] - v[1]`, so the partials are equal in
      magnitude and opposite in sign). Numerical proof for the
      AD path.

## Phase 6 — Documentation (~0.25 days)

### 6.1 `docs/pulsim-v2/layer2-ad-and-device-models.md`
- [x] 6.1.1 Section "The AD killer" — how `current<S>` instantiated
      for `double` is the forward eval AND for `ADRealN<N>` is the
      Jacobian extraction. Code example showing both calls.
- [x] 6.1.2 Section "Why forward mode" — Jacobian-vector vs
      vector-Jacobian arguments; for devices with ≤ ~8 terminals
      forward mode is faster (no tape, stack-allocated).
- [x] 6.1.3 Section "How to add a new device" — three steps: write
      a `Params` struct, write a templated `current<S>`, declare
      the static constants. The concept verifies the rest.

## Phase 7 — Validation

- [x] 7.1 `pulsim_v2_layer2_tests` MUST pass with zero failures.
      Initial target: ≥ 40 assertions / ≥ 15 test cases.
- [x] 7.2 `pulsim_v2_layer0_tests` AND `pulsim_v2_layer1_tests` MUST
      stay green. Layer 2 doesn't touch lower layers.
- [x] 7.3 v1 suites (`pulsim_tests`, `pulsim_simulation_tests`) MUST
      stay green.
- [x] 7.4 `openspec validate pulsim-v2-device-models-ad-driven
      --strict` MUST pass.
