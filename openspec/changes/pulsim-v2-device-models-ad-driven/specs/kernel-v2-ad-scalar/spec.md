## ADDED Requirements

### Requirement: Forward-Mode AD Scalar

`pulsim::v2::ad::ADRealN<N>` SHALL be a forward-mode automatic-
differentiation scalar carrying a `Real value_` and a
`std::array<Real, N>` of partial derivatives with respect to `N`
seeded inputs.

The type MUST support:
- Default construction (value = 0, derivs = all zero).
- Construction from `Real` (value = the real, derivs = all zero —
  representing a constant with no input dependence).
- Accessors: `value()`, `derivatives()`, `deriv(Size i)`.
- Arithmetic operators: binary `+ - * /` and unary `-`, mixing
  `ADRealN` and `Real`. Compound `+= -= *= /=` for self-assignment.
- Math functions via ADL: `exp`, `log`, `sqrt`, `tanh`, `sinh`,
  `cosh`, `abs`.
- Comparison operators (`< <= > >= == !=`) comparing values only.
- A `pulsim::v2::ad::seed<N>(Real v0, Real v1, ...)` helper that
  returns a `std::array<ADRealN<N>, N>` where element `i` has
  `value = vi` and the `i`-th derivative slot set to 1, all other
  slots 0.

All operations MUST propagate derivatives via chain rule. The
result type after any AD-vs-AD or AD-vs-Real operation MUST be
`ADRealN<N>` (no truncation, no implicit conversion to `Real`).

#### Scenario: Seeded inputs propagate through addition + multiplication

- **GIVEN** `auto [x, y] = ad::seed<2>(3.0, 4.0)`
- **WHEN** the user computes `z = x + y`
- **THEN** `z.value()` SHALL equal `7.0`
- **AND** `z.deriv(0)` SHALL equal `1.0`
- **AND** `z.deriv(1)` SHALL equal `1.0`
- **AND** when the user computes `w = x * y`, then
  `w.value()` SHALL equal `12.0`, `w.deriv(0)` SHALL equal `4.0`
  (= y), and `w.deriv(1)` SHALL equal `3.0` (= x).

#### Scenario: exp propagates self-derivative

- **GIVEN** `auto [x] = ad::seed<1>(2.0)`
- **WHEN** the user computes `y = exp(x)`
- **THEN** `y.value()` SHALL equal `exp(2.0)` within 1e-12
- **AND** `y.deriv(0)` SHALL equal `exp(2.0)` within 1e-12.

#### Scenario: Composition through chain rule

- **GIVEN** `auto [x, y] = ad::seed<2>(1.0, 2.0)`
- **WHEN** the user computes `f = exp(x) + y*y`
- **THEN** `f.value()` SHALL equal `e + 4` within 1e-12
- **AND** `f.deriv(0)` SHALL equal `e` (= d/dx exp(x))
- **AND** `f.deriv(1)` SHALL equal `4.0` (= 2·y at y=2).

#### Scenario: Comparison operators compare values only

- **GIVEN** two `ADRealN<2>` values `a` (value 3.0, derivs [1, 0])
  and `b` (value 5.0, derivs [0, 1])
- **WHEN** the user evaluates `a < b`
- **THEN** the result SHALL be `true`
- **AND** the derivatives SHALL play no role (operator returns
  `bool`, not an AD type).
