## ADDED Requirements

### Requirement: Numeric Scalar and Index Types

`pulsim::v2` SHALL define three core numeric type aliases that every
higher layer uses:

- `pulsim::v2::Real` — the simulator's working floating-point
  precision. Default `double`; overridable at build time via
  `-DPULSIM_V2_REAL_TYPE=float` for single-precision builds.
- `pulsim::v2::Index` — `std::int32_t`. Used for node IDs, branch
  IDs, sparse matrix row/column indices. Signed so that `-1` is a
  valid sentinel for ground / "no such device".
- `pulsim::v2::Size` — `std::size_t`. Used for container sizes and
  iteration counts.

The header MUST expose two compile-time sentinels:
- `pulsim::v2::kInvalidIndex == -1`
- `pulsim::v2::kGround == -1`

#### Scenario: Real defaults to double, can be flipped to float

- **GIVEN** a default Pulsim v2 build (no CMake overrides)
- **WHEN** the consumer queries `sizeof(pulsim::v2::Real)`
- **THEN** the result SHALL be `8` (double precision)
- **AND** a build with `cmake -DPULSIM_V2_REAL_TYPE=float ...` SHALL
  yield `sizeof(pulsim::v2::Real) == 4` (single precision).

#### Scenario: Index is signed int32 for cache and sentinel reasons

- **GIVEN** the `pulsim::v2::Index` type
- **WHEN** the consumer queries `sizeof(pulsim::v2::Index)`
- **THEN** the result SHALL be `4`
- **AND** `std::is_signed_v<pulsim::v2::Index>` SHALL be `true`
- **AND** `std::numeric_limits<pulsim::v2::Index>::max()` SHALL be
  ≥ `2^31 − 1`.

### Requirement: Dense Vector and Matrix Aliases

`pulsim::v2` SHALL expose dense linear-algebra type aliases that wrap
Eigen without adding intermediate abstractions:

- `pulsim::v2::Vector` = `Eigen::Matrix<Real, Eigen::Dynamic, 1>`
- `pulsim::v2::DenseMatrix` = `Eigen::Matrix<Real, Eigen::Dynamic,
  Eigen::Dynamic>`

No additional operator overloads are added — consumers use Eigen's
expression templates directly. The aliases exist for type-name
clarity ("this code operates on a Pulsim Vector") not for behaviour
change.

#### Scenario: Vector::Zero(N) creates an N-element zero vector

- **GIVEN** a positive integer N
- **WHEN** the user constructs `pulsim::v2::Vector::Zero(N)`
- **THEN** the result SHALL be an `N × 1` Eigen vector
- **AND** every element SHALL be `0.0` (within Real precision).

### Requirement: Numeric Concepts for Generic Templates

`pulsim::v2::numeric` SHALL expose two C++20 concepts that future
layers use to constrain templates:

- `pulsim::v2::numeric::FloatingPoint` — satisfied by `Real` and any
  standard `std::floating_point` type. Layer 2 (device models) uses
  this to write a single templated `current<S>(...)` function that
  accepts both `double` and AD scalar types.
- `pulsim::v2::numeric::IndexLike` — satisfied by any signed
  integer type of at least 32 bits. Used by Layer 3 (stamping) and
  Layer 4 (state-space cache) to constrain template indices.

#### Scenario: FloatingPoint accepts double and rejects int

- **GIVEN** the `pulsim::v2::numeric::FloatingPoint` concept
- **WHEN** evaluated against `double`
- **THEN** the result SHALL be `true`
- **AND** evaluated against `int` SHALL be `false`
- **AND** evaluated against `pulsim::v2::Real` SHALL be `true`.
