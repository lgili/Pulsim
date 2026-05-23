# Layer 0 — Numeric primitives + Sparse linear algebra

The foundation layer. Everything above uses it; nothing in it depends
on anything else above. Choosing it once, correctly, lets every layer
above compose naturally; choosing it poorly means v2 inherits v1's
mistakes.

## Public surface

```cpp
// pulsim/numeric/types.hpp
namespace pulsim {
    using Real  = double;          // overridable -DPULSIM_V2_REAL_TYPE=float
    using Index = std::int32_t;    // exactly 4 B, signed
    using Size  = std::size_t;
    inline constexpr Index kInvalidIndex = -1;
    inline constexpr Index kGround       = -1;
}

// pulsim/numeric/dense.hpp
namespace pulsim {
    using Vector      = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using DenseMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector2 / Vector3 / Vector4 / Matrix2 / Matrix3 / Matrix4;
}

// pulsim/numeric/concepts.hpp
namespace pulsim::numeric {
    template <typename T> concept FloatingPoint;
    template <typename T> concept IndexLike;
}

// pulsim/sparse/matrix.hpp
namespace pulsim::sparse {
    using Matrix  = Eigen::SparseMatrix<Real, Eigen::ColMajor, Index>;
    using Triplet = Eigen::Triplet<Real, Index>;
    Size stamp_dense(Matrix&, Index row, Index col, const DenseMatrix&);
    void reserve_capacity(Matrix&, Size nnz_estimate);
    void compress_in_place(Matrix&);
}

// pulsim/sparse/solver.hpp
namespace pulsim::sparse {
    class DirectSolver;            // abstract: analyze → factorize → solve
    class SparseLuSolver;          // concrete: Eigen::SparseLU
    std::unique_ptr<DirectSolver> make_default_solver();
}
```

That's the entire Layer 0 contract. ~250 LOC across 5 headers.

## Design decisions and why

### `Real = double` by default, overridable to `float`

Future-proofs the codebase for embedded HIL / FPGA targets without
making the default API template-parameterized. A build with
`-DPULSIM_V2_REAL_TYPE=float` flips Real to single precision; the
entire tree recompiles in float. Layer 2 + 3 templates accept any
floating-point type (so AD scalars and float work in the same
template), Layer 4 + 5 work in `Real` for numeric stability.

### `Index = std::int32_t` (signed, exactly 4 B)

- **4 B fits 2× more indices per cache line vs 8 B.** Cache density
  matters in stamping hot loops.
- **Matches int32 sparse-solver index arrays.** KLU, UMFPACK, MKL
  Pardiso expect int32 by default — no wrapper, no copy.
- **Signed allows `-1` as sentinel for ground / "not found"** without
  burning bit-31.
- **`2^31 − 1 ≈ 2 G nodes`** is plenty for any conceivable
  power-electronics circuit (large industrial inverter ≤ 1000 nodes).

Locked in via `static_assert(sizeof(Index) == 4)` at the top of
`numeric/types.hpp` — accidentally widening to 8 B halts the build.

### Sparse matrix: `ColMajor`, `Index` storage type

- **ColMajor matches every direct sparse solver's native input.**
  RowMajor would force a transpose-and-copy at every solver call.
- **ColMajor matches the MNA stamping pattern.** Each device touches
  a few entries in a few columns; ColMajor means those entries live
  next to each other in memory.
- **`Index` as the storage type keeps the matrix's index arrays
  packed at int32.**

### `DirectSolver` separates `analyze / factorize / solve`

The lifecycle contract is the foundation of the Layer 4 PWL
state-space cache:

| Step       | Frequency                                | Cost          |
|------------|------------------------------------------|---------------|
| `analyze`  | ONCE per topology change                 | O(n·log n)    |
| `factorize`| ONCE per matrix-value change             | O(n^1.5) typical |
| `solve`    | EVERY step                               | O(n) triangular |

For a stable switch combination, sparsity AND values are constant
across many steps. `analyze + factorize` run ONCE; many `solve` calls
reuse the cached factor. **This is the 5-10× speedup over v1's
current `linear_factor_cache` which conflates the two phases.**

Out-of-order calls throw `std::logic_error` with a clear message
naming the missing prerequisite. The contract is enforced at the
type level, not by documentation.

### `SparseLuSolver` as the reference backend

Layer 0 ships ONE concrete solver: `Eigen::SparseLU` with explicit
`COLAMDOrdering<Index>` (avoids the int32/int64 ordering-template
collision Eigen has by default).

Future backends (KLU, UMFPACK, MKL Pardiso, HYPRE) can register
through the same `DirectSolver` interface without modifying any
consumer. The factory `make_default_solver()` returns whatever the
runtime hint plus matrix-property heuristics select.

### Eigen wrapping is INTENTIONALLY thin

Pure aliases + a handful of stamping utilities. We do NOT re-implement
linear algebra. Eigen is mature, well-optimised, already a v1
dependency.

If Eigen ever needs replacing (e.g. GPU kernels via CUDA), Layer 0's
small surface IS the swap surface — not 50 files using `Eigen::VectorXd`
directly. **The smaller the surface, the cheaper the future swap.**

## What Layer 0 does NOT do

- No device models. Real + SparseMatrix is not a device.
- No Newton-Raphson, no integrator, no event detection — those live
  in Layer 5.
- No graph / topology / KCL / KVL — those live in Layer 1.
- No state space — Layer 4.

Future layers compose by ADDING capability above Layer 0; Layer 0
never grows to accommodate them (a banded solver, for example,
would be a NEW class implementing `DirectSolver`, not a modification
of `SparseLuSolver`).

## Validation

`pulsim_v2_layer0_tests` (built in `build/core/`) covers:

- Numeric type contracts (`sizeof(Index) == 4`, `is_signed`,
  `FloatingPoint` concept selectivity).
- Sparse matrix triplet assembly, `stamp_dense` accumulation,
  `compress_in_place` round-trip.
- `DirectSolver` lifecycle: SPD 3×3 solves to 1e-12, out-of-order
  calls throw with diagnostic, multiple factorize calls reuse
  symbolic cache, polymorphic dispatch through abstract interface.

Current: **80 assertions / 19 test cases, all green.**
