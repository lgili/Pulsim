# Build System

Pulsim is built with CMake. The kernel itself is **header-only**, so
everything except the Python extension is a chain of INTERFACE
libraries + Catch2 test executables.

## Targets

| Target | Kind | Notes |
|---|---|---|
| `pulsim::core` (alias for `pulsim_core`) | INTERFACE | The entire C++23 header tree under `core/include/pulsim/`. Carries the C++23 standard requirement plus `Eigen3::Eigen` + `yaml-cpp::yaml-cpp` + KLU (+ optional HYPRE) link lines. |
| `pulsim_layer{0..5}_tests`, `pulsim_layer4_v{1..3}_tests`, `pulsim_layer5_v{1..4}_tests`, `pulsim_builder_tests`, `pulsim_yaml_tests`, `pulsim_showcase_tests`, `pulsim_sources_tests`, `pulsim_analysis_tests` | executable | Catch2 test binaries, one per layer / layer-milestone. Each links only `pulsim::core` + `Catch2::Catch2WithMain`. |
| `_pulsim` | shared module | The pybind11 binding (`python/bindings.cpp`). Links `pulsim::core`; loaded by Python as `pulsim._pulsim`. |

Every test binary is registered with `catch_discover_tests(...)`, so
`ctest` picks them up automatically.

## Configure + build

```bash
# Configure (Ninja recommended)
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPULSIM_BUILD_PYTHON=ON

# Build everything
cmake --build build -j

# Or build just one layer's test binary
cmake --build build --target pulsim_layer4_tests

# Run the full test harness
ctest --test-dir build --output-on-failure
```

## Build options

| Option | Default | Effect |
|---|---|---|
| `PULSIM_BUILD_TESTS` | `ON` | Build the layer-by-layer Catch2 test binaries. |
| `PULSIM_BUILD_PYTHON` | `OFF` | Build the `_pulsim` pybind11 extension (in `build/python/pulsim/`). |
| `PULSIM_BUILD_BENCHMARKS` | `OFF` | Add the `benchmark_compile_time` custom target (clean rebuild timing). |
| `PULSIM_ENABLE_LTO` | `ON` (Release) | Enable link-time optimization. |
| `PULSIM_ENABLE_NATIVE` | `OFF` | `-march=native -mtune=native` (don't ship binaries built with this). |
| `PULSIM_ENABLE_PGO_GENERATE` / `..._USE` | `OFF` | Two-pass PGO workflow (see comments in `CMakeLists.txt`). |
| `PULSIM_SANITIZERS` | `OFF` | `-fsanitize=address,undefined` (works best with `Debug`). |

## Dependencies

All managed through `find_package` first, then `FetchContent` as a
last resort:

- **Eigen 3.4+** — header-only matrix/vector library. Also provides
  `Eigen::SparseLU`, the direct sparse solver used by every Pulsim
  transient + DC solve. System install or fetched from
  `https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz`.
- **yaml-cpp 0.8.0** — fetched via `FetchContent`. Used only by the
  YAML circuit loader (`pulsim/yaml/loader.hpp`). Forced to C++17 to
  dodge GCC 14 + libstdc++ regressions when compiled in C++23 mode.
- **Catch2 v3.8.0+** — fetched. Only built when `PULSIM_BUILD_TESTS=ON`.
  Same C++17 forcing as yaml-cpp.
- **pybind11 ≥ 2.10** — only needed when `PULSIM_BUILD_PYTHON=ON`.
  Picks up `find_package(pybind11)` first (typical when building via
  `pip install` or `scikit-build-core`); falls back to `FetchContent`
  pinned at `v2.12.0`.

Pulsim does **not** depend on SuiteSparse / KLU, HYPRE, BLAS, LAPACK,
MKL, PETSc, SUNDIALS, or any other external solver library. The
direct sparse path is `Eigen::SparseLU` (header-only).

## Toolchain

C++23 is non-negotiable. The detection in `CMakeLists.txt` warns
(but does not fail) when:

- AppleClang < 15.0 (Xcode 15+ recommended),
- LLVM Clang < 17.0,
- GCC < 13.0,
- MSVC < 17.7.

For an explicit Clang toolchain (auto-detects Homebrew LLVM on
macOS, standard paths on Linux/Windows):

```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-clang.cmake
```

## Build performance

Two knobs that move the needle:

- **Ninja** — parallel scheduler is far better than Make for
  templated code.
- **LTO** — on by default in Release. Disabled automatically for
  Linux Python builds (pybind11 thread-local-storage interaction).

A `PULSIM_BUILD_BENCHMARKS=ON` configure adds the
`benchmark_compile_time` target, which times a clean rebuild of
the heaviest test binary (Layer 5 V4 — Newton in run_transient
pulls the full template tree).

## Editable Python install

For local Python development without `pip install`:

```bash
cmake -S . -B build -DPULSIM_BUILD_PYTHON=ON
cmake --build build -j
export PYTHONPATH="$(pwd)/build/python:$PYTHONPATH"
python3 -c "import pulsim as p; print(p.__version__)"
```

For a proper wheel build (used by `pip install .` and by CI):

```bash
pip install scikit-build-core pybind11
pip install -e .         # editable; rebuilds on Python import
```

The wheel is driven by `pyproject.toml` + `scikit-build-core`,
which runs the same top-level `CMakeLists.txt` with `SKBUILD=ON`.
