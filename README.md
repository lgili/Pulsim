<div align="center">

```
██████╗ ██╗   ██╗██╗     ███████╗██╗███╗   ███╗
██╔══██╗██║   ██║██║     ██╔════╝██║████╗ ████║
██████╔╝██║   ██║██║     ███████╗██║██╔████╔██║
██╔═══╝ ██║   ██║██║     ╚════██║██║██║╚██╔╝██║
██║     ╚██████╔╝███████╗███████║██║██║ ╚═╝ ██║
╚═╝      ╚═════╝ ╚══════╝╚══════╝╚═╝╚═╝     ╚═╝
```

### Header-only C++23 + Python power-electronics simulator with an in-house sparse-LU kernel.

[![CI](https://github.com/lgili/Pulsim/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lgili/Pulsim/actions/workflows/ci.yml)
[![Docs](https://github.com/lgili/Pulsim/actions/workflows/docs.yml/badge.svg?branch=main)](https://lgili.github.io/Pulsim/)
[![PyPI](https://img.shields.io/pypi/v/pulsim.svg?logo=python&logoColor=white&color=3776ab)](https://pypi.org/project/pulsim/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776ab?logo=python&logoColor=white)](https://pypi.org/project/pulsim/)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599c?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/23)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Citation](https://img.shields.io/badge/cite-CITATION.cff-orange.svg)](CITATION.cff)
[![v1.8.0 release](https://img.shields.io/badge/release-v1.8.0-success.svg)](CHANGELOG.md)

[**Docs**](https://lgili.github.io/Pulsim/) ·
[**Tutorials**](docs/tutorials/) ·
[**API reference**](docs/api-reference.md) ·
[**How Pulsim works**](docs/how-pulsim-works/) ·
[**Examples**](examples/scripts/) ·
[**Changelog**](CHANGELOG.md)

</div>

---

## TL;DR

Pulsim is a SPICE alternative tuned for the **switched-mode power
electronics** workload: PWL devices, finite-state topologies,
$10^8$-step transients. Where SPICE pays full assembly + LU
factorization on every step, Pulsim caches one state-space
per switch combination and updates only the columns of $L+U$
that actually changed.

The v1.3.0 sparse LU rewrite + the v1.4.0 generalised path-based
update framework give:

| Workload                            | n_state | Speedup vs baseline       |
|-------------------------------------|--------:|---------------------------|
| Buck single-bit Gray-code           | 14      | **2.81×**                 |
| Multi-bit δ=2 switch transition     | 14      | **1.58×** vs Eigen LU     |
| Parametric sweep (R/L/C)            | 14–26   | **3.0–3.7×** vs rebuild   |
| AC sweep complex LU                 | 8–32    | rough parity with Eigen   |

Both single-bit and (v1.4.0+) multi-bit / parametric paths run
through an **in-house C++23 sparse LU**. No KLU, no SuiteSparse,
no `Eigen::SparseLU<complex>` in production — only Eigen as a
matrix container.

---

## Why Pulsim

| Feature | What it buys you |
|---|---|
| **PLECS-style PWL cache** (Layer 4) | switched-converter transient in milliseconds, not minutes |
| **Header-only C++23 kernel** | drop `pulsim/` into your CMake via `pulsim::core`; no static-library link |
| **In-house sparse LU** (Pulsim's own, v1.3.0+) | path-based partial refactor for switch flips + parameter sweeps; zero third-party LU |
| **Python-first ergonomics** | `CircuitBuilder` takes string node names + SI units; one `SimulationResult` for transients, AC, sweeps, MC |
| **Mixed-domain control chain** | PID / comparator / rate-limiter / FOC / thermal at kernel speed (no Python interpreter per step) |
| **Frequency-domain analysis** | small-signal MNA Bode + swept-sine FRA + closed-loop GM/PM in the same surface |
| **Drop-in parameter sweep + MC** (v1.4.0+) | `sweep_path_aware` / `monte_carlo_path_aware` reuse the cached LU across all sweep points |

---

## Performance

Numbers captured 2026-05-24 on macOS / Apple Silicon (-O3 -DNDEBUG).
Reproducible end-to-end via `cmake --build build --target pulsim_benchmarks`
followed by `./build/core/pulsim_benchmarks "[rank1][microbench]"` (and
the analogous tags `[multi_bit]`, `[parametric]`, `[ac_sweep]`).

### 1) Single-bit rank-1 (v1.3.0, N-switch chain)

3-backend microbench. (A) baseline `solve` per-mask cache,
(B) Eigen sliding solver, (C) Pulsim path-based partial refactor.

| N | n_state | µs/solve (A) | µs/eigen (B) | µs/pulsim (C) | **C/A** |
|---:|---:|---:|---:|---:|---:|
| 4  | 6  | 6.74  | 6.15 | 2.30 | **2.93×** |
| 12 | 14 | 10.01 | 5.41 | 3.56 | **2.81×** |
| 16 | 18 | 12.15 | 7.08 | 4.31 | **2.82×** |
| 20 | 22 | 13.85 | 8.24 | 5.08 | **2.73×** |
| 24 | 26 | 16.41 | 9.83 | 6.13 | **2.68×** |

Zero fallbacks across all 1999 single-bit Gray-code flips × 8 N values.

### 2) Multi-bit path-union (v1.4.0, Pulsim ÷ Eigen sliding solver)

| N (n_state) | δ=1 | δ=2 | δ=3 | δ=4 |
|---|---:|---:|---:|---:|
| 8 (10)  | **3.12×** | 1.62× | 1.61× | 1.42× |
| 12 (14) | 1.72×     | 1.58× | 1.58× | 1.42× |
| 16 (18) | 1.56×     | 1.28× | 1.51× | 1.25× |
| 20 (22) | 1.36×     | 1.42× | 1.54× | 1.51× |
| 24 (26) | 1.55×     | 1.46× | 1.33× | 1.42× |

Hit rate decays gracefully: ~45 % path-union at δ=2 → ~10 % at δ=4
(rest fall back cleanly to full factorize).

### 3) Parametric sweep / Monte Carlo (v1.4.0, Pulsim ÷ legacy rebuild)

| n_state | 50 pts | 100 pts | 500 pts | 1000 pts |
|---:|---:|---:|---:|---:|
| 8  | **5.18×** | 3.29× | 3.55× | 3.68× |
| 14 | 3.57×     | 3.02× | 3.51× | 3.35× |
| 26 | 3.53×     | 3.31× | 3.38× | 3.40× |

**Zero fallbacks** across all 12 cells.

### 4) AC sweep complex LU (v1.4.0, Pulsim ÷ Eigen)

| n | µs/freq Eigen | µs/freq Pulsim | Pulsim ÷ Eigen | Parity Δ |
|---:|---:|---:|---:|---:|
| 8   | 8.98  | 5.48  | **0.61× (Pulsim faster)** | 1×10⁻²² |
| 16  | 8.16  | 7.64  | 0.94× | 5×10⁻²² |
| 32  | 13.98 | 14.56 | 1.04× | 3×10⁻²¹ |
| 64  | 28.49 | 33.36 | 1.17× | 1×10⁻²¹ |
| 128 | 46.13 | 91.52 | 1.98× (Eigen faster) | 4×10⁻²¹ |

Both solvers numerically interchangeable. v1.4.0 contribution is
**no third-party LU on the production path**; `Backend::Eigen` is
kept explicitly available as a paper-comparison baseline.

---

## Quick start

### Prerequisites

Two native dependencies — everything else is header-only or vendored
at configure time:

| Dependency | Why |
|---|---|
| **Eigen 3.4+** | Header-only sparse linear algebra (matrix + vector containers). Pulsim ships its own sparse LU on top. |
| **C++23 compiler** | AppleClang 15+ / Clang 17+ / GCC 13+ |

```bash
# macOS (Homebrew)
brew install cmake ninja eigen

# Debian / Ubuntu
sudo apt-get install -y cmake ninja-build libeigen3-dev

# Fedora
sudo dnf install cmake ninja-build eigen3-devel
```

### Install from PyPI

```bash
pip install pulsim
```

### Or build from source

```bash
git clone https://github.com/lgili/Pulsim.git
cd Pulsim
pip install -e .
```

### First simulation — 8 lines

```python
import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "n0", "gnd", 5.0)
b.add_resistor      ("R1", "n0", "vc",   1000.0)
b.add_capacitor     ("C1", "vc", "gnd",  1e-6)

res = p.simulate(b, t_end=5e-3, dt=1e-5)
p.plot.scope(b, res, signals=["vc"])      # one-line waveform
```

### Parametric sweep — exploit the v1.4.0 path-based refactor

```python
import numpy as np
import pulsim as p

def make_rc(R=5.0, C=10e-6):
    b = p.CircuitBuilder()
    b.add_voltage_source("v1", "vin", "gnd", 10.0)
    b.add_resistor      ("R_load", "vin", "vout", R)
    b.add_capacitor     ("C_out", "vout", "gnd", C)
    return b

def steady_state_vout(res, _params):
    states = np.asarray(res.states)
    return {"vout": float(states[-1, 1])}

# Sweep R_load through 100 values. v1.4.0 path-based refactor:
# build the cache ONCE, then reuse the L+U factors at each point.
out = p.sweep_path_aware(
    make_rc(),                               # initial builder
    params={"R_load": np.linspace(1, 20, 100).tolist()},
    kpi_fn=steady_state_vout,
    t_end=5e-3, dt=1e-6,
)
print(out.to_dataframe().head())
```

For Monte Carlo, swap `sweep_path_aware` →
`monte_carlo_path_aware(builder, distributions={"R_load": lambda r:
r.uniform(1, 20)}, n_samples=1000, ...)` and get the same ~3.4×
speedup over the legacy rebuild-per-sample pattern.

---

## What's new in v1.4.0

Two algorithmic contributions ship in one release — see
[`CHANGELOG.md`](CHANGELOG.md) for the full breakdown.

### In-house complex sparse LU

`PulsimSparseLuSolver` is now templated on `Scalar`. The new
`PulsimComplexSparseLuSolver = PulsimSparseLuSolverT<std::complex<Real>>`
drives the AC-sweep production path — `Eigen::SparseLU<complex>` is
no longer compiled in. Backward-compat aliases keep the v1.3.0
real-scalar API source-identical:

```cpp
using PulsimSparseLuSolver        = PulsimSparseLuSolverT<Real>;
using PulsimComplexSparseLuSolver = PulsimSparseLuSolverT<std::complex<Real>>;
using Matrix                      = MatrixT<Real>;
using Vector                      = VectorT<Real>;
```

### Multi-bit + parametric path-based update

The v1.3.0 single-bit path-based partial refactor now generalises to
**three SMPS-relevant use cases**:

| Case                                | Mechanism                  | Speedup |
|-------------------------------------|----------------------------|--------:|
| Single-bit switch flip (v1.3.0)     | etree path of one column   | 2.7–2.9× |
| Multi-bit switch flip (v1.4.0+)     | **union** of etree paths   | 1.3–1.7× |
| Parametric value sweep (v1.4.0+)    | path of param-affected cols| 3.0–3.7× |

The `MAX_PATH_LENGTH_RATIO = 0.6` gate skips the path-based attempt
when the union path covers > 60 % of $n$ (the path-walk would cost
the same as a fresh factorise).

### Python helpers

Drop-in replacements for the legacy sweep / MC APIs:

```python
import pulsim as p
p.sweep_path_aware(builder, params={...}, kpi_fn=..., t_end=..., dt=...)
p.monte_carlo_path_aware(builder, distributions={...}, n_samples=..., ...)
```

Auto-fallback to the legacy code path when a parameter name isn't
recognised (warns + delegates) — drop-in safe.

---

## What's NOT in Pulsim (intentionally)

Many simulators inherit a sprawling dependency tree from their
sparse-LU backend. Pulsim deliberately doesn't:

| ❌ Not used | Why |
|---|---|
| **SuiteSparse / KLU** | The v1.3.0 in-house LU replaces KLU completely. No SuiteSparse linkage, no licensing complexity. |
| **Eigen::SparseLU<complex>** | Replaced in v1.4.0 by the in-house complex specialisation. `Backend::Eigen` is retained explicitly as a paper-comparison baseline. |
| **dpsim / dwf / PSCAD vendored code** | Pulsim's algorithms are entirely first-party; the methods paper claims ours, not someone else's. |
| **Runtime Python in kernel** | The C++ kernel has zero Python dependency. `pybind11` only crosses the boundary at the API surface. |
| **Heavy installer / multi-step build** | `pip install pulsim` or `cmake --build`; no make-then-make-install dance. |

What IS used: **Eigen 3.4+** (header-only, as a sparse matrix
container) and **pybind11** (for the Python binding). That's it.

---

## Architecture

10 layers, bottom-up:

```
Layer 0:  Sparse LU + matrix containers      (in-house, real + complex)
Layer 1:  Topology graph + switch state mask
Layer 2:  Device models (R, L, C, MOSFET, IGBT, source, ...)
Layer 3:  Symbolic stamping helpers (Newton refresh)
Layer 4:  PWL state-space cache  ← refactor_parametric + solve_rank1
Layer 5:  Solver loop + event detection + run_transient
Layer 6:  CircuitBuilder (ergonomic API surface)
Layer 7:  YAML loader + reference circuits
Layer 8:  AC analysis (small-signal MNA Bode + FRA)
Layer 9:  Python facade (CircuitBuilder, simulate, sweep, plot)
```

See [`docs/how-pulsim-works/`](docs/how-pulsim-works/) for the
chapter-by-chapter walkthrough (18 figures, 11 chapters, MNA basics
through path-based partial refactor with full provenance).

---

## Validation

```bash
# C++ kernel tests (Catch2 binaries per layer)
ctest --test-dir build --output-on-failure
# 498 / 498 tests pass on a fresh checkout

# Python runtime tests
PYTHONPATH=build/python pytest python/tests -v
# 6 / 6 path-aware sweep tests pass

# Benchmark suite (opt-in, not in default ctest)
./build/core/pulsim_benchmarks "[rank1][microbench]"
./build/core/pulsim_benchmarks "[multi_bit][microbench]"
./build/core/pulsim_benchmarks "[parametric][microbench]"
./build/core/pulsim_benchmarks "[ac_sweep][microbench]"
```

Reference CSV traces for the 10 converter showcases live under
`benchmarks/baselines/`. A 1.0-native regression runner that consumes
them is on the roadmap.

---

## Roadmap

| Version | Highlight | Status |
|---|---|---|
| **v1.3.0** | In-house real sparse LU (KLU removed) | ✅ shipped |
| **v1.4.0** | Complex sparse LU + multi-bit + parametric refactor + Python helpers | ✅ shipped (this release) |
| v1.5.0 | Reachability-based sparse triangular solve (close the n ≥ 64 Eigen gap on AC sweep) | planned |
| v1.6.0 | AC-sweep symbolic-reuse: amortise `analyze` across all frequencies | planned |
| v1.7.0 | BTF block-triangular ordering — composes with path-union for an extra ~2× on wide multi-bit transitions | planned |
| v2.0.0 | TBD — first breaking Python API change since v1.0.0 | TBD |

---

## Citation

If you use Pulsim in academic work, please cite:

```bibtex
@software{pulsim,
  author  = {Gili, Luiz Carlos},
  title   = {Pulsim: A piecewise-linear state-space simulator for switched-mode power electronics},
  version = {1.4.0},
  url     = {https://github.com/lgili/Pulsim},
  year    = {2026},
}
```

Machine-readable [`CITATION.cff`](CITATION.cff) is in the repo root.

---

## Contributing

PRs welcome. Workflow:

1. Open an issue describing the change (or pick one from the
   [issue tracker](https://github.com/lgili/Pulsim/issues)).
2. For non-trivial work, draft an OpenSpec proposal under
   `openspec/changes/<name>/` (see existing proposals under
   `openspec/changes/archive/` for the format).
3. Implement + add tests under `core/tests/` (C++) or
   `python/tests/` (Python).
4. Ensure all 498 C++ tests + 6 Python tests pass and
   `openspec validate --strict` is green.
5. Open the PR; CI runs on every push.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide.

---

## License

[MIT](LICENSE) © 2026 Luiz Carlos Gili. Free for academic +
commercial use; please cite when used in academic work.

---

## Documentation deployment

Docs are published by `.github/workflows/docs.yml` using
MkDocs Material + mike:

- **PR**: strict docs build (no deploy)
- **`main`**: deploys to the `dev` channel
- **`vX.Y.Z` tag**: deploys release docs and updates `latest`

In repository settings, set **Pages Source** to **GitHub Actions**.

Live docs: [https://lgili.github.io/Pulsim/](https://lgili.github.io/Pulsim/)
