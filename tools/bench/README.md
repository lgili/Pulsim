# pulsim-bench — developer benchmark CLI

Single entry point that unifies the historical collection of scripts
under `benchmarks/`. **Dev-only tool** — independent of the main
`pulsim` distribution, with its own `pyproject.toml` and deps.

## Install

From a Pulsim checkout:

```bash
pip install -e tools/bench
```

This installs the `pulsim-bench` console script and the `pulsim_bench`
Python package. The CLI does not require the Pulsim C++ extension to
*import*, but every subcommand that runs simulations needs it. Build
the extension first (see top-level [`README.md`](../../README.md)) and
make sure `build/python` is on `PYTHONPATH`, or `pip install -e .` the
main package.

Alternatively, without installing:

```bash
cd tools/bench && python -m pulsim_bench --help
```

### Tested setup (verified end-to-end)

```bash
# 1. Build the extension (Release, no tests)
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPULSIM_BUILD_PYTHON=ON \
  -DPULSIM_BUILD_TESTS=OFF
cmake --build build -j

# 2. Create a clean venv with the SAME Python that cmake picked
#    (check `grep _Python3_EXECUTABLE build/CMakeCache.txt`)
PY=/path/to/that/python3
"$PY" -m venv .venv && source .venv/bin/activate
pip install rich typer pyyaml numpy
pip install -e tools/bench

# 3. Run a real benchmark
PYTHONPATH=build/python pulsim-bench run --only rc_step \
    --output-dir /tmp/bench_real

# 4. Re-render later or export HTML
pulsim-bench show /tmp/bench_real --export-html /tmp/bench_real/report.html
```

### Common setup gotchas

- **Python version match.** The compiled `_pulsim.cpython-<ver>-darwin.so`
  is locked to the Python `cmake` chose. Use the same interpreter in
  the venv, or force one with `-DPython3_EXECUTABLE=/path/to/python3`.
- **Existing `pulsim` install shadowing the local build.** A previously
  `pip install -e .`'d pulsim in your global / framework Python can
  shadow `build/python/pulsim` even with `PYTHONPATH=build/python` (the
  `.pth` file wins). A fresh venv avoids this.
- **PEP 668 errors from Homebrew Python.** Same fix: use a venv (or
  `--break-system-packages` if you know what you're doing).
- **`Running cmake --build & --install …` noise on every Python
  startup.** Caused by `scikit-build-core`'s editable rebuild hook from
  the main `pulsim` install. Silence with `SKBUILD_NO_BUILD=1` or
  `pip install -e . --config-settings=editable.rebuild=false`.

## Commands

| Command | Wraps | Purpose |
|---|---|---|
| `pulsim-bench run` | [`benchmarks/benchmark_runner.py`](../../benchmarks/benchmark_runner.py) | Standard benchmark suite (per-scenario + KPIs) |
| `pulsim-bench parity` | [`benchmarks/benchmark_ngspice.py`](../../benchmarks/benchmark_ngspice.py) | Compare Pulsim vs ngspice / LTspice |
| `pulsim-bench stress` | [`benchmarks/stress_suite.py`](../../benchmarks/stress_suite.py) | Tiered stress validation |
| `pulsim-bench local-limit` | [`benchmarks/local_limit_suite.py`](../../benchmarks/local_limit_suite.py) | PC-local stress discovery |

Every subcommand inherits the same visual layer (rich-based header +
live progress bar + colored results table + summary panel) from
[`benchmarks/_console.py`](../../benchmarks/_console.py). All
subcommands honor:

- `--quiet` / `-q` — disable rich UI, emit ASCII + JSON summary only.
- `PULSIM_BENCH_PLAIN=1` env var — same as `--quiet`.
- `NO_COLOR=1` env var — rich respects this natively.

## Examples

```bash
# Standard benchmark suite, default output dir
pulsim-bench run

# Just one benchmark, with the variable-step gate
pulsim-bench run --only rc_step --force-adaptive

# Parity vs ngspice
pulsim-bench parity --backend ngspice

# Parity vs LTspice (executable path required)
pulsim-bench parity --backend ltspice \
    --ltspice-exe "/Applications/LTspice.app/Contents/MacOS/LTspice"

# Stress tier A only
pulsim-bench stress --tier A

# Local-limit suite, fixed-step only, 2x duration
pulsim-bench local-limit --mode fixed --duration-scale 2.0
```

## Environment variables

- `PULSIM_REPO_ROOT` — explicit path to the Pulsim checkout root.
  Useful when `pulsim-bench` is installed system-wide. Default:
  auto-discovered by walking upward from the CLI install location.
- `PULSIM_BENCH_PLAIN` — when `1`, force ASCII fallback (no rich UI).
- `NO_COLOR` / `FORCE_COLOR` — standard env vars, honored by rich.
- `PULSIM_KPI_DEBUG` — when set, KPI extraction failures print full
  tracebacks (legacy flag from `benchmark_runner.py`).

## Relationship to the legacy scripts

The scripts in [`benchmarks/`](../../benchmarks/) keep working
standalone — `pulsim-bench` is **strictly additive**. The Makefile
targets (`make benchmark-converters`, `make benchmark-table`, …) still
invoke them directly. Migration is opt-in: switch to `pulsim-bench`
when you prefer the unified entry point.

## Why a separate package

- **Dev-only dependencies** (`typer`, `rich`, `pyyaml`) don't leak into
  the published `pulsim` wheel.
- **Independent versioning** — bumping the CLI doesn't require a
  Pulsim release.
- **Installable in isolated venvs** — power users can have multiple
  Pulsim checkouts share one `pulsim-bench` install via
  `PULSIM_REPO_ROOT`.
