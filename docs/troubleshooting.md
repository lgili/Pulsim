# Troubleshooting

## `ModuleNotFoundError: pulsim._pulsim`

Cause:

- Python cannot find the compiled extension for your interpreter ABI.

Fix:

```bash
cmake -S . -B build -G Ninja -DPULSIM_BUILD_PYTHON=ON
cmake --build build -j
PYTHONPATH=build/python python3 -c "import pulsim; print(pulsim.__version__)"
```

## `ModuleNotFoundError: pulsim.signal_evaluator`

Cause:

- stale build/package path missing companion Python modules.

Fix:

- rebuild and ensure `build/python/pulsim/signal_evaluator.py` exists;
- avoid stale `build/cp*/python` directories from old builds.

## macOS import fails with C++/libc++ symbols

Cause:

- ABI mismatch when building bindings with non-Apple clang toolchain.

Fix:

- use AppleClang for macOS Python builds (project CMake already enforces this path when possible);
- clean and rebuild.

## Tests fail due to wrong module path

Cause:

- environment points to stale build artifacts.

Fix:

```bash
# Verify active module origin
python3 - <<'PY'
import pulsim
print(pulsim.__file__)
PY
```

Ensure it points to either:

- installed package in active venv/site-packages, or
- current `build/python/pulsim` tree.

## Docs build fails (`mkdocs --strict`)

Typical causes:

- broken links after page renaming
- missing dependencies in docs environment

Fix:

```bash
python3 -m pip install -r docs/requirements.txt
mkdocs build --strict
```

## Parity script fails with LTspice backend

Cause:

- LTspice executable path missing or incorrect.

Fix:

```bash
PYTHONPATH=build/python python3 benchmarks/benchmark_ngspice.py \
  --backend ltspice \
  --ltspice-exe "/Applications/LTspice.app/Contents/MacOS/LTspice" \
  --output-dir benchmarks/ltspice_out
```

## Slow or unstable switched-converter runs

Recommendations:

- use variable timestep first;
- inspect fallback and rejection telemetry before tuning manually;
- run benchmark matrix to compare integrators and solver stacks in a controlled way.
