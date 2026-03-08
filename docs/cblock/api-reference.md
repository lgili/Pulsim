# C-Block Python API Reference

All public symbols are importable from `pulsim.cblock` and re-exported from
`pulsim` directly.

```python
from pulsim.cblock import compile_cblock, CBlockLibrary, PythonCBlock
# or:
from pulsim import compile_cblock, CBlockLibrary, PythonCBlock
```

---

## `detect_compiler`

```python
def detect_compiler() -> str | None
```

Find a suitable C compiler on the current machine.

**Discovery order:**

1. `PULSIM_CC` environment variable (overrides everything).
2. POSIX: tries `cc`, `gcc`, `clang` in order (via `shutil.which`).
3. Windows: tries `cl.exe`, then `gcc`.

Returns the full path to the executable, or `None` if nothing is found.

---

## `compile_cblock`

```python
def compile_cblock(
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    name: str = "cblock",
    extra_cflags: list[str] | None = None,
    compiler: str | None = None,
) -> Path
```

Compile a `.c` source file (or inline C source string) into a shared library.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | `str` \| `Path` | — | Path to a `.c` file, **or** a string containing C source code |
| `output_dir` | `str` \| `Path` \| `None` | `None` | Directory for the output library; defaults to a temporary directory (`tempfile.mkdtemp()`) |
| `name` | `str` | `"cblock"` | Stem of the output filename (`name.so` / `name.dylib` / `name.dll`) |
| `extra_cflags` | `list[str]` \| `None` | `None` | Extra compiler flags appended after the defaults (e.g. `["-lm", "-I/usr/include/mylib"]`) |
| `compiler` | `str` \| `None` | `None` | Explicit compiler path; when `None`, falls back to `PULSIM_CC` env var, then `detect_compiler()` |

**Compiler priority:** explicit `compiler` argument → `PULSIM_CC` env var →
auto-detect (`detect_compiler()`).

**Default flags added automatically:**

- POSIX: `-O2 -shared -fPIC -std=c11 -Wall -Wextra`
- MSVC: `/LD /O2`

**Returns:** `Path` to the compiled shared library.

**Raises:**

- `CBlockCompileError` — compiler executable not found, or process exits non-zero.
  Attributes: `.compiler_path` (str | None), `.source` (str), `.stderr_output` (str).

---

## `CBlockLibrary`

```python
class CBlockLibrary:
    def __init__(
        self,
        lib_path: str | Path,
        n_inputs: int = 1,
        n_outputs: int = 1,
        name: str = "",
    ) -> None
```

Loads a compiled shared library and wraps it in the CBlock interface.

**Constructor behaviour:**

1. Loads the library via `ctypes.CDLL`.
2. Reads `pulsim_cblock_abi_version` and raises `CBlockABIError` if the value
   differs from `PULSIM_CBLOCK_ABI_VERSION`.
3. Resolves `pulsim_cblock_step` (required) and raises `CBlockABIError` if
   the symbol is absent.
4. Resolves `pulsim_cblock_init` and `pulsim_cblock_destroy` (optional).
5. Calls `pulsim_cblock_init` if present.

### Properties

| Property | Type | Description |
|---|---|---|
| `n_inputs` | `int` | Number of scalar input channels |
| `n_outputs` | `int` | Number of scalar output channels |
| `name` | `str` | Block name (from constructor or from library metadata) |

### `step`

```python
def step(self, t: float, dt: float, inputs: Sequence[float]) -> list[float]
```

Call `pulsim_cblock_step` with one simulation timestep.

- `t` — current simulation time [s]
- `dt` — elapsed time since previous accepted step [s]; pass `0.0` at first step
- `inputs` — sequence of `n_inputs` floats (list, tuple, or numpy array)

Returns a `list[float]` of length `n_outputs`.

Raises `CBlockRuntimeError` if the C function returns a nonzero value.

### `reset`

```python
def reset(self) -> None
```

Re-initialise the block state: calls `pulsim_cblock_destroy` (if present) then
`pulsim_cblock_init` (if present). Any internal state accumulated during
simulation is discarded.

### Context manager

```python
with CBlockLibrary(lib_path, n_inputs=1, n_outputs=1) as blk:
    blk.step(0.0, 1e-6, [1.0])
# destroy called, library unloaded
```

`__enter__` returns `self`; `__exit__` calls `pulsim_cblock_destroy` and
unloads the library.

---

## `PythonCBlock`

```python
class PythonCBlock:
    def __init__(
        self,
        fn: Callable[[dict, float, float, list[float]], list[float]],
        n_inputs: int,
        n_outputs: int,
        name: str = "",
    ) -> None
```

Wraps a Python callable so it can be used wherever a `CBlockLibrary` is
expected — no C compiler required.

**`fn` signature:**

```python
def fn(ctx: dict, t: float, dt: float, inputs: list[float]) -> list[float]:
    ...
```

- `ctx` is a plain `dict` shared across calls — store any state you need there.
- Must return a list of exactly `n_outputs` floats.

### Properties

Same as `CBlockLibrary`: `n_inputs`, `n_outputs`, `name`.

### `step`

```python
def step(self, t: float, dt: float, inputs: Sequence[float]) -> list[float]
```

Calls `fn(self._ctx, t, dt, list(inputs))` and returns its result.

### `reset`

```python
def reset(self) -> None
```

Clears the internal `ctx` dict (`self._ctx = {}`).

---

## Exceptions

### `CBlockCompileError`

Raised by `compile_cblock()` when the compilation fails.

| Attribute | Type | Description |
|---|---|---|
| `compiler_path` | `str` \| `None` | Path to the compiler that was used (or attempted) |
| `source` | `str` | Source text or source path that was compiled |
| `stderr_output` | `str` | Full stderr from the compiler process |

### `CBlockABIError`

Raised by `CBlockLibrary.__init__()` when the library's ABI version does not
match, or when a required symbol is missing.

| Attribute | Type | Description |
|---|---|---|
| `expected_version` | `int` | `PULSIM_CBLOCK_ABI_VERSION` at load time |
| `found_version` | `int` \| `None` | Version read from the library; `None` if the version symbol was absent |

### `CBlockRuntimeError`

Raised by `CBlockLibrary.step()` when `pulsim_cblock_step` returns a nonzero value.

| Attribute | Type | Description |
|---|---|---|
| `return_code` | `int` | Value returned by C `step` |
| `t` | `float` | Simulation time at which the error occurred [s] |
| `step_index` | `int` | Step counter (0-based) |

---

## `pulsim.capabilities`

```python
import pulsim
assert pulsim.capabilities["c_block"] is True
```

Module-level dict that advertises optional feature availability.
Downstream code and GUI tools can check `pulsim.capabilities["c_block"]` before
offering C-Block functionality to end users.
