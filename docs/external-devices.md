# External devices (C ABI)

An external device is an implicit Newton device whose law comes from a shared
library you build: Pulsim stamps it like a built-in device on the `pwl` and
`trbdf2` engines, in the DC operating point (`start_from_dc_op`,
`compute_dc_operating_point`) and in the kernel AC sweep. Status: stages 1–3
of `add-external-device-abi` (C ABI, checks, Python call, time, DC, AC),
internal unknowns and model state, named outputs, a YAML type behind an
allow list, inline `code=` (`pulsim.add_external_device`) and equations to C
(`pulsim.xdev`, SymPy). Not yet: event functions (zero crossings located by
the engine), OSDI.

## Loading a library runs native code

A shared library is arbitrary native code that runs with your privileges.
Pulsim checks the model's *mathematics* (below); it does not sandbox it and
cannot stop a library from crashing the process or reading your files. Load
only libraries you built or trust. Pulsim never loads a library on its own:
`add_external_device(..., lib=path)` loads the path you name, and a YAML
netlist's `external_device` loads only a library the load call lists:

```python
lc = pulsim.load_yaml_file("net.yaml", allow_native_code=["/opt/mymodels"])
```

`allow_native_code` is a list of library files or directories; the netlist's
`lib:` (relative to the YAML file's folder, or to the working directory for
`load_yaml_string`) and every list entry are resolved with symlinks followed
before the comparison, and a library outside the list is refused by name.
Without the list every `external_device` entry is refused, and
`allow_native_code=True` is refused too: a netlist from an untrusted source
must not choose the code it runs. A relative list entry is taken from the
working directory (not the YAML file's folder). A directory entry admits
everything under it, subfolders included: list only directories nobody else
can write (never `/tmp` or a shared download folder). The older
`wire_c_blocks_from_yaml` helper is a separate, explicit call and is not
covered by this list. The cache of compiled C blocks is per user
(mode 0700); a cached library another user owns or can write is rebuilt,
never loaded.

## The contract

- The stable contract is `core/include/pulsim/xdev/pulsim_xdev.h` (C99/C++),
  ABI version 1. The C++ helper `pulsim/xdev/adapter.hpp` compiles into your
  library and turns a law written like a built-in one (a template over the
  scalar, `f` and `q` per row) into the C table; it is not a stable API.
- Rows: pins first (KCL: `f` = current leaving the node into the device,
  `q` = charge), then internal unknowns; each row is `f(x) + dq(x)/dt = 0`.
  The integrator owns `d/dt`.
- `eval` must be pure (no hidden state) and reentrant. State changes only in
  `accept`, into the kernel-owned array of `n_state` reals.
- **Time**: `ctx->t` is the time of the solve (the end of the stage being
  solved; `t_eval` in the DC operating point) whenever
  `ctx->host_features & PULSIM_XDEV_HOST_TIME` — every host since this bit
  was named sets it, and a library built before it reads the same field (the
  bit is new, the field is not). The registration checks evaluate at t = 0:
  a law that is only valid at other times must be valid at 0 as well.
- **DC operating point**: the model's `f(x)` and `∂f/∂x` with the storage
  terms dropped (`dq/dt = 0`), solved by the same Newton and cascade as the
  built-in devices; the model state is the one `init(x = 0)` writes (the
  transient re-seeds it at the DC point). A model with no DC path (a pure
  `q` device between two otherwise floating nodes) makes the DC matrix
  singular and the DC solver says so.
- **AC sweep** (`run_mna_sweep(..., v_in_branch_id=...)`): linearised at the
  DC operating point of the given mask, `G += ∂f/∂x`, `E += ∂q/∂x`, from the
  model's declared Jacobians.
- **Smoothness**: a Newton device law must be smooth to about 1e-12 — a 2L/h
  factor turns a 1e-9 V tolerance into 6e-13 A, and a cancellation-prone
  formula (`coth(x) - 1/x` without its series) once made a law non-monotone
  at 1 pA. The checks below catch kinks and non-monotone regions they sample;
  they cannot certify 1e-12.

## What is checked (refused by name: device, model, property, numbers)

At registration, inside the model's declared probe box: buffers written
within their declared sizes; finite outputs; purity (two calls agree to the
bit); KCL (pin `f` and pin `q` sum to 0); the declared Jacobian against
central differences at 17 points and along a fine line per unknown (also
between the points, down to 1e-15 of the box around 0) and along two diagonals
and four fixed-seed directions through the box centre; no write through the
`const` inputs `x`, `ctx->p`, `ctx->state` (copies compared after every call);
the declared claims
SMOOTH, PASSIVE, RECIPROCAL and monotone pairs; model state written within
`n_state`. During a run: every evaluation is checked for non-finite output
(one comparison), every `accept` for writes past `n_state`;
`debug_jacobian=True` re-checks the Jacobian at every evaluation (tolerance:
1e-5 of the row plus the difference quotient's own rounding, so a 0.1 pF
capacitance is checked, not floored away) and the `const` inputs. A model
that does not claim SMOOTH may declare either one-sided derivative at a
kink (forward and backward Richardson quotients are accepted besides the
central one); a SMOOTH claim is held to the central quotient.

Always on during a run, besides finiteness: a stateful model's committed
state is compared after every evaluation (`n_state` reals; nothing for a
stateless model) and `ctx->p` once per accepted step — a write through
either is refused by name. A write into `x` reaches only the host's copy for
that call (it is gathered again before `accept`). Purity also means "not
depending on the previous call": registration evaluates x_a, x_b, x_a and
requires the first result again, and evaluates the probe points from four
threads at once against the serial results (a probe for a model that shares
scratch between calls, not a proof: a race that does not fire there is not
excluded). Every model must be reentrant; there is no serial-only mode.

## Internal unknowns, model state, outputs

- **Internal unknowns** are rows `n_pins …` (x, f, q); each is an unknown of
  the circuit on a branch named after the device (`name`, `name#1`, …), with
  the 1e-12 branch conductance a built-in inductor row has. The Lauritzen
  diode with its stored charge as one internal unknown
  (`core/tests/xdev/twin_models.cpp`, `lauritzen_twin`) gives the built-in's
  reverse recovery on `pwl` (both device paths) and `trbdf2`.
- **Model state** (`n_state` reals, `init`, `accept`) lives in the kernel's
  flat device-state vector: snapshots carry it, a rejected step restores it
  (`accept` runs for accepted steps only), a resumed run continues it.
- **Named outputs** (appended descriptor block 1, still ABI version 1:
  `n_outputs`, `outputs` = {name, unit}, `outputs_fn(ctx, x, out)` evaluated
  with the committed state). A library built against an older header has no
  outputs and loads unchanged. Add the device with `record=True` and read

  ```python
  res.device_output("M", "i(a)")     # current INTO the device at pin a: f + committed dq/dt
  res.device_output("M", "power")    # sum of pin voltage x pin current
  res.device_output("M", "energy")   # one of the model's named outputs
  ```

  at every stored sample of a `pwl` / `trbdf2` run (traced at each accepted
  step; nothing is traced without `record=True`). `power` and `i(<pin>)` are
  reserved keys; a model naming an output so is refused.
- **Events are reserved, not supported**: `n_events` must be 0. No engine
  locates an external device's zero crossing yet (the locators are
  per-kind: gate edges and switched-diode signals), so a model declaring one
  is refused instead of running without its events.

## Building a model, platforms

- Compile against the installed package: `cc -std=c99 -shared -fPIC
  -I"$(python -c 'import pulsim; print(pulsim.get_include())')" model.c -o
  libmodel.so` (Windows: a DLL exporting `pulsim_xdev_get`; `PULSIM_XDEV_EXPORT`
  is `__declspec(dllexport)` there). The wheel ships only the stable header.
- The header asserts its own layout at compile time (`offsetof` for every
  field group, 64-bit and 32-bit): no `long`, `size_t` or enum field, so
  Linux / macOS (LP64) and Windows (LLP64) share one layout, with no padding.
  A compiler that would lay it out differently fails to compile.
- One loaded copy per path per process. Rebuilding a library while a circuit
  in the same process still uses it is refused by name (its size,
  modification time or bytes changed — the file is hashed at each load, so a
  same-size same-second rebuild is seen too): the loader would otherwise keep
  running the old code. Release those circuits (or restart) to load the new
  build. Replace the file as a linker does (a new file renamed over the
  path): copying over a loaded library in place truncates it, and on Linux
  the process then crashes at its next call into the old image, which no
  host can prevent. Windows locks a loaded DLL, so it cannot change there; a path of
  MAX_PATH (260) characters or more is loaded with the `\\?\` prefix.

## Python

```python
b.add_external_device("L1", ["a", "gnd"], "/path/libmymodels.so", "satind_twin",
                      {"L_0": (1e-3, "H"), "I_sat": 1.0})
```

Parameters by field name; a `(value, unit)` pair must use the declared unit
(no conversion). `components()` lists the device as `external_device` with
its parameters, `model`, `library` (the canonical path loaded) and
`library_hash` (FNV-1a of the library's bytes at load), so a result can be
traced to the code that produced it. Engines: `pwl`, `trbdf2` run it (`device_path='legacy'`
routes it through its law too: it has no hand stamp), the DC operating
point and the kernel AC sweep; `dsed` (the exact engine) and `bdf1` refuse it
with `UnsupportedError`. The exact engine admits no Newton device at all: it
propagates each topology mode in closed form, which a law known only through
`f`, `q` and their Jacobians at sampled points cannot give.
The test models in `core/tests/xdev/` are illustrations, not part models.

`pulsim.add_external_device(builder, name, nodes, *, lib=None, code=None,
lang="c", model=None, params=None, check_jacobian=True, record=False)` does
the same from a library path or from source: `code=` is a whole C (or C++,
`lang="cpp"`) translation unit against `pulsim_xdev.h`, compiled through the
per-user native cache (the C block's: a private directory, mode 0700, a
cached library someone else owns or can write is rebuilt, never loaded).
`model` may be left out when the library holds one model.
`check_jacobian=False` skips the registration checks.

## Equations to C (`pulsim.xdev`)

Needs SymPy (`pip install pulsim[codegen]`) and a C compiler (`$CC`, `cc`,
`gcc` or `clang`); each missing piece is a clear error naming what to
install; a refused law or name is `xdev.XdevError` (a `ValueError`).
Windows: the native cache drives a gcc/clang-style command line
only (no MSVC yet).

```python
import sympy as sp
import pulsim as p
from pulsim import xdev

a, k = xdev.pins("anode", "cathode")
I_S, n, V_T, G_min, i_lim = xdev.params(          # (default, unit[, lo, hi])
    I_S=(1e-12, "A", 1e-30, 1.0), n=(1.0, "", 0.1, 10.0),
    V_T=(0.025852, "V", 1e-3, 1.0), G_min=(1e-12, "S", 0.0, 1.0),
    i_lim=(1e6, "A", 1e-6, 1e12))
vte = n * V_T
v_lim = vte * sp.log(i_lim / I_S + 1)
e_lim = sp.exp(v_lim / vte)
u = a - k
i = sp.Piecewise((I_S * (e_lim - 1) + I_S * e_lim / vte * (u - v_lim), u > v_lim),
                 (I_S * (sp.exp(u / vte) - 1), True)) + G_min * u
dev = xdev.compile("sympy_diode", pins=[a, k], f=[i, -i])   # q defaults to 0

b = p.CircuitBuilder()
b.add_sine_voltage_source("V", "in", "gnd", 0.0, 20.0, 1e3)
b.add_resistor("Rs", "in", "a", 1.0)
dev.add_to(b, "D", "a", "out")           # params by name: dev.add_to(..., I_S=2e-12)
b.add_capacitor("C", "out", "gnd", 10e-6)
b.add_resistor("R", "out", "gnd", 100.0)
res = p.simulate(b, t_end=2e-3, dt=1e-6, device_path="interface")
```

This is the built-in `add_shockley_diode` law written out; the two runs
agree to 1.6e-14 of the voltage scale (macOS arm64; the C uses `expm1`
where the built-in writes `exp − 1`). The full script, with the comparison
and the scans below, is `examples/scripts/run_xdev_sympy_diode.py`.

What `compile` does: `sympy.diff` of every row by every unknown; the
structurally non-zero derivatives are the fixed Jacobian pattern; common
sub-expressions are shared (`cse`); `exp(z) − 1` → `expm1(z)` and
`log(1 + z)` → `log1p(z)`; plain C99 (no VLAs, no extensions, no
`M_PI`, numbers as literals, FMA contraction off under clang so
`c·x0 − c·x1` is 0 at `x0 = x1`), compiled with `-std=c99`; the generated
source is deterministic (the same law gives the same bytes, so the cache
hits) and pure (no mutable statics; the registration's purity and
reentrancy probes run on it). The generated C of the test laws compiles
with `-std=c99 -pedantic -Wall -Wextra -Werror` on clang and gcc 15.

- `xdev.internal("i")`: an internal unknown (one more row in `f` and `q`);
  the saturable inductor is `f = [i, −i, v_from − v_to]`,
  `q = [0, 0, −λ(i)]`.
- `xdev.state("E", init=0)` with `compile(..., update={E: E + xdev.h·G·v²})`:
  a committed state (`init` at the start, `update` after each accepted step;
  the law may read it, it has no Jacobian column). `outputs={"energy": (E,
  "J")}`: named outputs, read with `res.device_output(name, key)` for a
  device added with `record=True`.
- `probe={unknown: (lo, hi)}`: the box the registration checks sample
  (default `(−1, 1)` per unknown).
- Parameters and states are found in the expressions, in declaration
  order; a symbol that is not one of the model's is refused by name.

**Smoothness lint.** `Piecewise`, `Abs`, `sign`, `Max`, `Min`,
`Heaviside`, `floor` (and `ceiling`, `frac`, `Mod`) **of an unknown** are
refused with the offending sub-expression. A function of parameters only
(`Abs(G)·v`, `Piecewise((…, BV > 0), (0, True))`) is no kink in the
unknowns and is accepted. A `Piecewise` is accepted as C¹ — and the model
keeps its SMOOTH claim — when it has two pieces, its condition is one
relational affine in one unknown (one boundary point per line), and the two pieces agree in
value and in gradient on that boundary (proved with `simplify`): the
tangent continuation above is such a case. Anything else (three pieces, an
`And`, a boundary SymPy cannot solve, a proof `simplify` cannot close) is
refused: the proof is conservative. `allow_nonsmooth=True` accepts the law;
the model then does not claim SMOOTH, the non-smooth functions are written
as `Piecewise`, and the Jacobian is the derivative of the branch the value
takes (one-sided at a kink, which the registration check accepts for a
model without the claim; `sign` and `Heaviside` take their right-hand value
at the jump, `frac` and `Mod` go through `floor`). Two kinks at the same point may take opposite
sides there (SymPy orders `Max`'s arguments canonically).

**Monotonicity.** The lint sees constructs, not rounding. A law written
with a cancellation — `coth(u) − 1/u` without its series — is smooth on
paper and non-monotone in doubles near 0. `xdev.scan_monotone(dev, pin, lo,
hi)` evaluates `f[row]` (default: that pin's own row) on a fine grid plus a
geometric run to 1e-15 of the width on each side of 0 and returns the
intervals where it decreases by more than a few ulps of `f`, and the points
where it is not finite: for that Langevin law it reports intervals within
|u| < 7e-8 and the NaN at 0; for the diode above, none. It works on any
library model too (`xdev.load(lib, model)`, no SymPy). (SymPy's C printer
writes `coth` as an `exp` ratio, a second cancellation; `pulsim.xdev`
prints `1/tanh`.)

**Trust.** Generated code is still native code, compiled and loaded into
your process. Names that reach the C (model, pins, parameters, states,
outputs) must match `[A-Za-z][A-Za-z0-9_]*` (≤ 63 characters) and not be C
keywords (nor `near` / `far`); units are ≤ 32 characters of letters,
digits, space and `_ . / * ^ ( ) % + -`; expressions reach the C only
through SymPy's printer from SymPy objects (a string law is refused, never
evaluated: `sympify(strict=True)`; no user text is pasted; unevaluated
`Derivative` / `Subs` and complex values are refused before printing); the
compiler runs from an argument list, never a shell; the cache rules above
apply unchanged.

## YAML

```yaml
- type: external_device
  name: D1
  nodes: [a, k]                 # the model's pin order
  lib: models/libmymodels.so    # relative to this file's folder
  model: my_diode
  params: {I_S: 1.0e-12, V_T: [0.025852, V]}
```

Keys are strict (`type`, `name`, `nodes`, `lib`, `model`, `params`); a
parameter must be one of the model's fields and a `[value, unit]` pair must
use its declared unit. Inline `code:` is not accepted. Loading needs
`allow_native_code=[...]` (above).
