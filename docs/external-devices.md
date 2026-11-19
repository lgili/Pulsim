# External devices (C ABI)

An external device is an implicit Newton device whose law comes from a shared
library you build: Pulsim stamps it like a built-in device on the `pwl` and
`trbdf2` engines, in the DC operating point (`start_from_dc_op`,
`compute_dc_operating_point`) and in the kernel AC sweep. Status: stages 1–3
of `add-external-device-abi` (C ABI, checks, Python call, time, DC, AC),
internal unknowns and model state, named outputs, and a YAML type behind an
allow list. Not yet: event functions (zero crossings located by the engine),
inline `code=`, equations-to-C (`pulsim.xdev`), OSDI.

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
must not choose the code it runs. The cache of compiled C blocks is per user
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
  build. Windows locks a loaded DLL, so it cannot change there; a path of
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
