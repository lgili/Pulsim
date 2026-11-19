# External devices (C ABI)

An external device is an implicit Newton device whose law comes from a shared
library you build: Pulsim stamps it like a built-in device on the `pwl` and
`trbdf2` engines. Status: stage 1 of `add-external-device-abi` (C ABI, checks,
Python call); no YAML type, no DC operating point yet.

## Loading a library runs native code

A shared library is arbitrary native code that runs with your privileges.
Pulsim checks the model's *mathematics* (below); it does not sandbox it and
cannot stop a library from crashing the process or reading your files. Load
only libraries you built or trust. Pulsim never loads a library on its own:
only `add_external_device(..., lib=path)` loads one, with the path you name.
There is no YAML device type, so a netlist cannot make Pulsim load code.

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
between the points, down to 1e-15 of the box around 0); the declared claims
SMOOTH, PASSIVE, RECIPROCAL and monotone pairs; model state written within
`n_state`. During a run: every evaluation is checked for non-finite output
(one comparison), every `accept` for writes past `n_state`;
`debug_jacobian=True` re-checks the Jacobian at every evaluation.

## Python

```python
b.add_external_device("L1", ["a", "gnd"], "/path/libmymodels.so", "satind_twin",
                      {"L_0": (1e-3, "H"), "I_sat": 1.0})
```

Parameters by field name; a `(value, unit)` pair must use the declared unit
(no conversion). Engines: `pwl`, `trbdf2` run it (`device_path='legacy'`
routes it through its law too: it has no hand stamp); `dsed`, `bdf1`, the AC
sweep and the DC operating point refuse it with `UnsupportedError`.
The test models in `core/tests/xdev/` are illustrations, not part models.
