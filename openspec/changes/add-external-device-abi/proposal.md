## Why

A Pulsim user cannot add an implicit device without rebuilding Pulsim.

Verified on this branch (review `raw/models.md` ranked item 11 and
section (c); synthesis M9):

- **The only user-code hook is explicit.** A C block reads signals,
  runs user code at its own sample time and drives controlled sources
  held zero-order between steps, on the fixed-step engine only
  (`python/pulsim/c_block.py:1-14`, ABI in
  `python/pulsim/cblock_abi.h`). It cannot contribute a residual or a
  Jacobian to the Newton solve, so a user diode, a nonlinear capacitor
  or a core model built with it lags by one sample and is unusable as a
  stiff device.
- **Python cannot supply a Newton refresh.** The binding builds one
  fixed chain, `make_combined_diode_mosfet_refresh()`
  (`python/bindings.cpp:2959-2969`).
- **A new built-in stateful device costs about 60 edit sites in 12–15
  files** and two hand-derived discretisations (the review's count for
  the Lauritzen diode).

`add-unified-device-interface` (Phase 3) reduces a Newton device to
`f(x, t) + d/dt q(x) = 0` with optional state, events and outputs, and
names this change as the consumer of that signature. Once a device is
those two functions and their Jacobians, they can come from a shared
library.

Other tools (general knowledge, not checked against current
documentation): SPICE-class simulators load compiled Verilog-A — ngspice
through OpenVAF and the OSDI interface, Xyce through its own compiler;
Simscape has a component language; PLECS and PSIM offer C-script and DLL
blocks, which are explicit signal blocks like Pulsim's C block. An
implicit user device in a PWL-cache power-electronics simulator is not a
common offering.

## What Changes

Three stages, each releasable.

1. **C ABI** (`python/pulsim/pulsim_xdev.h`): a versioned descriptor
   with the device name, pin and internal-unknown counts, a parameter
   field table (name, unit, default, bounds), fixed Jacobian sparsity
   patterns, and function pointers:
   - `eval(x, ctx, f, q, df, dq)` — required; residual contributions and
     their Jacobians in the row convention of the device interface;
   - `validate`, `init`, `accept`, `events`, `outputs` — optional.
   - Kernel side: one built-in kind `ExternalDevice` with run-time
     arity, stamped by the same time-formula code as every other Newton
     device.
   - Python: `pulsim.add_external_device(builder, name, nodes, *,
     lib=None, code=None, lang="c", params=None)`; compilation through
     the C block's compile cache, which moves to a per-user directory.
   - YAML type `external_device`; a netlist that loads native code is
     refused unless the caller passes `allow_native_code=True`.
   - Registration checks: ABI version, pattern bounds, finite outputs,
     and the supplied Jacobians against finite differences.
2. **Equations to C** (`pulsim.xdev`): the user declares pins, internal
   unknowns, parameters and the expressions `f` and `q` with SymPy;
   Pulsim differentiates, derives the sparsity pattern, generates C for
   the ABI and compiles it. Non-smooth constructs are refused unless
   explicitly allowed.
3. **OSDI host**: load an `.osdi` library compiled from Verilog-A and
   adapt its resistive and reactive residuals and Jacobians to `f` and
   `q`.

Acceptance for stage 1: a Shockley diode implemented externally matches
the built-in to 1e-12.

- **BREAKING** (minor): the C block's compile cache moves from the
  shared temporary directory to a per-user cache directory; cached
  libraries are rebuilt once.

## Impact

- Affected specs: `device-interface` (the capability created by
  `add-unified-device-interface`; this change adds requirements to it).
- Affected code: new `python/pulsim/pulsim_xdev.h`,
  `core/include/pulsim/device/external_device.hpp`,
  `python/pulsim/{xdev,_native_cache}.py`; `python/pulsim/c_block.py`
  (cache helper moved), `yaml/loader.hpp`, `python/bindings.cpp`,
  stubs, `docs/c-block.md`, new `docs/external-devices.md`.
- Dependencies:
  - `add-unified-device-interface` (Phase 3) — required, as the roadmap
    states. Nothing in-kernel can ship before it: without the f/q
    contract an external device would need its own history class and a
    slot in today's wrapper chain.
  - `add-engine-conformance-matrix` — rows: `external_device` ×
    engine.
  - `add-parameter-sensitivities` — will use a reserved descriptor slot
    for parameter derivatives; not implemented here.
  - `add-c-block` (open) — shares the compile cache; unchanged
    otherwise.
- New optional dependencies: SymPy (stage 2, extra `pulsim[codegen]`);
  a C compiler at run time (already needed for inline C blocks); an
  external Verilog-A compiler for stage 3 (the user's tool; nothing is
  bundled).
