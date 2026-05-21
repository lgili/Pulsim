# Pulsim v2 — User Guide

Pulsim v2 is a header-only C++23 circuit simulator with a Python frontend, designed for **power electronics and SMPS** workloads. The whole solver lives in a few `.hpp` files; the engine pre-factors every reachable switch configuration into a piecewise-linear state-space cache, then steps through time at near-machine-speed.

If you've used SPICE before, the mental model is similar but the implementation is very different — see [the mental model page](mental-model.md) for the one-page summary.

## When to use v2

- You're building **power-electronics converters** (buck, boost, flyback, full-bridge, 3-φ VSI, LDO, …).
- You want a **scriptable, Python-first** workflow with full access to the solver internals.
- You need to **run thousands of duty-cycle / load-line sweeps** quickly — v2's PWL cache makes each switch combo essentially free after the first solve.
- You're OK driving things from a `CircuitBuilder` (Python or C++) or YAML — there is no schematic GUI yet.

## When NOT to use v2

- You need a full SPICE language frontend with `.MODEL` cards — v2 has its own device catalogue, not SPICE-compatible.
- You're doing **RF / high-frequency PCB** simulations with distributed elements — v2 is lumped-only.
- You need built-in convergence retries that mask physically dubious circuits — v2's Newton is principled but unforgiving; expect to think about your circuit.

## Reading order

1. [**Getting started**](getting-started.md) — install, run the first transient, see a plot.
2. [**Mental model**](mental-model.md) — what the Graph, DevicePool, PwlStateSpaceCache, and Newton refresh do.
3. **Tutorials** — six walk-throughs from simplest to most involved:
   - [01 — RC charging from a pulse source](tutorials/01-rc-charging.md)
   - [02 — Buck converter (YAML + switch_fn)](tutorials/02-buck-converter.md)
   - [03 — Isolated flyback (transformer + commutation)](tutorials/03-flyback-isolated.md)
   - [04 — Three-phase voltage-source inverter (SPWM)](tutorials/04-3phase-vsi.md)
   - [05 — LDO with op-amp feedback (VCVS + MOSFET)](tutorials/05-ldo-feedback.md)
   - [06 — IGBT boost with realistic gate drive](tutorials/06-igbt-boost.md)
4. [**API reference**](api-reference.md) — Python surface, one page.
5. [**Gotchas**](gotchas.md) — Newton convergence corner cases and the workarounds that ship in v2.

## Where the source lives

| Tree | What's inside |
|---|---|
| `core/include/pulsim/v2/` | The entire C++23 solver (header-only) |
| `core/tests/v2/` | Catch2 unit + integration tests |
| `python/pulsim/v2.py` | Python re-export module + `simulate()` helper |
| `python/bindings_v2_kernel.cpp` | pybind11 bindings |
| `examples/v2/*.yaml` | YAML showcases referenced from these tutorials |
| `openspec/specs/pulsim-v2-*` | Authoritative capability specs |
| `openspec/changes/pulsim-v2-*` | Historical change proposals |
