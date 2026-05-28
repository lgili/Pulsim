# Pulsim — User Guide

Pulsim is a header-only C++23 circuit simulator with a Python frontend, designed for **power electronics and SMPS** workloads. The whole solver lives in a few `.hpp` files; the engine pre-factors every reachable switch configuration into a piecewise-linear state-space cache, then steps through time at near-machine-speed.

If you've used SPICE before, the mental model is similar but the implementation is very different — see [the mental model page](mental-model.md) for the one-page summary.

## When to use Pulsim

- You're building **power-electronics converters** (buck, boost, flyback, full-bridge, 3-φ VSI, LDO, …).
- You want a **scriptable, Python-first** workflow with full access to the solver internals.
- You need to **run thousands of duty-cycle / load-line sweeps** quickly — the kernel's PWL cache makes each switch combo essentially free after the first solve.
- You're OK driving things from a `CircuitBuilder` (Python or C++) or YAML — there is no schematic GUI yet.

## When NOT to use Pulsim

- You need a full SPICE language frontend with `.MODEL` cards — Pulsim has its own device catalogue, not SPICE-compatible.
- You're doing **RF / high-frequency PCB** simulations with distributed elements — Pulsim is lumped-only.
- You need built-in convergence retries that mask physically dubious circuits — the kernel's Newton is principled but unforgiving; expect to think about your circuit.

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
5. [**Gotchas**](gotchas.md) — Newton convergence corner cases and the workarounds that ship in 1.0.

### Post-hoc analysis (v1.5)

- [**Losses & junction temperature**](losses-and-thermal.md) — `device_loss_summary` + `device_thermal_summary`. Reads resistors, inductors (with optional Steinmetz / iGSE core loss), ideal switches (PSIM `E_on` / `E_off`), and switched diodes (`Q_rr` / `E_rr_ref`); pipes the per-device loss through a Foster network for `T_j(t)`.
- [**`@fast_block`**](fast-block.md) — Pulsim's PSIM-`Custom C Block` / PLECS-`C-Script` workalike via Numba LLVM JIT. Write the control law in Python, get native-speed code without a C compiler in the user's PATH.
- [**YAML composite devices + chain wiring**](yaml-chain.md) — `induction_motor` / `hysteretic_inductor` in `circuit:` plus `wire_chain_from_yaml(loaded, chain_spec)` for the control layer (IM mechanical, JA observer, SMO/MRAS sensorless).

## Where the source lives

| Tree | What's inside |
|---|---|
| `core/include/pulsim/` | The entire C++23 solver (header-only) |
| `core/tests/` | Catch2 unit + integration tests |
| `python/pulsim/__init__.py` | Python re-export module + `simulate()` helper |
| `python/bindings.cpp` | pybind11 bindings |
| `examples/*.yaml` | YAML showcases referenced from these tutorials |
| `openspec/specs/pulsim-*` | Authoritative capability specs |
| `openspec/changes/pulsim-*` | Historical change proposals |
