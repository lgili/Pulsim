## Why

PulsimGUI's migration to pulsim 1.4 (PulsimGUI PR #9, landed
2026-05-24 on `feat/ux-cleanup-and-scripts`) surfaced two
recurring access patterns that leak builder internals through
every Python consumer:

1. **State-vector index dance.** To plot `V(vout)`, callers
   cache `vout_idx = builder.node_id_of("vout")` and reach into
   `states[:, vout_idx]`. Branch currents are worse —
   `i_L1 = states[:, builder.graph.num_nodes + branch_idx]`.
   Forgetting the `+ num_nodes` offset is the most common bug:
   the math keeps working, the answer is just wrong.
   `scripts/validate_bridge_losses.py:155` is a real-world
   victim — it silently returned `V_bus = 0 V` because the
   alias-vs-builder lookup landed at index `-1` (the
   `.get(name, -1)` fallback) and the subtraction `s[-1] - s[-1]`
   yielded zero without any error.

2. **Device → switch_idx map.** v1.4 enumerates switching
   branches in `add_*` call order but exposes no inverse. To
   drive `Q1` with `make_pwm_switch_fn`, callers count
   `add_mosfet` calls themselves. PulsimGUI's compat shim had to
   invent `pending_gate_signals`, `switch_indices`, and
   `gate_node_indices` from scratch
   (`pulsim_v0_compat.py:352-418`) just to support this lookup.

Both leak the builder's internal coordinate system into every
script and notebook. They're also the single biggest source of
silent-numerical-wrong-answer bugs: when an index is off, the
simulation runs to completion, the plot looks plausible, and the
bug goes unnoticed until someone cross-checks against analytical
expectations.

## What Changes

- **NEW**: `SimulationResult.v(name, t=None)` — returns the node
  voltage trace for `name` (a node name registered with the
  builder). When `t=None`, returns the full series; when `t` is
  an `int` / slice / boolean mask, returns the corresponding
  subset.
- **NEW**: `SimulationResult.i(name, t=None)` — branch current
  trace. Accepts a branch name matching `add_resistor` /
  `add_capacitor` / `add_inductor` / `add_diode` / `add_mosfet*`
  / `add_*_voltage_source`'s `name=` argument.
- **NEW**: `SimulationResult.power(device_name)` — convenience
  wrapper around `device_loss_summary` returning the total
  conduction-loss power for a single device.
- **NEW**: `CircuitBuilder.switch_index_of(name)` — bit position
  in `SwitchStateMask` for the named switching device. Raises
  `KeyError` if `name` exists but isn't switching (passive,
  source, etc.).
- **NEW**: `CircuitBuilder.branch_index_of(name)` — branch
  offset into the state vector's post-node section. Hides the
  `num_nodes` shift from callers.
- **NEW**: `CircuitBuilder.devices()` — ordered list of
  `DeviceInfo(name, kind, terminals)` tuples for every device
  the builder has accepted, in `add_*` call order. Useful for
  GUI hosts and introspection.
- **MODIFIED**: Unknown-name lookups raise `pulsim.NameNotFoundError`
  (subclass of `KeyError`) with `suggestions: list[str]` (max 3
  via `difflib.get_close_matches`) instead of the bare
  `IndexError('CircuitBuilder::node_id_of: node "X" was never
  registered')` the C++ binding currently emits.

## Impact

- **Affected specs**: `python-bindings`
- **Affected code**:
  - `core/include/pulsim/v1/runtime_circuit.hpp` — add
    `switch_index_of`, `branch_index_of`, `devices()` member
    functions on `RuntimeCircuit` (the kernel class that
    `CircuitBuilder` forwards to).
  - `python/src/pulsim_bindings.cpp` (or wherever `node_id_of`
    is bound) — expose the three new methods.
  - `python/pulsim/__init__.py` — extend `SimulationResult`
    Python wrapper with `v` / `i` / `power` methods (the C++
    `SimulationResult` stays as-is; the Python wrapper holds
    a back-reference to the builder for name resolution).
  - `python/tests/test_named_lookups.py` (new).
- **Downstream**: PulsimGUI's `pulsim_v0_compat.py` collapses by
  ~150 lines (the `pending_gate_signals` / `switch_indices`
  invention disappears).
- **No breaking changes** — purely additive on the public
  surface. Existing scripts that compute indices manually
  continue to work.
