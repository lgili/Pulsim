## Why

Three smaller papercuts surfaced during PulsimGUI's migration to
pulsim 1.4 (PR #9). They don't share enough machinery to justify
separate proposals, but together they remove three common
GUI-integration headaches:

1. **Initial conditions on passives are write-only.** v0's
   `Circuit.add_capacitor(name, n1, n2, C, initial_voltage)` had a
   5th positional argument. v1.4's `CircuitBuilder.add_capacitor`
   is 4-arg only; ICs go via `simulate(initial_state=array)` —
   which requires the caller to flatten a `{branch_name: value}`
   dict into a flat ndarray after computing `num_nodes +
   branch_index_of(name)` offsets. PulsimGUI's compat shim was
   forced to accept the v0 5th-positional argument and warn that
   it was silently dropped
   (`pulsim_v0_compat.py:506-540` on PR #9). The IC information
   has nowhere to land in the builder.

2. **Human node aliases are an out-of-band concept.** The GUI
   maintains a `{gui_wire_id → human_label}` map ("n_dc_plus" →
   "node_42"), but the builder only knows the GUI's wire IDs.
   Looking up "n_dc_plus" requires a two-hop name resolution that
   the GUI has to maintain in parallel. The
   `validate_bridge_losses.py` script silently reports
   `V_bus = 0 V` when an alias points at a wire id no component
   referenced (known-limitation comment near line 175 of that
   script). A first-class alias map on the builder would let GUI
   metadata round-trip through pulsim files without a
   parallel registry.

3. **`should_continue` is on `simulate()` only.** PulsimGUI's
   cancel button works for transient simulations because
   `pulsim.simulate(should_continue=...)` accepts it. But
   `compute_dc_op`, `run_ac_sweep`, and `compute_temperature`
   ignore the GUI's cancel signal, so the GUI's
   `backend_adapter.py` had to add wrapper-level cancel checks
   that can't actually preempt the solver mid-run. Long
   sweeps (Monte Carlo Bode, 50+ frequency points each) hang
   the UI for several seconds after the user clicks Cancel.

## What Changes

- **MODIFIED**: `CircuitBuilder.add_capacitor(name, from, to, C,
  c0=None)` and `CircuitBuilder.add_inductor(name, from, to, L,
  i0=None)` accept optional initial-condition kwargs. When set,
  the builder records the IC; when `simulate(initial_state=None)`
  is later called, the builder synthesises a flat
  `initial_state` ndarray from the recorded ICs (zero for any
  branch / node without an IC).
- **NEW**: `CircuitBuilder.set_initial(device_name, value)` — same
  effect post-hoc, useful when the IC isn't known at `add_*` time
  (e.g., the GUI's "set IC" right-click menu).
- **NEW**: `CircuitBuilder.set_alias(human_name, *, node=None,
  branch=None)` — attaches a secondary human-friendly name that
  resolves through the same lookups as the canonical electrical
  name. Exactly one of `node=` / `branch=` must be set.
- **NEW**: `CircuitBuilder.aliases()` — returns the
  `dict[str, AliasTarget]` map for round-tripping through GUI
  file formats. `AliasTarget = (kind: Literal["node", "branch"],
  name: str)`.
- **MODIFIED**: `pulsim.compute_dc_op`, `pulsim.run_ac_sweep`,
  `pulsim.run_mna_sweep`, and `pulsim.compute_temperature`
  accept the same `should_continue: Callable[[], bool] | None =
  None` kwarg that `simulate` already exposes. The callback is
  invoked at well-defined checkpoints (Newton iterations,
  frequency-sweep points, Foster convolution chunks). When the
  callback returns `False`, the function raises
  `pulsim.Cancelled` (subclass of `RuntimeError`).

## Impact

- **Affected specs**: `python-bindings`, `dc-operating-point`,
  `ac-analysis`
- **Affected code**:
  - `core/include/pulsim/v1/runtime_circuit.hpp` — IC fields on
    capacitor/inductor records; `set_alias` and `aliases` storage.
  - `core/include/pulsim/analysis/dc_op.hpp` — `should_continue`
    callback in the Newton loop.
  - `core/include/pulsim/analysis/mna_sweep.hpp` and
    `analysis/ac_analysis.hpp` — `should_continue` between
    frequency points.
  - `core/include/pulsim/thermal/foster_network.hpp` —
    `should_continue` between Foster convolution chunks.
  - `python/pulsim/__init__.py:simulate` — auto-populate
    `initial_state` from builder ICs when caller passes `None`.
  - `python/pulsim/__init__.py:compute_dc_op`,
    `python/pulsim/ac_analysis.py:run_ac_sweep`,
    `python/pulsim/mna_sweep.py:run_mna_sweep`,
    `python/pulsim/thermal.py:compute_temperature` — accept
    `should_continue=` and propagate to the C++ binding.
  - `python/pulsim/__init__.py` — define
    `class Cancelled(RuntimeError)`.
- **Downstream**: PulsimGUI's `pulsim_v0_compat.py` IC handling
  drops the warning path
  (`_warn_initial_condition`); `backend_adapter.py` can pass
  `should_continue=lambda: not self._cancel_requested` to every
  analysis function (today only `simulate` is wired). The
  validate-bridge-losses V_bus probe can use `builder.aliases()`
  to find the right node without the GUI maintaining a parallel
  alias map.
- **No breaking changes** — every kwarg defaults to `None` /
  preserves prior behavior.
