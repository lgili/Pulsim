## 1. Initial conditions on passives

- [ ] 1.1 Extend `RuntimeCircuit::add_capacitor` (in
      `core/include/pulsim/v1/runtime_circuit.hpp`) with
      `std::optional<Real> c0 = std::nullopt`. Store on the
      per-branch metadata record.
- [ ] 1.2 Extend `RuntimeCircuit::add_inductor` with
      `std::optional<Real> i0 = std::nullopt`.
- [x] 1.3 Add `RuntimeCircuit::set_initial(std::string_view
      device_name, Real value)` that finds the device by name and
      sets its IC. Throws `std::out_of_range` if the name doesn't
      match a capacitor or inductor.
- [ ] 1.4 Add `RuntimeCircuit::initial_state() const ->
      std::vector<Real>` that synthesises the flat initial-state
      vector from recorded ICs (zero elsewhere). Returns
      `std::vector<Real>(num_nodes + num_branches, 0.0)` if no
      ICs are set.
- [ ] 1.5 Pybind: bind `c0=` / `i0=` kwargs to
      `CircuitBuilder.add_capacitor` / `.add_inductor` (default
      `None`); bind `set_initial(device_name, value)`; bind
      `initial_state()` for diagnostic use.
- [x] 1.6 In `python/pulsim/__init__.py:simulate`, when
      `initial_state=None`, call `builder.initial_state()` and
      pass the result. When `initial_state` is explicitly
      provided, use it as-is (caller wins).

## 2. Builder aliases

- [ ] 2.1 (Python-side via _builder_ergonomics) Add `RuntimeCircuit::set_alias(std::string_view human,
      std::optional<std::string_view> node,
      std::optional<std::string_view> branch)` (C++). Validate at
      most one of `node` / `branch` is set, neither is empty,
      and `human` doesn't collide with an existing canonical
      name. Store in `std::unordered_map<std::string,
      AliasTarget> aliases_`.
- [x] 2.2 Add `RuntimeCircuit::aliases() const ->
      std::unordered_map<std::string, AliasTarget>` (returns the
      stored map). `AliasTarget` is a `(kind: AliasKind, name:
      std::string)` pair with `enum class AliasKind { Node,
      Branch }`.
- [ ] 2.3 Make `RuntimeCircuit::node_id_of`,
      `branch_index_of`, and `switch_index_of` consult
      `aliases_` before raising. Order: canonical lookup first,
      then alias resolution.
- [x] 2.4 Pybind: bind `set_alias(human, *, node=None,
      branch=None)`, `aliases()`. Throw a Python
      `ValueError` from `set_alias` when both
      kwargs are set or both are `None`.
- [ ] 2.5 Update `SimulationResult.v` / `.i` in
      `python/pulsim/__init__.py` to leverage the same alias
      resolution (depends on
      `add-python-named-lookups` for the accessor methods, but
      the alias resolution lives in the C++
      `node_id_of` / `branch_index_of` themselves so
      consumers get it for free).

## 3. should_continue everywhere

- [ ] 3.1 Define `using ShouldContinueFn =
      std::function<bool()>` in
      `core/include/pulsim/analysis/cancellation.hpp` (new file).
      Define `class Cancelled : public std::runtime_error`.
- [ ] 3.2 Add `ShouldContinueFn should_continue = nullptr` param
      to `compute_dc_op` in
      `core/include/pulsim/analysis/dc_op.hpp`. Inside the Newton
      loop, after each `solve_step()`, invoke
      `if (should_continue && !should_continue()) throw
      Cancelled("compute_dc_op", iteration);`.
- [ ] 3.3 Same for `run_ac_sweep`
      (`core/include/pulsim/analysis/ac_analysis.hpp`) and
      `run_mna_sweep` (`core/include/pulsim/analysis/mna_sweep.hpp`)
      — call between frequency points / sweep iterations.
- [ ] 3.4 Same for `compute_temperature` in
      `core/include/pulsim/thermal/foster_network.hpp` — call
      every 1000 convolution samples or every 1 % of the trace,
      whichever is more frequent (cost-amortized for short
      traces).
- [x] 3.5 Define `pulsim.Cancelled(RuntimeError)` in
      `python/pulsim/__init__.py`. Pybind translates the C++
      `Cancelled` exception to this Python type with the
      `iteration` / `point_index` / `chunk_index` field exposed
      as an attribute.
- [ ] 3.6 Bind `should_continue=` kwarg on `compute_dc_op`,
      `run_ac_sweep`, `run_mna_sweep`, `compute_temperature` in
      the existing Python wrappers (`__init__.py`,
      `ac_analysis.py`, `mna_sweep.py`, `thermal.py`).

## 4. Tests

- [x] 4.1 `python/tests/test_initial_conditions.py`:
      - RLC step response with `c0=5.0` on `C1`: first
        `result.v("v")[0]` equals 5.0 V within 1 mV.
      - `add_inductor("L1", "a", "b", 1e-3, i0=0.5)` followed by
        `simulate(...)` produces `result.i("L1")[0] == 0.5`
        within 1 mA.
      - Explicit `initial_state=np.zeros(...)` overrides recorded
        ICs (the C1=5.0 example yields v[0]=0).
- [x] 4.2 `python/tests/test_builder_aliases.py`:
      - `builder.set_alias("vout", node="node_42")` lets
        `builder.node_id_of("vout") == builder.node_id_of("node_42")`.
      - Setting both `node=` and `branch=` raises `ValueError`.
      - Setting neither raises `ValueError`.
      - Alias name colliding with an existing canonical name
        raises `ValueError`.
      - `builder.aliases()` returns the registered map.
- [ ] 4.3 `python/tests/test_cancellation.py`:
      - `compute_dc_op(..., should_continue=)` that returns False
        after N calls raises `pulsim.Cancelled` with the right
        iteration count.
      - `run_ac_sweep(..., should_continue=)` cancels between
        frequency points; `point_index` attribute is set on the
        exception.
      - `compute_temperature(..., should_continue=)` cancels mid-
        convolution.
      - `should_continue=None` (default) preserves existing
        behavior — same result, no exception, same timing
        envelope.

## 5. Docs

- [ ] 5.1 Update `docs/tutorials/basics.md` to show
      `add_capacitor("Cout", "vout", "gnd", 220e-6, c0=12.0)` as
      the recommended IC pattern.
- [ ] 5.2 Add or extend `docs/tutorials/gui-integration.md`
      explaining how `set_alias` round-trips GUI human labels
      through pulsim — concrete example with a YAML round-trip.
- [ ] 5.3 Update each long-running entry-point's docstring with
      the `should_continue` kwarg, including a recipe for
      "wrap a `threading.Event` into the callback for GUI
      Cancel buttons".

## 6. Validation

- [x] 6.1 `openspec validate add-python-builder-ergonomics
      --strict` passes.
- [ ] 6.2 Existing pytest suite green (no regressions in any
      analysis entry-point's default behavior).
- [ ] 6.3 PulsimGUI sanity check: pass
      `should_continue=lambda: not cancel_event.is_set()` to
      `compute_dc_op` and confirm the GUI cancel button preempts
      a 50-frequency Bode sweep in under one frequency point
      (typically ≤ 200 ms).
