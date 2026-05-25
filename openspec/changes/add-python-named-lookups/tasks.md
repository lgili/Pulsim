## 1. CircuitBuilder lookup helpers (C++ + bindings)

- [x] 1.1 Add `BranchId branch_index_of(std::string_view name) const`
      to `RuntimeCircuit` in
      `core/include/pulsim/v1/runtime_circuit.hpp`. Implementation
      walks the existing `branches_` vector and returns the index
      of the entry whose `name == name`. Throw `std::out_of_range`
      with a name + suggestion message when no match.
- [x] 1.2 Add `SwitchIdx switch_index_of(std::string_view name) const`
      — looks up only switching branches (MOSFET / IGBT / diode
      / `add_switch`); raises `std::out_of_range` if `name` is
      a passive or source, with a message that names the device's
      actual kind.
- [x] 1.3 Add `std::vector<DeviceInfo> devices() const` returning
      `DeviceInfo { std::string name; std::string kind;
      std::vector<std::string> terminals; }` in `add_*` call
      order. `kind` is one of `"resistor" / "capacitor" /
      "inductor" / "diode" / "mosfet" / "igbt" / "switch" /
      "voltage_source" / "current_source" / "transformer" / …`.
- [x] 1.4 Expose all three via pybind11 in
      `python/src/pulsim_bindings.cpp` next to the existing
      `node_id_of` binding. `DeviceInfo` is bound as a named
      tuple (`py::class_<DeviceInfo>` with `.def_readonly`
      fields).
- [x] 1.5 Wire the three methods through `CircuitBuilder` (the
      Python-facing class) — they delegate to the underlying
      `RuntimeCircuit`.

## 2. SimulationResult named accessors (Python)

- [x] 2.1 In `python/pulsim/__init__.py`, extend `simulate()` to
      attach the source builder to the returned `SimulationResult`
      as `result._builder` so the accessor methods can resolve
      names without forcing callers to re-pass the builder.
- [x] 2.2 Add `SimulationResult.v(self, name, t=None) -> np.ndarray`
      that returns
      `np.asarray(self.states)[t_slice, self._builder.node_id_of(name)]`
      where `t_slice = slice(None) if t is None else t`. Returns a
      scalar `float64` when `t` is a single int.
- [x] 2.3 Add `SimulationResult.i(self, name, t=None) -> np.ndarray`
      analogously, but with the column index
      `self._builder.graph.num_nodes + self._builder.branch_index_of(name)`.
- [x] 2.4 Add `SimulationResult.power(self, device_name) -> float`
      returning the device's total power loss from the existing
      `device_loss_summary(self, self._builder)` helper. One-liner.
- [x] 2.5 Add `pulsim.NameNotFoundError(KeyError)` exception with
      attributes `name: str`, `kind: Literal["node", "branch",
      "switch"]`, `suggestions: list[str]`. Wrap the bare
      `KeyError` / `IndexError` from the C++ binding at the
      Python boundary (`SimulationResult.v` / `.i` / accessor
      methods) — translate via `difflib.get_close_matches(name,
      candidates, n=3, cutoff=0.6)`.

## 3. Tests

- [x] 3.1 `python/tests/test_named_lookups.py` — build a buck
      circuit (`V1`, `Q1`, `D1`, `L1`, `Cout`, `R_L`) and verify:
      - `np.array_equal(result.v("vout"),
        np.asarray(result.states)[:, builder.node_id_of("vout")])`
      - `np.array_equal(result.i("L1"),
        np.asarray(result.states)[:, builder.graph.num_nodes +
        builder.branch_index_of("L1")])`
      - `builder.switch_index_of("Q1") == 0` (single switching
        device)
      - `builder.devices()` enumerates 6 entries in `add_*` order.
- [x] 3.2 Negative-path: `result.v("typo_node")` raises
      `pulsim.NameNotFoundError` with
      `suggestions == ["vout"]` (the closest match).
- [x] 3.3 Type test: `result.v("vout", t=-1)` returns scalar
      `float`, not 0-d ndarray. Use `isinstance(..., float)` in
      the assertion.
- [x] 3.4 Power accessor: `result.power("Q1")` equals
      `device_loss_summary(result, builder)["Q1"]["P_total"]`
      within machine precision.

## 4. Docs + examples

- [ ] 4.1 Update `docs/tutorials/api-reference.md` to feature
      `result.v(name)` / `result.i(name)` / `result.power(name)`
      as the recommended pattern. Add a "deprecated pattern"
      callout for the raw `result.states[:, idx]` access (still
      supported, just less ergonomic).
- [ ] 4.2 Update `python/scripts/test_cl_buck.py` to use the new
      accessors — it's the canonical closed-loop sample, should
      demonstrate the idiom.
- [ ] 4.3 Add a docstring example to `SimulationResult.v` showing
      the typical buck-converter usage.

## 5. Validation

- [x] 5.1 `openspec validate add-python-named-lookups --strict`
      passes.
- [ ] 5.2 Existing `python/tests/` pytest suite green (no
      regressions in `device_loss_summary`, `node_id_of`, or
      transient solver behavior).
- [ ] 5.3 Build artefacts re-export the new symbols from
      `pulsim.__init__` (`v`, `i`, `power` are methods so no
      module-level exports needed; verify `pulsim.NameNotFoundError`
      is reachable as `pulsim.NameNotFoundError`).
