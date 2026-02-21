# Legacy Inventory Matrix

This matrix tracks legacy and stale surfaces that create ambiguity in Pulsim's
runtime contract. The migration rule is strict:

1. Port required behavior to v1.
2. Validate with tests/parity.
3. Remove legacy path.

## Inventory

| feature | current_path | target_v1_path | status |
|---|---|---|---|
| JSON netlist parser (`NetlistParser`) | `core/legacy/include/pulsim/parser.hpp`, `core/legacy/src/parser.cpp` | `core/src/v1/yaml_parser.cpp`, `python` `YamlParser` bindings | removed_phase_2 |
| Legacy SPICE parser/subcircuit parser | `core/legacy/src/parser/spice_parser.cpp`, `core/legacy/src/parser/subcircuit.cpp` | Python benchmark parity runners + explicit mapped comparator flows | removed_phase_2 |
| Legacy solver stack (`advanced_solver`, old convergence aids) | `core/legacy/src/advanced_solver.cpp`, `core/legacy/src/convergence_aids.cpp` | `core/include/pulsim/v1/solver.hpp`, `core/include/pulsim/v1/high_performance.hpp` | removed_phase_2 |
| Legacy thermal implementation | `core/legacy/src/thermal.cpp` | `core/include/pulsim/v1/thermal.hpp` and coupled simulation path in `v1::Simulator` | removed_phase_2 |
| Legacy FMU exporter placeholders | `core/legacy/src/fmu/fmu_export.cpp` | no direct v1 replacement planned for Python-only scope | removed_phase_2 |
| JSON dependency in top-level build | `CMakeLists.txt` (`FetchContent` nlohmann/json block) | no dependency in supported v1 YAML runtime path | removed_phase_2 |
| Legacy include path in Python build | `python/CMakeLists.txt` (`core/legacy/include`) | remove include, bind only to v1 headers | removed_phase_1 |
| Duplicate bindings translation unit | `python/bindings_v2.cpp` | consolidate on active `python/bindings.cpp` path | removed_phase_1 |
| Planned API tests permanently skipped | `python/tests/test_simulation.py`, `python/tests/test_circuit.py`, `python/tests/test_gui_integration.py`, `python/tests/test_thermal.py`, `python/tests/test_parser.py`, `python/tests/test_grpc_client.py` | implement supported API tests or remove stale planned suites | remove_or_replace |
| JSON benchmark fixtures | `benchmarks/circuits/*.json` | YAML-only benchmark corpus in `benchmarks/circuits/*.yaml` | removed_phase_2 |
| Legacy/stale docs suggesting unsupported user flows | `README.md`, `docs/user-guide.md`, `docs/python/index.rst` | Python-only user docs + migration notes | rewrite_required |

## Exit Criteria Per Row

- `migrate_then_remove` or `remove_after_migration`: equivalent v1 behavior exists and has regression coverage.
- `remove_blocker`: deleted before declaring Python-only build surface complete.
- `remove_or_replace`: no required suite may remain as placeholder skip for supported surface.
- `rewrite_required`: docs no longer advertise unsupported user-facing paths.
