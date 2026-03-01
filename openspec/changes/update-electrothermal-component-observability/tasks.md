## 1. Kernel Electrothermal Contract
- [ ] 1.1 Add a deterministic per-component electrothermal result surface in v1 simulation results.
- [ ] 1.2 Ensure all non-virtual components are represented, including zero-loss entries.
- [ ] 1.3 Merge loss and thermal data by component name from the same accepted-step/event commit timeline.
- [ ] 1.4 Add runtime guardrails that reject invalid thermal constants before transient stepping.

## 2. YAML Thermal-Port Validation
- [ ] 2.1 Enforce thermal-port capability checks (`component.thermal.enabled=true` only on thermal-capable models).
- [ ] 2.2 Implement strict-mode requirement for explicit `rth` and `cth` when thermal is enabled.
- [ ] 2.3 Keep non-strict compatibility by defaulting missing `rth`/`cth` from global defaults with diagnostics.
- [ ] 2.4 Add deterministic diagnostics for invalid ranges (`rth <= 0`, `cth < 0`, non-finite values).

## 3. Python Surface
- [ ] 3.1 Expose the unified per-component electrothermal telemetry in Python bindings.
- [ ] 3.2 Preserve backward compatibility for `loss_summary` and `thermal_summary`.
- [ ] 3.3 Update Python typing stubs and API documentation.

## 4. Validation & Benchmarking
- [ ] 4.1 Add parser unit tests for thermal-port capability and strict/non-strict parameter behavior.
- [ ] 4.2 Add integration tests that assert per-component losses and temperatures are emitted deterministically.
- [ ] 4.3 Add benchmark gate coverage for component-level electrothermal baselines and tolerance checks.
- [ ] 4.4 Run `openspec validate update-electrothermal-component-observability --strict` and resolve all issues.
