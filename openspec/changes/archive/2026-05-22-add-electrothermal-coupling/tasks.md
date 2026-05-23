## 1. Thermal-domain primitives
- [ ] 1.1 Audit existing `thermal_scope` and `electrical_scope` plumbing; document what's already implemented in `core/`.
- [ ] 1.2 Add (or expose via YAML) a `thermal_node` virtual component carrying a single temperature variable (initial T = 25 °C, heat capacity C in J/K).
- [ ] 1.3 Add `thermal_resistance` between two thermal nodes (or thermal-node-to-ambient), units K/W.
- [ ] 1.4 Add `power_dissipation_source` that takes a target electrical component name and injects ∫V·I·dt as heat into a named thermal node.

## 2. Temperature-dependent device parameters
- [ ] 2.1 Extend `IdealSwitch` / `MOSFET` / `IGBT` to accept an optional `r_on_tcr` (temperature coefficient) and `t_ref_celsius` parameter.
- [ ] 2.2 During Jacobian assembly, look up the device's bonded thermal node (if any) and compute `r_on(T) = r_on_ref · (1 + tcr·(T − T_ref))`.
- [ ] 2.3 Make the binding optional — devices without a thermal node behave identically to today (regression-safe).

## 3. Benchmark library
- [ ] 3.1 Create `electrothermal_buck_steady.yaml`: buck converter + one thermal node per MOSFET + thermal-network-to-ambient. Run to thermal steady-state.
- [ ] 3.2 Create `electrothermal_load_step.yaml`: same buck, but load resistor toggles mid-run; observe T_j transient.
- [ ] 3.3 Create `electrothermal_self_heating_resistor.yaml`: a high-current resistor with thermal binding and TCR; the resistance value changes as temperature rises.
- [ ] 3.4 Generate baselines, validate against analytical thermal RC predictions (within 3 %).

## 4. Documentation + smoke-run
- [ ] 4.1 Document the thermal-coupling model in `docs/ELECTROTHERMAL.md` with a diagram of the bonded electrical + thermal network for the buck example.
- [ ] 4.2 Run the regression dashboard; confirm baselines pass and existing benches don't regress.
