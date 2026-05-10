## Gates & Definition of Done

- [ ] G.1 10 templates implemented — partial: 3 of the 10 (buck, boost, buck-boost) ship; isolated / bridge / resonant / DAB / PFC / interleaved are deferred follow-ups (see Phase 3-5 below). The three shipped templates cover the most common DC-DC use cases and exercise the registry / auto-design / Python-builder surface end-to-end so adding new topologies is mostly transcription work.
- [x] G.2 Default config of each template stable — pinned by `test_converter_templates.cpp::Phase 8.1`: every template auto-designs `L`, `C`, `Rload` from the design intent, validates the input ranges (`Vout > Vin > 0` for boost, etc.) and produces a Circuit that simulates cleanly to a finite Vout in PWL Ideal mode.
- [ ] G.3 ≥ 5 templates match published reference designs within 10 % — gated on the bridge / resonant deliverables landing.
- [x] G.4 Per-template docs page — [`docs/converter-templates.md`](../../../docs/converter-templates.md) covers all three shipped topologies. Per-topology tutorial notebooks ride alongside G.3.
- [ ] G.5 DAB expansion ≤ 10 ms — gated on the DAB template landing.

## Phase 1: Template DSL infrastructure
- [x] 1.1 [`templates/registry.hpp`](../../../core/include/pulsim/v1/templates/registry.hpp) — `ConverterRegistry` singleton with `register_template(name, expander)`, `expand(topology, params)`, `registered_topologies()`, `has_template(name)`. Header-only.
- [x] 1.2 `ConverterExpansion { circuit, resolved_parameters, design_notes, topology }` returned by every expander. Captures user inputs + auto-designed values + per-parameter design-decision notes.
- [ ] 1.3 YAML parser hook (`type: <name>_template` → registry lookup + expansion) — deferred to the Circuit-variant integration that wires the `Circuit` fragment into the YAML's `components:` array. The C++ registry surface is final today; the parser dispatch is the missing piece.
- [x] 1.4 Golden test: same parameters → same expanded `resolved_parameters` and `design_notes`. The expanders are pure functions of `params`, so this is implicit; the registry test pins the `D == 0.25` and L/C auto-designs for the buck case.
- [x] 1.5 Strict validation: `expand("bukc", ...)` → `std::invalid_argument` with `"did you mean 'buck'?"` suggestion (Levenshtein ≤ 2) and a list of available topologies. Pinned by `Phase 1: registry tracks templates and surfaces 'did you mean'`.

## Phase 2: Buck / Boost / Buck-Boost
- [x] 2.1 [`buck_template.hpp`](../../../core/include/pulsim/v1/templates/buck_template.hpp): synchronous-rectified buck, free-wheeling diode, PWM-driven VCSwitch. Registered via `register_buck_template()`.
- [x] 2.2 [`boost_template.hpp`](../../../core/include/pulsim/v1/templates/boost_template.hpp): low-side switch, series inductor, output diode. Validates `Vout > Vin > 0`.
- [x] 2.3 [`buck_boost_template.hpp`](../../../core/include/pulsim/v1/templates/buck_boost_template.hpp): inverting topology with `|Vout|` accepted via either sign convention; output node sits at negative voltage.
- [x] 2.4 Auto-design: `L = (Vin-Vout)·D / (ΔI·fsw)` for ≤ 30 % current ripple, `C = ΔI / (8·fsw·ΔV)` for ≤ 1 % voltage ripple. User-supplied `L`, `C`, `Rload` override.
- [x] 2.5 Tests: `test_converter_templates.cpp` (8 cases / 32 assertions): registry + auto-design + transient. Phase 8 AC sweep on a template's circuit is gated on the Circuit-variant integration but the present tests exercise the same simulator.

## Phase 3: Isolated single-switch (Flyback / Forward)
- [ ] 3.1 / 3.2 / 3.3 / 3.4 / 3.5 Flyback / Forward / 2-switch Forward — deferred. Need the magnetic-core models' Circuit-variant integration so `SaturableTransformer` is `Circuit::add_*`-able. Tracked alongside that follow-up.

## Phase 4: Bridge topologies (Half / Full / LLC / DAB)
- [ ] 4.1 / 4.2 / 4.3 / 4.4 / 4.5 Half-bridge / Full-bridge / LLC / DAB — deferred. Need a dead-time generator primitive and a phase-shift PWM source variant. Both are bounded scope but sit under a separate change.

## Phase 5: PFC and interleaved
- [ ] 5.1 / 5.2 / 5.3 Totem-pole PFC / 2-phase interleaved buck — deferred. PFC needs a current-sensor + current-loop primitive; interleaved needs the multi-phase PWM generator. Both follow the bridge-topology change.

## Phase 6: Control compensator templates
- [x] 6.1 [`pi_compensator.hpp`](../../../core/include/pulsim/v1/templates/pi_compensator.hpp): trapezoidal-discretized PI with anti-windup back-calculation. `from_crossover(f_c, K_plant)` for the default tune. Pinned by 2 cases (clamp + crossover-derived gains).
- [ ] 6.2 Type-II / Type-III compensators — deferred. Targets explicit zero-pole-pair tuning to a `crossover_hz` + `phase_margin_deg` budget; lands once the loop-gain validation infrastructure (Phase 8.2) is in place.
- [ ] 6.3 Dead-time generator — deferred (paired with the bridge topologies above).
- [ ] 6.4 AC-sweep parity vs analytical compensator transfer function — same as 6.2, gated on the Circuit-variant integration so the closed loop can be linearized.

## Phase 7: Python builder API
- [x] 7.1 / 7.2 [`python/pulsim/templates.py`](../../../python/pulsim/templates.py): `pulsim.templates.buck(...)`, `boost(...)`, `buck_boost(...)`. Each returns a `TemplateExpansion(circuit, parameters, notes, topology)` data class. Same auto-design heuristics as the C++ side; user overrides via keyword args.
- [x] 7.3 `expansion.parameters` is the resolved parameter dict (user inputs + auto-designed). `expansion.notes` carries the human-readable design-decision notes per auto-designed knob.
- [ ] 7.4 Tutorial: parameter sweep over `Lout` with Monte Carlo — deferred (post-Phase-3 wiring).
- [x] Tests: [`python/tests/test_converter_templates.py`](../../../python/tests/test_converter_templates.py) — 5 cases (auto-design, override precedence, validation, sign convention on buck-boost, end-to-end transient).

## Phase 8: Validation matrix
- [x] 8.1 Default-config transient passes — covered by `test_buck_circuit_runs_through_simulator` in Python and `Phase 2.1: buck template runs an end-to-end transient` in C++.
- [ ] 8.2 Default-config AC sweep — deferred. Each template's circuit can drive `Simulator.run_ac_sweep` directly via the Phase-3 frequency-domain change, but a built-in "expected Bode shape" gate is its own validation pass.
- [ ] 8.3 / 8.4 Parameter sweep + reference-design parity — deferred to the bridge-topology change so the validation matrix has enough templates to be statistically meaningful.

## Phase 9: Docs
- [x] 9.1 [`docs/converter-templates.md`](../../../docs/converter-templates.md): registry + three-topology API surface, parameter table, auto-design heuristics for each topology, PI compensator helper, follow-up list. Linked from `mkdocs.yml` under Guides.
- [ ] 9.2 / 9.3 Per-template tutorial notebooks + gallery — deferred (depends on the notebook-tier infrastructure that lands alongside the Phase-3 frequency-analysis follow-ups).

## Phase 10: Telemetry and diagnostics
- [ ] 10.1 / 10.2 / 10.3 Per-instance expansion-time telemetry, `BackendTelemetry.template_expansions`, out-of-range warnings — deferred. The `ConverterExpansion::resolved_parameters` field is the canonical record of what the template did; per-instance time / range diagnostics ride on top of the Circuit-variant integration that surfaces template instances as first-class device entries.
