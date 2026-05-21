## Phase 1 — SimulationOptions extension (~0.1 days)

- [ ] 1.1 Add `Size max_event_iterations = 16` field to
      `SimulationOptions`.
- [ ] 1.2 Update `valid()` if needed (no, the field is purely
      additive and any non-negative value is fine).
- [ ] 1.3 Tests for default value.

## Phase 2 — SimulationResult extension (~0.1 days)

- [ ] 2.1 Add `std::vector<Size> event_iteration_count` parallel
      to `times` / `states`.
- [ ] 2.2 Update `reserve(n)` to also reserve the new vector.
- [ ] 2.3 Tests: default-constructed has empty vector; reserve
      does not change size.

## Phase 3 — Run_transient event-iteration loop (~0.5 days)

- [ ] 3.1 Move the cache.solve + diodes.update_from_state into a
      `do { ... } while (flipped && iters < max)` loop.
- [ ] 3.2 Throw `std::runtime_error` if loop hits max without
      converging.
- [ ] 3.3 Record `event_iteration_count.push_back(iters)` per
      sample.
- [ ] 3.4 Skip the iteration if `has_diodes == false`
      (degenerate case: no events possible).
- [ ] 3.5 Apply to BOTH the dynamic (cache.dt() > 0) and static
      (cache.dt() == 0) paths.

## Phase 4 — Boost converter test (~0.3 days)

- [ ] 4.1 Restore `tests/v2/layer5_v2/test_integration_boost.cpp`
      (deleted in V2). Use V_in=12, D=0.5, L=100µH, C=100µF,
      R_load=20Ω, f_sw=100kHz, dt=100ns.
- [ ] 4.2 Run for 10 ms.
- [ ] 4.3 Verify mean V_out ≈ 24 V within 10 % over the last
      1 ms.
- [ ] 4.4 Verify mean I_L ≈ 2.4 A within 10 %.
- [ ] 4.5 Verify `max(event_iteration_count)` is reasonable
      (≤ 4 expected).
- [ ] 4.6 Verify NO step hit the iteration limit.

## Phase 5 — CMake + regression sweep (~0.1 days)

- [ ] 5.1 Re-add `test_integration_boost.cpp` to the layer5_v2
      CMake target.
- [ ] 5.2 All Layer 0-5 V0/V1/V2 tests stay green.
- [ ] 5.3 v1 `pulsim_tests` stays green.
- [ ] 5.4 `openspec validate pulsim-v2-event-detection
      --strict` passes.

## Phase 6 — Documentation (~0.1 days)

- [ ] 6.1 `docs/pulsim-v2/layer5-v2.1-event-detection.md`
      explaining the iteration approach.
- [ ] 6.2 Update `docs/pulsim-v2/layer5-v2-ideal-diode.md`
      with a note about the V0 chatter limitation now fixed.
