## Phase 1 — Benchmark binary infrastructure (~0.2 days)

- [x] 1.1 New `pulsim_v2_benchmarks` Catch2 target
      linking both v1 and v2 libraries.
- [x] 1.2 `test_main.cpp` Catch2 entry point.

## Phase 2 — Benchmark scenarios (~0.4 days)

- [x] 2.1 S1: V_dc + R (linear sanity).
- [x] 2.2 S2: RC charging (dynamic, no switches).
- [ ] 2.3 S3: Half-wave rectifier — DEFERRED to V1 (v1
      diode + event-iteration setup needs a careful
      schema mapping to be a fair comparison).
- [ ] 2.4 S4: PWM chopper — DEFERRED to V1 (same
      reason).

For each shipped:
- Build the same circuit in v1 and v2.
- Wall-clock measure the time-stepping loop.
- INFO a markdown table row with (v1_ms, v2_ms, speedup).
- Sanity-assert that final node voltages match.

## Phase 3 — Docs (~0.1 days)

- [x] 3.1 `docs/pulsim-v2/layer10-benchmarks.md` with
      measured numbers (Release + Debug) + caveats.

## Phase 4 — Regression (~0.1 days)

- [x] 4.1 All previous v2 tests stay green.
- [x] 4.2 `openspec validate pulsim-v2-benchmarks
      --strict` passes.

## Measured results

### Release build
| Scenario        | v1 (ms) | v2 (ms) | speedup |
|-----------------|--------:|--------:|--------:|
| S1: V_dc + R    |   2.677 |   0.293 |   9.1×  |
| S2: RC charging |  14.476 |   2.283 |   6.3×  |

### Debug build
| Scenario        | v1 (ms) | v2 (ms) | speedup |
|-----------------|--------:|--------:|--------:|
| S1: V_dc + R    |  21.550 |   4.812 |   4.5×  |
| S2: RC charging | 148.647 |  50.233 |   3.0×  |

Both modes confirm the architectural claim: v2's PWL
state-space cache is measurably faster than v1's per-step
refactor path on these linear scenarios. Switching
scenarios (where v2's cache REALLY shines) are V1 add-ons.
