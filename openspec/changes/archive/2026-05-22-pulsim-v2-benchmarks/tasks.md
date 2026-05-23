## Phase 1 — Benchmark binary infrastructure (~0.2 days)

- [x] 1.1 New `pulsim_v2_benchmarks` Catch2 target
      linking both v1 and v2 libraries.
- [x] 1.2 `test_main.cpp` Catch2 entry point.

## Phase 2 — Benchmark scenarios (~0.4 days)

- [x] 2.1 S1: V_dc + R (linear sanity).
- [x] 2.2 S2: RC charging (dynamic, no switches).
- [x] 2.3 S3: Half-wave rectifier (1 switching diode).
- [ ] 2.4 S4: PWM chopper — DEFERRED to V1 (v1's
      `add_switch` needs external `set_switch_state`
      calls between solver steps, which doesn't compose
      cleanly with `run_transient` — a fair comparison
      would require a custom step-by-step run loop. S3
      already proves the architectural point with the
      diode auto-commutation pattern).

## Phase 3 — Docs (~0.1 days)

- [x] 3.1 `docs/pulsim-v2/layer10-benchmarks.md` with
      measured numbers (Release + Debug) + caveats.

## Phase 4 — Regression (~0.1 days)

- [x] 4.1 All previous v2 tests stay green.
- [x] 4.2 `openspec validate pulsim-v2-benchmarks
      --strict` passes.

## Measured results

### Release build (`-O3 -DNDEBUG`)
| Scenario                |  v1 (ms) |  v2 (ms) | speedup |
|-------------------------|---------:|---------:|--------:|
| S1: V_dc + R            |    2.919 |    0.297 |   9.8×  |
| S2: RC charging         |   14.925 |    1.956 |   7.6×  |
| S3: Half-wave rectifier |    3.863 |    0.304 |  12.7×  |

### Debug build (`-O0 -g`)
| Scenario                |  v1 (ms) |  v2 (ms) | speedup |
|-------------------------|---------:|---------:|--------:|
| S1: V_dc + R            |   22.483 |    4.678 |   4.8×  |
| S2: RC charging         |  151.867 |   50.974 |   3.0×  |
| S3: Half-wave rectifier |   49.229 |    7.146 |   6.9×  |

**S3's 12.7× Release speedup is the architectural claim
made concrete**: pre-factored cached segments mean
switch commutations are essentially free for v2, while
v1 refactors at every event. The biggest win is on the
scenario v2 was DESIGNED for.
