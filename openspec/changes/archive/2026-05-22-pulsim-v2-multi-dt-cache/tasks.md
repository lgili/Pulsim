## Phase 1 — Cache extension (~0.4 days)

- [ ] 1.1 Add private member `mutable
      std::unordered_map<Real, std::unordered_map<
        SwitchStateMask, PwlSegment>> alt_segments_`.
- [ ] 1.2 Add `solve_at(mask, dt, b_extra, x) const`:
      - If dt == dt_, delegate to solve().
      - Else, find-or-build in alt_segments_[dt].
- [ ] 1.3 Add `build_one_segment_at(mask, dt) → PwlSegment`
      private helper that returns a fresh segment for an
      auxiliary dt (without touching segments_).
- [ ] 1.4 Add `num_alt_dt_values()` and
      `num_alt_segments_at(dt)` accessors.

## Phase 2 — Tests (~0.3 days)

- [ ] 2.1 New file
      `tests/v2/layer4_v1/test_multi_dt_cache.cpp`.
- [ ] 2.2 Test: solve_at with dt == cache.dt() matches
      solve() bit-identical.
- [ ] 2.3 Test: solve_at with a NEW dt builds the alt-cache
      entry; subsequent calls reuse.
- [ ] 2.4 Test: solve_at on a cap-containing circuit
      with different dt values gives DIFFERENT numerical
      results (the trap companion's g_eq depends on dt).
- [ ] 2.5 Test: num_alt_dt_values() counts unique dt values
      visited.

## Phase 3 — Regression + docs (~0.15 days)

- [ ] 3.1 All previous tests stay green.
- [ ] 3.2 `openspec validate pulsim-v2-multi-dt-cache --strict`
      passes.
- [ ] 3.3 `docs/pulsim-v2/layer4-v7-multi-dt-cache.md`.
