## Phase 1 — Cache extension (~0.4 days)

- [ ] 1.1 Add `bool lazy_mode_ = false` member to
      `PwlStateSpaceCache`.
- [ ] 1.2 Add `build_lazy(Real dt)` method:
      - Clear `segments_`
      - Set `dt_ = dt`
      - Set `lazy_mode_ = true`
- [ ] 1.3 Modify `solve(mask, ...)`:
      - If `segments_.find(mask) == end()` AND `lazy_mode_`,
        build the segment on demand (private helper
        `build_one_segment(mask)`).
      - Else (eager mode + missing): throw as before.
- [ ] 1.4 Extract `build_one_segment(const SwitchStateMask&)`
      as a private helper used by BOTH `build()` (eager loop)
      and `solve()` (lazy on-demand).
- [ ] 1.5 Add `num_built_segments() const noexcept` accessor.

## Phase 2 — Test: lazy build on boost (~0.3 days)

- [ ] 2.1 New file
      `tests/v2/layer4_v1/test_lazy_cache.cpp`.
- [ ] 2.2 Build the chopper-PWM circuit (1 switch, 2
      segments).
- [ ] 2.3 Call `build_lazy(dt)`. Verify
      `num_built_segments() == 0`.
- [ ] 2.4 Call solve with mask=OFF. Verify count == 1.
- [ ] 2.5 Call solve with mask=ON. Verify count == 2.
- [ ] 2.6 Call solve again with mask=OFF. Verify count
      stays at 2.

## Phase 3 — Regression + docs (~0.15 days)

- [ ] 3.1 All previous tests stay green (eager build is
      default).
- [ ] 3.2 `openspec validate pulsim-v2-lazy-cache-build
      --strict` passes.
- [ ] 3.3 `docs/pulsim-v2/layer4-v6-lazy-cache.md`.
