# Decisions taken while implementing (parts 1-2 only)

Format: question → option taken → why → how to reverse.

1. **Scope** → parts 1 (byte-budgeted event table) and 2
   (allocation-free refactor) only. Part 3 (geometric step grid and
   factor slots, tasks 4-6) is not started. → Part 3 changes results
   and needs the owner's accuracy sign-off (task 5.6). → Implement
   tasks 4-6 on top of this change.

2. **One budget per table or one shared total (open question 4)** →
   one budget applied to each table, accounted separately, as the
   design's "Byte budget, one knob" decision proposes. → No new setter;
   the event table cannot starve the segment table or the reverse. →
   Sum both tables' bytes against the one value in the eviction loop.

3. **Eager mode** → the event table is bounded in eager and lazy mode
   alike (the segment table never evicts in eager mode). → Event
   entries are created on demand in both modes, so the eager argument
   ("the caller asked for all 2^N factors") does not cover them. →
   Guard the eviction loop with `lazy_`.

4. **Budget 0** → disables event eviction too (unbounded), as it does
   for segments. → Same meaning of the one knob. → None needed.

5. **What an entry stores** → `J` at its current `dt` plus `G` and `C`
   scattered onto `J`'s pattern (`std::vector<Real>`), instead of the
   `G` and `C` matrices. → The in-place combine needs the values on
   `J`'s pattern; keeping the matrices as well would double the
   memory. `approx_bytes()` counts `J` (values, indices, column
   pointers), both value arrays, `b` and the solver's `factor_bytes()`.
   → Store the matrices again and recombine through
   `SegmentAssembler::combine`.

6. **Static (`dt = 0`) entries** → still re-assembled from the static
   template on a refactor (not allocation-free). → TR-BDF2 never solves
   at `dt = 0`; the guarantee in task 3.3 is for a resident entry at a
   new `dt > 0`. → Store the static `J` values too and copy them.

7. **Error message on a failed refactor** → built before the entry is
   erased. → It reads `J`, which now lives in the entry (it was a local
   copy before). → None; required for correctness.

8. **Task 1.3 / 3.5 wall time** → measured with the golden probe's
   `perf` mode, interleaved with the parent commit, best of several
   runs; reported, not asserted. The machine is shared and loaded
   (run-to-run spread ±25 %), so the "≤ 0.7 × on the VSI" acceptance
   is judged on those numbers and recorded in tasks.md as measured.

9. **Task 1.5 (analytic reference errors)** → not recorded. Parts 1-2
   are bit-identical (goldens `np.array_equal`), so every error is
   unchanged; the baseline errors matter for part 3's gate. → Record
   them when part 3 starts.

10. **`docs/solvers.md`** → not changed: it has no text on TR-BDF2's
    factor caching to update. The design text lives in
    `docs/internals/layer4-v7-multi-dt-cache.md`,
    `layer4-pwl-state-space-cache.md` and `docs/performance-tuning.md`.
