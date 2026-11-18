# Decisions taken while implementing

## Step 1 — the history fold

1. `pwl::StatefulHistories` (`pwl/stateful_histories.hpp`) holds the five
   history objects as named members and maps the per-kind argument
   conventions once: `commit(x, h, TrBdf2Stage)` is the Coss's bool /
   BDF2 commit, the saturable inductor's no-`h` form and the other three's
   `(h, stage)` form. The Coss's TR-BDF2 stage charge (`q_gamma`, a
   driver-local vector before) and its BDF2 commit (`commit_coss`, a
   driver lambda) moved into `NonlinearCapacitorHistory`
   (`capture_gamma`, `commit_bdf2_stage2`) with the same arithmetic, so
   `capture_gamma` and `commit` are uniform over the five.
2. The `has_X` guards around single-history calls are dropped: every
   history method loops over its entries, so an empty history is a no-op.
3. The fold order is coss, laur, pmsm, tail, sat (the fixed engine's
   chain order). Commits and captures touch disjoint state, so the order
   is irrelevant for them.

## Step 2 — kernel-built refresh

4. One `build_device_refresh(base, histories, StepCtx&, sat_increment_form)`
   replaces both five-wrapper chains. The step context is one struct
   (`h`, `stage`) read by every stateful stamp; TR-BDF2's two copies
   (`dev_h/dev_stage` and `coss_h/coss_stage`) are gone, so a probe solve
   cannot set one and forget the other.
5. Stamp order is the fold order. On TR-BDF2 that swaps the PMSM and the
   IGBT tail (the chain had tail before PMSM). Both add into the same
   Jacobian entries only when a PMSM terminal is an IGBT terminal; the
   change can then move a result by reassociation (≤ 1 ulp per entry).
   No golden, harness or test circuit has both kinds.
6. `sat_increment_form` stays a parameter: the fixed engine stamps the
   direct flux residual, TR-BDF2 the closed-form increment (TR-BDF2
   track 4). Unifying them is a numerical change, not a refactor.

## Step 3 — flat device state and snapshot

7. The in-run rollback keeps the entry copies of step 1
   (`StatefulHistories::Snapshot`, assign into preallocated buffers, no
   allocation). The entries carry stage scratch (`lambda_gamma`,
   `ja_gamma`, `i_gamma`, the tail/Lauritzen derivative slots) that the
   flat state deliberately leaves out; rolling back through the flat
   vector would drop that scratch and is not bit-identical by
   construction. The histories' storage moves into the flat vector when
   each kind is ported (steps 5–7), where the scratch becomes workspace.
   **Deviation from task 3.2 ("rollback is one copy").**
8. The run's flat, layout-checked device state is
   `SolverSnapshot::dev_state` described by a `device::DeviceStateLayout`:
   blocks in a fixed order (linear companion, then the five stateful
   kinds), each block the history's own `to_flat` layout, plus
   `dev_layout` and `layout_signature`. No separate `DeviceState`
   struct: until the histories store into the vector (DECISIONS #7) only
   the snapshot holds it. A filled `dev_state` without a `dev_layout` is
   refused at resume.
9. `mask` carries the solver-owned conduction bits (what `diode_on`
   carried). Gate bits are a function of `switch_fn(t)`, not state, and
   are not stored. `diode_on` is a view of `mask`.
10. The seven named fields are views in Python. They stay WRITABLE for
    the compatibility release because `steady_state.py` builds snapshots
    by hand and tests write a field to forge an old layout: writing a view
    turns the snapshot into a legacy payload (named fields), which a
    resume converts per field with the histories' own refusals, exactly
    as before.
11. The cross-engine and split-resume tests use a circuit with the Coss,
    Lauritzen diode, saturable inductor, IGBT tail, PWL diodes and linear
    L / C — not the MNA PMSM: alone, the PMSM takes a dt/2 retry on every
    fixed step at dt = 1 µs, 100 ns and 10 ns (Newton stalls on its branch
    rows against the absolute tolerance; same on the tree before this
    change), so it cannot share a 20 ns grid with the others.
12. The spec scenario "Exact resume … bit-identical" is NOT met, and was
    not before this change: the resumed run's grid starts from t_split,
    its sample times differ from the continuous run's by ~1e-20 s, and a
    gate edge then lands a rounding apart (states equal up to the first
    edge, then within 8e-12 of scale). The test asserts 1e-9 of scale.
    Making it bit-exact is a time-grid question for the unified stepper.
13. Found, not fixed: a TR-BDF2 run that ends exactly ON a gate edge stores
    the pre-edge solve as `x`; the fixed engine resumed from it aborts at
    the next edge (Newton at 1e3 V/A residuals, Coss across the switch).
    Same on the tree before this change. Documented in api-reference.md.
