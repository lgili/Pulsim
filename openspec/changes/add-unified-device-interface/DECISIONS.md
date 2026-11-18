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

## Slice 2 — the kinds on one interface, resume

14. **The stateful kinds are one compile-time list.** `pwl::StatefulKinds`
    (`device::TypeList` of the five history classes, fold = stamp order:
    coss, laur, pmsm, tail, sat) is the registry; each class models
    `device::StatefulKind` (`device/kind.hpp`: `init`, `seed_from_dc_op`,
    `commit(x, h, stage)`, `capture_gamma`, `extrapolate_previous`,
    `refresh_derivatives(x, h)`, `to_flat` / `from_flat`,
    `for_each_state`, `kBlock`, `kFlatPerDevice`, `kStoredKind`) and has a
    `stamp_kind` overload (`pwl/device_refresh.hpp`). `static_assert`s:
    every kind models the concept and has a stamp; its `kFlatPerDevice` is
    its block's width; every state block but the linear companion is
    claimed by exactly one kind; every kind sits on a Newton row of
    `kKindCaps`. `StatefulHistories`, the refresh, rollback and the
    snapshot iterate the list; the drivers name no kind (TR-BDF2's five
    `has_X` flags became `!hs.any()`). The per-kind argument conventions
    (Coss bool / BDF2 commit, the saturable inductor's missing `h`, the
    probe-step-free derivative refreshes) moved into each class as
    `commit` / `refresh_derivatives` adapters with the old arithmetic.
15. **Rollback through one flat vector (task 3.2, DECISIONS #7
    resolved).** The scratch slice 1 found is NOT recomputable in general
    (the TR-BDF2 stage values `lambda_gamma`, `ja_gamma`, `i_gamma`,
    `q_m_gamma`, `q_gamma` are captured at the stage point; the Lauritzen /
    tail `k0`, `k1` are written by the stamp during the Newton iterations;
    the PMSM `dlam_old` is the derivative the formula produced, which
    `from_flat` only approximates from `v − R·i`), so it is put in the
    vector: `for_each_state` visits every real an entry carries, snapshot
    reals first, then the scratch, and the rollback point is one
    `std::vector<Real>` (cleared and refilled, so only the run's first
    rollback point allocates). Bit-identical by construction (every real
    round-trips; the JA sub-step count is an int, exact in a double).
    The Coss's `q_gamma_` vector is outside the entries and was not rolled
    back before; it still is not (written by `capture_gamma` before every
    read). **Deviation kept:** the live storage stays in each kind's
    entries; the vector is the rollback and snapshot medium. Moving the
    storage itself buys nothing measurable until the stamps move onto
    `Device::eval` (D5's `Instance::state_offset`).
16. **Steps 5–7 under the bit-identity gate.** The kinds are on the
    stateful-kind interface above, on every engine, bit-identical. Their
    hand stamps were NOT rewritten as `Device::eval` laws: an AD Jacobian
    reassociates the hand-derived sums (Open Question 2: 1e-10 budget vs
    bit identity) and this slice's gate was bit identity at every kind.
    The `Device` contract exists and is tested (step 4, #19); porting a
    kind onto it is the owner's call on OQ2.
17. **Defect (a), exact resume (supersedes #12).** The snapshot did not carry the time
    grid: the fixed engine computes `t_k = t_start + k·dt`, so a resumed
    run's grid started at `t_split` and its sample times differed from the
    continuous run's by ~1e-20 s; a gate edge (`switch_fn(t)`) then fell a
    rounding apart. Nothing else was missing: with the grid continued the
    split run is bit-identical (states, times, final `dev_state`) split
    between edges and exactly on an on- and an off-edge. The snapshot
    carries `grid_origin`, `grid_step`, `grid_dt`; a fixed-step resume with
    the same `dt` whose `t` is `grid_origin + grid_step·dt` to a few ulps
    (not exactly: the check and the loop may contract to FMA differently)
    computes `t_k = grid_origin + (grid_step + k)·dt`; otherwise (another
    `dt`, a TR-BDF2 or legacy snapshot) the grid starts at `t_start` as
    before. Pickle version 4 (version 3 + one field); version 3 and 2.0
    pickles load and resume exactly as before (restarted grid). TR-BDF2 →
    TR-BDF2 resume stays to tolerance: bit identity there needs the step
    controller's state (last `h`, LTE history, edge memory), which the
    spec leaves to `refactor-unified-transient-stepper`.
18. **Defect (b), a TR-BDF2 snapshot on a gate edge (supersedes #13).** Cause: when the run
    ends on an edge, `restart_after_gate_event` cannot commit its
    post-edge probe (no room before `t_end`) and replaces only the
    derivative slots; the snapshot then pairs the pre-edge `x` with
    post-edge derivatives (the Coss slot read −22,881 A: the 22.9 V Coss
    discharging through the closed 1 mΩ switch). TR-BDF2 resumes from that
    pair correctly; the fixed trapezoid, whose own edge step starts from
    pre-edge history, rang on it and aborted at the next edge (Newton at
    2.3e4 A). Only the Coss block matters (swapping in the fixed engine's
    Coss block alone made it run). Not "fixed" by saving pre-refresh
    slots: that would degrade the TR-BDF2 → TR-BDF2 resume, which is
    right. The snapshot carries `post_edge_derivatives` (set when the
    slots were replaced without a commit by a probe reaching within
    `h_min` of the run's last `t`: the traced run replaced them at t_end − 1e-14 and then committed
    a 1e-14 sliver step, which is where the −22,881 A comes from), and the
    fixed engine refuses it by name, pointing at `engine='trbdf2'`.
    Review (2026-11-18): the flag is not cleared by a committed step
    after the refresh. Kept: a commit carries the post-edge slot forward
    whatever its length (the trapezoid's q̇ = 2Δq/h − q̇_old), and the flag
    needs the probe (1e-14–1e-13 s) to reach within `h_min` of the end,
    so the edge was at t_end to within that probe.
19. **Step 4, the `Device` contract** (`device/device.hpp`, unused by the
    engines): `Formula`, `StepCtx` (by value; the only source of `h`),
    `Out<S, R>`, `history_term`, `formula_a`, a dense `stamp_fq` /
    `commit_fq`, `seed_params`. For `add-parameter-sensitivities` the
    parameters are `Params<S>`, a template over the scalar: standard
    layout, trivially copyable, `n_params` reals (`static_assert`ed in the
    concept), with `map(f)` to rebuild it over another scalar; `eval` is
    written once and must compile for (Real params, Real x), (Real params,
    AD x: the Newton Jacobian) and (AD params, Real x: the parameter
    derivative). `Params<Real>` is the plain struct, so the Real path is
    the engine's. The sparse `jac_slot` write and the registry of
    `Device` kinds wait for the first ported kind (#16). `ModeDevice`,
    `SelfCommutated`, `StaticDevice` and deriving `StoredKind` from the
    list are not done (tasks 4.2 / 4.4 partial).
20. **Task 2.3 closed.** The DC-only `resolve_refresh` of the binding
    (`compute_dc_op_with_strategy`, `compute_dc_operating_point`) is now
    the kernel's `pwl::dc_device_refresh(pool, enable)`, same rule (the
    standard chain when the pool has any Newton device and `enable` is
    unset); the binding no longer names `make_combined_diode_mosfet_refresh`.
21. **Defect (c).** The cross-engine test missed a dropped Coss block in
    the TR-BDF2 → fixed direction because the fixed engine moves the lost
    charge through the conducting D1, not through v(Cj) (2e-6 with the
    block zeroed). The seam check now reads i(D1) against the TR-BDF2
    continuous run (the fixed engine's own continuous run rings in i(D1)
    after the Coss turn-off: 8.39 / 6.86 A): intact 2.1e-7, block zeroed
    4.1e-2. A new test zeroes the block in both directions and requires
    the check to see it.
22. **Defect (d), the MNA PMSM retry — diagnosed, not fixed here.**
    Every fixed step of a PMSM-only circuit (200 of 200 at dt = 1 µs and
    100 ns) is retried at dt/2 with the reason "PwlStateSpaceCache:
    numerically singular … current through inductor M1_a has no equation".
    The nominal step takes the cached segment (`cache.lookup`), which
    FACTORISES the linear matrix; the PMSM phase-current rows are stamped
    only by the Newton refresh, so they are empty there and the factor
    fails before Newton runs. The dt/2 retry assembles a segment without
    factorising (`assemble_segment` + `solve_with_newton_b_extra`), so it
    succeeds. Not the refresh or the step context: `pwl/assemble.hpp`
    puts a 1e-12 placeholder on the saturable inductor's branch row and on
    a Coss's nodes (`BranchKind::Nonlinear`), not on a PMSM phase row. The
    fix is one line there (`|| src_kind == StoredKind::PmsmMna` on the
    branch-row placeholder); it moves every PMSM result (each step is
    taken at dt instead of as two dt/2 steps), so it belongs to the
    machines agent's rewrite (told), and the rule is in the "How to add a
    device kind" section.

## Slice 3 — the dual path (2026-11-18)

23. **Dual path (orchestrator's decision, not re-opened).** Porting a
    kind's law onto `Device` reassociates its arithmetic, which the
    bit-identity gate (#16) forbids on the DEFAULT path only. So the hand
    stamps stay the default (`device::DevicePath::Legacy`, bit-identical)
    and every ported kind ALSO runs through its law on the `Device`
    contract behind one explicit option: kernel `SimulationOptions::
    device_path`, `TrBdf2Options::device_path`, the DC operating point's
    `dc_device_refresh(pool, enable, path)`; Python `simulate(...,
    device_path='interface')` (default `'legacy'`). A run is all-legacy
    or all-interface: a circuit holding a Newton kind not yet ported
    refuses `'interface'` by name, listing the unported kinds. Gate
    between the paths: trajectories within 1e-12 of scale on the fixed
    engine at a fixed dt; Jacobian entries a few ulp apart at random
    states; strict reference cells equal in outcome and in error to 3
    significant digits; TR-BDF2 at an rtol-proportional tolerance (its
    accepted step sequence may differ by rounding). Speed within 3 %
    (`perf2.py`). The release flips the default and deletes the hand
    stamps (`benchmarks/DEVICE_PATH_EVIDENCE.md` and its deletion list).
24. **The controller-state snapshot block.** `SolverSnapshot::controller`
    is refactor-unified-transient-stepper's `step::ControllerState`
    (`solver/step/controllers.hpp`, taken verbatim from
    `feat/core-p3-stepper`): a `kind` (None / Fixed / Lte) and a flat
    vector `v` in that controller's layout (Lte: `[h, h_mem_gate0,
    h_mem_gate1, pending_gate_slot, diode_slot, gate_parity,
    h_mem_diode...]`). An EMPTY block means "restart the controller",
    which is what every older snapshot (2.0, v3, v4 with 13 or 14 fields)
    means, so they all load and resume as before. Pickle version 4 gains
    a 15th field `(kind, v)`, written only when the block is filled, so
    an empty block keeps the 14-field version 4 that earlier readers
    load; 13- and 14-field version-4 pickles (and a 15th field of `None`)
    still load; a kind other than 1 or 2 is refused. Python:
    `snapshot.controller_state` → `(kind, values)` or None. Filling it is
    the stepper's (TR-BDF2 saves `LteController` at the end of every run);
    this branch carries and pickles it. A first draft with named fields
    (engine, version, names, values) was dropped to carry the stepper's
    object exactly. Other blocks the coordinator named — per-rate-group
    tick origins (event queue), the gated diode's gate bit (machines) —
    are separate fields for their owners, not this one.
25. **What the flip deletes (prepared, not done).** When the release makes
    `'interface'` the default (and every Newton kind is ported, 15.6), the
    hand stamps go: `make_combined_diode_mosfet_refresh` and the
    `stamp_nonlinear_branches` helper (`pwl/nonlinear_refresh_device.hpp`,
    `nonlinear_refresh_mosfet_level1.hpp`, `nonlinear_refresh_diode.hpp`);
    `refresh_nonlinear_capacitors` (`nonlinear_refresh_nonlinear_capacitor.hpp`);
    `stamp_saturable_inductor` / `refresh_saturable_inductors`
    (`nonlinear_refresh_saturable_inductor.hpp`); `refresh_lauritzen_diodes`,
    `refresh_igbt_tails`, `refresh_pmsm_mna` once their kinds are ported;
    the `stamp_kind` overloads and the legacy branch of
    `build_device_refresh` (`pwl/device_refresh.hpp`); the per-kind
    `commit` / `capture_gamma` arithmetic of the histories
    (`NonlinearCapacitorHistory::update_from_state`, `commit_bdf2_stage2`,
    the non-hysteretic part of `SaturableInductorHistory::update_from_state`),
    keeping `seed_from_dc_op`, `extrapolate_previous`,
    `refresh_derivatives` (which move onto the laws then) and the
    Jiles–Atherton `accept_ja` / `capture_ja`; the Real-only duplicates
    `models::NonlinearCapacitor::charge` / `capacitance` and
    `models::SaturableInductor::flux` / `inductance` (the laws are the one
    body; the model functions become calls of it); `DevicePath` stays one
    release as an accepted no-op. Evidence: `benchmarks/DEVICE_PATH_EVIDENCE.md`.
26. **What the interface path does NOT route through the law yet.** The
    restart helpers of the ported kinds — `seed_from_dc_op`,
    `extrapolate_previous` (SDIRK pseudo-history), `refresh_derivatives`
    (TR-BDF2 post-edge slots) — keep the history's arithmetic, which
    evaluates the same model functions as the law's Real value. Moving them
    is part of the flip (#25). The live storage stays in each history's
    entries, which the flat device-state vector already serialises
    (rollback and snapshot); the law reads and writes it through the
    adapters (`pwl/interface_stamps.hpp`). The Coss's junction law below its
    floor (v/V0 < v_floor) has charge flat in v: the law's Jacobian there is
    0, the hand stamp's is C(floor) — the hand Jacobian is not the
    derivative of its own residual. Newton converges to the same root; no
    gate circuit reaches the floor. The 1e-12 fixed-step gate holds while
    both paths take the same Newton iterations (measured 1.8e-13 and
    1.8e-14 of scale on macOS); a path that stops one iteration apart
    differs by up to the Newton tolerance (~1e-10 of a 12 V scale), so a
    platform failure of that gate is a stopping difference, not a law
    difference — check the iteration counts first.
