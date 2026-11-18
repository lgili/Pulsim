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
    The first `DEVICE_PATH_EVIDENCE.md` (2026-11-18): 261 reference cells
    run on both paths, all with the same outcome and error, all with diff
    0 — the reference set's switched circuits use PWL devices, so it
    barely exercises the ported Newton laws; the laws' evidence is
    `test_device_path_equivalence.py`, the path cells of
    `test_engine_conformance.py` and `test_device_laws.cpp`. Its speed
    column was taken at load 15 (A/A noise up to 17 %); the 3 % gate
    (15.7) is open: the saturable-inductor stamp is ~1.4× the hand stamp
    per call (17.8 vs 12.8 ns), 7–11 % on the inductor-only case.

## Slice 4 — speed, the stored-charge kinds, AD, step 9 (2026-11-18)

27. **The speed gate (15.7): where the ns went.** Measured per stamp call
    in isolation (`.scratch/bench_stamp.cpp`: 64 instances, minimum of 41
    pinned repetitions, interleaved; the run-level A/A noise of 17 % does
    not reach it). Three causes, three fixes, none on the default path:
    (a) the AD ran at the ROW count (3 tangents for a saturable inductor)
    although λ(i) and Q(v) are univariate: `laws::univariate` runs such a
    law at one tangent and `ad::chain` spreads it (the seeds are ±1, so
    the derivatives are the same to the bit); (b) the junction Coss took
    two `pow` (value, then capacitance) like the model: the law now takes
    Q and dQ/dv from ONE `pow`, (1+u)·(1+u)^(−m) − 1 (a few ulp of the
    model's (1+u)^(1−m) − 1 against the C0·V0/(1−m) prefactor; both cancel
    alike near v = 0); (c) clang left the fold over the law's pattern
    (five inlined `coeffRef`) out of line, ~4 ns of a 14 ns stamp; the
    scatter is now unrolled at compile time and force-inlined
    (`PULSIM_IFACE_INLINE`, empty on MSVC), as are the per-entry stamps
    and `device::stamp_fq` / `stamp_fq_local` (`PULSIM_DEVICE_INLINE`;
    out of line, the Lauritzen stamp was 2.15× the hand one). A runtime
    loop over the pattern, force-inlined, was 2× slower (not unrolled).
    ns per call, legacy → interface:

    | kind | before | after |
    |---|---|---|
    | saturable inductor (atan) | 13.9 → 18.2 (1.30×) | 13.2 → 11.9 (0.90×) |
    | Coss (junction) | 21.7 → 24.9 (1.14×) | 20.7 → 15.4 (0.74×) |
    | MOSFET L1 | — | 23.4 → 22.3 (0.96×) |
    | Shockley | — | 9.4 → 9.4 (1.00×) |
    | Lauritzen (local q_M) | 16.5 → 35.5 (2.15×) | 17.2 → 20.5 (1.19×) |
    | IGBT tail (local Q) | 23.6 → 37.2 (1.57×) | 22.5 → 24.9 (1.10×) |

    Measured reason for the two over 3 %: the condensation (#28) runs the
    law at AD width 3 / 4 and a Schur step where the hand stamp reads
    K0 + K1·g(v) at width 1 / 3, and the interface stamp still calls the
    hand `begin_step` (the restart helpers read its K0/K1, #26) — ~3 ns per
    instance. Whole runs (`.scratch/perf_path.py`, 5 interleaved rounds,
    min of 7, under a parallel build, A/A noise several %): the static and
    mixed circuits 0.80–0.95×, saturable-inductor-only 1.06×, Coss-only
    1.09× (per call both are faster: the run-level gap is not in the stamp
    and is inside this machine's noise band; to re-measure on a quiet
    machine), the tail + Lauritzen chopper 1.13× (pwl) / 1.07× (trbdf2).
    No timing is asserted anywhere (reported only).
28. **A law with an eliminated internal state (Lauritzen q_M, IGBT tail
    Q): LOCAL rows.** The f/q contract gains `n_local`: rows after the
    internal ones whose unknowns the engines never see. `stamp_fq_local`
    condenses them (static condensation): the local rows and the global
    rows must be affine in the local unknown (`kLocalAffine`), so one
    Newton step from the committed charge solves it to rounding, F_G is
    taken at the solved value and J = J_GG − J_GL·J_LL⁻¹·J_LG. The hand
    stamps' q = K0 + K1·g(v) is exactly that solve; the law states the
    physics (Lauritzen: f_L = A·q_M − q_E/T_M, q_L = q_M; tail:
    f_L = Q/τ − k·i_ss, q_L = Q, branch current −dQ/dt) and the integrator
    keeps the formula. The commit is the contract's internal-row rule
    (q̇ ← −f_L), which the hand BDF2 commit equals up to the local
    residual. The committed charge stays in the history; the K0/K1 the
    restart helpers read (#26: `extrapolate_previous`) are still computed
    by `begin_step` on the interface path, and `capture_gamma` solves the
    local row under the history's `iface_ctx` (the step of the last
    interface stamp — the hand path reads its K0/K1 the same way).
    Shortcut: one local row; a block solve is the upgrade for more.
    Gates: law vs hand F, J, commit and capture within 64 ulp of the
    row's scale at random states, every stage (`test_device_laws.cpp`);
    `chopper_igbt_tail_lauritzen` in `test_device_path_equivalence.py`.
29. **The two reviewer points of #26, closed.** (1) The fixed-step gate
    asserts 1e-12 of scale only when both runs' `newton.iterations_per_
    solve` are equal (then the difference is rounding); otherwise it
    asserts the converged tolerance (10·tol_newton_dx absolute, 1e-8 in
    the conformance cells) — a platform where a convergence test decides
    one iteration apart passes on the contract, not on luck. The relaxed
    branch also requires the counts to differ on at most 1 % of the solves
    (min 3): a wrong Jacobian still converges, more slowly, and would move
    the counts everywhere (reviewer, slice 4). (2) Below its
    floor the junction law keeps the tangent C(floor) (its Q is flat
    there, as the hand stamp's): the closed-form tangent of #27 takes the
    floored u, so J = a·C(floor), never a zero row. Test: "below its floor
    the junction Coss keeps C(floor) on both paths".
30. **Discrete state (gate bit, latch) lives outside the Device contract.**
    The gated diode (PWL IGBT gate-held, thyristor latching) is a MODE
    kind: no Newton law; its gate bit is the user's switch mask and its
    latch the mode selector's state (`models/switched_diode.hpp`), stamped
    by `pwl/assemble.hpp` on both paths alike. So it runs on 'interface'
    unchanged, beside Newton kinds that do go through laws
    (`test_a_gated_diode_runs_on_interface_with_its_gate_bit`: the gate
    switches the device; gate off, the device blocks). A Newton law that
    ever needs a discrete input gets it through its `Aux` (read-only per
    solve, like the Jiles–Atherton base state) and commits it in the
    history, never inside `eval`.
31. **Step 9: readings, and the ONE stepper call.** `pwl/device_readings.hpp`:
    `for_each_stateful_reading(histories, x, fn)` and
    `for_each_static_reading(graph, pool, x, fn)` give, per instance of
    every ported kind, the pin currents (f_pin(x) + committed q̇_pin), the
    absorbed power Σ v·i and the law's f / q on the internal and local rows
    (v and −λ of a saturable inductor, the stored charges) — read-only
    functions of the committed state, no loop edit. What the stepper adds
    when it moves the loops (refactor-unified-transient-stepper), and
    nothing else: in `events/step_hooks.hpp`,
    `using DeviceStateFn = std::function<void(Real t, const Vector& x,
    const pwl::StatefulHistories&)>; void on_device_state(DeviceStateFn);`
    and ONE call `hooks.device_state(t, x, histories)` right AFTER
    `histories.commit(x, h, stage)` of every accepted step (both engines,
    after the TR-BDF2 second stage only; never after a rejected attempt
    or a probe), with `t` the accepted time. Empty hook list: no cost
    beyond the emptiness test. A recorder then calls the two visitors.
32. **Live storage in the flat vector (3.2 / 4.2 / 4.4): specified, not
    moved.** Today each history's entries hold the live reals and the
    one flat vector is their copy (rollback, snapshot). Moving the live
    storage means the entries keep OFFSETS into one `std::vector<Real>`
    owned by `StatefulHistories`, and every history member access (the
    hand stamps' too, which are the default path) reads through it. It
    needs no loop edit — only the five history headers, the adapters and
    `StatefulHistories` — but it rewrites the default path's state access
    in every Newton kind for no number change, which is the flip's job
    (#25 deletes most of that arithmetic). Order at the flip: delete the
    hand stamps, then make the entries offset views (`for_each_state`
    already enumerates exactly the reals and their order), then drop
    `snapshot_into` / `restore` copies for a pointer swap.
33. **AD (add-parameter-sensitivities 4.1–4.3).** Every ported law has
    `kParamNames` (declaration order), and `kLawLayout` asserts
    `Params<ADRealN<n>>` packs like `Params<Real>` (no padding, standard
    layout). `has_param_tangent<D>(aux)` / `require_param_tangent<D>(aux,
    device, param)`: a consumer asks per parameter and gets a refusal that
    names `device.param` for a table or Jiles–Atherton instance, never the
    0 its AD slots read. ∂f/∂p and ∂q/∂p of every ported law are checked
    against a fourth-order central difference within 1e-8 (Lauritzen and
    tail added). 4.3 is ticked: the layouts are asserted at compile time,
    and "bit-identical on the golden hashes" is the golden-21 + stateful-40
    byte compare of every commit (the default path never instantiates
    `Params<Real>` of a law, so nothing there can move); 4.1 / 4.2 stay open for the MNA
    PMSM and the induction machine (#34).
34. **Not ported: the MNA PMSM and the induction machine.** The machines
    branch's incremental stator rows were not pushed (origin
    `feat/core-p5-machines` = 7ff35556, the batch-12 merge), and porting
    today's form would be re-ported when they land; 'interface' keeps
    refusing a circuit with either, by name ("PmsmMna" / "pmsm_mna").
    What the port needs: a law of 9 rows (6 pins + 3 currents) for the
    PMSM and 10 (7 pins + 3) for the IM, f = KCL ±i, −(T − T_load),
    −ω on θ, v − R·i on the current rows, q = −λ(i, θ) there; λ and T
    from the model's Real functions through `ad::chain` (θ and i tangents
    exact, no parameter tangent — `param_tangent` false, refused by name
    per #33) unless the Park/harmonic expressions are templated on the
    scalar; `kPorted<PmsmMnaHistory>`; and the registry's stateful
    coverage test must accept `InductionMachineMna` through
    `PmsmMnaHistory` (its `kStoredKind` is `PmsmMna`). Speed: an AD width
    of 9–10 against a hand-derived stamp needs the per-call bench first.

## Slice 5 — slice 4 verified (2026-11-18)

35. **Slice 4's C++ tests, run; the missing ones added.** Full ctest on
    e9805fb4 gave 935/936: the Lauritzen law test compared the paths in
    ulps of the residual, a small difference of the stored-charge terms
    (3 of 100 trials); 8419f471 states the tolerance in ulps of the carried
    terms (q, q_γ, h·q̇) plus q/T_M (τ_tail) on the rate. It can still fail:
    with the Lauritzen local row's 1/τ dropped and the tail's k·i_ss sign
    flipped, both local-row tests fail (4754 of 10400 assertions). Added:
    (a) `Params<Real>` bit-identical to the model's own `Params` for the four
    static kinds (500 random points each, each instantiation out of line
    through a volatile function pointer; same expression, same scalar type,
    -ffp-contract=off) — memcmp; the AD value within 4 ulp; (b) the Lauritzen
    + IGBT-tail chopper allocates O(1) per step on BOTH paths: 91 operator-new
    at 1000 steps, 93 at 4000 (with 4x the gate edges) on each path — bounded
    at +6, a per-step allocation would add ≥ 3000; (c) every pickle layout
    (2.0 10/11/12 fields, v3 12/13, v4 13/14/15) resumes a Lauritzen +
    IGBT-tail snapshot taken on either path exactly as the snapshot it came
    from. Full ctest 937 (936 + the fixed bound), the law and perf targets on
    Eigen 3.4.0 green.
36. **Reviewer, slice 4 (fresh context, read-only).** No law defect found
    (Lauritzen and tail condensation, commit, capture, Coss floor, increment
    form checked against the hand stamps). Fixed: task 4.3 of
    add-parameter-sensitivities narrowed to what the memcmp test shows (four
    static kinds); the split-equals-continuous resume test now runs on
    'interface' too (the local charges survive a snapshot against the
    UNINTERRUPTED run, bit-for-bit); the `univariate` comment states its ±1/0
    seed condition. Left, stated: (1) `iface_ctx` is not in the rollback /
    snapshot state — capture after a rollback that sits between a stage-1
    stamp and its capture would read the newer step context (no engine order
    does that today; the stepper's rollback must keep stamp → capture
    adjacent, or the field joins `for_each_state`); (2) the step-9 readings
    test does not cover the IGBT tail reading nor `for_each_static_reading`;
    (3) the relaxed branch of the equivalence gate (#29) is not Jacobian
    evidence for a defect confined to a few transients — the C++ law tests
    are.
