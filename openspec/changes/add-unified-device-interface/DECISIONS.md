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
37. **Batch 13 merged (integration 02bf7506).** Pickle: integration's
    version 4 layout is adopted (fields 0–11, the controllers' marker at 12,
    the grid at 13), plus the step controller's block as an optional 15th
    field (#24); the 13-field version 4 of this branch (grid at 12, no
    marker) still loads. Batch 14a unifies the layout again; this branch
    takes its layout then. The saturable permeance (the nonlinear
    capacitor's `Tanh` law) runs on 'interface': the law takes the model's
    charge expression in the same operation order; the Coss law test gains
    it (stamp, commit, capture, ∂/∂p), the equivalence suite a saturating
    winding-on-core circuit (pwl, trbdf2, DC). Before the port the merged
    law would have fallen through to the junction formula silently. The
    batch made no registry entries for the gated diode or the induction
    machine to replace (kind.hpp is unchanged by it; the gated diode is a
    MODE kind, #30). Default path: golden 21 and stateful 40 identical to
    an integration-based build (p5-machines at 33dbd853 = 02bf7506 +
    machines) except the four PMSM cases that branch changes; against this
    branch's previous baselines, every pwl / dsed case identical and the
    trbdf2 cases moved by integration's TR-BDF2 tracks 7–8. New baselines
    `golden_m13` / `sbit_m13`.
38. **The whole-run gap of #27 (15.7): not in the device path; wall time
    on this machine cannot resolve 3 %.** Two load-independent
    measurements on e9805fb4 (same build, both paths):
    (a) instructions retired per accepted step (`/usr/bin/time -l`, 30 runs
    minus a 0-run baseline per path, `.scratch/instr.sh`):

    | case | legacy | interface | ratio |
    |---|---|---|---|
    | saturable inductor only (pwl) | 16 903 | 17 235 | 1.020 |
    | Coss only (pwl) | 13 900 | 13 659 | 0.983 |
    | IGBT tail + Lauritzen chopper (pwl) | 28 470 | 26 918 | 0.945 |
    | IGBT tail + Lauritzen chopper (trbdf2) | 96 710 | 91 416 | 0.945 |
    | boost MOSFET + Coss + saturable (pwl) | 33 777 | 29 299 | 0.867 |
    | boost MOSFET + Coss + saturable (trbdf2) | 132 431 | 114 144 | 0.862 |

    Same Newton iteration totals on both paths (one case 21 042 vs 21 051).
    (b) `sample` profiles (15 s each): the device code (stamps, laws,
    commits, `exp`/`pow`/`atan`) is 5–14 % of a run's samples on either
    path; the interface share exceeds the legacy one by 1–2 % of the run
    (saturable inductor: the law's commit re-evaluates λ through
    `commit_fq`, 159 vs 75 samples, and the stamp 662 vs 498), which is
    the +2.0 % instruction count of that case. The 1.06–1.13× of #27
    and of an in-process ABBA timing on this machine (load 14–46, eight
    other agents: the same probe gave 1.30× and 0.96× for one case on two
    passes) is scheduling noise: no case retires more than 2 % more
    instructions on 'interface'. 15.7 ticked on that measured reason;
    wall-time confirmation on a quiet machine stays a release-checklist
    item, and no timing is asserted.

## Slice 6 — MOSFET C¹, reviewer fixes (2026-11-18)

39. **The MOSFET law is C¹ through V_DS = 0 on 'interface' (the legacy law
    is not).** The model's forward law f(a, d) = α(a)·K·(1+λd)·[a² −
    β(a−d)(a−d)²] (a = V_OV, d = V_DS, α = β = σ(κ·)) leaves
    f(a, 0) = K·a²·σ(κa)·σ(−κa) at d = 0, so the mirrored reverse branch
    −f(a−d, −d) JUMPS by 2·f(a, 0) at V_DS = 0 — the near-threshold leak,
    peak 0.439·K/κ² at a = 2.4/κ (1.9e-3·K at κ = 15). The interface law
    (`device/laws.hpp` `MosfetLevel1`, no longer the model body) is
    h(a, d) = f(a, d) − f(a, 0)·e^{−κd}, mirrored the same way: h(·, 0) = 0
    and both one-sided derivatives at 0 equal ∂f/∂d(a, 0) + κ·f(a, 0)
    (∂/∂V_GS: 0 on both sides), so it is C¹ for ANY weight w(d) with
    w(0) = 1; e^{−κd} was chosen because it only RAISES the slope
    (f(a, 0) ≥ 0) and dies within a few 1/κ, so off-state leakage at
    |V_DS| ≫ 1/κ is the model's (κd > 1 evaluates the model's own forward
    law minus the term f(a, 0)·e^{−κ|V_DS|}, the whole difference). A
    first version, h = f(a, d) − f(a, 0) everywhere, was C¹ too but removed
    the model's off-state leakage K·a²·σ(κa) at every V_DS (28 µA on the
    boost's soft κ = 4 device at V_GS = 0): the boost's paths parted by
    4.9e-3 of scale. For κd ≤ 1 h is written without the subtraction
    (expm1 of the logistics' difference: no cancellation at small d);
    `ad::expm1` added.
    Tests (`test_device_laws.cpp` `[mosfet]`; three devices: default, the
    boost's K = 0.5 / κ = 4, K = 50 / κ = 15): (a) the law equals the model
    minus the weighted, mirrored leak — value and AD Jacobian — to 1e-9 of
    scale on 400 random points each, threshold band included; (b) i(0) = 0
    exactly; |i(±e) ∓ g₀e| ≤ O(e²) and the jump of every Jacobian column
    falls ≥ 500× per 1000× smaller e (e = 1e-5, 1e-8, 1e-11) on 12
    overdrives across the band; CONTROL: the model's own jump at a = 2.4/κ
    exceeds 0.86·K/κ²; (c) strictly increasing, positive slope and passive
    on |V_DS| ≤ 0.5/κ, log-spaced from 1e-13 V (≈ 1e-16 A on the default
    device), at overdrives 0.5, 2.4, 8, 60, −4, −8 /κ. NOT fixed, both
    paths (release list): the slope at V_DS = 0 is negative for
    a ∈ (≈ −1.3/κ, 0) — the triode polynomial 2·V_OV·V_DS at negative
    overdrive (asserted at a = −1/κ on both laws) — and the same
    σ(κu)·u² makes the triode/saturation corner non-monotone where λ·a² is
    small. A monotone overdrive (softplus², EKV-like) would fix both but
    moves the corner by ~1e-3·K at EVERY overdrive: a law change for the
    release, not this port. Readings (`for_each_static_reading`) report the
    interface law on either path.
    Equivalence (`test_device_path_equivalence.py`, pwl, legacy vs
    interface, of scale): the always-on AC reverse-conduction circuit (new;
    V_DS through zero both ways at V_OV = 9 V) 5.4e-15; the boost with 1 ns
    gate edges and κ = 15 (new control) 1.8e-14 (trbdf2 2.5e-8, gate 1e-5);
    the IGBT / Coss / permeance circuits unchanged (≤ 7.5e-14). The soft boost
    (κ = 4, 0.3 µs ramps) parts BY DESIGN: 2.5e-3 of scale on pwl, 2.1e-4 on
    trbdf2, largest on the first turn-on ramp (V_DS ≈ 0 while a crosses
    threshold); with 1 ns edges but κ = 4 it still parts by 5e-9 after every
    turn-off (the off-state leak at V_DS ≈ 0 is 28 µA there). It is a
    recorded cell (`RECORDED`; asserted visible and below 1e-2 of scale),
    not a gated one; its DC point is gated (V_DS = 12 V). The AC circuit has
    no gate edge to land both TR-BDF2 grids on (4 common times of 2757); its
    TR-BDF2 evidence is the conformance MosfetLevel1 cell (AC, interpolated).
40. **Reviewer (fresh context, read-only) on the three integration merges
    and the Tanh port.** Merges: no defect (c465ea51 resolves as the
    automatic merge; 0a56fec1's layout was superseded by f7547a33, whose
    13/14/15-field readers agree; the test deletions there are covered
    elsewhere). Tanh law: Q bit-identical, but the AD tangent was
    1 − tanh², which cancels in saturation ((C_max/C_min)·eps of J; exactly
    C_min past 19·V0). Fixed: with Real parameters the tangent is the
    model's `sech2_` through `ad::chain`; a permeance at C_max/C_min = 2000
    driven to 40·V0 joins the law test (law 4); with the fix disabled that
    law fails 708 of 59 030 `[laws]` assertions, with it 0.

41. **Evidence regenerated at this slice's code (2026-11-18).**
    (a) Instructions retired per accepted step (`.scratch/instr.sh`, K = 20
    runs minus a 0-run baseline per path, same build; the MOSFET law now
    costs 4 logistics + expm1 near V_DS = 0 and two model calls + exp away
    from it):

    | case | legacy | interface | ratio |
    |---|---|---|---|
    | saturable inductor only (pwl) | 16 827 | 17 306 | 1.028 |
    | Coss only (pwl) | 13 930 | 13 758 | 0.988 |
    | IGBT tail + Lauritzen chopper (pwl) | 28 429 | 27 272 | 0.959 |
    | IGBT tail + Lauritzen chopper (trbdf2) | 93 862 | 85 458 | 0.910 |
    | boost MOSFET + Coss + saturable (pwl) | 33 674 | 30 902 | 0.918 |
    | boost MOSFET + Coss + saturable (trbdf2) | 122 208 | 109 669 | 0.897 |
    | boost, 1 ns edges, κ = 15 (pwl, new) | 34 309 | 31 257 | 0.911 |
    | MOSFET on, AC reverse conduction (pwl, new) | 21 905 | 21 194 | 0.968 |

    Every case but the saturable inductor (below) is below legacy; the boost's ratio rose from 0.867
    (#38) to 0.918 with the C¹ law (its two paths now also run slightly
    different trajectories, #39); the saturable inductor read 1.020 in #38
    on unchanged code, and two repeats today read 1.023 and 1.031 — AT the
    3 % gate's edge, one sample over it; its law did not change here, so
    the saturable-inductor commit (#38: `commit_fq` re-evaluates λ) is the
    open item, not this slice (Coss: 0.988 and
    0.979; boost pwl 0.915, boost trbdf2 0.896, chopper trbdf2 0.939 against
    0.910) — the probe's spread is ~1 % on pwl, ~3 % on the 1298-step
    trbdf2 chopper.
    (b) `benchmarks/DEVICE_PATH_EVIDENCE.md` regenerated
    (`benchmarks/tools/device_path_evidence.py`). The tool's law-circuit
    section (added in e9805fb4) never ran: it put `benchmarks/python/tests`
    on the path (ROOT is `benchmarks/`) and could not import the
    equivalence circuits, and the file was last written at e17fc920, before
    it; fixed (ROOT.parent). Today's file: 277 cells run on both paths
    (12 `law:` rows new), 274 with the same outcome and error to 3
    significant digits, 4 refused by name (the induction machine). The 3
    others, all MOSFET, same outcome: `mosfet_nmos_dc` pwl and trbdf2
    (error 0 → 2.58e-9, paths 2.58e-12 of scale apart) and `ll08_buck_mosfet`
    trbdf2 (1.10e-4 → 1.13e-4, 2.35e-9 apart; its pwl cell 2.35e-21): the
    C¹ law is a different operation order from the model body (not
    bit-identical any more), which moves a Newton stop within its
    tolerance and the adaptive grid; no reference cell reaches the leak.
    `law:` rows: the soft boost 2.49e-3 (pwl) / 2.08e-4 (trbdf2), every
    other ≤ 1.81e-14 (pwl) and ≤ 1.0e-9 (trbdf2). The `speed` column is
    wall time under load (other agents, this slice's own C++ build): not
    evidence; (a) is.

## Slice 7 — the machines on the interface (2026-11-18/19)

42. **The MNA machines on the `Device` contract (15.6; supersedes #34).**
    Merged `origin/feat/core-p5-machines` 0e1f549b first (f9158ffb: the
    flux-map PMSM, the increment-form stator rows). `device/machine_laws.hpp`:
    `PmsmMna` (rows a, b, c, n, ω, θ | i_a, i_b, i_c; constant L(θ) with PM
    harmonics, or the flux map) and `InductionMachine` (a, b, c, n, ω, λ_rα,
    λ_rβ | i_a..i_c). The stator rows' conserved quantity is the FLUX:
    q = −λ(i, θ) / −λ(i, λ_r), f = (v − v_n) − R_s·i, so the row is
    conductance form J = −R·δ − a·∂λ/∂i (a = 2/h, 1/h, c1/h; never h-scaled)
    and the engines pass a·(q − q_n) as the history's closed-form increment
    −(Δλ − lam_off) (BDF2: −dlam_gamma) — the hand row term for term. The
    angular momentum J·ω, the angle θ and the IM rotor fluxes are the charges
    of the builder's linear capacitors on those nodes (the engines integrate
    them like any capacitor); the law injects their rates (−(T − T_load),
    −C_θ·ω, −C_r·g). λ, T, g and their partials are the model's Real
    functions in the hand stamp's operation order, spread over the AD slots by
    `chain_n` (exact on the engines' unit seeds); the law's parameters are
    R_s and T_load (exact ∂/∂p), the flux law sits in `Aux`. Commit through
    `commit_fq`: λ_n ← λ(x) (bitwise the hand commit's), dλ/dt_n ← f(x); the
    reference point and v_n stay the history's bookkeeping; `capture_gamma`
    is the history's increment. `iface::history_covers` lets the machine
    history cover the `InductionMachineMna` row (set_path, the binding).
    `PmsmMna::torque` gained overloads taking L(θ) and λ_pm′, λ_pm″ (one
    `lambda_pm_all` trig pass); the hand path routes through them (same
    values). The adapter computes L(θ_e) once for the law and the increment
    (`Aux::ind`, a cache of the model at that call's θ).
    Gates (`test_device_path_equivalence.py`, machine section; the machine
    tests' synthetic parameter sets): pwl legacy vs interface, same Newton
    counts, every state ≤ 2.5e-14 of scale — salient-PMSM sine start, locked
    rotor (ω held, θ0 = 0.3), saturated flux-map start, IM sine start, IM
    generator at 1.05·ω_sync held, IM six-step inverter start. A FLOATING
    neutral is physically 0 V (no zero-sequence flux) and reads ~1e-11 V of
    solve round-off on both paths: left out of the 1e-12 gate, checked ≤
    1e-9·v_phase on both (the inverter's neutral is a real 200 V state and
    stays gated). The held-speed IM uses J = 1e-3 (J = 1 made the source
    current round at ulp(2J/h·ω) = 9e-10 A, a 2e-11 single-step spike).
    TR-BDF2 (three circuits): end states ≤ 10·rtol. Energy balance of the
    saturated flux-map start: residual ≤ 2e-8 of the input on 'interface',
    equal to legacy's to 1e-3 of itself. C++ (`test_device_laws.cpp`): the
    laws stamp to 32 ulp (F) / 16 ulp (J), commit bitwise and capture
    identically against `refresh_pmsm_mna` for all three machines × three
    formulas (grounded star point included); the stator block equals −R·δ −
    a·∂λ/∂i (central difference); the whole x-Jacobian against central
    differences of the law's Real eval (independent of the hand partials);
    ∂/∂(R_s, T_load) by AD. Conformance: the PmsmMna / InductionMachineMna /
    harmonics / flux-map device-path cells run (none refused).
    Speed (instructions retired per accepted step, `.scratch/instr.sh`, 5×
    longer runs, K = 6): PMSM sine start 1.016, locked rotor 1.009,
    flux-map start 1.001–1.003; induction machine 1.04 (inverter) and 1.084–
    1.086 (sine start, held generator) — OVER the 3 % gate. Microbench: the
    IM stamp 131 → 227 ns, of which the law's eval at AD width 10 is 104 ns
    (the hand stamp writes ~55 known coefficients; the law propagates 10
    tangents through every operation); computing J on `kPattern` entries
    only changed nothing (reverted). Lever, not taken: evaluate at the 6
    inputs the IM's nonlinear terms read (i, λ_r, ω) and spread with the
    seed map known at compile time. Recorded for the release; the default
    path is untouched.
    Reviewer (fresh context, read-only): no law defect; fixed — test gaps
    (BDF2 commit, capture, grounded star point, independent FD Jacobian),
    the neutral's explanation, a hard-coded R_s in the energy test. Open:
    `run_transient.hpp:128` (`seed_held_machine_capacitors`) still calls the
    hand `refresh_pmsm_mna` on both paths (same values; loop freeze — the
    stepper track ports it before the hand stamps go); the "unported kind
    refuses by name" test lost its fixture (every Newton kind is ported now;
    the refusal stays covered by `set_path` code, not by a test).
43. **Readings describe the law that was stamped (item 4).**
    `for_each_static_reading(graph, pool, x, path, fn)` takes the run's
    `DevicePath`: a legacy run reads the MOSFET model body
    (`laws::MosfetLevel1Model`, leak included — what
    `refresh_mosfets_level1` stamps), an interface run the C¹ law; the other
    static kinds are one body on both. Test: at the leak's peak (a = 2.4/κ,
    V_DS = 0.1 mV) the legacy reading equals `models::MosfetLevel1::current`
    bitwise, the interface reading the law bitwise, and they differ by
    > 0.4·K/κ² (control). Also corrected here: #39's claim that the slope at
    V_DS = 0 is negative "on both paths" for a ∈ (−1.3/κ, 0) — on the law it
    is Kσa·[κ(1−σ²)a + 2σ] (λ aside), negative only for a ∈ (≈ −0.75/κ, 0)
    (+0.105·K/κ at a = −1/κ); the test asserted it at −1/κ and failed on all
    three devices (never run after #39). Now asserted at −0.5/κ (law) and
    −1/κ (model). (Superseded by #44: one body again, no `path`.)

## Slice 8 — one MOSFET law on both paths (2026-11-19)

44. **The C¹ correction moves into the model; both paths run it.** Two
    corrections existed: (a) this branch's interface law (#39),
    h = f − f(a, 0)·e^{−κ|V_DS|} mirrored, written with expm1 for κd ≤ 1;
    (b) `origin/feat/core-p5-passives-mosfet` 8aa40070, legacy model only:
    the V_DS < 0 branch adds `2·f(V_OV − V_DS/2, 0)`. Merged (b) with
    `--no-ff`, then measured both against the raw mirror
    (`.scratch/mosfet_laws_sweep.py`, K = 1.43, V_T = 3, λ = 0.01, κ = 15,
    V_OV ∈ [−10/κ, 10/κ] in 0.01/κ steps, |V_DS| ≤ 2 V on a 1 mV grid):

    | law | max abs i(0) | jump at 0 | max abs ∂i/∂V_DS(0⁺) − ∂i/∂V_DS(0⁻) | min i·v | ∂i/∂V_DS(0) < 0 for κ·V_OV in | min ∂i/∂V_DS |
    |---|---|---|---|---|---|---|
    | raw mirror | 2.79e-3 A | 5.6e-3 A | 2.9e-2 S | −2.91e-4 W | (jumps) | −4.43e-2 S |
    | (b) | 2.79e-3 A | 3.8e-13 A | 1.0e-6 S (FD noise) | −2.26e-3 W | [−2.20, 0] | −4.83e-2 S |
    | (a) | 0 | 3.8e-13 A | 1.1e-6 S (FD noise) | −2.91e-4 W | [−0.72, 0] | −3.95e-2 S |

    Same sweep at the soft boost's K = 0.5, κ = 4, λ = 0: max abs i(0)
    13.7 mA (raw, b) vs 0 (a); min i·v −5.35 mW (raw), −41.7 mW (b),
    −5.36 mW (a); slope-at-zero window κ·V_OV ∈ [−2.21, 0] (b) vs
    [−0.72, 0] (a).
    Both are C¹. (b) is not passive at V_DS = 0: i(0) = L(a), up to
    0.439·K/κ² (2.8 mA here), and its added term makes the worst i·v 7.8×
    the raw law's (−2.26 mW at κ·V_OV = −6.3, κ·V_DS = −6.5). (a) has
    i(0) = 0 exactly, a third of (b)'s slope-at-zero window, and its worst
    i·v is the raw forward law's own (the triode polynomial past
    saturation onset, κ·V_OV = −3.0, κ·V_DS = −3.2, mirrored), not one it
    adds. **Chosen: (a)**, as `MosfetLevel1::forward_c1_` in
    `models/mosfet_level1.hpp`, both quadrants. One addition keeps the model's
    bits where the correction is below rounding: when the subtracted term is
    ≤ 2⁻⁵⁴·|f| (≤ half an ulp) it returns f itself, so a fully-on channel
    (κ·V_OV ≳ 37) and any |V_DS| ≫ 1/κ are the raw forward law to the bit
    (and the third quadrant its exact mirror, as (b) was). `laws::MosfetLevel1`
    is the model body again (`CurrentLaw`); `MosfetLevel1Model` and the
    readings' `path` argument (#43) are removed — a reading is the law both
    paths stamp. The soft boost (`boost_mosfet_coss_saturable`, RECORDED since
    #39) is gated at 1e-12 again; `test_the_soft_boost_parts_by_the_mosfet_leak_only`
    is deleted (nothing parts). Not removed (release list, unchanged in
    kind): the negative slope near threshold and past saturation onset —
    the triode polynomial at small overdrive, in the forward law both
    corrections share; it is not free to remove (it changes the law at
    every V_DS, not only near 0).
    Legacy numbers that moved (default path, against (b) as the baseline):
    golden compare vs `golden_m13` — every case byte-identical except
    `mosl1_trbdf2`, whose golden is the pre-(b) abort at t = 0 (it runs
    under (b) and here). Against a (b) build of this tree (header swapped,
    `.scratch/mosl1_cmp.py`): `mosl1_pwl` byte-identical; `mosl1_trbdf2`
    3175 accepted points vs (b)'s 3191, states at the 118 common times
    within 4.8e-7 of each column's range, end state within 8.4e-10. Stateful
    bit-identity (`sbit_m14`) unchanged. Reference cells: one moved,
    `switching/basic/mosfet_nmos_dc` pwl error 0 → 3.55e-15 (trbdf2 0, dsed
    refused, unchanged). Every pinned Python and C++ test passes (5281
    pytest, 375 reference; layer2 and layer5 on Eigen 5 and 3.4.0).

45. **The saturable-inductor commit (item 3; the open item of #38/#41).**
    Measured first (`.scratch/bench_commit2.cpp`, 64 atan entries, best of
    31): hand commit 3.3–3.5 ns per entry, interface 10.2–11.1 ns, the law's
    eval alone 3.5–3.8 ns — so the 7 ns were `commit_fq`'s time formula, not
    λ. Hoisting the formula switch (`history_terms` once) changed nothing
    (10.7–11.1 ns; reverted). Fix: a law whose q lives on internal rows only
    declares `kPinQ = false` (saturable inductor, PMSM, induction machine);
    `commit_fq` then takes no formula (q̇_n ← 0 on a pin row, −f on an
    internal one; every caller discards the pin rows, debug-asserted q = 0
    there). Interface commit 3.4–3.6 ns (1.02–1.07× the hand); machines:
    PMSM commit 67 → 53 ns (hand 43), IM 23 → 15.6 ns (hand 6.6). The
    PMSM commit's h = 0 guard (a pin row's 2/h·0) is gone with it.
    Instructions per accepted step, satind_pwl (`.scratch/instr.sh`, K = 20,
    three runs): interface 17 115 / 17 109 / 17 107 (was 17 306 in #41),
    legacy 16 624 / 16 828 / 16 807 → ratios 1.030, 1.017, 1.018 (were
    1.023–1.031). Under the gate on two of three, NOT clearly under: the
    legacy baseline alone moves ±0.6 % run to run, and the interface's
    remaining ~300 instructions per step are in the stamp (per Newton
    iteration), not the commit. Recorded, not chased further here.

46. **Evidence regenerated at 2e60463b (item 5).** Instructions retired per
    accepted step (`.scratch/instr.sh`, K = 20; machines: 5× longer runs,
    K = 6):

    | case | legacy | interface | ratio | #41 / #42 |
    |---|---|---|---|---|
    | saturable inductor (pwl) | 16 722 | 16 974 | 1.015 | 1.023–1.031 |
    | Coss (pwl) | 13 808 | 13 711 | 0.993 | 0.988 |
    | IGBT tail + Lauritzen chopper (pwl) | 28 608 | 27 025 | 0.945 | 0.959 |
    | IGBT tail + Lauritzen chopper (trbdf2) | 90 719 | 85 143 | 0.939 | 0.910 |
    | soft boost MOSFET + Coss + saturable (pwl) | 35 083 | 30 463 | 0.868 | 0.918 |
    | soft boost (trbdf2) | 128 168 | 108 384 | 0.846 | 0.897 |
    | boost, 1 ns edges, κ = 15 (pwl) | 35 757 | 31 148 | 0.871 | 0.911 |
    | MOSFET on, AC reverse conduction (pwl) | 24 652 | 22 783 | 0.924 | 0.968 |
    | PMSM sine start | 64 694 | 64 854 | 1.002 | 1.016 |
    | PMSM locked rotor | 56 529 | 56 670 | 1.002 | 1.009 |
    | flux-map saturated start | 247 292 | 249 270 | 1.008 | 1.001–1.003 |
    | IM sine start | 63 664 | 68 255 | 1.072 | 1.086 |
    | IM generator, held speed | 65 491 | 69 630 | 1.063 | 1.084 |
    | IM six-step inverter start | 92 407 | 96 751 | 1.047 | 1.040 |

    The MOSFET rows' ratios fell because LEGACY got dearer with #44 (the
    model now evaluates forward(a, d), forward(a, 0) and an exp: soft boost
    legacy 33 674 → 35 083, +4.2 %; AC reverse conduction 21 905 → 24 652,
    +12.5 %) — a cost of the default path, not of the interface. The
    commit fix (#45) brought the PMSM to 1.002; the induction machine stays
    over the gate (the AD-width-10 stamp, #42).
    `benchmarks/DEVICE_PATH_EVIDENCE.md` regenerated: 289 cells on both
    paths, 289 with the same outcome and error to 3 significant digits, 0
    refused (#41: 277, 274, 4 refused — the MOSFET cells that parted under
    #39 agree again with one law). `law:` rows, pwl: soft boost 2.86e-14
    (was 2.49e-3), every other ≤ 7.5e-14 except the IM inverter start
    2.43e-13; trbdf2 ≤ 5.8e-7 (soft boost; was 2.08e-4). The tool now
    leaves a floating machine neutral out as the test gate does (it read
    0.15 "of scale" on a 1e-11 V node).

## Slice 9 — default-path cost, machine speed, stepper request (2026-11-19)

47. **The default path's MOSFET cost of #44, recovered where the law
    allows it — bit-identical to 4aa3b15e.** `models::MosfetLevel1::
    forward_c1_` keeps 4aa3b15e's arithmetic for every result and changes
    only shared and skipped work: (a) σ(z) and σ(−z) from one exponential
    (`LogisticPair`, each half formed on demand with `numeric::logistic`'s
    own operations; z = ±0 / NaN fall back to the two calls); (b) the exits
    are decided on Real values (an AD value is the Real computation's
    value), and the AD f and forward(a, 0) are formed only on the exits
    that return them; (c) a pre-test `leak_below_half_ulp_` — no exp, no
    forward(a, 0) — that is a SUFFICIENT condition for the "leak ≤ 2⁻⁵⁴·f"
    exit (bound fl(leak) ≤ 16·|K|·a²·e^{−κ|a|−κd}, tested with a factor 4
    of margin, f normal; proof in the header); (d) for κd ≤ 1 with
    |forward(a, 0)|/4 above half an ulp of f, the exit is impossible
    (e^{−κd} ≥ e^{−1}), so its exp is skipped. Proof of identity: the
    dense sweep `.scratch/mos_pretest_sweep.cpp` compares bit patterns
    against a verbatim copy of 4aa3b15e's body — five parameter sets
    (K, V_T, λ, κ) = (1.43, 3, 0.01, 15), (0.5, 3, 0, 4), (50, 3, 0.02, 15),
    (1e-3, 2, 0.02, 100), (2, 1.5, −0.01, 1); V_OV ∈ [−60/κ, 200/κ] step
    0.05/κ × V_DS ∈ [−3, 3] V on 1 mV, plus a log grid 1e-15…3 V of both
    signs, ±{10, 48, 400, 1e4} V, and, for every V_GS where the pre-test
    flips along V_DS, the 129 adjacent doubles around the flip on both
    signs (13 178 flips, 3.4 M points): 165 593 109 Real evaluations, the
    AD-x (width 3) and AD-parameter (width 4) tangents on a subset, 0
    differing bits, 0 points where the pre-test passed and the original
    exit did not (it catches 120.4 M of the 126.2 M original exits). C++
    test: `test_mosfet_level1.cpp` "shared exponentials and early exits"
    (the 4aa3b15e body inline, a few ulp — the bit gate is this machine's).
    Instructions per accepted step, legacy (`.scratch/instr1.sh`, K = 20,
    three runs; "before #44" = add3b90a built in `.scratch/src_pre44`):

    | case | before #44 | 4aa3b15e | this slice | vs before |
    |---|---|---|---|---|
    | soft boost (pwl) | 33 568 / 33 650 / 33 655 | 34 858 / 35 037 / 35 199 | 34 398 / 34 740 / 34 489 | +2.7 % |
    | AC reverse conduction (pwl) | 21 921 / 21 782 / 21 758 | 24 360 / 24 481 / 24 370 | 22 763 / 22 721 / 22 704 | +4.2 % |

    NOT within 1 %: in both circuits the on-state points carry a leak the
    law does not drop. With κ = 4 and V_GS = 12 V, κ·V_OV = 36 and the
    computed 1 − σ(36) = 2⁻⁵² (true e⁻³⁶ = 2.3e-16), so forward(a, 0) ≈
    0.5·81·2.2e-16 = 9e-15 A against half an ulp of a 1 A current
    (1.1e-16 A): the correction moves the result by ~50 ulp and 4aa3b15e's
    law computes it at every such point (κd ≤ 1: the expm1 form, three
    exponentials; κd > 1: f − leak, one more exp). Per law call (AD 3,
    `.scratch/bench_mos_instr.cpp`): raw mirror 339 instructions;
    4aa3b15e 504 where the exit is taken, 936 where the expm1 form runs;
    now 480 and 717. The rest is that arithmetic, which bit-identity
    forbids changing. Lever (a NUMBER change, not taken): drop the leak
    when it is below a relative 1e-12 of f instead of half an ulp — every
    on-state point then takes the cheap exit (≈ before-#44 cost) and the
    default numbers move by ≤ 1e-12 relative; an owner decision for the
    number-changing release.

48. **The induction machine off AD width 10 (#42's lever).** A law may
    now give `stamp_fq` its Jacobian by hand: `eval_jac(aux, p, x, a, o, J)`
    writes the Real values in `o` and the nonzero entries of J = ∂f/∂x +
    a·∂q/∂x (each as fd + a·qd of what the AD tangents were, an exact zero
    dropped — so the same values up to the sign of an exact zero); the
    AD branch is unchanged for every other law and stays the path for
    ∂/∂p. `InductionMachine::terms` computes λ, the rotor rate g, the
    torque and their partials in u = (i, λ_r, ω) once, in the hand stamp's
    operation order; `eval` spreads them with `chain_n` as before,
    `eval_jac` writes them (rows 0–3 ±1, stator rows 1, −1, −R_s + a·(−L),
    a·(−M); rotor rows −C_r·∂g; speed row −∂T). The commit gained
    `eval_internal` (q and f of the internal rows only — all `commit_fq`
    keeps when `kPinQ = false`), and `commit_fq` is force-inlined.
    Instructions per call (`.scratch/bench_im_instr.cpp`): stamp hand 3 305,
    interface 3 860 → 3 474; commit hand 184, interface 334 → 206.
    Interface results are BYTE-identical to 4aa3b15e's on every machine
    case (`.scratch/iface_ident.py`). Tests: `test_device_laws.cpp` —
    `eval_jac` against the AD eval for 20 random states × three a (1e-14),
    `eval_internal` against `eval`'s internal rows; the existing stamp /
    commit / capture gates against `refresh_pmsm_mna` (32 / 16 ulp) now run
    through the new branch. Instructions per accepted step, interface /
    legacy (`.scratch/instr.sh`, K = 6, 5× longer runs, PYTHONHASHSEED=0),
    runs 1–3: IM sine start 1.005 / 1.010 / 1.014 (was 1.059 / 1.070 /
    1.066 at 4aa3b15e, same probe); held generator 0.998 / 1.007 / 1.038,
    two more runs 0.984 / 1.006 (run 3's LEGACY baseline was 2.8 % below
    its other four — the probe's run-to-run noise here is ±1.5 %); six-step
    inverter 1.010 / 1.007 / 1.002 (was 1.038 / 1.055 / 1.040). Under the
    3 % gate on every run but the one legacy outlier.

49. **Requests to the stepper track (loop freeze).**
    (1) `solver/run_transient.hpp:128`, in `seed_held_machine_capacitors`:
    the line `(void)pwl::refresh_pmsm_mna(x, J_nl, f_nl, pmsm, Real{1});`
    becomes, behind the run's path,
    `if (path == device::DevicePath::Interface) (void)pwl::iface::stamp(pmsm,
    x, J_nl, f_nl, Real{1}, TrBdf2Stage::Trapezoidal, /*inc=*/true); else`
    the existing call — with `device::DevicePath path` added as the
    function's last parameter and passed `opts.device_path` (or
    `hs.path()`) at its one call site (`run_transient.hpp:782`).
    `iface::stamp` takes a `const PmsmMnaHistory&` since this slice for it.
    Only the mechanical rows (ω, θ) are read there, which carry no q, so h
    and the formula do not enter; the law's rows equal the hand rows to
    32 ulp (#42). TR-BDF2 has no such seed. Until it lands both paths seed
    from the hand stamp (same values to rounding).
    (2) Unchanged: `StepHooks::on_device_state` (#31).
    The "unported kind is refused by name" test: not rebuilt. The refusal
    in `StatefulHistories::set_path` walks the compile-time kind lists
    (`StatefulKinds`, `kPorted`, `static_kind_ported`); every shipped
    Newton kind is ported, and a test cannot add an unported dummy kind to
    those lists without a test-only kind in the production headers (task
    8.4's `TestRcDevice`, not built). The equivalence module's docstring no
    longer claims the test; the refusal stays exercised only by code review
    until 8.4.

50. **The saturable inductor at the gate: located, partly recovered.**
    A C++ driver of the golden `satind_pwl` circuit through
    `solver::run_transient` (`.scratch/satrun.cpp`, instructions per step
    from 20 000 vs 120 000 steps, minimum of five runs; same results to the
    bit on both paths) separates the phases by substitution: with the
    interface path's stamp swapped for the hand stamp the two paths cost
    the same (16 868 / 16 887); with the commit swapped nothing moves —
    the whole gap is the stamp. Per call in isolation the interface stamp
    is CHEAPER (289 vs 337 instructions, one grounded instance,
    `.scratch/bench_sat_instr.cpp`); in the run it is ~70 dearer per call
    (about 4.3 stamps per step): code generation inside the refresh, not
    arithmetic — forcing `iface::stamp` inline brought the isolated whole-
    kind call 436 → 294 and moved nothing in the run (reverted);
    `noinline` made it worse. Kept: the law's hand Jacobian (`eval_jac`,
    #48's mechanism; λ and dλ/di from the one-tangent flux `univariate`
    already runs, five entries): run 16 776 / 17 128 (1.021) → 16 838 /
    17 112 (1.016). Test: `test_device_laws.cpp`, against the AD eval at
    seven currents × three a (1e-15). Python probe (`.scratch/instr.sh`,
    K = 20, three runs): 1.020 / 1.021 / 1.016 (#46: 1.015; #45: 1.017–
    1.030). NOT clearly under 1.02: the remaining ~70 instructions per stamp
    call are in how the stamp is compiled inside the refresh, which the
    isolated benchmarks do not reproduce; next lever, untried: a
    `kPinQ = false` assembly that skips the pin rows' a·q + history (zeros
    there), and building the stamp's q_n / q̇_n arrays once per kind.

### Hand-off (2026-11-19, slice 9)

- Done: item-1 verification of 4aa3b15e (pytest 5281 passed / 108 skipped /
  7 xfailed; reference 375 passed; reference_status exit 0; goldens vs
  golden_m13 differ only in `mosl1_trbdf2`, as #44 recorded; stateful
  compare identical). #47 (legacy MOSFET cost, bit-identical), #48 (IM off
  AD width 10: 1.00–1.01×), #49 (stepper request; the refusal test not
  rebuilt), #50 (saturable inductor 1.016–1.021×). Default path: goldens
  and stateful compares BYTE-identical to 4aa3b15e; both paths' runs of the
  machine, saturable and MOSFET cases byte-identical to 4aa3b15e's.
- Open: the legacy MOSFET stays +2.7 % (soft boost) / +4.2 % (AC) over
  before-#44 — the rest is the C¹ correction's own arithmetic at points
  where it changes the result (#47; the lever is a number change). The
  saturable inductor is not clearly under 1.02 (#50, levers named).
- Open, unchanged: `StepHooks::on_device_state` (#31) and #49's seed call
  are the stepper track's.
- Release list: the MOSFET's negative slope near threshold and past
  saturation onset (#44), a quiet-machine wall-time check (#38), #47's
  relative leak cut-off.

## Slice 10 — integration merge, evidence labels, restart cause (2026-11-19)

51. **What the evidence table's above-1e-12 cells are.**
    (a) `analytic:im_locked_rotor` / `im_free_acceleration` pwl 0.0355 /
    0.0389 (trbdf2 1.3e-4 / 2.3e-5) were NOT the analytic error (that is the
    `err` columns, equal on both paths) and not a law difference: the worst
    column is state 3, the floating star point, max |v| 3.0e-11 / 5.8e-11 V
    (physically 0), measured against its own round-off through the 1e-9
    floor; the absolute difference 3.5e-11 / 3.9e-11 V is 2.5e-13 / 2.0e-13
    of the 141 / 195 V phases (`.scratch/probe_2a_im.py`). The law-circuit
    block already dropped such a neutral; the analytic block did not. Tool
    fixed: a state at round-off level (max ≤ 1e-12 of the circuit's largest)
    is measured against the circuit's largest — 1e-12, not a looser cut, so a
    small-unit state (a 3e-8 C stored charge next to 400 V) keeps its own
    scale (reviewer); the table gained a `grid` column and the
    header says what each column means.
    (b) The trbdf2 law cells (1.02e-12 … 5.77e-7) are grid-driven, not law
    differences (`.scratch/probe_2a.py`): in every one the accepted step
    sequences part early by a rounding of the error estimate — coss+saturable
    boost at step 6 (t = 23 ns; difference before the parting 7.9e-18 of
    scale), fast-gate boost step 60 (2.7e-15), IGBT+VDMOS chopper step 68
    (1.1e-11), permeance core step 3 (3.9e-22), tail+Lauritzen step 1 (0) —
    same accepted-point counts, Newton counts equal until well after the
    parting. After it each run carries its own truncation error; at the
    common times the difference is ≤ 0.58·rtol (rtol 1e-6). Gate: the
    existing `test_trbdf2_trajectories_agree_to_rtol` (10·rtol); the table
    now marks those cells `parted`.

52. **`mosl1_pwl_restart` moves ≤ 4.8e-6 of scale: the restart stage solves
    inside the gate ramp.** Experiment (`.scratch/probe_2b.py`, old law =
    `.scratch/pypre44`, add3b90a): the golden `mos_l1` circuit (0.1 µs gate
    edges) with `discontinuity_restart=True` differs from the old law from
    the first sample (4.83e-6 at t = 0.1 µs, V_DS = 345 µV, decaying to
    3.5e-6 by 0.4 µs); plain `mosl1_pwl` is byte-identical although it too
    sits at V_DS = 169 µV at t = 0.1 µs with V_GS = 10 V. With 1 ns gate edges
    (`.scratch/probe_2b_fast.py`) BOTH runs are byte-identical under the two
    laws. So the integration verifier's reading is half right: the leak band
    is V_DS ≈ 0, but the discriminator is V_GS — the SDIRK restart stage
    (2γ·dt after the edge) solves inside the 0.1 µs ramp, at an intermediate
    V_GS where κ·V_OV is small and the leak f(a, 0)·e^{−κ|V_DS|} is far above
    half an ulp of the current; at V_GS = 10 V (κ·V_OV = 15·7 = 105) it is below it
    and #47's exit returns the old value bit for bit. Expected, not chased.

53. **The machine commit gate no longer needs bit equality across two code
    paths.** `test_device_laws.cpp` "the machine laws stamp, commit and
    capture as the hand stamp": `dlam_gamma` stays `==` (`iface::capture_gamma`
    for the machine IS the history's `capture_gamma`, one function — a
    comment says so); `lambda_old` (law eval vs model) and `dlam_old`
    (v − R·i in two functions) now compare to 8 ulp of the three phases'
    flux terms (max(|λ_k|, 0.1 H·max|i|, 1 Wb): λ sums L·i terms that can
    cancel) / of max(|v|, |R·i|) — a compiler that contracts one side to an
    FMA passes. Built and run on Eigen 5 and 3.4.0. (The machines' ∂/∂p
    against central differences already existed, `check_param_tangent` at
    the end of the same test; add-parameter-sensitivities 4.2 is annotated
    with the coverage.)

54. **The saturable inductor after the integration merge: NOT under 1.02.**
    Python probe (`.scratch/instr.sh satind_pwl`, K = 20, merged tree
    d386a013): 16 843 → 17 283 (1.026), 16 790 → 17 228 (1.026),
    16 842 → 17 156 (1.019). The standalone C++ driver of the same circuit
    (`.scratch/satrun.cpp`, `-O3 -ffp-contract=off`, min of five) on the
    same sources: 16 889 → 17 087 (1.012). The interface path's extra cost
    per step is ≈ 440 instructions in the extension and ≈ 200 in the driver
    for the same code, so about half of it is how the extension's
    translation unit compiles the refresh (inlining budget in the large
    binding TU, the default `-ffp-contract`), not the law — consistent with
    #50 (cheaper in isolation, dearer in place). Not fixed this slice;
    next step: compare the two builds' `build_device_refresh` interface
    lambda (`-Rpass-missed=inline` on the binding TU that instantiates
    `run_transient`), then #50's untried levers.

### Hand-off (2026-11-19, slice 10)

- Merged `origin/feat/core-integration` e3ee5341 (d386a013, no conflicts;
  stubs `--check` clean, REFERENCE_STATUS.md regenerated unchanged). Python
  set on the merge: pytest 5524 passed / 107 skipped / 8 xfailed, reference
  377, reference_status exit 0, ruff, openspec 78/78; goldens vs
  `golden_4aa` differ only in the trbdf2 stat key (`n_newton_start_retries`,
  33 → 34 keys) and `syncbuck_dsed`'s zero Newton counters (both accepted in
  the integration); stateful compare identical.
- #51 (evidence labels: analytic IM cells were a floating neutral; trbdf2
  law cells are parted grids, tolerance-bounded; regenerated: 289/289 cells
  agree, 0 same-grid cells above 1e-12, 7 parted), #52 (restart cell: the
  restart stage solves inside the gate ramp), #53 (machine commit gate
  FMA-safe), #54 (saturable inductor 1.019–1.026, open).
- Task groups 5–14: every open box carries its reason in tasks.md. Next
  candidates without a loop edit: 10.x (AC/DC from f and q), 12.x (field
  tables). No new stepper request this slice (#49's two stand).

## Slice 11 — AC and DC from f and q (2026-11-19)

55. **`linearize` (task 10.1/10.2): what is linearised, where, and what is
    refused.** `analysis/linearize.hpp`. The linear part is the sweep's own
    assembly (`linear_part`: `assemble_segment_split` at the caller's mask,
    E = C_split/2), unchanged operation for operation; a circuit with no
    Newton device returns it before anything else runs (62 sweep outputs of
    the existing tests byte-identical against the merged tip; C++ test
    "a linear circuit is the sweep's linear part, bit for bit"). External
    devices keep their path (same calls, same order). Every built-in
    Newton kind adds ∂f/∂x to G and ∂q/∂x to C from ONE forward-AD
    evaluation of its `Device` law at x0. The operating point: the DC point
    of the caller's mask through the device laws
    (`dc_device_refresh(…, Interface)`), PWL diodes held at the mask (as the
    external path always did); internal unknowns (the saturable inductor's
    current, the PMSM's phase currents, θ, ω) are read from that x0; a
    LOCAL charge (Lauritzen q_M, IGBT-tail Q) is taken at its DC value
    (its affine row solved at q̇ = 0) and kept as an EXTRA unknown after the
    circuit's — condensing it would give an ω-dependent admittance, not a
    real (G, C) pair. Refused by name (`UnsupportedError(engine='ac_sweep')`):
    InductionMachineMna (kind table; no analytic small-signal test written);
    a Jiles–Atherton inductor (`feature='hysteretic_core'`: the incremental
    L depends on the direction of the excursion, a minor loop — no unique
    small-signal model); a nonlinear capacitor on a magnetic node
    (`feature='nonlinear_permeance'`: its DC point needs a nonlinear
    flux-conservation row the DC assembly lacks — it used to fail as "no DC
    operating point"); a PMSM held at ω ≠ 0 (abc quantities rotate: no
    equilibrium) or whose shaft row of the FREE system, (G·x0 + b)_ω +
    f_ω(x0), is not 0 to 1e-9 of its largest term (the DC point holds the
    mechanics, so it does not enforce T = T_load by itself). The two
    conformance PMSM cells (BLDC harmonics, flux map) are driven, so not in
    torque balance: they are refused with that reason. The IGBT-tail
    conformance cell's DC point has the gate at 0 V (IGBT off), so the tail
    has nothing to linearise there: presence is not asserted for it (the
    conducting tail is the analytic test).

56. **The DC point through the device contract (task 10.3).** What
    `Formula::Dc` (q̇ = 0) means per kind, and what the code does:
    static kinds — already their laws on the interface path; saturable
    inductor — the DC assembly's short IS the law at DC (f₂ = v = 0);
    nonlinear capacitor — open IS the law (q only); IGBT tail — its DC
    current is exactly 0 (Q = τ·k·i_ss, i_tail = Q/τ − k·i_ss): no stamp;
    Lauritzen diode — the hand path has NO DC stamp (an open circuit: a
    1.7 V source behind 1 Ω reads 1.700 V at the anode, the law 0.7242 V);
    `dc_device_refresh` on `DevicePath::Interface` now stamps its law at
    DC (local row condensed, `iface::stamp_local` at `Formula::Dc`), mpmath
    bisection agrees to 1e-12. The default path is untouched (no default
    number moves). NOT done: `run_transient(start_from_dc_op)` on the
    interface path builds its DC refresh inside the loop file
    (`nl_static`), so it still opens the Lauritzen diode — a one-line loop
    hunk (use `dc_device_refresh(pool, …, path)` there) left for the
    stepper owner / the release. PMSM — `Formula::Dc` would be WRONG at a
    held speed (q̇ = ω·∂λ/∂θ ≠ 0); the DC assembly's held-mechanics stamp
    (v = R·i + ω_e·(∂L·i + λ'_pm)) is the right law; tested against the
    closed form at ω0 = 40 rad/s on both paths (1e-12).

57. **trbdf2's error control did not refine a small sine** (observation for
    the TR-BDF2 track, not chased here). The junction-capacitor case at
    10 kHz with a 5 mV sine on 5 V: rtol 1e-8, 1e-10 and 1e-12 all gave
    1004 accepted points (span/1000, the default ceiling) and the same
    8.6e-4 error of the fitted fundamental; the ceiling alone sets the
    accuracy (1.5e-4 at h_max = 1 µs, 1.5e-6 at 0.1 µs, 9.8e-8 at 25 ns).
    The local error per step there is ~amp·(ωh)³/12 ≈ 1e-7 V against
    rtol·|x| = 5e-10 V, so the estimate does not see it. The cross-check
    tests pass `dt = T/400` (the ceiling).

58. **Measured errors (task 10.4).** |H_sweep − H_ref|/|H_ref| at
    100 Hz / 1 kHz / 10 kHz; closed form | small-signal trbdf2 transient
    (1e-3·bias sine, h_max = T/400, whole periods after ≥ 2 ms, cubic
    spline + Simpson, time-weighted):
    Shockley at 1 A 6.9e-13 6.8e-13 3.6e-13 | 6.8e-7 1.8e-6 8.5e-6;
    saturable L(I) 5.7e-12 5.7e-13 5.8e-14 | 1.1e-5 1.0e-5 5.0e-6;
    junction C(V) 1.0e-10 9.7e-11 5.3e-11 | 2.7e-7 2.5e-6 9.3e-6;
    transformer k = 0.99 1.4e-17 1.4e-16 5.9e-16 | 1.9e-6 1.3e-6 3.0e-7;
    RC through the closed switch < 1e-9 (asserted) | < 1e-3 (asserted);
    Lauritzen 7.0e-13 7.1e-13 7.1e-13 | 7.6e-7 3.1e-6 2.8e-6;
    IGBT static 1.3e-10 ×3 | 8.7e-9 ×3; IGBT + tail 1.3e-10 ×3 |
    3.2e-7 2.5e-6 1.5e-6; MOSFET L1 1.7e-9 1.5e-9 7.2e-10 |
    6.5e-7 5.0e-6 9.9e-6; smooth ideal diode 3.2e-10 3.2e-10 2.3e-10 |
    1.2e-7 9.8e-7 7.0e-6; PMSM at rest (motional impedance) < 1e-9 |
    2.2e-6 2.3e-7 7.5e-8. MOSFET / IGBT / ideal diode take g from a central
    difference of two DC points (their closed form needs the law's
    smoothing); the others from the law's own closed form. Tiny-bias
    linearisations: Shockley at 1 pA (g itself, 1.6e-9: the 1/H − 1 extraction cancels) and the saturable
    inductor at 1e-12 A (L_0 to 1e-9) match. Gates: closed form 1e-9
    (1e-6 for the central-difference kinds, 1e-10 Lauritzen), transient
    1e-3.

59. **Generic outputs (tasks 9.2-9.5) without the 9.1 loop recording.**
    `res.i` reads the kernel law for the static kinds through a new
    read-only binding, `static_device_readings(graph, pool, states,
    branch_id)`, which runs `pwl::static_reading` (the law the in-loop
    recorders read) on each stored state. No loop file is touched. The
    Python ladder stays for the kinds it already covered, so no existing
    `res.i` number moves (baseline identical); deleting it is 9.1's job.
    Uncovered, still raising with the 1 mOhm recipe: the Lauritzen diode
    and the IGBT tail (the law needs the stored charge, a local unknown
    the states do not hold), the nonlinear capacitor (its current is
    dq/dt, a step quantity), and the `power_mosfet` composite (its kind is
    not a pool device). Row 0 of a zero start is not a solve: the smooth
    diode's non-passive law reads -1.2e-5 A at v = 0 there, so the test
    compares from row 1. The loss summary integrates the product of the
    piecewise-linear v and i, so its reference uses the same quadrature
    (the trapezoid of v·i differs by 1.6e-4 on the MOSFET case). Measured
    against the sense resistor, max |i − i_Rs| / peak and the energy
    error: MOSFET 2e-13 / 4e-15, IGBT 1.7e-12 / 2e-13, smooth diode
    1.7e-12 / 3e-13, Shockley 1.5e-12 / 3e-13, saturable L 1.1e-12 /
    5e-14 (worst of pwl and trbdf2).

60. **Slice 11 review (fresh-context reviewer, read-only) and hand-off.**
    No existing AC, DC or `res.i` number changed; no refusal was lifted
    without an analytic test; no kind linearised at a wrong point. Fixed:
    the PMSM torque-balance check refused a 3e-14 N·m residual at a 1e-12 V
    bias (relative-only tolerance); it now has an absolute floor of the
    torque of 1 nA through the PM flux. Added a held-rotor test with a DC
    current and a balancing `T_load` (the θ stiffness S = pp²ψ cos θ_e i_a0
    in the shaft row, against the closed form to 1e-9). The
    `static_device_readings` binding refuses a states array narrower than
    the circuit's state; `device_thermal_summary` warns when it skips a
    tailed IGBT. Left: (a) `simulate(device_path='interface',
    start_from_dc_op=True)` still starts from `nl_static`, which does not
    stamp the Lauritzen law, so its DC point differs from
    `compute_dc_operating_point(device_path=Interface)` — the loop hunk of
    #56, not done (LOOP FREEZE); (b) the saturable table test uses a
    straight-line table and the 1 pA case sits at L_0, so only the 1 A case
    probes the operating point; (c) the conformance AC cells for the
    composites check finiteness and presence only (the analytic numbers are
    in test_ac_linearize.py). Next slice: 9.1 (per-sample pin currents in
    the loop) lets 9.2 delete the Python ladder and cover the stored-charge
    kinds; group 12 (field tables) not started.
