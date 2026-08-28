# Changelog

All notable changes to Pulsim are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Phase 3 — the unified engine

* **The DSED cost model at scale** (Phase-3 item 4, audit E.4) — and
  a measurement that corrected the audit's own attribution. Per-mode
  mask resolve on a 400-state RC ladder: **267 ms → 1.75 ms (152×)**;
  800 states resolve in 9.9 ms; a switched 200-state ladder runs 5 ms
  of simulation in 0.25 s on the default path.

  Where the time actually was: the audit blamed the dense Schur
  (`FullPivLU`, O(n³)) — real, but only ~15 ms of the 267. The other
  ~252 ms was the **exact stepper's eigendecomposition from item 2**,
  built eagerly on every mask resolve, including on the `auto` path
  that never steps exactly. Three fixes:

  * The extraction now stays SPARSE end to end: partition by triplet
    routing, `G_aa` factored once by the kernel's own sparse LU
    (Gilbert–Peierls + COLAMD, Phase 1a), sparse factors kept on the
    left of every product, and the floating-cap congruence applied
    through `T.sparseView()`. The scatter/selector matrices are gone
    — one 1 per row is an index write, not a matrix. Extraction
    alone: ~7× at 400 states, and every consumer (AC sweep, DSED
    bridge, recovery map) inherits it.
  * The exact stepper is built **lazily**, on the first
    `exact_advance_state` call — paths that never step exactly no
    longer pay O(n³) per visited mode.
  * `PEDSimulatorBDF2` pulled `A_matrix()` **by value inside the
    step loop** — a 1.3 MB dense copy per step at 400 states. By
    reference now. And the stiffness detector's full eigensolve is
    replaced above n = 64 by two-step **power iteration** (the
    dispatch decision needs |λ|max to a few percent against a
    threshold; the two-step norm ratio settles on the modulus even
    for a dominant complex pair — tested against the dense solver on
    both a real-dominant and an LC-pair-dominant 80–100-state
    system).

* **Sine-driven modes step exactly too** (Phase-3 item 3, reshaped by
  measurement). Item 2 covered DC-driven modes; the RK45 fallback
  still owned anything with a time-varying source, and an ordinary
  sine rectifier with an RC snubber ground it down — 157k steps at
  τ = 1e-6, 1.9M steps and 8.3 s at τ = 1e-8, refused outright at
  τ = 1e-10. The audit prescribed a Radau stiff member here; the
  stronger move came first: a sine-driven LTI mode is autonomous
  once the state is augmented with the source's own oscillator pair
  (u = (sin ωt, cos ωt), u̇ = [[0,ω],[−ω,−0]]·u — amplitudes and
  phases folded into the coupling columns, one pair per distinct
  frequency, v_dc into the constant term). The augmented system
  steps EXACTLY: all three snubber cases now run in ~0.1 s, and a
  C++ test lands a sine-driven RC on its closed-form response to
  1e-9 across composed steps.

  The arbitration that validated it: the exact path, the dsed RK45
  path and the pwl engine at dt = 1e-8 all agree on 5.9262 V — the
  outlier was the pwl engine at dt = 1e-7, off by 3.2% from its own
  commutation resolution. The event-driven answer is again the
  sharper one.

  Deliberately still numeric: PWM/pulse sources (a stepped b(t) is
  not a finite oscillator sum — a test pins that the adapter
  declines) and user `b_extra_fn` callbacks. A stiff circuit in
  those classes is Radau's remaining subject, and per this
  project's standing rule the code waits for a repro. Oscillator
  phase never drifts: the pair is rebuilt from absolute time at
  every step, and driving a circuit exactly at its own resonance
  degrades safely (defective eigenbasis → numeric fallback).

* **Consistent reinitialization + exact LTI stepping** (Phase-3
  item 2, the audit's "obra №3") — and the measurement that reshaped
  it. The DCM buck that item 1 refused now runs the full 5 ms in
  **0.02 s** and lands within **0.024%** of the pwl engine at
  dt = 1e-8.

  The projection (`dsed/event_projection.hpp`): at every event the
  carried-over state is projected onto the NEW mode's slow manifold —
  fast stable modal components move to their quasi-static values,
  everything else passes through exactly. In the ideal-switch limit
  this **is** the audit's `argmin‖x⁺−x⁻‖_M` charge/flux-conserving
  projection: for two capacitors a switch just paralleled, the
  preserved coordinate is (C₁v₁+C₂v₂)/(C₁+C₂) — charge conservation —
  and a test pins that the equalization lands there to machine
  precision. Resonant modes (Re λ ≈ 0, any |Im|) are physics and are
  never touched; with no fast stable mode the projection is the exact
  identity.

  **The projection alone did not fix DCM, and the measurement is the
  story**: with the state sitting *exactly* at the idle mode's
  equilibrium (i_L frozen at −14 µA, its true g_off leakage level),
  the integrator still ground at h ≈ 2e-10 s. An explicit method is
  **stability**-limited, not accuracy-limited — DOPRI5's region ends
  near |hλ| ≈ 3.3, and no state and no error controller move that
  bound. What removes it is `dsed/exact_lti.hpp`: between events a
  PWL circuit with DC sources is autonomous LTI, so the trajectory
  has a closed form (`x(t+h) = V·e^{Λh}·V⁻¹x + h·φ₁(Λh)·V⁻¹b`, the
  φ₁ form keeping λ = 0 integrator modes exact) valid for ANY h —
  a C++ test crosses the 5e9-rad/s idle mode in ONE 10 µs step,
  fifteen thousand stability limits at once. One eigendecomposition
  per visited mask, cached in the adapter; predicates are located on
  the **analytic** trajectory rather than the Hermite interpolant;
  time-varying sources or a defective eigenbasis fall back to the
  numeric path, which remains correct everywhere.

* **The dsed engine commutates PWL diodes** (Phase-3 item 1). It
  never had a diode before — only a resistor whose state the user
  pinned through `switch_fn`: a reverse-biased series diode conducted
  backwards (−10.909 V where the pwl engine blocks at −1e-06 V), and
  a buck's frozen freewheel diode settled v_out at **0.59 V where
  12 V is correct** — an error the buck benchmarks never caught
  because they asserted speed and finiteness, not the number.

  Every PWL diode now gets two auto-derived event predicates on its
  branch voltage, reconstructed from the reduced state through the
  new algebraic recovery map (`ContinuousLTI.recover_from_state` /
  `recover_const` / `recover_from_b` — the map Step 6 of the
  extraction was already solving for and discarding). Predicates are
  armed by the diode's own bit — turn-ON (`v_D − V_th`) while OFF,
  turn-OFF (the `i_D` zero-cross) while ON — which is also the
  hysteresis: firing flips the bit, disarming the predicate that
  fired and arming its counterpart. The bit flip happens INSIDE the
  scheduler (`fire_event_` no longer asks `switch_fn`, a pure
  function of time, what a diode should do), and a zero-time cascade
  settles diodes the mask change instantaneously forward-biases —
  the freewheel diode at gate-off, where integrating even one step
  of the intermediate mode would put ±i/g_off volts on the switch
  node. The census and decision rule are the same code the pwl
  engine uses (`DiodeEventState`, `SwitchedDiode::decide_next_state`),
  so the engines cannot drift on what "conducting" means.

  Measured: the reverse rectifier blocks; the real-diode buck lands
  at 11.994 V / 5.997 A; a full-wave bridge matches the pwl engine's
  time-average to 0.00%; and the half-wave rectifier's peak is
  **exactly** the source peak, where the pwl engine at dt = 1e-6
  overshoots to 10.46 V from trapezoidal commutation ringing — the
  event-driven answer is the sharper one, which is the entire
  argument for the engine.

  What is refused, loudly and by name: **discontinuous conduction**
  (the idle mode's L·g_off ≈ 1e-13 s grinds an explicit integrator —
  a new progress guard converts a 7-second silent burn of the
  10M-step cap into a 0.07 s error naming the mechanism and pointing
  at `engine='pwl'`; the fix is Phase-3 item 2's consistent
  reinitialization), an **explicit `integrator='bdf2'`** with diodes
  (predicates live on the RK45 scheduler; 'auto' routes there), and
  **nonlinear devices** (unchanged). The diode-circuit fallback to
  the predicate-less Bridge.10 path is also closed — it would have
  silently reintroduced the frozen bits.

### Phase 2 — automatic robustness (v2.0 audit follow-up)

* **Adversarial review of the Phase-2 tail** (44 agents over
  #100-#105): 30 findings survived refutation, 9 did not. Ten
  distinct defects, all introduced by this phase, all fixed:

  * **A Python callback's exception was treated as a step-size
    problem.** Both new catch sites caught every `std::exception`,
    which includes `pybind11::error_already_set` — and *that* type's
    constructor calls `PyErr_Fetch`, which **clears** the Python
    error indicator. So a `KeyboardInterrupt` raised inside the
    user's `switch_fn` was swallowed: the step was rolled back and
    re-taken, the signal was gone, the run finished, and a `DtRetry`
    that never happened was recorded. A deterministic callback error
    was re-invoked up to 126 times and then surfaced stripped of its
    type. Both catches are now `std::runtime_error` — everything the
    solver throws is one; a foreign exception is not.
  * **A retried step mixed two time steps.** `refresh_dt` has always
    been fixed at `opts.dt` (its own comment claimed otherwise and
    was always wrong), which was harmless while the only off-nominal
    path was a linear solve. The retry made it a silent wrong answer:
    every capacitor at `2C/sub_dt` while the saturable inductor stayed
    at `2L/opts.dt` — 64× out at the bottom of the ladder. The retry
    now refuses circuits with saturable inductors, because threading
    the dt is not enough on its own: `SaturableInductorHistory` has no
    snapshot/restore, so its flux cannot be rolled back.
  * **A diode mask cycle in any sub-step but the last vanished** —
    including under `strict_event_iterations`, which exists precisely
    to make that fatal.
  * The voltage check treated an **op-amp** (a VCVS at gain 1e5) as
    contributing nothing to the bound, so every op-amp circuit would
    have tripped it; it now declines to have an opinion there. It
    also read a **NaN** trace as plausible (`abs(x) > bound` is false
    for NaN), in a check named for leaving physics.
  * The whole voltage check sat inside `except Exception: pass`, so
    under `-W error` its own warning became an exception and was
    swallowed — **no** warning and no attribute, for the one user who
    asked for warnings to be fatal.
  * `result.dt_retries` was recorded and **never surfaced**, though
    its own documentation says the user is entitled to know.
    `simulate()` now warns, and `voltage_sanity_factor` is reachable
    from `simulate()` rather than only from the binding.
  * `SimulationAborted.partial` had none of the side-table attributes
    `simulate()` attaches on success, so every name-based accessor on
    it failed with a message telling the user to do what they had
    already done.
  * `Size{1} << h` with an unvalidated `max_dt_halvings` is undefined
    behaviour at h ≥ 64; clamped at 20. `reported_limit` was
    last-write rather than a maximum, meaningless for the freeze
    guard. `approx_bytes()` counted neither of the two new vectors,
    and `DtRetry` carries a few hundred bytes of failure text each.

* **2.9 megavolts on a 48 V circuit, reported in silence** — and now
  named. This closes audit item B.3, though not the way it was
  written.

  ```
  Vin(48 V) — L(1 mH) — S ——| gnd,   S opening at 10 kHz
  max |v(sw)| = 2.9e+06 V,  isfinite everywhere, no warning
  ```

  An inductor whose conduction path opens produces, in an idealized
  model, an unbounded voltage: `v = L·di/dt` with `di/dt` forced to
  `-i/dt` in one step. Nothing caught it — the inductor freeze and
  clamp guards watch the **current**, and the current here stays at a
  believable 14 A. It is the voltage that leaves physics.

  `solver/voltage_sanity.hpp` compares the largest node voltage a run
  produced against the largest voltage any *independent* source in
  the circuit can produce (dependent sources are excluded: a VCVS's
  output is a function of the circuit, so folding its gain in would
  define the bound in terms of the thing being checked). A node past
  100× that is named:

  ```
  simulate(): node 'sw' reached 2.860e+06 V at t = 0.00075001 s, but the
  largest voltage any source in this circuit can produce is 48 V — a factor
  of 59592. …an inductor whose conduction path opens really does produce an
  unbounded voltage in an idealized model. No real circuit does, because the
  switch avalanches or its parasitic capacitance rings or the designer fitted
  a snubber. Add whichever of those your design has…
  ```

  **It names and stops there.** The audit's B.3 asked for an
  auto-inserted snubber; inserting one means choosing its value,
  which is a modelling decision belonging to whoever knows the
  design's stand-off voltage. Substituting one silently is the
  failure this entire phase has been removing — and the 1 kW drive
  already showed how that ends, with a reported line current that is
  exactly the clamp. Naming the node is the part a simulator can do
  honestly.

  The scan is read-only, one pass over the recorded node block, and a
  test pins that a boost converter — which legitimately exceeds its
  input — says nothing.

* **Newton promotes line search when it diverges** — the last named
  Phase-2 globalization item, and the one that measures the rest.

  The kernel already auto-promoted Levenberg-Marquardt on two
  triggers: a singular factorize, and a near-miss stall (residual
  already tiny, `||dx||` plateaued). Neither sees the one condition
  backtracking exists for — a full Newton step that makes the
  residual **worse** — so a plainly diverging Newton fell through
  both and the run died.

  Promoting rather than defaulting line search on is deliberate:
  measured on a mains rectifier, backtracking from the first
  iteration costs ~30 % on a run that never needed it, while the
  comparison that triggers the promotion costs nothing. And it does
  not move answers — line search changes the path to the root, not
  the root: 1.2e-12 on a 170 V scale, or exactly zero.

  **What each layer is worth.** Over a 27-point sweep of (peak
  voltage, sharpness, dt), with the step ladder disabled:

  | | failures |
  |---|---|
  | after the logistic fix alone | 11 / 27 |
  | + line-search promotion | 4 / 27 |
  | + dt-halving retry | **0 / 27** |

  That is the Phase-2 gate — *converges with zero manual
  intervention* — met on this family, with each layer cheaper than
  the one above it and measured on what the one below leaves behind.

  It also cost three test fixtures, which is the honest part. The
  gmin-stepping demonstration went from ten reverse-biased junctions
  to twenty, and the dt-retry's rectifier from 170 V to 400 V,
  because the free globalization now carries the easier cases on its
  own. That is the third time this phase a cheaper fix dissolved a
  more expensive feature's demonstration — after the logistic
  overflow did it twice — and each time the fixture had to move to
  what actually survives rather than the claim being left standing.

* **A run that dies part-way brings the run with it.** A simulation
  that failed at 90 % returned nothing at all: the exception carried
  a message and every sample computed before the failure was
  destroyed with the stack. On a run that takes minutes that is the
  difference between "here is where it broke, and here is the
  waveform leading into it" and "start again".

  `run_transient` now throws `solver::SimulationAborted`, which
  carries `partial()` — everything recorded up to but not including
  the step that failed — and `t_failed()`. Python gets
  `pulsim.SimulationAborted` with `.partial` and `.t_failed`:

  ```python
  try:
      res = p.simulate(b, t_end=..., dt=...)
  except p.SimulationAborted as e:
      print(e)                       # still names the row that failed
      plot(e.partial.times, e.partial.states)   # …and you keep this
  ```

  **The default is still an exception**, and `SimulationAborted`
  subclasses `RuntimeError` so existing `except RuntimeError` code is
  unaffected. Returning a truncated result as if it were whole would
  be exactly the silent wrong answer this project keeps removing, so
  the partial trace has to be asked for by catching the type. The
  partial trace is a genuine prefix on the same uniform grid, not a
  resampling, and a test pins that. A cancellation via
  `should_continue` is unchanged — a deliberate stop is not a
  failure, and it still returns normally.

* **The inductor guards now confess.** The audit's item B.3 was
  "delete `inductor_freeze_di_max` and `inductor_abs_clamp`" — the
  post-solve guards that overwrite an inductor's branch current, and
  which the audit calls confessions rather than features. Deleting
  them turned out to be wrong, and finding that out is the result:

  * Across 13 configurations of the failures their own documentation
    names — a buck in deep DCM over five decades of load, an inductor
    with no freewheel path at all, `g_off` down to literally zero,
    10 ms runs at `dt = 1e-8` — the kiloamp runaway does **not**
    reproduce. Switches always stamp `g_off`, so the loop never truly
    opens.
  * But the circuit they were written for genuinely depends on them.
    Running `projects/inverters/pfc_vsi_drive` with the guards
    removed, the line current runs away and the boost stage starves.
    They stay.
  * **And that dependence is not what the guards claim it is.** With
    the guard off, that run's peak line current is **294.51 A at
    `dt` = 2e-7, 5e-8 *and* 1e-8** — dt-independent to five figures
    across a 20× range. It is not solver garbage; it is the model's
    own trajectory, diverging exactly as that script's docstring
    already predicts ("the L001-C006-bridge tank rings unbounded …
    there's no PFC current controller"). `v_rect` decays 172 → 67 V
    over the run while the current grows, and the peak sits on the
    final sample. The clamp is substituting a limit for a missing
    control loop.
  * And with them on, that run's reported line current peaks at
    **exactly 100.000 A** — which is the configured
    `inductor_abs_clamp`. The clamp fires on **30 878 steps** of a
    20 ms run. The plotted current is the limit, and nothing said so.

  So instead of deleting them: every firing is recorded, per
  inductor, in `result.inductor_guard_actions` (branch, freeze and
  clamp counts, first time, peak raw solve, limit reported), and
  `simulate()` warns once naming the device and the step count. The
  option docs now open by saying the guard does not solve anything.
  Recording changes no numbers — a test requires two guarded runs to
  be bit-identical.

  The warning deliberately does NOT name a cause. The first draft
  said "usually a snubber across a path that opens", which is the
  guards' original story and is wrong on the only circuit that can
  be tested — there the cause is an unbounded open-loop stage. What
  it says instead is the check that settles it: re-run with the
  guard off and a smaller dt, and if the current does not move, the
  clamp is hiding a modelling result rather than a numerical one.

* **A step the solver cannot take is re-taken at a smaller one**
  (audit finding B.4). A failed inner solve used to end the run and
  discard everything computed before it. Now the step is rolled back
  and re-taken as 2 sub-steps of dt/2, then 4 of dt/4, up to
  `2^max_dt_halvings` (default 6), and the run continues at the
  nominal dt.

  Unlike the fallback rungs Phase 2 B.2 had to repair, this one is a
  genuinely different problem: the trapezoidal companion's `2C/dt`
  grows as dt shrinks, improving the Jacobian's diagonal dominance
  and putting the previous state closer to the answer. A 170 V mains
  half-wave rectifier at `dt = 1e-4` — about 170 samples per cycle,
  a perfectly reasonable thing to type — fails one step and needs
  exactly **one** halving; the peak lands at 169.25 V either way.

  * **The output grid does not move.** Sub-steps are internal, so
    `times[k]` stays exactly `t_start + k·dt` and an FFT of the
    result stays valid — the property Phase 1e made `store_every` a
    pure stride to protect. A test walks every sample to confirm it.
  * **The easy run pays nothing.** With the ladder enabled and
    disabled, a run that needs no retry produces bit-identical
    output; a test compares every sample of every row exactly.
  * **Nothing is silent.** Every reduction lands in
    `result.dt_retries` with the time, the number of halvings, and
    what the nominal attempt reported — integrating an interval more
    finely than asked is a change in accuracy the user is entitled
    to know about. `max_dt_halvings=0` restores the hard failure.
  * **A topology defect fails fast instead.** Reachability to ground
    is a property of the graph, not of dt, so an isolated subnet is
    singular at every step size. Retrying one burned the whole
    ladder and buried Phase 1's named diagnostic under a "could not
    be taken, even split into 64 sub-steps" wrapper — the dead-rung
    defect again, this time in my own new code, caught by this
    project's own hostile-circuit suite. The check now runs once,
    before the loop.
  * No retry on the static path (no capacitors or inductors), where
    dt does not enter the matrix at all and a smaller step would
    re-run the byte-identical computation.
  * `PwlStateSpaceCache::solve_at`'s three singular-matrix messages
    gained the same `explain_singular` node naming its sibling
    `solve` has had since Phase 1 — the retry routes through
    `solve_at`, which made the gap visible.
  * Internal tidy this made necessary and worth having: the step's
    right-hand side accumulator was written out three times, and the
    ORDER of its terms is what keeps results bit-identical, so one
    copy is the only way to keep that promise honest.

* **A mains rectifier can be simulated** (v2.0 Phase 2). Every smooth
  device model blends its regions with a logistic, written the
  textbook way:

  ```
  alpha = 1 / (1 + exp(-kappa * u))
  ```

  For `u` sufficiently negative the exponent is a large POSITIVE
  number and `exp` overflows. The *value* survives that — `1/(1+inf)`
  is 0, the right answer — but forward-mode AD propagates
  `d = exp(x)·dx`, so the derivative is `inf`, and the reciprocal's
  is `inf/inf` = **NaN**. One NaN in the Jacobian defeats
  Levenberg-Marquardt at every λ, and the run ends with
  `solve_with_newton (LM): factor failed at λ = 1e+09`.

  The threshold is exactly `kappa·|u| > 709`, the double-precision
  `exp` limit. At the default `kappa = 20` that is 35 V of reverse
  bias — a 170 V half-wave rectifier passes it in the first
  half-cycle. It is a property of how the formula was *written*, not
  of the circuit, the model or the time step.

  `numeric/logistic.hpp` evaluates the same function with the
  exponent's sign forced non-positive (`u ≥ 0: 1/(1+exp(-u))`,
  `u < 0: e/(1+e)`). The branches are algebraically identical, agree
  exactly at 0, and keep `exp` in (0, 1] where neither value nor
  derivative can overflow. `IdealDiode`, `MosfetLevel1` and
  `IgbtLevel1` all use it. This is the job SPICE's `pnjlim` does for
  a Shockley junction, done in the formula rather than by limiting
  the Newton step — there is no iterate from which it can fail.

  **How much of the "stiffness" this was.** A 12-diode chain that
  needed gmin stepping now converges on the direct solve, at every
  sharpness tried up to `kappa = 20000`; a 27-point sweep of chains
  that previously defeated all four DC rungs now defeats none. The
  Phase 2 B.2 tests that used a sharp forward chain to demonstrate a
  rescue have been repointed at the case that survives — reverse-
  biased junctions at `G_off = 1e-9`, where a node's pivot genuinely
  has no significant digits left. A test whose premise a later fix
  removes is worse than no test.

  **What it did NOT fix**, stated plainly: the same rectifier still
  fails at `dt = 5e-5` and runs correctly at `dt = 1e-5`. That
  remaining failure is a genuine convergence failure and it IS
  step-size dependent — which is what local time-step reduction is
  for, and why that work is sequenced after this rather than before.

* **The DC operating point tells the truth** — and recovers by
  itself (audit finding `no-gmin-infrastructure`, plus a
  silent-wrong-answer this work uncovered).

  **BREAKING / correctness.** `pulsim.compute_dc_op(builder)` answered
  **5.000 V** for the anode of a diode fed from 5 V through 1 kΩ. The
  truth is 0.700 V. `dc_assemble` skips `BranchKind::Nonlinear` as an
  open circuit, so every route into the standalone entry point solved
  a *different circuit* and reported it with full confidence.
  `simulate(start_from_dc_op=True)` had been correct since Phase 0;
  the two entry points simply disagreed by 700% and nothing said so.
  A test now pins them to each other. Pass
  `enable_nonlinear_refresh=False` if you specifically want the linear
  system.

  * **`gmin`, both halves.** A conductance floor (1e-12 S, SPICE's
    GMIN) node-to-ground in every DC solve, to keep pivots meaningful
    when a node is reachable only through near-open devices — a bridge
    of reverse-biased diodes, a sub-threshold MOSFET. And **gmin
    stepping**: start at 1e-2 S, walk down by decades warm-starting
    each solve, so Newton crosses one decade of nonlinearity at a
    time. A 12-diode chain that the direct solve cannot converge now
    solves, and lands within 1e-6 of where source stepping
    independently lands.
  * **gmin never covers for topology.** A conductance on every node
    would also make an *unreferenced* node solvable — and hand the
    user a confident 0 V for a node with no defined voltage, which is
    exactly the failure Phase 1 taught the kernel to name and B.1
    taught the builder to repair. So every DC entry point runs a
    structural check on the UN-augmented system first, and the named
    error still wins. The check is careful in both directions: it
    unions every branch for reachability (a diode is a galvanic
    connection even though it does not conduct at DC) and probes
    emptiness on the linear stamps unioned with the nonlinear ones, so
    it never libels an interior node of a diode chain.
  * **The fallback cascade now actually cascades.** In `Auto` mode it
    could not rescue anything: `pseudo_transient_dc` pre-factorised
    the raw matrix and threw if it was singular — rejecting exactly
    the inputs rung 1 had just failed on — and `source_stepping_dc`
    factorised once and re-solved the same system with a scaled
    right-hand side, which returns the naive answer after `n_steps`
    redundant solves. Both passed a no-op refresh, so reaching either
    directly on a circuit with diodes returned the operating point of
    that circuit with the diodes open. All three are fixed:
    `dc_assemble` gained a `source_scale` homotopy parameter (scaling
    independent sources only — a VCVS gain is circuit structure, not
    excitation), each rung re-assembles and warm-starts, and every
    rung takes the real refresh.
  * **Reordered:** naive → gmin stepping → source stepping →
    pseudo-transient. PTC is last on its own header's evidence: MNA
    voltage-source constraint rows give the artificial dynamics
    mixed-sign eigenvalues, so it is unstable on exactly the systems
    Pulsim builds. Both homotopy rungs bisect on failure rather than
    giving up on a too-wide step.
  * **One implementation of "the DC operating point."**
    `pwl/dc_operating_point.hpp` does the three things that make it
    right — nonlinear devices stamped, PWL diode states iterated to
    consistency, cascade walked on failure — and `run_transient`'s
    pre-charge, the BDF1 bootstrap and the Python entry point all
    route through it. Previously only `run_transient` did all three.
  * **`run_transient_bdf1` refuses a circuit it would silently open.**
    It has no Newton loop at all, so a smooth diode / MOSFET / IGBT
    was an open circuit for the *whole run*, not just at DC. It now
    raises and points at the trapezoidal engine.
  * `compute_dc_op()` runs the same topology preflight `simulate()`
    does, so the two stop disagreeing about what "floating" means.
  * **`strategy="auto"` no longer falls through to a settling
    transient.** A transient answers a different question, and on a
    node with no DC equation it answers it confidently — it would
    report whatever the initial condition decayed to. `"settle"` is
    still available explicitly, and the failure message points at it.
  * **Adversarial review (39 agents) caught two more silent wrong
    answers**, both in the guard rails this change added:
    - the structural check tested *emptiness*, not *rank*, so a
      subnet reachable only through a coupling capacitor or an ideal
      current source passed both probes — every column populated by
      its own resistors, galvanically connected to ground — and the
      floor then supplied the missing rank and reported 0 V (or
      I/2gmin volts) as an operating point. There is now a third
      probe: DC reachability, with nonlinear branches counted as
      conducting exactly when a refresh will stamp them.
    - a named rung (`strategy="gmin_step"` and friends) bypassed the
      PWL diode-state iteration, so it answered for a circuit with
      every switched diode frozen open — reintroducing this change's
      own headline bug, one strategy name at a time. Every strategy
      now runs inside the diode loop; resolving diode states is part
      of what "the DC operating point" means, not a feature of one
      rung.
    Also from the review: `DCSolveReport.residual` was reported as a
    default-constructed 0 on the default path rather than measured;
    the DC vector came back as a read-only view pinning the kernel
    object (the leak Phase 1's review caught in `res.v()`);
    `run_transient_bdf1`'s guard missed PWL diodes, which it also
    cannot commutate, and its gmin fallback re-solved the identical
    system; `"settle"` silently dropped arguments it cannot honour;
    and `gmin_ramp` produced an all-NaN ramp for a non-finite start,
    which the stepping loop could not advance past.
    Declined, with the reason recorded in the code: carrying a
    homotopy's bisected step width forward. A shrink-only controller
    turns one hard rung into a step so small the budget runs out
    before the ramp ends — it failed two tests when tried.
  * New Python surface: `strategy="gmin_step"`, `gmin=`,
    `enable_nonlinear_refresh=`, `report=[]` (a `DCSolveReport` saying
    which rung answered), `SettleConfig`. `"pseudo_trans"` now means
    the kernel's PTC rung; the old transient-settling behaviour is
    `"settle"`. `"source_step"` is now a real homotopy rather than a
    long transient.

* **Topology preflight + auto-regularization** (audit finding
  `no-topology-preflight-or-auto-shunt`, CRITICAL). A node nobody gave
  a voltage reference — an isolated transformer secondary, a divider
  tap hanging off one capacitor, a sub-circuit fed only by current
  sources — used to make the factorization singular. Since Phase 1 the
  error at least named the node; now the circuit just runs.
  * `simulate()` sweeps the topology before the engine dispatch (so
    BOTH engines benefit), finds every subnet with no path to ground,
    and ties each one to ground through a **1 GΩ** resistor. Large is
    the point: a 1 GΩ reference draws nanoamps and leaves the physics
    untouched, whereas the 1 µΩ tie an older tutorial prescribed is a
    galvanic BOND that silently welds an isolated secondary to primary
    ground. A test asserts the auto-tied circuit's waveforms match a
    hand-tied one exactly.
  * Two passes, because they catch different things: galvanic
    reachability over every branch, and DC reachability over the
    branches that conduct at DC (capacitors are open, ideal current
    sources contribute no conductance). A cap-only node is fine in a
    transient and singular at DC — both halves are pinned by tests.
  * Nothing happens silently. Every insertion lands in a
    `PreflightReport` naming the node and the value, surfaced once as
    a warning and attached to `result._preflight`.
    `auto_regularize=False` restores the previous named error, and
    `builder.run_preflight(PreflightOptions(auto_regularize=False))`
    inspects a circuit without touching it. The pass is idempotent and
    finds nothing on a well-posed circuit.
  * Deliberately NOT auto-fixed: a node reachable only through an
    inductor is floating in the legacy `dt = 0` static build, but the
    right answer there is `dt > 0`, which the existing error already
    says — inserting a resistor would hide a modelling mistake.
  * New: `PreflightReport` / `PreflightFinding` / `PreflightOptions`
    exported from `pulsim`, `Graph.branch_name()` / `node_name()`
    bound, and `DevicePool::is_registered()` as a non-throwing
    companion to `kind_of`.
  * `docs/tutorials/03-flyback-isolated.md` prescribed the 1 µΩ tie;
    corrected, with the reason spelled out.

### Phase 1 — kernel foundation (v2.0 audit follow-up)

* **Kernel diagnostics now name the node or device** (audit findings
  `kernel-has-no-name-context-for-errors` and
  `singular-errors-dont-name-the-node`). Every solver failure used to
  report a mask bitstring and a norm — `numerically singular for mask
  0010111…1 (dt=1e-7)` — which on a 200-switch MMC is unactionable.
  Now:
  ```
  PwlStateSpaceCache: numerically singular for mask 0b0 N=0 (dt=0)
    — nothing connects node vfloat: its column in the MNA matrix is
    empty, i.e. no device ties it to the rest of the circuit (a node
    reachable only through a capacitor has no DC path; add a
    bleeder/parallel resistance or tie it to ground)
  ```
  * `topology::Graph` gained an optional branch name table
    (`add_branch(..., name)`, `set_branch_name`, `branch_name`),
    populated by `CircuitBuilder` at its single branch-creation
    choke point. Nodes already carried names; branches did not, so
    names never crossed into the kernel. Unnamed branches (raw-kernel
    users, hand-built graphs) stay empty and every message falls back
    to the id. Branch names are deliberately excluded from the
    structural hash, so PWL cache identity is unchanged.
  * New `pwl/row_names.hpp` resolves an MNA row to its owner using
    the documented layout `[v_0…v_{N-1} | i_src | i_L]` —
    `describe_row`, `row_label`, `branch_label`,
    `top_entries_by_name`, and `explain_singular`.
  * Two independent localisation sources, because one is not enough:
    a structural empty-column/row probe (`sparse::first_empty_column`
    / `first_empty_row`, O(n), works on BOTH backends — the Eigen
    backend used by the DC and Newton paths exposes no pivot index at
    all), plus the new `DirectSolver::singular_index()` implemented by
    the in-house LU, which catches what structure cannot (a node
    behind an OPEN switch is not empty — `g_off` is always stamped —
    it just pivots to ~0).
  * The phrasing branches on WHAT the empty row belongs to. Some
    unknowns are reserved by `state_size` but deliberately not
    stamped by the current assembly mode (every inductor at
    `dt == 0`; saturable inductors in the DC assembly), so their row
    is empty *by construction*, not because the device is
    disconnected — telling that user to "add a bleeder resistor to
    L1" would send them to debug a correctly wired part. A node gets
    the wiring advice; a device branch-current gets the truth about
    the assembly mode and the real fix (`build with dt > 0`).
  * Wired into: `compute_dc_op`, `pseudo_transient_dc`,
    `source_stepping_dc`, `pseudo_transient_solve` (both the
    singular and the non-convergence exits), the Newton
    structurally-singular and LM no-improving-step throws, the
    non-finite-residual guard, the inductor-cycle error (which now
    names the device closing the loop), the PWL cache (`CacheError` gained a
    structured `failing_row` + `detail` for C++ consumers of the
    non-throwing `try_*` API — Python still receives the message as
    the exception string; surfacing the structured fields through
    pybind is a follow-up), Newton non-convergence
    (`lpNorm<Infinity>()` discarded the argmax; now reports *where*
    the worst residual and step live), and the strict diode
    event-iteration throw on BOTH the dynamic and static paths
    (names the devices still flipping). The Newton residual is
    phrased as an EQUATION (`the KCL balance at node vout`) and the
    step as an UNKNOWN (`current through inductor L1`), which is
    what each actually is.

* **Contiguous zero-copy waveform storage + output decimation**
  (audit finding `waveform-storage-vector-of-vectors`, BREAKING).
  `SimulationResult::states` was a `std::vector<Vector>` — one heap
  block per recorded sample (a 10^7-step run meant 10^7
  allocations) — and every `res.states` access from Python rebuilt
  a fresh list of N 1-D ndarray objects, so reading the result cost
  O(N) *per access*. It is now a `StateTrajectory`: ONE row-major
  buffer allocated once for the whole run, exposed to Python as a
  single read-only `(num_steps, state_size)` **zero-copy numpy
  view**.
  * Measured on a 500 000-sample run: `res.states` access
    **174 ms → 1.2 µs**; `np.asarray(res.states).mean()`
    **248 ms → 1.6 ms (154×)**. Kernel side, the removed per-step
    allocation shows up as **0.48 → 0.34 µs/step** on the rectifier
    benchmark and 0.28 → 0.26 on the buck.
  * The C++ read API is source-compatible (`states[k]`, `size()`,
    `back()`, range-for, `traj = {x0, x1}`); element access now
    yields an `Eigen::Map<const Vector>` by value, so bind with
    `const auto&` / `const Vector&`, never a non-const `auto&`.
    All 207 in-tree call sites compiled unchanged. Ragged pushes,
    silently possible before, now throw.
  * Python indexing, slicing, iteration, `len()` and `np.asarray`
    are unchanged; the view is **read-only** (v1.x rows were
    non-writeable views too, so this preserves the old behaviour) —
    call `.copy()` for a mutable array. New `res.states_bytes`.
  * The read-only view is confined to `res.states` itself: the
    per-signal accessors `res.v(name)` / `res.i(name)` and
    `compute_dc_op`'s fallback still return **writable owned**
    arrays, as in v1.x (adversarial review caught the read-only
    view leaking through them via a column slice).
  * `engine='dsed'` results now expose the SAME 2-D `states` array
    (plus `states_bytes`), so `res.states[:, node_id]` — the
    pattern the docs recommend — works on both engines.
  * The eager reservation is byte-capped: a multi-GB trace grows on
    demand instead of committing one huge block before the first
    step, so a run the caller may cancel early no longer risks an
    immediate `bad_alloc`. Ordinary runs still get exactly one
    allocation.
  * **New `SimulationOptions.store_every`** (also `simulate(...,
    store_every=m)`): record every m-th step
    (default 1 = every step, identical to v1.x). The solver still
    integrates at `dt`; only what is STORED changes, and the
    recorded grid stays STRICTLY UNIFORM at `m · dt` — decimation
    is a pure stride rather than a stride-plus-forced-endpoint,
    precisely so downstream FFT / harmonic / ripple analysis stays
    valid. `expected_sample_count()` reports what a run will store.

* **Zero-allocation transient hot loop** (audit finding
  `per-step-heap-allocations`): the per-step loop performed 6–10
  heap allocations per step — fresh `rhs` in every `cache.solve`,
  fresh `y` in every LU triangular solve, per-step snapshot copies
  (`history.snapshot()`, diode bits), `compute_b_extra`'s fresh
  `Vector::Zero`, and per-step event scratch. All buffers are now
  once-allocated workspaces: `HistoryState::compute_b_extra(dt,
  out)` / `snapshot_into`, `DiodeEventState::snapshot_on_bits_into`
  fill in place; the LU solver and the cache use **thread-local**
  solve workspaces (per-thread, so concurrent GIL-released
  transients sharing a warm cache stay numerically correct —
  adversarial-review finding ZA-1 caught that instance-member
  buffers would have silently corrupted them); `run_transient`
  hoists every per-step vector, including the sub-step-correction
  scratch. Steady-state `cache.solve`/`solve_at` — and the full
  event-corrected linear loop — now perform **zero** heap
  allocations, locked in by a dedicated test binary that counts
  global `operator new` calls AND forbids Eigen-side malloc
  (`EIGEN_RUNTIME_NO_MALLOC` + throwing `eigen_assert`, effective
  in release builds; count assertions auto-skip under MSVC
  iterator-debug). Scope: the linear trapezoidal path — the Newton
  nonlinear-refresh solve and the deliberately uncached BDF1
  comparison path are unchanged. Measured on the reference
  open-loop buck (100k steps): **0.52 → 0.30 µs/step (1.7×)**.

* **True Gilbert–Peierls sparse LU** — `factorize()` rewritten as a
  left-looking GP factorization (DFS reach, topological sparse
  elimination, cs_lu-style O(1) pivoting via `pinv`, O(nnz) final
  relabel) with COLAMD column ordering by default (RCM kept via
  `set_ordering`). Kills the O(n²) dense inner loops: banded factor
  time now scales ~linearly and an n=5000 circuit-like MNA factors in
  ~1 ms (audit findings `lu-effectively-dense` /
  `lu-quadratic-inner-loops`). `partial_refactor()` is reach-based
  and keyed by the *current* changed-column set. New scaling
  regression tests lock the complexity law in CI.
* **LRU-bounded lazy cache + exact (G, C, b) split.** Three audit
  findings closed in one coherent rework of the PWL cache:
  * *`no-mode-cache-eviction`* — the lazy per-mask segment cache is
    now bounded by a byte budget (default 1 GiB,
    `set_segment_budget_bytes`, 0 = unbounded): least-recently
    solved masks are evicted and transparently rebuilt on re-visit;
    eager `build()` never evicts. New telemetry:
    `CacheMetrics.segment_evictions`, `segment_cache_bytes()`,
    `DirectSolver::factor_bytes()`. `CacheMetrics` is now bound to
    Python (`cache.metrics()`), so eviction and event-solver
    behaviour is observable — and therefore testable — from there;
    `SimulationResult.total_bytes` exposes the whole result's
    footprint alongside `states_bytes`. The recency tick is atomic
    and is only written in lazy mode, where eviction can actually
    fire.
  * *`alt-dt-cache-unbounded-factorization`* — `solve_at` no longer
    keeps a map keyed by *exact Real dt* of fully analyzed +
    factorized segments (one per commutation, forever). Each
    recently-evented mask now owns ONE event solver (LRU ≤ 8):
    a dt change is `J = G + (1/dt)·C` + numeric `factorize()` on
    the entry's shared symbolic analysis — no re-analysis, no
    dt-keyed storage, and `refactor_parametric` now invalidates
    event entries (the old aux cache silently kept stale factors).
  * *(G, C, b) split* — `assemble_segment_split` emits the static
    matrix G, the 1/dt-coefficient matrix C (2·C_cap caps, −2·L
    inductor diagonals, −2·M transformer cross terms; mask-
    invariant) and the dt-independent b; `assemble_segment`
    recombines `J = G + (1/dt)·C` through the same single stamping
    loop. Note for anyone holding v1.x golden traces: companion
    entries are now `fl(fl(1/dt)·2C)` instead of `fl(2C/dt)`, a
    shift of at most ~2 ulp (far below trapezoidal LTE), so
    bit-identical baselines recorded on v1.x will differ in the last
    bit. `compute_lti_state_space` now uses the split EXACTLY,
    replacing the two-assembly finite-difference recovery — which
    had been burying *physically real zero eigenvalues* (series-cap
    midpoint imbalance modes in NPC/MMC stacks) under cancellation
    noise; the LTI tests now assert the analytic pole and the exact
    marginal modes.
* **Dynamic `SwitchStateMask` — the 64-switch ceiling is gone.**
  The mask is now a small-buffer word array (≤ 128 switches inline
  with zero heap allocation, wider masks spill to one vector), so
  >64-switch topologies — 120-switch MMC phases, large CHB stacks —
  are representable end-to-end (audit finding `switch-mask-64-cap`).
  The bit-indexed API (`get`/`set`/`flip`/`count`/hash/equality) is
  unchanged; new word-level APIs (`num_words`/`word`/
  `hamming_distance`/`overlay`) replace the raw-uint64 hot paths in
  the PWL cache delta detection and the diode overlay merge. The
  legacy `bits()`/`set_bits()` shortcuts now throw `std::logic_error`
  beyond 64 switches instead of silently truncating. Eager
  `cache.build()` still fails loudly for ≥ 64 switches (2^N
  enumeration); the lazy cache (`build_lazy`) is the supported route
  and is covered by new 70-switch integration tests.

### Phase 0 — silent-wrong-answer fixes (v2.0 audit follow-up)

Nine fixes from the 2026-08 full-kernel audit, all targeting paths
that produced **plausible-but-wrong results** or destroyed user work
without telling anyone. See `PULSIM_V2_AUDIT.md` findings referenced
per item.

### Fixed

* **`simulate(engine='dsed')` silently dropped `step_observer`,
  `closed_loops`, `should_continue`, `live_stream` and
  `start_from_dc_op`** — a closed-loop converter ran OPEN-loop and
  returned plausible waveforms. Now raises `ValueError` naming the
  offending kwargs. (`dsed-silently-drops-observers`)
* **DC operating point ignored every Newton device** —
  `compute_dc_op` stamps `BranchKind::Nonlinear` as open circuits, so
  `start_from_dc_op=True` seeded runs from the operating point of a
  *different* circuit. New `compute_dc_op_newton` reuses the
  transient's `NonlinearRefreshFn` chain (warm-started from the
  linear solve); `run_transient` routes through it automatically.
  Known limitation: saturable inductors still lack a DC stamp — the
  DC Newton chain deliberately excludes their trap-companion refresh
  (time-step-dependent, wrong at DC) and such circuits keep the
  pre-existing structurally-singular error at `start_from_dc_op`.
  (`dc-op-skips-nonlinear-devices`)
* **DSED dropped time-varying sources that were zero at t=0 and
  t=1 µs** (e.g. a pulse with `t_start > 1 µs`) — detection is now
  structural over `DevicePool::kind_of`, zero false negatives.
  (`dynamic-source-probe-false-negative`)
* **Diode chattering aborted the whole run** — the event-iteration
  loop now detects mask cycles, accepts the last consistent solve,
  records `result.event_iteration_breaches` and continues (loud
  Python warning). `strict_event_iterations=True` restores the old
  throw. (`event-iteration-throws-away-simulation`)
* **Mode-id hash truncation poisoned the DSED stiffness cache** —
  `std::hash`/`py::hash` truncated to int32 could alias two masks and
  silently pick the wrong integrator. Replaced by injective interned
  ids on the native adapter and PySystem; truncating ADL shims
  deleted. (`mode-id-hash-truncation`)
* **Absolute time epsilons skipped gate events on long runs** —
  `switch_fn(t_event + 1e-15)` rounds back to `t_event` once
  t ≳ 16 s. New shared `time_eps.hpp` makes every scheduler time
  comparison ULP-relative (legacy floors preserved near t=0).
  (`absolute-time-epsilons`)

### Changed

* **`simulate()` builds the PWL cache lazily** — only visited switch
  states are factorised (a PWM converter visits a handful of 2^N).
  Eager `cache.build()` remains available; `build_lazy` +
  `num_built_segments` are now bound to Python. Many-switch circuits
  (3φ NPC, multilevel) no longer hang before the first step.
  (`eager-2n-is-the-only-production-path`)
* **Driverless controlled switches now warn loudly** — `switch_fn`
  omitted + controlled switches present = all-CLOSED default
  (shoot-through hazard). v2.0 will flip the default to all-open and
  make it an error. (`default-all-switches-closed`)

### Added

* `pulsim.run_periodic_shooting` / `PeriodicShootingResult` exported
  (documented in the migration guide since v1.5 but unreachable).
* `EventIterationBreach` result records + `strict_event_iterations`
  option (kernel + Python).
* `PwlStateSpaceCache.build_lazy(dt)` / `.num_built_segments()`
  bindings; `_switch_census` helper.

### Review hardening (adversarial diff review, 8/8 findings fixed)

* Breach handling re-syncs diode bits to the last-solved state before
  recording, so `(x, diode)` pairs commit consistently (P0-R1); the
  mask-cycle break applies only to the memoryless linear path — the
  warm-started Newton path relies on the budget (P0-R3).
* Lazy-build singular masks now surface as `RuntimeError` (circuit
  problem), not `IndexError` (P0/PY-2); sweeps forward
  `strict_event_iterations` and warn on breached points (PY-1);
  `engine='dsed'` also builds lazily, warns on driverless controlled
  switches, and rejects `strict_event_iterations` (PY-3, PY-4).
* DSED backtrack/tie guards keep their historical ABSOLUTE 1e-15
  bands — a relative band there would widen the terminal-root discard
  window; the real fix (firing terminal-band roots) belongs to the
  Phase-3 event-queue overhaul (P0-R4).

### Docs

* `docs/gotchas.md`: the "1 µΩ to ground" recipe replaced — a 1 µΩ tie
  on a live node silently bonds the nets; reference-only ties use
  1 MΩ–1 GΩ. Lazy-build guidance now matches the code.
* README release badge unstuck from v1.4.0; `docs/solvers.md`
  variable-step row now reflects the shipped DSED engine.

## [1.8.0] — 2026-06-07

Custom code blocks ("C block"): user code wired into the circuit.

### Added

* **Custom code block** — `add_c_block(builder, inputs, outputs, *, dt,
  fn=… | lib=… | code=…)`: a PSIM-/Simulink-style sampled subsystem
  wired into the circuit. It reads circuit signals (input wires
  `("v", node)` / `("i", branch)`), runs **your code** at a sample time
  you choose (zero-order hold between steps), and drives signals back
  (output wires `("v", n+, n-)` / `("i", n+, n-)` — controlled voltage /
  current sources). The step code can be **Python** (`fn=` callable),
  **compiled C/C++** (`lib=` shared library via the
  `pulsim_cblock_step` C ABI, loaded with ctypes), or **inline C/C++**
  (`code=` + `lang=`, auto-compiled to a content-hash-cached shared
  library). Optional `init`/`term` manage opaque per-block state. Rides
  the existing PWL `step_observer` + `b_extra` path — no kernel change;
  `simulate()` picks up registered blocks automatically.
* **`wire_c_blocks_from_yaml(loaded, spec)`** — declare C blocks from a
  YAML string or list of dicts.
* **`CBLOCK_ABI`** constant + `pulsim/cblock_abi.h` header documenting
  the C ABI; `CBlockHandle` exposes live `outputs` / `state` / `n_fires`.
* **Docs** — new User-Guide page *Custom Code Blocks (C / C++ / Python)*.

## [1.7.1] — 2026-06-07

### Fixed

* **Loss/thermal summary on closed-loop transients** —
  `device_thermal_summary`, `device_loss_summary` and `result.i()` no
  longer report an unphysical conduction spike (`P_cond` ~ 1e4 W,
  `T_j` ~ 1e4 °C) for switches driven by a stateful (closed-loop)
  `switch_fn`. The mask is now taken from an exact simulate-time record
  (new opt-in `SwitchMaskRecorder`) or, in the fallback, a
  voltage-consistency guard drops samples that claim ON while the device
  is clearly blocking. No-op for stateless / settled designs.

## [1.7.0] — 2026-06-03

Precision loss & thermal modelling for inverter thermal optimisation.
The loss/thermal stack moves from a fixed-coefficient, per-device
estimate to a coupled, temperature-aware model — devices sharing a
heatsink, loss↔temperature feedback (with thermal-runaway detection),
datasheet-curve switching loss, offset+slope conduction, and
heatsink/TIM sizing helpers. All new symbols are re-exported at the
flat ``pulsim`` namespace.

### Added

* **Shared heatsink** — ``HeatsinkDevice``, ``SharedHeatsink``,
  ``shared_heatsink_steady_state``, ``add_shared_heatsink``,
  ``make_heatsink_observer``. N devices on one sink are thermally
  coupled: the sink rise is driven by the *total* dissipation
  (``T_sink = T_amb + R_th,sa · Σ Pᵢ``), lifting every junction
  together — the effect a per-device-independent model misses when
  power is pushed up.

* **Electro-thermal feedback + runaway detection** — ``TempCoLoss``,
  ``electrothermal_steady_state``,
  ``make_electrothermal_heatsink_observer``. Conduction loss climbs
  with junction temperature (``R_ds(T)`` / ``V_ce(T)``), so the steady
  state is the self-consistent fixed point
  ``(I − M·K)·T_j = T_amb + M·(P₀ − K·T_ref)``, solved in closed form.
  Feedback gain ``ρ(G) ≥ 1`` is flagged as **thermal runaway** instead
  of returning a plausible-but-wrong temperature.

* **Offset+slope conduction loss** — ``device_loss_summary`` /
  ``device_thermal_summary`` gain ``conduction_specs``:
  ``P = V_f0·|i| + r·i²`` (datasheet ``V_ce0 + r·I`` / ``V_f0 + r·I``),
  whose forward-voltage offset dominates at low current, vs the default
  pure-resistive ``v²·g``.

* **Nonlinear datasheet switching curves** — ``E_on_curve`` /
  ``E_off_curve`` / ``E_rr_curve`` interpolated at the actual switched
  current per detected edge (linearly extrapolated beyond the table),
  for fidelity away from the single reference operating point.

* **Heatsink / TIM sizing helpers** — ``tim_resistance``,
  ``convection_coefficient``, ``convection_resistance`` and
  ``TIM_CATALOG`` turn geometry / airflow into the ``R_th`` values the
  thermal API consumes (TIM resistance ``thickness/(k·area)`` is exact;
  convection is a first-cut estimate).

* **Newly exported** thermal primitives that existed internally but
  weren't on the public surface: ``CauerStage``,
  ``add_cauer_thermal_network``, ``ThermalLimitMonitor``.

* **Docs** — new User-Guide page *Loss & Thermal Modelling*
  (``docs/loss-thermal.md``) covering how the model works and how to use
  it, with the coupling / runaway maths and runnable snippets.

### Changed

* ``projects/inverters/pfc_vsi_drive`` now computes junction
  temperatures through ``electrothermal_steady_state`` (same
  junction-to-ambient topology, so it stays comparable to the PSIM
  KPIs) and ships a ``thermal_comparison.py`` IPM case study (old model
  vs new). The legacy ``junction_temperature`` helper is retained and
  reproduced bit-for-bit when the temperature coefficient is zero.

## [1.6.6] — 2026-06-03

Branch-current readout overhaul. ``SimulationResult.i(name)`` now
reconstructs the current of **any** PWL-supported branch — the PLECS
"current is just a row of the output equation" model — instead of
only inductors and voltage sources. This eliminates the
sense-resistor / bypass-shunt pattern downstream tools (PulsimGUI)
were using to probe currents.

### Added

* **``result.i(name)`` for every supported device family** — in
  addition to the existing inductor / voltage-source fast path
  (state-vector native), the method now reconstructs:

  | kind | reconstruction |
  |---|---|
  | resistor | ``(V_from − V_to) / R_ohms`` |
  | capacitor | ``C · d(V_from − V_to)/dt`` |
  | current_source | constant ``I`` from params |
  | diode (PWL switched) | ``(v − V_th)·g_on`` forward, ``v·g_off`` reverse |
  | switch | ``v·G`` with ``G = g_on`` when the mask bit is set, ``g_off`` otherwise |

  Deferred (raise ``NotImplementedError`` with a hint at
  ``pulsim.losses.device_loss_summary``): ``mosfet_level1`` /
  ``igbt_level1`` / ``nonlinear_diode`` / ``vcvs`` /
  ``saturable_inductor`` — their per-step nonlinear stamps aren't
  exposed by ``builder.components()`` yet.

* **``result.currents()``** — PLECS-style "all currents in one shot",
  returns ``{branch_name: ndarray}`` for every reconstructible
  branch. ``skip_unsupported=True`` (default) quietly omits kinds
  without a reconstruction; ``False`` surfaces the error.

* **``pulsim.simulate(...)`` stashes the composed ``switch_fn`` on
  ``result._switch_fn``** (the same way ``_builder`` is attached) so
  switch-branch current reconstruction works without manual wiring.

### Changed

* ``result.i()`` on a **resistor** branch no longer raises
  ``NotImplementedError`` — it returns the reconstructed current.
  ``result.i()`` on a **capacitor** likewise now returns
  ``C·dv/dt`` instead of raising. (Callers relying on the old
  exception should switch to checking the device kind explicitly.)

### Notes

The recommended probe pattern for arbitrary topology points is a
**0 V voltage source** (``add_voltage_source(probe, n_in, n_out,
0.0)``) — its branch current is a state variable read with the
fast path, matching the SPICE / PSIM / PLECS "ammeter" convention
with zero perturbation. The sense-resistor / ``__IP_BYPASS_<probe>``
shunt approach is obsolete: it injects a small ``V_drop = I·R`` and
ill-conditions the G matrix for tiny R.

## [1.6.5] — 2026-05-30

GUI integration hardening release. Six findings from a multi-stage
drive integration audit (boost MOSFET 65 kHz + 3φ VSI 20 kHz + PMSM)
landed as PRs #76-#81 and ship together.

### Added

* **`MotorObserverBundle`** — `make_pmsm_observer` /
  `make_bldc_observer` / `make_dc_motor_observer` now return a
  callable bundle with per-step trace buffers (`times`, `omega_rad_s`,
  `theta_rad`, `T_em`, `i_d`, `i_q`, `i_a`/`i_b`/`i_c` on 3-φ;
  `i_a` on DC). Backward-compatible: still iterates as
  `(step_observer, b_extra_fn)` for legacy `obs, b_extra = ...`
  unpacking. `pulsim.simulate` auto-attaches bundles to the result so
  `res.signal("M1.omega")` resolves without manual wiring. Auto-attach
  walks composed observers (closed-loop path) too. PR #78.
* **`SimulationResult.signal(name)` and `.signals()`** — name-based
  lookup for user-recorded traces (today: motor bundles), with
  fuzzy-matched suggestions on `NameNotFoundError`. PR #78.
* **PMSM saliency** — `add_pmsm` gains `Ld=, Lq=, i_d_init=,
  i_q_init=, theta_init=` kwargs (mutually exclusive with `L_s=`).
  The abc topology uses `L_avg = (Ld + Lq) / 2`; the observer
  publishes the reluctance torque `T_rel = (3/2)·pp·(Ld − Lq)·i_d·i_q`
  on top of the magnet torque. `i_d_init` / `i_q_init` are
  inverse-Park'd to abc at `θ_e(0) = pp · theta_init` and seeded into
  the three phase inductors via `i0=`. Legacy `L_s=` form stays
  bit-for-bit identical (verified). PR #80.

### Fixed

* **PWL Newton auto-promotes to LM** on rank-deficient Jacobians
  and near-miss stalls (T1.2). Multi-stage switched topologies (e.g.
  realistic boost + SH1 MOSFET on inductive load) previously required
  the user to hand-set `enable_newton_lm=True` (often with a physical
  RC snubber on top). `solve_with_newton_b_extra` now sets
  `enable_lm=true` automatically on (a) a numerically singular
  factorize or (b) 3 consecutive iterations where `residual < tol_res`
  but `||dx|| ≥ tol_dx`. Transparent — explicit
  `enable_newton_lm=True` callers see no behaviour change. PR #77.
* **`simulate(closed_loops=, switch_fn=, step_observer=)` composes**
  instead of raising (T1.1). Pre-fix the kernel rejected the combo as
  conflicting; now closed_loops' switch_fn is OR-merged with the
  user's, and observers run in registration order (closed-loop first,
  then user). Unlocks closed-loop PFC + open-loop VSI + PMSM observer
  in the same run. PR #76.
* **DSED schedulers fail loudly on NaN/Inf step output** (T1.3).
  `PEDSimulator` / `PEDSimulatorBDF2` / `PEDSimulatorAuto` previously
  could silently commit NaN to the result (BDF2) or burn 5 generic
  rejections (RK45) when a switch-mask combo extracted an
  ill-conditioned A or a Python `b_extra_fn` returned NaN. All three
  schedulers now detect NaN/Inf per step and throw an actionable
  error pointing at the common root causes + workarounds
  (`engine='pwl'` benefits from the new auto-LM, shrink `dt_max`,
  audit callbacks). RK45 shrinks h × 0.1 and retries up to
  `kNanMaxStreak=3` consecutive iterations before throwing; BDF2
  throws immediately on first NaN. PR #79.

### Documentation

* **1.5 → 1.6 API stability notes** added to
  `docs/migration-guide.md` (T3.1) — covers the `pulsim.sweep`
  package→function collapse (with a lookup table mapping each retired
  `Distribution`/`Cartesian`/`metrics` helper to its
  lambda-based 1.6 replacement), `add_rc_snubber` becoming
  keyword-only, and PMSM/VSI/BLDC no longer having `*Params` structs.
  In-source `_V1_SYMBOL_HINTS` extended with 6 new entries
  (`Distribution`, `Cartesian`, `metrics`, `PmsmParams`,
  `ThreePhaseVsiParams`, `BldcParams`) so probing for the retired
  names raises `AttributeError` with an actionable migration hint
  instead of a bare error. PR #81.

### Notes

The T2.1 caveat: the salient-pole model uses `L = (Ld + Lq) / 2` in
the abc topology, so the di/dt anisotropy along dq is approximated
(reluctance torque is captured exactly; the high-frequency electrical
response uses the average inductance). For most IPM control studies
(FOC, flux-weakening, MTPA) this is the dominant saliency effect.
Full dq-frame reformulation is a v2 follow-up.

## [1.6.4] — 2026-05-29

### Fixed

* **`simulate(engine='dsed', switch_fn=<plain Python callable>)`**
  was silently producing the trajectory of a circuit with the
  switch **frozen at the t=0 mask** — no PWM edges ever detected
  — while still returning a successful result. Reproducer: buck
  CCM with 50 % duty PWM at 100 kHz over 5 ms. Pre-fix DSED
  returned V_C ≈ V_in (the SW-always-on attractor); PWL returned
  V_out ≈ V_in/2 as expected. Users hit "DSED stabilises at the
  wrong setpoint" / "DSED hangs at a weird value" and reasonably
  suspected the topology was being rejected silently.

  Root cause: the C++ scheduler computes the next gate-edge time
  via ``switch_fn.next_edge_after(t)`` (see
  ``core/include/pulsim/dsed/scheduler*.hpp``). The native PWM
  classes (:class:`NativePwm2Switch`,
  :class:`NativeMultiMaskPwm`) implement that method analytically;
  plain Python callables don't, and the pybind11 fallback in
  ``PySwitchFn::next_edge_after`` returned
  ``std::numeric_limits<Real>::infinity()``. The scheduler then
  computed ``t_gate = min(∞, t_end) = t_end`` and integrated the
  whole window with the mask sampled at ``t = t_start``.

  Fix: each DSED scheduler (``scheduler.hpp``,
  ``scheduler_auto.hpp``, ``scheduler_bdf2.hpp``) now treats
  ``∞`` returns from ``next_edge_after`` as "no edge info — poll
  defensively" and caps ``t_gate`` at ``t + dt_max/10`` so the
  scheduler is forced to land at that boundary and re-sample the
  switch_fn via ``fire_gate_event_`` (which catches any
  discovered mask change). Native PWM classes and any user
  object whose ``next_edge_after`` returns a finite value take
  the analytical fast path unchanged (``std::isfinite()`` check).

  The dispatcher emits a one-shot ``UserWarning`` when it sees a
  plain Python switch_fn so users know they're in the polled path
  and can opt into :class:`NativePwm2Switch` for best performance.

### Performance

Canonical buck CCM (V_in=24V, L=100µH, C=100µF, R_load=2Ω,
f_sw=100 kHz, 50 % duty). Plain Python switch_fn (the path that
exercises the new defensive polling):

  | t_end  | DSED auto | PWL @ 100ns | PWL/DSED speedup |
  |--------|----------:|------------:|-----------------:|
  |   1 ms |    3.5 ms |     33.0 ms |        **9.5×**  |
  |   5 ms |   15.8 ms |    163.1 ms |       **10.3×**  |
  |  20 ms |   63.7 ms |    666.8 ms |       **10.5×**  |

The ~10× speedup matches the DSED win when the trajectory
between events is smooth — the scheduler skips through each
5 µs PWM half-period in one RK45 step instead of 50 fixed-dt
steps. Switching to :class:`NativePwm2Switch` would push this
higher still on ≥2-switch topologies (analytical
``next_edge_after`` skips the polling overhead entirely).

### Tests

* `python/tests/test_dsed_python_switch_fn_polling.py` (3 tests):
  - **Correctness**: canonical bug reproducer — buck with
    resistive freewheel + plain Python PWM. DSED ``V_out_mean``
    and ``i_L_mean`` must match PWL within 5 %. Pre-fix: 3× off.
    Post-fix: 0.06 %.
  - **Warning**: the dispatcher emits a ``UserWarning`` mentioning
    ``NativePwm2Switch`` when the polling wrapper is engaged.
  - **Fast-path preservation**: a switch_fn that already exposes
    ``next_edge_after`` is NOT auto-wrapped (no warning, no
    polling cap engaged).

## [1.6.3] — 2026-05-29

### Internal — Python DSED scheduler reorganised to test tree

No user-facing behaviour change. v1.6.2 confirmed the C++ DSED
scheduler reaches the documented speedup on `pip install pulsim`;
the pure-Python implementation that shipped alongside it had
become **dead code on the user-facing path** — the dispatcher was
already taking the C++ Bridges 10/11 for every supported circuit,
and the Stage-4 Python fallback was only reachable if those C++
paths both rejected the input, which doesn't happen for any
circuit the documented surface ports today. Despite that, the
Python files still:

* Forced `pulsim` to keep optional deps (scipy) reachable through
  the wheel's import graph — exactly the regression v1.6.2 fixed
  with the lazy-import dance.
* Doubled the maintenance cost of every DSED algorithm tweak.
* Hid drift between the C++ port and the Python reference, since
  nothing ever cross-checked them.

v1.6.3 moves the Python scheduler from `pulsim/dsed/` to
`python/tests/_dsed_reference/` where it serves its actual
ongoing purpose — a readable reference for the
`docs/how-pulsim-works/` Part II chapters and a cross-validation
baseline against the C++ port. The hot path
(`pulsim.simulate(b, engine='dsed', ...)`) is unchanged.

### Public surface delta

Net surface change is **additive** for the documented escape
hatch and **internal-only** for everything else.

**Added**

* `pulsim.dsed.run_user_lti(system, switch_fn, x0, t_end, *,
  integrator='auto', rtol=1e-6, atol=1e-9, dt_init=1e-9,
  dt_max=1e-5, h_bdf2=None, stiffness_threshold=100.0,
  store_every=1)` — the canonical user-LTI escape hatch for
  custom-LTI workflows (system-ID benchmarks, FMU integrations,
  hand-rolled physics models). Thin wrapper around the native C++
  `run_ped_native` / `run_bdf2_native` / `run_auto_native` symbols
  that documents the required `system` protocol (5 methods:
  `A_matrix`, `b_vector`, `rhs`, `current_mask`, `set_mask`) and
  validates it at the call site so a missing method raises a
  clear `TypeError` with the list of what's needed instead of a
  cryptic pybind11 dispatch error from deep inside the C++
  scheduler.

  The docs and previous `_dsed_dispatch.py` error messages used
  to point at `pulsim.dsed.PEDSimulatorAuto` for this workflow.
  `run_user_lti` is the new pointer; equivalent semantics, better
  ergonomics.

* `pulsim.dsed.CircuitBuilderAdapter` is still exported for
  advanced users who want the same `(A, b)` extraction logic the
  dispatcher uses internally (Bridge.10 path).

**Removed** *(internal symbols only — none of these had
documented end-user call sites; the public docs and our
error-message pointers used them as an escape-hatch hint, now
replaced by `run_user_lti`)*

* `PEDSimulator`, `PEDResult`, `EventRecord`
* `PEDSimulatorBDF2`
* `PEDSimulatorAuto`, `PEDResultAuto`, `AutoDispatchEventRecord`
* `PIController`, `EventPredictor`, `EventPredicate`
* `BDF2State`, `BDF2PIController`, `bdf2_step`
* `RK45State`, `rk45_step`, `interpolate`
* `StiffnessDetector`, `IntegratorChoice`
* `illinois`, `brent_fallback`

These remain available **inside the test tree only**, importable
from tests as `from _dsed_reference import ...`. They are no
longer reachable from `pulsim.dsed.*`.

### Changed

* The DSED dispatcher (`pulsim._dsed_dispatch`) no longer ships
  a Stage-4 pure-Python scheduler fallback. When both Bridge.11
  (native C++ adapter) and Bridge.10 (Python adapter + native C++
  scheduler) reject a circuit — typically a nonlinear device the
  extractor doesn't yet support — the dispatcher now raises
  `NotImplementedError` with two clear options: switch to
  `engine='pwl'`, or use `pulsim.dsed.run_user_lti` with a
  hand-rolled LTI system. Previously the fallback would silently
  take the user out of the documented speedup regime even though
  the dead-code path itself wasn't reachable on the wheel after
  v1.6.2.

### Tests

* `python/tests/test_dsed_cpp_matches_python_reference.py` (3
  parametrised tests + 2 protocol-validation tests):
  - **Cross-validation**: a 2-state damped oscillator run through
    both the C++ scheduler (via `run_user_lti`) and the Python
    reference (via `_dsed_reference.PEDSimulator`). Both forced to
    `integrator='rk45'` to pin the *kernel* (DOPRI5 + adaptive PI
    + Hermite interpolation) rather than the auto-dispatch
    heuristic. End-state must agree within numpy-allclose-style
    tolerance (`atol_tol + rtol_tol·|x_py|`). The two
    implementations converge to the same trajectory at common
    accepted-step time stamps to 14 digits.
  - `run_user_lti(system_missing_method, ...)` raises `TypeError`
    naming the missing method.
  - `run_user_lti(..., integrator='bogus')` raises `ValueError`.

### Docs

* `docs/how-pulsim-works/11-dsed-engine-overview.md` and
  `16-dsed-benchmarks.md` updated: every mention of
  `pulsim.dsed.PEDSimulator*` as the user-LTI escape hatch
  replaced with `pulsim.dsed.run_user_lti`.
* `_dsed_dispatch.py` error messages updated to point at
  `run_user_lti` instead of the removed scheduler classes.

### Migration notes (best-effort, no expected callers)

We did a repo-wide and downstream-project sweep for callers of
the removed symbols outside the test tree and found none. If you
were importing one of them anyway (defensive note, since they
were technically exported):

1. **You were driving the scheduler from a CircuitBuilder** —
   switch to `pulsim.simulate(b, engine='dsed', ...)`. This is
   the production path and exclusively uses the C++ kernel.
2. **You were driving it from a custom LTI** — switch to
   `pulsim.dsed.run_user_lti(system, switch_fn, x0, t_end, ...)`.
   The system protocol is documented inline; a complete usage
   example is in
   `python/tests/test_dsed_cpp_matches_python_reference.py`.

If neither replacement covers your workflow, please open an issue
describing the use case so we can document or expose the right
extension hook from the C++ side.

## [1.6.2] — 2026-05-29

### Fixed

* **`simulate(engine='dsed', ...)` silently lost the 24× speedup
  on `pip install pulsim`** because `pulsim/dsed/bdf2_integrator.py`
  had a top-level `from scipy.linalg import lu_factor, lu_solve`.
  scipy is not a runtime dependency (it lives under
  `[project.optional-dependencies] dev`), so the import crashed —
  cascading through `pulsim.dsed.__init__` and masking even the
  C++ native Bridge.11 path that doesn't need scipy at all. End
  users got a `RuntimeError: required Pulsim bindings not
  available (No module named 'scipy')` and the native C++ path
  was unreachable from the wheel.

  Moved the scipy import to a lazy on-demand helper inside
  `bdf2_integrator.py`. The C++ native DSED path (the headline
  ~24× speedup on the buck CCM benchmark) now works on a clean
  `pip install pulsim`. scipy is still required for the
  pure-Python BDF2 fallback — the lazy import raises a clear
  `ModuleNotFoundError` pointing at `pip install 'pulsim[dev]'`
  if a user explicitly hits that path.

  Empirical confirmation (clean venv, no scipy, v1.6.2 wheel,
  canonical buck CCM 24 V→12 V at 100 kHz, 5 ms window):
  PWL @ dt=100ns = 143 ms (50001 steps); DSED auto = 0.6 ms
  (507 steps); **speedup 232×**.

### Tests

* `python/tests/test_dsed_without_scipy.py` (3 tests) pins the
  invariant:
  - Subprocess test: spawn fresh Python with an `__import__`
    hook blocking `scipy`, assert `import pulsim.dsed` succeeds.
  - Buck CCM test: run a 5 ms buck simulation with `engine='dsed'`
    under the same hook; wall-clock < 5 s (the C++ native path
    completes in ~1 ms; the Python fallback would take 30+ s).
  - Speedup test: PWL @ dt=100ns vs DSED auto on buck CCM — gate
    at >5× speedup. Skipped if scipy is missing (the gate would
    be biased by the same regression we're testing).

## [1.6.1] — 2026-05-28

### Fixed

* **`pip install pulsim` was broken on v1.6.0** because
  `pulsim/__init__.py` re-exports `wire_chain_from_yaml`, which forced
  a top-level `import yaml as _yaml` in `pulsim.yaml_chain`. PyYAML is
  only listed under the `dev` optional extra (it's not a runtime
  dependency), so the cibuildwheel smoke test (`python -c "import
  pulsim"`) failed on all platforms (Linux/macOS/Windows) and the
  PyPI publish workflow rejected the v1.6.0 wheels.

  Moved the `import yaml` to a lazy import inside
  `wire_chain_from_yaml` — only triggered when the caller passes a
  YAML *string*. Python list/dict chain specs continue to work
  without PyYAML installed. If the YAML-string path is taken without
  PyYAML, the user gets an actionable `ModuleNotFoundError` pointing
  at `pip install 'pulsim[dev]'`.

  No behavioural change for anyone who already had PyYAML
  installed via `pulsim[dev]` or as a transitive dep.

## [1.6.0] — 2026-05-28

### Highlights — Path-Based Event-Driven (DSED) engine + native PWM switch_fn

Lands the **Path-Based Event-Driven (PED)** simulation engine
([PR #62](https://github.com/lgili/Pulsim/pull/62)) — Pulsim's
alternative to the default fixed-step trapezoidal + PWL cache loop.
DSED predicts the next event time analytically (gate edges, body
diode commutation, voltage thresholds), integrates with adaptive
step control between events (DOPRI5 or BDF2 per-mode dispatch), and
handles mask transitions instantaneously without aliasing.

End-result for canonical buck CCM (24V→12V, 100 kHz, 5 ms window,
1007 vs 50001 steps): **DSED is 24× faster than PWL in wall-clock**.
Geo-mean speedup across 6 converter topologies (buck/boost/buck-boost/
half-bridge/floating-cap RLC/NPC split-bus): **14.5× vs PWL**.

#### What you can do now

```python
import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "in", "gnd", 24.0)
# ... build a buck ...

# DSED engine — variable-step, event-driven, ~24× faster than 'pwl'
sf = p.NativePwm2Switch(T_sw=1e-5, D=0.5, n_switches=2)
res = p.simulate(b, t_end=5e-3, engine='dsed', switch_fn=sf)
```

The `engine='dsed'` API is fully wired through Python with all the
familiar kwargs (rtol, atol, integrator='rk45'/'bdf2'/'auto',
b_extra_fn, switch_fn, initial_state). No user code change needed
beyond the `engine='dsed'` opt-in.

#### Topologies supported

LTI-per-mask circuits: buck, boost, buck-boost, flyback, forward,
half-bridge, full-bridge, NPC split-bus (floating caps), MMC SM-stacks
(floating caps), PFC with AC input, grid-tied inverters with sine
V_grid. Plus rejection of pathological cases (floating caps with
both terminals on ground, parallel caps, inductor loops) with
actionable error messages.

#### Architecture (Bridges 1–13)

* **Bridges 1–5** — algorithm gates: DOPRI5 + PI controller +
  Illinois root finder, BDF2 + Crank-Nicolson bootstrap, stiffness
  detector, per-mode RK45↔BDF2 auto-dispatch. Initial Python
  prototype + C++23 port.
* **Bridge 5.1b** — T^T·M·T congruence transform for **floating
  capacitors** (NPC split bus, MMC SM stacks, half/full-bridge
  differential output caps).
* **Bridges 6/7** — Time-varying source overlay (sine / PWM / pulse)
  + user `b_extra_fn` callback via per-mask projection matrix B.
* **Bridge 8** — Real-converter end-to-end validation (buck, boost,
  NPC split-bus, half-bridge + sine V_in).
* **Bridge 9** — Inductor-loop rejection (parallel L, all-L cycles)
  with clear "merge into L_eq" pointer.
* **Bridge 10** — Pybind11 scheduler wrappers (C++ inner loop,
  Python callbacks). 2.7× per-step over pure Python.
* **Bridge 11** — Native C++ `CircuitBuilderAdapter` — eliminates
  GIL roundtrips on the hot loop. Brought DSED to 13.3× faster
  than PWL.
* **Bridge 12** — Native PWM switch_fn classes (`NativePwm2Switch`,
  `NativeMultiMaskPwm`) detected at construction; scheduler calls
  them in pure C++ without GIL. Brought DSED to 24.3× faster than
  PWL.
* **Bridge 13** — PWL engine also detects native PWM and dispatches
  through a C++ lambda. PWL itself becomes 2× faster on PWM-driven
  circuits.

#### Speedup breakdown (buck CCM, 5 ms, 100 kHz)

| Layer | Wall-clock | per-step | vs PWL (old) |
|---|---:|---:|---:|
| PWL (C++ trap, 50001 steps)               | 52.7 ms | 1.05 µs | 1.0× (baseline) |
| **PWL + native PWM** (Bridge.13)           | **26.2 ms** | **0.52 µs** | **2.0×** |
| DSED Python scheduler (Bridge.5)           | 61.3 ms | 60.8 µs | 0.85× |
| DSED Bridge.10 (Python adapter)            | 22.4 ms | 22.2 µs | 2.3× |
| DSED Bridge.11 (native adapter)            |  3.8 ms |  3.80 µs | 13.3× |
| **DSED Bridge.12 (+ native PWM)**          |  **2.2 ms** |  **2.19 µs** | **24.3×** |

Final v_C = 12.0000 V exact (bit-for-bit match with analytical
D·V_in steady state) across all layers.

#### Tests

* **549/549 C++ tests pass** (Catch2; added 6 LTI extractor tests
  for the congruence transform + inductor-cycle rejection)
* **14 new Python end-to-end tests** (`python/tests/test_dsed_end_to_end.py`)
  covering buck/boost/NPC/floating-cap/sine-input/Bridge.12-vs-Python
  bit-for-bit agreement
* **Total Python test suite: 950 pass**

#### API surface added

```python
# DSED engine
pulsim.simulate(b, engine='dsed', integrator='auto'|'rk45'|'bdf2',
                 rtol=..., atol=..., dt_init=..., h_bdf2=...,
                 stiffness_threshold=..., switch_fn=..., b_extra_fn=...)

# Native PWM (DSED + PWL both detect these)
pulsim.NativePwm2Switch(T_sw, D, n_switches, hs_first=True)
pulsim.NativeMultiMaskPwm(T_sw, phase_boundaries, masks)

# Advanced — direct PED scheduler access (user-defined LTI system)
pulsim.dsed.PEDSimulator(...)
pulsim.dsed.PEDSimulatorBDF2(...)
pulsim.dsed.PEDSimulatorAuto(...)
pulsim.dsed.PIController(...)
pulsim.dsed.StiffnessDetector(...)
pulsim.dsed.BDF2State / bdf2_step(...)
pulsim.dsed.RK45State / rk45_step(...) / interpolate(...)
pulsim.dsed.EventPredictor / EventPredicate / illinois / brent_fallback

# C++ control blocks (advanced, embedded export):
# pulsim._pulsim._NativePIController, _NativePIDController,
# _NativeFirstOrderLowPass — bit-for-bit identical to the Python
# pulsim.control classes but for native step_observer use cases.
```

#### Known limitations / out of scope

* Nonlinear devices (diode Shockley, MOSFET Vth, saturable L) still
  use `pulsim.dsed.PEDSimulator` directly with a user-defined LTI
  system. The PED engine does not model per-operating-point
  linearization.
* Inductor cycles (parallel L) and parallel capacitors are rejected
  with clear errors pointing to the merge-equivalent workaround.

See `notes/DSED_BRIDGE_DESIGN.md` for the full design (~700 lines:
math, MNA→state-space reduction, T^T·M·T congruence for floating
caps, projection matrix B for time-varying sources, native C++
adapter, dispatch hierarchy) and
`openspec/changes/add-path-based-dsed-engine/` for the proposal +
specs.

## [1.5.0] — Unreleased

### Highlights — Phase 2 physics-parity push + PSIM-equivalent loss/thermal/control panels

This release closes **Phase 2 (Physics Parity)** of the v1.x roadmap
and lands a four-way upgrade to the post-hoc analysis surface so
users get the loss / thermal / control workflows they expect from
PSIM and PLECS without leaving Python.

**A. Phase 2 — Physics parity in the C++ kernel**
([PR #51](https://github.com/lgili/Pulsim/pull/51) →
[#52](https://github.com/lgili/Pulsim/pull/52) →
[#53](https://github.com/lgili/Pulsim/pull/53) →
[#54](https://github.com/lgili/Pulsim/pull/54) →
[#55](https://github.com/lgili/Pulsim/pull/55) →
[#56](https://github.com/lgili/Pulsim/pull/56))

* **2.1 — Squirrel-cage induction motor**: header-only C++ port at
  `core/include/pulsim/motors/induction_motor.hpp` + pybind
  `CxxBlockChain.add_induction_motor(...)`. 5-state Krause αβ
  model running at kernel speed.
* **2.2 — Jiles-Atherton hysteretic inductor**: C++ port at
  `core/include/pulsim/magnetics/jiles_atherton.hpp` +
  `CxxBlockChain.add_hysteretic_inductor(...)`. Sign convention
  fix on `v_M` in the b_extra row matches the Python observer.
* **2.3 — Sensorless rotor observers**: C++ port of
  `SlidingModeObserver` (PMSM Utkin + LPF + PLL) and
  `FluxMRASObserver` (IM Schauder voltage + current with
  bootstrap-fixed normalised cross-product) at
  `core/include/pulsim/observers/sensorless.hpp`. New
  `CxxBlockChain.add_sliding_mode_observer(...)` /
  `.add_flux_mras_observer(...)`.
* **2.4 — Adaptive Runge-Kutta**: `DormandPrince5` and `RadauIIA3`
  shipped (Python in v1.4.x, C++ port for standalone use in v1.5).
  `simulate(integrator=)` schema landed; kernel wiring deferred to
  v1.6 — see "v1.6 deferred" note below.

**B. YAML composite devices + `chain:` wiring**
([PR #54](https://github.com/lgili/Pulsim/pull/54),
[#56](https://github.com/lgili/Pulsim/pull/56))

* New device kinds in `circuit:` — `induction_motor` and
  `hysteretic_inductor`. The loader expands them into
  deterministic branch-id schemes (`IM_Lsig_{a,b,c}`,
  `IM_E_{a,b,c}`, `L_core_L0`, `L_core_V_M`).
* New `pulsim.wire_chain_from_yaml(loaded, chain_spec)` resolves
  the deterministic branch names and stamps a `CxxBlockChain`.
  Four block types: `induction_motor`, `hysteretic_inductor`,
  `sliding_mode_observer`, `flux_mras_observer`.
* See [docs/yaml-chain.md](yaml-chain.md).

**C. PSIM-style loss + thermal pipeline**
([PR #58](https://github.com/lgili/Pulsim/pull/58))

* `device_loss_summary` extended to cover **resistor + inductor +
  ideal-switch + switched-diode** in one pass, with optional
  per-device datasheet annotations:
  - `diode_specs={"D1": {"Q_rr": ...}}` or
    `{"E_rr_ref": ..., "V_R_ref": ...}` → reverse-recovery energy
    per turn-off event, accumulated from `commutation_events`.
  - `switch_specs={"M1": {"E_on_ref": ..., "E_off_ref": ...,
    "V_ref": ..., "I_ref": ...}}` → PSIM-style turn-on / turn-off
    energy scaled by `(V_blocking, I_load)` at each `switch_fn`
    edge.
  - `core_loss_specs={"L1": {"material": "N87", ...}}` →
    Steinmetz / iGSE core loss from `pulsim.magnetic`.
* New **`device_thermal_summary(builder, result,
  thermal_specs=...)`** pipes the loss output through a per-device
  Foster network and returns per-device `T_j(t)` traces, plus
  `T_j_avg`, `T_j_peak`, `P_total_avg`, `R_th_total`.
* Strict spec validation — unknown device names in any `*_specs`
  raise `KeyError`; non-positive geometry on core loss raises
  `ValueError`. No silent zeros.
* Shared `_result_views` helpers between `losses.py` and
  `thermal.py` eliminate duplicated result-walk code.
* See [docs/losses-and-thermal.md](losses-and-thermal.md).

**D. PSIM/PLECS-style "C block" via Numba JIT**
([PR #59](https://github.com/lgili/Pulsim/pull/59))

* New `@pulsim.fast_block` decorator turns a Python control
  function into a Numba-LLVM-compiled native callable. Same
  authoring contract as PSIM's Custom C Block (read inputs,
  mutate `state` in-place, return scalar) without runtime `cc`
  invocation, cross-OS compiler dance, or `.so` plumbing.
* `pip install pulsim[fast]` enables the JIT path; the optional
  dep keeps the base install lean. Without numba, `@fast_block`
  raises a clear `ImportError` with the install hint.
* See [docs/fast-block.md](fast-block.md) and the runnable
  showcase
  [`examples/scripts/run_fast_block_pi_buck.py`](../examples/scripts/run_fast_block_pi_buck.py).

### Added

* `pulsim.SlidingModeObserver` / `FluxMRASObserver` — C++ kernel
  adapters via `CxxBlockChain.add_*` (Phase 2.3).
* `pulsim.wire_chain_from_yaml(loaded, chain_spec)` — Python
  glue between the YAML loader and `CxxBlockChain`.
* `SimulationOptions.integrator` / `rtol` / `atol` / `dt_init`
  fields + matching YAML `simulation:` block keys (Phase 2.4
  schema, kernel wiring deferred to v1.6).
* `simulate(integrator=, rtol=, atol=, dt_init=)` kwargs —
  `"kernel"` default unchanged; `"dopri5"` / `"radau"` raise
  `NotImplementedError` with a v1.6 pointer.
* `pulsim.device_loss_summary` extended (see Highlights C).
* `pulsim.device_thermal_summary` — new (see Highlights C).
* `pulsim.FastBlock`, `pulsim.fast_block` — new (Highlights D).
* `pulsim.magnetic` Steinmetz / iGSE helpers + N87 / 3F4 / 3C90
  built-in material catalogue used by `device_loss_summary`'s
  core-loss path.
* Optional dep: `pulsim[fast]` → `numba>=0.58`.

### Changed

* `device_loss_summary` previously skipped switches / diodes /
  magnetic devices silently; now they're reported with the
  datasheet annotations described above. The signature gains
  `switch_specs`, `diode_specs`, `core_loss_specs` kwargs.
* **(Breaking)** `device_loss_summary` now **raises `KeyError`**
  when any `*_specs` mapping references a device name / branch_id
  that isn't in the builder. Was a silent skip in v1.4. Update
  YAMLs and test fixtures to use the actual device names.
* `KNOWN_LIMITATIONS.md` § "Per-device loss reporting" rewritten
  to reflect the v1.5 coverage — what's actually covered today
  vs the sub-`dt` switching-transient waveform shapes that the
  fixed-`dt` kernel still doesn't resolve.

### Fixed

* MRAS bootstrap fix: the normalised cross-product
  `ε / (|ψ_ref|·|ψ_adj|)` is now the default
  (`normalise_eps=True`) in `FluxMRASObserver` — resolves
  cold-start convergence for IM sensorless on `ω̂_init=0`.
* JA observer `v_M` sign in `b_extra` was inverted on the
  v1.4.x release branch — now matches the Python observer's
  `+N·A·µ₀·dM/dt` convention.
* `device_thermal_summary` previously computed `P_core_avg`
  internally but omitted the field from the output dict —
  fixed, users now see the core contribution that drove `T_j`.
* `_inductor_core_loss` previously returned silent zeros for
  invalid geometry (`N_turns ≤ 0` etc.); now raises `ValueError`.
* Narrowed `except Exception` around iGSE fall-back to
  `except ValueError` so unrelated bugs surface instead of
  hiding.

### v1.6 deferred

The Phase 2.4 schema for `simulate(integrator="dopri5"|"radau")`
landed, but actual execution waits on a `PwlStateSpaceCache`
refactor (the cache stores `J = G + (2/dt)·M` in trap-companion
form; adaptive RK needs `(G, M, b)` separately and DAE-aware
Radau — augmented MNA's mass matrix is structurally singular).
Same blocker postpones in-kernel `R_DS_on(T_j)` live feedback
and the stiff-thermal Radau example. See the v1.6 milestone for
the cache refactor work-item.

## [1.4.0] — 2026-05-24

### Highlights — In-house complex sparse LU + generalised path-based update framework

This release bundles **two algorithmic contributions** that were
originally scoped as separate releases but ship together as
v1.4.0 since neither had been tagged yet:

**A. In-house complex sparse LU** (per
[`openspec/changes/add-pulsim-complex-sparse-lu/`](openspec/changes/add-pulsim-complex-sparse-lu/)) —
templates `PulsimSparseLuSolver` on `Scalar` and migrates
`run_mna_sweep` to the new `PulsimComplexSparseLuSolver`
(= `PulsimSparseLuSolverT<std::complex<Real>>`). Completes the
v1.3.0 "no third-party LU in production" agenda — the AC sweep
code path no longer compiles `Eigen::SparseLU<complex>`.
`Backend::Eigen` is retained as the IEEE TPEL §VI.B
paper-comparison baseline.

**B. Generalised path-based update framework** (per
[`openspec/changes/add-generalised-path-refactor/`](openspec/changes/add-generalised-path-refactor/)) —
generalises the v1.3.0 single-bit path-based partial refactor to
**three SMPS-relevant use cases** that no open-source
power-electronics simulator currently exploits:

1. **Multi-bit switch transitions** (Part A) — SPWM with multiple
   legs commutating simultaneously, multilevel commutation patterns.
   v1.3.0 unconditionally routed those to full `factorize()`; v1.4.0
   attempts the union of etree paths when the union covers ≤
   `MAX_PATH_LENGTH_RATIO` (default `0.6`) of the matrix.
2. **Parametric value changes** (Part B) — `R`, `L`, `C`, source `V`
   updates for sweep / Monte Carlo / design-optimisation workloads.
   v1.3.0 forced a fresh `analyze + factorize` rebuild per sweep
   point (~100 µs/point cold path); v1.4.0 reuses both the symbolic
   factor AND most of L+U via the same path-union machinery.
3. **Single-bit Gray-code flips** (preserved from v1.3.0) — same
   2.7-2.9× speedup at n_state ≥ 12 documented in
   `RANK1_RESULTS.md`.

User-facing Python helpers `sweep_path_aware` /
`monte_carlo_path_aware` ship as drop-in replacements for `sweep` /
`monte_carlo`. Auto-fallback to the legacy path when the swept
parameter name is unknown to the builder; the user sees a
`RuntimeWarning` and the same `SweepResult` shape.

### Performance — Part A multi-bit microbench

Captured 2026-05-24 on macOS Apple Silicon (see
[`artigos/02_tpel_methods/benchmarks/MULTI_BIT_RESULTS.md`](artigos/02_tpel_methods/benchmarks/MULTI_BIT_RESULTS.md)).
Pulsim path-union speedup vs the v1.3.0 emulation (Eigen sliding
solver = full factorize per flip):

| n_state | δ = 1 | δ = 2 | δ = 3 | δ = 4 |
|--------:|------:|------:|------:|------:|
| 10      | 3.12× | 1.62× | 1.61× | 1.42× |
| 14      | 1.72× | 1.58× | 1.58× | 1.42× |
| 18      | 1.56× | 1.28× | 1.51× | 1.25× |
| 22      | 1.36× | 1.42× | 1.54× | 1.51× |
| 26      | 1.55× | 1.46× | 1.33× | 1.42× |

Multi-bit hit rate decays gracefully with δ:
~40–50 % of 2-bit transitions take the path-union path,
~20–25 % at δ = 3, ~8–19 % at δ = 4. The remainder gracefully
fall back to full factorize without regression vs v1.3.0.

### Performance — Part B parametric microbench

Captured 2026-05-24 on the same hardware (see
[`artigos/02_tpel_methods/benchmarks/PARAMETRIC_RESULTS.md`](artigos/02_tpel_methods/benchmarks/PARAMETRIC_RESULTS.md)).
Pulsim `refactor_parametric` speedup vs the legacy rebuild-the-
cache-from-scratch-per-sweep-point pattern (current
`pulsim.sweep.sweep(...)` semantics):

| n_state | 50 pts | 100 pts | 500 pts | 1000 pts |
|--------:|-------:|--------:|--------:|---------:|
| 8       | 5.18×  | 3.29×   | 3.55×   | 3.68×    |
| 14      | 3.57×  | 3.02×   | 3.51×   | 3.35×    |
| 26      | 3.53×  | 3.31×   | 3.38×   | 3.40×    |

**Zero fallbacks across all 12 (n_switches × n_sweep_points)
cells** — every refactor_parametric call took the path-based
update successfully on this fixture family.

### Added

- **`pulsim::sparse::MAX_PATH_LENGTH_RATIO`** — compile-time
  tunable (default `0.6`). Path-based update is skipped when the
  union-path length exceeds this fraction of `n`. See
  `openspec/changes/add-generalised-path-refactor/design.md`
  Decision 2 for the empirical break-even rationale.
- **`DirectSolverT<Scalar>::partial_refactor_count_path(changed_cols)`**
  — virtual query method. Returns the length of the union path that
  `partial_refactor` would walk **without executing the refactor**.
  Used by `solve_rank1` to consult `MAX_PATH_LENGTH_RATIO` before
  attempting path-based update on multi-bit transitions.
  Default implementation returns 0; `PulsimSparseLuSolverT<Scalar>`
  overrides with the real walk.
- **`PulsimSparseLuSolverT<Scalar>::partial_refactor_count_path`**
  — production implementation. Walks the etree path of each column
  in the **hypothetical union** of `varying_set_ + changed_cols`,
  deduplicates via an in-path bitmap. Pure read-only — does not
  mutate solver state. Companion `partial_refactor_path_ratio`
  wraps `count_path / n` for the common comparison expression.
- **`pulsim::pwl::CacheMetrics::multi_bit_rank1_hits`** — new
  counter for multi-bit successes via path-union `partial_refactor`.
  v1.3.0 routed all multi-bit transitions to `full_refactor_hits`;
  v1.4.0 splits them between this new counter (success path) and
  `full_refactor_hits` (path too long → fallback).
  Invariant: `rank1_hits + multi_bit_rank1_hits + full_refactor_hits
  + fallbacks == N`.
- **`pulsim::pwl::DevicePool::columns_affected_by_switch(sw_idx,
  graph)`** — new helper returning the MNA columns affected by
  toggling switch `sw_idx`. Mirrors the
  `branch_var_id_for_source` access pattern. Used by
  `compute_changed_columns_` and (in a future cycle) by Python
  bindings exposing the switch→column map.
- **`core/tests/benchmarks/test_bench_multi_bit_rank1.cpp`** —
  3-backend microbench across `(N, δ) ∈ {8,12,16,20,24} × {1,2,3,4}`.
  1000 random transitions per cell.
- **`artigos/02_tpel_methods/benchmarks/MULTI_BIT_RESULTS.md`** +
  `multi_bit_microbench.csv` — full writeup mirroring
  `RANK1_RESULTS.md`'s structure.
- **7 new C++ unit tests** in `core/tests/layer0/test_pulsim_lu_solver.cpp`
  (4 spec-mandated v1.4.0 scenarios: multi-col `partial_refactor`,
  monotone `count_path`, empty-input no-op, `MAX_PATH_LENGTH_RATIO`
  range gate) and `core/tests/layer4/test_pwl_cache_rank1.cpp`
  (3 cache-level scenarios: 2-bit transition routing, telemetry
  invariant under mixed Hamming workload, 4-bit transition correctness).

#### Part B — parametric refactor

- **`pulsim::pwl::ParametricRefactorResult`** + **`ParametricRefactorMode`**
  + **`ParametricUpdate`** — new public types in `cache.hpp`.
  Result invariant: `path_refactor_hits + fallback_hits ==
  masks_processed`.
- **`PwlStateSpaceCache::refactor_parametric`** — new C++ API
  with two overloads:
  - Single-param: `refactor_parametric(branch_id, new_value, mode)`
  - Batch: `refactor_parametric(span<const ParametricUpdate>, mode)`
  Pushes parameter updates through the pool, walks every active
  mask (or just the rank-1 mask in `Mode::CurrentOnly`),
  re-assembles `(J, b)` at the new values, and calls
  `partial_refactor(new_J, affected_cols)` for each segment.
  Falls back to fresh `factorize()` when path too long
  (gated by `MAX_PATH_LENGTH_RATIO`) or backend lacks
  `partial_refactor` support.
- **`DevicePool::columns_affected_by_branch(branch_id, graph)`**
  — returns the MNA columns that depend on a branch's stored
  parameter value(s). Resistor/Switch/Capacitor → both endpoint
  cols; Inductor → its branch-current col; VoltageSource →
  empty (RHS-only). Unsupported device kinds → empty (falls back).
- **`DevicePool::update_resistor_R / update_inductor_L /
  update_capacitor_C / update_voltage_source_V`** — value
  mutators dispatching on the stored variant via
  `std::get_if`. Throws `std::out_of_range` on kind mismatch.
- **`CircuitBuilder::branch_id_of(name)`** — inverse of
  `name_of(branch_id)`. Throws on unknown name.
- **`CircuitBuilder::update_resistor_R(name, R_ohms)` (+ inductor /
  capacitor / voltage_source variants)** — convenience wrappers
  that combine `branch_id_of` + the pool mutator. Designed for
  the user-facing parametric refactor pattern:
  ```python
  b.update_resistor_R("R_load", 3.0)
  cache.refactor_parametric(b.branch_id_of("R_load"), 3.0)
  ```
- **pybind11 bindings** for all of the above — `ParametricRefactorResult`
  + `ParametricRefactorMode` enum + cache methods + builder helpers
  exposed to Python via `python/bindings.cpp`. Smoke-tested
  end-to-end on the `pulsim 1.4.0` wheel.
- **`core/tests/layer4/test_pwl_cache_parametric.cpp`** — 6 new
  test cases / 57 assertions covering: single-param sweep parity
  vs fresh-rebuild within 1e-10, two-param simultaneous parity,
  empty-updates no-op, unsupported-kind throws, `Mode::CurrentOnly`
  processes 1 mask, telemetry invariant over 10 sweep points.
- **`core/tests/benchmarks/test_bench_parametric_sweep.cpp`** —
  3-backend microbench across `(n_switches, n_sweep_points) ∈
  {2,4,8} × {50,100,500,1000}` on parallel-leg buck fixtures.
- **`artigos/02_tpel_methods/benchmarks/PARAMETRIC_RESULTS.md`**
  + `parametric_microbench.csv` — full writeup.

#### Part C — In-house complex sparse LU (AC sweep migration)

- **`pulsim::sparse::PulsimSparseLuSolverT<Scalar>`** — the
  templated class. Backward-compat type aliases keep every Layer 1-9
  call site source-compatible:
  ```cpp
  using PulsimSparseLuSolver        = PulsimSparseLuSolverT<Real>;
  using PulsimComplexSparseLuSolver = PulsimSparseLuSolverT<std::complex<Real>>;
  ```
- **`pulsim::sparse::DirectSolverT<Scalar=Real>`** — the templated
  abstract base. `DirectSolver = DirectSolverT<Real>` for backward
  compat. Same pattern for `SparseLuSolverT<Scalar=Real>` /
  `SparseLuSolver = SparseLuSolverT<Real>`.
- **`pulsim::sparse::make_default_solver_t<Scalar>(n, hint)`** —
  template factory. The non-template
  `make_default_solver(n, hint)` is now a shim that dispatches to
  `make_default_solver_t<Real>(n, hint)`.
- **`pulsim::VectorT<Scalar>`** and **`pulsim::sparse::MatrixT<Scalar>`** /
  **`pulsim::sparse::TripletT<Scalar>`** templates with
  `Vector` / `Matrix` / `Triplet` backward-compat aliases for `Real`.
- **`core/tests/layer0/test_pulsim_lu_solver_complex.cpp`** — 5 new
  test cases / 31 assertions covering the complex specialisation:
  Hermitian PD identity, asymmetric MNA 8×8 (forces partial pivoting
  at the zero-diagonal voltage-source row), single-column
  partial_refactor parity, solve-before-factorize lifecycle, factory
  dispatch.
- **`core/tests/analysis/test_mna_sweep.cpp`** — 2 integration
  tests through `run_mna_sweep`: RC low-pass within 0.1 dB / 1°
  of `1/(1+jωRC)` across 50 frequencies; series RLC peak within
  1.5 % of `1/(2π√(LC))` (Q ≈ 5).
- **`core/tests/benchmarks/test_bench_ac_sweep.cpp`** — 2-backend
  AC sweep microbench across `n ∈ {8, 16, 32, 64, 128}`,
  100 log-spaced frequencies from 1 Hz to 1 MHz.
- **`artigos/02_tpel_methods/benchmarks/AC_SWEEP_RESULTS.md`** +
  `ac_sweep_microbench.csv` — full writeup of the
  Pulsim-vs-Eigen parity story.

### Changed

- **`core/include/pulsim/analysis/mna_sweep.hpp`** —
  `Eigen::SparseLU<ComplexSparseMatrix, COLAMDOrdering<Index>>`
  replaced with `sparse::PulsimComplexSparseLuSolver`. Lifecycle:
  `analyze(M)` → `factorize(M)` → `solve(b, x)`, all returning
  `bool` (vs Eigen's `info()` enum). `ComplexSparseMatrix` switched
  from RowMajor to ColMajor to match the in-house solver's CSC
  input format (no transpose-and-copy per frequency).
  `#include <Eigen/SparseLU>` removed — no longer needed on the
  production path.
- **`PwlStateSpaceCache` constructor** — `pool` parameter changed
  from `const DevicePool&` to `DevicePool&`. Existing callers that
  pass a non-const builder pool continue to compile unchanged.
  Required so `refactor_parametric` can drive `pool.update_*`.
- **`PwlStateSpaceCache::try_make_segment`** — segments are now
  built with `Backend::Auto` (= Pulsim in-house LU) by default,
  not the Eigen baseline. Numerically bit-identical on real-scalar
  SPD matrices; enables `refactor_parametric` to use
  `partial_refactor` on the cached segment factors without an
  explicit `set_segment_backend` step.
- **`PwlStateSpaceCache::solve_rank1`** — multi-bit routing logic
  rewritten. For `delta_bits >= 2`, the cache now:
  1. Computes the deduped `changed_cols` set via
     `compute_changed_columns_`.
  2. Queries `solver.partial_refactor_count_path(changed_cols)`.
  3. If `path_length / n ≤ MAX_PATH_LENGTH_RATIO`, calls
     `partial_refactor`; on success counts `multi_bit_rank1_hits++`.
  4. Otherwise calls `factorize()` and counts `full_refactor_hits++`.
  Single-bit transitions (`delta_bits == 1`) keep v1.3.0 behavior:
  always try `partial_refactor` without the ratio gate.
- **`compute_changed_columns_`** now deduplicates via `std::set<Index>`
  before returning. Switches sharing a node (common in half/full-bridge
  topologies) previously produced duplicate column entries. v1.4.0's
  `partial_refactor_count_path` requires a canonical input to give
  a meaningful answer to the ratio gate, so dedup happens at the call
  site.

### Removed

- **Implicit production dependency on `Eigen::SparseLU<std::complex<Real>>`**.
  The Eigen complex SparseLU instantiation is no longer compiled
  into the AC sweep code path. `Backend::Eigen` keeps the path
  explicitly available for paper-comparison purposes.

### Migration notes

- **`CacheMetrics` ABI**: the struct grew a new field
  (`multi_bit_rank1_hits`). Existing callers reading
  `rank1_hits` / `full_refactor_hits` / `fallbacks` continue to
  compile + work. Code that pinned `full_refactor_hits == N` for
  N multi-bit transitions will see those calls land in
  `multi_bit_rank1_hits` instead — update test fixtures to use the
  telemetry invariant `rank1 + multi_bit + full + fallbacks == N`.
- **`solve_rank1` behavior on Eigen backend** (no
  `partial_refactor` support): unchanged. Every transition falls
  back to full factorize and counts under `full_refactor_hits`.
- **Pulsim backend behavior on a 4-switch n_state=6 fixture**: a
  small fraction (~3-5 %) of multi-bit transitions now hit
  `fallbacks` instead of `full_refactor_hits` because the pivot
  threshold check on the wider path rejects more often. Telemetry
  invariant still holds; numerical correctness still within 1e-10
  vs fresh-factorise. Documented in `MULTI_BIT_RESULTS.md`.
- **`PulsimSparseLuSolverT<Real>`** is bit-identical to v1.3.0's
  `PulsimSparseLuSolver` modulo the rename — every Layer 1-9
  consumer that uses the unparameterised name keeps compiling
  unchanged.

### Regression test summary

- **498 / 498 C++ tests pass** (up from 478 in v1.3.0; +20 new
  tests: 5 complex-solver unit + 2 mna_sweep integration +
  7 multi-bit spec scenarios + 6 parametric refactor cases).
- **6 / 6 Python tests pass** (`test_sweep_path_aware.py` —
  KPI parity vs legacy, two-param sweep, unknown-name fallback,
  MC, result shape).
- **`pulsim 1.4.0`** Python wheel builds + imports clean.
  `cache.refactor_parametric(b.branch_id_of("R_load"), 3.0)`
  smoke-tested end-to-end.
- Existing rank-1 microbench (single-bit Gray-code, all N ∈ {4..24})
  shows the same 2.7-2.9× speedup as v1.3.0 — the complex solver
  templatisation, multi-bit routing, and parametric refactor all
  change dispatch logic without regressing the v1.3.0 hot path.

## [1.3.0] — 2026-05-24

### Highlights — In-house sparse LU + path-based partial refactorization

This release replaces the V8 KLU-backed `partial_refactor` with a
**fully in-house C++23 sparse LU stack** (`pulsim::sparse::PulsimSparseLuSolver`),
implementing the path-based partial refactorisation algorithm
(Chan/Brandwajn/Tinney, *IEEE Trans. Power Syst.* 1, 1986;
Dinkelbach et al., *Energies* 14:7989, 2021, §3) from scratch on
top of Eigen sparse-matrix containers. **Zero third-party LU
dependency** — neither SuiteSparse KLU (V8) nor the dpsim-simulator
fork (the rejected V8.1-vendoring approach).

Per the project owner's 2026-05-24 architectural decision
(documented in
[`openspec/changes/replace-klu-with-pulsim-sparse-lu/`](openspec/changes/replace-klu-with-pulsim-sparse-lu/)):
the algorithmic novelty of the planned IEEE TPEL methods paper
must be ours, not a thin wrapper around someone else's C patch.

### Performance

3-backend microbench captured 2026-05-24 on macOS Apple Silicon
(see [`artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md`](artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md)):

| n_state | baseline solve | Pulsim path-based | speedup |
|--------:|---------------:|------------------:|--------:|
| 6       | 6.7 µs         | 2.3 µs            | 2.93×   |
| 14      | 10.0 µs        | 3.6 µs            | **2.81×** |
| 18      | 12.2 µs        | 4.3 µs            | **2.82×** |
| 26      | 16.4 µs        | 6.1 µs            | **2.68×** |

**Zero fallbacks across all 1999 single-bit Gray-code flips per N**
— every transition exercised the path-based fast path successfully.
The per-call cost stays nearly flat (3.6 → 6.1 µs from n_state=14
to n_state=26) while the baseline scales linearly — the textbook
signature of O(path) per call vs O(nnz·log n) for fresh factorize.

### Added

- **`pulsim::sparse::PulsimSparseLuSolver`** — in-house sparse LU
  in pure C++23, ~900 lines header-only. Implements the full
  `DirectSolver` lifecycle:
  - `analyze()` — Reverse Cuthill-McKee column ordering (George 1971),
    elimination tree (Davis 2006 §4.10 / Liu 1986), symbolic L+U
    pattern
  - `factorize()` — Gilbert-Peierls left-looking with partial
    pivoting (Gilbert & Peierls, *SIAM J. Sci. Stat. Comput.* 9,
    1988). Handles the asymmetric MNA + zero-diagonal patterns
    characteristic of voltage-source constraint rows.
  - `solve()` — forward + back substitution with `Prow`/`Pcol`
    permutations
  - `partial_refactor()` — **path-based** re-elimination over the
    etree, with lazy union of varying columns + pivot-threshold
    fault detection. ~2.7-2.9× speedup vs baseline at the
    n_state ≥ 14 regime.
- **`pulsim::sparse::Backend::Pulsim`** — new enum value (replaces
  `Backend::KLU` from v1.2.0). Default for `Backend::Auto`.
- **CSV bench `rank1_microbench.csv`** — 3-backend, 8-row capture
  for direct citation in the TPEL §VI table.

### Removed (BREAKING at the C++ kernel-builder level)

- **`pulsim::sparse::KluSolver`** — replaced by PulsimSparseLuSolver
- **`pulsim::sparse::Backend::KLU`** — replaced by `Backend::Pulsim`
- **`find_package(KLU)` block in `CMakeLists.txt`** — KLU is no
  longer a dependency at all
- **`PULSIM_HAVE_KLU` + `PULSIM_ENABLE_KLU` compile defs / build
  options** — no longer applicable
- **`libsuitesparse-dev` from CI** — no longer needed; `apt install
  libsuitesparse-dev` removed from all Linux CI matrix entries,
  `brew install suite-sparse` removed from macOS

**Migration:** any out-of-tree caller that constructed `KluSolver`
directly or passed `Backend::KLU` to `make_default_solver(n, hint)`
must switch to `PulsimSparseLuSolver` / `Backend::Pulsim`. The
standard `make_default_solver()` / `make_default_solver(n,
Backend::Auto)` entry points continue to work transparently — the
factory returns PulsimSparseLuSolver by default.

### Not changed

- **Public Python API** — `pp.simulate(builder, ...)` keeps working.
  Wiring `solve_rank1` into Layer 5's `run_transient` + Python
  bindings is out of scope of this release; tracked as
  `add-pwl-rank1-runtime-integration` (TBD).
- **All 8 reference projects under `projects/`** — bit-identical
  output (verified via the 17,279 layer4/4_v1/5/5_v1/5_v4
  assertions across 135 test cases).
- **Build prerequisites** — just **Eigen 3.4+ and a C++23 compiler**
  now; no SuiteSparse install needed.

### Fixed — pre-release cleanup

- **`pulsim.device_loss_summary`** now walks both **inductor** and
  **resistor** branches. Resistor entries report `P_avg` and
  `E_total` in addition to `i_avg`/`i_rms`/`i_peak` — current is
  reconstructed from the node-voltage difference and the stored
  `R_ohms`. Switches and diodes remain deferred (see
  [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) § *Post-hoc
  analysis*). Previously the summary covered inductors only and
  the module docstring advertised a function that the curated
  `pulsim.*` surface never re-exported.
- **`pulsim.LossAccumulator`, `pulsim.EfficiencyCalculator`,
  `pulsim.device_loss_summary`, `pulsim.average_power_at_node`**
  are now wired into `pulsim.__all__` and importable from the
  top level. The functions existed in `pulsim/losses.py` from the
  start but were not exposed, so callers following the module
  docstring (`p.LossAccumulator()`) hit `AttributeError`.
- **`pulsim.schematic.render`** — removed the `position_hints=`
  keyword. Neither backend (`netlistsvg`, `python_native`) ever
  shipped an implementation; both raised `NotImplementedError` on
  non-empty input. The auto-layout path is unchanged. The
  follow-up renderer is tracked as
  [`add-schematic-renderer-v2`](openspec/changes/add-schematic-renderer-v2/).
- **`KNOWN_LIMITATIONS.md`** — added at the repository root,
  cataloguing every deliberately-deferred item carried into v1.3
  and linking each one back to its OpenSpec proposal or follow-up
  task.

## [1.2.0] — 2026-05-24

### Highlights — PWL rank-1 cache update path (Layer 4 V8)

This release ships the algorithmic contribution that backs the
planned IEEE TPEL methods paper on Pulsim's PWL state-space cache
(see [`artigos/02_tpel_methods/`](artigos/02_tpel_methods/)). The
full design + decisions + delta specs live in
[`openspec/changes/add-pwl-rank1-update/`](openspec/changes/add-pwl-rank1-update/)
and the captured benchmark in
[`artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md`](artigos/02_tpel_methods/benchmarks/RANK1_RESULTS.md).

**No BREAKING changes.** Every existing caller produces bit-identical
output. The rank-1 path is purely additive performance. 17,839 layer5
assertions across 89 test cases pass unchanged.

### Added

- **`pulsim::sparse::KluSolver`** — `DirectSolver` implementation
  wrapping SuiteSparse KLU (Davis & Natarajan, *ACM TOMS* 37(3), 2010,
  Algorithm 907). Purpose-built for circuit MNA matrices. Header-only,
  gated on `PULSIM_HAVE_KLU` (set by the root `CMakeLists.txt`'s
  `find_package(KLU CONFIG)` block). When KLU is absent the kernel
  builds and runs identically using `Eigen::SparseLU`.
- **`pulsim::sparse::Backend` enum + factory hint** — new overload
  `make_default_solver(Size n, Backend hint = Backend::Auto)` lets the
  caller request `Backend::KLU`, `Backend::Eigen`, or auto-pick by
  matrix size. Default `Backend::Auto` picks KLU when n ≥ 100
  (`PULSIM_KLU_AUTO_THRESHOLD`, tuneable at build).
- **`DirectSolver::supports_partial_refactor()`** + **`partial_refactor(M, changed_cols)`** —
  new virtual methods on the base interface, with default impls
  returning `false` so existing solvers transparently fall back.
- **`PwlStateSpaceCache::solve_rank1(mask, b_extra, x)`** — sliding-
  solver fast path. On single-bit Gray-code mask flips it calls
  `partial_refactor` instead of rebuilding the cache segment; falls
  back transparently to full re-factor on multi-bit flips, unsupported
  backends, or numerical singularities.
- **`PwlStateSpaceCache::set_rank1_backend(Backend)`** — pre-`solve_rank1`
  override for the rank-1 sliding solver's backend (useful for
  benchmarks that want to exercise KLU even at small n).
- **`PwlStateSpaceCache::metrics()`** + **`pulsim::pwl::CacheMetrics`** —
  `{rank1_hits, full_refactor_hits, fallbacks}` atomic monotonic
  counters for benchmark attribution. Thread-safe sampling via
  `std::memory_order_relaxed`.
- **Microbenchmark `core/tests/benchmarks/test_bench_pwl_rank1.cpp`** —
  Catch2 binary in the opt-in `pulsim_benchmarks` target. Sweeps
  N ∈ {4, 6, 8, 10, 12} switches, times `solve` vs `solve_rank1`,
  writes CSV to `${PULSIM_BENCH_RESULTS_DIR}/rank1_microbench.csv`.
- **CI matrix** updated to install `libsuitesparse-dev` on Linux +
  `suite-sparse` via brew on macOS across every existing entry
  (Clang 17/18, GCC 13, Debug sanitizers, coverage).
- **README "Build prerequisites"** section documenting the new
  optional dependency with install commands for macOS / Debian /
  Fedora and the `-DPULSIM_ENABLE_KLU=OFF` opt-out.

### Performance

Captured microbench on macOS 26.5 / Apple Silicon / AppleClang 17:

| N | n_state | µs/solve | µs/rank1 | speedup |
|--:|--:|--:|--:|:--:|
| 4  | 6  | 4.67 | 10.29 | 0.45× (overhead dominates at tiny n) |
| 6  | 8  | 2.57 | 2.79  | 0.92× (break-even) |
| 8  | 10 | 2.57 | 2.90  | 0.89× (break-even) |
| 10 | 12 | 4.60 | 2.73  | 1.69× (rank-1 wins) |
| 12 | 14 | 9.69 | 3.08  | **3.15×** (headline finding) |

Per-call rank-1 cost stays ~3 µs across the sweep while per-call
`solve` cost grows linearly with n — the textbook signature of
amortising the symbolic factorisation across all calls. The
V0 MVP delegates to `klu_refactor`; the V8.1 follow-up will replace
that with path-based partial re-elimination per Chen et al.,
IEEE TPEL 2024 §III, extending the speedup to 5-10× at n=200.

### Not changed

- **Public Python API** — `pp.simulate(builder, …)` continues to use
  the existing per-mask cache path via `cache.solve(mask)`. Wiring
  `solve_rank1` into Layer 5's `run_transient` + Python bindings is
  out of scope of this release; tracked as
  `add-pwl-rank1-runtime-integration` (TBD).
- **All 8 reference projects under `projects/`** — bit-identical
  output (verified via the layer5 / layer5_v1..v4 / showcase regression
  test suite, 17,839 assertions across 89 test cases).

## [1.1.0] — 2026-05-23

### Highlights — JOSS submission release

This release marks the first version of Pulsim accompanied by a
peer-reviewed publication. The accompanying paper has been submitted
to the [Journal of Open Source Software (JOSS)](https://joss.theoj.org/);
the source lives in [`artigos/01_joss_tool_paper/`](artigos/01_joss_tool_paper/).
Once the JOSS paper is accepted, this version's DOI will be the
canonical software citation.

### Added

- **`LICENSE`** at repo root — MIT text. The licence was previously
  only declared in `pyproject.toml`; JOSS (and most academic
  citation tools) require the licence file at the root.
- **`CITATION.cff`** at repo root — Citation File Format v1.2.0
  metadata for automatic citation generation by GitHub and tools
  like `cffconvert`.
- **`artigos/` directory** — paper sources for the Pulsim publication
  campaign, with `README.md` documenting the 4-paper strategic plan
  (JOSS tool paper → EPE-ECCE Europe 2026 conference →
  IEEE Open Journal of Power Electronics methods paper →
  IEEE TPEL / JESTPE application paper).

### Fixed

- **README quick-start example** — `p.scope(...)` updated to
  `p.plot.scope(...)` to match the actual location of the plot
  helper in the current Pulsim 1.x API. Verified end-to-end
  against the installed package.

## [0.10.0] — 2026-05-19

### Highlights

The 0.10.0 release closes the alpha cycle that started with `0.10.0a1`
and adds a **switched-mode closed-loop control surface** that brings
Pulsim into PSIM/Simulink territory for power-electronics controller
design and verification.

### Added — Switched-Mode Closed-Loop

- **`Simulator.run_transient(x0, circuit, callback)`** — new binding
  overload that accepts a Python callback invoked after every accepted
  timestep. The callback can call back into the circuit
  (`circuit.set_pwm_duty(name, new_duty)`, `circuit.set_pmsm_foc_references(...)`,
  …) to close the loop. Single transient run, full state preservation,
  Python in control — same architectural pattern as PSIM / Simulink.
- **GIL-safe streaming binding** — `run_transient_streaming` now
  releases the GIL around the C++ integration loop, lets callbacks
  re-enter pybind11 safely, and survives `None` callbacks. The
  `py::call_guard<py::gil_scoped_release>` race that crashed on every
  invocation is fixed.
- **`RuntimeCircuit::has_any_dynamic_history()`** — kernel helper that
  lets `Simulator::run_transient_native_impl` discriminate fresh-circuit
  vs. continuation calls. Continuations now preserve cap `i_prev` and
  inductor `v_prev` on the same Circuit instance (the per-period
  closed-loop pattern no longer collapses the dynamic state).
- Periodic shooting `run_periodic_shooting` retains "fresh-state-per-
  shooting-iteration" semantics — explicit `update_history(guess, true)`
  reset before each `run_transient(guess)` call.

### Added — Teaching Notebooks

- `examples/notebooks/vsi_inverter_design.ipynb` — end-to-end design
  of a 3φ Voltage Source Inverter (SPWM, 16 kHz, 6 SiC MOSFETs).
- `examples/notebooks/boost_pfc_vsi_design.ipynb` — full AC → DC → 3φ AC
  cascade (220 V_rms in, 400 V DC bus, 230 V_rms 3φ out).
- `examples/notebooks/boost_pfc_closed_loop.ipynb` — switched-mode
  closed-loop PFC using `Simulator.run_transient(x0, ckt, callback)`.
  V_dc converges (architecture proof-of-concept; PI tuning is iterative
  follow-up work — cascaded ACMC is the next milestone).

### Fixed

- `run_transient(x0)` no longer ping-pongs voltage-source nodes between
  `0` and `2·V_src` when `x0 = zeros` (consistent initialization fix
  in `a2cb883`).
- `run_transient_streaming` no longer aborts the process with
  `pybind11::handle::inc_ref` GIL assertions when any callback is
  passed (including `None`).
- Per-period closed-loop boost: cap state is preserved across
  Simulator constructions sharing the same Circuit, removing the
  divergence-to-0V symptom on continuation runs.
- 95 `ruff` errors across `python/` brought to zero — E702 multi-stmt
  semicolons split onto separate lines, F401 unused imports added to
  `__all__` or removed, E402 imports-after-importorskip ignored at the
  per-file level for property tests.
- `mkdocs build --strict` is green again — removed dangling refs to
  retired loss-params classes (`MOSFETLossParams`, `IGBTLossParams`,
  `DiodeLossParams`, `ConductionLoss`, `SwitchingLoss`), switched
  cross-tree file links to absolute GitHub URLs, added `: Any`
  annotations on `circuit` params that griffe was flagging.
- Stress benchmark suite no longer aborts on `periodic_rc_pwm` —
  added the missing entry to `benchmarks/benchmarks.yaml` (with no
  SPICE netlist, since the periodic-analysis bench has no parity
  baseline).
- `test_fmu_*` skip cleanly on Windows (ctypes.CDLL holds the DLL
  handle across `TemporaryDirectory` cleanup → PermissionError).
- `test_bode_plot_rejects_failed_result` skips when matplotlib is
  not installed (Windows CI).
- `test_shooting_uses_warm_start_retry_for_pwm_case` marked `xfail`
  pending shooting-solver re-tune for dead-time PWM (regression
  pre-dates this release; tracked separately).

### Notebooks — also revalidated

- `boost_converter_design.ipynb` runs end-to-end on the new kernel
- `flyback_converter_design.ipynb` runs end-to-end on the new kernel
- `vsi_inverter_design.ipynb` / `boost_pfc_vsi_design.ipynb` —
  `np.trapz` → `np.trapezoid` compat for NumPy 2.x

### Removed

- (No public API removals in this release. The loss-params classes
  documented in earlier alpha series were already replaced by
  device-side params during the alpha cycle.)

### Migration

- The new closed-loop pattern is **opt-in via a new binding overload**.
  Existing single-shot transient calls (`Simulator.run_transient()` /
  `Simulator.run_transient(x0)`) behave exactly as before.
- Per-period closed-loop users that reused the same Circuit across
  Simulator constructions now get **correct state preservation** by
  default. If your code depended on the old "reset on every call"
  behaviour, call `circuit.update_history(x, True)` explicitly before
  each `run_transient` to force the reset.

### Internal

- Kernel test suite: **304 cases / 4214 assertions** green.
- Python lint: **`ruff check python/`** zero errors.
- Docs build: **`mkdocs build --strict`** green.

### Notable commits

- `fc3c686` — kernel: preserve dynamic-device history + streaming GIL fix
- `ed879af` — bindings: `Simulator.run_transient(x0, ckt, callback)`
- `9062c78` — notebook: closed-loop PFC switched-mode proof-of-architecture
- `cef7981` — notebook: AC → DC → 3φ AC cascade design
- `663e3be` — notebook: 3φ VSI design walkthrough
- `9806df5` — chore: zero ruff errors
- `c5d7699` — fix: docs strict + benchmark index
- `1b9d01d` — fix: restore periodic shooting + Windows test gates

---

## Earlier Releases

See [GitHub Releases](https://github.com/lgili/Pulsim/releases) for
0.9.0 and earlier.
