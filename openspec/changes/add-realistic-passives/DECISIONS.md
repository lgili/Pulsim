# Decisions taken during implementation (owner away)

Each entry: question → option taken → why → how to reverse.

## D1. Where the composites live

- Question: tasks 1.1, 2.1 and 3.2 name the C++ `CircuitBuilder`; the
  composites need no new kernel code, and the C++ builder, bindings and
  YAML loader were being edited by other changes at the same time.
- Option: `python/pulsim/passives.py` wraps `CircuitBuilder.add_capacitor`
  and `add_inductor` with keyword-only `esr`, `esl`, `c_par`, `rac`, and
  installs `add_transmission_line` (same mechanism as
  `_builder_ergonomics` uses for `add_shockley_diode`). With every new
  parameter zero the wrapper calls the native method with the same
  arguments, so the result is the native one bit for bit.
- Why: one implementation, no rebuild, nothing to merge in shared C++.
- Reverse: port the expansion into `circuit_builder.hpp` with the same
  sub-element names, bind it, delete the wrappers; tests unchanged.

## D2. YAML keys `esr`, `esl`, `c_par`, `rac`; type `transmission_line`

- Option: `esr`, `esl` (capacitor) and `esr`, `c_par` (inductor) are
  read by the C++ loader, which expands them into the same elements as
  the Python keywords (pinned equal by
  `python/tests/test_yaml_composites.py`). `rac:` is refused by name
  (a ladder is a fitted object; fitting at load needs Python). No
  `transmission_line` YAML type (section 5.10 belongs to the Bergeron
  line).
- Risk recorded: the loader still does not check other per-device
  keys; an unknown key is ignored without a warning. That predates
  this change.

## D3. Where the ESR loss is reported (task 1.7)

- Option: as the resistor row `{name}.esr` of `device_loss_summary`
  (the sub-element is an ordinary resistor). No change to `losses.py`.
- Why: the row is named after the component and its value is exactly
  `i²·esr`; folding it into a `{name}` row would need a composite
  registry.
- Reverse: aggregate `{name}.*` rows in `losses.py`.

## D4. ESR as one number (Open Question 3)

- Option: one number (the design's text).

## D5. `c_par` on the inductor now (Open Question 4)

- Option: yes, as designed.

## D6. Bergeron line (section 5, Open Questions 1 and 2)

- Option: not implemented; `model="bergeron"` is refused by name with a
  pointer to `model="pi"`. `model="pi"` is the default of the Python
  call (the design's default is `bergeron`, which does not exist yet).
- Why: the design makes it a kernel device (`models/transmission_line.hpp`,
  `pwl/transmission_line_history.hpp`, a pool kind, a per-step history
  source in `run_transient`, a step cap in `trbdf2`). Built from
  today's `b_extra_fn` + `step_observer` hooks it would be a Python
  observer on `pwl` only — the pattern `add-mna-native-machines` is
  retiring — and invisible to `trbdf2`.
- Reverse: implement section 5 as designed; flip the default.

## D7. π line return conductor

- Option: shunt capacitances go to `k_neg`; when `m_neg != k_neg` a
  0 V source `{name}.ret` joins them (the return conductor's impedance
  is lumped into the forward conductor; differential mode only).

## D8. `fit_rac_ladder` R_dc

- Option: when `R_dc` is not given it is a non-negative least-squares
  unknown alongside the cells; the fit minimises the relative error.

## D9. DC-bias capacitance table (section 4)

- Option: not started; it depends on the `ChargeTable` /
  charge-law `NonlinearCapacitor` of `add-power-mosfet-datasheet-device`,
  being implemented by another change.
- Update (integration, after both changes met): 4.1 and 4.3 done as
  `passives.odd_charge_table(v, Q, C)` (knots for v >= 0 mirrored into
  a `ChargeTable`; tails are the table's end-knot slopes). Open: 4.2
  `fit_mlcc_dc_bias`, 4.4 (ripple, charge conservation at 1e-9, which
  add-power-mosfet-datasheet-device measured as unreachable on pwl,
  2.2e-8, its DECISIONS 17), 4.5 YAML.

## D10. Default `n_cells` (task 3.4)

- Measured on the round-wire skin-effect curve over two decades around
  the skin corner (R_ac/R_dc from 1.0002 to 1.853): 1 cell 39 %
  (refused), 2 cells 1.75 %, 4 cells 0.60 %, 6 cells 0.18 %. The
  non-negative fit keeps only 2 non-zero cells of the 4.
- Option: keep the design's default `n_cells = 4` (2 cells reach 5 % on
  two decades; 4 leave margin for wider bands at no cost, since unused
  cells come out zero and are dropped).
- Reverse: change the default.

## D11. Tolerances that differ from the validation plan

- Phasor test (task 1.4): 1e-3 as designed; measured 5.5e-5, 3.3e-9 and
  7.1e-5 at 20, 159.15 and 800 kHz (200 steps per period).
- ESR step (task 1.5): measured with a ±1000 V pulse behind 1 kΩ (no
  pulse current source yet), expected `esr·2000/(1000 + esr)`, 1e-6.
- ESR loss (task 1.7): the design's 1e-9 is a transient average; the
  test bound is 1e-6 (measured 50.000 mW to 6.5e-11 relative).
- Ladder resistance (task 3.3): the design's 10.009 / 55.00 / 99.11 mΩ
  are rounded; they are checked to 2e-4, and the AC sweep against the
  ladder's analytic impedance to 1e-9.
- `c_par` resonance (task 1.6): 5.033 MHz is the rounded value (1e-3);
  the peak of the AC sweep equals `1/(2π√(LC))` to 1e-6 (with
  `esr = 0.1 Ω` so the peak is finite).

## D12. Bergeron line (section 5) stays deferred to the device interface

- Question (2026-11-18 session): implement section 5 now in today's
  drivers, or after `add-unified-device-interface` lands?
- Option: deferred; `model="bergeron"` keeps refusing by name.
- Why: the line is a stateful kind (a ring buffer of both ends, commit
  per accepted step, rollback on a retried step, snapshot block, a
  step cap and stage-time lookup on `trbdf2`). In today's tree a
  stateful kind is wired by hand into `run_transient.hpp` (20 sites for
  the IGBT tail history) and `trbdf2_transient.hpp` (10), plus the
  snapshot struct, the pool, the bindings and the dsed refusals; the
  device-interface branch (`feat/core-p3-device-interface`, slice 2,
  "How to add a device kind") replaces exactly those sites with one
  `StatefulKinds` list, and is being merged in batch 12. Wiring the line
  into the old sites now would conflict with that merge and be ported
  again right after. Unlike the Newton kinds the line also needs a
  linear history source in `b` (constant `1/Z0` in the cached matrix),
  which the interface's kind categories do not describe yet.
- Reverse: implement section 5 as a stateful kind on the interface
  (`pwl/transmission_line_history.hpp`, a `StateBlock`, an entry in
  `StatefulKinds`, the `b`-source hook); tasks 5.1–5.10 unchanged.

## D13. YAML `rac:` reads ladder cells only

- Option: `rac: {R_dc, L_hf, R: [...], L: [...]}` is expanded by the C++
  loader exactly as `add_inductor(rac=)` (same names, `components()`
  equal); `L_hf + ΣL` must equal `L` to 1e-9, cells > 0, `esr` and `rac`
  exclusive (as in Python). An `(f, R_ac)` table is refused by name.
- Why: fitting is a non-negative least squares in Python
  (`fit_rac_ladder`); the loader is C++. Write the fitted cells.
- Reverse: a Python pre-pass over the YAML that fits and rewrites `rac:`.

## D14. Charge conservation measured with a charge meter

- Question: task 4.4 asks for zero net charge over a ±20 V cycle to 1e-9;
  the output-grid trapezoid is not the engine's quadrature (trbdf2) and
  the 1e-12 S placeholder that `pwl/assemble.hpp` stamps on every
  NonlinearCapacitor node leaks (add-power-mosfet-datasheet-device
  DECISIONS 20).
- Option: the law in series with a 10 µF linear capacitor, no bleeder
  (`auto_regularize=False`); both are stamped in charge form by the same
  rule and carry the same current, so the meter's C·Δv is the law's
  charge under the engine's own rule; the placeholder leak at the meter
  node (1e-12 S × ∫v dt) is added to the balance. Measured 1.6e-15 (pwl)
  and 2.5e-15 (trbdf2) of Q(20 V).
- Reverse: remove the placeholder from the assembly (changes every
  result with a nonlinear capacitor) and drop the leak term.

## D15. Session 2026-11-18 scope

- Done: 3.2 (cells), 4.2, 4.4, 4.5, 6.1. Not done: section 5 (D12),
  6.2's AI_GUIDE part (the guide is at its 32 KiB limit: it gained the
  `rac=` keyword and a pointer to `fit_mlcc_dc_bias`; the ESL step and
  π-versus-Bergeron guidance are in `docs/known-limitations.md` only).
- No frequency-dependent ESR, dielectric absorption, core loss or
  temperature dependence: non-goals of this change (design.md), not open
  tasks.

## D16. A STRICT analytic case for the ESR expansion

- Question: the realistic-passive tests compare against closed forms
  inside `test_passives.py`; the reference harness had no passive with
  a parasitic.
- Option: `rc_esr_charge` in `benchmarks/analytic` (10 V through 1 kΩ
  into 1 µF with `esr=250 Ω`, observed at the capacitor terminal:
  `v = 10(1 − 0.8·e^{−t/1.25 ms})`); `k` scales the ESR, so the 1 %
  non-vacuity check bites on the parasitic itself. The terminal voltage
  jumps at t = 0 (an algebraic observable), so `exclude_t0`. Ceilings
  at twice the measured error (pwl at dt = 1 µs: at 10 µs its first-order
  start error, 1.4e-3, hid a 1 % ESR change), and a pwl golden added.
  The manifest block was inserted by hand from `freeze_analytic.py`'s
  printed numbers: its `--write` also rewrites the dsed entries of the
  static and coupled cases as "ran although refused" (the case table and
  the manifest disagree there; not this change's to settle).
- Reverse: delete the case, its manifest entry and golden; count back to 20.

## D17. Bergeron line after the device interface landed (closeout 2026-11-19)

- Question: D12 deferred section 5 to the device interface; it has
  landed (`device::StatefulKind`, `pwl::StatefulKinds`). Is section 5
  doable now without touching a time loop?
- Finding: not on the contract as it stands. A stateful kind declares
  `kFlatPerDevice`, which must equal its block's compile-time
  `reals_per_instance` (`StatefulCheck::reals`); a delay line carries
  `ceil(τ/dt) + 1` samples of `(v, i)` per end, fixed per run but not per
  kind. And every stateful kind must sit on a Newton row of `kKindCaps`
  (`StatefulCheck::newton`); the line is linear (`1/Z0` in the cached
  matrix, history sources in `b`), so as a Newton kind it would push every
  circuit holding a line onto the Newton path.
- What closes it: a variable-length state block (the in-run rollback and
  the snapshot's flat vector are already `std::vector<Real>`; the block
  table and the reader need a per-instance length), a linear "b-source"
  category whose stamp adds to `b` only (the cached matrix keeps `1/Z0`
  from the build), then tasks 5.1–5.10 unchanged. The trbdf2 step cap at
  τ can be set by the run plan as `dt_max` (above the dispatch, no loop
  hunk); stage-time lookup reads `StepCtx::t`.
- Option: left open as NOT DONE (not attempted in the closeout session);
  `model="bergeron"` keeps refusing by name.

## D18. Bergeron line: the five design questions answered before code (2026-11-19, branch feat/core-p5-line-and-excess)

Status: DESIGN ONLY in this session (the excess-loss element of
add-permeance-magnetic-domain took it). No line code is on the branch;
`add_transmission_line(model="bergeron")` keeps raising
`UnsupportedError` naming the device (D6), there is no kernel kind to
refuse anywhere else. The answers below are what the implementation
follows; each names its truth.

Law (lossless, Z0, τ; i_k, i_m into the line at each port):
i_k(t) = v_k(t)/Z0 + I_k(t), I_k(t) = −a_m(t−τ)·2/Z0 with the outgoing
wave a_m(s) = (v_m(s) + Z0·i_m(s))/2, i.e. I_k(t) = −v_m(t−τ)/Z0 −
i_m(t−τ) (and k ↔ m). Exact for the line: given exact port values at
t − τ the relation at t has no discretisation error.

### (1) Fixed step (pwl)

- Matrix: 1/Z0 across each port, in the CACHED matrix (mask independent);
  I_k, I_m enter b at the solve time, as a time-varying source does: no
  refactorisation, ever.
- The committed times are NOT a uniform grid: dt-retry sub-steps,
  discontinuity-restart stages and the post-event second sample (2) land
  off k·dt (run_transient.hpp, the retry and restart paths). So the far
  end is found by TIME on both engines — binary search of the committed
  times for t − τ — and the index shortcut "sample n − k" is taken only
  while every committed time since t − τ is exactly on the k·dt grid
  (a flag cleared by any off-grid commit). From the review.
- Lookup of the far end at t − τ: k = floor(τ/dt), θ = τ/dt − k. If
  min(θ, 1 − θ) ≤ 64·eps·(τ/dt) (τ = k·dt, or (k+1)·dt, to rounding) the
  stored sample at that time is
  used: no interpolation, and a line between resistors / sources is then
  EXACT — the lattice plateaus to rounding (gate 1e-12 relative, not
  bit-equality: two code paths). Otherwise linear interpolation between
  samples n − k and n − k − 1 with weights (1 − θ, θ). Error, derived:
  the interpolant's response to e^{jωt} is H = (1 − θ) + θe^{−jωdt}
  against the exact e^{−jωθdt}; |H|² = 1 − 2θ(1−θ)(1 − cos ωdt), so each
  pass loses θ(1−θ)(ωdt)²/2 of a wave's amplitude (DISSIPATIVE, never
  amplifying) and its phase error is O((ωdt)³); after n = T/τ passes the
  amplitude error is n·θ(1−θ)(ωdt)²/2 — second order in dt at a fixed
  end time. A step (a discontinuous wave) is smeared over one dt; its
  plateaus are exact (a constant interpolates exactly), so the lattice
  plateau values hold to rounding even for non-integer τ/dt once the
  front has passed (task 5.5's 2 % bound is on the front only).
- dt > τ: refused by name (`UnsupportedError`, feature `dt`, the line):
  the far-end value would be in the future.

### (2) Variable step (trbdf2; esdirk43)

- Bound: every stage time t_s ≤ t_n + h must satisfy t_s − τ ≤ t_n (the
  far end must be committed), i.e. h ≤ τ_min over the circuit's lines.
  Registered in the RUN PLAN as `dt_max = min(dt_max, τ_min)` (above the
  dispatch, the plan rule of add-kernel-simulation-object; no loop
  hunk); the counter `diagnostics['line_step_bound'] = {tau_min,
  n_steps_at_bound}` is computed from the accepted step list after the
  run (h ≥ τ_min·(1 − 1e-12)), so the step controller is not touched.
- Lookup: binary search in the committed, time-ordered buffer of
  (t, v, i) per port, then linear interpolation (second order: the order
  of TR-BDF2). esdirk43 (order 4/3) would be cut to order 2 by it and its
  embedded estimate would then lie: REFUSED by name on esdirk43 in the
  first slice (cubic Hermite with the stored rates is the upgrade).
- The estimator must not see the source as a jump (the defect class of
  add-in-kernel-event-losses DECISIONS 16/17: a source evaluated as E/h
  over the attempt jumps at every step boundary and grows as 1/h on a
  retry). Here I_k(t) is a FIXED continuous function of t on each step —
  the interpolant of committed samples, independent of the attempted h —
  so its value at t_n used by the trapezoid's left limit equals the value
  it had when t_n was solved (both read the same immutable samples:
  t_n − τ ≤ t_{n−1} because h ≤ τ). A retry re-evaluates the same
  function. What remains is a REAL discontinuity: a front launched at an
  event t_e arrives at t_e + τ. Rule: every landed event (switch, diode,
  source edge) of the run schedules a time breakpoint t_e + τ per line in
  the kernel event queue (one hop: a breakpoint landing schedules none,
  so the queue stays bounded; later reflections are found by the
  ordinary error control, as any kink is).
- Committed state only: samples are appended in the shared COMMIT phase
  (`StatefulKinds::commit`, once per committed time) — never in a stage
  solve, a rejected attempt or a localisation trial, which only READ the
  buffer. A post-event commit at the same t appends the right limit as a
  second sample with the same t (a lookup at exactly t_e takes the right
  limit, matching the event rows' convention). Samples older than
  t − τ − (one sample) are dropped; the buffer is bounded by
  τ/h_min_seen + 2 and refused by name above 2^20 samples per port.

### (3) Exact engine

A delay e^{−sτ} is not a finite-dimensional LTI state: `kKindCaps` row
dsed = no (refused by name through the table, as every non-LTI kind);
`engine='auto'` routes away with the reason recorded by the existing
capability check in `_route` (the reason names the kind). bdf1 = no.

### (4) Snapshot / resume

The buffer is state: a new pickle block at the next free index (19 today:
18 is the recorder) `("pulsim.lines", 1, [per line: t, v_k, i_k, v_m,
i_m arrays])`, bounded by (2) above; readers accept the 17/18/19-field
layouts with no lines and refuse by name a snapshot with lines resumed
into a circuit whose lines differ (count, Z0, τ). Truth: a run split at
any t equals the uninterrupted one (pwl: to rounding at τ = k·dt;
trbdf2: within its tolerance, as the other resume tests).

### (5) AC sweep and DC point

- AC: the EXACT two-port of the transient model, stamped in ABCD form
  with the port-m current as an extra unknown (finite at every ω; the
  Y form 1/(Z0·sin ωτ) is singular at ωτ = kπ): rows
  V_k = cos(ωτ)·V_m + jZ0·sin(ωτ)·I_m', I_k = (j/Z0)·sin(ωτ)·V_m +
  cos(ωτ)·I_m' (I_m' = −i_m). Lossy (R lumped R/4 – R/2 – R/4): the
  product of the five ABCD matrices (the transient model's own small-
  signal model, not the distributed γℓ line — they differ by O(R²/Z0²)).
  Truths: lossless delay e^{−jωτ} into a matched load (1e-9); quarter-wave
  transformer Z_in = Z0²/R_L at ωτ = π/2 (1e-9); the same number from a
  long transient's steady-state amplitude (within the transient's
  tolerance).
- DC: the line is its series resistance; lossless, a wire — v_k = v_m,
  i_k = −i_m, stamped as a 0 V branch (plus R), so the existing posedness
  refusals apply by name: an ideal voltage loop through lines refuses as
  an ideal loop; two ports with separate references keep the common mode
  undefined and refuse as a floating island. The buffer is initialised
  from the DC solution (constant waves), so a line carrying DC starts
  with no wave (task 5.7: far-end voltage constant to 1e-9).

### Closed forms the tests will use (derived here)

- Matched source (R_s = Z0), open end, 1 V step: incident wave 1/2 V, far
  end 0 before τ and exactly 1.000 V after (Γ_L = 1, Γ_s = 0).
- R_s = 10 Ω, Z0 = 50 Ω, open end: incident V⁺ = Z0/(Z0 + R_s) = 5/6,
  Γ_s = (R_s − Z0)/(R_s + Z0) = −2/3, Γ_L = 1; far end after (2n+1)τ:
  V⁺·2·Σ_{j=0..n} Γ_s^j = 1.6667, 0.5556, 1.2963 V (n = 0, 1, 2) → 1 V.
  Shorted end (Γ_L = −1): far-end voltage 0, far-end current after
  (2n+1)τ (2V⁺/Z0)·Σ_{j=0..n}(−Γ_s)^j = 33.333, 55.556, 70.370 mA → V/R_s
  = 100 mA.
- Lossy, matched both ends, R = 1 Ω on 50 Ω: Z0/(2·Z0 + R) = 0.4950495 V.
- Energy: stored energy = (1/Z0)·[∫_{t−τ}^{t} a_k(s)² ds + ∫_{t−τ}^{t}
  a_m(s)² ds] (the waves in flight); energy in − energy out = Δstored, to
  the trapezoid's order on the buffer samples.

### Kind and hot path

One kind `TransmissionLine` (two branches, one per port) with a NEW
solve class `LinearDelay` (linear matrix, b-source, state): not Newton,
so a circuit with a line stays on the cached-LU path. D17's two contract
extensions are needed first: a variable-length state block (per-instance
length; the in-run rollback and the snapshot vector are already
`std::vector<Real>`) and a b-source stamp hook in `StatefulKinds` (one
run-invariant `if (pool.has_lines())` in the shared source phase, no
per-step branch or copy for circuits without a line; to be measured at
the cost gate). Loop hunks expected: that one call in the shared phase;
the commit rides the existing `StatefulKinds::commit`; the step bound is
in the plan.

### Implemented (2026-11-20, branch feat/core-p5-transmission-line): where the code departs from the design, with evidence

- **No variable-length stateful block (D17's extensions not built).** The
  line's past lives in `pwl::LineHistory` (`pwl/transmission_line_history.hpp`),
  one object per run, owned by the engine, outside `StatefulKinds`. Each
  engine makes one `std::optional<LineHistory>`, plus a pointer that is null
  when there is no line. The hot path pays one null-pointer test per solve
  (`inject`) and per commit. Measured on line-free circuits (instruction
  counts, min of 6, interleaved): buck pwl +0.22 %, vsi pwl +0.14 %,
  buck trbdf2 +0.15 %, bridge trbdf2 +0.05 %. Byte compares unchanged:
  goldens 22/22, stateful, engines and steps identical, pwl_identity
  390/390, trbdf2_identity 371/371. The new public snapshot attribute
  `line_state` (empty) and the stats key `n_line_bound_steps` (written
  only when non-zero) are ignored when they are absent from the baseline.
- **Stamp: Thevenin per port, not Norton.** Each port is a `Source` branch
  with its own current unknown: row `v − Z0·i = E(t)` (−Z0 on the
  diagonal of the cached matrix; E into b). Same law as (1), and both
  `res.i("T")` and `res.i("T.b")` are state columns. Port names: `name`
  and `name.b`.
- **pwl commit: at the outer step end only.** `step_commit` writes the
  buffer once per output sample. The dt-retry sub-steps, restart stages and
  event iterations READ it, but their intermediate states are not committed.
  So on the fixed engine the buffer grid is the output grid, and a front
  inside a step is smeared over that step (the plateaus stay exact).
  Truth: lattice 5/3, 5/9, 35/27 V to 2.2e-16. Non-integer τ/dt = 100.5:
  plateaus exact; the sine far-end error falls with order 1.97 and 1.99
  (5.93e-5, 1.51e-5, 3.82e-6 at τ/dt = 25.5, 50.5, 100.5).
- **pwl start rule.** The waves come from x when the run starts from a DC
  point or a given `initial_state`. Otherwise the line is at rest (zero
  waves). The static (no-reactive) path and `dt > τ` are refused by name.
- **trbdf2.** `h_max = min(h_max, τ_min)` at setup. The commit is the
  first line of `device_state_once`, the funnel every commit site calls,
  and a repeat at the same t is ignored. `inject` sits in
  `accumulate_sources`. `n_line_bound_steps` (stats) counts the commits
  that advanced by ≥ τ_min·(1 − 1e-12). esdirk43 is refused by name
  (feature `integrator=esdirk43`).
- **trbdf2 landings: kinks of the recorded waves, not source corners + τ
  (D18 (2) corrected).** The design's rule (every landed event schedules
  t_e + τ, one hop) fails on the evidence:
  - With one hop (source corners only), the rejects of a matched line
    + RC load equalled the line-free circuit's (38/6 vs 36/6, 61/4 vs
    59/4, 188/2 vs 180/2 accepts/rejects at rtol 1e-3, 1e-4, 1e-6;
    before any landing 46/17, 78/27, 210/30). A mismatched line still
    lost its second arrival. The reflection the far end's RC sends back
    is not a source corner, the near end is resistive, so trbdf2 stepped
    over [2τ, 3τ] in one step of τ.
  - With eight hops (c + k·τ): the plateau error after 3τ was 0.24 V at
    every rtol.
  - Implemented instead, two rules:
    1. Source corner c + τ (one hop), landed and restarted as the corner
       itself is.
    2. A committed sample k where a recorded wave, at either port,
       changes slope by Δs with |Δs|·h_max/4 > atol + rtol·|w| (the error
       of stepping over it) lands a step at t_k + τ, with no restart
       (`LineHistory::next_kink_after`, in `tb_clamp`). This covers
       reflections, curved transients and fronts from switch or diode
       events, and it decays with the waves.
  - Two rejected variants:
    - A chord-deviation test missed a corner whose restart probe sits
      0.1 fs after it: the resistive lattice was off by 0.46 at rtol 1e-3.
    - Restarting at every kink: rejects 13/11/8 vs 6/4/2, loose-rtol error
      7.9e-4, resume off by 2e-6.
  - Evidence for the final rules:
    - Matched line + RC load, accepts/rejects: 123/6 vs 36/6, 139/4 vs
      59/4, 390/2 vs 180/2 at rtol 1e-3, 1e-4, 1e-6. The rejects are
      equal; the extra accepts are the landings.
    - Mismatched line + RC load, far end against pwl at dt = 0.05 ns, worst
      of nine plateau points: 9.4e-5, 2.4e-5, 1.7e-5, 1.1e-6 at rtol 1e-3,
      1e-4, 1e-5, 1e-6. The floor near 1e-6 is the reference's own error.
    - Resistive lattice 5/3, 5/9, 35/27 V exact to 2.2e-16 at rtol 1e-3
      to 1e-8.
    - Resume vs uninterrupted run: 1.1e-15.
- **What the error control does not see (a limit, documented).** The
  line's interpolation error is not part of the TR-BDF2 estimate, and the
  landings above only carry what a step already recorded. On a circuit
  whose only dynamics is the line, a smooth wave is never refined: a
  3.3 MHz sine through 100 ns is off by 6.6e-3 V (of 0.5 V) at rtol 1e-3
  and 1e-6 alike (0.207 V before the kink landings). The cure is the
  user's step ceiling (`dt=`) or pwl. An error estimate for the line's
  interpolant is the upgrade.
- **Snapshot.** Pickle field 19 (0-based; the 20th field), `line_state`
  = [n_lines, per line Z0, τ, n, n × (t, w_a, w_b)]. It is appended only
  when non-empty, so a line-free snapshot keeps its 19-field tuple. The
  reader accepts 17 to 20 fields. A mismatch (count, Z0, τ, truncation, no
  state) is refused by name. For the integration batch: if another branch
  also takes index 19, renumber; the line block is the only one that is
  optional by length.
- **AC.** As (5), stamped as a per-ω correction (G's rows of the two
  ports are replaced by the ABCD rows). The lossy line is the builder
  composite, so its AC model is the five-matrix product, as designed.
- **DC.** As (5). Port b's row is `v_b − v_a = 0` and port a's is
  `i_a + i_b = 0`. The voltage-loop finder treats port b as a 1:1 winding
  of port a, and the structure analysis gives port b ZeroImp at DC, as for
  the ideal transformer.
- **Exact engine.** `engine='auto'` never routes to the exact engine
  (`_route`: "'auto' never routes to dsed"), so there is no route to
  record; `engine='exact'|'dsed'` and bdf1 refuse it through the table.
- **R ≤ Z0.** The lossy composite refuses R > Z0 by name (the lumped
  thirds are then a poor line). The `R ≪ Z0` warning of task 5.1 is this
  refusal.
