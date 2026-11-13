# Known limitations carried into v1.3

> **For Pulsim 2.0, read [`docs/known-limitations.md`](docs/known-limitations.md).**
> It lists the limitations re-verified on 2.0, each with its measured
> effect and a workaround. This file is the 1.3 catalogue. Its sections
> were not re-verified for 2.0, except two. The "Transient engines
> (2.0)" defects still reproduce; the chatter circuit aborts at 20, 10,
> 5 and 2 ns and runs at 50 ns, 2.4 % low. Its Shockley workaround is a
> model change: the output falls by 16.6 %. Two entries are out of date: `.subckt`
> is now flattened, and since 1.4.0 the AC sweep has factorised with
> the in-house `PulsimComplexSparseLuSolver`, not `Eigen::SparseLU`.
> `add_nonlinear_diode` is retired (its law sources power below
> `V_F0`); see the 2.0 page.

This file catalogues every gap we knowingly ship in **pulsim 1.3.0** —
deliberately deferred work, partial implementations, and features whose
roadmap is open in [`openspec/changes/`](openspec/changes/). Use it as the
honest companion to [`CHANGELOG.md`](CHANGELOG.md): the changelog says
what's *in*, this file says what's still *out*.

The list is organised by area. Each entry links to the tracking artefact
(OpenSpec proposal, task ID, or follow-up plan) so the next contributor
can pick it up without re-discovering the context.

---

## Post-hoc analysis

### Per-device loss reporting — sub-step transient resolution

`pulsim.device_loss_summary(builder, result)` walks every
**resistor**, **inductor**, **ideal-switch** and
**switched-diode** branch and reports `i_avg`, `i_rms`, `i_peak`,
plus `P_avg`/`E_total`. The switch path requires the caller to
forward the same `switch_fn=` that drove the simulation. The
diode path infers the conducting interval from the node-voltage
drop and `V_th`. PSIM-style datasheet annotations are accepted:

* `diode_specs={"D1": {"Q_rr": ...}}` or
  `{"E_rr_ref": ..., "V_R_ref": ...}` — reverse-recovery energy
  per turn-off event, accumulated from
  `SimulationResult.commutation_events`.
* `switch_specs={"M1": {"E_on_ref": ..., "E_off_ref": ...,
  "V_ref": ..., "I_ref": ...}}` — MOSFET/IGBT turn-on / turn-off
  energy scaled linearly by the (V_blocking, I_load) at each
  switch_fn edge.
* `core_loss_specs={"L1": {"material": "N87", ...}}` —
  Steinmetz / iGSE inductor core loss from `pulsim.magnetic`.

`pulsim.device_thermal_summary(builder, result, thermal_specs=...)`
pipes the same loss data through a per-device Foster network to
produce `T_j(t)`.

What is **still not** covered post-hoc:

* **In-sim switching-transient waveforms** (nanosecond V/I overlap
  at the switching edge) — the kernel's discrete trap step doesn't
  resolve sub-dt transitions, so the on-state `i_SW` and the
  off-state `v_SW` are interpreted as block-averaged samples.
  The PSIM-style annotation captures the right *energy*, not the
  *instantaneous* trajectory. For waveform-level accuracy on a
  given event, drop the dt to ≪ T_sw or use a sub-step model.

**When LossAccumulator still helps.** A state-aware
`step_observer` calling `LossAccumulator.add_sample(P_cond, dt)`
+ `add_switching_event(E_sw)` remains the right tool when you
need per-event energy bookkeeping driven by a custom datasheet
curve (e.g. `E_on(I, T_j)`) that doesn't fit the
linear-scaling `*_specs` shape.

---

## Schematic renderer

### Default backend and layout quality

`pulsim.schematic.render(builder, path)` uses the pure-Python layout
path by default: a topology template when a recogniser matches (RC /
RL / RLC filters, buck with `add_switch` + `add_diode`, flyback,
half-bridge, phase-shift full bridge, ...), force-directed placement
otherwise. Until 2.0.x the default was the `python_native` ELK backend,
which raised `AttributeError` on every 2.0 `CircuitBuilder` (0 of 25 v2
examples rendered); it is opt-in now (`PULSIM_SCHEMATIC_BACKEND=python_native`,
needs Node.js and `elkjs`) and was not exercised end to end when the
default changed, because no elkjs was installed.

Limits measured on the 25 v2 examples (all render to SVG):

- 9 of them match a template and lay out like a textbook drawing. The
  other 16 (MOSFET-with-body-diode bucks and boosts, three-phase
  stages, the op-amp LDO, ...) go through force-directed placement:
  symbols can overlap and a switch symbol's leads do not always touch
  its wires. Treat that output as a connectivity sketch.
- PNG needs `cairosvg` and the cairo C library; without the library an
  `ImportError` names it (the SVG is still written).
- The Tier-2 topology classifier calls the Anthropic API when
  `ANTHROPIC_API_KEY` is set, unless `PULSIM_LLM_LAYOUT_HINTS=0`.

### Per-cell position hints

`pulsim.schematic.render(...)` no longer exposes a `position_hints=`
keyword. The previous prototype always raised `NotImplementedError`
when a non-empty dict was passed — neither the deprecated `netlistsvg`
backend nor the new `python_native` backend ever shipped a working
implementation. v1.3 removes the dead kwarg so the public surface
matches what actually works (auto-layout only).

**Why the prototype never landed.** netlistsvg 1.0.2's `--layout` flag
has two real upstream bugs documented in the
[`add-schematic-position-hints`](openspec/changes/add-schematic-position-hints/)
analysis: the Promise path renders into `undefined` when `elkData` is
supplied, and overriding `(x, y)` on cells without recomputing the
cached edge `sections` produces tangled wires through empty space.

**Roadmap.** The full topology-aware renderer (template library +
LLM-assisted classification + constraint-aware ELK layout) is tracked
as [`add-schematic-renderer-v2`](openspec/changes/add-schematic-renderer-v2/).

---

## SPICE importer

`pulsim.spice_to_builder` parses the everyday subset — `R`, `C`, `L`,
`V` (DC/SINE/PULSE), `D`, `M` — and the standard engineering suffixes.
The intentional gaps for v1.3:

* ~~`.subckt` / `.ends` — subcircuit definitions are recognised in the
  lexer but not flattened into the builder.~~ Flattened since then
  (checked on 2.0: an `X` instance of a one-resistor subcircuit loads
  its node as expected).
* `Q` (BJT) / `J` (JFET) — device models are not implemented.
* `K` — coupled inductors are deferred until the magnetics layer
  exposes a stable Python-facing mutual-inductance helper.
* `.param` expressions — only numeric assignments; algebraic forms
  (`{R*2}`) raise.
* Behavioural sources (`B`, `E` with expressions) — not parsed.

For complex netlists, the YAML loader (`pulsim.load_yaml_file`) or
hand-built `CircuitBuilder` is the recommended path.

VDMOS cards keep their charge since add-power-mosfet-datasheet-device
(CGS, CGDMAX/CGDMIN/A, CJO/VJ/M, TT, RG → `add_power_mosfet`); still
dropped: subthreshold conduction (KSUBTHRES), FC linearisation, and BV
when TT makes the body diode Lauritzen. `charge="drop"` restores 2.0.

---

## MMC stack (Phase 20)

| Layer | Status | Notes |
|---|---|---|
| L0 — averaged arm | ✅ shipped + cross-validated against Sousa (2022) Cap 4.3 / 5 |
| L1 — discrete multilevel arm (PS-PWM, IPD) | ✅ shipped |
| L2 — SM-equivalent (dead-time + min-pulse-width) | ✅ shipped |
| L3 — detailed per-SM with balancing | ⚠️ shipped but slow; full O(N_sub²) sort each step |

The L3 layer is correct but not optimised for large submodule counts.
For N_sub ≥ 16 prefer L2 with a separate balancing observer until a
faster sort path lands.

---

## M3C converter (Phase 22)

The M3C project ships the heuristic Fast-SVM selector **and** the
Dead-Beat Predictive Controller (DBPC) introduced in Phase 22.13. They
target different operating regimes:

* **Heuristic SVM + cost-function selector** (default) — fast, works
  for f_out ≠ f_in, validated against Gili (2024) Tab. 16.
* **DBPC** (`m3c_dbpc.py`) — recommended when f_out crosses f_in (the
  Phase 22.14 motor-ramp scenario) or when capacitor pre-charge
  matters (Phase 22.15). Heavier per-step cost, but produces lower
  capacitor ripple at the f_out ≈ f_in singularity.

Pick the controller per-application; both are public API in v1.3.

### Input-side current + capacitor outer loop

The Phase 22.8/22.9 input-side dq controller and capacitor outer loop
are shipped as closed-loop infrastructure, but they are tuned for the
Tab. 16 operating point. Other operating points (large f_in/f_out
ratios, weak grids) need re-tuning; reproducible PI presets ship with
the DBPC variant instead.

---

## Sparse LU stack

### Complex-scalar path

`pulsim::sparse::PulsimSparseLuSolver` covers every real-scalar path
in production. The AC small-signal sweep
(`pulsim::analysis::mna_sweep`) is the **only** remaining call site
using `Eigen::SparseLU<std::complex<Real>>` instead of the in-house
solver. The directive is to drive that last third-party LU out of
production code paths.

**Roadmap.** Templating `PulsimSparseLuSolver` on `Scalar` is tracked
as [`add-pulsim-complex-sparse-lu`](openspec/changes/add-pulsim-complex-sparse-lu/);
the Gilbert-Peierls + path-based partial-refactor algorithms
generalise to `std::complex<Real>` with no algorithmic change.

### Runtime integration of `solve_rank1`

The path-based partial-refactor fast path is exposed at the
`PulsimSparseLuSolver` level and exercised by the benchmark suite,
but the Layer 5 `run_transient` driver does not yet call
`solve_rank1` for single-bit Gray-code switch transitions. Wiring
that integration is tracked as `add-pwl-rank1-runtime-integration`
(TBD — see CHANGELOG § 1.3.0 *Not changed*).

### First visit of a switch state: full factorisation

A lazy cache factorises every newly visited mask in full, on the
shared symbolic analysis. Tier 2 of `update-linear-time-assembly`
(`partial_refactor` on the nearest resident mask's factor) was
measured on 2026-11-13 and not landed: on the switched ladder most
single-switch flips exceed the 0.6·n path gate and the first visit got
slower. Tier 3 (stored reciprocal pivots, 1.2× on the triangular
solve) is held until the TR-BDF2 step grid stops amplifying last-bit
differences beyond the 1e-9 gate. See that change's DECISIONS.md,
items 13–14.

---

## LiveScope GUI

LiveScope (`pulsim.scope.LiveScope`) ships with single-panel real-time
plotting, GIL release, and downsampling. The polish backlog —
multi-panel layout, cursors, channel statistics — is queued as
**task #145** (`LiveScope polish — multi-panel + cursors + stats`)
for the v1.4 cycle.

---

## Controlled and waveform current sources (2.x)

- Linear only: VCVS, VCCS, CCCS, CCVS and the gyrator have constant
  gains. There is no behavioural (expression) source, and SPICE `B`
  sources are refused. Use `b_extra_fn` or a closed loop for a
  nonlinear law.
- The piecewise-linear table is a current source only
  (`add_pwl_current_source`); there is no PWL voltage source.
- Engines: `pwl`, `trbdf2`, `dsed` and the DC operating point run all
  seven kinds. The AC sweep (`run_mna_sweep`) runs the four controlled
  ones; the waveform current sources are independent and add no AC
  excitation. `trbdf2` lands on table knots and pulse corners through a
  local step clamp, until `update-trbdf2-error-control` provides
  breakpoints.
- A YAML `pwl_current_source` `file:` path is relative to the working
  directory, not to the netlist, and is not echoed by `components()`.

---

## Transient engines (2.0)

### Which engine runs which device: one table, typed refusals

The kernel's capability table (`pulsim.engine_capabilities()`, mirrored
in `docs/known-limitations.md` and checked by
`test_engine_conformance.py`) decides which engine simulates which
device kind. Newton devices (smooth / Shockley / Lauritzen diode,
level-1 MOSFET / IGBT, saturable inductor, junction capacitor, PMSM) run
on `pwl` and `trbdf2` only; `dsed`, `run_transient_bdf1` and the kernel
AC sweep refuse them with `pulsim.UnsupportedError`. Before the table, a
junction capacitor on `dsed` / `bdf1` was silently absent (6.3186 V
instead of 1.0216 V). The retired `add_nonlinear_diode` is opt-in only
(`allow_non_passive=True`). Feature keywords an engine cannot honour
refuse the same way, with `feature` set to the keyword. Controllers
(`closed_loops`, `step_observer`) run on `pwl` and `trbdf2` only; `dsed`
refuses them.

### Ideal switch (branch form): opt-in, first slice

`add_ideal_switch` / `add_diode(form='branch')` solve the switch current
as an unknown, so `R_on = 0` and `g_off = 0` are representable. This
slice runs the switch on `pwl` and `trbdf2` and the diode on `pwl`
(`docs/known-limitations.md`, branch-form table); other engines refuse
with `feature='branch_form'`. A mode that makes a state dependent (an
inductor opened by an ideal switch with `g_off = 0`, two capacitors
paralleled through `R_on = 0`) is refused with
`feature='dependent_states'` until the re-initialisation service lands;
an island bounded only by open ideal elements (OFF ideal diodes
included: the gauge of D3 is not implemented) with `floating_node`; a
loop of closed ideal elements and voltage sources with
`ideal_voltage_loop`. On `trbdf2` the dependent-state check is static
(any ideal element that could form one is refused up front).

### Fixed engine: first order after a discontinuity (by default)

`engine='pwl'` keeps the trapezoidal derivative across a gate edge, a
diode flip, a source jump and an inconsistent start: first order and a
Nyquist alternation after each event (buck CCM `i_L` error 1.18e-2 A
at T/50, order 1.00; DCM idle `v(sw)` alternating 0.48 / 26.1 V).
`discontinuity_restart=True` takes the step containing each one and the
next two as SDIRK(2,2) steps (1.17e-7 A, order 2.0; idle within 0.05 %)
and samples the gate at the step start. Off by default until the flip
of the migration plan.

Limits with the rule on:
- an edge off the `dt` grid stays first order unless
  `enable_substep_state_correction=True` splits the step at it (order
  2: RL edge +0.3 dt 1.01e-6 A at 2 µs; buck D = 0.4567 `v_C` 1.07e-6 V
  at T/50 instead of 8.08e-2 V). The split needs an edge oracle
  (`next_edge_after`): an opaque `switch_fn` is still taken at the next
  grid point (counted in `res.counters["unlocated_gate_steps"]`, one
  `UserWarning`);
- a `b_extra_fn` jump restarts, and with the split is split at, only
  when the callable declares it through `next_jump_after(t)` (RC step
  at 2.037 µs: order 1.97); undeclared ones stay first order;
- a step that needs the dt-retry ladder is re-taken without its
  interior events (the next step restarts on the new mask);
- the split is second order on the integrated devices (MLCC charge
  table, DC machine, controlled sources, pulse / sine / table current
  sources, diode with `V_f`: orders 1.9-2.0 to T/800, probe
  `split_devices.py`) but not on a power MOSFET switched through its
  gate: its channel crosses `V_T` between grid points, which is no
  event, and the run is not second order with the flag off, the rule
  or the split (identical with the edges on the grid);
- the event table keeps one factor per mask: the split's dt1 / dt2
  solves and the restart stages at 2γ·dt refactor the same entry in
  turn (VSI 10 ms at 1e-7 with the split: 2395 event refactors, 307
  ns/step against 252 with the rule alone);
- an MNA PMSM retries every fixed step at dt/2 with the rule on or off,
  so its steps are never split.
- a diode that turns off inside stage 2 of a restart step because of
  the SDIRK overshoot (dt/τ ≫ 1, a rectifier charging its bus) is
  re-tried as backward Euler from stage 1: that step is first order
  (bus within 1e-4 of the source peak, but 0.76 V max error vs trbdf2
  at dt/τ = 10, 0.36 V at dt/τ = 1).

Resolution diagnostics (`res.resolution`, `pulsim.ResolutionWarning`):
the truncation estimate covers linear capacitors and inductors only
(the stateful devices are seen by the ringing detector through their
node voltages and branch currents); it is skipped on steps whose
three-point stencil straddles a mask change, restart, split, retry or
built-in source jump (an undeclared `b_extra_fn` jump is not seen); it
is an estimate, not an error bound. Thresholds (1e-2; 1 % over four
steps) come from one prototype and five circuits. Cost with it on:
+5 % to +14 % ns/step (design gate: 3 %), so it is OFF by default
(`resolution_diagnostics=True` turns it on). With the shipped examples
under the example test's all-gates-OFF drive, six warn: buck,
buck_converter, stress_buck_switching (an inductor behind a 1e-9 S
switch rings at Nyquist: true), half_bridge_pwm and
stress_large_rc_ladder (truncation 0.67 / 0.056 at their dt: true), and
flyback (`T1.s` 1.27e3: the normalisation of a winding current that is
~0 everywhere — a false positive the 1e-3-of-kind floor does not
remove).

### DSED exact propagator: what is still numeric or refused

`engine='dsed'` steps a mask with constant sources exactly (tier 1
eigen, tier 2 Padé; `result.exact["masks"]`). A mask that passes
neither bound (tier 3: stiff and non-normal, e.g. a 1e12 Jordan block)
runs on the old numeric path for that mask, recorded with its reason;
the implicit hand-off and a strict refusal need the unified stepper
(`add-exact-propagator-strategy` task 2.4). Masks with PWM / pulse
voltage sources or a sine / pulse / table current source are numeric too
(task 8; tier 3 in `result.exact`); sine voltage sources are exact
(an augmented oscillator state). Capacitor loops (parallel
capacitors) are refused by the extractor, so the charge-conserving
start for inconsistent `c0=` values is not reachable yet. Coupled
inductors stay refused (task 7).

**Two diode-mode guards on `dsed`.** The zero-band settle of the
exact-propagator slice can keep a diode ON through reverse current (or
OFF while forward-biased). The complementarity guard (default; see the
next section) judges every settled mode and repairs the wrong ones with
the LCP: the boost (12 V, 100 µH, 47 µF, 100 kHz, D = 0.5, 3 ms) with
V_th = 0 at 20 / 50 / 500 Ω and V_th = 1e-3 at 20 / 50 Ω now ends on
trbdf2 rtol 1e-10 (7 / 16 / 23 / 1 / 3 repairs), and the half-wave RL
rectifiers run right (2 and 1 repairs). The batch-4 stored-sample guard
stays: it checks every stored sample against the diode law and raises
`DiodeModeInconsistentError` (a `RuntimeError`, in
`pulsim._dsed_dispatch`), and it catches what the kernel guard does
not: the `V_th = 0` LC tank of
`test_double_crossing_inside_one_step_is_seen` (the kernel guard repairs
one event and returns v(k) = -5.18 V at 10 µs; trbdf2 9.50 V), and every
case above when `MODE_GUARD = False`. It reads stored samples only, so a
wrong mode shorter than one stored step is not seen. Workaround for the
double crossing: `V_th = 1e-3`.

### Complementarity mode selection: dsed only; double crossings at `V_th = 0`

`mode_selection='lcp'` and the default-mode guard run on `engine='dsed'`
only; `pwl` and `trbdf2` still decide with their settle and refuse
`'shadow'`/`'lcp'` by name (hooks: add-complementarity-mode-selection
task 4.2). The guard and the LCP decide at events; a commutation the
event locator misses is not decided. With `V_th = 0` a diode that turns
on at exactly `v = 0` starts its next step with the turn-off predicate
at 0, which the dsed locator skips, so a return through zero inside that
step is lost: an LC tank into a series diode stays ON at −18 mA from
2.25 µs (`test_double_crossing_inside_one_step_is_seen`'s circuit). A
band of `V_th = 1e-3` gives the exact windows. The guard repairs only
the modes it judges inconsistent over `1e4·h_e`; a mode consistent that
long and wrong later is not caught.

### Fixed-step controller ticks land on the `dt` grid

On `pwl` a controller tick is delivered at the first grid point at or
after `t_start + k*T`. When `T` is not a multiple of `dt` each tick is
late by less than `dt` (`simulate` warns; `result.control_tick_jitter_max`
records the largest lateness); `T < dt` is refused. `trbdf2` lands every
tick exactly. Pick `dt = T/n` for an exact fixed-step cadence.

### TR-BDF2 stalls at `h_min` after a hard switching edge

With a Newton device in the circuit (a saturable transformer, a
saturable inductor, a smooth junction), `engine='trbdf2'` — and
`engine='auto'` when it routes there — can abort right after a hard
switching edge with `Newton failed at t = ... with h already at
h_min`. Both Newton norms are at their noise floor; the worst residual
sits on a LINEAR inductor row scaled by `2L/h`. Reproducer: the
snubbed flyback in
`python/tests/test_saturable_transformer.py::test_engines_agree_below_saturation`
(strict xfail), characterised in
`python/tests/test_newton_stall_and_mask_chatter.py`.

**Do not loosen `tol_newton_dx` / `tol_newton_res` to get past it.**
That flyback, re-run with `tol_newton_dx=1e-5, tol_newton_res=1e-4`,
completes and reports V_out = 3.53 V; the fixed engine
(`engine='pwl', dt=1e-8`) gives 15.31 V. The absolute step test is
load-bearing: making it scale-aware lets the DC cascade stop at a
spurious root 10.67 V away (CHANGELOG 2.0.0, *Two engine defects
characterised, and one tempting fix refuted*).

**Workaround.** Run the circuit on `engine='pwl'` with an explicit
`dt`. **Fix.** Needs the step controller not to drive `h` to `h_min`
after an event landing, or a row-equilibrated Jacobian — not a
tolerance change.

A saturable inductor used as a buck's own inductor loses its whole
current at a turn-off edge on TR-BDF2 (`v_sw` = 2.6e7 V; comment above
`_gated_buck` in `test_trbdf2_stateful_with_diodes.py`; a fix is in
progress). Use `engine='pwl'` with a `dt`.

### TR-BDF2 still refactors on every new step size

`engine='trbdf2'` solves through one event solver per switch state.
Since `update-trbdf2-factor-reuse` (parts 1-2) the table holds every
visited state within the segment byte budget (it used to hold 8, so
the three-phase VSI rebuilt entries 3,198 times in 20 ms) and a refactor
allocates nothing — but each new `h` or stage fraction is still one
numeric factorisation (10,716 refactors in a 10 ms buck run), because
the controller's step sizes are continuous. Landing steps at an event
refactor whatever the grid. Memory: one `J`, its `G`/`C` values and one
LU factor per visited state, released only by the budget's LRU
(`CacheMetrics.event_evictions`). **Fix.** Part 3 of the proposal (a
geometric step grid with factor slots) is a numerical change awaiting
the owner's accuracy sign-off.

### Newton: what the persistent analysis does not cover

Since `update-newton-persistent-solver` (stages A1 and A2) a run
analyses the Newton Jacobian once per sparsity pattern and refactorises
it in place with the in-house LU, without allocating. Still per
iteration: one full numeric factorisation, although only the device
columns change. Stage B (the low-rank update) was measured on
2026-11-13 and not landed: `partial_refactor` on the device columns
gains ≤ 1.26× and never runs on the rectifier; Woodbury gains 1.87× at
n = 483 but changes the iteration trace and diverges on two golden
circuits (see that change's DECISIONS.md, item 17). The Levenberg-Marquardt fallback and the line-search
trial refresh build their matrices afresh on every attempt. A user
refresher whose stamp positions depend on `x` changes the pattern and
re-analyses on each change (correct, just slower); a refresher that
stamps nothing on a call keeps the explicit zeros of earlier calls in
`J_nl`. The DC operating point and continuation (calls without a run
workspace) still use `Eigen::SparseLU`.

### Fixed engine: diode-mask chatter inside Newton at a small `dt`

With PWL diodes (`add_diode`) in the same circuit as a Newton device
(saturable or hysteretic magnetics, a smooth junction),
`engine='pwl'` can abort with a Newton failure that survives every dt
halving (`run_transient: the step ending at t = ... could not be
taken`). The diode mask is re-decided inside every Newton iteration,
and at a stiff step no single mask is self-consistent: the iterate
bounces while the residual sits near zero (at `dt = 10 ns`, after 64
sub-steps: `||dx|| = 1.46 V` at `vout` against a residual of
1.9e-9). Reproducer: the PSFB rectifier in
`python/tests/test_newton_stall_and_mask_chatter.py` (strict xfail
`test_a_newton_device_survives_a_stiff_step`).

**A smaller `dt` makes it worse, not better**: that circuit aborts at
`dt` = 10, 5 and 2 ns and runs at 50 ns. Loosening the Newton
tolerances is not the remedy either (see the section above).

**Workarounds.** A coarser `dt`, or smooth Shockley junctions
(`add_shockley_diode`) for those diodes — with the four rectifier
diodes swapped, the same circuit runs at 10 ns and agrees with the
linear reference to 0.13 %. That reference also uses Shockley
diodes. The swap changes the model: each junction drops about 0.7 V
and a PWL diode drops none. The output, `v(vout) - v(rect_neg)` over
1.5-2 ms, is 5.785 V against 6.938 V with PWL diodes (-16.6 %).
At 50 ns with PWL diodes it is 6.771 V, 2.4 % low.

### Reference benchmark suite: what runs on 2.0

The harness reader maps 111 of the 126 `pulsim-v1` benchmark circuits;
108 complete on `engine='pwl'`. Skipped (15): eight closed loops built
from control blocks other than PI → PWM (`hysteresis`, `integrator`,
`limiter`, `lookup_table`, `gain`, `pid_controller`, `rate_limiter`,
`sample_hold`), four PLL / dq circuits, a p-channel MOSFET, the v1
saturable-inductor law, and a MOSFET with a loss/thermal block. Failing
on `pwl` (3): `buck_igbt` (Newton does not converge),
`ll05_diode_rectifier_filter` (no consistent DC diode configuration),
`ll13_active_clamp_forward` (no DC operating point).
`ll16_flyback_closed_loop` completes since the Newton roundoff
acceptance, through 1,645 roundoff accepts with a diverged state
(I(Lm_core) ≈ 2.4e13 A from t = 0); a fix that makes it loud again is in
progress.
The differential matrix (strict gate: every observable, every interval)
records 65 `disagree` / `error` cells over 55 circuits as `known:` in
`benchmarks/manifests/v2_triage.yaml`,
plus `sepic` on pwl and trbdf2, whose trbdf2 reference grinds to
`max_steps` (a TR-BDF2 regression, not frozen; integrated tree,
2026-11-13, `benchmarks/REFERENCE_STATUS.md`). Each is an
open finding, not a tolerance. Circuits whose v1 file has no `uic: true`
start from the DC operating point, which `dsed` cannot take (refused).

### First stored sample of algebraic nodes

On `engine='pwl'` and `'trbdf2'` the first stored sample is the initial
state, not the solution at `t = 0⁺`: an RL step reads 0 V across the
inductor at `t = 0` where the answer is 10 V (`dsed` returns 10 V).
Comparisons in the reference harness exclude that sample; the fix is the
consistent-initial-conditions solve.

---

## Machines (`add-mna-native-machines`, step 1)

`b.add_dc_machine` is solved in the matrix; the PMSM (observer path),
BLDC and induction-motor observers are not yet, and stay first order
on `pwl`. Not done: `theta_node` (refused; integrate `ω` with `add_vccs` into a 1 F capacitor), the induction
machine, BLDC harmonics, flux maps. None of the machine tests
is a validation against a measured machine: they check analytic
properties of the equations.

---

## Passives (`add-realistic-passives`, composites)

ESR/ESL/`c_par`, the `R_ac` ladder and the π-section line are networks
of R, L, C (YAML reads `esr`, `esl`, `c_par`; `rac:` is refused by
name). Not done: YAML `rac:` and `transmission_line`, the MLCC DC-bias
table (needs the charge-law `NonlinearCapacitor`), the
Bergeron line (needs a kernel device; `model="bergeron"` is refused).
The π line overshoots an ideal step by 27 % at any section count (the
exact line does not): give the source a rise time and use
`recommend_line_sections`. A fitted ladder reproduces its own data; it
is not a validation, and it extrapolates outside its band as its last
cell does.

---

### Composite MOSFET / IGBT diodes: a zero-width band

The body / freewheeling diodes of `add_mosfet`, `add_mosfet_with_body_diode`,
`add_mosfet_level1` and `add_igbt_level1(with_fwd=True)` are
`add_diode(V_th=0, V_f=V_F)`: they turn on and off at the same voltage
(no hysteresis band). A diode that turns on from zero current at a
dead time could chatter on `dsed`; an H-bridge of four `add_mosfet` with
200 ns dead time settles (measured), and a run that does not settle
or settles a mode its law forbids raises a named error
(`DiodeModeInconsistentError` for the latter). For a band, build the device from `add_switch` and
`add_diode(V_th=V_f + 1e-3, V_f=...)`. The composites keep a fixed
1 mΩ diode slope (`g_on_diode` only on `add_mosfet_with_body_diode`).
`add_igbt` is still a bidirectional switch with no knee and no diode;
`add_igbt_level1` is the IGBT that blocks reverse current.

## Open OpenSpec proposals as of v1.3.0

The following proposals are accepted as in-flight work but **not**
required for the v1.3 release. They will land incrementally on `main`
without blocking the tag.

| Proposal | What it adds |
|---|---|
| [`add-long-simulation-stress`](openspec/changes/add-long-simulation-stress/) | Multi-second / high-freq stress benches + drift-conservation KPI |
| [`add-pulsim-complex-sparse-lu`](openspec/changes/add-pulsim-complex-sparse-lu/) | Templated `PulsimSparseLuSolver` for AC sweep |
| [`add-schematic-renderer-v2`](openspec/changes/add-schematic-renderer-v2/) | Topology-aware renderer + LLM-augmented classifier |
| [`add-soft-switching-validation`](openspec/changes/add-soft-switching-validation/) | ZVS / ZCS KPI helpers + LLC / PSFB benches |
| [`add-textbook-reference-cases`](openspec/changes/add-textbook-reference-cases/) | Erickson / Mohan / Kassakian reference comparisons |
| [`replace-klu-with-pulsim-sparse-lu`](openspec/changes/replace-klu-with-pulsim-sparse-lu/) | Already shipped via v1.3.0 — proposal kept as historical record |
| [`update-electrothermal-component-observability`](openspec/changes/update-electrothermal-component-observability/) | Per-component electrothermal contract in `loss_summary` / `thermal_summary` |

---

## How to add to this list

When you hit a "we'll fix this later" decision while implementing a
change, capture it here in the same commit as the change itself. A
clear entry has:

1. **What is missing** — the user-visible gap, one paragraph.
2. **Workaround** — what the user can do *today* to get the same
   result, even if less ergonomic.
3. **Roadmap** — the OpenSpec proposal, task ID, or named follow-up
   that owns the closure.

Keeping the catalogue honest is cheaper than answering the same Slack
question for the third time.
