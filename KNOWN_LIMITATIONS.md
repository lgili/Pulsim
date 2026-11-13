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

---

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
