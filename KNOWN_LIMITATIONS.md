# Known limitations carried into v1.3

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

* `.subckt` / `.ends` — subcircuit definitions are recognised in the
  lexer but not flattened into the builder.
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
