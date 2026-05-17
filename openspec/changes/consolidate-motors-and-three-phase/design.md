## Context

The motor and three-phase work in Pulsim grew organically: the `motor-models` and `three-phase-grid` specs landed in May 2026 as part of the original 16-change roadmap, but only the simplest pieces (PMSM dynamic, DC motor, balanced 3φ source) were wired through to the `DeviceVariant` union and the YAML / Python surface. The rest of the spec — BLDC, induction, three-phase VSI, RL load, saturable transformer, hysteresis inductor — accumulated as math-object headers under `motors/`, `grid/`, `magnetic/` without Circuit integration.

When the team opened four parallel proposals (`add-motor-drive-benchmarks`, `add-three-phase-control-blocks`, `integrate-three-phase-motors-magnetics-into-circuit-variant`, `add-three-phase-grid-tied-suite`), each proposal made a different assumption about whether the math objects would be available as device-variant primitives or had to be hand-assembled from voltage sources + inductors. The benchmark and grid-tied changes assumed the latter — they planned to **rebuild** what should have come from finishing the device-variant integration.

This document records the consolidation decisions.

## Goals / Non-Goals

**Goals:**
- One source of truth per motor model. No `PmsmSteadyStateParams` vs `PmsmDevice` ambiguity.
- One source of truth per three-phase source. `ThreePhaseSourceParams` composes the `grid::` math object instead of duplicating fields.
- Close the `motor-models` spec gap: ship `BldcMotorDevice` and `InductionMotorDevice`.
- Finish the device-variant integration for the remaining motors (FOC, mechanical) and magnetics (saturable transformer, hysteresis inductor).
- Author the missing benchmark YAMLs against the consolidated device API.
- Single PR, single review, single release window.

**Non-Goals:**
- New analytical motor physics. BLDC trapezoidal back-EMF and induction-motor dq equations are textbook; no novel modeling.
- Real-time / HIL code generation for these motors (owned by the separate `add-realtime-code-generation` change, already archived).
- AC small-signal analysis on the new devices (owned by `add-frequency-domain-analysis`; will work once the AD path is the default — separate concern).
- Replacing the shipped `add-three-phase-control-blocks` virtual-block implementation. Those blocks are already done and used here.

## Decisions

### Decision 1: `PmsmSteadyStateParams` is removed, not deprecated

- **What**: Delete the struct, the `add_pmsm_steady_state()` method, the pybind11 binding, and the seven tests built around it.
- **Why**: The struct exists only because PmsmDevice wasn't integrated when it shipped. The source comment (`runtime_circuit.hpp:2849`) explicitly says "use `add_pmsm()` (future)". There are no production callers — only test fixtures of itself. A deprecation cycle adds noise without removing the duplication; a single PR removal is cleaner.
- **Alternatives considered**:
  - *Deprecate with warning for one release* — Rejected. There are no external callers to warn (the type is only used by its own tests). Deprecation does not eliminate duplication, it merely defers it.
  - *Keep as a reduced-complexity variant* — Rejected. The dynamic `PmsmDevice` with ω fixed via IC reproduces the steady-state op-point exactly. Keeping a second model invites drift.

### Decision 2: `ThreePhaseSourceParams` composes `grid::ThreePhaseSource`; the existing API is preserved

- **What**: Refactor the struct to embed a `grid::ThreePhaseSource` member; rewrite the accessors (`v_rms`, `frequency`, `phase_rad`, `sequence`) as forwarding accessors. Add a new C++/Python overload `Circuit::add_three_phase_source(name, nodes, grid::ThreePhaseSource)` for callers that prefer the math object directly.
- **Why**: ~13 C++ and ~12 Python call sites use `ThreePhaseSourceParams` today. Removing the struct outright would break the surface for no benefit. Composition removes the duplicated field set while preserving every existing call. The new overload gives forward-looking code (e.g., the FOC-tuning examples) the canonical math object.
- **Alternatives considered**:
  - *Remove ThreePhaseSourceParams entirely* — Rejected. Too many callers; cost of migration not justified by the structural cleanup.
  - *Make `ThreePhaseSourceParams` an alias of `grid::ThreePhaseSource`* — Rejected. The Circuit-side struct will eventually grow Circuit-specific fields (node sign convention, neutral handling, balance enforcement) that don't belong on the pure math object.

### Decision 3: BLDC and induction land as proper device-variant classes, not YAML primitive macros

- **What**: Implement `BldcMotorDevice` (trapezoidal back-EMF with six-step commutation awareness) and `InductionMotorDevice` (squirrel-cage in stationary αβ frame with rotor flux as state) as `DynamicDeviceBase` subclasses in `components/`, with math objects in `motors/`.
- **Why**: The `motor-models` spec lists both as shall-provide. Building them as YAML primitives (per the `add-motor-drive-benchmarks` original plan) would:
  1. Create a second parallel implementation per motor (the primitive macro vs the real C++ class that someone will eventually add).
  2. Lose the integration with mechanical-load coupling that `Mechanical` device requires.
  3. Fail when the catalog work later wants to attach manufacturer parameters (Kv, pole pairs, slip curves).
- **Alternatives considered**:
  - *Defer BLDC / induction to a separate follow-up change* — Rejected per user instruction; in scope here.
  - *Reuse `PmsmDevice` for BLDC with a trapezoidal-back-EMF flag* — Tempting but rejected. BLDC commutation logic, current waveform shape, and torque ripple model differ enough from PMSM that a shared class would carry two distinct branches throughout; cleaner as separate types.

### Decision 4: Single PR, even though it's large

- **What**: All four phases (A code dedup, B integration finish, C BLDC + induction, D benchmarks) land in one PR on branch `feat/consolidate-motors-and-three-phase`.
- **Why**: The user explicitly chose this. Phase A's removal of `PmsmSteadyStateParams` breaks the steady-state tests; if A and B/C ship in separate PRs, the intermediate state has a broken or skipped test. Phase D benchmark YAMLs only make sense against the devices added in C. A single coherent merge avoids partial states.
- **Alternatives considered**:
  - *Three sequential PRs (A+B / C / D)* — Rejected per user. Adds review overhead and time to each phase.

### Decision 5: The three absorbed changes are **deleted**, not archived under `archive/`

- **What**: `openspec/changes/add-motor-drive-benchmarks/`, `add-three-phase-grid-tied-suite/`, and `integrate-three-phase-motors-magnetics-into-circuit-variant/` are removed with `rm -rf`. Their content is captured in this change's `proposal.md` for traceability.
- **Why**: These were proposals that had not yet shipped meaningful code; deleting is cleaner than carrying a `superseded-by` archive entry. The 5/24 already-shipped tasks of `integrate-...` are captured by the device-variant work that already exists in `runtime_circuit.hpp` (`add_three_phase_source`, `add_pmsm`, `add_dc_motor`) — those stay, and the remaining 19/24 tasks land here.
- **Alternatives considered**:
  - *Move to `openspec/changes/archive/2026-05-17-superseded-by-consolidate/`* — Rejected per user; adds noise to the archive directory.

### Decision 6: `add-three-phase-control-blocks` is archived separately, ahead of this change

- **What**: The 15/15-shipped control-blocks change moves to `openspec/changes/archive/2026-05-17-add-three-phase-control-blocks/` as its own commit, the `kernel-v1-core` spec is updated to record the virtual-block requirements, all before this consolidation's Phase A begins.
- **Why**: It's actually done. Archiving it independently keeps this consolidation focused on the motor/3φ device integration; the virtual-block work is a separate (already-complete) story.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Removing `PmsmSteadyStateParams` breaks downstream forks/tests we don't see | The struct has no production callers and the comment already flags it as temporary; if any external user complains post-release, the dynamic `PmsmDevice` covers the use case. |
| BLDC trapezoidal-EMF model isn't accurate for some motors | We pick the canonical Krause trapezoidal model (120° flat-top, 60° linear). Catalog-specific waveform tuning is the catalog tier's job. |
| Induction-motor dq integration on stiff op-points needs Newton damping | Already handled by the existing Newton stack. If a benchmark requires it, the robustness profile knobs are enough. |
| Large PR (10+ files, 3000+ lines) is hard to review | Commit history is structured per phase (A.1, A.2, B.1..., C.1, C.2, D.1...) so reviewers can read sequentially. |
| Existing `ThreePhaseSourceParams` callers depend on a specific field order or POD layout | The struct is not part of any binary ABI (header-only). Composition preserves the field accessors that callers use. |

## Migration Plan

1. **Phase 0 (this PR's first commits)**:
   - Archive `add-three-phase-control-blocks` (already done at 15/15).
   - Delete the 3 absorbed change folders.
   - Land this change's scaffolding (`proposal.md`, `tasks.md`, `design.md`, spec deltas).
2. **Phase A**: Code dedup — refactor `ThreePhaseSourceParams`; remove `PmsmSteadyStateParams` and migrate its 7 tests to `PmsmDevice`. Build + tests green.
3. **Phase B**: Device-variant integration completion for the remaining motors and magnetics. New `Mechanical` domain in scheduler.
4. **Phase C**: BLDC + induction motor implementations with unit + transient tests.
5. **Phase D**: New benchmark YAMLs + KPI baselines.

Rollback: revert the merge commit. Composition is a non-breaking change; the device additions are pure additions; the only breaking removal is `PmsmSteadyStateParams`, whose absence does not affect any production path.

## Open Questions

- Should `PMSM_FOC` device be a separate type from `PmsmDevice + PIController + Clarke/Park virtual blocks`? *Lean: separate type, since FOC tuning lives at the controller level and the device should expose the dq currents directly to whichever controller the user wires.*
- Three-phase VSI: should it stay as the `ThreePhaseVsiParams` POD structure or also get the composition treatment with `grid::ThreePhaseInverter` (if one exists)? *Lean: same treatment as the 3φ source — compose if the math object exists; otherwise defer to a follow-up.*
- Induction-motor field-oriented control: should the FOC controller for IM be the same as for PMSM (different rotor flux model)? *Lean: separate `IndirectFocController` since the field orientation logic is the distinguishing piece, but reuse the Clarke/Park virtual blocks unchanged.*
