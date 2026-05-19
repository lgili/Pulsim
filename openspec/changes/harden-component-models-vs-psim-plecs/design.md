## Design Notes

This change is structured to **land convergence wins early** (Phase A
ships in 1.5 days and immediately unblocks the audit's V_gs = 0 cutoff
failure and the IGBT V_CE_sat fidelity bug), then add the larger
**user-facing model fidelity work** (Phase B, body diodes + motor
thermal — 3 days) and **magnetics fidelity** (Phase C, saturable
transformer + Steinmetz — 4 days) as separate landings with their own
validation runs.

The three phases are intentionally independent — Phase A can land
without B or C; B can land without C. This lets us merge incrementally,
keep CI green between landings, and decouple the body-diode work (which
touches every MOSFET / IGBT integration test) from the magnetics work
(which is mostly net-new device variants).

### Why Norton-shift, not piecewise-linear, for IGBT V_CE_sat

The IGBT today computes `i_C = g_eff · V_CE` with `g_eff` smoothly
interpolated by σ_g · σ_d sigmoids. To match the datasheet drop, two
options:

1. **Piecewise:** when ON, force `V_CE = V_CE_sat`. Doesn't work with
   Newton — the device's row would be a hard equality constraint, and
   you'd need a separate branch variable. Breaks the MNA structure.
2. **Norton-shift (chosen):** when σ_g approaches 1 (gate fully on),
   the conductance line passes through `(V_CE_sat, 0)` instead of
   `(0, 0)`. Implemented as a current offset: `i_C = g_eff · (V_CE −
   V_CE_sat · σ_g)`. At V_CE = V_CE_sat with σ_g = 1, i_C = 0 — matches
   the datasheet curve. At V_CE >> V_CE_sat, i_C ≈ g_on · V_CE — same
   slope as before (R_CE_on). At V_CE < V_CE_sat (reverse conduction
   without body diode), i_C goes negative through the linear extension
   — this is the same artifact MOSFET has today, addressed separately
   in Phase B.

This mirrors the existing diode pattern (`ideal_diode.hpp:619`) and
keeps the AD path's autodiff identity-true to the manual stamp.

### Why default `enable_auto = true` + `apply_only_in_recovery = false`

The `ModelRegularizationOptions` struct exists but its main feature
(per-device-class `g_off_min` floors) is gated behind two opt-in flags
that effectively keep it OFF by default. The audit shows that flipping
both defaults to ON addresses the floating-gate singularity (Phase A2's
gate anchor is the model-level fix; the regularization is a
solver-level safety net).

Trade-off: tests that rely on exact `g_off = 1e-12` will see `g_off ≥
1e-7` (the MOSFET floor) and might assert at the 13th decimal. We
inspect those tests in A6.3 and either widen the tolerance or pin the
regularization OFF for the specific test.

### Why body diode default ON

The audit (§ 1.1, § 1.2) notes that every real power MOSFET / IGBT has
an intrinsic body / antiparallel diode. Users who model with PSIM /
PLECS expect it. Defaulting to ON makes new circuits "just work" for
synchronous-rectification topologies. Users who explicitly want the
no-body-diode behaviour (rare — only for SPICE-parity bench tests) can
set `body_diode_enable = false`.

This default flip is the only behavioural-incompatibility risk in
Phase B. Existing tests that assume V_sw goes below GND during dead
time will need updating to either expect the clamp or set
`body_diode_enable = false`. We document this in the migration guide.

### Why a separate `SaturableTransformer` device, not extending Transformer

`Transformer` is 122 LOC of ideal-N-turns math. Bolting saturation +
leakage + winding R + core loss onto that file would balloon it past
800 LOC and break the simple `add_transformer(name, np, nn, ns, sn,
N)` constructor that's used in dozens of existing tests.

A separate `SaturableTransformer` device wrapping
`magnetic/saturable_transformer.hpp` (168 LOC math object that already
exists) keeps:
- Existing tests untouched (Transformer stays as the ideal-N case).
- The math object as the single source of truth.
- Discoverability: a user typing `Circuit::add_saturable_transformer`
  knows immediately they're getting the saturable model.

The pattern parallels `HysteresisInductorDevice` wrapping
`magnetic/hysteresis_inductor.hpp` — same shape.

### Why phase the work into A → B → C (not A → C → B)

The audit ranks B (body diode + motor thermal) ahead of C (saturable
transformer) by user impact for typical converter sims. But B touches
every MOSFET / IGBT integration test in the suite (default flip on
`body_diode_enable`), so we deliberately land Phase A first (low risk —
all opt-in additions or single-line defaults that are easy to revert),
then Phase B (one default flip, but heavy test coverage required),
then Phase C (net-new device, doesn't touch existing tests at all).

This sequencing minimises rollback exposure: if B's default-flip causes
unexpected regression we can revert B alone without touching A or C.

### Out of scope

The audit lists 10 items; this change implements all 10. But three of
the audit's "nice-to-have" finer points are deferred to follow-on
proposals:

1. **DC motor field-winding variants** (separately-excited / shunt /
   series). Audit § 3.1. Adds 4 new motor types' worth of math.
   ~5 days. Better as its own change `add-dc-motor-field-windings`.
2. **Switched reluctance motor (SRM).** Audit § 3.4 #1. ~7 days.
   Better as `add-srm-motor-model`.
3. **Coupled inductor saturable variant.** PSIM ships this; we have
   the math via `magnetic/saturable_inductor.hpp` but the saturable
   inductor *device wrapper* is also missing. ~2 days. Roll into a
   follow-on change `wire-saturable-inductor-device`.

Each is a clean addition that doesn't depend on this change. We surface
them in the audit's Top 10 → follow-on cross-reference table at the end
of `docs/component-models-audit.md`.
