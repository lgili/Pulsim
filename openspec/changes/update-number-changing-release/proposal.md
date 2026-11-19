## Why

CORE_ROADMAP decision 3: every number-changing default flips together, in
one release, with one re-baseline through the reference harness. Until then
each improvement sits behind an option that defaults off. The options exist
on the integrated tree (`feat/core-integration` fe75ee01). This change is that
release. It decides nothing new: each flipped value was decided by the change
that owns it. It lists them in one place, gates the flip on the blockers the
rehearsal found, and carries the re-baseline.

The rehearsal (`benchmarks/tools/next_defaults_rehearsal.py`,
`benchmarks/NEXT_DEFAULTS_REHEARSAL.md`) ran every reference circuit and
analytic case under today's defaults, each option alone, the whole profile and
`engine='auto'` under the release route. Its ranked blocker list is task group 1.

## What Changes

The profile is data in `benchmarks/tools/next_defaults.py`. The release makes it
the library default:

- **BREAKING** fixed-step engine: `discontinuity_restart=True`
  (`add-discontinuity-restart-rule`, DECISIONS 4; roadmap decisions 2–3:
  SDIRK(2,2)×3 and the gate mask sampled at the step start).
- **BREAKING** fixed-step engine: `enable_substep_state_correction=True`
  (`add-fixed-step-event-splitting`, DECISIONS 1; it requires the rule).
- **BREAKING** switch / diode model: `add_switch` / `add_diode` build the branch
  form (`add-branch-form-ideal-switch`, DECISIONS 23, 39; design D6: a given
  `g_on` means `R_on = 1/g_on`; `R_on = 0`, `g_off = 0` legal).
- **BREAKING** `mode_selection='auto'` on every engine
  (`add-complementarity-mode-selection`, DECISIONS F5).
- **BREAKING** `control_tick='state'` (`add-compiled-control-graph`, G30/G32).
- **BREAKING** `device_path='interface'` on pwl and trbdf2, and the hand stamps
  deleted (`add-unified-device-interface`, DECISIONS 23/25).
- **BREAKING** `engine='auto'` without `dt` routes an exact-eligible run to the
  exact engine, the rest to trbdf2; with `dt` to pwl
  (`add-exact-propagator-strategy` D11, task 19.1, its default-flip gate).
- TR-BDF2 step grid (`update-trbdf2-factor-reuse` task 6.1) flips only if its
  option exists by then; the C++-only TR-BDF2 options
  (`update-trbdf2-post-event-restart` DECISIONS 9–10) stay as decided there.
- The composite diode drop (`update-composite-diode-forward-drop`) is already the
  default on fe75ee01 (the builders' `V_F` is the conduction drop): nothing to
  flip; its CHANGELOG entry moves under this release's heading.
- NOT flipped: `integrator='esdirk43'` stays opt-in
  (`add-high-order-stiff-integrator`).
- Owner decisions carried here: the eight per-engine position-dependent blocks
  of `add-kernel-simulation-object` DECISIONS 23 (undriven switches first).

## Impact

- Affected specs: `transient-timestep` (the profile is the default set).
- Affected code: the defaults in `python/pulsim/__init__.py` (`simulate()`),
  `CircuitBuilder.add_switch` / `add_diode`, the option table
  (`sim::kRunOptionTable`), the `engine='auto'` route; the deletions the owning
  changes list (hand stamps).
- Regenerated: goldens, identity baselines, `REFERENCE_STATUS.md` and the
  triage freezes, `EXACT_SCOREBOARD.md`, docs and examples that print numbers;
  CHANGELOG migration notes with the old values reachable by keyword.
