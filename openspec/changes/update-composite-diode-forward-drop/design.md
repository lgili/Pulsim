## Context

`SwitchedDiode` has two voltages since
`add-switched-diode-forward-drop`:

- `V_th` — commutation threshold. It only decides OFF → ON.
- `V_f` — conduction drop. ON law `i = g_on·(v − V_f)`; turn-on at
  `V_on = max(V_th, V_f)` (`V_th` alone when `V_f = 0`); turn-off at
  zero ON current.

The composites were written before `V_f` existed:

| Builder | Diode branch | Today |
|---|---|---|
| `add_mosfet`, `add_mosfet_with_body_diode` | `<name>_body` | `g_on_diode` (1e3 S), `V_th = V_F` (0.7), `V_f = 0` |
| `add_mosfet_level1(with_body_diode=True)` | `<name>_body` | 1e3 S, `V_th = 0.5`, `V_f = 0` |
| `add_igbt_level1(with_fwd=True)` | `<name>_fwd` | 1e3 S, `V_th = 0.5`, `V_f = 0` |
| `add_igbt` | none | bare switch |

In PR #145 the composites kept `V_F` as a threshold as a deferred
default, not as a decision; the review (D9) asks to finish it.

Probe scripts: `core_review_2026-11/models/p6b.py` (0.010 V / 0.10 W),
and two written for this proposal, in `probes/` of this change folder:
`d9b_forced.py` (gate on / off, level-1) and `d9_buck_emulated.py` (the
buck, with the future composite emulated as `add_switch` +
`add_diode(V_th=0, V_f=0.7)`). They run read-only against the built
extension.

## Goals / Non-Goals

- Goals: the diode inside every composite drops `V_F` while it
  conducts; one meaning of "forward voltage" across `add_diode`, the
  bridge, the VSI and the composites; the turn-on voltage does not move.
- Non-Goals:
  - An IGBT with `V_ce0` or reverse blocking
    (`add-pwl-igbt-and-thyristor`).
  - Reverse recovery, junction capacitance, temperature.
  - A `V_f(I, T)` table (`add-in-kernel-event-losses`).
  - Renaming keywords (see Open Questions).

## Decisions

- **Mapping: `V_f = V_F`, `V_th = 0`.** Pure PSIM, the same choice as
  `add_three_phase_vsi`. `V_on = max(0, V_F) = V_F`, so a diode that
  turned on at 0.7 V still turns on at 0.7 V; only the ON law changes.
  - Alternative *`V_th = V_F`, `V_f = V_F`*: identical behaviour
    (`max` of equal numbers); rejected as it suggests two knobs.
  - Alternative *`V_th = V_F + 1 mV`* (a hysteresis band for DSED and
    bridges): the measured buck settles on `dsed` without it; a hidden
    1 mV is a number the user did not write. Left as an Open Question.
- **Level-1 devices get a `V_F` keyword, default 0.7 V**, appended after
  the existing arguments so positional calls keep working:
  `add_mosfet_level1(..., with_body_diode=True, V_F=0.7)`,
  `add_igbt_level1(..., with_fwd, tau_tail, k_tail, V_F=0.7)`.
  - Alternative *keep 0.5 V as the default drop*: 0.5 V was a
    commutation threshold chosen for robustness, not a diode voltage.
  - With the level-1 MOSFET gated on in the third quadrant, the channel
    and the diode are in parallel; the diode now conducts only when the
    channel drop exceeds `V_F`, which is the physical behaviour.
- **Validation** is the diode's own: `V_F` finite and ≥ 0, error naming
  the composite and `V_F`.
- **No change in Python consumers.** `_diode_state`, `losses`, `thermal`
  and `res.i` already take `V_f` from `components()`.
- **Zero-drop identity.** With `V_F = 0` the diode is `V_th = 0`,
  `V_f = 0`: it turns on at 0 V, not at the old threshold. This is a
  different diode from today's default, so "bit-identical at the old
  default" is not offered; the migration snippet in proposal.md is.

## Affected files (grep of proposal.md)

- `examples/` (30): `boost_realistic_igbt.yaml`,
  `boost_realistic_mosfet_v14.yaml`, `boost_saturable_inductor.yaml`,
  `buck.yaml`, `common_source_amplifier.yaml`, `flyback.yaml`,
  `ldo_with_opamp.yaml`, `pwm_chopper_realistic_mosfet.yaml`; notebooks
  `05_nonlinear_devices`, `14_api_reference`,
  `15_new_components_catalog`, `18_buck_mosfet_pwm_block`,
  `29_electrothermal_yaml_validation_modes`, `buck_converter_design`;
  scripts `debug_buck_bode_offset`, `run_ac_sweep_buck`,
  `run_adversarial_sweep`, `run_auto_tune_buck`,
  `run_boost_realistic_igbt`, `run_boost_realistic_mosfet_v14`,
  `run_boost_saturable_inductor`, `run_buck`, `run_buck_chain_logged`,
  `run_buck_closed_loop`, `run_buck_closed_loop_chain`,
  `run_buck_with_helper`, `run_common_source_amplifier`,
  `run_ldo_with_opamp`, `run_live_scope_buck`,
  `run_pwm_chopper_realistic_mosfet`.
- `python/tests/` (24): `test_builder_kwargs`, `test_closed_loop_helper`,
  `test_engine_auto`, `test_engine_auto_closed_loop`,
  `test_engine_default_routing`, `test_igbt_no_reverse_conduction`,
  `test_igbt_tail`, `test_loss_window`,
  `test_mosfet_body_diode_default`, `test_mosfet_third_quadrant`,
  `test_named_lookups`, `test_newton_auto_lm_promotion`,
  `test_pwm_grid_edges`, `test_python_bindings`,
  `test_result_i_all_kinds`, `test_saturable_transformer`,
  `test_schematic_render_default`, `test_shipped_yaml_examples`,
  `test_snapshot_resume_stateful`,
  `test_stateful_device_stamps_compose`, `test_stored_kind_coverage`,
  `test_switch_state_record`, `test_trbdf2_robustness`,
  `test_trbdf2_stateful_devices`.
- `core/tests/` (5): `builder/test_circuit_builder.cpp`,
  `dsed/test_trbdf2_transient.cpp`, `layer2/test_igbt_level1.cpp`,
  `layer2/test_mosfet_level1.cpp`, `yaml/test_yaml_loader.cpp`.
- `docs/` (12): `api-reference.md`, `gotchas.md`,
  `migration-from-v1.md`, `migration-guide.md`,
  `internals/layer2-v1-mosfet-igbt.md`,
  `internals/layer2-v2-transformer.md`,
  `internals/layer8-yaml-parser.md`,
  `internals/layer9-smps-showcase.md`, `tutorials/02`, `04`, `05`,
  `06`.
- `projects/` (10): the `*_pulsim_validation.py` scripts under
  `converters/{boost,boost_pfc,buck,buck_boost,flyback,forward,half_bridge}`
  and `inverters/pfc_vsi_drive`, plus the buck notebook and its builder.
- `python/pulsim/` (4): `AI_GUIDE.md`, `__init__.py`, `discovery.py`,
  `spice_import.py`.

Not every file moves: only those where the diode conducts. The count is
the search space, not the damage.

## Risks / Trade-offs

- A script that passes `V_F=0.7` and was tuned against the old numbers
  shifts by a few percent with no error → CHANGELOG entry at the top of
  the release, the buck figures quoted, and the same wording in
  `AI_GUIDE.md` trap 1.
- Level-1 devices: the turn-on voltage of `_body` / `_fwd` rises from
  0.5 V to 0.7 V. Circuits that relied on the lower threshold to catch
  an inductive kick see a 0.2 V higher clamp.
- `V_th = 0` with `V_f > 0` is a zero-width band. The previous change
  recorded that a zero-current turn-on on `dsed` may fail to settle
  (named error suggesting `V_th = V_f + 1e-3`). The buck probe settles;
  a bridge of composites is the case to test.
- Closed-loop examples re-converge to the same set-point with a larger
  duty; their expected duty values move.

## Migration Plan

1. Land in a minor release with the BREAKING note first in the
   CHANGELOG, together with or right after
   `add-switched-diode-forward-drop`, so users see one diode migration.
2. Update the 85-file search space: re-derive expectations where a
   diode conducts.
3. PulsimGUI: its composite devices call these builders; its expected
   conduction values need the same pass (separate repository).

Rollback: the four builder bodies and the two keywords; no data format
changes.

## Open Questions

1. **Keyword name.** Keep `V_F` on the composites (proposed: no rename,
   only the meaning changes), or rename to `V_f` to match `add_diode`,
   `add_bridge_rectifier` and `add_three_phase_vsi`, accepting `V_F` as
   a deprecated alias for one major?
2. **Threshold keyword.** Add `V_th=0.0` to the four builders (as
   `add_three_phase_vsi` got), giving a one-keyword way back to the old
   diode (`V_F=0.0, V_th=0.7`) and a hysteresis band for DSED? Not
   proposed: the explicit `add_diode` covers it.
3. **Default for the level-1 diodes.** 0.7 V for both (proposed), or
   keep 0.5 V, or a different default for the IGBT co-pack diode?
4. **Default for `add_mosfet`.** Keep `V_F = 0.7` (proposed; every
   default call changes), or make the default 0 so that only explicit
   `V_F=` calls change?
5. **Slope.** The composites keep `g_on_diode = 1e3 S` (1 mΩ). Should
   they also accept an `R_d` so a datasheet `V_F0 + r_d·I` line can be
   entered, as the VSI helper does?
6. **Release.** Ship as 2.1 with the other Phase 0 number changes, or
   hold for the release that carries `add-discontinuity-restart-rule`?
