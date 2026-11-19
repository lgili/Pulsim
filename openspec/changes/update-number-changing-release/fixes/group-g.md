# Group G — the thermal summary on branch-form devices (F-2-1 = F-6-3), F-3-2, F-3-5

Branch `fix/core-release-g-thermal-summary`, on the PR head `origin/feat/core-integration` = 27c00520
(extension `so_27c00520`, Python only, nothing built). Tests: `python/tests/test_release_findings_g.py`.

## F-2-1 / F-6-3 — `device_thermal_summary` ignored `conduction_specs` on a branch-form device

Reproduction (PR head, `form='branch'` explicit): the DC diode of `test_conduction_model` (10 V,
g_on = 1e3, 1 Ohm), `conduction_specs={'D': {V_f0: 0.7, r_on: 0.02}}`: thermal `P_cond_avg`
0.0998 W, loss summary 8.98901 W, closed form `V_f0*I + r_on*I^2` with `I = 10/1.001` = 8.98901 W.

Trace (`.scratch/thermal_digest.py` + `agree.py`: both summaries over a device kind x form x engine
matrix — DC diode, buck with the MOSFET helper + body diode / IGBT / MOSFET-with-body-diode / ideal
switch, the bridge-rectifier composite, a Shockley diode; pwl, trbdf2, dsed; with and without
`conduction_specs`; whole run and `t_from` window): 592 device rows, 76 disagree, and ALL 76 are
`form='branch'` WITH `conduction_specs` (diodes, switches, body diodes; e.g. buck Q1 1.010 W vs
2.564 W, D1 0.0203 W vs 2.576 W). Zero disagreements in the conductance form, in the branch form
without specs, or for the ideal (R_on = 0) elements.

Cause: the per-kind dispatch in `thermal.py` tested `params['branch_form']` FIRST and read the
device's solved `v*i` (the averages from the loss summary's `E_total`, i.e. `P_avg`), while
`device_loss_summary` tests `finite(g_on, g_off)` first: a branch-form device with `R_on > 0` goes
through its conductance law there and gets the conduction model. Two dispatches, two answers.

Fix (at the cause, once): `losses.conduction_power_avg(entry)` — `P_cond_model_avg` where the model
was applied, else `P_avg` — is the ONE reading of a device's conduction loss; the thermal summary's
`P_cond_avg` is that of the device's loss-summary entry (its own `E_cond` integrals are deleted), and
its `P_cond(t)` trace (the T_j driver) picks the model from the same entry (`form == 'branch'` = the
ideal element's solved v*i; otherwise the kind's law with `conduction_specs`). Silent fall-backs
removed: a `thermal_specs` device of a kind with no loss model (source, capacitor, a future kind
such as `excess_loss`) is refused by name (ValueError); a supported device the loss summary left out
warns by name; the loss summary itself now warns where it dropped a device silently (a level-1
IGBT with a tail; a switch with neither a recorded mask nor `switch_fn`).

Truth: closed form on the DC diode (conductance and branch form): `P_cond_avg` = 8.98901 W to
1e-9 and `T_j = 25 + R_th*P` (tau << dt) to 1e-9. Every kind x form x engine (pwl, dsed) x specs x
window: thermal `P_cond_avg` == loss entry to 1e-12, and the T_j trace's time average ==
`R_th*P_total_avg` (tau << dt) within 3 % of the largest device power (the sample mean of a
switching waveform) — before the fix these were off by 100x.

Default-path numbers: none moved (the PR line's default form is the conductance form; the digest's
conductance rows are byte-identical, `P_cond_avg` included). Branch-form rows WITHOUT specs: averages
unchanged; the T_j trace now reads the conductance-law current instead of the solved branch current
(they agree to 1.6e-10 A): T_j moves <= 3e-13 K on pwl / trbdf2; on dsed the step-start insertion
of `_with_step_starts` (exact `!=`) picks a different set of samples, T_j_avg moves <= 5.2e-6 K and
T_j_peak <= 2.5e-4 K (bridge D4, rise 0.088 K: 0.3 %, the documented resample accuracy on a variable grid).

## F-3-2 — the LCP refusal's `.device` is `'D (branch #1)'` — NEEDS C++, left xfail

Reproduced on the PR head: `mode_selection='legacy'` -> `'D'`; `'lcp'` / `'auto'` -> `'D (branch #1)'`.
Cause (code): `run_transient.hpp:961` fills `mode_diode_names` with `pwl::branch_label(graph, id)`
(`row_names.hpp:141`, "name (branch #id)") and passes `mode_diode_names[j]` as the
`UnsupportedError`'s device (`run_transient.hpp:1178`); the same vector feeds
`events::raise_if_refused` (1215) and the repair list (1341); `trbdf2_transient.hpp:1251` builds the
same vector. The fix needs C++: pass `std::string(graph.branch_name(e.branch_id))` as the device
(keep the label in the message), in both engines and in `raise_if_refused`'s device field; then
drop the xfail of `test_ideal_voltage_loop_refusal_names_the_device[lcp]` and extend it over every
refusal of `raise_if_refused` that carries a device.

## F-3-5 — a control-graph build refusal arrived as `SimulationAborted` under `control_tick='state'`

Reproduced: `Rate(5e-8)` graph, `dt=1e-7`: `label` -> `ControlGraphBuildError` (raised by
`set_solver_dt` in `run_transient_with_chain`, before the run); `state` -> `SimulationAborted`, same
text. Cause: a queue-driven graph is not given the solver step before the run; `BlockChain` runs
`check_periods()` at its first in-run call (`chain.hpp:786`), inside the kernel loop, which re-wraps
the exception. Fix: `_control_plan.for_fixed_step` runs the same check on each queue-driven graph's
group periods before the run (`_check_graph_periods`, `ControlGraphBuildError`). It only raises;
`set_solver_dt` is NOT called on that path, so no state of the graph changes (byte-identical runs).

## Review (fresh-context, read-only) and what was done
- The matrix test's average check was a tautology after the fix: it now checks the thermal summary's
  OWN T_j trace (tau << dt: T_j - T_amb = R_th*P(t)) against R_th times the LOSS summary's power, per
  device, 3 %. Against the PR head's thermal.py this check alone fails all 16 branch-form x
  conduction_specs cases (and passes the other 48). The closed form now also covers a switch
  (conductance and branch form) through `conduction_specs`.
- The "no entry" warning no longer points to a loss-summary warning that is not always raised.
- Left: a conductance-form device booked impulsive energy would not get it in the thermal offset
  (only branch-form devices add `P_impulse_avg`, as before; not verified whether that can happen).
  The F-3-5 pre-check refuses three edge cases the in-run C++ check let through (a one-step run, a
  resume, a graph reused after a label run) — in each the refusal is the physically right verdict.
  CHANGELOG not edited (shared file; for the integration batch): `device_thermal_summary` now refuses
  `thermal_specs` on kinds without a loss model (lauritzen diode, saturable inductor, machines ...)
  where it used to return no row.
