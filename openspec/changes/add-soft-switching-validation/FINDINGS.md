# Findings — add-soft-switching-validation

Tree: `feat/core-integration` b9e46778 (extension `so_b9e46778`). Tests:
`python/tests/test_soft_switching_validation.py`. Truths are closed forms
in charge / energy, mpmath quadrature and mpmath's Taylor ODE solver on the
dead-time state equations; none comes from a Pulsim run.

## New

### F-S-1 — esdirk43 cannot start a charge-form capacitor whose initial current is not zero
`engine='trbdf2', integrator='esdirk43'` aborts at t = 0 (Newton fails, h at
h_min) when a `add_nonlinear_capacitor` carries current at t = 0. Minimal: 0 V
source, 1 mOhm, `add_nonlinear_capacitor(C0=1 nF, V0=25, m=0)` (a LINEAR law),
10 A current source: `||dx|| 3.7e-6`, `||residual|| 7.0e-6` at the KCL row.
`add_capacitor` in its place runs; plain trbdf2 runs both. In the half-bridge
of case B (inductor i0 = -0.97 A) it aborts the same way (`||residual||
1.45e-6`); with i0 = +1.03 A it runs and agrees with the ODE truth.
Strict xfail: `test_esdirk43_starts_a_current_driven_nonlinear_capacitor`.

### F-S-2 — `compute_switching_loss_per_event` was not an energy (FIXED here)
`benchmarks/kpi.compute_switching_loss_per_event` summed `0.5*|v*i|` of the
pre-transition samples (watts) and divided by the span: W/s, reported as
"switching_loss_w_avg". Now `int v*i dt` over a required `window_s` after
each transition (`switching_energy_j` added); `benchmark_runner` passes the
KPI entry's `window_s`. Synthetic test against 50/6 uJ.

## Known items measured here (cited, not new)

Hard turn-on of the half-bridge, total switch + diode energy in a 2 ns + 20 ns
window from `device_loss_summary`, against the closed form:

| engine | case A (truth 62.83 uJ) | case B, ZVS lost at 61.8 V (truth 1.688 uJ) |
|---|---|---|
| pwl, restart off | 2.57 uJ | 0.091 uJ |
| pwl, restart on | 1.44 uJ | 0.065 uJ |
| trbdf2 | 1381.7 uJ (22x) | 2.196 uJ (+30 %) |
| trbdf2 + esdirk43 | 62.92 uJ (+0.15 %) | F-S-1 |

* fixed engine: books almost none of E_oss (the 1 ps collapse under a 1 ns
  step; open item);
* trbdf2: overbooks (fixed on `fix/core-trbdf2-state-error-norm`, not merged);
* fixed engine without `discontinuity_restart`: the gate is read at the
  step's end, so a transition starts one step early (crossing times
  -0.49 ns on 31 ns); a diode clamp rings step by step (5 A / 0 A around
  2.5 A); both are AI_GUIDE trap 9.

## Passing (truth within the stated tolerance)

Transition time to 300/200/100 V (charge closed form): pwl+restart, trbdf2,
esdirk43 within 0.05 ns (seen <= 4 ps); pwl within one step. ZVS fraction
(low side 1.0, high side 0.0) on all four. Resonant trajectory, trough v_min
(energy closed form, 61.8 V) and full-ZVS clamp on pwl+restart and trbdf2
within 0.5 V; esdirk43 for the full-ZVS case (F-S-1 for the partial one).
`engine='dsed'` refuses the nonlinear C_oss by name.
