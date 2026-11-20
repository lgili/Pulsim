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

### F-S-3 — the exact-truth tool lost ~18 digits in its slow coordinates (truth tool, FIXED here)
`benchmarks/tools/exact_truth.py` raised "no consistent diode mode" at the
first gate-off of an LLC half-bridge with an ideal 4:1 transformer and a
diode bridge (t = 3.7 us, dps 30) and of a PSFB (the draft's 1.8 us; the
benchmark netlist raises at its start, t = 0).
Cause, measured: `Struct` took the basis of the slow subspace from the
columns of Eh^(index+1) (Eh = (lam E - A)^-1 E, lam = 1e4). Those columns
are a power iteration of Eh, aligned to the slowest mode to
(lam/|mu_fast|)^(index+1) ~ 1e-27 (fast modes to 3e14 /s: 1 mOhm x 200 pF,
the g_off-loaded transformer between L_r and L_m). In the LLC's first mode
the slow coordinates were ~1.5e20 for a 400 V state (max |ProjE| 4.5e42):
18 digits cancelled at dps 30, the eigenvector condition exceeded
10^(dps/2) (expm path), and each root test's noise bound read 3e4 V. So the
rectifier stayed OFF from t = 0 with 17.7 V forward on D1 and D4 for 2.5 us
(the tool's own law broken, silently), and at 3.7 us every candidate's sign
test read noise. On a PSFB variant (L_lk 0 A at t = 0, 150 ns dead time)
the same loss hid the rectifier's commutation: L_lk ramped to 74 A in a
mode with all four rectifier diodes ON before the run raised at 5 us.
Not the cause (checked with numbers): an inductor cut-set / state jump —
x(3.7 us-) satisfies the new mode's algebraic rows to 1.9e-14 and the
conductance form has no exact cut-set (g_off closes it; an inconsistent
start, L_lk -4 A against L_o 10 A, is resolved inside the first segment by
a ~1 ps g_off transient to the flux-conserving -2.5455 / 10.1818 A, which
`engine='dsed'` reproduces to 1e-6); the enumeration (it tried all 2^6); a
threshold rounding (the bound itself was inflated 1e18-fold).
Fix: the basis is the reduced echelon rows of Pk^T (exact rationals, an
identity at the pivots: z_j is a state variable's own value). The LLC and
the PSFB solve at dps 30 (|y| 200 for the 400 V state); hand closed forms
(matrix exponentials of the interval's own state equations, written from
KCL/KVL, not the tool's MNA) agree within 6e-19 (LLC dead time + rectifier
turn-on) and 2.8e-21 (PSFB lagging-leg transition + D4 turn-on). The
closed-form validation table is unchanged (all checks <= 1.3e-37; buck DCM
improves 1e-41 -> 1e-50); every stored truth record re-solved with the old
and the new tool: TRUTH_COMPARE.
Tests: `test_exact_tool_settles_the_llc_with_a_transformer_rectifier`,
`test_exact_tool_psfb_lagging_leg_transition_closed_form` (both also check
the truth's diode law on every segment).

### F-S-4 — the exact engine drops an inductor's initial current at a rectifier start (simulator, OPEN)
The predecessor's `DiodeModeInconsistentError` on its draft PSFB at
t = 10 ns was not an inconsistent start. The PSFB started in power transfer
(S1, S4 ON, L_lk 0 A, L_o 24 A: consistent, the four rectifier diodes share
L_o) raises it (body diode D4 ON at -7.8 mA at 20 ns); the truth runs.
Minimal (11 elements): 400 V -> L_lk 10 uH (0 A) -> ideal 4:1 primary (p, b);
b to ground through 1 mOhm || a diode (gnd -> b) || 200 pF; full bridge into
L_o 20 uH (24 A) to ground. `engine='dsed'` starts L_o at -2.8e-9 A and
reaches i_Llk(200 ns) = 0.2424 A (truth 6.0605 A, pwl+restart and trbdf2
agree with it): SILENT, no warning. Without the 200 pF or the diode it
agrees; with a diode b -> 400 V added it raises (the loud form).
Strict xfail: `test_exact_engine_keeps_the_output_inductor_current_at_a_rectifier_start`.
The benchmark PSFB starts in its freewheel state (L_lk -4 A, L_o 18 A) and
runs on dsed within 1.8e-7 A of the truth.

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

* LLC (`llc_half_bridge_zvs.yaml`) on the fixed engine without restart: the
  switch and its body diode ring 14 A / 33 A step by step through the on
  time and the turn-off dumps C_oss in ONE step (truth: 66 ns); transition
  times off by up to 27 ns (2.7 steps), i_Lr at turn-off by 0.019 A. Trap 9.
  The ZVS fraction still reads 100 %.

## Passing (truth within the stated tolerance)

PSFB (`psfb_full_bridge_zvs.yaml`) against the exact tool, 12 gate-off
edges over 3 periods; means over [20, 30] us time weighted:

| engine | i_Llk at turn-off (tol 5e-3 A) | transition samples (tol 0.5 V) | mean V(out) / I(Lo) (tol 1e-3 rel) | ZVS 4 switches |
|---|---|---|---|---|
| pwl, restart off | 0.11 A | 245 V | 1.2e-3 / 2.4e-3 | S3 2 of 3 |
| pwl, restart on | 6.7e-4 A | 0.46 V | 2.8e-5 / 1.3e-4 | 100 % |
| trbdf2 | 1.7e-5 A | 0.026 V | 1.2e-6 / 4.1e-6 | 100 % |
| trbdf2 + esdirk43 | 6.0e-6 A | 0.010 V | 3.6e-7 / 2.8e-6 | 100 % |
| dsed | 1.8e-7 A | 2.0e-5 V | 1.2e-8 / 4.1e-7 | 100 % |

pwl without restart: strict xfails (trap 9: the gate read at the step's end,
C_oss dumped in about one step, the clamp ringing -0.004 / 8.5 / -0.001 V).
trbdf2 accepts 7 steps above its error estimate here (warning) and still
passes: the forced steps at the primary cut-set currents (known) cost no
accuracy at these bounds.

LLC half-bridge against the exact tool (10 turn-off edges over 5 periods):
i_Lr at turn-off within 2.5e-4 A (seen: pwl+restart 3.4e-5, trbdf2 2.2e-5,
esdirk43 1.3e-7, dsed 8.9e-10) and every engine sample inside a transition
within 0.5 V of the truth on pwl+restart, trbdf2, esdirk43, dsed; ZVS
fraction 100 % on both switches on all five configurations and in the truth.


Transition time to 300/200/100 V (charge closed form): pwl+restart, trbdf2,
esdirk43 within 0.05 ns (seen <= 4 ps); pwl within one step. ZVS fraction
(low side 1.0, high side 0.0) on all four. Resonant trajectory, trough v_min
(energy closed form, 61.8 V) and full-ZVS clamp on pwl+restart and trbdf2
within 0.5 V; esdirk43 for the full-ZVS case (F-S-1 for the partial one).
`engine='dsed'` refuses the nonlinear C_oss by name.
