# DECISIONS — add-high-order-stiff-integrator

Code: `core/include/pulsim/solver/step/esdirk43.hpp` (the integrator),
`core/tests/step/mna_stage_body.hpp` (a test body that plays the engine on
dense MNA systems), `core/tests/step/test_esdirk43.cpp` (tests; the hidden
tags `[.esdirk_measure]` and `[.esdirk_measure_long]` print every number
below). Truths: `probes/esdirk43_truth.py` (mpmath expm for the ladder,
mpmath odefun for the RC closed form, scipy Radau/DOP853 at 1e-13 for the
Shockley clamp), closed forms in the test, `probes/esdirk43_dense.py`.

## 1. Method: ESDIRK4(3)6L[2]SA as designed (D1)

Kept the design's tableau. Fourth order, stiffly accurate, L-stable, stage
order 2, embedded order 3, one real factor per step (G + C/(γh), the
trapezoidal factor of h/2). Radau IIA(5) would give stage order 3 and
order 5 but needs a 3n real system (or one real + one complex n-system)
per Newton iteration; the kernel's sparse LU is real and keyed by
(mask, dt), so that is a new solver path, not a new stage table.
- Coefficients equal SUNDIALS `ARKODE_ARK436L2SA_DIRK_6_3_4`
  (arkode_butcher_dirk.def, main branch, fetched 2026-11-18): every A, b,
  d(=b̂), c entry identical. The 2003 paper itself was not consulted.
- Exact rational verification: design probes plus the reviewer's
  independent script (`.scratch/review/tableau_exact.py`, not committed):
  all literals of the header transcribe the rationals; 8 order-4
  conditions of b are 0; b̂ orders 1–3 are 0 (order-4 residual 1.445e-3);
  A·c = c²/2 on all six rows, A·c² ≠ c³/3 (stage order exactly 2);
  R(−∞) = 0, R̂(−∞) = −3/20; R = −4(7z⁴+8z³−96z²−192z+768)/(3(z−4)⁵);
  E(y) = y⁶(9y⁴ − 64y² + 512) ≥ 0 for all real y (A-stability proved, not
  sampled).

## 2. Plugging into the skeleton: five BE stages + an explicit-stage weight

`Esdirk43` satisfies `step::Integrator` unchanged: `n_stages = 5` implicit
stages, each `{BackwardEuler, factor_dt = h/2, ctx_h = γh, t_off = c·h}`,
`weight(s, j) = a/γ` on the increments (q(Y_j) − ψ_j) (the Sdirk22
convention). The explicit first stage needs one more trait,
`explicit_weight(s) = a_{s+1,1}`, the weight of h·q̇0 with
q̇0 = −f(x0, t0): the stateful devices' trapezoidal history at x0, an
evaluation, never a solve. Everything is in q space, so algebraic
unknowns never need a derivative. No shared header was edited
(`integrators.hpp`, `controllers.hpp`, `stepper.hpp` untouched: the
existing step sequences are byte-identical by construction; the 25
pre-existing step tests pass).

## 3. Dense output: stage-value form, order 4 on q; algebraic by re-solve

x(θ) = x0 + Σ w_s(θ)(Y_s − x0) with b(θ) = Σ w_s(θ)·a_{s+1}; the five
order-4 conditions that survive stage order 2 determine w uniquely
(degree-4 polynomials, `probes/esdirk43_dense.py`, exact; the other three
order-4 conditions hold identically). w(0) = 0, w(1) = e₅: continuous.
Measured local errors (one step from the exact state, h = 0.4 … 0.025,
slope = local order; 5 = an order-4 interpolant):

| unknown | raw combination | re-solved at t0+θh |
|---|---|---|
| differential (index-1 RC network v3) | 5.1–5.2 | 5.1–5.2 |
| algebraic (capacitor-free node v2, source current iV) | 2.8–2.9 | 5.1–5.3 |
| Prothero–Robinson λ = −1 | 4.9 | — |
| stiff mode (PR λ = −1e6, reviewer) | 3.0 | n/a (it is differential) |
| x of a nonlinear q = x + x³/3 (reviewer) | 3.1 | combine q, invert: 5.0 |
| index-2 unknowns | 1.9 | n/a (not projectable) |

Reason for the 3: Σ w·c³ ≠ θ³ (stage order 2). The order-4 claim is
therefore for q on non-stiff modes; the header says so. Event location
must not rely on the interpolant alone (DECISIONS 6).

## 4. Error estimate: embedded difference, stiff filter, LteClock as is

e_q = (b₁−b̂₁)·h·q̇0 + Σ (b−b̂)_{s+1}/γ·(q(Y_s)−ψ_s); filter
(G + C/(γh))·ẽ = e_q/(γh) (one back-solve on the step's factor; the
reviewer confirmed it is (C − γh∂f)⁻¹ = (I − γhJ)⁻¹ for M x' = f); RMS
over the differential unknowns of ẽ/(atol + rtol·max(|x0|,|x1|)).
`err_exponent = −1/4`; `LteClock<Esdirk43>`/`LteController` are used
unchanged (exponent from the integrator: nothing to wire, nothing changed
for the other integrators).
Filter evidence: the ladder with its source toggling every 5 µs (each
event leaves va 10 V off its 1e-13 s manifold), rtol 1e-6: filtered 108
steps + 25 rejects; raw 453 + 102 (4.2× the attempts).

## 5. Observed orders (fixed steps, halving, global error at T = 2)

| problem | unknown | ESDIRK | TR-BDF2 |
|---|---|---|---|
| index-1 RC network | v3 (differential) | 4.00 | 2.00 |
| | v2 (capacitor-free node), iV (source current) | 4.00 | 2.00 |
| capacitor across a voltage source (index 2) | iV | 1.69→1.99 (2.89e-5 at N=320) | 2.00 (6.98e-5) |
| inductor in a current-source cutset (index 2) | v_L | 1.69→1.99 | 2.00 |
| same, v_L feeding a VCCS into an RC state | the fed state vo | 4.00 | 2.00 |
| Prothero–Robinson λ = −1 | y | 4.00 | 2.00 |
| Prothero–Robinson λ = −1e6 (stiff) | y | 2.59, 2.42, 2.27, 2.16, 2.08 → 2 | 2.35 → 2.03 |
| LC tank, 10 periods at T0/20, /40, /80 | amplitude | 5.164e-4, 3.235e-5, 2.023e-6 (3.99) | 0.247, 0.0625, 0.0157 |

Say it plainly: order 4 on ODEs and on index-1 MNA (differential AND
algebraic unknowns); order 2 on index-2 unknowns (a source current that
closes a capacitor loop, an inductor voltage in a current-source cutset)
— still 2.4× below TR-BDF2's error there — while the differential states
they feed stay order 4 (measured on one circuit); in the stiff limit
(Prothero–Robinson, |hλ| ≥ 1e4) order 2 with errors ~1e-13 (h²/|λ|).
The index-1 iV check is not independent of v2 (R1 = 1, v1 exact).

## 6. Events: bracket on the dense output, land by re-steps

Test `Lander` (an event policy on the skeleton, `localizes = true`): a
sign change over an accepted step is bracketed by bisection on the raw
dense output, then landed by Illinois on the END values of re-steps from
x0 (each a full fourth-order step; the skeleton's cut/re-solve path).
Measured (seed error → landed error, cuts):
PR λ=−1e6, y = ½ (t* = π/6): h=0.2: 4.4e-4 → 4.7e-10, 5; h=0.1:
6.9e-6 → 1.6e-11, 3. RC network v2 = 0.2: h=1/10: 3.7e-4 → 1.4e-11, 5.
Cost: 3–5 extra step solves per event (each a factor of a new h).
This is the design's D4 "landing re-step", made the locator rather than
a final correction, because the interpolant is order 3 on what events
watch (device voltages: algebraic nodes, snubber modes).

## 7. Work-precision against TR-BDF2 (same body, same filtered estimator kind)

Linear solves (stage + filter), Newton iterations, factorizations; max
error against the truth at the sample times. TR-BDF2's estimate is the
Bank et al. LTE in q space with the same kind of filter on its factor.

LC tank, 1000 periods (`[.esdirk_measure_long]`), per period:

| rtol | ESDIRK steps, solves, max err | TR-BDF2 steps, solves, max err | solves ratio |
|---|---|---|---|
| 1e-5 | 15.0, 90, 1.83 V | 154, 506, 5.37 V | 5.6 |
| 1e-6 | 27.5, 165, 0.178 V | 336, 1051, 1.16 V | 6.4 |
| 1e-8 | 84.0, 504, 1.68e-3 V | 1577, 4751, 5.3e-2 V | 9.4 |

(task 2.4 bounds hold: 27.5 ≤ 32 and 0.178 ≤ 0.25 V at 1e-6; 84 ≤ 100
and 1.68e-3 ≤ 2.5e-3 V at 1e-8; energy loss not measured.)

Stiff ladder (spread 1e9), 0–50 µs, error vs mpmath:

| rtol | ESDIRK steps/rej, solves, err | TR-BDF2 steps/rej, solves, err |
|---|---|---|
| 1e-5 | 45/11, 336, 6.1e-4 | 220/8, 684, 1.2e-3 |
| 1e-6 | 67/8, 450, 6.6e-5 | 456/8, 1392, 2.6e-4 |
| 1e-7 | 112/7, 714, 7.0e-6 | 964/9, 2919, 5.7e-5 |
| 1e-8 | 188/7, 1170, 7.4e-7 | 2066/9, 6225, 1.2e-5 |
| 1e-9 | 324/7, 1986, 7.5e-8 | 4430/9, 13317, 2.6e-6 |

Shockley clamp (Newton device), 0–2 ms, error vs scipy Radau:

| rtol | ESDIRK steps/rej, solves, Newton, err | TR-BDF2 steps/rej, solves, Newton, err |
|---|---|---|
| 1e-6 | 131/41, 2622, 2450, 3.0e-6 | 719/55, 4064, 3290, 3.0e-5 |
| 1e-7 | 229/42, 3729, 3458, 3.3e-7 | 1564/37, 8130, 6529, 6.3e-6 |
| 1e-8 | 416/40, 5923, 5467, 4.7e-8 | 3409/33, 17216, 13774, 1.3e-6 |
| 1e-9 | 754/35, 10083, 9294, 5.2e-9 | 7392/27, 37096, 29677, 2.8e-7 |

At equal error ESDIRK needs 4–10× fewer solves (factorizations ≈ steps +
rejects for both). Not measured: an event-dominated circuit (the
proposal's honest limit: 6 solves per step against 3).

## 8. Newton: one Jacobian per attempt, stage predictor, one refresh

Simplified Newton per stage on the step's factor (Jacobian at x0);
predictor = quadratic extrapolation through the last three known
(t, value) points (previous step's c = 0.62, 0.85 stages and x0, then
this step's stages); after an event only x0 (no history). A stage that
stalls (8 iterations) gets one Jacobian refresh at its last iterate, then
fails the step (h/4). Clamp at rtol 1e-7: 5 stages per attempt, 2.55
Newton iterations per stage, max 5, 0 refreshes, 0 failures; at rtol
1e-4…1e-6 refreshes appear (34, 24, 3) and a few failures (3, 1, 0).
Test bounds carry ≥ 30 % margin.

## 9. Restart

The step reads only (x0, q̇0 = −f(x0, t0)): the first step after an
event equals the first step of a fresh body and clock (test, ≤ 1e-14).
q̇0 uses every unknown of x0 (G·x0), so x0 must be consistent at t0+: the
engine's post-event solve (test: null-space projection keeping q). The
step end is consistent (stiffly accurate; algebraic residual < 1e-12 ×
100 A at the first step after each of 9 events).

## Hand-off (what the engine-switch step needs)

1. Body services: `solve_stage` for `{BackwardEuler, h/2, γh}` with an
   arbitrary ψ (the device interface's general stage (βh, ψ)); h·q̇0 from
   the trapezoidal history at x0 (the explicit stage); one extra
   back-solve for the filter; the five increments kept per step.
2. Factor key h/2 in `solve_at`; nonlinear: one Jacobian per attempt,
   refresh on stall (DECISIONS 8).
3. Events: dense output only to bracket; land by re-steps (DECISIONS 6);
   combine q, not x, when q is nonlinear, then re-solve x (DECISIONS 3).
4. Post-event: consistent solve before q̇0 (DECISIONS 9).
5. Option `integrator='esdirk43'` default off; refuse with the fixed
   clock (design applicability rules). Not done here: no engine runs on
   the skeleton yet (loop freeze).
6. Open: 2003-paper cross-check of the tableau (SUNDIALS match only);
   energy-loss bound of task 2.4; event-dominated cost measurement;
   tasks 3–5 (nonlinear devices through `f`, `q`, resonant converters).
