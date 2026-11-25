# DECISIONS — add-high-order-stiff-integrator

Code: `core/include/pulsim/solver/step/esdirk43.hpp` (the integrator),
`core/tests/step/mna_stage_body.hpp` (a test body that plays the engine on
dense MNA systems), `core/tests/step/test_esdirk43.cpp` (tests; the hidden
tags `[.esdirk_measure]`, `[.esdirk_measure_long]`, `[.esdirk_measure_nlq]`
(entries 10–12) and `[.esdirk_measure_energy]` (13–14) print every number
below). Truths: `probes/esdirk43_truth.py` (mpmath expm for the ladder,
mpmath odefun for the RC closed form, scipy Radau/DOP853 at 1e-13 for the
Shockley clamp), closed forms in the test, `probes/esdirk43_dense.py`,
`probes/esdirk43_nonlinear_q.py` (saturable inductor, Coss, buck).

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

## 8. Newton: one Jacobian per attempt, stage predictor, refresh on slow contraction

Simplified Newton per stage on the step's factor (Jacobian at x0);
predictor = quadratic extrapolation through the last three known
(t, value) points (previous step's c = 0.62, 0.85 stages and x0, then
this step's stages); after an event only x0 (no history).

Revised 2026-11-19 (entry 12): the first version refreshed once, after 8
stalled iterations, then failed the step. A saturable inductor
(λ' falls by 1e3 across the knee) defeats that at every fixed step
size: the stale Jacobian contracts at ~0.999, and the refreshed full
Newton 2-cycles on the tanh law (−11 A ↔ +261 A, seen). Now: refresh at
the last finite iterate as soon as an iteration contracts by less than
θ = 0.5 (or a pass runs out), up to 10 refreshes per stage; after a
refresh, backtrack on the residual norm; convergence is judged on the
FULL (undamped) step, and a refresh with no descent direction counts as
a stall. The refreshed factor is kept for the rest of the step.
Clamp at rtol 1e-7 unchanged: 2.55 iterations per stage, max 5, 0
refreshes, 0 failures; at rtol 1e-4…1e-6 refreshes 56, 27, 3 (were 34,
24, 3) and failures 0, 0, 0 (were 3, 1, 0); max iterations per stage
(across passes) 18 at 1e-4. Test bounds carry ≥ 30 % margin.

## 9. Restart

The step reads only (x0, q̇0 = −f(x0, t0)): the first step after an
event equals the first step of a fresh body and clock (test, ≤ 1e-14).
q̇0 uses every unknown of x0 (G·x0), so x0 must be consistent at t0+: the
engine's post-event solve (test: null-space projection keeping q). The
step end is consistent (stiffly accurate; algebraic residual < 1e-12 ×
100 A at the first step after each of 9 events).

## 10. Nonlinear charge and flux: stages in q, step end on the q quadrature

Body: `q(x) = C·x + q_n(x)`, Jacobian `G + ∂g + dq/dx·(2/factor_dt)`;
stage residual `(q(Y) − ψ)/(γh) + f(Y, t)`; x recovered by the stage
Newton with the device's own dq/dx. Truths:
`probes/esdirk43_nonlinear_q.py` (scipy Radau and LSODA at rtol 1e-13,
spread printed; mpmath odefun 25 digits agrees with Radau within 4e-14
on A, k = 10, at 5 and 10 ms).

Circuits: saturable inductor `λ = Ls·i + (L0 − Ls)·Is·tanh(i/Is)`,
`Ls = L0/k` (smooth, monotone, no cancellation; λ' = Ls + (L0 − Ls)/cosh²);
junction Coss `q = 2·C0·v/(√(1 + v/φ) + 1)`.
A `sat_series`: 10 sin(2π50 t) − 1 Ω − L (10 mH, 1 A), k = 10 and 1000
(iL to 10·Is; λ' to L0/k). Algebraic vs, vm, iV are LINEAR in iL.
B `nl_tank`: 10 V − 0.1 Ω − node m − L (100 µH, 1 A, k 100; iL to 16 A)
− node v − Coss (10 µF, 0.7 V) ∥ 100 Ω, plus a limiter 2 A·tanh(vm/5 V)
from m to ground: the algebraic vm is a NONLINEAR function of iL.
`sat_ideal`: the inductor straight across the sine (R = 0), closed form
λ = (V/ω)(1 − cos ωt), iL = λ⁻¹ (to ~5000 A).

Observed global orders (fixed steps, rtol = atol = 1e-10, Newton 0.01:
1e-12 A absolute; the default 1e-9·1e-6 floor is BELOW the conditioning
of i(λ) in saturation, ulp(λ)/Ls ≈ 1.7e-13 A, and stalls Newton):

| circuit | n (steps) | errors | orders |
|---|---|---|---|
| A k=10, iL and vm at 20 ms | 80…1280 | 7.8e-6 … 1.2e-10 | 4.05, 4.01, 4.01, 4.00, 4.00 |
| A k=1000 | 80…1280 | 1.7e-5 … 3.0e-10 | 3.90, 3.95, 3.97, 3.99 |
| A k=1000, carry_q | 160…2560 | identical to 3 digits | same |
| B iL / v / vm at 200 µs | 3200→6400 | 3.9e-9→2.3e-10 (iL) | 4.11 / 4.08 / 4.11 |

B is pre-asymptotic below 3200 steps (slopes 2.5–7.3) and the 12800
point (8.7e-12) is below the reference spread (3.4e-11): ONE clean ratio.
The last A points (n = 2560, 7e-12 / 2e-11) sit at the reference spread
(3.5e-12) and the Newton floor: not used. The fixed clock fails Newton on
B at n = 100 (no retry ladder; the adaptive clock recovers by h/4).

Dense output, local error at θ = 0.5 of one step from the exact state on
the knee (t0 = 2.6 ms), h = 4e-4 … 2.5e-5:

| | combined in q, then x by from_q | combined in x |
|---|---|---|
| sat_ideal k=10 | slopes 5.44, 5.27, 5.08, (5.46) | 4.39, 1.48, 2.88, 3.01 |
| sat_ideal k=1000 | 5.04, 5.02, 5.01, 5.00 | 0.083 A at every h (no convergence: Δi per step 15–300 A) |
| sat_series k=10 (state-coupled) | 5.18, 5.02, 5.16, 5.10 | 2.40, 3.20, 3.14, 3.08 |

## 11. Flux balance: what is structural, what leaks, and the x form

Reviewed (fresh context, 2026-11-19); the reviewer's corrections are
folded in. Per step, with x1 = Y6 (stiffly accurate):

- `q(x1) − q(x0)` equals the b-weighted stage increments to 5e-16
  relative. TRUE BY CONSTRUCTION (the increments are q(Y) − ψ and ψ is
  built from them): a wiring check only.
- Against the quadrature `h·Σ b_i·q̇(Y_i)` (q̇ = −f at the stage values)
  it is off by the stages' Newton residuals: 5.9e-10 of V/ω per step at
  Newton tolerance 0.03, 1.8e-11 at 1e-3; they do NOT cancel: λ drifts
  1.2e-6 (V/ω units) after 1000 periods of `sat_ideal` (40 steps per
  period), 1.0e-8 at 1e-3.
- `carry_q`: end the step on `q1 = q0 + h·Σ b_i·q̇(Y_i)` and recover x1
  by `from_q` (1.2 Newton iterations per step, damped): λ back to 0
  within 3.7e-13 after 1000 periods. On `sat_ideal` the source does not
  depend on the state, so ANY q-form update with an exact quadrature
  returns (the reviewer's forward Euler in q: 2e-12): this checks the
  recovery, not accuracy. On the state-coupled `sat_series`, `carry_q`
  has the q form's errors to 3 digits and keeps order 4 (tests).
- The same tableau on x (`i' = v/λ'(i)`), and the trapezoidal rule on x:
  on `sat_ideal` below ~2560 steps per period the stage equation near the
  knee has several roots (or none) and the solution jumps branch: λ off
  by 0.55–1.27 of V/ω after one period (trapezoid at 160 per period:
  −0.59 with every stage solve converged; ESDIRK at 40/160/640: 1/1/2
  stage solves without a root). From 2560 per period both converge
  (ESDIRK 2.2e-5, trapezoid 1.5e-13). With damping (`sat_series`) the x
  form is NOT worse: same order, error 0.79× (k = 1000) to 2.5× (k = 10)
  the q form's. The q form matters where nothing restores the flux: an
  inductor (magnetizing branch) driven by a voltage source.

## 12. Newton through the knee, events landing in saturation

Adaptive, against TR-BDF2 (solves = stage + filter; factors include
refreshes). `sat_series` k = 1000, 0–20 ms, max |ΔiL| at 5/10/15/20 ms:

| rtol | ESDIRK steps/rej, solves, it/stage, factors/attempt, err | TR-BDF2 steps/rej, solves, err |
|---|---|---|
| 1e-5 | 104/50, 4252, 5.32, 2.10, 8.2e-6 | 333/83, 3332, 5.1e-5 |
| 1e-6 | 153/56, 5443, 5.01, 1.41, 7.0e-6 | 709/32, 4943, 1.2e-5 |
| 1e-7 | 274/68, 8235, 4.62, 1.14, 3.1e-7 | 1568/31, 10148, 2.6e-6 |
| 1e-8 | 502/77, 12654, 4.17, 1.01, 1.0e-7 | 3452/26, 20783, 5.8e-7 |

`nl_tank` (max of |ΔiL|, |Δv|/10): ESDIRK 1e-7 4.7e-6 at 9222 solves
against TR-BDF2 1e-8 9.3e-6 at 30850; at 1e-6 9.4e-5 / 6179 against
1.4e-4 / 7711. Square-wave drive (±10 V every 5 ms, each edge lands at
iL = ±10 A, λ' = L0/1000), edges handled as events (land, flip,
consistent solve, restart):

| rtol | ESDIRK solves, it/stage, refreshes, fails, err; 4 edges: attempts, it/stage | TR-BDF2 solves, err |
|---|---|---|
| 1e-5 | 8617, 5.89, 578, 0, 6.6e-5; 26, 9.0 | 5946, 3.5e-5 |
| 1e-6 | 9054, 5.64, 333, 1, 1.2e-6; 18, 8.7 | 8174, 9.2e-6 |
| 1e-7 | 13421, 5.31, 232, 0, 2.3e-7; 15, 8.3 | 15203, 2.2e-6 |
| 1e-8 | 19097, 4.82, 77, 0, 2.6e-8; 13, 8.3 | 29198, 4.9e-7 |

Every run converges (one stage failure at 1e-6, retried at h/4). The
price of the long steps is Newton: 4–6 iterations per stage and up to
2.1 factors per attempt at loose tolerance; TR-BDF2 stays at 2.5–3.5 and
1.0. At equal error ESDIRK wins 2.5–3.3× at rtol ≤ 1e-7, 1.0–1.25× at
1e-5…1e-6, and LOSES ~1.5× on the square wave at 1e-5.

## 13. Energy on the lossless LC (task 2.4)

Fixed steps at equal solves per period (no filter solve): loss per period
`1 − ρ^(2n)`, ρ the spectral radius of the one-step propagator in energy
coordinates; a 1000-period simulation reproduces it to 4 digits.

| solves/period | ESDIRK (steps) | TR-BDF2 (steps) | trapezoidal (steps) |
|---|---|---|---|
| 50 | 2.82e-5 (10) | 7.26e-4 (25) | 0 (50) |
| 100 | 9.99e-7 (20) | 9.15e-5 (50) | 0 (100) |
| 200 | 3.23e-8 (40) | 1.15e-5 (100) | 0 (200) |
| 400 | 1.02e-9 (80) | 1.43e-6 (200) | 0 (400) |

(trapezoidal: |loss| ≤ 4e-14, round-off; it conserves energy and rings
on stiff modes, 5b.2). Adaptive, 1000 periods, total energy loss:
ESDIRK 7.1e-2 / 4.7e-3 / 2.69e-4 / 1.48e-5 / 8.1e-7 at rtol 1e-4…1e-8
(53 / 90 / 165 / 282 / 504 solves per period); TR-BDF2 4.9e-2 / 4.9e-3 /
4.8e-4 / 4.7e-5 / 4.7e-6 (257 / 506 / 1051 / 2221 / 4751). Task 2.4:
2.69e-4 = 0.027 % ≤ 0.035 % at rtol 1e-6. At rtol 1e-4 ESDIRK loses MORE
energy than TR-BDF2 (7.1 % vs 4.9 %) on 5× fewer solves.

## 14. Event-dominated: buck at 100 kHz

Synchronous buck, ideal switch node (a PWL source 12 V / 0), D = 0.4,
L 10 µH + 10 mΩ, C 10 µF ∥ 1 Ω, from rest, 20 periods (40 edges, each
an event: land, flip, consistent solve, restart). Error: max of |ΔiL|,
|Δvo| at the end of periods 5/10/15/20 against mpmath expm.

| rtol | ESDIRK steps/period, solves, err | TR-BDF2 steps/period, solves, err |
|---|---|---|
| 1e-1 | 2.2, 264, 4.4e-4 | 2.2, 135, 5.0e-2 |
| 1e-2 | 2.2, 264, 4.4e-4 | 4.0, 237, 5.5e-2 |
| 1e-3 | 2.2, 264, 4.4e-4 | 5.5, 375, 6.4e-3 |
| 1e-5 | 3.1, 378, 5.5e-5 | 21.6, 1356, 3.2e-4 |
| 1e-6 | 5.2, 624, 9.7e-6 | 44.3, 2715, 7.3e-5 |
| 1e-7 | 9.2, 1104, 1.0e-6 | 95.4, 5787, 1.5e-5 |
| 1e-8 | 14.5, 1746, 8.5e-8 | 203.9, 12300, 3.3e-6 |
| 1e-9 | 24.8, 2988, 8.8e-9 | 437.0, 26280, 7.1e-7 |

Solves per accepted step: ESDIRK 6.00–6.06, TR-BDF2 3.00–3.44. At the
event floor (2.2 steps per period: one per interval plus the start)
ESDIRK costs 1.96× the solves — for 114× less error. At equal error it
needs 2.5–15× fewer. The case where it loses is a user tolerance looser
than what one ESDIRK step per interval delivers (here 4.4e-4).
Not measured: state events (DCM, diode commutation) located by re-steps,
6 solves per re-step against 3.

## 15. On the engine: `integrator='esdirk43'` on `engine='trbdf2'` (2026-11-19)

**Selection.** `TrBdf2Options::integrator` (0 TR-BDF2, 1 ESDIRK4(3)) from
`simulate(integrator='esdirk43')` → `sim::RunOptions::integrator_esdirk43`
→ `to_trbdf2_options`; the `integrator` row of `kRunOptionTable` is now
honoured on trbdf2 and the exact engine (values: simulate() refuses the
exact engine's names on trbdf2 and 'esdirk43' on pwl / exact, by name,
`feature='integrator'`). The engine's `tb_take`, `tb_error`, `tb_after`,
`tb_commit` are template lambdas `<bool kE = false>`; `run_twin` picks the
instantiation ONCE per run (`step::run<Esdirk43>` with four adapters, or
the unchanged TrBdf2 body). Everything else — poll, begin, clamp, the
crossing detection, Illinois, the landing path, the restart after events,
mode selection, snapshot, ticks, hooks, the chatter guard, the controller
object and its accounting — is the same code. The integrator-specific
parts, all under `if constexpr (kE)`:
- take: `ed_stages` — five backward-Euler stages of γh on the factor of
  h/2 (the engine's `(mask, dt)` cache key: one factor per step), each
  through the linear companion histories with the entries' state slots set
  to ψ_s and derivative slots 0 (`compute_b_extra` at h/2 = BE of γh,
  the mutual term included); h·ẏ0 from the committed derivative slots
  (capacitor current, coupling-aware inductor di/dt: an evaluation);
  x_g = stage 1 (c = 1/2, `h_pt = h/2`) for the crossing detector,
  x_1 = stage 5 (stiffly accurate).
- error: the embedded difference h·Σ(b − b̂)·ẏ per entry, ALWAYS filtered
  on the stage factor (`filter_estimate(mask, h/2)` = (G + C/(γh))⁻¹·C/(γh));
  exponent −1/4 in reject and accept; the round-off floor (TR-BDF2's
  constants) is skipped.
- after_solve: Illinois trials and the landing are ESDIRK re-steps of the
  bracket (a trapezoidal trial over an ESDIRK-sized bracket put the
  rectifier's events 1.4e-9 s off; re-steps: 2.1e-10).
- commit: the last stage's base restored, `update_from_state_backward_euler
  (x_1, γh)`: slots = (Y5 − ψ5)/(γh) and the inductor terminal voltage,
  KCL-consistent at x_1. The landing commits the same way.

**Restart (hand-off item v).** The explicit stage reads the derivative
slots, and the embedded estimate is nearly blind to them (b0 − b̂0 = 0.0032
against b0 = 0.158). A diode landing commits under the pre-commutation
mask, so after it the slots belong to another mode: measured, the diode
rectifier ran 298 diode events instead of 199 and 6× the error. So a step
whose mask differs from the slots' mask (`x_mask`), and every restart /
re-initialisation step, is the shared SDIRK(2,2) step (BE stages, no slot
read); its commit leaves slots of the new mode. Its first-order estimate is
filtered when kE forced it (fresh reviewer: raw, it reads the damped jump
as error at any h; rectifier error at rtol 1e-6 2.7e-7 → 8.2e-8). A gate
edge with the committed refresh probe already leaves slots of the new mode
(`x_mask = m_probe`): ESDIRK continues directly.

**Devices.** Run: R, C, L, coupled windings / two-winding transformer,
sources, ideal and PWL switches, PWL diodes (events), static Newton
devices (Shockley diode, the Level-1 MOSFET law without Coss: the engine's
own Newton per stage solve). Refused by name (`UnsupportedError('trbdf2',
'integrator=esdirk43', device, kind)`): every `StatefulKinds` history —
NonlinearCapacitor (Coss), LauritzenDiode, PmsmMna (machines), IgbtTail,
SaturableInductor, ExternalDevice — whose stage takes no arbitrary ψ yet;
also the legacy loop. Not done: carry_q, stages in q, the Newton refresh
rule of entry 8 (one Jacobian per attempt).

**Orders through the engine** (fixed ceiling, rtol 1, LC 100 µH / 100 nF
from a landed 10 V step, error after 10 periods): 5.165e-3, 3.240e-4,
2.050e-5, 1.400e-6 V at T0/20, /40, /80, /160 — ratios 15.9, 15.8, 14.6
(order 4; the test body's 5.16e-4, 3.24e-5, 2.02e-6 relative to 10 V).
TR-BDF2 ratios 3.96, 3.99, 4.00. Index-2 and the stiff limit: not
re-measured on the engine (entry 5 stands). A step from `initial_state`
with derivative slots 0 is an inconsistent start for BOTH integrators
(first-order global error, measured): start from a DC point and a landed
source step.

**Work-precision against truths** (atol = 1e-3 rtol; solves = linear
solves incl. rejects; harness dt is the ceiling in the scoreboard rows).
LC tank, 1000 periods (solves / period, energy drift, max error):
rtol 1e-5 TR-BDF2 283, −7.4e-3, 7.0 V | ESDIRK 113, −4.5e-3, 1.7 V;
1e-6 609, −7.3e-4, 1.5 V | 203, −2.7e-4, 0.17 V; 1e-8 2825, −7.2e-6,
6.9e-2 V | 573, −8.2e-7, 1.6e-3 V. Rejects: ESDIRK 4.5–8 per period (the
shared elementary controller on an oscillator; the test body's 72 solves
per period at 1e-5 had none).
Stiff ladder (spread 1e9, mpmath expm, ceiling = span): 1e-4 8.3e-4 / 195
vs 8.5e-4 / 71; 1e-6 4.3e-5 / 721 vs 1.2e-5 / 231; 1e-8 2.0e-6 / 3141 vs
1.5e-7 / 756. Shockley clamp (scipy Radau 1e-13, 2 ms): 1e-4 1.4e-4 / 450
vs 3.7e-5 / 350; 1e-6 6.0e-6 / 1602 vs 6.1e-7 / 925; 1e-8 2.7e-7 / 6962
vs 9.3e-9 / 2380.
Exact scoreboard measure (`exact`, `.scratch/ed/wp_score.py`), TR-BDF2 |
ESDIRK, error / solves:

| circuit | rtol 1e-4 | rtol 1e-6 | rtol 1e-8 |
|---|---|---|---|
| rlc_series_resonant | 1.6e-3 / 4342 \| 2.1e-6 / 5015 | 7.3e-5 / 17610 \| 2.1e-6 / 5040 | 3.3e-6 / 80000 \| 9.7e-8 / 11875 |
| high_q_rlc_ringdown | 2.7e-2 / 6515 \| 3.8e-4 / 5026 | 1.2e-3 / 27951 \| 1.3e-4 / 7171 | 5.6e-5 / 128061 \| 1.3e-6 / 21696 |
| buck_switching | 4.9e-7 / 2460 \| 1.4e-7 / 5674 | 4.7e-7 / 2855 \| 7.3e-9 / 5946 | 2.9e-7 / 5725 \| 3.0e-9 / 6577 |
| buck_dcm | 3.0e-6 / 4104 \| 1.0e-7 / 6810 | 2.5e-6 / 6494 \| 1.2e-8 / 10909 | 2.7e-7 / 21456 \| 3.8e-9 / 13853 |
| boost_classic | 7.8e-6 / 5423 \| 1.6e-8 / 9804 | 6.4e-6 / 8707 \| 1.5e-8 / 13863 | 4.4e-7 / 26459 \| 1.5e-8 / 17307 |
| diode_rectifier | 1.5e-6 / 8204 \| 6.7e-6 / 20734 | 4.9e-8 / 10877 \| 8.2e-8 / 24124 | 2.5e-9 / 21121 \| 4.8e-9 / 29492 |
| half_wave_rectifier_sine | 4.8e-7 / 2219 \| 8.3e-8 / 5252 | 1.5e-8 / 2505 \| 1.3e-8 / 5375 | 8.1e-10 / 4302 \| 1.9e-9 / 5876 |

(the switching rows: buck_switching, buck_dcm, boost_classic, rectifiers
before the reviewer's filter fix for buck_switching / half_wave; the three
re-run rows after it.) As D5b predicts: resonant — ESDIRK wins outright
(high-Q ringdown at the default-like 1e-4: 2.7e-2 → 3.8e-4 with fewer
solves; at equal error 6–13× fewer solves). Hard-switched buck / boost:
ESDIRK at rtol 1e-4 is more accurate than TR-BDF2 at 1e-8 with 2.7–3×
fewer solves; at equal rtol it costs 1.2–2.3× the solves. Rectifiers
(event-dominated, diode commutations every few steps): ESDIRK LOSES —
equal error costs ~2× the solves (each commutation is followed by a forced
SDIRK step with a filtered estimate, and trials / landings cost 5 solves).

**Events** (truth event times, nearest landed sample): diode_rectifier
rtol 1e-6 max 1.2e-11 s (TR-BDF2) vs 2.1e-10 s (ESDIRK), 1e-8 4.6e-13 vs
7.6e-12; buck_dcm 1e-6 7.3e-10 vs 2.8e-13, 1e-8 5.9e-11 vs 1.8e-13. Event
counts equal the truth's (199; 34 diode + 80 gate). Landing re-steps per
event: not counted (no stat key added: the stats keys are compared by the
identity gates).

**Fuzz** (every value × (1 + 1e-13·u), 30 draws each): DCM buck 0 bad
(1287 steps every draw), floating-bus diode bridge 0 bad (1041–1049).

**Option off.** Byte-identical to 39450607 on the four compares (goldens,
stateful, engines, step sequences); instructions per accepted step (min of
3, two runs each, base | this): buck_trbdf2 10962 / 10965 | 10992 / 10989
(+0.25 %), shockbuck_trbdf2 72526 / 72509 | 72553 / 72569 (+0.06 %),
bridge 12735 / 12705 | 12748 / 12756 (+0.2 %).

## Hand-off (what the engine-switch step needs)

1. Body services: `solve_stage` for `{BackwardEuler, h/2, γh}` with an
   arbitrary ψ (the device interface's general stage (βh, ψ)); h·q̇0 from
   the trapezoidal history at x0 (the explicit stage); one extra
   back-solve for the filter; the five increments kept per step.
2. Factor key h/2 in `solve_at`; nonlinear: one Jacobian per attempt,
   refresh on stall (DECISIONS 8).
3. Events: dense output only to bracket; land by re-steps (DECISIONS 6);
   combine q, not x, when q is nonlinear, then re-solve x (DECISIONS 3,
   10).
4. Post-event: consistent solve before q̇0 (DECISIONS 9).
5. Option `integrator='esdirk43'` default off; refuse with the fixed
   clock (design applicability rules). Not done here: no engine runs on
   the skeleton yet (loop freeze).
6. Nonlinear q (DECISIONS 10–12): stage unknowns in q; END THE STEP ON
   THE q QUADRATURE and recover x (`carry_q`): with x1 = Y6 the flux
   balance leaks the Newton residuals (1.2e-6 of V/ω per 1000 periods).
   Newton: refresh on slow contraction, backtracking after a refresh,
   convergence on the undamped step; the absolute Newton tolerance must
   stay above ulp(q)/q'(x) (1.7e-13 A for this inductor in saturation).
7. Selection guidance: design.md D5b.
8. Open: 2003-paper cross-check of the tableau (SUNDIALS match only);
   state-event-dominated cost (DCM, diode commutations); tasks 3–5 on
   the engine (devices through `f`, `q`; resonant converters).

Hand-off after entry 15 (the option runs on the engine for linear
companions, PWL switches / diodes and static Newton devices):
- DONE on the engine: items 1 (linear companions; one factor per step),
  2 (dt-keyed cache; Newton per stage is the engine's own), 3 (bracket on
  stage values, land by ESDIRK re-steps), 4 (the SDIRK step after a mode
  change or restart), 5 (the option, refusals).
- NEXT: (a) stateful Newton histories (Coss, saturable inductor, machines,
  Lauritzen, IGBT tail, external) need the device interface's general
  stage (βh, ψ) + carry_q (item 6) — refused by name today; (b) Newton:
  one Jacobian per attempt with the refresh rule of entry 8; (c) the
  round-off floor for the ESDIRK estimate (skipped: a SEPIC-like node held
  by g_off may grind to h_lo); (d) the reject rate on oscillators (shared
  elementary controller; a PI controller is a number change for TR-BDF2
  too); (e) task 4.3 / 5.x gates; (f) dense-output bracketing (stage values
  only today).

## 16. Stateful kinds, readings and losses on esdirk43 (2026-11-19)

**Capacitor committed current (KCL).** The in-loop `i()` / `p()` reading
of a capacitor missed KCL by 7e-5 (buck) and 7e-6 (rectifier) of the node
scale on esdirk43. Every bad sample sat at an interpolated floor landing
(a turn-off at the bracket floor, the state interpolated between two ESDIRK
states, `interp_land`). The trapezoidal reconstruction
`2C/dt_land·(v − v_n) − i_n` over that sliver does not belong to ESDIRK
states. The landing now commits the same interpolation of the two bracket
ends' own stage currents `C·d_s/(γh)` (the committed slot at the left end,
the last trial's stage 5 or the step's stage 1/5 at the right). KCL is
linear under one mask, so the interpolation is exact up to the state's own
interpolation. Measured (rtol 1e-7, atol 1e-10, residual / node scale):
buck 3.9e-13 (in), 7.7e-13 (sw), 1.3e-8 (out; TR-BDF2 1.8e-7, the
round-off term C·eps·|v|/h dominates); rectifier 5.9e-12, 1.5e-10,
1.7e-10. No sample over 1e-9·scale + 8·C·eps·max|v|/h on either. The
recorder's esdirk43 capacitor refusal is gone.
Against `engine='dsed'` (end values, `.scratch/s4/acc.py`): esdirk43 is
within 6e-10 – 2.5e-8 of the exact engine from rtol 1e-5 to 1e-9 on the
DCM buck and the rectifier, flat in rtol — the exact engine's own floor
at its defaults; the order after events is not resolvable against it
(TR-BDF2 at rtol 1e-9: 2.2e-8 / 6.9e-8). The state trajectory did not
move with the fix (same steps, same errors to 3 digits): only the
committed slot after an interpolated landing changed.

**Loss tracker.** `accept_step<N>` integrates the conduction power on the
ESDIRK weights b over (x0, Y1..Y4, x1); the N = 1 overload keeps the
TR-BDF2 expressions verbatim (byte compares identical). Heat is advanced
once per accepted step (`advance(h)` in `ed_take`), events are booked at
the TR-BDF2 sites. An ESDIRK landing integrates on its own stages; an
interpolated or trapezoidal landing keeps the trapezoidal weights.
Agreement with TR-BDF2 at rtol 1e-8 / atol 1e-11 (rtol 1e-9 grinds
TR-BDF2 itself on the buck's capacitance-free switch node): equal n_on /
n_off, worst 5.1e-7 (buck) and 2.8e-6 (VSI) of the device's largest
energy; at rtol 1e-7 1.3e-5. Closed-form ramp (`test_device_losses_variable_step`)
and the ledger balance pass on esdirk43.

**Stateful kinds: the stage form (βh, ψ).** The nonlinear capacitor
(Coss and the explicit C_gd / junction laws), the saturable inductor (atan
and table laws) and external devices take the five stages. Each exposes
its q rows to `StatefulHistories` (`stage_rows / stage_start / stage_base
/ stage_increment / stage_commit / stage_clear`): q = charge, flux, or the
external model's q per row; q̇ = the committed current, the terminal
voltage, or the formula derivative (−f on internal rows). The engine
builds ψ_s − q0 from the increments q(Y_j) − ψ_j and h·q̇0 with the same
tableau weights as the linear entries (no cancellation of q0), sets the
base on the histories, solves the stage with the backward-Euler stamp of
γh (the stamps' own BE rule: `q_prev = ψ` for the capacitor and the
external cells; `lambda_old = ψ` and `lambda_offset = ψ − λ(i_L_old)` for
the saturable inductor, whose increment form subtracts the offset), and
reads the increment back. `ed_stages` snapshots `hs` first and restores
it after the last stage and on every throw: no stage, failed Newton,
rejected step or Illinois trial leaves a trace. The commit (accepted
step, or ESDIRK landing re-step) re-applies the last stage's base and
commits backward Euler of γh (`stage_commit`; an external model's
`accept()` advances the whole step h). The forced SDIRK(2,2) step on kE
takes BE of gs·h for the q rows too: stage 1 from the committed state,
stage 2 from ψ = q0 + ((1 − gs)/gs)·(q(x_g) − q0) set through the same
stage form (`ed_sdirk_base`; the kinds' own `extrapolate_previous` moves
λ_old but not the increment form's i_L_old), restored after the solve and
re-applied at the commit. Off esdirk43 every new path is either under
`if constexpr (kE)` or behind `kE && ed_sf`; the saturable offset is 0.0
outside a stage (x − 0.0 is exact), the external `h_accept` defaults to
the formula's h: byte compares identical.
Kept refused by name (`integrator=esdirk43`): Lauritzen diode, IGBT tail,
machines (PmsmMna), and the Jiles-Atherton saturable law (its base is a
magnetisation state, not a q). The interpolated floor landing never meets
a stateful kind (a stateful kind makes the run nonlinear, and the
interpolated landing is linear-only).
Not done: `carry_q` (the step ends on x1 = Y5 and q1 = Q(x1); in exact
arithmetic the same as the q quadrature, it leaks the Newton residual per
step — measured harmless here: a Newton tolerance of 1e-12 instead of
1e-9 changes the saturable LC's error in the fourth digit); the Newton
refresh rule of entry 8 (the engine's own Newton per stage solve); the
stateful rows in the error norm (the estimate reads the linear entries
only, as TR-BDF2's does).

Measured (tests in `test_esdirk43_engine.py`):
- Saturable LC (100 µH → 20 µH atan law, I_sat 0.1 A, 100 nF, 10 V landed
  step) against scipy DOP853 at rtol 1e-13: fixed ceiling, end error after
  4 periods 2.03e-3, 8.06e-5, 4.72e-6, 2.94e-7 at T0/20, /40, /80, /160 —
  ratios 25.2, 17.1, 16.1 (observed order 4.0–4.6). Adaptive, rtol 1e-5 …
  1e-10: esdirk43 1005 … 2943 steps, error 8.4e-8 … 2.4e-8; TR-BDF2 1358
  … 50922 steps, 1.2e-3 … 1.1e-6. The esdirk43 floor (2e-8 … 8e-8, flat
  in rtol and in the Newton tolerance) sits there from 0.01·T0 on and the
  linear LC shows it too (2.4e-8): the landed source step's 7e-14 s
  sliver, not the stages. Order by halving the tolerance is therefore not
  measurable on this case; the fixed-ceiling halving is the measure.
  Rejects: 0–32 per 4 periods at rtol 1e-5 … 1e-8, 106 at 1e-10 (TR-BDF2
  24–104).
- Junction-law capacitor charged through 1 kOhm from a landed 10 V step,
  against scipy DOP853: fixed ceiling, max error 6.07e-6, 3.91e-7,
  2.81e-8, 2.26e-9 at 3 µs/20 … /160 (ratios 15.5, 13.9, 12.4; the last
  nears the source-step floor of 1e-9); at its own steps 2.1e-9 (TR-BDF2
  8.6e-7). (A constant-current charge, q = I·t, is exact for any
  consistent method and was dropped as a truth on the reviewer's note.)
- Saturable table law (the atan curve as 41 knots, C¹ monotone cubic):
  2.5e-6 of 0.5 A from TR-BDF2 under a T0/4000 ceiling, whose own error
  is ~1e-6.
- MOSFET L1 + Coss + saturable + Shockley boost, and IGBT + VDMOS C_gd +
  saturable chopper, against TR-BDF2 at rtol 1e-9 under a 2.5 ns ceiling:
  esdirk43 (rtol 1e-9, own steps) 6.3e-6 / 7.2e-6 of each state's scale;
  TR-BDF2 at its own steps 8.3e-5 / 3.5e-5. Under a 2.5 ns ceiling
  esdirk43 is 1.6e-6 / 3.4e-7 from it (the reference's own error).
- DCM buck with a junction Coss and a saturable inductor (diode events →
  19 forced SDIRK steps with the q rows): v(out) 3.1e-6, i(L) 3.2e-6
  against TR-BDF2 under a 2 ns ceiling (whose own error is ~4e-6); equal
  event counts.

**External devices.** The twins against the built-ins on esdirk43
(`test_external_device.py`): saturable-inductor twin 1e-15, Shockley twin
0 (same steps); built-in against TR-BDF2 at rtol 1e-10 under a 100 ns
ceiling 1.3e-8 / 5.3e-9, the reference's own error (50 ns: 3.3e-9 /
1.7e-9). The time-dependent law reads each stage's own time (the ramp
test passes on esdirk43 to 1e-9).

**Slice 3's open list.**
- Stale `ed_hist` / `ed_spsi5` at a landing: impossible by construction.
  Every path that runs a trial (`ed_stages` of dts) ends in a landing;
  an ESDIRK landing re-runs `ed_stages(dt_land)` immediately before its
  commit, the interpolated landing does not read `ed_hist`, a failed
  landing rejects (the next take re-runs the stages), and the chatter
  guard's fall-through accept runs no trial. The KCL test on the
  rectifier (Illinois trials at every commutation) is the test that would
  fail: a trial's base committed with dt_land puts (Y5 − ψ5(dts))/(γ·
  dt_land) in the slot, off KCL by O(1). Not mutation-tested.
- The estimate's round-off floor: not done (no grind seen on the cases
  here; the SEPIC-like g_off node is still the open case).
- The oscillator reject rate: measured above and in entry 15 (4.5–8 per
  period on the LC); the shared elementary controller is unchanged.
- Dense output bracketing an event: not done (stage values only).

**Cost.** Option off (default integrator), instructions per accepted step,
min of 3 interleaved runs, pre-change build | this tip: buck_pwl 3090.9 |
3094.6 (+0.12 %), vsi_pwl 4765.1 | 4772.4 (+0.15 %), buck_trbdf2 11017.4 |
11017.6 (+0.00 %), bridge_trbdf2 12775.5 | 12780.8 (+0.04 %).
Resonant cases (scoreboard `exact` measure, `.scratch/s4/res_cost.py`;
instructions per accepted step include the harness call, min of 3):

| case, rtol | TR-BDF2 err / steps / instr per step | esdirk43 err / steps / instr per step |
|---|---|---|
| lcc_resonant_inverter 1e-4 | 2.6e-4 / 2894 / 15.1k | 3.2e-6 / 1009 / 20.9k |
| lcc_resonant_inverter 1e-6 | 1.2e-5 / 13751 / 12.8k | 1.4e-6 / 1372 / 27.2k |
| lcc_resonant_inverter 1e-8 | 5.4e-7 / 64440 / 12.4k | 1.8e-8 / 4305 / 23.8k |
| rlc_series_resonant 1e-4 | 1.6e-3 / 1773 / 9.8k | 2.1e-6 / 1003 / 12.6k |
| rlc_series_resonant 1e-6 | 7.3e-5 / 8476 / 7.8k | 2.1e-6 / 1008 / 12.7k |
| rlc_series_resonant 1e-8 | 3.3e-6 / 39821 / 7.4k | 9.7e-8 / 2248 / 15.1k |

At equal or better accuracy on the LCC inverter: esdirk43 at 1e-4
(3.2e-6, 1009 steps, 2.1e7 instructions) against TR-BDF2 at 1e-6
(1.2e-5, 13751 steps, 1.8e8): 13.6× fewer steps, 8.4× fewer
instructions; esdirk43 at 1e-8 (1.8e-8, 4305, 1.0e8) against TR-BDF2 at
1e-8 (5.4e-7, 64440, 8.0e8): 15× fewer steps, 7.8× fewer instructions.
One esdirk43 step costs 1.4–2.1× a TR-BDF2 step.
The esdirk43 scoreboard configuration (`.scratch/ed/wp_score.py`, the
seven circuits of entry 15) before (pre-change build) and after: every
error identical to its printed digits; solves identical except
diode_rectifier (20734 → 20736, 24124 → 24146, 29492 → 29482 at rtol
1e-4, 1e-6, 1e-8: the interpolated landing's committed slot). (Entry 15's
buck_switching and half_wave rows predate the filter fix; the pre-change
build gives today's values for them.)

**Review** (fresh-context, read-only, on f220a1da..HEAD). No blocking
defect; B (state left behind, stale bases) clean on every traced path;
C byte-identical by construction. Acted on: the constant-current
capacitor truth (replaced by an ODE truth with an order check), the
untested table law (test added), the interpolated-landing fix's coverage
(the per-sample KCL test failed there before the fix: noted in the
test). Left: the event-row KCL tolerance (`floor / 1e-15`, pre-existing)
is loose; the forced-SDIRK stage-2 base is checked only through end
values at 2e-5 (measured 3.1e-6); a stage-2 or non-singular stage-1
throw in the forced SDIRK step skips the `hs` restore, harmless because
it ends the run; an external model whose eval reads ctx.h sees γh in the
stages and h in accept() (by design, untested).

Hand-off after entry 16:
- DONE: capacitor readings KCL-consistent on esdirk43 (refusal gone);
  the loss tracker on esdirk43 (refusal gone); the stage form for the
  nonlinear capacitor, the saturable inductor (atan, table) and external
  devices, with the forced SDIRK step (refusals gone).
- NEXT: (a) Lauritzen diode, IGBT tail, machines, the JA law: their
  Newton histories need a q-row form (refused by name); (b) carry_q;
  (c) the Newton refresh rule of entry 8; (d) the estimate's round-off
  floor; (e) dense-output bracketing; (f) task 4.3 and 5.x gates (the
  LLC with nonlinear Coss, the PSFB with a saturable inductor are now
  runnable on esdirk43).

Verification on this tip: pytest 5722 passed, 2 failed only from the
python/ working directory (`test_ai_guide` CLI resolves a stale build
path, `test_exact_route` opens an example by a root-relative path; both
pass from the repository root); `pytest -m reference` 382 passed;
`reference_status` exit 0 with the strict counts unchanged (pwl
81/27/1/0/0, pwl+restart 89/19/1/0/0, pwl+split 91/17/1/0/0, trbdf2
106/2/1/0/0, dsed 100/0/0/9/0); C++ step, dsed, perf, thermal, record,
layer5_v3, xdev pass on Eigen 5 and on Eigen 3.4.0; g++-15 syntax check
clean on the touched headers; stubs up to date; ruff clean. Full ctest
not run.

## 17. Every stateful kind on the stage form; the q rows in the estimate; one round-off floor (2026-11-19)

**Stage form for the charge kinds and the machines.** The Lauritzen
diode (q_M) and the IGBT tail (Q) commit through the affine relation
q = K0 + K1·s(x) that `begin_step(h, stage)` rebuilds at every stamp; a
stage's increment is `K0 + K1·s(x) − ψ` with the K0/K1 of the stage's own
last stamp, and `stage_commit` calls `begin_step(h_e, BE)` before the
hand-path commit because the rollback restored the pre-step pair. The
PMSM and induction machine carry λ (3 rows per machine) through
`lam_off` / `delta_lambda`. The `kStageForm` trait and `stage_form()`
are gone: every kind is on the form. Truths (`test_esdirk43_stateful_
kinds.py`), esdirk43 vs TR-BDF2 at the same rtol:

| case | truth | esdirk43 | TR-BDF2 | order (fixed ceiling) |
|---|---|---|---|---|
| Lauritzen reverse recovery | closed form | 3.2e-6 (rtol 1e-8; source-edge floor) | 2.0e-1 at any rtol | 3.97, 4.05 (TR-BDF2 2.1) |
| IGBT tail turn-off | closed form | 1.4e-7 (1e-8) | 5.2e-3 | 3.88, 3.74 |
| PMSM locked rotor | closed form | 8.5e-10 (1e-9) | 9.3e-8 | 3.96 (then a 4e-9 floor) |
| PMSM free acceleration | TR-BDF2 at 1e-10 | rel 9e-7 on ω_end (1e-8) | — | — |
| induction machine | pinned SciPy | ew 5.3e-8, ei 4.5e-6 (1e-8, 20 µs ceiling) | 5.7e-6, 4.3e-6 | — |
| DC machine | closed form | 8.2e-10 (1e-8) | 3.2e-6 | — |

A rejected step leaves no device state behind: with `dt_init = t_end/4`
the Lauritzen, IGBT and PMSM-free runs reject 12, 11, 9 times (1, 0, 0 without) and are
bit-identical, times and states, to the runs started at the step they
accepted (`ed_stages` snapshots and restores `hs` on every path; K0/K1
are part of the rollback). The interface path agrees with the legacy
path on the Lauritzen and IGBT runs.

**Jiles-Atherton stays refused (by name, a new reason).** Loop energy
against the fixed engine at dt 1e-6 (W = 1.78336e-4 J): esdirk43 at a
5e-6 ceiling −2.4e-7, 2e-5 −1.5e-5, 1e-4 −3e-4..−4.8e-4: first order,
and TR-BDF2 is first order on the same law, even with c = 1 (ratios
~2.0). The in-loop JA law is the limiter; a fourth-order integrator buys
nothing there.

**The q rows in the error norm (esdirk43).** A circuit whose dynamics
live only in the stateful kinds (a Lauritzen recovery, an IGBT tail) had
no estimate: the embedded difference covered the linear entries only,
so steps sat at the ceiling for any rtol. The q rows now enter the norm,
scale `rtol·max(|q0|, |q1|, 1e-3·max|q0|)` (the max over committed
step-start values only, so a rejected trial cannot loosen later scales). FINDING, left unchanged: the
same gap exists on TR-BDF2 (its Lauritzen error is 2e-1 at any rtol);
closing it changes default numbers, so it needs its own change.
(2026-11-19: closed by update-trbdf2-error-control DECISIONS 10. The
rows now sit once in `tb_error` for both integrators, scale
`atol·s + rtol·max(|q0|, |q1|)` with s the device's own C / L / τ instead
of the 1e-3·max|q| floor, which was a vanishing scale for a row that never
moved: a nonlinear capacitor from x = 0 aborted at t = 0 here. They join
the entries by max of the two RMS, not one RMS. The esdirk43 numbers above
move within each case's floor band; that entry has the table.)

**carry_q (measured, not ported).** 10^4 periods, rtol 1e-6. Flux (a
saturable inductor across a cosine source, exact invariant λ(kT) =
λ(0)): drift −2.8e-12 of V/ω per period on esdirk43. Charge (a tanh-law
C1 and 1 µF C2 exchanging charge through a periodic switch, no DC path
from their top plates): the raw drift −4.7e-7 of Q0 per period is a
modeled leak, the same with linear capacitors (−3.3e-7), on both
integrators, at any g_off, R_b, rtol (1e-6..1e-8) or Newton tolerance;
a 5e-10 S leak through ∫(v_a + v_c − 2v_b)dt removes it to a residual of
−4.7e-10 (esdirk43) / −5.0e-10 (TR-BDF2) per period, also independent of
rtol. The engine's Newton tolerance is absolute and tight, so the
committed q(x) differs from the carried stage value at round-off; the
step-layer measurement (test_esdirk43.cpp, newton_tol 0.03) is where
carry_q pays. Not ported.

**One round-off floor.** The estimate's floor (noise = J^-1(|J||x| +
|b|)·eps) now runs once in `tb_error` for both integrators. The q rows
keep their share of the norm without a floor, so the floor never accepts
a step on a stateful kind's error (today the floor does not run at all
with a stateful kind: they set nl_effective). ESDIRK reads
each entry's state noise times Σ|e_j| (+|e_0|/γ); the noise right-hand
side alternates signs (an all-positive one cancels through J^-1 where
the elimination's round-off does not: PSFB defect (2), update-trbdf2-
post-event-restart DECISIONS 21). TR-BDF2 keeps today's rule unless
`roundoff_floor = 1` (low-level `run_transient_trbdf2` only; default 0:
byte-identical by construction). PSFB 0–2 ms, mean over 1.8–2.0 ms vs
the exact engine's 24.0661693 V:

| run | error | accepted | rejected | forced | floor accepts |
|---|---|---|---|---|---|
| TR-BDF2 rtol 1e-5, floor 0 | −9.14e-2 | 23797 | 8233 | 2336 | 8 |
| TR-BDF2 rtol 1e-5, floor 1 | −8.92e-2 | 24379 | 8152 | 2210 | 3 |
| TR-BDF2 rtol 1e-6, floor 0 | −7.56e-2 | 20996 | — | 3884 | 30 |
| TR-BDF2 rtol 1e-6, floor 1 | −7.02e-2 | 20871 | — | 3779 | 3 |
| esdirk43 rtol 1e-5 | −1.02e-1 | 9634 | 1239 | 112 | 0 |

Both complete either way; the floor is not what limits the PSFB (the
forced accepts at h_min are). LC tank, 1000 periods, esdirk43: rejects
15 / 7 / 6 / 6 at rtol 1e-5 / 1e-8 / 1e-10 / 1e-12 (≤ 0.001 per step),
identical before and after, 0 floor accepts: the 4.5–8 per period of
entry 15 was the prototype's.

**Task 3.4.** A Level-1 MOSFET buck (high side, Coss, Shockley
freewheel) and the IGBT-tail + Lauritzen chopper against TR-BDF2 at rtol
1e-9 under a 2.5 ns ceiling: esdirk43 1.1e-7 / 5.5e-8 of the state
scale (TR-BDF2 at its own steps 1.3e-7 / 3.8e-7). `engine='auto'` never
selects esdirk43, so its results are unchanged by construction.

**Device DC start (add-unified-device-interface #56, same branch).**
`simulate(start_from_dc_op=True)` now solves the DC point with the
Lauritzen law (it was open: 2.000 V where the law gives 0.73096 V) and
seeds every stateful kind from it (`StatefulHistories::seed_from_dc_op`).
Default-path number moved, truth-backed: `compute_dc_operating_point`
on the legacy path with a Lauritzen diode, 2.000 → 0.73096 V (bisection
of the law). See that change's DECISIONS, slice 11c.

Hand-off after entry 17:
- DONE: the Lauritzen diode, IGBT tail, PMSM, induction and DC machines
  on esdirk43 (refusals gone) with closed-form / reference truths, orders
  and the reject-leaves-no-state test; the q rows in the esdirk43 norm;
  one round-off floor in `tb_error` (TR-BDF2 default unchanged, opt-in
  `roundoff_floor = 1` on the low-level binding); task 3.4.
- NOT DONE: (a) dense-output bracketing of events (stage values only);
  (b) the Jiles-Atherton law stays refused (first order on both
  integrators: the in-loop law, not the integrator, needs work);
  (c) the q rows missing from TR-BDF2's own estimate (a default-number
  change: Lauritzen 2e-1 at any rtol there); (d) carry_q in the engine
  (no measurable drift to remove); (e) tasks 4.3, 5.x; (f) the Newton
  refresh rule of entry 8; (g) the SEPIC-like g_off node for the floor;
  (h) commutation energy as a thermal state jump: `feat/core-p2-trbdf2-
  thermal` was not in the integration branch at the start of this step;
  it is now (origin/feat/core-integration a32b97ed), and this branch
  does not have it yet: merge first, then start.

Verification on this tip: pytest 5971 passed (110 skipped, 8 xfailed)
from the repository root; `pytest -m reference` 383 passed;
`reference_status` exit 0, REFERENCE_STATUS.md unchanged (strict counts
unchanged); C++ step, dsed, perf, thermal, record, layer5_v3, xdev pass
on Eigen 5 and on Eigen 3.4.0 (trees deleted); g++-15 syntax check of
bindings.cpp and test_recorder.cpp: no error, no warning in a touched
header; stubs up to date; ruff clean; openspec strict valid. Byte
compares against the pre-change build: goldens, stateful 41, engines,
steps, pwl_identity 390/390, trbdf2_identity 371/371, AC 57/57, the
recorder set and the esdirk43 scoreboard identical; only esdirk43 runs
with stateful kinds moved (saturable LC, saturable table, the device
circuits: the q rows in the norm), all inside their test bounds. Cost
(instructions per accepted step, min of 3 interleaved runs, base | tip):
buck_pwl 3100.2 | 3101.5, vsi_pwl 4769.8 | 4769.5, buck_trbdf2 11038.5 |
11043.8, bridge_trbdf2 12799.4 | 12810.5 (all within +0.09 %). Reviewer
(fresh context, read-only): one MSVC capture (`ed_sf` not named with
`(void)` in tb_error), fixed; doubts left: the PMSM `stage_base`
overwrites a non-zero step-start `lam_off` (only on a resume from a
snapshot not taken at a commit), `ed_sqmax` is not in the resume
snapshot, a q row that has always been exactly 0 has a scale of
`numeric_limits::min()`, no test drives the new kinds through a diode
event localisation, a failed stage or the forced SDIRK step. Full ctest
not run.
