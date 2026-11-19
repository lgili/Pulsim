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
