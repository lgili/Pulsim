# F-S-5 — trbdf2 / esdirk43 abort on a half bridge with nonlinear C_oss (case A)

Branch `fix/core-trbdf2-coss-abort`, base cedac36b (`feat/core-integration`).
Test: `python/tests/test_release_findings_fs5.py` (case A of
`test_soft_switching_validation.py` on feat/core-integration-14t, copied with its
closed forms, plus rtol 1e-4 / 1e-8, `dt` 1e-10 and a 50 ns run).

## Reproduction (stored extensions)

| extension | trbdf2 | esdirk43 |
|---|---|---|
| b9e46778 | runs (books 1381.7 uJ: energy cell fails) | runs; rtol 1e-8 aborts at 1 us |
| be5454f3 | aborts at 1 us | runs |
| cedac36b (PR head) | aborts at t = 0 | aborts at t = 0 |
| D3 gate-edge branch (p5-igbt 6dffe5f4 extension) | aborts at t = 0 | aborts at t = 0 |

Reduced (no switch, no diode): `Vdc` ramp 0 -> 400 V in 100 ns, 1 mOhm vdc -> sw,
ONE nonlinear capacitor sw -> gnd, optional 10 A ramp out of sw. Runs at b9e46778,
aborts at t = 0 on cedac36b on both integrators, for m = 0.5 AND m = 0 (a linear
charge law). No dependent states: the t = 0 abort is not the capacitor loop.

Amplitude / stiffness scan of the reduced case (m = 0, linear law; cedac36b):
ramp 1 V and 4 V run; 40 / 400 / 4000 V abort (residual at h_min 2.9e-6 / 2.8e-6 /
3.6e-6 A, dx 3e-13 .. 3e-12); R = 1 mOhm and 10 mOhm abort, 100 mOhm runs; a DC
400 V source (inconsistent start) runs; the same ramp delayed by 10 ns runs; a dt
ceiling of 1e-13 .. 1e-16 s still aborts (residual 1.0e-9 .. 1.3e-9 against tol 1e-9).

Hand solution at t = 0+ (case A, Sh ON = 1 mOhm across Ch, all else OFF):
vdc = v_sw = 0 and I = 0 at t = 0, so the switch carries no current and the two
capacitors split the bus slope: C0 (dvdc/dt − v') = C0 v' ⇒ v'(0+) = 2e9 V/s,
i_Ch = i_Cl = 2 A, I(Vdc) = 2 A. In the reduced case (one capacitor) i_C(0+) = 0:
the zero derivative slots ARE consistent there, and it still aborts.

## Cause (b), the t = 0 abort since cedac36b: the charge law's cancellation near 0 V

Not the slots and not the dependent states (above). The abort signature is a Newton
that converged in dx (6e-13 .. 3e-12, halving) while the residual at node sw sits
at 2.8e-6 .. 3.5e-6 A, a backward error of 2e9 eps against the start state: the
residual does not move when v moves. With m = 0 the law is linear, so the only
non-linearity left is how Q is computed:

    Q = C0·V0/(1−m)·[pow(1 + v/V0, 1−m) − 1]

which cancels near 0 V: Q is held to C0·V0·eps absolute (1e-9·25·2.2e-16 = 5.5e-24 C;
m = 0.5: 1.1e-23 C), whatever v is. The companion divides it by the step
(i = (Q − Q_n)/(γs·h)), so the residual floor is C0·V0·eps/(γs·h): 1.9e-8 A at
h = 2e-15 s, 1e-5 A at h_min = 3e-18 s — above tol_res = 1e-9 A at every
h < 3.8e-14 s.

Why only since cedac36b: the run's first step is now the SDIRK restart step and it
carries the q-row estimate (its embedded first-order estimate). From rest the charge
grows as t² (the ramp behind a 1 ps RC), so that estimate is O(1) relative to Q at
every h and only the absolute part atol·C(0) = 1e-17 C accepts it: h ≈ 2e-15 s
(hand model of the stepper on the reduced case, `.scratch/f5/model2.py`: the step
sequence 3e-10 → … → 3.2e-15 (err 2.0) → 2.0e-15 (err 0.81, accepted), then TR-BDF2
steps of 2e-15 .. 1e-13 s, all with err < 1). There the law's floor (1.9e-8 A)
fails Newton; each Newton failure quarters h down to h_min. At b9e46778 the first
step was a TR step with no q estimate: no femtosecond step, no floor met.

Decisive experiment (cedac36b extension, reduced case, both integrators): the
cancellation quantum is C0·V0·eps, so with the LINEAR law (m = 0) V0 changes
nothing but the quantum. V0 = 2500 / 25 V abort; V0 = 2.5 / 0.25 / 0.025 /
0.0025 V run (m = 0.5: the same split). `.scratch/f5/v0scan.py`.

Fix: Q = C0·V0/(1−m)·expm1((1−m)·log1p(u)) — relative to Q to a few ulp at every
bias — in the model (`models/nonlinear_capacitor.hpp`, every engine's hand stamp)
and the interface law (`device/laws.hpp`, the same form, tangent C0·exp(−m·log1p(u))).
