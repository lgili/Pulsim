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

Trace with the debug build (per attempt: t, h, SDIRK?, Newton ok; per q row the
estimate ee and its scale; `PULSIM_OLDLAW=1` restores the old law), reduced case,
m = 0.5, trbdf2:

| law | first-step attempts (h: q-row err, row 0 = the capacitor) |
|---|---|
| pow − 1 (cedac36b) | 3e-12: 5.8e4; 3e-13: 1.3e4; 3e-14: 177; 3e-15: Newton fails (residual stuck at 3.0e-9 A at node sw, dx 2.6e-12), then 7.5e-16 … 3e-18 = h_min: Newton fails → abort |
| expm1(log1p) | 3e-12 … 3e-14 as above; 3e-15: 1.80; 2.0e-15: 0.81, accepted; then TR-BDF2 at err 1.3e-4 and growing; runs (1069 samples) |

The first-step sequence is the hand model's to the digit (`.scratch/f5/model2.py`:
err 5.845e4 at 3e-12). With the law fixed, the reduced case runs on both integrators
(m = 0 and 0.5), and case A runs on esdirk43 at the defaults, rtol 1e-4, rtol 1e-8
(which aborted at 1 us even at b9e46778) and dt 1e-10. TR-BDF2 now aborts at 1 us:

## Cause (a), the 1 us abort since be5454f3: the probe's slot after a hard turn-on

Trace (law fixed, trbdf2 defaults), the steps after the high-side turn-on at 1 us
(400 V across Ch shorted by Sh's 1 mOhm; Cl at −0.47 V). All TR-BDF2 steps (s = 0),
rows 0 = Ch (q 1.56e-7 C, scale 1.56e-12), 1 = Cl (q −4.7e-10 C, scale 4.7e-15):

| h | ee row 0 | ee row 1 | err (row 1 / its scale) | |
|---|---|---|---|---|
| 6.0e-15 | −4.92e-13 | −4.80e-13 | 32.7 | reject |
| 1.9e-15 | −1.48e-13 | −1.48e-13 | 31.4 | reject |
| 6.1e-16 | −4.72e-14 | −4.72e-14 | 10.0 | reject |
| 2.9e-16 | — | — | — | Newton fails |
| 7.1e-17 | −5.53e-15 | −5.53e-15 | 1.18 | reject |
| 6.8e-17, 2.2e-17 = h_min | — | — | — | Newton fails → abort |

The estimate is proportional to h (ee/h ≈ 80 A, constant): not a truncation error
(that falls as h³) but a wrong derivative slot f_n, the O(h) term
2·clte·h·f_n/γ. Its source: at a gate edge `restart_after_gate_event` commits a
TRAPEZOIDAL probe (or rewrites the slots by the trapezoidal rule), so a kind's
slot becomes 2·i+ − i− instead of the post-edge current i+, off by the jump; TR-BDF2's
first stage then integrates from it. Row 1 names it first only because Cl's charge
(near 0 V) gives it the smaller scale; both rows carry the same ee (the loop current
Vdc–Ch–Cl). At h ≈ 1e-16 the companion's representation floor eps·Q_h/(γh)
(3.5e-23 C / 4e-17 s ≈ 1e-6 A) is above Newton's 1e-9 A: abort. Before be5454f3 no
q row existed, nothing shrank the step, and the run booked 1381.7 uJ for 62.8 uJ.
Not the dependent states: per row the estimates are each right for a wrong slot.

Not this finding (recorded): two nonlinear capacitors in a loop with the bus and a
10 A source out of their mid node, NO diode or switch (`.scratch/f5/loop2.py`)
aborts at 54 ns on every build (b9e46778 too): the source drags Cl to −22.29 V,
the law's floor (u = −0.9, −22.5 V), where Q is flat by design (DECISIONS #26) and
nothing can sink the current. Case A's diodes clamp the node at −0.7 V.

## Fixes

- (b) `models/nonlinear_capacitor.hpp` and `device/laws.hpp` (model and interface
  path, Real and AD): Q = C0·V0/(1−m)·expm1((1−m)·log1p(u)). Same function, relative
  to Q to a few ulp at every bias; C++ test in `test_device_laws.cpp`.
- (a) `trbdf2_transient.hpp`: a mode-changing gate edge in a run with a stateful kind
  makes the next step the SDIRK restart (no slot read), TR-BDF2 only. On ESDIRK4(3)
  forcing it broke the defaults at 1.5 us: the SDIRK estimate reads the 700 A ZVS
  spike (0.7 V on Cl through Sl's 1 mOhm, tau 1 ps) as an O(jump) error, h fell to
  ~1e-16 s, below Ch's companion floor, Newton stalled. ESDIRK4(3) keeps its own
  restart (explicit stage from the committed probe), which ran with fix (b) alone.

Case A after both (2.2 us): trbdf2 crossings −12.34 / −0.51 / +1.28 ps,
E 62.8490 uJ; esdirk43 −7.62 / +0.54 / +15.39 ps, E 62.9259 uJ (closed form
62.8335; b9e46778 62.9246: 1.3 nJ further, within the 5 % cell). Knobs: all pass
except trbdf2 rtol 1e-8.

## Remaining: trbdf2 at rtol 1e-8 (strict xfail)

Aborts at the 1.5 us edge (Sl on across conducting Dl). The post-edge h is the
remembered same-parity h (`h_mem_gate`), 2.2e-17 s = h_min, because resolving the
1 ps edge to rtol 1e-8 needs it. There Ch's charge (1.56e-7 C) gives the companion a
floor eps·Q/(γh) ≈ 1e-6 A against Newton's absolute 1e-9 A. Roundoff acceptance
sees omega 0.23 eps (would accept) but declines on "still contracting": a period-~7
limit cycle whose last ||dx|| (5.4e-6, 5.2e-7, 3.1e-7) contract by chance. Next
step: a stagnation test over the cycle (e.g. a longer window) in
`pwl/nonlinear_solve.hpp`, a change to add-newton-roundoff-acceptance DECISIONS 8/10b
(maintainer criterion), so not made here.
