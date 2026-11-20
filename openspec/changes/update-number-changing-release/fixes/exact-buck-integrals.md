# Exact engine vs trbdf2 / fixed engine on the buck window integrals (finding F-B2-1)

Suspicion (fixes/group-b2.md, "Numbers"): on the buck of
`test_recorder_device_readings._buck` the exact engine at `output_rtol=1e-10`
differs by 1.1e-4 to 2.0e-4 on ∫i(L), ∫v(out), ∫p(R) over [A, B] from
trbdf2 (rtol 1e-11) and from the fixed engine with the restart rule at dt/16,
which agree with each other to 3e-8; called "a modelling difference of the
exact route".

**Verdict: (c), an artefact of how the integral is taken — but not the
left-limit one. The exact engine's states are exact; its window integrals are
chords over its 10 us steps.** trbdf2 and the fixed engine are right.

Extension: `/Users/lgili/pulsim_wt/p3-devices/python/pulsim/_pulsim.cpython-313-darwin.so`
(branch `fix/core-release-b2-recorder-restart` at f0d3eaf9, as built).
Probes: `.scratch/exact_buck/` (`truth.py`, `vs_truth.py`, `integrals.py`,
`states.py`, `nearA.py`); `.scratch/ref44.py` is the report's own probe.

## Referee (independent of the three engines)

Two mpmath truths, no pulsim import (`.scratch/exact_buck/truth.py`):

1. `benchmarks/tools/exact_truth.py` from `origin/feat/core-soft-switching-validation`
   (fa4b5c71), the buck as a netlist (vcswitch driven by a 20 kHz, D = 0.4
   PWM source; diode g_on 1e3 / g_off 1e-9, V_th = V_f = 0; uic, zero IC).
2. By hand: x = [iL, vC], v_sw = (24 g_s − iL)/(g_s + g_d), x' = M x + c per
   (S, D) mode, x(t) = e^{Mt}(x0 + xp) − xp by eigendecomposition; diode
   OFF→ON iff v_sw ≤ 0, ON→OFF iff g_on·(−v_sw) ≤ 0 (bisection); window
   integrals by `mp.quad` on the closed form.

They agree to 18 digits on the integrals and 15 on the states. The one diode
turn-off (DCM at start-up) is at 147.784821853551 us in both and in the exact
engine's `event_times`. Same circuit everywhere: no diode-law, switch-form,
conductance or gate-timing difference (gate edges at k·50 us and k·50 + 20 us
on all engines).

## States at the window ends (a run that ENDS there; no integral, no interpolation)

Relative error vs the truth:

| engine | iL(A) | vC(A) | iL(B) | vC(B) |
|---|---|---|---|---|
| exact (default) | −1.1e-16 | +2.2e-16 | −6.1e-15 | −5.8e-15 |
| exact output_rtol 1e-10 | −3.2e-13 | −9.2e-13 | +1.4e-12 | +3.1e-12 |
| trbdf2 rtol 1e-11 | −4.4e-9 | +2.1e-10 | −4.5e-9 | −1.2e-9 |
| pwl rule dt/16 (interpolated) | −3.4e-7 | −1.1e-7 | −2.1e-7 | −2.8e-7 |

The truth sides with the exact engine to 14–15 digits.

## Window integrals over [A, B], relative error vs the truth (i(L), v(out), p(R))

| how | i(L) | v(out) | p(R) |
|---|---|---|---|
| recorder, exact, output_rtol 1e-10 (the report's call; feed in-loop, 101 steps) | −1.08e-4 | +1.60e-4 | +2.02e-4 |
| recorder, exact default (101 steps) | −1.08e-4 | +1.60e-4 | +2.02e-4 |
| stored record (102 rows), plain trapezoid = `with_event_limits` = `res.integral` | −1.08e-4 | +1.60e-4 | +2.02e-4 |
| stored record, output_rtol 1e-10 (165 269 rows, `with_event_limits`) | +8.9e-6 | −3.95e-4 | −1.40e-3 |
| recorder / stored, exact dt = 1e-7 | −1.08e-8 | +1.60e-8 | +3.30e-8 |
| recorder / stored, exact dt = 1e-8 | −1.08e-10 | +1.59e-10 | +3.29e-10 |
| recorder, trbdf2 rtol 1e-11 | −1.1e-9 | −1.9e-9 | −3.8e-9 |
| recorder, pwl rule dt/16 | +1.8e-9 | −1.3e-9 | −5.8e-9 |

- The event limits change nothing here: i(L), v(out), p(R) do not jump
  (`n_jumps` 0); the left-limit convention is not the cause.
- The error is the chord's: exactly h² (10 us → 0.1 us → 10 ns:
  1e-4 → 1e-8 → 1e-10, the same digits).
- The report's `output_rtol=1e-10` never reached the engine: `simulate()`
  drops it when `recorders` is given (`__init__.py`: "a recorder samples on
  its own grid: the output-grid rule stays off"), with no warning (checked
  with `-W always`). The feed then sees the exact engine's default steps:
  one per 10 us cap (`recorder_feed['n_steps'] = 101`).
- Even the stored record at output_rtol 1e-10 is 1.4e-3 off on ∫p(R): the
  4096-sample cap bit in 40 intervals (`diagnostics['output_grid']
  ['n_output_capped'] = 40`), leaving 10 us chords (e.g. 203.2 → 213.2 us,
  across A). That one warns (RuntimeWarning).

## What a user following the documentation gets

Silently the wrong one, on every route a user would take on `engine='exact'`
without `dt`:

- `Recorder(windows=...)` (`_recorders.py`): "statistics over each window,
  clipped exactly"; the module docstring sends the exact engine through the
  in-loop feed ("events at their exact instants … they see every accepted
  step"). Nothing says an accepted step is a 10 us chord.
- `res.avg` / `res.integral` (`_recorders.result_avg`): "windows clipped
  exactly. **Correct on any grid, unlike `.mean()`.**" — the misleading
  sentence: on the exact engine's default grid it is 1.1e-4 off on this buck.
- `output_rtol`'s docstring does say "Ignored with `dt` / `store_every` /
  recorders", but the run does not warn.

## Finding F-B2-1 and what should change

The exact engine (states exact to 1e-14) hands back window statistics with
the trapezoid error of 10 us steps, silently. Options, smallest first:

1. Documentation: `result_avg`'s "Correct on any grid" → "exact for the
   piecewise-linear interpolant of the samples; on `engine='exact'` without
   `dt` the samples are 10 us apart and the error is the chord's (h²)"; the
   Recorder `windows` doc and the AI guide the same; warn when `output_rtol`
   is dropped for recorders.
2. Engine: the exact engine integrates each segment in closed form (it has
   the per-mode decomposition; `exact_truth.py` does exactly that), so its
   recorder windows / periods are exact like its states.

Until then: on `engine='exact'`, pass `dt` (1e-8 gives 1e-10 here) for
window statistics.

The note in `test_release_findings_b2.py` (test of the window integrals vs
trbdf2: "a model difference of its own") is wrong on the cause; it is the
chord, not the model.

Test: `python/tests/test_release_findings_b2_exact.py` — strict xfail
(raises=AssertionError) on the recorder integrals within 1e-6 of the truth;
its control (states within 1e-12 of the truth) raises RuntimeError so a state
regression fails instead of being absorbed by the xfail.

## Run / not run

Run: `.scratch/ref44.py` (reproduces 1.08e-4 / −1.60e-4 / −2.02e-4 vs trbdf2
exactly), the probes above, the new test (1 xfailed; `--runxfail` fails on the
integral assertion with the numbers above), ruff on the test. Not run: the
full pytest, ctest (no C++ touched), any other circuit.

---

# Fix (branch `fix/core-exact-window-integrals`)

**Result: the exact engine's in-loop recorder windows and periods are the
trajectory's, to 4.4e-16 on F-B2-1's buck, independent of the output grid.**

## The closed forms (`core/include/pulsim/dsed/segment_integrals.hpp`)

Augmented energy coordinates of `period_integrals.hpp`: x̄ = (x, 1),
x̃ = d∘x̄, As = d·[[A, b], [0, 0]]·d⁻¹ = V·diag(μ)·V⁻¹, z = V⁻¹x̃0, and at a
piece starting s0 into the step ζ = z∘e^{μ s0}. A probe that is a polynomial
of degree ≤ 2 in readings affine in the state (v(), i() of a state, a
resistor, a PWL switch / diode under the step's mask with g_on / g_off /
V_f, a constant current source; p() = product) is

| degree | y(s) | ∫₀ᴸ y | ∫₀ᴸ y² |
|---|---|---|---|
| ≤ 1: l·x̄ | Σ α_i e^{μ_i s}, α = (Vᵀ(l/d))∘ζ | Σ α_i Lφ₁(μ_i L) | Σ α_iα_j Lφ₁((μ_i+μ_j)L) |
| 2: x̄ᵀQx̄ | Σ β_ij e^{(μ_i+μ_j)s}, β = (Vᵀ D⁻¹QD⁻¹ V)∘ζζᵀ | Σ β_ij Lφ₁((μ_i+μ_j)L) | Σ_pq γ_pγ_q Lφ₁((σ_p+σ_q)L), p = (i ≤ j), σ_p = μ_i+μ_j, γ_p = β_ii or β_ij+β_ji |

Lφ₁(νL) = ∫₀ᴸ e^{νs} ds through `phi1` (expm1 form): λ = 0 gives L exactly
and |λ|L = 1e-12 loses nothing to cancellation (test: 1e-13).

**Conditioning.** A sum over a product of p modal expansions carries
u·κ₂(V)^p of the natural scale ‖l̃‖^k‖x̃‖^k: p = 1 the state, 2 a degree-1
∫y² and a degree-2 ∫y, 4 a degree-2 ∫y². Measured on a near-defective pair
δ/a = 1e-3 (κ ≈ 1e3): 6e-13, 8e-10, 6e-6. Each is modal only under
u·κ^p ≤ 1e-10 (the propagator's bar). Otherwise the matrix form, no
eigenvectors: Φ = e^{As L}, I = ∫Φ, W_Q = ∫ΦᵀQΦ by a Taylor start at
max(‖As‖₁, ‖As‖∞)·h ≤ 1/4 (24 terms) and k doublings Φ(2h) = Φ², I(2h) =
I + ΦI, W(2h) = W + ΦᵀWΦ — only forward propagations, so a 5e12 1/s mode
cannot overflow (test). Degree 1: ∫y = l̃·I x̃, ∫y² = x̃ᵀW_{l̃l̃ᵀ}x̃; degree 2:
∫y = x̃ᵀW_Q̃x̃, ∫y² by the same Gramian one level up (w = x̃⊗x̃, w' =
(As⊗I + I⊗As)w, y = vec(Q̃)·w) for n + 1 ≤ 8; past that the piece's ∫y² is
the line, counted. Defective pairs (Jordan block, an integrator's λ = 0 with
the augmented constant), δ/a = 1e-8 and δ/a = 3e-2 (κ ≈ 1e2: modal
products, Kronecker square) are all within 2.3e-13 of 50-digit mpmath
truths (`core/tests/dsed/test_segment_integrals.cpp`; worst 2.3e-13 is the
κ ≈ 1e2 modal product, u·κ²; the rest ≤ 7.6e-15).

**Windows inside a segment.** The recorder clips; the source integrates
[lo, hi] ⊂ [ta, tb] with s0 = lo − ta (ζ above, or Φ(s0)x̃0 in the matrix
form) and returns y(lo), y(hi) too: min / max take the samples at the step's
ends and the closed form at a clip point (v(out)'s window min at A moved
from the chord's 7.84945933 to 7.84937116894253, the truth 7.8493711689425).
Period statistics are windows of the same pieces.

**Checked, counted.** Each step's closed form must END at the engine's end
state (energy norm of x̃, 1e-9), so a numeric or corrected step is never
integrated as a mode. (Not per probe against the recorder's samples: a
reading like v(in, sw) = 12 − 11.999999999 carries the sample's own
cancellation, 1e-6 relative; that check refused 23 good pieces of the DCM
boost.) A piece with no closed form — a probe that is not such a
polynomial (abs, min, a quotient of states; also a unary function of a
constant, conservatively), a circuit with time-varying sources (PWM /
pulse / sine / table: the step's input is not the mode's constant b), a
degree-2 square past n + 1 = 8 on an ill-conditioned basis — is the
recorder's line, counted in `res.recorder_feed['closed_form']`
(`n_closed`, `n_line`, first `why`) and warned (RuntimeWarning).

**One implementation.** `detail::phi1_kernel` / `phi1_pair_kernel` and
`detail::modal_basis` in `period_integrals.hpp` are THE kernels: the
period map, the loss port (`loss_port.hpp` `energies_`) and the recorders
call them. Loss port energies and the period map's integrals are byte-
identical to the base (74/74 arrays, `.scratch/wi/port_bytes.py`).

## The ledger: window integrals and impulses

Between events the window integral of every element's absorbed power plus
the storage's ΔE closes to rounding. At a state jump (an ideal switch
closing a capacitor loop) the energy ½·C1C2/(C1+C2)·ΔV² leaves in zero
time: no window integral sees it (the integrand has no Dirac part; the
pieces end at the left limit and restart from the right limit). The loss
port books it as the switch's `E_impulse`. So the ledger over [0, T] is
ΔE_C + Σ ∫p (recorders) + Σ E_impulse (loss port) = 0. Measured (default
grid): buck 1.3e-13 of ∫p(V); DCM boost (57 events, diode turn-offs)
2.7e-14; capacitor pair + R through an ideal switch at 1 us: 6.9e-16 of
∫p(R) (E_impulse 2.42e-5 J), with the switch's own g_off leak ∫p(S) =
6.4e-14 J before it closes needed to get there (tests hold 1e-11).

## Decision 2: stored-record integrals stay the trapezoid (docs)

`res.integral` / `result_avg` / `with_event_limits` / the stored-arrays
pass integrate the stored samples. Computing them from segment data would
mean carrying every step's mode (A, b, mask, start state) in the result and
re-deriving modal forms post-run for arbitrary probes — the in-loop source
already does this at no storage cost. So the docs say what they are:
"trapezoid of the STORED samples: exact for the fixed engine's rule, O(h²)
otherwise (`engine='exact'` without `dt` stores one sample per 10 us); for
the exact average use an in-loop `Recorder(windows=...)`" (`result_avg`,
`_recorders` module notes, `docs/api-reference.md`, AI_GUIDE trap 2).

## Decision 3: no silent drop

- `output_rtol` with recorders is kept: the run is the run without them
  (samples, states, event location and projection: the rule sets those
  too), and the statistics stay closed forms (identical to the run without
  the rule to 1e-12; F-B2-1 buck: bit-identical).
- The sample-cap / unbounded warning now says the stored-record integrals
  (res.avg / res.integral / with_event_limits) are chords over those gaps
  and that in-loop recorders integrate the exact trajectory.

## Numbers against the truth (F-B2-1 buck, window [A, B])

| | i(L) | v(out) | p(R) |
|---|---|---|---|
| before (chords) | −1.08e-4 | +1.60e-4 | +2.02e-4 |
| after, default grid | 4.4e-16 | 4.4e-16 | 4.4e-16 |
| after, output_rtol = 1e-10 | 4.4e-16 | 4.4e-16 | 4.4e-16 |

Test asserts < 1e-11 (the xfail is gone). What limits it: the modal sums'
rounding, u·κ·Σ|terms| per piece, summed over the pieces (u·κ^2 for the
products); here a few u. Grid independence (default vs dt = 1e-7 vs
output_rtol = 1e-8): 2.0e-15 / 1.2e-14 / 1.3e-14 and 0 / 0 / 0 (test at
1e-12). Over many pieces it adds up: buck_dcm's v(vout)*i(L) RMS differs
1.7e-11 between the default grid and dt = 1e-7 (3021 steps), 1.3e-10 at
dt = 1.5e-8 (20049) — the test of `test_exact_recorder_loop.py` holds 1e-10.

## Moved recorder values (probe_vals.py, F-B2-1 buck, w0 = [A, B], w1 = [0, 1 ms])

Old = base 19e4f1c5, new = this branch, ref = stored-line on dt = 1e-8
(itself O(h²): its error vs dt = 1e-7 is shown so new − ref ≈ (d7 − d8)/100
reads as the reference's own error).

| probe | stat | old | new | new vs ref | old vs ref | truth |
|---|---|---|---|---|---|---|
| i(L) | ∫ w0 | 1.281344126572e-3 | 1.281482929636e-3 | 1.1e-10 | −1.1e-4 | 1.28148292963602e-3 (new 4.4e-16) |
| v(out) | ∫ w0 | 6.325529461529e-3 | 6.324519877616e-3 | −1.6e-10 | 1.6e-4 | 6.32451987761644e-3 (4.4e-16) |
| p(R) | ∫ w0 | 1.218860865823e-2 | 1.218614789826e-2 | −3.3e-10 | 2.0e-4 | 1.21861478982593e-2 (4.4e-16) |
| i(L) | rms w1 | 2.229983026504 | 2.233809745122 | 1.9e-9 | −1.7e-3 | |
| v(out) | rms w1 | 9.709328797545 | 9.718791916879 | 1.0e-9 | −9.7e-4 | |
| p(V) | ∫ w1 | −1.930383192614e-2 | −1.932647772373e-2 | −1.1e-9 | 1.2e-3 | ledger |
| p(S1) | ∫ w1 | 2.147431071104e-6 | 1.997555116794e-6 | −7.5e-8 | 7.5e-2 | ledger |
| p(D1) | ∫ w1 | 3.088455403649e-6 | 2.992926250797e-6 | −3.2e-8 | 3.2e-2 | ledger |
| p(L) | ∫ w1 | −2.903109763604e-5 | 8.353442491743e-6 | 4.4e-6 | −4.5 | ½L·iL(T)², 1e-11 |
| i(D1) | ∫ w1 | 1.202338935643e-3 | 1.202240228223e-3 | −1.0e-10 | 8.2e-5 | |
| v(sw) | ∫ w1 | 9.618999720892e-3 | 9.618995440744e-3 | −7.3e-12 | 4.4e-7 | |
| v(out) | min w0 | 7.84945933003 | 7.84937116894 | | | vC(A) 7.8493711689425 |

(all 9 probes × 2 windows × integral / rms moved the same way; min / max
moved only at clip points, by ≤ 1e-5.) Test values that moved:
`test_exact_recorder_loop.py` — in-loop statistics no longer equal the
stored pass's line where closed (compared instead with a dt = 1e-7 run, 1e-10;
min / max vs the stored pass 1e-9: clip points); the trbdf2 comparison now
runs on the default grid (was dt = 1e-7). `test_exact_output_grid.py` /
`test_exact_output_rule_combinations.py` — recorders no longer drop
output_rtol.

## Cost (instructions retired, `/usr/bin/time -l`, F-B2-1 buck 1 ms, per run)

| | base 19e4f1c5 | this branch (merged tip) |
|---|---|---|
| exact run, no recorder (1000 runs) | 4.2066 M | 4.2140 M (+0.18 %, run-to-run spread 0.2 %) |
| one recorder, 9 probes x 2 windows (500 runs) | 13.20 M | 17.18 M (+30 %) |

With no recorder the source is never built (the feed adapter is not
attached); the +0.18 % is within the measurement's spread and includes the
merged F-branch commits.

## Run / not run

Run on the final tip: extension build (Eigen 5, clang, ccache);
`pulsim_core_dsed_tests` (148 cases, incl. the 6 new segment-integral cases)
and `pulsim_core_record_tests` (18); the exact-engine, recorder, F-finding,
period-KPI and AI-guide pytest files (578 passed); one full pytest on
d5310f5d (6123 passed, 110 skipped, 8 xfailed); dsed bit-identity 36/36 and golden baseline identical vs
the base build of 19e4f1c5; loss port energies + period map integrals
74/74 byte-identical; ruff on python/. Not run: Eigen 3.4.0, g++, full
ctest, the full exact scoreboard (it grades stored samples only, which are
bit-identical on the compared set); stubs (no binding signature changed:
the native result gains a `recorder_closed_form` dict key).

---

# Time-varying sources (branch `fix/core-exact-integrals-sources`)

**Result: with a sine, pulse, PWM or table source the in-loop recorder
windows and periods are closed forms too — the line (1e-4 off on the default
grid, warned) is gone for every source kind the engine propagates exactly.**

## How the engine propagates each source kind (builder_adapter.hpp)

| kind | propagation | exact? |
|---|---|---|
| sine V / I (`f > 0`) | one oscillator pair (sin ωt, cos ωt) per distinct ω augmented to the mode (`build_augmented_exact_`): amplitude and phase folded into its two coupling columns, v_dc into b | yes (tier 1 modal / tier 2 Padé; tier 3 numeric, refused unless `allow_implicit_segments`) |
| PWM V | level per step (`pwl_cols`), a corner is a time event | yes |
| pulse V / I, PWL table I | level + ramp per step (`exact_advance_affine_into`), never across a corner | yes, degree 1 |
| user `b_extra_fn` | numeric | no |

## Design: the inputs as an exosystem (make_exo_mode)

Every exactly-propagated source is a state of an autonomous u' = S·u
appended to x: per ω the engine's (s, c) pair (modes ±jω), per PWL source
its level v (mode 0) and, on a step where some slope is not 0, its slope σ
(v' = σ, σ' = 0: a Jordan block at 0, degree 1 = what the engine steps
with). With x̄ = (x, u, 1):

  As = d·[[A, Bx, b_dc], [0, S, 0], [0, 0, 0]]·d⁻¹,  x_full = [Rx, Ru]·(x, u) + rc,

Bx = b_projection·G, Ru = recover_from_b·G, G the sources' b_extra stamps per
unit of u (`NativeCircuitBuilderAdapter::exo_inputs`; `exo_state` /
`exo_advance` give u at a step's start and end: the oscillators at t, the
levels v + σL, a corner's left limit). The source's forced response is then
part of the modal sums; nothing of SegmentMode changed except the gate below:
the same kernels (`phi1_kernel`, `phi1_pair_kernel`, `modal_basis`), the same
u·κ^p rule, the same Gramian / Kronecker fallback, the same end check (the
closed form's end against the engine's end state, now with u). A waveform
current source's own reading (i(I), p(I)) is a row of u. Modes are cached
per (mask, ramp). The DC path is untouched (`make_period_mode`, byte-
identical: only its observable rows moved into `observable_rows`).

**A second conditioning gate (`sums_ok_`).** κ₂(V) does not see the
cancellation of a near-resonant drive: a series LC driven 1e-3 off its
frequency has κ = 10 (u·κ⁴ = 2e-12, modal accepted) but the forced and free
responses are each ~30× the state, and the degree-2 square's modal sum was
6.6e-10 off (numpy vs mpmath on one 1 µs step; 2.6e-10 on the window rms).
Each modal sum now also checks u·Σ|terms| ≤ 1e-10 · (the probe's own scale
over the piece, from the states at its ends: |l̃|·|x̃|, or |x̃|ᵀ|Q̃||x̃|, times
L, squared for ∫y²) and takes the matrix form otherwise: 2.6e-10 → 1.1e-13.
The gate applies to exosystem modes only: on DC modes the ends' scale is ~0
at every diode's zero current and it re-routed pieces whose two forms agree
to the circuit's rounding (DCM buck i(D1) period rms 8e-10 of 3e-4 A); DC
recorder values stay byte-identical (48/48 arrays, ccm + dcm buck).

**End check on (x, u).** The closed form's end is judged on the circuit's
states and the constant only (`SegmentMode` `nx`): a source level in volts
and a slope in V/s (5e5 on a 20 µs edge) next to √m-scaled states would let
any gap pass (reviewer finding D1). u's own end is not judged either way
(sin ωt1 recomputed vs rotated differ by u·ωt).

## Numbers against the truths (default grid unless named)

By hand (python/tests/test_exact_integrals_sources.py, mpmath quadrature of
the closed forms, 40 digits): RL on two sines (50 Hz + DC offset, 150 Hz,
phases), a three-phase set into star RLs, a series LC driven at, 1e-8 from,
1e-3 from and 1.7× its frequency — all integrals and rms ≤ 1e-11 (∫y relative
to ∫|y|: near resonance ∫v(c) is 2600× below ∫|v(c)|). RL probe:
6.3e-13 (∫i), 2.2e-13 (rms i); the resonant LC passes at 1e-11 (defective
augmented mode: Gramian / Kronecker form).

Kernel (core/tests/dsed/test_segment_integrals.cpp, 50-digit mpmath of
expm, `fixes/exact_sources_truth/seg_truth_exo.py`): resonant, 1e-8, 1e-3,
far, damped resonant over a piece, a ramp and a ramp's piece: ∫y, ∫y², a
degree-2 ∫y and ∫y², all ≤ 1e-12.

mpmath engine (`benchmarks/tools/exact_truth.py` at fa4b5c71,
`fixes/exact_sources_truth/diode_truths.py`): window mean / rms, and every
element's window energy (relative to the window's source energy):

| circuit | events | worst column | worst energy |
|---|---|---|---|
| half-wave RLC, 1 kHz | 12 | 2.2e-14 | 4.7e-12 of p(D1) itself |
| 1-ph bridge, Ls 1 mH, floating bus | 7 | 7.2e-12 (mean i(Ls), of itself) | p(D4) 5.9e-11 of itself = 1e-15 of the source; p(Rtie) 4.5e-8 of its 5e-4 J (the floating common mode) |
| 3-ph bridge, Ls 0.5 mH | 10 | 2.5e-9 | 9.5e-8 of a 2e-3 J diode energy (6e-12 of the source) |
| RLC on a pulse with 20/30 µs edges | 12 corners | 4.2e-14 | 1.4e-14 |

The 3-phase bridge is limited by the TRAJECTORY, not the integrals: the
engine's end states are off the truth by up to 1.8e-9 (i(La) at 8 ms; bus
7.7e-11 at 4.9 ms), and the truth's D2 turn-on at 5.0000003 ms and D5 turn-off
1.35 µs later are one event in the engine (8 events vs 10; the same at
11.6667 ms). The test holds it at 1e-8. FINDING for the exact-engine
owner (scheduler, not touched here).

Floating-bus bridge of the recorder-loop tests (40 ms): the 30–40 ms bus
average is 6.6e-14 off its 50-digit truth (311.343881428885841697 V) on the
default grid (the old line: −1.4e-12 on this window); at dt = 1e-7 it is
1.9e-12 (3.5e7 pieces of rounding).

Ledger (1-ph bridge, [0, 30 ms], default grid): Σ∫p(Rs, RL, Rtie, D1–4,
Ls, Vac) + ½C·v_C(T)² = 1.0e-12 of the source's 54.38 J; the source energy
is the truth's (54.384383077361804 J) to 1e-11. Half-wave RLC: 1.3e-14.

Grid independence (half-wave RLC): default / dt = 1e-7 / output_rtol = 1e-8
agree to 2.4e-12 (test 1e-10). dt = 1e-7 adds up rounding: 2e-12 (half-wave),
3e-11 (RL, 6.7e6 pieces).

## What still takes the line (counted, warned)

- a user `b_extra_fn` (the engine steps it numerically);
- a mode the engine stepped numerically (tier 3, `allow_implicit_segments`):
  the end check refuses it;
- a capacitor's own in-loop i(C) / p(C) (its companion law) and Newton
  laws, as before (use ½C·v² for the ledger);
- a degree-2 probe's ∫y² on a mode with no acceptable modal sum when
  n + 1 > 8, where n now counts the exosystem's states too (two per
  frequency, one per PWL source, one more per ramping source; the 3-phase
  bridge is 4 + 2 + 1 = 7) — its ∫y stays closed.

NOT routed: the default `integrator='auto'` of a diode-free circuit runs
PEDSimulatorAuto, which has no step hooks: its recorders take the stored-
array pass (trapezoid of the stored samples) for DC and AC alike. Pass
`integrator='rk45'` (the tests do). Routing recorders to the event-driven
integrator would move that run's stored samples: left to the owner.

FINDING (pre-existing, base 20affaf9 too): an RC driven by a pulse with a
finite rise into a diode (`v_D` = 0 exactly at the ramp's start, the
diode's first derivative 0) is refused: `DiodeModeInconsistentError ...
OFF while forward-biased by 2.4e-2 V at t = 6e-6 s`. The scheduler's
crossing scan misses a second-order turn-on; the pulse test uses an RLC
without a diode.

## Moved recorder values (base 20affaf9 → this branch, `.scratch/src/moved.py`)

| circuit | statistic | old | new | truth |
|---|---|---|---|---|
| floating-bus bridge | ∫v(dp,dn) 30–40 ms | 3.113438814285 | 3.113438814289 | 3.11343881428886 |
| floating-bus bridge | rms i(Ls) 30–40 ms | 7.30723062451 | 7.307231627426 | (old −1.4e-7) |
| floating-bus bridge | rms v·i 20–40 ms | 2151.413405541 | 2151.41372192 | |
| rectifier (R load) | ∫v(dp,dn) 7.3–33.1 ms | 1.520410095004 | 1.520410107508 | (chord h² 8.2e-9) |
| pulse RC (zero edges) | ∫v(in)·v(a) 0.15–0.45 ms | 0.01198522798102 | 0.01198689086503 | (old −1.4e-4) |
| pulse RC | rms v(a) 0.15–0.45 ms | 5.92279079959 | 5.923002449319 | |

(every integral / rms of those three circuits moved by the chord's h²:
1e-12 … 1.4e-4; min / max only at clip points.) Test values that moved:
`test_exact_recorder_loop.py` — pulse_corners, rectifier and bridge are now
closed (compared with a finer grid, dt = max(1e-7, t_end / 2e4): at 1e-7 the
bridge's 3.5e7 pieces add 1.4e-10 of rounding).

## Byte compares (base 20affaf9)

Goldens with counters: all identical; dsed bit-identity 36/36 arrays; loss
port energies + period map integrals 74/74 arrays; DC recorder statistics
(F-B2-1 ccm buck, DCM buck: windows + periods, integral / rms / min / max)
48/48 arrays. No binding signature changed (stubs not regenerated).

## Reviewer (fresh context, opus, medium) — what it found, what was done

- D1 end check diluted by the source's level / slope (fixed: x and the
  constant only).
- R1 sin ωt1 recomputed vs rotated (moot after D1: u's end is not judged).
- R2 the cancellation gate over-rejects at a DC diode's zero current and
  moved DC values ≤ 8e-10 of a small rms (fixed: exosystem modes only).
- R3 every ramp step is defective: a Gramian (and, for a degree-2 probe,
  the Kronecker 64×64) doubling per new step length — left; cost below.
- R4 untested: a waveform current source's own reading (exo_current_row),
  a PWM source's level, a sine with f ≤ 0, mixed PWM + ramping pulse —
  left untested (no truth built for them here).
- nits: kNoFourth message updated; unused exo_may_ramp removed.

## Cost (instructions retired, min of six, base 20affaf9 rebuilt with the tip's flags)

20 single-phase bridges × 20 ms, engine='exact'. A process includes the
import (n=0: base 3.13e9, tip 2.31e9 — the base runs from a `git archive`
tree; the difference is in the import, not in the engine), so per run =
(total − import) / 20.

| | base | tip | |
|---|---|---|---|
| no recorder, per run | 1.705e9 | 1.703e9 | −0.2 % |
| one recorder (4 probes, window + period), per run | 1.928e9 | 3.463e9 | +80 % |

The recorder's own share went from 0.22e9 to 1.76e9 per run: the exact
∫ / rms with the source modes (n + 2 states per piece, a Gramian per new
step length on the Padé tier) instead of the trapezoid. R3 (a cache of
the doubled Gramian per (mask, h)) is where that would come back.

## Full pytest on 3a5949bc

6135 passed, 110 skipped, 8 xfailed (11 min 46 s), build/python and
python/pulsim the same .so. Not run: Eigen 3.4.0, g++, full ctest.

# Every route, the cost, the source paths (same branch, from f7a7aeba)

## 1. The diode-free route (`integrator='auto'`, PEDSimulatorAuto)

A circuit with no diode, loss port, `output_rtol` or ticks runs on
PEDSimulatorAuto (`dsed/scheduler_auto.hpp`): per step the closed-form
propagator when the mode has one (`exact_advance_state_into`, the same as
PEDSimulator's), DOPRI5 / BDF2 otherwise. It had no step hooks, so its
recorders took the stored-array pass (the trapezoid of the 10 µs samples),
silently. Now it drives `events::StepHooks` with PEDSimulator's conventions:
`start` (before a resume's landing, as there), `event(t, left, old, right,
new)` from inside `fire_gate_event_` (a copy of the left limit only when an
event listener is attached), `accepted(t0, t, pre-event state, step mask)`
after both the exact and the numeric branch. `run_auto_native_builder`
takes `loop_feed=` and attaches the same `ExactFeedAdapter` +
`ExactSegmentSource`; a numeric (non-exact) step fails the closed form's end
check and is counted as the line, warned, as before. Nothing stepped or
stored changes: the hooks only read.

The remaining stored-array routes of the exact engine (`integrator='bdf2'`,
a fresh `t_start` run, a step hook that did not fire) now WARN
(RuntimeWarning, "trapezoid of the stored samples") when a recorder has
windows or periods; `res.recorder_feed['mode'] == 'stored-arrays'` counts
it. `test_exact_recorder_routes.py` enumerates auto (diode-free), rk45,
auto→rk45 (diode), output_rtol, bdf2, t_start × {with, without recorder}:
closed form with no warning, or stored arrays with exactly one warning;
times and states bit-identical with and without the recorder on each.

Numbers (window [0.21 ms + 1.234 ns, 0.87 ms + 4.321 ns], hand mpmath truth
of a SYNCHRONOUS buck — two ideal switches, 24 V, 100 µH, 10 µF, 5 Ω,
20 kHz, D = 0.4, from rest; one affine law per mode, M the same in both):

| statistic | f7a7aeba (stored arrays) | now (default route) | truth |
|---|---|---|---|
| ∫i(L) | 1.2848344194661705e-3 | 1.2850038019123695e-3 | 1.285003801912367e-3 |
| ∫v(out) | 6.324751201621357e-3 | 6.323796867570363e-3 | 6.323796867570354e-3 |
| ∫p(R) | 1.2189152099455192e-2 | 1.2186816641668244e-2 | 1.2186816641668225e-2 |
| series LC, detune 1.7, ∫v(c)·i(L) | −1.0103357289111828e-7 | −9.979694316339019e-8 | −9.979694316339956e-8 |

Tests at 1e-11 (the predecessor's tolerance): the sync buck (∫ and rms of
i(L), v(out), ∫p(R)), the series LC at detune 0, 1e-3, 1.7 and the RL on
two sines — the predecessor's own truths, now on the default route.

MOVED test values (the RC of `test_recorders.py`, engine='exact', window
[0.37 ms + 1.234 ns, 2.71 ms + 4.321 ns]; truth = the closed form in the test):

| | old (stored arrays) | new (in the loop) | truth |
|---|---|---|---|
| avg v(out) | 7.332494924704019 | 7.332495146996288 | 7.332495146996273 |
| rms v(out) | 7.533260060824895 | 7.533260244177804 | 7.533260244177797 |
| avg v(in,out)/1e3 | 2.667505075295974e-3 | 2.667504853003703e-3 | 2.6675048530037273e-3 |
| min v(out) (at the clipped start) | 3.0926652130277494 (chord) | 3.0926652172829012 | V(1 − e^{−A/τ}) |

`FEEDING['exact']` is 'in-loop'; `RC_BOUND['exact']` 2e-5·V → 1e-12·V; the
stored-sample `time_average` comparisons stay for pwl / trbdf2 only.

Byte compares against f7a7aeba (base rebuilt from `git archive` with the
same flags): goldens with counters ALL IDENTICAL; dsed bit-identity 36/36;
auto-route stored times + states (sync buck, LC ×2, pulse RLC) 8/8 arrays;
loss port energies + period map 74/74; recorder values of the rk45 route
(the predecessor's AC/pulse 64 and DC 48) 64/64 and 48/48 identical.
