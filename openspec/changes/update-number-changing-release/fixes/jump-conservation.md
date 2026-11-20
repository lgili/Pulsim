# Jump conservation under the restart rule (two fast modes in one cut-set; a fast move that stops inside stage 1)

Branch `fix/core-restart-jump-conservation` from `fix/core-restart-nonlinear-charge` b57ce86f (code 6a950022).
Marks: **[M]** measured, **[R]** read in the code, **[D]** derived.
Functions touched in `run_transient.hpp`: `consistent_start` only (and its scratch vectors).

## Reproduction on the base [M]
Reviewer's island (`.scratch/rev_fs6/t1_three.py`): C1 1 nF at 100 V –S (1 mΩ, closes at 2 µs)– C2 2 nF –R2– C3 1 nF,
all to ground, the island {n1, n2, n3} floating; dt 10 ns (γh = 2.93 ns). Truth: Σq conserved, all three end at
v∞ (linear 25 V, junction laws 19.14745364 V), record = E_before − E_after (linear 3.75 µJ, laws 2.141524 µJ).
| R2 | linear v_end | record | laws v_end | record |
|---|---|---|---|---|
| 0.05 | 25.0992 | 3.7596e-6 | 19.1957 | 2.1444e-6 |
| 0.4  | 25.7786 | 3.8114e-6 | 19.5507 | 2.1628e-6 |
| 0.8  | 26.4408 | 3.8408e-6 | 19.9297 | 2.1775e-6 |

## Derivation [D]
Probe: BE stages of γh on one factor, x_k = T x_{k−1} + f (linear new mode, T = (M + γhG)⁻¹M). Stage differences
Δ_k = x_k − x_{k−1}, second differences D_k = Δ_k − Δ_{k+1} (D1 = d, D2 = d' of the rule). Then D_{k+1} = T·D_k
exactly (the forcing drops out of a difference of differences when it is constant over the probe). A mode of T
with factor ρ (ρ = 1/(1 + γh/τ)) and amplitude c contributes Δ_k = c(ρ − 1)ρ^{k−1}, D_k = −c(1 − ρ)²ρ^{k−1}; its whole
jump is −c = Σ_k k·D_k = D_1/(1 − ρ)². Summed over modes: **J = Σ_{k≥1} k·D_k = (I − T)⁻² D1 on the fast subspace.**
The rule's per-state a = D2_i/D1_i is the one-mode model of that series state by state: exact when each state sees
one mode, wrong when a state is moved by two modes of different ρ (the island's C2 sees ρ1 ≈ 2.3e-4 and ρ2 = 0.013 …
0.17; C3 only ρ2). J_i = D1_i/(1 − a_i)² is then a different multiple per state, and Σ C_i J_i ≠ 0 although every
D_k conserves the charge of the floating island exactly (each BE stage is a KCL solve: P·M·Δ_k = 0, so P·M·D_k = 0).

Options judged on the linear model of the island (`.scratch/jc/proto.py`, exact BE map):
- (A) per-state J, then project onto Σq = 0: v_end right (the island's end value is fixed by Q alone), but the
  three voltages after the step are not equal (24.16 / 26.14 / 23.56 V at R2 = 0.8 instead of 25) and the record
  stays 2.3 % high (3.837e-6 vs 3.750e-6): the projection fixes one constraint, not the mode mixing. Rejected.
- (C) one stage with a large ε: needs a new factorization and is exact only as τ_fast/ε and ε/τ_slow → 0. Rejected.
- (B) the series as a VECTOR: since D_{k+1} = T·D_k, D1, D2, … is a Krylov sequence of T. On span{D1 … D_m}
  T acts as an m×m matrix H (column j+1 of [D2 … D_{m+1}] = H applied to column j; least squares for the last
  column), and J = [D1 … D_m]·(I − H)⁻²·e1, the Ritz values ρ_i of H clamped as the rule clamps a. m modes need
  D1 … D_{m+1}, i.e. m + 2 stages: ONE mode needs the three stages the rule has; TWO modes need a fourth stage
  (D3) — a 2-term scalar Aitken on three stages cannot separate them; the island has two; the vector fit is exact
  there (`proto.py`: v+ = 25.000000 at R2 = 0.05 / 0.4 / 0.8, record 3.750000e-6, Ritz values 2.3e-4 and
  0.0127 / 0.093 / 0.170). Chosen.
What is conserved after (B): J ∈ span{D1 … D_m} for any coefficients, so every LINEAR invariant that each BE stage
keeps — the charge of a node set the event leaves floating (Σ of capacitor charges, nonlinear ones in charge), the
flux of an inductor loop — is kept by J to rounding, whatever the fit's accuracy; it holds over the set of states
the fit includes (the jumping states). A non-jumping state keeps d (O(h²) on a smooth trajectory, as before) — an
invariant that also involves a non-jumping state is kept to that state's detection threshold.
Details: the fit uses the energy norm (√C·v, q/√C(v) for a nonlinear capacitor's charge, √L·i), the jumping states
only, m grows from 1 while the fit's residual of D_{m+1} is above the stages' noise (the rule's 1e-8 floor per state),
up to m = min(#states, 6); one jumping state keeps the rule's scalar code bit for bit (it IS the m = 1 case).
Above 6 unresolved modes the per-state a is kept (shortcut: exact for uncoupled modes, not conserving when coupled).

## The fast move that stops inside stage 1 [R][M][D]
Reproduction (`.scratch/jc/clamp.py`, test `test_a_fast_move_that_a_diode_stops_inside_stage_1_is_no_jump`):
a 10 A source pulls node n; Sl (1 mΩ) holds n at −0.01 V; Dl (V_f 0.7 V) clamps at −0.71 V; C 2 nF.
Sl opens at 1 µs (a TIME event, δt_loc = 0): n falls 0.7 V at I/C in 140 ps, then Dl conducts.
Base: a record of 4.9000e-10 J = ½·2 nF·0.7² exactly (linear; 4.9241e-10 the junction law), split
on and off — i.e. the allowance was ZERO.
Cause: the allowance is δt_loc·|ẋ(0−)| and ẋ(0−) is read from the history slot BEFORE the event:
i_C = 0, Sl held the node. The stage-1 widening (δt_loc → γh on a stage-1 diode flip) applies, but
multiplies zero. The rate that bounds a stop inside stage 1 is the rate AFTER the event in the mode
the event started in (the new gates, the diode bits before stage 1: Dl off): |d| ≤ θ·|ẋ(0+)|, θ ≤ γh.
A finite-rate move has ẋ(0+) = I/C there; a real jump is a fast mode of that start mode, after which
its rate is the slow one — so the start mode's own probe separates them: two single stages from x0
in it, Δ2 = γh·ẋ(0+) after its fast modes. Fix: when the test fired and stage 1's diode bits differ
from the event start's, that probe runs (two solves; only then), and each state's allowance is
max(δt_loc·|ẋ(0−)|, |Δ2 of the start mode|). A state over the old bound but within the new one is a
MOVE: carried to x0 + J like a jump (else stage 2's pseudo-state carries it: −0.71067 V at the next
sample instead of −0.71, measured with a version that left it unmoved), recorded by nobody, no
right limit at t_from.

## Implementation as landed (09126877) [R]
- Detection unchanged (the rule's per-state test; ẋ(0−) allowance), plus the start-mode allowance above.
- The series runs over ALL C / L / nonlinear-capacitor entries with one coefficient vector (charge and flux
  structural), whenever a state is carried and there are ≥ 2 entries; one entry keeps the scalar code bit for bit.
  A Ritz value outside the fast range (real ∉ [0, γ/(1+γ)], complex of modulus > γ/(1+γ)) is held at ρ = 0
  (factor 1: the rule's d for a state that does not jump). Fallbacks: the flagged states only (> 6 modes over
  all), then the per-state a (a stage failed, a source jump inside the extra stages, > 6 modes among the flagged).
- The start-mode probe: three single stages in (new gates, diode bits before stage 1); per state a smooth move
  (|Δ1 − Δ2| ≤ |Δ2|/γ + noise) allows |Δ2|; a jump of that mode only its Aitken slow rate (≤ |Δ2|); a third
  stage that fails (a junction law driven out of its range: the clamp's 14 V stages) gives 0 for a jump.
- Flags: 1 = jump (recorded, energy), 2 = move (carried, unrecorded, no right limit, no energy, mutual term
  skipped), 3 = a nonlinear capacitor the series carried without a flag (moved, unrecorded).

## Numbers at 09126877 against the truths [M]
Island (dt 10 ns; `.scratch/jc/nums.py`), after the event's step / record vs E_before − E_after:
| | R2 | v+ (all three) | record | truth | rel |
|---|---|---|---|---|---|
| linear | 1e-6 / 0.05 / 0.4 / 0.8 | 25.0000000 | 3.749999988e-6 | 3.749999988e-6 | ≤ 1.0e-13 |
| linear | 1e9 | 33.3333333, C3 3e-7 | 3.333333320e-6 | 3.333333320e-6 | 2.2e-13 |
| laws | 1e-6 / 0.05 | 19.1474536 | 2.141523786e-6 | 2.141523786e-6 | ≤ 5.7e-12 |
| laws | 0.4 | 19.1474539 / …539 / …528 | 2.141523982e-6 | 2.141523786e-6 | +9.2e-8 |
| laws | 0.8 | 19.1474635 / …634 / …176 | 2.141528859e-6 | 2.141523786e-6 | +2.4e-6 |
| laws | 1e9 | 25.9608554, C3 3e-7 | 1.946686259e-6 | 1.946686259e-6 | 3e-12 |
Base: +0.099 / +0.40 / +0.78 / +1.44 V (linear, R2 0.05/0.2/0.4/0.8), record +2.4 %; laws +0.048…+0.78 V, +1.7 %.
The laws' residual at R2 = 0.4/0.8 is the series' secant model of a nonlinear stage map (exact for a linear one).
ΣQ over the event step: ≤ 7e-19 C (linear; = Ssrc's OFF leakage 1e-12 S·100 V·dt), ≤ 6e-18 C (laws, Newton).
Inductor dual (L1 1 µH at 10 A through S, L2 2 µH, L3 1 µH, Rm): i+ = 2.5 / −2.5 / −2.5 A at Rm 1e9 / 2e4 /
2500 / 1250 Ω (truth −φ/ΣL = 2.5), Δφ ≤ 5e-21 Wb, record 3.740013322e-5 J (truth = ½Σ L(i² − i+²), test rel 1e-6);
Rm 1e-6: 3.33333332 / −3.33333332, L3 −4e-8, record 3.323348319e-5. Base at Rm 2e4: 2.5105 A, Δφ −4.2e-8 Wb.
Two snubbers (test_release_findings_a): record 1.000778e-5 J [½Cs(J1² + J2²) = 1.000778e-5]; Cs2 after the step
0.0099990 V [v_on 0.0099990]; base 1.000782e-5 and Cs2 +0.010220 V.
Clamp: no record at 1 µs (base 4.9e-10 / 4.924e-10 J), v = −0.71 V from the next sample on (1e-9).

## Reviewer (opus, fresh context, read-only; scripts `.scratch/rev_jc/`) on d7100f19 [M]
1. FIXED (09126877): a real jump hidden as a move — C1 1 nF/100 V –S– C2 10 nF clamped by Dl, R_on 0.3–0.85 Ω
   (ρ ≈ 0.04–0.12): the start-mode allowance |y2 − y1| held the fast mode's tail; records lost C2 or vanished.
   Now (window.py) every R_on 0.3–0.85: [C1, C2], 4.932695e-6 J [closed form 4.93270e-6].
   OPEN (pre-existing in kind): R_on 0.90–1.10 (ρ 0.12–0.15): the base already missed C1 (free probe after a
   stage-1 flip: not geometric) and recorded C2 alone, 1.38–1.69 µJ of 4.93; now 0.90 the same, 0.95–1.10 no
   record; the waveform is the base's (vm 1.94 → 0.736 V at R_on 0.95, identical). Cause not traced.
2. FIXED (09126877): ladders of 5–10 capacitors lost 8–11 % of their charge (unflagged island states kept d):
   now ΔQ/Q over the event 5e-14 … 6e-13 for N = 3, 5, 7, 8, 10 (ladder.py); base −4 … −9 %.
   island_hidden (C3 at a 1e7 V offset, hidden by the 1e-8 floor): Q to 1e-16 rel, v1+ 0.099801 [0.099800]
   (base and d7100f19: Q off 2.7e-8 C, v1+ 0.0687).
3. Minor, done: mutual term skipped for a moved winding; fallbacks named in the shortcut comment.
Held (reviewer): default path bit-identical (md5); ringing modes (pH loop inductance) Q to 5e-13, dW in 0.8 %.

## Gates at 09126877 [M]
- Byte compares vs a FRESH base recording (6a950022 extension, `.scratch/jc/pkgbase`; the predecessor's c_base
  was stale — trbdf2 entries differed there that this change cannot reach): goldens all identical except
  pyduty_split (rule on: states max|d| 24 V, restart steps 718 → 719) and clmon_split (rule on: 8.9e-11);
  pwl_identity 363/396 — all 33 differing entries are pwl+restart / pwl+split, every default entry identical;
  trbdf2_identity 377/377; dsed bit-identity 36/36 + shadow 36/36; stateful identical except
  everything_pwl_restart (rule on, max|d| 16.8); engines ALL IDENTICAL.
  pyduty_split / clmon_split against the same circuit at dt/16 (rule off): vout p99 6.59e-3, sw 0.382, out
  1.64e-3 — identical to 3 digits base and new (moved, not worse).
- reference_status (base vs new, --json): NO outcome moved. err_rms moved in 31 cells; up: half_bridge_transformer_dc
  2.663e-6 → 2.815e-6 (restart, +5.7 %) / 2.786e-6 → 2.967e-6 (split, +6.5 %), forward_converter +0.5 %, flyback
  +0.12 %, cascaded_boost_buck split +0.005 %, others ≤ 1e-6 relative; down: synchronous_flyback 1.619e-6 →
  1.288e-6 (−20 %, both), stiff_two_timescale_rc −0.02 %, ll15 restart −0.004 %. NOT traced: which event moves
  half_bridge_transformer_dc (a transformer pair at its first commutation; t_estimated of event 0 moved).
  Both runs exit 1 on the base's two lines (llc / psfb new outcome on pwl), as at the base.
- Cost, instructions per accepted step, min of six interleaved, vs the 6a950022 extension: buck_pwl 0.9991,
  vsi_pwl 1.0008, buck_trbdf2 0.9997, bridge_trbdf2 1.0000; rule on buck_pwl_restart 1.0068, vsi_pwl_restart 1.0025.
- The 60 restart / split test files at d7100f19: 1685 passed, 23 skipped, 33 xfailed, 2 failed (the two
  test_f24_exact_thermal_ratio… of the base).
