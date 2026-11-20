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
