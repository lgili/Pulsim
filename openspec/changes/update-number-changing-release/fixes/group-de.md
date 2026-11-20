# Release findings, groups D and E (+ F-6-1): trace evidence and fixes

Branch `fix/core-release-de-trbdf2-tick`, base the PR head b9e46778 plus
`fix/core-trbdf2-state-error-norm` (ecb07507, merged first: it enters the PR
line in batch 14q and touches the trbdf2 error phase). Tests:
`python/tests/test_release_findings_de.py`.

## F-4-5 — trbdf2 loss port books ~1000x the switching energy with a nonlinear C_oss

Call: `test_loss_port_seams._stateful_with_loss(True, True)`, `engine='trbdf2'`.

Sweep on b9e46778 (heat_injected, J; truth 4.04e-3): rtol 1e-3 6.56, 1e-4 6.56,
1e-5 6.97, 1e-6 7.40, 1e-7 6.44, 1e-8 5.56. Wrong at EVERY tolerance, the
defaults included (6.97 J), not from some onset. The ledgers split it:
E_on = E_off = 2e-3 J (right: 20 edges each at the 1e-4 J knot);
E_cond_electrical = 6.968 J against 2.9e-5 J on pwl. With a linear C_oss the
same run is right (3.95e-3 J), with 2419 accepted steps against 1076.

Trace of the turn-on at 20 us (rtol 1e-3 defaults, samples):

| t − 20 us | h | v_S | i_S | v·i |
|---|---|---|---|---|
| 0 (left) | 9.8e-10 | 48.004 | 4.8e-8 | 2.3e-6 |
| +9.8e-16 (right limit) | 9.8e-16 | 47.964 | 47964 | 2.30e6 |
| +9.8e-10 | 9.8e-10 | −0.165 | −165 | 27 |
| +5.9e-9 | 4.9e-9 | 0.0045 | 4.45 | 0.020 |

T_j jumps 25.099 → 27.448 °C over the 9.8e-10 step: 2.35e-3 J booked by the
TR-BDF2 quadrature h·(w·p0 + w·p_γ + d·p1) from the right limit's 2.3 MW (the
TR stage rings to v_γ ≈ −v0, so p_γ is as large). The channel–C_oss loop has
tau = R_on·C_oss(48 V) = 1e-3·5.85e-10 = 0.6 ps; the step is 1600 tau.

Cause: the TR-BDF2 error estimate had no row for the nonlinear capacitor's
charge (a stateful kind), so nothing made the controller resolve the
discharge; the state comes out right (L-stable BDF2 stage), the energy
integral does not. NOT the tracker's dissipating-current definition (i is the
channel's own branch current, the capacitor's current is not in it), NOT the
event reading (E_on / E_off exact), NOT an impulse double count (E_impulse 0).

Fixed by ecb07507 (q rows in the error norm). On the merged tree:
heat 3.94310e-3 / 3.94308e-3 / 3.94303e-3 / 3.94311e-3 / 3.94236e-3 / 3.94211e-3
at rtol 1e-3 .. 1e-8; T_j 28.57 °C; the discharge is resolved (407 samples in
the first ns, steps 1.5e-14 s).

Truth (closed form + the run's own electrical trajectory, which the junction
does not feed back): E_cond = R_on·∫_ON i_L² + g_off·∫_OFF v² + Σ E_oss(v_k),
E_oss(v) = ∫_0^v u·C(u)du = C0·V0²·[(2/3)w^1.5 − 2w^0.5]_1^(1+v/V0)
(7.7649e-7 J at 48.0043 V, quad 7.76494e-7). Whole run: truth 4.12e-5 J;
trbdf2 4.30e-5 (defaults), 4.21e-5 (rtol 1e-8), esdirk43 4.29e-5; pwl 2.92e-5.
One edge (19.9 .. 21 us): truth 7.981e-7 J; trbdf2 8.149e-7 (rtol 1e-6),
8.099e-7 (1e-9), esdirk43 1e-9 8.116e-7.

Remaining 1.5 % per edge (recorded, not fixed): the first step after the gate
event (h1 = 1.5e-14 s) starts the capacitors from their PRE-event current (the
first step's decay is half the exact one, 47.41 V where the ODE gives 46.79 V;
the same with a linear C_oss), while the tracker reads the right limit's power
at the start: + p_r·h1/2 = 1.7e-8 J per edge. O(h1), and h1 hardly moves with
rtol (1.5e-14 → 1.0e-14 from 1e-6 to 1e-9). Total heat to 5e-4 of the truth.

Fixed engine (strict xfail kept): pwl at dt = 1e-7 collapses the 0.6 ps
discharge inside one step and books none of E_oss (2.92e-5 against 4.17e-5;
the restart rule 3.01e-5): the energy is the step's numerical damping. Needs an
impulsive booking of an unresolved RC collapse; the restart rule's
ReinitRecord covers ideal devices only. The total heat agrees to 0.35 % (the
tables dominate).

## F-4-7 — control_tick='state' never fires the tick at t_end − dt

Fixed-engine observer protocol (measured): calls labelled t_0 (priming, x_0),
t_1 (x_0), …, t_N = t_end (x_{N−1}); x_N is never observed. Tick counts of a
rate group against t_end = n·dt (b9e46778), label / state:

| period, offset | n = 1 | 2 | 6 | 10 | 11 |
|---|---|---|---|---|---|
| dt, 0 | 1 / 0 | 2 / 1 | 6 / 5 | 10 / 9 | 11 / 10 |
| 2.5 dt, 0 | 1 / 0 | 1 / 1 | 3 / 2 | 4 / 4 | 5 / 4 |
| 3 dt, dt | 0 / 0 | 1 / 0 | 2 / 2 | 3 / 3 | 4 / 3 |

The END is dropped, never t0: the label path fires every tau ≤ t_{N−1} (the
graph's state time t_n − dt at the last call); the state path queues each
group instant at tau + dt (`fixed_tick_kwargs`: origin + dt, "one step late")
on run_transient's queue, whose run end is half-open (t_end − tol), so the
last servable tick, tau = t_{N−1} queued AT t_end, was cut. Fix: those
sources' Rate has `closed_end` (the queue's existing opt-in, t_end + tol),
passed through a new `tick_closed_ends` argument of `run_transient`. After:
label = state in every cell above (n = 1..11); a tick strictly inside the last
step (2.5 dt at t_end = 3 dt) is the resumed run's first on both modes, and a
run split at 3, 4, 5 or 6 dt fires the uninterrupted run's count. The count
identity is a test on pwl (tau_k ≤ t_end − dt), trbdf2 and dsed
(ceil((t_end − t0 − offset)/T), already exact there).

## F-6-1 — steady_state's default probe inside the restart engine's round-off

`steady_state(_buck, period=1e-5, dt=2e-8, grid gate)`, residual against
probe_scale (b9e46778):

| probe_scale | legacy | discontinuity_restart=True |
|---|---|---|
| 1e-6 | 1.9e-8 | 3.7e-7 |
| 1e-5 | 1.9e-9 | 1.1e-7 |
| 1e-4 (default) | 7.6e-10 | 1.9e-8 |
| 1e-3 | 9.8e-11 | 3.6e-10 |
| 1e-1 | 2.9e-14 | 2.0e-11 |

Residual ∝ 1/probe: the columns carry the period map's round-off divided by
the probe. Fix: ONE chord step on the verify run's measured residual,
h' = h* + (I − Φ)^-1 (P(h*) − h*), kept only when a second verify run measures
it smaller. It removes the columns' error whatever its source (round-off or a
smooth non-affinity) for one more period run, where the brief's two-probe rule
costs n more runs and two probe sizes cannot separate round-off (∝ 1/s) from
curvature (∝ s). After: residual at the default probe 3.6e-14 (legacy) and
2.8e-12 (restart); every probe 1e-6 .. 0.3 at 1e-11 or below. Moves on the
default path: `steady_state(dt=...)`'s history by ≤ its old residual
(≤ 1e-9), its residual / drift (to the floor), n_period_runs + 1.

## F-3-4 — trbdf2 aborts at h_min with branch-form switch and diode

On b9e46778: branch/branch aborts at 0.41 ms; the other three form pairs run
(15.31039 V; pwl dt 1e-8: 15.3095 V). On the merged tree (ecb07507):
branch/branch RUNS (15.310393 V) and branch-switch / conductance-diode ABORTS
at 0.47 ms: Newton at h_min, ||dx|| stagnated at 4.442e-8 on node T1.pm,
residual 6.1e-5 at KCL sec_neg, res/|b| = 1.1e-16 with |b| up to 5.5e11,
row backward error 67 eps at KCL sec_anode (roundoff acceptance limit 8 eps).
Trace (temporary per-attempt trace in the trbdf2 loop, removed): 53171
accepted steps, 1784 error rejects, 20 Newton rejects before the abort. Each
of the 47 diode landings (D1) is followed by an error-reject cascade whose
largest entry is branch 4 = T1.lp (the primary leakage, on the all-inductor
node T1.pm); the estimate is linear in h (the abort's landing: err 16295 at
h = 3.3e-9, 1632 at 3.3e-10, 163 at 3.3e-11, 27 at 5.4e-12, accepted at
2.0e-13): a stale derivative slot, not a stiff mode (which reads constant in
h). The min step after a landing: 1.0e-14 .. 2.5e-14 (median 8.4e-13) on this
pair, 3.2e-14 .. 4.8e-14 (median 4.1e-13) on conductance/conductance, which
survives the same collapse only because Newton still converges at 3e-14.
At 1e-14 = h_min Newton stalls (||dx|| 4.44e-8, three equal iterates; the
2L/h rows put |b| at 5.5e11): the memory's "the h_min stall is the absolute
dx test" — the stall is the symptom, the collapse is the cause.

Cause: TR-BDF2 restarts a GATE edge by committing a trapezoidal probe (fresh
derivative slots), and a DIODE turn-off by one SDIRK(2,2) step (L-stable, no
derivative slots) — but `sdirk_possible = !hs.any()`: with any stateful kind
(here the saturable transformer) the TR-BDF2 stage pair has no SDIRK step, so
the step after every diode commutation reads the pre-event slots. The error
norm with q rows (ecb07507) only moved which pair aborts.

Not fixed in this slice (strict xfail with the cause). It needs the SDIRK
restart with stateful kinds on the TR-BDF2 stage pair: the esdirk43
integrator already has it (`ed_sdirk_base` / `ed_hs_commit`: the q rows'
backward Euler from the linear entries' pseudo previous state), and runs all
four pairs to 15.31034 V in 0.34 s. Porting it lifts `sdirk_possible`'s
stateful exclusion, which also turns today's `dependent_states` refusal
(trbdf2 + an ideal-device re-initialisation + a stateful kind) into a run:
a default-path change of every trbdf2 circuit with a stateful kind and a
diode, to be measured against esdirk43 at tight tolerance. Not a refusal by
name: three of four pairs (the default conductance pair included) run today,
by the margin above. What an ideal-device circuit needs on this engine today:
no stateful kind, or integrator='esdirk43'.
