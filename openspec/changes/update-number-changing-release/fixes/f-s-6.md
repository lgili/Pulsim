# F-S-6 — a jump across a nonlinear capacitor under the restart rule

Branch `fix/core-restart-nonlinear-charge` from `origin/feat/core-integration-14t` (d9d32b88).
Marks: **[M]** measured, **[R]** read in the code.

## Functions touched (for the agents editing `run_transient.hpp` in parallel)
- `run_transient.hpp`: `consistent_start` only (the saturable / nonlinear-capacitor block that was
  `refuse_jump`, the record's dominant state, the x0+ write before the stage-1 re-solve), plus one
  scratch vector `ip_nc` next to `ip_d` / `ip_r`. Not touched: `run_restart`, the right limit
  (`event_rl_x`), closures' construction, the stepper phases.
- `models/nonlinear_capacitor.hpp`: `voltage_of_charge` (Q⁻¹, bracketed Newton) and `jump_energy`
  (∫(Q(v) − Q(v⁻)) dv: closed form for the junction law, Gauss–Legendre for the others).

## The rule (as implemented)
(filled in below with the measurements)
- Detection unchanged (the predecessor's rule, in voltage): |d| above max(|Δ2|/γ, (1+1/γ)|d'|) +
  1e-8·(1+|v0|+|v1|) + δt_loc·|i/C(v)| in both source readings. What changed is what follows a
  detection: the old code threw `UnsupportedError` ("moves nonlinear_capacitor"); now the jump is
  CARRIED. Per jumping capacitor, the stages' charges q_k = Q(v_k) (the stamps are charge-form
  Newton solves of the law, so every stage conserves the charge of a capacitor cut-set exactly):
  d_q = 2q1 − q2 − q0, a = clamp(d'_q/d_q, 0, γ/(1+γ)), J_q = d_q/(1−a)², q0⁺ = q0 + J_q,
  v0⁺ = Q⁻¹(q0⁺). The history entry (v_prev, q_prev) is set to (v0⁺, q0⁺) before stage 1 is solved
  again from x0⁺ (the same place the linear histories move). The dissipated energy per capacitor is
  ∫_{v⁻}^{v⁺}(v⁺ − v)C(v)dv = ∫(Q(v) − Q(v⁻))dv (Tellegen with the post-event voltages, the
  impulse path's final voltage → 0; ½CJ² for a linear law — the linear entries keep that code).
  Record: the state names (linear + nonlinear), `lambda_` = the dominant state's moved charge
  (C), `dW` = Σ energies.
- Refused by name (what is not solved here): a saturable inductor's jump (dual: flux conserved,
  i⁺ from λ(i) — not the same code: separate history, and the Jiles–Atherton law's λ(i) depends
  on its (H, M) path, no inverse); a nonlinear capacitor that jumps in a mode with ideal elements
  (the structural tracker re-initialises linear states only); a charge after the jump below a
  clamped floor of the law (Q flat there: no voltage reaches it).

## Case A's −0.69 V: a jump, not a ZVS artefact [M]
Trace (refusal text and the run after the fix): the refusal was at t = 0.5 µs, the LOW-side gate
turn-on, 100 ns after the high side's turn-off. By then the 10 A current-driven transition is over
(~180 ns): the node is clamped by Dl at −(V_f + I/g_on) = −0.71 V. Sl (1 mΩ) closing in parallel
with a diode that has a 0.7 V conduction drop pulls the node to −I/g_on = −0.01 V through 1 mΩ:
τ = R_on·(C(−0.4) + C(400.4)) ≈ 1.2 ps ≪ γh = 0.29 ns. That is a real impulse of the model (the
diode's V_f is the PSIM-style drop decided 2026-11-06), 0.70 V on both capacitors, not a probe
artefact of a strongly varying C(v) — the stages are charge-form Newton solves of the law, there is
no linearisation of C in them. Closed form of its energy (mpmath, Cl −0.71 → −0.01 V, Ch 400.71 →
400.01 V): 3.0676173e-10 J; the record reads 3.06762e-10 J. The ZVS turn-off at 0.4 µs (the
current-driven 31–180 ns transition) is NOT flagged (no record there).
So "no impulse record in case A" does not hold for this circuit: there are 6 records in 2.2 µs —
0.5 / 1.5 µs (real: V_f handover, 3.07e-10 J), 1.0 / 2.0 µs (real: the hard high-side turn-on,
6.28142e-5 J), and 0.9 / 1.9 µs (NOT real, see below).

### Open (pre-existing in the rule, not F-S-6): a finite-rate move that ends inside stage 1
0.9 µs is the low side's turn-off: the 10 A source moves the node −0.01 → −0.71 V in
ΔQ/I ≈ 70–87 ps, then Dl clamps it. No resistive impulse — the energy goes from the current source
into the capacitors. The probe sees a kink at θ ≪ γh after a TIME event (δt_loc = 0) with
ẋ(0−) = 0 (the capacitors carried no current before), so d ≈ the whole move reads as a jump:
3.06e-10 J recorded. The SAME circuit with linear 1 nF capacitors records the same event
(4.9e-10 J = ½·2 nF·0.7²) at the base rule — it is the rule's own limit (the post-event rate is
finite but the move ends inside the probe), only visible now that the nonlinear circuit runs.
Distinguishing it from the 0.5 µs impulse needs the move's time scale at the grid's resolution
(both are ≪ γh) — e.g. a second evaluation of ẋ(0+) under a scaled g_on (an impulse scales with
it, a source-limited move does not). Not in this slice. Size: 4.9e-6 of the hard turn-on's energy.

## Case B and the hard turn-on of case A: numbers vs the closed forms [M]
| case | record dW | regular part (window) | total | truth (suite) | rel |
|---|---|---|---|---|---|
| A hard turn-on 1.0 µs | 6.28142e-5 J | 1.62069e-8 J | 6.28304e-5 | 6.28335e-5 J | −4.9e-5 |
| B (0.7·i_crit) low-side turn-on 0.5 µs | 1.68796e-6 J | 9.8e-12 J | 1.68797e-6 | 1.68781e-6 J | +9.3e-5 |
The suite's `_window_loss` now sums E_total + E_impulse_total (the window / impulse convention:
a jump is reported by its record). The other F-S-6 cells (transition times 300/200/100 V, ZVS
fractions 1.0/0.0, the partial-ZVS trough, the resonant trajectory, the full-ZVS clamp) pass on
their unchanged bounds. The `KNOWN_LOSS pwl` xfail is the restart-OFF fixed engine: this change
does not touch it (the rule is off there), it stays an xfail; the restart rule's cell books the
energy now.

## Hand case and the junction-capacitance cases [M]
- `test_restart_nonlinear_charge.py`: C1 (1 nF, 25 V, m 0.5) at 100 V shares through a 1 mΩ switch
  with C2 (2 nF, 10 V, m 1/3) at 0 V. mpmath: v⁺ = 25.96085552 V, E = 1.946686278e-6 J. Run (split
  off and on): v⁺ = 25.96085552 V, Σq conserved to 2e-17 C over the event, one record 1.9466862586e-6 J
  (rel 1e-8), states {C1, C2}. ½C(v⁻)ΔV² at the pre-event voltages is −3 %.
- Cj (test_release_findings_a): the 5.02 µs turn-off kink — no record, no refusal; the 10 µs
  shorting — carried: Cj 24.0011933 → 0.0011981 V, record Δq −7.99904569e-9 C [closed form
  −7.99904571e-9], dW 7.46628485e-8 J [closed form 7.46628486e-8; ½·C(v⁻)·24² would be 5.76e-8].
