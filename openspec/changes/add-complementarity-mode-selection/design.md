## Context

Four places decide which diodes conduct, all through
`SwitchedDiode::decide` (`core/include/pulsim/models/switched_diode.hpp`)
and a re-solve loop around it:

| Site | Loop | Guard |
|---|---|---|
| fixed-step event iteration (`solver/run_transient.hpp`) | re-solve the step, re-decide | round cap, mask-cycle detector |
| TR-BDF2 settle (`solver/trbdf2_transient.hpp`) | re-solve at the event, re-decide | cascade cap |
| DSED settle (`dsed/scheduler.hpp`) | 64 rounds | chatter guard, 32 commutations in a zero-length interval |
| DC seeding (`pwl/dc_operating_point.hpp`) | re-solve, re-decide | round cap |

The value law decides each diode from its own voltage or current with
the others held. Two diodes that must switch together (the two arms of
a bridge, a freewheeling diode and the diode it relieves) see a
solution that is wrong for both until both have moved, and with
`V_th = 0` a diode at rest sits exactly on the decision boundary. The
remedy so far has been a band (`V_th > V_f`), which changes the circuit,
or, on the exact strategy only, a look-ahead along the OFF trajectory
(`add-exact-propagator-strategy` slice 1).

Owner decisions that bound this change: PSIM-style diode with a
conduction drop `V_f` (owner note *pwl-diode-vth-is-not-a-drop*); typed
refusals (`pulsim.UnsupportedError(engine, feature, device, kind)` and
the per-kind capability table); consistent initial conditions, never
the trapezoid-at-2h start; the restart rule after a discontinuity is
three consecutive Alexander SDIRK(2,2) steps (decision of 2026-11-13,
superseding the backward-Euler pair); every number-changing default
flips in one release with one re-baseline.

## Goals / Non-Goals

- Goals: one decision rule for natural commutation, shared by every
  engine and the DC operating point; a termination guarantee instead
  of caps; a named error when no consistent mode exists; no tuning
  band; a shadow period that measures disagreement before any number
  changes.
- Non-Goals: gate logic (gates stay inputs: the PWM, the controller,
  the user's `switch_fn`); devices with smooth laws (Shockley diodes,
  MOSFET channels — the Newton segments of `add-exact-propagator-strategy`
  keep them); thyristor latching and reverse recovery as part of the
  LCP (they are state machines layered on top: a thyristor is a diode
  port whose OFF→ON is additionally gated); friction-type set-valued
  laws other than the diode.

## Decisions

### D1. The PWL diode as a complementarity element

```
              i_d →
   a ──[ ideal ]──[ V_f ]──[ R_on ]── k          plus g_off from a to k
        0 ≤ i_d  ⟂  w = V_f + R_on·i_d − v_ak ≥ 0
```

The ideal element is either conducting (`w = 0`, `i_d ≥ 0`) or
blocking (`i_d = 0`, `w ≥ 0`). The terminal current is
`i = i_d + g_off·v_ak`.

Relation to today's conductance law (`i_ON = g_on·(v − V_f)`,
`i_OFF = g_off·v`): with `R_on = 1/g_on` the ON branch here is
`g_on·(v − V_f) + g_off·v`. The difference is `g_off·v`, 7e-10 A at
0.7 V with the default leak, and it removes a jump of `g_off·V_f` that
today's law has at `v = V_f`. Today's law is not monotone across that
jump; the new one is. This is the only change for conductance-form
diodes and it is **BREAKING** (rounding-level) at the default flip.

The turn-on threshold: with the LCP the effective turn-on voltage of a
diode without hysteresis is `V_f` (`w = 0` at `i_d = 0`).
`V_th > V_f` keeps a meaning — hysteresis, D8. `V_th < V_f`, including
the negative thresholds that today make a diode with `V_f = 0` conduct
early, has none; such a value is accepted and ignored with a
deprecation warning during the legacy period and refused afterwards
(`UnsupportedError(engine, 'V_th', device, 'diode')`).

### D2. The LCP: ports of the companion network at the event step

At an event at time `t` the state `x(t⁻)` (capacitor voltages,
inductor currents) is known. The question "which diodes conduct right
after `t`" is answered on the backward-Euler companion network of a
step `h_e`: each capacitor becomes `C/h_e` in parallel with a current
source `C·v(t⁻)/h_e`, each inductor `L/h_e` in series with a voltage
source `L·i(t⁻)/h_e`, sources take their values at `t⁺`, gates their
values at `t⁺`. Every diode's ideal element is cut out and becomes a
port. The remaining network is linear and resistive, so between the
port currents `i_d` and the port voltages there is an affine relation
whose Schur complement form is

```
w = M·i_d + q,     M = diag(R_on) + Eᵀ·K⁻¹·E,     q = V_f − Eᵀ·K⁻¹·b
```

with `K` the MNA matrix of the companion network with every port open,
`E` the port incidence and `b` the right-hand side holding the state.
This is the rational complementarity problem of linear complementarity
systems at `σ = 1/h_e` (Heemels, Camlibel, Schumacher; general
knowledge): for small `h_e` its solution is the mode the trajectory
takes on `(t, t + ε)`, including the impulsive ones.

`K` does not exist when a port is the only path (two diodes in series
with nothing between them, an inductor whose only path is a diode). The
prototype therefore writes the relation in **hybrid coordinates**: for
each port either the current or the voltage is the independent
variable, chosen by a pivoted greedy basis that prefers the previous
mode (an ON diode is current-controlled, an OFF diode
voltage-controlled). In the previous mode's coordinates, `K` is the
companion matrix of the previous mode itself, which the factor cache
already holds (per mask and `dt`, `pwl/cache.hpp`); the LCP in other
coordinates is reached by principal pivots, one per Lemke step (D4).

The DC operating point is the same LCP with the `σ = 0` port model
(capacitors open, inductors short).

Measured cost of the dense prototype is irrelevant (numpy); what
matters is the count of factor solves, D9.

### D3. Passive networks: PSD, copositive-plus, a solution exists

For a network of resistors `R ≥ 0`, capacitors, inductors and coupled
inductors with a positive-semidefinite inductance matrix, independent
sources and ideal switches, the companion network is a resistive
passive network, so `i_dᵀ·(M·i_d) ≥ 0` for every `i_d`: `M` is positive
semidefinite in the non-symmetric sense. A principal pivot transform
(the change to hybrid coordinates) preserves the product `⟨i, w⟩` of
the exchanged pairs and therefore preserves this property. Consequences
(standard LCP theory; Cottle, Pang, Stone):

- PSD matrices are copositive-plus, so Lemke's method with a
  lexicographic ratio test terminates either at a solution or at a
  secondary ray, and for a copositive-plus matrix a secondary ray
  proves that no solution exists (no consistent mode: the event is a
  modelling error, reported by name).
- With every `R_on > 0` the matrix is positive definite, a P-matrix,
  and the solution is unique.
- With `R_on = 0` the port currents can be non-unique (two ideal
  diodes in parallel); the port voltages are not affected, and the
  state trajectory is not affected; the kernel splits the current by
  the minimum-norm rule and reports the split as non-unique in the
  event record.

**Non-passive networks**: controlled sources with gain, negative
resistors, an op-amp macro-model. `M` is then not certified, Lemke can
end at a secondary ray that proves nothing, and the solution can be
non-unique in the voltages. The kernel classifies a circuit as
passive structurally at build time (device kinds and parameter signs).
For a non-passive circuit it uses Lemke first and, on a secondary ray,
enumerates the `2^k` modes for `k ≤ 12` (4096 solves, each one
back-substitution in the event's factor); above 12 it raises
`UnsupportedError(engine, 'mode_selection', device, 'non_passive_ports')`
naming the controlled sources that make it non-passive. Enumeration is
also the test oracle for every `k ≤ 12` circuit in the suite.

### D4. Solver: Lemke with the lexicographic rule

Lemke's complementary pivoting with covering vector `e = 1` and the
lexicographic minimum ratio test (`(b_i, B⁻¹_i·)` compared
lexicographically). Lexicographic degeneracy resolution is what makes
the termination argument hold without perturbing `q`. Tested on random
LCPs in the prototype: every PSD instance solved; the only secondary
rays were on non-PSD instances.

Warm start: the initial basis is the previous mode. If `q ≥ −tol` in
the previous mode's coordinates (the previous mode is still
consistent), the answer is the previous mode at zero pivots, which is
the common case: 8001 of 11 980 LCPs of the DCM buck and 40 000 of
40 180 of the CCM boost (`probes/lcp_cases.out`).

Tolerances are absolute, at circuit scale: `vscale` (largest source
voltage magnitude, 1 V floor), `iscale = vscale/zscale` with `zscale`
the impedance scale of the circuit; `tol_w = 1e-9·vscale`,
`tol_i = 1e-9·iscale + tol_w·σ·C_max` (the current a voltage error of
`tol_w` drives through the largest companion capacitor). Relative
tolerances failed in the prototype (ties at rest were decided by
rounding).

### D5. Ties and the choice of `h_e`

A tie is a port with `i_d ≤ tol_i` and `w ≤ tol_w`: the first-order
answer does not decide. The prototype re-solves the same LCP at
`100·h_e` and then `1e4·h_e`; the larger step carries the next terms of
the RCP's expansion (the derivative of the gap), which decide whether
the port moves into conduction. A port still tied after `1e4·h_e` is
taken OFF (no current flows, both modes are consistent; this happened
only at the rest state and at series-pair turn-offs).
`h_e = 1e-6·T_ref`, `T_ref` the shortest declared period (PWM, source)
or `t_end`. Measured ties: 15 of 40 events in the single-phase bridge
(the turn-off of a series pair at the same instant), 18 of 60 in the
three-phase bridge, 0 in the DCM buck, 1 in the boost.

The choice of `h_e` is the weakest point of the design: too small and
rounding decides, too large and a mode lasting less than `h_e` is
missed. The widening is the fallback; task 3 measures the band of
`h_e` over which every family gives the same modes, and the default is
set from that measurement, not from this estimate.

### D6. One event, in order

1. Gates and sources take their `t⁺` values (D7).
2. One LCP (D2–D5) gives the diode mode.
3. If the mode makes states dependent (an inductor cut-set the new mode
   opens, a capacitor loop it closes), the re-initialisation service of
   `add-branch-form-ideal-switch` projects the state and records the
   impulse.
4. Consistency check on the exact (or the first implicit) step: every
   ON port has `i_d ≥ −tol_i` and every OFF port `w ≥ −tol_w` right
   after `t`. A violation is a named error with the LCP data, not a
   second round: the LCP answer is final.

There is no iteration over steps 2–4. Simultaneous events (a gate edge
and a diode crossing at the same instant, three diodes crossing at
once) are one LCP: they share `t`, so they share the problem.

Zero-duration modes cannot be selected: a mode that lasts zero time
fails step 4 at the next event at the same `t`. The kernel counts
consecutive events at the same `t`; two LCPs at the same `t` returning
different modes for the same gate mask is reported as a zero-time loop
error (the prototype counter `immediate`; 0 over all families).

### D7. Gate-forced switches, the ideal switch, impulses

- A gated switch is not a complementarity variable: its gate bit fixes
  its port (`v = R_on·i + V_f` ON, `i = g_off·v` OFF). In the LCP it is
  part of `K`.
- A switch with an antiparallel diode is two branches: the gated switch
  and a diode port. A MOSFET with body diode in PWL form is that pair.
- The ideal switch (`R_on = 0`, `g_off = 0`) is the branch form of
  `add-branch-form-ideal-switch`; the LCP is unchanged, `K` just has
  the branch rows.
- A gate that opens an inductor current with no diode path leaves the
  LCP with no mode that conserves flux; with the BE companion the LCP
  still has a solution (the inductor's companion is a resistance
  `L/h_e`), and step 3 detects the cut-set and records the impulse; the
  `on_impulse` policy of `add-branch-form-ideal-switch` decides whether
  the run continues.
- Impulses chosen by the diodes themselves (a bridge closing on an
  uncharged capacitor) are recorded the same way: prototype (v-b),
  14.758493 J at t = 0.

### D8. Hysteresis `V_th > V_f`

Not prototyped. A diode that is OFF with `V_th > V_f` is held OFF in
the LCP (its port fixed OFF) while its open-circuit voltage is below
`V_th`; an event is scheduled at `v = V_th` and the LCP is posed again
there with the port free. A diode that is ON stays a free port, so
turn-off is at `i_d = 0` as for every diode. This reproduces today's
meaning of a threshold above the drop. Task 6 adds the test before the
behaviour is claimed.

### D9. Cost per event

Per event: one solve with the cached factor of the previous mode gives
`q`. If the previous mode is consistent, that is all. Otherwise each
Lemke pivot needs one column of `M` in the current coordinates (one
more solve with the same factor, or a rank-one-updated factor after a
principal pivot) and O(k) dense work on the tableau. Measured pivots
per event (first LCP; with tie re-solves in brackets):

| Family | max | mean |
|---|---|---|
| buck at rest | 2 | 2.00 |
| DCM buck, 4000 periods | 2 | 0.66 |
| boost CCM / DCM | 2 / 2 | 0.01 / 0.66 |
| single-phase bridge | 6 (15) | 3.61 (7.07) |
| three-phase, overlap | 4 (8) | 2.84 (4.14) |
| three-phase, C filter | 5 (15) | 3.03 (5.90) |

Today's settle on the same events costs one full re-solve per round
and up to 64 rounds. An estimate, not a claim: in C++ a pivot costs
about one back-substitution of the event's factor (sub-microsecond on
a 20-unknown converter); task 8 measures it against the settle rounds
on the same events.

### D10. Floating islands are a gauge

With ideal diodes OFF, a sub-network that only diodes and open
switches connect to the rest (the DC side of a rectifier between
conduction intervals) has an undefined potential: the dynamics do not
depend on it and the diode gaps constrain it only to an interval.
Today the leak picks a value and the engines disagree (1.9e-10 V on
`trbdf2`, 47.586 V on `pwl` and `dsed`, same `v_C = 95.1711 V`;
`probes/today_dsed.py`). The kernel:

- detects the gauge directions per mode (the null space of the mode's
  constraint Jacobian that moves no state; prototype `Mode`);
- uses gauge-free event predicates: the gap of a diode on the island is
  combined with the gap of a diode on the opposite side of it (pairwise
  sums with opposite gauge sign), so a crossing is located on a
  quantity the dynamics determine;
- reports the island's potential at the centre of the interval its
  OFF diodes allow, and marks it in `result.diagnostics['gauge_nodes']`.

This supersedes "a floating node is always an error" of the earlier
`add-branch-form-ideal-switch` D3 for islands bounded by diodes; an
island bounded only by open gated switches is still a floating-node
error (its potential has no interval).

### D11. One service, every engine

```cpp
struct ModeSelection {
    SwitchStateMask mode;      // diode bits
    int pivots, ties, widenings;
    bool needs_reinit;         // dependent states in the new mode
    bool nonunique_currents;
};
ModeSelection ModeSelector::select(const SwitchStateMask& gates,
                                   std::span<const Real> held_state,
                                   const SwitchStateMask& previous,
                                   Real t);
```

Callers: the four sites of the Context table, and
`periodic_steady_state` (the saltation matrix at a diode event uses the
selected mode). After Phase 3 it is the decision rule behind
`ModeSet::redecide` of `add-unified-event-queue` D6.

### D12. Migration

| Stage | Option | Numbers change? |
|---|---|---|
| 1 shadow | `mode_selection='shadow'`: today's settle decides; the LCP runs on the same held state; disagreements in `result.diagnostics['mode_shadow']` (`t`, settle mode, LCP mode, settle rounds, pivots) | no |
| 2 opt-in | `mode_selection='lcp'` on every engine | yes, for that run |
| 3 default on the exact strategy | `dsed` / `engine='auto'` exact route | yes (in the single number-changing release) |
| 4 default everywhere | `pwl`, `trbdf2`, DC | yes (same release) |
| 5 removal | the settle loops, caps and chatter guards are deleted; `mode_selection='legacy'` warns for one release, then is refused | — |

Stage 1 runs on the conductance form (`R_on = 1/g_on`), where `M` is a
P-matrix: the shadow comparison needs nothing from the branch form.
Stages 3 and 4 flip together with the other number-changing defaults
(roadmap decision of 2026-11-13) and are re-baselined through the
reference harness.

## Risks / Trade-offs

- `h_e` (D5) → measured band per family before the default is set; the
  widening as fallback; a tie still unresolved is OFF and counted.
- Non-passive ports → enumeration to `k = 12`, a named refusal above;
  the passive classification is structural and conservative.
- Many diodes (a 200-diode rectifier bank, an MMC): Lemke is
  exponential in the worst case and roughly linear in practice (general
  knowledge); the counter `max_pivots_per_event` is in the result, and
  task 8 measures a 60-diode bank.
- Gauge convention (D10) → reported, documented, marked; users who
  probe a floating DC bus see a different number than today.
- Shadow disagreements may be many on circuits that today run only
  thanks to a band → the shadow report is the evidence for each one;
  none is silently accepted.
- Hysteresis not prototyped (D8) → a test precedes the claim.

## Open Questions

None for the owner. Technical points to settle by measurement: the
default `h_e` (task 3), the enumeration limit (task 5), whether the
lazy-column implementation needs a rank-one update of the factor or a
dense `k × k` tableau suffices below some `k` (task 8).
