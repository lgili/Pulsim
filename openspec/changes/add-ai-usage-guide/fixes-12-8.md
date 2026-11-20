# Task 12.8 — the soft edge of a zero-dead-time leg

Branch `fix/core-commutation-edge-rule` from `feat/core-bindings-closeout`
01ea9c7d. Test: `python/tests/test_commutation_edge_rule.py`.

## Evidence (base 01ea9c7d)

Guide 3.7 VSI, `switch_specs` / loss port on `INV__HSa`, `INV__LSa`
(4 / 3.5 mJ at 600 V, 50 A, bilinear tables), 20-40 ms, W:

| route | dead 0 HSa / LSa | dead 1 us HSa / LSa |
|---|---|---|
| table path (`device_loss_summary` on pwl) | 22.6355 / 22.6390 | 11.2824 / 10.6846 |
| loss port pwl | 22.6355 / 22.6390 | 11.2824 / 10.6846 |
| loss port trbdf2 | 22.6535 / 22.6531 | 11.2843 / 10.6833 |
| loss port dsed | 22.6366 / 22.6383 | 11.2841 / 10.6832 |

One leg (600 V; HS dcp->a, LS a->gnd, R_on 1 mOhm, antiparallel diodes
V_f 0.7 V / R_d 50 mOhm; R 10 Ohm + L 5 mH from `a` to Vm; 10 kHz, D 0.5),
no dead time: Vm = 0 (i > 0) LS 45.14 W where the truth is 0;
Vm = 300 V (zero-mean triangle) HS 2.24 W where the truth is 1.049 W.
With 1 us dead time every engine matched the closed form already.

## Cause

`thermal::commutation_point` read `|v|` on the open side and `|i|` on the
closed side of each device's edge, independently per device. With no dead
time the complementary switch's soft edge also sees the bus (open side)
and the load current (closed side), so both edges of one commutation were
charged. Signs tell them apart: the soft switch's open-side `v` is +bus
while its closed-side `i` flows in its diode direction (negative) — a
device passing from `(v, 0)` to `(0, i)` with `v i < 0` would deliver
energy, so another device carried the commutation. With dead time the
antiparallel diode clamps it at `-V_f` first: signs agree, the edge is
charged at the diode drop (unchanged).

## Fix (one rule, four callers)

`commutation_point(turn_on, v0, i0, v1, i1, gated)` returns `soft` when
`gated && v_open * i_closed < 0`: recorded at `v = 0`, energy 0. Diodes
(`gated` false) are never soft (their turn-off is the recovery).
Callers: `LossTracker` (pwl `accept` now calls `book_`; trbdf2/dsed
events and `accept_step` via `book_`), `_commutation_points` binding
(period path; returns `(v, i, soft)`), `design_batch::switching_energy`
(had an inline copy; now calls the rule), `losses._switch_switching_loss`
(numpy mirror, `gated=`; the bit-for-bit test vs the scalar path now
draws random signs, both `gated` values).

## Truth (independent of the simulator)

Datasheet tables E(v, i) = E_ref (v / 600)(|i| / 50). Per commutation one
E_on (the device that takes the current, against the voltage it blocked),
E_off only where a channel interrupts current in its forward direction;
the complementary edge is soft (0 with no dead time; at V_f + R_d|i| with
one). The leg's edge currents: periodic solution of the piecewise R-L
circuit in closed form (`truth()` in the test), both current signs and a
zero crossing inside the period (Vm = 0 / 600 / 300 V).

Sinusoidal load (VSI): the switch carrying i forward over its half
cycle is the hard one; per switch P = f (E_on + E_off)(V/600)(I_pk/50)/pi
= 11.32 W, leg 22.65 W (I_pk from the phasor, 23.71 A).

## Results (tip)

One leg, P_sw HS / LS (W), truth from the closed form:

| Vm, td | truth HS / LS | dsed | trbdf2 | pwl = table path |
|---|---|---|---|---|
| 0, 1 us | 44.09965 / 0.16054 | 44.09878 / 0.16054 | 44.09878 / 0.16054 | 44.09931 / 0.16054 |
| 0, 100 ns | 44.91891 / 0.16561 | 44.91802 / 0.16560 | 44.91802 / 0.16560 | - |
| 0, 10 ns | 45.00085 / 0.16612 | 44.99996 / 0.16611 | 44.99996 / 0.16611 | - |
| 0, 0 | 44.84563 / 0 | 44.84698 / 0 (was 45.14) | 44.84698 / 0 (was 45.14) | 44.84751 / 0 (was 45.14) |
| 300, 1 us | 1.05201 / 1.05201 | 1.05203 / 1.05199 | 1.05204 / 1.05199 | 1.05003 / 1.04999 |
| 300, 100 ns | 1.05203 / 1.05203 | 1.05205 / 1.05201 | 1.05205 / 1.05201 | - |
| 300, 10 ns | 1.05203 / 1.05203 | 1.05205 / 1.05201 | 1.05206 / 1.05201 | - |
| 300, 0 | 1.04913 / 1.04913 | 1.04915 / 1.04911 (was 2.248) | 1.04916 / 1.04911 | 1.04715 / 1.04711 (was 2.244) |

pwl reads one 100 ns step either side (first order; 0.19 % at the 1.5 A
edges of Vm = 300). Vm = 600 (i < 0) is the mirror, in the test.

Cost (instructions per accepted step, min of six, same Release+LTO
build flags, base .so vs tip .so): port off buck_pwl 3094.3 -> 3087.2
(-0.23 %), vsi_pwl 4772.6 -> 4761.5 (-0.23 %); port on (the leg, pwl)
3294.0 -> 3298.8 (+0.15 %).

VSI 3.7, dead 0: table/pwl 11.6068 / 10.9900, trbdf2 11.6082 / 11.0035,
dsed 11.6080 / 10.9888; leg 22.60 W vs closed form 22.65 W (0.2 %:
the (E_off - E_on) x ripple term); per device equal to the classification
by the load current's sign (11.6076 / 10.9885, 3e-5). Dead 1 us: every
value byte-identical to base. The HS/LS split is not 50/50 (it is not at
1 us either): the floating neutral bends the ripple, biasing the edge
samples per half cycle.

Continuity (dsed VSI, HSa / LSa): 1 us 11.284 / 10.683, 100 ns 11.631 /
11.013, 10 ns 11.665 / 11.046, 0: 11.608 / 10.989 (was 22.64 / 22.64).
The remaining gap at td -> 0 is the diode drop: with any dead time the
hard edge commutates V + V_f + R_d|i| and the soft one V_f + R_d|i|;
the ideal zero-dead-time model has no diode interval. Bounded by
2 (V_f + R_d i_max)/V of the total (leg test).

## Moved values

Found by dumping every loss record, `loss_rows` row and table-path result
read by 29 loss / thermal / design / examples-claims / guide / doc test
files on base and tip (`.scratch/c128/lossdump.py`) and diffing: 19
records differ, all below. Everything else (incl. every dead-time result
of the guide VSI and the leg, and the guide/doc/example printed numbers)
is identical. No test assertion needed a change except the call-signature
update in `test_period_kernel_kpis.py`.

| where | old | new | truth / why |
|---|---|---|---|
| guide 3.7 VSI, `INV__HSa` P_sw, no dead time (docstring, guide text, known-limitations row removed) | 22.64 W | 11.61 W | leg closed form 22.65 W, measured leg 22.60 W |
| same, `INV__LSa` | 22.64 W | 10.99 W | load-current classification 10.9885 W |
| one leg, LS, Vm = 0, no dead time (new test) | 45.14 W | 0 | closed form 0 |
| one leg, HS, Vm = 300 V, no dead time (new test) | 2.24 W | 1.049 W | closed form 1.0491 W |
| `test_vsi_parity...` (1 us dead time, dt = 1 us): total E_on of three devices | 0.64921 / 0.65209 / 0.65689 J | 0.64484 / 0.64773 / 0.65375 J | four edges became soft: LSc at 0.381 ms (the dead time is one step, so the sample before shows HS still on: an effective zero-dead-time edge, read +600 V against a reverse 10.8 A, the double count) and three at the zero crossing (0.05 A, v -0.70 V with i after +0.058 A: the current reversed inside the step). The random table is 1-5 mJ at i = 0, so each such edge carried ~4 mJ. Parity kernel = table path still holds (1e-9). |
| same test, one device E_off | 0.59157 J | 0.58733 J | HSc turn-off at 13.551 ms, 0.05 A crossing zero (v +86 V after, i before -0.05 A) |
| `test_dsed_switch_edge_energy_reads_the_edge_itself[20]` E_on | 4.4387713e-4 J | 4.4387378e-4 J | one edge at the current's zero crossing (DCM boost) |
| same `[500]` E_on | 2.1810214e-4 J | 2.1802933e-4 J | same |
| buck_pareto example (synchronous buck, NO dead time, `Q1` specs): `Q1` E_on per period in 3 of 32 designs | 1.547e-5 / 4.327e-6 / 6.935e-6 J | 0 / 0 / 0 | the inductor current is negative at Q1's turn-on (small L): Q1 turns on at zero voltage, Q2's turn-off is the hard edge (Q2 has no spec). Moves those designs' loss / `r_th_sa` / volume. Records still equal kernel batch = Python path. |
| `test_a_capacitor_loop_impulse...` (dsed) S turn-on event | v 8.0 V, E 1e-9 J | v 0, E 0 | NOT a soft edge physically: the switch closes a capacitor loop (8 V); the after-current read at the projected right limit is -8397 A (numerical, the impulse is booked as E_impulse, unchanged). Known corner, see below. |


## Scope notes (not changed)

- Unequal dead times: each edge is classified on its own readings; a
  transition with dead time keeps the diode-drop charge, one without is
  soft/hard by sign. Same code.
- Outgoing device a separate antiparallel diode (switch that never
  carries reverse current): the switch's own edge reads ~0 current, E ~ 0
  either way; the incoming switch is hard (signs agree); the diode's
  recovery is the diode's own spec (`diode_specs` / a diode `E_off`
  table), never soft.
- Three-level leg (NPC / T-type): the rule is per device and local; an
  outgoing device whose readings disagree in sign is soft, the device
  that interrupts forward current or takes current against a blocked
  voltage is hard. Not tested here.
- Synchronous rectification (channel in reverse): zero dead time, the
  SR channel's edges are soft and the main switch's hard (the buck_pareto
  example); with dead time the body diode clamps it, charged at V_f.

## Reviewer (fresh context, opus, read-only) and what is left

Asked to construct a leg where the rule charges zero for a physically hard
edge. Found none in a well-resolved topology (AC switch pairs, matrix /
CSI reverse-blocking, NPC / T-type both current signs, overlap, one-step
dead time all classify correctly). Left open, with what each needs:

1. NOISE-FLOOR EDGES (medium-low). A hard edge at ~0 A (a DCM turn-on, a
   zero crossing) has a current whose sign is rounding noise (dsed /
   trbdf2 right limit) or already reversed (pwl, one step late); about
   half of them now read soft and lose the table's E(v, 0) intercept
   (the output-capacitance energy of a real datasheet table). With
   tables that vanish at 0 A (all shipped tests and the guide) the
   effect is nil (DCM boost test: 3e-9 J of 4.4e-4 J). Fix: soft only
   when |i_closed| exceeds the device's own off-state current g_off |v|
   — needs g_off plumbed into all four callers (tracker: the law;
   design batch: SwitchingEntry; `_commutation_points`: an argument;
   losses.py: the component's params).
2. A switch that closes a capacitor loop reads a delta current after the
   edge (-8397 A on dsed, `test_exact_loss_port`): its table E_on is now
   0 (was the table at 8 V). E_impulse (1/2 C dV^2) is unchanged.
3. An infinite reading against a finite current of opposite sign is now
   soft (v = 0) instead of refused / fallback, identically in all four
   callers.
4. User-visible: with specs on only one switch of a leg (buck_pareto's
   Q1), the hard edge of a soft commutation lands on the switch without
   a spec, so that energy is not reported at all. Correct per device;
   give both switches of a leg their tables.

## Verification run (tip)

Extension build (Release+LTO); C++ `pulsim_core_thermal_tests` (114
assertions) and `pulsim_core_dsed_tests` (41265) on Eigen 5, pass;
`test_commutation_edge_rule.py` 23 pass; the 29 loss / thermal / design /
examples-claims / guide / doc files pass; ONE full pytest from
`python/tests`: 6214 passed, 110 skipped, 8 xfailed, 1 failed
(`test_exact_route` three_phase_diode_rectifier: a repo-root-relative
YAML path; passes from the repo root, 5/5). Byte compares vs 01ea9c7d:
`pwl_identity --goldens --reference` 379/379 identical; 18 loss-port runs
(VSI + leg, dead 0 / 1 us, pwl / trbdf2 / dsed) times, states,
switch_states identical. Ruff clean; stubs regenerated, `--check` clean;
`openspec validate add-ai-usage-guide --strict` and
`add-in-kernel-event-losses --strict` valid. NOT run: full ctest, Eigen
3.4.0, g++, trbdf2_identity, the dsed bit-identity set.
