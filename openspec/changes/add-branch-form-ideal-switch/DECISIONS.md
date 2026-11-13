# Decisions taken while implementing (feat/core-a-ideal-switch)

Each entry: question → option taken → evidence → how to reverse.

1. **`g_off = 0` in the conductance form.** Design "Decisions taken" #2
   says refuse it until the flip; tasks 2.1 / 1.1 say `g_off` finite and
   `>= 0`. → Kept legal (tasks 2.1). → Refusing it would remove the
   characterisation of task 1.1 (±396.8 V on `pwl`, 2.35e9 V on
   `trbdf2`), which is the measured reason the branch form exists, and
   would change no valid circuit's numbers either way; the branch form
   is the remedy the refusal would point to, and it is opt-in in this
   slice. → Reverse: one line in `CircuitBuilder::require_conductances_`
   (`g_off > 0`) and flip `test_goff_zero_*` to expect the refusal.

2. **Where the branch form lives in the capability table.** A row per
   `StoredKind` (D9) would need new kinds, and the branch form is a form
   of the Switch and Diode kinds (same mask bit, same parameters). →
   A separate table `pwl::kBranchFormCaps` ("Switch": pwl, trbdf2;
   "Diode": pwl), checked by `require_branch_form_supported` first in
   `require_supported`, by `run_transient_bdf1` and the kernel AC sweep;
   bound as `_pulsim._branch_form_capabilities()`, mirrored in a second
   docs block and checked by `test_engine_conformance.py` (one cell per
   form and engine; no silent cell). → The diode is pwl-only because its
   decisions are re-read through the solved current only in the fixed
   engine's sites (`DiodeEventState`, the split's crossings);
   `trbdf2_transient.hpp` is not ours to edit. → Reverse: add the engine
   bit to the row once its decision sites read `i_s`.

3. **Branch-form diode decisions.** The voltage law `decide(on, v)` reads
   `i = g_on·(v − V_f)` when ON. → An ON branch-form diode is entered in
   `DiodeEventState` with `g_on = 1` and reads `v = V_f + i_s`
   (`DiodeEntry::law_voltage`), so `decide` and `crossing_signal` get
   `i_s` (amperes) and every decision site stays the same code; OFF reads
   `v_a − v_k` as before. → Conductance-form entries are untouched
   (bit-identical: golden probe). → Reverse: a dedicated
   `decide_branch(i_s, w)` in `models::SwitchedDiode`.

4. **Islands bounded by OFF ideal diodes: refused, not a gauge (task
   4.2).** The design's gauge (complementarity D10) reports the potential
   at the centre of the interval the diodes allow. That is a per-mode
   replacement of the island's rows, which breaks D1's mask-invariant
   pattern (one symbolic analysis), and the centre equation couples the
   diode thresholds into the matrix. → `floating_node` with `kind =
   'Diode'` when any boundary element is a diode, naming every boundary
   element and the nodes; the message says the gauge is not implemented
   and gives the remedies (`g_off > 0`, a component to ground). With
   `g_off > 0` (the default 1e-9 in every test circuit) no island exists.
   → Reverse: implement D10 as a per-mode gauge row on a reserved
   structural slot (keeps the pattern), then drop the refusal for
   diode-bounded islands.

5. **TR-BDF2 dependent states (task 5.5 refused).** The per-mode pass
   needs a hook in the TR-BDF2 step loop (`trbdf2_transient.hpp`, not
   ours). → A static over-approximation at entry: any branch-form element
   with `g_off = 0`, or with `R_on = 0` closing a loop of voltage sources,
   capacitors or other `R_on = 0` elements, refuses with
   `dependent_states`. A circuit it refuses might never visit the
   dependent mode; none it accepts can. → Reverse: run
   `BranchStructureChecker` per mode inside the TR-BDF2 loop.

6. **One structural pass per mode, memoised.** → `BranchStructureChecker`
   keeps a map mask → result and a last-mask fast path; it is inactive
   (zero cost, zero checks) without an ideal branch element, so every
   conductance-form circuit skips it; `counters['mode_structure_checks']`
   counts the passes (4 for two ideal switches visiting their 4 masks,
   however often). → Reverse: drop the memo (the pass is O(branches)).

7. **Rank-one row update vs one factor per mask (task 3.2), by
   measurement.** Single-flip walk of 4000 masks on a 12-bit
   three-phase inverter in branch form (n = 24), best of 5, A/B/A/B
   (`[rank1_measure]` in `core/tests/layer4_v2/test_branch_form_switch.cpp`):
   the engines' per-mask factor cache 206/206 ns per solve (a hit after
   the first visit of each mask); `solve_rank1` 2113/2113 ns, and WRONG
   for the branch form: its path refactor keeps the previous pivot order
   and a flip swaps `(s, a)` between 1 and −g_off, so it lands on a
   g_off-sized pivot (`|x − x_ref| = 1.3e4` against 1.3e-8 allowed on the
   16-mask bridge; 5859 of 20000 path attempts fell back). → The per-mask
   cache stays the engines' path (it already holds one factor per visited
   mask on the one shared analysis); `solve_rank1` refactors in full when
   the circuit has a branch-form element (2195/2259 ns, equal to the
   per-mask factor within 1e-12). The conductance form keeps the path
   refactor. Dense partial-pivot growth max|U|/max|A| over the 16 masks:
   1. Same numbers with Eigen 3.4.0 (206/206 vs 2157/2209 ns, growth 1).
   → Reverse: a path refactor that re-pivots inside the path.

8. **Residual of task 3.3 on an all-OFF bridge.** With every switch open
   the bridge's inner nodes are tied by `g_off = 1e-9` alone; a 1–2 A
   injection into every row gives `‖Jx − b‖/‖b‖ = 8e-8` by conditioning,
   not by the factor. → The test checks the circuit's own right-hand side
   (`‖Jx − b‖/‖b‖ ≤ 1e-12`, measured ≤ 1.1e-15) and the normwise backward
   error under the injection (≤ 1e-15, measured ≤ 1.7e-18). Task 3.4's
   bridge rectifier is compared on its bus `v(p) − v(n)`: with every
   diode OFF its common mode is set through `Rg = 1 MΩ` by currents that
   cancel to the solver residual (7.6e-12 A in both forms; the gap scales
   with `Rg`: 7.6e-6 V at 1 MΩ, 7.8e-9 V at 1 kΩ), differential mode
   1.3e-11 V. → Reverse: none needed; the absolute node is ill-posed.
