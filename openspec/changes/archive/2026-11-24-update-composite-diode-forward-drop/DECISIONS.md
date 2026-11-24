# Decisions taken during implementation (2026-11-13)

The design's Open Questions were unanswered; each takes the option marked
"proposed", else the most conservative one.

| # | Question | Option taken | Why | How to reverse |
|---|---|---|---|---|
| OQ1 | Keyword name `V_F` or `V_f` | `V_F` kept, only its meaning changes | Proposed; a rename breaks every keyword call for no behaviour | Add `V_f` to the four builders with `V_F` as a deprecated alias |
| OQ2 | `V_th` keyword on the four builders | Not added | Proposed ("not proposed"), and task 1.9 measured the case that would need it: an H-bridge of four `add_mosfet` with 200 ns dead time settles on `dsed` (rk45, 2402 steps over 200 periods, body diodes conducting at the 0.7 V drop), no "did not settle" error. `add_diode(V_th=, V_f=)` beside `add_switch` covers a band | Add `V_th=0.0` after `V_F` in the builders, loader and bindings, pass it to `pool_.add_diode` |
| OQ3 | Level-1 diode default | 0.7 V for both the level-1 MOSFET body diode and the IGBT co-pack diode | Proposed | Change the `V_F` defaults of `add_mosfet_level1` / `add_igbt_level1` (header, bindings, loader) |
| OQ4 | `add_mosfet` default `V_F` | Kept at 0.7 V (every default call now has a 0.7 V drop) | Proposed | Default `V_F = 0` in `add_mosfet` |
| OQ5 | Slope `R_d` keyword | Not added; `g_on_diode` stays the slope (`add_mosfet_with_body_diode` only) | Not proposed; the conservative choice adds no API | Add `R_d` mapped to `g_on = 1/R_d` |
| OQ6 | Release | Not decided here: CHANGELOG `[Unreleased]`, marked BREAKING | No option marked proposed; release timing is the owner's | — |

Implementation choices not covered by the design:

1. **Validation only where a diode is built.** `add_mosfet(body_diode=False)`,
   `add_mosfet_level1(with_body_diode=False)` and `add_igbt_level1(with_fwd=False)`
   ignore `V_F`, so a bad value there is not refused. The error names the
   builder, the device and `V_F` (`add_mosfet` reports through
   `add_mosfet_with_body_diode`, which it calls).
2. **YAML.** `mosfet` gains `V_F`; `mosfet_level1` gains `with_body_diode`
   and `V_F`; `igbt_level1` gains `v_knee`, `with_fwd` and `V_F` (`V_F`
   would be inert without `with_fwd`, and the positional call needs
   `v_knee`). Tail parameters stay out of YAML.
3. **Task 1.8's loss figure.** The tasks quote `P(QL_body) = 3.479 W`
   "time-weighted over the last 100 periods". 3.479 W is the 0–400-period
   average (start-up included, measured 3.4792 W). Over the last 100
   periods it is 3.4422 W, which matches `(V_F·I_L + I_L²/g_on)/2` =
   3.44214 W at the measured `I_L` = 9.700 A. The test asserts the window
   the task names, 3.442 W ± 0.01 W, and the analytic value within
   0.2 %; the spec scenario and the proposal table are corrected to match.
4. **Task 1.2's loss figure.** The test asserts `P(Q)` = 0.1 W within 1 %
   (10 A through 1 mΩ) instead of the quoted 0.0993 W.
