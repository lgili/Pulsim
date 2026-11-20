# Group C — the new mode selection (`mode_selection='auto'`)

Branch `fix/core-release-c-mode-selection` on the PR head 27c00520. Findings F-2-3, F-5-9, F-4-1,
F-3-2, F-4-2, F-4-3, F-5-3, and the F-3-3 flyback side observation. Reproductions and
regression tests: `python/tests/test_release_findings_c.py`. Marks: **[M]** measured, **[R]** read
in the code, **[H]** hypothesis not shown by a trace.

`'auto'` resolves in `_resolve_mode_auto` (python/pulsim/__init__.py): `'lcp'` on pwl (unless a gated /
Newton device or `max_event_iterations=0`) and on the exact engine (any PWL diode); trbdf2 keeps the
settle. Every finding below reproduces with `mode_selection='auto'` alone (no other release option).

## F-2-3 / F-5-9 — a bridge diode held OFF while forward-biased (pwl, 'lcp') [M] FIXED

Reproduction (dt = 1 µs, 60 ms): grounded-DC bridge at V_th = 0, 'auto': 23 samples, BR__D1 OFF up to
1.64 mV forward; floating-DC bridge at V_th = 0.7: 2 samples, up to 13.5 mV. Legacy: none.

Trace (grounded bridge, AC side floats when all four diodes are OFF), samples around the peak:

| t (s) | bits | v_s | v_b | v_p | v_a − v_p |
|---|---|---|---|---|---|
| 0.005024 | 1000 | 324.990826 | +6.4e-5 | 324.990826 | 5.7e-14 |
| 0.005025 | 0000 | 324.991774 | **+1.7978e-3** | 324.990135 | **+1.64e-3** |
| 0.005026 | 0000 | 324.991019 | +1.8610e-3 | 324.989444 | +1.58e-3 |

With all four OFF the island {s, a, b} is held only by four 1e-9 S leaks; its KCL gives
v_a + v_b = v_p, i.e. v_b = (v_p − Vac)/2 = +0.08 mV and v_a − v_p = −0.08 mV (reverse: OFF is right).
The stored v_b is 1.8 mV — the factor's rounding of the island's common mode
(ε·internal current / G_boundary, DECISIONS F1 / #30 of add-complementarity-mode-selection). The LCP
decides on the island-corrected solve (`PwlModePorts::site` / `correct`), so its answer (OFF) is
complementary on the network it solved; the STEP then stored the raw, uncorrected solve
(`mode_lcp_site`: `x = *x_start` / `solve(mask, x)`), on which the decided-OFF diode reads forward-biased.
F-5-9 is the same on the floating DC bus (its common mode jumps 29 mV between two all-OFF samples).
Neither a stale RHS, nor a different step network, nor a hysteresis override: the decided mode was
right, the stored state was not the state it was decided on.

Legacy has the same rounding (v_b = 1.8 mV at 5.025 ms too) but its settle reads the raw solve,
turns D1 back ON on the noise and chatters ON/OFF (bits 1000/0000/1000/1000/0000 …): no stored
violation because the noise decides. Not changed (default path).

Fix: `mode_lcp_site` imposes the islands' KCL on the state it stores (`mode_ports->correct(mask,
for_dt, bx, x)`, the same correction the site's port problem and quiet test use) on all three exits
(quiet, cached, decided). 'lcp' only; no-op without an island.

Truth: the closed form v_a + v_b = v_p on every all-OFF sample of the grounded bridge:
max |v_a + v_b − v_p| ≤ 3.25e-7 V (1e-9·325 V) after; 1.6e-3 V before. Both bridges: no
forward-biased OFF sample (the loss summary's rule check is silent).

## F-3-2 — a refusal's `.device` is an internal label [M] FIXED

`run_transient.hpp` / `trbdf2_transient.hpp` kept one vector of `pwl::branch_label` ('D (branch #1)') for
both the message and `.device`. Now a second vector of `graph.branch_name` feeds `.device` (the
`ideal_voltage_loop` refusal of the shorted port and `raise_if_refused`'s new `names` argument);
the label stays in the message. C++ test in `test_mode_selector.cpp` ("F-3-2" section), Python
`test_a_refusal_device_is_the_user_name`. Group G parks the same test as a strict xfail
(`test_release_findings_g.py::test_ideal_voltage_loop_refusal_names_the_device[lcp]`): it XPASSes
after this branch merges — remove the mark at integration.

## F-4-1 — an ideal voltage loop refused as `consistent_mode` under 'auto' (exact engine) [M] FIXED

pwl's LCP path already classified a failed selection (`mode_lcp_site`: a shorted ideal port, or the
mode with every not-reverse-biased ideal port closed run through the structure check). The exact
engine had no such step: Lemke ends on a secondary ray with the ideal diode posed OFF and
`raise_if_refused` says `consistent_mode` with the "add the missing path" advice; the settle names
`ideal_voltage_loop` because its extraction of the ON mode throws it (`rejected_loop_`).

Fix (one rule, both engines): `ModeSelector::closure_candidate(ports, base, tol_w)` — the base mode
with every ideal port whose first-pass posed gap q_j < tol_w closed. pwl's second classification
block now calls it (it built the same set from u − V_f inline); the exact engine's `decide_modes_`
classifies a `consistent_mode` refusal on that candidate (`classify_refusal_`: the extraction of the
candidate mode; a candidate refusal — `ideal_voltage_loop` / `source_capacitor_loop` — becomes the
refusal's name, as `rejected_loop_` already did for the settle). pwl's LCP solves now go through the
settle's structural pass first, so `floating_node` is named there too (it was a singular factor and
an abort after 64 sub-steps).

## F-4-2 — PSFB, branch form R_on = 1e-6, refused at 339 µs under 'auto' [M] FIXED

Truth: conductance twin at g_on = 1e6, 24.071424 V; the settle (branch) 24.071411 V; at R_on = 1e-3
both selectors agree (24.066152 / 24.066152). Trace of the refusal (env-guarded print of the first
pass, not committed), t = 339.0278 µs, prev / anchor mode 00010110 (D_LS_B, D2, D3 ON), k = 8,
h_e = 2.5e-13, Lemke status = secondary ray. The hybrid-coordinate matrix H has

    H(5,5) = +4.66e-9,  H(6,6) = -1.28e-7  (D2, D3: ON ports, R_on = 1e-6)
    H(4,4) = H(7,7) = 1.74e6               (D1, D4: OFF)
    q(D2) = -1.06e-7 A, q(D3) = -9.6e-8 A  (the D2/D3 pair at its zero-current turn-off)

A negative diagonal: H is not P0 / copositive-plus, so the ray proves nothing. Cause: an ON
port's hybrid row is A − B·(I+ZΓ)⁻¹Z·Da with A = B = Da = 1/R_on, i.e. 1/R_on − (1/R_on)²·Z_jj; for
a series pair behind 1e9 Ω of leaks Z_jj ≈ R_on and the result (~1e-9 S) is a difference of two
1e6 S numbers through a solve of condition ~1e15: rounding. At R_on = 1e-3 the same cancellation
leaves ~1e-4 relative error, which is why the shipped g_on = 1e3 runs. Not a degenerate tie rule,
not the gauge: the first-pass port matrix itself is wrong in sign.

Fix (selector, one place): before a first-pass failure (secondary ray / pivot cap) is believed, an
anchored selector (the exact engine's) poses the LCP again with every port in OFF coordinates (z =
i, H = Z_open + R_on: no 1/R_on term, no cancellation) — the anchor the ideal ports already use —
and solves that. Only a run that would have been refused takes the new path (counter
`ModeSelection::recoordinated`). The fixed engine's selector is not anchored (it poses its site
once): not covered there; no fixed-engine refusal of this kind was found.

## F-5-3 — a 1 ps gate pulse dropped by the exact engine under 'auto' [M] FIXED

Not a merged simultaneity: both edges are events (799 event times on 400 periods, the falling edge
at k·T + 1e-12 s on both selectors). Trace (`diagnostics['mode_selection']`, 3 periods): h_e = 1e-12,
3 ties, 3 pin judgements, 0 leak pins. At the falling edge the inductor carries 2.4e-7 A
(24 V · 1 ps / 100 µH); the stored samples show i_L 2.4e-7 A at the edge and 2.4e-8 A (= 24 V · g_off,
the switch's leak) at the next — the pulse's current is gone, v_sw ≈ 0.
- The selector's current tolerance at h_e is tol_i = tol·iscale + tol_w·C_max/h_e = 1.1 A (C_max =
  47 µF, h_e = 1 ps); at 1e4·h_e still 1.1e-4 A. 2.4e-7 A is a tie at every pass.
- The tie pins (removing it opens a gap i·R_jj ≫ tol_w), so the judge arbitrates: D1 OFF is judged
  on the exact trajectory at 100·h_e and 1e4·h_e only, after the 0.2 ps g_off transient that
  dissipates the current (τ = L·g_off): "consistent", and D1 goes OFF. The settle reads v_D at t+
  (D1 OFF: v_sw = −i_L/(2 g_off) = −120 V, forward) and turns D1 ON.
- Threshold measured (100 periods, v_out 'auto'/legacy): D = 1e-7 0.010, 1e-6 0.0010, 3e-6 3.4e-4,
  1e-5 1.0e-4, 3e-5 3.5e-5 (every pulse up to 300 ps lost: its current is under the tie band);
  D = 1e-4 (1 ns, 2.4e-6 A … the band's widened value): identical.
Fix: the pin judge (the selector's `judge` on the exact engine, `run_selector_`) first checks the
candidate at t itself (`off_forward_at_`: an OFF diode of the candidate breaking its law on the held
state, the right limit before any parasitic transient); a candidate that does is inconsistent and
the pins go ON. Only the pin arbitration changes (the guard's judge and the horizon are untouched).

## Side observation — F-3-3 flyback on the exact engine with the LEGACY selector [M]

`_flyback(False, 1.0)` of test_saturable_transformer, 100 kHz D = 0.5, `engine='dsed'`, 1 ms:
legacy (today's default) peak i_p 2.0614 A, v_out(0.8–1 ms) 0.000 V (also at rtol 1e-9); 'auto'
v_out 16.357 V. Independent truths: the restart fixed step on group A's branch (peak 16.38199 A at
dt 10/5/2.5 ns, v_out 16.365 V) and bucket 3's exact-engine value 16.365 V. Every period legacy shows
i_p 2.06 A at the gate-off and 9.6e-8 A at the next sample, the secondary current 0: the
magnetising energy is lost at each turn-off. Mechanism [H, partly read]: at t+ with D1 OFF the
secondary current is a state at 0, so v_D1 = 0 — the settle keeps D1 OFF; the forward bias exists
only inside the 1e-14 s R_off transient (L_p / 1e9 Ω) in which the coupling would carry the current
over, and the guard's judge (horizons past the parasitic transient) confirms the mode (5 checks, 0
repairs). The LCP poses the BE companion over h_e, where the coupling is visible, and turns D1 ON.
A default-path defect the release's flip to 'auto' fixes; not changed here (not a one-line fix).
Test: `test_the_flyback_secondary_conducts_on_the_exact_engine` ('auto' passes, 'legacy' strict xfail).

## F-4-3 — ideal PSFB with C_oss / C_j: D1 left ON at −2.2 A under 'auto' [M] FIXED

Truth: the settle runs to 30 µs, v_out(27–30 µs) 1.0358473 V; the conductance twins at R_on = 1e-3 /
1e-4 give 1.0357080 / 1.0358334 V on both selectors (Δ per decade 1.25e-4 → 1.25e-5: limit
≈ 1.035846, the settle's value). So the circuit is well posed and the refusal was the selector's.

Trace (`D1..D4` = the rectifier, bits in the order D1 D2 D3 D4): the two selectors pick different
representatives of the rectifier overlap (all four conducting is a loop of ideal diodes; any three ON
carry it): legacy 1011 → 1110 → 0110, 'auto' 1101 → 0111 → ?. At t = 14.96740 µs (auto: D4's current
reaches 0 in 0111, the pair current is zero: the overlap ends; legacy ends it at the same instant)
the LCP returns **0110** (D1 decided OFF with gap +3.4e-4 V at h_e, D4 OFF, D2/D3 ON — the settle's
mode) but the applied mode was **1110**. Who wrote it: `release_band_ties_` (after `lcp_apply_`'s
selection) — D1's signal at t is its C_j state, 1.31e-13 V, inside the tie band (1e-10 V), and the
look-ahead `zero_band_on_` read the ladder's first rung (1.6e-20 s) as significant (the residue
itself, not a slope) and turned D1 ON; D4, residue −, stayed OFF. In 1110 D1's current starts at 0
and goes negative at once (−2.2 A after 20 ns); a predicate that starts on its event side never
fires, and the end-of-run law check aborted the run.
So: the LCP was right; its result was overridden by a post-step that is meant for the selector's
remaining TIES (ports no pass decided) and reopened a port the first pass had decided OFF.

Fix: `ModeSelection::tied` (per port: left OFF as a remaining tie); `release_band_ties_` considers only
those ports (both callers: `lcp_apply_` and the guard's repair).
After: 'auto' runs to 30 µs, v_out(27–30 µs) 1.0358472569 V against the settle's 1.0358473 V and the
twins' limit ≈ 1.035846 V. The gated bridge from rest (`test_a_gated_bridge_from_rest_turns_its_tie_on_under_the_lcp`,
the case `release_band_ties_` was written for) still passes: its D1 is a remaining tie.

## Verification (tip after the WIP safety push b8fc0e0b, trace prints stripped)

- Extension build; C++ `pulsim_core_mode_tests` (9206 assertions / 40 cases, includes the new F-3-2
  section) and `pulsim_core_dsed_tests` (40935 / 141) pass on Eigen 5. Full ctest not run.
- `test_release_findings_c.py` 19 pass + 1 strict xfail (legacy flyback); area files (18) 397 pass.
- Byte compares against the PR head 27c00520 (`.scratch/c_base`): golden ALL IDENTICAL; trbdf2 identity
  371/371 (modes default, lcp, shadow); bit-identity 36/36 default and 36/36 shadow; pwl identity
  389/390. The one difference is `static rect_lcp` (`mode_selection=2`, opt-in, not the default): one
  bit, x[0] at t = 0 is +0.0 instead of −0.0 (the island correction on the quiet exit adds a zero);
  every other value and every mode bit identical. No default-path number moved.
- ruff clean; stubs `--check` up to date.
- Cost, instructions per accepted step (interleaved head/branch, min of 3): buck pwl 3102.8 → 3088.0
  (−0.48 %), vsi pwl 4770.8 → 4767.4 (−0.07 %), buck trbdf2 11033.4 → 11032.2 (−0.01 %), bridge trbdf2
  12785.3 → 12807.6 (+0.17 %). All within 0.5 %.
