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

## F-4-1 — an ideal voltage loop refused as `consistent_mode` under 'auto' (exact engine) [M]

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
