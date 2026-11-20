# Findings: add-textbook-reference-cases

Cases and truths: `python/tests/test_textbook_reference_cases.py` (mpmath,
independent of pulsim). Every engine (`pwl` with and without
`discontinuity_restart`, `trbdf2`, `trbdf2` + `esdirk43`, `dsed`) met the
truth within the tolerance its method explains on every case; the two
findings below are the only defects seen.

## F-T-1 (benchmark / spec, not an engine): the "CCM" Erickson buck is in DCM

`benchmarks/circuits/switching/buck/textbook_erickson_buck_ccm.yaml`
(24 V, D = 0.5, 50 kHz, 47 uH, 100 uF, Rload = 10 ohm) has
K = 2 L f / R = 0.47 < K_crit = 1 - D = 0.5: discontinuous conduction.
Exact orbit: V_out = 12.2516 V (Erickson ch. 5 DCM formula 12.2483 V),
inductor current zero for 0.42 us a period; D * V_in = 12.0 V is 2.1 %
off, outside the proposal's +-2 %. Every engine agrees with the DCM orbit
(worst 5.0e-4, fixed step without restart). The spec scenario's numbers
(V_out 12.0, delta_il 1.6, delta_vc 0.024) match neither this circuit
(peak-to-peak 2.553 A and 63.8 mV by the CCM formulas) nor its DCM state;
the spec now uses the same converter at Rload = 5 ohm (K = 0.94, CCM),
where the exact orbit meets 12.0 V / 2.553 A / 63.8 mV within 2 %. The
YAML carries a note.

## F-T-2: `engine='dsed'` never returns on a non-advancing `next_edge_after`

A `switch_fn` whose `next_edge_after(t)` returns `t` itself (here a float
round-off in a periodic gate: `k*T1 + e[j]` equal to `t` at an edge) makes
`simulate(engine='dsed')` spin at 100 % CPU indefinitely (a 0.1 ms RC
switch run, killed after 30 s; the same run with a correct gate takes
milliseconds). `engine='trbdf2'` returns (with a warning);
`periodic_steady_state` refuses such a gate by name. Expected: a refusal
by name, or progress. CLOSED on the integration line (batch 14t):
`events::require_after` refuses such an oracle by name at once
(`UnsupportedError`, 0.3 s); the test is now the direct assertion
`test_dsed_refuses_a_non_advancing_next_edge_after`.

## Observations (not defects)

- At the same `rtol = 1e-6`, `integrator='esdirk43'` is sometimes less
  accurate than the default TR-BDF2 (buck R = 5: 5.2e-6 vs 6.6e-7; boost:
  5.4e-6 vs 1.1e-6). Both are within 50 rtol; an error-controlled method
  is held to its tolerance, not its order.
- Metric traps met while writing the cases (not engine errors): an
  un-clipped Fourier window on the exact engine's grid (1.2e-4 of I1),
  the trapezoid of i^2 on a sparse adaptive grid (1e-4 of I_rms; the test
  integrates the linear interpolant's square exactly), and the bridge's
  two-diode commutation overlap, which the first truth ignored (1.4e-5 of
  the line current RMS; the exact engine exposed it, the truth now
  subtracts it).
