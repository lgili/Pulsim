# Findings — add-long-simulation-stress

Each finding is parked as a strict xfail in
`python/tests/test_long_simulation_stress.py` (its reason starts with the
finding id). Numbers measured on the extension of b9e46778 (macOS arm64).

## F-L-1 — pwl `discontinuity_restart=True`: one sample per gate edge, integrals off by O(dt) per edge

The restart rule evolves the state correctly (the buck's period-average
output matches the closed-form periodic steady state to 1.4e-12 relative),
but the record holds ONE sample at each gate edge, the left limit (turn-on:
the switch current reads 0 at the edge, its value one step later). Linear
interpolation through the samples turns every edge into a one-step ramp, so
every integral over the record — the in-loop recorder's statistics too —
misses O(dt) per edge, and the error grows with the run length.

| quantity (buck 24 V -> 12 V, 100 kHz, dt = T/100) | measured | method bound |
|---|---|---|
| last-period input power vs closed form | 0.0720 W of 28.743 W (2.5e-3) | (w0 dt)^2 = 2.1e-6 |
| energy ledger over 10 ms from rest | residual 2.44e-3 of the input energy | 2.1e-6 |

`discontinuity_restart=False` closes the same ledger to 6e-9. Same root as
the release's group B (event samples / right limits, 36836230); this pins it
on a long run.

Tests: `test_buck_input_power_matches_the_closed_form[pwl-restart]`,
`test_buck_energy_ledger_closes_over_the_record[pwl-restart]`,
`test_long_run_drift_buck_one_second[pwl-restart]`.

## F-L-2 — dsed: one sample per gate edge (the left limit), with or without `output_rtol`

The exact engine's states are exact (period average 3.5e-13 off the closed
form), but its record carries a single sample at each gate edge, the left
limit. Without `output_rtol` (this head's default) the record is the event
instants only, so the switch current reads as a ramp across the whole
on-time.

| buck energy ledger over 10 ms from rest | residual / input energy | grid bound (w0 h_max)^2 |
|---|---|---|
| dsed, `output_rtol` unset | 0.109 (above the spec's 1 %) | 5.3e-3 |
| dsed, `output_rtol=1e-6` | 5.6e-3 | 1.4e-4 |
| trbdf2 (a right-limit sample 1e-12 s after each edge) | 2.6e-6 | 1e-5 + 10 rtol |

Tests: `test_buck_energy_ledger_closes_over_the_record[dsed]` and
`[dsed-output_rtol]`. Related to the release's group F (output rule).

## Observed, not a defect of this change

- `simulate(engine='dsed', output_rtol=0)` is accepted with `recorders=` and
  refused ("must be > 0") without them.
- trbdf2 / dsed refuse `store_every`: their records hold every accepted
  step, so a long run's record grows with its length by design; only pwl
  can bound it (`test_store_every_bounds_the_record_not_the_statistics`).
- pwl without the restart rule reads each gate at the step's end: the
  sample at an edge instant is one step past the edge (trap 9).

## F-L-1 / F-L-2 on the other circuits (task 5.2)

Energy ledger over a record from rest, residual / input energy:

| circuit | pwl | pwl restart (F-L-1) | trbdf2 | esdirk43 | dsed (F-L-2) |
|---|---|---|---|---|---|
| 1 MHz GaN buck, 50 us | closes | 5.2e-3 (bound 1e-4) | closes | closes | 0.206 |
| stiff 1 GOhm leak + 5 MHz loop, 20 us | closes | 0.0499 | 1.5e-4 | not a KPI (*) | 0.0564 |
| 8-cell MMC, 5 ms | closes | 1.75e-6 (bound 1e-8) | closes | closes | 0.0304 |

(*) esdirk43 crosses the 10 ns RC transient in 0.16-tau steps and records
only those (5711 rows over 20 us; TR-BDF2 44809): its states are right (the
period-start value matches the closed form to 5e-5), but a trapezoid
through that record misses 1.25 % of the input energy. That is ordinary
quadrature error of a coarse, correct record ((w h_max)^2 = 3.3), not an
engine defect: the case is skipped with that reason. Integrals over a
high-order engine's record need its interpolant or a denser record.
