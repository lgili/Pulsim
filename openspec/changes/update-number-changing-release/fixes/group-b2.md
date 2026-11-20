# Release finding F-4-4 — the in-loop recorder at restart steps (group B, part 2)

Branch `fix/core-release-b2-recorder-restart`, from group B's finished
branch `fix/core-release-b-event-samples` (36836230). Tests:
`python/tests/test_release_findings_b2.py` (the release options explicit:
`engine='pwl', discontinuity_restart=True, enable_substep_state_correction=True`).

## Reproduction (base 36836230)

| check | truth | base |
|---|---|---|
| buck window charge, windows on stored samples | C·Δv of the stored samples (identity, round-off) | 4.0e-4 off |
| rectifier, same | same | 2.6e-8 off (> 1e-11 of scale) |
| buck / rectifier mid-step windows (test_f44's) | C·Δv, 2e-4 | 4.1e-4 / 4.4e-3 |
| buck energy balance from recorder outputs | supplied = dissipated + stored change | 3.57e-2 of supplied |
| in-loop vs stored-array feed, same run | equal (1e-12) | differ |

## Trace (buck, dt = 1 us, in-loop i(C) and v(out) kept every step)

Per step, `h/2·(i_C(k−1) + i_C(k)) − C·(v(k) − v(k−1))`: 123 steps of 1000
exceed 1e-15; the run has 122 restart steps and 1 inner event; all 122
restart steps start at a sample in `right_limit_index`; every other step is
exact. First rows: (0, 1 us] −2.1e-10, (1, 2 us] −2.2e-10, (2, 3 us]
−2.2e-10 (the run's start), (20, 21 us] −5.7e-11 ... (a gate edge's three).

Cause (group B's reading, confirmed): the feed sees a restart step's ENDS
only, and a capacitor reading is the companion's END current; an SDIRK(2,2)
step's charge is h·((1 − γ)·i1 + γ·i2), not the trapezoid of its ends.
