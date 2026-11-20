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
