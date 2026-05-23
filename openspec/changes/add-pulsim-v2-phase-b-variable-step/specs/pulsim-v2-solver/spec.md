## ADDED Requirements

### Requirement: Adaptive Time-Stepping with Local Truncation-Error Control

Pulsim v2 SHALL provide an adaptive time-step driver that adjusts `dt` between solver steps based on a local truncation-error (LTE) estimate, accepting or rejecting each step so the propagated error stays under user-supplied absolute + relative tolerances.

The driver SHALL:
- Use step doubling + Richardson extrapolation for the LTE estimate: take one step of size `dt` and two steps of size `dt/2`, then estimate `LTE ≈ ||x_double − x_single|| / (2^p − 1)` where `p = 2` is the order of the trapezoidal companion method.
- Accept a step when `LTE / (atol + rtol · ||x||) ≤ 1`, rejecting otherwise.
- Update the next-step size via the standard PI step controller `dt_new = dt · safety · (1 / err_norm)^(1/(p+1))`, clamped to a configurable `[dt_min, dt_max]` interval and a maximum step-growth ratio (default 5×).
- Emit BOTH the accepted samples (parallel `times` and `states`) AND the history of `dt` choices (so users can plot the step-size adaptation).
- Be wrapped by a Python entry point `pulsim.v2.run_transient_adaptive(builder, t_start, t_end, dt_init=..., atol=..., rtol=..., ...)`.

#### Scenario: RL settling — dt grows toward the steady state

- **GIVEN** an RL step response with `τ = L/R = 10 µs` simulated for 500 ms
- **WHEN** `run_transient_adaptive(builder, t_end=500e-3, dt_init=100e-9, atol=1e-6, rtol=1e-4)` is called
- **THEN** the final dt SHALL grow to at least 10 × the initial step (typically ~50 µs) by the end of the transient
- **AND** the worst-case `||x(t) − (1 − e^(−t/τ))||_∞` over the recorded samples SHALL stay below `rtol · ||x||_∞`
- **AND** the total number of accepted steps SHALL be at most 10 % of the fixed-dt step count (~500e-3 / 100e-9 = 5 M)

#### Scenario: Adaptive solver does not corrupt switched simulations

- **GIVEN** the existing buck closed-loop circuit (PWM at 100 kHz, fixed-dt at 100 ns)
- **WHEN** the same simulation is run with `run_transient_adaptive` using `dt_max ≤ 100 ns` and the same `switch_fn`
- **THEN** the KPI (output voltage mean over the last ms) SHALL match the fixed-dt result within 5 mV
- **AND** `n_rejected / n_accepted` SHALL stay below 5 % (no thrash near PWM edges)
