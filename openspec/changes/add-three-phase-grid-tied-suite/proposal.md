## Why
Today we have `three_phase_diode_rectifier` and `triple_sync_buck` (three independent legs), but no real three-phase **inverter** driving a balanced wye load with **sine-PWM** modulation. Without that, we can't claim parity with PSIM/PLECS on grid-tied applications — which is where the regulatory action is (IEEE 1547, grid codes).

The two industry-defining benchmarks here are:
1. **Three-phase 6-switch inverter** producing a sine-PWM AC output, observing line-to-line voltage and load currents.
2. **Grid-tied inverter** that synchronizes to a reference grid voltage via a **PLL** (Phase-Locked Loop), so the inverter's output is in phase with the grid — the foundational building block for solar/wind inverters.

## What Changes
- Build three new benchmarks:
  - `three_phase_inverter_svpwm` — three half-bridges, gates driven by **Space-Vector PWM** (sine reference × triangle carrier per phase, 120° phase-shifted). Wye-connected RL load. Observe i_a, i_b, i_c plus V(neutral).
  - `grid_tied_single_phase_pll` — a single-phase H-bridge inverter trying to track a reference sine "grid" voltage. Closed-loop with a software PLL implemented from `pi_controller` + `integrator` + `sine` blocks. Observe phase error vs grid.
  - `back_to_back_rectifier_inverter` — full-bridge rectifier on the input + three-phase sine-PWM inverter on the output, both sharing a DC link cap. Tests bidirectional power flow handling.
- New helper inside `pi_controller` or via a new control block: **PLL_SOGI** (Second-Order Generalized Integrator-based PLL). If a clean implementation requires new infrastructure, document the approach taken (likely build it from existing `integrator` + `gain` + `sum` blocks).
- KPIs: output current THD per phase, phase error in steady state (for the PLL benchmark), DC-link ripple (for the back-to-back).

## Impact
- Affected specs: `three-phase-grid` (concrete benchmark requirements), `benchmark-suite`.
- Affected code: new YAML circuits + baselines, optional new PLL composite block.
- Builds on existing primitives — no C++ changes anticipated.
