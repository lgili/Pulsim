## ADDED Requirements

### Requirement: Closed Electrothermal Loop

The simulator SHALL push the integrated junction temperature `T_i(t)`
from `DefaultThermalService` back into each participating device's
internal `T_j_` at the end of every accepted simulation step, so that
device accessors (`<dev>_junction_temperature(name)`) and the
temperature-corrected device methods (`Rds_on_at_Tj`, `V_ce_sat_at_Tj`,
`R_at_Tj`, `DCR_at_Tj`, `ESR_at_Tj`, `V_F0_at_Tj`) MUST observe the
same `T_i(t)` that the service uses for the
`set_device_temperature_scales` Newton-stamp feedback path.

A device participates in the closed loop iff
`device_traits<T>::has_thermal_model == true` AND the device class
exposes a `void set_T_j_init(Real)` method. Devices that fail either
predicate MUST silently skip the dispatch (SFINAE / concept gating).

#### Scenario: MOSFET temperature walks under continuous conduction

- **GIVEN** a MOSFET with `R_th_ja = 1 K/W`, `T_amb = 25 °C`,
  `Rds_on_tc = 5e-3 1/K`, drawing 10 A continuous
- **AND** the simulator is run for `t_stop ≥ 5·R_th_ja·C_th` so the
  thermal ODE reaches steady state
- **WHEN** the simulation completes
- **THEN** `circuit.mosfet_junction_temperature("M1")` SHALL return a
  value strictly greater than `T_amb + 50 °C` (the device dissipates
  ≥ 50 W continuously, so T_j SHALL be ≥ T_amb + P·R_th_ja − ε)
- **AND** `circuit.mosfet_junction_temperature("M1")` SHALL equal
  `result.thermal_summary.device_temperatures["M1"]` within 1e-9 °C
  at the final step
- **AND** `circuit.mosfet_steady_state_junction_temperature("M1")`
  SHALL agree with the final `mosfet_junction_temperature` within
  5 % (steady state has converged).

#### Scenario: Circuit with R_th_ja = 0 sees no loop effect (back-compat)

- **GIVEN** a MOSFET / IGBT / Diode / R / L / C device with
  `R_th_ja = 0` (the default for MOSFET, IGBT, R, L, C; opt-out for
  Diode)
- **WHEN** the simulator runs the closed-loop dispatch
- **THEN** the device's `junction_temperature()` SHALL stay at `T_amb`
  for the whole simulation
- **AND** the simulator output SHALL be bit-identical to a run with
  the closed loop disabled (`R_th = 0` short-circuits the ODE).

#### Scenario: Devices without `set_T_j_init` silently skip the dispatch

- **GIVEN** a circuit containing motors (DcMotorDevice, PmsmDevice,
  etc.) that do NOT expose `set_T_j_init`
- **WHEN** the closed-loop walker iterates the device variant
- **THEN** the walker SHALL compile + run without errors
- **AND** the motor devices SHALL skip the `set_T_j_init` dispatch
  branch (concept-gated)
- **AND** the simulator SHALL run to completion with no diagnostic
  about the missing setter (it is by-design for non-thermal devices).

## ADDED Requirements

### Requirement: Thermal-Service Dispatch Order

The `DefaultThermalService::commit_accepted_segment` SHALL execute the
following ordered sub-steps per accepted simulation step:

1. Integrate `T_i(t)` via the Euler update
   `T_i ← T_i + dt·(P_i·R_th_i − (T_i − T_amb_i))/τ_i` for every device
   with `has_thermal_model == true`.
2. Compute `scale_i = clamp(1 + α_i·(T_i − T_ref_i), 0.05, 4)` for the
   same devices.
3. Push `scale_i` into the stamp via
   `circuit_.set_device_temperature_scales(scale_i)`.
4. Push `T_i` into the device-internal `T_j_` via `set_T_j_init(T_i)`
   for every device that exposes the setter.
5. Mirror `T_i`, `peak_T_i`, `avg_T_i` into the
   `thermal_summary.device_temperatures[i]` accumulator.

Steps (1)-(3) preserve the existing closed loop on the stamp side.
Step (4) is the new sub-step this proposal adds. Step (5) is unchanged
and SHALL continue to reflect the post-step `T_i` value.

#### Scenario: Dispatch order is deterministic across runs

- **GIVEN** the same circuit + options run twice in succession
- **WHEN** the dispatch executes
- **THEN** the per-step `T_i`, `scale_i`, and device-internal `T_j_`
  values SHALL match between the two runs bit-for-bit (no random
  ordering, no map iteration order leak).
