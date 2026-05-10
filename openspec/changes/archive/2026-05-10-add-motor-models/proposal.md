## Why

Motor drives are roughly half of the power-electronics market PSIM addresses. Without motor models, Pulsim is unusable for:
- Drives engineering (PMSM, induction, BLDC, SRM, DC)
- EV/e-mobility powertrain simulation
- Servo controller validation
- Field-oriented control (FOC) and direct-torque control (DTC) prototyping

PSIM ships a Motor Drive Module with detailed models. PLECS has the Electrical Machines library. OpenModelica's `Modelica.Electrical.Machines` library is the open-source reference. To compete in this segment, Pulsim must ship at least:

- Three-phase PMSM (sinusoidal, surface and interior — SPM/IPM)
- Three-phase induction motor (squirrel cage)
- Brushless DC motor (BLDC, trapezoidal back-EMF)
- DC motor (separately and series excited)
- Mechanical load: shaft, friction, gear, flywheel

The implementation pattern (dq decomposition, abc/dq transforms, mechanical coupling) is well-known.

## What Changes

### Motor Device Models
- `PmsmDevice` — three-phase PMSM in dq frame with optional saturation. Parameters: `Rs`, `Ld`, `Lq`, `psi_pm`, `pole_pairs`, `inertia_J`, `friction_b`, optional `saturation_table`.
- `InductionMotorDevice` — three-phase squirrel cage in dq frame. Parameters: `Rs`, `Rr_referred`, `Ls`, `Lr`, `Lm`, `pole_pairs`, mechanical.
- `BldcMotorDevice` — trapezoidal back-EMF, hall sensors. Parameters: `Rs`, `Ls`, `Ke`, `pole_pairs`.
- `DcMotorDevice` — separately or series excited. Parameters: `Ra`, `La`, `Kt`, `Ke`, optional field winding.
- `SrmDevice` (stretch goal, deferred to v2) — switched reluctance.

### Mechanical Subsystem
- `Shaft` 1-D mechanical block with inertia, friction (linear + Coulomb), torque ports.
- `GearBox { ratio, efficiency }`.
- `FlywheelLoad { J }`, `ConstantTorqueLoad { tau }`, `FanLoad { Kp, Kw }` (proportional / quadratic).
- Mechanical-electrical coupling via shared `omega` / `tau` signals.

### Frame Transformations
- `AbcToDq { theta_signal }`, `DqToAbc { theta_signal }` blocks (Park / Clarke).
- Available as both signal-domain blocks (for FOC controllers) and electrical-domain blocks (for analytical motor terminals).

### FOC Controller Sub-Template
- `pmsm_foc_template` instantiates current loops (id, iq), speed loop, position loop, with id-flux-weakening option.
- Default tuning derived from motor parameters.

### Encoder / Hall Models
- `EncoderQuadrature { ppr, gain }` for FOC.
- `HallSensor { transitions_table }` for BLDC commutation.
- `Resolver { excitation_freq, ratio }` for high-precision applications.

### YAML Schema
- New types: `pmsm`, `induction_motor`, `bldc_motor`, `dc_motor`, `shaft`, `gearbox`, `flywheel_load`, `constant_torque_load`, `fan_load`, `encoder_quadrature`, `hall_sensor`, `resolver`, `abc_to_dq`, `dq_to_abc`, `pmsm_foc_template`.
- Mechanical-port concept: `terminals: { electrical: [a, b, c], mechanical: [shaft] }`.

### Validation
- Each motor model has at least one analytical-reference parity test (e.g., DC motor speed step matches first-order analytical response).
- FOC template validated: id/iq decoupling, speed-loop step response matches design.
- Locked-rotor and no-load tests for induction motor.

## Impact

- **New capability**: `motor-models`.
- **Affected specs**: `motor-models` (new), `netlist-yaml`.
- **Affected code**: new `core/include/pulsim/v1/motors/`, `core/include/pulsim/v1/mechanical/`, parser additions.
- **Performance**: motor models are smooth nonlinear; AD path handles. Mechanical subsystem is small (≤5 states), no impact on linear solver.

## Success Criteria

1. **Coverage**: PMSM, IM, BLDC, DC motor models implemented and validated.
2. **Mechanical**: shaft + gearbox + load blocks compose correctly.
3. **FOC template**: speed-loop step response matches designed bandwidth within 20%.
4. **Reference parity**: PMSM no-load + locked-rotor tests match analytical formulas within 5%.
5. **Tutorial**: end-to-end PMSM-FOC drive with three-phase inverter (uses `add-three-phase-grid-library`) demonstrating speed-trapezoidal command response.
