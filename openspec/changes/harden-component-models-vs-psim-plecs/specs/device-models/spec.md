## ADDED Requirements

### Requirement: IGBT V_CE_sat in Stamping Path

The Behavioral and AD stamping paths of the IGBT device SHALL produce a
collector-emitter voltage drop matching the `Params::v_ce_sat` field
when the IGBT is in the ON state, not the previous `I_C / g_on` ≈ 5 mV
artefact.

#### Scenario: IGBT carries 50 A with v_ce_sat = 1.5 V

- **GIVEN** an `IGBT` with `Params::v_ce_sat = 1.5 V`, `Params::R_CE_on = 25 mΩ`
- **AND** a stiff DC bus driving 50 A through the IGBT (collector → emitter), gate held high (V_gs = 15 V)
- **WHEN** the simulator stamps and solves the steady-state operating point
- **THEN** the measured `V_CE` SHALL equal `1.5 V + 50 A · 25 mΩ = 2.75 V` (±5 mV)
- **AND** NOT the legacy `≈ 5 mV` from `I_C / g_on = 50 / 1e4`.

#### Scenario: IGBT cross-validates AD path

- **GIVEN** the same IGBT in cutoff (V_gs = 0), at triode boundary (V_CE = V_CE_sat), and at saturation (V_CE >> V_CE_sat)
- **WHEN** the manual and AD stamps are evaluated at each op-point
- **THEN** Jacobian and residual entries SHALL agree within 1e-12 margin (matches the existing `test_ad_igbt_stamp.cpp` contract).

### Requirement: Gate-Row Diagonal Anchor on MOSFET + IGBT

The MOSFET and IGBT stamping paths (PWL Ideal, Behavioral, and AD) SHALL
unconditionally stamp a small leakage conductance on the gate-row
diagonal to prevent a structurally singular Jacobian when the gate node
is otherwise floating.

#### Scenario: NMOS with floating gate produces a well-conditioned matrix

- **GIVEN** an NMOS MOSFET with the gate node NOT connected to any PWM source, resistor to ground, or other device
- **AND** the drain connected to a stiff 100 V DC bus through a 10 Ω pull-up
- **AND** the source connected directly to ground
- **WHEN** the simulator runs a transient with `Params::g_gate_leak = 1e-9` (the new default)
- **THEN** the linear solve SHALL succeed without "matrix is singular" failures
- **AND** the steady-state V_drain SHALL be 100 V (within 0.1 V) — MOSFET in cutoff, no current flows.

### Requirement: Smooth-Blend Sharpness Stable Under Float Precision

The Behavioral MOSFET and IGBT stamps SHALL keep the smooth-blend sigmoid `sigma_g = 1/(1 + exp(κ · (V_th − V_gs)))` numerically non-denormal at typical operating points in single-precision builds, by using a sharpness `κ` no greater than 20 (down from the legacy 50).

#### Scenario: σ_g at V_gs = 0, V_th = 4 V stays above the float denormal floor

- **GIVEN** the Behavioral MOSFET stamp evaluated at V_gs = 0, V_th = 4 V in a Pulsim build compiled with `Real = float`
- **WHEN** the smooth-blend sharpness is `kSmoothRegionSharpness = 20` (the new default, down from 50)
- **THEN** `sigma_g = 1/(1 + exp(20·4)) = 1/(1 + exp(80)) ≈ 1.8e-35` SHALL be in the normal float range (> 1.4e-45, the IEEE-754 single-precision denormal floor)
- **AND** the derivative chain `dsigma_g_d_vgs = κ·σ_g·(1−σ_g) ≈ 3.6e-34` SHALL also be normal-range.

### Requirement: Diode Threshold Hysteresis Wide Enough for Bus Noise

The `IdealDiode::event_hysteresis_` default SHALL be wide enough that
typical ESL-induced ringing on a converter bus does NOT cause the diode
to chatter (flip ON/OFF every Newton iteration).

#### Scenario: 20 mV pp noise on a 400 V bus does not chatter the diode

- **GIVEN** an `IdealDiode` with default `event_hysteresis_` (now 50 mV, up from 10 mV)
- **AND** a 20 mV pp triangular noise signal injected at the anode (modeling ESL kick during commutation on a 400 V converter bus)
- **WHEN** the simulator runs a transient over several noise cycles
- **THEN** the diode `pwl_state_` SHALL NOT toggle (no spurious switching events)
- **AND** `backend_telemetry.pwl_event_commutations` from the diode SHALL be 0.

### Requirement: Voltage-Controlled Switch Hysteresis Family-Consistent

`VoltageControlledSwitch::event_hysteresis_` default SHALL be on the
same numeric scale as the other PWL switching devices (MOSFET / IGBT /
diode), not 7 orders of magnitude tighter.

#### Scenario: VCSwitch hysteresis default is 1 mV (not 1 nV)

- **GIVEN** a freshly-constructed `VoltageControlledSwitch`
- **WHEN** `vcswitch.event_hysteresis()` is queried
- **THEN** the returned value SHALL be 1e-3 V (1 mV)
- **AND** SHALL NOT be the legacy 1e-9 V (1 nV).

### Requirement: Trapezoidal Integration on Induction Motor Rotor Flux

The 3-phase `InductionMotorDevice::advance_state` method SHALL integrate
the rotor flux `ψ_r` via the trapezoidal rule (one fixed-point
iteration), not forward-Euler, to stabilise high-slip start
transients.

#### Scenario: DOL start of 3φ IM converges monotonically

- **GIVEN** a 3-phase induction motor with `R_s = 0.5 Ω, L_s = 5 mH, L_m = 4.8 mH, L_r = 5 mH, R_r = 0.3 Ω, J = 0.01 kg·m²`, started DOL at full voltage from standstill
- **WHEN** the simulator runs a 500 ms transient at `dt = 100 µs`
- **THEN** the rotor angular velocity `ω` SHALL increase monotonically
- **AND** the ψ_r magnitude SHALL NOT oscillate by more than 1% peak-to-peak at any time
- **AND** no Newton convergence failures SHALL be reported in the fallback trace.

### Requirement: Trapezoidal Integration on Single-Phase PSC Motor Run Capacitor

The `SinglePhaseInductionMotorDevice::advance_state` method SHALL
integrate the run-capacitor voltage `V_cap` via the trapezoidal rule
using the previous and current auxiliary winding currents, not
forward-Euler.

#### Scenario: Run-cap voltage waveform is smooth on starting transient

- **GIVEN** a single-phase PSC induction motor with `C_run = 4 µF`, started at t=0 with motor standing still
- **AND** simulator timestep `dt = 100 µs`
- **WHEN** the simulation runs the first 50 ms of the starting transient
- **THEN** consecutive samples of `V_cap` SHALL NOT differ by more than 1 V (the FE artefact was up to 25 V per step at peak aux current).

### Requirement: MOSFET Body Diode

The MOSFET device SHALL support an optional intrinsic body diode
(antiparallel diode anode = source, cathode = drain) configurable via
`Params::body_diode_enable` (default true) and the
`body_diode_{V_F0, R_d, Qrr}` field group. When enabled, the body diode
participates in MNA stamping using the same `R_th_ja` and `T_amb` as
the parent MOSFET unless `body_diode_R_th_ja > 0` is set explicitly.

#### Scenario: Synchronous buck with body diode enabled clamps V_sw

- **GIVEN** a half-bridge synchronous buck with high-side MOSFET ON and low-side MOSFET OFF during dead time
- **AND** load inductor pushes positive current into the switching node
- **AND** both MOSFETs have `body_diode_enable = true` (default)
- **WHEN** dead time occurs (both gates LOW for 1 µs)
- **THEN** V_sw SHALL clamp at `+V_F` of the high-side body diode (not −V_dc as before)
- **AND** the conducting body diode's loss accumulator SHALL register the freewheel current.

#### Scenario: SPICE-parity test mode opts out of body diode

- **GIVEN** a user explicitly sets `MOSFETParams::body_diode_enable = false`
- **WHEN** the MOSFET is added to the circuit
- **THEN** no antiparallel diode SHALL be stamped (legacy behaviour preserved).

### Requirement: IGBT Antiparallel Diode

The IGBT device SHALL support an optional antiparallel diode (anode = emitter, cathode = collector) configurable via `IGBTParams::antiparallel_diode_enable` (default true) and the `antiparallel_diode_{V_F0, R_d, Qrr}` field group, mirroring the MOSFET body diode contract.

#### Scenario: 3φ inverter on RL load freewheels through antiparallel diodes

- **GIVEN** a 3φ IGBT inverter driving an RL load, dead time = 2 µs
- **WHEN** an inverter leg's IGBTs both turn OFF (during dead time) and the load current is positive
- **THEN** the corresponding antiparallel diode SHALL conduct the load current
- **AND** the simulation SHALL NOT report "matrix singular" or "Newton failed" errors.

### Requirement: Motor Winding Thermal Model

The five motor device families (DC, PMSM, BLDC, 3φ Induction, Single-phase PSC) SHALL each support an optional winding thermal model via Params fields `R_th_winding_to_ambient` (K/W, default 0), `T_amb` (°C), `R_s_tc` (1/K), and `T_ref_winding` (°C, default 20); when `R_th_winding_to_ambient > 0`, the stator resistance SHALL be scaled by `(1 + R_s_tc · (T_winding − T_ref_winding))` and the winding temperature SHALL be derived from the I²·R conduction loss via the unified `accumulate_loss` pipeline.

#### Scenario: PMSM under rated load reaches expected steady-state T_winding

- **GIVEN** a PMSM with `R_s = 0.5 Ω, R_th_winding_to_ambient = 5 K/W, T_amb = 25 °C, R_s_tc = 3.9e-3/K`, drawing 10 A line current in steady state
- **WHEN** the simulator runs long enough to reach thermal steady state (>20 thermal time constants)
- **THEN** `circuit.pmsm_steady_state_winding_temperature("M1")` SHALL equal `25 + 3·10²·0.5·5 / (1 - 3·10²·0.5·3.9e-3·5/R_th)` ≈ 100 °C (closed-form Foster steady-state with R(T) feedback) within 5%.

### Requirement: Automatic Shaft Coupling API

The `Circuit` SHALL provide a `couple_shaft(motor_name,
mechanical_name, gear_ratio = 1.0)` method that wires
`MechanicalDevice::set_tau_input(motor.tau_em() / gear_ratio)` and
`motor.set_tau_load(mechanical.reaction_torque() · gear_ratio)`
automatically each timestep, using `motors::GearBox::reflect_load` for
the gear-ratio reflection.

#### Scenario: Coupled PMSM + shaft responds to load step

- **GIVEN** a PMSM coupled to a `MechanicalDevice` (J=0.01, b=0.001) via `circuit.couple_shaft("M1", "shaft1", gear_ratio=1.0)`
- **AND** the PMSM is at steady-state ω = 100 rad/s, τ_em ≈ 0.1 N·m (just overcoming friction)
- **WHEN** the user applies a 5 N·m load step at t=1.0 s via the mechanical device's external load input
- **THEN** the shared shaft state SHALL update with `τ_input = motor.tau_em()` and `τ_load = 5 N·m + friction`
- **AND** ω SHALL drop to a new steady-state value consistent with the motor's `τ_em(ω)` characteristic.

### Requirement: SaturableTransformer Device Variant

A new `SaturableTransformer` device variant SHALL wrap
`magnetic/saturable_transformer.hpp` and support multi-winding turns,
per-winding leakage inductance, per-winding resistance, and saturable
magnetizing branch with Jiles-Atherton or polynomial-S B-H curve.

#### Scenario: Saturable transformer exhibits peaky magnetizing current at high flux

- **GIVEN** a 2-winding saturable transformer with `B_sat = 0.3 T`, primary turns N1 = 100, secondary turns N2 = 50, secondary open-circuited
- **WHEN** a sinusoidal voltage at low frequency drives the primary to peak flux density 0.5 · B_sat
- **THEN** the primary magnetizing current waveform SHALL be approximately sinusoidal
- **WHEN** the primary drive is increased to peak flux density 2 · B_sat
- **THEN** the magnetizing current SHALL exhibit visible saturation peaks (peak-to-RMS ratio > 2.0).

#### Scenario: Saturable transformer total flux linkage is conserved across windings

- **GIVEN** a 2-winding saturable transformer with `N1/N2 = 2.0` and turns ratio convention `v1/v2 = N1/N2`
- **WHEN** the primary is energised with a sinusoid and the secondary terminates in a resistor
- **THEN** in steady state, `V_secondary_rms / V_primary_rms` SHALL equal `1/2.0` within 1%
- **AND** the per-winding leakage drop SHALL be reflected in the actual transformer regulation.

### Requirement: Steinmetz / iGSE Core Loss on Inductor and SaturableTransformer

The `Inductor` and `SaturableTransformer` devices SHALL support
optional Steinmetz / iGSE core-loss accumulation via Params fields
`core_loss_k_h`, `core_loss_alpha`, `core_loss_beta`, `core_volume`,
`effective_area`. When `core_volume > 0` AND `core_loss_k_h > 0`, the
device SHALL compute the flux density `B(t) = λ(t) / (N · A_e)` per
step and apply iGSE per `magnetic/bh_curve.hpp::igse_loss_density`
to derive the core loss, integrating into `e_cond_` via the unified
`accumulate_loss` pipeline.

#### Scenario: Inductor under square-wave excitation accrues Steinmetz core loss

- **GIVEN** a 100 µH inductor with `core_loss_k_h = 1.5e-3, core_loss_alpha = 1.4, core_loss_beta = 2.5, core_volume = 1e-6 m³, effective_area = 5e-5 m²`
- **WHEN** the inductor is driven by a 100 kHz, 10 V peak square-wave voltage source
- **THEN** `circuit.inductor_average_power("L1")` SHALL include both the I²·DCR conduction loss and the Steinmetz core loss
- **AND** the predicted Steinmetz loss density at 100 kHz, ΔB ≈ V·T/(2·N·A_e) SHALL match the accumulator output within 10%.

### Requirement: HysteresisInductor Incremental L_eff

The `HysteresisInductorDevice` MNA stamp SHALL recompute its
effective inductance `L_eff = ∂λ/∂i` per step from the current
Jiles-Atherton operating point, NOT use a constant `L_eff` from
Params.

#### Scenario: Saturable inductor V_L drops at peak current

- **GIVEN** a `HysteresisInductor` with Jiles-Atherton params giving `B_sat = 0.4 T, L_linear = 10 mH at low I`
- **AND** the inductor driven by a triangular current waveform 0 → 1.5 · I_sat → 0
- **WHEN** the simulator runs the transient
- **THEN** the measured `V_L = L_eff(i) · dI/dt` SHALL drop sharply when `|i| > I_sat`
- **AND** the inductor's `total_energy()` SHALL include both hysteresis loss (loop area) AND optional Steinmetz core loss (if Params are set).
