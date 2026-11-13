## ADDED Requirements

### Requirement: MNA-Native DC Machine

The library SHALL provide a DC machine that is solved inside the circuit
matrix: armature resistance and inductance, an ideal electromechanical
coupling `v_emf = K·ω` and `T = K·i`, and a speed node whose voltage is
`ω` [rad/s], carrying the inertia `J` as a capacitor, the friction `B`
as a resistor and the load torque as a current source. It SHALL contain
no explicit one-step coupling, SHALL be second-order accurate in the
fixed step, and SHALL run on `pwl`, on `trbdf2` and in the DC operating
point. It SHALL be available through the builder, Python and the YAML
type `dc_machine`. Invalid parameters SHALL be refused naming the
machine.

#### Scenario: Second-order accuracy against the exact solution
- **GIVEN** `R_a = 0.5 Ω`, `L_a = 1 mH`, `K = 0.1 V·s/rad`,
  `J = 1e-5 kg·m²`, a 10 V, 200 Hz sine source, 20 ms
- **WHEN** it runs on `pwl` at dt = 100, 50, 25 and 12.5 µs
- **THEN** the speed error against the exact solution at 12.5 µs is at
  most 1e-4 of the peak speed
- **AND** each halving of the step reduces the error by at least 3.5

#### Scenario: DC operating point
- **GIVEN** 10 V on the armature, `K = 0.1`, `R_a = 0.5 Ω`,
  `T_load = 0.2 N·m`
- **WHEN** the DC operating point is computed
- **THEN** `i = 2 A` and `ω = 90 rad/s` within 1e-9

#### Scenario: Variable-step engine
- **WHEN** the sine-driven case runs on `trbdf2`
- **THEN** the speed error against the exact solution is at most 2e-4

### Requirement: MNA-Native Induction Machine

The library SHALL provide a squirrel-cage induction machine solved as an
implicit device: three stator phase currents and two rotor-flux
components are unknowns of the circuit matrix, the electromagnetic
torque `(3/2)·pp·(L_m/L_r)·(λ_rα·i_sβ − λ_rβ·i_sα)` enters the speed
node, and no quantity is taken from the previous step. Parameters SHALL
be `R_s`, `L_ls`, `R_r`, `L_lr`, `L_m` (referred to the stator),
`pole_pairs`, `J` and `B`. It SHALL run on `pwl` and `trbdf2`, SHALL be
refused by name on an engine that cannot solve it, and SHALL be
available through the builder, Python and the YAML type
`induction_machine`.

#### Scenario: Locked rotor
- **GIVEN** the test machine `R_s = 0.5 Ω`, `L_ls = 2 mH`,
  `R_r = 0.4 Ω`, `L_lr = 2 mH`, `L_m = 60 mH`, 2 pole pairs, fed with
  100 V rms per phase at 60 Hz, speed node held at 0
- **WHEN** the run reaches steady state
- **THEN** the stator current is 57.88 A rms within 0.5 %

#### Scenario: Synchronous speed
- **WHEN** the speed node is held at 188.50 rad/s
- **THEN** the stator current is 4.277 A rms within 0.5 % and the mean
  torque magnitude is below 0.01 N·m

#### Scenario: Torque at slip 0.05
- **WHEN** the speed node is held at 179.07 rad/s
- **THEN** the mean torque is 16.12 N·m and the stator current 12.29 A
  rms, each within 0.5 %

#### Scenario: Engines agree on a start-up
- **WHEN** a free acceleration from standstill runs on `pwl` and on
  `trbdf2`
- **THEN** the speed waveforms differ by less than 1e-3 of the final
  speed

### Requirement: Non-Sinusoidal PM Flux

The MNA-native PMSM SHALL accept the PM flux linkage as a harmonic
series `λ_pm,a(θ_e) = Σ_h ψ_h·cos(h·θ_e + φ_h)`, with phases b and c
displaced by ∓120° electrical, and SHALL compute the back-EMF and the
torque from that series and its exact derivatives. A series with the
fundamental only SHALL be bit-identical to the existing device. A
helper SHALL build the series of a trapezoidal back-EMF from the
back-EMF constant and the flat-top width.

#### Scenario: Trapezoidal back-EMF
- **GIVEN** a BLDC built with a 120° flat top and harmonics up to the
  25th, driven at constant speed with open terminals
- **WHEN** the phase back-EMF is recorded over one electrical period
- **THEN** its RMS deviation from the ideal trapezoid is below 0.5 % of
  the flat-top value

#### Scenario: Fundamental only is unchanged
- **WHEN** a PMSM is built with `ψ_1 = psi_pm` and no other harmonic
- **THEN** its run is bit-identical to the same machine built without
  the harmonic list

### Requirement: Saturation Flux Maps

The MNA-native PMSM SHALL accept flux-linkage maps `λ_d(i_d, i_q)` and
`λ_q(i_d, i_q)`. The maps SHALL be represented as the gradient of one
co-energy function so that `∂λ_d/∂i_q = ∂λ_q/∂i_d`, and the torque SHALL
be `(3/2)·pp·(λ_d·i_q − λ_q·i_d)` from the same function. The fit
residual SHALL be reported. Input tables whose fit residual exceeds the
tolerance, or whose incremental inductance matrix is not positive
definite, SHALL be refused naming the machine.

#### Scenario: Linear map equals the linear machine
- **GIVEN** maps generated from `L_d = 1 mH`, `L_q = 3 mH`,
  `ψ_pm = 0.05 Wb`
- **WHEN** the machine runs with the maps and with the constants
- **THEN** the two runs agree within 1e-9

#### Scenario: Non-reciprocal tables refused
- **WHEN** the `λ_q` table is scaled by 1.1 and the maps are built
- **THEN** construction fails with an error naming the machine and the
  fit residual

### Requirement: DC Operating Point with Machines

The DC operating point SHALL hold each machine's speed and angle nodes
at their initial conditions, when the circuit contains an induction
machine or a PMSM, and solve the electrical rows with all flux
derivatives zero.

#### Scenario: PMSM at standstill
- **GIVEN** a PMSM with `ω0 = 0`, `θ0 = 0.3 rad`, DC voltages on its
  phases
- **WHEN** the DC operating point is computed
- **THEN** each phase current equals its phase-to-neutral voltage
  divided by `R_s` within 1e-9, where today the solve fails

### Requirement: Observer Machine Deprecation

The observer-based machines SHALL emit a `DeprecationWarning` that
names the MNA-native replacement (`add_dc_motor`, `add_pmsm`,
`add_bldc`, `add_induction_motor` and their `make_*_observer`
functions). They SHALL keep their behaviour until removed.

#### Scenario: Warning names the replacement
- **WHEN** `add_dc_motor(...)` is called
- **THEN** a `DeprecationWarning` is emitted whose message contains
  `add_dc_machine`
