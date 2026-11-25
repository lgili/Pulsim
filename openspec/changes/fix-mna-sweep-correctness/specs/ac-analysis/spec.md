## ADDED Requirements

### Requirement: Kernel MNA Sweep Shares the Transient Dynamic Matrix

The kernel MNA sweep SHALL build its frequency-dependent part from the
same dynamic matrix the transient assembler produces for the given
switch state, so that every linear energy-storage stamp of the
transient — capacitors, inductors and the mutual terms of coupled
inductors — is present in the sweep. No second assembler for those
stamps SHALL exist.

#### Scenario: Coupled transformer
- **GIVEN** `add_transformer` with `L_p = L_s = 1 mH`, `k = 0.99`, a
  10 Ω source resistance and a 10 Ω load on the secondary
- **WHEN** the kernel sweep runs at 100 Hz, 1 kHz and 10 kHz
- **THEN** `|H|` is 0.0617, 0.3885 and 0.4949 and `H` equals
  `jωM·R_L / ((R_s + jωL_p)(R_L + jωL_s) + ω²M²)` within 1e-9
  relative (it returned 0 at all three before)

#### Scenario: Linear circuits without coupling are unchanged
- **WHEN** an RC, an RL and a series RLC circuit are swept
- **THEN** each `H` equals its pre-change value within 1e-12 relative

### Requirement: Kernel MNA Sweep Uses the Caller's Switch State

The kernel MNA sweep SHALL linearise the circuit at the switch state the
caller supplies through `mask=`. When the circuit contains controlled
switches or PWL diodes and no mask is supplied, the sweep SHALL raise an
error naming those branches; it SHALL NOT assume a state.

#### Scenario: Closed switch
- **GIVEN** 100 Ω, a switch (`g_on = 1e3 S`, `g_off = 1e-9 S`), then
  1 µF in parallel with 1 MΩ to ground
- **WHEN** the sweep runs with the switch bit set
- **THEN** `|H|` is 0.998, 0.8467 and 0.1572 at 100 Hz, 1 kHz and
  10 kHz, within 1e-9 relative of the divider formula

#### Scenario: Open switch
- **WHEN** the same sweep runs with the switch bit clear
- **THEN** `|H|` at 100 Hz is below 1e-5

#### Scenario: No mask is refused
- **WHEN** the same sweep runs without `mask=`
- **THEN** it raises an error whose message names the switch

### Requirement: Kernel MNA Sweep Refuses Newton Devices

The kernel MNA sweep SHALL raise an error naming the device and its
kind when the circuit contains a Newton device (smooth, Shockley or
Lauritzen diode, level-1 MOSFET or IGBT, saturable inductor, junction
capacitor, MNA-native PMSM), because it does not linearise such devices
at the operating point. The error SHALL name a time-domain alternative.

#### Scenario: Biased Shockley diode
- **GIVEN** 1.7 V, 1 Ω and a Shockley diode `D` to ground
- **WHEN** the kernel sweep is requested
- **THEN** it raises an error containing `D` and `ShockleyDiode` (it
  returned `|H| = 1` where the small-signal value is 0.0252)

#### Scenario: Saturable inductor and junction capacitor
- **WHEN** the sweep is requested on a circuit with
  `add_saturable_inductor` or `add_nonlinear_capacitor`
- **THEN** each call raises an error naming the device (they returned
  0.5 and 0.9999 at every frequency)

### Requirement: Kernel MNA Sweep Failures Propagate

A failure of the kernel MNA sweep SHALL reach the caller unchanged. The
wrapper SHALL select the impulse-FFT path only from its arguments
(`use_kernel=False`, or no `freqs` / `v_in_branch_id`) and SHALL NOT
fall back to it after a kernel error.

#### Scenario: Kernel error is not swallowed
- **GIVEN** a kernel sweep that raises `RuntimeError("boom")`
- **WHEN** `run_mna_sweep` is called with `freqs`, `v_in_branch_id`,
  `dt` and `t_end`
- **THEN** the caller receives that `RuntimeError` and no time-domain
  simulation is started
