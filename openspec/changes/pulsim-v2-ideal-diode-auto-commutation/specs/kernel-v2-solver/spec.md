## ADDED Requirements

### Requirement: DiodeEventState — Per-Step Diode Tracker

`pulsim::v2::pwl::DiodeEventState` SHALL track the current
on/off state of every `IdealDiode` registered in the
`DevicePool`. Layer 5 V2's `run_transient` constructs one
instance per simulation and updates it after each `cache.solve`.

The class MUST expose:

```cpp
class DiodeEventState {
public:
    DiodeEventState(const topology::Graph& graph,
                    const DevicePool& pool);

    /// Returns a SwitchStateMask of pool.num_switches() bits
    /// with the diode bits set per the current state and the
    /// non-diode bits cleared.
    [[nodiscard]] topology::SwitchStateMask
    current_diode_mask() const;

    /// Returns a SwitchStateMask of pool.num_switches() bits
    /// with 1s at diode-owned bit positions and 0s elsewhere.
    /// Layer 5 uses this to mask out user-supplied diode bits.
    [[nodiscard]] topology::SwitchStateMask
    diode_owned_bits() const;

    /// Re-decide each diode's state based on the just-computed
    /// state vector `x`. Returns true if any diode flipped.
    bool update_from_state(const Vector& x);

    /// Reset all diodes to OFF (V0 initial state).
    void reset() noexcept;

    /// Diagnostic.
    [[nodiscard]] Size num_diodes() const noexcept;
};
```

All diodes start OFF on construction (and after reset).

#### Scenario: No diodes → empty tracker

- **GIVEN** a Graph + DevicePool with no diodes
- **WHEN** the user constructs `DiodeEventState`
- **THEN** `num_diodes()` SHALL be `0`
- **AND** `current_diode_mask()` SHALL be all-zero of the right
  width.

#### Scenario: One diode, OFF initial state

- **GIVEN** a Graph with one Switch-kind branch and a
  DevicePool registering it as a diode
- **WHEN** the user constructs `DiodeEventState`
- **THEN** `num_diodes()` SHALL be `1`
- **AND** `current_diode_mask().get(0)` SHALL be `false`.

#### Scenario: update_from_state flips the diode

- **GIVEN** a diode currently OFF
- **WHEN** the user calls `update_from_state(x)` with an `x`
  where v_anode − v_cathode > V_th
- **THEN** the call SHALL return `true` (a diode flipped)
- **AND** `current_diode_mask().get(diode_switch_idx)` SHALL be
  `true`.

### Requirement: run_transient — Diode-Aware Loop

`run_transient` SHALL be extended so that, when the
DevicePool contains diodes, each diode's state is automatically
managed per simulation step. The implementation MUST:

1. Construct a `DiodeEventState` from `(graph, pool)`. If
   `pool.num_diodes() == 0` the state is empty and the loop's
   behaviour MUST be bit-identical to Layer 5 V1.
2. Each step, before `cache.solve`, combine the user-supplied
   `switch_fn(t)` mask with `diodes.current_diode_mask()` —
   replacing the diode-owned bits in the user mask with the
   diode's own bits, leaving non-diode bits intact.
3. Each step, after `cache.solve`, call
   `diodes.update_from_state(x)` so the next step's mask
   reflects any commutation that just happened.

#### Scenario: Buck converter test (no diodes) unchanged

- **GIVEN** the Layer 5 V1.5 buck converter (no diodes — both
  switches are controlled MOSFETs)
- **WHEN** the user runs the existing buck integration test
  through `run_transient`
- **THEN** the results SHALL be bit-identical to the V1
  behaviour (diode infrastructure is silent for diode-less
  circuits).

#### Scenario: Half-wave rectifier — diode auto-commutates

- **GIVEN** a sinusoidal source (V_amp = 10 V, f = 60 Hz) →
  diode → resistor (10 Ω) → GND
- **WHEN** the user runs a 2-cycle transient
- **THEN** at least 99 % of positive-half-cycle samples SHALL
  match the source voltage within 0.5 V
- **AND** at least 99 % of negative-half-cycle samples SHALL be
  within 0.1 V of zero
- **AND** the mean output power over a full cycle SHALL be
  within 5 % of `V_amp² / (4·R) = 2.5 W`.

Note: The boost converter integration test mentioned in the
proposal is DEFERRED to the follow-up
`pulsim-v2-event-detection` OpenSpec. The V0 per-step diode
state-decision logic chatters during the DCM/CCM boundary
transient with idealized switches, requiring sub-step
bisection to resolve correctly.
