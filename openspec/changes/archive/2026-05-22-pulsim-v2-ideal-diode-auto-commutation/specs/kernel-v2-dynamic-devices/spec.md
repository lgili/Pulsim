## ADDED Requirements

### Requirement: IdealDiode Device Model

`pulsim::v2::models::IdealDiode` SHALL be a 2-terminal device
representing an ideal diode with binary on/off states. The
device's stamping is identical to `IdealSwitch` — `g_on` between
terminals when ON, `g_off` when OFF — but the device EXPOSES a
state-decision helper that Layer 5 V2 uses to auto-commutate
the diode based on circuit conditions.

The struct SHALL expose:

```cpp
struct IdealDiode {
    struct Params {
        Real g_on;   // forward conductance
        Real g_off;  // reverse conductance (≪ g_on)
        Real V_th;   // forward threshold voltage (V_th = 0 for
                     // perfectly ideal; 0.7 for Si-behavioral)
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Switch;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_dynamic = false;
    static constexpr bool is_linear  = true;

    /// Decide the NEXT step's state given the current step's
    /// state + (v_diode, i_diode) at the just-computed solution.
    ///
    /// Transition rules:
    ///   OFF → ON  iff v_diode ≥ V_th
    ///   ON  → OFF iff i_diode ≤ 0
    ///   Otherwise unchanged.
    static bool decide_next_state(bool currently_on,
                                    Real v_diode,
                                    Real i_diode,
                                    const Params& p) noexcept;

    /// Static current contract (returns 0 — switches are
    /// stamped via stamp_switch_fixed, not stamp_device).
    template <numeric::FloatingPoint S>
    static constexpr S current(const S* /*v*/,
                                const Params& /*p*/) noexcept;
};
```

#### Scenario: OFF + forward bias above threshold → ON

- **GIVEN** an IdealDiode with `V_th = 0`, currently OFF
- **WHEN** the user calls `decide_next_state(false, v_diode =
  +0.5, i_diode = 0, p)`
- **THEN** the result SHALL be `true` (turn ON).

#### Scenario: OFF + forward bias below threshold → stays OFF

- **GIVEN** an IdealDiode with `V_th = 0.7`, currently OFF
- **WHEN** the user calls `decide_next_state(false, v_diode =
  +0.5, i_diode = 0, p)`
- **THEN** the result SHALL be `false` (stay OFF).

#### Scenario: ON + reverse current → OFF

- **GIVEN** an IdealDiode currently ON
- **WHEN** the user calls `decide_next_state(true, v_diode =
  +1.0, i_diode = -0.1, p)`
- **THEN** the result SHALL be `false` (turn OFF).

#### Scenario: ON + forward current → stays ON

- **GIVEN** an IdealDiode currently ON
- **WHEN** the user calls `decide_next_state(true, v_diode =
  +1.0, i_diode = +0.5, p)`
- **THEN** the result SHALL be `true` (stay ON).
