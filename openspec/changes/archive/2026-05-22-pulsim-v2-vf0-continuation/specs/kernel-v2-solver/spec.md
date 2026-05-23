## ADDED Requirements

### Requirement: make_vf0_override_refresh helper

`make_vf0_override_refresh(Real V_F0_override)` SHALL return a
`NonlinearRefreshFn` that stamps the smooth-blend `IdealDiode`
using `V_F0_override` for ALL diodes (instead of each diode's
pool-stored V_F0). Other parameters (R_d, G_off, kappa) MUST
come from the pool unchanged.

The helper enables V_F0 parameter sweeps and V_F0-component
continuation chains. V9 ships this as a composable primitive;
no claim is made that V_F0 continuation solves arbitrarily-
stiff problems from `x = 0`.

#### Scenario: Override refresh uses override V_F0

- **GIVEN** a `DevicePool` with one smooth-blend IdealDiode at
  pool-stored V_F0 = 0.7
- **AND** an override refresh built with V_F0_override = −5.0
- **WHEN** the user computes the residual at a known x using
  BOTH refreshes
- **THEN** the residual at the override refresh SHALL differ
  from the pool-default refresh (the sigmoid centre is shifted).

#### Scenario: V_F0 sweep on a DC load-line matches analytical

- **GIVEN** a DC diode load-line (V_dc=2V → smooth diode
  (κ=20) → R(1kΩ) → GND)
- **WHEN** the user evaluates the circuit with V_F0 overrides
  ∈ {0.3, 0.5, 0.7}
- **THEN** the converged `v_n1` SHALL match the analytical
  load-line `V_dc − V_F0` within 50 mV at each sweep point.

### Requirement: make_kappa_vf0_override_refresh helper

`make_kappa_vf0_override_refresh(Real kappa_override, Real V_F0_override)` SHALL return a `NonlinearRefreshFn` that stamps the smooth-blend `IdealDiode` using BOTH `kappa_override` AND `V_F0_override`.

Other parameters (R_d, G_off) MUST come from the pool unchanged.
The combined override is the building block for joint κ + V_F0
homotopy chains.

#### Scenario: Combined override differs from each single override

- **GIVEN** a smooth-blend IdealDiode with pool params
  (V_F0=0.7, κ=20) and a fixed test point x
- **AND** three refresh functions: κ-only override (κ=5),
  V_F0-only override (V_F0=0), and the combined override
  (κ=5, V_F0=0)
- **WHEN** the user computes the residual at x for each
- **THEN** the combined residual SHALL differ from BOTH
  single-override residuals (proving both parameter overrides
  take effect simultaneously).
