# Compressor + Refrigerant Load

> **Status: shipped.** Refrigeration / heat-pump compressor torque
> models for the three dominant hermetic-compressor topologies
> (Reciprocating, Rotary, Scroll), with a curated refrigerant-property
> table (R600a, R134a, R290, R32, R744) and YAML-first wiring through
> the `compressor_load` component type.

Domestic refrigerators, freezers, and heat-pump water-heaters share a
common topology: a hermetic compressor (the motor and the
gas-compression mechanism live inside the same sealed shell) driven by
either a fixed-frequency line-voltage motor (the *compressor
convencional* — CC) or by a variable-frequency inverter feeding a BLDC
/ PMSM (the *VCC* / inverter-compressor — VCC).

Pulsim's compressor layer separates the **mechanical load** (what the
gas-compression mechanism demands from the shaft) from the **electric
motor** (which provides the shaft torque). You attach a
`compressor_load` to any registered motor by name; the simulation
walker pushes the compressor's angle / speed dependent torque demand
into the motor's `set_load_torque(...)` setter each accepted step.

## Quick start — typical Embraco R600a domestic fridge

### YAML

```yaml
schema: pulsim-v1
version: 1
simulation:
  tstop: 1.0
  dt: 5e-5
components:
  # Line supply
  - { type: sine_voltage_source, name: V_line, nodes: [line, 0],
      amplitude: 311.0, frequency: 60.0 }
  - { type: voltage_source, name: V_n, nodes: [0, 0], value: 0.0 }

  # Compressor motor — fixed-frequency CC convencional (1φ PSC IM)
  - type: single_phase_induction_motor
    name: M_compressor
    nodes: [line, 0]
    pole_pairs: 2          # 4-pole at 60 Hz → ω_sync ≈ 188.5 rad/s

  # Mechanical load — Reciprocating piston compressor with R600a
  - type: compressor_load
    name: COMP1
    motor: M_compressor
    refrigerant: R600a            # seeds polytropic_n + cycle pressures
    topology: Reciprocating
    displacement_m3: 6.0e-6       # 6 cm³ per revolution
    # P_suction_Pa / P_discharge_Pa inherited from R600a table
```

### Python

```python
import pulsim as ps

ckt = ps.Circuit()
line = ckt.add_node("line")

ckt.add_sine_voltage_source("V_line", line, ps.Circuit.ground(),
                             220.0 * (2**0.5), 60.0)

# 1φ PSC CC compressor motor (defaults are Embraco-style).
motor_p = ps.SinglePhaseInductionMotorParams()
ckt.add_single_phase_induction_motor(
    "M_compressor", line, ps.Circuit.ground(), motor_p)

# Refrigeration load — start from the R600a refrigerant defaults,
# override only the per-machine fields (displacement, topology).
comp = ps.compressor_defaults_for(ps.Refrigerant.R600a)
comp.topology = ps.CompressorTopology.Reciprocating
comp.displacement_m3 = 6e-6
ckt.attach_compressor_load("M_compressor", comp)

# Inspect the static analytical numbers BEFORE running:
print(f"Mean compression torque: {ckt.compressor_mean_torque('M_compressor'):.3f} N·m")
print(f"Indicated work / cycle:  {ckt.compressor_indicated_work('M_compressor'):.3f} J")

opts = ps.SimulationOptions()
opts.tstop = 1.0
opts.dt    = 5e-5
ps.Simulator(ckt, opts).run_transient()
```

---

## Architecture

```
 ┌─────────────────────────┐         ┌──────────────────────┐
 │  Refrigerant table      │         │  CompressorParams    │
 │  (R600a, R134a, R290,   │  seeds  │  (topology, V_d,     │
 │   R32, R744)            │ ──────► │   P_suc, P_disch,    │
 │  - polytropic_n         │         │   polytropic_n, ...) │
 │  - typical pressures    │         └────────┬─────────────┘
 │  - critical constants   │                  │
 └─────────────────────────┘                  │ ctor / set_params
                                              ▼
                                  ┌──────────────────────┐
                                  │  CompressorLoad      │
                                  │  load_torque(θ, ω)   │
                                  └────────┬─────────────┘
                                           │ attach_compressor_load(motor_name, params)
                                           ▼
                          ┌────────────────────────────────────┐
                          │  Circuit::compressor_loads_        │
                          │  (motor_name → CompressorLoad)     │
                          └────────────────┬───────────────────┘
                                           │
                                           │ Walker, every accepted step:
                                           │   dev.set_load_torque(
                                           │       load.load_torque(θ, ω))
                                           ▼
                                  ┌──────────────────────┐
                                  │  Motor (BLDC, PMSM,  │
                                  │   3φ IM, 1φ PSC,     │
                                  │   DC, Mechanical)    │
                                  └──────────────────────┘
```

The compressor load lives in a side-table on the Circuit — not in the
`DeviceVariant` itself. This keeps the mechanical layer composable
across motor families: any motor type that exposes a `set_load_torque`
slot can host a compressor load, and the same `CompressorParams` /
`CompressorLoad` shape works for all of them.

---

## Topologies

The `CompressorTopology` enum covers the three dominant hermetic
refrigeration compressor families:

### `Reciprocating` (piston)

The classic Embraco / Secop / Aspera "compressor convencional" — and
the inverter "VCC" variant. A piston driven by a crank from the rotor
shaft pulls refrigerant from the suction port during one half of the
revolution and pushes it out the discharge port during the other half.

**Torque shape:** strong ripple with `2·N` peaks per revolution
(where `N = num_cylinders`), each corresponding to a
suction → compression → discharge stroke.

```
T(θ) = T_mean · (1 + α · cos(2·N·θ))
```

Default `α = 0.5` produces 30–70 % torque ripple, matching real
single-cylinder fridge compressors.

### `Rotary` (rolling-piston)

An eccentric rotor in a fixed chamber. Used in some inverter freezer
and air-conditioning units. Smoother torque, smaller ripple than
reciprocating.

```
T(θ) = T_mean · (1 + 0.2·α · cos(2·θ))
```

### `Scroll`

Two interleaved spirals. Near-constant torque with small
high-frequency ripple. Common in commercial chillers and some
high-end heat pumps; rare in domestic appliances.

```
T(θ) = T_mean · (1 + 0.05·α · cos(8·θ))
```

## Polytropic physics

The **mean compression torque** per revolution is derived from the
polytropic indicated work:

```
W_ind  = P_suction · V_d / (n − 1) · [(P_discharge / P_suction)^((n−1)/n) − 1]
T_mean = W_ind · num_cylinders / (2π)
```

For the isothermal limit `n = 1` (rare in real compressors), the
formula degenerates to `W_ind = P_s · V_d · ln(P_d / P_s)`.

The angle-dependent torque is the mean plus a topology-specific
ripple term, plus a Newton friction term:

```
T_friction(ω) = b_friction · ω + tau_coulomb · sign(ω)
T_load(θ, ω) = T_compression(θ) + T_friction(ω)
```

---

## Refrigerant table

The `Refrigerant` enum and `refrigerant(...)` lookup table cover the
five fluids that account for ≈ 95 % of domestic + commercial
refrigeration:

| Refrigerant | `polytropic_n` | Typical P_s | Typical P_d | T_crit | Use |
|---|---|---|---|---|---|
| **R600a** (isobutane) | 1.13 | 0.59 bar | 5.30 bar | 134.7 °C | **Modern domestic fridge / freezer** (post-2015 EU + LATAM) |
| **R134a** (1,1,1,2-tetrafluoroethane) | 1.30 | 1.64 bar | 10.17 bar | 101.1 °C | Legacy domestic / automotive AC |
| **R290** (propane) | 1.18 | 2.03 bar | 13.69 bar | 96.7 °C | EU domestic freezer / commercial chiller |
| **R32** (difluoromethane) | 1.30 | 9.51 bar | 27.78 bar | 78.1 °C | Residential split AC, HFC blends |
| **R744** (CO₂) | 1.30 | 40.0 bar | 90.0 bar | 31.0 °C | **Transcritical** heat-pump water heater / commercial cascade |

Values are typical defaults at design evaporator / condenser
temperatures — real systems vary with ambient, refrigerant charge,
and design margin. Users should override the cycle pressures when
they have system-level data.

### Python

```python
import pulsim as ps

# 1. Just read the curated properties.
props = ps.refrigerant(ps.Refrigerant.R600a)
print(props.polytropic_n)            # 1.13
print(props.typical_P_suction_Pa)    # 59000.0
print(props.critical_temperature_K)  # 407.85

# 2. Build a CompressorParams pre-seeded for a refrigerant.
p = ps.compressor_defaults_for(ps.Refrigerant.R290)
p.displacement_m3 = 12e-6            # override only what changes
p.topology = ps.CompressorTopology.Rotary

# 3. Swap refrigerants in-place on an existing CompressorParams.
ps.apply_refrigerant(p, ps.Refrigerant.R134a)   # n + pressures only
```

### C++

```cpp
#include "pulsim/v1/loads/refrigerants.hpp"
#include "pulsim/v1/loads/compressor_load.hpp"
using namespace pulsim::v1;

// Lookup
constexpr auto r290 = loads::refrigerant(loads::Refrigerant::R290);
static_assert(r290.polytropic_n == 1.18);   // constexpr-friendly

// Build params from refrigerant
auto params = loads::compressor_defaults_for(loads::Refrigerant::R600a);
params.displacement_m3 = 6e-6;
ckt.attach_compressor_load("M_compressor", params);

// Or apply in-place
loads::apply_refrigerant(params, loads::Refrigerant::R134a);
```

### YAML

```yaml
- type: compressor_load
  name: COMP1
  motor: M_compressor
  refrigerant: R600a       # seeds polytropic_n + typical pressures
  displacement_m3: 6.0e-6
```

Override per-field after the refrigerant seed:

```yaml
- type: compressor_load
  name: COMP1
  motor: M_compressor
  refrigerant: R134a
  P_suction_Pa: 1.20e5     # custom evap pressure
  P_discharge_Pa: 11.0e5   # custom cond pressure
```

---

## `CompressorParams` reference

| Field | Unit | Default | Meaning |
|---|---|---|---|
| `topology` | enum | `Reciprocating` | One of `Reciprocating`, `Rotary`, `Scroll` |
| `num_cylinders` | – | 1 | Cylinders / chambers per revolution |
| `displacement_m3` | m³ | 6.0e-6 | Swept volume per revolution (6 cm³ typical fridge) |
| `P_suction_Pa` | Pa | 7.0e4 | Low-side absolute pressure |
| `P_discharge_Pa` | Pa | 8.0e5 | High-side absolute pressure |
| `polytropic_n` | – | 1.13 | Compression polytropic exponent (R600a default) |
| `b_friction` | N·m·s | 1e-3 | Viscous friction on the shaft |
| `tau_coulomb` | N·m | 0.05 | Coulomb friction torque |
| `ripple_amplitude` | – | 0.5 | Torque ripple as a fraction of `T_mean`, 0..1 |

Set `ripple_amplitude = 0` to get a constant-torque idealization (useful for control-loop sanity tests).

## YAML `compressor_load` reference

| YAML field | C++ analog | Required |
|---|---|---|
| `motor` | (target motor name) | **Yes** |
| `refrigerant` | seed via `apply_refrigerant` | No (defaults to R600a) |
| `topology` | `topology` | No |
| `num_cylinders` | `num_cylinders` | No |
| `displacement_m3` | `displacement_m3` | No |
| `P_suction_Pa` | `P_suction_Pa` | No (refrigerant-seeded) |
| `P_discharge_Pa` | `P_discharge_Pa` | No (refrigerant-seeded) |
| `polytropic_n` | `polytropic_n` | No (refrigerant-seeded) |
| `b_friction` | `b_friction` | No |
| `tau_coulomb` | `tau_coulomb` | No |
| `ripple_amplitude` | `ripple_amplitude` | No |

Aliases: `compressor`, `refrigeration_compressor` all resolve to
`compressor_load`.

---

## Circuit API

```cpp
// Attach (replaces any existing load on the motor).
ckt.attach_compressor_load("M_compressor", params);

// Detach.
ckt.detach_compressor_load("M_compressor");

// Analytical introspection (returns NaN if no load attached).
Real T_mean = ckt.compressor_mean_torque("M_compressor");
Real W_ind  = ckt.compressor_indicated_work("M_compressor");

// Evaluate the angle/speed dependent torque demand (analytical).
Real T_inst = ckt.compressor_load_torque("M_compressor",
                                          theta_m, omega_m);
```

Python mirrors all of these on the `Circuit` class.

---

## Choosing the motor

| Compressor variant | Motor model | Notes |
|---|---|---|
| Pre-inverter domestic fridge / freezer ("CC convencional") | `single_phase_induction_motor` (PSC) | Fixed 50 / 60 Hz line, motor self-starts via run cap |
| Modern inverter fridge / freezer ("VCC") | `bldc_motor` | Variable-frequency 6-step / sinusoidal drive |
| Commercial chiller / heat pump | `pmsm` + `pmsm_foc` | FOC controller with current loops |
| Cascade R744 commercial supermarket | `pmsm` + `pmsm_foc` | Same as chillers, just R744 refrigerant |
| Reciprocating air compressor (non-refrigeration) | `induction_motor` (3φ) | Industrial 3φ grid feed |

Any of these motor types can host a `compressor_load` of any topology
— the load just supplies the shaft torque demand.

---

## Validation

Test files:

| Concern | Test file | Cases |
|---|---|---|
| Polytropic formula match | `test_compressor_load.cpp` | 1 |
| Per-topology ripple shape (Reciprocating / Rotary / Scroll) | `test_compressor_load.cpp` | 3 |
| Friction terms (viscous + Coulomb) | `test_compressor_load.cpp` | 1 |
| End-to-end Circuit on BLDC motor | `test_compressor_load.cpp` | 1 |
| Refrigerant table values (5 refrigerants) | `test_refrigerants.cpp` | 1 |
| String round-trip / refrigerant_from_string | `test_refrigerants.cpp` | 1 |
| `compressor_defaults_for(...)` | `test_refrigerants.cpp` | 1 |
| `apply_refrigerant(...)` preserves non-refrigerant fields | `test_refrigerants.cpp` | 1 |
| YAML parse: `single_phase_induction_motor` | `test_yaml_compressor_load.cpp` | 1 |
| YAML parse: `compressor_load` with explicit fields | `test_yaml_compressor_load.cpp` | 1 |
| YAML parse: defaults to R600a refrigerant | `test_yaml_compressor_load.cpp` | 1 |
| YAML parse: rejects missing `motor:` field | `test_yaml_compressor_load.cpp` | 1 |
| YAML parse: `refrigeration_compressor` alias + R290 swap | `test_yaml_compressor_load.cpp` | 1 |

Run them with:

```bash
ctest --test-dir build -R "compressor_load|refrigerant|single_phase_im"
```

---

## Limitations / future work

- **Pressure dynamics not modelled.** Suction and discharge pressures
  are treated as constants on the time scale of one revolution. Real
  systems show pressure-pulsation transients at the cylinder ports —
  out of scope for the mechanical-torque view we're modelling.
- **Constant volumetric efficiency assumed.** The displacement
  `V_d` is the swept volume; we ignore re-expansion losses from
  clearance volume. For accurate COP estimates, multiply
  `indicated_work_per_cycle()` by a user-supplied η_vol.
- **No reverse-flow / startup pressure equalization.** The model
  assumes the compressor is already running at design `P_s` / `P_d`.
  Cold-start scenarios (where high-side pressure equalizes through
  the cap-tube) need a separate pre-charge ramp on the user side.
- **No oil / acoustic dynamics.** Sound, vibration, oil-foaming are
  outside the simulator's scope.

---

## See also

- **[Motor Models](motor-models.md)** — full motor family reference
  including the 1φ PSC IM (CC motor) and BLDC (VCC motor).
- **[Components Reference](components-reference.md)** — YAML
  components table.
- **[Three-Phase Grid Library](three-phase-grid.md)** — programmable
  3φ sources / VSI inverter for commercial-compressor demos.
