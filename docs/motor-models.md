# Motor Models

> **Status: shipped.** Seven motor families are wired all the way through
> `Circuit::DeviceVariant`, the YAML parser, and the Python bindings —
> DC, PMSM, PMSM-FOC (signal-domain controller), `MechanicalDevice`
> (shaft-only signal device for multi-shaft builds), BLDC (trapezoidal
> back-EMF), 3φ squirrel-cage Induction, and 1φ PSC Induction (the
> *compressor convencional* / CC motor used in legacy refrigeration
> compressors).

Motor models bridge the electrical and mechanical domains. Every model
is `pulsim::v1::DeviceVariant`-aware: the Circuit reserves MNA branch
rows, the stamping walker assembles the trapezoidal-companion or
back-EMF terms each step, and the advance walker integrates the
mechanical state forward.

## Quick chooser

| You want… | Use | YAML `type:` | Pins |
|---|---|---|---|
| Brushed DC motor (battery + armature) | `DcMotorDevice` | `dc_motor` | 2 |
| Sinusoidal-back-EMF PMSM (SPM / IPM) | `PmsmDevice` | `pmsm` | 4 (3φ + neutral) |
| Trapezoidal-back-EMF BLDC | `BldcMotorDevice` | `bldc_motor` | 4 |
| 3φ squirrel-cage induction motor | `InductionMotorDevice` | `induction_motor` | 4 |
| 1φ PSC induction motor for legacy compressors | `SinglePhaseInductionMotorDevice` | `single_phase_induction_motor` | 2 |
| FOC current loop (controller only, no electrical pins) | `PmsmFocDevice` | `pmsm_foc` | 0 |
| Standalone shaft / multi-shaft / pure-mechanical load | `MechanicalDevice` | `mechanical` | 0 |

Plus the compressor / refrigerant load layer that attaches to any of
the above by name — see [Compressor + Refrigerant Load](compressor-and-refrigerant-load.md).

## TL;DR — three end-to-end snippets

### Python (BLDC drive at 6 V on phase A)

```python
import pulsim as ps

ckt = ps.Circuit()
a = ckt.add_node("a"); b = ckt.add_node("b"); c = ckt.add_node("c")
n = ckt.add_node("n")
ckt.add_voltage_source("V_a", a, ps.Circuit.ground(), 6.0)
ckt.add_voltage_source("V_b", b, ps.Circuit.ground(), 0.0)
ckt.add_voltage_source("V_c", c, ps.Circuit.ground(), 0.0)
ckt.add_voltage_source("V_n", n, ps.Circuit.ground(), 0.0)

p = ps.BldcMotorParams()
p.R_s = 5.0; p.L_s = 8e-3; p.K_e_peak = 0.012
p.pole_pairs = 2; p.J = 5e-5; p.b_friction = 1e-5
ckt.add_bldc_motor("M1", a, b, c, n, p)

opts = ps.SimulationOptions()
opts.tstop = 5e-3
opts.dt = 1e-5
ps.Simulator(ckt, opts).run_transient()

print("ω =", ckt.bldc_omega("M1"), "rad/s")
print("τ_em =", ckt.bldc_torque("M1"), "N·m")
```

### YAML (220 V / 60 Hz 1φ PSC compressor motor)

```yaml
schema: pulsim-v1
version: 1
simulation:
  tstop: 0.2
  dt: 5e-5
components:
  - { type: sine_voltage_source, name: V_line, nodes: [line, 0],
      amplitude: 311.0, frequency: 60.0 }
  - { type: voltage_source, name: V_n, nodes: [0, 0], value: 0.0 }
  - type: single_phase_induction_motor
    name: M_cc
    nodes: [line, 0]
    R_s_main: 10.0
    L_s_main: 50e-3
    R_s_aux:  20.0
    L_s_aux:  80e-3
    C_run:    4e-6
    R_r:      8.0
    L_r:      55e-3
    L_m:      50e-3
    pole_pairs: 2
    J: 1e-4
```

### C++ (PMSM with the FOC current loop)

```cpp
#include "pulsim/v1/runtime_circuit.hpp"
using namespace pulsim::v1;

Circuit ckt;
auto a = ckt.add_node("a"); auto b = ckt.add_node("b");
auto c = ckt.add_node("c"); auto n = ckt.add_node("n");
// ... voltage sources ...

motors::PmsmParams motor_p{};
motor_p.Rs = 0.5; motor_p.Ld = motor_p.Lq = 2e-3;
motor_p.psi_pm = 0.05; motor_p.pole_pairs = 4;
motor_p.J = 1e-3; motor_p.b_friction = 1e-4;
ckt.add_pmsm("M1", a, b, c, n, motor_p);

PmsmFocDevice::Params foc_p{};
foc_p.motor = motor_p;
foc_p.foc.bandwidth_hz = 1000.0;
foc_p.foc.Vd_min = -50.0; foc_p.foc.Vd_max = 50.0;
foc_p.foc.Vq_min = -50.0; foc_p.foc.Vq_max = 50.0;
ckt.add_pmsm_foc("Ctrl1", foc_p);
```

---

## Architecture: math object + device wrapper

Every motor model has two layers:

| Layer | Purpose | Location |
|---|---|---|
| **Math object** | Standalone integrator — usable in isolation for control-design unit tests | `core/include/pulsim/v1/motors/*.hpp` |
| **Device wrapper** | `DeviceVariant` entry — handles MNA stamping, branch-row reservation, walker integration | `core/include/pulsim/v1/components/*_device.hpp` |

The math objects (`motors::Pmsm`, `motors::BldcMotor`,
`motors::InductionMotor`, `motors::DcMotor`,
`motors::SinglePhaseInductionMotor`) compose freely in user-space C++
code without involving the Circuit at all — useful for offline tuning
of FOC gains or for building custom benchmarks.

The device wrappers (`PmsmDevice`, `BldcMotorDevice`,
`InductionMotorDevice`, `DcMotorDevice`,
`SinglePhaseInductionMotorDevice`, `MechanicalDevice`, `PmsmFocDevice`)
implement `DynamicDeviceBase<DeviceT>` — they expose `stamp_impl`,
`jacobian_pattern_impl`, and `update_history_impl`, and the
`RuntimeCircuit` walker drives them through the canonical
DC-OP → transient → advance pipeline.

## Per-model reference

### DC motor (`dc_motor`)

Standard separately-excited armature model:

```
v_a  = R_a · i_a + L_a · di_a/dt + K_e · ω_m
τ_em = K_t · i_a
J · dω/dt = τ_em − τ_load − b · ω
```

**Pins**: `[armature+, armature−]`. The motor stamps as an
inductor-in-series-with-a-back-EMF source via trapezoidal companion;
the mechanical state is forward-Euler-integrated per accepted step.

| Field | Unit | Default | Note |
|---|---|---|---|
| `R_a` | Ω | 0.5 | Armature resistance |
| `L_a` | H | 1e-3 | Armature inductance |
| `K_e` | V·s/rad | 0.05 | Back-EMF constant |
| `K_t` | N·m/A | 0.05 | Torque constant (`K_e = K_t` for ideal motor) |
| `J` | kg·m² | 1e-4 | Rotor inertia |
| `b` | N·m·s | 1e-5 | Viscous friction |
| `tau_load_quad_coeff` | N·m·s² | 0 | Optional `τ = k·ω²` load |
| `omega_init` | rad/s | 0 | Initial speed |
| `i_a_init` | A | 0 | Initial armature current |

```python
ckt.set_dc_motor_tau_load("M1", 0.5)   # constant 0.5 N·m brake
ω = ckt.motor_omega("M1")
i = ckt.motor_armature_current("M1")
```

Closed-form steady-state speed:
`ω_ss = (V·K_t − τ_load·R_a) / (K_t·K_e + b·R_a)`.

### PMSM (`pmsm`) — sinusoidal back-EMF

Standard rotor-frame (αβ → dq) equations:

```
v_d = R_s·i_d + L_d·di_d/dt − ω_e · L_q · i_q
v_q = R_s·i_q + L_q·di_q/dt + ω_e · (L_d · i_d + ψ_PM)
τ_em = (3/2) · p · (ψ_PM · i_q + (L_d − L_q) · i_d · i_q)
```

`L_d ≠ L_q` produces the reluctance torque term — relevant for
**Interior Permanent Magnet (IPM)** machines where saliency is
significant. Surface PMSM (`L_d = L_q`) reduces to the classical
`τ_em = (3/2)·p·ψ_PM·i_q`.

**Pins**: `[A, B, C, N]` (4-wire wye). Each phase reserves one MNA
branch row stamped via per-phase trapezoidal companion + back-EMF
source.

| Field | Unit | Default |
|---|---|---|
| `Rs` | Ω | 0.5 |
| `Ld`, `Lq` | H | 2e-3 |
| `psi_pm` | Wb | 0.05 |
| `pole_pairs` | – | 4 |
| `J`, `b_friction` | kg·m², N·m·s | 1e-3, 1e-4 |
| `omega_init`, `theta_init` | rad/s, rad | 0, 0 |

```python
ckt.set_pmsm_tau_load("M1", 1.0)
ω = ckt.pmsm_omega("M1")
i_a = ckt.pmsm_i_a("M1")
```

### PMSM-FOC controller (`pmsm_foc`) — 0-pin signal device

Cascaded PI for `i_d` / `i_q` with **pole-zero cancellation** tuning
auto-derived from the motor's `(R_s, L_d, L_q)` and a target bandwidth:

| Gain | Formula |
|---|---|
| `K_p_d` | `ω_c · L_d` |
| `K_i_d` | `K_p_d · R_s / L_d` |
| `K_p_q` | `ω_c · L_q` |
| `K_i_q` | `K_p_q · R_s / L_q` |

The device has **zero electrical pins**: it's a signal-domain block
that reads the motor's `(i_d, i_q)` and produces `(V_d_ref, V_q_ref)`
each step. Feed those into an inverter / PWM modulator (or directly
into a PMSM in bench mode).

```yaml
- type: pmsm_foc
  name: Ctrl1
  Rs: 0.5
  Ld: 2e-3
  Lq: 2e-3
  psi_pm: 0.05
  pole_pairs: 4
  J: 1e-3
  b_friction: 1e-4
  bandwidth_hz: 1000
  Vd_min: -50
  Vd_max: 50
  Vq_min: -50
  Vq_max: 50
```

```python
ckt.set_pmsm_foc_iq_ref("Ctrl1", 5.0)   # 5 A torque-producing reference
ckt.set_pmsm_foc_id_ref("Ctrl1", 0.0)
Vd_ref = ckt.pmsm_foc_vd_ref("Ctrl1")
Vq_ref = ckt.pmsm_foc_vq_ref("Ctrl1")
```

### BLDC motor (`bldc_motor`) — trapezoidal back-EMF

Same 3φ pin layout as PMSM but back-EMF is **trapezoidal** (six 60°
flat-top segments per electrical revolution). Suited for 6-step
commutation drives common in low-cost BLDC controllers.

```
e_k(θ_e) = K_e_peak · trap(θ_e − φ_k)       # k ∈ {a, b, c}, φ_k = 0°, 120°, 240°
τ_em = (e_a·i_a + e_b·i_b + e_c·i_c) / ω_m
```

| Field | Unit | Default |
|---|---|---|
| `R_s` | Ω | 5.0 |
| `L_s` | H | 8e-3 |
| `K_e_peak` | V·s/rad | 0.012 |
| `pole_pairs` | – | 2 |
| `J` | kg·m² | 5e-5 |
| `b_friction`, `friction_coulomb` | N·m·s, N·m | 1e-5, 0 |

```python
ckt.set_bldc_tau_load("M1", 0.1)
ω = ckt.bldc_omega("M1")
τ = ckt.bldc_torque("M1")
```

### 3φ Induction motor (`induction_motor`) — squirrel cage

αβ-frame stationary model with rotor flux state `ψ_r = (ψ_rα, ψ_rβ)`:

```
v_α = R_s·i_α + σ·L_s·di_α/dt + (L_m/L_r) · dψ_rα/dt
v_β = R_s·i_β + σ·L_s·di_β/dt + (L_m/L_r) · dψ_rβ/dt
dψ_rα/dt = −R_r/L_r·ψ_rα + R_r·L_m/L_r·i_α − ω_e·ψ_rβ
dψ_rβ/dt = −R_r/L_r·ψ_rβ + R_r·L_m/L_r·i_β + ω_e·ψ_rα
τ_em = (3/2) · p · (L_m/L_r) · (ψ_rα·i_β − ψ_rβ·i_α)
```

`σ = 1 − L_m² / (L_s · L_r)` is the leakage coefficient.

**Pins**: `[A, B, C, N]`. Three MNA branch rows (per-phase trapezoidal
companion). At DC OP, rotor flux is zero → no back-EMF, so each phase
reduces to `R_s`.

| Field | Unit | Default |
|---|---|---|
| `R_s` | Ω | 1.0 |
| `R_r` | Ω | 1.5 |
| `L_s`, `L_r` | H | 0.15 |
| `L_m` | H | 0.14 |
| `pole_pairs` | – | 2 |
| `J`, `b_friction` | kg·m², N·m·s | 0.01, 1e-3 |

```python
ckt.set_induction_tau_load("M1", 2.0)
slip = ckt.induction_slip("M1", 2*math.pi*50)  # 50 Hz synchronous
ω = ckt.induction_omega("M1")
```

### Single-phase Induction motor (`single_phase_induction_motor`) — **NEW**

The *compressor convencional* (CC) motor used in pre-inverter domestic
refrigerators and freezers. Permanent Split Capacitor (PSC) topology:
main winding on α-axis, auxiliary winding on β-axis (90° spatial), with
a permanent run capacitor in series with the auxiliary winding. The
phase shift created by the cap turns a single-phase line voltage into
a quasi-2φ rotating field, enabling self-start without an external
inverter.

**Pins: 2** — `[line, neutral]`. The run capacitor and aux winding are
**internal to the device**. Two MNA branch rows (`i_main`, `i_aux`).

```
v_main = R_s_main·i_main + L_s_main·di_main/dt + (L_m/L_r)·dψ_rα/dt
v_aux − V_cap = R_s_aux·i_aux + L_s_aux·di_aux/dt + (L_m/L_r)·dψ_rβ/dt
dV_cap/dt = i_aux / C_run
dψ_rα/dt = −R_r/L_r·ψ_rα + R_r·L_m/L_r·i_main − ω_e·ψ_rβ
dψ_rβ/dt = −R_r/L_r·ψ_rβ + R_r·L_m/L_r·i_aux  + ω_e·ψ_rα
τ_em = p · (L_m/L_r) · (ψ_rα·i_aux − ψ_rβ·i_main)
```

| Field | Unit | Default | Note |
|---|---|---|---|
| `R_s_main` | Ω | 10.0 | Main winding resistance |
| `L_s_main` | H | 50e-3 | Main winding self-inductance |
| `R_s_aux` | Ω | 20.0 | Auxiliary winding resistance |
| `L_s_aux` | H | 80e-3 | Auxiliary winding self-inductance |
| `C_run` | F | 4e-6 | Permanent run capacitor in aux branch |
| `R_r` | Ω | 8.0 | Rotor resistance (referred to stator) |
| `L_r` | H | 55e-3 | Rotor self-inductance |
| `L_m` | H | 50e-3 | Mutual inductance |
| `pole_pairs` | – | 2 | 4-pole = 2 pole pairs (typical fridge) |
| `J` | kg·m² | 1e-4 | Rotor + shaft inertia |
| `b_friction` | N·m·s | 1e-4 | Viscous friction |
| `friction_coulomb` | N·m | 0.05 | Coulomb friction |

```python
import pulsim as ps

ckt = ps.Circuit()
line = ckt.add_node("line"); neutral = ckt.add_node("neutral")

# 220 V (RMS) / 60 Hz line.
ckt.add_sine_voltage_source("V_line", line, neutral,
                             220.0 * (2**0.5), 60.0)
ckt.add_voltage_source("V_n", neutral, ps.Circuit.ground(), 0.0)

# CC compressor motor with default Embraco-style PSC params.
p = ps.SinglePhaseInductionMotorParams()
ckt.add_single_phase_induction_motor("M_cc", line, neutral, p)

opts = ps.SimulationOptions()
opts.tstop = 0.2
opts.dt = 5e-5
ps.Simulator(ckt, opts).run_transient()

print("ω      =", ckt.single_phase_im_omega("M_cc"), "rad/s")
print("i_main =", ckt.single_phase_im_i_main("M_cc"), "A")
print("V_cap  =", ckt.single_phase_im_V_cap("M_cc"), "V")
print("τ_em   =", ckt.single_phase_im_torque("M_cc"), "N·m")
```

**Rotation direction is a wiring choice.** The PSC topology can rotate
in either direction depending on the relative phase produced by the
run capacitor; real CC compressors pick a direction by mechanical
winding sense. Tests should compare magnitudes, not signed values.

To attach a refrigeration-cycle load:

```python
comp = ps.compressor_defaults_for(ps.Refrigerant.R600a)
comp.displacement_m3 = 6e-6
ckt.attach_compressor_load("M_cc", comp)
```

See [Compressor + Refrigerant Load](compressor-and-refrigerant-load.md)
for the full compressor docs.

### MechanicalDevice (`mechanical`) — 0-pin shaft

Standalone shaft / inertia block with no electrical pins. Useful for
multi-shaft mechanical topologies (e.g. coupling two motors through a
gear, or modelling a flywheel) without forcing a motor into the
build.

```
J · dω_m/dt = τ_input − τ_load_const − τ_load_quad·ω·|ω|
              − b_friction·ω − τ_coulomb·sign(ω)
θ_m += dt · ω_m
```

```yaml
- type: mechanical
  name: Shaft1
  J: 1e-3
  b_friction: 1e-4
  tau_load_const: 0.5
```

```python
ckt.set_mechanical_tau_input("Shaft1", 1.0)
ω = ckt.mechanical_omega("Shaft1")
```

---

## Mechanical primitives (header-only)

The free-standing math helpers in `motors/mechanical.hpp` are still
the simplest way to build custom mechanical topologies in user-space:

| Helper | Math | Use |
|---|---|---|
| `Shaft` | `J·dω/dt = τ_net − b·ω − τ_C·sign(ω)` | Foundation block — used inside every motor |
| `GearBox` | `ω_out = ω_in / ratio`, `τ_out = τ_in · ratio · η` | Ideal speed reducer |
| `ConstantTorqueLoad` | `τ_load = τ` | Industrial brake / constant friction |
| `FanLoad` | `τ_load = k · ω · |ω|` | Quadratic torque profile (fans, blowers, props) |
| `FlywheelLoad` | `τ_load = 0`, `J += J_extra` | Pure-inertia load |

Compose them in user-space when you need a mechanical layout that
doesn't fit `MechanicalDevice` directly.

## Frame transforms

`motors/frame_transforms.hpp` provides Clarke / Park transforms in the
**amplitude-invariant** convention (Clarke prefactor `2/3`), matching
Texas Instruments / STMicro motor-control library conventions. Round-tripping `abc → dq → abc` at any rotor angle is identity within
machine precision (pinned by `test_motor_models.cpp`).

```cpp
auto [d, q] = motors::abc_to_dq(va, vb, vc, theta_e);
auto [Va, Vb, Vc] = motors::dq_to_abc(Vd_ref, Vq_ref, theta_e);
```

---

## Validation gates

Each motor model has at least one physics gate test in the suite. The
list is non-exhaustive — see `core/tests/test_*motor*` and
`test_pmsm_*` for the full coverage.

| Model | Gate | Test file |
|---|---|---|
| DC motor | Step response → analytical `ω_ss` within ±5 % | `test_dc_motor_device.cpp` |
| PMSM | No-load: `back_emf_peak = ψ_PM · p · ω_m` to FP precision | `test_motor_models.cpp` |
| PMSM | Locked-rotor: `i_q → V_q / R_s` within ±1 % | `test_motor_models.cpp` |
| PMSM | Full transient at 100 V / 200 rad/s → analytical match | `test_pmsm_dynamic.cpp` |
| PMSM-FOC | Bandwidth tracking: `i_q` follows `i_q_ref` within ±5 % at design BW | `test_pmsm_foc_device.cpp` |
| BLDC | Trapezoidal back-EMF shape vs analytical at no load | `test_bldc_motor_device.cpp` |
| BLDC | Full YAML benchmark — 6-step commutation under load | `test_bldc_benchmark_yaml.cpp` |
| Induction | Locked-rotor inrush current envelope | `test_induction_motor_device.cpp` |
| Induction | No-load: rotor accelerates toward synchronous | `test_induction_motor_device.cpp` |
| Induction | Loaded: slip stabilises at positive value | `test_induction_motor_device.cpp` |
| 1φ PSC IM | 60 Hz sine drive: currents bounded, rotor self-starts | `test_single_phase_induction_motor.cpp` |
| 1φ PSC IM | Circuit-level 220 V transient: rotor reaches sync magnitude | `test_single_phase_induction_motor.cpp` |
| MechanicalDevice | Shaft step → `ω_ss = τ/b` within ±2 % at 5·τ_m | `test_mechanical_device.cpp` |

## When to use which model

| Application | Suggested model |
|---|---|
| Bench-test a DC motor / brushed gearbox | `dc_motor` |
| FOC PMSM design / current-loop tuning | `pmsm` + `pmsm_foc` |
| BLDC controller bring-up (6-step commutation) | `bldc_motor` |
| Industrial 3φ AC motor on a grid / inverter | `induction_motor` |
| **Legacy fixed-frequency refrigerator / freezer compressor** | `single_phase_induction_motor` |
| Inverter-driven modern compressor (variable-speed) | `bldc_motor` or `pmsm` + `pmsm_foc` |
| Multi-shaft mechanical topology / no electrical pins | `mechanical` |
| Standalone control-loop / FOC bench without electrical model | `pmsm_foc` |

## See also

- **[Compressor + Refrigerant Load](compressor-and-refrigerant-load.md)**
  — attach a polytropic refrigeration-cycle load to any motor; covers
  the *compressor convencional* (CC) flow end-to-end.
- **[Three-Phase Grid Library](three-phase-grid.md)** — programmable
  3φ sources, harmonics, VSI inverter helpers that feed the motors.
- **[Converter Templates](converter-templates.md)** — PI compensators
  shared by the FOC current loop.
- **[Components Reference](components-reference.md)** — full YAML
  component table.
