# Motor Models

> Status: shipped — mechanical primitives + Park/Clarke + PMSM + DC
> motor + PMSM-FOC current loop. Induction / BLDC / sensors are the
> natural follow-ups.

Motor models are the mechanical-electrical bridge of the simulator.
Phase 1.2's `add-motor-models` lays down the rotor-frame primitives
that every drive design needs:

| Layer | Header | Purpose |
|---|---|---|
| Mechanical primitives | `motors/mechanical.hpp` | `Shaft`, `GearBox`, `ConstantTorqueLoad`, `FanLoad`, `FlywheelLoad` |
| Frame transforms | `motors/frame_transforms.hpp` | Clarke / Park / inverse, plus composite `abc_to_dq` and `dq_to_abc` |
| PMSM | `motors/pmsm.hpp` | `Pmsm` device in dq-frame, including saliency (Ld ≠ Lq) for IPM machines |
| DC motor | `motors/dc_motor.hpp` | Separately-excited armature equations + closed-form steady-state speed |
| FOC current loop | `motors/pmsm_foc.hpp` | Cascaded PI for id / iq, auto-tuned from the motor's (Rs, Ld, Lq) and a target bandwidth |

Like the other Fase 1 / 2 changes, the motor layer is **header-only**.
Full Circuit-side integration (registering motors as
`Circuit::DeviceVariant` entries that stamp the right phase voltages
into MNA) lands in a follow-up; today the math objects compose freely
in user-space code and tests.

## TL;DR — PMSM-FOC current loop

```cpp
#include "pulsim/v1/motors/pmsm.hpp"
#include "pulsim/v1/motors/pmsm_foc.hpp"
using namespace pulsim::v1::motors;

PmsmParams motor_p{
    .Rs = 0.5, .Ld = 2e-3, .Lq = 2e-3, .psi_pm = 0.05,
    .pole_pairs = 4, .J = 1e-3, .b_friction = 1e-4,
};
Pmsm motor(motor_p);

// 500 Hz current-loop bandwidth, ±50 V output clamp.
PmsmFocCurrentLoop foc(motor_p, {.bandwidth_hz = 500.0,
                                  .Vd_min = -50, .Vd_max = 50,
                                  .Vq_min = -50, .Vq_max = 50});

const Real iq_ref = 5.0, id_ref = 0.0;     // 5 A torque-producing
const Real dt = 1e-5;                       // 100 kHz control loop
for (int k = 0; k < N_steps; ++k) {
    auto [Vd, Vq] = foc.step(id_ref, iq_ref,
                              motor.i_d(), motor.i_q(), dt);
    motor.step(Vd, Vq, /*tau_load*/0.0, dt);
}
```

The PI tuning `K_p = ω_c · L`, `K_i = K_p · R_s / L` cancels the plant
pole at `R_s/L` and places unity-gain crossover at `ω_c`. Both d-axis
and q-axis loops use the same recipe with their respective inductances.

## Mechanical primitives

### `Shaft`

Implements `J · dω/dt = τ_input − τ_load − b·ω − τ_coulomb·sign(ω)` via
forward-Euler. Used as the mechanical port for every motor model.

```cpp
Shaft sh{.J = 1e-3, .b_friction = 1e-3};
sh.advance(/*tau_net*/0.5, /*dt*/1e-5);
const Real omega = sh.omega;
```

Time constant: `τ_m = J / b`. After `5·τ_m` of constant input torque
the shaft settles within 1 % of `ω_ss = τ/b`.

### `GearBox`

Ideal speed reducer. `ratio = ω_in / ω_out`, `efficiency` ∈ (0, 1].

```cpp
GearBox gb{.ratio = 10.0, .efficiency = 0.95};
const Real omega_load_side = gb.omega_out(motor.omega_m());
const Real torque_load_side = gb.torque_out(motor_torque);
```

Reflecting load torque back to the motor side: `gb.reflect_load(τ_out) =
τ_out / (ratio · η)`.

### Loads

| Load | Math | Use |
|---|---|---|
| `ConstantTorqueLoad` | `τ_load = τ` | Industrial motor with a fixed mechanical brake |
| `FanLoad` | `τ_load = k · ω · |ω|` | Fans / blowers / propellers / quadratic torque profiles |
| `FlywheelLoad` | `τ_load = 0`, `J += J_extra` | Pure-inertia load (sums into the shaft's J) |

## Frame transforms

The Clarke / Park transforms convert between three coordinate frames
power-electronics engineers reach for hourly:

| Frame | Variables | Purpose |
|---|---|---|
| 3φ stationary | `(a, b, c)` | Inverter output / motor terminals |
| 2φ stationary | `(α, β)` | Compact transient analysis |
| 2φ rotating | `(d, q)` | dq-frame steady-state for FOC / control design |

Pulsim uses the **amplitude-invariant** convention (Clarke prefactor
`2/3`) — the same one Texas Instruments / STMicro motor-control
libraries use. Round-tripping `abc → dq → abc` at any rotor angle is
identity within numerical noise (pinned by the unit tests).

```cpp
// abc → dq at rotor angle θ_e
auto [d, q] = abc_to_dq(va, vb, vc, motor.theta_electrical());

// dq → abc (PWM modulator gets these phase references)
auto [Va, Vb, Vc] = dq_to_abc(Vd_ref, Vq_ref, motor.theta_electrical());
```

## PMSM

Standard rotor-frame equations:

```
v_d = R_s · i_d + L_d · di_d/dt − ω_e · L_q · i_q
v_q = R_s · i_q + L_q · di_q/dt + ω_e · (L_d · i_d + ψ_PM)
τ_em = (3/2) · p · (ψ_PM · i_q + (L_d − L_q) · i_d · i_q)
```

`L_d ≠ L_q` produces the reluctance torque term `(L_d − L_q)·i_d·i_q`
— relevant for interior-permanent-magnet (IPM) machines where saliency
is significant. Surface-PMSM / `L_d = L_q` reduces to the classical
`τ_em = (3/2)·p·ψ_PM·i_q`.

### No-load gate (G.4)

Spinning the unexcited rotor produces back-EMF
`v_back = ψ_PM · ω_e = ψ_PM · p · ω_m` per phase (peak). Pinned by
`Phase 3.4` in the test suite.

### Locked-rotor gate (G.4)

With ω_m forced to zero (huge inertia), applying `V_q` drives `i_q`
toward `V_q / R_s` on the L_q time constant. Pinned by `Phase 3.5`.

## DC motor

```
v_a  = R_a · i_a + L_a · di_a/dt + K_e · ω
τ_em = K_t · i_a
J · dω/dt = τ_em − τ_load − b · ω
```

Closed-form steady-state speed:

```
ω_ss = (V·K_t − τ_load·R_a) / (K_t·K_e + b·R_a)
```

Mechanical time constant: `τ_m ≈ J · R_a / (K_t · K_e)`. Both helpers
are exposed on the `DcMotor` class so the test gate ("speed step
matches first-order analytical within ≤ 5 %") is direct.

## PMSM-FOC current loop

`PmsmFocCurrentLoop` wraps two `PiCompensator` instances (from the
`add-converter-templates` change) with the canonical "pole-zero
cancellation" tuning. Bandwidth defaults to 1 kHz — fast enough for
typical PWM rates of 10–20 kHz.

| Tuning rule | Formula |
|---|---|
| `K_p_d` | `ω_c · L_d` |
| `K_i_d` | `K_p_d · R_s / L_d` |
| `K_p_q` | `ω_c · L_q` |
| `K_i_q` | `K_p_q · R_s / L_q` |

`retune(motor, foc_params)` rebuilds the gains from updated motor
parameters or a new bandwidth target — useful for adaptive control or
parameter identification flows.

The loop's `step(id_ref, iq_ref, id_meas, iq_meas, dt)` returns
`(Vd_ref, Vq_ref)` which the user feeds either into a PMSM model
directly (for control-loop bench tests) or into an inverse-Park +
Space-Vector-Modulator chain (for full PWM-on-inverter simulation).

## Validation

Six gate-level tests (all in `test_motor_models.cpp`):

| Gate | Test | Result |
|---|---|---|
| **G.1** Shaft / friction model | `Phase 1: shaft + flywheel + step torque` | ω_ss within ±2 % at 5·τ_m for J=b=1e-3 |
| **G.1** Park round-trip | `Phase 2: Clarke / Park identity` | Sinusoidal balanced 3φ → (d=1, q=0) within 1e-12 |
| **G.4** PMSM no-load | `Phase 3.4` | `back_emf_peak() == ψ_PM · p · ω_m` to machine precision |
| **G.4** PMSM locked-rotor | `Phase 3.5` | i_q converges to V_q / R_s within ±1 % over 0.5 s |
| **G.1** DC motor speed step | `Phase 5.2` | ω matches `(V·K_t − τ·R_a) / (K_t·K_e + b·R_a)` within ±5 % at 12·τ_m |
| **G.3** PMSM-FOC tracking | `Phase 7` | i_q tracks i_q_ref within ±5 % after ≈ 50 ms at 500 Hz bandwidth |

## Follow-ups

- **Induction motor** (`InductionMotorDevice`): rotor flux observer,
  slip computation, V/f start-up tutorial. Math is well-known; ships
  alongside the `add-three-phase-grid-library` change which provides
  the inverter side.
- **BLDC / trapezoidal back-EMF**: hall-sensor commutation table,
  6-step modulation. Tracked separately from the sinusoidal PMSM.
- **Sensor models**: `EncoderQuadrature`, `HallSensor`, `Resolver`.
  Encoder is straightforward; resolver wants a sinusoidal-modulator +
  demodulator pair that justifies its own change.
- **Saturation `L_d(i_d), L_q(i_q)`**: lookup-table extension of the
  PMSM's constant inductances. Lands when the magnetic catalog's
  `BHCurveTable` is wired through to motor stators.
- **Speed / position outer loops**: the FOC current loop is final;
  the cascaded speed PI + position-loop PI follow alongside the
  closed-loop benchmark suite (`add-closed-loop-benchmarks`).
- **`Circuit::DeviceVariant` integration**: register `Pmsm` /
  `DcMotor` so a YAML netlist can declare `type: pmsm, parameters:
  {Rs: ..., Ld: ..., ...}` and the existing parser dispatches to
  these models. Pairs with the `add-three-phase-grid-library` change.

## See also

- [`converter-templates.md`](converter-templates.md) — the PMSM-FOC
  loop's `PiCompensator` ships with the converter-templates change.
- [`ac-analysis.md`](ac-analysis.md) — when the motor + inverter is
  part of a closed loop, the AC sweep helper lets you verify the
  measured loop-gain margin against the FOC's design target.
