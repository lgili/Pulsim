## Context

Review items: `raw/models.md` finding 4 and ranked item 8; synthesis M6.
`PmsmMna` is listed among the things that are already excellent
(flux-state, full L(θ), co-energy torque, mechanics as nodes); this
change extends that pattern to the other machines and does not alter
its equations for existing parameter sets.

Owner notes that bind this design:

- *flux-row-keep-conductance-form*: flux rows keep the `2/h` (or
  `c1/h`) factor; they are not rescaled into flux units.
- *newton-devices-need-smooth-laws-to-1e-12*: laws smooth and accurate
  to 1e-12, Jacobian the derivative of the evaluated expression.
- *nonlinear-refresh-wrapper-chain*: a new stateful kind added to
  today's chain must capture the accumulated refresh; the reason this
  change waits for the device interface for anything stateful.
- *motor-inductance-is-the-free-parameter*: a machine "validated"
  against data its parameters were tuned on is not validated.

## Goals / Non-Goals

- Goals: second-order machines solved with the circuit on `pwl` and
  `trbdf2`; a DC operating point; YAML; a deprecation path for the
  observers.
- Non-Goals: iron loss, cogging torque, slot harmonics; saturation of
  the induction machine's magnetising branch; wound-rotor and
  doubly-fed machines; synchronous-frame variants; arbitrary Python
  load callbacks inside the solve (a load is an element on the speed
  node); removing the observers in this change.

## Decisions

### Mechanical nodes

As in `PmsmMna`: the speed `ω` [rad/s] and the angle `θ` [rad] are node
voltages; torque [N·m] is current. Inertia is a capacitor `J` [kg·m²]
from the speed node to the reference, friction a resistor `1/B`, a
constant load torque a current source, a shaft coupling a wire. A
speed-dependent load is another element on the node.

### DC machine (linear composite)

```
a_pos ─ R_a ─ L_a ─ e ─[ secondary ]─ a_neg        v_e − v_a_neg = K·ω
                       ideal transformer, n = K
omega ─[ primary ]─ ref      torque K·i_a into omega
omega ─ C = J ─ ref,  omega ─ R = 1/B ─ ref,  I = T_load out of omega
```

- Equations: `v_a = R_a·i + L_a·di/dt + K·ω`,
  `J·dω/dt = K·i − B·ω − T_load`. Power is conserved by construction
  (`v_emf·i = K·ω·i = T·ω`).
- Only one constant: in SI units `K_e = K_t`. A legacy call with
  `Kt ≠ Ke` is refused by the native path with that explanation.
- Optional angle node: a 1 F capacitor charged by a unit-gain VCCS from
  `ω` (`add-controlled-source-set`).
- Engines: linear, so `pwl` uses the cached factor with no Newton;
  `trbdf2`; DC operating point (inductor short, capacitor open) gives
  `i = T_load/K`, `ω = (V − R_a·i)/K`; AC through the linear sweep.
  `dsed` refuses the ideal transformer today and stays refused.

### Induction machine (Newton device, rotor-flux states)

Stationary αβ frame, amplitude-invariant Clarke transform `T`, rotor
quantities referred to the stator. With `L_s = L_ls + L_m`,
`L_r = L_lr + L_m`, `σ = 1 − L_m²/(L_s·L_r)`, `τ_r = L_r/R_r`,
`ω_e = pp·ω`, `J₂ = [[0, −1], [1, 0]]`:

```
λ_s,αβ = σ·L_s·i_s,αβ + (L_m/L_r)·λ_r          λ_0 = L_ls·i_0
v_k − v_n = R_s·i_k + dλ_k/dt,   λ_abc = T⁻¹·[λ_sα, λ_sβ, λ_0]     (3 rows)
dλ_r/dt + λ_r/τ_r − (L_m/τ_r)·i_s,αβ − ω_e·J₂·λ_r = 0             (2 rows)
T_em = (3/2)·pp·(L_m/L_r)·(λ_rα·i_sβ − λ_rβ·i_sα)   into the speed node
```

- Unknowns: the three stator branch currents (as in `PmsmMna`) and two
  internal unknowns `λ_rα`, `λ_rβ`.
- In the device interface's terms: stator rows `q = λ_k`,
  `f = R_s·i_k − (v_k − v_n)`; rotor rows `q = λ_r`, `f` the remaining
  terms; speed node `f = −T_em`.
- Nonlinearity is bilinear only (`ω·λ_r`, `λ_r × i_s`): polynomial,
  C^∞, exact closed-form Jacobian, no special functions.
- Flux rows stay in conductance form.
- The rotor currents are eliminated, so a short-circuited cage needs no
  rotor nodes.

### BLDC and non-sinusoidal PM flux

`λ_pm,a(θ_e) = Σ_h ψ_h·cos(h·θ_e + φ_h)`, phases b and c shifted by
∓2π/3 in `θ_e`. The existing co-energy torque
`pp·(dλ_pm/dθ_e)ᵀ·i` applies unchanged.

- A **harmonic series, not a table**: the torque row needs `dλ/dθ` in
  the residual and `d²λ/dθ²` in the Jacobian. An ideal trapezoidal
  back-EMF has corners, so its flux is only C¹ and a table of it would
  give a discontinuous Jacobian. The truncated series is C^∞, periodic
  by construction and evaluated directly.
- Trapezoid with a 120° flat top (ramp half-width `γ = π/6`), back-EMF
  constant `k_e` [V·s/rad, phase, flat-top]:
  `ψ_h = (k_e/pp)·4·sin(h·γ)/(π·γ·h³)`, odd `h`. Truncated at the 25th
  harmonic the back-EMF's RMS deviation from the ideal trapezoid is
  0.40 % of the flat-top value (max 2.4 %, at the corners); at the 49th
  0.14 % (computed from the series).
- A measured back-EMF waveform is converted by FFT to `(h, ψ_h, φ_h)`.
- `ψ_1 = psi_pm`, no other harmonic, is bit-identical to today.

### Saturation: flux maps through a co-energy spline

`λ_d(i_d, i_q)`, `λ_q(i_d, i_q)` replace `L_d·i_d + ψ_pm`, `L_q·i_q`.

- The maps must be the gradient of a co-energy `W'(i_d, i_q)`
  (reciprocity, `∂λ_d/∂i_q = ∂λ_q/∂i_d`); otherwise the machine creates
  or destroys energy over a current cycle and the torque
  `(3/2)·pp·(λ_d·i_q − λ_q·i_d)` is inconsistent with the fluxes.
- Decision: fit a C² bicubic spline `W'` to both tables by least
  squares; use `λ = ∇W'` (C¹) and the Hessian (continuous, symmetric)
  as the incremental inductance matrix. The fit residual is reported; a
  residual above `tol` (default 2 % of the flux range) or a Hessian
  that is not positive definite at a knot is refused by name.
- The map replaces only the dq-frame constitutive law; the abc
  transformation, the flux rows and the mechanics are unchanged.

### DC operating point

Machines have dynamic mechanical rows (`dθ/dt = ω`), so a spinning
machine has no DC solution. Rule: the DC operating point holds every
machine's speed and angle nodes at their initial conditions and solves
the electrical rows with `dq/dt = 0`. For the linear DC machine the
ordinary DC solve already gives the steady speed and no hold is
applied.

### Engines

| Device | `pwl` | `trbdf2` | `dsed` | DC | AC |
|---|---|---|---|---|---|
| DC machine composite | cached LU | yes | refused (ideal transformer) | yes | linear sweep |
| Induction machine | Newton | Newton | refused | held mechanics | from f and q |
| PMSM with harmonics or maps | Newton | Newton | refused | held mechanics | from f and q |

### Alternatives considered

- *Keep the observers and raise their order (Heun, or a one-step
  extrapolation of the back-EMF).* Still explicit, still `pwl`-only,
  still invisible to steady-state solvers and the DC solve.
- *Induction machine in the synchronous frame.* Fewer harmonics in the
  states but the terminals stop being physical node voltages.
- *Rotor windings as circuit branches.* Three more unknowns and a
  singular cage when shorted; the rotor-flux form needs two.
- *PM flux as a periodic Hermite table.* Rejected (C¹ at best for a
  trapezoid; see above).
- *Independent splines for `λ_d` and `λ_q`.* Rejected: nothing enforces
  reciprocity, and measured maps are not exactly reciprocal.
- *Implementing the induction machine on today's structure.* Possible
  (a sixth history class and wrapper), rejected for the cost and the
  silent-failure risk the owner's wrapper-chain note records.

## Parameterisation & provenance

- DC machine: `R_a`, `L_a` measured at the terminals; `K` from the
  no-load speed or back-EMF constant; `J` from the datasheet or a
  run-down test.
- Induction machine: the five equivalent-circuit parameters from the
  no-load and locked-rotor tests. `im_parameters_from_nameplate`
  (`python/pulsim/motors.py:1158`) estimates them from a nameplate; its
  result is labelled `estimated`, and the documentation says a model
  built from it is not validated against that nameplate.
- PMSM: `L_d`, `L_q`, `psi_pm` measured or from the manufacturer; flux
  maps from FEA or a locked-rotor measurement, with the source recorded
  on the `FluxMapDQ` object.
- No machine parameter set ships with this change. The test machines
  below are declared test parameters, not real machines.

## Validation plan

| Check | Reference | Tolerance |
|---|---|---|
| DC machine, sine drive, dt = 12.5 µs | exact solution of the linear system (`p5b_dcmotor.py`) | 1e-4 (measured 7.45e-5) |
| DC machine, order: errors at 100, 50, 25, 12.5 µs | ratio 4 per halving (measured 3.99, 4.00, 4.00) | each ratio ≥ 3.5 |
| DC machine step response | `ω_ss = (V − R_a·T_load/K)/K`, time constants from the 2×2 system | 1e-4 |
| DC machine DC operating point, 10 V, K = 0.1, `T_load = 0.2 N·m`, `R_a = 0.5 Ω` | `i = 2 A`, `ω = 90 rad/s` | 1e-9 |
| Induction machine, test parameters (0.5 Ω, 2 mH, 0.4 Ω, 2 mH, 60 mH, pp = 2), 100 V rms phase, 60 Hz, speed node held | per-phase equivalent circuit | |
| — locked rotor | 57.88 A rms, `|Z| = 1.728 Ω ∠59.6°` | 0.5 % |
| — synchronous speed (188.50 rad/s) | 4.277 A rms, zero mean torque | 0.5 %, 0.01 N·m |
| — slip 0.05 (179.07 rad/s) | 16.12 N·m, 12.29 A rms | 0.5 % |
| Induction machine free acceleration | independent integration of the same equations (SciPy, rtol 1e-10) | 1e-4 on `trbdf2`; order ≥ 3.5 per halving on `pwl` |
| PMSM, `ψ_1` only | existing `PmsmMna` tests | bit-identical |
| BLDC back-EMF at constant speed, harmonics to the 25th | ideal trapezoid | RMS 0.5 % of flat top |
| Flux map built from a linear machine (`L_d`, `L_q`, `ψ_pm`) | the linear `PmsmMna` | 1e-9 |
| Flux map reciprocity | `∂λ_d/∂i_q = ∂λ_q/∂i_d` at 100 random points | 1e-9 |
| Engine agreement, each machine | `pwl` against `trbdf2` | 1e-3 |

Every row above is verification: analytic limits of the model or an
independent integration of the same equations. None is validation of a
machine. Validation needs measured data that was not used to set the
parameters (Open Questions).

## Risks / Trade-offs

- A capacitor of 1e-5 F (J) beside nanofarad electrical capacitors and
  a 1 F angle capacitor widens the matrix scale → the PMSM already
  does this; the conformance tests include a machine next to a
  switching stage.
- The DC machine composite adds two nodes and a branch unknown per
  machine → negligible.
- Changing the YAML default changes results → staged, warned, and the
  old path selectable.
- Users with `T_load_fn` callbacks have no native equivalent → the
  observers stay until a replacement (a C block or control-graph source
  on the speed node) is documented.
- A flux map with noise can fail the positive-definite check →
  refusal with the offending knot, and a `smooth=` option in the fit.

## Migration Plan

1. `add_dc_machine` composite, tests, binding, YAML `dc_machine`
   (today's structure).
2. After `add-unified-device-interface`: `InductionMachineMna`;
   `induction_machine` YAML; `induction_motor` gains `model:`.
3. PM-flux harmonics and `add_bldc_mna`; DC operating point with held
   mechanics; `pmsm`, `bldc` YAML types.
4. `FluxMapDQ` and saturation.
5. Deprecation warnings on the observers; one release later the YAML
   default flips (BREAKING).

Rollback: `model: observer`, or the legacy `add_*` functions, at every
step.

## Open Questions

- Does the owner have a machine with measured no-load, locked-rotor and
  load points (or a PMSM with a measured phase inductance) that could
  serve as held-out validation data?
- `Kt ≠ Ke` in legacy calls: refuse in the native path, or support it
  with a VCVS/CCCS pair at the price of a machine that does not
  conserve power?
- How long do the observers stay after the warning: one minor release,
  or until 3.0?
- Should the wound-field (separately and series excited) DC machine,
  which the existing `DC Motor Device` requirement names, be part of
  this change (a bilinear Newton device) or a follow-up?
