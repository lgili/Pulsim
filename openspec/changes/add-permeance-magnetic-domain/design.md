## Context

Review items: `raw/models.md` ranked item 10 and section (d); synthesis
M8. The existing magnetic models are winding-level (`λ(i)` in a flux
row). This change adds a core-level description beside them and reuses
their laws.

Owner notes that bind this design:

- *flux-row-keep-conductance-form*: the saturable inductor's row is not
  rescaled; that row and its devices are untouched here.
- *newton-devices-need-smooth-laws-to-1e-12*: every new law is smooth
  and accurate to 1e-12, with the Jacobian the derivative of the
  evaluated expression; each gets the picoampere-spacing monotonicity
  test.
- *shipped-parts-are-synthetic*: the four core files are illustrations
  whose parameters disagree between `magnetic.py`, `hysteresis.py` and
  the YAMLs; this change neither uses them as references nor edits
  them.
- *motor-inductance-is-the-free-parameter*: a loss model fitted to a
  datasheet loss curve is not validated by that curve.

## Goals / Non-Goals

- Goals: multi-path cores and integrated magnetics from a small set of
  primitives; saturation and hysteresis per core segment; eddy and
  excess loss inside the solve; equivalence with the T-model where both
  apply.
- Non-Goals: a library of core shapes (EE, EI, three-limb, five-limb
  helpers) — the primitives and two worked examples only; fringing and
  3-D leakage estimation; temperature-dependent material laws;
  frequency-dependent permeability; winding AC resistance
  (`add-realistic-passives`); replacing the T-model.

## Decisions

### Analogy

Permeance–capacitor (gyrator–capacitor):

| Magnetic | Network |
|---|---|
| magnetomotive force `F` [A] | node potential |
| flux rate `dΦ/dt` [V] | through variable |
| permeance `P` [H]: `Φ = P·F` | capacitor `C = P` |
| winding, N turns | gyrator `g = 1/N`: `i_e = F/N`, `dΦ/dt = v_e/N` |
| loss: `F = R_m·dΦ/dt` [S] | resistor |

`F·dΦ/dt` is power, so energy and loss bookkeeping are the network's
own. The resistance–reluctance analogy was rejected because its
"resistors" store energy and its losses have no element.

### Elements

- **Winding**: `add_gyrator(e_pos, e_neg, m_pos, m_neg, g = 1/N)`.
- **Linear permeance / air gap**: `add_capacitor` with `C = P`;
  `P_gap = μ₀·A/l_g`.
- **Saturable permeance**: charge-type device `q = Φ(F)` with the
  material law of `GappedCore` (`models/gapped_core.hpp:26-37`) made
  explicit by the segment having no gap of its own:

  ```
  H = F/l,  M_s = B_sat/μ₀,  H₀ = M_s/(μ_r0 − 1)
  Φ(F) = μ₀·A·[F/l + M_s·tanh(F/(l·H₀))]
  P(F) = dΦ/dF = (μ₀·A/l)·[1 + (μ_r0 − 1)/cosh²(F/(l·H₀))]
  ```

  Odd, C^∞, `P(F) > μ₀·A/l > 0`, `P(0) = μ₀·μ_r0·A/l`. Or a
  `FluxTable` with knots `(F_k, Φ_k, P_k)`, which already has exactly
  this symmetry.
  - Stamp: the flux-form companion,
    `dΦ/dt|_{n+1} = (2/h)·(Φ(F_{n+1}) − Φ(F_n)) − dΦ/dt|_n` (and the
    BDF2 stage form). It is the `NonlinearCapacitor` stamp with an odd
    law. Written on `Φ`, never on `P·dF/dt`: the flux through a node is
    then conserved exactly.
  - A gap is a separate linear permeance in series, so no inversion or
    sweep is needed; the network does what `GappedCore` does by
    sweeping.
- **Hysteretic permeance**: `Φ = μ₀·A·(H + M)`, `H = F/l`, `M` from
  the Jiles–Atherton state `(H, M)` with the exact tangent of
  `models/jiles_atherton.hpp`; state advanced in `accept` after each
  accepted step.
- **Magnetic resistance** (eddy): `F = R_m·dΦ/dt`, an ordinary
  resistor. For a laminated segment of conductivity `σ`, lamination
  thickness `d`, length `l`, area `A`: classical loss density
  `σ·d²·(dB/dt)²/12` gives `R_m = σ·d²·l/(12·A)`.
- **Excess loss**: loss density `k_x·|dB/dt|^1.5` gives
  `F = c·sign(dΦ/dt)·|dΦ/dt|^0.5` with `c = k_x·l/√A`, whose slope is
  infinite at zero. The element is a branch with the flux rate `u` as
  its unknown and the smooth law

  ```
  F(u) = c·u/(u² + ε²)^(1/4)
  dF/du = c·(u²/2 + ε²)/(u² + ε²)^(5/4)  > 0
  ```

  which equals the square-root law for `|u| ≫ ε` (−0.25 % in power at
  `|u| = 10·ε`) and is linear inside `ε`. `ε` [V] is a required
  parameter.
- Series order of a lossy segment: permeance — eddy resistance —
  excess element, between the segment's two magnetic nodes.

### Reference and preflight

Every magnetic island needs one node tied to the reference; the
gyrator does not connect the two sides galvanically. The builder keeps
a registry of magnetic nodes (those touched by a magnetic element);
the preflight refuses an island without a reference by name instead of
inserting its usual tie resistor, which would be a spurious magnetic
loss.

### DC operating point

At DC a capacitor is open, so a magnetic node whose branches are all
permeances has an empty row. The physical condition is flux
continuity. Rule: for each such node, the DC system uses the row
`Σ_k Φ_k(F) = Φ₀` (the net initial flux at the node, default 0) in
place of the rate row. A node with a winding keeps its rate row, which
at DC states that the winding voltage is zero. The result is the
magnetostatic reluctance-network solution, and the transient starts
from consistent fluxes.

### Engines

| Element | `pwl` | `trbdf2` | `dsed` | DC | AC |
|---|---|---|---|---|---|
| winding, linear permeance, air gap, eddy resistance | cached LU | yes | as the gyrator | flux rows | linear |
| saturable permeance | Newton | Newton | refused | flux rows (Newton) | at the operating point, via the device interface |
| hysteretic permeance, excess loss | Newton | Newton | refused | static law | via the device interface |

### Outputs

Per element: flux `Φ` (the charge of the permeance), `B = Φ/A`,
`H = F/l`, and for loss elements the power `F·dΦ/dt`. Hysteresis loss
of a JA permeance is the energy absorbed over a closed cycle,
`∮F·dΦ`.

### Alternatives considered

- *Extend the T-model with more magnetising branches.* Works for a
  specific core by duality but each topology needs its own derivation;
  the network does it by construction.
- *Reluctance–resistance network with controlled sources.* Needs an
  integrator per winding and has no natural loss element.
- *Loss as an equivalent resistor on the electrical side.* Correct for
  one winding on one path; ambiguous for multi-limb cores, where the
  loss belongs to a segment.
- *Excess loss with the raw square-root law.* Not C¹ at zero.
- *Excess loss as a voltage-controlled element `u(F) = F·|F|/c²`.*
  Smooth, but its conductance is zero at `F = 0`, which leaves the
  series node undetermined at every flux reversal.
- *A tie resistor for floating magnetic nodes.* Gives the wrong DC flux
  distribution and a trapped flux that never decays.

## Parameterisation & provenance

- Geometry (`A`, `l`, gap) from the core datasheet; `μ_r0` and `B_sat`
  from the material datasheet at the operating temperature, as
  `GappedCore` already documents.
- JA parameters from `fit_ja_from_bh_curve`
  (`python/pulsim/hysteresis.py:545`).
- `R_m`: from `σ`, `d` for laminations; for ferrites, where bulk eddy
  loss is not a lamination formula, fitted to loss data. `c`: fitted.
- A new fitter `fit_core_loss_elements(f, B, P_v, *, geometry)` returns
  `(R_m, c)` with its residual, and records which datasheet points it
  used. Points used in the fit are excluded from any reported
  agreement.
- No core data ships; the four synthetic core files are untouched.

## Validation plan

| Check | Reference | Tolerance |
|---|---|---|
| Single winding on a gapped core, linear: N = 25, A = 76 mm², l = 72 mm, gap 0.5 mm, μ_r0 = 2000 | `N²/(1/P_core + 1/P_gap)` = 111.36 µH (`P_core` = 2.653 µH, `P_gap` = 191.0 nH) | 1e-9 |
| Two-winding transformer, permeance form against `add_saturable_transformer` (T-model), below saturation, same grid | each other | 1e-6 |
| Same, driven to 3× the knee current | each other | 1e-3 (the T-model's table accuracy) |
| Saturated limit | `N²·μ₀·A/(l + l_g)` = 0.823 µH | 1 % |
| Three-limb core, N = 100, `P_L` = 5 µH, `P_air` = 50 nH | `L_1 = 50 mH`, `L_0 = 166.1 µH` | 1e-6 |
| Coupled inductor on an EE core, N = 10, outer legs 2 µH, centre leg 0.5 µH | `L = 111.1 µH`, `k = 0.8` | 1e-6 |
| Eddy resistance, sine: N = 10, 100 V peak, `R_m = 0.01 S` | `R_m·(V̂/N)²/2` = 0.500 W | 1e-4 |
| Excess loss, sine: `c = 0.01`, `ε = 0.01 V`, same drive | `c·(V̂/N)^1.5·0.5564` = 0.1759 W | 0.5 % |
| JA permeance against `add_hysteretic_core_inductor`, same core | each other | 1e-6 |
| Flux conservation at an internal node | `Σ Φ = 0` | 1e-12 of the largest flux |
| DC operating point of a core with a gap and a DC-biased winding | reluctance-network solution | 1e-9 |
| Engine agreement | `pwl` against `trbdf2` | 1e-4 |

`0.5564 = Γ(1.25)/(√π·Γ(1.75))` is the mean of `|cos|^1.5`.

All rows are analytic or cross-model checks. Core-loss validation needs
measured loss under non-sinusoidal excitation that was not used to fit
`R_m`, `c` or the JA parameters. Agreement with iGSE on a triangular
flux is a consistency check between two models built from the same
datasheet, and is reported as such. An external data set of measured
B–H loops under sine, triangular and trapezoidal excitation exists
(the Princeton MagNet data; general knowledge, licence to be checked);
using it is an owner decision.

## Risks / Trade-offs

- Capacitors of 1e-7 F with node potentials of hundreds of ampere-turns
  beside electrical nodes → scaling comparable to the mechanical nodes
  the PMSM already adds; covered by engine-agreement tests with a
  switching stage.
- The flux-conservation DC rows are a new kind of row in `dc_assemble`
  → limited to nodes detected as purely capacitive; the same rule is
  what the device interface needs for any q-only node, so it is
  written once.
- Users can build a network with a flux loop that has no winding and
  no loss (a floating "current" loop) → harmless in transient, and the
  preflight reports it.
- More unknowns than the T-model for the same two-winding transformer
  → the T-model stays the default helper.
- JA and excess elements wait for the device interface → the linear
  and saturable parts are useful alone.

## Migration Plan

1. After `add-controlled-source-set`: `add_winding`, `add_permeance`,
   `add_air_gap`, `add_magnetic_resistance`; preflight; DC flux rows;
   linear acceptance tests (three-limb, coupled inductor, T-model
   equivalence).
2. `add_saturable_permeance` on the charge-law variant; saturation
   tests against the T-model.
3. After `add-unified-device-interface`: `add_hysteretic_permeance`,
   `add_excess_loss`, outputs and loss rows, `fit_core_loss_elements`.
4. YAML, tutorials (three-limb transformer; integrated coupled
   inductor).

Nothing existing changes; rollback is removal.

## Open Questions

- May the MagNet measurements (or another measured data set the owner
  has) be used as the held-out validation reference for core loss?
- Should helpers for common core shapes follow in a separate change,
  or are worked examples enough?
- Should `add_saturable_transformer` gain a `model="permeance"` option
  once the equivalence is established, or stay T-model only?
- Unit of the magnetic resistance in the API: siemens (the network
  unit) or the lamination data (`sigma`, `d`) only?
