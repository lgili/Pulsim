## Context

Review: `raw/models.md` ranked item 13 ("Ideal passives; no line
model"; verification "impedance against the analytic curve"); synthesis
M12. The goal is ripple, loss and EMI-band realism from datasheet
numbers a designer already has, with the smallest kernel footprint.

Owner notes that bind this design:

- *newton-devices-need-smooth-laws-to-1e-12*: the DC-bias table is a
  Newton charge law and inherits the `ChargeTable` rules.
- *shipped-parts-are-synthetic*: no capacitor, inductor or cable data
  ships as if it were a part.
- *motor-inductance-is-the-free-parameter*: a ladder fitted to an
  `R_ac(f)` curve reproduces that curve by construction; that is a fit,
  not a validation.

## Goals / Non-Goals

- Goals: series parasitics on C and L; bias-dependent capacitance;
  frequency-dependent winding resistance; a line model that is exact in
  delay.
- Non-Goals: dielectric absorption, ageing and temperature dependence
  of capacitors; frequency-dependent ESR; core loss (the magnetic
  domain); multi-conductor and frequency-dependent cables (modal
  decomposition, skin-effect lines); common-mode propagation; proximity
  loss computed from geometry (Dowell) — the ladder is fitted to a
  curve the user supplies.

## Decisions

### ESR, ESL, DCR, parallel capacitance: composites

```
capacitor:  from ─ R(esr) ─ L(esl) ─ C ─ to        names: {name}.esr, {name}.esl, {name}
inductor:   from ─ R(esr) ─ L ─ to ,  C(c_par) from → to   names: {name}.esr, {name}, {name}.cpar
```

- Zero parameters add nothing: no internal node, same stamp, same
  result bits.
- `{name}` stays the ideal element, so `c0`, `i0`, `res.i(name)` and
  existing code keep their meaning; `c0` is the voltage of the ideal
  capacitor.
- `Z_C(f) = esr + j(2πf·esl − 1/(2πf·C))`, series resonance at
  `1/(2π√(esl·C))`. `Z_L`: `esr + j2πf·L` in parallel with `c_par`,
  self-resonance near `1/(2π√(L·c_par))`.
- Validation: each parameter finite and ≥ 0, error names the component.
- Loss: `i²·esr`, reported under the component.

Alternative considered: a single-branch series R–C companion
(`G_eq = 1/(esr + h/(2C))`) with no internal node. It saves a node but
needs a new stateful kind with its own history; rejected for the first
version.

### Capacitance against DC bias

- Law: odd `Q(v)`, even `C(v) > 0`, as knots `(v_k, Q_k, C_k)` for
  `v ≥ 0` extended as an odd function — the structure of `FluxTable`
  with `(i, λ, L) → (v, Q, C)`; linear tail with the last `C`.
- Stamped through the charge form of `NonlinearCapacitor`, so charge is
  conserved; `C(v)` is the exact derivative of the evaluated `Q(v)`.
- `fit_mlcc_dc_bias(v_bias, c_ratio, C_nominal)`: integrates
  `C_nominal·c_ratio(v)` (monotone interpolation) into the table and
  returns the residual on the input points.
- What the datasheet curve is: a small-signal capacitance at a DC
  bias. Integrating it is exact for the small-signal behaviour around
  any bias and an approximation for large swings (a real ceramic also
  has hysteresis). The documentation says so.

### AC resistance ladder

```
from ─ R_dc ─ L_hf ─ [R_1 ∥ L_1] ─ … ─ [R_n ∥ L_n] ─ to
R_ac(ω) = R_dc + Σ R_k·(ωτ_k)²/(1 + (ωτ_k)²),   τ_k = L_k/R_k
L(ω)    = L_hf + Σ L_k/(1 + (ωτ_k)²)
```

- Fit: corner frequencies `1/(2πτ_k)` log-spaced across the data band;
  non-negative least squares for `R_k`; `L_k = R_k·τ_k`;
  `L_hf = L − Σ L_k`, where `L` is the low-frequency inductance.
- Passive by construction (`R_k, L_k ≥ 0`). Refused by name: `L_hf ≤ 0`
  (the requested resistance rise needs more inductance than the
  component has), or a residual above `tol`.
- The inductance falls with frequency as the cells bypass, which is
  the physical direction (internal inductance decreases with skin
  depth).
- In the time domain the loss of every harmonic is accounted for by the
  network, with no post-processing.

Alternatives: a Cauer ladder (series L, shunt R) represents the same
impedance class and is harder to fit with a linear solve; a
frequency-domain loss post-processor cannot load the circuit.

### Transmission line

Two-conductor line, differential mode, characteristic impedance `Z0`,
one-way delay `τ`. From per-length data: `Z0 = √(L'/C')`,
`τ = len·√(L'·C')`.

**π-sections** (`model="pi"`): `N` cells, each a series `Z0·τ/N` with
`τ/(Z0·N)` shunt split at its ends; series resistance distributed.
Cut-off `N/(π·τ)`. Existing elements only, so every analysis works.
Its limit is the measurement in the proposal: an ideal edge overshoots
27 % at any `N`. `recommend_line_sections(delay, t_rise)` returns
`ceil(11·delay/t_rise)`, the ratio at which the probe measured 5 %
(50 sections for `t_rise = delay/5`); the rule is empirical, from two
measured points, and documented as such.

**Bergeron** (`model="bergeron"`), the method of characteristics for a
lossless line. With `i_k`, `i_m` the currents entering the line at each
end:

```
i_k(t) = v_k(t)/Z0 + I_k(t),   I_k(t) = −v_m(t − τ)/Z0 − i_m(t − τ)
i_m(t) = v_m(t)/Z0 + I_m(t),   I_m(t) = −v_k(t − τ)/Z0 − i_k(t − τ)
```

- Stamp: a constant conductance `1/Z0` across each port (mask
  independent, in the cached factor) and a history current source into
  `b` each step. The two ends are coupled only through the delay.
- State: a history buffer of `(v, i)` at both ends covering the last
  `τ`. The delayed value is the stored sample when `τ` is a multiple
  of the step, otherwise a linear interpolation between the two
  neighbouring samples.
- Constraint: the step must not exceed `τ` (the delayed value must
  already exist). The fixed engine refuses `dt > τ` by name; `trbdf2`
  caps its step at `τ` for circuits with a line.
- Loss: series resistance `R` lumped as `R/4` at each end and `R/2`
  between two half-length lossless lines. Valid for `R ≪ Z0`; a larger
  ratio is refused with a pointer to `model="pi"`.
- DC operating point: the line is its series resistance (a short when
  lossless); the history is initialised from that solution so the
  transient starts with no artificial wave.
- Not representable: a delay has no finite-dimensional LTI state, so
  `dsed`, the exact propagator and periodic-steady-state shooting
  refuse it; the AC sweep refuses it and names `model="pi"`.

| Element | `pwl` | `trbdf2` | `dsed` | DC | AC |
|---|---|---|---|---|---|
| ESR/ESL/c_par composites, R_ac ladder, π line | yes | yes | as R, L, C | yes | yes |
| DC-bias capacitor | Newton | Newton | refused | open | via the device interface |
| Bergeron line | yes (`dt ≤ τ`) | yes (step ≤ `τ`) | refused | series R | refused |

Alternatives: only π-sections (rejected: the 27 % artefact); a
frequency-dependent line model (out of scope; the Bergeron line is the
element such models are built on).

## Parameterisation & provenance

- `esr`, `esl`: the capacitor datasheet's ESR at the stated frequency
  and ESL (or the series-resonant frequency). ESR varies with
  frequency and temperature; one value is used and the documentation
  says at which frequency to read it.
- `c_par`: from the inductor's self-resonant frequency.
- DC-bias curve: the manufacturer's curve for the exact part, with its
  measurement condition recorded on the table object.
- `R_ac(f)`: measured with an impedance analyser, or computed by the
  user's winding tool. The ladder's source and residual are stored.
- `Z0`, `delay`, `R`: the cable datasheet (`L'`, `C'`, `R'`, length).
- Nothing ships as part data.

## Validation plan

All references are analytic except the two labelled measurements.

| Check | Expected | Tolerance |
|---|---|---|
| `esr = esl = 0`, `c_par = 0` | bit-identical to today | exact |
| Capacitor 100 µF, `esr` 10 mΩ, `esl` 10 nH, AC | minimum `|Z|` = 10 mΩ at 159.15 kHz | 1e-9 |
| Same, transient, sine drive at 20, 159.15 and 800 kHz | phasor amplitude of the terminal voltage | 1e-3 |
| Ripple: ±1 A square current into `esr` = 50 mΩ + 100 µF | 100 mV step at each edge | 1e-6 |
| Inductor 100 µH, `c_par` 10 pF, AC | impedance peak at 5.033 MHz | 1e-6 in frequency |
| ESR loss, 1 A rms through 50 mΩ | 50.0 mW under the component | 1e-9 |
| DC-bias table from `C0/(1 + (v/V1)²)`, `C0` = 10 µF, `V1` = 10 V, 64 knots | `Q(20 V)` = 110.71 µC; `C(20 V)` = 2.000 µF; energy 0.8047 mJ | 1e-4, 1e-3, 1e-4 |
| Charge conservation over a ±20 V cycle | zero net charge | 1e-9 |
| Ladder identity: `R_dc` 10 mΩ, one cell 90 mΩ, 143.24 nH | `R_ac` = 10.009, 55.00, 99.11 mΩ at 1 kHz, 100 kHz, 1 MHz | 1e-9 (AC) |
| Ladder loss, 1 A peak at 1 kHz plus 1 A peak at 100 kHz | 32.50 mW | 1e-3 |
| `fit_rac_ladder` round trip on ladder-generated data | parameters recovered | 1e-6 |
| `fit_rac_ladder` on the round-wire skin-effect curve (Bessel form), two decades around the skin corner | residual ≤ `tol` with the cell count reported | 5 % (target, to be confirmed by task 3.4) |
| π line, 50 Ω, 100 ns, 50 sections, open end, AC | `Z_in = −j·Z0·cot(ωτ)` = −j91.52 Ω at `ωτ = 0.5` | 1e-3 |
| π line step (measurement, regression value) | peak 1.271 V, 50 % crossing at 1.015·τ for 50 sections | 1 % |
| Bergeron line, matched source, open end, `τ/dt` integer | 0 V before `τ`, 1.000 V after, no overshoot | 1e-9 |
| Bergeron, source 10 Ω, open end | lattice: `v_far` = 1.667, 0.556, 1.296 V after `τ`, `3τ`, `5τ` | 1e-9 |
| Bergeron, `τ/dt` not an integer | same steps, interpolation error bounded | 2 % of the step |
| Bergeron on `trbdf2` | same lattice | 1e-3 |
| Bergeron DC | far end equals near end minus `R·I` | 1e-9 |

Lattice values: source reflection `Γ_s = (10 − 50)/(10 + 50) = −2/3`,
launch `50/60` V, open-end reflection `+1`:
`2·(5/6) = 1.667`, `1.667 − (2/3)·1.667 = 0.556`,
`0.556 + (4/9)·1.667 = 1.296`.

## Risks / Trade-offs

- ESL in series with a capacitor creates a resonance the fixed step
  may not resolve (ringing at the Nyquist rate) → the resolution
  diagnostic warns; documentation gives the step needed
  (`dt ≤ 1/(20·f_res)`).
- Internal nodes change the node count of circuits that use the new
  parameters → only when they are non-zero.
- Interpolated Bergeron history damps and disperses when `τ/dt` is not
  an integer → documented; `dt` is suggested as a divisor of `τ`.
- The lumped-resistance loss model is valid for `R ≪ Z0` only →
  refusal above a ratio, pointer to the π model.
- Three independent lines do not model a three-phase cable's coupling
  → stated in the limitations.
- A fitted ladder extrapolates outside its data band as its last cell
  dictates → the ladder stores its band and warns when the run's
  spectrum is known to exceed it (documentation only in the first
  version).

## Migration Plan

1. ESR, ESL, `c_par` composites; loss rows; YAML keys.
2. π-section line and `recommend_line_sections`.
3. `fit_rac_ladder` and `rac=`.
4. DC-bias table and `fit_mlcc_dc_bias` (after the charge-law variant).
5. Bergeron line: kernel kind, history, `pwl` then `trbdf2`,
   conformance refusals.

Each step is independent and releasable; nothing existing changes.

## Open Questions

- Is the single-pair Bergeron line enough for the cable-fed drive
  studies the owner has in mind, or is a coupled three-conductor line
  (modal decomposition) the real requirement?
- Should the Bergeron line be the default, with π-sections chosen
  automatically for the analyses that refuse a delay?
- ESR as one number, or a small `(f, ESR)` table collapsed at the
  switching frequency the user states?
- Is `c_par` on the inductor wanted now, or should the inductor get
  only `esr` in the first step?
