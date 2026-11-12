## Context

The review (`raw/models.md`, finding 3 and items 4, 12, 14; synthesis
items M2, M10, M11) found that the device-level transient works and
that two engines agree on it to 0.06 %, but that it is reachable only by
hand composition with linear capacitors and invented parameters. The
probe itself states its limit: "the double-pulse test used arbitrary
parameters, not a real part".

Owner notes that bind this design:

- *shipped-parts-are-synthetic*: the shipped YAMLs are illustrations;
  no file may claim `transcribed` data it does not carry, and real data
  enters through an importer or a user directory.
- *newton-devices-need-smooth-laws-to-1e-12*: a Newton law must be
  smooth to about 1e-12 relative and its Jacobian must be the derivative
  of the exact expression the residual evaluates.
- *motor-inductance-is-the-free-parameter*: a model checked against the
  data it was tuned on is not validated.
- *nonlinear-refresh-wrapper-chain*: every stateful kind added to the
  wrapper chain is a silent-failure risk.

## Goals / Non-Goals

- Goals: one call builds a power MOSFET from datasheet curves; SPICE
  VDMOS cards import with their charge; each shipped part is checked
  against held-out datasheet switching energies; the Newton devices get
  an ngspice reference.
- Non-Goals: p-channel devices; live `T_j` inside the Newton law
  (arrives with `add-in-kernel-event-losses` and the device interface);
  drain inductance, Kelvin-source pins and package RLC networks;
  avalanche, short-circuit and SiC/GaN-specific effects (dynamic
  R_DS(on), gate-threshold drift); a datasheet IGBT (same structure,
  separate change once this one is proven); the PWL body-diode drop
  (Phase 0).

## Decisions

### Topology

```
D ─ R_d ─ d' ─┬─ channel (d' → s')      G ─ R_g ─ g'
              ├─ C_ds(v_d's')           C_gd(v_d'g') between d' and g'
              └─ body diode (s' → d')   C_gs between g' and s'
s' ─ L_s ─ S
```

Internal nodes exist only when the element is non-zero (`R_d = 0` ties
`d'` to `D`, and so on). Sub-element names follow the importer's
convention: `{name}` (channel), `{name}.rd`, `{name}.rg`, `{name}.ls`,
`{name}.cgs`, `{name}.cgd`, `{name}.cds`, `{name}_body`.

### Parameters

| Name | Unit | Meaning | Source |
|---|---|---|---|
| `K`, `V_T`, `lambda_`, `kappa` | A/V², V, 1/V, 1/V | Level-1 channel (existing law, third quadrant by terminal swap) | transfer curve |
| `R_d` | Ω | drift resistance in series with the channel | R_DS(on) at the datasheet V_GS |
| `C_gs` | F | constant | C_iss − C_rss at the highest tabulated V_DS |
| `C_gd` | charge law of `v_dg` | Miller capacitance | C_rss(V_DS) |
| `C_ds` | charge law of `v_ds` | output junction capacitance | C_oss − C_rss |
| `R_g` | Ω | internal gate resistance | datasheet value |
| `L_s` | H | common source inductance, default 0 | package data, if given |
| `body_diode` | Lauritzen / Shockley params, or `None` | anti-parallel junction | V_SD(I_S) curve, Q_rr, I_rrm |
| `T_j` | °C | run-constant junction temperature | user |
| `rds_tc1`, `rds_tc2` | 1/K, 1/K² | `r(T) = 1 + tc1·ΔT + tc2·ΔT²`, ΔT = T_j − 25 | normalised R_DS(on)(T) |
| `vt_tc` | V/K | `V_T(T) = V_T + vt_tc·ΔT`, default 0 | V_GS(th)(T) |

`r(T)` scales `R_d` by `r` and `K` by `1/r`, so the device's
small-signal `R_DS(on)(T) = r(T)·R_DS(on)(25 °C)` holds at the datasheet
gate voltage (exactly in the limit `V_DS → 0`; the triode law's `V_DS²`
term adds a deviation of order `V_DS / (2·V_OV)`). `R_d = R_DS(on) − 1/(2·K·(V_GS,spec − V_T))` must be ≥ 0;
a negative value means the transfer curve and R_DS(on) disagree and is
refused with both numbers.

### Charge laws (state: the charge)

All three are evaluated through the existing charge-form companion
(`i_{n+1} = (2/h)(Q(v_{n+1}) − Q(v_n)) − i_n`, and the BDF2 stage form),
so charge is conserved whatever `C(v)` does.

1. `junction` — unchanged: `Q = C0·V0/(1−m)·[(1+v/V0)^(1−m) − 1]`.
2. `table` — `ChargeTable`: knots `(v_k, Q_k, C_k)`, `v` strictly
   increasing, one knot at `v = 0` with `Q = 0`, `C_k > 0`; cubic
   Hermite; `C(v)` is the derivative of the same polynomial; the
   Fritsch–Carlson condition is checked at construction and a violating
   tangent is clamped (as in `FluxTable`); linear tails with slopes
   `C_lo`, `C_hi`. Unlike `FluxTable` the table is not odd.
   - From `(v, C)` pairs (a digitised datasheet curve): `Q_k` is the
     integral of the monotone interpolant of `C`; the fitter adds knots
     until the table's `C(v)` reproduces every input point within
     `tol` (default 1 %) and returns the residual.
3. `vdmos_cgd` — closed form, for imported cards. With
   `s = (CGDMAX − CGDMIN)/(1 + π/2)` and `y = CGDMAX − s`, and
   `u = v_gd`:
   - `u > 0`: `C = y + s·tanh(a·u)`, `Q = y·u + (s/a)·ln cosh(a·u)`
   - `u ≤ 0`: `C = y + s·atan(a·u)`,
     `Q = y·u + s·[u·atan(a·u) − ln(1 + a²u²)/(2a)]`

   `C → CGDMAX` for `u → +∞` and `→ CGDMIN` for `u → −∞`; `C` is C¹ at
   0, so `Q` is C². This form was checked against ngspice 45.2 at ten
   bias points from V_DS = −5 V to 400 V (CGDMAX = 1 nF, CGDMIN = 50 pF,
   A = 0.5): agreement to the six digits ngspice prints (probe `probes/cgd.cir` in this
   change). `ln cosh(x)` is evaluated as
   `|x| + log1p(exp(−2|x|)) − ln 2` to hold 1e-12.

Smoothness: every `Q(v)` is at least C¹ with a continuous, strictly
positive `C(v)`; each law gets the picovolt-spacing monotonicity test
the owner's note prescribes for flux laws.

### Channel

- Step 1: the existing `MosfetLevel1` law, unchanged, evaluated on
  `(v_g' − v_s', v_d' − v_s')`.
- Step 4: `ChannelTable` on a rectilinear `(V_GS, V_DS ≥ 0)` grid with
  `I_D(V_GS, 0) = 0`, non-decreasing in both arguments, tensor-product
  monotone cubic Hermite, third quadrant by the same terminal swap
  (`i(V_DS < 0) = −I(V_GS − V_DS, −V_DS)`), which is C¹ at `V_DS = 0`
  because the `V_DS = 0` row is identically zero. Deferred because a
  table has no AD form and needs the device interface's hand-Jacobian
  path.

### Body diode

Lauritzen by default (`I_S`, `n`, `tau`, `T_M`), Shockley for a part
with no stored charge, `None` for a device without a p-n body diode.
For an imported card, SPICE's `TT` is the `T_M → 0` limit of the
Lauritzen model (`i = i_dc + TT·di_dc/dt`), so the importer sets
`tau = TT` and `T_M = r·TT`; `r` is a numerical ratio, not a datasheet
number, fixed by task 3.4 against ngspice and reported in the import
notes.

### State, events and engines

State: the three charges (through the capacitor histories), the body
diode's stored charge and the `L_s` current. The device has no discrete
state and declares no events; its gate is a node, not a mask bit.

| Engine | Behaviour |
|---|---|
| `pwl` | Newton refresh; the time step must resolve the gate transient (`add-fixed-step-resolution-diagnostics` warns) |
| `trbdf2` | supported |
| `dsed` | refused by the conformance matrix error (Newton kinds) |
| DC operating point | charges open, channel and diode static |
| AC | refused until `add-unified-device-interface` derives it from f and q |

### Alternatives considered

- *A new monolithic `PowerMosfet` kernel device now.* Rejected for
  steps 1–3: it needs a history class with hand-derived TR and BDF2
  forms and about 60 edit sites; the composite reuses devices that
  already pass their tests. It becomes the right form once the device
  interface exists (step 5).
- *Sampling the VDMOS law into a `ChargeTable`.* Rejected: the closed
  form allows an ngspice comparison at 1e-6 instead of table accuracy.
- *Fitting an `m`-law to C_oss.* Rejected as the only option: measured
  curves are not a single power law; kept as `junction` for cards.
- *One `C(v)·dv/dt` stamp.* Rejected: it does not conserve charge
  (`models/nonlinear_capacitor.hpp:13-37`).

## Parameterisation & provenance

- Datasheet capacitance curves are small-signal values at V_GS = 0;
  each part file records the frequency and bias conditions printed on
  the datasheet figure.
- A part file's `power_mosfet` block carries, per curve: document
  number, revision, date, figure or table number, and
  `method: transcribed | digitised | vendor_spice`.
- `fitted_on` lists the curves the parameters were fitted to;
  `validated_on` lists the held-out data. The loader refuses a file in
  which the two lists intersect. Switching energies (E_on, E_off) and
  E_oss are never in `fitted_on`.
- The gate-charge curve is held out when the C_rss curve alone
  determines `C_gd`; if it is used to set the `v_dg < 0` extension
  (which a C_rss curve at V_GS = 0 does not cover), it moves to
  `fitted_on` and no longer counts.
- No synthetic part is relabelled. The existing six shipped files are
  not touched.
- Which real parts ship, and whether their curve data may be
  redistributed, is an owner decision (Open Questions).

## Validation plan

| Check | Reference | Tolerance |
|---|---|---|
| `ChargeTable` from the junction law C0 = 2 nF, V0 = 25 V, m = 0.5, 64 knots, exact slopes | closed form: Q(400 V) = 312.31 nC, C(400 V) = 485.07 pF | 1e-4 in Q, 1e-3 in C |
| `vdmos_cgd` law | ngspice 45.2 VDMOS, ten bias points | 1e-6 |
| Charge conservation over a 0 → 400 V → 0 cycle | ∮ i dt = 0 | 1e-9 relative |
| Composite equals the hand-built circuit of `p4_dpt.py` | E_on 136.14 µJ, E_off 158.96 µJ, plateau 64.8 ns | 1e-9 (same stamps) |
| Miller plateau | `Q_gd / I_g`, `I_g = (V_drv − V_pl)/R_g,tot`, `V_pl = V_T + sqrt(I_L/K)` | 10 % |
| `pwl` against `trbdf2` on the double pulse | each other | 0.5 % in E_on, E_off |
| Level-1 static curves, Shockley static curve | ngspice | stated per region (task 7.1) |
| Imported VDMOS card, double pulse | ngspice, same card and circuit | target 5 % in E_on, E_off; 3 % in plateau (to be confirmed by task 7.3) |
| Shipped part | its datasheet E_on, E_off at the datasheet test condition | the part's declared tolerance, at most 25 % (proposed) |

The first four are verification of the numerics. Only the last row is
validation, and it uses data outside `fitted_on`.

## Risks / Trade-offs

- A nanosecond transient on the fixed engine needs sub-nanosecond steps
  (7501 steps for 1.5 µs in the probe). Mitigation: documentation
  points to `trbdf2`; the fixed engine's resolution diagnostic warns.
- A Level-1 channel with a constant K does not reproduce the
  transconductance roll-off of a real part; E_on can be off by more
  than the ceiling. Mitigation: such a part is not labelled
  `validated`; the table channel (step 4) is the remedy.
- Datasheet E_on depends on the test fixture's stray inductance and
  freewheel diode, often unstated. Mitigation: the part file records
  what the datasheet states, the harness takes `L_stray`, and a missing
  condition is recorded as assumed, with the tolerance widened.
- The importer's new default changes results for existing imported
  decks. Mitigation: one summary warning naming the kept parameters,
  and `charge="drop"`.

## Migration Plan

1. `ChargeTable` and the law variant in `NonlinearCapacitor` (bit-identical
   for `junction`).
2. `add_power_mosfet` composite, YAML, bindings, `double_pulse_test`.
3. VDMOS import with charge; ngspice cross-validation netlists.
4. Fitters, part schema, PLECS formula evaluation, first real parts.
5. After `add-unified-device-interface`: `ChannelTable`, the device as
   one f/q device with generic outputs and AC.

Each step is releasable; rollback of step 3 is `charge="drop"`.

## Open Questions

- Which three to five real parts ship first (one Si super-junction, one
  SiC, one GaN would cover the classes), and may their digitised
  datasheet curves be redistributed under the repository's licence, or
  must parts stay in a user directory on `PULSIM_PARTS_PATH` with only
  the fitter and the acceptance harness in the repository?
- Is 25 % the right ceiling for the `validated` label, or should the
  label carry the measured error instead of a pass/fail?
- Argument order: `(drain, source, gate)` matches `add_mosfet_level1`;
  SPICE and most datasheets use D, G, S. Keep the house order?
- Should the same structure be proposed for a datasheet IGBT
  immediately after, or wait for the device interface?
