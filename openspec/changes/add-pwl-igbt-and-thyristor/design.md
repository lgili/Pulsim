## Context

Review items: `raw/models.md` ranked item 6 and the inventory rows for
`add_igbt` and the thyristor; synthesis M5. The diode half of M5 (body
and freewheel drops) is Phase 0's `update-composite-diode-forward-drop`.

Owner notes that bind this design:

- *pwl-diode-vth-is-not-a-drop*: PSIM style — an ideal switch plus a
  series drop; a threshold is not a drop. The same note records that a
  gated-on `add_igbt` conducts in both directions.
- *switchstatemask-takes-a-size*: `SwitchStateMask(n)` is a size with
  every bit off; every device probe needs a control that proves the
  device can conduct before a blocking claim is made.
- *motor-inductance-is-the-free-parameter*: a missing conduction term
  can be absorbed by a tuned free parameter and go unnoticed.

## Goals / Non-Goals

- Goals: an IGBT with a knee and reverse blocking; a thyristor, GTO and
  TRIAC whose latch lives in the kernel; all on every engine and in
  YAML; losses from the same law.
- Non-Goals: turn-off time `t_q` and commutation failure; dv/dt and
  break-over triggering; latching current distinct from holding
  current; a pulse-commanded GTO with separate fire and quench inputs;
  gate current and gate-drive power; reverse recovery (use the
  Lauritzen diode); switching energies (loss tables, in
  `add-in-kernel-event-losses`).

## Decisions

### One law

`v = v_anode − v_cathode` (collector − emitter), `c` the conduction
state, `g` the gate bit, `V_f` the drop (`V_ce0` for the IGBT),
`V_on = (V_f == 0) ? V_th : max(V_th, V_f)` as in `SwitchedDiode`.

| | Rule |
|---|---|
| ON law | `i = g_on·(v − V_f)` |
| OFF law | `i = g_off·v` (no offset) |
| `gate_held`: next state | `c⁺ = g ∧ (c ? i > 0 : v ≥ V_on)` |
| `latching`: next state | `c⁺ = c ? (g ? i > 0 : i > I_H) : (g ∧ v ≥ V_on)` |
| Crossing signal | ON: `i − I_x` with `I_x = 0`, or `I_H` when latched with the gate off; OFF with `g`: `v − V_on`; OFF without `g`: none |
| Power while ON | `V_f·i + i²/g_on` |
| Validation | `V_f`, `I_H` finite and ≥ 0; `V_f > 0` requires reverse blocking; errors name the device |

- **`gate_held` is the diode rule AND the gate.** With the gate on it
  behaves exactly as `SwitchedDiode` with `V_f = V_ce0`; when the gate
  goes off it opens whatever the current (hard turn-off; the load
  current needs its freewheel path).
- **`latching` keeps the diode rule while the gate is on** (`i > 0`),
  and uses the holding current only once the gate is removed. A latch
  that compared against `I_H` during the gate pulse would drop out on
  the first step of an inductive load, whose current starts at zero.
- **`I_H = 0` by default**: turn-off is the diode's current-zero event,
  detected by the engines' existing crossing machinery, not by a
  sampled `|i| < ε` test. `I_H > 0` chops the current into `g_off` and,
  with an inductive load and no snubber, produces the voltage
  `I_H / g_off`; this is documented and the voltage-sanity check
  reports it.
- **`reverse_blocking=False`, `V_ce0 = 0`** reduces `gate_held` to
  `c = g`: the legacy switch, stamped through the same code, so the old
  behaviour is one flag away and bit-identical.

### Stamp and mask

- The stamp is `stamp_switch_fixed(..., v_drop_on)`
  (`stamping/stamp_switch.hpp:47-56`): `g_on` plus `∓g_on·V_f` on `b`
  when `c = 1`, `g_off` when `c = 0`. `J` and its sparsity do not
  depend on the mode, so cached factorisations are keyed as today.
- The user's mask bit is the gate `g`, at the same switch index the
  device has today. The cache is keyed on the **effective** mask, in
  which the device contributes `c`. Two stamps per device, as before.
- The result records both `g` and `c` per step; `res.i`, the loss and
  thermal summaries use `c`.

### Decision sites (every engine)

The same sites as the PWL diode, through one `GatedDiode::decide` /
`crossing_signal`: pwl event iteration and sub-step watcher, DC
operating point and seeding, TR-BDF2 crossing / held / forbidden
checks, DSED predicates and settle. A gate edge is a commanded event; a
change of `c` with `g` constant is a state event.

| Engine | Behaviour |
|---|---|
| `pwl` | supported; commutation instants on the grid (exact with `add-fixed-step-event-splitting`) |
| `trbdf2` | supported; state events located by the existing crossing search |
| `dsed` | supported through the diode predicates; a zero-width band has the diode's known settle limit |
| DC operating point | gate from the mask at `t = 0`, `c` decided; a latching device starts OFF unless `initial_on=True` |
| AC sweep | linear at the supplied mask (`fix-mna-sweep-correctness`) |

### Composites

- `add_gto` = `gate_held` with `V_f`; its mask bit is the gate-drive
  state (on-drive / off-drive).
- `add_triac` = two `latching` branches, anti-parallel, with two gate
  bits reported by `switch_indices_of(name)`.
- `add_igbt` adds no diode. The co-packaged diode is a separate
  `add_diode(V_f=...)`, as today.

### Firing schedule

`sources/phase_control_fn.hpp`: for a p-pulse bridge synchronised to a
line frequency and phase, gate `k` is high for `width` after
`α + k·2π/p`. It lives beside the existing C++ PWM schedules and is
superseded by the event queue's modulators.

### Alternatives considered

- *Series composite today: `add_switch` + `add_diode(V_f)` with an
  internal node.* Works with no kernel change, but adds a node per
  device, leaves that node on a `g_off`/`g_off` divider when both are
  off, and doubles the state count. Kept as the documented stopgap
  until this ships.
- *A Newton IGBT for everything.* `add_igbt_level1` already has the
  knee and blocks reverse current; it is 28 µs per step against about
  1 µs for a linear circuit (`raw/models.md`, per-step cost) and is
  refused by `dsed`.
- *Keep the latch in a schedule or observer.* That is today's closure:
  one-step lag, `pwl` only, and the measured failure above.
- *Default `reverse_blocking=False`.* Keeps old results but leaves the
  wrong physics as the default; see Open Questions.

## Parameterisation & provenance

- `V_ce0` and `R_on` come from the datasheet output characteristic at
  the operating gate voltage and temperature: a straight line through
  two points of `V_CE(I_C)`. The docstring states this and gives no
  default part values; the default `V_ce0 = 0` is "not specified", not
  a typical device.
- `V_f`, `R_on`, `I_H` of a thyristor: datasheet `V_T0`, `r_T`, `I_H`.
- No part data ships with this change.

## Validation plan

| Check | Expected | Tolerance |
|---|---|---|
| IGBT forward, control: 10 A, `V_ce0 = 0.9 V`, `R_on = 10 mΩ` | `v_CE = 1.000 V`, `P = 10.0 W` | 1e-9 |
| IGBT reverse: −100 V applied, gate on | `i = −1e-7 A` (`R_off = 1e9`) | 1e-9 relative |
| Legacy flag: `reverse_blocking=False`, `V_ce0 = 0` | bit-identical to the current `add_igbt` | exact |
| Hold at `V_ce0 ± ε`, gate on | ON above, OFF below; `min(i) ≥ −1e-9 A` | — |
| Half-wave thyristor rectifier, 100 V peak, 10 Ω, 60 Hz, dt = 10 µs, α = 30° and 90° | 29.699 V and 15.915 V | 0.5 % |
| Latch: 100 µs gate pulse, RL load | conducts until the current zero, not until the gate edge | — |
| Holding current: gate removed, current decays through `I_H = 0.05 A` | turns off at the first accepted state with `i ≤ 0.05 A` | — |
| Six-pulse bridge, 400 V line-to-line, 60 Hz, 20 A constant-current load, gate edges on the grid, α = 0°, 30°, 60° | `(3√2/π)·V_LL·cos α` (≈ `1.35·V_LL·cos α`) = 540.2, 467.8, 270.1 V | 0.5 % |
| Same with 1 mH source inductance per phase, α = 30° | `467.8 − (3/π)·ω·L_s·I_d` = 460.6 V | 0.5 % |
| Engine agreement on the six-pulse case | mean output | 0.5 % |
| Loss summary: IGBT at 10 A DC | `V_ce0·I + R_on·I²` = 10.0 W | 1e-9 |

All references are analytic (ideal-converter formulas); none uses a
fitted parameter. Means are time-weighted.

## Risks / Trade-offs

- **Existing circuits without a freewheel path stop working.** That is
  the intended trade (as for `add_igbt_level1`), but it is a behaviour
  change → BREAKING note, migration flag, and the preflight names the
  IGBT when the run aborts on an inductive cut-set.
- A decision site that misses the gate lets a gated-off device conduct.
  Mitigation: one helper, a test per site that holds the gate off with
  forward bias and asserts `|i| ≤ g_off·|v|` on every engine.
- `V_th ≤ V_f` gives a zero-width band and can chatter at a current
  zero, exactly as the diode does; documented, same remedy.
- The effective mask differs from the commanded mask; code that reads
  the mask record must say which it wants. Mitigation: both are
  recorded under distinct names.
- TRIAC as two indices is less convenient than one gate; accepted for
  the first version.

## Migration Plan

1. `GatedDiode` law, stamp reuse, pool and event state; `add_igbt`
   parameters with `reverse_blocking` defaulting to **False** behind a
   `FutureWarning` for one release.
2. Decision sites on `pwl`, DC, `trbdf2`, `dsed`; conformance rows;
   Python law and summaries.
3. `add_thyristor`, `add_gto`, `add_triac`, firing schedule, YAML;
   deprecate the Python closure and the chain block.
4. Next release: flip the `add_igbt` default to `True` (BREAKING,
   CHANGELOG).

`add_igbt(..., reverse_blocking=False)` restores the old device at any
point.

## Open Questions

- Flip the `add_igbt` default in the same release as the feature (pure
  correctness, as was done for the VSI's `V_f`), or after one release
  of warnings as the plan above proposes?
- Is the gate-drive-state GTO enough, or is a pulse-latched GTO with a
  separate turn-off command needed?
- Should `add_igbt` grow `fwd=True, V_f=, R_d=` for the co-packaged
  diode once Phase 0's drop lands, or stay a single device?
- Is thyristor turn-off time `t_q` (commutation failure in inverter
  mode) wanted in a follow-up?
