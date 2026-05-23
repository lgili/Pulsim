# Half-Bridge Converter — Reference Project

Sixth entry under `projects/converters/`. First **multi-switch** topology.

## Why the half-bridge

The half-bridge is the natural next step after the forward. **Same
small-signal model**, but the implementation moves from a single-switch
ON/OFF chop to **two switches that alternate** around a rail-split
input. That single change unlocks:

| What changes | Forward | Half-bridge |
|---|---|---|
| Switch count | 1 | 2 (S1 high-side + S2 low-side) |
| Transformer reset | needs reset winding | **automatic** (symmetric flux) |
| Per-switch voltage stress | 2·V_g (during reset) | **V_g** (rail-split halves it) |
| Output ripple frequency | $f_{sw}$ | **$2 f_{sw}$** (both half-cycles rectified) |
| Per-switch duty cap | $D \le 0.5$ (reset margin) | $D \le 0.5$ (dead-time margin) |
| Average $V_o$ | $n \cdot V_g \cdot D$ | $n \cdot V_g \cdot D$ (identical) |

The headline takeaway: the half-bridge **inherits the forward's small-
signal model** — same $G_{vd}(s)$, same $G_{vg}(s)$, same $Z_{out}(s)$,
**no RHP zero**. What you learn from designing a forward compensator
transfers immediately. The pedagogy here is about the IMPLEMENTATION,
not new control math.

## The implementation story (what's actually new)

### Two switches alternating

S1 conducts during the first half of the period, S2 during the second
half. A short **dead-time** ($t_{dead}$, typically 50–200 ns) between
them prevents **shoot-through** — both on simultaneously would short
$V_g$ directly through the switches and destroy them.

The per-switch duty $D$ is defined per FULL period (not per half-
period). So $D \in [0, 0.5]$ with the dead-time as the actual upper
constraint. We cap at $D_{max} = 0.45$.

### Rail-split input

Two series caps (C1, C2) across $V_g$ produce a midpoint at $V_g/2$.
The transformer primary connects between the **switching node** (the
S1/S2 midpoint) and the **rail midpoint**, so the primary sees
$\pm V_g/2$ — half of what a full-bridge would deliver.

A common student question: "If the primary sees half the voltage, why
is $V_o = n V_g D$ (not $n V_g D / 2$)?" Answer: both half-cycles
deliver energy to the output (the rectified output sees 2D-fraction
of conduction per period), so the factor of 2 from rectification
exactly cancels the factor of 1/2 from rail-split. Net:
$V_o = n \cdot V_g \cdot D$ — same as the forward.

### Center-tapped secondary

Two rectifier diodes — one for each half-cycle. The output filter
inductor sees a positive voltage during BOTH halves of the period
(the rectification flips the negative half), so the **output ripple
frequency doubles** to $2 f_{sw}$. That lets the LC filter be smaller
for the same ripple spec.

### No reset winding

Each half-cycle resets the transformer flux symmetrically (S1 push +
S2 push = balanced flux excursion). No third winding needed. Compare
the forward's reset-winding constraint ($D \le 0.5$) — the half-bridge
has the same constraint, but it's now a **dead-time** constraint, not
a reset constraint.

## Files

| File | Role |
|---|---|
| `01_half_bridge_modeling.ipynb` | Derivation, 4-phase switched model, state-space matching forward, self-consistency. |
| `02_half_bridge_controller.ipynb` | K-factor Type-III + switched closed-loop with 4-phase switching logic. |
| `half_bridge_model.py` | `HalfBridgeParams` + state-space + 3 reference TFs + dead-time check. |
| `_build_notebooks.py` | Generator. |

## Reference parameters

`HalfBridgeParams()` default: 48 V → 5 V at 2 A (10 W), 1:0.25
transformer ($n = 0.25$), 50 µH filter inductor, 200 µF cap, 100 kHz
per-switch switching (200 kHz output ripple), 100 ns dead-time.

- $D \approx 0.417$ — comfortable margin below the 0.45 cap
- $f_n \approx 1.6$ kHz, $Q \approx 5$ (same form as the buck/forward)
- **No RHP zero** — bandwidth target $f_{sw}/20 = 5$ kHz (same anti-
  saturation choice as the forward)

## What you'll learn

- The half-bridge's small-signal model is **literally identical** to
  the forward's — the two-switch implementation doesn't show up in
  the average model.
- The implementation differences (rail-split, alternating switches,
  dead-time, center-tap rectifier, $2 f_{sw}$ ripple) all show up in
  the **switched** model and in the **stress / sizing** equations,
  not in $G_{vd}(s)$.
- Why production designs move from forward to half-bridge as power
  increases: lower switch stress, no reset winding, smaller output
  filter for the same ripple.

## Comparison across the library

| Converter | Settling | RHP zero | Isolation? | Switches | Topology family |
|---|---|---|---|---|---|
| buck | 1.4 ms | none | no | 1 | buck |
| boost | 25 ms | 4.6 kHz | no | 1 | boost |
| buck-boost | 25 ms | 9.5 kHz | no | 1 | buck-boost |
| flyback | 15 ms | 12.7 kHz | yes | 1 | buck-boost (isolated) |
| forward | 1.14 ms | none | yes | 1 | buck (isolated) |
| **half-bridge** | **~1.1 ms (predicted)** | **none** | **yes** | **2** | **buck (isolated, multi-switch)** |

The half-bridge should land in the same neighborhood as the forward —
the model is identical.

## How to run

```bash
pip install numpy scipy matplotlib
jupyter lab projects/converters/half_bridge/01_half_bridge_modeling.ipynb
```

## Bibliography

- Erickson & Maksimović, *Fundamentals of Power Electronics*, 3rd
  ed., **Section 6.3** (half-bridge derivation) and **Section 6.4**
  (push-pull, full-bridge — the topology siblings).
- TI / Infineon half-bridge application notes — dead-time selection,
  bootstrap drive for the high-side switch, MOSFET vs IGBT selection
  for $V_g \ge 200$ V designs.
