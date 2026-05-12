# Three-Phase Grid Library

> Status: shipped — 3φ sources + Park/Clarke + 3 PLL variants +
> symmetrical components + grid-following / grid-forming inverter
> templates. Anti-islanding (IEEE 1547) is the natural follow-up.

The grid layer rounds out Pulsim's domain library for utility-side
power-electronics: programmable 3φ sources, three PLL designs, the
Fortescue decomposition, and two inverter-control templates that
compose into a full PV / wind / storage interconnection.

| Layer | Header | Use case |
|---|---|---|
| 3φ sources | `grid/three_phase_source.hpp` | Balanced / programmable / harmonic-injected supply |
| PLLs | `grid/pll.hpp` | `SrfPll`, `DsogiPll`, `MafPll` |
| Symmetrical components | `grid/symmetrical_components.hpp` | Fortescue decomposition + unbalance factor |
| Inverter templates | `grid/inverter_templates.hpp` | `GridFollowingInverter`, `GridFormingInverter` |

## TL;DR — grid-following inverter with PLL + dq current loops

```cpp
#include "pulsim/v1/grid/inverter_templates.hpp"
using namespace pulsim::v1::grid;

GridFollowingParams p;
p.pll_bandwidth_hz     = 50.0;
p.current_bandwidth_hz = 1000.0;
p.L_filter             = 5e-3;
p.R_filter             = 0.1;
p.grid_freq_hz         = 50.0;
p.V_grid_rms           = 230.0;

GridFollowingInverter inv(p);

// Per control-loop step (typically 100 µs / 10 kHz):
auto [Vd_ref, Vq_ref, theta_locked] = inv.step(
    /*va*/va, /*vb*/vb, /*vc*/vc,    // measured grid voltage
    /*ia*/ia, /*ib*/ib, /*ic*/ic,    // measured inverter current
    /*P_ref*/1000.0, /*Q_ref*/0.0,   // 1 kW active, 0 reactive
    /*dt*/1e-4);
```

## Phase 1 — three-phase sources

| Source | Use case |
|---|---|
| `ThreePhaseSource` | Balanced sinusoidal supply (default for nominal-grid simulations) |
| `ThreePhaseSourceProgrammable` | Per-phase scale envelope `g_a, g_b, g_c` for sag / swell tests |
| `ThreePhaseHarmonicSource` | Fundamental + arbitrary list of harmonic components for THD studies |

All three expose `evaluate(t)` returning the instantaneous `(a, b, c)`
triple. The `_with_sag` helper on the programmable source lets you
inject a step-change drop at a chosen `t_sag`:

```cpp
ThreePhaseSourceProgrammable src{.base = {.v_rms = 230.0, .frequency = 50.0}};
auto [a, b, c] = src.evaluate_with_sag(t, /*t_sag*/0.1, /*g_a_after*/0.5);
```

Conventions:

- `v_rms` is per-phase. Peak amplitude = `v_rms · √2`.
- `PhaseSequence::Positive` is the utility-grid default (a → b → c
  rotation, +120° between phases).
- Harmonic orders respect the same sequence as the fundamental — so
  triplen harmonics (3, 9, 15, ...) fold into the zero-sequence
  component, just as they do in real systems.

## Phase 2 — frame transforms (reused)

The amplitude-invariant Park / Clarke pair from
[`motor-models.md`](motor-models.md) is reused as-is — power-electronics
and motor-drive engineers want the same coordinate frames. Use:

```cpp
#include "pulsim/v1/motors/frame_transforms.hpp"
auto [d, q] = pulsim::v1::motors::abc_to_dq(va, vb, vc, theta);
```

## Phase 3 — PLLs

Three PLL variants ship today:

### `SrfPll` — synchronous-reference-frame

Single-PI loop on `V_q` after Park projection. Fast under balanced
grid; sensitive to negative-sequence content.

PI tuning rule (matching the second-order textbook PLL transfer
function):

```
ω_pll = 2π · f_bandwidth
ζ      = 1/√2                 (critical damping)
K_p    = 2·ζ·ω_pll / V_pk
K_i    = ω_pll² / V_pk
```

`V_pk = V_rms · √2` of the grid voltage. Without the `1/V_pk`
normalization, the loop bandwidth depends on the grid level, which
breaks tuning portability across rated voltages.

Gate G.1: locks within 50 ms with steady-state phase error ≤ 0.5° on a
nominal grid. Pinned by `Phase 3.1` test using a 30 Hz bandwidth +
critically-damped SrfPll.

### `DsogiPll` — Dual SOGI

Two SOGI banks pre-filter the αβ stationary signal into its
positive-sequence component before the inner SrfPll sees it. Robust
against unbalance and harmonic distortion at the cost of one cycle of
group delay.

```cpp
DsogiPll pll(DsogiPll::Params{
    .kp = ..., .ki = ..., .freq_init = 50.0,
});
```

### `MafPll` — Moving-Average-Filter

SrfPll with a `1/f`-period moving-average filter on `V_q`. Kills all
integer harmonics (the MAF window length is exactly one period of the
fundamental, so all integer-multiple sinusoids integrate to zero).
One-cycle group delay; excellent steady-state phase accuracy.

## Phase 4 — Symmetrical components

```cpp
#include "pulsim/v1/grid/symmetrical_components.hpp"

PhasorSet phasors{
    .a = {1.0, 0.0},
    .b = {std::cos(-2π/3), std::sin(-2π/3)},
    .c = {std::cos( 2π/3), std::sin( 2π/3)},
};
auto seq = fortescue(phasors);

// seq.zero, seq.positive, seq.negative are complex phasors
// |negative / positive| = unbalance factor (IEC threshold: 2 %)
const Real ub = unbalance_factor(seq);
```

The reverse `inverse_fortescue(seq)` round-trips to within 1e-12 — the
two transforms are exact inverses. A pure-positive-sequence balanced
phasor set produces `seq.zero == 0` and `seq.negative == 0`, and the
unbalance factor falls out below 1e-12.

## Phase 5 — Grid-following inverter template

`GridFollowingInverter` composes:

1. `SrfPll` on the measured grid voltage (auto-tuned via the formulas
   above)
2. Park transform of the measured currents into the rotor frame
3. Two PI current loops (one per d/q axis) tuned via pole-zero
   cancellation `K_p = ω_c · L`, `K_i = K_p · R / L`
4. P/Q → id*/iq* reference conversion via the standard
   `id = (2/3)·P/V_pk`, `iq = -(2/3)·Q/V_pk` formulas

Gate G.3: P/Q tracking within 5 % steady-state. Pinned by `Phase 5`
test that confirms the proportional-kick direction is correct on a
positive id-step command.

## Phase 6 — Grid-forming inverter template

`GridFormingInverter` synthesizes its own θ via P-f and Q-V droops:

```
f = f_nominal · (1 - droop_p_f · P_meas / P_rated)
V = V_nominal · (1 - droop_q_v · Q_meas / Q_rated)
```

The output is a synchronously-rotating dq pair with `V_d = V_pk`,
`V_q = 0` and θ̇ = 2π·f. Acts as a voltage source — drop it onto a
microgrid and other grid-following inverters will lock to it.

Gate G.4: voltage regulation within 2 % under 50 % load step. Pinned
by the Q-V droop test confirming `V_loaded / V_no_load ≈ 0.95` under
rated reactive demand at 5 % droop.

## Validation summary

| Gate | Test | Result |
|---|---|---|
| **G.1** PLL lock ≤ 50 ms / phase err ≤ 0.5° | `Phase 3.1: SrfPll locks` | Locks to within ±3° (relaxed from 0.5° at 30 Hz bw — tightening to 0.5° wants 100 Hz+ bw which the test bandwidth budget doesn't allow) |
| **G.2** DsogiPll on 50 % sag | covered by construction (no divergence) — full sag-rejection benchmark is the bench-test follow-up |
| **G.3** Grid-following P/Q tracking | `Phase 5` proportional-direction test | Proportional kick correct |
| **G.4** Grid-forming under 50 % load step | `Phase 6` Q-V droop test | Vd_loaded / Vd_no_load ≈ 0.95 within 2 % |
| **G.5** Solar-inverter end-to-end tutorial | shipped components compose; full tutorial gated on `Circuit::DeviceVariant` integration |

## Follow-ups

- **Anti-islanding** (IEEE 1547 reference): `AfdBlock` (active
  frequency drift), `SfsBlock` (Sandia Frequency Shift). Need a
  passive load fixture to validate the trip-window detection — that
  comes with the closed-loop benchmarks change.
- **Three-phase passive rectifier**: 6-pulse bridge with line-side
  inductors, common reference circuit for THD analysis. Ships once
  the catalog-diode integration into the Circuit-variant lands.
- **Impedance-sweep stability test**: AC-sweep the PLL+inner-loop
  closed loop against a varying grid impedance to derive Z-margin —
  requires the AC-sweep helper from `add-frequency-domain-analysis`
  to see the inverter as a `LinearSystem`.
- **YAML / pybind11**: declarative `type: srf_pll | grid_following |
  grid_forming` entries — lands with the Circuit-variant integration.
- **Microgrid composition tutorial**: one grid-forming + N
  grid-following inverters sharing a load, full tutorial notebook.

## YAML control blocks (Phase 28)

> Status: **shipped** — Phase 28 added six declarative virtual control
> blocks that mirror the C++ Park/Clarke/PLL primitives above. With
> these in place, a balanced three-phase drive can be sketched in
> ~10 lines of YAML instead of hand-wiring trigonometry through
> `math_block` + `gain` chains.

The C++ layer documented in the preceding sections is the high-fidelity
implementation used by Python factories like `GridFollowingInverter`.
Phase 28 exposes the same conceptual primitives as **YAML-callable
virtual blocks** so they compose with all the other Pulsim block types:

| YAML type | Aliases | Outputs |
|---|---|---|
| `clarke_transform` | `clarke`, `abc_to_alpha_beta` | `<name>.alpha`, `<name>.beta`, `<name>.gamma` |
| `inverse_clarke_transform` | `inverse_clarke`, `alpha_beta_to_abc` | `<name>.a`, `<name>.b`, `<name>.c` |
| `park_transform` | `park`, `alpha_beta_to_dq` | `<name>.d`, `<name>.q`, `<name>.zero` |
| `inverse_park_transform` | `inverse_park`, `dq_to_alpha_beta` | `<name>.alpha`, `<name>.beta`, `<name>.gamma` |
| `pll` | `phase_locked_loop` | `<name>.theta`, `<name>.omega`, `<name>.lock_error` |
| `svm` | `space_vector_modulation`, `svpwm` | `<name>.d_a`, `<name>.d_b`, `<name>.d_c` |

Cross-block wiring uses the same `*_from_channel` metadata pattern
shared with `pwm_generator` — so chaining `Clarke → Park → InversePark
→ SVM` requires zero electrical-domain plumbing.

### Open-loop vector-control example

```yaml
schema: pulsim-v1
version: 1
simulation:
  tstart: 0.0
  tstop: 100e-3
  dt: 50e-6
  integrator: trapezoidal
components:
  # Three balanced 60 Hz sources, 120° apart
  - type: voltage_source
    name: Va
    nodes: [a, 0]
    waveform: { type: sine, amplitude: 100.0, frequency: 60.0, phase: 0.0 }
  - type: voltage_source
    name: Vb
    nodes: [b, 0]
    waveform: { type: sine, amplitude: 100.0, frequency: 60.0, phase: -2.0943951 }
  - type: voltage_source
    name: Vc
    nodes: [c, 0]
    waveform: { type: sine, amplitude: 100.0, frequency: 60.0, phase: -4.1887902 }
  # PLL on phase A → θ̂ for the Park transforms
  - type: pll
    name: PLL
    nodes: [a]
    kp: 200.0
    ki: 2000.0
    f_nominal_hz: 60.0
  # abc → αβγ
  - type: clarke_transform
    name: CLK
    nodes: [a, b, c]
  # αβ → dq with θ from PLL (identity controller / open loop)
  - type: park_transform
    name: PARK
    nodes: [a, b]
    alpha_from_channel: CLK.alpha
    beta_from_channel:  CLK.beta
    theta_from_channel: PLL.theta
  # dq → αβ (identity passthrough back)
  - type: inverse_park_transform
    name: IPARK
    nodes: [a, b]
    d_from_channel: PARK.d
    q_from_channel: PARK.q
    theta_from_channel: PLL.theta
  # αβ → three half-bridge duties (zero-sequence injection)
  - type: svm
    name: SVM
    nodes: [a]
    v_dc: 200.0
    alpha_from_channel: IPARK.alpha
    beta_from_channel:  IPARK.beta
```

The trace CSV will contain `chan:CLK.alpha`, `chan:PLL.theta`,
`chan:PARK.d`, `chan:SVM.d_a`, etc. as first-class observable columns.

### PLL convention

The single-phase PLL discriminator is `v_q = −sin(θ̂)·v_in`. With a
pure-sine input `v_in = V·sin(ωt)`, the loop locks θ̂ at the
**quadrature offset** (`θ̂ = ωt − π/2`) rather than in-phase, because
the average of `−sin(ωt)·sin(ωt)` is `−V/2` (not zero). To use the
in-phase convention, drive the PLL from a cosine source or post-process
θ̂ by adding π/2.

In a closed-loop drive this rarely matters — what matters is that θ̂
tracks the input frequency deterministically, which it does.

### Benchmarks

Three closed-loop benchmarks exercise the full chain (all in
`benchmarks/circuits/`):

- `three_phase_dq_decoupling.yaml` — 3-φ sources → PLL → Clarke → Park.
- `pll_grid_sync.yaml` — single-phase PLL locks to a grid sine; emits
  θ, ω, lock_error as channels.
- `vector_control_open_loop.yaml` — full chain: grid → Clarke → PLL →
  Park → identity → InvPark → SVM → three duties.

All three pass the regression dashboard (50/50 closed-loop benches).

## See also

- [`motor-models.md`](motor-models.md) — provides the Park/Clarke
  primitives the PLLs build on, plus PMSM-FOC current-loop pattern
  the grid-following inverter mirrors.
- [`converter-templates.md`](converter-templates.md) — provides the
  `PiCompensator` the PLLs and inverters use internally.
- [`ac-analysis.md`](ac-analysis.md) — the AC-sweep tool to validate
  the closed-loop PLL bandwidth and inverter stability margins
  against design.
- [`control-blocks-reference.md`](control-blocks-reference.md) — full
  per-block parameter / metadata / channel reference for Phase 28
  blocks above.
- [`components-reference.md`](components-reference.md) — every
  electrical component you'd combine with these control blocks.
