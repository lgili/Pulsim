# Magnetic Models

> Status: shipped — header-only math primitives + four reference cores.
> Full Circuit-variant integration (saturable devices on the MNA stamp
> surface) is the natural follow-up.

Pulsim's magnetic-modeling layer covers four kinds of components a power
engineer reaches for when laser-printing a converter:

| Component | Header | Use case |
|---|---|---|
| **B-H curves** | `magnetic/bh_curve.hpp` | Material characterization (table from datasheet, or arctan / Langevin analytical fits). |
| **Saturable inductor** | `magnetic/saturable_inductor.hpp` | Output chokes, PFC inductors, flyback primary — anything where the ferrite knee matters. |
| **Saturable transformer** | `magnetic/saturable_transformer.hpp` | Multi-winding power transformers with shared core flux + per-winding leakage. |
| **Hysteretic inductor** | `magnetic/hysteresis_inductor.hpp` | Mains-frequency cores where the actual hysteresis loop drives loss, demagnetization current, and inrush asymmetry — Jiles-Atherton 5-parameter model. |

Plus the loss / catalog plumbing:

| Helper | Header | Use case |
|---|---|---|
| **Steinmetz / iGSE** | `magnetic/bh_curve.hpp` | Cycle-averaged specific loss `P_v = k·f^α·B^β`. iGSE handles non-sinusoidal flux. |
| **Jiles-Atherton step** | `magnetic/bh_curve.hpp` | One-step ODE evolution of the hysteretic state for Phase 4 devices. |
| **Core catalog loader** | `magnetic/core_catalog.hpp` | Parses `<vendor>/<material>.yaml` into `BHCurveTable + SteinmetzLoss + JilesAthertonParams`. Four reference cores ship under `devices/cores/`. |

Everything in the layer is **header-only** and self-contained — no link
to `pulsim.lib` is required for tests or downstream code that just wants
the math objects. The full Circuit-side integration (registering
`SaturableInductor` as a stamp-able variant alongside `Inductor`,
`Capacitor`, etc.) is tracked as a follow-up so the public API at the
device-class level is final.

## TL;DR

Build a saturable inductor from a TDK N87 core data sheet:

```cpp
#include "pulsim/v1/magnetic/core_catalog.hpp"
#include "pulsim/v1/magnetic/saturable_inductor.hpp"

using namespace pulsim::v1::magnetic;

const CatalogCore core = load_core_catalog_file(
    "devices/cores/TDK/N87.yaml");

SaturableInductor<BHCurveTable> Lo(
    {.turns = 50.0, .area = core.area_m2, .path_length = core.path_length_m},
    core.bh_curve);

// Apply a constant 12 V across Lo for 5 µs:
Lo.advance_trapezoidal(12.0, 5e-6);

// Read back state:
const Real i = Lo.current();                        // A
const Real L_d = Lo.differential_inductance();      // H, falls off at the knee
```

## Math reference

### B-H curves

All three implementations expose the same surface:

```cpp
Real h_from_b(Real B);     // forward characteristic
Real b_from_h(Real H);     // inverse characteristic
Real dbdh(Real H);         // differential permeability dB/dH
Real saturation_density(); // Bs (T)
```

| Curve | When to use |
|---|---|
| `BHCurveTable` | You have a measured (B, H) sequence from datasheet — most accurate, no analytical fitting drift. |
| `BHCurveArctan` | Quick first-cut for soft ferrites; only need `Bs` and `Hc` (= H at half saturation). |
| `BHCurveLangevin` | Para- / superparamagnetic materials with a softer roll-off than arctan can model. |

### Saturable inductor

The geometry × curve combo gives:

```
λ  = N · A_e · B                (flux linkage)
i  = (l_e / N) · h_from_b(B)    (Ampère's law)
L_d= (N² · A_e · dB/dH) / l_e   (differential inductance)
```

`L_d` is what the MNA companion stamp's `g_eq = dt / (2·L_d)` uses. In
deep saturation `dB/dH → 0` and `L_d` floors at the air-core inductance
`μ₀·N²·A_e/l_e` so the simulator stays numerically stable.

### Saturable transformer

N windings share one magnetic core. Each winding has its own turns
count and leakage inductance. The model treats the first winding as the
turns-ratio reference; voltage on winding `k` is

```
v_k = (N_k / N_ref) · dλ_m/dt + L_leak[k] · di_leak[k]/dt
```

Use cases: flyback (1 primary + 1 secondary, possibly with
auxiliary), forward / push-pull (1 primary + center-tapped secondary),
DAB, multi-output flyback. Saturation on the magnetizing branch is
where the model earns its keep — linear coupled-inductor models miss
the inrush behavior entirely.

### Steinmetz / iGSE

Two flavors of cycle-averaged core-loss density:

```cpp
SteinmetzLoss s{.k = 1.5e-3, .alpha = 1.6, .beta = 2.7};
const Real P_v_sin = s.cycle_average(f_hz, B_pk);   // sinusoidal flux
const Real P_v_arb = igse_specific_loss(B_t, dt, s); // arbitrary B(t)
```

iGSE (Improved Generalized Steinmetz Equation) integrates `|dB/dt|^α`
over one cycle then folds in the peak-to-peak swing — necessary for
non-sinusoidal converter waveforms (square, triangular) where the
plain Steinmetz over-/under-counts depending on duty cycle.

For sinusoidal inputs the two agree within ≤ 5 % (gate G.2 tolerance).

### Jiles-Atherton hysteresis

Five parameters describe the loop shape:

| Parameter | Range | Role |
|---|---|---|
| `Ms` | 1e5 – 1e6 A/m | Saturation magnetization |
| `a` | 10 – 1000 A/m | Domain density (Langevin scale) |
| `alpha` | 1e-5 – 1e-3 | Inter-domain coupling |
| `k` | 10 – 1000 A/m | Pinning coefficient (loop width) |
| `c` | 0.05 – 0.5 | Reversibility coefficient |

`HysteresisInductor` wraps this with the geometry / turns plumbing.
Each `apply_flux_step(λ)` advances the J-A state to match the new flux,
producing path-dependent current responses that match measured loops
within typical 10 % envelope after parameter fitting.

## Catalog

Four reference cores ship under `devices/cores/`:

| Vendor | Material | Geometry | Notes |
|---|---|---|---|
| TDK | `N87` | 1.5 cm² × 4.5 cm | Standard MnZn power ferrite, datasheet rev 2017-11 |
| Ferroxcube | `3C90` | 1.0 cm² × 4.0 cm | Lower-loss MnZn, datasheet rev 2008-09 |
| Magnetics | `MPP_60u` | 0.85 cm² × 5.5 cm | Distributed-air-gap powder, μ=60, catalog rev 2019 |
| EPCOS | `N97` | 1.4 cm² × 4.4 cm | Re-branded TDK family, data brief rev 2014-10 |

Catalog files are YAML maps with `vendor`, `material`, `geometry`,
`bh_curve`, `steinmetz`, and optional `jiles_atherton` blocks — see any
of the shipped files for a complete example. Loader signature:

```cpp
CatalogCore core = load_core_catalog_file(
    "devices/cores/TDK/N87.yaml");
SaturableInductor<BHCurveTable> Lo(
    {.turns = 50, .area = core.area_m2, .path_length = core.path_length_m},
    core.bh_curve);
```

The loader rejects malformed YAML (missing geometry, fewer than two
B-H points, etc.) with `std::invalid_argument`.

## Validation

Three end-to-end checks pin the gates from `add-magnetic-core-models`:

| Gate | Test | Result |
|---|---|---|
| **G.1** Inrush ≤ 20 % vs Faraday | `test_magnetic_phase6_validation::Phase 6 G.1` | Mains transformer (Bs=1.5T, 800 turns) at 110V·5ms applied integral → measured `i_actual` matches the arctan-inverse closed-form `(l_e/N)·Hc·tan(π·B/(2Bs))` within ±20 % |
| **G.2** Steinmetz ≤ 10 % | `Phase 6 G.2` | `iGSE` sweeps 25 / 100 / 500 kHz on a sinusoid → matches `cycle_average(f, B_pk)` within ±10 % |
| **G.3** Saturation onset ≤ 5 % | `Phase 6 G.3` | 5 % flux-linkage increment past the knee produces ≥ 5× the linear-regime current change — saturation is real |
| **G.4** ≥ 3 of 4 cores load | `test_magnetic_phase5_catalog` | All four shipped catalog files parse cleanly |

## Follow-ups

- **Circuit-variant integration**: register `SaturableInductor` and
  friends in the `DeviceVariant` so the existing MNA stamp pipeline
  picks them up. Today they're math objects; the integration layer is
  the next change.
- **Datasheet PDF importer**: the `pulsim.import_core_datasheet(pdf,
  vendor)` helper from the proposal is deferred — the YAML manifest
  approach already covers the catalog use case without OCR / vendor-
  specific table extraction.
- **Coupled-loop hysteresis loss**: `iGSE` covers eddy + classical
  Steinmetz loss but not hysteresis loop area. Tracked alongside
  Jiles-Atherton parameter fitting.

## See also

- [`automatic-differentiation.md`](automatic-differentiation.md) — the
  AD path that the saturable devices' Newton Jacobian will land on
  once they're wired into the MNA stamp surface.
- [`linear-solver-cache.md`](linear-solver-cache.md) — the per-key LRU
  that benefits when transformer cycling produces the same `(topology,
  dt)` matrix across PWM cycles.
