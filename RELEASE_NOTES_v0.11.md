# Release Notes — Pulsim v0.11

> **Source change:** `simplify-and-harden-numerical-surface` (PR #13,
> archived 2026-05-18 under
> `openspec/changes/archive/2026-05-18-simplify-and-harden-numerical-surface/`).
> This file mirrors that change's release notes at the repo root for
> discoverability; the final v0.11 release notes will fold these in
> alongside other landed work.

## Highlights

Pulsim's numerical configuration surface drops from **~50
hand-tunable knobs** to **one `Preset` decision**, and the
convergence engine gains four automatic aids that close standing gaps
on multilevel converters (NPC, T-type, flying-cap, MMC).

| What changed | Who benefits |
|---|---|
| `Preset` enum + `SimulationOptions.from_preset(...)` factory | Every user — no more 50-field tuning |
| 4 convergence aids on by default (Armijo, simultaneous events, iterative refinement, homotopy) | Multilevel + nonlinear circuit users |
| MMC topology template (single-arm + 3φ + cap balancing) | Anyone simulating an MMC |
| 5 deprecated integrators (BDF3-5, Gear, SDIRK2) marked + warned | Cleaner enum surface |
| Friendly `LinearSolverKind.{Auto, Direct, Iterative}` + `SolverQuality.{Fast, Default, Best}` + `DCStrategy.Override` | Smaller user-facing API, same internal capability |

## New user-facing API

### `Preset` — single named numerical profile

```python
import pulsim as ps
opts = ps.SimulationOptions.from_preset(ps.Preset.Auto,
                                          dt=1e-6, tstop=1e-3)
ps.Simulator(circuit, opts).run_transient()
```

```yaml
simulation:
  preset: robust          # auto | fast | robust | high_fidelity
  tstop: 1e-3
  dt: 1e-6
```

Four profiles:

- **`Auto`** (default) — currently maps to `Robust`. Tracks the
  production recommendation.
- **`Fast`** — pure-switching topologies (buck, boost, 3φ VSI).
  PWL Ideal + Trapezoidal + Fixed step + KLU.
- **`Robust`** — motor drives, mixed-domain, magnetics, thermal.
  TRBDF2 + Variable step + stiffness on + 12 retries.
- **`HighFidelity`** — PLECS / PSIM / SPICE parity runs. Robust +
  10× tighter LTE + small `dt_max` + 24 retries.

Full reference: [docs/numerical-configuration.md](../docs/numerical-configuration.md).

### Convergence aids (all default-on)

Four aids run automatically when their trigger conditions fire — no
user config required:

1. **Armijo line search inside Newton** — uses the textbook
   `||f(x+α·dx)|| ≤ (1−σ·α)·||f(x)||` criterion (σ = 1e-4) instead
   of the legacy "any reduction" check. Recovers from oversized
   Newton steps that previously diverged.
2. **Simultaneous event coalescence in PWL engine** — when ≥ 2
   switches commute within bisection tolerance, they're applied
   atomically in one Newton solve instead of serialised across N
   steps. Fixes MMC and 3φ-VSI synchronous-gate convergence.
3. **Iterative refinement on KLU** — automatic post-solve residual
   check + one round of refinement when ill-conditioned. Triggers on
   floating-cap topologies.
4. **Homotopy continuation as 5th DC strategy** — λ-stepping from
   linear MNA to full nonlinear model. Last-resort in
   `DCStrategy.Auto` for hard cold-starts (NPC, MMC, flying-cap).

New telemetry counters:

```python
result.newton_result.telemetry.line_search_backtracks
result.backend_telemetry.simultaneous_event_groups
result.linear_solver_telemetry.linear_refinement_steps
result.dc_result.homotopy_steps
result.dc_result.homotopy_ladder_completed
```

### MMC topology template

```cpp
#include "pulsim/v1/templates/mmc.hpp"
using namespace pulsim::v1;

// Single-arm half-bridge submodule chain.
auto [ckt, h] = templates::mmc_arm(templates::MmcArmParams{
    .num_submodules = 9,
    .V_dc           = 900.0,
    .L_arm          = 1e-3,
    .C_submodule    = 2e-3,
});

// Full 3φ MMC inverter: 6 arms (upper + lower per phase).
auto [ckt2, h2] = templates::mmc_3phase_inverter(
    templates::Mmc3PhaseParams{
        .num_submodules_per_arm = 4,
        .V_dc                   = 600.0,
});

// Round-robin cap-balancing controller (pure helper).
std::vector<templates::MmcSubmoduleState> states = { ... };
auto cmds = templates::mmc_balance_submodules(states,
                                                 arm_current,
                                                 num_inserted);
```

Reference YAML netlist: `templates::mmc_example_yaml()` returns a
9-submodule arm string users can copy as a starting point.

Full reference: [docs/multilevel-converters.md](../docs/multilevel-converters.md).

### Collapsed public enums (additive)

`LinearSolverKind` gains 3 friendly values alongside the existing 6
concrete engines:

```python
ps.LinearSolverKind.Auto       # let runtime pick by system size
ps.LinearSolverKind.Direct     # force KLU (or EnhancedSparseLU/SparseLU)
ps.LinearSolverKind.Iterative  # force GMRES
# + ps.LinearSolverKind.{SparseLU, EnhancedSparseLU, KLU, GMRES, BiCGSTAB, CG}
```

`DCStrategy` gains `Override` (skip the Auto ladder, run only
`strategy_override`).

`SolverQuality` (new enum) replaces the leaky `PreconditionerKind`
enum in the user-facing API:

```python
cfg = ps.LinearSolverStackConfig()
cfg.solver_quality = ps.SolverQuality.Best   # Fast / Default / Best
cfg.apply_solver_quality()
```

## Deprecations

The following are deprecated in v0.11 — they emit warnings now, and
will be **removed in the next major release**:

| Field / value | Replacement |
|---|---|
| `Integrator.BDF3` / `BDF4` / `BDF5` | `Integrator.BDF2`, `TRBDF2`, or `RosenbrockW` |
| `Integrator.Gear` | `Integrator.BDF2` (literal alias) |
| `Integrator.SDIRK2` | `Integrator.TRBDF2` |
| `SimulationOptions.adaptive_timestep: bool` | `step_mode: StepMode.Fixed \| Variable` |
| `SimulationOptions.direct_formulation_fallback: bool` | (always-on internally now) |

## BREAKING changes — none in this release

The Phase 11 design originally called for flipping `SwitchingMode.Auto`
to resolve to `Ideal` instead of `Behavioral`. The flip was attempted
and revealed real PWL Ideal stability gaps on legacy buck-converter +
diode-loss circuits (vout overshoot to 20V, V_diode spikes to −1067V).
**Phase 11 is BLOCKED** pending PWL Ideal hardening on those topologies;
documented in OpenSpec tasks.md § 11 with unblock criteria.

`SwitchingMode.Auto` continues to resolve to `Behavioral` in v0.11 for
back-compat.

## What's still pending

| Phase | Status |
|---|---|
| **3** — `AdvancedOptions` namespace refactor | Scoped for dedicated PR (~4-6h work) |
| **8.6** — migrate every example to use the new abstract enum values | Partial (12 examples done; ~6 remaining + notebooks) |
| **11** — `SwitchingMode.Auto → Ideal` flip | BLOCKED on PWL Ideal hardening |
| **13** — PLECS/PSIM benchmarks vs golden CSVs | Needs external license + manual export |

## Migration

See [docs/migration-guide.md § 8](../docs/migration-guide.md#8-numerical-surface--v010--v011)
for per-deprecation migration recipes.

Quick recipe for the common case:

```python
# Before:
opts = ps.make_robust_options(circuit, 0.0, 1e-3, 1e-6,
                                ps.NewtonOptions(),
                                ps.LinearSolverStackConfig.defaults())

# After:
opts = ps.SimulationOptions.from_preset(ps.Preset.Robust,
                                          dt=1e-6, tstop=1e-3)
```

## Numbers

- **Code shipped**: ~5500 lines added (across 11 commits in PR #13)
- **Tests added**: 39 new test cases / 200+ assertions
- **OpenSpec tasks closed**: 70 / 99 (was 0 at proposal time)
- **Phases complete**: 10 of 15 (1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14
  fully or partially)
- **Phases blocked**: 1 (Phase 11)
- **Phases deferred**: 3 (Phase 3 — too big; Phase 13 — needs external
  resources; Phase 15 — depends on Phase 13)

## Acknowledgements

The numerical-surface audit document
(`docs/numerical-modes-audit.md`) catalogued the 50-field state of
the old API and motivated this change. The four convergence aids
(Armijo, simultaneous events, iterative refinement, homotopy) were
all in the OpenSpec proposal up front; this PR proves they land
without breaking the regression suite.

Co-authored with Claude (Anthropic).
