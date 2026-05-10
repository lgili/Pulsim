# Converter Templates

> Status: shipped — three fundamental DC-DC topologies + a PI
> compensator. Bridge / resonant / PFC / interleaved templates are the
> natural follow-up.

A converter template is a parametric `Circuit` factory: you supply
high-level design intent (`Vin`, `Vout`, `Iout`, `fsw`) and the
template produces a fully-wired `Circuit` with auto-sized passives, a
PWM-driven switch, free-wheeling diode, and resistive load. The
auto-design heuristics target standard ripple bounds (≤ 30 % current,
≤ 1 % voltage by default) so the resulting circuit converges in
simulation without parameter tuning.

| Topology | C++ helper | Python helper | Notes |
|---|---|---|---|
| Buck | `expand_buck(params)` in `templates/buck_template.hpp` | `pulsim.templates.buck(...)` | Sync rectified; PWM control; free-wheeling diode |
| Boost | `expand_boost(params)` | `pulsim.templates.boost(...)` | Vout > Vin; low-side switch; series inductor |
| Buck-Boost | `expand_buck_boost(params)` | `pulsim.templates.buck_boost(...)` | Inverting; |Vout| can be > or < Vin |
| PI compensator | `PiCompensator` (math class) | n/a today | Use `from_crossover(f_c, K_plant)` for the default tune |

## TL;DR

```python
import pulsim

exp = pulsim.templates.buck(
    Vin=24, Vout=5, Iout=2, fsw=100_000,    # design intent
    # ripple_pct=0.30, vout_ripple_pct=0.01 (defaults)
)

# Switch to PWL Ideal mode so segment-primary engine resolves switching:
exp.circuit.set_switching_mode_for_all(pulsim.SwitchingMode.Ideal)
exp.circuit.set_pwl_state("Q1", False)
exp.circuit.set_pwl_state("D1", False)

opts = pulsim.SimulationOptions()
opts.tstop = 1e-3
opts.dt = 1e-7
opts.adaptive_timestep = False
opts.switching_mode = pulsim.SwitchingMode.Ideal
opts.newton_options.num_nodes    = exp.circuit.num_nodes()
opts.newton_options.num_branches = exp.circuit.num_branches()

sim = pulsim.Simulator(exp.circuit, opts)
result = sim.run_transient(sim.dc_operating_point().newton_result.solution)
```

The `expansion` object exposes:

| Field | Type | Use |
|---|---|---|
| `circuit` | `pulsim.Circuit` | Pass to `Simulator` constructor |
| `parameters` | `dict[str, float]` | Resolved values (user inputs + auto-design) — useful for reproducibility |
| `notes` | `dict[str, str]` | Per-parameter explanation of auto-design decisions |
| `topology` | `str` | `"buck"`, `"boost"`, `"buck_boost"` |

## Parameters

All three topologies share the same call shape:

| Parameter | Required? | Default | Auto-designed when omitted? |
|---|---|---|---|
| `Vin` | ✓ | — | n/a |
| `Vout` | ✓ | — | n/a |
| `Iout` | ✓ | — | n/a |
| `fsw` | ✓ | — | n/a |
| `ripple_pct` | — | 0.30 | n/a |
| `vout_ripple_pct` | — | 0.01 | n/a |
| `L` | — | — | ✓ from ripple_pct |
| `C` | — | — | ✓ from vout_ripple_pct |
| `Rload` | — | — | ✓ from `Vout / Iout` |
| `q_g_on` | — | 1e3 (S) | n/a — switch on-conductance |
| `q_g_off` | — | 1e-9 (S) | n/a — switch off-conductance |

User-supplied `L`, `C`, or `Rload` always override the auto-design.

## Auto-design heuristics

### Buck

```
D     = Vout / Vin
ΔI    = ripple_pct · Iout
L     = (Vin - Vout) · D / (ΔI · fsw)
ΔV    = vout_ripple_pct · Vout
C     = ΔI / (8 · fsw · ΔV)
Rload = Vout / Iout
```

### Boost

```
D       = 1 - Vin / Vout
I_in    = Iout · (Vout / Vin)
ΔI      = ripple_pct · I_in
L       = Vin · D / (ΔI · fsw)
ΔV      = vout_ripple_pct · Vout
C       = Iout · D / (fsw · ΔV)
Rload   = Vout / Iout
```

### Buck-Boost (inverting)

```
D       = |Vout| / (Vin + |Vout|)
I_in    = Iout · |Vout| / Vin
ΔI      = ripple_pct · I_in
L       = Vin · D / (ΔI · fsw)
ΔV      = vout_ripple_pct · |Vout|
C       = Iout · D / (fsw · ΔV)
```

The PWM duty in each template is pre-set to the steady-state `D` so
the open-loop circuit converges to the target `Vout` without a
controller. Plug a `PiCompensator` (or your own controller) into the
`Vctrl` source to close the loop.

## PI compensator helper

```cpp
#include "pulsim/v1/templates/pi_compensator.hpp"
using namespace pulsim::v1::templates;

// Default tune: unity crossover at 1 kHz on a single-pole plant.
auto pi = PiCompensator::from_crossover(/*f_c*/1e3, /*K_plant*/1.0/24.0);

// In the control loop:
const Real u = pi.step(/*error*/Vref - Vmeas, /*dt*/1e-5);
```

The PI uses trapezoidal-discretized integration with anti-windup
back-calculation: when the unclamped output exceeds `[u_min, u_max]`,
the integrator state is rolled back so the next step starts clean.

`from_crossover(f_c, K_plant)` sets `Kp = 1/K_plant` and `Ki = 2π·f_c·Kp`
— the classical first-cut tune for a single-pole plant with DC gain
`K_plant`. For converter loops, `K_plant` is typically `1/Vin` for
duty-to-`Vout` (buck) or `Vout/Vin·(Vout-Vin)/Vin` for boost — see the
auto-tuning notes in the proposal.

## Validation

| Gate | Test | Result |
|---|---|---|
| **G.1 / G.2** Default-config transient + parameter validation | `test_converter_templates.cpp` | All three templates auto-design correctly, validate input ranges, and produce a Circuit that simulates cleanly to a finite `Vout` |
| **G.4** Per-template docs | This page | Three topologies covered today |

## Topology registry

The `ConverterRegistry` (header `templates/registry.hpp`) is the
runtime dispatch surface. C++ users can:

```cpp
using namespace pulsim::v1::templates;

register_buck_template();    // pulls expand_buck into the global registry
register_boost_template();
register_buck_boost_template();

const auto exp = ConverterRegistry::instance().expand("buck", {
    {"Vin",  24.0},
    {"Vout",  5.0},
    {"Iout",  2.0},
    {"fsw", 100e3},
});
```

The registry surfaces "did you mean" suggestions for typos
(`expand("bukc", ...)` → `"did you mean 'buck'?"`) and lists the
available topologies on unknown lookups.

## Follow-ups

- **Isolated topologies**: `flyback_template`, `forward_template`,
  `two_switch_forward_template`. Need the magnetic-core models'
  Circuit-variant integration so `SaturableTransformer` is registerable
  in YAML / `add_*` calls. Tracked alongside the Phase-3 deliverables
  in the magnetic change.
- **Bridge / resonant / DAB**: `half_bridge`, `full_bridge`,
  `llc_half_bridge`, `dab_template`. Need a dead-time generator
  primitive plus a phase-shift PWM source.
- **PFC / interleaved**: `totem_pole_pfc_template`,
  `interleaved_buck_2ph_template`. Need the current-loop helpers from
  the compensator library.
- **Type-II / Type-III compensators**: extend `PiCompensator` with
  zero-pole pairs. Targets crossover + phase-margin tuning.
- **YAML schema**: `type: buck_template` with `parameters: {...}`
  block, dispatched by the existing parser. Lands together with the
  Circuit-variant integration follow-up that wires the `Circuit`
  fragment into the YAML's `components:` array.
- **Telemetry**: per-template-instance expansion time, auto-designed
  parameters reported in `BackendTelemetry.template_expansions`.

## See also

- [`magnetic-models.md`](magnetic-models.md) — provides the saturable
  inductors / transformers that isolated converter templates will use.
- [`catalog-devices.md`](catalog-devices.md) — provides the
  vendor-data MOSFETs / IGBTs / diodes that templates will pull
  switching elements from once the Circuit-variant integration lands.
- [`ac-analysis.md`](ac-analysis.md) — how to do `run_ac_sweep` on a
  template's circuit to validate loop-gain margins.
