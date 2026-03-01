# YAML Netlist and Solver Reference

## Table of Contents
- [Netlist Skeleton](#netlist-skeleton)
- [Top-Level Keys](#top-level-keys)
- [Component Types](#component-types)
- [Waveform Keys](#waveform-keys)
- [Simulation and Solver Keys](#simulation-and-solver-keys)
- [Validation Checklist](#validation-checklist)

## Netlist Skeleton

```yaml
schema: pulsim-v1
version: 1
simulation:
  tstart: 0.0
  tstop: 1e-3
  dt: 1e-6
components:
  - type: voltage_source
    name: V1
    nodes: [in, 0]
    waveform: {type: dc, value: 12}
```

## Top-Level Keys

- `schema` (required): must be `pulsim-v1`.
- `version` (required): currently `1`.
- `components` (required): list of components.
- `simulation` (optional): transient/solver options.
- `models` (optional): reusable model templates.

## Component Types

- Passive:
- `resistor` (`R`), key: `value`
- `capacitor` (`C`), keys: `value`, optional `ic`
- `inductor` (`L`), keys: `value`, optional `ic`
- Sources:
- `voltage_source` (`V`) with `waveform`
- `current_source` (`I`) with `value` (DC)
- Switching:
- `diode` (`D`) with `g_on`, `g_off`
- `switch` (`S`) with `ron/roff` or `g_on/g_off`, `initial_state`
- `vcswitch` with `v_threshold`, `g_on`, `g_off`
- Power devices:
- `mosfet`, `nmos`, `pmos` (`M`)
- `igbt` (`Q`)
- Transformer:
- `transformer` (`T`) with `turns_ratio`

## Waveform Keys

- `dc`: `value`
- `pulse`: `v_initial`, `v_pulse`, `t_delay`, `t_rise`, `t_fall`, `t_width`, `period`
- `sine`: `amplitude`, `frequency`, `offset`, `phase`
- `pwm`: `v_high`, `v_low`, `frequency`, `duty`, `dead_time`, `phase`

## Simulation and Solver Keys

Common keys under `simulation`:

- `tstart`, `tstop`, `dt`, `dt_min`, `dt_max`
- `adaptive_timestep`
- `enable_events`
- `enable_losses`
- `integrator` (e.g. `trapezoidal`, `bdf2`, `trbdf2`, `rosenbrockw`, `sdirk2`)

Solver stack (`simulation.solver`):

- `order`, `fallback_order`, `allow_fallback`, `auto_select`
- `size_threshold`, `nnz_threshold`, `diag_min_threshold`
- `preconditioner`, `ilut_drop_tolerance`, `ilut_fill_factor`
- `iterative.max_iterations`, `iterative.tolerance`, `iterative.restart`
- `iterative.enable_scaling`, `iterative.scaling_floor`
- `nonlinear.anderson.*`, `nonlinear.broyden.*`, `nonlinear.newton_krylov.enable`
- `nonlinear.trust_region.radius/shrink/expand/min/max`
- `reuse_jacobian_pattern`

Newton alternative path (`simulation.newton`):

- `max_iterations`
- `enable_newton_krylov`
- `krylov_residual_cache_tolerance`
- `reuse_jacobian_pattern`

Periodic options:

- `shooting.period`, `shooting.max_iterations`, `shooting.tolerance`, `shooting.relaxation`
- `harmonic_balance.period`, `num_samples`, `max_iterations`, `tolerance`, `relaxation`

## Validation Checklist

- `schema` and `version` are correct.
- Every component has valid `name`, `type`, and `nodes`.
- Units/prefixes are intentional (`u`, `m`, `k`, `meg`, etc.).
- `dt` is much smaller than switching period for switched converters.
- For parser path: use `YamlParserOptions`, then `YamlParser.load()`.
- For runtime path: set `options.newton_options.num_nodes` and `num_branches` from circuit before simulation.
