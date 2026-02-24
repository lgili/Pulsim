# Configuration Guide

Este guia concentra as configurações que mais impactam convergência, tempo de simulação e fidelidade.

## Bloco `simulation`

Configuração base:

```yaml
simulation:
  tstart: 0.0
  tstop: 2e-3
  dt: 1e-6
  step_mode: variable   # fixed | variable
  dt_min: 1e-10
  dt_max: 1e-4
  integrator: trbdf2
```

Campos práticos:

- `tstop`, `dt`: resolução e duração.
- `step_mode`: `fixed` para grade determinística; `variable` para adaptação automática.
- `adaptive_timestep`: override avançado (evite no fluxo canônico; prefira `step_mode`).
- `integrator`: `trapezoidal`, `bdf1..bdf5`, `trbdf2`, `rosenbrockw`, `sdirk2`.

Observação de migração:

- `simulation.backend` e `simulation.sundials` são campos legados para transiente e não
  fazem parte do caminho suportado; use `simulation.step_mode`.

## Solver linear em runtime

```yaml
simulation:
  solver:
    order: [klu, gmres]
    fallback_order: [sparselu]
    allow_fallback: true
    auto_select: true
    size_threshold: 400
    nnz_threshold: 2500
    diag_min_threshold: 1e-12
```

Quando usar:

- `KLU`/`SparseLU`: redes pequenas/médias com robustez alta.
- `GMRES`/`BiCGSTAB`: redes grandes e esparsas.
- `fallback_order`: proteção para casos difíceis.

## Iterativo e precondicionador

```yaml
simulation:
  solver:
    iterative:
      max_iterations: 300
      tolerance: 1e-8
      restart: 40
      preconditioner: ilut
      ilut_drop_tolerance: 1e-3
      ilut_fill_factor: 10
      enable_scaling: true
      scaling_floor: 1e-12
```

Regras rápidas:

- comece com `ilut` em conversores maiores;
- aumente `max_iterations` se ficar próximo de convergir;
- `enable_scaling` ajuda matrizes mal condicionadas.

## Newton, fallback e robustez

```yaml
simulation:
  newton:
    max_iterations: 60
    enable_limiting: true
    max_voltage_step: 2.0
    max_current_step: 5.0
```

Complementos úteis:

- `max_step_retries`: quantas tentativas por passo.
- `gmin_fallback`: reforço de robustez em regiões difíceis.
- `fallback_policy`: rastreio e política de retry (`fallback_trace`).

## Perdas e térmico

```yaml
simulation:
  enable_losses: true
  thermal:
    enable: true
    ambient: 25.0
    policy: loss_with_temperature_scaling
    default_rth: 1.5
    default_cth: 0.02
```

Saídas relevantes:

- `result.loss_summary`
- `result.thermal_summary`
- `result.events` para inspeção de chaveamento/eventos.

## Estratégias de configuração

### Debug rápido

- `step_mode: fixed`
- `integrator: trapezoidal`
- `order: [sparselu]`

### Produção robusta

- `step_mode: variable`
- `integrator: trbdf2` ou `rosenbrockw`
- `order: [klu, gmres]`
- `fallback_order: [sparselu]`
