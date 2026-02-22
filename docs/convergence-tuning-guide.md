# Convergence Tuning Guide (Python + YAML)

Este guia foca no runtime suportado: Python + YAML (`pulsim-v1`).

## Sintomas comuns

- Falha no ponto de operação DC (`dc_operating_point`).
- Muitos `timestep_rejections` no transiente.
- Queda para fallback linear com frequência alta.
- Passos muito pequenos e simulação lenta.

## Checklist rápido

1. Garanta caminho DC para o terra (evite nós flutuantes).
2. Use `simulation.solver.order` e `fallback_order` explícitos.
3. Em circuitos stiff, prefira `integrator: trbdf2` ou `rosenbrockw`.
4. Ative precondicionador ILUT para malhas grandes.
5. Use `adaptive_timestep: false` quando quiser baseline determinístico.

## Verificar backends compilados

No Python:

```python
import pulsim as ps
print(ps.backend_capabilities())
# {'klu': True, 'hypre_amg': True/False, 'sundials': True/False}
```

Observação: `sundials=True` indica suporte compilado, mas use primeiro o pipeline robusto padrão (`run_transient`) e o YAML v1 com fallback de solver para casos de chaveamento.

## Exemplo de configuração robusta

```yaml
schema: pulsim-v1
version: 1
simulation:
  tstart: 0.0
  tstop: 2e-3
  dt: 1e-6
  dt_min: 1e-10
  dt_max: 5e-5
  adaptive_timestep: true
  integrator: trbdf2

  solver:
    order: [klu, gmres]
    fallback_order: [sparselu]
    allow_fallback: true
    auto_select: true
    size_threshold: 400
    nnz_threshold: 2500
    diag_min_threshold: 1e-12
    iterative:
      max_iterations: 300
      tolerance: 1e-8
      restart: 40
      preconditioner: ilut
      ilut_drop_tolerance: 1e-3
      ilut_fill_factor: 10
      enable_scaling: true
      scaling_floor: 1e-12

  newton:
    max_iterations: 60
    enable_limiting: true
    max_voltage_step: 2.0
    max_current_step: 5.0
```

## Ajustes por cenário

### Buck/boost com chaveamento rápido

- `integrator: trbdf2`
- `dt` inicial pequeno (ordem de 1/50 a 1/200 do período de comutação)
- `enable_events: true`

### Malha grande com muitos passivos

- `order: [gmres, klu]` ou `[klu, gmres]` (teste as duas)
- ILUT ligado
- aumente `iterative.max_iterations`

### Problema quase singular

- aumente `diag_min_threshold`
- reduza `max_voltage_step`
- aumente `gmin_fallback` e `max_step_retries`

## Observabilidade para diagnóstico

No resultado transiente, acompanhe:

- `result.newton_iterations_total`
- `result.timestep_rejections`
- `result.linear_solver_telemetry.total_fallbacks`
- `result.fallback_trace`

Esses campos são essenciais para comparar tuning entre cenários e evitar regressão.

## Estratégia de validação

1. Ajuste em circuito pequeno de referência.
2. Replique no benchmark matrix.
3. Confirme em `ngspice` e depois `LTspice`.
4. Execute stress tiers A/B/C antes de promover configuração.
