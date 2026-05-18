# Convergence Tuning Guide (Python + YAML)

Este guia foca no runtime suportado: Python + YAML (`pulsim-v1`).

## ⚠️ Primeiro passo — escolha um preset

Antes de tunar campo a campo, **escolha um preset numérico** em
[`numerical-configuration.md`](numerical-configuration.md). 95% dos
casos de não-convergência se resolvem trocando `Preset.Auto` por
`Preset.Robust` (que liga TRBDF2 + stiffness + 12 retries + Armijo
line search + homotopy DC):

```python
opts = ps.SimulationOptions.from_preset(ps.Preset.Robust, 1e-6, 1e-3)
```

```yaml
simulation:
  preset: robust
  tstop: 1e-3
  dt: 1e-6
```

O resto deste guia é para os casos restantes que precisam de tuning
fino. Se o preset resolver, **não passe daqui**.

## Sintomas comuns

- Falha no ponto de operação DC (`dc_operating_point`).
- Muitos `timestep_rejections` no transiente.
- Queda para fallback linear com frequência alta.
- Passos muito pequenos e simulação lenta.

## Convergence aids automáticas (ligadas por default)

Pulsim ship com quatro aids automáticas que disparam em casos
difíceis sem o usuário precisar configurar — todas chegaram com o
release `simplify-and-harden-numerical-surface`:

| Aid | Quando dispara | Sintoma de que ajudou |
|---|---|---|
| **Armijo line search** (Newton) | Sempre que o full step Newton aumenta o resíduo | `result.newton_result.telemetry.line_search_backtracks > 0` |
| **Coalescência de eventos simultâneos** (PWL) | ≥ 2 switches comutam dentro da tolerância de bisection | `result.backend_telemetry.simultaneous_event_groups > 0` |
| **Iterative refinement** (KLU/SparseLU) | Resíduo pós-back-solve > 10·ε_machine | `result.linear_solver_telemetry.linear_refinement_steps > 0` |
| **Homotopy continuation** (DC OP) | Direct → Source → Gmin → PseudoTransient all fail | `result.dc_result.strategy_used == DCStrategy.Homotopy` |

Veja [Numerical Configuration](numerical-configuration.md#convergence-aids)
para detalhes e tuning de cada uma.

## Checklist rápido

1. **Pegue o preset certo** (`Preset.Robust` é o ponto de partida).
2. Garanta caminho DC para o terra (evite nós flutuantes).
3. Use `simulation.solver.order` e `fallback_order` explícitos só se
   precisar mais que o auto-selector.
4. Em circuitos stiff, o preset Robust já escolhe `integrator: trbdf2`
   automaticamente. Override para `rosenbrockw` só se TRBDF2 falhar.
5. `Preset.Fast` quando o circuito é só switching.

## Verificar backends compilados

No Python:

```python
import pulsim as ps
print(ps.backend_capabilities())
# {'klu': True, 'hypre_amg': True/False, 'sundials': True/False}
```

Observação: `sundials=True` indica apenas suporte compilado opcional. No caminho
suportado de transiente, use o core nativo com `simulation.step_mode`
(`fixed|variable`), sem seleção de backend legado.

## Exemplo de configuração robusta

```yaml
schema: pulsim-v1
version: 1
simulation:
  tstart: 0.0
  tstop: 2e-3
  dt: 1e-6
  step_mode: variable
  dt_min: 1e-10
  dt_max: 5e-5
  adaptive_timestep: true  # override avançado
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
