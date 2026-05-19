# Configuration Guide

Este guia concentra as configurações que mais impactam convergência, tempo de simulação e fidelidade.

## ⚠️ Primeiro passo — escolha um `preset`

Antes de configurar qualquer outro campo, pegue um dos quatro presets
em [`numerical-configuration.md`](numerical-configuration.md). 95%
dos casos se resolvem com uma única decisão:

```yaml
simulation:
  preset: robust          # auto | fast | robust | high_fidelity
  tstop: 1e-3
  dt: 1e-6
```

O resto deste guia é override avançado que só precisa quando o
preset não basta.

## Bloco `simulation` — referência completa

Tabela canônica de cada campo `simulation.*` reconhecido pelo parser
YAML (`pulsim-v1`).

### Top-level (recomendado para usuários)

| Campo | Valores | Default | Notas |
|---|---|---|---|
| `preset` | `auto` / `fast` / `robust` / `high_fidelity` | — | Materializa um perfil numérico completo. Outros campos `simulation.*` aplicam ON TOP do preset. |
| `tstart` | float | `0.0` | Tempo inicial (s) |
| `tstop` | float | `1e-3` | Tempo final (s) |
| `dt` | float | `1e-6` | Passo inicial / fixo (s) |
| `dt_min` | float | `1e-12` | Limite mínimo do passo adaptativo |
| `dt_max` | float | `1e-3` | Limite máximo do passo adaptativo |
| `step_mode` | `fixed` / `variable` | `variable` | Canônico — substitui `adaptive_timestep` |
| `switching_mode` | `auto` / `ideal` / `behavioral` | `auto` (resolve para `behavioral` hoje) | PWL fast path = `ideal` |
| `integrator` | ver tabela abaixo | `trapezoidal` (`trbdf2` em Robust) | |
| `enable_events` | bool | `true` | |
| `enable_losses` | bool | `true` | |
| `enable_bdf_order_control` | bool | `false` (`true` em Robust) | Ordem BDF adaptativa |
| `formulation` | `projected_wrapper` / `direct` | `projected_wrapper` | DAE assembly strategy |

### Integradores aceitos

| Valor | Status |
|---|---|
| `trapezoidal` | ✅ Default — order 2, A-stable |
| `bdf1` (ou `be`) | ✅ Backward Euler — L-stable fallback |
| `bdf2` | ✅ Order 2, A-stable |
| `trbdf2` | ✅ Híbrido trapezoidal + BDF2 — default Robust |
| `rosenbrockw` (ou `rosenbrock`) | ✅ Linearly implicit stiff |
| `bdf3`, `bdf4`, `bdf5` | ⚠️ Deprecated — emitem warning, removidos no próximo major |
| `gear` | ⚠️ Deprecated — alias literal de `bdf2` |
| `sdirk2` | ⚠️ Deprecated — sem cobertura de benchmark |

### Sub-blocos avançados

> Status: ainda **não** colapsados sob `advanced.*` na release atual
> (Phase 3 do `simplify-and-harden-numerical-surface` change). Os
> campos abaixo seguem reconhecidos no top-level `simulation.*` por
> compatibilidade. Quando Phase 3 ship, eles migrarão.

| Sub-bloco | Conteúdo | Quem usa |
|---|---|---|
| `newton` | `max_iterations`, `armijo_line_search`, `armijo_sigma`, ... | Tuning do Newton interno |
| `timestep` | `error_tolerance`, `growth_factor`, `safety_factor`, ... | Controlador adaptativo de `dt` |
| `lte` | `voltage_tolerance`, `current_tolerance`, `use_weighted_norm` | Estimação de erro local |
| `bdf` | `min_order`, `max_order`, `initial_order` | Controle de ordem BDF |
| `solver` | `order`, `fallback_order`, `direct_config`, `iterative_config` | Stack de solvers lineares |
| `dc_config` | `strategy`, `homotopy_config`, sub-configs | DC operating point |
| `stiffness` | `enable`, `switch_integrator`, `stiff_integrator`, thresholds | Detecção e mitigação de stiffness |
| `fallback` | `enable_transient_gmin`, `gmin_initial`, etc. | Recuperação de divergência |

### Campos depreciados

| Campo | Substituto | Quando some |
|---|---|---|
| `adaptive_timestep: bool` | `step_mode: fixed \| variable` | Próximo major |
| `direct_formulation_fallback: bool` | (sempre on internamente) | Próximo major |
| `integrator: bdf3 \| bdf4 \| bdf5` | `bdf2` / `trbdf2` / `rosenbrockw` | Próximo major |
| `integrator: gear` | `bdf2` | Próximo major |
| `integrator: sdirk2` | `trbdf2` | Próximo major |
| `simulation.backend`, `simulation.sundials` | use `step_mode` + `formulation` no caminho canônico | Já não é caminho suportado |

Veja [Migration Guide § 8](migration-guide.md#8-numerical-surface--v010--v011)
para recipes de migração concretos por campo.

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
- `formulation: projected_wrapper`
- `integrator: trbdf2` ou `rosenbrockw`
- `order: [klu, gmres]`
- `fallback_order: [sparselu]`

### DAE direto (diagnóstico/perfil)

- `step_mode: fixed` ou `variable`
- `formulation: direct`
- opcional: `direct_formulation_fallback: false` para comparar somente rota direta
