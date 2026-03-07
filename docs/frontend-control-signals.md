# Frontend Control and Signal Contract

Este documento define o contrato que o frontend deve usar para consumir sinais de controle, eventos, instrumentação e térmico no `SimulationResult`.

Objetivo: evitar heurísticas frágeis no front e garantir que o que é plotado corresponde ao que o core realmente simula.

## 1) Base temporal única

- A base temporal oficial é `result.time`.
- O backend adiciona amostras de canais a cada passo **aceito** de transiente.
- Regra de consumo no front:
  - sempre use `result.time` no eixo X;
  - assuma que cada série em `result.virtual_channels[...]` é alinhada com `result.time`.

## 2) Ordem de execução mixed-domain por amostra

Para cada amostra, o scheduler executa nesta ordem:

1. `electrical`
2. `control`
3. `events`
4. `instrumentation`

Isso importa para interpretação:

- PI/PID são atualizados antes dos blocos de evento daquele mesmo instante.
- `pwm_generator.target_component` já afeta estado do switch no passo mixed-domain.

## 3) Modos de atualização de controle

Configuração: `simulation.control.mode` (`auto|continuous|discrete`) e `sample_time`.

- `continuous`:
  - blocos de controle (PI/PID) atualizam em todo passo aceito.
- `discrete`:
  - PI/PID atualizam apenas quando `dt_acumulado >= sample_time`;
  - entre amostras, o canal mantém o último valor (hold).
- `auto`:
  - usa `sample_time` se definido;
  - senão tenta inferir por frequência PWM.

## 4) Canais canônicos por tipo

### Controle

- `PI1`: saída do `pi_controller`.
- `PID1`: saída do `pid_controller`.
- `PWM1`: estado PWM (0/1).
- `PWM1.duty`: duty efetivo já clampado (`duty_min`/`duty_max`).
- `PWM1.carrier`: portadora triangular normalizada.

Para malha fechada com PWM:

- use `PWM1.duty` para gráfico de duty;
- não derive duty de `PWM1` (é estado lógico, não razão cíclica).

### Eventos

- relay:
  - `K1.state`, `K1.no_state`, `K1.nc_state`
- fuse/circuit_breaker/thyristor/triac:
  - `X.state`
- thyristor/triac também expõem:
  - `X.trigger`, `X.i_est`

### Instrumentação

- `voltage_probe`, `current_probe`, `power_probe`, `electrical_scope` etc.

### Térmico (junção/dispositivo)

Canal térmico oficial de transiente:

- `T(<component_name>)`
  - exemplos: `T(M1)`, `T(D1)`, `T(Rload)`

Condições para existir:

- `simulation.enable_losses: true`
- `simulation.thermal.enabled: true`
- componente com thermal habilitado no runtime (`component.thermal.enabled`/config equivalente)

Importante:

- `THERMAL_SCOPE` **não** é a temperatura de junção do componente.
- para temperatura de dispositivo, use sempre `T(<component>)`.

### Perdas (potência por componente)

Canais canônicos de perdas instantâneas (amostrados por passo aceito):

- `Pcond(<component_name>)`: potência de condução (W)
- `Psw_on(<component_name>)`: potência equivalente de turn-on no passo (W)
- `Psw_off(<component_name>)`: potência equivalente de turn-off no passo (W)
- `Prr(<component_name>)`: potência equivalente de reverse recovery no passo (W)
- `Ploss(<component_name>)`: potência total (`Pcond + Psw_on + Psw_off + Prr`) (W)

Condição para existir:

- `simulation.enable_losses: true`

## 5) Metadata obrigatória para roteamento no front

Não use heurística por nome; use `result.virtual_channel_metadata[channel]`.

Para canal térmico canônico:

- `domain == "thermal"`
- `component_type == "thermal_trace"`
- `source_component == "<component_name>"`
- `unit == "degC"`

Para canal canônico de perdas:

- `domain == "loss"`
- `source_component == "<component_name>"`
- `unit == "W"`

Para controle/eventos/instrumentação:

- use `domain` para escolher painel/eixo.

## 6) Consistência numérica que o front pode validar

Para qualquer componente térmico `X`:

- `len(T(X)) == len(result.time)`
- `component_electrothermal[X].final_temperature == last(T(X))`
- `component_electrothermal[X].peak_temperature == max(T(X))`
- `component_electrothermal[X].average_temperature == mean(T(X))`

Essas relações são o contrato canônico do backend.

## 7) Fluxo recomendado no frontend

1. Executar simulação e ler `result.time`.
2. Enumerar `result.virtual_channels`.
3. Para cada canal:
  - ler `meta = result.virtual_channel_metadata[channel]`;
  - roteamento por `meta.domain`:
    - `control` -> painel de controle;
    - `events` -> painel de estados;
    - `instrumentation` -> painel elétrico;
    - `thermal` -> painel térmico.
4. Para térmico, exibir unidade `degC` e mapear componente por `meta.source_component`.

## 8) Erros comuns no front

- Usar `THERMAL_SCOPE` para temperatura de junção.
- Plotar `PWM1` como se fosse duty (o correto é `PWM1.duty`).
- Ignorar `control.mode=discrete` e esperar atualização contínua de PI/PID.
- Ignorar `virtual_channel_metadata` e inferir domínio por regex de nome.
