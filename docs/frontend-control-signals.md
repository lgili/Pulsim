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

## 3) Agendamento de controle (Ts por bloco + fallback legado)

O contrato atual é **por bloco de controle**:

- use `component.sample_time` (ou aliases `ts`/`Ts`) em cada bloco de controle;
- `Ts = 0` (ou ausente): execução contínua (a cada passo aceito);
- `Ts > 0`: execução discreta com hold entre atualizações (`dt_acumulado >= Ts`).

Blocos suportados para `sample_time`: PI/PID, PWM, C-block e demais blocos de controle virtuais.
Scopes/probes não aceitam `sample_time`.

Compatibilidade legada (`simulation.control.mode/sample_time`) continua ativa como fallback:

- `continuous`: força contínuo global.
- `discrete`: `sample_time` global é usado apenas quando o bloco não declara `sample_time` local.
- `auto`:
  - se `sample_time` global > 0, usa fallback global;
  - se existir qualquer `sample_time` local, desabilita inferência global por PWM;
  - sem `sample_time` local/global, ainda pode inferir `Ts = 1/f_pwm_max` para fallback legado.

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

### Averaged converter (planta média)

Quando `simulation.averaged_converter.enabled: true`, os canais canônicos são:

- `Iavg(<inductor_name>)`
- `Vavg(<output_node>)`
- `Davg`

Modos suportados no backend:

- `operating_mode: ccm | dcm | auto`
- `topology: buck | boost | buck_boost`

Contrato de consumo no front:

- plote estes canais diretamente quando existirem;
- mantenha alinhamento estrito com `result.time`;
- use metadata para roteamento (`component_type == "averaged_converter"` e `source_component == "averaged_state"`).

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

Para averaged converter:

- `component_type == "averaged_converter"`
- `source_component == "averaged_state"`
- `domain == "time"`
- unidades esperadas:
  - `Iavg(...)`: `A`
  - `Vavg(...)`: `V`
  - `Davg`: `ratio`

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
- Ignorar `sample_time` por bloco e assumir uma taxa global única.
- Ignorar `virtual_channel_metadata` e inferir domínio por regex de nome.
- Sintetizar `Iavg/Vavg/Davg` localmente quando backend não exportou o canal.

## 9) Responsabilidades GUI-only

- UX de entrada de dados (formularios, validações visuais, assistentes).
- Importação/digitalização de curvas para gerar arrays numéricos de entrada.
- Organização de painéis, legenda, escala e interação de plots.
- Conversão/apresentação de unidades no nível de interface.

## 10) Comportamentos proibidos no GUI

- Criar curva térmica sintética quando `T(...)` não existe no resultado.
- Inferir perdas por heurística local sem usar canais `Pcond/Psw_on/Psw_off/Prr/Ploss`.
- Reescrever, filtrar ou suavizar fisicamente os traços do backend sem sinalizar que é pós-processamento.
- Corrigir inconsistências de resumo no frontend em vez de reportar erro de contrato.
