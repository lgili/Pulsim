# Matriz de Paridade: PulsimGui -> Backend v1

Esta página documenta a paridade entre componentes do catálogo do PulsimGui e
o backend unificado v1 usado por YAML + Python.

- Snapshot de inventário: `2026-02-22`
- Escopo desta matriz: 34 componentes que antes eram ausentes no backend
- Resultado atual: 34/34 cobertos no backend (modelo físico, surrogate ou virtual)

## Status por componente

| Componente GUI | Tipo YAML canônico | Representação backend | Status |
| --- | --- | --- | --- |
| `BJT_NPN` | `bjt_npn` | surrogate sobre `mosfet` | supported |
| `BJT_PNP` | `bjt_pnp` | surrogate sobre `mosfet` | supported |
| `THYRISTOR` | `thyristor` | `switch` + controlador de latch/evento | supported |
| `TRIAC` | `triac` | `switch` + controlador bidirecional de latch/evento | supported |
| `SWITCH` | `switch` | dispositivo elétrico direto | supported |
| `FUSE` | `fuse` | `switch` + evento de trip por I²t | supported |
| `CIRCUIT_BREAKER` | `circuit_breaker` | `switch` + evento de overcurrent/delay | supported |
| `RELAY` | `relay` | par `switch` (NO/NC) + evento de bobina | supported |
| `OP_AMP` | `op_amp` | bloco virtual de controle | supported |
| `COMPARATOR` | `comparator` | bloco virtual de controle | supported |
| `PI_CONTROLLER` | `pi_controller` | bloco virtual de controle | supported |
| `PID_CONTROLLER` | `pid_controller` | bloco virtual de controle | supported |
| `MATH_BLOCK` | `math_block` | bloco virtual de controle | supported |
| `PWM_GENERATOR` | `pwm_generator` | bloco virtual de controle | supported |
| `INTEGRATOR` | `integrator` | bloco virtual de controle | supported |
| `DIFFERENTIATOR` | `differentiator` | bloco virtual de controle | supported |
| `LIMITER` | `limiter` | bloco virtual de controle | supported |
| `RATE_LIMITER` | `rate_limiter` | bloco virtual de controle | supported |
| `HYSTERESIS` | `hysteresis` | bloco virtual de controle | supported |
| `LOOKUP_TABLE` | `lookup_table` | bloco virtual de controle | supported |
| `TRANSFER_FUNCTION` | `transfer_function` | bloco virtual de controle | supported |
| `DELAY_BLOCK` | `delay_block` | bloco virtual de controle | supported |
| `SAMPLE_HOLD` | `sample_hold` | bloco virtual de controle | supported |
| `STATE_MACHINE` | `state_machine` | bloco virtual de controle | supported |
| `SATURABLE_INDUCTOR` | `saturable_inductor` | `inductor` + controlador virtual não linear | supported |
| `COUPLED_INDUCTOR` | `coupled_inductor` | 2x `inductor` + controlador virtual de acoplamento | supported |
| `SNUBBER_RC` | `snubber_rc` | macro expandida para ramo R-C | supported |
| `VOLTAGE_PROBE` | `voltage_probe` | instrumento virtual (sem stamp MNA) | supported |
| `CURRENT_PROBE` | `current_probe` | instrumento virtual (sem stamp MNA) | supported |
| `POWER_PROBE` | `power_probe` | instrumento virtual (sem stamp MNA) | supported |
| `ELECTRICAL_SCOPE` | `electrical_scope` | instrumento virtual/canais | supported |
| `THERMAL_SCOPE` | `thermal_scope` | instrumento virtual/canais térmicos | supported |
| `SIGNAL_MUX` | `signal_mux` | roteamento virtual | supported |
| `SIGNAL_DEMUX` | `signal_demux` | roteamento virtual | supported |

## Notas de execução

- Fase do scheduler mixed-domain por passo aceito:
  `electrical -> control -> events -> instrumentation`.
- Blocos virtuais não alteram topologia MNA.
- Estados de evento (trip/latch/contato) são determinísticos.

## Diagnósticos e compatibilidade

- Use `YamlParserOptions(strict=True)` para validações de pinagem/parâmetros.
- Todos os tipos da matriz acima não devem gerar `Unsupported component type`.
- Aliases de GUI são normalizados para nomes canônicos no parser.

## Gate de regressão recomendado (CI)

```bash
PYTHONPATH=build/python pytest -q python/tests/test_gui_component_parity.py
PYTHONPATH=build/python pytest -q python/tests/test_runtime_bindings.py
./build-test/core/pulsim_simulation_tests "[v1][yaml][gui-parity]"
```
