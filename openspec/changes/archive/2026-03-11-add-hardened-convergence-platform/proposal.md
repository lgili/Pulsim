## Why
Os cenarios atuais mostram um padrao recorrente: melhorias locais para um circuito dificil resolvem um caso, mas podem degradar outros (ex.: muitos diodos, cruzamento por zero, comutacao intensa, magneticos nao lineares, e malhas de controle PI/C-Block).

Precisamos evoluir de ajustes ad-hoc para uma plataforma de convergencia orientada por politica, com classificacao de falha, gates por fase e observabilidade robusta para evitar regressao cruzada.

## Scope Strategy
- Entregar em fatias incrementais (observacao -> classificacao -> politica ativa), evitando rollout big-bang.
- Tornar `strict` o contrato de referencia para determinismo; `balanced` e `robust` entram como perfis progressivos.
- Tratar trilha de solver avancado (SUNDIALS/PETSc/KINSOL/IDA) como faixa de avaliacao separada (Gate ADV), sem bloquear o hardening do kernel atual.

## What Changes
- Introduzir uma arquitetura de convergencia por politicas no kernel v1 (Convergence Policy Engine) com classificacao de falhas e recovery orientado por contexto.
- Fortalecer o transient para cenarios de event burst, zero-crossing e chattering com guardas especificas.
- Evoluir a estrategia de solver linear para selecao condicionada por saude numerica e fallback deterministico com telemetria rica.
- Padronizar regularizacao numerica fisicamente limitada para dispositivos de comutacao e elementos magneticos nao lineares.
- Definir semantica multirate explicita para controle (PI/C-Block) com contratos de estabilidade entre dominio eletrico e de controle.
- Expor configuracao e diagnostico de convergencia nas Python bindings de forma estruturada.
- Criar matriz de benchmark/gates por dificuldade (diode-heavy, switch-heavy, zero-cross, magnetic-nonlinear, closed-loop control).
- Publicar playbook de convergencia com documentacao e exemplos referenciais reproduziveis.
- Avaliar e definir trilha de integracao para bibliotecas mais avancadas (SUNDIALS/PETSc/KINSOL/IDA) com criterios objetivos de adocao.

## Success Criteria (Program-Level)
- Robustez cross-cenario: reduzir falhas terminais de convergencia em classes-alvo da matriz de estresse, sem otimizar apenas um circuito.
- Nao regressao funcional: manter suites existentes verdes (kernel/testes de controle/PI/C-Block) ao longo de todas as fases.
- Nao regressao de performance: manter degradacao dentro de budget definido por fase e por classe (p50/p95).
- Auditabilidade: diagnostico estruturado deve ser suficiente para triagem sem parsing textual.
- Determinismo: repetir run com fingerprint equivalente deve manter classificacao e acao de politica dentro de tolerancias definidas.

## Impact
- Affected specs:
  - `kernel-v1-core`
  - `transient-timestep`
  - `linear-solver`
  - `device-models`
  - `python-bindings`
  - `benchmark-suite`
  - `convergence-playbook` (nova capacidade)
- Affected code:
  - `core/src/v1/simulation.cpp`
  - `core/src/v1/transient_services.cpp`
  - `core/include/pulsim/v1/simulation.hpp`
  - `core/include/pulsim/v1/high_performance.hpp`
  - `core/include/pulsim/v1/runtime_circuit.hpp`
  - `python/*` (bindings/result schemas)
  - `benchmarks/*` (catalogo e gates)
  - `docs/*` (novo playbook)

## Breaking / Compatibility
- Nao introduz quebra obrigatoria imediata de API publica.
- Novas opcoes de convergencia serao adicionadas com defaults backward-compatible.
- Gates novos entram em rollout por fases para evitar bloqueio abrupto de CI.

## Risks
- Escopo excessivo para uma unica iteracao.
  - Mitigacao: fases curtas com criterio de saida claro e rollback por feature flags.
- Regressao silenciosa em casos estaveis devido a heuristicas agressivas.
  - Mitigacao: matriz de nao-regressao obrigatoria por classe + budgets de performance.
- Complexidade de configuracao para usuario final.
  - Mitigacao: perfis canonicos (`strict`, `balanced`, `robust`) e knobs avancados opcionais.
