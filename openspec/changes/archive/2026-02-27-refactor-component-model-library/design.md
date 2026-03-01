## Context
O simulador está evoluindo para cenários mais rígidos de eletrônica de potência (buck/boost/inversores com comutação dura), onde:
- o modelo idealizado de alguns dispositivos gera sistemas com condicionamento ruim;
- o fluxo de correção fica caro (muitas rejeições/retries);
- a base de modelos em arquivo único torna manutenção e iteração lentas.

## Goals / Non-Goals
- Goals:
  - Modularizar modelos por componente (cohesão alta, baixo acoplamento).
  - Melhorar convergência de topologias com comutação sem exigir tuning por netlist.
  - Preservar compatibilidade pública (API e include legado) durante migração.
  - Manter gates de KPI para evitar regressão de acurácia/runtime.
- Non-Goals:
  - Reescrever o kernel MNA completo.
  - Introduzir novos dispositivos de potência nesta mudança.

## Decisions
- Decision: manter `device_base.hpp` como agregador estável e mover implementação para `core/include/pulsim/v1/components/`.
  - Rationale: reduz blast radius e permite migração incremental sem quebrar consumidores.

- Decision: tratar parasíticos como regularização numérica controlada por política.
  - Rationale: convergência melhora quando há descontinuidades ideais; porém precisa limites e telemetria para não mascarar erros físicos.

- Decision: rollout em fases com baseline KPI congelada.
  - Rationale: separar risco estrutural (refactor) de risco comportamental (modelo/parasitismo).

## Parasitic Strategy (Proposed)
- Device classes alvo inicial: `IdealDiode`, `VoltageControlledSwitch`, `MOSFET`, `IGBT`.
- Parâmetros de regularização pequenos e limitados:
  - `gmin_branch_floor` (condutância mínima por ramo não linear).
  - `cpar_switch_floor` (capacitância efetiva mínima para transições abruptas).
  - `rseries_floor` opcional em dispositivos ideais para reduzir rigidez algébrica extrema.
- Política automática:
  - Ativar apenas quando detectado padrão de falha/rejeição repetitiva.
  - Escalar de forma monotônica e limitada.
  - Registrar telemetria explícita (contador, último valor aplicado, energia/erro estimado).

## Risks / Trade-offs
- Risco de alterar resposta física em alta frequência: mitigar com limites baixos, defaults conservadores e opção de desligar.
- Risco de overhead: mitigar com aplicação condicional (somente em estágio de recuperação).
- Risco de regressão em casos fáceis: mitigar com regressão matrix e critérios de aceitação por classe de circuito.

## Migration Plan
1. Fase 1: modularização estrutural (sem mudança funcional de modelo).
2. Fase 2: introdução de API/política de regularização e telemetria.
3. Fase 3: calibrar limites com benchmark local-limit e KPI gate.
4. Fase 4: documentação e recomendação operacional.

## Open Questions
- Quais limites padrão de `cpar_switch_floor` e `rseries_floor` garantem robustez sem impacto perceptível de acurácia?
- Em quais famílias de circuito devemos habilitar auto-regularização por default?
