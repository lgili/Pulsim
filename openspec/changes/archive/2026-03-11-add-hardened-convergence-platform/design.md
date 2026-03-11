## Context
O backend v1 ja possui mecanismos relevantes (stiffness handling, gmin fallback, regularizacao de modelo, event clipping e stack de solver linear com fallback), mas o comportamento global ainda depende fortemente de regras estaticas por tentativa.

Em circuitos desafiadores (muitos diodos/chaves, cruzamento por zero, magneticos nao lineares, controle PI/C-Block), isso gera risco de:
- ajustes locais que resolvem um caso e degradam outro,
- colapso de timestep em janelas de eventos densos,
- retries custosos sem classificacao de causa,
- baixa previsibilidade para usuario final.

## Goals
- Tornar convergencia orientada por politica e classe de falha, nao apenas por contador de retries.
- Melhorar robustez inter-cenarios sem regressao sistematica em performance.
- Preservar determinismo e contratos de strict mode.
- Expor diagnostico estruturado para Python/GUI/benchmark sem parsing textual.
- Definir gates incrementais para rollout seguro.

## Non-Goals
- Substituir integralmente o kernel atual em um unico ciclo.
- Introduzir dependencia externa pesada sem gate de decisao/ADR.
- Alterar semantica fisica dos modelos sem limites e auditoria.

## Proposed Architecture
### 1) Convergence Policy Engine
Novo modulo de politica no runtime transiente com duas etapas:
1. `FailureClassifier`: classifica a tentativa atual com base em telemetria numerica/topologica.
2. `PolicySelector`: escolhe acao de recovery conforme classe, contexto e budget.

Classes iniciais de falha:
- `event_burst_zero_cross`
- `switch_chattering`
- `nonlinear_magnetic_stiffness`
- `control_discrete_stiffness`
- `linear_breakdown`
- `newton_globalization_failure`

Acoes de politica:
- ajuste de dt direcionado por classe,
- tuning de Newton/trust-region por classe,
- escalonamento gmin e regularizacao bounded por familia,
- mudanca de integrador/perfil em janelas especificas,
- abort deterministico com diagnostico tipado.

### 2) Event/Timestep Hardening
- `Event Burst Handler`: detecta densidade de eventos por janela e ativa guard profile.
- `Zero-Cross Guard`: tolerancias temporais/histerese para evitar ping-pong em torno de limiar.
- `Chattering Guard`: taxa maxima de toggles por janela + policy de merge/split segura.
- Acoplamento LTE/Newton/eventos com prioridade contextual (evitar feedback instavel de LTE perto de descontinuidade).

### 3) Linear/Nonlinear Robustness
- Selecao de solver linear orientada por sinais de saude (nao apenas size/nnz).
- Estados de saude e razoes de invalidacao padronizados.
- Regularizacao por familia com envelopes fisicos:
  - diodo/chave: condutancias limite,
  - magnetico: clamps e smooth transitions,
  - controle: limites de slew/clamp para evitar disrupcao numerica.

### 4) Control Robustness
- Contrato multirate explicito entre solver eletrico e controle discreto.
- Validacao de compatibilidade de `control_sample_time` global vs per-block.
- Detecao de loops algebraicos control-driven e estrategia de quebra deterministica.

### 5) Telemetry and API Surface
Esquema canonico de convergencia:
- `failure_class`, `recovery_stage`, `policy_action`, `dt_before/after`,
- `gmin_level`, `regularization_intensity`, `linear_solver_path`,
- `event_density`, `toggle_rate`, `control_sync_mode`.

Esse esquema deve ser exposto em:
- `SimulationResult` (core),
- bindings Python,
- artefatos de benchmark e GUI adapter.

## Profile Contract (Strict / Balanced / Robust)
### Strict
- Objetivo: determinismo maximo e diagnostico reprodutivel.
- Nao permite transicoes de fallback global quando `allow_fallback=false`.
- Permite apenas estabilizacao interna bounded explicitamente permitida pelo contrato.
- Sempre emite diagnostico tipado terminal ao esgotar budget.

### Balanced
- Objetivo: robustez geral com custo moderado.
- Permite politica adaptativa de recovery por classe de falha.
- Respeita budgets conservadores de retries/escalacao e evita heuristicas agressivas por default.

### Robust
- Objetivo: maximizar taxa de convergencia em casos extremos.
- Permite estrategias mais agressivas (desde que bounded/auditaveis).
- Deve manter contrato de telemetria completo para auditoria de impacto.

## Determinism and Reproducibility Contract
- Todos os campos de classificacao e acao da politica devem usar enums/ids estaveis (nao texto livre).
- Politicas devem ser puramente deterministicas dado: circuito + opcoes + fingerprint de ambiente.
- Fingerprint minimo para reproducao: versao backend, build flags relevantes, solver stack order, perfil de convergencia.
- Reproducibilidade deve ser avaliada por tolerancias declaradas de KPI por classe (nao igualdade bit-a-bit).

## Anti-Overfitting Rules
- Nenhuma heuristica pode ser promovida sem passar matriz multi-classe.
- Melhoria em uma classe nao pode degradar classes estaveis acima do budget de regressao da fase.
- Heuristicas localizadas devem ter guardas de contexto explicitos e telemetria de ativacao.
- Toda nova regra deve ter teste de nao-ativacao em cenarios fora do alvo.

## KPI Framework for Gates
- Robustez:
  - taxa de sucesso por classe,
  - taxa de falha terminal por classe,
  - repetibilidade (desvio em repeticoes equivalentes).
- Custo numerico:
  - `timestep_rejections`,
  - `newton_iterations_total`,
  - contagem de escalacoes por estagio.
- Performance:
  - tempo total p50/p95 por classe/perfil.
- Qualidade de diagnostico:
  - cobertura de campos tipados obrigatorios por run.

## Advanced Libraries Track
### Candidate Matrix
- `SUNDIALS IDA/CVODE/KINSOL`: DAE stiff e ecossistema maduro.
- `PETSc SNES/KSP`: Newton-Krylov com preconditioning avancado.

### Decision Criteria
- ganho de robustez em matriz estendida,
- impacto de performance p50/p95,
- complexidade de build/portabilidade,
- custo de manutencao/dep support,
- interoperabilidade com arquitetura atual.

## Phase Gates (Acceptance)
- Gate A: classificacao + policy engine ativos com telemetria tipada.
- Gate B: reducao estatisticamente significativa de falhas em zero-cross/event burst.
- Gate C: sem regressao de performance acima do threshold acordado em suites estaveis.
- Gate D: closed-loop PI/C-Block passa matriz completa com determinismo.
- Gate E: Python/tooling consomem schema novo sem parsing textual.
- Gate F: docs/playbook + exemplos executaveis e validados.
- Gate ADV: ADR de solver avancado aprovado com dados.

## Incremental Rollout Plan
1. M0 (Observacao): coletar classificacao/telemetria sem alterar decisoes do solver.
2. M1 (Policy Passive): habilitar seletor em dry-run e comparar acao recomendada vs acao aplicada.
3. M2 (Policy Active Balanced): ativar acoes em `balanced` sob feature flag.
4. M3 (Policy Active Robust): ativar faixa robusta em classes mais desafiadoras.
5. M4 (Default Review): decidir elevacao de defaults apenas com dados de matriz completa.

## Risks and Mitigations
- Risco: excesso de knobs e comportamento dificil de prever.
  - Mitigacao: perfis canonicos (`strict`, `balanced`, `robust`) e defaults conservadores.
- Risco: regressao de performance em casos simples.
  - Mitigacao: classificador habilita perfis pesados apenas quando necessario.
- Risco: regressao cruzada entre familias de circuito.
  - Mitigacao: matriz de stress por classe + gates por fase.

## Rollout Strategy
1. Introduzir telemetria e classificador em modo observacao (sem mudar decisao).
2. Ativar politica adaptativa em `balanced/robust` mantendo `strict` deterministico.
3. Promover gates progressivamente em CI.
4. Documentar tuning e migrar exemplos oficiais.
