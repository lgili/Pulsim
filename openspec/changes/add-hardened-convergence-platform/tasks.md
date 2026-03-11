## 1. Baseline and Measurement Contract
- [x] 1.1 Congelar baseline de robustez/performance por classe: `diode-heavy`, `switch-heavy`, `zero-cross`, `magnetic-nonlinear`, `closed-loop-control`.
- [x] 1.2 Definir KPI canonico por run: sucesso/falha terminal, `timestep_rejections`, `newton_iterations_total`, tempo p50/p95, campos de diagnostico tipado.
- [x] 1.3 Definir budget inicial de nao-regressao por fase (funcional e performance) e registrar em artefato versionado.
- [x] 1.4 Montar catalogo de circuitos de estresse reprodutiveis com parametros fixos e fingerprint de ambiente.

## 2. M0 - Observation-Only Instrumentation (Sem mudar decisoes)
- [x] 2.1 Implementar schema tipado de convergencia no runtime (`failure_class`, `recovery_stage`, `policy_action`, contexto numerico minimo).
- [x] 2.2 Instrumentar classificador em modo observacao (somente recomendacao, sem atuar no solver).
- [x] 2.3 Expor telemetria no `SimulationResult` e preservar compatibilidade com campos legados.
- [x] 2.4 Gate A: cobertura de schema obrigatorio >= 99% nos casos da matriz minima e 0 regressao funcional.

## 3. M1 - Policy Passive Validation
- [x] 3.1 Implementar `PolicySelector` em dry-run comparando acao recomendada vs acao aplicada.
- [x] 3.2 Adicionar verificador de anti-overfitting (ganho local nao pode violar budget em classes estaveis).
- [x] 3.3 Definir e validar contrato de perfis `strict`, `balanced`, `robust` (sem ativar mudancas agressivas ainda).
- [x] 3.4 Gate B: recomendacoes de politica melhoram KPI de robustez em classes-alvo sem degradar classes estaveis alem do budget.

## 4. M2 - Policy Active (Balanced)
- [x] 4.1 Ativar politicas contextuais para `event_burst_zero_cross` e `switch_chattering` com guardas bounded.
- [x] 4.2 Refinar arbitragem LTE/Newton/evento para reduzir loops de rejeicao sem progresso fisico.
- [ ] 4.3 Garantir contrato de strict mode (`allow_fallback=false`) com estabilizacao interna bounded e diagnostico terminal deterministico.
- [ ] 4.4 Gate C: queda estatisticamente significativa de falhas terminais nas classes-alvo e performance p95 dentro do budget.

## 5. M3 - Solver and Model Hardening
- [ ] 5.1 Evoluir politica de solver linear orientada por saude numerica (com transicoes auditaveis).
- [ ] 5.2 Padronizar regularizacao bounded por familia (`diode`, `switch`, `magnetic`) com trilha de auditoria por intensidade.
- [ ] 5.3 Incluir testes de nao-ativacao de heuristicas fora do contexto alvo.
- [ ] 5.4 Gate D: matriz estendida passa com 0 regressao funcional em suites de controle PI/C-Block.

## 6. M4 - Control Robustness and Multirate Contract
- [ ] 6.1 Definir semantica multirate explicita entre controle discreto e solver eletrico (global/per-block sample time).
- [ ] 6.2 Adicionar diagnostico tipado para eventos de rigidez de controle e risco de loop algebraico.
- [ ] 6.3 Validar suite fechada PI/C-Block com criterios de estabilidade e repetibilidade.
- [ ] 6.4 Gate E: closed-loop suite completa verde + determinismo confirmado em repeticoes equivalentes.

## 7. API / Tooling / Benchmarks
- [x] 7.1 Expor configuracao de politica de convergencia nas bindings Python com validacao de ranges.
- [x] 7.2 Expor telemetria tipada de convergencia/fallback sem parsing textual.
- [x] 7.3 Atualizar benchmark runner, artefatos e CI para gates por fase e budgets versionados.
- [x] 7.4 Gate F: tooling/GUI consomem schema novo sem quebrar compatibilidade.

## 8. Documentation and Reference Corpus
- [x] 8.1 Publicar `Convergence Playbook` com matriz de triagem por classe de falha.
- [x] 8.2 Publicar exemplos referenciais executaveis por classe com KPI esperado.
- [x] 8.3 Publicar guia de migracao/tuning entre perfis `strict`, `balanced`, `robust`.
- [x] 8.4 Validar docs e exemplos em pipeline automatizado.

## 9. Advanced Solver Evaluation Track (Isolado)
- [ ] 9.1 Definir benchmark de decisao para SUNDIALS/PETSc/KINSOL/IDA com criterios objetivos.
- [ ] 9.2 Implementar prototipo controlado para pelo menos 1 backend avancado sem impactar fluxo principal.
- [ ] 9.3 Registrar ADR de adocao/nao adocao com custo de manutencao e portabilidade.
- [ ] 9.4 Gate ADV: decisao formal aprovada com evidencias reproduziveis.
