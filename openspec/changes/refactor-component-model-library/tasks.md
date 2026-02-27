## 1. Refactor Estrutural
- [x] 1.1 Criar namespace/diretório `core/include/pulsim/v1/components/`.
- [x] 1.2 Extrair modelos existentes para arquivos dedicados (um por componente) preservando comportamento.
- [x] 1.3 Manter `device_base.hpp` como agregador compatível para evitar quebra de include público.
- [x] 1.4 Validar build + testes críticos de kernel/eventos/fallback após modularização.

## 2. Modelo e Convergência
- [ ] 2.1 Definir contrato de regularização numérica por componente (floors e limites).
- [ ] 2.2 Implementar aplicação condicional de pequenos parasíticos em estágio de recuperação.
- [ ] 2.3 Expor telemetria de regularização (contadores e intensidade aplicada).
- [ ] 2.4 Adicionar testes unitários para garantir limites e determinismo da política.

## 3. YAML / Configuração
- [ ] 3.1 Adicionar bloco YAML para configuração de regularização de modelos.
- [ ] 3.2 Implementar parsing, validação e defaults seguros.
- [ ] 3.3 Cobrir com testes de parser e bindings Python.

## 4. KPI / Benchmark Gate
- [ ] 4.1 Rodar matriz local-limit (fixed + variable) com artefatos versionados.
- [ ] 4.2 Verificar non-regression em benchmark suite e KPI gate.
- [ ] 4.3 Ajustar limites de regularização com base nos deltas de acurácia/runtime.
