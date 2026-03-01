<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

## Diretrizes de Testes para Agentes de IA

Estas regras são obrigatórias ao criar ou modificar testes.

### NÃO fazer

- Não introduzir testes não determinísticos.
  - Não usar aleatoriedade sem seed fixa.
  - Não depender de ordem implícita de dicionário/lista sem garantir ordenação.
- Não criar testes que dependam de rede, internet, clock real, timezone local, hostname, ou recursos externos instáveis.
- Não usar caminhos absolutos de máquina local (ex.: `/Users/...`) dentro de testes.
- Não depender de artefatos de build “stale” (`build/`, `build/cp*/python`) sem validar que o pacote está completo.
- Não quebrar a matriz CI (Linux/macOS/Windows, Python 3.10+): evitar assumptions específicas de SO/ABI.
- Não “mascarar” falhas:
  - Não adicionar `xfail`/`skip` genérico para esconder regressão.
  - Não desabilitar sanitizers, warnings, ou gates de qualidade para “fazer passar”.
- Não introduzir testes com timeout agressivo/sensível a carga de runner.
- Não aumentar custo da suíte padrão com cenários pesados sem marcar/segmentar adequadamente.

### PODE fazer

- Criar testes determinísticos, com entradas explícitas e asserts objetivos.
- Usar diretórios/arquivos temporários (`tmp_path`, fixtures) e limpar estado global após cada teste.
- Isolar dependências opcionais com `skip` explícito e justificativa técnica clara.
- Separar testes por escopo:
  - unitários rápidos na suíte padrão;
  - stress/performance em suites dedicadas.
- Validar localmente antes de commit (mínimo):
  - `pytest python/tests -v --ignore=python/tests/validation`
  - `ctest --test-dir build --output-on-failure` (quando aplicável)
- Atualizar testes de configuração/workflow quando alterar CI/docs para manter contratos explícitos.

### Regras de estabilidade específicas deste repositório

- Ao mexer em import path de `pulsim`, garantir fallback correto entre pacote instalado e build local.
- Ao adicionar módulo Python no pacote (`python/pulsim`), refletir também no empacotamento/instalação (`python/CMakeLists.txt`) e cobertura de testes.
- Em testes de benchmark/performance, usar thresholds realistas e com margem para runners compartilhados.
