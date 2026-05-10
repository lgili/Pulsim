# Pulsim Roadmap — Caminho para Bater PSIM e PLECS

> **Documento estratégico.** Mapa de execução das 15 propostas OpenSpec recém-criadas, organizado em 5 fases com dependências, marcos e KPIs explícitos. Cada change tem `proposal.md`, `tasks.md`, deltas de spec e (quando crítico) `design.md` em `openspec/changes/`.

## Resumo executivo

A análise técnica do código identificou que o motor de chaveamento de Pulsim, embora arquiteturalmente próximo de PLECS no papel, está **inativo na prática** para conversores reais — qualquer netlist com diodo/MOSFET/IGBT cai no caminho DAE+Newton, perdendo o speedup state-space que PLECS entrega. O roadmap atual prioriza:

1. **Fase 0 (3-6 meses)** — virar a chave do motor PWL de verdade. **Sem isso, não há competição com PSIM/PLECS.**
2. **Fase 1 (3-6 meses)** — fechar gap de fidelidade (catálogo de devices, magnéticos, análise frequencial).
3. **Fase 2 (6-12 meses)** — biblioteca de domínio (templates de conversor, motores, three-phase/grid).
4. **Fase 3 (6-12 meses)** — diferenciação técnica (HIL/code-gen, FMI, Monte Carlo, property-based testing).
5. **Fase 4 (contínuo)** — higiene técnica (build modular, política de robustez unificada).

Marcos demoláveis em cada fase, com métricas concretas. Sem dependências circulares — ordem é seguir a numeração.

---

## Fase 0 — Killer Feature (3-6 meses)

> **Mensagem:** este é o trio que decide se Pulsim tem chance ou não. Se os 3 itens entregarem os KPIs prometidos, o produto está pronto pra demo viral. Se não, nada do resto importa.

### 0.1 — `refactor-pwl-switching-engine`
**O que:** ativa o motor PWL state-space de verdade para diodo/MOSFET/IGBT em modo `Ideal`. Topology bitmask como cache key. Cache de fatoração KLU por topologia. Step sem Newton em janelas de topologia estável.

**Dependências:** nenhuma. **Pode começar imediatamente.**

**KPIs (gates):**
- ≥10× speedup vs main em `buck_switching.yaml`, `boost_switching_complex.yaml`, `interleaved_buck_3ph.yaml`
- 0 iterações Newton em janelas de topologia estável
- ≤0.5% de erro vs LTspice/NgSpice
- ≥95% cache hit rate de topologia em steady-state

**Esforço estimado:** 8-12 sprints (2-3 meses, 1 dev sênior).

**Demo de saída:** "buck 100 kHz, 10 ms simulado em 50 ms wallclock, batendo PLECS em 1.2× no mesmo cenário."

---

### 0.2 — `add-automatic-differentiation`
**O que:** Eigen `AutoDiffScalar` para Jacobianas de devices não-lineares em modo Behavioral. Validation layer FD vs AD. Custom devices em Python via `pulsim.register_device()`.

**Dependências:** 0.1 (PWL bypassa AD; saber a fronteira é importante). **Pode rodar em paralelo nas Phases 1-3 do 0.1.**

**KPIs:**
- AD Jacobian ≡ manual stamp dentro de 1e-12 em 5 op-points por device
- Convergência Newton ≤ baseline ± 5%
- Build time ≤+15%
- Custom device exemplo (JFET) em 30 linhas Python

**Esforço:** 4-6 sprints (1.5-2 meses, 1 dev pleno).

**Bug-killer collateral:** valida todos os manual stamps existentes — provável que algum sign error apareça.

---

### 0.3 — `refactor-linear-solver-cache`
**O que:** corrigir o cache de fatoração que está broken. Sparsity-pattern hash (sem valores). Symbolic factor reusado entre Newton iterations. Numeric factor por topologia. Workspace pré-alocado no construtor.

**Dependências:** 0.1 estrutura compartilhada (topology bitmask). **Implementar logo após 0.1 chegar à Phase 4 (cache key).**

**KPIs:**
- ≥95% cache hit rate em steady-state
- 0 alocações por step no hot loop (heap profiler verifies)
- ≥3× speedup adicional vs baseline em `mosfet_buck.yaml` modo Behavioral

**Esforço:** 3-4 sprints (1 mês, 1 dev pleno).

**Combinado com 0.1**: target total = ≥30× speedup em conversores chaveados.

---

### Marco da Fase 0: "Demo PSIM-killer"

Ao final da Fase 0, deve existir:
- 3 benchmarks (buck, boost, interleaved-3φ) rodando em modo `Ideal` 10× mais rápido que main, com parity vs LTspice <0.5%.
- 1 vídeo curto e 1 blog post mostrando o speedup numa conversora realista.
- Documentação de migração `docs/pwl-switching-migration.md`.

**Decision gate:** se os KPIs não baterem, **pausar Fase 1+ e investigar a causa antes de seguir.** Sem o motor PWL, o resto é cosmético.

---

## Fase 1 — Fechar Gap de Fidelidade (3-6 meses, em paralelo)

> Engenheiros de potência avaliam um simulador pelo realismo dos modelos. Sem isso, sabe-se ler dados, mas não tomar decisões de design.

### 1.1 — `add-catalog-device-models`
**O que:** MOSFET/IGBT/diodo de catálogo com Coss/Ciss/Crss tabelados, body diode com Qrr, Rds_on(Tj), tail current. Importadores de SPICE, PLECS XML e PDF datasheet. 6 devices de referência (Si/SiC/GaN MOSFET, Si IGBT, SiC Schottky, fast-recovery diode).

**Dependências:** 0.2 (AD para Jacobianas com Coss(Vds) suaves). **Iniciar quando 0.2 chegar à Phase 4.**

**KPIs:** switching loss ≤10% vs LTspice vendor model em 6 referências; conduction loss ≤5% sobre 25-125 °C.

**Esforço:** 8-10 sprints (2-2.5 meses).

---

### 1.2 — `add-magnetic-core-models`
**O que:** `SaturableInductor` e `SaturableTransformer` com B-H table/arctan/Langevin, Steinmetz / iGSE para core loss, Jiles-Atherton hysteresis opt-in, eddy-current lumped. 4 cores de referência (Magnetics, TDK, Ferroxcube, EPCOS).

**Dependências:** 0.2 (AD para `i(λ)` saturável). **Pode rodar em paralelo com 1.1.**

**KPIs:** inrush ≤20% vs Faraday analítico; Steinmetz loss ≤10% vs vendor calculator.

**Esforço:** 6-8 sprints (1.5-2 meses).

---

### 1.3 — `add-frequency-domain-analysis`
**O que:** AC small-signal sweep com linearização em torno do OP. FRA via injeção senoidal. Bode/Nyquist plot helpers Python. Multi-input/multi-output transfer function matrix.

**Dependências:** 0.1 (state-space matrices reusadas), 0.3 (factorization cache reuso entre frequências).

**KPIs:** AC vs FRA paridade ≤1 dB / 5° em buck open-loop; AC vs analítico ≤0.1 dB / 1° em RLC.

**Esforço:** 6-8 sprints (1.5-2 meses).

---

### Marco da Fase 1: "Engenheiro de controle adota Pulsim"

- 6 devices de catálogo prontos com side-by-side waveform vs LTspice
- 4 magnetic cores de catálogo
- AC sweep + FRA produzindo Bode em 1 chamada Python
- Tutorial: "design e validação de um buck converter end-to-end" usando catalog MOSFET + AC sweep para tunar PI

---

## Fase 2 — Library de Domínio (6-12 meses)

> Deixar o usuário sair de "0 a buck convertendo" em 5 linhas de YAML. Deixar entrar em motor drives e grid.

### 2.1 — `add-converter-templates`
**O que:** 10 templates parametrizados (buck, boost, buck-boost, flyback, forward, 2sw-forward, half-bridge, full-bridge, LLC, DAB, totem-pole PFC, 2φ interleaved). Auto-design de defaults. Compensadores embutidos (PI, type-II/III). Reference parity vs TI/Infineon AN.

**Dependências:** 0.1, 1.1, 1.2, 1.3 idealmente.

**KPIs:** 10 templates implementados; default config estável; ≥5 com parity vs published reference.

**Esforço:** 10-12 sprints (2.5-3 meses).

---

### 2.2 — `add-motor-models`
**O que:** PMSM, induction, BLDC, DC motor em frame dq. Mecânica (shaft, gearbox, loads). Park/Clarke. PMSM-FOC template auto-tunado. Encoder, hall, resolver.

**Dependências:** 0.2 (AD para nonlinear motor models), 1.3 (AC sweep for FOC tuning), 2.1 (template DSL).

**KPIs:** PMSM/IM/BLDC/DC parity ≤5% vs analytical; FOC speed-loop bandwidth ≤20% off design.

**Esforço:** 10-12 sprints (2.5-3 meses).

---

### 2.3 — `add-three-phase-grid-library`
**O que:** Three-phase sources (programmable, harmonic). PLLs (SrfPll, DsogiPll, MafPll). Symmetrical components. Grid-following e grid-forming inverter templates. Anti-islanding (informativo).

**Dependências:** 2.1 (template DSL), 2.2 (Park/Clarke).

**KPIs:** SrfPll lock ≤50 ms; DsogiPll resiste a sag 50%; grid-following P/Q tracking ≤5%.

**Esforço:** 8-10 sprints (2-2.5 meses).

---

### Marco da Fase 2: "Pulsim cobre 80% dos casos PSIM"

- 10 templates de conversor + 4 motores + grid library com tutoriais.
- Solar inverter end-to-end (PV + DC link + 3φ inverter + grid follow) compliant IEEE 1547.
- PMSM-FOC drive completo com 3φ inverter, demonstrando trapezoidal speed command.

---

## Fase 3 — Diferenciação Técnica (6-12 meses)

> Aqui Pulsim deixa de "alcançar" PSIM/PLECS e começa a oferecer coisas que **não estão lá** no open-source — ou só em produtos caros.

### 3.1 — `add-realtime-code-generation`
**O que:** geração de C99 fixed-step a partir do circuito PWL (state-space discretizado por topologia via `expm`). Targets: c99, ARM Cortex-M7, Zynq baremetal. PIL test bench. Stability check `|λ_max·Ts| ≤ 0.5`.

**Dependências:** 0.1 (PWL state-space), 0.3 (cache de fatoração).

**KPIs:** buck PIL parity ≤0.1%; Cortex-M7 step latency ≤500 ns @ 240 MHz; ROM ≤8 KB / RAM ≤512 B.

**Esforço:** 10-12 sprints (2.5-3 meses).

**Diferencial:** começa a competir com PLECS RT Box em performance/preço. Open-source HIL é muito raro.

---

### 3.2 — `add-fmi-export`
**O que:** export FMI 2.0/3.0 Co-Simulation. Import FMI 2.0 CS para usar FMUs PLECS/Modelica como blocos. Validação `fmuCheck` em CI.

**Dependências:** 0.1, 0.2, 0.3 — base sólida; 1.x desejável (FMU mais útil com mais devices).

**KPIs:** buck FMU passa `fmuCheck` e roda em OMSimulator com parity ≤1%; PLECS FMU importável.

**Esforço:** 8-10 sprints (2-2.5 meses).

**Diferencial:** ANSYS Twin Builder, MATLAB Simulink, OpenModelica, Dymola — todos consomem FMI. Pulsim entra em todos os fluxos system-level.

---

### 3.3 — `add-monte-carlo-parameter-sweep`
**O que:** `pulsim.sweep()` com Cartesian/MC/LHS/Sobol. Executores serial, joblib, dask, GPU (PWL only). Métricas (steady-state, peak, THD, settling time, efficiency). Pandas/xarray export. Sensitivity (Sobol indices) e optimization (Optuna) wrappers.

**Dependências:** 0.1 (GPU executor depende de PWL state-space). Restante pode rodar antes.

**KPIs:** 1000-sample LHS produz percentile bands úteis; 10000-sample GPU sweep ≥10× speedup vs CPU joblib.

**Esforço:** 6-8 sprints (1.5-2 meses).

**Diferencial:** PSIM tem parametric sweep mas é manual. Pulsim oferece API limpo Python + GPU.

---

### 3.4 — `add-property-based-testing`
**O que:** Hypothesis-based property tests para KCL, KVL, Tellegen, energy, passivity, periodicity, reciprocity. RapidCheck no C++ para invariants em MNA. Regression corpus auto-shrinking.

**Dependências:** nenhuma forte — pode rodar em paralelo desde o começo. Mais útil **depois** de Fase 0 (catches PWL bugs cedo).

**KPIs:** 7 invariants implementados; ≥3 latent bugs descobertos; regressões corpus ativo em CI.

**Esforço:** 4-6 sprints (1-1.5 meses).

**Diferencial:** rara entre simuladores — você pode comunicar isso como **rigor científico** que PSIM/PLECS não têm visibilidade.

---

### Marco da Fase 3: "Pulsim no estado da arte open-source"

- HIL real-time code generation funcionando em Cortex-M7
- FMU export validado em OpenModelica + Simulink
- Sweep API com 1k samples Latin Hypercube + sensitivity analysis
- Property-based test suite encontrando regressões antes de PRs mergearem

---

## Fase 4 — Higiene Técnica (contínuo, 1-3 meses inicial)

> Não bloqueante para release, mas reduz custo de cada PR daqui pra frente. Idealmente concorre com as fases acima como background work.

### 4.1 — `refactor-modular-build-split`
**O que:** dividir `python/bindings.cpp` em ~9 arquivos por domínio; mover impl de `runtime_circuit.hpp` (3110 linhas) para `.cpp`; auditar `high_performance.hpp` (2544 linhas) e `integration.hpp` (1862 linhas).

**KPIs:** clean build ≤75% baseline; incremental rebuild ≤10% do clean.

**Esforço:** 4-6 sprints (1-1.5 meses), zero risco de quebrar API.

---

### 4.2 — `refactor-unify-robustness-policy`
**O que:** consolidar `apply_robust_*_defaults` (3 lugares duplicados) em um único `RobustnessProfile` no kernel. Deprecar wrapper retry layer Python depois que PWL é default.

**Dependências:** 0.1 chegar à Phase 7 (PWL default).

**KPIs:** uma única definição site; comportamento preservado.

**Esforço:** 3-4 sprints (1 mês).

---

## Diagrama de dependências

```
Phase 0 (Killer)
    0.1 PWL Engine ────────┬──────────┐
                            │          │
    0.2 Auto-Diff ──────────┤          │
                            │          │
    0.3 Solver Cache ───────┤          │
                            │          │
                            ▼          ▼
Phase 1 (Fidelity)         Phase 3.1 / 3.2
    1.1 Catalog Devices ◄── 0.2
    1.2 Magnetic Cores ◄── 0.2
    1.3 AC/FRA ◄── 0.1, 0.3
                  │
                  ▼
Phase 2 (Library)
    2.1 Templates ◄── 0.1, 1.1, 1.2, 1.3
    2.2 Motors ◄── 0.2, 1.3, 2.1
    2.3 Three-Phase ◄── 2.1, 2.2
                  │
                  ▼
Phase 3 (Differentiation)
    3.1 Code-Gen ◄── 0.1, 0.3
    3.2 FMI ◄── 0.1, 0.2, 0.3
    3.3 Sweep ◄── 0.1 (GPU only)
    3.4 Property Tests ◄── (no hard deps)

Phase 4 (Hygiene, parallel)
    4.1 Modular Build (no deps)
    4.2 Robustness Policy ◄── 0.1
```

---

## Cronograma agregado (otimista, 2 devs sêniores + 1 pleno)

| Mês | Marcos paralelos |
|-----|------------------|
| **M1-M3** | Fase 0.1 PWL Engine PoC → impl → benchmark |
| **M2-M4** | Fase 0.2 Auto-Diff em paralelo com 0.1 |
| **M3-M4** | Fase 0.3 Solver Cache (depende de 0.1 estrutura) |
| **M4-M6** | Fase 1.1 Catalog devices + 1.2 Magnetic cores + 1.3 AC/FRA em paralelo |
| **M5-M9** | Fase 2.1 Templates → 2.2 Motors → 2.3 Three-Phase (sequencial) |
| **M7-M9** | Fase 3.4 Property tests + 4.1 Modular build (in parallel, baixo risco) |
| **M8-M11** | Fase 3.1 Code-gen + 3.2 FMI (paralelo) |
| **M10-M12** | Fase 3.3 Sweep + 4.2 Robustness policy |

Pessimista: dobrar tudo. Realista: 18 meses até Fase 4 completa.

---

## KPIs comparativos vs benchmarks comerciais (target final)

| Métrica | PSIM | PLECS | Pulsim hoje | Pulsim alvo |
|---------|------|-------|-------------|-------------|
| Speedup conversor chaveado | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ (Fase 0) |
| Fidelidade device | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ (Fase 1.1) |
| Magnetic models | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ (Fase 1.2) |
| AC small-signal | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ⭐⭐⭐⭐ (Fase 1.3) |
| Converter templates | ⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | ⭐⭐⭐⭐ (Fase 2.1) |
| Motor library | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | ⭐⭐⭐⭐ (Fase 2.2) |
| 3φ / grid library | ⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | ⭐⭐⭐⭐ (Fase 2.3) |
| HIL/code-gen | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ⭐⭐⭐⭐ (Fase 3.1) |
| FMI export | ⭐⭐ | ⭐⭐⭐ | ❌ | ⭐⭐⭐⭐ (Fase 3.2) |
| Monte Carlo / Sweep | ⭐⭐⭐ | ⭐⭐⭐ | ❌ | ⭐⭐⭐⭐ (Fase 3.3) |
| Property-based tests | ❌ | ❌ | ⭐ | ⭐⭐⭐⭐ (Fase 3.4) |
| Open-source | ❌ | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Python-first | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Lista das 15 propostas OpenSpec criadas

Todas validadas com `openspec validate <id> --strict`.

| # | Change ID | Capability(s) | Has design.md |
|---|-----------|---------------|----------------|
| 0.1 | `refactor-pwl-switching-engine` | kernel-v1-core, device-models, transient-timestep | ✅ |
| 0.2 | `add-automatic-differentiation` | device-models, python-bindings | ✅ |
| 0.3 | `refactor-linear-solver-cache` | linear-solver | — |
| 1.1 | `add-catalog-device-models` | device-models, netlist-yaml | — |
| 1.2 | `add-magnetic-core-models` | magnetic-models (new), netlist-yaml | — |
| 1.3 | `add-frequency-domain-analysis` | ac-analysis (new), python-bindings, netlist-yaml | — |
| 2.1 | `add-converter-templates` | converter-templates (new), python-bindings | — |
| 2.2 | `add-motor-models` | motor-models (new), netlist-yaml | — |
| 2.3 | `add-three-phase-grid-library` | three-phase-grid (new), netlist-yaml | — |
| 3.1 | `add-realtime-code-generation` | code-generation (new), python-bindings | — |
| 3.2 | `add-fmi-export` | fmi-export (new), python-bindings | — |
| 3.3 | `add-monte-carlo-parameter-sweep` | parameter-sweep (new), python-bindings | — |
| 3.4 | `add-property-based-testing` | benchmark-suite | — |
| 4.1 | `refactor-modular-build-split` | python-bindings, kernel-v1-core | — |
| 4.2 | `refactor-unify-robustness-policy` | kernel-v1-core, python-bindings | — |

**Como usar:**
```bash
# Listar
openspec list

# Detalhes de uma change
openspec show refactor-pwl-switching-engine --type change

# Aprovar (por enquanto a aprovação é manual via PR review do team)
# Depois implementar seguindo tasks.md
# Após deploy, archivar:
openspec archive refactor-pwl-switching-engine --yes
```

---

## Princípios não-negociáveis durante a execução

1. **Não pular o approval gate.** Toda change abre como proposal, vai pra PR, é revisada antes de qualquer commit de implementação.
2. **KPI gates em CI.** Cada Phase tem gates `G.1..G.N` numerados em `tasks.md`. Failure = PR não merge.
3. **Backward compat preservado** até a próxima major version. Deprecation warnings antes de remover.
4. **Cada change é mergeable independentemente.** Se 0.1 atrasa, 0.2 não bloqueia (apenas perde sinergia até 0.1 chegar).
5. **Documentação como first-class citizen.** Toda change inclui tutorial/notebook na sua tasks.md. Sem docs, não fecha.
6. **Bench antes de otimizar.** Não aceitar speedup baseado em "intuition" — sempre número before/after numa máquina de referência.
7. **Cuidado com `runtime_circuit.hpp` e `bindings.cpp`** — são pontos de fricção. Faça `refactor-modular-build-split` cedo (paralelo a Fase 0) se quiser velocidade de iteração.

---

## Quando reabrir / dividir / arquivar

- **Se 0.1 estourar 50% do esforço estimado**, parar e reavaliar (talvez state-space matrices via Modelica reduction são melhores que MNA-direct).
- **Se 1.1 datasheet importer for mais complexo do que parece**, dividir em `add-catalog-device-models-core` (devices) e `add-datasheet-importer` (tooling).
- **Se 3.1 code-gen virar 6+ meses**, considerar começar só com `c99` target e adiar Cortex-M7/Zynq para change separado.

---

## Próximos passos imediatos

1. **Revisão técnica deste roadmap** — está alinhado com sua visão de produto?
2. **Aprovação da Fase 0** — abrir as 3 proposals como issues e atribuir donos.
3. **PoC de `refactor-pwl-switching-engine` Phase 1** (PWL device contract) — 1 sprint, low-risk, valida a arquitetura antes do refactor pesado.
4. **Setup de KPI dashboard** — antes de medir speedup vs main, precisamos de baseline numbers per platform.

Quer que eu detalhe uma das proposals (descer para pseudo-código), ou prefere começar pela PoC da 0.1 Phase 1?
