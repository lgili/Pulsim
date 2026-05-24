# CMC 3-φ Trifásico — Modelagem e Validação

Pasta de projeto que **modela e valida** o **Conversor Matricial Convencional**
(CMC, 3×3) — ponto de partida para topologias CA-CA diretas no pulsim
(Phase 21).

A motivação é estabelecer infraestrutura validada para conversores
matriciais antes de atacar topologias mais complexas como o M3C
(Modular Multilevel Matrix Converter — Phase 22).

A referência teórica é o **Capítulo 2 da tese de Luiz Carlos Gili**
(*Contribuições Para o Conversor Modular Matricial Multinível — M3C*,
UFSC, 2024 — `artigos/Luiz Carlos Gili-1.pdf`), Seção 2.2 "Conversor
Matricial Convencional - CMC", que sintetiza a teoria clássica do CMC
a partir de Venturini (1980) [18], Huber & Borojevic (1995) [20] e
Wheeler et al. (2002) [19].

## Topologia

3 fases de entrada (A, B, C) × 3 fases de saída (a, b, c), conectadas
por 9 chaves bidirecionais $S_{ij}$:

```
        A      B      C
        │      │      │
    ┌───┴──┐ ──┴─── ──┴───┐
    │  S1  │  S4  │  S7  │ ─→ a
    │  S2  │  S5  │  S8  │ ─→ b
    │  S3  │  S6  │  S9  │ ─→ c
```

**Convenção da tese** (Tab. 1, 2, 3): cada **coluna** corresponde a
uma fase de entrada (A → S1-S3, B → S4-S6, C → S7-S9), cada **linha**
a uma fase de saída.

**Restrições operacionais**:
1. Em cada fase de saída (linha), exatamente **uma** chave conduz —
   evita curto entre fases de entrada (fonte de tensão).
2. Continuidade de corrente — todas as 3 saídas devem ter caminho de
   condução em qualquer instante — exige carga indutiva (ou filtro).

Resultado: $3^3 = 27$ estados de comutação válidos:
- **3 estados nulos** (Tab. 1): todas as 3 saídas tied to mesma entrada.
- **18 estados ativos** (Tab. 2): magnitude fixa ($\frac{2}{3} V_{LL}$),
  ângulo variável conforme a permutação.
- **6 estados rotacionais** (Tab. 3): saída = permutação completa da
  entrada.

## Chave bidirecional: configuração comum-emissor

Cada chave $S_{ij}$ é implementada como 2 IGBTs em anti-série
(emissores conectados) + 2 diodos antiparalelos:

```
node_in ─[Q1]─┬─[Q2]─ node_out
              │
            (CE)
              │
              ┴ (não conectado)
```

- $Q_1$ ON: corrente $node\_in \to node\_out$
- $Q_2$ ON: corrente $node\_out \to node\_in$
- Diodos garantem condução reversa quando o IGBT correspondente está OFF

**Total no CMC**: 9 chaves × (2 IGBTs + 2 diodos) = **18 IGBTs + 18 diodos**
+ 9 nós internos de emissor comum.

## Modulação SVM (Sec 2.2.1 da tese)

Para sintetizar um vetor de saída $\vec{V}_o$ (ângulo $\alpha_o$) com
fator de deslocamento de corrente de entrada $\varphi_i$:

1. **Setores** $K_v, K_i \in \{1, \ldots, 6\}$ — cada um cobre 60°.
2. **Razões cíclicas** (Eqs 7a-7d da tese) — 4 vetores ativos
   selecionados da Tab. 4:
   $$\delta^I = (-1)^{K_v+K_i+1} \cdot \frac{2}{\sqrt{3}} \cdot m \cdot
   \frac{\cos(\tilde{\alpha}_o - \pi/3) \cos(\tilde{\beta}_i - \pi/3)}{\cos(\varphi_i)}$$
   (e análogas para II, III, IV)
3. **Sequência simétrica** (Fig 5 — "Sequência I"):
   $$T_a/2,\ T_b/2,\ T_c/2,\ T_d/2,\ T_0,\ T_d/2,\ T_c/2,\ T_b/2,\ T_a/2$$
4. **Limite teórico**: $m \le \frac{\sqrt{3}}{2} \approx 0{,}866$ a FP unitário.

## Conteúdo

```
cmc_3phase_model.py            — CmcParams + vector tables + SVM helper +
                                  build_l0_plant + build_l1_plant
_build_notebooks.py            — gerador
01_cmc_topology_modeling.ipynb — topologia + 27 vetores + SVM analítica
02_cmc_svm_switched.ipynb      — chaveado vs analítico (validação L0↔L1)
03_cmc_inductive_load_validation.ipynb — métricas (THD, FP) vs forma fechada
```

## Plano de validação (em 4 tiers, espelhando o MMC)

| Tier | O que valida | Critério |
|------|--------------|----------|
| **1 — Analítico** | SVM: razões cíclicas, setores, limite m | Eqs 7a-7d da tese, m ≤ 0,866 |
| **2 — L0 ↔ L1** | Mesmo fundamental, ripple-only difference | AVG(i_a) erro < 1%, fundamental L0=L1 |
| **3 — Cross-sim** | Comparar com SPICE/PSIM | THD < 5%, picos < 2% diferença |
| **4 — Sweep** | Operação em variar f_out, m, FP_in | sem divergência em todo o range |

**Tier 5** (pytest regression) será adicionado em `python/tests/test_cmc_baseline.py`.

## Pontos de operação validados

| Caso | $V_{in}$ | $f_{in}$ | $f_{out}$ | $m$ | $\varphi_i$ | Carga |
|------|---:|---:|---:|---:|---:|---|
| Step-down 60→60 | 380 V | 60 Hz | 60 Hz | 0,6 | 0° (FP unitário) | RL 5Ω/10mH |
| Motor drive | 380 V | 60 Hz | 30 Hz | 0,5 | 0° | RL 5Ω/10mH |
| Limite teórico | 380 V | 60 Hz | 60 Hz | 0,866 | 0° | RL 5Ω/10mH |

## Como rodar

```bash
# Construir/atualizar os notebooks:
python projects/inverters/cmc_3phase/_build_notebooks.py

# Executar:
jupyter nbconvert --to notebook --execute --inplace \
    projects/inverters/cmc_3phase/01_cmc_topology_modeling.ipynb

# Regressão (pytest):
pytest python/tests/test_cmc_baseline.py -v
```
