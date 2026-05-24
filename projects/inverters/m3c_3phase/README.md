# M3C 3-φ — Conversor Modular Matricial Multinível

Pasta de projeto que implementa o **Conversor Modular Matricial
Multinível** (M3C) — a topologia central da **tese de doutorado
do autor** (Gili, *Contribuições Para o Conversor Modular Matricial
Multinível - M3C*, UFSC 2024 — `artigos/Luiz Carlos Gili-1.pdf`).

O M3C é uma evolução do CMC (Phase 21) onde **cada chave bidirecional
da matriz 3×3 é substituída por um módulo de submódulos full-bridge
em cascata**, capacitando o conversor a operar em alta tensão / alta
potência com semicondutores de baixa tensão e produzir formas de
onda multinível.

## Topologia

```
                    Fonte trifásica
                    A    B    C
                    │    │    │
                  L_in L_in L_in (filtro entrada)
                    │    │    │
            ┌───────┼────┼────┼───────┐
            │  ┌────M_Aa───────────M_Ab──── M_Ac─┐ ─── a
            │  ├────M_Ba───────────M_Bb──── M_Bc─┤ ─── b
            │  └────M_Ca───────────M_Cb──── M_Cc─┘ ─── c
            │              ↑                       │
            │   9 módulos M_xy onde x ∈ {A,B,C}    │
            │   é a fase de entrada e y ∈ {a,b,c}  │
            │   é a fase de saída                  │
            └──────────────────────────────────────┘
                    Motor/Carga trifásica
```

Cada **módulo M_xy** é uma cascata de N **submódulos (SMs)** em
série, onde cada SM é uma ponte completa (4 IGBTs + 4 diodos
antiparalelos + 1 capacitor grampeado):

```
   ┌──[Q1]──┬──[Q2]──┐
   X        C_SM     Y
   └──[Q3]──┴──[Q4]──┘
```

Estados do SM (Fig 37-40 da tese, Sec 4.2):
* **Estado 1**: Q1+Q4 ON → V_XY = +V_cap
* **Estado 2**: Q2+Q3 ON → V_XY = -V_cap
* **Estado 3**: Q1+Q2 ou Q3+Q4 ON → V_XY = 0 (curto-circuito interno)
* **Estado 4**: todos OFF → SM aberto (V_XY < V_cap)

### Configuração desta implementação

Seguindo a tese (Cap 4-7):

| Parâmetro | Valor |
|---|---:|
| Módulos (matriz 3×3) | 9 |
| Submódulos por módulo (N) | 6 |
| **Total SMs full-bridge** | **54** |
| Capacitância por SM | 680 µF |
| Tensão alvo do capacitor | 4 kV |
| Níveis de tensão linha-linha | 13 |
| Potência nominal | 2 MVA |
| Sistema 1 (entrada) | 13,8 kV / 50 Hz |
| Sistema 2 (saída) | 11 kV / 5-30-45-55 Hz |

## Contribuições da tese — implementadas neste projeto

A tese identifica 4 contribuições principais (Sec 1.1 + 5.6):

1. **Fast SVM no plano lgγ** (Sec 3.2) — modulação por vetores
   espaciais usando uma transformação não-ortogonal que gera
   vetores **inteiros** no plano `(l, g, γ)`. Não precisa de
   trigonometria, escala para qualquer número de níveis com as
   **mesmas equações** (Eqs 29-30). Ideal para FPGA.

2. **Cálculo da tensão dos módulos** (Sec 4.3) — algoritmo que,
   dadas as razões cíclicas e a configuração de 5 módulos ativos,
   calcula a tensão que **cada um dos 9 módulos** deve gerar.

3. **Função custo de balanceamento** (Sec 5.5.3, Eq 163) —
   $C = \sum_{xy} (\epsilon_{xy} + \Delta V_{xy})^2$ — avalia o
   impacto de cada uma das 45 conexões viáveis sobre a tensão dos
   caps e seleciona a melhor.

4. **Balanço interno dos SMs por sorting** (Sec 5.5.3, Alg 2-3) —
   dentro de cada módulo, escolhe quais SMs comutam para equalizar
   as tensões dos N capacitores internos.

## Conteúdo (em construção)

```
m3c_3phase_model.py            — M3cParams + Fast SVM + plant builders
_build_notebooks.py            — gerador dos notebooks
01_m3c_fast_svm.ipynb          — Fast SVM no plano lgγ (Sec 3)
02_m3c_module_voltages.ipynb   — cálculo das tensões dos módulos (Sec 4.3)
03_m3c_cost_function.ipynb     — função custo + balanço (Sec 5.5)
04_m3c_l0_averaged.ipynb       — L0 plant simplificado
05_m3c_l1_switched.ipynb       — L1 com 54 SMs
06_m3c_closed_loop.ipynb       — Cap 5 controle dq
07_m3c_validation_thesis.ipynb — comparação contra HIL OPAL-RT da tese
```

## Plano de validação (multi-tier)

| Tier | O que valida | Quando |
|------|--------------|--------|
| **1 — Fast SVM analítica** | Eqs 26-30, vetores inteiros, soma δ ≤ 1 | 22.1 |
| **2 — Conexões e cost function** | 81 → 45 conexões válidas, função custo | 22.2 |
| **3 — L0 plant** | Internal consistency Venturini-style | 22.3 |
| **4 — L1 switched (54 SMs)** | L0 ↔ L1 fundamental match | 22.4 |
| **5 — Closed-loop** | Step de potência, freq variável | 22.5 |
| **6 — Cross-validation tese** | Tabela 16, Figs 87-122 | 22.7 |

## Como rodar

```bash
# Construir/atualizar os notebooks:
python projects/inverters/m3c_3phase/_build_notebooks.py

# Executar (após cada stage):
jupyter nbconvert --to notebook --execute --inplace \
    projects/inverters/m3c_3phase/01_m3c_fast_svm.ipynb

# Regressão (pytest):
pytest python/tests/test_m3c_fast_svm.py -v
```
