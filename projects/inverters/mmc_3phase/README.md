# MMC Trifásico DC/AC — Validação contra Sousa (2022)

Pasta de projeto que valida a infraestrutura MMC do pulsim
(`pulsim.MmcArm{Average,Multilevel,Equivalent,Detailed}` — Phase 20)
contra o experimento da Seção 4.1 da tese de doutorado de
**Gean Jacques Maia de Sousa**
(*Sistemas de Controle para a Operação Eficiente de Conversores
Modulares Multiníveis em Acionamentos Elétricos*, UFSC, 2022 —
`artigos/Gean Jacques Maia de Sousa.pdf`).

## Por que essa validação

O Capítulo 4 da tese apresenta o protótipo experimental (15 kVA,
V_dc = 640 V, N = 5 SMs/braço, V_CSM = 128 V) e usa medidas de
laboratório como referência para validar **dois** modelos de
simulação propostos na própria tese (o detalhado e o SM-equivalente).

A Tabela 4.2 reporta as métricas chave:

| Métrica | Experimental | Sim 1 (com t_d) | Sim 2 (sem t_d) |
|---|---:|---:|---:|
| THD(i_a) [%] | 1.11 | 0.706 | 0.709 |
| RMS(i_ca) [A] | 4.60 | 4.55 | 8.67 |
| RMS(CA(i_cc)) [A] | 1.30 | 1.14 | 0.50 |

Replicamos exatamente o mesmo ponto de operação (V_dc, M, N, C_SM,
R/L da carga, T_d) com nossos modelos L1/L2 e comparamos os
resultados.

## Conteúdo

```
mmc_3phase_model.py        — Helpers: GeanThesisParams, build_l*_plant,
                              run_mmc_open_loop, THD/RMS/circulating
01_mmc_validation_gean.ipynb — Notebook único:
                                 1. Topologia + parâmetros do experimento
                                 2. Referências de modulação
                                 3. Simulação L1 (PS-PWM, sem t_d) ≡ Sim 2
                                 4. Simulação L2 (PS-PWM + t_d) ≡ Sim 1
                                 5. Tabela comparativa vs Tabela 4.2
                                 6. Discussão honesta das diferenças
_build_notebooks.py         — Gerador do notebook (edite aqui, regenere com
                                 `python _build_notebooks.py`).
```

## Resultados — match qualitativo, gap quantitativo

**O que bate com a tese**:

* Topologia (6 braços + 6 indutores + carga Y) e dinâmica básica ✓
* Tensão média do capacitor próxima de V_dc ✓
* Trinca de correntes balanceada a 60 Hz ✓
* Modelo L2 mostra os "notches" do tempo morto exatamente como
  previsto na Fig 3.12 da tese ✓
* Presença de 2ª harmônica nas correntes de circulação ✓

**Onde divergimos** (notebook 01, L1 sem tempo morto):

* Ondulação de v_C (soma do braço): 185 V pkpk vs ~50 V da Fig 4.2.
  O balanço de energia do braço dá 281 V sem corrente de circulação e
  178 V com a 2ω que a simulação desenvolve (10 A pico); nenhuma 2ω o
  leva abaixo de 172 V. Os ~50 V não são alcançáveis pela soma do braço
  com esses parâmetros (talvez a figura mostre um submódulo: ÷5 ≈ 36 V;
  não conferido, o PDF não está no repositório).
* THD da corrente de fase: 1,7 % vs 0,71 % (simulação da tese) e
  1,11 % (experimento). É a 5ª harmônica que a ondulação de v_C
  produz em m·v_C; o modelo médio L0, sem PWM, tem a mesma THD.
* Pico da corrente de fase: 26,6 A (fundamental 26,1 A) vs ~22 A. V̂/|Z_AC| = 26,8 A com os
  parâmetros da legenda.

Até 2026-11 o notebook reportava THD ≈ 80 %: `i_a` lia uma corrente
de braço, por um índice de estado tirado antes de os nós das fases b/c
existirem (`branch_var_id_for_inductor` = num_nodes + k).

## Próximos passos sugeridos

1. ✅ ~~Implementar IPD no pulsim~~ — feito (Phase 20.14).
2. ✅ IGBT level-1 — notebook 04 (não fecha o gap: ele não vem de
   perdas de condução).
3. Controle de energia e de circulação do Cap. 5 — notebooks 02/03
   têm a estrutura; com os ganhos usados, os blocos não mudam o regime
   permanente.

## Como rodar

```bash
# Construir/atualizar o notebook a partir da fonte Python:
python projects/inverters/mmc_3phase/_build_notebooks.py

# Executar o notebook:
jupyter nbconvert --to notebook --execute --inplace \
    projects/inverters/mmc_3phase/01_mmc_validation_gean.ipynb

# Ou abrir interativamente:
jupyter notebook projects/inverters/mmc_3phase/01_mmc_validation_gean.ipynb
```
