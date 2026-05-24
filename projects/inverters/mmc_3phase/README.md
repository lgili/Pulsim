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

**Onde divergimos**:

* Ondulação de v_C é maior (~185 V pkpk vs ~50 V da tese).
* THD da corrente de fase é maior (~80% vs ~0.7% da tese).
* Pico da corrente de fase é ~30% maior (~28 A vs ~22 A da tese).

A causa principal das duas primeiras divergências é uma escolha
de modulação diferente: **a tese usa In-Phase Disposition (IPD)**,
enquanto o pulsim L1/L2 implementa **Phase-Shifted PWM (PS-PWM)**.
Com N = 5 (ímpar), PS-PWM tem mais conteúdo sub-harmônico que IPD,
resultando em maior ondulação de v_C que por sua vez modula a saída
e cria distorção na corrente.

A diferença na corrente de fase ainda fica em parte sem explicar —
o protótipo da tese tem perdas adicionais (semicondutores não-ideais)
que a Seção 4.1 modela aproximadamente ajustando o R_b parasita; nosso
half-bridge é mais idealizado.

## Próximos passos sugeridos

1. **Implementar IPD no pulsim** (`MmcArmMultilevelParams.modulation_scheme = "ipd"`)
   para fechar o gap de v_C ripple.
2. **Substituir o half-bridge ideal por IGBT level-1** (já existe no
   pulsim) para capturar V_F0 e r_on.
3. **Replicar o Capítulo 5** (controle de energia + frequência variável)
   para extender a validação ao regime closed-loop.

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
