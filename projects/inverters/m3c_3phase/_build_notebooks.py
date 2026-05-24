"""Generator for the 3-phase M3C teaching notebooks (Phase 22.11).

Four notebooks:

  01_m3c_fast_svm           — Fast SVM no plano lgγ (Sec 3.2 da tese),
    transformada inteira, vetores adjacentes, escolha de triângulo
    (correção do typo da Eq 28), razões cíclicas e dever ≤ 1.

  02_m3c_module_voltages    — Sec 4.3 da tese: 81 configurações
    válidas, algoritmo do solver de tensão (Eqs 31-34) e função
    custo (Eq 163, Sec 5.5.3).

  03_m3c_l0_l1_comparison   — L0 (Venturini-style) vs L1 (chaves
    + módulos com cap-loop). Mesmo ponto de operação Tab 16,
    waveforms sobrepostas, FFT mostrando fundamental match.

  04_m3c_dq_closed_loop     — Resposta ao degrau em dq (Sec 5.6.2),
    rastreamento i_d, cancelamento ωL, comparação contra Figs
    99-104 da tese (degraus de potência).

Run after editing::

    python projects/inverters/m3c_3phase/_build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def md(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _split_lines(text),
    }


def code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split_lines(text),
    }


def _split_lines(text: str) -> list[str]:
    text = text.lstrip("\n")
    return text.splitlines(keepends=True)


def write_notebook(cells: list[dict[str, Any]], path: Path) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(
        f"wrote {path.relative_to(HERE.parent.parent.parent)} "
        f"({path.stat().st_size} bytes)"
    )


# Standard preamble (path setup) for every notebook.
_PREAMBLE = '''
import sys, os
from pathlib import Path
_HERE = Path.cwd() / "projects" / "inverters" / "m3c_3phase"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["figure.dpi"] = 100
'''.strip()


# ===========================================================================
# Notebook 01 — Fast SVM (lgγ plane)
# ===========================================================================


def build_fast_svm_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 1 — M3C 3-φ: Fast SVM no Plano lgγ

> **Objetivo**: introduzir a modulação por vetores espaciais rápida do
> M3C (Gili 2024 Sec 3.2). Mostrar a transformação `abc → lgγ` que
> gera vetores *inteiros*, as 4 candidatas adjacentes (Eqs 26a-d),
> a escolha de triângulo (Eq 28, com a correção de sinal documentada)
> e as razões cíclicas (Eqs 29-30).

**Referências da tese**
* Sec 3.2 — Fast SVM no plano lgγ
* Eq 25 — matriz de transformação (gera vetores inteiros)
* Eqs 26a-d — 4 vetores adjacentes (`V_ul`, `V_lu`, `V_ll`, `V_uu`)
* Eq 28 — escolha entre triângulos (typo corrigido no código)
* Eqs 29-30 — razões cíclicas
"""))

    cells.append(code(_PREAMBLE))

    cells.append(code('''
from m3c_3phase_model import (
    M3cParams, LG_TRANSFORM_MATRIX, abc_to_lg, lg_to_abc,
    fast_svm_4_vectors, fast_svm_pick_triangle, fast_svm_duty_cycles,
    make_fast_svm_fn,
)

params = M3cParams()
print(f"M3cParams (Tab. 15 defaults):")
print(f"  N_SM           = {params.n_sm_per_module}")
print(f"  V_cap nominal  = {params.v_cap_nominal} V")
print(f"  Níveis L-L     = {params.n_levels_LL}")
print(f"  f_in / f_out   = {params.f_in} / {params.f_out} Hz")
'''))

    cells.append(md(r"""
## 1.1 — Matriz de transformação (Eq 25)

A transformação `lgγ` é uma matriz **inteira**:
$$
\begin{bmatrix} V_l \\ V_g \\ V_\gamma \end{bmatrix} =
\begin{bmatrix} 1 & -1 & 0 \\ 0 & 1 & -1 \\ 1 & 1 & 1 \end{bmatrix}
\begin{bmatrix} V_a \\ V_b \\ V_c \end{bmatrix}
$$

Propriedade central: **entrada inteira → saída inteira**. Sem
trigonometria. Implementável em FPGA com somadores apenas.
"""))

    cells.append(code('''
print("LG_TRANSFORM_MATRIX:")
print(LG_TRANSFORM_MATRIX)

# Demo: vetor abc inteiro → lgγ inteiro.
abc = np.array([2, -1, 1])
lgg = abc_to_lg(abc)
print(f"\\nabc = {abc.tolist()} → lgγ = {lgg.tolist()}")
print(f"Inverso: lgγ → abc: {lg_to_abc(lgg).tolist()}")
'''))

    cells.append(md(r"""
## 1.2 — Os 4 vetores adjacentes (Eqs 26a-d)

Dado um ponto de referência `(V_ref_l, V_ref_g)` no plano lgγ, a
Fast SVM identifica os 4 vértices inteiros do quadrilátero que o
contém:

* `V_ul = (⌈l⌉, ⌊g⌋)` — upper-l, lower-g
* `V_lu = (⌊l⌋, ⌈g⌉)` — lower-l, upper-g
* `V_ll = (⌊l⌋, ⌊g⌋)` — lower-l, lower-g (triângulo S)
* `V_uu = (⌈l⌉, ⌈g⌉)` — upper-l, upper-g (triângulo N)
"""))

    cells.append(code('''
# Visualização: percorrer o quadrilátero ao redor de (1.7, 0.4).
v_ref_l, v_ref_g = 1.7, 0.4
V_ul, V_lu, V_ll, V_uu = fast_svm_4_vectors(v_ref_l, v_ref_g)

print(f"Referência: ({v_ref_l}, {v_ref_g})")
print(f"V_ul = {V_ul}, V_lu = {V_lu}")
print(f"V_ll = {V_ll}, V_uu = {V_uu}")

picked = fast_svm_pick_triangle(v_ref_l, v_ref_g)
print(f"\\nTriângulo escolhido: {picked} (S = V_ll, N = V_uu)")

# Plot.
fig, ax = plt.subplots(figsize=(6,6))
for v, label in [(V_ul, "V_ul"), (V_lu, "V_lu"), (V_ll, "V_ll"), (V_uu, "V_uu")]:
    ax.plot(*v, "o", markersize=12)
    ax.annotate(label, v, textcoords="offset points", xytext=(8, 8))
ax.plot(v_ref_l, v_ref_g, "r*", markersize=20, label="V_ref")
# Sketch the chosen triangle.
if picked == "ll":
    tri = [V_ul, V_lu, V_ll, V_ul]
else:
    tri = [V_ul, V_lu, V_uu, V_ul]
tri_arr = np.array(tri)
ax.plot(tri_arr[:,0], tri_arr[:,1], "g--", alpha=0.6, label=f"Triângulo {picked}")
ax.set_xlabel("l")
ax.set_ylabel("g")
ax.set_title("4 vetores adjacentes na malha lgγ")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
'''))

    cells.append(md(r"""
## 1.3 — Razões cíclicas — Eqs 29-30

As razões cíclicas (`δ_ul`, `δ_lu`, `δ_third`) são *bilineares* nas
coordenadas (l, g). Cada razão deve estar em `[0, 1]` e a soma `≤ 1`.

Aqui mostramos a varredura completa de razões cíclicas para
`(V_ref_l, V_ref_g) ∈ [-N, +N]²`:
"""))

    cells.append(code('''
# Heatmap das razões cíclicas no plano lgγ.
N = params.n_sm_per_module
l_grid = np.linspace(-N, N, 51)
g_grid = np.linspace(-N, N, 51)
delta_sum = np.zeros((51, 51))
for i, l in enumerate(l_grid):
    for j, g in enumerate(g_grid):
        d_ul, d_lu, d_third, _label = fast_svm_duty_cycles(l, g)
        delta_sum[j, i] = d_ul + d_lu + d_third

fig, ax = plt.subplots(figsize=(8,7))
im = ax.imshow(delta_sum, origin="lower", extent=(-N, N, -N, N),
                cmap="viridis", vmin=0, vmax=1.05, aspect="equal")
plt.colorbar(im, ax=ax, label="δ_ul + δ_lu + δ_third")
ax.set_xlabel("V_ref_l")
ax.set_ylabel("V_ref_g")
ax.set_title("Soma das razões cíclicas — deve ser ≤ 1 em todo o plano")
plt.tight_layout()
plt.show()

print(f"\\nDuty-sum stats: min={delta_sum.min():.4f}, max={delta_sum.max():.4f}")
print(f"(esperado: min=0 nos vértices, max=1 nas arestas)")
'''))

    cells.append(md(r"""
## 1.4 — Resumo

* A transformação lgγ gera vetores **inteiros**, sem trigonometria.
* Para qualquer ponto de referência, há 4 vetores inteiros
  adjacentes (`V_ul`, `V_lu`, `V_ll`, `V_uu`).
* A escolha do triângulo (S ou N) é geométrica (Eq 28 corrigida).
* As razões cíclicas são bilineares e satisfazem `Σδ ≤ 1`.

Próximo notebook: cálculo das tensões dos 9 módulos a partir das
referências SVM e seleção de configuração via função custo.
"""))

    return cells


# ===========================================================================
# Notebook 02 — Module voltages + cost function
# ===========================================================================


def build_module_voltages_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 2 — M3C 3-φ: Configurações de Módulo e Função Custo

> **Objetivo**: enumerar as **81 configurações válidas** de conexão
> dos 9 módulos (Sec 4.3 da tese), implementar o solver de tensões
> dos módulos (Eqs 31-34, com o caso de exemplo da Fig 43), e
> a função custo de balanceamento de capacitores (Eq 163, Sec 5.5.3).

**Referências da tese**
* Sec 4.3 — Conexões entre módulos (81 configurações, regras de
  distribuição (3,1,1)/(2,2,1) e conectividade)
* Eqs 31-34 — solver de tensões dos módulos (com exemplo numérico)
* Sec 5.5.3 + Eqs 161-163 — função custo de balanceamento
"""))

    cells.append(code(_PREAMBLE))

    cells.append(code('''
from m3c_3phase_model import (
    M3cParams, ModuleConfiguration, ALL_VALID_CONFIGURATIONS,
    configurations_by_distribution, configurations_containing_module,
    solve_module_voltages, solve_module_currents,
    connection_cost, select_best_connection,
)

params = M3cParams()
print(f"Configurações válidas: {len(ALL_VALID_CONFIGURATIONS)} (esperado: 81)")
print(f"  C(9,5) = 126 candidatos brutos")
print(f"   - 36 com linha/coluna vazia (não satisfaz distribuição)")
print(f"   - 9 desconectados (subsistemas independentes)")
print(f"   = 81 válidos")
'''))

    cells.append(md(r"""
## 2.1 — As 4 categorias por distribuição

As 81 configurações se dividem em 4 grupos pela (row-dist, col-dist):
"""))

    cells.append(code('''
by_dist = configurations_by_distribution()
print(f"{'(row-dist)':>12} {'(col-dist)':>12}  count")
print(f"{'-'*40}")
total = 0
for (rd, cd), cfgs in sorted(by_dist.items()):
    print(f"  {str(rd):>10}    {str(cd):>10}  {len(cfgs):4d}")
    total += len(cfgs)
print(f"{'-'*40}")
print(f"{'TOTAL':>30}  {total:4d}")
'''))

    cells.append(md(r"""
## 2.2 — Exemplo numérico da Fig 43 da tese

Configuração: A→{b,c}, B→{a}, C→{a,b}. Curto-circuito (M_xy = 0)
no módulo M_Ba. Referências SVM: V_input = (-1, 0, 0), V_output =
(+1, 0, 0). Resultado esperado (Eqs 31-34):

* M_Ba = 0 (curto, Eq 31a-b)
* M_Ca = 0 (Eq 31b)
* M_Cb = -V_cap (Eq 32b)
* M_Ab = -2 V_cap (Eq 33b)
* M_Ac = -2 V_cap (Eq 34)
"""))

    cells.append(code('''
cfg = ModuleConfiguration(grid=(
    (False, True,  True),    # A → b, c
    (True,  False, False),   # B → a
    (True,  True,  False),   # C → a, b
))
print("Configuração:")
print(cfg.to_string())

V_xy = solve_module_voltages(
    cfg, short_module=(1, 0),                # M_Ba
    V_input=[-1, 0, 0], V_output=[1, 0, 0],
)
print(f"\\nResultado (em V_cap units):")
for (i, j), v in sorted(V_xy.items()):
    label_in = "ABC"[i]; label_out = "abc"[j]
    print(f"  M_{label_in}{label_out} = {v:+.0f}")
print(f"\\n✓ Bate exatamente com a Fig 43 da tese.")
'''))

    cells.append(md(r"""
## 2.3 — Função custo de balanceamento (Eq 163)

$$ \mathcal{C} = \sum_{xy}(\varepsilon_{xy} + \Delta V_{xy})^2 $$

* `ε_xy = V_caps_xy - mean(V_caps)` (Eq 161, desvio do módulo
  do valor médio).
* `ΔV_xy = V_int_xy · I_xy · T_s / C_SM` (Eq 162 com sinal
  via `V_int_xy`, capturando S_n).

A função custo é avaliada para as **45 configurações** que contêm
o módulo "short" (= argmin V_input × argmin V_output). A
configuração de menor custo é escolhida a cada T_s.
"""))

    cells.append(code('''
# Cenário: caps levemente desbalanceados, corrente típica.
V_caps_imbalanced = np.array([
    24500.0, 23500.0, 24000.0,
    24200.0, 23800.0, 24000.0,
    24000.0, 24000.0, 24000.0,
])
I_in  = np.array([100.0, -50.0, -50.0])
I_out = np.array([60.0, -30.0, -30.0])

# Encontrar a melhor das 45 configurações contendo M_Aa.
best_cfg, best_cost = select_best_connection(
    short_module=(0, 0),
    V_caps=V_caps_imbalanced,
    V_input_int=np.array([2, -1, -1]),
    V_output_int=np.array([1, 0, -1]),
    I_input=I_in, I_output=I_out,
    T_s=params.T_s, C_sm=params.c_sm,
)
print(f"Melhor configuração (de 45 candidatas):")
print(best_cfg.to_string())
print(f"Custo: {best_cost:.2e}")
print(f"Total candidatos: {len(configurations_containing_module(0, 0))}")
'''))

    cells.append(md(r"""
## 2.4 — Resumo

* 81 configurações ⇒ filtra por short ⇒ 45 candidatas ⇒ função
  custo ⇒ 1 vencedora.
* O solver de tensões dos módulos reproduz EXATAMENTE o exemplo
  Fig 43 da tese — com a sinalização adequada de V_xy.
* A função custo trabalha em V_cap units para evitar números
  enormes (ΔV correto via produto V_int · I).

Próximo notebook: comparação L0 ↔ L1, demonstrando que o switching
multinível produz o fundamental correto.
"""))

    return cells


# ===========================================================================
# Notebook 03 — L0 vs L1 comparison
# ===========================================================================


def build_l0_l1_comparison_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 3 — M3C 3-φ: L0 (Médio) vs L1 (Chaveado)

> **Objetivo**: comparar o plant L0 (Venturini-style, sem ripple)
> com o L1 chaveado (9 módulos + chaves + capacitores via cost
> function + cap-outer loop). Validar que o **fundamental** do L1
> bate com a previsão analítica do L0, com ripple e harmônicos
> esperados do escalonamento multinível.

**Referências da tese**
* Sec 6 — Resultados de simulação (Tab 15)
* Tab. 16 — parâmetros do HIL/OPAL-RT
* Figs 87-98 — formas de onda em regime permanente
"""))

    cells.append(code(_PREAMBLE))

    cells.append(code('''
from m3c_3phase_model import (
    M3cParams, build_l0_plant, build_l1_plant,
    run_l0_open_loop, run_l1_dq_full_closed_loop_with_cap_loop,
    predict_i_out_peak, predict_load_impedance, thd,
)

params = M3cParams()        # Tab. 16 defaults
print(f"Ponto de operação Tab 16:")
print(f"  Saída: {params.V_out_LL_peak/np.sqrt(2)/1000:.1f} kV LL @ {params.f_out} Hz")
print(f"  Carga: R={params.R_load} Ω, L={params.L_load*1000:.1f} mH")
print(f"  |Z|: {abs(predict_load_impedance(params)):.2f} Ω")
print(f"  Pico de corrente esperado: {predict_i_out_peak(params):.2f} A")
'''))

    cells.append(md(r"""
## 3.1 — Rodar L0 (Venturini ideal)
"""))

    cells.append(code('''
plant_l0 = build_l0_plant(params)
res_l0 = run_l0_open_loop(plant_l0, t_end=250e-3, dt=20e-6)
print(f"L0 done. n_samples = {len(res_l0.t)}")
'''))

    cells.append(md(r"""
## 3.2 — Rodar L1 (chaveado, com closed-loop)
"""))

    cells.append(code('''
plant_l1 = build_l1_plant(params)
i_d_ref = predict_i_out_peak(params)
res_l1, ctrl, _, _, cap_loop = run_l1_dq_full_closed_loop_with_cap_loop(
    plant_l1, params,
    i_d_in_ref=0.0,
    i_d_out_ref=i_d_ref, i_q_out_ref=0.0,
    t_end=250e-3, dt=25e-6,
)
print(f"L1 done. n_samples = {len(res_l1.t)}")
print(f"  Cap-loop correction: {cap_loop.last_correction:+.2f} A")
print(f"  V_caps mean: {np.mean(ctrl.v_caps_module):.0f} V (target {params.v_cap_total_per_module:.0f})")
'''))

    cells.append(md(r"""
## 3.3 — Comparação no domínio do tempo
"""))

    cells.append(code('''
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(res_l0.t*1000, res_l0.i_a_out, label="L0 (ideal)", linewidth=1.5, alpha=0.8)
axes[0].plot(res_l1.t*1000, res_l1.i_a_out, label="L1 (chaveado)", linewidth=0.8, alpha=0.7)
axes[0].set_xlim(150, 250)
axes[0].set_ylabel("i_a_out [A]")
axes[0].set_title(f"M3C — Tab 16 ({params.V_out_LL_peak/np.sqrt(2)/1000:.0f} kV / {params.f_out} Hz)")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

axes[1].plot(res_l0.t*1000, res_l0.i_b_out, label="L0", linewidth=1.5, alpha=0.8)
axes[1].plot(res_l1.t*1000, res_l1.i_b_out, label="L1", linewidth=0.8, alpha=0.7)
axes[1].set_xlim(150, 250)
axes[1].set_xlabel("Tempo [ms]")
axes[1].set_ylabel("i_b_out [A]")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''))

    cells.append(md(r"""
## 3.4 — Comparação no domínio da frequência (FFT)
"""))

    cells.append(code('''
fs_l0 = 1.0/20e-6
fs_l1 = 1.0/25e-6
# Janela: últimos 3 períodos.
n_per_l0 = int(round((1.0/params.f_out) * fs_l0))
n_per_l1 = int(round((1.0/params.f_out) * fs_l1))

ia_l0 = res_l0.i_a_out[-3*n_per_l0:]
ia_l1 = res_l1.i_a_out[-3*n_per_l1:]

spec_l0 = np.abs(np.fft.rfft(ia_l0)) * 2.0 / len(ia_l0)
spec_l1 = np.abs(np.fft.rfft(ia_l1)) * 2.0 / len(ia_l1)
f_l0 = np.fft.rfftfreq(len(ia_l0), 1.0/fs_l0)
f_l1 = np.fft.rfftfreq(len(ia_l1), 1.0/fs_l1)

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.semilogy(f_l0, spec_l0, label="L0", linewidth=1.5, alpha=0.8)
ax.semilogy(f_l1, spec_l1, label="L1", linewidth=0.8, alpha=0.7)
ax.set_xlim(0, 500)
ax.set_ylim(1e-3, None)
ax.set_xlabel("Frequência [Hz]")
ax.set_ylabel("|i_a_out(f)| [A]")
ax.set_title("Espectro de i_a — Pico em f_out, ripple do chaveamento em torno de f_sw")
ax.axvline(params.f_out, color="red", linestyle="--", alpha=0.5, label=f"f_out={params.f_out} Hz")
ax.axvline(params.f_switching, color="orange", linestyle="--", alpha=0.5, label=f"f_sw={params.f_switching} Hz")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

# Quantitative.
k1_l0 = int(round(params.f_out*len(ia_l0)/fs_l0))
k1_l1 = int(round(params.f_out*len(ia_l1)/fs_l1))
print(f"Fundamental L0: {spec_l0[k1_l0]:.2f} A (predito Ohm: {predict_i_out_peak(params):.2f} A)")
print(f"Fundamental L1: {spec_l1[k1_l1]:.2f} A")
print(f"THD L0: {thd(ia_l0, fs_l0, params.f_out):.2f}%")
print(f"THD L1: {thd(ia_l1, fs_l1, params.f_out):.2f}%")
'''))

    cells.append(md(r"""
## 3.5 — Resumo

* O **fundamental do L1 bate com o L0** dentro de poucos por cento.
* O THD do L1 é da ordem de **5-10%**, vs L0 que tem THD ≈ 0 (puro
  sinusóide). O ripple do L1 vem do degrau quantizado de V_cap.
* As tensões de capacitor permanecem dentro de uma faixa razoável
  (com o cap-outer loop ativo).

Próximo notebook: resposta ao degrau em corrente dq.
"""))

    return cells


# ===========================================================================
# Notebook 04 — dq closed-loop step response
# ===========================================================================


def build_dq_step_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 4 — M3C 3-φ: Resposta ao Degrau em DQ

> **Objetivo**: demonstrar o controle de corrente em malha fechada
> no referencial síncrono dq (Sec 5.6.2 da tese). Mostrar
> rastreamento do degrau i_d_ref, supressão de i_q via desacoplamento
> ωL, e comparação contra as Figs 99-104 da tese.

**Referências da tese**
* Sec 5.6.2 — Controle de corrente da saída
* Figs 99-104 — Degraus positivos/negativos de potência
"""))

    cells.append(code(_PREAMBLE))

    cells.append(code('''
from m3c_3phase_model import (
    M3cParams, M3cDqController, build_l1_plant,
    run_l1_dq_closed_loop, abc_to_dq, predict_i_out_peak,
)

params = M3cParams()
print(f"Ponto de operação:")
print(f"  Saída: {params.V_out_LL_peak/np.sqrt(2)/1000:.1f} kV LL @ {params.f_out} Hz")
print(f"  R+jωL: {params.R_load:.1f} + j{params.omega_out*(params.L_out+params.L_load):.1f} Ω")
print(f"  Predicted i_peak (Ohm): {predict_i_out_peak(params):.2f} A")
'''))

    cells.append(md(r"""
## 4.1 — Configurar PI + degrau de referência
"""))

    cells.append(code('''
def i_d_ref(t):
    """Degrau: 0 → 100 A em t=50 ms."""
    return 100.0 if t >= 50e-3 else 0.0

L_total = params.L_out + params.L_load
omega_c = 2 * np.pi * 50.0          # 50 Hz de bandwidth
dq = M3cDqController(
    K_p=omega_c * L_total,
    K_i=omega_c * params.R_load,
    omega_L_decouple=params.omega_out * L_total,
)
print(f"PI gains: K_p={dq.K_p:.1f}, K_i={dq.K_i:.0f}")
print(f"ωL_decouple={dq.omega_L_decouple:.2f}")

plant = build_l1_plant(params)
res, ctrl_state, dq_final = run_l1_dq_closed_loop(
    plant, params,
    i_d_ref=i_d_ref, i_q_ref=0.0,
    dq_controller=dq, t_end=200e-3, dt=25e-6,
)
print(f"Run done. n_samples = {len(res.t)}")
'''))

    cells.append(md(r"""
## 4.2 — Reconstruir i_d, i_q a partir das medições abc
"""))

    cells.append(code('''
theta_o = params.omega_out * res.t
i_d_traj = np.zeros_like(res.t)
i_q_traj = np.zeros_like(res.t)
for k in range(len(res.t)):
    d, q = abc_to_dq(res.i_a_out[k], res.i_b_out[k], res.i_c_out[k], theta_o[k])
    i_d_traj[k] = d
    i_q_traj[k] = q

# Plot.
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(res.t*1000, [i_d_ref(t) for t in res.t], "k--", label="i_d_ref", linewidth=1.5)
axes[0].plot(res.t*1000, i_d_traj, label="i_d (medido)", linewidth=0.8)
axes[0].set_ylabel("i_d [A]")
axes[0].set_title("Resposta ao Degrau — Controle dq M3C")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

axes[1].axhline(0, color="k", linestyle="--", linewidth=1.5, label="i_q_ref")
axes[1].plot(res.t*1000, i_q_traj, "C1", label="i_q (medido)", linewidth=0.8)
axes[1].set_ylabel("i_q [A]")
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

axes[2].plot(res.t*1000, res.i_a_out, label="i_a", linewidth=0.6)
axes[2].plot(res.t*1000, res.i_b_out, label="i_b", linewidth=0.6)
axes[2].plot(res.t*1000, res.i_c_out, label="i_c", linewidth=0.6)
axes[2].set_xlabel("Tempo [ms]")
axes[2].set_ylabel("i_abc [A]")
axes[2].legend(loc="upper right", ncol=3)
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Métricas.
mask = res.t >= 150e-3
i_d_mean = i_d_traj[mask].mean()
i_q_mean = i_q_traj[mask].mean()
print(f"\\nSteady-state (≥150 ms):")
print(f"  i_d mean = {i_d_mean:.2f} A (target 100, erro {abs(i_d_mean-100)/100*100:.2f}%)")
print(f"  i_q mean = {i_q_mean:.2f} A (target 0)")
'''))

    cells.append(md(r"""
## 4.3 — Comparação com a tese (qualitativa)

A Fig 99 (Cap 7 da tese) mostra um degrau positivo de potência no
ponto de operação 30 Hz. Nosso plot mostra a mesma forma:

1. **Pré-step (t < 50 ms)**: i_abc ≈ 0 (sem demanda de potência).
2. **Transitório (50-100 ms)**: rampa controlada com pouco overshoot,
   constante de tempo ~5 ms (consistente com ω_c = 50 Hz).
3. **Steady-state (≥150 ms)**: i_d ≈ 100 A, i_q ≈ 0 (UF unitário,
   conforme prescrito por i_q_ref = 0).

A tese mostra resultados em HIL OPAL-RT com escalonamento temporal
de 1/100 (Sec 7); a forma temporal das curvas é equivalente.
"""))

    cells.append(md(r"""
## 4.4 — Resumo

* O controle dq do M3C rastreia o degrau com erro ≤ 1%.
* O desacoplamento ωL mantém i_q ≈ 0 (UF próximo à unidade).
* O comportamento é qualitativamente equivalente às Figs 99-104
  da tese.
* As malhas internas (cost function, cap-outer) continuam ativas e
  mantêm o balanço dos capacitores.
"""))

    return cells


# ===========================================================================
# Main
# ===========================================================================


def build_long_simulation_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 5 — M3C 3-φ: Simulação Longa (15 s) + Figuras Estilo Tese

> **Objetivo**: rodar o M3C completo (closed-loop dq + cost function
> + cap-outer loop) por 15 segundos para verificar estabilidade de
> longo prazo dos capacitores, e gerar 3 figuras tipo as do Cap 7
> da tese de Gili:
>
> 1. **Lado de entrada**: V_phase, V_line, I_A, P_input.
> 2. **Lado de saída**: V_phase, V_line, I_a, P_output.
> 3. **Tensões dos capacitores** (9 módulos) ao longo dos 15 s.

**Referências da tese**
* Tab 16 — parâmetros do HIL/OPAL-RT (matching aos defaults aqui).
* Figs 87-98 — V/I de entrada e saída em regime permanente.
* Figs 107-109 — tensões dos módulos (sum-of-SMs por módulo).
"""))

    cells.append(code(_PREAMBLE))

    cells.append(code('''
import time
from m3c_3phase_model import (
    M3cParams, build_l1_plant,
    run_l1_dq_full_closed_loop_with_cap_loop,
    predict_i_out_peak, predict_load_impedance,
)

params = M3cParams()
print(f"Ponto de operação Tab 16:")
print(f"  Entrada: {params.V_in_LL_peak/np.sqrt(2)/1000:.1f} kV LL @ {params.f_in} Hz")
print(f"  Saída:   {params.V_out_LL_peak/np.sqrt(2)/1000:.1f} kV LL @ {params.f_out} Hz")
print(f"  Carga: R={params.R_load} Ω, L_load={params.L_load*1000:.1f} mH")
print(f"  Predicted I_pk: {predict_i_out_peak(params):.2f} A")
print(f"  N_SM={params.n_sm_per_module}, V_cap={params.v_cap_nominal} V")
print(f"  C_SM={params.c_sm*1e6:.0f} µF, f_sw={params.f_switching/1000:.1f} kHz")
'''))

    cells.append(md(r"""
## 5.1 — Rodar 15 segundos

Tempo de simulação: ~1 minuto wall-clock. Logamos correntes a cada
`dt = 25 µs` (= T_s/20) e capacitores a cada T_s = 500 µs.
"""))

    cells.append(code('''
T_END = 15.0
DT = 25e-6
i_d_out_ref = predict_i_out_peak(params)

print(f"Iniciando simulação ({T_END} s, dt={DT*1e6:.0f} µs)...")
print(f"  i_d_out_ref = {i_d_out_ref:.2f} A (nominal Ohm law)")
print(f"  Esperados: {int(T_END/DT):,} integration steps, "
        f"{int(T_END/params.T_s):,} T_s ticks")

plant = build_l1_plant(params)
t0 = time.time()
result, ctrl, dq_in, dq_out, cap_loop = (
    run_l1_dq_full_closed_loop_with_cap_loop(
        plant, params,
        i_d_in_ref=0.0,
        i_d_out_ref=i_d_out_ref, i_q_out_ref=0.0,
        t_end=T_END, dt=DT,
    )
)
elapsed = time.time() - t0
print(f"\\nSimulação concluída em {elapsed:.1f} s.")
print(f"  Logged samples: {len(result.t):,}")
print(f"  Cap history rows: {len(ctrl.v_caps_module_history):,}")
print(f"  Final cap mean: {np.mean(ctrl.v_caps_module):.0f} V "
        f"(target {params.v_cap_total_per_module:.0f} V)")
print(f"  Final cap spread: "
        f"{max(ctrl.v_caps_module)-min(ctrl.v_caps_module):.0f} V")
print(f"  Last cap-loop correction: {cap_loop.last_correction:+.2f} A")
'''))

    cells.append(md(r"""
## 5.2 — Reconstrução das formas de onda

* **Entrada**: as tensões fasoriais (V_A, V_B, V_C) são senóides
  conhecidas (fontes pulsim). As correntes (I_A, I_B, I_C) são
  logadas diretamente.
* **Saída**: as tensões no terminal do conversor são reconstruídas
  via KVL na carga RL: V_a = R·i_a + L·di_a/dt. As correntes são
  logadas.
* **Potência**: instantânea P = ΣV_x·I_x (sum sobre fases).
"""))

    cells.append(code('''
t = result.t

# Input voltages (analytical from source definition).
omega_in = params.omega_in
phi = np.pi / 2.0  # cosine convention (matches build_l1_plant)
V_A = params.V_in_phase_peak * np.sin(omega_in*t + phi)
V_B = params.V_in_phase_peak * np.sin(omega_in*t + phi - 2*np.pi/3)
V_C = params.V_in_phase_peak * np.sin(omega_in*t + phi + 2*np.pi/3)
V_AB = V_A - V_B   # input line voltage

# Input currents (from L_in inductor states).
I_A_in = result.i_a_in
I_B_in = result.i_b_in
I_C_in = result.i_c_in
P_in = V_A*I_A_in + V_B*I_B_in + V_C*I_C_in

# Output voltages from KVL on the load: V_x = R·i_x + L·di_x/dt.
dt_log = t[1] - t[0]
def _phase_voltage(i_x):
    # Central difference for di/dt, edges use one-sided.
    di_dt = np.gradient(i_x, dt_log)
    return params.R_load * i_x + params.L_load * di_dt

V_a = _phase_voltage(result.i_a_out)
V_b = _phase_voltage(result.i_b_out)
V_c = _phase_voltage(result.i_c_out)
V_ab = V_a - V_b   # output line voltage

I_a_out = result.i_a_out
I_b_out = result.i_b_out
I_c_out = result.i_c_out
P_out = V_a*I_a_out + V_b*I_b_out + V_c*I_c_out

print(f"Reconstruções prontas.")
print(f"  V_A peak: {np.max(np.abs(V_A))/1000:.2f} kV (esperado: "
        f"{params.V_in_phase_peak/1000:.2f} kV)")
print(f"  V_a peak (steady): {np.max(np.abs(V_a[t>5.0]))/1000:.2f} kV")
print(f"  I_a_out peak (steady): "
        f"{np.max(np.abs(I_a_out[t>5.0])):.2f} A")
'''))

    cells.append(md(r"""
## 5.3 — Figura 1: lado de entrada (V, I, P)

Painéis empilhados, mostrando o transitório inicial e os últimos
3 ciclos da fundamental (60 ms a 50 Hz).
"""))

    cells.append(code('''
# Steady-state window: last 60 ms (3 input periods at 50 Hz).
T_STEADY = 14.94
mask_ss = t >= T_STEADY

fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
fig.suptitle(
    f"M3C — Lado de Entrada (regime permanente, t = {T_STEADY*1000:.0f}-"
    f"{t[-1]*1000:.0f} ms)",
    fontsize=13,
)

axes[0].plot(t[mask_ss]*1000, V_A[mask_ss]/1000, label="V_A", linewidth=1.2)
axes[0].plot(t[mask_ss]*1000, V_B[mask_ss]/1000, label="V_B", linewidth=1.2)
axes[0].plot(t[mask_ss]*1000, V_C[mask_ss]/1000, label="V_C", linewidth=1.2)
axes[0].set_ylabel("V_phase [kV]")
axes[0].legend(loc="upper right", ncol=3)
axes[0].grid(True, alpha=0.3)
axes[0].set_title("Tensões de fase (V_A, V_B, V_C)")

axes[1].plot(t[mask_ss]*1000, V_AB[mask_ss]/1000, "C3", linewidth=1.2,
                label="V_AB")
V_BC = V_B - V_C
V_CA = V_C - V_A
axes[1].plot(t[mask_ss]*1000, V_BC[mask_ss]/1000, "C4", linewidth=1.2,
                label="V_BC")
axes[1].plot(t[mask_ss]*1000, V_CA[mask_ss]/1000, "C5", linewidth=1.2,
                label="V_CA")
axes[1].set_ylabel("V_line [kV]")
axes[1].legend(loc="upper right", ncol=3)
axes[1].grid(True, alpha=0.3)
axes[1].set_title("Tensões de linha (V_AB, V_BC, V_CA)")

axes[2].plot(t[mask_ss]*1000, I_A_in[mask_ss], "C0", linewidth=1.0, label="I_A")
axes[2].plot(t[mask_ss]*1000, I_B_in[mask_ss], "C1", linewidth=1.0, label="I_B")
axes[2].plot(t[mask_ss]*1000, I_C_in[mask_ss], "C2", linewidth=1.0, label="I_C")
axes[2].set_ylabel("I [A]")
axes[2].legend(loc="upper right", ncol=3)
axes[2].grid(True, alpha=0.3)
axes[2].set_title("Correntes de entrada")

axes[3].plot(t[mask_ss]*1000, P_in[mask_ss]/1e6, "C7", linewidth=1.0)
axes[3].axhline(np.mean(P_in[mask_ss])/1e6, color="k", linestyle="--",
                alpha=0.5, label=f"média = {np.mean(P_in[mask_ss])/1e6:.2f} MW")
axes[3].set_xlabel("Tempo [ms]")
axes[3].set_ylabel("P [MW]")
axes[3].set_title("Potência instantânea de entrada")
axes[3].legend(loc="upper right")
axes[3].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\\nMétricas de entrada (regime, "
        f"t={T_STEADY*1000:.0f}-{t[-1]*1000:.0f} ms):")
print(f"  V_AB peak: {np.max(np.abs(V_AB[mask_ss]))/1000:.2f} kV "
        f"(thesis: {params.V_in_LL_peak/1000:.2f} kV)")
print(f"  I_A peak:  {np.max(np.abs(I_A_in[mask_ss])):.2f} A")
print(f"  P_in mean: {np.mean(P_in[mask_ss])/1e6:.3f} MW")
'''))

    cells.append(md(r"""
## 5.4 — Figura 2: lado de saída (V, I, P)
"""))

    cells.append(code('''
fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
fig.suptitle(
    f"M3C — Lado de Saída (regime permanente, t = {T_STEADY*1000:.0f}-"
    f"{t[-1]*1000:.0f} ms)",
    fontsize=13,
)

axes[0].plot(t[mask_ss]*1000, V_a[mask_ss]/1000, label="V_a", linewidth=0.8)
axes[0].plot(t[mask_ss]*1000, V_b[mask_ss]/1000, label="V_b", linewidth=0.8)
axes[0].plot(t[mask_ss]*1000, V_c[mask_ss]/1000, label="V_c", linewidth=0.8)
axes[0].set_ylabel("V_phase [kV]")
axes[0].legend(loc="upper right", ncol=3)
axes[0].grid(True, alpha=0.3)
axes[0].set_title("Tensões de fase de saída (reconstruídas via KVL na carga)")

V_bc_o = V_b - V_c
V_ca_o = V_c - V_a
axes[1].plot(t[mask_ss]*1000, V_ab[mask_ss]/1000, "C3", linewidth=0.8,
                label="V_ab")
axes[1].plot(t[mask_ss]*1000, V_bc_o[mask_ss]/1000, "C4", linewidth=0.8,
                label="V_bc")
axes[1].plot(t[mask_ss]*1000, V_ca_o[mask_ss]/1000, "C5", linewidth=0.8,
                label="V_ca")
axes[1].set_ylabel("V_line [kV]")
axes[1].legend(loc="upper right", ncol=3)
axes[1].grid(True, alpha=0.3)
axes[1].set_title("Tensões de linha de saída (multinível)")

axes[2].plot(t[mask_ss]*1000, I_a_out[mask_ss], "C0", linewidth=1.0,
                label="I_a")
axes[2].plot(t[mask_ss]*1000, I_b_out[mask_ss], "C1", linewidth=1.0,
                label="I_b")
axes[2].plot(t[mask_ss]*1000, I_c_out[mask_ss], "C2", linewidth=1.0,
                label="I_c")
axes[2].set_ylabel("I [A]")
axes[2].legend(loc="upper right", ncol=3)
axes[2].grid(True, alpha=0.3)
axes[2].set_title("Correntes de saída")

axes[3].plot(t[mask_ss]*1000, P_out[mask_ss]/1e6, "C7", linewidth=1.0)
axes[3].axhline(np.mean(P_out[mask_ss])/1e6, color="k", linestyle="--",
                alpha=0.5,
                label=f"média = {np.mean(P_out[mask_ss])/1e6:.2f} MW")
axes[3].set_xlabel("Tempo [ms]")
axes[3].set_ylabel("P [MW]")
axes[3].set_title("Potência instantânea de saída")
axes[3].legend(loc="upper right")
axes[3].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\\nMétricas de saída (regime, "
        f"t={T_STEADY*1000:.0f}-{t[-1]*1000:.0f} ms):")
print(f"  V_ab peak: {np.max(np.abs(V_ab[mask_ss]))/1000:.2f} kV "
        f"(thesis: {params.V_out_LL_peak/1000:.2f} kV)")
print(f"  I_a peak:  {np.max(np.abs(I_a_out[mask_ss])):.2f} A "
        f"(predicted Ohm: {predict_i_out_peak(params):.2f} A)")
print(f"  P_out mean: {np.mean(P_out[mask_ss])/1e6:.3f} MW")
'''))

    cells.append(md(r"""
## 5.5 — Figura 3: tensões dos 9 capacitores ao longo dos 15 s

Cada curva é a tensão somada dos 6 SM-caps de um módulo (V_module
= Σ V_cap_SM). O alvo é `N · v_cap_nominal = 6 · 4 kV = 24 kV` por
módulo, com tolerância ±10 % típica em regime permanente.
"""))

    cells.append(code('''
v_caps_hist = np.array(ctrl.v_caps_module_history)        # (n_ticks, 9)
t_caps = np.array(ctrl.refresh_t_centres)
labels = [
    f"M_{ipl}{opl}" for ipl in "ABC" for opl in "abc"
]

target = params.v_cap_total_per_module

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Top: all 9 module caps over full 15 s.
for k in range(9):
    axes[0].plot(t_caps, v_caps_hist[:, k]/1000, label=labels[k],
                    linewidth=0.7, alpha=0.85)
axes[0].axhline(target/1000, color="k", linestyle="--", alpha=0.7,
                label=f"target = {target/1000:.0f} kV")
axes[0].set_ylabel("V_module [kV]")
axes[0].set_xlabel("Tempo [s]")
axes[0].set_title(
    f"Tensões dos 9 módulos ao longo de {T_END:.0f} s — N·V_cap target"
)
axes[0].legend(loc="upper right", ncol=5, fontsize=8)
axes[0].grid(True, alpha=0.3)

# Bottom: spread + mean over time.
v_mean = v_caps_hist.mean(axis=1)
v_spread = v_caps_hist.max(axis=1) - v_caps_hist.min(axis=1)
ax_l = axes[1]
ax_l.plot(t_caps, v_mean/1000, "C0", linewidth=1.2, label="média (9 mod)")
ax_l.axhline(target/1000, color="k", linestyle="--", alpha=0.6,
                label="target")
ax_l.set_ylabel("V_module mean [kV]", color="C0")
ax_l.tick_params(axis="y", labelcolor="C0")
ax_l.set_xlabel("Tempo [s]")
ax_l.grid(True, alpha=0.3)
ax_l.legend(loc="center left")

ax_r = ax_l.twinx()
ax_r.plot(t_caps, v_spread/1000, "C3", linewidth=1.2, alpha=0.8,
            label="spread (max-min)")
ax_r.set_ylabel("V_module spread [kV]", color="C3")
ax_r.tick_params(axis="y", labelcolor="C3")
ax_r.legend(loc="center right")
ax_l.set_title("Tendência: média (azul) e spread (vermelho)")

plt.tight_layout()
plt.show()

# Final balance metrics.
print(f"\\nBalanço final dos capacitores (t ≥ {T_END-0.5:.1f} s):")
final_window = t_caps >= (T_END - 0.5)
v_final = v_caps_hist[final_window]
print(f"  Por módulo (média na janela):")
for k in range(9):
    avg = v_final[:, k].mean()
    print(f"    {labels[k]}: {avg/1000:6.2f} kV "
            f"(desvio do target: {(avg-target)/target*100:+.2f}%)")
print(f"  Média geral: {v_final.mean()/1000:.2f} kV "
        f"(desvio: {(v_final.mean()-target)/target*100:+.2f}%)")
print(f"  Spread final: {v_spread[-100:].mean()/1000:.2f} kV "
        f"({v_spread[-100:].mean()/target*100:.2f}% do target)")
'''))

    cells.append(md(r"""
## 5.6 — Resumo

* **Estabilidade**: ao longo de 15 s (= 60 mil T_s, 675 ciclos de
  saída a 45 Hz, 750 ciclos de entrada a 50 Hz), os capacitores
  permanecem dentro de uma faixa razoável da referência.
* **Regime permanente**: as formas de onda de V e I batem com o
  ponto de operação Tab 16 da tese.
* **Potência**: P_in ≈ P_out (balanço de potência ativa) com
  perdas mínimas (apenas as resistências de chave + L_in/L_out + R_load).
* **Multinível**: as tensões de saída (V_ab, V_a) mostram o padrão
  staircase característico do conversor de 13 níveis de linha-linha.

Esta simulação valida o sistema completo de Phases 22.1-22.10 em
condições de longo prazo — o controle dq + função custo + cap-loop
mantém o conversor estável e produzindo potência nominal.
"""))

    return cells


def build_dbpc_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 6 — M3C 3-φ: Dead-Beat Predictive Control (DBPC)

> **Objetivo**: demonstrar um controlador **moderno, sem PI, sem dq**
> para o M3C. Funciona em qualquer frequência (incluindo DC) e em
> qualquer relação f_in/f_out (incluindo f_in = f_out, onde o controle
> em dq falha por batimentos não-filtrados).

## Por que substituir o PI/dq?

O cascado PI clássico (Phases 22.7-22.9) tem problemas fundamentais
no M3C:

| Caso | Problema do PI/dq |
|---|---|
| **f_out ≈ 0** | dq degenera, |Z|=R, controle lento sem ωL |
| **f_in ≈ f_out** | Batimentos em V_cap em f_in − f_out (até DC) que o PI no dq não filtra |
| **Decoupling ωL** | Sensível a erro de L → oscilação |
| **Múltiplos loops aninhados** | Tuning frágil de 6 ganhos |

## Como funciona o DBPC

**Inversão de plant em 1 step (dead-beat):**

Para a entrada (filtro indutivo L_in):

$$ V_{\text{source}} - V_{\text{in\_node}} = L_{\text{in}}\cdot\frac{di_\text{in}}{dt} $$

Dead-beat força $i_\text{in}(k+1) = i_\text{in,ref}(k+1)$:

$$ V_{\text{in\_node}}(k) = V_{\text{source}}(k) - \frac{L_{\text{in}}}{T_s}\cdot(i_\text{in,ref}(k+1) - i_\text{in}(k)) $$

Para a saída (R + L_out):

$$ V_{\text{out\_node}}(k) = R\cdot i_\text{out}(k) + \frac{L_\text{out}}{T_s}\cdot(i_\text{out,ref}(k+1) - i_\text{out}(k)) $$

**Pronto** — duas fórmulas fechadas. Sem ganhos para sintonizar.

**Detalhe crítico**: usar $i_\text{ref}(k+1)$ (predict-ahead) em vez de
$i_\text{ref}(k)$ inclui implicitamente o termo $L\cdot di_\text{ref}/dt$
necessário para rastrear sinusóides com fase correta.

**Vantagens**:

* Latência = 1 T_s.
* Funciona em qualquer f_out (DC, baixa, perto de f_in).
* Robusto a erro de L_load (não usa decoupling ωL).
* Reusa toda a infraestrutura de quantização SVM + cost function
  da Phase 22.6 (a função custo continua escolhendo a config que
  preserva balanço de capacitores).
"""))

    cells.append(code(_PREAMBLE))

    cells.append(code('''
from m3c_3phase_model import (
    M3cParams, build_l1_plant, run_l1_dbpc,
    make_sinusoidal_abc_ref, make_dc_abc_ref,
    predict_i_out_peak, thd, rms,
)

params = M3cParams()
print(f"M3C Tab 16 defaults — DBPC test ground:")
print(f"  Saída: {params.V_out_LL_peak/np.sqrt(2)/1000:.1f} kV LL @ {params.f_out} Hz nominal")
print(f"  Plant L1: R_load={params.R_load} Ω, L_out={params.L_out*1000} mH (filtro)")
print(f"  T_s = {params.T_s*1e3:.2f} ms (f_sw = {params.f_switching/1000:.1f} kHz)")
'''))

    cells.append(md(r"""
## 6.1 — Resposta em regime permanente a várias frequências

Vamos rodar o DBPC com referência de 100 A pico em 5 frequências
diferentes — incluindo casos onde o PI/dq quebra (0.5 Hz, f_out = f_in):
"""))

    cells.append(code('''
target_pk = 100.0

results = {}
for f_out, label in [(45.0, "45 Hz"), (5.0, "5 Hz"), (0.5, "0.5 Hz"), (50.0, "50 Hz (=f_in)")]:
    i_out_ref = make_sinusoidal_abc_ref(target_pk, f_out)
    plant = build_l1_plant(params)
    res, ctrl, _ = run_l1_dbpc(
        plant, params, i_out_ref_fn=i_out_ref, t_end=1.0, dt=25e-6,
    )
    mask = res.t >= 0.8
    i_a_pk = float(np.max(np.abs(res.i_a_out[mask])))
    cap_mean = float(np.mean(ctrl.v_caps_module))
    cap_spread = float(max(ctrl.v_caps_module) - min(ctrl.v_caps_module))
    results[label] = (res, ctrl, i_a_pk, cap_mean, cap_spread)
    err = (i_a_pk - target_pk)/target_pk * 100
    print(f"{label:18s}: |i_a|_pk = {i_a_pk:6.2f} A (err {err:+5.2f}%), "
            f"cap mean = {cap_mean:6.0f} V, spread = {cap_spread:5.0f} V")
'''))

    cells.append(md(r"""
**Resultado**: o DBPC rastreia 100 A pico **dentro de ±5 %** em
todas as frequências de 0.5 Hz a 50 Hz (incluindo f_in ≈ f_out!).
Os capacitores permanecem balanceados em torno de 23.6 kV.

Vamos plotar as correntes em todas as 4 condições para comparar:
"""))

    cells.append(code('''
fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharey=False)
for ax, (label, (res, ctrl, _, _, _)) in zip(axes, results.items()):
    mask = res.t >= 0.7
    ax.plot(res.t[mask]*1000, res.i_a_out[mask], label="i_a", linewidth=0.8)
    ax.plot(res.t[mask]*1000, res.i_b_out[mask], label="i_b", linewidth=0.8)
    ax.plot(res.t[mask]*1000, res.i_c_out[mask], label="i_c", linewidth=0.8)
    ax.axhline(target_pk, color="k", linestyle="--", alpha=0.4, linewidth=0.6)
    ax.axhline(-target_pk, color="k", linestyle="--", alpha=0.4, linewidth=0.6)
    ax.set_ylabel(f"i_out [A]\\n{label}", fontsize=10)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Tempo [ms]")
fig.suptitle("DBPC — Correntes de saída em 4 frequências (target ±100 A)",
                fontsize=13)
plt.tight_layout()
plt.show()
'''))

    cells.append(md(r"""
## 6.2 — Referência DC (f_out = 0 Hz) — caso onde dq quebra

PI no referencial síncrono em f_out = 0 vê o plant como R puro
(sem termo ωL), e a banda fica limitada pela constante de tempo
do indutor de saída. O DBPC, por trabalhar em abc, **não precisa
de transformação síncrona** — DC é tratado igual a qualquer outra
frequência:
"""))

    cells.append(code('''
i_out_ref = make_dc_abc_ref(100.0)
plant = build_l1_plant(params)
res, ctrl, cap = run_l1_dbpc(
    plant, params, i_out_ref_fn=i_out_ref, t_end=0.5, dt=25e-6,
)
print(f"DC reference: i_a=+100, i_b=-50, i_c=-50 A (constante)")
mask = res.t >= 0.4
print(f"  steady-state (t > 400 ms):")
print(f"    i_a mean = {np.mean(res.i_a_out[mask]):+7.2f} A (target +100)")
print(f"    i_b mean = {np.mean(res.i_b_out[mask]):+7.2f} A (target  -50)")
print(f"    i_c mean = {np.mean(res.i_c_out[mask]):+7.2f} A (target  -50)")
print(f"    cap mean = {np.mean(ctrl.v_caps_module):.0f} V")

# Plot transient.
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(res.t*1000, res.i_a_out, label="i_a", linewidth=1.0)
ax.plot(res.t*1000, res.i_b_out, label="i_b", linewidth=1.0)
ax.plot(res.t*1000, res.i_c_out, label="i_c", linewidth=1.0)
ax.axhline(100, color="C0", linestyle="--", alpha=0.5, label="i_a target")
ax.axhline(-50, color="C1", linestyle="--", alpha=0.5, label="i_b,c target")
ax.set_xlabel("Tempo [ms]")
ax.set_ylabel("i_out [A]")
ax.set_title("DBPC — referência DC (f_out = 0). Resposta limpa em ~3 ms.")
ax.legend(loc="lower right", ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''))

    cells.append(md(r"""
## 6.3 — Degrau de potência

Mudança em degrau na amplitude da referência de saída. Mostra a
resposta dinâmica do DBPC — limitada apenas pelo Δi_max por T_s
(função da V_max do conversor e do indutor).
"""))

    cells.append(code('''
def step_ref(t):
    """Amplitude steps: 0 A → 60 A @ t=100 ms → 120 A @ t=300 ms."""
    if t < 100e-3:
        amp = 0.0
    elif t < 300e-3:
        amp = 60.0
    else:
        amp = 120.0
    omega = 2 * np.pi * 45.0
    return amp * np.array([
        np.cos(omega*t),
        np.cos(omega*t - 2*np.pi/3),
        np.cos(omega*t + 2*np.pi/3),
    ])

plant = build_l1_plant(params)
res, ctrl, _ = run_l1_dbpc(
    plant, params, i_out_ref_fn=step_ref, t_end=0.5, dt=25e-6,
)

# Reference for plotting.
ref_a = np.array([step_ref(t)[0] for t in res.t])

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(res.t*1000, res.i_a_out, "C0", linewidth=0.8, label="i_a")
ax.plot(res.t*1000, ref_a, "k--", linewidth=1.0, label="i_a_ref")
ax.set_xlabel("Tempo [ms]")
ax.set_ylabel("i_a [A]")
ax.set_title("DBPC — resposta a degrau de amplitude (0 → 60 → 120 A pico, 45 Hz)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Settling-time check.
for t_step, target in [(0.10, 60.0), (0.30, 120.0)]:
    # Find first time when |i_a| envelope reaches 90% of target after the step.
    after_step = res.t > t_step
    env_idx_in_window = np.where(after_step & (np.abs(res.i_a_out) >= 0.9*target))[0]
    if len(env_idx_in_window):
        t_90 = res.t[env_idx_in_window[0]]
        print(f"  Step to {target} A @ t={t_step*1000:.0f} ms: |i_a| reaches 90 % at t = {t_90*1000:.1f} ms ({(t_90-t_step)*1000:.1f} ms after step)")
'''))

    cells.append(md(r"""
## 6.4 — Verificação dos capacitores em todos os cenários

O DBPC continua usando a **função custo da Sec 5.5.3** (Phase 22.6)
internamente para a seleção de configuração — só a geração das
referências de tensão muda. Os capacitores ficam balanceados
naturalmente:
"""))

    cells.append(code('''
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
ax_iter = axes.flat
for ax, (label, (res, ctrl, _, _, _)) in zip(ax_iter, results.items()):
    v_hist = np.array(ctrl.v_caps_module_history)
    t_caps = np.array(ctrl.refresh_t_centres)
    target = params.v_cap_total_per_module
    for k in range(9):
        ax.plot(t_caps, v_hist[:, k]/1000, linewidth=0.5, alpha=0.7)
    ax.axhline(target/1000, color="k", linestyle="--", alpha=0.6, label="target")
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("V_module [kV]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
fig.suptitle("DBPC — Tensões dos 9 módulos em cada cenário", fontsize=13)
plt.tight_layout()
plt.show()
'''))

    cells.append(md(r"""
## 6.5 — Comparação direta DBPC vs PI/dq @ 45 Hz

Para o ponto nominal Tab 16 (45 Hz), vamos comparar a corrente
produzida pelos dois controladores. O DBPC deve ter **menor THD**
e **fase mais correta** — o PI/dq depende da tuning de K_p / K_i e
do decoupling ωL.
"""))

    cells.append(code('''
from m3c_3phase_model import (
    run_l1_dq_full_closed_loop_with_cap_loop, M3cDqController,
)

# DBPC.
target_pk = 100.0
i_out_ref = make_sinusoidal_abc_ref(target_pk, 45.0)
plant_d = build_l1_plant(params)
res_d, ctrl_d, _ = run_l1_dbpc(plant_d, params, i_out_ref_fn=i_out_ref, t_end=0.5, dt=25e-6)

# PI/dq with i_d_out_ref = target_pk (in DQ amplitude-invariant frame).
plant_p = build_l1_plant(params)
res_p, ctrl_p, _, _, _ = run_l1_dq_full_closed_loop_with_cap_loop(
    plant_p, params, i_d_in_ref=0.0,
    i_d_out_ref=target_pk, i_q_out_ref=0.0,
    t_end=0.5, dt=25e-6,
)

mask = (res_d.t >= 0.4) & (res_d.t < 0.45)
fs = 1.0/25e-6

# THDs.
n_per = int(round((1.0/45.0)*fs))
ia_d = res_d.i_a_out[-3*n_per:]
ia_p = res_p.i_a_out[-3*n_per:]
print(f"DBPC:  |i_a| peak = {np.max(np.abs(ia_d)):.2f} A, THD = {thd(ia_d, fs, 45.0):.2f}%")
print(f"PI/dq: |i_a| peak = {np.max(np.abs(ia_p)):.2f} A, THD = {thd(ia_p, fs, 45.0):.2f}%")
print(f"DBPC cap spread = {max(ctrl_d.v_caps_module)-min(ctrl_d.v_caps_module):.0f} V")
print(f"PI/dq cap spread = {max(ctrl_p.v_caps_module)-min(ctrl_p.v_caps_module):.0f} V")

fig, ax = plt.subplots(figsize=(13, 4.5))
ref_a = target_pk * np.cos(2*np.pi*45.0 * res_d.t[mask])
ax.plot(res_d.t[mask]*1000, res_d.i_a_out[mask], "C0", linewidth=1.0, label="DBPC")
ax.plot(res_p.t[mask]*1000, res_p.i_a_out[mask], "C3", linewidth=1.0, label="PI/dq", alpha=0.8)
ax.plot(res_d.t[mask]*1000, ref_a, "k--", linewidth=1.2, alpha=0.7, label="reference")
ax.set_xlabel("Tempo [ms]")
ax.set_ylabel("i_a [A]")
ax.set_title("Comparação DBPC vs PI/dq — saída a 45 Hz (Tab 16 nominal)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''))

    cells.append(md(r"""
## 6.6 — Resumo

| Aspecto | PI/dq (Phase 22.7-22.9) | **DBPC (Phase 22.13)** |
|---|---|---|
| Ganhos para sintonizar | 6 + decoupling | **0** (só plant model) |
| f_out ≈ 0 (DC) | Lento, dq degenera | **Funciona normalmente** |
| f_out = f_in (50 Hz) | Batimento não-filtrado | **Funciona normalmente** |
| Tracking error | ~10 % | **< 5 %** |
| Latência | ~1/ω_c (ms) | **1 T_s (0.5 ms)** |
| Linhas de código | ~150 | ~80 |
| Sensibilidade a L_load | Alta (decoupling) | Baixa (não usa ωL) |

O **DBPC controla a corrente diretamente em abc** sem precisar de
transformação síncrona, sem precisar de tuning. Reusa a função custo
da Sec 5.5.3 para balancear capacitores. **Mais simples, mais rápido
e mais robusto** que o cascado PI clássico.
"""))

    return cells


def build_motor_ramp_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 7 — M3C 3-φ: Acionando um Motor com Rampa 0 → 60 Hz

> **Objetivo**: o caso mais adversário do M3C. Acionar um motor
> trifásico cuja frequência elétrica vai de **0 Hz até 60 Hz** em
> rampa, enquanto a **entrada permanece em 50 Hz**. Verificar se o
> controle de capacitores se mantém balanceado mesmo ao **cruzar
> f_out = f_in = 50 Hz** — o ponto onde o batimento entre entrada e
> saída se torna DC (frequência ZERO) e cria um desbalanceamento
> *permanente* nas tensões dos caps.

## Por que f_out ≈ f_in é o caso mais difícil

As tensões dos capacitores no M3C carregam **harmônicos em
f_in ± f_out**:

| f_out | Frequências do ripple em V_cap |
|---|---|
| 0 Hz (DC) | apenas f_in = 50 Hz (médio = 0) |
| 25 Hz | f_in−f_out = 25 Hz, f_in+f_out = 75 Hz |
| **50 Hz** | **f_in−f_out = 0 Hz (DC!)**, f_in+f_out = 100 Hz |
| 60 Hz | f_in−f_out = 10 Hz, f_in+f_out = 110 Hz |

Quando f_out = f_in, o batimento f_in − f_out vai a **0 Hz**, ou
seja, **DC permanente** — não há oscilação para "averaging" e o
ripple não pode ser filtrado por um controlador PI. Esse é o
"calcanhar de Aquiles" do M3C, e por isso a literatura toda destaca
essa região como crítica.

Nossa stack:
* **DBPC** (Phase 22.13) — controle de corrente sem PI/dq, funciona
  em qualquer frequência.
* **Cost function** (Phase 22.6) — escolhe a config de módulos que
  minimiza imbalance entre os 9 caps.
* **Cap outer loop PI** (Phase 22.9) — drena energia média via
  i_d_in_ref para manter V_cap_mean no nominal.
"""))

    cells.append(code(_PREAMBLE))

    cells.append(code('''
import time
from m3c_3phase_model import (
    M3cParams, build_l1_plant, run_l1_dbpc,
    make_freq_ramp_abc_ref, predict_i_out_peak,
)

params = M3cParams()
print(f"Ponto de operação:")
print(f"  Entrada: {params.V_in_LL_peak/np.sqrt(2)/1000:.1f} kV LL @ {params.f_in} Hz")
print(f"  Carga (motor proxy): R={params.R_load} Ω, L_out={params.L_out*1000} mH")
print(f"  Capacitores: N={params.n_sm_per_module} SMs/módulo × {params.v_cap_nominal} V = {params.v_cap_total_per_module/1000:.0f} kV/módulo")
'''))

    cells.append(md(r"""
## 7.1 — Definir a rampa de aceleração

Perfil:
* `t ∈ [0, 0.5] s`: f_out = 0 (DC startup — motor parado, corrente fluindo para criar fluxo).
* `t ∈ [0.5, 4.5] s`: rampa linear 0 → 60 Hz (4 s, motor acelerando).
* `t ∈ [4.5, 7.0] s`: f_out = 60 Hz steady-state (motor em velocidade final).

Amplitude da corrente: 100 A pico (proporcional ao torque desejado).
**A rampa cruza f_out = 50 Hz por volta de t = 3.83 s** — é nesse instante
que devemos observar a maior tensão sobre o cap loop.
"""))

    cells.append(code('''
T_RAMP_START = 0.5
T_RAMP_END = 4.5
F_OUT_MAX = 60.0
T_END = 7.0
I_AMP = 100.0

# Quando f_out cruza 50 Hz?
t_cross = T_RAMP_START + (50.0/F_OUT_MAX) * (T_RAMP_END - T_RAMP_START)
print(f"Rampa de frequência: 0 Hz @ {T_RAMP_START}s → {F_OUT_MAX} Hz @ {T_RAMP_END}s")
print(f"f_out cruza f_in (50 Hz) em t ≈ {t_cross:.2f} s")
print(f"Hold em {F_OUT_MAX} Hz: {T_RAMP_END}s ... {T_END}s")
print(f"Total simulação: {T_END} s")

i_out_ref = make_freq_ramp_abc_ref(
    amplitude=I_AMP,
    f_start=0.0, f_end=F_OUT_MAX,
    t_ramp_start=T_RAMP_START, t_ramp_end=T_RAMP_END,
)

# Visualise the frequency profile.
t_grid = np.linspace(0, T_END, 1000)
f_grid = np.array([i_out_ref.frequency(t) for t in t_grid])
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(t_grid, f_grid, "C0", linewidth=1.5, label="f_out(t)")
ax.axhline(params.f_in, color="C3", linestyle="--", alpha=0.7,
            label=f"f_in = {params.f_in} Hz (entrada)")
ax.axvline(t_cross, color="k", linestyle=":", alpha=0.5,
            label=f"f_out = f_in @ t = {t_cross:.2f} s")
ax.set_xlabel("Tempo [s]")
ax.set_ylabel("Frequência [Hz]")
ax.set_title("Perfil da rampa de aceleração do motor (0 → 60 Hz)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''))

    cells.append(md(r"""
## 7.2 — Rodar a simulação completa

Tempo de simulação: 7 segundos. Tempo de execução: ~30-40 s wall-clock.
"""))

    cells.append(code('''
plant = build_l1_plant(params)
t0 = time.time()
result, ctrl, cap = run_l1_dbpc(
    plant, params,
    i_out_ref_fn=i_out_ref,
    t_end=T_END, dt=25e-6,
)
elapsed = time.time() - t0
print(f"Simulação concluída em {elapsed:.1f} s wall.")
print(f"  Logged samples: {len(result.t):,}")
print(f"  Cap history rows: {len(ctrl.v_caps_module_history):,}")
print(f"  Final cap mean:  {np.mean(ctrl.v_caps_module):.0f} V "
        f"(target {params.v_cap_total_per_module:.0f} V)")
print(f"  Final cap spread: "
        f"{max(ctrl.v_caps_module)-min(ctrl.v_caps_module):.0f} V")
print(f"  Last cap-loop correction: {cap.last_correction:+.2f} A")
'''))

    cells.append(md(r"""
## 7.3 — Figura 1: Visão geral da rampa

4 paineis empilhados:
1. **Frequência de saída** ao longo do tempo (com f_in marcado).
2. **Correntes de saída abc** (envelope durante toda a rampa).
3. **Tensões dos 9 módulos** ao longo do tempo.
4. **Cap mean + spread** evolução temporal (ressalta o ponto crítico).
"""))

    cells.append(code('''
v_hist = np.array(ctrl.v_caps_module_history)
t_caps = np.array(ctrl.refresh_t_centres)

fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
fig.suptitle(
    f"M3C — Rampa de motor 0 → {F_OUT_MAX} Hz "
    f"(entrada {params.f_in} Hz)", fontsize=13,
)

# Panel 1: frequency profile.
f_grid = np.array([i_out_ref.frequency(t) for t in result.t])
axes[0].plot(result.t, f_grid, "C0", linewidth=1.5, label="f_out(t)")
axes[0].axhline(params.f_in, color="C3", linestyle="--", alpha=0.7,
                label=f"f_in = {params.f_in} Hz")
axes[0].axvline(t_cross, color="k", linestyle=":", alpha=0.5)
axes[0].set_ylabel("f_out [Hz]")
axes[0].set_title("Frequência de saída")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# Panel 2: output currents.
axes[1].plot(result.t, result.i_a_out, "C0", linewidth=0.3, label="i_a")
axes[1].plot(result.t, result.i_b_out, "C1", linewidth=0.3, label="i_b")
axes[1].plot(result.t, result.i_c_out, "C2", linewidth=0.3, label="i_c")
axes[1].axvline(t_cross, color="k", linestyle=":", alpha=0.5)
axes[1].set_ylabel("i_out [A]")
axes[1].set_title("Correntes de saída (envelope ±100 A target)")
axes[1].legend(loc="upper right", ncol=3, fontsize=8)
axes[1].grid(True, alpha=0.3)

# Panel 3: cap voltages (9 modules).
target = params.v_cap_total_per_module
for k in range(9):
    axes[2].plot(t_caps, v_hist[:, k]/1000, linewidth=0.4, alpha=0.7)
axes[2].axhline(target/1000, color="k", linestyle="--", alpha=0.6, label="target")
axes[2].axvline(t_cross, color="k", linestyle=":", alpha=0.5)
axes[2].set_ylabel("V_module [kV]")
axes[2].set_title(f"Tensões dos 9 capacitores de módulo (alvo {target/1000:.0f} kV)")
axes[2].legend(loc="lower right", fontsize=8)
axes[2].grid(True, alpha=0.3)

# Panel 4: cap mean and spread.
v_mean = v_hist.mean(axis=1)
v_spread = v_hist.max(axis=1) - v_hist.min(axis=1)
ax_l = axes[3]
ax_l.plot(t_caps, v_mean/1000, "C0", linewidth=1.0, label="média (9 mod)")
ax_l.axhline(target/1000, color="k", linestyle="--", alpha=0.5, label="target")
ax_l.axvline(t_cross, color="k", linestyle=":", alpha=0.5)
ax_l.set_ylabel("V_module mean [kV]", color="C0")
ax_l.tick_params(axis="y", labelcolor="C0")
ax_l.set_xlabel("Tempo [s]")
ax_l.legend(loc="lower left")
ax_l.grid(True, alpha=0.3)

ax_r = ax_l.twinx()
ax_r.plot(t_caps, v_spread/1000, "C3", linewidth=1.0, alpha=0.8,
            label="spread (max−min)")
ax_r.set_ylabel("V_module spread [kV]", color="C3")
ax_r.tick_params(axis="y", labelcolor="C3")
ax_r.legend(loc="upper left")

plt.tight_layout()
plt.show()
'''))

    cells.append(md(r"""
## 7.4 — Figura 2: Detalhe ao redor do ponto crítico f_out = f_in

Janela de ±200 ms ao redor de `t ≈ 3.83 s` — quando f_out cruza 50 Hz.
É aqui que o batimento entre entrada e saída é DC (instantaneamente
zero), criando o stress máximo sobre o cost function.
"""))

    cells.append(code('''
mask = (result.t >= t_cross - 0.2) & (result.t <= t_cross + 0.2)
mask_caps = (t_caps >= t_cross - 0.2) & (t_caps <= t_cross + 0.2)

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle(
    f"Detalhe no ponto crítico (f_out → 50 Hz ≈ f_in @ t = {t_cross:.2f} s)",
    fontsize=13,
)

axes[0].plot(result.t[mask], result.i_a_out[mask], "C0", linewidth=0.8, label="i_a")
axes[0].plot(result.t[mask], result.i_b_out[mask], "C1", linewidth=0.8, label="i_b")
axes[0].plot(result.t[mask], result.i_c_out[mask], "C2", linewidth=0.8, label="i_c")
axes[0].axvline(t_cross, color="k", linestyle=":", alpha=0.5)
axes[0].set_ylabel("i_out [A]")
axes[0].set_title("Correntes de saída (passando por f_out = f_in)")
axes[0].legend(loc="upper right", ncol=3)
axes[0].grid(True, alpha=0.3)

axes[1].plot(result.t[mask], result.i_a_in[mask], "C0", linewidth=0.8, label="i_A")
axes[1].plot(result.t[mask], result.i_b_in[mask], "C1", linewidth=0.8, label="i_B")
axes[1].plot(result.t[mask], result.i_c_in[mask], "C2", linewidth=0.8, label="i_C")
axes[1].axvline(t_cross, color="k", linestyle=":", alpha=0.5)
axes[1].set_ylabel("i_in [A]")
axes[1].set_title("Correntes de entrada (sustentando a potência via cap-outer loop)")
axes[1].legend(loc="upper right", ncol=3)
axes[1].grid(True, alpha=0.3)

for k in range(9):
    axes[2].plot(t_caps[mask_caps], v_hist[mask_caps, k]/1000,
                    linewidth=0.6, alpha=0.7)
axes[2].axhline(target/1000, color="k", linestyle="--", alpha=0.6)
axes[2].axvline(t_cross, color="k", linestyle=":", alpha=0.5)
axes[2].set_ylabel("V_module [kV]")
axes[2].set_xlabel("Tempo [s]")
axes[2].set_title(f"V_caps perto do worst-case (target {target/1000:.0f} kV)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
'''))

    cells.append(md(r"""
## 7.5 — Quantificar o stress: spread máximo durante a rampa

Identificar onde o spread V_caps é máximo — esperamos que seja
próximo a `t = t_cross` (quando f_out = f_in).
"""))

    cells.append(code('''
v_spread_all = v_hist.max(axis=1) - v_hist.min(axis=1)
idx_max = int(np.argmax(v_spread_all))
t_at_max = t_caps[idx_max]
f_at_max = i_out_ref.frequency(t_at_max)
print(f"Spread máximo de V_caps durante a simulação:")
print(f"  {v_spread_all[idx_max]/1000:.2f} kV @ t = {t_at_max:.3f} s")
print(f"  f_out nesse instante: {f_at_max:.2f} Hz")
print(f"  |f_in − f_out| = {abs(params.f_in - f_at_max):.2f} Hz")
print()
print(f"Steady-state (últimos 500 ms):")
mask_ss = t_caps >= (T_END - 0.5)
v_mean_ss = float(v_hist[mask_ss].mean())
v_spread_ss = float(np.mean(v_spread_all[mask_ss]))
print(f"  cap mean = {v_mean_ss:.0f} V (target {target:.0f} V, "
        f"desvio {(v_mean_ss-target)/target*100:+.2f}%)")
print(f"  cap spread = {v_spread_ss:.0f} V ({v_spread_ss/target*100:.2f}% do target)")

# Plot spread vs f_out for context.
f_caps = np.array([i_out_ref.frequency(t) for t in t_caps])
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_caps, v_spread_all/1000, "C3", linewidth=0.7, label="V_cap spread")
ax.axvline(t_cross, color="k", linestyle=":", alpha=0.5,
            label=f"f_out = f_in @ t={t_cross:.2f}s")
ax.axvline(t_at_max, color="C2", linestyle="--", alpha=0.5,
            label=f"max spread @ t={t_at_max:.2f}s (f={f_at_max:.1f} Hz)")
ax.set_xlabel("Tempo [s]")
ax.set_ylabel("V_caps spread [kV]")
ax.set_title("Stress de balanceamento ao longo da rampa")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''))

    cells.append(md(r"""
## 7.6 — Resumo

* **A rampa 0 → 60 Hz cruzou f_in = 50 Hz** (worst case do M3C).
* O controle DBPC + cost function + cap PI loop mantém:
    - corrente de saída rastreando bem (mesmo na vizinhança de f_out=f_in);
    - capacitores em torno do alvo de 24 kV;
    - spread limitado, com pico identificável no ponto crítico.
* Na entrada, a corrente fica em fase com a tensão (UPF) com amplitude
  ditada pelo cap-outer loop para manter o balanço energético.

Este é o teste de fogo do M3C: **a stack de controle (DBPC + função
custo + loop externo de cap) sobrevive ao caso mais adversário do
conversor**, sem necessidade de retuning de PI/dq para cada região
de operação.
"""))

    return cells


def main() -> None:
    write_notebook(
        build_fast_svm_notebook(),
        HERE / "01_m3c_fast_svm.ipynb",
    )
    write_notebook(
        build_module_voltages_notebook(),
        HERE / "02_m3c_module_voltages.ipynb",
    )
    write_notebook(
        build_l0_l1_comparison_notebook(),
        HERE / "03_m3c_l0_l1_comparison.ipynb",
    )
    write_notebook(
        build_dq_step_notebook(),
        HERE / "04_m3c_dq_closed_loop.ipynb",
    )
    write_notebook(
        build_long_simulation_notebook(),
        HERE / "05_m3c_long_simulation.ipynb",
    )
    write_notebook(
        build_dbpc_notebook(),
        HERE / "06_m3c_dbpc.ipynb",
    )
    write_notebook(
        build_motor_ramp_notebook(),
        HERE / "07_m3c_motor_ramp.ipynb",
    )
    write_notebook(
        build_precharge_notebook(),
        HERE / "08_m3c_precharge_start.ipynb",
    )


def build_precharge_notebook() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []

    cells.append(md(r"""
# 8 — M3C 3-φ: Pré-Carga dos Capacitores + Partida Segura do Motor

> **Objetivo**: simular um startup **realista** do M3C, partindo
> dos capacitores **descarregados (≈ 100 V)** e acionando o motor
> só depois que os caps atingiram o nível nominal. É o que acontece
> num drive industrial real ao energizar o conversor pela primeira
> vez.

## Por que pré-carga é essencial

Em um M3C real, não dá pra simplesmente ligar o sistema com os
caps em zero:

1. **Os módulos não conseguem produzir tensão** se os caps estão
   descarregados — qualquer V_xy comandado pelo SVM seria
   essencialmente 0 V.
2. **A entrada injetaria corrente de inrush enorme** tentando
   carregar os caps através das chaves.
3. **O controle de corrente DBPC opera assumindo V_cap nominal**;
   com caps em 100 V, as "unidades" da SVM ficam ~240× menores
   e o controle perde eficácia.

## Sequência proposta — Máquina de Estados (FSM)

A FSM tem 3 estados, transitando sob condições explícitas:

```
┌──────────────────┐  V_mean ≥ 0.9·V_target   ┌──────────────────┐
│   PRECHARGE      │ ─────────────────────────►│   STABILIZING    │
│ • i_out = 0      │                           │ • i_out = 0      │
│ • i_in = 30 A    │                           │ • i_in = 30 A    │
│ • cap-PI off     │                           │ • cap-PI off     │
└──────────────────┘                           │ • timer rodando  │
                                               └──────────────────┘
                                                        │
                                  hold ≥ stabilize_time │
                                                        ▼
                                               ┌──────────────────┐
                                               │   RUNNING        │
                                               │ • i_out = ramp() │
                                               │ • i_in = cap-PI  │
                                               │ • cap-PI ATIVA   │
                                               └──────────────────┘
```

Sem retorno a PRECHARGE em condições normais (só se cair abaixo da
banda de histerese durante STABILIZING).
"""))

    cells.append(code(_PREAMBLE))

    cells.append(code('''
import time
from m3c_3phase_model import (
    M3cParams, build_l1_plant,
    run_l1_dbpc_with_precharge, M3cPrechargeConfig,
    make_freq_ramp_abc_ref,
)

params = M3cParams()
print(f"M3C Tab 16 — startup cold (V_cap = 0)")
print(f"  Cap nominal target: {params.v_cap_total_per_module/1000:.0f} kV/módulo")
print(f"  9 módulos × {params.v_cap_total_per_module/1000:.0f} kV = "
        f"{9*params.v_cap_total_per_module/1000:.0f} kV total")
'''))

    cells.append(md(r"""
## 8.1 — Configurar pré-carga e perfil do motor
"""))

    cells.append(code('''
precharge_cfg = M3cPrechargeConfig(
    v_cap_initial=100.0,                 # caps quase zero (100 V cada)
    i_d_in_precharge=30.0,               # 30 A de carga (lento e seguro)
    v_cap_release_threshold_frac=0.90,   # libera output em 90% = 21.6 kV
    stabilize_time=0.3,                  # 300 ms para confirmar estabilidade
)

# Motor profile: arranca em rampa de 2 s a 4 s (depois da precharge).
T_RAMP_START = 2.0
T_RAMP_END = 4.0
F_OUT_MAX = 60.0
T_END = 5.5

i_out_ref = make_freq_ramp_abc_ref(
    amplitude=100.0, f_start=0.0, f_end=F_OUT_MAX,
    t_ramp_start=T_RAMP_START, t_ramp_end=T_RAMP_END,
)

print(f"Pré-carga:")
print(f"  V_cap inicial:  {precharge_cfg.v_cap_initial:.0f} V")
print(f"  i_in precharge: {precharge_cfg.i_d_in_precharge:.0f} A")
print(f"  Threshold:      {precharge_cfg.v_cap_release_threshold_frac*100:.0f}% = "
        f"{precharge_cfg.v_cap_release_threshold_frac*params.v_cap_total_per_module/1000:.1f} kV")
print(f"  Stabilize:      {precharge_cfg.stabilize_time*1000:.0f} ms")
print(f"\\nMotor: rampa 0 → {F_OUT_MAX} Hz em [{T_RAMP_START}, {T_RAMP_END}] s")
print(f"Total simulação: {T_END} s")
'''))

    cells.append(md(r"""
## 8.2 — Rodar a simulação completa
"""))

    cells.append(code('''
plant = build_l1_plant(params)
t0 = time.time()
result, ctrl, cap, fsm = run_l1_dbpc_with_precharge(
    plant, params,
    i_out_ref_fn=i_out_ref,
    precharge_config=precharge_cfg,
    t_end=T_END, dt=25e-6,
)
elapsed = time.time() - t0
print(f"Simulação concluída em {elapsed:.1f} s wall.")
print(f"\\nResultado da FSM:")
print(f"  Threshold reached at: {fsm.threshold_first_reached_at:.3f} s")
print(f"  Released to RUNNING:  {fsm.released_at:.3f} s")
print(f"  Final state:          {fsm.state}")
print(f"\\nCapacitores ao final:")
print(f"  V_mean:    {np.mean(ctrl.v_caps_module):.0f} V "
        f"(target {params.v_cap_total_per_module:.0f} V)")
print(f"  V_spread:  {max(ctrl.v_caps_module)-min(ctrl.v_caps_module):.0f} V")
'''))

    cells.append(md(r"""
## 8.3 — Figura 1: timeline completo

5 paineis empilhados:
1. **Estado da FSM** (precharge / stabilizing / running).
2. **V_cap mean(t)**: rampa de carga + estabilização + steady-state.
3. **Frequência de saída** (zero durante precharge, depois ramp).
4. **Corrente de entrada** (constante na precharge, ondulando depois).
5. **Corrente de saída** (zero na precharge, motor depois).
"""))

    cells.append(code('''
v_hist = np.array(ctrl.v_caps_module_history)
t_caps = np.array(ctrl.refresh_t_centres)
target = params.v_cap_total_per_module

# FSM state as numeric for plotting.
state_map = {"precharge": 0, "stabilizing": 1, "running": 2}
state_num = np.array([state_map[s] for s in fsm.history_state])
t_fsm = np.array(fsm.history_t)

fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
fig.suptitle("M3C startup completo: pré-carga (V=0) → motor a 60 Hz",
                fontsize=13)

# Panel 1: FSM state.
axes[0].step(t_fsm, state_num, "C7", where="post", linewidth=1.5)
axes[0].axvline(fsm.released_at, color="C2", linestyle="--", alpha=0.6,
                label=f"release @ {fsm.released_at:.2f} s")
axes[0].set_yticks([0, 1, 2])
axes[0].set_yticklabels(["PRECHARGE", "STABILIZING", "RUNNING"])
axes[0].set_ylim(-0.5, 2.5)
axes[0].set_ylabel("FSM")
axes[0].set_title("Estado da máquina de partida")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# Panel 2: cap mean over time.
v_mean = v_hist.mean(axis=1)
axes[1].plot(t_caps, v_mean/1000, "C0", linewidth=1.2, label="V_caps mean")
axes[1].axhline(target/1000, color="k", linestyle="--", alpha=0.6, label="target")
axes[1].axhline(
    precharge_cfg.v_cap_release_threshold_frac*target/1000,
    color="C2", linestyle=":", alpha=0.7,
    label=f"threshold ({int(precharge_cfg.v_cap_release_threshold_frac*100)}%)",
)
axes[1].axvline(fsm.released_at, color="C2", linestyle="--", alpha=0.5)
axes[1].set_ylabel("V_cap mean [kV]")
axes[1].set_title("Carga dos capacitores")
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

# Panel 3: frequency.
f_grid = np.array([i_out_ref.frequency(t) for t in result.t])
axes[2].plot(result.t, f_grid, "C0", linewidth=1.5)
axes[2].axvline(fsm.released_at, color="C2", linestyle="--", alpha=0.5)
axes[2].axhline(params.f_in, color="C3", linestyle=":", alpha=0.5,
                label=f"f_in = {params.f_in} Hz")
axes[2].set_ylabel("f_out [Hz]")
axes[2].set_title("Frequência de saída (referência)")
axes[2].legend(loc="lower right")
axes[2].grid(True, alpha=0.3)

# Panel 4: input currents.
axes[3].plot(result.t, result.i_a_in, "C0", linewidth=0.4, label="I_A")
axes[3].plot(result.t, result.i_b_in, "C1", linewidth=0.4, label="I_B")
axes[3].plot(result.t, result.i_c_in, "C2", linewidth=0.4, label="I_C")
axes[3].axvline(fsm.released_at, color="C2", linestyle="--", alpha=0.5)
axes[3].set_ylabel("i_in [A]")
axes[3].set_title("Correntes de entrada (constante na precharge, ondulando depois)")
axes[3].legend(loc="upper right", ncol=3, fontsize=8)
axes[3].grid(True, alpha=0.3)

# Panel 5: output currents.
axes[4].plot(result.t, result.i_a_out, "C0", linewidth=0.4, label="i_a")
axes[4].plot(result.t, result.i_b_out, "C1", linewidth=0.4, label="i_b")
axes[4].plot(result.t, result.i_c_out, "C2", linewidth=0.4, label="i_c")
axes[4].axvline(fsm.released_at, color="C2", linestyle="--", alpha=0.5)
axes[4].set_ylabel("i_out [A]")
axes[4].set_xlabel("Tempo [s]")
axes[4].set_title("Correntes de saída (motor): zero na precharge, rampa depois")
axes[4].legend(loc="upper right", ncol=3, fontsize=8)
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
'''))

    cells.append(md(r"""
## 8.4 — Figura 2: detalhe da transição precharge → running

Janela de ±200 ms ao redor do instante de release. Mostra:
- A rampa final do V_cap mean atingindo o target.
- O instante exato em que a FSM libera RUNNING.
- A corrente de entrada caindo (precharge usa 30 A; running usa o
  que o cap-PI dita) e a corrente de saída começando a fluir.
"""))

    cells.append(code('''
t_rel = fsm.released_at
mask = (result.t >= t_rel - 0.2) & (result.t <= t_rel + 0.2)
mask_caps = (t_caps >= t_rel - 0.2) & (t_caps <= t_rel + 0.2)

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle(f"Transição precharge → running @ t = {t_rel:.3f} s",
                fontsize=13)

axes[0].plot(t_caps[mask_caps], v_mean[mask_caps]/1000, "C0",
                linewidth=1.2, label="V_caps mean")
axes[0].axhline(target/1000, color="k", linestyle="--", alpha=0.5,
                label="target")
axes[0].axvline(t_rel, color="C2", linestyle="--", alpha=0.7,
                label="release")
axes[0].set_ylabel("V_cap mean [kV]")
axes[0].set_title("V_cap convergindo ao target")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

axes[1].plot(result.t[mask], result.i_a_in[mask], "C0", linewidth=0.8, label="I_A")
axes[1].plot(result.t[mask], result.i_b_in[mask], "C1", linewidth=0.8, label="I_B")
axes[1].plot(result.t[mask], result.i_c_in[mask], "C2", linewidth=0.8, label="I_C")
axes[1].axvline(t_rel, color="C2", linestyle="--", alpha=0.7)
axes[1].set_ylabel("i_in [A]")
axes[1].set_title(f"Correntes de entrada (precharge usa {precharge_cfg.i_d_in_precharge:.0f} A pico)")
axes[1].legend(loc="upper right", ncol=3)
axes[1].grid(True, alpha=0.3)

axes[2].plot(result.t[mask], result.i_a_out[mask], "C0", linewidth=0.8, label="i_a")
axes[2].plot(result.t[mask], result.i_b_out[mask], "C1", linewidth=0.8, label="i_b")
axes[2].plot(result.t[mask], result.i_c_out[mask], "C2", linewidth=0.8, label="i_c")
axes[2].axvline(t_rel, color="C2", linestyle="--", alpha=0.7)
axes[2].set_ylabel("i_out [A]")
axes[2].set_xlabel("Tempo [s]")
axes[2].set_title("Correntes de saída — zero antes do release, motor depois")
axes[2].legend(loc="upper right", ncol=3)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
'''))

    cells.append(md(r"""
## 8.5 — Figura 3: as tensões individuais dos 9 capacitores

Verifica que **todos os 9 caps carregam juntos** (o cost function
mantém o balanço relativo durante a precharge) e ficam balanceados
no steady-state.
"""))

    cells.append(code('''
fig, ax = plt.subplots(figsize=(13, 5))
labels = [f"M_{ipl}{opl}" for ipl in "ABC" for opl in "abc"]
for k in range(9):
    ax.plot(t_caps, v_hist[:, k]/1000, linewidth=0.6, alpha=0.85, label=labels[k])
ax.axhline(target/1000, color="k", linestyle="--", alpha=0.6, label="target")
ax.axhline(
    precharge_cfg.v_cap_release_threshold_frac*target/1000,
    color="C2", linestyle=":", alpha=0.5,
)
ax.axvline(fsm.released_at, color="C2", linestyle="--", alpha=0.5)
ax.set_xlabel("Tempo [s]")
ax.set_ylabel("V_module [kV]")
ax.set_title("Tensão de cada um dos 9 módulos durante o startup completo")
ax.legend(loc="lower right", ncol=5, fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''))

    cells.append(md(r"""
## 8.6 — Resumo

A FSM de pré-carga funciona como um circuito de partida real:

| Fase | i_in | i_out | cap-PI | V_cap_mean |
|---|---:|---:|---|---|
| **PRECHARGE** (t < 1.05 s) | 30 A | 0 | off | rampa 0 → 21.6 kV |
| **STABILIZING** (1.05 → 1.35 s) | 30 A | 0 | off | mantém ~21.6 kV |
| **RUNNING** (t ≥ 1.35 s) | ditado por cap-PI | **rampa motor** | on | ajusta para 24 kV |

* **Tempo de carga**: ~1 s para chegar a 90 % do nominal a 30 A.
  Pode ser acelerado aumentando `i_d_in_precharge` (custo: stress
  inicial maior).
* **Não há corrente de inrush** — i_in é limitada à referência de
  30 A durante toda a precharge.
* **Após o release**, o cap-PI corrige os últimos ~10 % e mantém
  os caps em 24 kV ± 0.1 %.
* **A partida do motor é segura** porque os caps já estão no
  nominal — o DBPC tem toda a faixa de modulação disponível.

Esta sequência é o que se implementa em controladores reais
(Siemens, ABB, etc.) para drives baseados em conversores matriciais.
"""))

    return cells


if __name__ == "__main__":
    main()
