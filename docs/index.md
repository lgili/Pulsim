# PulsimCore

<div class="pulsim-hero">
  <h1>PulsimCore Documentation</h1>
  <p>Simulação de eletrônica de potência com foco em <strong>Python-first runtime</strong>, netlists YAML e validação contra ferramentas de referência.</p>
  <p>API suportada: <code>import pulsim</code> + schema <code>pulsim-v1</code>.</p>
  <div class="pulsim-hero-actions">
    <a class="md-button md-button--primary" href="getting-started/">Começar agora</a>
    <a class="md-button" href="python-api/">API Python</a>
    <a class="md-button" href="versioning-and-release/">Versionamento</a>
  </div>
</div>

## Trilha recomendada

<div class="grid cards" markdown>

- :material-rocket-launch-outline: **Primeiro contato**

  ---

  Build local, primeiro circuito e primeiro `run_transient`.

  [Getting Started](getting-started.md)

- :material-sine-wave: **Simulação de conversores**

  ---

  Buck/Boost/Flyback/Forward, malha fechada e tuning de controle.

  [Notebooks](notebooks.md)

- :material-memory: **Solver + convergência**

  ---

  Estratégias de fallback, robustez numérica e tuning de timestep.

  [Convergence Tuning](convergence-tuning-guide.md)

- :material-file-code-outline: **YAML e componentes**

  ---

  Formato de netlist, componentes avançados e paridade GUI/backend.

  [Netlist YAML](netlist-format.md)

- :material-chart-line: **Validação e benchmark**

  ---

  Métricas de precisão/performance e comparação com LTspice/ngspice.

  [Benchmarks and Parity](benchmarks-and-parity.md)

- :material-history: **Docs por versão**

  ---

  Cada tag `vX.Y.Z` gera documentação versionada com histórico preservado.

  [Versionamento e Release](versioning-and-release.md)

</div>

## Quick start (2 comandos)

```bash
cmake -S . -B build -G Ninja -DPULSIM_BUILD_PYTHON=ON
PYTHONPATH=build/python python3 -c "import pulsim; print(pulsim.__version__)"
```

## O que esta documentação cobre

- API Python completa (configuração de solver, integração, análise e callbacks).
- Fluxo YAML `pulsim-v1` com componentes elétricos, térmicos e mixed-domain.
- Exemplos reproduzíveis com notebooks para projeto e validação.
- Diretrizes de release para manter docs por versão no GitHub Pages.
