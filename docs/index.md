# PulsimCore Documentation

```{raw} html
<div class="hero">
  <h1>PulsimCore</h1>
  <p>Simulador de eletrônica de potência com runtime <strong>Python-first</strong> e netlists em YAML.</p>
  <p>Superfície suportada: <code>import pulsim</code> + parser YAML + benchmark/paridade/stress suite.</p>
</div>
```

## Comece aqui

```{toctree}
:maxdepth: 2
:caption: Uso da biblioteca

getting-started
user-guide
python-api
netlist-format
convergence-tuning-guide
benchmarks-and-parity
notebooks
```

```{toctree}
:maxdepth: 1
:caption: Referência e migração

device-models
migration-guide
```

## Fluxo recomendado (rápido)

1. Build local das bindings Python.
2. Carregar um YAML (`schema: pulsim-v1`) com `YamlParser`.
3. Rodar `Simulator(...).run_transient(...)`.
4. Validar com benchmark/paridade (`ngspice`/`LTspice`) e stress suite.
