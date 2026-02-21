# PulsimCore Documentation

<div class="pulsim-hero">
  <h1>PulsimCore</h1>
  <p>Simulação de eletrônica de potência com runtime <strong>Python-first</strong>.</p>
  <p>Uso suportado: <code>import pulsim</code> + netlists YAML <code>pulsim-v1</code>.</p>
</div>

## O que você encontra aqui

<div class="pulsim-grid">
  <div class="pulsim-card">
    <strong>Getting Started</strong><br>
    Build, primeiro circuito e primeiro resultado.
  </div>
  <div class="pulsim-card">
    <strong>API + Configuração</strong><br>
    Classes Python, integradores, solver stack, térmico e fallback.
  </div>
  <div class="pulsim-card">
    <strong>Exemplos + Resultados</strong><br>
    Conversores, benchmark, paridade com SPICE e stress tiers.
  </div>
  <div class="pulsim-card">
    <strong>Versões</strong><br>
    Cada release tag gera uma versão navegável da documentação.
  </div>
</div>

## Fluxo recomendado

1. Faça build local das bindings Python.
2. Carregue YAML com `YamlParser`.
3. Execute `Simulator(...).run_transient(...)`.
4. Valide com benchmark/paridade/stress suite.

## Navegação

Use o menu superior para:

- começar rapidamente;
- configurar simulações robustas;
- consultar API e formato YAML;
- comparar resultados com `ngspice`/`LTspice`;
- selecionar versões antigas da documentação.
