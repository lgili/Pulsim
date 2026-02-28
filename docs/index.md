# PulsimCore Documentation

<div class="pulsim-hero">
  <h1>PulsimCore Backend</h1>
  <p>High-performance power electronics simulation backend with a <strong>Python-first runtime</strong>, versioned YAML netlists, and robust convergence tooling.</p>
  <p>Recommended surface: <code>import pulsim</code> + <code>schema: pulsim-v1</code>.</p>
  <div class="pulsim-hero-actions">
    <a class="md-button md-button--primary" href="getting-started/">Get Started</a>
    <a class="md-button" href="api-reference/">API Reference</a>
    <a class="md-button" href="examples-and-results/">Examples</a>
  </div>
</div>

## What You Can Do With PulsimCore

<div class="grid cards" markdown>

- :material-rocket-launch-outline: **Run simulations from Python**

  ---

  Build locally or install from package, then execute transient runs through `Simulator`.

  [Getting Started](getting-started.md)

- :material-file-code-outline: **Drive the backend with YAML netlists**

  ---

  Keep simulations reproducible and versioned with `pulsim-v1` schema.

  [Netlist YAML Format](netlist-format.md)

- :material-sine-wave: **Model switched converters and control loops**

  ---

  Use mixed-domain blocks, event handling, and duty callbacks in production-like scenarios.

  [Examples and Results](examples-and-results.md)

- :material-chart-line: **Measure performance and parity**

  ---

  Validate runtime and waveform fidelity against benchmark baselines and SPICE tools.

  [Benchmarks and Parity](benchmarks-and-parity.md)

- :material-api: **Integrate through a typed API**

  ---

  Navigate classes, options, and enums generated from the package interface.

  [API Reference](api-reference.md)

- :material-tools: **Operate docs and release pipeline**

  ---

  Publish versioned docs in GitHub Pages and keep strict docs checks in CI.

  [Docs Versioning and Release](versioning-and-release.md)

</div>

## Backend In One Command

```bash
PYTHONPATH=build/python python3 -c "import pulsim as ps; print(ps.__version__)"
```

## Recommended Learning Path

1. Start with [Getting Started](getting-started.md).
2. Follow the [User Guide](user-guide.md) to understand the canonical runtime flow.
3. Run [Examples and Results](examples-and-results.md) end-to-end.
4. Use [Benchmarks and Parity](benchmarks-and-parity.md) to set CI quality gates.
5. Integrate against [API Reference](api-reference.md).
