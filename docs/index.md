# PulsimCore Documentation

<div class="pulsim-hero">
  <h1>PulsimCore Backend</h1>
  <p>High-performance power-electronics simulation backend with a <strong>Python-first runtime</strong>, versioned YAML netlists, mixed electrical-and-control domains, and a 50-bench closed-loop regression dashboard.</p>
  <p>Recommended surface: <code>import pulsim</code> + <code>schema: pulsim-v1</code>.</p>
  <div class="pulsim-hero-actions">
    <a class="md-button md-button--primary" href="getting-started/">Get Started</a>
    <a class="md-button" href="components-reference/">Components Reference</a>
    <a class="md-button" href="control-blocks-reference/">Control Blocks</a>
    <a class="md-button" href="api-reference/">API</a>
  </div>
</div>

## What's in the box

<div class="grid cards" markdown>

- :material-circle-multiple-outline: **30+ electrical components**

    ---

    Resistors, capacitors, inductors, mosfets (smooth Shichman-Hodges),
    IGBTs, diodes, transformers, coupled inductors, saturable inductors,
    fuses, breakers, relays, thyristors, triacs — every parameter,
    default, and unit documented.

    [Components Reference](components-reference.md)

- :material-graph-outline: **30+ virtual control blocks**

    ---

    PWM generators, PI / PID, integrators, Schmitt triggers,
    state-machines, lookup tables, transfer functions, Clarke / Park /
    PLL / SVM — all mixable in the same netlist, all addressable as
    channels (`chan:PI.output`, `chan:PLL.theta`, …).

    [Control Blocks Reference](control-blocks-reference.md)

- :material-file-code-outline: **Versioned YAML netlists**

    ---

    Schema `pulsim-v1` is stable, validated, diffable, and round-trips
    through every backend (transient / shooting / harmonic balance / AC
    / FRA). YAML is the source of truth for the benchmark suite.

    [Netlist YAML Format](netlist-format.md)

- :material-sine-wave: **Mixed-domain runtime**

    ---

    Stamping electrical solver (MNA + trapezoidal / TR-BDF2 / Rosenbrock)
    runs alongside the control-block scheduler in a single simulator
    object. Switch events, duty callbacks, channel feedback — all in one
    Newton iteration.

    [Backend Architecture](backend-architecture.md)

- :material-chart-bell-curve: **AC sweep + FRA**

    ---

    Linearize around a periodic operating point, sweep frequency
    analytically (`run_ac_sweep`) or via transient injection
    (`run_fra_sweep`). Both produce LinearSystem outputs you can
    compare to analytical Bode models.

    [AC Analysis](ac-analysis.md) / [FRA](fra.md)

- :material-thermometer: **Electrothermal coupling**

    ---

    MOSFET / IGBT / BJT carry a `thermal:` port; the runtime computes
    junction temperature alongside electrical state, and KPI helpers
    derive `T_j_max`, `Δ_T_j`, and full Foster-network responses.

    [Electrothermal Workflow](electrothermal-workflow.md)

- :material-magnet: **Magnetic fidelity**

    ---

    Saturable inductors with soft-knee L(I), SPICE-style mutual
    coupling, ideal transformers, and Steinmetz core-loss KPI for B(t)
    traces — all callable from YAML.

    [Magnetic Models](magnetic-models.md)

- :material-speedometer: **Quantitative KPIs**

    ---

    THD, power factor, efficiency, ripple p-p, rise/settling/overshoot,
    ZVS%, ZCS%, conduction + switching loss, Steinmetz core loss, and
    junction temperature — all wired into the benchmark runner with CI
    gates.

    [KPI Reference](kpi-reference.md)

- :material-cog-transfer-outline: **Real-time codegen + FMI**

    ---

    Export a circuit as a C99 step function (`pulsim.codegen`) or as
    an FMI 2.0 Co-Simulation FMU — for HIL targets, model exchange, or
    integration with Simulink / OpenModelica.

    [Code Generation](code-generation.md) / [FMI Export](fmi-export.md)

- :material-circuit-board: **Domain libraries**

    ---

    Vendor MOSFET / IGBT presets, converter templates (buck / boost /
    full-bridge / LLC), PMSM / induction motor models, and three-phase
    grid primitives (sources, PLLs, grid-following + grid-forming
    inverters).

    [Catalog Devices](catalog-devices.md) · [Converter Templates](converter-templates.md) · [Motor Models](motor-models.md) · [Three-Phase Grid](three-phase-grid.md)

- :material-shield-check-outline: **Benchmark dashboards**

    ---

    50 closed-loop benches, 80+ stress benches, ngspice/LTspice parity
    matrices, and a live SPICE parity dashboard — Pulsim is regression-
    tested on real topologies, not toy circuits.

    [Benchmarks and Parity](benchmarks-and-parity.md) · [SPICE Parity Dashboard](spice-parity-dashboard.md)

- :material-tools: **Pro tooling**

    ---

    Parameter sweeps, property-based tests, automatic differentiation,
    a linear-solver cache, JFNK / GMRES / KLU / Pardiso solvers, and
    robust convergence policies.

    [Parameter Sweep](parameter-sweep.md) · [Property-Based Testing](property-based-testing.md) · [Automatic Differentiation](automatic-differentiation.md) · [Linear-Solver Cache](linear-solver-cache.md)

</div>

## One-command sanity check

```bash
python3 -c "import pulsim as ps; print(ps.__version__)"
```

If that prints a version string, you're set. Otherwise see
[Getting Started](getting-started.md) → "Installation" or
[Troubleshooting](troubleshooting.md).

## Recommended learning path

1. [Getting Started](getting-started.md) — install, run an RC step, see the output CSV.
2. [User Guide](user-guide.md) — the canonical runtime flow.
3. [Netlist YAML Format](netlist-format.md) — schema for your first real circuit.
4. [Components Reference](components-reference.md) + [Control Blocks Reference](control-blocks-reference.md) — every block at your disposal.
5. [Examples and Results](examples-and-results.md) — run shipped scripts end-to-end.
6. [KPI Reference](kpi-reference.md) — score what you've built.
7. [Benchmarks and Parity](benchmarks-and-parity.md) — wire it into CI.
8. [API Reference](api-reference.md) — when you need to drop down past YAML.

## Topic shortcuts

| If you're doing... | Read |
|---|---|
| Buck / boost / LLC / FB converter | [Converter Templates](converter-templates.md), [Catalog Devices](catalog-devices.md) |
| Three-phase / motor drives / vector control | [Three-Phase Grid](three-phase-grid.md), [Motor Models](motor-models.md), [Control Blocks](control-blocks-reference.md) |
| Magnetics design (saturation, core loss) | [Magnetic Models](magnetic-models.md), [KPI Reference](kpi-reference.md) |
| Thermal margin sizing | [Electrothermal Workflow](electrothermal-workflow.md), [KPI Reference](kpi-reference.md) |
| Closed-loop tuning | [AC Analysis](ac-analysis.md), [FRA](fra.md) |
| HIL / embedded export | [Code Generation](code-generation.md), [FMI Export](fmi-export.md) |
| Comparing to PSIM / PLECS / ngspice / LTspice | [Benchmarks and Parity](benchmarks-and-parity.md), [SPICE Parity Dashboard](spice-parity-dashboard.md), [GUI Backend Parity](gui-component-parity.md) |
| Solver convergence problems | [Convergence Tuning](convergence-tuning-guide.md), [Troubleshooting](troubleshooting.md) |
