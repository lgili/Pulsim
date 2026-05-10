## Why

System integrators rarely use a single simulator. They build the converter in PSIM/Pulsim, the motor in PLECS, the grid in PSCAD, the controller in Simulink/Modelica/Dymola, and then need to **co-simulate**. The standard for this is FMI (Functional Mock-up Interface), versions 2.0 and 3.0, supported by Modelica/OpenModelica, MATLAB/Simulink, Dymola, ANSYS Twin Builder, GT-Suite, and a long tail of automotive and aerospace tooling.

Without FMI, Pulsim simulations cannot be embedded in larger system-level studies. With it, Pulsim becomes a citizen of every multi-physics workflow.

OpenSource references:
- **fmilibrary** (Modelon, BSD-3) — production-grade FMI 2.0/3.0 master and slave.
- **PythonFMU** — Python-based FMU export with optional Cython.
- **OMSimulator** (OpenModelica) — multi-FMU co-simulation orchestrator.

## What Changes

### FMU Export (Co-Simulation FMU)
- New CLI: `pulsim fmu-export <netlist.yaml> --version 2.0|3.0 --type cs --out my_model.fmu`.
- Internally: Pulsim runs as the FMU's solver; FMI exposes `set_input(name, value)`, `do_step(t, dt)`, `get_output(name)`.
- Generated `modelDescription.xml` with declared inputs/outputs/parameters/states.
- Packaging: zip with `binaries/<platform>/`, `sources/`, `resources/`, `modelDescription.xml`.

### Model Exchange FMU (Stretch)
- ME-FMU exposes derivatives `f(x, u, t)` for the host's solver to integrate.
- Most useful when host is Modelica/Dymola with stiffer integrator than Pulsim.
- Requires AD/Jacobian export (Phase 0 dependency).

### FMU Import (Co-Simulation Master)
- `pulsim fmu-import <foreign.fmu>` parses `modelDescription.xml`, instantiates as a custom signal-domain block.
- Useful for integrating PLECS-exported FMU into a Pulsim study.
- Master orchestration: Gauss-Seidel for now; Jacobi (parallel) optional.

### Variable and Parameter Mapping
- Auto-derive FMU inputs from netlist sources marked `is_fmu_input: true`.
- Auto-derive FMU outputs from `measurement_nodes` and `measurement_signals` declared in netlist.
- Parameters: any YAML field can be marked `is_fmu_parameter: true`.

### Compliance
- Pass FMI Compliance Checker (`fmuCheck` from Modelon) on each generated FMU.
- Cross-test against ANSYS / Modelica importers via CI nightly job.

### YAML Schema
- `fmu_export:` section optional:
```yaml
fmu_export:
  name: pulsim_buck_module
  version: "2.0"
  type: co_simulation
  inputs:
    - { name: vref, source: Vref }
    - { name: load_ohm, parameter: Rload.value }
  outputs:
    - { name: vout, node: vout }
    - { name: i_inductor, branch_current: L1 }
  parameters:
    - { name: kp_ctrl, parameter: PI_ctrl.kp }
```

## Impact

- **New capability**: `fmi-export`.
- **Affected specs**: `fmi-export` (new), `python-bindings` (FMU import block).
- **Affected code**: new `python/pulsim/fmu/` (FMU pack/unpack with `fmilibrary` Python bindings or pure-Python implementation), new C entry point `core/src/fmu_entry.c` packaging Pulsim runtime as the FMU's solver.
- **Performance**: co-simulation step latency ≤Pulsim transient step latency + FMI overhead (typically <50 µs per step on host hardware).

## Success Criteria

1. **FMI 2.0 CS export**: buck-template FMU passes `fmuCheck` and runs in OpenModelica's OMSimulator.
2. **FMI 2.0 CS import**: PLECS-exported FMU imports into Pulsim and exchanges values correctly.
3. **Cross-tool parity**: same buck simulated end-to-end in Pulsim native vs Pulsim-as-FMU-in-OMSimulator within 1% on output voltage.
4. **Documentation**: tutorial showing Pulsim+OpenModelica co-sim of motor drive (Pulsim does converter + control, OpenModelica does mechanical load).
5. **CI**: nightly `fmuCheck` job; Modelica/ANSYS interop tested where licenses available.
