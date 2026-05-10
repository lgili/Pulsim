# FMI Export

> Status: shipped — FMI 2.0 Co-Simulation export pipeline. ME export +
> FMU import + FMI 3.0 + cross-tool validation are the natural
> follow-ups.

`pulsim.fmu` packages a Pulsim Circuit as a self-contained
**FMI 2.0 Co-Simulation** `.fmu` file. The exported FMU drops into any
FMI master that speaks 2.0 — OMSimulator, Dymola, MATLAB/Simulink (via
the Co-Simulation block), FMPy, ANSYS Twin Builder. The internal solver
is the discrete state-space machinery shipped in
[`add-realtime-code-generation`](code-generation.md), so the FMU runs
deterministically at the codegen-fixed dt.

## TL;DR

```python
import pulsim

ckt = ...   # your Pulsim Circuit
summary = pulsim.fmu.export(
    ckt,
    dt=1e-6,
    out_path="gen/buck.fmu",
    model_name="buck",
    outputs=["vout"],
)
print(summary.path)         # gen/buck.fmu
print(summary.guid)         # auto-generated UUID
```

The exported file is a valid `.fmu` archive with the standard layout:

```
buck.fmu
├── modelDescription.xml          # FMI 2.0 metadata
├── binaries/
│   └── <platform>/buck.<ext>      # compiled shared library
└── sources/
    ├── model.c                    # codegen output
    ├── model.h
    └── fmu_entry.c                # FMI 2.0 CS callback wrapper
```

## What's inside

`pulsim.fmu.export` orchestrates four steps:

1. **Codegen** the discrete state-space C via
   `pulsim.codegen.generate(circuit, dt, out_dir, target='c99')`.
2. **Emit** an FMI 2.0 CS wrapper (`fmu_entry.c`) implementing the
   minimal callback set:
    - `fmi2GetVersion`, `fmi2GetTypesPlatform`
    - `fmi2Instantiate`, `fmi2FreeInstance`, `fmi2Reset`,
      `fmi2Terminate`
    - `fmi2SetupExperiment`, `fmi2EnterInitializationMode`,
      `fmi2ExitInitializationMode`
    - `fmi2SetReal`, `fmi2GetReal`
    - `fmi2DoStep`, `fmi2CancelStep`
3. **Generate** `modelDescription.xml` with the model's metadata,
   inputs / outputs / states declared with FMI value references.
4. **Compile** `model.c + fmu_entry.c` into a shared library
   (`.so` / `.dylib` / `.dll` per platform) and **zip** everything
   into the `.fmu`.

## Value reference layout

| Range | Variable kind |
|---|---|
| `1, 2, ...` | Inputs (FMI causality `input`) |
| `1000, 1001, ...` | Outputs (FMI causality `output`) |
| `2000, 2001, ...` | Internal state (FMI causality `local`) |

The exporter writes one `<ScalarVariable>` per requested input /
output node plus one `x_i` for each internal state coordinate so
masters can inspect the model's evolution.

## fmi2DoStep semantics

Each call to `fmi2DoStep(c, t_now, h, ...)` advances the FMU's
internal state by `floor(h / dt_internal + 0.5)` substeps of the
codegen-fixed `dt_internal`. For an FMU built with `dt=1e-6` and a
master calling `fmi2DoStep(..., 1e-4, ...)`, each call runs 100
internal substeps.

The FMU advertises `canHandleVariableCommunicationStepSize="true"` —
the master may pick its own `h` per call — but `h` should be a
multiple of `dt_internal` (or close to it) for bit-deterministic
behavior.

## Validation gates

| Gate | Test | Status |
|---|---|---|
| **G.1** Valid FMI 2.0 layout | `test_fmu_export_produces_valid_zip_layout` | Zip contains `modelDescription.xml` + `binaries/<platform>/<lib>` + `sources/` |
| **G.1 partial** Compliance via XML structure | `test_model_description_xml_is_well_formed` | XML parses; `fmiVersion="2.0"`; `<CoSimulation>` block present with required attributes |
| **G.1 partial** Symbol exports | `test_fmu_shared_library_exports_fmi2_symbols` | All 13 FMI 2.0 CS callback symbols load via `dlsym` |
| **G.3** Cross-tool parity | `test_fmu_round_trip_step_via_ctypes` | ctypes-loaded library executes `fmi2Instantiate` + 10 `fmi2DoStep` calls without crashing |

The full `fmuCheck` integration + cross-tool benchmark in OMSimulator
/ Simulink / Dymola is deferred to Phase 7 (cross-tool validation).
The structural + ctypes-roundtrip tests above catch regressions on
the export pipeline itself; tool compatibility lives downstream.

## Limitations / follow-ups

- **Model Exchange export** (Phase 3): would expose
  `fmi2GetDerivatives` / `fmi2GetEventIndicators` /
  `fmi2NewDiscreteStates` / `fmi2EnterEventMode`. Requires the AD
  Jacobian path from `add-automatic-differentiation` — deferred.
- **FMU import** (Phase 4): load a third-party FMU as a Pulsim
  signal-domain block via DLL/dylib loading + Gauss-Seidel master
  loop. Deferred (separate change for the master orchestration).
- **FMI 3.0 support** (Phase 6): `fmi3Float64` types, updated XML
  schema, terminal kinds. Deferred until FMI 3.0 ecosystem support
  is broader.
- **Cross-tool validation** (Phase 7): explicit OMSimulator /
  Simulink / Dymola / Twin Builder runs in CI — license-permitting.
- **Performance optimization** (Phase 8): `fmi2DoStep` overhead ≤
  50 µs per call on buck-size models. Today's wrapper does one
  matrix-vector multiply per substep with no allocation; the basic
  performance is fine. The cycle-budget enforcement that codegen
  Phase 6 covers carries over directly here.
- **YAML `fmu_export:` section** (Phase 5.1): declarative export
  config alongside the simulation block. Lands once the
  Circuit-variant integration's parser dispatch is final.
- **Multi-topology FMU** (per-topology switching matrices): same
  follow-up as the multi-topology codegen tracked under
  `add-realtime-code-generation`.

## See also

- [`code-generation.md`](code-generation.md) — the discrete
  state-space pipeline this module sits on top of.
- [`linear-solver-cache.md`](linear-solver-cache.md) — the per-key
  cache that the multi-topology FMU export will reuse via the
  topology bitmask.
