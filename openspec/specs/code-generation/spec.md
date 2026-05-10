# code-generation Specification

## Purpose
TBD - created by archiving change add-realtime-code-generation. Update Purpose after archive.
## Requirements
### Requirement: Topology-Indexed Discrete State-Space Codegen
The codegen pipeline SHALL produce C99 code in which `model_step()` selects pre-computed discrete state-space matrices via topology bitmask and performs matrix-vector products without dynamic allocation.

#### Scenario: Generated step for stable topology
- **GIVEN** a buck netlist with two topologies (high-side ON, low-side ON)
- **WHEN** `pulsim codegen --target c99 --step 1e-6` runs
- **THEN** generated `pulsim_model.c` contains a `switch (topology_bitmask)` with two cases
- **AND** each case is a matrix-vector product against `static const` `A_d`, `B_d` matrices

#### Scenario: No allocation in generated code
- **WHEN** generated `model_step()` is compiled and analyzed
- **THEN** no `malloc`, `calloc`, `realloc`, `free`, or VLA appears
- **AND** all state lives on the caller-supplied pointer; matrices are `static const`

### Requirement: Stability Pre-Codegen Check
The codegen pipeline SHALL refuse to generate code when any topology fails the discrete-time stability criterion `|eig(A) · Ts| ≤ 0.5`.

#### Scenario: Step too large
- **GIVEN** a netlist whose fastest pole has time constant 100 ns and `--step 5e-6`
- **WHEN** codegen runs
- **THEN** the pipeline aborts with `unstable_step_size` reason
- **AND** the diagnostic reports the offending topology, eigenvalue, and a recommended `Ts ≤ 250 ns`

#### Scenario: Step within bound
- **GIVEN** the same netlist with `--step 50e-9`
- **WHEN** codegen runs
- **THEN** the pipeline proceeds and stability is recorded as `passed` in summary

### Requirement: Discrete Matrix Computation Accuracy
The pipeline SHALL compute discrete-time matrices `A_d = expm(A·Ts)` and corresponding `B_d` integrals to a verifiable accuracy.

#### Scenario: Round-trip vs native
- **GIVEN** a generated buck model with `Ts = 1 µs`
- **WHEN** the same input trace is fed to native Pulsim and the generated C
- **THEN** the output difference is below `0.1%` peak relative on `vout` and `i_inductor` over 100 ms simulated

### Requirement: Code Size and Cycle Budget Reporting
The pipeline SHALL report ROM size, RAM size, and worst-case cycle count for `model_step()` in a structured summary.

#### Scenario: Default budget reporting
- **WHEN** codegen completes
- **THEN** the summary includes `rom_bytes`, `ram_bytes`, `cycles_estimate_per_step` per target
- **AND** the values are derived from compiled output size and target-specific cycle model

#### Scenario: Budget hard-fail
- **GIVEN** `--max-rom 4096` and a model exceeding 4 KB
- **WHEN** codegen runs
- **THEN** the pipeline aborts with `rom_budget_exceeded`
- **AND** the diagnostic reports actual vs limit

### Requirement: Target Profile Support
The pipeline SHALL support target profiles: `c99` (portable), `arm-cortex-m7`, `zynq-baremetal`, with documented characteristics.

#### Scenario: Cortex-M7 target
- **WHEN** `--target arm-cortex-m7` is specified
- **THEN** generated code uses CMSIS-DSP matrix-vector primitives optionally
- **AND** memory alignment hints (`__attribute__((aligned(32)))`) are emitted

### Requirement: PIL Test Bench Generation
The pipeline SHALL optionally produce a Processor-In-the-Loop test bench that compiles generated C and runs it against native Pulsim on the same input trace.

#### Scenario: PIL bench parity
- **GIVEN** `--pil-bench` flag
- **WHEN** the bench is compiled and executed (via `gcc` or `qemu-arm`)
- **THEN** the generated C output matches Pulsim native within 0.1% on the configured output channels
- **AND** the test integrates with the existing CI as a `codegen-pil` job

### Requirement: Reachable-Topology Reduction
The pipeline SHALL accept an explicit list of reachable topologies or auto-discover via simulation, generating code only for those topologies.

#### Scenario: Auto-discovery
- **GIVEN** `reachable_topologies: auto`
- **WHEN** codegen runs
- **THEN** the pipeline runs a short simulation to identify visited topologies
- **AND** generates code for only those topologies, reporting the count

#### Scenario: Explicit list
- **GIVEN** `reachable_topologies: [0b0001, 0b0010, 0b1010]`
- **WHEN** codegen runs
- **THEN** only the listed topologies appear in generated code
- **AND** any visited topology not in the list at runtime triggers a `topology_unreachable` C-side error code

