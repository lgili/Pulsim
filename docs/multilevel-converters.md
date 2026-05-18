# Multilevel Converters

> **Status: MVP.** Single-arm MMC builder shipped. 3φ + upper/lower-arm
> variant + PLECS/PSIM golden-CSV benchmarks are the next phase.

Multilevel converters (NPC, T-type, flying-cap, MMC) are the
dominant topologies for grid-scale STATCOMs, HVDC links, modular
motor drives, and high-power inverters. They're also notoriously
hard to simulate from cold start because they combine:

- **Hundreds of switches** that commute synchronously on PWM edges
- **Floating-cap networks** producing ill-conditioned MNA matrices
- **Multi-second startup transients** with stiff cap-balancing
  controllers

Pulsim ships four convergence aids that target exactly these failure
modes — see [Numerical Configuration](numerical-configuration.md) for
the full menu. This page covers the **topology builders** that make
multilevel circuits easy to construct, plus the convergence-validation
gates they pass.

## The MMC arm builder

```cpp
#include "pulsim/v1/templates/mmc.hpp"
using namespace pulsim::v1;

auto [ckt, h] = templates::mmc_arm(templates::MmcArmParams{
    .num_submodules = 9,        // chain of 9 half-bridge submodules
    .V_dc           = 900.0,    // 100 V per submodule nominal
    .L_arm          = 1e-3,
    .C_submodule    = 2e-3,
    .name_prefix    = "armA",
});

// Wire DC + gate signals + load externally.
ckt.add_voltage_source("V_dc", h.v_top, h.v_bot, 900.0);
for (int k = 0; k < 9; ++k) {
    PulseParams gate{};
    gate.v_initial = 0.0;
    gate.v_pulse   = 5.0;
    gate.t_delay   = 1e-4 * k;        // staggered gates
    gate.t_width   = 1e-3;
    ckt.add_pulse_voltage_source(
        "Gate_" + std::to_string(k),
        h.gate_nodes[k], Circuit::ground(), gate);
}
```

The returned `MmcArmHandles` exposes everything the user needs to wire
up:

| Field | Purpose |
|---|---|
| `v_top`, `v_bot` | DC link rails |
| `mid_nodes[N]` | Per-submodule output / midpoint (for probing or chaining) |
| `gate_nodes[N]` | Per-submodule gate (user drives these) |
| `cap_top_nodes[N]` | Above the floating cap (probe cap voltage as `V(cap_top[k]) - V(mid[k])`) |

## The HBSM topology

Each submodule is a **Half-Bridge SubModule** (HBSM):

```
   v_in ──┬── S_high ── cap_top ── C_sm ── v_out
          └── S_low ─────────────────────── v_out
```

- **S_high ON, S_low OFF**: cap is INSERTED in the arm current path
  (`V_in − V_out = V_cap`)
- **S_high OFF, S_low ON**: cap is BYPASSED (`V_in − V_out ≈ 0`)

The submodule output `v_out` becomes the input of the next submodule,
forming a chain of N HBSMs in series with the arm inductor.

## Convergence on multilevel — what runs under the hood

When you simulate a 9-submodule MMC arm with `Preset::Robust`, Pulsim
silently engages **four** convergence aids that earlier releases
didn't have. The current MMC test in
`core/tests/test_mmc_arm_template.cpp` validates each of them on a
4-submodule arm; full 9-submodule MMC at 100 µs steps passes
identically.

### 1. Armijo line search

8 switches + 4 floating caps creates a Newton state space where the
full-length step often overshoots the descent direction. The Armijo
condition catches this and halves the step until the residual
decreases. **Without Armijo, the cold-start often diverges**; with
it, Newton converges in ≤ 20 iterations on the first timestep.

### 2. Simultaneous event coalescence

The MMC test ships with a synchronous-gate edge — all 4 submodules
commute at the same PWM rising edge. **Without coalescence**, the
simulator processes them one at a time across 4 consecutive steps,
which on a 9-submodule arm becomes 9 steps and often loses
convergence. **With coalescence**, all gates fire atomically in a
single Newton solve at the bisected event time.

Confirmed via telemetry:

```python
result.backend_telemetry.simultaneous_event_groups  # ≥ 1 per gate edge
result.backend_telemetry.pwl_event_commutations     # ≥ N per gate edge
```

### 3. Iterative refinement on KLU

Floating-cap networks produce **ill-conditioned MNA submatrices**
(cap-to-cap loops). KLU's partial-pivoting back-substitution
accumulates round-off; iterative refinement recovers the lost
precision in one cheap re-solve using the existing factorization.
Triggers automatically when the post-solve residual exceeds
`10·ε_machine`.

Confirmed via telemetry:

```python
result.linear_solver_telemetry.linear_refinement_steps  # > 0 on flying-cap / MMC
```

### 4. Homotopy continuation (DC OP)

Cold-start MMCs often hang at the DC operating point because the
nonlinear residual landscape has too many local minima. When
`DCStrategy::Auto` exhausts Direct → Source → Gmin →
PseudoTransient, the orchestrator falls back to **homotopy
continuation**: λ-steps from 0 (linear MNA, all FETs in g_off) to 1
(full nonlinear model) in 5 increments, warm-starting Newton at each
step.

Confirmed via telemetry:

```python
result.dc_result.strategy_used == DCStrategy.Homotopy
result.dc_result.homotopy_ladder_completed == True
```

## What's NOT yet shipped (Phase 13 roadmap)

The MVP MMC arm is single-arm — the canonical 3φ + upper/lower-arm
MMC inverter (where the AC output is at the midpoint between each
phase's upper and lower arm) is a follow-up. Plus:

- **3-level NPC topology** template (with paired clamp diodes)
- **5-level flying-cap** template
- **T-type 3-level** template
- **PLECS / PSIM golden-CSV benchmarks** for each topology gating on
  ≤ 0.5% RMS error vs the external reference

These ship in Phase 13 of the `simplify-and-harden-numerical-surface`
OpenSpec change. The numerical machinery (Armijo + coalescence +
iterative refinement + homotopy) is ALREADY in place — the
benchmarks just need the topology builders + the external golden
CSVs.

If you have an existing PLECS or PSIM netlist for one of these
topologies you'd like to validate Pulsim against, see
[Benchmarks and Parity](benchmarks-and-parity.md) for the
contribution protocol.

## End-to-end example (C++)

```cpp
#include "pulsim/v1/core.hpp"
#include "pulsim/v1/templates/mmc.hpp"

using namespace pulsim::v1;

int main() {
    // 4-submodule arm, V_dc = 400 V (100 V per submodule nominal).
    auto [ckt, h] = templates::mmc_arm(templates::MmcArmParams{
        .num_submodules = 4,
        .V_dc           = 400.0,
        .V_cap_init     = 100.0,
        .L_arm          = 1e-3,
        .C_submodule    = 2e-3,
    });

    // DC supply.
    ckt.add_voltage_source("V_dc", h.v_top, h.v_bot, 400.0);
    ckt.add_resistor("R_gnd", h.v_bot, Circuit::ground(), 1e-3);

    // Shared PWM gate for all 4 submodules (synchronous commutation).
    PulseParams gate{};
    gate.v_initial = 0.0;
    gate.v_pulse   = 5.0;
    gate.t_delay   = 100e-6;
    gate.t_rise    = 1e-9;
    gate.t_fall    = 1e-9;
    gate.t_width   = 1e-3;
    ckt.add_pulse_voltage_source("PWM", h.gate_nodes[0],
                                   Circuit::ground(), gate);
    // Tie the other 3 gates to the same control signal.
    for (int k = 1; k < 4; ++k) {
        ckt.add_voltage_source(
            "Tie_" + std::to_string(k),
            h.gate_nodes[k], h.gate_nodes[0], 0.0);
    }

    // Numerical configuration: Robust preset gives us all four
    // convergence aids.
    auto opts = SimulationOptions::from_preset(Preset::Robust,
                                                 /*dt=*/1e-5,
                                                 /*tstop=*/5e-4);
    opts.switching_mode = SwitchingMode::Ideal;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    auto result = sim.run_transient();

    if (result.success) {
        // Check the convergence-aid telemetry.
        std::cout << "PWM coalesced groups: "
                  << result.backend_telemetry.simultaneous_event_groups
                  << "\n";
        std::cout << "KLU refinement steps: "
                  << result.linear_solver_telemetry
                        .linear_refinement_steps
                  << "\n";
    }
    return 0;
}
```

## See also

- [Numerical Configuration](numerical-configuration.md) — the
  `Preset` enum and convergence-aid reference
- [Convergence Tuning Guide](convergence-tuning-guide.md) — what to
  do when a circuit doesn't converge
- [Components Reference](components-reference.md) — `vcswitch`,
  `capacitor`, `inductor` (the primitives the MMC builder uses)
- [Motor Models](motor-models.md) — multilevel inverters often
  drive motor loads
