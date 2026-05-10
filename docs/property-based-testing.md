# Property-Based Testing

> Status: shipped — Hypothesis infrastructure + 5 invariant families
> on randomized circuits. C++ RapidCheck integration + reciprocity
> are the natural follow-ups.

Property-based testing flips the traditional unit-test model: instead
of hand-crafting one input + one expected output, you state an
*invariant* the simulator must always satisfy ("KCL holds at every
node", "the resistor dissipates non-negative power") and let
Hypothesis generate hundreds of randomized circuits to challenge it.
When something breaks, Hypothesis automatically *shrinks* the failing
input to a minimal counter-example.

`add-property-based-testing` Phase 1+ ships the harness, randomized
circuit strategies, and four invariant families:

| Invariant | Test file | Pinned for |
|---|---|---|
| KCL at DC + after transient | `properties/test_kcl.py` | Voltage divider; midpoint current balance |
| Energy / steady-state | `properties/test_energy.py` | RC charging V_C → V_src; monotone |
| Passivity (resistor) | `properties/test_passivity.py` | `P_R ≥ 0` per step |
| PWL switching invariants | `properties/test_pwl_invariants.py` | Cache hit rate ≥ 92 %; no DAE fallback for linear PWL |

## TL;DR

```python
from hypothesis import given, settings
from python.tests.properties.strategies import gen_passive_rc, make_quick_options
import pulsim, math

@given(gen_passive_rc())
@settings(max_examples=25, deadline=None)
def test_my_invariant(generated):
    sim = pulsim.Simulator(generated.circuit, make_quick_options())
    dc  = sim.dc_operating_point()
    run = sim.run_transient(dc.newton_result.solution)
    # ... assert invariants hold ...
```

Hypothesis runs `25` randomized examples per `@given`, each one a
full circuit-+-transient. On failure, it auto-shrinks to the simplest
parameter combination that still triggers the assertion.

## Strategies

`properties/strategies.py` ships three randomized-circuit factories:

| Factory | What it generates |
|---|---|
| `gen_passive_rc()` | RC low-pass with `R ∈ [1, 1e6]` Ω, `C ∈ [1pF, 1mF]`, `V ∈ [0.1, 100]` V |
| `gen_passive_rlc()` | RLC with R/L/C in physical ranges |
| `gen_resistor_divider()` | Two-resistor divider for KCL pinning |

Each factory returns a `GeneratedCircuit { circuit, parameters,
description }` triple — `parameters` is the dict the test reads to
re-derive analytical references.

Range bounds are picked so the simulator doesn't pathologize on
extreme values (1e-15 capacitors trigger numerical underflow that's
unrelated to the invariant being tested). When you want to push the
limits, swap the strategy or pass tighter `min_value/max_value` into
`Distribution.uniform`.

## Reproducibility (gate G.3)

Every Hypothesis test has a deterministic seed derived from the
example index. Failures print the **exact seed** that produced them,
plus the **shrunken example** in YAML form. CI runs log every seed so
a flaky-but-real bug can be re-derived bit-for-bit.

```
Falsifying example: test_kcl_holds_at_dc_op(
    generated=GeneratedCircuit(parameters={'R1': 1.0, 'R2': 1e-6, 'V': 50.0}),
)
```

When a property test catches a real bug, copy the shrunken parameters
into a regression test under `tests/properties/regressions/` so the
ratchet sticks (Phase 7.4 contract).

## Validation summary

| Gate | Result |
|---|---|
| **G.1** 7 invariants — partial: 5 invariant families ship today (KCL DC, KCL transient, energy, monotonicity, passivity, PWL cache, PWL no-fallback). KVL via incidence-matrix, Tellegen full-form, and reciprocity follow alongside the C++ RapidCheck integration |
| **G.2** Latent-bug discoveries — the property suite is in CI from day 1; bugs found and pinned ride into `regressions/` |
| **G.3** Determinism — Hypothesis seeds are deterministic + logged |
| **G.4** Regression corpus active in CI — `tests/properties/` runs in the standard `pytest` invocation |
| **G.5** ≤ 30 s suite runtime — current 7-test suite runs in < 1 s on M-series |

## Limitations / follow-ups

- **C++ RapidCheck integration** (Phase 8): same property checks at
  the MNA / linear-solver level, independent of Python. Catches bugs
  one layer deeper. Tracked alongside the GoogleTest-vs-Catch2
  conversation in `refactor-modular-build-split`.
- **Reciprocity invariant** (Phase 5): linear AC admittance matrix
  must be symmetric. Pairs with `add-frequency-domain-analysis`'s
  AC sweep machinery.
- **KVL via incidence matrix** (Phase 2.2): the explicit branch-
  voltage formulation. Today we exercise KCL via the divider; the
  full incidence-matrix extraction sits with the topology-bitmask
  exposure work.
- **Energy conservation full-form** (Phase 3.3): currently we pin the
  steady-state corollary; the full per-step
  `stored + ∫R·I² dt − ∫P_src dt ≈ 0` integral requires
  per-element power telemetry which the loss-summary surface partly
  provides.
- **Periodicity** (Phase 4.3): `x(t+T) ≈ x(t)` after the periodic
  shooting solver runs. Pairs with the periodic-steady-state
  validation tests.
- **Auto-shrunk YAML emission** (Phase 9.1): on failure, dump the
  shrunken circuit to a YAML file alongside the Hypothesis report.

## See also

- [`parameter-sweep.md`](parameter-sweep.md) — sweep covers the
  *design-space exploration* axis; property tests cover the
  *correctness invariants* axis. They're complementary.
- [`code-generation.md`](code-generation.md) — the same invariants
  translate into PIL-side property tests (run the generated C, check
  KCL on the recovered states).
