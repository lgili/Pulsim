# Troubleshooting and Tuning

## Table of Contents
- [Convergence Failure Patterns](#convergence-failure-patterns)
- [High-Impact Fixes](#high-impact-fixes)
- [Solver and Integrator Tuning](#solver-and-integrator-tuning)
- [Power-Electronics-Specific Checks](#power-electronics-specific-checks)
- [Answer Template](#answer-template)

## Convergence Failure Patterns

- DC operating point does not converge.
- Newton oscillates or diverges.
- Transient repeatedly rejects timesteps.
- Simulation progresses with unrealistic values.

Typical root causes:

- Floating nodes (especially gates/control nodes).
- Too-aggressive timestep or tolerance settings.
- Discontinuous switching without damping/snubber paths.
- Poor initial condition for strongly nonlinear circuits.

## High-Impact Fixes

- Add high-value bias resistors for floating nodes.
- Use `dc_config.strategy = DCStrategy.Auto` with fallback aids enabled.
- Reduce aggressiveness: tighter `dt_max`, conservative timestep config.
- Enable/retune damping and retry policy (`fallback_policy`, `gmin_fallback`).
- Start from known initial state and verify node/reference polarity.

## Solver and Integrator Tuning

Linear solver stack strategy:

- Small systems: direct solvers early in `order`.
- Large sparse systems: include `gmres`/iterative path with preconditioner.
- Use fallback order and keep `allow_fallback = true` for robustness.

Iterative tuning knobs:

- `max_iterations`, `tolerance`, `restart`
- Preconditioner (`ilu0`, `ilut`, optionally `amg` if available)
- Scaling (`enable_scaling`, `scaling_floor`)

Integrator/timestep knobs:

- `integrator`, `adaptive_timestep`, `timestep_config`
- `lte_config` tolerances and method
- `stiffness_config` for hard switching/stiff dynamics

## Power-Electronics-Specific Checks

- Ensure `dt` resolves switching edges and dead-time events.
- Validate duty-cycle and gate timing consistency.
- Add realistic parasitics/snubbers when ideal switching destabilizes numerics.
- For loss/thermal runs, confirm `enable_losses`, `thermal.enable`, and per-device thermal config.

## Answer Template

Use this structure when diagnosing user issues:

1. Scope: topology, operating point, solver path.
2. Expected behavior: quick physical sanity checks.
3. Findings: root-cause hypotheses with evidence.
4. Parameter changes: smallest high-impact adjustments first.
5. Re-test plan: exact signals and pass/fail criteria.
