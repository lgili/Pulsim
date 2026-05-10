"""Linear-solver cache hit-rate telemetry.

`refactor-linear-solver-cache` Phase 6 ships a per-key LRU keyed on
matrix_hash so repeat factorizations of the same (sparsity, topology)
collapse to a single key. This script demonstrates the cache by running a
PWL switching converter and reading the hit / miss counters off
``BackendTelemetry``.

Expected outcome: a buck converter switching at 100 kHz over 1 ms (100
periods, 200 commutations) hits a small set of distinct topology states
(2 in the simple case: switch closed / switch open). After warm-up, the
cache hit rate climbs above 99 percent.

Run::

    python 10_linear_solver_cache.py

See also: docs/linear-solver-cache.md
"""

from __future__ import annotations

import pulsim


def main() -> None:
    exp = pulsim.templates.buck(
        Vin=24.0, Vout=5.0, Iout=2.0, fsw=100_000.0,
    )
    exp.circuit.set_switching_mode_for_all(pulsim.SwitchingMode.Ideal)
    exp.circuit.set_pwl_state("Q1", False)
    exp.circuit.set_pwl_state("D1", False)

    opts = pulsim.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = 5e-4                                 # 50 PWM periods
    opts.dt = 1e-7
    opts.dt_min = 1e-12
    opts.dt_max = 1e-7
    opts.adaptive_timestep = False
    opts.integrator = pulsim.Integrator.BDF1
    opts.switching_mode = pulsim.SwitchingMode.Ideal
    opts.newton_options.num_nodes = exp.circuit.num_nodes()
    opts.newton_options.num_branches = exp.circuit.num_branches()

    sim = pulsim.Simulator(exp.circuit, opts)
    dc = sim.dc_operating_point()
    if not dc.success:
        raise SystemExit(f"DC OP failed: {dc.message}")
    result = sim.run_transient(dc.newton_result.solution)
    if not result.success:
        raise SystemExit(f"transient failed: {result.message}")

    bt = result.backend_telemetry

    # Numeric (per-key) factor cache.
    hits = bt.linear_factor_cache_hits
    miss = bt.linear_factor_cache_misses
    inval = bt.linear_factor_cache_invalidations
    total = hits + miss
    hit_rate = (hits / total * 100.0) if total > 0 else 0.0

    # PWL segment-model cache (coarser — caches the linearization, not
    # just the factorization).
    seg_hits = bt.segment_model_cache_hits
    seg_miss = bt.segment_model_cache_misses
    seg_total = seg_hits + seg_miss
    seg_hit_rate = (seg_hits / seg_total * 100.0) if seg_total > 0 else 0.0

    # Linear-solver wallclock.
    lst = result.linear_solver_telemetry
    avg_factor = (lst.total_factorize_time_seconds / max(lst.total_factorize_calls, 1) * 1e6)
    avg_solve  = (lst.total_solve_time_seconds  / max(lst.total_solve_calls,  1) * 1e6)

    print("Linear factor cache (per-key LRU keyed on matrix_hash):")
    print(f"  hits:           {hits}")
    print(f"  misses:         {miss}")
    print(f"  invalidations:  {inval}    (last reason: "
          f"{bt.linear_factor_cache_last_invalidation_reason!r})")
    print(f"  hit rate:       {hit_rate:.2f} %    "
          f"(target ≥ 90 % once warm)")
    print()
    print("PWL segment-model cache (caches A_red / B_red linearization):")
    print(f"  hits:    {seg_hits}    misses: {seg_miss}    "
          f"hit rate: {seg_hit_rate:.2f} %")
    print()
    print("Linear solver throughput:")
    print(f"  total factorize calls: {lst.total_factorize_calls}    "
          f"avg {avg_factor:.2f} µs/call")
    print(f"  total solve calls:     {lst.total_solve_calls}    "
          f"avg {avg_solve:.2f} µs/call")
    print(f"  fallbacks (KLU → SparseLU → GMRES): {lst.total_fallbacks}")
    print()
    print(f"PWL switching activity:")
    print(f"  topology transitions:    {bt.pwl_topology_transitions}")
    print(f"  commutations within step: {bt.pwl_event_commutations}")
    print(f"  state-space-primary steps: {bt.state_space_primary_steps}    "
          f"(of total {result.total_steps})")
    print(f"  non-admissible-fallback steps: {bt.segment_non_admissible_steps}")


if __name__ == "__main__":
    main()
