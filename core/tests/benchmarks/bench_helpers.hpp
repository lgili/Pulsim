#pragma once

// =============================================================================
// Pulsim — benchmark + analytical-validation helpers
// =============================================================================
//
// Provides:
//   * ``Stopwatch`` — RAII timer with steady_clock.
//   * ``BenchReport`` — POD aggregate (cache_build, per-step, total,
//     max abs/rel error vs analytical).
//   * ``time_and_validate`` — drives one transient simulation, times
//     it, and (if a closed-form is supplied) measures error against
//     it.
//   * ``print_report_row`` — pretty-prints a one-liner row for the
//     bench summary table.
//
// Each bench test in this directory is a thin Catch2 wrapper that
// builds a ``CircuitBuilder``, defines the analytical reference
// (or returns ``{}`` to skip), calls ``time_and_validate(...)``,
// and ``REQUIRE``s that the error is within a tolerance.

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/options.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::bench {

/// Wall-clock timer. ``elapsed_ms()`` returns milliseconds since
/// construction.
class Stopwatch {
public:
    Stopwatch() : t0_(std::chrono::steady_clock::now()) {}

    [[nodiscard]] double elapsed_ms() const {
        const auto t1 = std::chrono::steady_clock::now();
        using ms = std::chrono::duration<double, std::milli>;
        return std::chrono::duration_cast<ms>(t1 - t0_).count();
    }

private:
    std::chrono::steady_clock::time_point t0_;
};

/// One row of the bench summary table.
struct BenchReport {
    std::string  name;
    int          num_switches  = 0;   ///< topology size, controls cache cost
    int          num_steps     = 0;
    int          state_size    = 0;
    double       cache_build_ms = 0.0;
    double       run_total_ms   = 0.0;
    double       per_step_us    = 0.0;
    double       max_abs_error  = std::numeric_limits<double>::quiet_NaN();
    double       max_rel_error  = std::numeric_limits<double>::quiet_NaN();
    bool         analytical_checked = false;
};

/// Maximum absolute error between two equal-length samples and
/// the corresponding maximum relative error (denominator floored
/// at 1e-9 to avoid div-by-zero on samples that legitimately
/// pass through zero).
inline std::pair<double, double> max_errors(
    const std::vector<double>& measured,
    const std::vector<double>& analytical) {
    double max_abs = 0.0;
    double max_rel = 0.0;
    const std::size_t n = std::min(measured.size(), analytical.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double err = std::abs(measured[i] - analytical[i]);
        max_abs = std::max(max_abs, err);
        const double denom = std::max(std::abs(analytical[i]), 1e-9);
        max_rel = std::max(max_rel, err / denom);
    }
    return {max_abs, max_rel};
}

/// Drive a simulation through the explicit cache + run_transient
/// path so we can time the two phases separately.
///
/// ``probe_state_idx`` is the state-vector index whose trajectory
/// we compare against ``analytical_fn`` (typically a node voltage).
/// Pass ``analytical_fn = {}`` to skip the validation step (pure
/// timing).
///
/// ``switch_fn`` / ``b_extra_fn`` are forwarded to ``run_transient``
/// so PWM / time-varying excitations work. If ``switch_fn`` is left
/// empty, we synthesize a closed-all-switches mask (correct for
/// static / passive circuits).
inline BenchReport time_and_validate(
    std::string                                   name,
    builder::CircuitBuilder&                       cb,
    const solver::SimulationOptions&               opts,
    int                                           probe_state_idx,
    std::function<double(double)>                 analytical_fn = {},
    solver::SwitchScheduleFn                       switch_fn      = {},
    solver::BExtraFn                               b_extra_fn     = {}) {

    BenchReport rpt;
    rpt.name = std::move(name);
    rpt.num_switches = static_cast<int>(cb.graph().num_switches());
    rpt.state_size   = static_cast<int>(cb.pool().state_size(cb.graph()));

    // ---- Phase 1: build the PWL cache -----------------------------------
    pwl::PwlStateSpaceCache cache(cb.graph(), cb.pool());
    {
        Stopwatch sw;
        cache.build(opts.dt);
        rpt.cache_build_ms = sw.elapsed_ms();
    }

    // Default switch_fn: all switches closed (correct for circuits with
    // no controlled switches, e.g. passive RC / RLC / diode-only paths).
    if (!switch_fn) {
        const auto n_sw = cb.graph().num_switches();
        topology::SwitchStateMask m(n_sw);
        for (Size i = 0; i < n_sw; ++i) m.set(i, true);
        switch_fn = [m](Real) { return m; };
    }

    // ---- Phase 2: drive the transient loop -------------------------------
    solver::SimulationResult sim;
    {
        Stopwatch sw;
        sim = solver::run_transient(
            cache, cb.graph(), cb.pool(), opts,
            switch_fn, b_extra_fn);
        rpt.run_total_ms = sw.elapsed_ms();
    }
    rpt.num_steps   = static_cast<int>(sim.times.size());
    rpt.per_step_us = rpt.num_steps > 0
        ? (rpt.run_total_ms * 1000.0) / rpt.num_steps
        : 0.0;

    // ---- Phase 3: error vs analytical -----------------------------------
    if (analytical_fn && probe_state_idx >= 0
            && probe_state_idx < rpt.state_size) {
        std::vector<double> measured;
        std::vector<double> analytical;
        measured.reserve(sim.times.size());
        analytical.reserve(sim.times.size());
        for (std::size_t i = 0; i < sim.times.size(); ++i) {
            measured.push_back(static_cast<double>(
                sim.states[i][probe_state_idx]));
            analytical.push_back(analytical_fn(
                static_cast<double>(sim.times[i])));
        }
        const auto [a, r] = max_errors(measured, analytical);
        rpt.max_abs_error      = a;
        rpt.max_rel_error      = r;
        rpt.analytical_checked = true;
    }
    return rpt;
}

/// Format and print a single-line summary row. Print the header
/// once with ``print_report_header()`` before the first row.
inline void print_report_row(const BenchReport& r) {
    std::printf(" %-36s | %3d sw | %6d steps | "
                "build %7.2f ms | run %8.2f ms | "
                "step %7.2f us",
                r.name.c_str(),
                r.num_switches,
                r.num_steps,
                r.cache_build_ms,
                r.run_total_ms,
                r.per_step_us);
    if (r.analytical_checked) {
        std::printf(" | err_abs %.3e | err_rel %.3e\n",
                    r.max_abs_error, r.max_rel_error);
    } else {
        std::printf("\n");
    }
}

inline void print_report_header() {
    std::printf("\n");
    std::printf(" %-36s | %8s | %12s | %15s | %15s | %14s | %s\n",
                "case", "switches", "steps", "cache build",
                "run total", "per-step", "max abs/rel error (vs analytical)");
    std::printf(" %s\n",
                "----------------------------------------------------"
                "----------------------------------------------------"
                "----------------------------");
}

}  // namespace pulsim::bench
