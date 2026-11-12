// =============================================================================
// Grid-exact PWM edges and the `SwitchSchedule` edge oracle
// =============================================================================
//
// Before: every built-in PWM generator compared
// `fmod(t / T, 1) < duty` in floating point. Evaluated on the
// fixed-step grid t = k * dt, an edge that sits exactly on a grid
// point flipped a step early or late at random:
//
//   make_pwm_switch_fn, 100 kHz, D = 0.5,  dt = 1e-7:  49/50/51 on-steps
//   make_pwm_switch_fn, 100 kHz, D = 0.25, dt = 1e-7:  24/25/26
//   make_dead_time_pwm_pair_fn, D = 0.5, 2e-7 dead:   HS 48/49 (want 48)
//
// After: an edge within a few ulp of a grid point takes effect at
// that point, so an on-time of N whole steps is N steps in EVERY
// period, and `next_edge_after(t)` names each edge exactly.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/models/pulse_voltage_source.hpp"
#include "pulsim/models/pwm_voltage_source.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/combined_switch_fn.hpp"
#include "pulsim/sources/dead_time_pwm_pair_fn.hpp"
#include "pulsim/sources/native_pwm_switch_fn.hpp"
#include "pulsim/sources/phase_shift_full_bridge_fn.hpp"
#include "pulsim/sources/pwm_switch_fn.hpp"
#include "pulsim/sources/spwm_pair_fn.hpp"
#include "pulsim/sources/switch_schedule.hpp"
#include "pulsim/sources/three_phase_spwm_fn.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <set>
#include <vector>

using namespace pulsim;
using namespace pulsim::sources;
using topology::SwitchStateMask;

namespace {

struct GridCase {
    Real f;
    Real duty;
    Real dt;
};

// (f, D, dt) with D * T / dt and T / dt whole numbers.
const std::vector<GridCase> kCases = {
    {100e3, 0.50, 1e-7},  {100e3, 0.25, 1e-7},  {100e3, 0.50, 2e-7},
    {50e3, 0.30, 1e-7},   {200e3, 0.37, 5e-8},  {20e3, 0.60, 2.5e-7},
    {100e3, 0.13, 1e-8},  {150e3, 0.40, 1e-7 / 1.5},
};

// Number of grid points k*dt + t0 in period j at which `bit` is set.
template <class Fn>
std::set<long> on_counts(const Fn& fn, Size bit, Real f, Real dt,
                         Real t0, int periods) {
    const long n = std::lround(Real{1} / (f * dt));
    std::set<long> counts;
    for (int j = 0; j < periods; ++j) {
        long c = 0;
        for (long k = j * n; k < (j + 1) * n; ++k) {
            if (fn(t0 + static_cast<Real>(k) * dt).get(bit)) ++c;
        }
        counts.insert(c);
    }
    return counts;
}

}  // namespace

TEST_CASE("make_pwm_switch_fn — whole-step on-time is exact in every period",
          "[v2][pwm_grid][unit]") {
    for (const auto& c : kCases) {
        const long want = std::lround(c.duty / (c.f * c.dt));
        auto sw = make_pwm_switch_fn(c.f, c.duty, 0, 1);
        INFO("f=" << c.f << " D=" << c.duty << " dt=" << c.dt);
        // From t = 0 and from a resumed-run offset (t_start + k dt).
        REQUIRE(on_counts(sw, 0, c.f, c.dt, 0.0, 300)
                == std::set<long>{want});
        const Real t0 = 2000 * (Real{1} / c.f);
        REQUIRE(on_counts(sw, 0, c.f, c.dt, t0, 50)
                == std::set<long>{want});
    }
}

TEST_CASE("make_pwm_switch_fn — a phase that is a whole number of steps",
          "[v2][pwm_grid][unit]") {
    // phase = 37 steps: still exactly 25 on-steps per period.
    const Real dt = 1e-7;
    auto sw = make_pwm_switch_fn(100e3, 0.25, 0, 1, 37 * dt);
    REQUIRE(on_counts(sw, 0, 100e3, dt, 0.0, 300) == std::set<long>{25});
}

TEST_CASE("make_dead_time_pwm_pair_fn — whole-step on-times are exact",
          "[v2][pwm_grid][unit]") {
    struct C { Real f, duty, dt, td; };
    for (const C c : {C{100e3, 0.5, 1e-7, 2e-7}, C{100e3, 0.25, 1e-7, 3e-7},
                      C{50e3, 0.3, 1e-7, 1e-7}}) {
        const Real T = Real{1} / c.f;
        auto sw = make_dead_time_pwm_pair_fn(c.f, c.duty, 0, 1, 2, c.td);
        const long hs = std::lround((c.duty * T - c.td) / c.dt);
        const long ls = std::lround(((1 - c.duty) * T - c.td) / c.dt);
        INFO("D=" << c.duty << " td=" << c.td);
        REQUIRE(on_counts(sw, 0, c.f, c.dt, 0.0, 300)
                == std::set<long>{hs});
        REQUIRE(on_counts(sw, 1, c.f, c.dt, 0.0, 300)
                == std::set<long>{ls});
    }
}

TEST_CASE("NativePwm2Switch and PWMVoltageSource — whole-step on-time exact",
          "[v2][pwm_grid][unit]") {
    for (const auto& c : kCases) {
        const long want = std::lround(c.duty / (c.f * c.dt));
        INFO("f=" << c.f << " D=" << c.duty << " dt=" << c.dt);
        NativePwm2Switch native{Real{1} / c.f, c.duty, 2};
        REQUIRE(on_counts(native, 0, c.f, c.dt, 0.0, 200)
                == std::set<long>{want});
        models::PWMVoltageSource::Params prm;
        prm.v_high = 1.0;
        prm.v_low = 0.0;
        prm.frequency = c.f;
        prm.duty = c.duty;
        auto src = [&prm](Real t) {
            SwitchStateMask m{1};
            m.set(0, models::PWMVoltageSource::value_at(prm, t) > 0.5);
            return m;
        };
        REQUIRE(on_counts(src, 0, c.f, c.dt, 0.0, 200)
                == std::set<long>{want});
    }
}

TEST_CASE("PulseVoltageSource — periodic pulse width exact on the grid",
          "[v2][pwm_grid][unit]") {
    // Before (fmod of the elapsed time): period 20 us, width 6 us,
    // dt 0.2 us read 29 or 30 grid points at v_pulsed per period.
    struct C { Real period, width, dt, t_start, rise; };
    for (const C c : {C{20e-6, 6e-6, 2e-7, 0.0, 0.0},
                      C{10e-6, 2.5e-6, 1e-7, 0.0, 0.0},
                      C{20e-6, 6e-6, 2e-7, 13 * 2e-7, 0.0},
                      C{20e-6, 6e-6, 2e-7, 0.0, 4e-7}}) {
        models::PulseVoltageSource::Params prm;
        prm.v_pulsed = 1.0;
        prm.pulse_width = c.width;
        prm.period = c.period;
        prm.t_start = c.t_start;
        prm.rise_time = c.rise;
        auto src = [&prm](Real t) {
            SwitchStateMask m{1};
            // Strictly above the ramp: only the flat top counts.
            m.set(0, models::PulseVoltageSource::value_at(prm, t)
                         > 1.0 - 1e-9);
            return m;
        };
        const long want = std::lround(c.width / c.dt);
        INFO("period=" << c.period << " width=" << c.width
             << " t_start=" << c.t_start << " rise=" << c.rise);
        REQUIRE(on_counts(src, 0, 1 / c.period, c.dt, c.t_start, 300)
                == std::set<long>{want});
    }
}

TEST_CASE("make_phase_shift_full_bridge_fn — grid-exact legs",
          "[v2][pwm_grid][unit]") {
    // 100 kHz, quarter shift (2.5 us lag = 25 steps), 2e-7 dead time.
    const Real dt = 1e-7;
    auto sw = make_phase_shift_full_bridge_fn(
        100e3, 0.5 * std::numbers::pi_v<Real>, 0, 1, 2, 3, 4, 2e-7);
    for (Size bit : {Size{0}, Size{1}, Size{2}, Size{3}}) {
        INFO("bit " << bit);
        REQUIRE(on_counts(sw, bit, 100e3, dt, 0.0, 200)
                == std::set<long>{48});
    }
}

TEST_CASE("next_edge_after — dead-time pair names its four edges exactly",
          "[v2][pwm_grid][switch_schedule][unit]") {
    const Real f = 100e3, T = 1 / f, D = 0.25, td = 1e-7;
    auto sw = make_dead_time_pwm_pair_fn(f, D, 0, 1, 2, td);
    REQUIRE(sw.has_edges());
    std::vector<Real> want;
    for (int k = 0; k < 50; ++k) {
        for (Real o : {D * T - td, D * T, T - td, T}) {
            want.push_back(k * T + o);
        }
    }
    Real t = 0;
    for (Real w : want) {
        const Real e = sw.next_edge_after(t);
        REQUIRE(e > t);
        REQUIRE(e == Catch::Approx(w).epsilon(1e-12));
        // A real edge: the mask AT the edge differs from just before.
        REQUIRE_FALSE(sw(e) == sw(0.5 * (t + e)));
        t = e;
    }
}

TEST_CASE("next_edge_after — single PWM: edges at k T and k T + D T",
          "[v2][pwm_grid][switch_schedule][unit]") {
    auto sw = make_pwm_switch_fn(100e3, 0.3, 0, 1, 1e-6);
    Real t = 0;
    std::vector<Real> got;
    for (int i = 0; i < 6; ++i) {
        t = sw.next_edge_after(t);
        got.push_back(t);
    }
    // phase = 1 us: on at 1, 11, 21 us; off at 4, 14, 24 us.
    const std::vector<Real> want{1e-6, 4e-6, 11e-6, 14e-6, 21e-6, 24e-6};
    for (std::size_t i = 0; i < want.size(); ++i) {
        REQUIRE(got[i] == Catch::Approx(want[i]).epsilon(1e-12));
    }
    REQUIRE(sw(got[1]).get(0) == false);   // at the off edge: OFF
    REQUIRE(sw(got[2]).get(0) == true);    // at the on edge: ON
}

namespace {

// Walk the edges of `sw` over [0, t_end]: every returned edge must be a
// mask change, and the mask must be constant between consecutive edges
// (sampled on `n_probe` points per interval).
template <class Sched>
void check_edges_complete(const Sched& sw, Real t_end, int n_probe,
                          std::size_t min_edges) {
    Real t = 0;
    std::size_t n_edges = 0;
    while (true) {
        const Real e = sw.next_edge_after(t);
        REQUIRE(e > t);
        if (e > t_end) break;
        const auto inside = sw(0.5 * (t + e));
        for (int i = 1; i < n_probe; ++i) {
            const Real tau = t + (e - t) * i / n_probe;
            REQUIRE(sw(tau) == inside);
        }
        t = e;
        ++n_edges;
    }
    REQUIRE(n_edges >= min_edges);
}

}  // namespace

TEST_CASE("next_edge_after — SPWM pair edges are complete and exact",
          "[v2][pwm_grid][switch_schedule][unit]") {
    // 10 kHz carrier, 50 Hz, M = 0.9, 1 us dead time: 4 edges per
    // carrier period.
    auto sw = make_spwm_pair_fn(10e3, 50.0, 0.9, 0, 1, 2, 1e-6);
    check_edges_complete(sw, 20e-3, 64, 4 * 200 - 4);
}

TEST_CASE("next_edge_after — three-phase SPWM edges are complete",
          "[v2][pwm_grid][switch_schedule][unit]") {
    ThreePhaseLegIndices legs{0, 1, 2, 3, 4, 5};
    auto sw = make_three_phase_spwm_fn(5e3, 50.0, 0.8, legs, 6, 2e-6);
    // Distinct edge times per carrier period: the period start (every
    // HS on) and 1 - d (every LS off) are shared by the three legs,
    // plus each leg's own HS off and LS on: 2 + 3 * 2 = 8.
    check_edges_complete(sw, 20e-3, 32, 8 * 100 - 8);
}

TEST_CASE("make_combined_switch_fn — edge oracle is the earliest part",
          "[v2][pwm_grid][switch_schedule][unit]") {
    auto a = make_pwm_switch_fn(100e3, 0.5, 0, 2);
    auto b = make_pwm_switch_fn(30e3, 0.2, 1, 2);
    auto combo = make_combined_switch_fn(2, {a, b});
    REQUIRE(combo.has_edges());
    for (Real t : {0.0, 1e-6, 4.9e-6, 6.3e-6, 33.3e-6}) {
        INFO("t = " << t);
        REQUIRE(combo.next_edge_after(t) ==
                std::min(a.next_edge_after(t), b.next_edge_after(t)));
    }
    // A plain lambda in the mix: edges unknown.
    solver::SwitchScheduleFn plain = [](Real) { return SwitchStateMask{2}; };
    auto mixed = make_combined_switch_fn(2, {a, plain});
    REQUIRE_FALSE(mixed.has_edges());
    REQUIRE(std::isinf(mixed.next_edge_after(0.0)));
    // A SwitchSchedule stored in a SwitchScheduleFn is recoverable.
    solver::SwitchScheduleFn erased = a;
    REQUIRE(as_schedule(erased) != nullptr);
    REQUIRE(as_schedule(plain) == nullptr);
}

TEST_CASE("run_transient — recorded duty equals D for whole-step on-times",
          "[v2][pwm_grid][integration]") {
    // Switched resistor: the stamped mask record of the fixed-step
    // engine must show exactly D * T / dt ON samples per period.
    for (const auto& c : {GridCase{100e3, 0.25, 1e-7},
                          GridCase{100e3, 0.5, 2e-7},
                          GridCase{50e3, 0.3, 1e-7}}) {
        builder::CircuitBuilder b;
        b.add_voltage_source("V", "a", "gnd", 10.0);
        b.add_switch("S", "a", "b", 1e3, 1e-9);
        b.add_resistor("R", "b", "gnd", 10.0);
        const Size n_sw = b.graph().num_switches();
        pwl::PwlStateSpaceCache cache(b.graph(), b.pool());
        cache.build(c.dt);
        solver::SimulationOptions opts{
            .t_start = 0.0, .t_end = 40 / c.f, .dt = c.dt};
        auto sw = make_pwm_switch_fn(c.f, c.duty, 0, n_sw);
        auto res = solver::run_transient(cache, b.graph(), b.pool(), opts,
                                         sw);
        REQUIRE(res.switch_states_complete());
        const long n = std::lround(1 / (c.f * c.dt));
        const long want = std::lround(c.duty * n);
        std::set<long> counts;
        // Sample k carries the mask of the step that ENDED at t_k.
        for (long j = 0; (j + 1) * n < static_cast<long>(res.num_steps());
             ++j) {
            long on = 0;
            for (long k = j * n + 1; k <= (j + 1) * n; ++k) {
                on += res.switch_state(static_cast<Size>(k), 0) ? 1 : 0;
            }
            counts.insert(on);
        }
        INFO("f=" << c.f << " D=" << c.duty << " dt=" << c.dt);
        REQUIRE(counts == std::set<long>{want});
    }
}
