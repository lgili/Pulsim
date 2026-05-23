// =============================================================================
// Layer 5 V1 — Integration: RLC underdamped ringdown
// =============================================================================
//
// Circuit:  V_dc ──[Source]── n0 ──[R]── n1 ──[L]── n2 ──[C]── GND
//
// Series RLC. From rest with V_dc applied, the cap voltage rises
// and oscillates as it converges to V_dc with damped ringing.
//
// Analytical underdamped step response:
//   ω_n = 1/√(LC),  ζ = (R/2)·√(C/L)
//   ω_d = ω_n · √(1−ζ²)
//   V_C(t) = V_dc · [1 − e^{−ζω_n t} · (cos(ω_d t) +
//                                       (ζω_n/ω_d) · sin(ω_d t))]
//
// With L=1µH, C=1µF, R=0.5Ω:
//   ω_n  = 10^6 rad/s
//   ζ   = 0.25 (underdamped)
//   ω_d ≈ 0.968·10^6 rad/s
//   T_d ≈ 6.49 µs
//
// Trap rule is energy-preserving for L/C; the analytical match
// should be within a few % over several periods.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
#include <memory>
#include <numbers>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::topology;
using Catch::Approx;

namespace {

struct RLCCircuit {
    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;
    Index n0 = -1, n1 = -1, n2 = -1;
    Index i_L_idx = -1;
    Real V_dc = 10.0;
    Real R    = 0.5;
    Real L    = 1e-6;
    Real C    = 1e-6;

    explicit RLCCircuit(Real dt) {
        n0 = g.add_node("n0");
        n1 = g.add_node("n1");
        n2 = g.add_node("n2");
        g.add_branch(n0, g.ground(), BranchKind::Source);
        g.add_branch(n0, n1,         BranchKind::PassiveLinear);
        g.add_branch(n1, n2,         BranchKind::PassiveLinear);
        g.add_branch(n2, g.ground(), BranchKind::PassiveLinear);

        pool.add_voltage_source(0, {.V = V_dc});
        pool.add_resistor(1, {.G = Real{1} / R});
        pool.add_inductor(2, {.L = L});
        pool.add_capacitor(3, {.C = C});

        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
        cache->build(dt);
        i_L_idx = pool.branch_var_id_for_inductor(2, g);
    }

    [[nodiscard]] Real omega_n() const noexcept {
        return Real{1} / std::sqrt(L * C);
    }
    [[nodiscard]] Real zeta() const noexcept {
        return (R / Real{2}) * std::sqrt(C / L);
    }
    [[nodiscard]] Real omega_d() const noexcept {
        const Real z = zeta();
        return omega_n() * std::sqrt(Real{1} - z * z);
    }
};

}  // namespace

TEST_CASE("RLC integration: ω_d and ζ from parameters",
          "[v2][layer5_v1][integration][rlc]") {
    RLCCircuit rlc(1e-8);
    REQUIRE(rlc.omega_n() == Approx(1.0e6).epsilon(1e-6));
    REQUIRE(rlc.zeta() == Approx(0.25).epsilon(1e-6));
    REQUIRE(rlc.omega_d() == Approx(0.968e6).epsilon(1e-3));
}

TEST_CASE("RLC integration: v_C(t) matches analytical step response",
          "[v2][layer5_v1][integration][rlc]") {
    RLCCircuit rlc(1e-8);              // dt = 10 ns
    const Real T_d = 2 * std::numbers::pi_v<Real> / rlc.omega_d();

    SimulationOptions opts{
        .t_start = 0,
        .t_end   = 4.0 * T_d,           // 4 damped periods
        .dt      = 1e-8,
    };
    SwitchScheduleFn fn = [](Real) {
        return SwitchStateMask(0);
    };

    auto result = run_transient(*rlc.cache, rlc.g, rlc.pool, opts, fn);

    const Real omega_n = rlc.omega_n();
    const Real omega_d = rlc.omega_d();
    const Real zeta    = rlc.zeta();

    // Step response of an underdamped 2nd-order system:
    //   V_C(t)/V_dc = 1 − e^{−ζω_n t} · (cos(ω_d t) +
    //                                    (ζω_n/ω_d)·sin(ω_d t))
    auto v_C_analytical = [&](Real t) {
        const Real env = std::exp(-zeta * omega_n * t);
        const Real osc = std::cos(omega_d * t) +
                          (zeta * omega_n / omega_d) *
                          std::sin(omega_d * t);
        return rlc.V_dc * (Real{1} - env * osc);
    };

    // Check the bulk of the waveform (skip the first few samples
    // where trap rule has a known boundary error).
    Real max_abs_err = 0;
    for (Size k = 10; k < result.num_steps(); ++k) {
        const Real t = result.times[k];
        const Real v_num = result.states[k][rlc.n2];
        const Real v_ana = v_C_analytical(t);
        const Real abs_err = std::abs(v_num - v_ana);
        max_abs_err = std::max(max_abs_err, abs_err);
    }
    INFO("RLC: max abs error over 4 periods = " << max_abs_err);
    // 4% of V_dc is roughly the expected envelope amplitude
    // resolution at dt=T_d/100 — comfortable margin for the
    // trapezoidal rule's accuracy.
    REQUIRE(max_abs_err < Real{0.04} * rlc.V_dc);
}

TEST_CASE("RLC integration: undamped (R=0) oscillates at ω_n",
          "[v2][layer5_v1][integration][rlc]") {
    // With R=0, the system is purely lossless. Trap rule
    // preserves energy → oscillation amplitude is constant (no
    // numerical decay).
    //
    // Topology:   V_dc ──[Source]── n0 ──[L]── n1 ──[C]── GND
    // (no resistor — the L provides the algebraic-loop closure)
    Graph g;
    auto n0 = g.add_node("n0");
    auto n1 = g.add_node("n1");
    g.add_branch(n0, g.ground(), BranchKind::Source);
    g.add_branch(n0, n1,         BranchKind::PassiveLinear);  // L
    g.add_branch(n1, g.ground(), BranchKind::PassiveLinear);  // C

    const Real V_dc = 10.0;
    const Real L    = 1e-6;
    const Real C    = 1e-6;
    const Real omega_n = Real{1} / std::sqrt(L * C);
    const Real T = 2 * std::numbers::pi_v<Real> / omega_n;

    DevicePool pool;
    pool.add_voltage_source(0, {.V = V_dc});
    pool.add_inductor(1, {.L = L});
    pool.add_capacitor(2, {.C = C});

    const Real dt = T / 200;            // 200 samples / period

    PwlStateSpaceCache cache(g, pool);
    cache.build(dt);

    SimulationOptions opts{.t_start = 0, .t_end = 3 * T, .dt = dt};
    SwitchScheduleFn fn = [](Real) { return SwitchStateMask(0); };

    auto result = run_transient(cache, g, pool, opts, fn);

    // Undamped LC step response peaks at 2·V_dc every odd
    // half-period (T/2, 3T/2, 5T/2, ...). Trap rule preserves
    // energy so every peak has essentially the same amplitude;
    // FP noise can make a later peak fractionally higher than
    // the first. Find the FIRST local maximum (sample where the
    // next sample is lower) instead of the global max.
    Real first_peak_v   = 0;
    Real first_peak_t   = 0;
    for (Size k = 1; k + 1 < result.num_steps(); ++k) {
        const Real v_prev = result.states[k - 1][n1];
        const Real v_here = result.states[k    ][n1];
        const Real v_next = result.states[k + 1][n1];
        if (v_here > v_prev && v_here > v_next && v_here > V_dc) {
            first_peak_v = v_here;
            first_peak_t = result.times[k];
            break;
        }
    }
    INFO("Undamped: first peak v_C = " << first_peak_v
         << " at t = " << first_peak_t
         << ", expected ~" << T / 2 << " s ≈ 2·V_dc = "
         << 2 * V_dc);
    REQUIRE(first_peak_v == Approx(2 * V_dc).epsilon(0.05));
    REQUIRE(first_peak_t == Approx(T / 2).epsilon(0.05));
}
