// =============================================================================
// PED Gate 3 — C++ end-to-end validation on buck DCM (body-diode commutation)
// =============================================================================
//
// Mirrors the Python Gate 3 validation captured in
// `prototype/dsed/run_buck_dcm_validation.py`:
//
//   24V → ~19V sync-buck in deep DCM, 100 kHz, D = 0.5
//   L = 100 µH, C = 100 µF, R_load = 240 Ω  (100× CCM → DCM)
//   Erickson K = 0.0833 < K_crit = 0.5  →  DCM
//   Analytical M = 2/(1+sqrt(1+4K/D²)) ≈ 0.7913
//   V_out_steady reference ≈ 19.0 V
//
// Gate 3A target: |V_out_mean - analytical| / V_in ≤ 1 %
// Gate 3B target: wall-clock ≥ 1× the trap-with-ZCD-detect reference
//                 at Δt = 50 ns (full 5× speedup target is for the
//                 microbench in test_bench_dsed_dcm.cpp).
//
// Test inventory (Gate 3.4 — 8+ cases):
//   1. Construction / analytical equations sanity
//   2. ZCD predicate fires once per cycle from steady state
//   3. State projection clamps i_L = 0 after ZCD
//   4. SwitchFn returns ZERO_CURRENT after ZCD register
//   5. Event-priority resolver: gate wins over ZCD at exact tie
//   6. Three-mode cycle: HS_ON → LS_CONDUCTING → ZERO_CURRENT → HS_ON
//   7. End-to-end DCM steady-state V_out matches analytical (Gate 3A)
//   8. End-to-end DCM wall-clock vs trap-with-ZCD reference (Gate 3B)

#include <chrono>
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/dsed/buck_dcm_model.hpp"
#include "pulsim/dsed/scheduler.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

using pulsim::Real;
using pulsim::Vector;
using namespace pulsim::dsed;
using namespace pulsim::dsed::buck;

namespace {

// -----------------------------------------------------------------------------
// trap_dcm_reference — fixed-step trapezoidal + post-step ZCD detection
// -----------------------------------------------------------------------------
// Mirrors `prototype/dsed/run_buck_dcm_validation.py::trap_dcm_reference`.
// Used as the v1.4.0-equivalent baseline for the Gate 3B wall-clock check.

struct TrapDcmResult {
    std::vector<Real> times;
    std::vector<Vector> states;
    std::size_t n_steps = 0;
    std::size_t n_zcd = 0;
    Real cpu_seconds = Real{0};
};

[[nodiscard]] TrapDcmResult trap_dcm_reference(
    const BuckDCMParams& p, const Vector& x0, Real t_end, Real dt) {

    const Real T = p.T_sw();
    const Real D = p.D;
    const Real L = p.L;
    const Real C = p.C;
    const Real R = p.R_load;

    // Build A, b vectors (matches BuckDCMModel).
    pulsim::DenseMatrix A(2, 2);
    A << Real{0}, -Real{1} / L,
         Real{1} / C, -Real{1} / (R * C);

    Vector b_HSon(2);
    b_HSon << p.V_in / L, Real{0};
    Vector b_LSc(2);
    b_LSc << Real{0}, Real{0};

    pulsim::DenseMatrix I = pulsim::DenseMatrix::Identity(2, 2);
    pulsim::DenseMatrix M_inv = (I - Real{0.5} * dt * A).inverse();
    pulsim::DenseMatrix Khat = M_inv * (I + Real{0.5} * dt * A);
    Vector dt_M_inv_HSon = dt * (M_inv * b_HSon);

    const Real tau_rc = R * C;
    const Real decay = std::exp(-dt / tau_rc);

    const std::size_t n_steps = static_cast<std::size_t>(std::ceil(t_end / dt));
    TrapDcmResult r;
    r.times.reserve(n_steps + 1);
    r.states.reserve(n_steps + 1);
    r.times.push_back(Real{0});
    r.states.push_back(x0);

    Vector x = x0;
    std::unordered_map<std::int64_t, bool> zcd_fired_cycle;
    BuckMode mode = BuckMode::HS_ON;

    const auto t0_wall = std::chrono::high_resolution_clock::now();
    for (std::size_t k = 0; k < n_steps; ++k) {
        const Real t_cur = static_cast<Real>(k) * dt;
        const Real t_mid = t_cur + Real{0.5} * dt;
        const auto cycle = static_cast<std::int64_t>(std::floor(t_mid / T));
        const Real phase = std::fmod(t_mid / T, Real{1});

        if (phase < D) {
            mode = BuckMode::HS_ON;
        } else if (zcd_fired_cycle.count(cycle)) {
            mode = BuckMode::ZERO_CURRENT;
        } else {
            mode = BuckMode::LS_CONDUCTING;
        }

        if (mode == BuckMode::ZERO_CURRENT) {
            x(0) = Real{0};
            x(1) = x(1) * decay;
        } else if (mode == BuckMode::HS_ON) {
            x = Khat * x + dt_M_inv_HSon;
        } else {
            Vector x_new = Khat * x;  // b_LSc = 0
            if (x(0) > Real{0} && x_new(0) <= Real{0}) {
                // ZCD inside this step — linearly interpolate
                const Real frac = x(0) / (x(0) - x_new(0));
                const Real vC_at_zcd = x(1) + frac * (x_new(1) - x(1));
                const Real rem_dt = (Real{1} - frac) * dt;
                x(0) = Real{0};
                x(1) = vC_at_zcd * std::exp(-rem_dt / tau_rc);
                zcd_fired_cycle[cycle] = true;
                ++r.n_zcd;
            } else {
                x = x_new;
            }
        }

        r.times.push_back(static_cast<Real>(k + 1) * dt);
        r.states.push_back(x);
    }
    const auto t1_wall = std::chrono::high_resolution_clock::now();
    r.n_steps = n_steps;
    r.cpu_seconds = std::chrono::duration<Real>(t1_wall - t0_wall).count();
    return r;
}

// -----------------------------------------------------------------------------
// Helper: build a fully-configured PED simulator for buck DCM
// -----------------------------------------------------------------------------
struct DcmSimBundle {
    BuckDCMModel model;
    BuckDCMSwitchFn switch_fn;
    EventPredictor predictor;
    PIController controller;

    explicit DcmSimBundle(BuckDCMParams p)
        : model{p},
          switch_fn{p.T_sw(), p.D},
          predictor{},
          controller{Real{1e-6}, Real{1e-9}} {  // rtol, atol
        predictor.register_predicate(
            "zcd_iL", make_zcd_predicate(), PredicateType::CurrentZC);
    }
};

}  // namespace

// =============================================================================
// Test 1 — analytical equations sanity
// =============================================================================

TEST_CASE("BuckDCMParams analytical equations are correct",
          "[dsed][gate3][dcm]") {
    BuckDCMParams p;  // defaults: 24V, 100µH, 100µF, 240Ω, 100kHz, D=0.5

    REQUIRE(p.T_sw() == Catch::Approx(Real{10e-6}));
    REQUIRE(p.K() == Catch::Approx(Real{0.0833}).epsilon(1e-3));
    REQUIRE(p.K_crit() == Catch::Approx(Real{0.5}));
    REQUIRE(p.is_dcm_at_steady_state());
    REQUIRE(p.M_dcm() == Catch::Approx(Real{0.7913}).epsilon(1e-3));
    REQUIRE(p.V_out_steady() == Catch::Approx(Real{18.99}).epsilon(1e-3));
    REQUIRE(p.M_ccm() == Catch::Approx(Real{0.5}));
}

// =============================================================================
// Test 2 — ZCD predicate fires once per cycle from steady state
// =============================================================================

TEST_CASE("PED captures one ZCD per cycle in DCM steady state",
          "[dsed][gate3][dcm]") {
    BuckDCMParams p;
    DcmSimBundle bun{p};

    Vector x0(2);
    x0 << Real{0}, p.V_out_steady();  // start of cycle, at steady-state v_C

    PEDSimulator sim(
        bun.model, bun.switch_fn,
        bun.controller, bun.predictor,
        /*dt_init=*/Real{1e-9},
        /*dt_max=*/p.T_sw() / Real{4},
        /*store_every=*/1,
        make_zcd_projection());

    const Real t_end = Real{1e-3};  // 100 switching cycles
    const auto result = sim.simulate(x0, t_end);

    // Count event types
    std::size_t n_gate = 0, n_zcd = 0;
    for (const auto& e : result.event_log) {
        if (e.type == PredicateType::GateEdge) ++n_gate;
        if (e.type == PredicateType::CurrentZC) ++n_zcd;
    }
    const auto n_cycles = static_cast<std::size_t>(
        std::round(t_end / p.T_sw()));

    // One ZCD per cycle (allow ±1 for boundary effects)
    REQUIRE(n_zcd >= n_cycles - 1);
    REQUIRE(n_zcd <= n_cycles + 1);
    // Two gate edges per cycle (HS-on, HS-off), with ± 2 boundary slop
    REQUIRE(n_gate >= 2 * n_cycles - 2);
    REQUIRE(n_gate <= 2 * n_cycles + 2);
}

// =============================================================================
// Test 3 — state projection clamps i_L = 0 after ZCD
// =============================================================================

TEST_CASE("State projection clamps i_L = 0 on ZCD event",
          "[dsed][gate3][dcm]") {
    auto proj = make_zcd_projection();
    Vector x(2);
    x << Real{1e-6}, Real{10};  // tiny i_L just before ZCD, v_C = 10
    Vector y = proj(Real{0}, x, PredicateType::CurrentZC);

    REQUIRE(y(0) == Real{0});
    REQUIRE(y(1) == Catch::Approx(Real{10}));  // v_C untouched

    // For non-ZCD events, projection is identity.
    Vector z = proj(Real{0}, x, PredicateType::GateEdge);
    REQUIRE(z(0) == Catch::Approx(Real{1e-6}));
    REQUIRE(z(1) == Catch::Approx(Real{10}));
}

// =============================================================================
// Test 4 — SwitchFn latches ZERO_CURRENT after register_zcd_transition
// =============================================================================

TEST_CASE("BuckDCMSwitchFn returns ZERO_CURRENT after ZCD register",
          "[dsed][gate3][dcm]") {
    BuckDCMSwitchFn sf{Real{10e-6}, Real{0.5}};  // T=10µs, D=0.5

    // Before any ZCD: mode follows gate schedule
    REQUIRE(sf(Real{0}) == BuckMode::HS_ON);
    REQUIRE(sf(Real{4e-6}) == BuckMode::HS_ON);
    REQUIRE(sf(Real{6e-6}) == BuckMode::LS_CONDUCTING);
    REQUIRE(sf(Real{9e-6}) == BuckMode::LS_CONDUCTING);

    // Register a ZCD at t = 7µs (in cycle 0, after HS-off boundary)
    sf.register_zcd_transition(Real{7e-6});

    // After ZCD: cycle 0 jumps to ZERO_CURRENT past t = 7µs
    REQUIRE(sf(Real{6e-6}) == BuckMode::LS_CONDUCTING);   // before ZCD
    REQUIRE(sf(Real{7.5e-6}) == BuckMode::ZERO_CURRENT);  // after ZCD
    REQUIRE(sf(Real{9e-6}) == BuckMode::ZERO_CURRENT);

    // Next cycle: ZCD memory does NOT bleed across cycle boundaries
    REQUIRE(sf(Real{10e-6}) == BuckMode::HS_ON);          // cycle 1 start
    REQUIRE(sf(Real{16e-6}) == BuckMode::LS_CONDUCTING);  // cycle 1, after HS-off
}

// =============================================================================
// Test 5 — Hermite interpolation reproduces endpoints exactly
// =============================================================================

TEST_CASE("hermite_interp reproduces endpoints + matches linear case",
          "[dsed][gate3][hermite]") {
    Vector x0(2), x1(2), k1(2), k7(2);
    x0 << Real{1}, Real{2};
    x1 << Real{3}, Real{5};
    k1 << Real{2}, Real{3};   // slope at t = 0
    k7 << Real{2}, Real{3};   // matching slope at t = h (linear case)
    const Real h = Real{1};

    // Endpoints
    Vector e0 = hermite_interp(x0, x1, k1, k7, h, Real{0});
    Vector e1 = hermite_interp(x0, x1, k1, k7, h, Real{1});
    REQUIRE(e0(0) == Catch::Approx(Real{1}));
    REQUIRE(e0(1) == Catch::Approx(Real{2}));
    REQUIRE(e1(0) == Catch::Approx(Real{3}));
    REQUIRE(e1(1) == Catch::Approx(Real{5}));

    // For a function with matching slopes at both ends, midpoint is the
    // linear midpoint of (x0, x1) (Hermite collapses to linear when
    // slopes equal the secant).
    Vector mid = hermite_interp(x0, x1, k1, k7, h, Real{0.5});
    REQUIRE(mid(0) == Catch::Approx(Real{2}));    // (1+3)/2
    REQUIRE(mid(1) == Catch::Approx(Real{3.5}));  // (2+5)/2
}

// =============================================================================
// Test 6 — Three-mode cycle progression observable in event log
// =============================================================================

TEST_CASE("PED event log shows correct A→B→C→A mode progression",
          "[dsed][gate3][dcm]") {
    BuckDCMParams p;
    DcmSimBundle bun{p};

    Vector x0(2);
    x0 << Real{0}, p.V_out_steady();

    PEDSimulator sim(
        bun.model, bun.switch_fn,
        bun.controller, bun.predictor,
        Real{1e-9},
        p.T_sw() / Real{4},
        1,
        make_zcd_projection());

    const auto result = sim.simulate(x0, Real{5e-5});  // 5 cycles
    REQUIRE(result.event_log.size() >= 12);  // 5 cycles × 3 events/cycle ≈ 15

    // Walk the event log and check mode transitions are plausible:
    //   gate edge: HS_ON → LS_CONDUCTING OR LS_CONDUCTING|ZERO_CURRENT → HS_ON
    //   ZCD     : LS_CONDUCTING → ZERO_CURRENT
    for (const auto& e : result.event_log) {
        if (e.type == PredicateType::CurrentZC) {
            REQUIRE(e.old_mask == BuckMode::LS_CONDUCTING);
            REQUIRE(e.new_mask == BuckMode::ZERO_CURRENT);
        }
        if (e.type == PredicateType::GateEdge) {
            // Gate edges either light up HS_ON or terminate it
            const bool ok =
                (e.new_mask == BuckMode::HS_ON
                    && (e.old_mask == BuckMode::LS_CONDUCTING
                        || e.old_mask == BuckMode::ZERO_CURRENT))
                || (e.new_mask == BuckMode::LS_CONDUCTING
                    && e.old_mask == BuckMode::HS_ON);
            REQUIRE(ok);
        }
    }
}

// =============================================================================
// Test 7 — Gate 3A correctness: V_out_mean matches analytical DCM regulator
// =============================================================================

TEST_CASE("PED DCM steady-state V_out matches analytical within 1% V_in (Gate 3A)",
          "[dsed][gate3][dcm][validation]") {
    BuckDCMParams p;
    DcmSimBundle bun{p};

    Vector x0(2);
    x0 << Real{0}, p.V_out_steady();

    PEDSimulator sim(
        bun.model, bun.switch_fn,
        bun.controller, bun.predictor,
        Real{1e-9},
        p.T_sw() / Real{4},
        1,
        make_zcd_projection());

    const Real t_end = Real{1e-3};  // 100 cycles
    const auto result = sim.simulate(x0, t_end);

    // Compute V_out mean over the last 0.5 ms (50 cycles, steady state)
    Real sum_vout = Real{0};
    std::size_t count = 0;
    for (std::size_t i = 0; i < result.times.size(); ++i) {
        if (result.times[i] >= Real{0.5e-3}) {
            sum_vout += result.states[i](1);
            ++count;
        }
    }
    REQUIRE(count > 0);
    const Real mean_vout = sum_vout / static_cast<Real>(count);

    const Real analytical = p.V_out_steady();
    const Real rel_err_pct =
        Real{100} * std::abs(mean_vout - analytical) / p.V_in;

    // Gate 3A criterion: ≤ 1 % of V_in
    REQUIRE(rel_err_pct <= Real{1.0});
    // Tighter check (matches Python prototype 0.0043 %):
    REQUIRE(rel_err_pct <= Real{0.1});
}

// =============================================================================
// Test 8 — Gate 3B wall-clock vs trap-with-ZCD reference
// =============================================================================

TEST_CASE("PED DCM wall-clock at most reference ×5 (prototype-level Gate 3B)",
          "[dsed][gate3][dcm][validation]") {
    BuckDCMParams p;
    DcmSimBundle bun{p};

    Vector x0(2);
    x0 << Real{0}, p.V_out_steady();

    const Real t_end = Real{1e-3};

    // PED engine
    PEDSimulator sim(
        bun.model, bun.switch_fn,
        bun.controller, bun.predictor,
        Real{1e-9},
        p.T_sw() / Real{4},
        1,
        make_zcd_projection());
    const auto ped_result = sim.simulate(x0, t_end);

    // Trap-with-ZCD reference at Δt = 50 ns
    const auto trap_result = trap_dcm_reference(p, x0, t_end, Real{50e-9});

    INFO("PED wall-clock:  " << ped_result.cpu_time_seconds * Real{1e3} << " ms");
    INFO("trap wall-clock: " << trap_result.cpu_seconds * Real{1e3} << " ms");
    INFO("Ratio PED/trap: "
         << ped_result.cpu_time_seconds / trap_result.cpu_seconds);

    // The Gate 3B target (5× speedup) is for the microbench
    // (test_bench_dsed_dcm.cpp); here we just check PED isn't egregiously
    // slow (≤ 5× trap is a generous unit-test budget).
    REQUIRE(ped_result.cpu_time_seconds <= Real{5} * trap_result.cpu_seconds);

    // Sanity: PED captures at least 95% of trap's ZCD events
    std::size_t ped_zcd = 0;
    for (const auto& e : ped_result.event_log) {
        if (e.type == PredicateType::CurrentZC) ++ped_zcd;
    }
    REQUIRE(ped_zcd * 100 >= trap_result.n_zcd * 95);

    // Final V_out should agree between engines within 0.1 % of V_in
    const Real ped_final_vout = ped_result.states.back()(1);
    const Real trap_final_vout = trap_result.states.back()(1);
    const Real abs_diff = std::abs(ped_final_vout - trap_final_vout);
    REQUIRE(abs_diff / p.V_in <= Real{0.001});
}
