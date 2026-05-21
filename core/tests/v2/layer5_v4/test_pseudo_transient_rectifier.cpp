// =============================================================================
// Layer 4 V10 — PTC primitive + smart-warm-start helper
// =============================================================================
//
// V8 (κ-homotopy) and V9 (V_F0-homotopy + combined) shipped
// continuation primitives but found that NO Newton-globalization
// chain we tried (line search, LM, κ chain, V_F0 chain, combined)
// can robustly solve the κ=20 stiff sinusoidal rectifier from
// `x_init = 0`. The smooth-blend sigmoid at κ=20 creates a wall
// that Newton's step jumps over into a bad basin.
//
// V10 ships two complementary tools:
//
//   1. `pseudo_transient_solve` — converts F(x)=0 into an
//      artificial ODE dx/dt = -F(x) with implicit Euler +
//      adaptive dt. Robust for LINEAR circuits and MILDLY
//      stiff problems. For DAE-MNA systems with sharp
//      sigmoids, the vanilla PTC dt schedule is fragile (the
//      I/dt term doesn't respect the algebraic/dynamic row
//      mismatch).
//
//   2. `make_diode_aware_initial_guess(graph, pool, b_extra)` —
//      a smart warm-start helper that walks the DevicePool,
//      reads voltage-source effective voltages from
//      pool.V + b_extra, and writes them onto the source's
//      "from" node. For canonical source→diode→load circuits,
//      this puts Newton inside the correct basin of
//      attraction.
//
// THIS file's KEY TEST: combine `make_diode_aware_initial_guess`
// with plain Newton + line search to solve the κ=20 stiff
// sinusoidal rectifier from `x = 0` (no manual load-line).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/initial_guess.hpp"
#include "pulsim/v2/pwl/nonlinear_refresh_diode.hpp"
#include "pulsim/v2/pwl/nonlinear_solve.hpp"
#include "pulsim/v2/pwl/pseudo_transient.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <cmath>
#include <memory>
#include <numbers>
#include <vector>

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::topology;
using Catch::Approx;

// Note: PTC ALONE (without smart warm-start) is unsuitable
// for Pulsim MNA systems with voltage-source constraints —
// the artificial dynamics dx/dt = -F is unstable for J's
// negative-eigenvalue subspaces. PTC is shipped in
// `pseudo_transient.hpp` for users with well-behaved
// Jacobians, but no PTC unit test ships in V10 since
// canonical Pulsim circuits all have MNA constraints.

TEST_CASE("make_diode_aware_initial_guess writes source values onto from-node",
          "[v2][layer4_v10][initial_guess]") {
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1,          BranchKind::Nonlinear);
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 3.5});
    const models::IdealDiode::Params dp{
        .V_F0 = 0.7, .R_d = 0.01,
        .G_off = 1e-9, .kappa = 20.0};
    pool.add_nonlinear_diode(1, dp);
    pool.add_resistor(2, {.G = 1.0 / 100.0});

    const Size n = pool.state_size(g);
    Vector b_extra = Vector::Zero(static_cast<Index>(n));

    Vector guess =
        make_diode_aware_initial_guess(g, pool, b_extra);

    REQUIRE(static_cast<Size>(guess.size()) == n);
    // v_n0 should be set to source voltage (3.5).
    REQUIRE(guess[0] == Approx(3.5).margin(1e-9));
    // v_n1 (diode cathode) should be untouched at 0.
    REQUIRE(guess[1] == Approx(0.0).margin(1e-9));
}

TEST_CASE("make_diode_aware_initial_guess folds b_extra into the source",
          "[v2][layer4_v10][initial_guess]") {
    // For a sinusoidal-source-via-b_extra circuit (the
    // canonical pulsim pattern), the effective voltage at
    // time t is pool.V − b_extra[source_var]. The helper
    // must compute this correctly.
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 0.0});   // pool baseline

    const Size n = pool.state_size(g);
    const Index src_var = pool.branch_var_id_for_source(0, g);

    Vector b_extra = Vector::Zero(static_cast<Index>(n));
    b_extra[src_var] = -7.5;   // overlay V_sine = +7.5 V

    Vector guess =
        make_diode_aware_initial_guess(g, pool, b_extra);

    // V_effective = 0.0 − (−7.5) = +7.5.
    REQUIRE(guess[0] == Approx(7.5).margin(1e-9));
}

// -----------------------------------------------------------------------------
// THE motivating test — κ=20 stiff sinusoidal rectifier from
// AUTO warm-start (no manual load-line trickery in the test
// driver). The driver just calls
// `make_diode_aware_initial_guess` per step and feeds the
// result into plain Newton + line search.
// -----------------------------------------------------------------------------
namespace {

struct AutoWSStiffRectifier {
    static constexpr Real V_amp  = 10.0;
    static constexpr Real f_line = 60.0;
    static constexpr Real R_load = 10.0;
    static constexpr Real V_F0   = 0.7;
    static constexpr Real R_d    = 0.01;
    static constexpr Real G_off  = 1e-9;
    static constexpr Real kappa  = 20.0;

    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;
    Index n0 = -1, n1 = -1;
    Index source_branch_var = -1;

    AutoWSStiffRectifier() {
        n0 = g.add_node("n0");
        n1 = g.add_node("n1");
        g.add_branch(n0, g.ground(), BranchKind::Source);
        g.add_branch(n0, n1,         BranchKind::Nonlinear);
        g.add_branch(n1, g.ground(), BranchKind::PassiveLinear);

        pool.add_voltage_source(0, {.V = 0.0});
        const models::IdealDiode::Params dp{
            .V_F0 = V_F0, .R_d = R_d,
            .G_off = G_off, .kappa = kappa};
        pool.add_nonlinear_diode(1, dp);
        pool.add_resistor(2, {.G = 1.0 / R_load});

        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
        cache->build();
        source_branch_var = pool.branch_var_id_for_source(0, g);
    }

    Vector b_extra_at(Real t) const {
        Vector b = Vector::Zero(
            static_cast<Eigen::Index>(pool.state_size(g)));
        const Real omega =
            2.0 * std::numbers::pi_v<Real> * f_line;
        const Real V_sine = V_amp * std::sin(omega * t);
        b[source_branch_var] = -V_sine;
        return b;
    }
};

}  // namespace

TEST_CASE("κ=20 stiff rectifier solves from AUTO smart warm-start "
          "(no manual load-line)",
          "[v2][layer4_v10][auto_ws][rectifier]") {
    AutoWSStiffRectifier rect;
    const auto& seg = rect.cache->lookup(SwitchStateMask(0));

    constexpr Real dt_sim = 1.0e-4;
    constexpr Real T_line = 1.0 / AutoWSStiffRectifier::f_line;
    constexpr Real t_end = 2.0 * T_line;
    const Size n_steps = static_cast<Size>(t_end / dt_sim) + 1;
    const Real omega =
        2.0 * std::numbers::pi_v<Real> *
        AutoWSStiffRectifier::f_line;

    std::vector<Real> times;
    std::vector<Vector> states;
    times.reserve(n_steps);
    states.reserve(n_steps);

    for (Size k = 0; k < n_steps; ++k) {
        const Real t = static_cast<Real>(k) * dt_sim;
        const Vector b_extra = rect.b_extra_at(t);

        // AUTOMATIC smart warm-start: walks the pool, reads
        // source values, writes them onto source nodes.
        const Vector x_init =
            make_diode_aware_initial_guess(
                rect.g, rect.pool, b_extra);

        Vector x = solve_with_newton_b_extra(
            seg, &refresh_smooth_diodes,
            rect.g, rect.pool,
            x_init, b_extra,
            /*max_iters=*/100,
            /*tol_dx=*/1e-7, /*tol_res=*/1e-5,
            /*enable_line_search=*/true,
            /*enable_lm=*/false);

        times.push_back(t);
        states.push_back(x);
    }

    REQUIRE(times.size() == n_steps);

    // Verify output across the LAST full cycle.
    const Size k_start = static_cast<Size>(T_line / dt_sim);
    Size n_pos_checked = 0, n_pos_match = 0;
    Size n_neg_checked = 0, n_neg_match = 0;
    Real sum_power = 0;
    Size n_power_samples = 0;

    for (Size k = k_start; k < times.size(); ++k) {
        const Real t = times[k];
        const Real v_sine =
            AutoWSStiffRectifier::V_amp * std::sin(omega * t);
        const Real v_out = states[k][rect.n1];

        sum_power += (v_out * v_out) /
                     AutoWSStiffRectifier::R_load;
        ++n_power_samples;

        if (v_sine > 1.0) {
            ++n_pos_checked;
            const Real v_expected = std::max(
                v_sine - AutoWSStiffRectifier::V_F0,
                Real{0});
            if (std::abs(v_out - v_expected) < 1.0) {
                ++n_pos_match;
            }
        } else if (v_sine < -1.0) {
            ++n_neg_checked;
            if (std::abs(v_out) < 0.5) {
                ++n_neg_match;
            }
        }
    }

    INFO("κ=20 auto-WS rectifier: pos-half match "
         << n_pos_match << "/" << n_pos_checked
         << ", neg-half match "
         << n_neg_match << "/" << n_neg_checked);

    REQUIRE(n_pos_checked > 0);
    REQUIRE(n_neg_checked > 0);
    REQUIRE(n_pos_match * 100 >= n_pos_checked * 95);
    REQUIRE(n_neg_match * 100 >= n_neg_checked * 95);

    const Real V_eff =
        AutoWSStiffRectifier::V_amp -
        AutoWSStiffRectifier::V_F0;
    const Real P_ana = V_eff * V_eff /
                       (4.0 * AutoWSStiffRectifier::R_load);
    const Real mean_power = sum_power /
                            static_cast<Real>(n_power_samples);
    INFO("κ=20 auto-WS rectifier: mean P = " << mean_power
         << " W (analytical: " << P_ana << " W)");
    REQUIRE(mean_power == Approx(P_ana).epsilon(0.15));
}
