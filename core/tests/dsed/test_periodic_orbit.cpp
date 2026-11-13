// =============================================================================
// DSED — the Newton shooting loop in the kernel
// =============================================================================
//
// add-design-optimization-workflow DECISIONS 18
// (pulsim/dsed/periodic_orbit.hpp): the stopping rule and every typed
// outcome on affine maps whose fixed point is known, then the whole loop
// on the DCM buck of add-periodic-steady-state-shooting (orbit and
// multipliers of the change's 50-digit prototype).

#include <catch2/catch_test_macros.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/dsed/builder_adapter.hpp"
#include "pulsim/dsed/periodic_orbit.hpp"
#include "pulsim/pwl/cache.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace pulsim;

namespace {

using SSM = topology::SwitchStateMask;
using Map = dsed::PeriodMapResult<SSM>;

/// P(x) = A·x + b as a period run: x_end, M = A, scale = |x| and |P(x)|.
struct Affine {
    DenseMatrix A;
    Vector b;
    int* calls = nullptr;
    Map operator()(const Vector& x, bool /*obs*/) const {
        if (calls) ++*calls;
        Map r;
        r.x_end = A * x + b;
        r.monodromy = A;
        r.scale = x.cwiseAbs().cwiseMax(r.x_end.cwiseAbs());
        r.n_steps = 1;
        return r;
    }
};

Affine affine(DenseMatrix A, Vector b) { return Affine{std::move(A), std::move(b)}; }

Vector v2(Real a, Real c) {
    Vector v(2);
    v << a, c;
    return v;
}

DenseMatrix m2(Real a, Real b, Real c, Real d) {
    DenseMatrix m(2, 2);
    m << a, b, c, d;
    return m;
}

struct Gate {
    Real T, D;
    std::size_t n_sw;
    SSM operator()(Real t) const {
        SSM m(n_sw);
        const Real ph = t - T * std::floor(t / T);
        if (n_sw) m.set(0, ph < D * T);
        return m;
    }
    Real next_edge_after(Real t) const {
        const Real k = std::floor(t / T);
        for (Real c : {k * T + D * T, (k + 1) * T, (k + 1) * T + D * T}) {
            if (c > t * (1 + 1e-15) + 1e-300) return c;
        }
        return (k + 2) * T;
    }
};

}  // namespace

TEST_CASE("Orbit loop: an affine map converges in one Newton step",
          "[v2][dsed][periodic_orbit]") {
    const auto f = affine(m2(0.5, 0.1, 0.0, 0.3), v2(1.0, 2.0));
    const auto r = dsed::periodic_orbit_newton<SSM>(f, Vector::Zero(2), 1.0, {});
    REQUIRE(r.status == dsed::OrbitStatus::Converged);
    const Vector xs = (DenseMatrix::Identity(2, 2) - f.A).partialPivLu().solve(f.b);
    CHECK((r.x0 - xs).cwiseAbs().maxCoeff() <= 1e-15 * xs.cwiseAbs().maxCoeff());
    // run from 0, the Newton step, the verification; two iterations
    CHECK(r.n_runs == 3);
    CHECK(r.n_iterations == 2);
    CHECK(r.history.size() == 2);
    CHECK(r.has_period);
    REQUIRE(r.multipliers.size() == 2);
    std::vector<Real> mu{r.multipliers[0].real(), r.multipliers[1].real()};
    std::sort(mu.begin(), mu.end());
    CHECK(mu == std::vector<Real>{0.3, 0.5});
}

TEST_CASE("Orbit loop: every non-converged outcome is a status, never an orbit",
          "[v2][dsed][periodic_orbit]") {
    SECTION("singular I - M (a multiplier equal to 1)") {
        const auto r = dsed::periodic_orbit_newton<SSM>(
            affine(m2(1.0, 0.0, 0.0, 0.5), v2(1.0, 1.0)), Vector::Zero(2), 1.0, {});
        CHECK(r.status == dsed::OrbitStatus::Singular);
        REQUIRE(r.marginal_direction.size() == 2);
        CHECK(r.marginal_direction[0] > 0.99);   // the integrating state
    }
    SECTION("budget") {
        dsed::OrbitOptions o;
        o.max_period_runs = 1;
        const auto r = dsed::periodic_orbit_newton<SSM>(
            affine(m2(0.5, 0.0, 0.0, 0.5), v2(1.0, 1.0)), Vector::Zero(2), 1.0, o);
        CHECK(r.status == dsed::OrbitStatus::Budget);
        CHECK_FALSE(r.verify_budget);
        CHECK(r.n_runs == 1);
    }
    SECTION("no run left for the verification period") {
        dsed::OrbitOptions o;
        o.max_period_runs = 2;
        const auto r = dsed::periodic_orbit_newton<SSM>(
            affine(m2(0.5, 0.0, 0.0, 0.5), v2(1.0, 1.0)), Vector::Zero(2), 1.0, o);
        CHECK(r.status == dsed::OrbitStatus::Budget);
        CHECK(r.verify_budget);
    }
    SECTION("a NaN period is never accepted") {
        auto nan_map = [](const Vector& x, bool) {
            Map r;
            r.x_end = Vector::Constant(x.size(), std::numeric_limits<Real>::quiet_NaN());
            r.monodromy = DenseMatrix::Identity(2, 2) * 0.5;
            r.scale = Vector::Ones(2);
            return r;
        };
        const auto r = dsed::periodic_orbit_newton<SSM>(nan_map, Vector::Zero(2), 1.0, {});
        CHECK(r.status == dsed::OrbitStatus::NoProgress);
        // 1 + 4 rounds of 9 backtracking runs + 3 x 5 relaxation periods
        CHECK(r.n_runs == 1 + 4 * 9 + 3 * 5);
        CHECK_FALSE(r.has_period);
    }
    SECTION("a failed run") {
        int calls = 0;
        auto f = [&](const Vector& x, bool o) {
            if (++calls == 2) throw dsed::PeriodRunFailed("did not settle");
            return affine(m2(0.5, 0.0, 0.0, 0.5), v2(1.0, 1.0))(x, o);
        };
        const auto r = dsed::periodic_orbit_newton<SSM>(f, Vector::Zero(2), 1.0, {});
        CHECK(r.status == dsed::OrbitStatus::RunFailed);
        CHECK(r.message == "did not settle");
    }
    SECTION("a period without M (grazing)") {
        auto f = [](const Vector& x, bool) {
            Map r;
            r.x_end = x;
            r.scale = Vector::Ones(2);
            r.grazing = "grazing event: D1 turns OFF";
            return r;
        };
        const auto r = dsed::periodic_orbit_newton<SSM>(f, v2(1.0, 1.0), 1.0, {});
        CHECK(r.status == dsed::OrbitStatus::Grazing);
        CHECK(r.message == "grazing event: D1 turns OFF");
    }
    SECTION("a multiplier on the unit circle (flip)") {
        const auto r = dsed::periodic_orbit_newton<SSM>(
            affine(m2(-1.0, 0.0, 0.0, 0.5), v2(1.0, 1.0)), Vector::Zero(2), 1.0, {});
        CHECK(r.status == dsed::OrbitStatus::Marginal);
        CHECK(r.marginal_mu.real() == -1.0);
    }
    SECTION("a verification period that does not return") {
        auto f = [](const Vector& x, bool obs) {
            auto r = affine(m2(0.5, 0.0, 0.0, 0.5), v2(1.0, 1.0))(x, obs);
            if (obs) r.x_end[0] += 1e-3;
            return r;
        };
        const auto r = dsed::periodic_orbit_newton<SSM>(f, Vector::Zero(2), 1.0, {});
        CHECK(r.status == dsed::OrbitStatus::Irreproducible);
        CHECK(r.residual > 0.9e-3);
    }
}

TEST_CASE("Orbit loop: the DCM buck orbit of the shooting prototype",
          "[v2][dsed][periodic_orbit]") {
    const Real T = 1e-5;
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 24.0);
    b.add_switch("S", "vin", "sw", 1e3, 1e-9);
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9);
    b.add_inductor("L", "sw", "vout", 100e-6);
    b.add_capacitor("C", "vout", "gnd", 47e-6);
    b.add_resistor("RL", "vout", "gnd", 50.0);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    dsed::NativeCircuitBuilderAdapter<> ad{b.graph(), b.pool(), cache};
    const Gate g{T, 0.3, static_cast<std::size_t>(b.graph().num_switches())};
    auto run = [&](const Vector& x, bool) {
        dsed::EventPredictor pred;
        auto diodes = dsed::add_diode_predicates(ad, pred);
        dsed::PEDSimulator<decltype(ad), Gate> sim(
            ad, g, dsed::PIController(1e-11, 1e-14), std::move(pred), T * 1e-6, T, 1);
        sim.enable_diode_commutation(diodes.owned,
                                     SSM(b.graph().num_switches()), diodes.list);
        sim.set_event_projection(false);
        const auto res = sim.simulate(x, T);
        return dsed::period_map_from_run(ad, res, diodes, {}, {}, true);
    };
    ad.set_mask(g(0.0));
    const auto r = dsed::periodic_orbit_newton<SSM>(run, Vector::Zero(ad.n_state()), T, {});
    REQUIRE(r.status == dsed::OrbitStatus::Converged);
    // (i_L, v_C) in either order: the reduced state is (v_C, i_L) here
    const Real vc = r.x0.cwiseAbs().maxCoeff();
    CHECK(std::abs(vc - 8.99624708728943) <= 1e-9 * 8.99624708728943);
    // the same Newton path as the Python loop: 6 runs, 5 iterations
    CHECK(r.n_runs == 6);
    CHECK(r.n_iterations == 5);
    std::vector<Real> mu{std::abs(r.multipliers[0]), std::abs(r.multipliers[1])};
    std::sort(mu.begin(), mu.end());
    CHECK(mu[0] < 1e-6);
    CHECK(std::abs(mu[1] - 0.988977) <= 1e-6);
    CHECK(r.residual <= 1e-9 * vc);
}
