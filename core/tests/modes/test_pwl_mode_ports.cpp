// add-complementarity-mode-selection task 4.2 on pwl — the diode k-port of a
// fixed-step decision site (pwl/mode_ports.hpp) against the cache's own solve.
#include <catch2/catch_test_macros.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/mode_ports.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>

using namespace pulsim;

namespace {

// A two-diode rectifier with a floating DC bus held by a 1 MOhm tie: the
// rounding case of the history form (C/h·x much larger than the answer).
builder::CircuitBuilder rectifier() {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 100.0);
    b.add_diode("D1", "in", "p", 1e3, 1e-9);
    b.add_diode("D2", "n", "gnd", 1e3, 1e-9);
    b.add_capacitor("C", "p", "n", 1e-3);
    b.add_resistor("R", "p", "n", 50.0);
    b.add_resistor("Rg", "n", "gnd", 1e6);
    b.add_inductor("L", "in", "gnd", 1e-3);
    return b;
}

Real vp(const Vector& s, Index a, Index c) {
    return stamping::read_node_voltage(s, a) - stamping::read_node_voltage(s, c);
}

}  // namespace

TEST_CASE("pwl port model: u and Z reproduce the cache's own site solve",
          "[modes][pwl_ports]") {
    auto b = rectifier();
    const auto& g = b.graph();
    auto& pool = b.pool();
    const Real dt = 1e-5;
    pwl::PwlStateSpaceCache cache(g, pool);
    cache.build_lazy(dt);
    pwl::DiodeEventState diodes(g, pool);
    pwl::PwlModePorts ports(cache, g, pool, diodes);
    REQUIRE(ports.branch_form_diode().empty());
    REQUIRE(ports.ports().size() == 2);

    const auto n = static_cast<Index>(pool.state_size(g));
    Vector x0 = Vector::LinSpaced(n, 3.0, 97.0);            // any step-start state
    Vector b_extra = Vector::LinSpaced(n, -2.0, 5.0) * 1e-3; // any site rhs
    topology::SwitchStateMask mask(g.num_switches());         // both OFF
    for (const Real f : {Real{1}, Real{0.5}, Real{1e-3}}) {   // step, γ-like stage, tiny sub-step
        const Real h = dt * f;
        DenseMatrix Z;
        Vector u;
        ports.site(mask, h, b_extra, x0, Z, u);
        Vector s;
        cache.solve_at(mask, h, b_extra, s);                  // the site's own solve
        {
            sparse::Matrix G, C;
            Vector bc;
            pwl::assemble_segment_split(g, pool, mask, G, C, bc);
            using LD = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;
            LD J = (DenseMatrix(G).cast<long double>() + DenseMatrix(C).cast<long double>() / (long double)h);
            Eigen::Matrix<long double, Eigen::Dynamic, 1> r = -(bc + b_extra).cast<long double>();
            Eigen::Matrix<long double, Eigen::Dynamic, 1> sl = J.partialPivLu().solve(r);
            const auto& es0 = diodes.entries();
            for (std::size_t j = 0; j < es0.size(); ++j) {
                long double rl = (es0[j].from >= 0 ? sl[es0[j].from] : 0) - (es0[j].to >= 0 ? sl[es0[j].to] : 0);
                std::printf("f=%g j=%zu  u-exact=%.3Le  cache-exact=%.3Le\n", (double)f, j,
                    (long double)u[static_cast<Index>(j)] - rl, (long double)vp(s, es0[j].from, es0[j].to) - rl);
            }
        }
        const auto& es = diodes.entries();
        for (std::size_t j = 0; j < es.size(); ++j) {
            const Real ref = vp(s, es[j].from, es[j].to);
            INFO("f=" << f << " j=" << j << " u=" << u[static_cast<Index>(j)] << " ref=" << ref);
            CHECK(std::abs(u[static_cast<Index>(j)] - ref) <= 1e-9 * (1 + std::abs(ref)));
        }
        // a current e drawn from each anode into its cathode: v = u − Z·e
        Vector e(2);
        e << 0.3, -0.7;
        Vector bx = b_extra;
        for (std::size_t j = 0; j < es.size(); ++j) {
            if (stamping::node_is_active(es[j].from)) bx[es[j].from] += e[static_cast<Index>(j)];
            if (stamping::node_is_active(es[j].to)) bx[es[j].to] -= e[static_cast<Index>(j)];
        }
        cache.solve_at(mask, h, bx, s);
        const Vector v = u - Z * e;
        for (std::size_t j = 0; j < es.size(); ++j) {
            const Real ref = vp(s, es[j].from, es[j].to);
            REQUIRE(std::abs(v[static_cast<Index>(j)] - ref) <= 1e-9 * (1 + std::abs(ref)));
        }
    }
}

TEST_CASE("pwl port model: the widened pass is backward Euler from x0",
          "[modes][pwl_ports]") {
    auto b = rectifier();
    const auto& g = b.graph();
    auto& pool = b.pool();
    pwl::PwlStateSpaceCache cache(g, pool);
    cache.build_lazy(1e-5);
    pwl::DiodeEventState diodes(g, pool);
    pwl::PwlModePorts ports(cache, g, pool, diodes);
    const auto n = static_cast<Index>(pool.state_size(g));
    const Vector x0 = Vector::LinSpaced(n, 1.0, 40.0);
    const Vector b_src = Vector::Zero(n);
    topology::SwitchStateMask mask(g.num_switches());
    mask.set(0, true);
    const Real h = 1e-3;
    DenseMatrix Z;
    Vector u;
    ports.widened(mask, h, b_src, x0, Z, u);
    // History form, dense: (G + C/(2h))·s = (C/(2h))·x0 − b_c − b_src
    sparse::Matrix G, C;
    Vector bc;
    pwl::assemble_segment_split(g, pool, mask, G, C, bc);
    const DenseMatrix J = DenseMatrix(G) + DenseMatrix(C) / (2 * h);
    const Vector s = J.partialPivLu().solve(DenseMatrix(C) / (2 * h) * x0 - bc - b_src);
    const auto& es = diodes.entries();
    for (std::size_t j = 0; j < es.size(); ++j) {
        const Real ref = vp(s, es[j].from, es[j].to);
        REQUIRE(std::abs(u[static_cast<Index>(j)] - ref) <= 1e-9 * (1 + std::abs(ref)));
    }
    REQUIRE(ports.solves() == 3);
}

TEST_CASE("pwl judge: the engine's own diode law with circuit-scale tolerance",
          "[modes][pwl_ports]") {
    auto b = rectifier();
    const auto& g = b.graph();
    auto& pool = b.pool();
    pwl::DiodeEventState diodes(g, pool);
    const auto n = static_cast<Index>(pool.state_size(g));
    Vector x = Vector::Zero(n);
    const auto& d1 = diodes.entries()[0];
    events::CircuitScale sc;                 // 1 V, 1 A
    // D1 forward-biased by 0.5 V: OFF violates (0.5 V), ON is consistent
    x[d1.from] = 0.5;
    Real w = 0;
    int wj = -1;
    REQUIRE(pwl::judge_mode(diodes, {0, 0}, x, sc, 1e-9, &w, &wj) == 0);
    REQUIRE(wj == 0);
    REQUIRE(std::abs(w - 0.5) < 1e-12);
    REQUIRE(pwl::judge_mode(diodes, {1, 0}, x, sc, 1e-9) == 1);
    // reverse-biased: ON violates (current −g_on·0.5)
    x[d1.from] = -0.5;
    REQUIRE(pwl::judge_mode(diodes, {1, 0}, x, sc, 1e-9) == 0);
    REQUIRE(pwl::judge_mode(diodes, {0, 0}, x, sc, 1e-9) == 1);
}
