// =============================================================================
// Layer 4 — gmin conductance floor + gmin stepping
// =============================================================================
//
// v2.0 Phase 2 (B.2), audit finding `no-gmin-infrastructure`.
//
// Two bars these tests hold:
//   * the floor must be electrically invisible AND must never stand
//     in for a topology defect;
//   * the ramp must rescue a circuit the direct solve cannot solve,
//     and land on the same answer every other route lands on.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/dc_assemble.hpp"
#include "pulsim/pwl/dc_operating_point.hpp"
#include "pulsim/pwl/dc_strategy.hpp"
#include "pulsim/models/ideal_diode.hpp"
#include "pulsim/pwl/gmin.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/solver/bdf1.hpp"
#include "pulsim/sparse/matrix.hpp"

#include <string>

using namespace pulsim;
using namespace pulsim::pwl;
using Catch::Approx;

namespace {

bool has(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

/// R + L to ground: the inductor gives us a branch-current row to
/// prove gmin stays away from.
builder::CircuitBuilder rl_circuit() {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b.add_resistor("R1", "vin", "vout", 4.0);
    b.add_inductor("L1", "vout", "gnd", 1e-3);
    return b;
}

/// A chain of smooth diodes sharp enough that plain Newton from the
/// linear warm start cannot close. This is the circuit gmin stepping
/// exists for.
builder::CircuitBuilder stiff_diode_chain(Size n = 12,
                                            Real kappa = 50.0) {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "vin", "gnd", 20.0);
    b.add_resistor("R", "vin", "n0", 100.0);
    for (Size i = 0; i < n; ++i) {
        models::IdealDiode::Params p;
        p.kappa = kappa;
        const std::string from = "n" + std::to_string(i);
        const std::string to =
            (i + 1 == n) ? std::string{"gnd"}
                          : ("n" + std::to_string(i + 1));
        b.add_nonlinear_diode("D" + std::to_string(i), from, to, p);
    }
    // Exactly what simulate() and compute_dc_op() do before solving:
    // the chain's interior nodes touch nothing that conducts at DC,
    // so preflight gives each one a 1 GΩ reference. Without this the
    // fixture would be testing a circuit no user ever runs.
    (void)b.run_preflight();
    return b;
}

}  // namespace

TEST_CASE("stamp_gmin touches node rows and nothing else",
          "[v2][layer4][gmin]") {
    auto b = rl_circuit();
    const auto& g = b.graph();
    const auto& pool = b.pool();
    const Index n_nodes = g.num_nodes();
    const Index n_total = static_cast<Index>(pool.state_size(g));
    REQUIRE(n_total > n_nodes);   // there ARE branch rows to protect

    topology::SwitchStateMask m(0);
    sparse::Matrix J_plain, J_gmin;
    Vector b_plain, b_gmin;
    dc_assemble(g, pool, m, J_plain, b_plain, 0.0, Real{0});
    dc_assemble(g, pool, m, J_gmin,  b_gmin,  0.0, Real{1e-3});
    sparse::compress_in_place(J_plain);
    sparse::compress_in_place(J_gmin);

    for (Index i = 0; i < n_total; ++i) {
        for (Index j = 0; j < n_total; ++j) {
            const Real want = J_plain.coeff(i, j) +
                ((i == j && i < n_nodes) ? Real{1e-3} : Real{0});
            INFO("entry (" << i << ", " << j << ")");
            REQUIRE(J_gmin.coeff(i, j) == Approx(want).margin(1e-18));
        }
    }
    // The right-hand side is a property of the sources; a shunt to
    // ground is not a source.
    REQUIRE((b_gmin - b_plain).cwiseAbs().maxCoeff() == Approx(0.0));
}

TEST_CASE("gmin does not cancel the inductor's DC branch epsilon",
          "[v2][layer4][gmin]") {
    // `dc_assemble` puts -1e-12 on every inductor's branch-current
    // diagonal, and the default floor is +1e-12: the SAME magnitude,
    // opposite sign. A gmin that walked the whole diagonal instead
    // of the node block would cancel it to exactly zero and make the
    // row rank-deficient — every inductor DC test would start
    // throwing, for a reason nobody would guess from the message.
    // Pin it.
    auto b = rl_circuit();
    const auto& g = b.graph();
    const auto& pool = b.pool();
    // Branch 2 is L1 (0 = Vin, 1 = R1).
    const Index l_row = pool.branch_var_id_for_inductor(2, g);
    REQUIRE(l_row >= g.num_nodes());

    topology::SwitchStateMask m(0);
    sparse::Matrix J;
    Vector rhs;
    dc_assemble(g, pool, m, J, rhs, 0.0, kDefaultGmin);
    sparse::compress_in_place(J);
    REQUIRE(J.coeff(l_row, l_row) == Approx(-1e-12).epsilon(1e-12));
}

TEST_CASE("The gmin floor is electrically invisible",
          "[v2][layer4][gmin]") {
    auto b = rl_circuit();
    topology::SwitchStateMask m(0);
    const Vector x_off =
        compute_dc_op(b.graph(), b.pool(), m, 0.0, Real{0});
    const Vector x_on =
        compute_dc_op(b.graph(), b.pool(), m, 0.0, kDefaultGmin);
    REQUIRE(x_off.size() == x_on.size());
    const Real scale = x_off.cwiseAbs().maxCoeff();
    REQUIRE(scale > Real{1});
    REQUIRE((x_on - x_off).cwiseAbs().maxCoeff() / scale <
             Real{1e-9});
}

TEST_CASE("The floor never substitutes for a missing equation",
          "[v2][layer4][gmin][diagnostics]") {
    // `vfloat` hangs off nothing but a capacitor, so its DC column is
    // empty. A conductance to ground would give it a diagonal, the
    // factorization would succeed, and the user would be handed
    // v = 0 for a node with no defined voltage — the exact silent
    // wrong answer Phase 1 taught the kernel to name. The structural
    // probe must run FIRST and the named error must still win.
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b.add_resistor("R1", "vin", "gnd", 10.0);
    b.add_capacitor("Cfloat", "vin", "vfloat", 1e-6);
    topology::SwitchStateMask m(0);

    // Prove the premise: with the floor stamped the matrix IS
    // factorizable, so only the probe stands between the user and a
    // fabricated answer.
    sparse::Matrix J;
    Vector rhs;
    dc_assemble(b.graph(), b.pool(), m, J, rhs, 0.0, kDefaultGmin);
    sparse::compress_in_place(J);
    REQUIRE(sparse::first_empty_column(J) == kInvalidIndex);

    for (const Real gmin : {Real{0}, kDefaultGmin, Real{1e-6}}) {
        bool threw = false;
        try {
            (void)compute_dc_op(b.graph(), b.pool(), m, 0.0, gmin);
        } catch (const std::runtime_error& e) {
            threw = true;
            const std::string msg = e.what();
            INFO("gmin = " << gmin << " message: " << msg);
            REQUIRE(has(msg, "vfloat"));
        }
        INFO("gmin = " << gmin);
        REQUIRE(threw);
    }

    // Same rule for the stepped solver: it refuses rather than
    // papering over, and says so.
    bool threw = false;
    try {
        (void)compute_dc_op_gmin_stepped(b.graph(), b.pool(), m);
    } catch (const std::runtime_error& e) {
        threw = true;
        const std::string msg = e.what();
        INFO(msg);
        REQUIRE(has(msg, "vfloat"));
        REQUIRE(has(msg, "gmin cannot substitute"));
    }
    REQUIRE(threw);
}

TEST_CASE("gmin_ramp descends by decades and lands on the floor",
          "[v2][layer4][gmin]") {
    GminConfig cfg;              // start 1e-2, floor 1e-12, 10 steps
    const auto ramp = gmin_ramp(cfg);
    REQUIRE(ramp.size() == 11);
    REQUIRE(ramp.front() == Approx(1e-2));
    REQUIRE(ramp.back() == Approx(cfg.floor));
    for (Size k = 1; k < ramp.size(); ++k) {
        INFO("rung " << k);
        REQUIRE(ramp[k] < ramp[k - 1]);
    }
    // One decade per rung with the defaults.
    for (Size k = 1; k + 1 < ramp.size(); ++k) {
        INFO("rung " << k);
        REQUIRE(ramp[k - 1] / ramp[k] == Approx(10.0).epsilon(1e-9));
    }

    // Degenerate configs still yield a usable single-rung ramp.
    GminConfig none;
    none.steps = 0;
    REQUIRE(gmin_ramp(none).size() == 1);
    GminConfig inverted;
    inverted.start = 1e-15;      // below the floor
    REQUIRE(gmin_ramp(inverted).size() == 1);
    REQUIRE(gmin_ramp(inverted).front() == Approx(inverted.floor));
}

TEST_CASE("source_scale ramps independent sources, not dependent ones",
          "[v2][layer4][gmin][homotopy]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 10.0);
    b.add_resistor("R1", "vin", "gnd", 1000.0);
    b.add_current_source("Iin", "gnd", "nb", 1e-3);
    b.add_resistor("R2", "nb", "gnd", 1000.0);
    topology::SwitchStateMask m(0);

    const Vector x_full =
        compute_dc_op(b.graph(), b.pool(), m, 0.0, Real{0});
    REQUIRE(x_full[0] == Approx(10.0));

    // α = 0 kills every excitation, so the only solution is zero.
    sparse::Matrix J;
    Vector rhs;
    dc_assemble(b.graph(), b.pool(), m, J, rhs, 0.0, Real{0},
                 /*source_scale=*/0.0);
    REQUIRE(rhs.cwiseAbs().maxCoeff() == Approx(0.0));

    // α = 0.5 halves both the voltage and the current source.
    sparse::Matrix J_half;
    Vector rhs_half, rhs_one;
    dc_assemble(b.graph(), b.pool(), m, J_half, rhs_half, 0.0,
                 Real{0}, 0.5);
    dc_assemble(b.graph(), b.pool(), m, J, rhs_one, 0.0, Real{0},
                 1.0);
    REQUIRE((rhs_half - 0.5 * rhs_one).cwiseAbs().maxCoeff() ==
             Approx(0.0).margin(1e-15));
    // Scaling excitation must not touch the matrix.
    sparse::compress_in_place(J);
    sparse::compress_in_place(J_half);
    REQUIRE((sparse::Matrix(J - J_half)).norm() ==
             Approx(0.0).margin(1e-18));
}

TEST_CASE("gmin stepping agrees with the direct solve when both work",
          "[v2][layer4][gmin]") {
    auto b = rl_circuit();
    topology::SwitchStateMask m(0);
    const Vector direct =
        compute_dc_op(b.graph(), b.pool(), m, 0.0, kDefaultGmin);
    DCSolveReport report;
    const Vector stepped = compute_dc_op_gmin_stepped(
        b.graph(), b.pool(), m, {}, 0.0, GminConfig{}, &report);
    REQUIRE((stepped - direct).cwiseAbs().maxCoeff() ==
             Approx(0.0).margin(1e-12));
    REQUIRE(report.strategy == DCStrategy::GminStepping);
    // No nonlinear devices → the ramp collapses to one solve at the
    // floor, because a direct solve has no basin to miss.
    REQUIRE(report.rungs_attempted == 1);
    REQUIRE(report.final_gmin == Approx(kDefaultGmin));
}

TEST_CASE("gmin stepping solves a chain the direct solve cannot",
          "[v2][layer4][gmin][integration]") {
    // THE point of the rung. 12 smooth diodes with a sharp knee:
    // Newton from the linear warm start diverges, the conductance
    // homotopy walks in.
    auto b = stiff_diode_chain();
    topology::SwitchStateMask m(b.graph().num_switches());
    const auto refresh = make_combined_diode_mosfet_refresh();

    bool direct_threw = false;
    try {
        (void)compute_dc_op_newton(b.graph(), b.pool(), m, refresh);
    } catch (const std::exception&) {
        direct_threw = true;
    }
    REQUIRE(direct_threw);        // the premise of the test

    DCSolveReport report;
    const Vector x = compute_dc_op_gmin_stepped(
        b.graph(), b.pool(), m, refresh, 0.0, GminConfig{}, &report);
    REQUIRE(x.allFinite());
    REQUIRE(report.strategy == DCStrategy::GminStepping);
    REQUIRE(report.rungs_attempted >= 11);
    // 12 forward drops of ~0.70 V each.
    REQUIRE(x[0] == Approx(20.0));               // the source node
    REQUIRE(x[1] == Approx(8.4).margin(0.2));    // top of the chain

    // The residual reported is against the ORIGINAL circuit, and the
    // answer must satisfy it — not merely the regularized one.
    REQUIRE(report.residual < 1e-6);

    // Source stepping, an independent route, must land in the same
    // place. Two homotopies agreeing is far stronger evidence than
    // either one converging.
    DCSolveReport ss_report;
    const Vector x_ss = compute_dc_op_with_strategy(
        b.graph(), b.pool(), m, DCStrategy::SourceStepping, 0.0,
        PseudoTransientConfig{}, SourceSteppingConfig{},
        analysis::ShouldContinueFn{}, refresh, GminConfig{},
        &ss_report);
    REQUIRE(ss_report.strategy == DCStrategy::SourceStepping);
    REQUIRE((x_ss - x).cwiseAbs().maxCoeff() < 1e-6);
}

TEST_CASE("Auto falls through to the rung that can answer",
          "[v2][layer4][gmin][integration]") {
    auto b = stiff_diode_chain();
    topology::SwitchStateMask m(b.graph().num_switches());
    const auto refresh = make_combined_diode_mosfet_refresh();

    DCSolveReport report;
    const Vector x = compute_dc_op_with_strategy(
        b.graph(), b.pool(), m, DCStrategy::Auto, 0.0,
        PseudoTransientConfig{}, SourceSteppingConfig{},
        analysis::ShouldContinueFn{}, refresh, GminConfig{},
        &report);
    REQUIRE(x.allFinite());
    // Naive is rung 1 and fails here, so the report must NOT claim it.
    REQUIRE(report.strategy != DCStrategy::Naive);
    REQUIRE(x[1] == Approx(8.4).margin(0.2));
}

TEST_CASE("The DC operating point honours nonlinear devices",
          "[v2][layer4][gmin][dc_op]") {
    // The regression that motivated `dc_operating_point.hpp`: the raw
    // linear solve treats a diode as an open circuit and answers
    // 5.000 V where the truth is ~0.70 V, with no warning.
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "vin", "gnd", 5.0);
    b.add_resistor("R", "vin", "na", 1000.0);
    b.add_nonlinear_diode("D", "na", "gnd",
                            models::IdealDiode::Params{});
    topology::SwitchStateMask m(b.graph().num_switches());

    const Vector linear = compute_dc_op(b.graph(), b.pool(), m);
    REQUIRE(linear[1] == Approx(5.0).margin(1e-6));   // the old lie

    const auto refresh = make_combined_diode_mosfet_refresh();
    const auto op = compute_dc_operating_point(
        b.graph(), b.pool(), m, refresh);
    REQUIRE(op.x[1] == Approx(0.70).margin(0.02));
    REQUIRE(op.report.strategy == DCStrategy::Naive);
}

TEST_CASE("BDF1 refuses a circuit it would silently open",
          "[v2][layer4][gmin][bdf1]") {
    // run_transient_bdf1 has no Newton loop at all, so a nonlinear
    // device would be an open circuit for the WHOLE run, not just at
    // DC. Refusing beats a plausible wrong answer.
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "vin", "gnd", 5.0);
    b.add_resistor("R", "vin", "na", 1000.0);
    b.add_nonlinear_diode("D", "na", "gnd",
                            models::IdealDiode::Params{});

    solver::SimulationOptions opts;
    opts.t_start = 0.0;
    opts.t_end = 1e-5;
    opts.dt = 1e-6;
    solver::SwitchScheduleFn sw = [](Real) {
        return topology::SwitchStateMask(0);
    };
    bool threw = false;
    try {
        (void)solver::run_transient_bdf1(b, opts, sw);
    } catch (const std::invalid_argument& e) {
        threw = true;
        const std::string msg = e.what();
        INFO(msg);
        REQUIRE(has(msg, "no Newton iteration"));
        REQUIRE(has(msg, "trapezoidal"));
    }
    REQUIRE(threw);
}
