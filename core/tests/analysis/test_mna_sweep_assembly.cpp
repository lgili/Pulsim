// =============================================================================
// fix-mna-sweep-correctness — the sweep's matrices are the transient's
// =============================================================================
//
// Before the fix the sweep built its mass matrix E with a private
// stamper that knew capacitors and inductors only; a coupled pair's
// mutual terms were missing (|H| = 0 for a k = 0.99 transformer whose
// exact |H| is 0.0617 / 0.3885 / 0.4949 at 100 Hz / 1 kHz / 10 kHz), and
// it silently treated Newton branches as absent.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/analysis/mna_sweep.hpp"
#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pulsim;
using Complex = std::complex<Real>;

namespace {

builder::CircuitBuilder coupled_rlc() {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", Real{0});
    b.add_resistor("Rs", "in", "a", Real{10});
    b.add_transformer("T", "a", "gnd", "s", "gnd", 1e-3, 1e-3, 0.99);
    b.add_resistor("RL", "s", "gnd", Real{10});
    b.add_capacitor("C", "s", "gnd", 1e-7);
    b.add_inductor("L", "a", "m", 1e-4);
    b.add_resistor("Rm", "m", "gnd", Real{100});
    return b;
}

}  // namespace

TEST_CASE("sweep E equals C/2 of the transient split, mutual terms included",
          "[analysis][mna_sweep][fix-mna-sweep-correctness]") {
    auto b = coupled_rlc();
    topology::SwitchStateMask mask(b.graph().num_switches());
    const auto sm = analysis::detail::assemble_sweep_matrices(b.graph(), b.pool(), mask);
    sparse::Matrix G, C;
    Vector rhs;
    pwl::assemble_segment_split(b.graph(), b.pool(), mask, G, C, rhs);
    REQUIRE(sm.E.rows() == C.rows());
    int n_offdiag_inductive = 0;
    for (Index i = 0; i < C.rows(); ++i) {
        for (Index j = 0; j < C.cols(); ++j) {
            REQUIRE(sm.E.coeff(i, j) == Real{0.5} * C.coeff(i, j));
            REQUIRE(sm.G.coeff(i, j) == G.coeff(i, j));
            if (i != j && sm.E.coeff(i, j) < Real{0}) ++n_offdiag_inductive;
        }
    }
    // The capacitor contributes no negative off-diagonal on a grounded
    // node; the two -M entries of the coupled pair must be there.
    REQUIRE(n_offdiag_inductive == 2);
}

TEST_CASE("sweep of a coupled transformer matches the closed form",
          "[analysis][mna_sweep][fix-mna-sweep-correctness]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", Real{0});
    b.add_resistor("Rs", "in", "a", Real{10});
    b.add_transformer("T", "a", "gnd", "s", "gnd", 1e-3, 1e-3, 0.99);
    b.add_resistor("RL", "s", "gnd", Real{10});
    topology::SwitchStateMask mask(b.graph().num_switches());
    const std::vector<Real> f{100.0, 1e3, 1e4};
    const auto src = b.pool().branch_var_id_for_source(0, b.graph());
    const auto res = analysis::run_mna_sweep(
        b.graph(), b.pool(), mask, f, static_cast<Size>(src),
        static_cast<Size>(b.node_id_of("s")));
    const Real M = 0.99e-3;
    for (std::size_t k = 0; k < f.size(); ++k) {
        const Real w = 2 * std::numbers::pi * f[k];
        const Complex jw{0, w};
        const Complex H = jw * M * Real{10} /
            ((Real{10} + jw * 1e-3) * (Real{10} + jw * 1e-3) + w * w * M * M);
        INFO("f = " << f[k] << "  H = " << res.H[k] << "  exact = " << H);
        REQUIRE(std::abs(res.H[k] - H) <= 1e-9 * std::abs(H));
    }
}

TEST_CASE("sweep of RC, RL and RLC matches the closed form to 1e-12",
          "[analysis][mna_sweep][fix-mna-sweep-correctness]") {
    // Series R, shunt (L + 1 ohm) || C || RL at the output.
    const Real R = 50, L = 1e-3, Cv = 1e-6, RL = 1e3;
    for (int variant = 0; variant < 3; ++variant) {
        builder::CircuitBuilder b;
        b.add_voltage_source("V", "in", "gnd", Real{0});
        b.add_resistor("R", "in", "o", R);
        b.add_resistor("RL", "o", "gnd", RL);
        if (variant != 0) {   // RL, RLC
            b.add_inductor("L", "o", "m", L);
            b.add_resistor("Rm", "m", "gnd", Real{1});
        }
        if (variant != 1) {   // RC, RLC
            b.add_capacitor("C", "o", "gnd", Cv);
        }
        topology::SwitchStateMask mask(b.graph().num_switches());
        const std::vector<Real> f{10.0, 1e3, 1e5};
        const auto src = b.pool().branch_var_id_for_source(0, b.graph());
        const auto res = analysis::run_mna_sweep(
            b.graph(), b.pool(), mask, f, static_cast<Size>(src),
            static_cast<Size>(b.node_id_of("o")));
        for (std::size_t k = 0; k < f.size(); ++k) {
            const Complex jw{0, 2 * std::numbers::pi * f[k]};
            Complex Y = Real{1} / RL;
            if (variant != 0) Y += Real{1} / (Real{1} + jw * L);
            if (variant != 1) Y += jw * Cv;
            const Complex Z = Real{1} / Y;
            const Complex H = Z / (R + Z);
            INFO("variant " << variant << " f = " << f[k]);
            REQUIRE(std::abs(res.H[k] - H) <= 1e-12 * std::abs(H));
        }
    }
}

TEST_CASE("sweep refuses a Newton branch by name and kind",
          "[analysis][mna_sweep][fix-mna-sweep-correctness]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", Real{1.7});
    b.add_resistor("R", "in", "a", Real{1});
    b.add_shockley_diode("Dj", "a", "gnd");
    topology::SwitchStateMask mask(b.graph().num_switches());
    const auto src = b.pool().branch_var_id_for_source(0, b.graph());
    try {
        (void)analysis::run_mna_sweep(b.graph(), b.pool(), mask, {100.0},
                                      static_cast<Size>(src),
                                      static_cast<Size>(b.node_id_of("a")));
        FAIL("expected a refusal");
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        INFO(msg);
        REQUIRE(msg.find("'Dj'") != std::string::npos);
        REQUIRE(msg.find("ShockleyDiode") != std::string::npos);
    }
}
