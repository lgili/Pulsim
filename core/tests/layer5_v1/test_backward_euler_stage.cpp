// =============================================================================
// add-discontinuity-restart-rule — the BackwardEuler stage is the
// trapezoidal stamp at 2·h_e with the stored derivative left out.
//
// The oracle the design keeps (D3): for the linear companions and for
// each of the five stateful devices, stamping under BackwardEuler at
// h_e must equal stamping under Trapezoidal at 2·h_e from a history
// whose derivative slot was zeroed — same Jacobian (so the nominal
// factor serves), same residual. A missed stage would give a right
// matrix, a converging Newton and a wrong answer; this catches it.
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/history_state.hpp"
#include "pulsim/pwl/igbt_tail_history.hpp"
#include "pulsim/pwl/lauritzen_diode_history.hpp"
#include "pulsim/pwl/nonlinear_capacitor_history.hpp"
#include "pulsim/pwl/nonlinear_refresh_igbt_tail.hpp"
#include "pulsim/pwl/nonlinear_refresh_lauritzen_diode.hpp"
#include "pulsim/pwl/nonlinear_refresh_nonlinear_capacitor.hpp"
#include "pulsim/pwl/nonlinear_refresh_pmsm_mna.hpp"
#include "pulsim/pwl/nonlinear_refresh_saturable_inductor.hpp"
#include "pulsim/pwl/pmsm_mna_history.hpp"
#include "pulsim/pwl/saturable_inductor_history.hpp"

#include <cmath>

using namespace pulsim;
using pulsim::pwl::TrBdf2Stage;

namespace {

constexpr Real kHe = 0.37e-6;   // the half-step; the trapezoidal oracle runs at 2·kHe

Vector state(Size n, Real scale, Real phase) {
    Vector x(static_cast<Index>(n));
    for (Index i = 0; i < x.size(); ++i) {
        x[i] = scale * std::sin(phase + 0.7 * static_cast<Real>(i)) + 0.1 * scale;
    }
    return x;
}

bool close(const Vector& a, const Vector& b) {
    return (a - b).norm() <= 1e-14 * std::max(Real{1}, a.norm());
}

bool close(const sparse::Matrix& a, const sparse::Matrix& b) {
    const sparse::Matrix d = a - b;
    return d.norm() <= 1e-14 * std::max(Real{1}, a.norm());
}

struct Stamp {
    sparse::Matrix J;
    Vector f;
    explicit Stamp(Size n)
        : J(static_cast<Index>(n), static_cast<Index>(n)),
          f(Vector::Zero(static_cast<Index>(n))) {}
};

}  // namespace

TEST_CASE("BackwardEuler: linear companions and a transformer equal the "
          "zero-slot trapezoid at 2·h_e", "[v2][layer5_v1][restart]") {
    builder::CircuitBuilder b;
    b.add_capacitor("C", "a", "gnd", 1e-6);
    b.add_inductor("L", "a", "m", 1e-3);
    b.add_resistor("R", "m", "gnd", 1.0);
    b.add_transformer("T", "a", "gnd", "s", "gnd", 2e-3, 0.5e-3, 0.9);
    b.add_resistor("Rs", "s", "gnd", 3.0);
    const Size n = b.pool().state_size(b.graph());

    pwl::HistoryState h(b.graph(), b.pool());
    h.update_from_state(state(n, 4.0, 0.1), 2 * kHe);
    h.update_from_state(state(n, 5.0, 0.9), 2 * kHe);

    // The oracle: the same history with i_C and v_L zeroed.
    pwl::HistoryState z = h;
    auto flat = z.to_flat();
    for (Size i = 0; i < z.entries().size(); ++i) {
        // (v_prev, i_prev): a capacitor's slot is i_prev, an
        // inductor's v_prev.
        if (z.entries()[i].kind == pwl::DevicePool::StoredKind::Capacitor) {
            flat[2 * i + 1] = 0;
        } else {
            flat[2 * i] = 0;
        }
    }
    z.from_flat(flat);
    REQUIRE(z.transformer_couplings().size() == 1);

    Vector be, tr;
    h.compute_b_extra_backward_euler(kHe, be);
    z.compute_b_extra(2 * kHe, tr);
    REQUIRE(be.norm() > 0);
    CHECK(close(be, tr));

    const Vector x = state(n, 3.0, 2.2);
    h.update_from_state_backward_euler(x, kHe);
    z.update_from_state(x, 2 * kHe);
    for (Size i = 0; i < h.entries().size(); ++i) {
        CHECK(h.entries()[i].v_prev == z.entries()[i].v_prev);
        CHECK(std::abs(h.entries()[i].i_prev - z.entries()[i].i_prev)
              <= 1e-14 * std::abs(h.entries()[i].i_prev) + 1e-300);
    }
}

TEST_CASE("BackwardEuler: nonlinear capacitor stamp and commit",
          "[v2][layer5_v1][restart]") {
    builder::CircuitBuilder b;
    b.add_nonlinear_capacitor("Coss", "n", "gnd", 2e-9, 25.0, 0.5);
    b.add_resistor("R", "n", "gnd", 1e3);
    const Size n = b.pool().state_size(b.graph());
    pwl::NonlinearCapacitorHistory h;
    h.init(b.graph(), b.pool());
    h.update_from_state(state(n, 30.0, 0.3), 2 * kHe);
    h.update_from_state(state(n, 40.0, 0.5), 2 * kHe);
    auto z = h;
    auto flat = z.to_flat();
    flat[2] = 0;   // (v, q, i): the slot is i_prev
    z.from_flat(flat);

    const Vector x = state(n, 35.0, 1.1);
    Stamp a(n), c(n);
    pwl::refresh_nonlinear_capacitors(x, a.J, a.f, h, kHe,
                                      pwl::CossStage::BackwardEuler);
    pwl::refresh_nonlinear_capacitors(x, c.J, c.f, z, 2 * kHe,
                                      pwl::CossStage::Trapezoidal);
    CHECK(close(a.J, c.J));
    CHECK(close(a.f, c.f));

    h.update_from_state(x, kHe, /*backward_euler=*/true);
    z.update_from_state(x, 2 * kHe);
    CHECK(std::abs(h.entries()[0].i_prev - z.entries()[0].i_prev)
          <= 1e-14 * std::abs(h.entries()[0].i_prev));
}

TEST_CASE("BackwardEuler: saturable inductor keeps the conductance-form row",
          "[v2][layer5_v1][restart]") {
    builder::CircuitBuilder b;
    b.add_saturable_inductor("Ls", "a", "gnd", 1e-3, 2.0, 5e-5);
    b.add_resistor("R", "a", "gnd", 1.0);
    const Size n = b.pool().state_size(b.graph());
    pwl::SaturableInductorHistory h;
    h.init(b.graph(), b.pool());
    h.update_from_state(state(n, 3.0, 0.4));
    auto z = h;
    auto flat = z.to_flat();
    flat[2] = 0;   // (λ, i, V, H, M, dir, n): the slot is V_L_old
    z.from_flat(flat);
    REQUIRE(h.entries()[0].V_L_old != 0);

    const Vector x = state(n, 2.5, 1.3);
    Stamp a(n), c(n);
    pwl::stamp_saturable_inductor(h.entries()[0], x, a.J, a.f, kHe,
                                  TrBdf2Stage::BackwardEuler);
    pwl::stamp_saturable_inductor(z.entries()[0], x, c.J, c.f, 2 * kHe,
                                  TrBdf2Stage::Trapezoidal);
    CHECK(close(a.J, c.J));
    CHECK(close(a.f, c.f));
    // The diagonal is −(2/dt)·L(i), dt = 2·h_e: conductance form.
    const Index r = h.entries()[0].branch_var_id;
    const Real L = models::SaturableInductor::inductance(
        x[r], h.entries()[0].params);
    CHECK(std::abs(a.J.coeff(r, r) + L / kHe) <= 1e-14 * L / kHe);
}

TEST_CASE("BackwardEuler: Lauritzen diode and IGBT tail charge pairs",
          "[v2][layer5_v1][restart]") {
    builder::CircuitBuilder b;
    b.add_lauritzen_diode("D", "a", "gnd");
    b.add_igbt_level1("Q", "c", "gnd", "g", 1.5, 0.05, 5.0, 10.0, 0.01,
                      false, 1e-6, 0.3);
    b.add_resistor("Ra", "a", "gnd", 10.0);
    b.add_resistor("Rc", "c", "gnd", 10.0);
    b.add_resistor("Rg", "g", "gnd", 10.0);
    const Size n = b.pool().state_size(b.graph());
    pwl::LauritzenDiodeHistory hd;
    pwl::IgbtTailHistory ht;
    hd.init(b.graph(), b.pool());
    ht.init(b.graph(), b.pool());
    REQUIRE(!hd.empty());
    REQUIRE(!ht.empty());
    // Nodes in order of appearance: a, c, g. A forward-biased
    // junction and an IGBT gated on: both charges nonzero.
    Vector x0 = state(n, 0.6, 0.2);
    x0[0] = 0.65;
    x0[1] = 3.0;
    x0[2] = 12.0;
    Stamp s0(n);
    pwl::refresh_lauritzen_diodes(x0, s0.J, s0.f, hd, 2 * kHe);
    pwl::refresh_igbt_tails(x0, s0.J, s0.f, ht, 2 * kHe);
    hd.update_from_state(x0, 2 * kHe);
    ht.update_from_state(x0, 2 * kHe);

    auto zd = hd;
    auto zt = ht;
    auto fd = zd.to_flat();
    auto ft = zt.to_flat();
    fd[1] = 0;   // (q, f): the slot is f_prev
    ft[1] = 0;
    zd.from_flat(fd);
    zt.from_flat(ft);
    REQUIRE(hd.to_flat()[1] != 0);

    Vector x = state(n, 0.7, 0.9);
    x[1] = 2.0;
    x[2] = 12.0;
    Stamp a(n), c(n);
    pwl::refresh_lauritzen_diodes(x, a.J, a.f, hd, kHe,
                                  TrBdf2Stage::BackwardEuler);
    pwl::refresh_igbt_tails(x, a.J, a.f, ht, kHe,
                            TrBdf2Stage::BackwardEuler);
    pwl::refresh_lauritzen_diodes(x, c.J, c.f, zd, 2 * kHe,
                                  TrBdf2Stage::Trapezoidal);
    pwl::refresh_igbt_tails(x, c.J, c.f, zt, 2 * kHe,
                            TrBdf2Stage::Trapezoidal);
    CHECK(close(a.J, c.J));
    CHECK(close(a.f, c.f));
    CHECK(std::abs(hd.entries()[0].k0 - zd.entries()[0].k0)
          <= 1e-14 * std::abs(hd.entries()[0].k0));
    CHECK(std::abs(ht.entries()[0].k1 - zt.entries()[0].k1)
          <= 1e-14 * std::abs(ht.entries()[0].k1));
}

TEST_CASE("BackwardEuler: MNA-native PMSM flux rows",
          "[v2][layer5_v1][restart]") {
    builder::CircuitBuilder b;
    b.add_pmsm_mna("M", "a", "b", "c", "n", "w", "th",
                   0.2, 1e-3, 1.5e-3, 0.05, 4.0, 1e-3);
    b.add_resistor("Ra", "a", "gnd", 5.0);
    b.add_resistor("Rb", "b", "gnd", 5.0);
    b.add_resistor("Rc", "c", "gnd", 5.0);
    b.add_resistor("Rn", "n", "gnd", 5.0);
    b.add_resistor("Rw", "w", "gnd", 5.0);
    b.add_resistor("Rt", "th", "gnd", 5.0);
    const Size n = b.pool().state_size(b.graph());
    pwl::PmsmMnaHistory h;
    h.init(b.graph(), b.pool());
    h.update_from_state(state(n, 2.0, 0.6), 2 * kHe);
    // The oracle zeroes dλ_n = v − R·i by setting v_old = R·i_old.
    auto z = h;
    auto flat = z.to_flat();   // (λ_abc, i_abc, v_abc)
    for (Size k = 0; k < 3; ++k) flat[6 + k] = 0.2 * flat[3 + k];
    z.from_flat(flat);
    REQUIRE(h.entries()[0].dlam_old[0] != 0);

    const Vector x = state(n, 1.5, 1.7);
    Stamp a(n), c(n);
    pwl::refresh_pmsm_mna(x, a.J, a.f, h, kHe, TrBdf2Stage::BackwardEuler);
    pwl::refresh_pmsm_mna(x, c.J, c.f, z, 2 * kHe, TrBdf2Stage::Trapezoidal);
    CHECK(close(a.J, c.J));
    CHECK(close(a.f, c.f));
}
