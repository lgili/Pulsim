// =============================================================================
// Test: SwitchingMode contract (Phase 1 of refactor-pwl-switching-engine)
// =============================================================================
//
// Validates the foundation for the PWL switching engine:
//   * SwitchingMode enum and resolve_switching_mode() helper.
//   * supports_pwl trait propagated through device_traits.
//   * Per-device pwl_state / commit_pwl_state / should_commute contract.
//   * Ideal-mode stamping is sharp (no derivative-of-conductance term).
//   * Behavioral-mode stamping retains legacy semantics (regression guard).
//
// These tests run against the header-only `pulsim::core` target. The PWL
// segment engine itself (Phase 2/3) is out of scope here; we only verify the
// device-level contract that the segment engine will consume.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/components/ideal_diode.hpp"
#include "pulsim/v1/components/ideal_switch.hpp"
#include "pulsim/v1/components/voltage_controlled_switch.hpp"
#include "pulsim/v1/components/mosfet.hpp"
#include "pulsim/v1/components/igbt.hpp"
#include "pulsim/v1/components/resistor.hpp"

#include <Eigen/Sparse>
#include <array>

using namespace pulsim::v1;
using Catch::Approx;

// -----------------------------------------------------------------------------
// SwitchingMode enum + resolve helper
// -----------------------------------------------------------------------------

TEST_CASE("SwitchingMode enum has compact storage and string mapping",
          "[switching_mode][api]") {
    STATIC_REQUIRE(sizeof(SwitchingMode) == 1);
    STATIC_REQUIRE(static_cast<int>(SwitchingMode::Auto) == 0);
    STATIC_REQUIRE(static_cast<int>(SwitchingMode::Ideal) == 1);
    STATIC_REQUIRE(static_cast<int>(SwitchingMode::Behavioral) == 2);

    REQUIRE(to_string(SwitchingMode::Auto) == "Auto");
    REQUIRE(to_string(SwitchingMode::Ideal) == "Ideal");
    REQUIRE(to_string(SwitchingMode::Behavioral) == "Behavioral");
}

TEST_CASE("resolve_switching_mode falls back deterministically",
          "[switching_mode][api]") {
    SECTION("explicit device mode wins over circuit default") {
        REQUIRE(resolve_switching_mode(SwitchingMode::Ideal,
                                       SwitchingMode::Behavioral) ==
                SwitchingMode::Ideal);
        REQUIRE(resolve_switching_mode(SwitchingMode::Behavioral,
                                       SwitchingMode::Ideal) ==
                SwitchingMode::Behavioral);
    }
    SECTION("Auto inherits from circuit default") {
        REQUIRE(resolve_switching_mode(SwitchingMode::Auto,
                                       SwitchingMode::Ideal) ==
                SwitchingMode::Ideal);
        REQUIRE(resolve_switching_mode(SwitchingMode::Auto,
                                       SwitchingMode::Behavioral) ==
                SwitchingMode::Behavioral);
    }
    SECTION("Auto + Auto resolves to Behavioral (Phase 1 backward-compat)") {
        REQUIRE(resolve_switching_mode(SwitchingMode::Auto,
                                       SwitchingMode::Auto) ==
                SwitchingMode::Behavioral);
    }
    SECTION("default circuit-default argument is Behavioral") {
        REQUIRE(resolve_switching_mode(SwitchingMode::Auto) ==
                SwitchingMode::Behavioral);
    }
}

// -----------------------------------------------------------------------------
// supports_pwl trait
// -----------------------------------------------------------------------------

TEST_CASE("supports_pwl trait labels switching devices correctly",
          "[switching_mode][traits]") {
    STATIC_REQUIRE(supports_pwl_v<IdealDiode>);
    STATIC_REQUIRE(supports_pwl_v<IdealSwitch>);
    STATIC_REQUIRE(supports_pwl_v<VoltageControlledSwitch>);
    STATIC_REQUIRE(supports_pwl_v<MOSFET>);
    STATIC_REQUIRE(supports_pwl_v<IGBT>);

    // Non-switching devices opt out by default.
    STATIC_REQUIRE_FALSE(supports_pwl_v<Resistor>);
}

// -----------------------------------------------------------------------------
// IdealDiode: Ideal-mode stamp is sharp; Behavioral-mode preserves tanh
// -----------------------------------------------------------------------------

TEST_CASE("IdealDiode SwitchingMode default is Auto",
          "[switching_mode][diode]") {
    IdealDiode d;
    REQUIRE(d.switching_mode() == SwitchingMode::Auto);
}

TEST_CASE("IdealDiode Ideal-mode forward stamp is exactly g_on, no smoothing",
          "[switching_mode][diode][ideal]") {
    IdealDiode d(/*g_on=*/1e3, /*g_off=*/1e-9);
    d.set_switching_mode(SwitchingMode::Ideal);
    d.commit_pwl_state(true);  // device "is on"

    Eigen::SparseMatrix<double> J(2, 2);
    J.setZero();
    Eigen::VectorXd f = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd x(2);
    x << 5.0, 0.0;  // strong forward bias

    std::array<Index, 2> nodes{0, 1};
    d.stamp_jacobian(J, f, x, nodes);
    J.makeCompressed();

    // Pure linear conductance: J(i,i) == g_on exactly. No v*dg/dv term.
    REQUIRE(J.coeff(0, 0) == Approx(1e3));
    REQUIRE(J.coeff(0, 1) == Approx(-1e3));
    REQUIRE(J.coeff(1, 0) == Approx(-1e3));
    REQUIRE(J.coeff(1, 1) == Approx(1e3));
    REQUIRE(f[0] == Approx(5000.0));
    REQUIRE(f[1] == Approx(-5000.0));
}

TEST_CASE("IdealDiode Ideal-mode off stamp uses g_off",
          "[switching_mode][diode][ideal]") {
    IdealDiode d(1e3, 1e-9);
    d.set_switching_mode(SwitchingMode::Ideal);
    d.commit_pwl_state(false);

    Eigen::SparseMatrix<double> J(2, 2);
    J.setZero();
    Eigen::VectorXd f = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd x(2);
    x << 5.0, 0.0;  // forward voltage applied but device committed off

    std::array<Index, 2> nodes{0, 1};
    d.stamp_jacobian(J, f, x, nodes);

    REQUIRE(J.coeff(0, 0) == Approx(1e-9));
    REQUIRE(J.coeff(1, 1) == Approx(1e-9));
    // Stamping must not flip the committed state.
    REQUIRE_FALSE(d.is_conducting());
}

TEST_CASE("IdealDiode Behavioral mode preserves smoothed transition",
          "[switching_mode][diode][behavioral]") {
    IdealDiode d(1e3, 1e-9);
    d.set_switching_mode(SwitchingMode::Behavioral);
    d.set_smoothing(0.1);

    SECTION("forward saturates near g_on (legacy regression)") {
        Eigen::SparseMatrix<double> J(2, 2);
        J.setZero();
        Eigen::VectorXd f = Eigen::VectorXd::Zero(2);
        Eigen::VectorXd x(2);
        x << 5.0, 0.0;

        std::array<Index, 2> nodes{0, 1};
        d.stamp_jacobian(J, f, x, nodes);
        REQUIRE(d.is_conducting());
        REQUIRE(J.coeff(0, 0) == Approx(1e3));
    }

    SECTION("near-threshold injects derivative-of-conductance term") {
        Eigen::SparseMatrix<double> J(2, 2);
        J.setZero();
        Eigen::VectorXd f = Eigen::VectorXd::Zero(2);
        Eigen::VectorXd x(2);
        x << 0.0, 0.0;  // exactly at threshold: tanh derivative is maximal

        std::array<Index, 2> nodes{0, 1};
        d.stamp_jacobian(J, f, x, nodes);

        // J(0,0) = g + v*dg/dv. At v=0, g = (g_on+g_off)/2 ≈ 500.
        // The dg/dv term goes to zero through `v` factor, so J is exactly g.
        REQUIRE(J.coeff(0, 0) == Approx(500.0).epsilon(1e-3));
    }
}

TEST_CASE("IdealDiode should_commute respects state and hysteresis",
          "[switching_mode][diode][events]") {
    IdealDiode d;
    d.set_switching_mode(SwitchingMode::Ideal);
    d.set_event_hysteresis(1e-6);  // 1 µV band

    SECTION("off device commutes when forward voltage exceeds hysteresis") {
        d.commit_pwl_state(false);
        PwlEventContext ctx;
        ctx.voltage = 1e-3;
        ctx.event_hysteresis = 0.0;
        REQUIRE(d.should_commute(ctx));
    }

    SECTION("off device stays off inside hysteresis band") {
        d.commit_pwl_state(false);
        PwlEventContext ctx;
        ctx.voltage = 1e-9;  // tiny forward bias inside band
        ctx.event_hysteresis = 0.0;
        REQUIRE_FALSE(d.should_commute(ctx));
    }

    SECTION("on device commutes when current goes negative beyond hysteresis") {
        d.commit_pwl_state(true);
        PwlEventContext ctx;
        ctx.current = -1.0;
        ctx.event_hysteresis = 0.0;
        REQUIRE(d.should_commute(ctx));
    }

    SECTION("on device stays on with positive current") {
        d.commit_pwl_state(true);
        PwlEventContext ctx;
        ctx.current = 5.0;
        ctx.event_hysteresis = 0.0;
        REQUIRE_FALSE(d.should_commute(ctx));
    }
}

// -----------------------------------------------------------------------------
// IdealSwitch: PWL by construction; should_commute returns false (externally
// commanded).
// -----------------------------------------------------------------------------

TEST_CASE("IdealSwitch exposes SwitchingMode but does not auto-commute",
          "[switching_mode][switch]") {
    IdealSwitch sw(1e6, 1e-12, false);
    REQUIRE(sw.switching_mode() == SwitchingMode::Auto);

    sw.set_switching_mode(SwitchingMode::Ideal);
    REQUIRE(sw.switching_mode() == SwitchingMode::Ideal);

    PwlEventContext ctx;
    ctx.control_voltage = 100.0;
    ctx.voltage = 50.0;
    ctx.current = 10.0;
    REQUIRE_FALSE(sw.should_commute(ctx));

    // pwl_state mirrors is_closed.
    REQUIRE_FALSE(sw.pwl_state());
    sw.commit_pwl_state(true);
    REQUIRE(sw.is_closed());
    REQUIRE(sw.pwl_state());
}

// -----------------------------------------------------------------------------
// VoltageControlledSwitch: Ideal vs Behavioral path
// -----------------------------------------------------------------------------

TEST_CASE("VoltageControlledSwitch Ideal stamp is sharp, no tanh derivative",
          "[switching_mode][vcswitch][ideal]") {
    VoltageControlledSwitch vcsw(/*v_th=*/2.5, /*g_on=*/1e3, /*g_off=*/1e-9);
    vcsw.set_switching_mode(SwitchingMode::Ideal);
    vcsw.commit_pwl_state(true);

    Eigen::SparseMatrix<double> J(3, 3);
    J.setZero();
    Eigen::VectorXd f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);
    x << 5.0, 1.0, 0.0;  // ctrl at 5V (above threshold), t1=1V, t2=0V

    std::array<Index, 3> nodes{0, 1, 2};
    vcsw.stamp_jacobian(J, f, x, nodes);

    // No control-coupling Jacobian entries in Ideal mode.
    REQUIRE(J.coeff(1, 0) == Approx(0.0));
    REQUIRE(J.coeff(2, 0) == Approx(0.0));
    REQUIRE(J.coeff(1, 1) == Approx(1e3));
    REQUIRE(J.coeff(1, 2) == Approx(-1e3));
}

TEST_CASE("VoltageControlledSwitch Behavioral stamp injects dg/dv_ctrl",
          "[switching_mode][vcswitch][behavioral]") {
    VoltageControlledSwitch vcsw(2.5, 1e3, 1e-9);
    vcsw.set_switching_mode(SwitchingMode::Behavioral);

    Eigen::SparseMatrix<double> J(3, 3);
    J.setZero();
    Eigen::VectorXd f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);
    x << 2.5, 1.0, 0.0;  // exactly at threshold: tanh derivative maximal

    std::array<Index, 3> nodes{0, 1, 2};
    vcsw.stamp_jacobian(J, f, x, nodes);

    // Control-coupling Jacobian entries are non-zero in Behavioral.
    REQUIRE(std::abs(J.coeff(1, 0)) > 0.0);
    REQUIRE(std::abs(J.coeff(2, 0)) > 0.0);
}

TEST_CASE("VoltageControlledSwitch should_commute crosses threshold",
          "[switching_mode][vcswitch][events]") {
    VoltageControlledSwitch vcsw(/*v_th=*/2.5);
    vcsw.set_event_hysteresis(0.0);

    SECTION("off device closes when control rises above threshold") {
        vcsw.commit_pwl_state(false);
        PwlEventContext ctx;
        ctx.control_voltage = 3.0;
        REQUIRE(vcsw.should_commute(ctx));
    }
    SECTION("on device opens when control drops below threshold") {
        vcsw.commit_pwl_state(true);
        PwlEventContext ctx;
        ctx.control_voltage = 1.0;
        REQUIRE(vcsw.should_commute(ctx));
    }
    SECTION("threshold band ignored without crossing") {
        vcsw.commit_pwl_state(false);
        PwlEventContext ctx;
        ctx.control_voltage = 2.5;  // exactly at threshold but not above
        REQUIRE_FALSE(vcsw.should_commute(ctx));
    }
}

// -----------------------------------------------------------------------------
// MOSFET: Ideal stamp is purely resistive; Behavioral keeps Shichman-Hodges
// -----------------------------------------------------------------------------

TEST_CASE("MOSFET Ideal-mode on-state is a pure linear conductance",
          "[switching_mode][mosfet][ideal]") {
    MOSFET::Params params{
        .vth = 2.0, .kp = 0.1, .lambda = 0.01,
        .g_off = 1e-12, .is_nmos = true, .g_on = 1e3,
    };
    MOSFET m(params, "M1");
    m.set_switching_mode(SwitchingMode::Ideal);
    m.commit_pwl_state(true);

    Eigen::SparseMatrix<double> J(3, 3);
    J.setZero();
    Eigen::VectorXd f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);
    x << 10.0, 5.0, 0.0;  // gate=10, drain=5, source=0

    std::array<Index, 3> nodes{0, 1, 2};
    m.stamp_jacobian(J, f, x, nodes);

    // Only drain-source path is stamped. No gm coupling (J(drain,gate) = 0).
    REQUIRE(J.coeff(1, 0) == Approx(0.0));
    REQUIRE(J.coeff(2, 0) == Approx(0.0));
    REQUIRE(J.coeff(1, 1) == Approx(1e3));
    REQUIRE(J.coeff(1, 2) == Approx(-1e3));
}

TEST_CASE("MOSFET Ideal-mode off uses g_off and ignores triode/saturation",
          "[switching_mode][mosfet][ideal]") {
    MOSFET::Params params{
        .vth = 2.0, .kp = 0.1, .lambda = 0.01,
        .g_off = 1e-12, .is_nmos = true, .g_on = 1e3,
    };
    MOSFET m(params, "M_off");
    m.set_switching_mode(SwitchingMode::Ideal);
    m.commit_pwl_state(false);

    Eigen::SparseMatrix<double> J(3, 3);
    J.setZero();
    Eigen::VectorXd f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);
    x << 10.0, 5.0, 0.0;  // gate above threshold but committed off

    std::array<Index, 3> nodes{0, 1, 2};
    m.stamp_jacobian(J, f, x, nodes);

    REQUIRE(J.coeff(1, 1) == Approx(1e-12));
    REQUIRE(J.coeff(2, 2) == Approx(1e-12));
}

TEST_CASE("MOSFET Behavioral mode keeps Shichman-Hodges regions",
          "[switching_mode][mosfet][behavioral]") {
    MOSFET m(/*vth=*/2.0, /*kp=*/0.1, /*is_nmos=*/true);
    m.set_switching_mode(SwitchingMode::Behavioral);

    Eigen::SparseMatrix<double> J(3, 3);
    J.setZero();
    Eigen::VectorXd f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);

    SECTION("cutoff: Vgs <= Vth") {
        x << 1.0, 5.0, 0.0;
        std::array<Index, 3> nodes{0, 1, 2};
        m.stamp_jacobian(J, f, x, nodes);
        REQUIRE_FALSE(m.is_conducting());
    }

    SECTION("on: Vgs > Vth") {
        x << 5.0, 5.0, 0.0;
        std::array<Index, 3> nodes{0, 1, 2};
        m.stamp_jacobian(J, f, x, nodes);
        REQUIRE(m.is_conducting());
    }
}

TEST_CASE("MOSFET should_commute uses Vgs vs vth with NMOS/PMOS sign",
          "[switching_mode][mosfet][events]") {
    SECTION("NMOS commute on rising Vgs") {
        MOSFET m(/*vth=*/2.0, /*kp=*/0.1, /*is_nmos=*/true);
        m.set_event_hysteresis(0.0);
        m.commit_pwl_state(false);
        PwlEventContext ctx;
        ctx.control_voltage = 3.0;  // Vgs > vth
        REQUIRE(m.should_commute(ctx));
    }
    SECTION("PMOS commute on falling Vgs (sign flipped)") {
        MOSFET m(/*vth=*/2.0, /*kp=*/0.1, /*is_nmos=*/false);
        m.set_event_hysteresis(0.0);
        m.commit_pwl_state(false);
        PwlEventContext ctx;
        ctx.control_voltage = -3.0;  // Vgs effectively +3 for PMOS
        REQUIRE(m.should_commute(ctx));
    }
}

// -----------------------------------------------------------------------------
// IGBT
// -----------------------------------------------------------------------------

TEST_CASE("IGBT Ideal stamp is purely linear collector-emitter",
          "[switching_mode][igbt][ideal]") {
    IGBT::Params params{.vth = 5.0, .g_on = 1e4, .g_off = 1e-12, .v_ce_sat = 1.5};
    IGBT ig(params, "Q1");
    ig.set_switching_mode(SwitchingMode::Ideal);
    ig.commit_pwl_state(true);

    Eigen::SparseMatrix<double> J(3, 3);
    J.setZero();
    Eigen::VectorXd f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);
    x << 15.0, 10.0, 0.0;

    std::array<Index, 3> nodes{0, 1, 2};
    ig.stamp_jacobian(J, f, x, nodes);

    REQUIRE(J.coeff(1, 1) == Approx(1e4));
    REQUIRE(J.coeff(1, 2) == Approx(-1e4));
    REQUIRE(J.coeff(2, 2) == Approx(1e4));
    // No saturation drop applied in Ideal mode: residual = g * vce.
    REQUIRE(f[1] == Approx(1e4 * 10.0));
    REQUIRE(f[2] == Approx(-1e4 * 10.0));
}

TEST_CASE("IGBT Behavioral mode preserves Vce_sat saturation drop",
          "[switching_mode][igbt][behavioral]") {
    IGBT::Params params{.vth = 5.0, .g_on = 1e4, .g_off = 1e-12, .v_ce_sat = 1.5};
    IGBT ig(params, "Q_beh");
    ig.set_switching_mode(SwitchingMode::Behavioral);

    Eigen::SparseMatrix<double> J(3, 3);
    J.setZero();
    Eigen::VectorXd f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);
    x << 15.0, 10.0, 0.0;  // Vge=15 > vth -> on

    std::array<Index, 3> nodes{0, 1, 2};
    ig.stamp_jacobian(J, f, x, nodes);

    REQUIRE(ig.is_conducting());
    REQUIRE(J.coeff(1, 1) == Approx(1e4).margin(1.0));
}
