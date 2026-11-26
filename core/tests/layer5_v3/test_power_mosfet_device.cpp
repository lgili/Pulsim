// add-power-mosfet-datasheet-device 5.1 / 5.2: the tabulated channel and
// the power MOSFET as one f/q device.
//
// Truths: the table's own grid values (interpolation reproduces knots), the
// two branch formulas of the terminal swap evaluated independently at
// V_DS = 0 (C¹), central differences of the residual for every Jacobian
// entry (the device's AD never checks itself), and the host's contract check.

#include "pulsim/models/channel_table.hpp"
#include "pulsim/models/power_mosfet_device.hpp"
#include "pulsim/xdev/checks.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using pulsim::Real;
using pulsim::models::ChannelTable;
namespace pmd = pulsim::models::power_mosfet_device;

namespace {

// A square-law-like synthetic table (Level-1, K = 1.43, V_T = 3) on a coarse grid.
ChannelTable square_law_table() {
    std::vector<Real> vgs{2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0};
    std::vector<Real> vds{0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0};
    std::vector<std::vector<Real>> I(vgs.size(), std::vector<Real>(vds.size()));
    for (std::size_t i = 0; i < vgs.size(); ++i) {
        const Real vov = std::max(Real{0}, vgs[i] - 3.0);
        for (std::size_t j = 0; j < vds.size(); ++j) {
            const Real v = std::min(vds[j], vov);
            I[i][j] = 1.43 * (2.0 * vov * v - v * v) * (1.0 + 0.01 * vds[j]);
        }
    }
    return ChannelTable(vgs, vds, I);
}

Real rel(Real a, Real b) { return std::abs(a - b) / std::max({std::abs(a), std::abs(b), 1e-300}); }

}  // namespace

TEST_CASE("ChannelTable reproduces its grid and its first-quadrant gradient",
          "[power_mosfet][channel_table]") {
    const auto t = square_law_table();
    for (std::size_t i = 0; i < t.v_gs().size(); ++i) {
        for (std::size_t j = 0; j < t.v_ds().size(); ++j) {
            CHECK(t.current(t.v_gs()[i], t.v_ds()[j]).i == t.current_grid()[i][j]);
        }
    }
    // gradient = central difference of the value (inside cells and outside the grid)
    for (const Real vgs : {2.7, 4.4, 7.3, 11.1, 13.5, 1.0}) {
        for (const Real vds : {0.3, 1.7, 6.0, 12.0, 20.0}) {
            const auto e = t.current(vgs, vds);
            const Real h = 1e-6;
            const Real fg = (t.current(vgs + h, vds).i - t.current(vgs - h, vds).i) / (2 * h);
            const Real fd = (t.current(vgs, vds + h).i - t.current(vgs, vds - h).i) / (2 * h);
            CHECK(std::abs(e.di_dvgs - fg) <= 1e-6 * (1 + std::abs(fg)));
            CHECK(std::abs(e.di_dvds - fd) <= 1e-6 * (1 + std::abs(fd)));
        }
    }
}

TEST_CASE("ChannelTable third quadrant is C1 at V_DS = 0 (task 5.1)",
          "[power_mosfet][channel_table]") {
    const auto t = square_law_table();
    for (int k = 0; k <= 200; ++k) {
        const Real vgs = 1.0 + 0.06 * k;   // 1 .. 13 V, inside and past the grid
        const auto r = t.forward(vgs, 0.0);   // the V_DS >= 0 branch at 0
        const auto l = t.reverse(vgs, 0.0);   // the terminal-swap branch at 0
        CHECK(r.i == 0.0);
        CHECK(l.i == 0.0);
        CHECK(r.di_dvgs == 0.0);
        CHECK(l.di_dvgs == 0.0);
        CHECK(rel(r.di_dvds, l.di_dvds) <= 1e-12);
        // and through the selector, a picovolt either side
        const auto a = t.current(vgs, 1e-12), b = t.current(vgs, -1e-12);
        CHECK(std::abs(a.i + b.i) <= 1e-9 * std::abs(a.i) + 1e-300);
        CHECK(rel(a.di_dvds, b.di_dvds) <= 1e-9);
    }
    // the swap itself: i(V_GS, -x) = -I(V_GS + x, x)
    CHECK(t.current(6.0, -1.5).i == -t.forward(7.5, 1.5).i);
}

TEST_CASE("ChannelTable refuses a non-monotone table naming the point",
          "[power_mosfet][channel_table]") {
    std::vector<Real> vgs{3.0, 4.0, 5.0}, vds{0.0, 1.0, 2.0, 4.0};
    std::vector<std::vector<Real>> I{{0, 0, 0, 0}, {0, 1.0, 1.5, 1.6}, {0, 2.0, 3.0, 3.2}};
    auto bad = I;
    bad[2][3] = 2.9;   // falls along V_DS at V_GS = 5
    try {
        ChannelTable(vgs, vds, bad);
        FAIL("not refused");
    } catch (const pulsim::NetlistError& e) {
        const std::string m = e.what();
        CHECK(m.find("not monotone in V_DS") != std::string::npos);
        CHECK(m.find("V_GS = 5") != std::string::npos);
        CHECK(m.find("[2][3]") != std::string::npos);
        CHECK(e.device() == "ChannelTable");
    }
    bad = I;
    bad[1][2] = 3.1;   // above the V_GS = 5 row at V_DS = 2 (row 1 stays monotone)
    bad[1][3] = 3.15;
    try {
        ChannelTable(vgs, vds, bad);
        FAIL("not refused");
    } catch (const pulsim::NetlistError& e) {
        CHECK(std::string(e.what()).find("not monotone in V_GS at V_DS = 2") != std::string::npos);
        CHECK(std::string(e.what()).find("[2][2]") != std::string::npos);
    }
    bad = I;
    bad[1][0] = 1e-3;
    CHECK_THROWS_AS(ChannelTable(vgs, vds, bad), pulsim::NetlistError);
    CHECK_THROWS_AS(ChannelTable(vgs, {0.1, 1.0, 2.0, 4.0}, I), pulsim::NetlistError);
    CHECK_THROWS_AS(ChannelTable({3.0, 3.0, 5.0}, vds, I), pulsim::NetlistError);
    CHECK_NOTHROW(ChannelTable(vgs, vds, I));
}

namespace {
pmd::Core full_core(bool table) {
    pmd::Core c;
    c.ch = {1.43, 3.0, 0.01, 15.0};
    if (table) c.table = std::make_shared<const ChannelTable>(square_law_table());
    c.gd = pulsim::models::NonlinearCapacitor::Params{};
    c.gd->law = pulsim::models::ChargeLawKind::VdmosCgd;
    c.gd->cgd_max = 1e-9;
    c.gd->cgd_min = 50e-12;
    c.gd->a = 0.5;
    c.ds = pulsim::models::NonlinearCapacitor::junction(2e-9, 25.0, 0.5);
    c.c_gd = 30e-12;   // constant parts on top of the laws (core_of uses one or the other)
    c.c_ds = 40e-12;
    c.body = pulsim::models::PowerMosfet::BodyDiode::Lauritzen;
    c.la = pulsim::device::laws::LauritzenDiode::from(
        pulsim::models::LauritzenDiode::Params{.I_S = 1e-12, .n = 1.0, .V_T = 0.025852,
                                              .tau = 50e-9, .T_M = 10e-9, .G_min = 1e-12});
    return c;
}
}  // namespace

TEST_CASE("power MOSFET device: Jacobian = central differences of its residual",
          "[power_mosfet][device]") {
    for (const bool table : {false, true}) {
        auto c = full_core(table);
        c.pat = pmd::pattern(c);
        const std::vector<std::array<Real, 4>> pts{
            {400.0, 0.0, 0.0, 0.0},     {0.4, 0.0, 12.0, 0.0},  {-0.7, 0.0, 0.0, 1e-8},
            {-0.3, 0.0, 6.0, 2e-9},     {5.0, 0.0, 4.2, 0.0},   {1e-3, 0.0, 3.1, 0.0},
            {30.0, 1.0, 9.0, 0.0},      {-0.5, 0.2, 5.5, 5e-9}};
        // (v_SD <= 0.7 V: at 2.5 V forward the junction's exp(96) makes the
        // central difference itself meaningless, not the AD)
        for (const auto& x : pts) {
            const auto d = pmd::evaluate(c, x.data());
            for (std::size_t k = 0; k < 4; ++k) {
                const Real h = (k == pmd::QM ? 1e-15 : 1e-6) * std::max(1.0, std::abs(x[k]));
                auto xp = x, xm = x;
                xp[k] += h;
                xm[k] -= h;
                const auto p = pmd::evaluate(c, xp.data()), m = pmd::evaluate(c, xm.data());
                for (std::size_t r = 0; r < 4; ++r) {
                    const Real fdf = (p.f[r] - m.f[r]) / (2 * h);
                    const Real fdq = (p.q[r] - m.q[r]) / (2 * h);
                    // + the difference's own round-off: eps·|f| / h
                    const Real nf = 8e-16 * std::max(std::abs(p.f[r]), std::abs(m.f[r])) / h;
                    const Real nq = 8e-16 * std::max(std::abs(p.q[r]), std::abs(m.q[r])) / h;
                    const Real sf = std::max({std::abs(fdf), std::abs(d.df[r][k]), 1e-12});
                    const Real sq = std::max({std::abs(fdq), std::abs(d.dq[r][k]), 1e-21});
                    INFO("table " << table << " x = " << x[0] << "," << x[1] << "," << x[2] << ","
                                  << x[3] << " row " << r << " col " << k);
                    CHECK(std::abs(d.df[r][k] - fdf) <= 1e-5 * sf + nf);
                    CHECK(std::abs(d.dq[r][k] - fdq) <= 1e-5 * sq + nq);
                }
            }
            // every entry outside the declared pattern is zero
            for (std::size_t r = 0; r < 4; ++r) {
                for (std::size_t k = 0; k < 4; ++k) {
                    bool in_f = false, in_q = false;
                    for (std::size_t e = 0; e < c.pat.fr.size(); ++e)
                        in_f |= c.pat.fr[e] == r && c.pat.fc[e] == k;
                    for (std::size_t e = 0; e < c.pat.qr.size(); ++e)
                        in_q |= c.pat.qr[e] == r && c.pat.qc[e] == k;
                    if (!in_f) CHECK(d.df[r][k] == 0.0);
                    if (!in_q) CHECK(d.dq[r][k] == 0.0);
                }
            }
        }
    }
}

TEST_CASE("power MOSFET device: descriptor, and the contract check's known refusal",
          "[power_mosfet][device]") {
    for (const bool lauritzen : {true, false}) {
        auto c = full_core(false);
        if (!lauritzen) c.body = pulsim::models::PowerMosfet::BodyDiode::None;
        auto [model, prm] = pmd::make_model(c);
        const pulsim::xdev::Instance in("M", model, prm);
        CHECK(in.rows() == (lauritzen ? 4u : 3u));
        // KNOWN GAP (measured 2026-11-26): xdev::check_contract's heuristics
        // over its +-1e3 V probe grid refuse this device even for the bare
        // Level-1 channel with constant capacitances (secant vs mean tangent
        // over 1 V cells at v_DS = -15 V: 30.54 vs 30.08 .. 30.87; a 5e-18
        // tangent "jump" where the row scale is ~0), and with the diode
        // exponential or the junction floor (DECISIONS #26). The builder
        // runs it with check = false; the Jacobian test above is the truth.
        CHECK_THROWS_WITH(pulsim::xdev::check_contract(in),
                          Catch::Matchers::ContainsSubstring("external device 'M'"));
    }
}
