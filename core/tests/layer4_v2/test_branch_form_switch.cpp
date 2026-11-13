// =============================================================================
// add-branch-form-ideal-switch tasks 3.3: the branch stamp (design D1, D2)
// =============================================================================
//
// A four-switch bridge, every switch in branch form with R_on = 0 and
// g_off = 1e-9 (each arm in series with its own resistor, so no mode is a
// loop of ideal switches). For all 16 masks: the assembled sparsity pattern
// (column pointers and row indices) is identical; the cache shares ONE
// symbolic analysis (no fallback); each mask's factor solves with a relative
// residual <= 1e-12, all-ON (R_on = 0, zero (s, s) diagonals) included; the
// single-flip `solve_rank1` path equals a full refactor within 1e-12.

#include <catch2/catch_test_macros.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/cache.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace pulsim;

namespace {

builder::CircuitBuilder bridge(Real g_off = 1e-9) {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "vs", "gnd", 100.0);
    b.add_resistor("Rs", "vs", "p", 0.1);
    b.add_capacitor("Cb", "p", "gnd", 10e-6);
    const char* arms[4][3] = {{"S1", "p", "x1"}, {"S2", "a", "x2"},
                              {"S3", "p", "x3"}, {"S4", "b", "x4"}};
    for (auto& a : arms) b.add_ideal_switch(a[0], a[1], a[2], 0.0, g_off);
    b.add_resistor("R1", "x1", "a", 0.05);
    b.add_resistor("R2", "x2", "gnd", 0.05);
    b.add_resistor("R3", "x3", "b", 0.05);
    b.add_resistor("R4", "x4", "gnd", 0.05);
    b.add_resistor("Rl", "a", "m", 10.0);
    b.add_inductor("Ll", "m", "b", 1e-3);
    return b;
}

topology::SwitchStateMask mask_of(unsigned bits) {
    topology::SwitchStateMask m(4);
    for (Size i = 0; i < 4; ++i) m.set(i, ((bits >> i) & 1u) != 0);
    return m;
}

}  // namespace

TEST_CASE("Branch form: one sparsity pattern for all 16 masks of a bridge",
          "[branch_form][d1]") {
    auto b = bridge();
    const auto& g = b.graph();
    auto& pool = b.pool();
    REQUIRE(pool.num_branch_switches() == 4);
    REQUIRE(pool.state_size(g) ==
            static_cast<Size>(g.num_nodes()) + 1 + 1 + 4);
    sparse::Matrix J0;
    Vector b0;
    pwl::assemble_segment(g, pool, mask_of(0), 1e-6, J0, b0);
    J0.makeCompressed();
    const auto nnz = J0.nonZeros();
    for (unsigned m = 1; m < 16; ++m) {
        sparse::Matrix J;
        Vector bv;
        pwl::assemble_segment(g, pool, mask_of(m), 1e-6, J, bv);
        J.makeCompressed();
        REQUIRE(J.nonZeros() == nnz);
        REQUIRE(std::equal(J.outerIndexPtr(), J.outerIndexPtr() + J.cols() + 1,
                           J0.outerIndexPtr()));
        REQUIRE(std::equal(J.innerIndexPtr(), J.innerIndexPtr() + nnz,
                           J0.innerIndexPtr()));
    }
    // All OFF: the (s, a) entries hold -g_off; all ON with R_on = 0 the
    // (s, s) diagonals are stored zeros.
    sparse::Matrix Jon;
    Vector bon;
    pwl::assemble_segment(g, pool, mask_of(15), 1e-6, Jon, bon);
    const Index s0 = pool.branch_var_id_for_switch(b.branch_id_of("S1"), g);
    REQUIRE(Jon.coeff(s0, s0) == 0.0);
}

TEST_CASE("Branch form: every mask factorises on the shared analysis",
          "[branch_form][d2]") {
    auto b = bridge();
    const auto& g = b.graph();
    auto& pool = b.pool();
    pwl::PwlStateSpaceCache cache(g, pool);
    cache.build_lazy(1e-6);
    const auto n = static_cast<Index>(pool.state_size(g));
    Vector rhs = Vector::LinSpaced(n, 1.0, 2.0);
    // Two residuals per mask: the circuit's own right-hand side relative to
    // it (the task's ||Jx - b|| / ||b||), and the normwise backward error
    // ||r|| / (||J|| ||x|| + ||b||) under a 1..2 A injection into EVERY row,
    // floating islands included (all-OFF leaves x1, a, m, b, x3 tied by
    // g_off = 1e-9 alone: ||r|| / ||b|| reaches 8e-8 there by conditioning).
    const Vector zero = Vector::Zero(n);
    for (unsigned m = 0; m < 16; ++m) {
        const auto& seg = cache.lookup(mask_of(m));
        Vector x(n);
        cache.solve(mask_of(m), zero, x);
        const Real rel = (seg.J * x + seg.b_constant).norm() / seg.b_constant.norm();
        cache.solve(mask_of(m), rhs, x);
        const Vector b = seg.b_constant + rhs;
        const Real be = (seg.J * x + b).norm() /
            (Eigen::MatrixXd(seg.J).norm() * x.norm() + b.norm());
        INFO("mask " << m << " residual " << rel << " backward error " << be);
        REQUIRE(rel <= 1e-12);
        REQUIRE(be <= 1e-15);
    }
    // Pivot growth max|U| / max|A| of dense partial pivoting per mask
    // (the in-house LU does not expose its U; recorded, 3.3).
    Real growth = 0;
    for (unsigned m = 0; m < 16; ++m) {
        const Eigen::MatrixXd A(cache.lookup(mask_of(m)).J);
        const Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
        const Eigen::MatrixXd U = lu.matrixLU().triangularView<Eigen::Upper>();
        growth = std::max(growth, U.cwiseAbs().maxCoeff() / A.cwiseAbs().maxCoeff());
    }
    std::printf("[branch_form] max pivot growth over 16 masks: %.6g\n", growth);
    REQUIRE(growth <= 2.0);
    const auto met = cache.metrics();
    REQUIRE(met.symbolic_analyses == 1);
    REQUIRE(met.symbolic_fallbacks == 0);

    // Single flips: the sliding rank-one path against the per-mask factor.
    const unsigned walk[] = {0, 1, 3, 7, 15, 14, 12, 8, 9, 13, 5, 4, 6, 2, 10, 11};
    for (unsigned m : walk) {
        Vector x1(n), x2(n);
        cache.solve_rank1(mask_of(m), rhs, x1);
        cache.solve(mask_of(m), rhs, x2);
        REQUIRE((x1 - x2).norm() <= 1e-12 * x2.norm());
    }
    const auto m2 = cache.metrics();
    INFO("rank1_hits " << m2.rank1_hits << " full " << m2.full_refactor_hits
                       << " fallbacks " << m2.fallbacks);
    REQUIRE(m2.rank1_hits + m2.full_refactor_hits + m2.multi_bit_rank1_hits >= 15);
}

TEST_CASE("Branch form with R_on, g_off reproduces the conductance form",
          "[branch_form][d1]") {
    // RL behind a switch: the DC solution of the closed and open circuit.
    for (bool closed : {true, false}) {
        builder::CircuitBuilder c, br;
        for (auto* bb : {&c, &br}) {
            bb->add_voltage_source("V", "vin", "gnd", 10.0);
            bb->add_resistor("R", "a", "gnd", 1.0);
        }
        c.add_switch("S", "vin", "a", 1e3, 1e-9);
        br.add_ideal_switch("S", "vin", "a", 1e-3, 1e-9);
        topology::SwitchStateMask m(1);
        m.set(0, closed);
        sparse::Matrix Jc, Jb;
        Vector bc, bb;
        pwl::assemble_segment(c.graph(), c.pool(), m, Jc, bc);
        pwl::assemble_segment(br.graph(), br.pool(), m, Jb, bb);
        const Vector xc = Eigen::MatrixXd(Jc).fullPivLu().solve(-bc);
        const Vector xb = Eigen::MatrixXd(Jb).fullPivLu().solve(-bb);
        REQUIRE(std::abs(xc[c.node_id_of("a")] - xb[br.node_id_of("a")]) <= 1e-12 * std::max(1.0, std::abs(xc[c.node_id_of("a")])));
    }
}

// Measurement for tasks 3.2 (hidden: run with "[rank1_measure]"): the per-mask
// factor cache against the sliding rank-one path on a 12-bit three-phase
// inverter in branch form, single-flip walk. Prints ns/solve and counters;
// asserts only that both paths agree.
TEST_CASE("Branch form: rank-one vs per-mask factor (measurement)",
          "[.][rank1_measure]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vdc", "dc", "gnd", 400.0);
    for (const char* ph : {"a", "b", "c"}) {
        const std::string p = ph;
        b.add_ideal_switch("S" + p + "h", "dc", p, 1e-3, 1e-9);
        b.add_diode_branch("D" + p + "h", p, "dc", 1e-3, 1e-9);
        b.add_ideal_switch("S" + p + "l", p, "gnd", 1e-3, 1e-9);
        b.add_diode_branch("D" + p + "l", "gnd", p, 1e-3, 1e-9);
        b.add_resistor("R" + p, p, "m" + p, 5.0);
        b.add_inductor("L" + p, "m" + p, "star", 2e-3);
    }
    const auto& g = b.graph();
    auto& pool = b.pool();
    const auto n = static_cast<Index>(pool.state_size(g));
    const Size ns = g.num_switches();
    Vector rhs = Vector::LinSpaced(n, 1.0, 2.0);
    // A single-flip walk that revisits masks (as a converter does).
    std::vector<topology::SwitchStateMask> walk;
    topology::SwitchStateMask m(ns);
    std::uint64_t s = 12345;
    for (int k = 0; k < 4000; ++k) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        const Size bit = static_cast<Size>((s >> 33) % 6) * 2;  // switches only
        m.set(bit, !m.get(bit));
        walk.push_back(m);
    }
    auto run = [&](bool rank1, Real& ns_per) {
        pwl::PwlStateSpaceCache cache(g, pool);
        cache.build_lazy(1e-6);
        Vector x(n);
        Real best = 1e30;
        for (int rep = 0; rep < 5; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            for (const auto& mk : walk) {
                if (rank1) cache.solve_rank1(mk, rhs, x);
                else cache.solve(mk, rhs, x);
            }
            const auto t1 = std::chrono::steady_clock::now();
            best = std::min(best, std::chrono::duration<Real, std::nano>(t1 - t0).count());
        }
        ns_per = best / static_cast<Real>(walk.size());
        return cache.metrics();
    };
    Real a = 0, r = 0, a2 = 0, r2 = 0;
    const auto ma = run(false, a);
    const auto mr = run(true, r);
    run(false, a2);
    run(true, r2);
    std::printf("[rank1_measure] n=%ld per-mask: %.0f/%.0f ns/solve (best of 5, A/B/A/B); "
                "rank-one: %.0f/%.0f ns/solve; rank1_hits=%llu full=%llu multi=%llu fallbacks=%llu "
                "symbolic=%llu\n",
                static_cast<long>(n), a, a2, r, r2,
                (unsigned long long)mr.rank1_hits, (unsigned long long)mr.full_refactor_hits,
                (unsigned long long)mr.multi_bit_rank1_hits, (unsigned long long)mr.fallbacks,
                (unsigned long long)ma.symbolic_analyses);
    SUCCEED();
}
