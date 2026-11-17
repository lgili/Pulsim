// =============================================================================
// update-linear-time-assembly, tier 1 — stamp-slot map and shared symbolic
// analysis. Everything here is a BIT-identity check: the template assembly,
// the slot combine and the adopted analysis must reproduce the bytes of the
// fresh path, signed zeros included.
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/sparse/pulsim_lu_solver.hpp"

#include <cstring>
#include <vector>

using namespace pulsim;
using namespace pulsim::pwl;
using topology::SwitchStateMask;

namespace {

/// Two half-bridge legs with anti-parallel diodes (8 mask bits), an
/// R-L-C load, a coupled transformer, a current source and a saturable
/// inductor (its G_min stamp lands in the linear pattern).
builder::CircuitBuilder bridge() {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vdc", "p", "gnd", 400.0);
    b.add_switch("S1", "p", "a", 1e3, 1e-9);
    b.add_switch("S2", "a", "gnd", 1e3, 1e-9);
    b.add_switch("S3", "p", "c", 1e3, 1e-9);
    b.add_switch("S4", "c", "gnd", 1e3, 1e-9);
    b.add_diode("D1", "a", "p", 1e3, 1e-9, 0.0, 0.7);
    b.add_diode("D2", "gnd", "a", 1e3, 1e-9);
    b.add_diode("D3", "c", "p", 1e3, 1e-9);
    b.add_diode("D4", "gnd", "c", 1e3, 1e-9, 0.0, 0.7);
    b.add_resistor("R1", "a", "m", 2.0);
    b.add_inductor("L1", "m", "x", 1e-3);
    b.add_transformer("T1", "x", "c", "s", "gnd", 2e-3, 1e-3, 0.95);
    b.add_resistor("RL", "s", "gnd", 10.0);
    b.add_capacitor("Cs", "s", "gnd", 1e-6);
    b.add_current_source("Ib", "s", "gnd", 0.5);
    b.add_saturable_inductor("Lsat", "p", "q", 1e-3, 2.0);
    b.add_resistor("Rq", "q", "gnd", 100.0);
    return b;
}

SwitchStateMask mask_of(Size nbits, unsigned v) {
    SwitchStateMask m(nbits);
    for (Size i = 0; i < nbits; ++i) m.set(i, ((v >> i) & 1U) != 0U);
    return m;
}

bool same_bytes(const sparse::Matrix& A, const sparse::Matrix& B) {
    if (!A.isCompressed() || !B.isCompressed()) return false;
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    if (A.nonZeros() != B.nonZeros()) return false;
    const auto nnz = static_cast<std::size_t>(A.nonZeros());
    const auto no  = static_cast<std::size_t>(A.outerSize() + 1);
    return std::memcmp(A.outerIndexPtr(), B.outerIndexPtr(), no * sizeof(Index)) == 0 &&
           std::memcmp(A.innerIndexPtr(), B.innerIndexPtr(), nnz * sizeof(Index)) == 0 &&
           std::memcmp(A.valuePtr(), B.valuePtr(), nnz * sizeof(Real)) == 0;
}

bool same_bytes(const Vector& a, const Vector& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(),
                       static_cast<std::size_t>(a.size()) * sizeof(Real)) == 0;
}

}  // namespace

TEST_CASE("slot-map assembly is byte-equal to fresh assembly for every mask, both regimes",
          "[v2][layer4][assembly][slot_map]") {
    auto b = bridge();
    const auto& g = b.graph();
    const auto& pool = b.pool();
    const Size nbits = g.num_switches();
    REQUIRE(nbits == 8);
    SegmentAssembler sa;
    for (unsigned v = 0; v < (1U << nbits); ++v) {
        const auto mask = mask_of(nbits, v);
        for (const Real dt : {Real{0}, Real{1e-7}, Real{3.3e-6}}) {
            sparse::Matrix J_ref, J;
            Vector b_ref, bb;
            assemble_segment(g, pool, mask, dt, J_ref, b_ref);
            sparse::compress_in_place(J_ref);
            sa.assemble(g, pool, mask, dt, J, bb);
            REQUIRE(same_bytes(J, J_ref));
            REQUIRE(same_bytes(bb, b_ref));
        }
        sparse::Matrix G_ref, C_ref, G, C;
        Vector b_ref, bb;
        assemble_segment_split(g, pool, mask, G_ref, C_ref, b_ref);
        sa.split(g, pool, mask, G, C, bb);
        REQUIRE(same_bytes(G, G_ref));
        REQUIRE(same_bytes(C, C_ref));
        REQUIRE(same_bytes(bb, b_ref));
    }
    // One structure per regime, whatever the number of masks.
    REQUIRE(sa.structure_builds() == 2);
}

TEST_CASE("in-place combine equals G + (1/dt)·C byte for byte over 100 dt values",
          "[v2][layer4][assembly][slot_map]") {
    auto b = bridge();
    SegmentAssembler sa;
    const auto mask = mask_of(8, 0b10010110U);
    sparse::Matrix G, C, J;
    Vector bb;
    sa.split(b.graph(), b.pool(), mask, G, C, bb);
    Real dt = 1e-9;
    for (int k = 0; k < 100; ++k, dt *= 1.137) {
        const sparse::Matrix J_ref = G + (Real{1} / dt) * C;
        sa.combine(G, C, dt, J);
        REQUIRE(same_bytes(J, J_ref));
    }
}

TEST_CASE("scattered in-place combine equals G + (1/dt)·C byte for byte",
          "[v2][layer4][assembly][slot_map][event]") {
    // update-trbdf2-factor-reuse 3.1: the event entry's refactor path.
    auto b = bridge();
    SegmentAssembler sa;
    for (unsigned v = 0; v < 256; v += 17) {
        const auto mask = mask_of(8, v);
        sparse::Matrix G, C, J;
        Vector bb;
        std::vector<Real> gs, cs;
        sa.split(b.graph(), b.pool(), mask, G, C, bb);
        sa.combine(G, C, 1e-6, J);
        sa.scatter(G, C, gs, cs);
        REQUIRE(gs.size() == static_cast<std::size_t>(J.nonZeros()));
        Real dt = 1e-9;
        for (int k = 0; k < 100; ++k, dt *= 1.137) {
            const sparse::Matrix J_ref = G + (Real{1} / dt) * C;
            SegmentAssembler::combine_scattered(gs, cs, dt, J);
            REQUIRE(same_bytes(J, J_ref));
        }
    }
}

TEST_CASE("an adopted symbolic analysis gives the factor of a private one",
          "[v2][layer0][sparse][pulsim_lu][symbolic]") {
    auto b = bridge();
    SegmentAssembler sa;
    sparse::Matrix JA, JB;
    Vector bb;
    sa.assemble(b.graph(), b.pool(), mask_of(8, 0x0FU), 1e-7, JA, bb);
    sa.assemble(b.graph(), b.pool(), mask_of(8, 0xA5U), 1e-7, JB, bb);

    sparse::PulsimSparseLuSolver first;
    REQUIRE(first.analyze(JA));
    const auto sym = first.export_symbolic(JA);
    REQUIRE(sym);

    sparse::PulsimSparseLuSolver shared, priv;
    REQUIRE(shared.adopt_symbolic(*sym, JB));
    REQUIRE(priv.analyze(JB));
    REQUIRE(std::ranges::equal(shared.column_permutation(), priv.column_permutation()));
    REQUIRE(std::ranges::equal(shared.etree_parent(), priv.etree_parent()));
    REQUIRE(shared.factorize(JB));
    REQUIRE(priv.factorize(JB));
    REQUIRE(same_bytes(shared.extract_L_matrix(), priv.extract_L_matrix()));
    REQUIRE(same_bytes(shared.extract_U_matrix(), priv.extract_U_matrix()));

    // Another pattern is refused (the caller then analyses privately).
    sparse::Matrix JS;
    sa.assemble(b.graph(), b.pool(), mask_of(8, 0xA5U), Real{0}, JS, bb);
    sparse::PulsimSparseLuSolver other;
    REQUIRE_FALSE(other.adopt_symbolic(*sym, JS));
    REQUIRE_FALSE(other.is_analyzed());
}

TEST_CASE("a cache builds one structure and one analysis per regime",
          "[v2][layer4][assembly][symbolic][counters]") {
    auto b = bridge();
    const auto& g = b.graph();
    PwlStateSpaceCache cache(g, b.pool());
    constexpr Real dt = 1e-7;
    cache.build_lazy(dt);
    const Index n = static_cast<Index>(b.pool().state_size(g));
    const Vector be = Vector::LinSpaced(n, -1.0, 1.0);
    for (unsigned v = 0; v < 64; ++v) {
        const auto mask = mask_of(8, v * 3U);
        Vector x;
        cache.solve(mask, be, x);
        // Reference: fresh assembly, private analysis, same arithmetic.
        sparse::Matrix J;
        Vector bc;
        assemble_segment(g, b.pool(), mask, dt, J, bc);
        sparse::compress_in_place(J);
        sparse::PulsimSparseLuSolver ref;
        REQUIRE(ref.analyze(J));
        REQUIRE(ref.factorize(J));
        const Vector rhs = -(bc + be);
        Vector x_ref;
        ref.solve(rhs, x_ref);
        REQUIRE(same_bytes(x, x_ref));
    }
    auto m = cache.metrics();
    REQUIRE(m.structure_builds == 1);
    REQUIRE(m.symbolic_analyses == 1);
    REQUIRE(m.symbolic_fallbacks == 0);

    // Event entries at another dt share the dynamic structure and analysis.
    for (unsigned v = 0; v < 16; ++v) {
        Vector x;
        cache.solve_at(mask_of(8, v * 7U), 2.5e-7, be, x);
    }
    m = cache.metrics();
    REQUIRE(m.structure_builds == 1);
    REQUIRE(m.symbolic_analyses == 1);
    REQUIRE(m.symbolic_fallbacks == 0);
}

TEST_CASE("static and dynamic regimes: exactly two analyses",
          "[v2][layer4][assembly][symbolic][counters]") {
    // No inductor, so the static (dt = 0) system is solvable.
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "p", "gnd", 10.0);
    b.add_switch("S1", "p", "a", 1e3, 1e-9);
    b.add_switch("S2", "a", "gnd", 1e3, 1e-9);
    b.add_diode("D1", "gnd", "a", 1e3, 1e-9);
    b.add_resistor("R", "a", "o", 3.0);
    b.add_resistor("RL", "o", "gnd", 5.0);
    b.add_capacitor("C", "o", "gnd", 1e-6);
    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build_lazy(1e-7);
    const Vector be = Vector::Zero(
        static_cast<Index>(b.pool().state_size(b.graph())));
    for (unsigned v = 0; v < 8; ++v) {
        Vector x;
        cache.solve(mask_of(3, v), be, x);
        cache.solve_at(mask_of(3, v), Real{0}, be, x);
        cache.solve_at(mask_of(3, v), 3e-7, be, x);
    }
    const auto m = cache.metrics();
    REQUIRE(m.structure_builds == 2);
    REQUIRE(m.symbolic_analyses == 2);
    REQUIRE(m.symbolic_fallbacks == 0);
}

TEST_CASE("a structure change is detected: new template, private analysis, fallback counted",
          "[v2][layer4][assembly][symbolic][counters]") {
    auto b = bridge();
    PwlStateSpaceCache cache(b.graph(), b.pool());
    constexpr Real dt = 1e-7;
    cache.build_lazy(dt);
    const Index n = static_cast<Index>(b.pool().state_size(b.graph()));
    Vector x;
    cache.solve(mask_of(8, 1U), Vector::Zero(n), x);
    // A resistor between existing nodes: same unknowns, new entries.
    b.add_resistor("Rnew", "a", "s", 7.0);
    const auto mask = mask_of(8, 2U);
    cache.solve(mask, Vector::Zero(n), x);
    const auto m = cache.metrics();
    REQUIRE(m.structure_builds == 2);
    REQUIRE(m.symbolic_fallbacks == 1);
    REQUIRE(m.symbolic_analyses == 2);

    sparse::Matrix J;
    Vector bc;
    assemble_segment(b.graph(), b.pool(), mask, dt, J, bc);
    sparse::compress_in_place(J);
    sparse::PulsimSparseLuSolver ref;
    REQUIRE(ref.analyze(J));
    REQUIRE(ref.factorize(J));
    Vector x_ref;
    ref.solve(Vector(-bc), x_ref);
    REQUIRE(same_bytes(x, x_ref));
}

TEST_CASE("solve_rank1 builds no structure after the first call",
          "[v2][layer4][rank1][slot_map]") {
    auto b = bridge();
    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build_lazy(1e-7);
    const Index n = static_cast<Index>(b.pool().state_size(b.graph()));
    Vector x;
    cache.solve_rank1(mask_of(8, 0U), Vector::Zero(n), x);
    const auto builds = cache.metrics().structure_builds;
    const auto analyses = cache.metrics().symbolic_analyses;
    for (unsigned k = 1; k <= 1000; ++k) {
        const unsigned gray = (k ^ (k >> 1)) & 0xFFU;
        cache.solve_rank1(mask_of(8, gray), Vector::Zero(n), x);
        REQUIRE(x.allFinite());
    }
    REQUIRE(cache.metrics().structure_builds == builds);
    REQUIRE(cache.metrics().symbolic_analyses == analyses);
}

TEST_CASE("the assembler's per-branch lookups follow parameter updates and added devices",
          "[v2][layer4][assembly]") {
    // SegmentAssembler caches each branch's kind, parameter pointer and
    // branch-current unknown. An in-place update must be read through the
    // pointer; an added device (a pool insert moves the parameters) must
    // rebuild the table. Both checked byte for byte against a fresh build.
    auto b = bridge();
    SegmentAssembler sa;
    const auto mask = mask_of(8, 0xA5U);
    auto check = [&] {
        sparse::Matrix J, J_ref;
        Vector bb, b_ref;
        sa.assemble(b.graph(), b.pool(), mask, 1e-7, J, bb);
        assemble_segment(b.graph(), b.pool(), mask, 1e-7, J_ref, b_ref);
        sparse::compress_in_place(J_ref);
        REQUIRE(same_bytes(J, J_ref));
        REQUIRE(same_bytes(bb, b_ref));
    };
    check();
    b.pool().update_resistor_R(b.branch_id_of("R1"), 7.5);
    check();
    b.add_resistor("Rx", "m", "gnd", 33.0);
    b.add_inductor("Lx", "x", "gnd", 2e-3);
    check();
}
