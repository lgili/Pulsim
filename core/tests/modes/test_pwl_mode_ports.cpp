// add-complementarity-mode-selection task 4.2 on pwl — the diode k-port of a
// fixed-step decision site (pwl/mode_ports.hpp) against an extended-precision
// solve of the same discrete system (DECISIONS F1).
#include <sstream>
#include <string>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/mode_ports.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <vector>

using namespace pulsim;

namespace {

// A two-diode rectifier with a floating DC bus held by a tie to ground: with
// both diodes OFF the bus {p, n} is an island whose common mode the double
// factor rounds to eps·(C/h·v)/G_tie.
builder::CircuitBuilder rectifier(Real r_tie) {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 100.0);
    b.add_diode("D1", "in", "p", 1e3, 1e-9);
    b.add_diode("D2", "n", "gnd", 1e3, 1e-9);
    b.add_capacitor("C", "p", "n", 1e-3);
    b.add_resistor("R", "p", "n", 50.0);
    b.add_resistor("Rg", "n", "gnd", r_tie);
    b.add_inductor("L", "in", "gnd", 1e-3);
    return b;
}

// Two such buses, each on its own tie, joined by a 1 MΩ resistor: with every
// diode OFF they are two islands that see each other through a weak edge.
builder::CircuitBuilder two_buses(Real r_tie) {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 100.0);
    b.add_inductor("L", "in", "gnd", 1e-3);
    for (const char* k : {"1", "2"}) {
        const std::string p = std::string("p") + k, n = std::string("n") + k;
        b.add_diode(std::string("Da") + k, "in", p, 1e3, 1e-9);
        b.add_diode(std::string("Db") + k, n, "gnd", 1e3, 1e-9);
        b.add_capacitor(std::string("C") + k, p, n, k[0] == '1' ? 1e-3 : 4.7e-4);
        b.add_resistor(std::string("R") + k, p, n, k[0] == '1' ? 50.0 : 120.0);
        b.add_resistor(std::string("Rg") + k, n, "gnd", k[0] == '1' ? r_tie : 3.3 * r_tie);
    }
    b.add_resistor("Rx", "p1", "p2", 1e6);
    return b;
}

Real vp(const Vector& s, Index a, Index c) {
    return stamping::read_node_voltage(s, a) - stamping::read_node_voltage(s, c);
}

// ---- double-double arithmetic (~32 digits): long double is double on arm64 --
struct DD {
    double hi = 0, lo = 0;
};
DD two_sum(double a, double b) {
    const double s = a + b, bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}
DD norm(double hi, double lo) { return two_sum(hi, lo); }
DD operator+(DD a, DD b) {
    DD s = two_sum(a.hi, b.hi);
    return norm(s.hi, s.lo + a.lo + b.lo);
}
DD operator-(DD a) { return {-a.hi, -a.lo}; }
DD operator-(DD a, DD b) { return a + (-b); }
DD operator*(DD a, DD b) {
    const double p = a.hi * b.hi;
    const double e = std::fma(a.hi, b.hi, -p);
    return norm(p, e + a.hi * b.lo + a.lo * b.hi);
}
DD operator/(DD a, DD b) {
    const double q1 = a.hi / b.hi;
    const DD r = a - b * DD{q1, 0};
    const double q2 = r.hi / b.hi;
    const DD r2 = r - b * DD{q2, 0};
    return norm(q1, q2) + DD{r2.hi / b.hi, 0};
}
DD dd(double x) { return {x, 0}; }

// Gaussian elimination with partial pivoting in DD; A is overwritten.
std::vector<DD> dd_solve(std::vector<std::vector<DD>> A, std::vector<DD> b) {
    const std::size_t n = b.size();
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t p = k;
        for (std::size_t i = k + 1; i < n; ++i)
            if (std::abs(A[i][k].hi) > std::abs(A[p][k].hi)) p = i;
        std::swap(A[k], A[p]);
        std::swap(b[k], b[p]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const DD l = A[i][k] / A[k][k];
            for (std::size_t j = k; j < n; ++j) A[i][j] = A[i][j] - l * A[k][j];
            b[i] = b[i] - l * b[k];
        }
    }
    std::vector<DD> x(n);
    for (std::size_t i = n; i-- > 0;) {
        DD s = b[i];
        for (std::size_t j = i + 1; j < n; ++j) s = s - A[i][j] * x[j];
        x[i] = s / A[i][i];
    }
    return x;
}

// J(mask, h) = G + C/h in DD, every node-row diagonal summed exactly from the
// branches (the double assembly rounds g_off + 1/R + …, which moves the
// island's common mode by the same order as the factor's rounding).
std::vector<std::vector<DD>> dd_matrix(const topology::Graph& g, const pwl::DevicePool& pool,
                                       const topology::SwitchStateMask& mask, Real h) {
    sparse::Matrix G, C;
    Vector bc;
    pwl::assemble_segment_split(g, pool, mask, G, C, bc);
    const DenseMatrix Gd(G), Cd(C);
    const auto n = static_cast<std::size_t>(Gd.rows());
    const Index nn = g.num_nodes();
    std::vector<DD> ground(static_cast<std::size_t>(nn));   // conductance to ground
    Size sw = 0;
    for (Index b = 0; b < g.num_branches(); ++b) {
        const auto& br = g.branch(b);
        Real gg = 0;
        if (br.kind == topology::BranchKind::Switch) {
            const auto& p = pool.diode_params(b);
            gg = mask.get(sw++) ? p.g_on : p.g_off;
        } else if (pool.kind_of(b) == pwl::DevicePool::StoredKind::Resistor) {
            gg = pool.resistor_params(b).G;
        } else {
            continue;
        }
        if (br.from >= 0 && br.to < 0) ground[static_cast<std::size_t>(br.from)] = ground[static_cast<std::size_t>(br.from)] + dd(gg);
        if (br.to >= 0 && br.from < 0) ground[static_cast<std::size_t>(br.to)] = ground[static_cast<std::size_t>(br.to)] + dd(gg);
    }
    std::vector<std::vector<DD>> A(n, std::vector<DD>(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A[i][j] = dd(Gd(static_cast<Index>(i), static_cast<Index>(j)));
            if (Cd(static_cast<Index>(i), static_cast<Index>(j)) != 0)
                A[i][j] = A[i][j] + dd(Cd(static_cast<Index>(i), static_cast<Index>(j))) / dd(h);
        }
        if (static_cast<Index>(i) >= nn) continue;
        DD diag = ground[i];
        Real csum = 0;
        for (Index j = 0; j < nn; ++j) {
            if (static_cast<std::size_t>(j) == i) continue;
            diag = diag - dd(Gd(static_cast<Index>(i), j));
            csum -= Cd(static_cast<Index>(i), j);
        }
        // this circuit has no capacitor to ground: C's diagonal is one element
        REQUIRE(csum == Cd(static_cast<Index>(i), static_cast<Index>(i)));
        A[i][i] = diag + dd(csum) / dd(h);
    }
    return A;
}

DD dd_vp(const std::vector<DD>& s, Index a, Index c) {
    return (a >= 0 ? s[static_cast<std::size_t>(a)] : DD{}) -
           (c >= 0 ? s[static_cast<std::size_t>(c)] : DD{});
}

}  // namespace

// u and Z of one site against the double-double solve of the same system.
void check_site(const topology::Graph& g, pwl::DevicePool& pool, pwl::PwlModePorts& ports,
                const topology::SwitchStateMask& mask, Real h, bool expect_fix,
                Real& worst_u, Real& worst_z, const std::string& what) {
    pwl::DiodeEventState diodes(g, pool);
    const auto& es = diodes.entries();
    const auto n = static_cast<Index>(pool.state_size(g));
    sparse::Matrix G, C;
    Vector bc;
    pwl::assemble_segment_split(g, pool, mask, G, C, bc);
    // a step-start state and its backward-Euler history −(C/h)·x0
    const Vector x0 = Vector::LinSpaced(n, 3.0, 97.0);
    const Vector b_extra = -(C * x0) / h;
    DenseMatrix Z;
    Vector u;
    const Size fixes0 = ports.island_corrections();
    ports.site(mask, h, b_extra, Z, u);
    INFO(what);
    REQUIRE((ports.island_corrections() > fixes0) == expect_fix);
    const auto A = dd_matrix(g, pool, mask, h);
    std::vector<DD> rhs(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) rhs[static_cast<std::size_t>(i)] = -(dd(bc[i]) + dd(b_extra[i]));
    const auto s = dd_solve(A, rhs);
    for (std::size_t j = 0; j < es.size(); ++j) {
        const DD ref = dd_vp(s, es[j].from, es[j].to);
        const Real err = std::abs((dd(u[static_cast<Index>(j)]) - ref).hi);
        worst_u = std::max(worst_u, err);
        INFO("j=" << j << " u=" << u[static_cast<Index>(j)] << " err=" << err);
        CHECK(err <= 1e-10);
    }
    // Z: J·z = −E_k, Z(i, k) = −v_i(z)
    const auto k_ports = static_cast<Index>(es.size());
    Real zmax = 0;
    DenseMatrix Zref(k_ports, k_ports);
    for (std::size_t k = 0; k < es.size(); ++k) {
        std::vector<DD> e(static_cast<std::size_t>(n));
        if (es[k].from >= 0) e[static_cast<std::size_t>(es[k].from)] = dd(-1);
        if (es[k].to >= 0) e[static_cast<std::size_t>(es[k].to)] = dd(1);
        const auto z = dd_solve(A, e);
        for (std::size_t i = 0; i < es.size(); ++i) {
            Zref(static_cast<Index>(i), static_cast<Index>(k)) = -dd_vp(z, es[i].from, es[i].to).hi;
            zmax = std::max(zmax, std::abs(Zref(static_cast<Index>(i), static_cast<Index>(k))));
        }
    }
    const Real zerr = (Z - Zref).cwiseAbs().maxCoeff() / zmax;
    worst_z = std::max(worst_z, zerr);
    INFO("Z rel err=" << zerr);
    CHECK(zerr <= 1e-12);
}

TEST_CASE("pwl port model: u and Z against a double-double solve of the site's system",
          "[modes][pwl_ports]") {
    const Real dt = 1e-5;
    Real worst_u = 0, worst_z = 0;
    for (const Real r_tie : {1e4, 1e5, 1e6, 1e7, 1e8, 1e9}) {
        auto b = rectifier(r_tie);
        const auto& g = b.graph();
        auto& pool = b.pool();
        pwl::PwlStateSpaceCache cache(g, pool);
        cache.build_lazy(dt);
        pwl::DiodeEventState diodes(g, pool);
        pwl::PwlModePorts ports(cache, g, pool, diodes);
        REQUIRE(ports.branch_form_diode().empty());
        REQUIRE(ports.ports().size() == 2);
        const auto& es = diodes.entries();
        for (const bool d1_on : {false, true}) {   // floating island / bus tied to the source
            topology::SwitchStateMask mask(g.num_switches());
            mask.set(es[0].switch_idx, d1_on);
            for (const Real f : {Real{1}, Real{0.5}, Real{1e-1}, Real{1e-2}, Real{1e-3}}) {
                std::ostringstream w;
                w << "rectifier tie=" << r_tie << " d1_on=" << d1_on << " f=" << f;
                check_site(g, pool, ports, mask, dt * f,
                           !d1_on && r_tie >= 1.0 / pwl::PwlModePorts::kWeakResistor,
                           worst_u, worst_z, w.str());
            }
        }
    }
    INFO("worst |u - u_ref| = " << worst_u << " V, worst Z rel = " << worst_z);
    CHECK(worst_u <= 1e-10);
}

TEST_CASE("pwl port model: two islands joined by a weak edge are corrected jointly",
          "[modes][pwl_ports]") {
    const Real dt = 1e-5;
    Real worst_u = 0, worst_z = 0;
    for (const Real r_tie : {1e5, 1e7, 1e9}) {
        auto b = two_buses(r_tie);
        const auto& g = b.graph();
        auto& pool = b.pool();
        pwl::PwlStateSpaceCache cache(g, pool);
        cache.build_lazy(dt);
        pwl::DiodeEventState diodes(g, pool);
        pwl::PwlModePorts ports(cache, g, pool, diodes);
        REQUIRE(ports.ports().size() == 4);
        const topology::SwitchStateMask mask(g.num_switches());   // all OFF: two islands
        for (const Real f : {Real{1}, Real{1e-1}, Real{1e-3}}) {
            std::ostringstream w;
            w << "two buses tie=" << r_tie << " f=" << f;
            check_site(g, pool, ports, mask, dt * f, true, worst_u, worst_z, w.str());
        }
    }
    INFO("worst |u - u_ref| = " << worst_u << " V, worst Z rel = " << worst_z);
    CHECK(worst_u <= 1e-10);
}

TEST_CASE("pwl port model: v = u − Z·e reproduces the cache's solve with injected currents",
          "[modes][pwl_ports]") {
    auto b = rectifier(1e6);
    const auto& g = b.graph();
    auto& pool = b.pool();
    const Real dt = 1e-5;
    pwl::PwlStateSpaceCache cache(g, pool);
    cache.build_lazy(dt);
    pwl::DiodeEventState diodes(g, pool);
    pwl::PwlModePorts ports(cache, g, pool, diodes);
    const auto n = static_cast<Index>(pool.state_size(g));
    const auto& es = diodes.entries();
    topology::SwitchStateMask mask(g.num_switches());
    mask.set(es[0].switch_idx, true);   // no island: the cache's own solve is the reference
    sparse::Matrix G, C;
    Vector bc;
    pwl::assemble_segment_split(g, pool, mask, G, C, bc);
    const Vector x0 = Vector::LinSpaced(n, 3.0, 97.0);
    const Vector b_extra = -(C * x0) / dt;
    DenseMatrix Z;
    Vector u;
    ports.site(mask, dt, b_extra, Z, u);
    Vector e(2);
    e << 0.3, -0.7;
    Vector bx = b_extra;
    for (std::size_t j = 0; j < es.size(); ++j) {
        if (stamping::node_is_active(es[j].from)) bx[es[j].from] += e[static_cast<Index>(j)];
        if (stamping::node_is_active(es[j].to)) bx[es[j].to] -= e[static_cast<Index>(j)];
    }
    Vector s;
    cache.solve_at(mask, dt, bx, s);
    const Vector v = u - Z * e;
    for (std::size_t j = 0; j < es.size(); ++j) {
        const Real ref = vp(s, es[j].from, es[j].to);
        REQUIRE(std::abs(v[static_cast<Index>(j)] - ref) <= 1e-9 * (1 + std::abs(ref)));
    }
}

TEST_CASE("pwl judge: the engine's own diode law with circuit-scale tolerance",
          "[modes][pwl_ports]") {
    auto b = rectifier(1e6);
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
