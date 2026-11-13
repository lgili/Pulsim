// add-complementarity-mode-selection task 2.2 / 3.1: the dense selector.
//
// The instances are the LCPs the numpy prototype posed on the five
// circuit families (probes/dump_lcp_instances.py); the prototype's Lemke
// is the oracle for pivots and solution, enumeration for the mode.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <random>
#include <string>
#include <vector>

#include "pulsim/events/mode_selector.hpp"

using pulsim::DenseMatrix;
using pulsim::Real;
using pulsim::Vector;
using namespace pulsim::events;

namespace {

struct Instance {
    const char* family;
    int k;
    std::vector<double> M, q;
    int pivots;
    std::vector<double> z, w;
};

const std::vector<Instance>& instances() {
    static const std::vector<Instance> v = {
#include "lcp_instances.inc"
    };
    return v;
}

DenseMatrix mat(int k, const std::vector<double>& a) {
    DenseMatrix M(k, k);
    for (int r = 0; r < k; ++r)
        for (int c = 0; c < k; ++c) M(r, c) = a[static_cast<std::size_t>(r * k + c)];
    return M;
}

Vector vec(const std::vector<double>& a) {
    Vector v(static_cast<Eigen::Index>(a.size()));
    for (std::size_t i = 0; i < a.size(); ++i) v[static_cast<Eigen::Index>(i)] = a[i];
    return v;
}

}  // namespace

TEST_CASE("Lemke reproduces the prototype on every family LCP",
          "[modes][lemke]") {
    LemkeSolver lemke;
    int n = 0, nontrivial = 0, max_piv = 0;
    for (const auto& in : instances()) {
        const DenseMatrix M = mat(in.k, in.M);
        const Vector q = vec(in.q);
        Vector z, w;
        const auto o = lemke.solve(M, q, z, w);
        INFO(in.family << " #" << n << " k=" << in.k);
        REQUIRE((o.status == LcpStatus::Solution || o.status == LcpStatus::Trivial));
        CHECK(std::abs(o.pivots - in.pivots) <= 1);
        CHECK(o.pivots == in.pivots);   // a line-by-line port: same count
        const Real sc = std::max({Real{1}, q.cwiseAbs().maxCoeff(), vec(in.z).cwiseAbs().maxCoeff()});
        CHECK((z - vec(in.z)).cwiseAbs().maxCoeff() <= 1e-9 * sc);
        CHECK((w - vec(in.w)).cwiseAbs().maxCoeff() <= 1e-9 * sc);
        // the enumeration oracle finds a complementary solution too
        Vector ze, we;
        CHECK(enumerate_lcp(M, q, ze, we) >= 1);
        nontrivial += o.pivots > 0;
        max_piv = std::max(max_piv, o.pivots);
        ++n;
    }
    CHECK(n == 218);
    CHECK(nontrivial >= 180);
    CHECK(max_piv == 6);       // design D9: single-phase bridge, max 6
}

TEST_CASE("Lemke agrees with enumeration on random positive definite LCPs",
          "[modes][lemke]") {
    std::mt19937 rng(20261113);
    std::normal_distribution<Real> N(0.0, 1.0);
    LemkeSolver lemke;
    for (int t = 0; t < 200; ++t) {
        const int k = 1 + t % 8;
        DenseMatrix A(k, k), S(k, k);
        Vector q(k);
        for (int r = 0; r < k; ++r) {
            q[r] = 3.0 * N(rng);
            for (int c = 0; c < k; ++c) { A(r, c) = N(rng); S(r, c) = N(rng); }
        }
        DenseMatrix M = A * A.transpose() + (S - S.transpose());
        M.diagonal().array() += 1e-2;
        Vector z, w, ze, we;
        const auto o = lemke.solve(M, q, z, w);
        REQUIRE((o.status == LcpStatus::Solution || o.status == LcpStatus::Trivial));
        REQUIRE(enumerate_lcp(M, q, ze, we) == 1);   // P-matrix: unique
        CHECK((z - ze).cwiseAbs().maxCoeff() <= 1e-8 * std::max(1.0, ze.cwiseAbs().maxCoeff()));
        CHECK(w.minCoeff() >= -1e-9 * std::max(1.0, q.cwiseAbs().maxCoeff()));
    }
}

TEST_CASE("Lemke reports a secondary ray on an infeasible PSD LCP; zero pivots when q >= 0",
          "[modes][lemke]") {
    LemkeSolver lemke;
    Vector z, w;
    DenseMatrix M(2, 2);
    M << 1, -1, -1, 1;              // PSD; w1 + w2 = -2 for every z
    Vector q(2);
    q << -1, -1;
    CHECK(lemke.solve(M, q, z, w).status == LcpStatus::SecondaryRay);
    Vector ze, we;
    CHECK(enumerate_lcp(M, q, ze, we) == 0);
    q << 0.0, 2.0;
    const auto o = lemke.solve(M, q, z, w);
    CHECK(o.status == LcpStatus::Trivial);
    CHECK(o.pivots == 0);
    CHECK(z.cwiseAbs().maxCoeff() == 0.0);
}

// ---------------------------------------------------------------------------
// Port model (task 3.1): a resistive network with k diode ports. For every
// previous mode P the network with P's stamps gives (Z, u); the physical
// answer (i, w) must not depend on the coordinates P picks.
// ---------------------------------------------------------------------------
namespace {

struct Net {
    DenseMatrix G;                  // node conductances (ground eliminated)
    Vector Iin;                     // currents injected into the nodes
    std::vector<std::pair<int, int>> ports;   // (anode, cathode), -1 = ground
    std::vector<DiodePort> p;
};

// (Z, u) of the network with port j stamped g_on (V_f source) if prev_on, else g_off
void port_problem(const Net& n, const std::vector<char>& on, DenseMatrix& Z, Vector& u) {
    DenseMatrix J = n.G;
    Vector rhs = n.Iin;
    const auto k = static_cast<Eigen::Index>(n.ports.size());
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(J.rows(), k);
    for (Eigen::Index j = 0; j < k; ++j) {
        const auto [a, c] = n.ports[static_cast<std::size_t>(j)];
        if (a >= 0) E(a, j) = 1;
        if (c >= 0) E(c, j) = -1;
        const auto& d = n.p[static_cast<std::size_t>(j)];
        const Real g = on[static_cast<std::size_t>(j)] ? 1.0 / d.R_on : d.g_off;
        J += g * E.col(j) * E.col(j).transpose();
        if (on[static_cast<std::size_t>(j)]) rhs += (d.V_f / d.R_on) * E.col(j);
    }
    const Eigen::PartialPivLU<DenseMatrix> lu(J);
    u = E.transpose() * lu.solve(rhs);
    Z = E.transpose() * lu.solve(E);
}

}  // namespace

TEST_CASE("The hybrid port LCP gives the same physical answer from every previous mode",
          "[modes][port]") {
    // a bridge-like 3-node network with 3 diode ports and a 10 V drive
    Net n;
    n.G = DenseMatrix::Zero(3, 3);
    auto R = [&](int a, int b, Real g) {
        if (a >= 0) n.G(a, a) += g;
        if (b >= 0) n.G(b, b) += g;
        if (a >= 0 && b >= 0) { n.G(a, b) -= g; n.G(b, a) -= g; }
    };
    R(0, -1, 1e-1); R(1, 2, 1e-2); R(2, -1, 5e-2); R(1, -1, 1e-6);
    n.Iin = Vector::Zero(3);
    n.Iin[0] = 2.0;                               // 20 V open-circuit at node 0
    n.ports = {{0, 1}, {-1, 1}, {2, 0}};
    n.p = {{1e-3, 1e-9, 0.7, 0.0}, {1e-3, 1e-9, 0.0, 0.0}, {1e-3, 1e-9, 0.7, 0.0}};
    ModeSelector sel;
    CircuitScale sc{20.0, 2.0, 0.0};
    std::vector<char> ref;
    Vector i_ref, w_ref;
    for (int m = 0; m < 8; ++m) {
        std::vector<char> prev = {char(m & 1), char(m >> 1 & 1), char(m >> 2 & 1)};
        auto out = sel.select(n.p, prev,
            [&](Real, DenseMatrix& Z, Vector& u) { port_problem(n, prev, Z, u); },
            1e-9, sc);
        REQUIRE(out.status != LcpStatus::SecondaryRay);
        if (m == 0) {
            ref = out.on;
            i_ref = sel.last_current();
            w_ref = sel.last_gap();
            CHECK(out.on == std::vector<char>{1, 0, 0});   // 20 V forward on port 0 only
            continue;
        }
        INFO("previous mode " << m);
        CHECK(out.on == ref);
        CHECK((sel.last_current() - i_ref).cwiseAbs().maxCoeff() <= 1e-9);
        CHECK((sel.last_gap() - w_ref).cwiseAbs().maxCoeff() <= 1e-9);
        // warm start: from the answer's own coordinates, zero pivots
        if (prev == ref) CHECK(out.pivots == 0);
    }
}

TEST_CASE("A hysteresis port is held OFF below V_th and released at V_th",
          "[modes][hysteresis]") {
    // one diode from a source V through 1 ohm: v_ak = V when OFF
    for (const Real V : {0.3, 0.6}) {
        Net n;
        n.G = DenseMatrix::Constant(1, 1, 1.0);
        n.Iin = Vector::Constant(1, V);
        n.ports = {{0, -1}};
        n.p = {{1e-3, 1e-9, 0.0, 0.5}};          // V_f = 0, V_th = 0.5
        ModeSelector sel;
        std::vector<char> prev = {0};
        auto out = sel.select(n.p, prev,
            [&](Real, DenseMatrix& Z, Vector& u) { port_problem(n, prev, Z, u); },
            1e-9, CircuitScale{1.0, 1.0, 0.0});
        CHECK(out.on[0] == (V > 0.5 ? 1 : 0));
        CHECK(out.released == (V > 0.5 ? 1 : 0));
        // an ON diode is a free port whatever V_th: stays ON while i > 0
        prev = {1};
        out = sel.select(n.p, prev,
            [&](Real, DenseMatrix& Z, Vector& u) { port_problem(n, prev, Z, u); },
            1e-9, CircuitScale{1.0, 1.0, 0.0});
        CHECK(out.on[0] == 1);
    }
}

// add-complementarity-mode-selection slice 2, D5 (DECISIONS.md #13): the
// three-phase diode bridge into C || R at a pair turn-off (dsed state at
// t = 5.0035e-4 s, V_f = 0.7, g_on = 1e3, g_off = 1e-9). The k-port at
// h_e = 2e-9 s, 100·h_e and 1e4·h_e, dumped with full precision. Port 3
// (Dbb) carries ~49 nA, the leak current that holds the floating DC bus:
// slice 1 called it a tie and took it OFF, and the leak divider then
// forward-biased an OFF diode by 8.14 V.
TEST_CASE("A leak-scale current pinning a floating island is ON unless the judge accepts OFF",
          "[modes][ties]") {
    const std::vector<std::vector<double>> Zs = {
        {0.0005004995004485765, -0.0004995004995493759, 0.0005004995004485765, -0.0004995004995493759, 0.0005004995004485765, -0.0004995004995493759, -0.0004995004995493758, 0.0005004995004485765, -0.0004995004995493758, 0.0005004995004485765, -0.0004995004995493758, 0.0005004995004485765, 0.0005004995004486049, -0.0004995004995494043, 0.0005004995004486049, -0.0004995004995494043, 0.0005004995004486049, -0.0004995004995494043, -0.0004995004995494042, 0.0005004995004486049, -0.0004995004995494042, 0.0005004995004486049, -0.0004995004995494042, 0.0005004995004486049, 0.0005004995004485481, -0.0004995004995493475, 0.0005004995004485481, -0.0004995004995493475, 0.0005004995004485481, -0.0004995004995493475, -0.0004995004995493474, 0.0005004995004485481, -0.0004995004995493474, 0.0005004995004485481, -0.0004995004995493474, 0.0005004995004485481},
        {0.0005454541322341528, -0.00045454586776384634, 0.0005454541322341528, -0.00045454586776384634, 0.0005454541322341528, -0.00045454586776384634, -0.00045454586776384634, 0.0005454541322341528, -0.00045454586776384634, 0.0005454541322341528, -0.00045454586776384634, 0.0005454541322341528, 0.0005454541322341528, -0.00045454586776384634, 0.0005454541322341528, -0.00045454586776384634, 0.0005454541322341528, -0.00045454586776384634, -0.00045454586776384634, 0.0005454541322341528, -0.00045454586776384634, 0.0005454541322341528, -0.00045454586776384634, 0.0005454541322341528, 0.0005454541322341532, -0.00045454586776384634, 0.0005454541322341532, -0.00045454586776384634, 0.0005454541322341532, -0.00045454586776384634, -0.0004545458677638468, 0.0005454541322341528, -0.0004545458677638468, 0.0005454541322341528, -0.0004545458677638468, 0.0005454541322341528},
        {0.0009545041359858115, -4.5495864012188805e-05, 0.0009545041359858115, -4.5495864012188805e-05, 0.0009545041359858115, -4.5495864012188805e-05, -4.5495864012188805e-05, 0.0009545041359858112, -4.5495864012188805e-05, 0.0009545041359858112, -4.5495864012188805e-05, 0.0009545041359858112, 0.0009545041359858115, -4.5495864012188886e-05, 0.0009545041359858115, -4.5495864012188886e-05, 0.0009545041359858115, -4.5495864012188886e-05, -4.549586401218882e-05, 0.0009545041359858113, -4.549586401218882e-05, 0.0009545041359858113, -4.549586401218882e-05, 0.0009545041359858113, 0.0009545041359858113, -4.5495864012188805e-05, 0.0009545041359858113, -4.5495864012188805e-05, 0.0009545041359858113, -4.5495864012188805e-05, -4.549586401218858e-05, 0.0009545041359858112, -4.549586401218858e-05, 0.0009545041359858112, -4.549586401218858e-05, 0.0009545041359858112}};
    const std::vector<std::vector<double>> us = {
        {-61.35336207255125, -108.31629525801563, -170.3696572960298, 0.6999999654629221, 0.699999965404615, -170.36965729597148},
        {-61.34330643553156, -108.32467052062981, -170.36797364808365, 0.6999966919222942, 0.6999966918751852, -170.36797364803655},
        {-60.33639903216793, -109.15992319087536, -170.19612943570587, 0.699807212662563, 0.6998072126137345, -170.19612943565704}};
    std::vector<DiodePort> ports(6, DiodePort{1e-3, 1e-9, 0.7, 0.0});
    const std::vector<char> prev = {0, 0, 0, 1, 1, 0};
    const CircuitScale sc{100.0, 5.0, 1e-3};
    const Real h_e = 2e-9;
    auto port_at = [&](Real h, DenseMatrix& Z, Vector& u) {
        const int p = h < 1e-8 ? 0 : (h < 1e-6 ? 1 : 2);
        Z = mat(6, Zs[static_cast<std::size_t>(p)]);
        u = vec(us[static_cast<std::size_t>(p)]);
    };
    ModeSelector sel;
    const std::vector<char> pinned = {0, 0, 0, 1, 0, 0};
    // no judge: the resistance test alone pins it
    auto a = sel.select(ports, prev, port_at, h_e, sc);
    CHECK(a.on == pinned);
    CHECK(a.leak_pins == 1);
    CHECK(a.ties == 0);
    // the judge rejects the all-OFF mode: pinned
    int asked = 0;
    auto b = sel.select(ports, prev, port_at, h_e, sc, [&](const std::vector<char>& m) {
        ++asked;
        CHECK(m == std::vector<char>(6, 0));
        return 0;
    });
    CHECK(asked == 1);
    CHECK(b.on == pinned);
    // the judge accepts all-OFF (as at a symmetric single-phase turn-off): OFF, counted as a tie
    auto c = sel.select(ports, prev, port_at, h_e, sc, [](const std::vector<char>&) { return 1; });
    CHECK(c.on == std::vector<char>(6, 0));
    CHECK(c.leak_pins == 0);
    CHECK(c.ties == 1);
    CHECK(c.pin_judgements == 1);
}

// task 5 (D3): passivity, enumeration and the named refusals. The k-port
// Z = −(1 + R_on)·I, u = 1 of OFF ports gives H = −I, q = −1 in OFF
// coordinates: no complementary solution, Lemke ends at a secondary ray.
#include "pulsim/events/mode_refusal.hpp"

namespace {
ModeSelection select_neg_identity(int k, bool passive) {
    std::vector<DiodePort> ports(static_cast<std::size_t>(k), DiodePort{1e-3, 0.0, 0.0, 0.0});
    const std::vector<char> prev(static_cast<std::size_t>(k), 0);
    ModeSelector sel;
    sel.passive = passive;
    return sel.select(ports, prev,
                      [&](Real, DenseMatrix& Z, Vector& u) {
                          Z = -(1.0 + 1e-3) * DenseMatrix::Identity(k, k);
                          u = Vector::Ones(k);
                      },
                      1e-9, CircuitScale{1.0, 1.0, 0.0});
}

std::vector<std::string> names(int k) {
    std::vector<std::string> v;
    for (int j = 0; j < k; ++j) v.push_back("D" + std::to_string(j + 1));
    return v;
}
}  // namespace

TEST_CASE("Named refusals of the mode selection", "[modes][refusal]") {
    const PassivityClass vcvs{"E1", "VCVS"};
    SECTION("passive, secondary ray: no consistent mode, every diode named") {
        const auto s = select_neg_identity(2, true);
        CHECK(s.status == LcpStatus::SecondaryRay);
        CHECK(s.enumerations == 0);
        try {
            raise_if_refused(s, "dsed", 1.5e-3, names(2), {});
            FAIL("no refusal");
        } catch (const pulsim::UnsupportedError& e) {
            CHECK(e.engine() == "dsed");
            CHECK(e.feature() == "consistent_mode");
            CHECK(e.device() == "D1, D2");
            CHECK(e.kind() == "Diode");
            CHECK(std::string(e.what()).find("1.500000000e-03") != std::string::npos);
        }
    }
    SECTION("non-passive, 2 ports: enumerated, still none -> no consistent mode") {
        const auto s = select_neg_identity(2, false);
        CHECK(s.status == LcpStatus::SecondaryRay);
        CHECK(s.enumerations == 1);
        CHECK_THROWS_AS(raise_if_refused(s, "dsed", 0.0, names(2), vcvs), pulsim::UnsupportedError);
    }
    SECTION("non-passive, 13 ports: beyond the enumeration limit -> non_passive_ports") {
        const auto s = select_neg_identity(13, false);
        CHECK(s.status == LcpStatus::Unenumerable);
        try {
            raise_if_refused(s, "dsed", 0.0, names(13), vcvs);
            FAIL("no refusal");
        } catch (const pulsim::UnsupportedError& e) {
            CHECK(e.feature() == "non_passive_ports");
            CHECK(e.device() == "E1");
            CHECK(e.kind() == "VCVS");
        }
    }
    SECTION("a solution raises nothing") {
        ModeSelection ok;
        CHECK_NOTHROW(raise_if_refused(ok, "dsed", 0.0, names(1), vcvs));
    }
}

TEST_CASE("Non-passive ports: enumeration finds the mode Lemke cannot certify", "[modes][refusal]") {
    // M = [[-1, 2], [2, -1]], q = (-1, -1): not copositive; z = (1, 1), w = 0 solves it
    const DenseMatrix M = mat(2, {-1.0, 2.0, 2.0, -1.0});
    const Vector q = vec({-1.0, -1.0});
    Vector z, w;
    const int n = enumerate_lcp(M, q, z, w);
    CHECK(n >= 1);
    CHECK((M * z + q - w).cwiseAbs().maxCoeff() < 1e-12);
    CHECK(z.minCoeff() >= 0.0);
    CHECK(w.minCoeff() >= -1e-12);
}
