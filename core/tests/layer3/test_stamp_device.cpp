// =============================================================================
// Layer 3 — generic 2-terminal device stamper (the v1-killer)
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/models/ideal_diode.hpp"
#include "pulsim/models/resistor.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/stamp_device.hpp"

using namespace pulsim;
using namespace pulsim::stamping;
using namespace pulsim::models;
using Catch::Approx;

namespace {

constexpr Index N_NODES = 3;

void reset(sparse::Matrix& J, Vector& f) {
    J = sparse::Matrix(N_NODES, N_NODES);
    f = Vector::Zero(N_NODES);
}

}  // namespace

TEST_CASE("Resistor stamping: standard 2x2 conductance block",
          "[v2][layer3][stamp_device]") {
    sparse::Matrix J(N_NODES, N_NODES);
    Vector f = Vector::Zero(N_NODES);
    Vector x(N_NODES);
    x << Real{0}, Real{3}, Real{1};

    Resistor::Params p{Real{2.0}};
    BranchCoord coord{Index{1}, Index{2}, Index{0}};

    stamp_device<Resistor>(J, f, x, coord, p);

    // Current: G·(v[1] - v[2]) = 2·(3 - 1) = 4
    REQUIRE(f[1] == Approx(Real{4}));
    REQUIRE(f[2] == Approx(Real{-4}));
    REQUIRE(f[0] == Approx(Real{0}));            // untouched node

    REQUIRE(J.coeff(1, 1) == Approx(Real{ 2}));  // +G
    REQUIRE(J.coeff(1, 2) == Approx(Real{-2}));  // -G
    REQUIRE(J.coeff(2, 1) == Approx(Real{-2}));
    REQUIRE(J.coeff(2, 2) == Approx(Real{ 2}));
}

TEST_CASE("Resistor stamping to ground skips ground entries",
          "[v2][layer3][stamp_device]") {
    sparse::Matrix J(1, 1);
    Vector f = Vector::Zero(1);
    Vector x(1);
    x << Real{5};

    Resistor::Params p{Real{1.0}};
    BranchCoord coord{Index{0}, kGround, Index{0}};

    stamp_device<Resistor>(J, f, x, coord, p);

    // i = G·(v[0] - 0) = 5; only f[0] gets the +5 contribution.
    REQUIRE(f[0] == Approx(Real{5}));
    REQUIRE(J.coeff(0, 0) == Approx(Real{1}));   // only the diagonal
    REQUIRE(J.nonZeros() == 1);                  // no other entries
}

TEST_CASE("Resistor stamping is additive (parallel resistors)",
          "[v2][layer3][stamp_device]") {
    sparse::Matrix J(2, 2);
    Vector f = Vector::Zero(2);
    Vector x(2);
    x << Real{0}, Real{0};

    // Two resistors in parallel between (0, 1): G = 1.0 + 2.0
    Resistor::Params p1{Real{1.0}};
    Resistor::Params p2{Real{2.0}};
    BranchCoord c1{Index{0}, Index{1}, Index{0}};
    BranchCoord c2{Index{0}, Index{1}, Index{1}};

    stamp_device<Resistor>(J, f, x, c1, p1);
    stamp_device<Resistor>(J, f, x, c2, p2);

    // J(0,0) accumulates G1 + G2 = 3.
    REQUIRE(J.coeff(0, 0) == Approx(Real{3}));
    REQUIRE(J.coeff(0, 1) == Approx(Real{-3}));
    REQUIRE(J.coeff(1, 0) == Approx(Real{-3}));
    REQUIRE(J.coeff(1, 1) == Approx(Real{3}));
}

TEST_CASE("IdealDiode stamping: AD partials match KCL sanity",
          "[v2][layer3][stamp_device][diode]") {
    sparse::Matrix J(2, 2);
    Vector f = Vector::Zero(2);
    Vector x(2);
    x << Real{1.0}, Real{0};   // forward-biased

    IdealDiode::Params p;
    BranchCoord coord{Index{0}, Index{1}, Index{0}};

    stamp_device<IdealDiode>(J, f, x, coord, p);

    // KCL sanity: residual at the two nodes is equal and opposite.
    REQUIRE((f[0] + f[1]) == Approx(Real{0}).margin(1e-12));

    // Jacobian-row sums are zero (the diode current depends only on
    // v_diode = v[0] - v[1], so column-pair sums vanish per row).
    REQUIRE((J.coeff(0, 0) + J.coeff(0, 1)) == Approx(Real{0}).margin(1e-9));
    REQUIRE((J.coeff(1, 0) + J.coeff(1, 1)) == Approx(Real{0}).margin(1e-9));
}

TEST_CASE("IdealDiode stamping populates the expected 4 Jacobian entries",
          "[v2][layer3][stamp_device][diode]") {
    sparse::Matrix J(2, 2);
    Vector f = Vector::Zero(2);
    Vector x(2);
    x << Real{1.0}, Real{0};

    IdealDiode::Params p;
    BranchCoord coord{Index{0}, Index{1}, Index{0}};

    stamp_device<IdealDiode>(J, f, x, coord, p);

    // 4 entries: (0,0), (0,1), (1,0), (1,1). All non-zero in the
    // forward-biased regime.
    REQUIRE(J.nonZeros() == 4);
    REQUIRE(std::abs(J.coeff(0, 0)) > Real{1});   // ≈ 1/R_d, large
    REQUIRE(std::abs(J.coeff(0, 1)) > Real{1});
    REQUIRE(std::abs(J.coeff(1, 0)) > Real{1});
    REQUIRE(std::abs(J.coeff(1, 1)) > Real{1});
}
