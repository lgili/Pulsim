// =============================================================================
// Layer 4 — named kernel diagnostics (MNA row → node / device)
// =============================================================================
//
// v2.0 Phase 1, audit findings `kernel-has-no-name-context-for-errors`
// (Graph now carries branch names) and
// `singular-errors-dont-name-the-node` (the error paths use them).
//
// The bar these tests hold: a user who hits an unsolvable circuit must
// be told WHICH node or device is at fault, not just that some mask
// was singular.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/dc_assemble.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/topology/graph.hpp"

#include <set>
#include <string>

using namespace pulsim;
using namespace pulsim::pwl;
using Catch::Approx;

namespace {

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

}  // namespace

TEST_CASE("Graph carries branch names; unnamed branches stay empty",
          "[v2][layer4][row_names]") {
    topology::Graph g;
    g.add_node("vin");
    const Index b_named =
        g.add_branch(0, g.ground(), topology::BranchKind::Source, "Vdc");
    const Index b_plain =
        g.add_branch(0, g.ground(), topology::BranchKind::PassiveLinear);

    REQUIRE(g.branch_name(b_named) == "Vdc");
    REQUIRE(g.branch_name(b_plain).empty());
    // Out-of-range is empty, not UB — diagnostics call this freely.
    REQUIRE(g.branch_name(99).empty());
    REQUIRE(g.branch_name(-1).empty());

    g.set_branch_name(b_plain, "R1");
    REQUIRE(g.branch_name(b_plain) == "R1");
}

TEST_CASE("CircuitBuilder pushes device names into the kernel Graph",
          "[v2][layer4][row_names]") {
    // The whole point of the plumbing: names entered by the user
    // must be visible to the SOLVER, not only to the schematic layer.
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "n1", "gnd", 12.0);
    b.add_resistor("Rload", "n1", "gnd", 10.0);
    b.add_inductor("L1", "n1", "gnd", 1e-3);

    const auto& g = b.graph();
    bool saw_vin = false, saw_rload = false, saw_l1 = false;
    for (Index i = 0; i < g.num_branches(); ++i) {
        const auto n = g.branch_name(i);
        if (n == "Vin")   saw_vin = true;
        if (n == "Rload") saw_rload = true;
        if (n == "L1")    saw_l1 = true;
    }
    REQUIRE(saw_vin);
    REQUIRE(saw_rload);
    REQUIRE(saw_l1);
}

TEST_CASE("describe_row maps every MNA row segment to its owner",
          "[v2][layer4][row_names]") {
    // Layout: [v_0 .. v_{N-1}] [i_src ...] [i_L ...]
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b.add_resistor("R1", "vin", "vout", 1.0);
    b.add_inductor("L1", "vout", "gnd", 1e-3);

    const auto& g = b.graph();
    const auto& pool = b.pool();
    const auto n_nodes = g.num_nodes();
    const auto n_total = static_cast<Index>(pool.state_size(g));

    // Node rows.
    for (Index r = 0; r < n_nodes; ++r) {
        const auto info = describe_row(g, pool, r);
        REQUIRE(info.kind == RowKind::NodeVoltage);
        REQUIRE(info.index == r);
        REQUIRE(contains(info.label, "node "));
        // Guard against the tautology: contains(x, "") is always
        // true, so only assert the name when there IS one.
        REQUIRE_FALSE(g.node(r).name.empty());
        REQUIRE(contains(info.label, g.node(r).name));
    }

    // Every remaining row is a branch current and must be attributed
    // to a NAMED device — no "unattributed" leftovers.
    int sources = 0, inductors = 0;
    for (Index r = n_nodes; r < n_total; ++r) {
        const auto info = describe_row(g, pool, r);
        REQUIRE(info.kind != RowKind::OutOfRange);
        REQUIRE(contains(info.label, "current through "));
        if (info.kind == RowKind::SourceCurrent) {
            ++sources;
            REQUIRE(contains(info.label, "Vin"));
        } else {
            ++inductors;
            REQUIRE(contains(info.label, "L1"));
        }
    }
    REQUIRE(sources == 1);
    REQUIRE(inductors == 1);

    // Out-of-range degrades gracefully instead of throwing.
    const auto bad = describe_row(g, pool, n_total + 5);
    REQUIRE(bad.kind == RowKind::OutOfRange);
    REQUIRE(contains(bad.label, "out of range"));
}

TEST_CASE("describe_row falls back to ids on a raw unnamed graph",
          "[v2][layer4][row_names]") {
    // Raw-kernel users (tests, hand-built graphs) never set names;
    // every diagnostic must still produce something printable.
    topology::Graph g;
    g.add_node("");
    DevicePool pool;
    g.add_branch(0, g.ground(), topology::BranchKind::Source);
    pool.add_voltage_source(0, {.V = 1.0});

    REQUIRE(contains(row_label(g, pool, 0), "node #0"));
    REQUIRE(contains(row_label(g, pool, 1), "branch #0"));
    REQUIRE(contains(branch_label(g, 0), "branch #0"));
}

TEST_CASE("Empty-column/row probes find the floating unknown",
          "[v2][layer4][row_names]") {
    sparse::Matrix M(3, 3);
    std::vector<sparse::Triplet> t;
    // Column (and row) 1 left entirely empty.
    t.emplace_back(0, 0, 1.0);
    t.emplace_back(2, 2, 1.0);
    t.emplace_back(0, 2, 0.5);
    M.setFromTriplets(t.begin(), t.end());
    M.makeCompressed();
    REQUIRE(sparse::first_empty_column(M) == 1);
    REQUIRE(sparse::first_empty_row(M) == 1);

    sparse::Matrix full(2, 2);
    std::vector<sparse::Triplet> t2{{0, 0, 1.0}, {1, 1, 1.0}};
    full.setFromTriplets(t2.begin(), t2.end());
    full.makeCompressed();
    REQUIRE(sparse::first_empty_column(full) == kInvalidIndex);
    REQUIRE(sparse::first_empty_row(full) == kInvalidIndex);

    // An UNCOMPRESSED matrix must report "unknown", never a false
    // positive — claiming a node is floating when it is not would be
    // worse than saying nothing.
    sparse::Matrix un(2, 2);
    un.coeffRef(0, 0) = 1.0;
    REQUIRE_FALSE(un.isCompressed());
    REQUIRE(sparse::first_empty_column(un) == kInvalidIndex);
    REQUIRE(sparse::first_empty_row(un) == kInvalidIndex);
}

TEST_CASE("DC singular error names the floating node",
          "[v2][layer4][row_names][diagnostics]") {
    // The classic: a node tied to the circuit ONLY through a
    // capacitor. Capacitors are open at DC, so `vfloat` gets an empty
    // MNA column — before this change the user got
    // "DC matrix numerically singular for mask 0b N=0" and nothing
    // else.
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b.add_resistor("R1", "vin", "gnd", 10.0);
    b.add_capacitor("Cfloat", "vin", "vfloat", 1e-6);

    topology::SwitchStateMask m(0);
    bool threw = false;
    try {
        (void)compute_dc_op(b.graph(), b.pool(), m);
    } catch (const std::runtime_error& e) {
        threw = true;
        const std::string msg = e.what();
        INFO("message: " << msg);
        REQUIRE(contains(msg, "vfloat"));          // THE point
        REQUIRE(contains(msg, "no device ties it"));
        REQUIRE(contains(msg, "DC path"));
    }
    REQUIRE(threw);
}

TEST_CASE("Cache singular error carries a localised detail + row",
          "[v2][layer4][row_names][diagnostics]") {
    // Same floating-node circuit through the PWL cache: the
    // structured CacheError must expose both the human sentence and
    // the row index (so a GUI can highlight the element rather than
    // parse what()).
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b.add_resistor("R1", "vin", "gnd", 10.0);
    b.add_capacitor("Cfloat", "vin", "vfloat", 1e-6);

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(Real{0});   // static build: caps are skipped

    topology::SwitchStateMask m(0);
    auto r = cache.try_lookup(m);
    REQUIRE_FALSE(r.has_value());
    const auto err = r.error();
    INFO("what(): " << err.what());
    REQUIRE(contains(err.what(), "vfloat"));
    REQUIRE(err.failing_row != kInvalidIndex);
    // The row it points at really is the floating node.
    REQUIRE(contains(row_label(b.graph(), b.pool(), err.failing_row),
                      "vfloat"));
}

TEST_CASE("Row→device mapping survives many sources and inductors",
          "[v2][layer4][row_names]") {
    // The single-source/single-inductor tests above cannot catch an
    // OFF-BY-ONE in the reverse lookup: with one of each, any shift
    // still lands on "the" source or "the" inductor. Here each row
    // must resolve to a SPECIFIC named device, so a shifted segment
    // boundary or a miscounted source kind fails loudly.
    //
    // The source band also mixes kinds on purpose — DC, PWM, sine and
    // pulse sources all own a branch-current row, and all of them
    // must be counted by `num_voltage_sources()` for the
    // source/inductor boundary to be right.
    builder::CircuitBuilder b;
    b.add_voltage_source("Vdc",  "a", "gnd", 12.0);
    b.add_pwm_voltage_source("Vpwm", "b", "gnd",
                              /*v_high=*/5.0, /*v_low=*/0.0,
                              /*frequency=*/1e3, /*duty=*/0.5);
    b.add_sine_voltage_source("Vac", "c", "gnd",
                               /*v_dc=*/0.0, /*v_amplitude=*/1.0,
                               /*frequency=*/50.0);
    b.add_inductor("La", "a", "b", 1e-3);
    b.add_inductor("Lb", "b", "c", 2e-3);
    b.add_inductor("Lc", "c", "gnd", 3e-3);
    b.add_resistor("Rload", "a", "gnd", 10.0);

    const auto& g = b.graph();
    const auto& pool = b.pool();

    // Every named source/inductor must be found at EXACTLY the row
    // the pool says it owns, and the label must name that device.
    auto check = [&](const char* name, Index row, RowKind kind) {
        const auto info = describe_row(g, pool, row);
        INFO("row " << row << " -> " << info.label
             << " (expected " << name << ")");
        REQUIRE(info.kind == kind);
        REQUIRE(contains(info.label, name));
    };

    for (Index bid = 0; bid < g.num_branches(); ++bid) {
        const auto name = g.branch_name(bid);
        if (name.empty()) continue;
        if (pool.is_voltage_source(bid)) {
            check(std::string{name}.c_str(),
                  pool.branch_var_id_for_source(bid, g),
                  RowKind::SourceCurrent);
        } else if (pool.is_inductor(bid)) {
            check(std::string{name}.c_str(),
                  pool.branch_var_id_for_inductor(bid, g),
                  RowKind::InductorCurrent);
        }
    }

    // And no two devices may claim the same row (a shifted boundary
    // would alias an inductor onto a source row).
    std::set<Index> claimed;
    for (Index bid = 0; bid < g.num_branches(); ++bid) {
        if (pool.is_voltage_source(bid)) {
            REQUIRE(claimed.insert(
                pool.branch_var_id_for_source(bid, g)).second);
        } else if (pool.is_inductor(bid)) {
            REQUIRE(claimed.insert(
                pool.branch_var_id_for_inductor(bid, g)).second);
        }
    }
    // Together the branch-current rows exactly fill [N, state_size).
    REQUIRE(claimed.size() ==
            static_cast<std::size_t>(pool.state_size(g)) -
                static_cast<std::size_t>(g.num_nodes()));
}

TEST_CASE("A device row empty BY CONSTRUCTION is not blamed as unwired",
          "[v2][layer4][row_names][diagnostics]") {
    // Adversarial-review finding
    // `dt0-static-build-blames-a-perfectly-connected-inductor`
    // (HIGH). At dt = 0 the assembler deliberately omits inductor
    // companion stamps, yet `state_size` still reserves the
    // inductor's branch-current row — so that row is empty because
    // of the ASSEMBLY MODE, not because the device is disconnected.
    //
    // The first version of this feature told such a user "nothing
    // connects current through inductor L1 ... add a bleeder
    // resistance", sending them to debug a correctly wired part.
    // That is worse than the old uninformative message. The
    // phrasing must branch on what the empty row belongs to.
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b.add_resistor("R1", "vin", "gnd", 10.0);
    b.add_inductor("L1", "vin", "gnd", 1e-3);   // perfectly wired

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    bool threw = false;
    try {
        cache.build(Real{0});      // static build: no L stamps
    } catch (const std::runtime_error& e) {
        threw = true;
        const std::string msg = e.what();
        INFO("message: " << msg);
        // It may name L1 — that IS where the empty row is — but it
        // must NOT claim the device is disconnected or tell the user
        // to rewire it.
        REQUIRE_FALSE(contains(msg, "no device ties it"));
        REQUIRE_FALSE(contains(msg, "bleeder"));
        // Instead it must explain the real cause and the real fix.
        REQUIRE(contains(msg, "no equation in this system"));
        REQUIRE(contains(msg, "dt > 0"));
    }
    REQUIRE(threw);
}

TEST_CASE("CacheError.detail and .failing_row always agree",
          "[v2][layer4][row_names][diagnostics]") {
    // Adversarial-review finding
    // `cache-error-detail-and-failing-row-use-inverted-precedence`
    // (HIGH): the sentence and the machine-readable index were
    // resolved independently with OPPOSITE priority, so a GUI could
    // highlight one device while the text named another. One
    // resolution now feeds both.
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 12.0);
    b.add_resistor("R1", "vin", "gnd", 10.0);
    b.add_capacitor("Cfloat", "vin", "vfloat", 1e-6);

    PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(Real{0});
    auto r = cache.try_lookup(topology::SwitchStateMask(0));
    REQUIRE_FALSE(r.has_value());
    const auto err = r.error();
    REQUIRE(err.failing_row != kInvalidIndex);
    REQUIRE_FALSE(err.detail.empty());
    // The row the tooling would highlight must be the one the
    // sentence talks about.
    const auto label = row_label(b.graph(), b.pool(), err.failing_row);
    INFO("detail: " << err.detail << "\n  row label: " << label);
    REQUIRE(contains(err.detail, label));
}
