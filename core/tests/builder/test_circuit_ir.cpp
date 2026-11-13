// =============================================================================
// add-immutable-circuit-ir — the immutable Circuit and its lowered form
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/ir/circuit.hpp"

#include <bit>
#include <cstdint>
#include <string>
#include <vector>

using namespace pulsim;
using pulsim::builder::CircuitBuilder;

namespace {

CircuitBuilder mixed_circuit() {
    CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 24.0);
    b.add_mosfet("Q1", "vin", "sw");                 // switch + body diode
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9);
    b.add_inductor("L", "sw", "vout", 100e-6);
    b.add_capacitor("C", "vout", "gnd", 47e-6);
    b.add_resistor("RL", "vout", "gnd", 10.0);
    b.add_transformer("T1", "vout", "gnd", "s", "gnd", 1e-3, 4e-3, 0.98);
    b.add_resistor("Rs", "s", "gnd", 20.0);
    b.add_vcvs("E1", "vout", "gnd", "e", "gnd", 2.0);
    b.add_resistor("Re", "e", "gnd", 1e3);
    b.set_initial("L", 1.5);
    b.set_initial("C", 3.0);
    b.set_alias("output", std::string_view{"vout"}, std::nullopt);
    return b;
}

std::vector<std::uint64_t> pool_bits(const CircuitBuilder& b) {
    std::vector<std::uint64_t> out;
    const auto& pool = b.pool();
    for (Index i = 0; i < b.graph().num_branches(); ++i) {
        const auto k = pool.kind_of(i);
        out.push_back(static_cast<std::uint64_t>(k));
        for (const auto& f : ir::fields_of(k)) {
            out.push_back(std::bit_cast<std::uint64_t>(f.get(pool.params_of(i))));
        }
    }
    return out;
}

}  // namespace

TEST_CASE("build(): the lowered form is the builder's graph and pool, in order",
          "[ir][circuit]") {
    const auto b = mixed_circuit();
    const auto c = b.build();
    const auto& lo = c.lowered();

    REQUIRE(lo.graph().num_nodes() == b.graph().num_nodes());
    REQUIRE(lo.graph().num_branches() == b.graph().num_branches());
    REQUIRE(lo.graph().id() == b.graph().id());
    for (Index i = 0; i < b.graph().num_branches(); ++i) {
        const auto& x = b.graph().branch(i);
        const auto& y = lo.graph().branch(i);
        CHECK(x.from == y.from);
        CHECK(x.to == y.to);
        CHECK(x.kind == y.kind);
        CHECK(b.graph().branch_name(i) == lo.graph().branch_name(i));
    }
    CHECK(pool_bits(b) == pool_bits(lo));
    CHECK(lo.pool().state_size(lo.graph()) == b.pool().state_size(b.graph()));
    CHECK(lo.graph().num_switches() == b.graph().num_switches());
    CHECK(lo.switch_index_of("D1") == b.switch_index_of("D1"));
    CHECK(lo.initial_state() == b.initial_state());
    CHECK(lo.state_var_names() == b.state_var_names());

    // IR tables.
    REQUIRE(c.devices().size() == static_cast<std::size_t>(b.graph().num_branches()));
    CHECK(c.devices()[static_cast<std::size_t>(b.branch_id_of("L"))].name == "L");
    CHECK(c.devices()[static_cast<std::size_t>(b.branch_id_of("Q1"))].kind ==
          static_cast<std::uint16_t>(pwl::DevicePool::StoredKind::Switch));
    CHECK(c.nodes().size() == static_cast<std::size_t>(b.graph().num_nodes()));
    REQUIRE(c.initial_conditions().size() == 2);
    REQUIRE(c.aliases().size() == 1);
    CHECK(c.aliases()[0].target == "vout");
}

TEST_CASE("Circuit: copies share data; the builder and to_builder() are independent",
          "[ir][circuit]") {
    auto b = mixed_circuit();
    const auto c = b.build();
    const auto c2 = c;  // NOLINT(performance-unnecessary-copy-initialization)
    CHECK(c2.same_data(c));
    const auto h = c.value_hash();
    const auto s = c.structure_hash();

    // Editing the builder after build() leaves the snapshot unchanged.
    b.add_resistor("Rx", "vout", "gnd", 5.0);
    b.set_value("RL.R", 20.0);
    CHECK(c.value_hash() == h);
    CHECK(c.lowered().graph().num_branches() + 1 == b.graph().num_branches());
    CHECK(b.build().structure_hash() != s);

    // A run's private copy is a different object.
    auto run = c.to_builder();
    run.set_value("RL.R", 30.0);
    CHECK(c.lowered().get_value("RL.R") == 10.0);
    CHECK(run.build().value_hash() != h);
    CHECK(run.build().structure_hash() == s);
}

TEST_CASE("Circuit: equal builds hash equal; a value change moves value_hash only",
          "[ir][circuit]") {
    const auto c1 = mixed_circuit().build();
    const auto c2 = mixed_circuit().build();
    CHECK(c1.structure_hash() == c2.structure_hash());
    CHECK(c1.value_hash() == c2.value_hash());
    auto b = mixed_circuit();
    b.set_value("C.C", 22e-6);
    CHECK(b.build().structure_hash() == c1.structure_hash());
    CHECK(b.build().value_hash() != c1.value_hash());
}

TEST_CASE("set_value: equals rebuilding; refusals name device and field",
          "[ir][set_value]") {
    auto edited = mixed_circuit();
    edited.set_value("RL.R", 4.7);
    edited.set_value("L.L", 220e-6);         // self-inductance only
    edited.set_value("T1.p.L", 2e-3);        // a coupled winding: coupling re-derived
    auto rebuilt = CircuitBuilder{};
    rebuilt.add_voltage_source("Vin", "vin", "gnd", 24.0);
    rebuilt.add_mosfet("Q1", "vin", "sw");
    rebuilt.add_diode("D1", "gnd", "sw", 1e3, 1e-9);
    rebuilt.add_inductor("L", "sw", "vout", 220e-6);
    rebuilt.add_capacitor("C", "vout", "gnd", 47e-6);
    rebuilt.add_resistor("RL", "vout", "gnd", 4.7);
    rebuilt.add_transformer("T1", "vout", "gnd", "s", "gnd", 2e-3, 4e-3, 0.98);
    rebuilt.add_resistor("Rs", "s", "gnd", 20.0);
    rebuilt.add_vcvs("E1", "vout", "gnd", "e", "gnd", 2.0);
    rebuilt.add_resistor("Re", "e", "gnd", 1e3);
    rebuilt.set_initial("L", 1.5);
    rebuilt.set_initial("C", 3.0);
    rebuilt.set_alias("output", std::string_view{"vout"}, std::nullopt);
    CHECK(edited.build().value_hash() == rebuilt.build().value_hash());
    CHECK(edited.pool().transformer_couplings()[0].params.L_p == 2e-3);

    using Catch::Matchers::ContainsSubstring;
    auto b = mixed_circuit();
    CHECK_THROWS_WITH(b.set_value("nope.R", 1.0), ContainsSubstring("no device named \"nope\""));
    CHECK_THROWS_WITH(b.set_value("RL.X", 1.0),
                      ContainsSubstring("has no value field \"X\"") &&
                      ContainsSubstring("R, G"));
    CHECK_THROWS_WITH(b.set_value("RL.R", -1.0),
                      ContainsSubstring("field R of device \"RL\" must be a finite number > 0"));
    CHECK_THROWS_WITH(b.set_value("RL", 1.0), ContainsSubstring("<device>.<field>"));
    CHECK(b.get_value("RL.R") == 10.0);  // refusals left it unchanged
}

TEST_CASE("IR: with_preflight regularises a copy, never the receiver", "[ir]") {
    CircuitBuilder b;
    b.add_voltage_source("V1", "a", "gnd", 5.0);
    b.add_resistor("R1", "a", "gnd", 10.0);
    b.add_resistor("Ri", "iso1", "iso2", 10.0);   // floating island
    const auto c = b.build();
    const auto h = c.value_hash();
    const auto nb = c.graph().num_branches();

    auto r = c.with_preflight(pwl::PreflightOptions{});
    REQUIRE(r.ties.size() == 1);
    CHECK(r.ties[0].name == "R_auto_iso_iso1");
    CHECK(r.ties[0].resistance == 1e9);
    CHECK(r.report.findings.size() == 1);
    CHECK(r.circuit.graph().num_branches() == nb + 1);
    CHECK(c.value_hash() == h);                    // receiver unchanged
    CHECK(c.graph().num_branches() == nb);
    CHECK(b.graph().num_branches() == nb);         // builder unchanged

    // Equal to the in-place preflight it replaces, and to with_ties.
    auto inplace = b.clone();
    (void)inplace.run_preflight(pwl::PreflightOptions{});
    CHECK(r.circuit.value_hash() == inplace.build().value_hash());
    CHECK(c.with_ties(r.ties).value_hash() == r.circuit.value_hash());

    // Well posed: same data, no copy, no ties.
    auto ok = r.circuit.with_preflight(pwl::PreflightOptions{});
    CHECK(ok.ties.empty());
    CHECK(ok.report.findings.empty());
    CHECK(ok.circuit.same_data(r.circuit));

    // auto_regularize off: report only, receiver returned as is.
    pwl::PreflightOptions off;
    off.auto_regularize = false;
    auto rep = c.with_preflight(off);
    CHECK(rep.ties.empty());
    CHECK(rep.report.findings.size() == 1);
    CHECK(rep.circuit.same_data(c));

    using Catch::Matchers::ContainsSubstring;
    const std::vector<ir::Tie> bad{{"Rx", 99, 1e9}};
    CHECK_THROWS_WITH(c.with_ties(bad), ContainsSubstring("tie \"Rx\" names node 99"));
}
