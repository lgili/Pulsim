// =============================================================================
// add-immutable-circuit-ir — the immutable Circuit and its lowered form
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/ir/circuit.hpp"
#include "pulsim/pwl/cache.hpp"

#include <cstring>

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

TEST_CASE("IR: with_params / with_value derive, never mutate", "[ir]") {
    auto make = [](Real L, Real R, bool bind) {
        CircuitBuilder b;
        b.add_voltage_source("Vs", "a", "gnd", 12.0);
        b.add_inductor("L1", "a", "o", L);
        b.add_resistor("RL", "o", "gnd", R);
        b.add_resistor("R2", "o", "gnd", R);
        if (bind) {
            (void)b.param("Lbase", L, "H", Real{1e-9});
            (void)b.param("Rload", R, "Ohm");
            b.bind_param("L1.L", "Lbase");
            b.bind_param("RL.R", "Rload");
            b.bind_param("R2.R_ohms", "Rload");   // alias
        }
        return b.build();
    };
    const auto c = make(1e-3, 5.0, true);
    const auto h = c.value_hash();
    const std::pair<std::string_view, Real> vs[]{{"Lbase", 2.2e-3}, {"Rload", 8.0}};
    const auto d = c.with_params(vs);
    CHECK(d.value_hash() == make(2.2e-3, 8.0, true).value_hash());
    CHECK(d.structure_hash() == c.structure_hash());
    CHECK(c.value_hash() == h);
    CHECK(d.lowered().get_value("R2.R") == 8.0);
    CHECK(std::holds_alternative<ir::ParamId>(d.devices()[3].fields[0].value));
    CHECK(d.evaluate(d.devices()[3].fields[0].value) == 8.0);
    // A binding is structure; a literal circuit hashes differently.
    CHECK(make(1e-3, 5.0, false).structure_hash() != c.structure_hash());

    // with_value: a literal again, equal to the rebuilt literal circuit.
    const auto lit = make(1e-3, 5.0, false);
    CHECK(lit.with_value("RL.R", 7.0).value_hash() ==
          make(1e-3, 7.0, false).with_value("R2.R", 5.0).value_hash());
    CHECK(std::holds_alternative<Real>(
        c.with_value("RL.R", 5.0).devices()[2].fields[0].value));

    using Catch::Matchers::ContainsSubstring;
    const std::pair<std::string_view, Real> bad[]{{"Rload", -1.0}};
    CHECK_THROWS_WITH(c.with_params(bad),
                      ContainsSubstring("parameter \"Rload\" = -1: field R of device \"RL\""));
    const std::pair<std::string_view, Real> low[]{{"Lbase", 0.0}};
    CHECK_THROWS_WITH(c.with_params(low),
                      ContainsSubstring("parameter \"Lbase\" must be a finite number >= 1e-09"));
    const std::pair<std::string_view, Real> unknown[]{{"X", 1.0}};
    CHECK_THROWS_WITH(c.with_params(unknown),
                      ContainsSubstring("no parameter named \"X\"; parameters: Lbase, Rload"));
    CHECK_THROWS_WITH(c.with_value("RL.Q", 1.0),
                      ContainsSubstring("with_value(\"RL.Q\"): device \"RL\" (Resistor) has no value field \"Q\""));
}

TEST_CASE("IR: circuits of one structure share one symbolic analysis", "[ir]") {
    using pwl::PwlStateSpaceCache;
    const auto c = mixed_circuit().build();
    const auto d = c.with_value("RL.R", 4.0).with_value("L.L", 220e-6);
    REQUIRE(d.symbolic_plan() == c.symbolic_plan());
    constexpr Real dt = 1e-7;
    auto solve = [&](const ir::Circuit& k, bool share, Index& analyses) {
        auto b = k.to_builder();   // a run owns its pool, as simulate() does
        PwlStateSpaceCache cache(b.graph(), b.pool());
        if (share) cache.share_symbolic(k.symbolic_plan());
        cache.build_lazy(dt);
        const Index n = static_cast<Index>(b.pool().state_size(b.graph()));
        const Vector be = Vector::LinSpaced(n, -1.0, 1.0);
        std::vector<Vector> xs;
        for (unsigned v = 0; v < 8; ++v) {
            topology::SwitchStateMask m(b.graph().num_switches());
            for (Size i = 0; i < m.size(); ++i) m.set(i, ((v >> (i % 3)) & 1U) != 0);
            Vector x;
            cache.solve(m, be, x);
            xs.push_back(x);
        }
        analyses = static_cast<Index>(cache.metrics().symbolic_analyses);
        return xs;
    };
    Index a0 = 0, a1 = 0, a2 = 0;
    (void)solve(c, true, a0);                  // analyses and offers
    const auto shared = solve(d, true, a1);    // adopts the plan's analysis
    const auto own = solve(d, false, a2);      // private analysis
    CHECK(a0 == 1);
    CHECK(a1 == 0);
    CHECK(a2 == 1);
    REQUIRE(shared.size() == own.size());
    for (std::size_t i = 0; i < own.size(); ++i) {
        REQUIRE(shared[i].size() == own[i].size());
        CHECK(std::memcmp(shared[i].data(), own[i].data(),
                          sizeof(Real) * static_cast<std::size_t>(own[i].size())) == 0);
    }
    // Another structure starts its own plan; build() always does.
    auto other = mixed_circuit();
    other.add_resistor("Rx", "vout", "gnd", 1e3);
    CHECK(other.build().symbolic_plan() != c.symbolic_plan());
    CHECK(mixed_circuit().build().symbolic_plan() != c.symbolic_plan());
}
