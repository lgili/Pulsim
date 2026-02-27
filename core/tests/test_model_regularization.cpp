#include "pulsim/v1/runtime_circuit.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <string>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

std::size_t find_device_index_by_name(const Circuit& circuit, const std::string& name) {
    const auto& conns = circuit.connections();
    const auto it = std::find_if(conns.begin(), conns.end(), [&](const DeviceConnection& conn) {
        return conn.name == name;
    });
    REQUIRE(it != conns.end());
    return static_cast<std::size_t>(std::distance(conns.begin(), it));
}

}  // namespace

TEST_CASE("Numerical regularization clamps switching models deterministically",
          "[v1][regularization][models]") {
    Circuit circuit;
    const auto n_gate = circuit.add_node("gate");
    const auto n_ctrl = circuit.add_node("ctrl");
    const auto n_drain = circuit.add_node("drain");
    const auto n_source = circuit.add_node("source");

    MOSFET::Params mosfet;
    mosfet.kp = 100.0;
    mosfet.g_off = 1e-14;
    circuit.add_mosfet("M1", n_gate, n_drain, n_source, mosfet);

    circuit.add_diode("D1", n_source, Circuit::ground(), 1e6, 1e-14);

    IGBT::Params igbt;
    igbt.g_on = 1e7;
    igbt.g_off = 1e-14;
    circuit.add_igbt("Q1", n_gate, n_drain, n_source, igbt);

    circuit.add_switch("S1", n_drain, n_source, false, 1e8, 1e-14);
    circuit.add_vcswitch("S2", n_ctrl, n_drain, n_source, 2.5, 1e7, 1e-14);

    const int first_changed = circuit.apply_numerical_regularization(
        8.0,
        1e-7,
        300.0,
        1e-9,
        5e3,
        1e-9,
        5e5,
        1e-9,
        5e5,
        1e-9);
    CHECK(first_changed >= 5);

    const auto& devices = circuit.devices();

    const auto m1_idx = find_device_index_by_name(circuit, "M1");
    const auto d1_idx = find_device_index_by_name(circuit, "D1");
    const auto q1_idx = find_device_index_by_name(circuit, "Q1");
    const auto s1_idx = find_device_index_by_name(circuit, "S1");
    const auto s2_idx = find_device_index_by_name(circuit, "S2");

    const auto* m1 = std::get_if<MOSFET>(&devices[m1_idx]);
    REQUIRE(m1 != nullptr);
    CHECK(m1->params().kp <= Approx(8.0));
    CHECK(m1->params().g_off >= Approx(1e-7));

    const auto* d1 = std::get_if<IdealDiode>(&devices[d1_idx]);
    REQUIRE(d1 != nullptr);
    CHECK(d1->g_on() <= Approx(300.0));
    CHECK(d1->g_off() >= Approx(1e-9));

    const auto* q1 = std::get_if<IGBT>(&devices[q1_idx]);
    REQUIRE(q1 != nullptr);
    CHECK(q1->params().g_on <= Approx(5e3));
    CHECK(q1->params().g_off >= Approx(1e-9));

    const auto* s1 = std::get_if<IdealSwitch>(&devices[s1_idx]);
    REQUIRE(s1 != nullptr);
    CHECK(s1->g_on() <= Approx(5e5));
    CHECK(s1->g_off() >= Approx(1e-9));

    const auto* s2 = std::get_if<VoltageControlledSwitch>(&devices[s2_idx]);
    REQUIRE(s2 != nullptr);
    CHECK(s2->g_on() <= Approx(5e5));
    CHECK(s2->g_off() >= Approx(1e-9));

    const int second_changed = circuit.apply_numerical_regularization(
        8.0,
        1e-7,
        300.0,
        1e-9,
        5e3,
        1e-9,
        5e5,
        1e-9,
        5e5,
        1e-9);
    CHECK(second_changed == 3);
}
