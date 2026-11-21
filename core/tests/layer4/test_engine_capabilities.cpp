// add-engine-conformance-matrix: the device-kind capability table and the
// typed refusal raised from it.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/dsed/builder_adapter.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/bdf1.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/solver/unsupported.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

using namespace pulsim;
using namespace pulsim::pwl;
using Catch::Approx;

namespace {

builder::CircuitBuilder junction_cap_rc() {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 10.0);
    b.add_resistor("R", "in", "o", 1e3);
    b.add_capacitor("C", "o", "gnd", 100e-9);
    b.add_nonlinear_capacitor("Cn", "o", "gnd", 1e-6, 1.0, 0.5);
    return b;
}

}  // namespace

TEST_CASE("kKindCaps has one row per StoredKind, named as the enum",
          "[layer4][capabilities]") {
    STATIC_REQUIRE(kKindCaps.size() == DevicePool::kNumKinds);
    REQUIRE(stored_kind_name(DevicePool::StoredKind::Resistor) == "Resistor");
    REQUIRE(stored_kind_name(DevicePool::StoredKind::NonlinearCapacitor)
            == "NonlinearCapacitor");
    REQUIRE(stored_kind_name(DevicePool::StoredKind::IdealTransformer)
            == "IdealTransformer");
    // add-controlled-source-set appended kinds 20-26 after the table.
    REQUIRE(stored_kind_name(DevicePool::StoredKind::VCCS) == "VCCS");
    REQUIRE(stored_kind_name(DevicePool::StoredKind::PwlCurrentSource)
            == "PwlCurrentSource");
    // add-realistic-passives D18: the Bergeron line, the one LinearDelay kind
    REQUIRE(stored_kind_name(DevicePool::StoredKind::TransmissionLine)
            == "TransmissionLine");
    {
        const auto& tl = kKindCaps[static_cast<std::size_t>(
            DevicePool::StoredKind::TransmissionLine)];
        REQUIRE(tl.cls == SolveClass::LinearDelay);
        REQUIRE(tl.supports(Engine::TrBdf2));
        REQUIRE(tl.supports(Engine::AcSweep));
        REQUIRE_FALSE(tl.supports(Engine::Dsed));
        REQUIRE_FALSE(tl.supports(Engine::Bdf1));
        std::size_t n_delay = 0;
        for (const auto& row : kKindCaps) n_delay += row.cls == SolveClass::LinearDelay;
        REQUIRE(n_delay == 1);
    }
    for (const auto& row : kKindCaps) {
        INFO(row.name);
        REQUIRE(row.supports(Engine::FixedTrap));  // the reference engine
        if (row.newton()) {
            REQUIRE_FALSE(row.supports(Engine::Dsed));
            REQUIRE_FALSE(row.supports(Engine::Bdf1));
            // add-external-device-abi / add-unified-device-interface 10.2:
            // the AC sweep linearises exactly these Newton kinds at the DC
            // point (each has an analytic small-signal test); the induction
            // machine is refused (no small-signal test yet). A new Newton
            // kind must be added here or refused on purpose.
            // ExcessLoss: at the DC point (zero flux rate) the regularised
            // core, a conductance c/√ε (add-permeance-magnetic-domain 4.2).
            constexpr std::array<std::string_view, 10> kAcNewton{
                "NonlinearDiode", "MosfetLevel1", "IgbtLevel1",
                "SaturableInductor", "NonlinearCapacitor", "ShockleyDiode",
                "LauritzenDiode", "PmsmMna", "ExternalDevice", "ExcessLoss"};
            const bool listed = std::find(kAcNewton.begin(), kAcNewton.end(),
                                          std::string_view(row.name)) != kAcNewton.end();
            REQUIRE(row.supports(Engine::AcSweep) == listed);
            REQUIRE((listed || std::string_view(row.name) == "InductionMachineMna"));
        }
    }
    // the BRANCH form of Switch / Diode (update-number-changing-release
    // 1.7): every engine but bdf1 (retired, refused by name), the AC sweep
    // included (stamped at the caller's mask)
    for (const auto& row : kBranchFormCaps) {
        INFO(row.name);
        for (const Engine e : {Engine::FixedTrap, Engine::TrBdf2, Engine::Dsed, Engine::AcSweep}) {
            REQUIRE(row.supports(e));
        }
        REQUIRE_FALSE(row.supports(Engine::Bdf1));
    }
}

TEST_CASE("the junction capacitor is a Newton kind",
          "[layer4][capabilities]") {
    auto b = junction_cap_rc();
    REQUIRE(b.pool().requires_newton());
    REQUIRE(b.pool().has_nonlinear_devices());   // the hand list missed it
    REQUIRE_FALSE(b.pool().requires_static_refresh());
}

TEST_CASE("require_supported names the branch, the kind and the engine",
          "[layer4][capabilities]") {
    auto b = junction_cap_rc();
    for (const Engine e : {Engine::Dsed, Engine::Bdf1}) {
        bool threw = false;
        try {
            require_supported(b.graph(), b.pool(), e);
        } catch (const UnsupportedError& err) {
            threw = true;
            REQUIRE(err.engine() == std::string(engine_name(e)));
            REQUIRE(err.device() == "Cn");
            REQUIRE(err.kind() == "NonlinearCapacitor");
            REQUIRE(err.feature().empty());
        }
        REQUIRE(threw);
    }
    REQUIRE_NOTHROW(require_supported(b.graph(), b.pool(), Engine::FixedTrap));
    REQUIRE_NOTHROW(require_supported(b.graph(), b.pool(), Engine::TrBdf2));
}

TEST_CASE("the exact engine and bdf1 refuse the Bergeron line by name",
          "[layer4][capabilities]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "s", "gnd", 1.0);
    b.add_resistor("Rs", "s", "k", 50.0);
    b.add_transmission_line("T", "k", "gnd", "m", "gnd", 50.0, 1e-7);
    b.add_resistor("RL", "m", "gnd", 50.0);
    for (const Engine e : {Engine::Dsed, Engine::Bdf1}) {
        bool threw = false;
        try {
            require_supported(b.graph(), b.pool(), e);
        } catch (const UnsupportedError& err) {
            threw = true;
            REQUIRE(err.device() == "T");
            REQUIRE(err.kind() == "TransmissionLine");
        }
        REQUIRE(threw);
    }
    REQUIRE_NOTHROW(require_supported(b.graph(), b.pool(), Engine::FixedTrap));
    REQUIRE_NOTHROW(require_supported(b.graph(), b.pool(), Engine::TrBdf2));
}

TEST_CASE("bdf1 and the DSED adapter refuse the junction capacitor",
          "[layer4][capabilities]") {
    auto b = junction_cap_rc();
    solver::SimulationOptions opts;
    opts.t_start = 0.0;
    opts.t_end = 1e-5;
    opts.dt = 1e-6;
    solver::SwitchScheduleFn sw = [](Real) {
        return topology::SwitchStateMask(0);
    };
    // Before: 6.3002 V, the circuit without the capacitor.
    REQUIRE_THROWS_AS(solver::run_transient_bdf1(b, opts, sw),
                      UnsupportedError);
    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build_lazy(1e-6);
    using Adapter = dsed::NativeCircuitBuilderAdapter<topology::SwitchStateMask>;
    REQUIRE_THROWS_AS(Adapter(b.graph(), b.pool(), cache), UnsupportedError);
}

TEST_CASE("run_transient stamps Newton devices when no refresh is passed",
          "[layer4][capabilities]") {
    // 10 V, 1 kΩ, Shockley clamp, 100 nF. With no callback the kernel
    // used to simulate the diode as an open circuit (10.000 V).
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 10.0);
    b.add_resistor("R", "in", "o", 1e3);
    b.add_shockley_diode("D", "o", "gnd");
    b.add_capacitor("C", "o", "gnd", 100e-9);
    REQUIRE(b.pool().requires_static_refresh());
    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build(1e-6);
    solver::SimulationOptions opts;
    opts.t_start = 0.0;
    opts.t_end = 1e-3;
    opts.dt = 1e-6;
    const auto r = solver::run_transient(
        cache, b.graph(), b.pool(), opts,
        [](Real) { return topology::SwitchStateMask(0); });
    const Index o = b.node_id_of("o");
    REQUIRE(r.states.back()[o] == Approx(0.594).margin(0.01));
}
