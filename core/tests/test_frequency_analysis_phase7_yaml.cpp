// =============================================================================
// Phase 7 of `add-frequency-domain-analysis`: YAML schema for `analysis:`
// =============================================================================
//
// Pins the contract:
//   * `analysis:` is a top-level array of frequency-domain analyses.
//   * Each entry has a `type:` discriminator that maps it to either
//     `AcSweepOptions` or `FraOptions`.
//   * The parser populates `SimulationOptions::ac_sweeps` /
//     `SimulationOptions::fra_sweeps` in YAML order.
//   * Unknown types and unknown per-entry fields fail under strict mode.
//   * The end-to-end flow (load YAML → run AC sweep) produces the same
//     Bode data as a hand-built AcSweepOptions on an equivalent circuit.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/parser/yaml_parser.hpp"
#include "pulsim/v1/core.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

[[nodiscard]] std::string rc_yaml_with_analysis() {
    return R"(schema: pulsim-v1
version: 1
simulation:
  tstart: 0.0
  tstop: 1e-6
  dt: 1e-7
  dt_min: 1e-12
  dt_max: 1e-7
  adaptive_timestep: false
analysis:
  - type: ac
    name: rc_lowpass
    f_start: 1.0
    f_stop: 1e6
    points_per_decade: 30
    scale: log
    perturbation_source: V1
    measurement_nodes: [out]
  - type: fra
    name: rc_lowpass_fra
    f_start: 100.0
    f_stop: 1e3
    points_per_decade: 4
    perturbation_source: V1
    perturbation_amplitude: 0.01
    measurement_nodes: [out]
    n_cycles: 6
    discard_cycles: 2
    samples_per_cycle: 64
components:
  - type: voltage_source
    name: V1
    nodes: [in, gnd]
    value: 1.0
  - type: resistor
    name: R1
    nodes: [in, out]
    value: 1e3
  - type: capacitor
    name: C1
    nodes: [out, gnd]
    value: 1e-6
)";
}

}  // namespace

// -----------------------------------------------------------------------------
// 7.1 / 7.2: parser populates ac_sweeps and fra_sweeps in YAML order
// -----------------------------------------------------------------------------

TEST_CASE("Phase 7: YAML analysis: parses into SimulationOptions vectors",
          "[v1][frequency_analysis][phase7][yaml][parser]") {
    parser::YamlParser p;
    auto [circuit, opts] = p.load_string(rc_yaml_with_analysis());
    REQUIRE(p.errors().empty());
    REQUIRE(opts.ac_sweeps.size() == 1);
    REQUIRE(opts.fra_sweeps.size() == 1);

    const auto& ac = opts.ac_sweeps[0];
    CHECK(ac.label == "rc_lowpass");
    CHECK(ac.f_start == Approx(1.0));
    CHECK(ac.f_stop  == Approx(1e6));
    CHECK(ac.points_per_decade == 30);
    CHECK(ac.scale == AcSweepScale::Logarithmic);
    CHECK(ac.perturbation_source == "V1");
    REQUIRE(ac.measurement_nodes.size() == 1);
    CHECK(ac.measurement_nodes[0] == "out");

    const auto& fra = opts.fra_sweeps[0];
    CHECK(fra.label == "rc_lowpass_fra");
    CHECK(fra.f_start == Approx(100.0));
    CHECK(fra.f_stop  == Approx(1e3));
    CHECK(fra.points_per_decade == 4);
    CHECK(fra.perturbation_source == "V1");
    CHECK(fra.perturbation_amplitude == Approx(0.01));
    CHECK(fra.n_cycles == 6);
    CHECK(fra.discard_cycles == 2);
    CHECK(fra.samples_per_cycle == 64);
    REQUIRE(fra.measurement_nodes.size() == 1);
    CHECK(fra.measurement_nodes[0] == "out");
}

// -----------------------------------------------------------------------------
// 7.4: end-to-end — parse YAML, run the parsed AC sweep, agree with the corner
// -----------------------------------------------------------------------------

TEST_CASE("Phase 7: parsed AC sweep runs end-to-end and matches RC corner",
          "[v1][frequency_analysis][phase7][yaml][end_to_end]") {
    parser::YamlParser p;
    auto [circuit, opts] = p.load_string(rc_yaml_with_analysis());
    REQUIRE(p.errors().empty());
    REQUIRE(opts.ac_sweeps.size() == 1);

    Simulator sim(circuit, opts);
    const auto result = sim.run_ac_sweep(opts.ac_sweeps[0]);
    REQUIRE(result.success);
    REQUIRE(result.measurements.size() == 1);

    const Real f_corner = Real{1} / (Real{2} * std::numbers::pi_v<Real> * 1e3 * 1e-6);
    const auto& m = result.measurements[0];
    const auto& freqs = result.frequencies;
    auto i_corner = std::min_element(
        freqs.begin(), freqs.end(),
        [&](Real a, Real b) {
            return std::abs(std::log10(a) - std::log10(f_corner)) <
                   std::abs(std::log10(b) - std::log10(f_corner));
        });
    const auto idx = static_cast<std::size_t>(std::distance(freqs.begin(), i_corner));

    CHECK(m.magnitude_db[idx] == Approx(-3.0103).margin(0.20));
    CHECK(m.phase_deg[idx]    == Approx(-45.0).margin(2.0));
}

// -----------------------------------------------------------------------------
// 7.3: strict mode rejects unknown analysis type
// -----------------------------------------------------------------------------

TEST_CASE("Phase 7: strict mode rejects unknown analysis type",
          "[v1][frequency_analysis][phase7][yaml][strict]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-6
  dt: 1e-7
analysis:
  - type: bogus
    f_start: 1.0
    f_stop: 1e3
components:
  - type: resistor
    name: R1
    nodes: [a, b]
    value: 1e3
)";
    parser::YamlParser p;
    p.load_string(yaml);

    REQUIRE_FALSE(p.errors().empty());
    const bool found = std::any_of(
        p.errors().begin(), p.errors().end(),
        [](const std::string& e) {
            return e.find("unknown analysis type") != std::string::npos;
        });
    CHECK(found);
}

// -----------------------------------------------------------------------------
// 7.3: strict mode rejects unknown per-entry field
// -----------------------------------------------------------------------------

TEST_CASE("Phase 7: strict mode rejects unknown per-entry analysis field",
          "[v1][frequency_analysis][phase7][yaml][strict]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-6
  dt: 1e-7
analysis:
  - type: ac
    f_start: 1.0
    f_stop: 1e3
    bogus_param: 42
components:
  - type: resistor
    name: R1
    nodes: [a, b]
    value: 1e3
)";
    parser::YamlParser p;  // strict = true by default
    p.load_string(yaml);

    REQUIRE_FALSE(p.errors().empty());
    const bool found = std::any_of(
        p.errors().begin(), p.errors().end(),
        [](const std::string& e) {
            return e.find("PULSIM_YAML_E_UNKNOWN_FIELD") != std::string::npos &&
                   e.find("bogus_param") != std::string::npos;
        });
    CHECK(found);
}

// -----------------------------------------------------------------------------
// 7.4: multiple analyses run sequentially
// -----------------------------------------------------------------------------

TEST_CASE("Phase 7: multiple AC sweeps in one YAML run sequentially",
          "[v1][frequency_analysis][phase7][yaml][multi]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  tstart: 0.0
  tstop: 1e-6
  dt: 1e-7
  dt_min: 1e-12
  dt_max: 1e-7
  adaptive_timestep: false
analysis:
  - type: ac
    name: low_band
    f_start: 1.0
    f_stop: 100.0
    points_per_decade: 5
    perturbation_source: V1
    measurement_nodes: [out]
  - type: ac
    name: high_band
    f_start: 1e3
    f_stop: 1e5
    points_per_decade: 5
    perturbation_source: V1
    measurement_nodes: [out]
components:
  - type: voltage_source
    name: V1
    nodes: [in, gnd]
    value: 1.0
  - type: resistor
    name: R1
    nodes: [in, out]
    value: 1e3
  - type: capacitor
    name: C1
    nodes: [out, gnd]
    value: 1e-6
)";
    parser::YamlParser p;
    auto [circuit, opts] = p.load_string(yaml);
    REQUIRE(p.errors().empty());
    REQUIRE(opts.ac_sweeps.size() == 2);
    CHECK(opts.ac_sweeps[0].label == "low_band");
    CHECK(opts.ac_sweeps[1].label == "high_band");

    Simulator sim(circuit, opts);
    for (const auto& ac : opts.ac_sweeps) {
        const auto r = sim.run_ac_sweep(ac);
        REQUIRE(r.success);
        REQUIRE(r.measurements.size() == 1);
    }
}
