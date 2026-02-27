#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>

#include "pulsim/v1/high_performance.hpp"
#include "pulsim/v1/parser/yaml_parser.hpp"

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("YAML solver order keeps separate fallback order", "[v1][solver][yaml]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-6
  dt: 1e-6
  solver:
    order: [gmres]
    fallback_order: [klu]
components:
  - type: voltage_source
    name: V1
    nodes: [n1, 0]
    waveform:
      type: dc
      value: 1.0
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    parser::YamlParser parser;
    auto [circuit, options] = parser.load_string(yaml);

    REQUIRE(options.linear_solver.order.size() == 1);
    REQUIRE(options.linear_solver.fallback_order.size() == 1);
    CHECK(options.linear_solver.order[0] == LinearSolverKind::GMRES);
    CHECK(options.linear_solver.fallback_order[0] == LinearSolverKind::KLU);
}

TEST_CASE("YAML integrator selection supports stiff methods", "[v1][integrator][yaml]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-6
  dt: 1e-6
  integrator: tr-bdf2
components:
  - type: voltage_source
    name: V1
    nodes: [n1, 0]
    waveform:
      type: dc
      value: 1.0
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    parser::YamlParser parser;
    auto [circuit, options] = parser.load_string(yaml);
    CHECK(options.integrator == Integrator::TRBDF2);

    const std::string yaml_ros = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-6
  dt: 1e-6
  integrator: rosenbrock
components:
  - type: voltage_source
    name: V1
    nodes: [n1, 0]
    waveform:
      type: dc
      value: 1.0
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    auto [circuit2, options2] = parser.load_string(yaml_ros);
    CHECK(options2.integrator == Integrator::RosenbrockW);
}

TEST_CASE("YAML periodic solver options are parsed", "[v1][steady][yaml]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-6
  dt: 1e-6
  shooting:
    period: 1e-3
    max_iterations: 5
    tolerance: 1e-4
    relaxation: 0.8
    store_last_transient: false
  hb:
    period: 2e-3
    num_samples: 16
    max_iterations: 7
    tolerance: 1e-3
    relaxation: 0.9
    initialize_from_transient: false
components:
  - type: voltage_source
    name: V1
    nodes: [n1, 0]
    waveform:
      type: dc
      value: 1.0
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    parser::YamlParser parser;
    auto [circuit, options] = parser.load_string(yaml);

    REQUIRE(options.enable_periodic_shooting);
    CHECK(options.periodic_options.period == Approx(1e-3));
    CHECK(options.periodic_options.max_iterations == 5);
    CHECK(options.periodic_options.tolerance == Approx(1e-4));
    CHECK(options.periodic_options.relaxation == Approx(0.8));
    CHECK(options.periodic_options.store_last_transient == false);

    REQUIRE(options.enable_harmonic_balance);
    CHECK(options.harmonic_balance.period == Approx(2e-3));
    CHECK(options.harmonic_balance.num_samples == 16);
    CHECK(options.harmonic_balance.max_iterations == 7);
    CHECK(options.harmonic_balance.tolerance == Approx(1e-3));
    CHECK(options.harmonic_balance.relaxation == Approx(0.9));
    CHECK(options.harmonic_balance.initialize_from_transient == false);
}

TEST_CASE("CG is rejected for non-SPD matrices", "[v1][solver][cg]") {
    LinearSolverStackConfig cfg;
    cfg.order = {LinearSolverKind::CG, LinearSolverKind::SparseLU};
    cfg.fallback_order = {LinearSolverKind::SparseLU};
    cfg.auto_select = false;
    cfg.allow_fallback = true;
    cfg.iterative_config.enable_scaling = false;
    cfg.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::Jacobi;

    RuntimeLinearSolver solver(cfg);

    SparseMatrix A(2, 2);
    A.insert(0, 0) = 4.0;
    A.insert(0, 1) = 1.0;
    A.insert(1, 0) = 0.0;  // break symmetry
    A.insert(1, 1) = 2.0;
    A.makeCompressed();

    REQUIRE(solver.factorize(A));
    auto active = solver.active_kind();
    REQUIRE(active.has_value());
    CHECK(*active == LinearSolverKind::SparseLU);
}

TEST_CASE("ILUT preconditioner is applied and tracked", "[v1][solver][ilut]") {
    LinearSolverStackConfig cfg;
    cfg.order = {LinearSolverKind::GMRES};
    cfg.fallback_order = {};
    cfg.auto_select = false;
    cfg.allow_fallback = false;
    cfg.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::ILUT;
    cfg.iterative_config.ilut_drop_tolerance = 1e-3;
    cfg.iterative_config.ilut_fill_factor = 5.0;

    RuntimeLinearSolver solver(cfg);

    SparseMatrix A(2, 2);
    A.insert(0, 0) = 4.0;
    A.insert(0, 1) = 1.0;
    A.insert(1, 0) = 2.0;
    A.insert(1, 1) = 3.0;
    A.makeCompressed();

    Vector b(2);
    b << 1.0, 2.0;

    REQUIRE(solver.factorize(A));
    auto result = solver.solve(b);
    REQUIRE(result);
    auto telemetry = solver.telemetry();
    REQUIRE(telemetry.last_preconditioner.has_value());
    CHECK(*telemetry.last_preconditioner == IterativeSolverConfig::PreconditionerKind::ILUT);
}

TEST_CASE("GMRES solve is stable after factorize with temporary matrix",
          "[v1][solver][gmres][lifetime]") {
    LinearSolverStackConfig cfg;
    cfg.order = {LinearSolverKind::GMRES};
    cfg.fallback_order = {};
    cfg.auto_select = false;
    cfg.allow_fallback = false;
    cfg.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::None;
    cfg.iterative_config.max_iterations = 64;
    cfg.iterative_config.tolerance = 1e-10;
    cfg.iterative_config.restart = 8;

    RuntimeLinearSolver solver(cfg);

    auto make_matrix = []() {
        SparseMatrix A(2, 2);
        A.insert(0, 0) = 4.0;
        A.insert(0, 1) = 1.0;
        A.insert(1, 0) = 2.0;
        A.insert(1, 1) = 3.0;
        A.makeCompressed();
        return A;
    };

    Vector b(2);
    b << 1.0, 2.0;

    REQUIRE(solver.factorize(make_matrix()));
    auto result = solver.solve(b);
    REQUIRE(result);
    REQUIRE(result.solution.has_value());
    CHECK(result.solution->size() == 2);
}

TEST_CASE("AMG preconditioner is feature-flagged and unavailable by default", "[v1][solver][amg]") {
    LinearSolverStackConfig cfg;
    cfg.order = {LinearSolverKind::GMRES};
    cfg.fallback_order = {};
    cfg.auto_select = false;
    cfg.allow_fallback = false;
    cfg.iterative_config.preconditioner = IterativeSolverConfig::PreconditionerKind::AMG;

    RuntimeLinearSolver solver(cfg);

    SparseMatrix A(2, 2);
    A.insert(0, 0) = 2.0;
    A.insert(0, 1) = 1.0;
    A.insert(1, 0) = 1.0;
    A.insert(1, 1) = 2.0;
    A.makeCompressed();

#ifdef PULSIM_HAS_HYPRE
    REQUIRE(solver.factorize(A));
    auto result = solver.solve(Vector::Ones(2));
    REQUIRE(result);
#else
    CHECK_FALSE(solver.factorize(A));
#endif
}

TEST_CASE("YAML parser keeps legacy SI suffix compatibility", "[v1][yaml][legacy]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 2milli
  dt: 10micro
components:
  - type: voltage_source
    name: V1
    nodes: [in, 0]
    waveform:
      type: dc
      value: 12
  - type: resistor
    name: R1
    nodes: [in, out]
    value: 2kilo
  - type: resistor
    name: R2
    nodes: [out, 0]
    value: 2K
  - type: capacitor
    name: C1
    nodes: [out, 0]
    value: 1uF
)";

    parser::YamlParser parser;
    auto [circuit, options] = parser.load_string(yaml);
    CHECK(options.tstop == Approx(2e-3));
    CHECK(options.dt == Approx(10e-6));

    options.tstart = 0.0;
    options.dt_min = options.dt;
    options.dt_max = options.dt;
    options.adaptive_timestep = false;
    options.enable_bdf_order_control = false;
    options.newton_options.num_nodes = circuit.num_nodes();
    options.newton_options.num_branches = circuit.num_branches();
    options.linear_solver.order = {LinearSolverKind::KLU, LinearSolverKind::SparseLU};
    options.linear_solver.auto_select = false;
    options.linear_solver.allow_fallback = true;

    Simulator sim(circuit, options);
    auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);

    const auto& node_names = circuit.node_names();
    auto it = std::find(node_names.begin(), node_names.end(), "out");
    REQUIRE(it != node_names.end());
    const auto out_idx = static_cast<Index>(std::distance(node_names.begin(), it));
    CHECK(dc.newton_result.solution[out_idx] == Approx(6.0).margin(1e-6));
}

TEST_CASE("YAML parser maps electro-thermal configuration and rejects JSON input",
          "[v1][yaml][thermal]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  tstart: 0
  tstop: 1e-4
  dt: 1e-6
  thermal:
    enabled: true
    ambient: 30
    policy: loss_with_temperature_scaling
    default_rth: 1.5
    default_cth: 0.2
components:
  - type: voltage_source
    name: V1
    nodes: [in, 0]
    waveform: {type: dc, value: 12}
  - type: mosfet
    name: M1
    nodes: [gate, in, 0]
    vth: 2.5
    kp: 5
    thermal:
      rth: 0.8
      cth: 0.05
      temp_init: 35
      temp_ref: 25
      alpha: 0.006
)";

    parser::YamlParser parser;
    auto [circuit, options] = parser.load_string(yaml);
    REQUIRE(parser.errors().empty());
    REQUIRE(options.thermal.enable);
    CHECK(options.thermal.ambient == Approx(30.0));
    CHECK(options.thermal.default_rth == Approx(1.5));
    CHECK(options.thermal.default_cth == Approx(0.2));
    CHECK(options.thermal.policy == ThermalCouplingPolicy::LossWithTemperatureScaling);

    REQUIRE(options.thermal_devices.contains("M1"));
    const auto& cfg = options.thermal_devices.at("M1");
    CHECK(cfg.rth == Approx(0.8));
    CHECK(cfg.cth == Approx(0.05));
    CHECK(cfg.temp_init == Approx(35.0));
    CHECK(cfg.temp_ref == Approx(25.0));
    CHECK(cfg.alpha == Approx(0.006));

    parser::YamlParser parser_json;
    parser_json.load_string(R"({"schema":"pulsim-v1","version":1,"components":[]})");
    REQUIRE_FALSE(parser_json.errors().empty());
    CHECK(parser_json.errors().front().find("JSON netlists are unsupported") != std::string::npos);
}

TEST_CASE("YAML parser maps fallback policy controls", "[v1][yaml][fallback]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-4
  dt: 1e-6
  max_step_retries: 4
  fallback:
    trace_retries: true
    enable_transient_gmin: true
    gmin_retry_threshold: 2
    gmin_initial: 1e-8
    gmin_max: 1e-4
    gmin_growth: 5
components:
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    parser::YamlParser parser;
    auto [circuit, options] = parser.load_string(yaml);
    REQUIRE(parser.errors().empty());
    CHECK(options.max_step_retries == 4);
    CHECK(options.fallback_policy.trace_retries);
    CHECK(options.fallback_policy.enable_transient_gmin);
    CHECK(options.fallback_policy.gmin_retry_threshold == 2);
    CHECK(options.fallback_policy.gmin_initial == Approx(1e-8));
    CHECK(options.fallback_policy.gmin_max == Approx(1e-4));
    CHECK(options.fallback_policy.gmin_growth == Approx(5.0));
    CHECK(circuit.num_devices() == 1);
}

TEST_CASE("YAML parser maps model_regularization controls with safe defaults",
          "[v1][yaml][model-regularization]") {
    const std::string yaml_with_block = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-4
  dt: 1e-6
  model_regularization:
    enable_auto: true
    apply_only_in_recovery: true
    retry_threshold: 2
    max_escalations: 3
    escalation_factor: 1.5
    mosfet_kp_max: 6.0
    mosfet_g_off_min: 1e-8
    diode_g_on_max: 200.0
    diode_g_off_min: 1e-10
    igbt_g_on_max: 4000.0
    igbt_g_off_min: 1e-10
    switch_g_on_max: 1e5
    switch_g_off_min: 1e-10
    vcswitch_g_on_max: 2e5
    vcswitch_g_off_min: 1e-10
components:
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    parser::YamlParser parser;
    auto [circuit_with_block, options_with_block] = parser.load_string(yaml_with_block);
    REQUIRE(parser.errors().empty());
    CHECK(options_with_block.model_regularization.enable_auto);
    CHECK(options_with_block.model_regularization.apply_only_in_recovery);
    CHECK(options_with_block.model_regularization.retry_threshold == 2);
    CHECK(options_with_block.model_regularization.max_escalations == 3);
    CHECK(options_with_block.model_regularization.escalation_factor == Approx(1.5));
    CHECK(options_with_block.model_regularization.mosfet_kp_max == Approx(6.0));
    CHECK(options_with_block.model_regularization.vcswitch_g_on_max == Approx(2e5));
    CHECK(circuit_with_block.num_devices() == 1);

    const std::string yaml_without_block = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-4
  dt: 1e-6
components:
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    auto [circuit_without_block, options_without_block] = parser.load_string(yaml_without_block);
    REQUIRE(parser.errors().empty());
    CHECK_FALSE(options_without_block.model_regularization.enable_auto);
    CHECK(options_without_block.model_regularization.retry_threshold == 2);
    CHECK(options_without_block.model_regularization.max_escalations == 4);
    CHECK(circuit_without_block.num_devices() == 1);
}

TEST_CASE("YAML parser rejects invalid model_regularization ranges in strict mode",
          "[v1][yaml][model-regularization]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-4
  dt: 1e-6
  model_regularization:
    retry_threshold: 0
    max_escalations: -1
    escalation_factor: 0
components:
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    parser::YamlParser parser;
    parser.load_string(yaml);
    REQUIRE_FALSE(parser.errors().empty());
    std::string joined;
    for (const auto& err : parser.errors()) {
        joined += err;
        joined.push_back('\n');
    }
    CHECK(joined.find("simulation.model_regularization.retry_threshold") != std::string::npos);
    CHECK(joined.find("simulation.model_regularization.max_escalations") != std::string::npos);
    CHECK(joined.find("simulation.model_regularization.escalation_factor") != std::string::npos);
}

TEST_CASE("YAML parser emits migration warnings for legacy backend controls in non-strict mode",
          "[v1][yaml][backend]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-4
  dt: 1e-6
  backend: auto
  sundials:
    enabled: true
  fallback:
    enable_backend_escalation: true
    backend_escalation_threshold: 3
    enable_native_reentry: true
    sundials_recovery_window: 5e-6
components:
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    parser::YamlParserOptions parser_options;
    parser_options.strict = false;
    parser::YamlParser parser(parser_options);
    auto [circuit, options] = parser.load_string(yaml);
    REQUIRE(parser.errors().empty());
    CHECK(options.step_mode == TransientStepMode::Variable);
    CHECK(options.fallback_policy.trace_retries);
    REQUIRE_FALSE(parser.warnings().empty());
    std::string joined;
    for (const auto& warning : parser.warnings()) {
        joined += warning;
        joined.push_back('\n');
    }
    CHECK(joined.find("simulation.backend") != std::string::npos);
    CHECK(joined.find("simulation.sundials") != std::string::npos);
    CHECK(joined.find("simulation.fallback.enable_backend_escalation") != std::string::npos);
    CHECK(joined.find("simulation.fallback.backend_escalation_threshold") != std::string::npos);
    CHECK(joined.find("simulation.fallback.enable_native_reentry") != std::string::npos);
    CHECK(joined.find("simulation.fallback.sundials_recovery_window") != std::string::npos);
    CHECK(circuit.num_devices() == 1);
}

TEST_CASE("YAML parser applies canonical step_mode and advanced overrides",
          "[v1][yaml][step-mode]") {
    const std::string yaml_fixed = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 2e-5
  dt: 1e-6
  step_mode: fixed
  advanced:
    solver:
      order: [klu]
components:
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    parser::YamlParser parser;
    auto [circuit_fixed, options_fixed] = parser.load_string(yaml_fixed);
    REQUIRE(parser.errors().empty());
    CHECK(options_fixed.step_mode == TransientStepMode::Fixed);
    CHECK(options_fixed.step_mode_explicit);
    CHECK_FALSE(options_fixed.adaptive_timestep);
    REQUIRE(options_fixed.linear_solver.order.size() == 1);
    CHECK(options_fixed.linear_solver.order.front() == LinearSolverKind::KLU);
    CHECK(circuit_fixed.num_devices() == 1);

    const std::string yaml_variable = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 2e-5
  dt: 1e-6
  step_mode: variable
components:
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    auto [circuit_var, options_var] = parser.load_string(yaml_variable);
    REQUIRE(parser.errors().empty());
    CHECK(options_var.step_mode == TransientStepMode::Variable);
    CHECK(options_var.step_mode_explicit);
    CHECK(options_var.adaptive_timestep);
    CHECK(options_var.enable_bdf_order_control);
    CHECK(options_var.integrator == Integrator::TRBDF2);
    CHECK(circuit_var.num_devices() == 1);
}

TEST_CASE("YAML parser emits strict migration diagnostics for legacy backend keys",
          "[v1][yaml][migration]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-4
  dt: 1e-6
  backend: auto
  advanced:
    backend: sundials
    sundials:
      enabled: true
components:
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    parser::YamlParser parser;
    parser.load_string(yaml);
    REQUIRE_FALSE(parser.errors().empty());
    std::string joined;
    for (const auto& err : parser.errors()) {
        joined += err;
        joined.push_back('\n');
    }
    CHECK(joined.find("simulation.backend") != std::string::npos);
    CHECK(joined.find("simulation.advanced.backend") != std::string::npos);
    CHECK(joined.find("simulation.advanced.sundials") != std::string::npos);
    CHECK(joined.find("simulation.step_mode") != std::string::npos);
}

TEST_CASE("YAML parser emits coded unknown-field diagnostics with field paths",
          "[v1][yaml][diagnostic]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  unknown_field: 123
components:
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    parser::YamlParser parser;
    parser.load_string(yaml);
    REQUIRE_FALSE(parser.errors().empty());
    CHECK(std::any_of(parser.errors().begin(), parser.errors().end(),
                      [](const std::string& msg) {
                          return msg.find("PULSIM_YAML_E_UNKNOWN_FIELD") != std::string::npos &&
                                 msg.find("simulation.unknown_field") != std::string::npos;
                      }));
}

TEST_CASE("YAML parser emits typed mismatch diagnostics for known fields",
          "[v1][yaml][diagnostic]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  dt: {invalid: true}
components:
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    parser::YamlParser parser;
    parser.load_string(yaml);
    REQUIRE_FALSE(parser.errors().empty());
    CHECK(std::any_of(parser.errors().begin(), parser.errors().end(),
                      [](const std::string& msg) {
                          return msg.find("PULSIM_YAML_E_TYPE_MISMATCH") != std::string::npos &&
                                 msg.find("simulation.dt") != std::string::npos &&
                                 msg.find("expected number") != std::string::npos;
                      }));
}

TEST_CASE("YAML parser emits migration-window warnings for deprecated canonical aliases",
          "[v1][yaml][migration]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  adaptive_timestep: true
  tstop: 1e-5
  dt: 1e-6
components:
  - type: resistor
    name: R1
    nodes: [n1, 0]
    value: 1k
)";

    parser::YamlParserOptions parser_options;
    parser_options.strict = false;
    parser::YamlParser parser(parser_options);
    parser.load_string(yaml);
    REQUIRE(parser.errors().empty());
    CHECK(std::any_of(parser.warnings().begin(), parser.warnings().end(),
                      [](const std::string& msg) {
                          return msg.find("PULSIM_YAML_W_DEPRECATED_FIELD") != std::string::npos &&
                                 msg.find("simulation.adaptive_timestep") != std::string::npos &&
                                 msg.find("simulation.step_mode") != std::string::npos;
                      }));
}

TEST_CASE("YAML parser accepts GUI parity slice with virtual and surrogate components",
          "[v1][yaml][gui-parity]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 2e-5
  dt: 1e-6
components:
  - type: voltage_source
    name: V1
    nodes: [in, 0]
    waveform: {type: dc, value: 24}
  - type: switch
    name: SW1
    nodes: [in, sw]
    initial_state: true
  - type: BJT_NPN
    name: QN1
    nodes: [ctrl, sw, 0]
    beta: 120
  - type: THYRISTOR
    name: SCR1
    nodes: [gate, sw, 0]
    gate_threshold: 0.8
  - type: FUSE
    name: F1
    nodes: [sw, load]
    initial_state: true
  - type: SATURABLE_INDUCTOR
    name: Lsat
    nodes: [load, 0]
    inductance: 1m
    saturation_current: 2
    saturation_inductance: 200u
  - type: COUPLED_INDUCTOR
    name: Lcpl
    nodes: [in, 0, aux, 0]
    l1: 1m
    l2: 2m
    coupling: 0.95
  - type: OP_AMP
    name: A1
    nodes: [ref, fb, ctrl]
    gain: 1e4
  - type: VOLTAGE_PROBE
    name: VP1
    nodes: [sw, 0]
  - type: CURRENT_PROBE
    name: IP1
    nodes: [in, 0]
    target_component: V1
  - type: POWER_PROBE
    name: PP1
    nodes: [sw, 0]
    target_component: V1
  - type: SIGNAL_MUX
    name: MUX1
    nodes: [sig_a, sig_b, sig_out]
)";

    parser::YamlParser parser;
    auto [circuit, options] = parser.load_string(yaml);
    INFO("Errors: " << parser.errors().size());
    INFO("Warnings: " << parser.warnings().size());
    REQUIRE(parser.errors().empty());
    CHECK(circuit.num_devices() >= 7);
    CHECK(circuit.num_virtual_components() >= 5);
    CHECK(std::any_of(parser.warnings().begin(), parser.warnings().end(),
                      [](const std::string& msg) {
                          return msg.find("PULSIM_YAML_W_COMPONENT_VIRTUAL") != std::string::npos;
                      }));
    CHECK(std::any_of(parser.warnings().begin(), parser.warnings().end(),
                      [](const std::string& msg) {
                          return msg.find("PULSIM_YAML_W_COMPONENT_SURROGATE") != std::string::npos;
                      }));
}

TEST_CASE("YAML parser emits coded diagnostics for invalid parity descriptors",
          "[v1][yaml][gui-parity][diagnostics]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
components:
  - type: RELAY
    name: K1
    nodes: [coil_p, coil_n, com]
)";

    parser::YamlParser parser;
    parser.load_string(yaml);
    REQUIRE_FALSE(parser.errors().empty());
    CHECK(std::any_of(parser.errors().begin(), parser.errors().end(),
                      [](const std::string& msg) {
                          return msg.find("PULSIM_YAML_E_PIN_COUNT") != std::string::npos;
                      }));
}

TEST_CASE("YAML parser accepts declared GUI parity type matrix",
          "[v1][yaml][gui-parity][matrix]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
components:
  - {type: voltage_source, name: V1, nodes: [n_in, 0], waveform: {type: dc, value: 24}}
  - {type: SWITCH, name: SW1, nodes: [n_in, n_sw], initial_state: true}
  - {type: BJT_NPN, name: QN1, nodes: [n_ctrl, n_sw, 0], beta: 100}
  - {type: BJT_PNP, name: QP1, nodes: [n_ctrl, n_in, n_sw], beta: 80}
  - {type: THYRISTOR, name: SCR1, nodes: [n_gate, n_sw, 0]}
  - {type: TRIAC, name: TRI1, nodes: [n_gate, n_sw, n_aux]}
  - {type: FUSE, name: F1, nodes: [n_sw, n_load], initial_state: true}
  - {type: CIRCUIT_BREAKER, name: CB1, nodes: [n_load, n_out], initial_state: true}
  - {type: RELAY, name: K1, nodes: [n_cp, n_cn, n_com, n_no, n_nc]}
  - {type: SATURABLE_INDUCTOR, name: Lsat, nodes: [n_out, 0], value: 1m, saturation_current: 2, saturation_inductance: 200u}
  - {type: COUPLED_INDUCTOR, name: Lcpl, nodes: [n_in, 0, n_aux, 0], l1: 1m, l2: 2m, coupling: 0.95}
  - {type: SNUBBER_RC, name: Sn1, nodes: [n_sw, 0], resistance: 100, capacitance: 1n}
  - {type: OP_AMP, name: OA1, nodes: [n_ref, n_fb, n_ctrl]}
  - {type: COMPARATOR, name: CMP1, nodes: [n_ref, n_fb, n_cmp]}
  - {type: PI_CONTROLLER, name: PI1, nodes: [n_ref, n_fb, n_ctrl]}
  - {type: PID_CONTROLLER, name: PID1, nodes: [n_ref, n_fb, n_ctrl]}
  - {type: MATH_BLOCK, name: M1, nodes: [n_a, n_b, n_c]}
  - {type: PWM_GENERATOR, name: PWM1, nodes: [n_ctrl, n_pwm]}
  - {type: INTEGRATOR, name: INT1, nodes: [n_a, n_b]}
  - {type: DIFFERENTIATOR, name: DIF1, nodes: [n_a, n_b]}
  - {type: LIMITER, name: LIM1, nodes: [n_a, n_b]}
  - {type: RATE_LIMITER, name: RL1, nodes: [n_a, n_b]}
  - {type: HYSTERESIS, name: HYS1, nodes: [n_a, n_b]}
  - {type: LOOKUP_TABLE, name: LUT1, nodes: [n_a, n_b], x: [0, 1], y: [0, 1]}
  - {type: TRANSFER_FUNCTION, name: TF1, nodes: [n_a, n_b]}
  - {type: DELAY_BLOCK, name: DLY1, nodes: [n_a, n_b]}
  - {type: SAMPLE_HOLD, name: SH1, nodes: [n_a, n_b]}
  - {type: STATE_MACHINE, name: SM1, nodes: [n_state]}
  - {type: VOLTAGE_PROBE, name: VP1, nodes: [n_sw, 0]}
  - {type: CURRENT_PROBE, name: IP1, nodes: [n_in, 0]}
  - {type: POWER_PROBE, name: PP1, nodes: [n_sw, 0]}
  - {type: ELECTRICAL_SCOPE, name: ES1, nodes: [n_a, n_b]}
  - {type: THERMAL_SCOPE, name: TS1, nodes: [n_a, n_b]}
  - {type: SIGNAL_MUX, name: MUX1, nodes: [n_a, n_b, n_c]}
  - {type: SIGNAL_DEMUX, name: DMX1, nodes: [n_a, n_b, n_c]}
)";

    parser::YamlParser parser;
    auto [circuit, options] = parser.load_string(yaml);
    INFO("Errors: " << parser.errors().size());
    INFO("Warnings: " << parser.warnings().size());
    REQUIRE(parser.errors().empty());
    CHECK(std::none_of(parser.errors().begin(), parser.errors().end(),
                       [](const std::string& msg) {
                           return msg.find("PULSIM_YAML_E_COMPONENT_UNSUPPORTED") != std::string::npos;
                       }));
}

TEST_CASE("Runtime virtual probe evaluation derives signal values", "[v1][virtual][probe]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    const auto gnd = Circuit::ground();
    circuit.add_voltage_source("V1", n_in, gnd, 5.0);
    circuit.add_virtual_component("voltage_probe", "VP", {n_in, gnd});
    circuit.add_virtual_component("current_probe", "IP", {n_in, gnd}, {},
                                  {{"target_component", "V1"}});
    circuit.add_virtual_component("power_probe", "PP", {n_in, gnd}, {},
                                  {{"target_component", "V1"}});

    Vector x = Vector::Zero(circuit.system_size());
    x[n_in] = 10.0;
    x[circuit.num_nodes()] = 0.25;  // V1 branch current
    const auto signals = circuit.evaluate_virtual_signals(x);

    REQUIRE(signals.contains("VP"));
    REQUIRE(signals.contains("IP"));
    REQUIRE(signals.contains("PP"));
    CHECK(signals.at("VP") == Approx(10.0));
    CHECK(signals.at("IP") == Approx(0.25));
    CHECK(signals.at("PP") == Approx(2.5));
}
