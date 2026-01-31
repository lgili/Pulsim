#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

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
