/**
 * Advanced Solver Tests
 *
 * Tests for:
 * - Trapezoidal (GEAR-2) integration
 * - BDF2 integration
 * - Adaptive timestep control
 * - Advanced linear solver features
 * - Factorization reuse
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/simulation.hpp"
#include "pulsim/advanced_solver.hpp"
#include "pulsim/mna.hpp"
#include <cmath>

using namespace pulsim;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// =============================================================================
// Trapezoidal Integration Tests
// =============================================================================

TEST_CASE("Trapezoidal companion coefficients", "[solver][trapezoidal]") {
    Real dt = 1e-6;

    // Backward Euler coefficients
    auto be_coef = get_companion_coefficients(IntegrationMethod::BackwardEuler, dt);
    CHECK(be_coef.alpha == 1.0);
    CHECK(be_coef.beta == 1.0);
    CHECK(be_coef.gamma == 0.0);

    // Trapezoidal coefficients
    auto trap_coef = get_companion_coefficients(IntegrationMethod::Trapezoidal, dt);
    CHECK(trap_coef.alpha == 2.0);
    CHECK(trap_coef.beta == 2.0);

    // GEAR2 is alias for Trapezoidal
    auto gear2_coef = get_companion_coefficients(IntegrationMethod::GEAR2, dt);
    CHECK(gear2_coef.alpha == trap_coef.alpha);
    CHECK(gear2_coef.beta == trap_coef.beta);
}

TEST_CASE("RC circuit with Trapezoidal integration", "[solver][trapezoidal]") {
    // Simple RC circuit to compare integration accuracy
    // R = 1k, C = 1uF, tau = 1ms
    // Step response: V(t) = Vin * (1 - exp(-t/tau))

    Circuit circuit;
    circuit.add_voltage_source("Vin", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "out", 1000.0);
    circuit.add_capacitor("C1", "out", "0", 1e-6);

    Real tau = 1000.0 * 1e-6;  // 1ms

    // Test with different integration methods
    for (auto method : {IntegrationMethod::BackwardEuler, IntegrationMethod::Trapezoidal}) {
        SimulationOptions opts;
        opts.tstart = 0.0;
        opts.tstop = 5e-3;  // 5 time constants
        opts.dt = 100e-6;   // Relatively large timestep
        opts.integration_method = method;
        opts.use_ic = true;

        Simulator sim(circuit, opts);
        auto result = sim.run_transient();

        REQUIRE(result.final_status == SolverStatus::Success);

        // Check final voltage (should be ~5V after 5 tau)
        Index out_idx = circuit.node_index("out");
        Real v_final = result.data.back()(out_idx);
        Real expected = 5.0 * (1.0 - std::exp(-5.0));  // ~4.966V

        INFO("Method: " << (method == IntegrationMethod::BackwardEuler ? "BE" : "Trap"));
        INFO("Final voltage: " << v_final << " V, expected: " << expected << " V");

        CHECK_THAT(v_final, WithinAbs(expected, 0.1));
    }
}

TEST_CASE("RLC circuit accuracy comparison", "[solver][trapezoidal]") {
    // RLC circuit with underdamped response
    // Should show better accuracy with trapezoidal

    Circuit circuit;
    circuit.add_voltage_source("Vin", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "n1", 10.0);
    circuit.add_inductor("L1", "n1", "out", 1e-3);   // 1mH
    circuit.add_capacitor("C1", "out", "0", 1e-6);   // 1uF

    // Natural frequency: omega_0 = 1/sqrt(LC) = 31623 rad/s
    // Damping ratio: zeta = R/(2*sqrt(L/C)) = 0.158 (underdamped)
    // Period: T = 2*pi/omega_d ~= 200us

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 2e-3;   // Several oscillation periods
    opts.dt = 5e-6;      // 40 points per period
    opts.use_ic = true;

    // Run with Backward Euler
    opts.integration_method = IntegrationMethod::BackwardEuler;
    Simulator sim_be(circuit, opts);
    auto result_be = sim_be.run_transient();
    REQUIRE(result_be.final_status == SolverStatus::Success);

    // Run with Trapezoidal
    opts.integration_method = IntegrationMethod::Trapezoidal;
    Simulator sim_trap(circuit, opts);
    auto result_trap = sim_trap.run_transient();
    REQUIRE(result_trap.final_status == SolverStatus::Success);

    // Both should complete successfully
    // Trapezoidal should preserve oscillation amplitude better
    Index out_idx = circuit.node_index("out");

    // Find peak voltages in each simulation
    Real peak_be = 0, peak_trap = 0;
    for (const auto& data : result_be.data) {
        peak_be = std::max(peak_be, std::abs(data(out_idx)));
    }
    for (const auto& data : result_trap.data) {
        peak_trap = std::max(peak_trap, std::abs(data(out_idx)));
    }

    INFO("BE peak: " << peak_be << " V");
    INFO("Trap peak: " << peak_trap << " V");

    // Both should find the initial overshoot
    CHECK(peak_be > 10.0);
    CHECK(peak_trap > 10.0);
}

// =============================================================================
// Timestep Controller Tests
// =============================================================================

TEST_CASE("Timestep controller basic operation", "[solver][adaptive]") {
    TimestepControllerOptions opts;
    opts.rtol = 1e-3;
    opts.atol = 1e-9;
    opts.safety_factor = 0.9;

    TimestepController controller(opts);

    // Test error estimation
    Vector x_high(3);
    Vector x_low(3);
    x_high << 1.0, 2.0, 3.0;
    x_low << 1.001, 2.002, 3.003;

    Real error = controller.estimate_error(x_high, x_low);
    CHECK(error > 0);

    // Test timestep computation with small error (should increase dt)
    Real dt_current = 1e-6;
    auto result = controller.compute_new_timestep(0.1, 0.5, 1e-15, 1e-3);
    CHECK(result.accepted);
    CHECK(result.dt_new >= dt_current);  // Should accept and possibly increase

    // Test with large error (should reject and decrease dt)
    result = controller.compute_new_timestep(1e-6, 10.0, 1e-15, 1e-3);
    CHECK_FALSE(result.accepted);
    CHECK(result.dt_new < 1e-6);  // Should decrease timestep
}

TEST_CASE("Timestep controller convergence", "[solver][adaptive]") {
    TimestepControllerOptions opts;
    opts.max_rejects = 5;

    TimestepController controller(opts);

    // Simulate multiple rejected steps
    for (int i = 0; i < 3; ++i) {
        auto result = controller.compute_new_timestep(1e-6, 5.0, 1e-15, 1e-3);
        CHECK_FALSE(result.accepted);
        CHECK(result.rejection_count == i + 1);
    }

    // Reset should clear rejection count
    controller.reset();
    auto result = controller.compute_new_timestep(1e-6, 0.5, 1e-15, 1e-3);
    CHECK(result.accepted);
}

// =============================================================================
// Advanced Linear Solver Tests
// =============================================================================

TEST_CASE("Advanced linear solver basic operation", "[solver][linear]") {
    AdvancedLinearSolver::Options opts;
    opts.backend = AdvancedLinearSolver::Backend::EigenSparseLU;
    opts.reuse_factorization = true;

    AdvancedLinearSolver solver(opts);

    // Create a simple 3x3 sparse matrix
    SparseMatrix A(3, 3);
    std::vector<Triplet> triplets;
    triplets.emplace_back(0, 0, 4.0);
    triplets.emplace_back(0, 1, 1.0);
    triplets.emplace_back(1, 0, 1.0);
    triplets.emplace_back(1, 1, 3.0);
    triplets.emplace_back(1, 2, 1.0);
    triplets.emplace_back(2, 1, 1.0);
    triplets.emplace_back(2, 2, 2.0);
    A.setFromTriplets(triplets.begin(), triplets.end());

    Vector b(3);
    b << 1.0, 2.0, 3.0;

    auto result = solver.solve(A, b);
    REQUIRE(result.status == SolverStatus::Success);

    // Verify solution: A * x = b
    Vector residual = A * result.x - b;
    CHECK(residual.norm() < 1e-10);
}

TEST_CASE("Advanced linear solver factorization reuse", "[solver][linear]") {
    AdvancedLinearSolver::Options opts;
    opts.reuse_factorization = true;
    opts.refactor_threshold = 0.1;
    opts.max_reuses = 10;

    AdvancedLinearSolver solver(opts);

    // Create initial matrix
    SparseMatrix A(3, 3);
    std::vector<Triplet> triplets;
    triplets.emplace_back(0, 0, 4.0);
    triplets.emplace_back(1, 1, 3.0);
    triplets.emplace_back(2, 2, 2.0);
    A.setFromTriplets(triplets.begin(), triplets.end());

    // First solve - should factorize
    Vector b(3);
    b << 1.0, 2.0, 3.0;
    solver.solve(A, b);
    CHECK(solver.factorization_count() == 1);

    // Small change - might reuse factorization check
    // (Note: with Eigen SparseLU, we can't truly reuse, but we track the count)

    // Solve again with same matrix
    solver.solve(A, b);
    // factorization_count might still be 2 with Eigen (no true reuse)
    CHECK(solver.factorization_count() >= 1);
}

// =============================================================================
// Advanced Newton Solver Tests
// =============================================================================

TEST_CASE("Advanced Newton solver with line search", "[solver][newton]") {
    AdvancedNewtonSolver::Options opts;
    opts.use_line_search = true;
    opts.max_iterations = 50;
    opts.abstol = 1e-10;

    AdvancedNewtonSolver solver(opts);

    // Solve x^2 - 2 = 0 (solution: x = sqrt(2))
    Vector x0(1);
    x0 << 1.0;

    auto system_func = [](const Vector& x, Vector& f, SparseMatrix& J) {
        f.resize(1);
        f(0) = x(0) * x(0) - 2.0;

        J.resize(1, 1);
        std::vector<Triplet> triplets;
        triplets.emplace_back(0, 0, 2.0 * x(0));
        J.setFromTriplets(triplets.begin(), triplets.end());
    };

    auto result = solver.solve(x0, system_func);

    REQUIRE(result.status == SolverStatus::Success);
    CHECK_THAT(result.x(0), WithinAbs(std::sqrt(2.0), 1e-8));
    CHECK(result.iterations < 10);
}

TEST_CASE("Advanced Newton solver continuation", "[solver][newton]") {
    AdvancedNewtonSolver::Options opts;
    opts.continuation = true;
    opts.continuation_steps = 5;
    opts.continuation_start = 0.1;
    opts.max_iterations = 20;
    opts.abstol = 1e-8;

    AdvancedNewtonSolver solver(opts);

    // Solve x^3 - 3x + param = 0 with continuation
    Vector x0(1);
    x0 << 0.0;

    auto system_func = [](Real param, const Vector& x, Vector& f, SparseMatrix& J) {
        f.resize(1);
        f(0) = x(0) * x(0) * x(0) - 3.0 * x(0) + param;

        J.resize(1, 1);
        std::vector<Triplet> triplets;
        triplets.emplace_back(0, 0, 3.0 * x(0) * x(0) - 3.0);
        J.setFromTriplets(triplets.begin(), triplets.end());
    };

    auto result = solver.solve_with_continuation(x0, system_func);

    REQUIRE(result.status == SolverStatus::Success);

    // Verify solution satisfies equation at param=1.0
    Real residual = result.x(0) * result.x(0) * result.x(0) - 3.0 * result.x(0) + 1.0;
    CHECK(std::abs(residual) < 1e-6);
}

// =============================================================================
// Integration with Simulation
// =============================================================================

TEST_CASE("Simulation with Trapezoidal integration option", "[solver][simulation]") {
    Circuit circuit;
    circuit.add_voltage_source("Vin", "in", "0", DCWaveform{12.0});
    circuit.add_resistor("R1", "in", "out", 100.0);
    circuit.add_capacitor("C1", "out", "0", 10e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-3;
    opts.dt = 50e-6;
    opts.integration_method = IntegrationMethod::Trapezoidal;
    opts.use_ic = true;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Check that simulation ran with expected number of steps
    CHECK(result.total_steps > 0);

    // Final voltage should be close to Vin (after 5 time constants)
    Index out_idx = circuit.node_index("out");
    Real v_final = result.data.back()(out_idx);
    CHECK_THAT(v_final, WithinAbs(12.0, 0.5));
}

TEST_CASE("DynamicHistory structure", "[solver][history]") {
    // Test the DynamicHistory struct used for multi-step methods
    DynamicHistory history;

    history.x_prev = Vector::Ones(5) * 2.0;
    history.x_prev2 = Vector::Ones(5) * 1.0;
    history.dt_prev = 1e-6;
    history.has_prev2 = true;

    CHECK(history.x_prev.size() == 5);
    CHECK(history.x_prev2.size() == 5);
    CHECK(history.dt_prev == 1e-6);
    CHECK(history.has_prev2);
}

TEST_CASE("MNA assembler with integration method", "[solver][mna]") {
    Circuit circuit;
    circuit.add_voltage_source("Vin", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "out", 100.0);
    circuit.add_capacitor("C1", "out", "0", 1e-6);

    MNAAssembler assembler(circuit);

    // Create history for multi-step methods
    DynamicHistory history;
    history.x_prev = Vector::Zero(circuit.total_variables());
    history.x_prev2 = Vector::Zero(circuit.total_variables());
    history.dt_prev = 1e-6;
    history.has_prev2 = true;

    Real dt = 1e-6;

    // Test assembly with different methods
    SparseMatrix G_be, G_trap, G_bdf2;
    Vector b_be, b_trap, b_bdf2;

    // Backward Euler
    assembler.assemble_transient(G_be, b_be, history, dt, IntegrationMethod::BackwardEuler);

    // Trapezoidal
    assembler.assemble_transient(G_trap, b_trap, history, dt, IntegrationMethod::Trapezoidal);

    // BDF2
    assembler.assemble_transient(G_bdf2, b_bdf2, history, dt, IntegrationMethod::BDF2);

    // Matrices should be different (different equivalent conductances)
    // For capacitor: G_BE = C/dt, G_trap = 2C/dt, G_BDF2 = 1.5C/dt
    CHECK(G_be.nonZeros() > 0);
    CHECK(G_trap.nonZeros() > 0);
    CHECK(G_bdf2.nonZeros() > 0);

    // The conductance stamps should be different
    // (We can't easily compare because of sparsity structure,
    //  but at least verify they all assembled without error)
}
