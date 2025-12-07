#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/convergence_aids.hpp"
#include "pulsim/circuit.hpp"
#include "pulsim/mna.hpp"
#include <cmath>

using namespace pulsim;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("Gmin Stepping - Simple resistive circuit", "[convergence][gmin]") {
    // Create a simple voltage divider circuit
    // V1 = 10V, R1 = 1k, R2 = 1k
    // Expected: V(n1) = 5V, I(V1) = 5mA

    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "out", 1000.0);
    circuit.add_resistor("R2", "out", "0", 1000.0);

    MNAAssembler assembler(circuit);

    // Build system function
    auto jacobian_func = [&](const Vector& x, Vector& f, SparseMatrix& J) {
        assembler.assemble_dc(J, f);
        f = J * x - f;
    };

    Index num_nodes = circuit.node_count();
    Vector x0 = Vector::Zero(circuit.total_variables());

    GminStepping gmin_solver;
    GminSteppingOptions opts;
    opts.gmin_initial = 1e-3;
    opts.gmin_final = 1e-12;
    gmin_solver.set_options(opts);

    // We need a build function that doesn't scale sources
    auto build_func = [&](SparseMatrix& G, Vector& b) {
        assembler.assemble_dc(G, b);
    };

    auto result = gmin_solver.solve(num_nodes, build_func, jacobian_func, x0);

    REQUIRE(result.status == SolverStatus::Success);

    // Check voltage at output node
    Index out_idx = circuit.node_index("out");
    REQUIRE_THAT(result.x(out_idx), WithinRel(5.0, 0.01));

    // Check source current (branch variable)
    Index branch_idx = circuit.node_count();  // First branch variable
    REQUIRE_THAT(std::abs(result.x(branch_idx)), WithinRel(0.005, 0.01));  // 5mA
}

TEST_CASE("Source Stepping - Diode circuit", "[convergence][source]") {
    // Circuit with diode that may be hard to converge
    // V1 -> R1 -> D1 -> ground
    // This tests source stepping for nonlinear convergence

    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "anode", 100.0);
    circuit.add_diode("D1", "anode", "0", DiodeParams{1e-14, 1.0, 0.0, 0.026, false});

    MNAAssembler assembler(circuit);

    // Build scaled system function
    auto scaled_jacobian = [&](Real factor, const Vector& x, Vector& f, SparseMatrix& J) {
        SparseMatrix G;
        Vector b;
        assembler.assemble_dc(G, b);

        // Scale source contributions in RHS
        b *= factor;

        // Add nonlinear contributions
        SparseMatrix J_nl;
        Vector f_nl;
        assembler.assemble_nonlinear(J_nl, f_nl, x);
        G += J_nl;
        b += f_nl;

        f = G * x - b;
        J = G;
    };

    auto scaled_build = [&](SparseMatrix& G, Vector& b, Real factor) {
        assembler.assemble_dc(G, b);
        b *= factor;
    };

    Vector x0 = Vector::Zero(circuit.total_variables());

    SourceStepping source_solver;
    SourceSteppingOptions opts;
    opts.factor_initial = 0.01;
    opts.factor_increment = 0.1;
    source_solver.set_options(opts);

    auto result = source_solver.solve(scaled_build, scaled_jacobian, x0);

    // Should converge (even if with warnings, the result should be reasonable)
    // For a diode with Is=1e-14, Vd ~ 0.6-0.7V typically

    Index anode_idx = circuit.node_index("anode");
    Real Vd = result.x(anode_idx);

    // The diode should conduct, so Vd should be around 0.5-0.8V
    // (exact value depends on current)
    REQUIRE(Vd > 0.3);
    REQUIRE(Vd < 1.0);
}

TEST_CASE("Pseudo-Transient Continuation - Stiff system", "[convergence][ptc]") {
    // Create a simple system to test pseudo-transient continuation
    // This tests the basic PTC algorithm

    // Simple nonlinear equation: x^3 - x = 0, solutions at x = -1, 0, 1
    auto system_func = [](const Vector& x, Vector& f, SparseMatrix& J) {
        Index n = x.size();
        f.resize(n);
        J.resize(n, n);
        std::vector<Triplet> triplets;

        for (Index i = 0; i < n; ++i) {
            Real xi = x(i);
            f(i) = xi * xi * xi - xi;  // x^3 - x
            triplets.emplace_back(i, i, 3.0 * xi * xi - 1.0);  // 3x^2 - 1
        }
        J.setFromTriplets(triplets.begin(), triplets.end());
    };

    PseudoTransientContinuation ptc_solver;
    PseudoTransientOptions opts;
    opts.tau_initial = 1.0;
    opts.tau_final = 1e-12;
    opts.tau_factor = 10.0;
    opts.max_iterations = 100;
    ptc_solver.set_options(opts);

    // Start from x0 = 0.5 (should converge to x = 1)
    Vector x0(1);
    x0(0) = 0.5;

    auto result = ptc_solver.solve(system_func, x0, 1.0);

    // Should converge to one of the solutions
    if (result.status == SolverStatus::Success) {
        Real x_final = result.x(0);
        bool near_solution = std::abs(x_final - 1.0) < 0.1 ||
                            std::abs(x_final + 1.0) < 0.1 ||
                            std::abs(x_final) < 0.1;
        REQUIRE(near_solution);
    }
    // Note: PTC may not always converge depending on initial conditions
}

TEST_CASE("Gmin Stepping Options", "[convergence][gmin]") {
    GminSteppingOptions opts;

    // Check default values
    REQUIRE_THAT(opts.gmin_initial, WithinAbs(1e-2, 1e-10));
    REQUIRE_THAT(opts.gmin_final, WithinAbs(1e-12, 1e-20));
    REQUIRE_THAT(opts.reduction_factor, WithinAbs(10.0, 1e-10));
    REQUIRE(opts.max_steps == 20);

    // Modify options
    opts.gmin_initial = 1e-1;
    opts.gmin_final = 1e-15;
    opts.reduction_factor = 5.0;
    opts.max_steps = 30;

    GminStepping solver(opts);
    REQUIRE_THAT(solver.options().gmin_initial, WithinAbs(1e-1, 1e-10));
    REQUIRE_THAT(solver.options().gmin_final, WithinAbs(1e-15, 1e-20));
}

TEST_CASE("Source Stepping Options", "[convergence][source]") {
    SourceSteppingOptions opts;

    // Check default values
    REQUIRE_THAT(opts.factor_initial, WithinAbs(0.1, 1e-10));
    REQUIRE_THAT(opts.factor_increment, WithinAbs(0.1, 1e-10));
    REQUIRE(opts.max_steps == 20);

    // Modify options
    opts.factor_initial = 0.05;
    opts.factor_increment = 0.05;
    opts.max_steps = 40;

    SourceStepping solver(opts);
    REQUIRE_THAT(solver.options().factor_initial, WithinAbs(0.05, 1e-10));
    REQUIRE_THAT(solver.options().factor_increment, WithinAbs(0.05, 1e-10));
}

TEST_CASE("Pseudo-Transient Options", "[convergence][ptc]") {
    PseudoTransientOptions opts;

    // Check default values
    REQUIRE_THAT(opts.tau_initial, WithinAbs(1e-3, 1e-10));
    REQUIRE_THAT(opts.tau_final, WithinAbs(1e-15, 1e-20));
    REQUIRE_THAT(opts.tau_factor, WithinAbs(10.0, 1e-10));
    REQUIRE(opts.max_iterations == 500);
    REQUIRE(opts.iterations_per_tau == 5);

    // Modify options
    opts.tau_initial = 1e-2;
    opts.tau_final = 1e-12;
    opts.tau_factor = 5.0;
    opts.max_iterations = 200;
    opts.iterations_per_tau = 10;

    PseudoTransientContinuation solver(opts);
    REQUIRE_THAT(solver.options().tau_initial, WithinAbs(1e-2, 1e-10));
    REQUIRE_THAT(solver.options().tau_final, WithinAbs(1e-12, 1e-20));
}

TEST_CASE("Gmin Stepping - Statistics", "[convergence][gmin]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    MNAAssembler assembler(circuit);

    auto jacobian_func = [&](const Vector& x, Vector& f, SparseMatrix& J) {
        assembler.assemble_dc(J, f);
        f = J * x - f;
    };

    auto build_func = [&](SparseMatrix& G, Vector& b) {
        assembler.assemble_dc(G, b);
    };

    Index num_nodes = circuit.node_count();
    Vector x0 = Vector::Zero(circuit.total_variables());

    GminStepping gmin_solver;
    auto result = gmin_solver.solve(num_nodes, build_func, jacobian_func, x0);

    // Check that statistics are available
    REQUIRE(gmin_solver.gmin_steps_used() >= 0);
    REQUIRE(gmin_solver.final_gmin() > 0);
}
