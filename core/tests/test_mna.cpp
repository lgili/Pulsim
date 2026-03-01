#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/mna.hpp"

using namespace pulsim;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("MNA resistor stamp", "[mna]") {
    // Simple voltage divider: V1 - R1 - R2 - GND
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "out", 1000.0);
    circuit.add_resistor("R2", "out", "0", 1000.0);

    MNAAssembler assembler(circuit);
    SparseMatrix G;
    Vector b;

    assembler.assemble_dc(G, b);

    // System size: 2 nodes + 1 branch current = 3
    CHECK(G.rows() == 3);
    CHECK(G.cols() == 3);

    // Solve the DC system
    Vector x = Eigen::SparseLU<SparseMatrix>(G).solve(b);

    // V(in) should be 10V
    CHECK_THAT(x(0), WithinRel(10.0, 1e-6));
    // V(out) should be 5V (voltage divider)
    CHECK_THAT(x(1), WithinRel(5.0, 1e-6));
    // Current should be 10/(1000+1000) = 5mA
    // In MNA, branch current for voltage source is defined as current leaving
    // the positive terminal (into the circuit). The voltage source supplies
    // current, so the branch current is negative (current enters positive terminal).
    CHECK_THAT(x(2), WithinRel(-5e-3, 1e-6));
}

TEST_CASE("MNA capacitor companion model", "[mna]") {
    // RC circuit
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "out", 1000.0);
    circuit.add_capacitor("C1", "out", "0", 1e-6);

    MNAAssembler assembler(circuit);
    SparseMatrix G;
    Vector b;

    // Start with capacitor at 0V
    Vector x_prev = Vector::Zero(3);  // 2 nodes + 1 branch
    Real dt = 1e-6;

    assembler.assemble_transient(G, b, x_prev, dt);
    assembler.evaluate_sources(b, 0.0);

    // Solve one timestep
    Vector x = Eigen::SparseLU<SparseMatrix>(G).solve(b);

    // V(in) should be 5V
    CHECK_THAT(x(0), WithinRel(5.0, 1e-6));

    // V(out) should be between 0 and 5V
    CHECK(x(1) > 0.0);
    CHECK(x(1) < 5.0);
}

TEST_CASE("MNA inductor companion model", "[mna]") {
    // RL circuit
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "out", 100.0);
    circuit.add_inductor("L1", "out", "0", 10e-3);

    MNAAssembler assembler(circuit);
    SparseMatrix G;
    Vector b;

    // Start with inductor current at 0
    Vector x_prev = Vector::Zero(4);  // 2 nodes + 2 branches (V1, L1)
    Real dt = 1e-6;

    assembler.assemble_transient(G, b, x_prev, dt);
    assembler.evaluate_sources(b, 0.0);

    // Solve one timestep
    Vector x = Eigen::SparseLU<SparseMatrix>(G).solve(b);

    // V(in) should be 10V
    CHECK_THAT(x(0), WithinRel(10.0, 1e-6));

    // Inductor current should be small but positive
    CHECK(x(3) >= 0.0);
}

TEST_CASE("Waveform evaluation", "[mna]") {
    Circuit circuit;

    SECTION("Pulse waveform") {
        PulseWaveform pulse{0.0, 5.0, 0.0, 1e-9, 1e-9, 0.5e-3, 1e-3};
        circuit.add_voltage_source("V1", "out", "0", pulse);
        circuit.add_resistor("R1", "out", "0", 100.0);

        MNAAssembler assembler(circuit);
        Vector b = Vector::Zero(2);

        // At t=0, should be at v2 (after rising edge)
        assembler.evaluate_sources(b, 0.0);
        // The source equation is in b(1) for branch current
        // V(out) = V, so after solving, V(out) should be close to 5V

        // At t=0.6ms, should be at v1 (low)
        b.setZero();
        assembler.evaluate_sources(b, 0.6e-3);
    }

    SECTION("Sine waveform") {
        SineWaveform sine{2.5, 2.5, 1000.0, 0.0, 0.0};
        circuit.add_voltage_source("V1", "out", "0", sine);
        circuit.add_resistor("R1", "out", "0", 100.0);

        MNAAssembler assembler(circuit);
        Vector b = Vector::Zero(2);

        // At t=0, sin(0) = 0, so V = offset = 2.5V
        assembler.evaluate_sources(b, 0.0);
        CHECK_THAT(b(1), WithinAbs(2.5, 1e-10));

        // At t=0.25ms (quarter period of 1kHz), sin(pi/2) = 1, V = 2.5 + 2.5 = 5V
        b.setZero();
        assembler.evaluate_sources(b, 0.25e-3);
        CHECK_THAT(b(1), WithinAbs(5.0, 1e-6));
    }
}
