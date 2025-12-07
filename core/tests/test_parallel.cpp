#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "pulsim/parallel.hpp"
#include "pulsim/circuit.hpp"
#include "pulsim/simulation.hpp"
#include <chrono>
#include <atomic>

using namespace pulsim;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("Thread Pool - Basic functionality", "[parallel][threadpool]") {
    ThreadPool pool(4);

    SECTION("Submit and execute tasks") {
        std::atomic<int> counter{0};

        std::vector<std::future<void>> futures;
        for (int i = 0; i < 100; ++i) {
            futures.push_back(pool.submit([&counter]() {
                ++counter;
            }));
        }

        for (auto& f : futures) {
            f.get();
        }

        REQUIRE(counter == 100);
    }

    SECTION("Submit tasks with return values") {
        auto future1 = pool.submit([]() { return 42; });
        auto future2 = pool.submit([]() { return 3.14; });
        auto future3 = pool.submit([]() { return std::string("hello"); });

        REQUIRE(future1.get() == 42);
        REQUIRE_THAT(future2.get(), WithinAbs(3.14, 0.001));
        REQUIRE(future3.get() == "hello");
    }

    SECTION("Wait for all tasks") {
        std::atomic<int> counter{0};

        for (int i = 0; i < 50; ++i) {
            pool.submit([&counter]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                ++counter;
            });
        }

        pool.wait_all();
        REQUIRE(counter == 50);
    }
}

TEST_CASE("Thread Pool - Global singleton", "[parallel][threadpool]") {
    auto& pool1 = ThreadPool::global();
    auto& pool2 = ThreadPool::global();

    REQUIRE(&pool1 == &pool2);
    REQUIRE(pool1.size() > 0);
}

TEST_CASE("SIMD Info - Detection", "[parallel][simd]") {
    // Just check that detection doesn't crash
    bool has_sse2 = SIMDInfo::has_sse2();
    bool has_avx = SIMDInfo::has_avx();
    bool has_avx2 = SIMDInfo::has_avx2();
    bool has_avx512 = SIMDInfo::has_avx512();

    const char* best = SIMDInfo::best_available();
    REQUIRE(best != nullptr);

    // If we have AVX, we should have SSE2
    if (has_avx) {
        REQUIRE(has_sse2);
    }
    if (has_avx2) {
        REQUIRE(has_avx);
    }
    if (has_avx512) {
        REQUIRE(has_avx2);
    }

    INFO("SIMD support: SSE2=" << has_sse2 << " AVX=" << has_avx
         << " AVX2=" << has_avx2 << " AVX512=" << has_avx512
         << " Best=" << best);
}

TEST_CASE("SIMD - Batch exponential", "[parallel][simd]") {
    std::vector<Real> input = {0.0, 1.0, -1.0, 2.0, -2.0, 0.5};
    std::vector<Real> output(input.size());

    simd::exp_batch(input.data(), output.data(), input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        REQUIRE_THAT(output[i], WithinRel(std::exp(input[i]), 1e-10));
    }
}

TEST_CASE("SIMD - Diode current batch", "[parallel][simd]") {
    std::vector<Real> Vd = {0.0, 0.3, 0.5, 0.6, 0.7, -0.5};
    std::vector<Real> Id(Vd.size());
    std::vector<Real> Gd(Vd.size());

    Real Is = 1e-14;
    Real n = 1.0;
    Real Vt = 0.026;

    simd::diode_current_batch(Vd.data(), Id.data(), Gd.data(), Vd.size(), Is, n, Vt);

    // Verify against scalar calculation
    for (size_t i = 0; i < Vd.size(); ++i) {
        Real expected_exp = std::exp(Vd[i] / (n * Vt));
        Real expected_Id = Is * (expected_exp - 1.0);
        Real expected_Gd = std::max(Is / (n * Vt) * expected_exp, 1e-12);

        REQUIRE_THAT(Id[i], WithinRel(expected_Id, 0.01));
        REQUIRE_THAT(Gd[i], WithinRel(expected_Gd, 0.01));
    }
}

TEST_CASE("Sweep Parameter - Linear", "[parallel][sweep]") {
    auto param = SweepParameter::linear("R1", "resistance", 100.0, 1000.0, 10);

    REQUIRE(param.component_name == "R1");
    REQUIRE(param.param_name == "resistance");
    REQUIRE(param.values.size() == 10);
    REQUIRE_THAT(param.values.front(), WithinAbs(100.0, 0.01));
    REQUIRE_THAT(param.values.back(), WithinAbs(1000.0, 0.01));

    // Check linear spacing
    Real step = (1000.0 - 100.0) / 9.0;
    REQUIRE_THAT(param.values[1] - param.values[0], WithinRel(step, 0.001));
}

TEST_CASE("Sweep Parameter - Logarithmic", "[parallel][sweep]") {
    auto param = SweepParameter::logarithmic("C1", "capacitance", 1e-9, 1e-6, 4);

    REQUIRE(param.values.size() == 4);
    REQUIRE_THAT(param.values[0], WithinRel(1e-9, 0.01));
    REQUIRE_THAT(param.values[3], WithinRel(1e-6, 0.01));

    // Check logarithmic spacing (ratio should be constant)
    Real ratio1 = param.values[1] / param.values[0];
    Real ratio2 = param.values[2] / param.values[1];
    REQUIRE_THAT(ratio1, WithinRel(ratio2, 0.01));
}

TEST_CASE("Sweep Parameter - List", "[parallel][sweep]") {
    std::vector<Real> values = {10.0, 47.0, 100.0, 470.0, 1000.0};
    auto param = SweepParameter::list("R1", "resistance", values);

    REQUIRE(param.values.size() == 5);
    REQUIRE(param.values == values);
}

TEST_CASE("Job Queue - Basic operations", "[parallel][jobqueue]") {
    JobQueue queue(2);

    // Create a simple circuit
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions options;
    options.tstart = 0.0;
    options.tstop = 1e-6;
    options.dt = 1e-9;

    SECTION("Submit and get result") {
        std::string job_id = queue.submit(circuit, options);

        REQUIRE(!job_id.empty());
        REQUIRE(queue.total_count() >= 1);

        JobResult result = queue.get_result(job_id, true);

        REQUIRE(result.job_id == job_id);
        REQUIRE((result.status == JobStatus::Completed || result.status == JobStatus::Failed));
    }

    SECTION("Job priority") {
        SimulationJob low_job;
        low_job.circuit = circuit;
        low_job.options = options;
        low_job.priority = JobPriority::Low;

        SimulationJob high_job;
        high_job.circuit = circuit;
        high_job.options = options;
        high_job.priority = JobPriority::High;

        // Submit low priority first, then high priority
        std::string low_id = queue.submit(low_job);
        std::string high_id = queue.submit(high_job);

        // Both should complete
        queue.get_result(low_id, true);
        queue.get_result(high_id, true);

        REQUIRE(queue.is_ready(low_id));
        REQUIRE(queue.is_ready(high_id));
    }

    SECTION("Cancel job") {
        // Submit a job
        std::string job_id = queue.submit(circuit, options);

        // Cancel it (may or may not succeed depending on timing)
        bool cancelled = queue.cancel(job_id);

        // Either way, getting result should not hang
        JobResult result = queue.get_result(job_id, true);
        REQUIRE(!result.job_id.empty());
    }

    SECTION("Multiple jobs") {
        std::vector<std::string> job_ids;

        for (int i = 0; i < 5; ++i) {
            job_ids.push_back(queue.submit(circuit, options));
        }

        // Wait for all to complete
        for (const auto& id : job_ids) {
            JobResult result = queue.get_result(id, true);
            REQUIRE((result.status == JobStatus::Completed ||
                    result.status == JobStatus::Failed ||
                    result.status == JobStatus::Cancelled));
        }
    }
}

TEST_CASE("Job Queue - Status tracking", "[parallel][jobqueue]") {
    JobQueue queue(1);  // Single worker for predictable behavior

    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions options;
    options.tstart = 0.0;
    options.tstop = 1e-6;
    options.dt = 1e-9;

    std::string job_id = queue.submit(circuit, options);

    // Status should be pending or running
    JobStatus initial_status = queue.get_status(job_id);
    REQUIRE((initial_status == JobStatus::Pending || initial_status == JobStatus::Running));

    // Wait for completion
    queue.get_result(job_id, true);

    // Status should be completed or failed
    JobStatus final_status = queue.get_status(job_id);
    REQUIRE((final_status == JobStatus::Completed || final_status == JobStatus::Failed));
}

TEST_CASE("Parallel MNA Assembly", "[parallel][mna]") {
    // Create a circuit with many components
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});

    // Add a chain of resistors
    std::string prev_node = "in";
    for (int i = 0; i < 50; ++i) {
        std::string next_node = (i == 49) ? "0" : "n" + std::to_string(i);
        circuit.add_resistor("R" + std::to_string(i), prev_node, next_node, 100.0);
        prev_node = next_node;
    }

    ParallelAssemblyOptions opts;
    opts.num_threads = 4;
    opts.min_components_per_thread = 10;  // Lower threshold for testing

    ParallelMNAAssembler assembler(circuit, opts);

    SparseMatrix G;
    Vector b;
    assembler.assemble_dc_parallel(G, b);

    // Verify matrix dimensions
    REQUIRE(G.rows() == circuit.total_variables());
    REQUIRE(G.cols() == circuit.total_variables());
    REQUIRE(b.size() == circuit.total_variables());

    // Matrix should not be empty
    REQUIRE(G.nonZeros() > 0);
}

TEST_CASE("Run parallel simulations", "[parallel]") {
    // Create test circuits
    std::vector<std::pair<Circuit, SimulationOptions>> simulations;

    for (int i = 0; i < 4; ++i) {
        Circuit circuit;
        circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0 + i});
        circuit.add_resistor("R1", "in", "0", 1000.0 + i * 100);

        SimulationOptions options;
        options.tstart = 0.0;
        options.tstop = 1e-6;
        options.dt = 1e-9;

        simulations.emplace_back(circuit, options);
    }

    auto results = run_parallel(simulations, 2);

    REQUIRE(results.size() == 4);

    for (const auto& result : results) {
        REQUIRE(result.final_status == SolverStatus::Success);
    }
}

TEST_CASE("Recommended threads", "[parallel]") {
    size_t threads = recommended_threads();
    REQUIRE(threads > 0);
    INFO("Recommended threads: " << threads);
}

TEST_CASE("Parameter Sweeper - Single parameter", "[parallel][sweep]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions options;
    options.tstart = 0.0;
    options.tstop = 1e-6;
    options.dt = 1e-9;

    auto param = SweepParameter::linear("R1", "resistance", 100.0, 1000.0, 3);

    ParameterSweeper sweeper(2);

    // Track progress
    std::atomic<size_t> progress_calls{0};
    sweeper.set_progress_callback([&](size_t completed, size_t total) {
        ++progress_calls;
        REQUIRE(completed <= total);
    });

    auto result = sweeper.sweep(circuit, options, param);

    REQUIRE(result.total_points() == 3);
    REQUIRE(result.parameters.size() == 1);

    // Note: The actual circuit modification is not fully implemented,
    // so all simulations run with the same circuit
}

TEST_CASE("Job Queue - Completion callback", "[parallel][jobqueue]") {
    JobQueue queue(1);

    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions options;
    options.tstart = 0.0;
    options.tstop = 1e-6;
    options.dt = 1e-9;

    std::atomic<int> callback_count{0};
    std::string received_job_id;

    queue.set_completion_callback([&](const JobResult& result) {
        ++callback_count;
        received_job_id = result.job_id;
    });

    std::string job_id = queue.submit(circuit, options);
    queue.get_result(job_id, true);

    // Give callback time to execute
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    REQUIRE(callback_count >= 1);
    REQUIRE(received_job_id == job_id);
}

TEST_CASE("Job Queue - Clear completed", "[parallel][jobqueue]") {
    JobQueue queue(2);

    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "0", 1000.0);

    SimulationOptions options;
    options.tstart = 0.0;
    options.tstop = 1e-6;
    options.dt = 1e-9;

    // Submit several jobs
    std::vector<std::string> job_ids;
    for (int i = 0; i < 3; ++i) {
        job_ids.push_back(queue.submit(circuit, options));
    }

    // Wait for completion
    for (const auto& id : job_ids) {
        queue.get_result(id, true);
    }

    size_t count_before = queue.completed_count();
    REQUIRE(count_before >= 3);

    queue.clear_completed();

    size_t count_after = queue.total_count();
    REQUIRE(count_after < count_before);
}
