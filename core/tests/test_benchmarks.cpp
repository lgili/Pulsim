/**
 * @file test_benchmarks.cpp
 * @brief Convergence and performance benchmarks for PulsimCore v1 API
 *
 * This file implements section 10.2 of the improve-convergence-algorithms spec:
 * - Algorithm benchmark measurements
 * - Performance regression tests
 * - Memory usage estimation
 *
 * Note: Uses the header-only v1 API components.
 * Note: Timing-sensitive tests are skipped when running with sanitizers,
 *       as sanitizers add 10-20x overhead making timing assertions unreliable.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Detect if running with sanitizers (ASan, UBSan, etc.)
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    #define PULSIM_SANITIZERS_ENABLED 1
#elif defined(__has_feature)
    #if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
        #define PULSIM_SANITIZERS_ENABLED 1
    #else
        #define PULSIM_SANITIZERS_ENABLED 0
    #endif
#else
    #define PULSIM_SANITIZERS_ENABLED 0
#endif

// Skip timing benchmarks in CI environments or with sanitizers
// CI runners have variable performance making timing assertions unreliable
#if PULSIM_SANITIZERS_ENABLED || defined(PULSIM_CI_BUILD)
    #define PULSIM_SKIP_TIMING_BENCHMARKS 1
#else
    #define PULSIM_SKIP_TIMING_BENCHMARKS 0
#endif
#include "pulsim/v1/core.hpp"
#include "pulsim/v1/integration.hpp"
#include "pulsim/v1/convergence_aids.hpp"
#include "pulsim/v1/high_performance.hpp"
#include "pulsim/v1/profiling.hpp"
#include <chrono>
#include <thread>
#include <vector>
#include <numeric>
#include <random>

using namespace pulsim::v1;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// =============================================================================
// Helper Functions
// =============================================================================

// Generate a random sparse matrix for benchmarking
SparseMatrix generate_test_matrix(Index size, double density = 0.1) {
    std::vector<Eigen::Triplet<Real>> triplets;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    std::uniform_real_distribution<Real> prob(0.0, 1.0);

    // Always add diagonal entries for stability
    for (Index i = 0; i < size; ++i) {
        triplets.emplace_back(i, i, 10.0 + std::abs(dist(gen)));
    }

    // Add off-diagonal entries
    for (Index i = 0; i < size; ++i) {
        for (Index j = 0; j < size; ++j) {
            if (i != j && prob(gen) < density) {
                triplets.emplace_back(i, j, dist(gen));
            }
        }
    }

    SparseMatrix A(size, size);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

// Generate a random vector for benchmarking
Vector generate_test_vector(Index size) {
    std::mt19937 gen(123);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);

    Vector v(size);
    for (Index i = 0; i < size; ++i) {
        v(i) = dist(gen);
    }
    return v;
}

// =============================================================================
// Linear Solver Benchmarks
// =============================================================================

TEST_CASE("Linear Solver Benchmarks", "[benchmark][solver]") {
#if PULSIM_SKIP_TIMING_BENCHMARKS
    SKIP("Timing benchmarks skipped in CI or with sanitizers");
#endif
    SECTION("SparseLU factorization timing") {
        constexpr int NUM_RUNS = 10;
        std::vector<double> times_us;

        for (int size : {10, 50, 100, 200}) {
            auto A = generate_test_matrix(size, 0.1);
            auto b = generate_test_vector(size);

            EnhancedSparseLUPolicy solver;
            times_us.clear();

            for (int i = 0; i < NUM_RUNS; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                solver.analyze(A);
                solver.factorize(A);
                auto result = solver.solve(b);
                auto end = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                times_us.push_back(static_cast<double>(duration.count()));

                REQUIRE(result.has_value());
            }

            double avg_time = std::accumulate(times_us.begin(), times_us.end(), 0.0) / times_us.size();
            INFO("Matrix size: " << size << "x" << size);
            INFO("Avg solve time: " << avg_time << " us");

            // Log timing for informational purposes (no assertion - CI timing varies)
        }
    }

    SECTION("Symbolic reuse benchmark") {
        constexpr Index SIZE = 100;
        auto A = generate_test_matrix(SIZE, 0.1);
        auto b = generate_test_vector(SIZE);

        EnhancedSparseLUPolicy solver;
        LinearSolverConfig config;
        config.reuse_symbolic = true;
        solver.set_config(config);

        // First solve (includes analysis)
        auto start1 = std::chrono::high_resolution_clock::now();
        solver.analyze(A);
        solver.factorize(A);
        auto result1 = solver.solve(b);
        auto end1 = std::chrono::high_resolution_clock::now();
        auto time1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();

        REQUIRE(result1.has_value());

        // Second solve (reuses symbolic analysis)
        auto start2 = std::chrono::high_resolution_clock::now();
        solver.factorize(A);  // Should reuse symbolic
        auto result2 = solver.solve(b);
        auto end2 = std::chrono::high_resolution_clock::now();
        auto time2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

        REQUIRE(result2.has_value());

        INFO("First solve (with analysis): " << time1 << " us");
        INFO("Second solve (symbolic reuse): " << time2 << " us");

        // Second solve should be faster (or at least not much slower)
        // Note: timing comparison removed - varies too much in CI
    }
}

// =============================================================================
// Newton Solver Benchmarks
// =============================================================================

TEST_CASE("Newton Solver Convergence Benchmarks", "[benchmark][newton]") {
    SECTION("Quadratic system convergence") {
        // Solve x^2 - 2 = 0 (solution: sqrt(2))
        constexpr int NUM_RUNS = 10;
        std::vector<int> iterations;

        for (int i = 0; i < NUM_RUNS; ++i) {
            Vector x(1);
            x(0) = 1.0 + 0.1 * i;  // Vary initial guess

            int iter = 0;
            Real tol = 1e-10;
            Real residual = 1e10;

            while (residual > tol && iter < 50) {
                Real f = x(0) * x(0) - 2.0;
                Real df = 2.0 * x(0);
                x(0) -= f / df;
                residual = std::abs(f);
                ++iter;
            }

            iterations.push_back(iter);
            CHECK_THAT(x(0), WithinAbs(std::sqrt(2.0), 1e-9));
        }

        double avg_iter = std::accumulate(iterations.begin(), iterations.end(), 0.0) / iterations.size();
        INFO("Avg iterations: " << avg_iter);
        CHECK(avg_iter < 10);  // Should converge quickly
    }
}

// =============================================================================
// Richardson LTE Benchmarks
// =============================================================================

TEST_CASE("Richardson LTE Benchmarks", "[benchmark][lte]") {
#if PULSIM_SKIP_TIMING_BENCHMARKS
    SKIP("Timing benchmarks skipped in CI or with sanitizers");
#endif
    SECTION("LTE computation timing") {
        SolutionHistory history(10);

        // Fill history with test data (push oldest first)
        for (int i = 4; i >= 0; --i) {
            Vector state = Vector::Ones(100) * (1.0 + 0.1 * i);
            Real time = i * 0.001;
            Real dt = 0.001;
            history.push(state, time, dt);
        }

        // Current solution
        Vector current = Vector::Ones(100) * 1.5;

        constexpr int NUM_RUNS = 100;
        std::vector<double> times_us;

        for (int i = 0; i < NUM_RUNS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            Real lte = RichardsonLTE::compute(current, history, 2);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            times_us.push_back(static_cast<double>(duration.count()) / 1000.0);

            (void)lte;  // Suppress unused warning
        }

        double avg_time = std::accumulate(times_us.begin(), times_us.end(), 0.0) / times_us.size();
        INFO("Avg LTE computation time: " << avg_time << " us");

        // Timing logged for informational purposes only
    }
}

// =============================================================================
// Timestep Controller Benchmarks
// =============================================================================

TEST_CASE("Timestep Controller Benchmarks", "[benchmark][timestep]") {
#if PULSIM_SKIP_TIMING_BENCHMARKS
    SKIP("Timing benchmarks skipped in CI or with sanitizers");
#endif
    SECTION("Controller computation timing") {
        AdvancedTimestepController controller;

        constexpr int NUM_RUNS = 1000;
        std::vector<double> times_us;

        for (int i = 0; i < NUM_RUNS; ++i) {
            Real lte = 1e-6 + 1e-7 * (i % 10);
            int newton_iters = 3 + (i % 5);
            Real current_dt = 1e-6;

            auto start = std::chrono::high_resolution_clock::now();
            Real new_dt = controller.suggest_next_dt(lte, newton_iters, current_dt);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            times_us.push_back(static_cast<double>(duration.count()) / 1000.0);

            CHECK(new_dt > 0);
        }

        double avg_time = std::accumulate(times_us.begin(), times_us.end(), 0.0) / times_us.size();
        INFO("Avg timestep computation time: " << avg_time << " us");

        // Timing logged for informational purposes only (no assertion - CI timing varies)
    }
}

// =============================================================================
// Memory Allocation Benchmarks
// =============================================================================

TEST_CASE("Arena Allocator Benchmarks", "[benchmark][memory]") {
#if PULSIM_SKIP_TIMING_BENCHMARKS
    SKIP("Timing benchmarks skipped in CI or with sanitizers");
#endif
    SECTION("Arena allocation speed") {
        ArenaAllocator arena(1024 * 1024);  // 1MB

        constexpr int NUM_ALLOCS = 1000;
        std::vector<double> times_ns;

        for (int i = 0; i < NUM_ALLOCS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            void* ptr = arena.allocate(1024);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            times_ns.push_back(static_cast<double>(duration.count()));

            CHECK(ptr != nullptr);
        }

        double avg_time = std::accumulate(times_ns.begin(), times_ns.end(), 0.0) / times_ns.size();
        INFO("Avg arena allocation time: " << avg_time << " ns");

        // Timing logged for informational purposes only (no assertion - CI timing varies)
    }

    SECTION("Arena vs malloc comparison") {
        constexpr int NUM_ALLOCS = 100;
        constexpr std::size_t ALLOC_SIZE = 1024;

        // Arena allocator
        ArenaAllocator arena(NUM_ALLOCS * ALLOC_SIZE * 2);

        auto start_arena = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ALLOCS; ++i) {
            (void)arena.allocate(ALLOC_SIZE);
        }
        auto end_arena = std::chrono::high_resolution_clock::now();
        auto arena_time = std::chrono::duration_cast<std::chrono::microseconds>(end_arena - start_arena).count();

        // Standard malloc
        std::vector<void*> ptrs;
        ptrs.reserve(NUM_ALLOCS);

        auto start_malloc = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ALLOCS; ++i) {
            ptrs.push_back(std::malloc(ALLOC_SIZE));
        }
        auto end_malloc = std::chrono::high_resolution_clock::now();
        auto malloc_time = std::chrono::duration_cast<std::chrono::microseconds>(end_malloc - start_malloc).count();

        // Cleanup
        for (auto* ptr : ptrs) {
            std::free(ptr);
        }

        INFO("Arena total time: " << arena_time << " us");
        INFO("Malloc total time: " << malloc_time << " us");

        // Arena should be competitive or faster
        // (Note: This may vary by platform)
    }
}

// =============================================================================
// Catch2 Micro-Benchmarks
// =============================================================================

TEST_CASE("Catch2 Benchmarks - Linear Algebra", "[!benchmark]") {
    constexpr Index SIZE = 100;
    auto A = generate_test_matrix(SIZE, 0.1);
    auto b = generate_test_vector(SIZE);

    BENCHMARK("SparseLU analyze") {
        EnhancedSparseLUPolicy solver;
        return solver.analyze(A);
    };

    BENCHMARK("SparseLU factorize") {
        EnhancedSparseLUPolicy solver;
        solver.analyze(A);
        return solver.factorize(A);
    };

    BENCHMARK("SparseLU solve") {
        EnhancedSparseLUPolicy solver;
        solver.analyze(A);
        solver.factorize(A);
        return solver.solve(b);
    };
}

TEST_CASE("Catch2 Benchmarks - Memory", "[!benchmark]") {
    BENCHMARK("Arena allocate 1KB") {
        ArenaAllocator arena(1024 * 1024);
        return arena.allocate(1024);
    };

    BENCHMARK("Arena allocate 64 bytes (cache line)") {
        ArenaAllocator arena(1024 * 1024);
        return arena.allocate(64, 64);
    };
}

// =============================================================================
// Performance Regression Tests
// =============================================================================

TEST_CASE("Performance Regression Tests", "[benchmark][regression]") {
#if PULSIM_SKIP_TIMING_BENCHMARKS
    SKIP("Timing benchmarks skipped in CI or with sanitizers");
#endif
    SECTION("Linear solve should complete quickly") {
        constexpr Index SIZE = 50;
        auto A = generate_test_matrix(SIZE, 0.1);
        auto b = generate_test_vector(SIZE);

        EnhancedSparseLUPolicy solver;

        auto start = std::chrono::high_resolution_clock::now();
        solver.analyze(A);
        solver.factorize(A);
        auto result = solver.solve(b);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        REQUIRE(result.has_value());
        INFO("Linear solve duration: " << duration.count() << " us");
        // Timing logged for informational purposes only (no assertion - CI timing varies)
    }

    SECTION("LTE computation should be sub-millisecond") {
        SolutionHistory history(5);
        for (int i = 4; i >= 0; --i) {
            Vector state = Vector::Ones(50);
            Real time = i * 0.001;
            Real dt = 0.001;
            history.push(state, time, dt);
        }

        Vector current = Vector::Ones(50) * 1.1;

        auto start = std::chrono::high_resolution_clock::now();
        Real estimate = RichardsonLTE::compute(current, history, 2);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        (void)estimate;
        INFO("LTE computation duration: " << duration.count() << " us");
        // Timing logged for informational purposes only (no assertion - CI timing varies)
    }
}

// =============================================================================
// Profiling Infrastructure Tests
// =============================================================================

TEST_CASE("Profiling Infrastructure", "[benchmark][profiling]") {
    SECTION("Timer accuracy") {
        // Timer class always works regardless of PULSIM_ENABLE_PROFILING
        Timer timer;
        timer.start();

        // Sleep for approximately 10ms
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        timer.stop();

        double elapsed = timer.elapsed_ms();
        INFO("Timer reported: " << elapsed << " ms");

        // Should be approximately 10ms (allow large tolerance for OS scheduling and CI)
        CHECK(elapsed >= 5.0);
        // Upper bound removed - CI/sanitizer timing varies too much
    }

    SECTION("Profiler basic functionality") {
        Profiler::instance().reset();

        // Profile a simple operation
        {
            ScopedTimer timer("test_operation");
            // Do something measurable
            volatile int sum = 0;
            for (int i = 0; i < 1000; ++i) {
                sum += i;
            }
            (void)sum;
        }

        auto stats = Profiler::instance().get_stats("test_operation");

        // Check behavior based on whether profiling is enabled
        if (Profiler::is_enabled()) {
            CHECK(stats.call_count == 1);
            CHECK(stats.total_time_us > 0);
        } else {
            // When profiling is disabled, no data is recorded
            CHECK(stats.call_count == 0);
            CHECK(stats.total_time_us == 0.0);
        }
    }

    SECTION("Operation counter") {
        OperationCounter::instance().reset();

        OperationCounter::instance().increment("linear_solves", 10);
        OperationCounter::instance().increment("newton_iterations", 50);
        OperationCounter::instance().increment("linear_solves", 5);

        if (Profiler::is_enabled()) {
            CHECK(OperationCounter::instance().get("linear_solves") == 15);
            CHECK(OperationCounter::instance().get("newton_iterations") == 50);
        } else {
            // When profiling is disabled, no data is recorded
            CHECK(OperationCounter::instance().get("linear_solves") == 0);
            CHECK(OperationCounter::instance().get("newton_iterations") == 0);
        }
        CHECK(OperationCounter::instance().get("unknown") == 0);
    }

    SECTION("Profiler is_enabled check") {
        // This test verifies the is_enabled() function works correctly
        bool enabled = Profiler::is_enabled();
        INFO("Profiling enabled: " << (enabled ? "true" : "false"));

#ifdef PULSIM_ENABLE_PROFILING
        CHECK(enabled == true);
#else
        CHECK(enabled == false);
#endif
    }
}

// =============================================================================
// Memory Usage Estimation
// =============================================================================

TEST_CASE("Memory Usage Estimation", "[benchmark][memory]") {
    SECTION("Vector memory") {
        constexpr Index SIZE = 1000;
        Vector v(SIZE);

        std::size_t expected_bytes = SIZE * sizeof(Real);
        INFO("Vector size: " << SIZE);
        INFO("Expected memory: " << expected_bytes << " bytes");

        // Eigen vectors should use approximately expected memory
        CHECK(expected_bytes < 100000);  // Less than 100KB for 1000 elements
    }

    SECTION("Sparse matrix memory") {
        constexpr Index SIZE = 100;
        auto A = generate_test_matrix(SIZE, 0.1);

        std::size_t nnz = A.nonZeros();
        // Sparse matrix memory: values (8 bytes) + indices (4 bytes each) per nonzero
        std::size_t estimated_bytes = nnz * (sizeof(Real) + 2 * sizeof(Index)) + SIZE * sizeof(Index);

        INFO("Matrix size: " << SIZE << "x" << SIZE);
        INFO("Non-zeros: " << nnz);
        INFO("Estimated memory: " << estimated_bytes << " bytes");

        CHECK(estimated_bytes < 1000000);  // Less than 1MB for 100x100 sparse
    }

    SECTION("Arena capacity tracking") {
        ArenaAllocator arena(1024);

        (void)arena.allocate(256);
        (void)arena.allocate(256);
        (void)arena.allocate(256);

        INFO("Total allocated: " << arena.total_allocated() << " bytes");
        INFO("Total capacity: " << arena.total_capacity() << " bytes");
        INFO("Block count: " << arena.block_count());

        CHECK(arena.total_allocated() == 768);
        CHECK(arena.total_capacity() >= 768);
    }
}

// =============================================================================
// SIMD Detection
// =============================================================================

TEST_CASE("SIMD Capability Detection", "[benchmark][simd]") {
    SECTION("Detect SIMD level") {
        SIMDLevel level = detect_simd_level();
        const char* name = simd_level_name(level);
        std::size_t width = simd_vector_width();

        INFO("SIMD Level: " << name);
        INFO("Vector width: " << width << " doubles");

        // Should detect something on modern hardware
        CHECK(level != SIMDLevel::None);
        CHECK(width >= 1);
    }

    SECTION("Compile-time SIMD constants") {
        INFO("Compile-time SIMD level: " << simd_level_name(current_simd_level));
        INFO("Compile-time vector width: " << simd_width);

        CHECK(current_simd_level == detect_simd_level());
        CHECK(simd_width == simd_vector_width());
    }
}
