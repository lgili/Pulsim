// =============================================================================
// PulsimCore v2 - Python Bindings for High-Performance API
// =============================================================================
// Phase 7: Python Integration
// - 7.1.1: pybind11 bindings for new API
// - 7.1.2: Compatibility layer for old API
// - 7.1.3: New solver configuration options
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "pulsim/v1/core.hpp"

namespace py = pybind11;
using namespace pulsim::v1;

// =============================================================================
// Helper: Convert std::expected to Python (raise exception on error)
// =============================================================================

template<typename T>
T unwrap_expected(const std::expected<T, std::string>& result, const char* context) {
    if (!result) {
        throw std::runtime_error(std::string(context) + ": " + result.error());
    }
    return *result;
}

// =============================================================================
// Module Definition
// =============================================================================

void init_v2_module(py::module_& v2) {
    v2.doc() = R"pbdoc(
        PulsimCore v2 - High-Performance Circuit Simulation API

        This module provides the new v2 API with:
        - C++23 features and modern C++ design
        - CRTP-based device models for zero-overhead abstraction
        - Policy-based solver configuration
        - Advanced convergence aids
        - Validation and benchmarking utilities

        Example:
            import pulsim.v2 as pv2

            # Create devices
            r = pv2.Resistor(1000.0, "R1")
            c = pv2.Capacitor(1e-6, 0.0, "C1")

            # Configure solver
            opts = pv2.NewtonOptions()
            opts.max_iterations = 50

            # Use convergence aids
            config = pv2.DCConvergenceConfig()
            config.strategy = pv2.DCStrategy.Auto
    )pbdoc";

    // =========================================================================
    // Enums
    // =========================================================================

    py::enum_<DeviceType>(v2, "DeviceType", "Device type enumeration")
        .value("Resistor", DeviceType::Resistor)
        .value("Capacitor", DeviceType::Capacitor)
        .value("Inductor", DeviceType::Inductor)
        .value("VoltageSource", DeviceType::VoltageSource)
        .value("CurrentSource", DeviceType::CurrentSource)
        .value("Diode", DeviceType::Diode)
        .value("Switch", DeviceType::Switch)
        .value("MOSFET", DeviceType::MOSFET)
        .value("IGBT", DeviceType::IGBT)
        .value("Transformer", DeviceType::Transformer);
        // Don't export_values() - conflicts with class names

    py::enum_<SolverStatus>(v2, "SolverStatus", "Newton solver status")
        .value("Success", SolverStatus::Success)
        .value("MaxIterationsReached", SolverStatus::MaxIterationsReached)
        .value("SingularMatrix", SolverStatus::SingularMatrix)
        .value("NumericalError", SolverStatus::NumericalError)
        .value("ConvergenceStall", SolverStatus::ConvergenceStall)
        .value("Diverging", SolverStatus::Diverging)
        .export_values();

    py::enum_<DCStrategy>(v2, "DCStrategy", "DC analysis convergence strategy")
        .value("Direct", DCStrategy::Direct)
        .value("GminStepping", DCStrategy::GminStepping)
        .value("SourceStepping", DCStrategy::SourceStepping)
        .value("PseudoTransient", DCStrategy::PseudoTransient)
        .value("Auto", DCStrategy::Auto)
        .export_values();

    py::enum_<RLCDamping>(v2, "RLCDamping", "RLC circuit damping type")
        .value("Underdamped", RLCDamping::Underdamped)
        .value("Critical", RLCDamping::Critical)
        .value("Overdamped", RLCDamping::Overdamped)
        .export_values();

    py::enum_<DeviceHint>(v2, "DeviceHint", "Node initialization hints")
        .value("None_", DeviceHint::None)
        .value("DiodeAnode", DeviceHint::DiodeAnode)
        .value("DiodeCathode", DeviceHint::DiodeCathode)
        .value("MOSFETGate", DeviceHint::MOSFETGate)
        .value("MOSFETDrain", DeviceHint::MOSFETDrain)
        .value("MOSFETSource", DeviceHint::MOSFETSource)
        .value("BJTBase", DeviceHint::BJTBase)
        .value("BJTCollector", DeviceHint::BJTCollector)
        .value("BJTEmitter", DeviceHint::BJTEmitter)
        .value("SupplyPositive", DeviceHint::SupplyPositive)
        .value("SupplyNegative", DeviceHint::SupplyNegative)
        .value("Ground", DeviceHint::Ground)
        .export_values();

    // =========================================================================
    // CRTP Devices (7.1.1)
    // =========================================================================

    py::class_<Resistor>(v2, "Resistor", "CRTP Resistor device")
        .def(py::init<Real, const std::string&>(),
             py::arg("resistance"), py::arg("name") = "")
        .def("resistance", &Resistor::resistance)
        .def("name", &Resistor::name);

    py::class_<Capacitor>(v2, "Capacitor", "CRTP Capacitor device")
        .def(py::init<Real, Real, const std::string&>(),
             py::arg("capacitance"), py::arg("initial_voltage") = 0.0,
             py::arg("name") = "")
        .def("capacitance", &Capacitor::capacitance)
        .def("name", &Capacitor::name)
        .def("set_timestep", &Capacitor::set_timestep);

    py::class_<Inductor>(v2, "Inductor", "CRTP Inductor device")
        .def(py::init<Real, Real, const std::string&>(),
             py::arg("inductance"), py::arg("initial_current") = 0.0,
             py::arg("name") = "")
        .def("inductance", &Inductor::inductance)
        .def("name", &Inductor::name)
        .def("set_timestep", &Inductor::set_timestep);

    py::class_<VoltageSource>(v2, "VoltageSource", "CRTP Voltage source")
        .def(py::init<Real, const std::string&>(),
             py::arg("voltage"), py::arg("name") = "")
        .def("voltage", &VoltageSource::voltage)
        .def("name", &VoltageSource::name);

    py::class_<CurrentSource>(v2, "CurrentSource", "CRTP Current source")
        .def(py::init<Real, const std::string&>(),
             py::arg("current"), py::arg("name") = "")
        .def("current", &CurrentSource::current)
        .def("name", &CurrentSource::name);

    // =========================================================================
    // Solver Configuration (7.1.3)
    // =========================================================================

    py::class_<ConvergenceChecker::Tolerances>(v2, "Tolerances",
        "Convergence tolerance configuration")
        .def(py::init<>())
        .def_readwrite("voltage_abstol", &ConvergenceChecker::Tolerances::voltage_abstol)
        .def_readwrite("voltage_reltol", &ConvergenceChecker::Tolerances::voltage_reltol)
        .def_readwrite("current_abstol", &ConvergenceChecker::Tolerances::current_abstol)
        .def_readwrite("current_reltol", &ConvergenceChecker::Tolerances::current_reltol)
        .def_readwrite("residual_tol", &ConvergenceChecker::Tolerances::residual_tol)
        .def_static("defaults", &ConvergenceChecker::Tolerances::defaults);

    py::class_<NewtonOptions>(v2, "NewtonOptions", "Newton solver options")
        .def(py::init<>())
        .def_readwrite("max_iterations", &NewtonOptions::max_iterations)
        .def_readwrite("initial_damping", &NewtonOptions::initial_damping)
        .def_readwrite("min_damping", &NewtonOptions::min_damping)
        .def_readwrite("auto_damping", &NewtonOptions::auto_damping)
        .def_readwrite("track_history", &NewtonOptions::track_history)
        .def_readwrite("check_per_variable", &NewtonOptions::check_per_variable)
        .def_readwrite("num_nodes", &NewtonOptions::num_nodes)
        .def_readwrite("num_branches", &NewtonOptions::num_branches)
        .def_readwrite("tolerances", &NewtonOptions::tolerances);

    py::class_<NewtonResult>(v2, "NewtonResult", "Newton solver result")
        .def(py::init<>())
        .def_readonly("solution", &NewtonResult::solution)
        .def_readonly("status", &NewtonResult::status)
        .def_readonly("iterations", &NewtonResult::iterations)
        .def_readonly("final_residual", &NewtonResult::final_residual)
        .def_readonly("final_weighted_error", &NewtonResult::final_weighted_error)
        .def_readonly("error_message", &NewtonResult::error_message)
        .def("success", &NewtonResult::success);

    // =========================================================================
    // Convergence Aids (Phase 5 exposed)
    // =========================================================================

    py::class_<GminConfig>(v2, "GminConfig", "Gmin stepping configuration")
        .def(py::init<>())
        .def_readwrite("initial_gmin", &GminConfig::initial_gmin)
        .def_readwrite("final_gmin", &GminConfig::final_gmin)
        .def_readwrite("reduction_factor", &GminConfig::reduction_factor)
        .def_readwrite("max_steps", &GminConfig::max_steps)
        .def_readwrite("enable_logging", &GminConfig::enable_logging)
        .def("required_steps", &GminConfig::required_steps);

    py::class_<SourceSteppingConfig>(v2, "SourceSteppingConfig",
        "Source stepping configuration")
        .def(py::init<>())
        .def_readwrite("initial_scale", &SourceSteppingConfig::initial_scale)
        .def_readwrite("final_scale", &SourceSteppingConfig::final_scale)
        .def_readwrite("initial_step", &SourceSteppingConfig::initial_step)
        .def_readwrite("min_step", &SourceSteppingConfig::min_step)
        .def_readwrite("max_step", &SourceSteppingConfig::max_step)
        .def_readwrite("max_steps", &SourceSteppingConfig::max_steps)
        .def_readwrite("max_failures", &SourceSteppingConfig::max_failures)
        .def_readwrite("enable_logging", &SourceSteppingConfig::enable_logging);

    py::class_<PseudoTransientConfig>(v2, "PseudoTransientConfig",
        "Pseudo-transient continuation configuration")
        .def(py::init<>())
        .def_readwrite("initial_dt", &PseudoTransientConfig::initial_dt)
        .def_readwrite("max_dt", &PseudoTransientConfig::max_dt)
        .def_readwrite("min_dt", &PseudoTransientConfig::min_dt)
        .def_readwrite("dt_increase", &PseudoTransientConfig::dt_increase)
        .def_readwrite("dt_decrease", &PseudoTransientConfig::dt_decrease)
        .def_readwrite("convergence_threshold", &PseudoTransientConfig::convergence_threshold)
        .def_readwrite("max_iterations", &PseudoTransientConfig::max_iterations)
        .def_readwrite("enable_logging", &PseudoTransientConfig::enable_logging);

    py::class_<InitializationConfig>(v2, "InitializationConfig",
        "Robust initialization configuration")
        .def(py::init<>())
        .def_readwrite("default_voltage", &InitializationConfig::default_voltage)
        .def_readwrite("supply_voltage", &InitializationConfig::supply_voltage)
        .def_readwrite("diode_forward", &InitializationConfig::diode_forward)
        .def_readwrite("mosfet_threshold", &InitializationConfig::mosfet_threshold)
        .def_readwrite("use_zero_init", &InitializationConfig::use_zero_init)
        .def_readwrite("use_warm_start", &InitializationConfig::use_warm_start)
        .def_readwrite("max_random_restarts", &InitializationConfig::max_random_restarts)
        .def_readwrite("random_seed", &InitializationConfig::random_seed)
        .def_readwrite("random_voltage_range", &InitializationConfig::random_voltage_range);

    py::class_<DCConvergenceConfig>(v2, "DCConvergenceConfig",
        "DC solver configuration with convergence aids")
        .def(py::init<>())
        .def_readwrite("strategy", &DCConvergenceConfig::strategy)
        .def_readwrite("gmin_config", &DCConvergenceConfig::gmin_config)
        .def_readwrite("source_config", &DCConvergenceConfig::source_config)
        .def_readwrite("pseudo_config", &DCConvergenceConfig::pseudo_config)
        .def_readwrite("init_config", &DCConvergenceConfig::init_config)
        .def_readwrite("enable_random_restart", &DCConvergenceConfig::enable_random_restart)
        .def_readwrite("max_strategy_attempts", &DCConvergenceConfig::max_strategy_attempts);

    py::class_<DCAnalysisResult>(v2, "DCAnalysisResult", "DC analysis result")
        .def(py::init<>())
        .def_readonly("newton_result", &DCAnalysisResult::newton_result)
        .def_readonly("strategy_used", &DCAnalysisResult::strategy_used)
        .def_readonly("random_restarts", &DCAnalysisResult::random_restarts)
        .def_readonly("total_newton_iterations", &DCAnalysisResult::total_newton_iterations)
        .def_readonly("success", &DCAnalysisResult::success)
        .def_readonly("message", &DCAnalysisResult::message);

    // =========================================================================
    // Validation Framework (Phase 6 exposed)
    // =========================================================================

    py::class_<RCAnalytical>(v2, "RCAnalytical", "RC circuit analytical solution")
        .def(py::init<Real, Real, Real, Real>(),
             py::arg("R"), py::arg("C"), py::arg("V_initial"), py::arg("V_final"))
        .def("tau", &RCAnalytical::tau)
        .def("voltage", &RCAnalytical::voltage, py::arg("t"))
        .def("current", &RCAnalytical::current, py::arg("t"))
        .def("waveform", &RCAnalytical::waveform,
             py::arg("t_start"), py::arg("t_end"), py::arg("dt"));

    py::class_<RLAnalytical>(v2, "RLAnalytical", "RL circuit analytical solution")
        .def(py::init<Real, Real, Real, Real>(),
             py::arg("R"), py::arg("L"), py::arg("V_source"), py::arg("I_initial"))
        .def("tau", &RLAnalytical::tau)
        .def("I_final", &RLAnalytical::I_final)
        .def("current", &RLAnalytical::current, py::arg("t"))
        .def("voltage_R", &RLAnalytical::voltage_R, py::arg("t"))
        .def("voltage_L", &RLAnalytical::voltage_L, py::arg("t"))
        .def("waveform", &RLAnalytical::waveform,
             py::arg("t_start"), py::arg("t_end"), py::arg("dt"));

    py::class_<RLCAnalytical>(v2, "RLCAnalytical", "RLC circuit analytical solution")
        .def(py::init<Real, Real, Real, Real, Real, Real>(),
             py::arg("R"), py::arg("L"), py::arg("C"),
             py::arg("V_source"), py::arg("V_initial"), py::arg("I_initial"))
        .def("omega_0", &RLCAnalytical::omega_0)
        .def("zeta", &RLCAnalytical::zeta)
        .def("alpha", &RLCAnalytical::alpha)
        .def("damping_type", &RLCAnalytical::damping_type)
        .def("voltage", &RLCAnalytical::voltage, py::arg("t"))
        .def("current", &RLCAnalytical::current, py::arg("t"))
        .def("waveform", &RLCAnalytical::waveform,
             py::arg("t_start"), py::arg("t_end"), py::arg("dt"));

    py::class_<ValidationResult>(v2, "ValidationResult_v2", "Validation result with metrics")
        .def(py::init<>())
        .def_readwrite("test_name", &ValidationResult::test_name)
        .def_readwrite("passed", &ValidationResult::passed)
        .def_readwrite("max_error", &ValidationResult::max_error)
        .def_readwrite("rms_error", &ValidationResult::rms_error)
        .def_readwrite("max_relative_error", &ValidationResult::max_relative_error)
        .def_readwrite("mean_error", &ValidationResult::mean_error)
        .def_readwrite("num_points", &ValidationResult::num_points)
        .def_readwrite("error_threshold", &ValidationResult::error_threshold)
        .def("to_string", &ValidationResult::to_string);

    v2.def("compare_waveforms", &compare_waveforms,
           py::arg("name"), py::arg("simulated"), py::arg("analytical"),
           py::arg("threshold") = 0.001,
           "Compare simulated vs analytical waveforms");

    v2.def("export_validation_csv", &export_validation_csv,
           py::arg("results"),
           "Export validation results to CSV");

    v2.def("export_validation_json", &export_validation_json,
           py::arg("results"),
           "Export validation results to JSON");

    // =========================================================================
    // Benchmark Framework
    // =========================================================================

    py::class_<BenchmarkTiming>(v2, "BenchmarkTiming", "Benchmark timing result")
        .def(py::init<>())
        .def_readwrite("name", &BenchmarkTiming::name)
        .def_readwrite("iterations", &BenchmarkTiming::iterations)
        .def("average_ms", &BenchmarkTiming::average_ms)
        .def("min_ms", &BenchmarkTiming::min_ms)
        .def("max_ms", &BenchmarkTiming::max_ms)
        .def("total_ms", &BenchmarkTiming::total_ms);

    py::class_<BenchmarkResult>(v2, "BenchmarkResult", "Complete benchmark result")
        .def(py::init<>())
        .def_readwrite("circuit_name", &BenchmarkResult::circuit_name)
        .def_readwrite("num_nodes", &BenchmarkResult::num_nodes)
        .def_readwrite("num_devices", &BenchmarkResult::num_devices)
        .def_readwrite("num_timesteps", &BenchmarkResult::num_timesteps)
        .def_readwrite("timing", &BenchmarkResult::timing)
        .def_readwrite("simulation_time", &BenchmarkResult::simulation_time)
        .def("timesteps_per_second", &BenchmarkResult::timesteps_per_second)
        .def("to_string", &BenchmarkResult::to_string);

    v2.def("export_benchmark_csv", &export_benchmark_csv,
           py::arg("results"),
           "Export benchmark results to CSV");

    v2.def("export_benchmark_json", &export_benchmark_json,
           py::arg("results"),
           "Export benchmark results to JSON");

    // =========================================================================
    // Integration Methods
    // =========================================================================

    py::class_<BDFOrderConfig>(v2, "BDFOrderConfig", "BDF order controller configuration")
        .def(py::init<>())
        .def_readwrite("min_order", &BDFOrderConfig::min_order)
        .def_readwrite("max_order", &BDFOrderConfig::max_order)
        .def_readwrite("initial_order", &BDFOrderConfig::initial_order)
        .def_readwrite("order_increase_threshold", &BDFOrderConfig::order_increase_threshold)
        .def_readwrite("order_decrease_threshold", &BDFOrderConfig::order_decrease_threshold)
        .def_readwrite("steps_before_increase", &BDFOrderConfig::steps_before_increase)
        .def_readwrite("enable_auto_order", &BDFOrderConfig::enable_auto_order);

    py::class_<TimestepConfig>(v2, "TimestepConfig", "Adaptive timestep controller configuration")
        .def(py::init<>())
        .def_readwrite("dt_min", &TimestepConfig::dt_min)
        .def_readwrite("dt_max", &TimestepConfig::dt_max)
        .def_readwrite("dt_initial", &TimestepConfig::dt_initial)
        .def_readwrite("safety_factor", &TimestepConfig::safety_factor)
        .def_readwrite("error_tolerance", &TimestepConfig::error_tolerance)
        .def_readwrite("growth_factor", &TimestepConfig::growth_factor)
        .def_readwrite("shrink_factor", &TimestepConfig::shrink_factor)
        .def_readwrite("max_rejections", &TimestepConfig::max_rejections)
        .def_readwrite("k_p", &TimestepConfig::k_p)
        .def_readwrite("k_i", &TimestepConfig::k_i)
        .def_static("defaults", &TimestepConfig::defaults)
        .def_static("conservative", &TimestepConfig::conservative)
        .def_static("aggressive", &TimestepConfig::aggressive);

    // =========================================================================
    // High-Performance Features (Phase 4 exposed)
    // =========================================================================

    py::class_<LinearSolverConfig>(v2, "LinearSolverConfig",
        "Linear solver configuration")
        .def(py::init<>())
        .def_readwrite("pivot_tolerance", &LinearSolverConfig::pivot_tolerance)
        .def_readwrite("reuse_symbolic", &LinearSolverConfig::reuse_symbolic)
        .def_readwrite("detect_pattern_change", &LinearSolverConfig::detect_pattern_change)
        .def_readwrite("deterministic_pivoting", &LinearSolverConfig::deterministic_pivoting);

    // SIMD detection
    py::enum_<SIMDLevel>(v2, "SIMDLevel", "SIMD capability level")
        .value("None_", SIMDLevel::None)
        .value("SSE2", SIMDLevel::SSE2)
        .value("SSE4", SIMDLevel::SSE4)
        .value("AVX", SIMDLevel::AVX)
        .value("AVX2", SIMDLevel::AVX2)
        .value("AVX512", SIMDLevel::AVX512)
        .value("NEON", SIMDLevel::NEON)
        .export_values();

    v2.def("detect_simd_level", &detect_simd_level,
           "Detect CPU SIMD capability level");

    v2.def("simd_vector_width", &simd_vector_width,
           "Get SIMD vector width for current CPU level");

    // =========================================================================
    // Utility Functions
    // =========================================================================

    v2.def("solver_status_to_string", &to_string,
           py::arg("status"),
           "Convert SolverStatus to string");

    // Version info
    v2.attr("__version__") = "2.0.0";
}

// =============================================================================
// Module Registration
// =============================================================================

PYBIND11_MODULE(_pulsim, m) {
    m.doc() = "PulsimCore High-Performance Circuit Simulation (C++ extension)";
    init_v2_module(m);
}
