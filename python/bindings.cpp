#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include "pulsim/types.hpp"
#include "pulsim/circuit.hpp"
#include "pulsim/parser.hpp"
#include "pulsim/simulation.hpp"
#include "pulsim/simulation_control.hpp"
#include "pulsim/metadata.hpp"
#include "pulsim/validation.hpp"
#include "pulsim/thermal.hpp"
#include "pulsim/devices.hpp"

namespace py = pybind11;
using namespace pulsim;

PYBIND11_MODULE(_pulsim, m) {
    m.doc() = "Pulsim - High-performance circuit simulator for power electronics";

    // --- Enums ---
    py::enum_<ComponentType>(m, "ComponentType")
        .value("Resistor", ComponentType::Resistor)
        .value("Capacitor", ComponentType::Capacitor)
        .value("Inductor", ComponentType::Inductor)
        .value("VoltageSource", ComponentType::VoltageSource)
        .value("CurrentSource", ComponentType::CurrentSource)
        .value("Diode", ComponentType::Diode)
        .value("Switch", ComponentType::Switch)
        .value("MOSFET", ComponentType::MOSFET)
        .value("Transformer", ComponentType::Transformer)
        .export_values();

    py::enum_<SolverStatus>(m, "SolverStatus")
        .value("Success", SolverStatus::Success)
        .value("MaxIterationsReached", SolverStatus::MaxIterationsReached)
        .value("SingularMatrix", SolverStatus::SingularMatrix)
        .value("NumericalError", SolverStatus::NumericalError)
        .export_values();

    py::enum_<MOSFETType>(m, "MOSFETType")
        .value("NMOS", MOSFETType::NMOS)
        .value("PMOS", MOSFETType::PMOS)
        .export_values();

    py::enum_<ThermalNetworkType>(m, "ThermalNetworkType")
        .value("Foster", ThermalNetworkType::Foster)
        .value("Cauer", ThermalNetworkType::Cauer)
        .value("Simple", ThermalNetworkType::Simple)
        .export_values();

    py::enum_<SimulationState>(m, "SimulationState",
        "Simulation execution states for GUI status display.\n\n"
        "States:\n"
        "    Idle: Simulation not started\n"
        "    Running: Simulation actively executing\n"
        "    Paused: Simulation paused, can be resumed\n"
        "    Stopping: Stop requested, finishing current step\n"
        "    Completed: Simulation finished successfully\n"
        "    Error: Simulation terminated with error")
        .value("Idle", SimulationState::Idle)
        .value("Running", SimulationState::Running)
        .value("Paused", SimulationState::Paused)
        .value("Stopping", SimulationState::Stopping)
        .value("Completed", SimulationState::Completed)
        .value("Error", SimulationState::Error)
        .export_values();

    py::enum_<IntegrationMethod>(m, "IntegrationMethod")
        .value("BackwardEuler", IntegrationMethod::BackwardEuler)
        .value("Trapezoidal", IntegrationMethod::Trapezoidal)
        .value("BDF2", IntegrationMethod::BDF2)
        .value("GEAR2", IntegrationMethod::GEAR2)
        .export_values();

    py::enum_<DiagnosticSeverity>(m, "DiagnosticSeverity",
        "Severity levels for circuit validation diagnostics.\n\n"
        "Levels:\n"
        "    Error: Must be fixed before simulation can run\n"
        "    Warning: May cause issues but simulation can proceed\n"
        "    Info: Informational message only")
        .value("Error", DiagnosticSeverity::Error)
        .value("Warning", DiagnosticSeverity::Warning)
        .value("Info", DiagnosticSeverity::Info)
        .export_values();

    py::enum_<DiagnosticCode>(m, "DiagnosticCode",
        "Specific diagnostic codes for programmatic error handling.\n\n"
        "Errors (E_xxx):\n"
        "    E_NO_GROUND: No ground reference node\n"
        "    E_VOLTAGE_SOURCE_LOOP: Voltage sources form a loop\n"
        "    E_INDUCTOR_LOOP: Inductors/V-sources form a loop\n"
        "    E_NO_DC_PATH: Node has no DC path to ground\n"
        "    E_INVALID_PARAMETER: Parameter value out of range\n"
        "    E_UNKNOWN_NODE: Referenced node doesn't exist\n"
        "    E_DUPLICATE_NAME: Component name already used\n"
        "    E_NO_COMPONENTS: Circuit has no components\n\n"
        "Warnings (W_xxx):\n"
        "    W_FLOATING_NODE: Node has only one connection\n"
        "    W_SHORT_CIRCUIT: Very low impedance path detected\n"
        "    W_HIGH_VOLTAGE: Unusually high voltage expected\n"
        "    W_MISSING_IC: Capacitor/inductor has no initial condition\n"
        "    W_LARGE_TIMESTEP: Timestep may be too large\n\n"
        "Info (I_xxx):\n"
        "    I_IDEAL_SWITCH: Using ideal switch model\n"
        "    I_NO_LOSS_MODEL: Power loss calculation not available\n"
        "    I_PARALLEL_SOURCES: Parallel voltage sources detected")
        .value("E_NO_GROUND", DiagnosticCode::E_NO_GROUND)
        .value("E_VOLTAGE_SOURCE_LOOP", DiagnosticCode::E_VOLTAGE_SOURCE_LOOP)
        .value("E_INDUCTOR_LOOP", DiagnosticCode::E_INDUCTOR_LOOP)
        .value("E_NO_DC_PATH", DiagnosticCode::E_NO_DC_PATH)
        .value("E_INVALID_PARAMETER", DiagnosticCode::E_INVALID_PARAMETER)
        .value("E_UNKNOWN_NODE", DiagnosticCode::E_UNKNOWN_NODE)
        .value("E_DUPLICATE_NAME", DiagnosticCode::E_DUPLICATE_NAME)
        .value("E_NO_COMPONENTS", DiagnosticCode::E_NO_COMPONENTS)
        .value("W_FLOATING_NODE", DiagnosticCode::W_FLOATING_NODE)
        .value("W_SHORT_CIRCUIT", DiagnosticCode::W_SHORT_CIRCUIT)
        .value("W_HIGH_VOLTAGE", DiagnosticCode::W_HIGH_VOLTAGE)
        .value("W_MISSING_IC", DiagnosticCode::W_MISSING_IC)
        .value("W_LARGE_TIMESTEP", DiagnosticCode::W_LARGE_TIMESTEP)
        .value("I_IDEAL_SWITCH", DiagnosticCode::I_IDEAL_SWITCH)
        .value("I_NO_LOSS_MODEL", DiagnosticCode::I_NO_LOSS_MODEL)
        .value("I_PARALLEL_SOURCES", DiagnosticCode::I_PARALLEL_SOURCES)
        .export_values();

    py::enum_<ParameterType>(m, "ParameterType",
        "Data types for component parameters in property editors.\n\n"
        "Types:\n"
        "    Real: Floating-point number (use spin box or slider)\n"
        "    Integer: Integer value (use integer spin box)\n"
        "    Boolean: True/false (use checkbox)\n"
        "    Enum: One of several choices (use dropdown)\n"
        "    String: Text input (e.g., model name)")
        .value("Real", ParameterType::Real)
        .value("Integer", ParameterType::Integer)
        .value("Boolean", ParameterType::Boolean)
        .value("Enum", ParameterType::Enum)
        .value("String", ParameterType::String)
        .export_values();

    // --- Waveforms ---
    py::class_<DCWaveform>(m, "DCWaveform")
        .def(py::init<Real>(), py::arg("value"))
        .def_readwrite("value", &DCWaveform::value);

    py::class_<PulseWaveform>(m, "PulseWaveform")
        .def(py::init<>())
        .def_readwrite("v1", &PulseWaveform::v1)
        .def_readwrite("v2", &PulseWaveform::v2)
        .def_readwrite("td", &PulseWaveform::td)
        .def_readwrite("tr", &PulseWaveform::tr)
        .def_readwrite("tf", &PulseWaveform::tf)
        .def_readwrite("pw", &PulseWaveform::pw)
        .def_readwrite("period", &PulseWaveform::period);

    py::class_<SineWaveform>(m, "SineWaveform")
        .def(py::init<>())
        .def_readwrite("offset", &SineWaveform::offset)
        .def_readwrite("amplitude", &SineWaveform::amplitude)
        .def_readwrite("frequency", &SineWaveform::frequency)
        .def_readwrite("delay", &SineWaveform::delay)
        .def_readwrite("damping", &SineWaveform::damping);

    py::class_<PWLWaveform>(m, "PWLWaveform")
        .def(py::init<>())
        .def_readwrite("points", &PWLWaveform::points);

    py::class_<PWMWaveform>(m, "PWMWaveform")
        .def(py::init<>())
        .def_readwrite("v_off", &PWMWaveform::v_off)
        .def_readwrite("v_on", &PWMWaveform::v_on)
        .def_readwrite("frequency", &PWMWaveform::frequency)
        .def_readwrite("duty", &PWMWaveform::duty)
        .def_readwrite("dead_time", &PWMWaveform::dead_time)
        .def_readwrite("phase", &PWMWaveform::phase)
        .def_readwrite("complementary", &PWMWaveform::complementary)
        .def("period", &PWMWaveform::period)
        .def("t_on", &PWMWaveform::t_on);

    // --- Component Parameters ---
    py::class_<DiodeParams>(m, "DiodeParams")
        .def(py::init<>())
        .def_readwrite("is_", &DiodeParams::is)
        .def_readwrite("n", &DiodeParams::n)
        .def_readwrite("rs", &DiodeParams::rs)
        .def_readwrite("vt", &DiodeParams::vt)
        .def_readwrite("ideal", &DiodeParams::ideal);

    py::class_<SwitchParams>(m, "SwitchParams")
        .def(py::init<>())
        .def_readwrite("ron", &SwitchParams::ron)
        .def_readwrite("roff", &SwitchParams::roff)
        .def_readwrite("vth", &SwitchParams::vth)
        .def_readwrite("initial_state", &SwitchParams::initial_state);

    py::class_<MOSFETParams>(m, "MOSFETParams")
        .def(py::init<>())
        .def_readwrite("type", &MOSFETParams::type)
        .def_readwrite("vth", &MOSFETParams::vth)
        .def_readwrite("kp", &MOSFETParams::kp)
        .def_readwrite("lambda_", &MOSFETParams::lambda)
        .def_readwrite("w", &MOSFETParams::w)
        .def_readwrite("l", &MOSFETParams::l)
        .def_readwrite("body_diode", &MOSFETParams::body_diode)
        .def_readwrite("rds_on", &MOSFETParams::rds_on)
        .def("kp_effective", &MOSFETParams::kp_effective);

    py::class_<IGBTParams>(m, "IGBTParams")
        .def(py::init<>())
        .def_readwrite("vth", &IGBTParams::vth)
        .def_readwrite("vce_sat", &IGBTParams::vce_sat)
        .def_readwrite("rce_on", &IGBTParams::rce_on)
        .def_readwrite("rce_off", &IGBTParams::rce_off)
        .def_readwrite("tf", &IGBTParams::tf)
        .def_readwrite("tr", &IGBTParams::tr)
        .def_readwrite("cies", &IGBTParams::cies)
        .def_readwrite("body_diode", &IGBTParams::body_diode)
        .def_readwrite("is_diode", &IGBTParams::is_diode)
        .def_readwrite("n_diode", &IGBTParams::n_diode)
        .def_readwrite("vf_diode", &IGBTParams::vf_diode);

    py::class_<TransformerParams>(m, "TransformerParams")
        .def(py::init<>())
        .def_readwrite("turns_ratio", &TransformerParams::turns_ratio)
        .def_readwrite("lm", &TransformerParams::lm)
        .def_readwrite("ll1", &TransformerParams::ll1)
        .def_readwrite("ll2", &TransformerParams::ll2);

    // --- Simulation Options ---
    py::class_<SimulationOptions>(m, "SimulationOptions")
        .def(py::init<>())
        .def_readwrite("tstart", &SimulationOptions::tstart)
        .def_readwrite("tstop", &SimulationOptions::tstop)
        .def_readwrite("dt", &SimulationOptions::dt)
        .def_readwrite("dtmin", &SimulationOptions::dtmin)
        .def_readwrite("dtmax", &SimulationOptions::dtmax)
        .def_readwrite("abstol", &SimulationOptions::abstol)
        .def_readwrite("reltol", &SimulationOptions::reltol)
        .def_readwrite("max_newton_iterations", &SimulationOptions::max_newton_iterations)
        .def_readwrite("damping_factor", &SimulationOptions::damping_factor)
        .def_readwrite("use_ic", &SimulationOptions::use_ic)
        .def_readwrite("integration_method", &SimulationOptions::integration_method)
        .def_readwrite("adaptive_timestep", &SimulationOptions::adaptive_timestep)
        .def_readwrite("lte_rtol", &SimulationOptions::lte_rtol)
        .def_readwrite("lte_atol", &SimulationOptions::lte_atol)
        .def_readwrite("output_signals", &SimulationOptions::output_signals)
        // Streaming configuration for GUI
        .def_readwrite("streaming_decimation", &SimulationOptions::streaming_decimation,
            "Store every Nth point (1 = all points, 10 = every 10th)")
        .def_readwrite("streaming_rolling_buffer", &SimulationOptions::streaming_rolling_buffer,
            "If true, keep only the last streaming_max_points")
        .def_readwrite("streaming_max_points", &SimulationOptions::streaming_max_points,
            "Maximum points to store when using rolling buffer")
        // Progress callback throttling
        .def_readwrite("progress_min_interval_ms", &SimulationOptions::progress_min_interval_ms,
            "Minimum milliseconds between progress callbacks")
        .def_readwrite("progress_min_steps", &SimulationOptions::progress_min_steps,
            "Minimum simulation steps between progress callbacks")
        .def_readwrite("progress_include_memory", &SimulationOptions::progress_include_memory,
            "Track memory usage in progress callbacks (slower)");

    // --- Simulation Controller ---
    py::class_<SimulationController>(m, "SimulationController",
        "Thread-safe simulation controller for pause/resume/stop operations.\n\n"
        "Use this class in GUI applications to control long-running simulations\n"
        "from a separate thread. The controller implements a state machine with\n"
        "states: Idle -> Running <-> Paused -> Completed/Error\n\n"
        "Example:\n"
        "    controller = pulsim.SimulationController()\n\n"
        "    # In simulation thread:\n"
        "    result = sim.run_transient_with_progress(\n"
        "        control=controller,\n"
        "        progress_callback=lambda p: update_progress(p.progress_percent)\n"
        "    )\n\n"
        "    # In GUI thread:\n"
        "    controller.request_pause()   # Pause button\n"
        "    controller.request_resume()  # Resume button\n"
        "    controller.request_stop()    # Stop button")
        .def(py::init<>())
        .def_property_readonly("state", &SimulationController::state,
            "Get the current simulation state (SimulationState enum).")
        .def("is_idle", &SimulationController::is_idle,
            "Check if simulation is in Idle state (not started).")
        .def("is_running", &SimulationController::is_running,
            "Check if simulation is actively running.")
        .def("is_paused", &SimulationController::is_paused,
            "Check if simulation is paused.")
        .def("is_stopping", &SimulationController::is_stopping,
            "Check if simulation is stopping (finishing current step).")
        .def("is_completed", &SimulationController::is_completed,
            "Check if simulation completed successfully.")
        .def("is_error", &SimulationController::is_error,
            "Check if simulation terminated with an error.")
        .def("request_pause", &SimulationController::request_pause,
            "Request simulation to pause at next checkpoint.")
        .def("request_resume", &SimulationController::request_resume,
            "Request simulation to resume after pause.")
        .def("request_stop", &SimulationController::request_stop,
            "Request simulation to stop. Results up to stop point will be available.")
        .def("reset", &SimulationController::reset,
            "Reset controller to Idle state before starting a new simulation.")
        .def("wait_for_state", &SimulationController::wait_for_state,
             py::arg("target"), py::arg("timeout_ms") = -1,
            "Wait for simulation to reach a specific state.\n\n"
            "Args:\n"
            "    target: The SimulationState to wait for\n"
            "    timeout_ms: Maximum time to wait (-1 for infinite)\n\n"
            "Returns:\n"
            "    True if target state was reached, False if timeout occurred");

    // --- Simulation Progress ---
    py::class_<SimulationProgress>(m, "SimulationProgress",
        "Real-time progress information during simulation.\n\n"
        "Passed to progress callbacks to provide current simulation status.\n"
        "Use this to update progress bars, show elapsed/remaining time,\n"
        "display convergence warnings, and monitor memory usage.\n\n"
        "Attributes:\n"
        "    current_time: Current simulation time (seconds)\n"
        "    total_time: Target end time (seconds)\n"
        "    progress_percent: Progress from 0.0 to 100.0\n"
        "    steps_completed: Number of timesteps completed\n"
        "    total_steps_estimate: Estimated total steps (-1 if unknown)\n"
        "    newton_iterations: Newton iterations for current step\n"
        "    convergence_warning: True if current step needed >10 iterations\n"
        "    elapsed_seconds: Wall-clock time elapsed\n"
        "    estimated_remaining_seconds: Estimated time remaining (-1 if unknown)\n"
        "    memory_bytes: Current memory usage (-1 if not tracked)")
        .def(py::init<>())
        .def_readwrite("current_time", &SimulationProgress::current_time)
        .def_readwrite("total_time", &SimulationProgress::total_time)
        .def_readwrite("progress_percent", &SimulationProgress::progress_percent)
        .def_readwrite("steps_completed", &SimulationProgress::steps_completed)
        .def_readwrite("total_steps_estimate", &SimulationProgress::total_steps_estimate)
        .def_readwrite("newton_iterations", &SimulationProgress::newton_iterations)
        .def_readwrite("convergence_warning", &SimulationProgress::convergence_warning)
        .def_readwrite("elapsed_seconds", &SimulationProgress::elapsed_seconds)
        .def_readwrite("estimated_remaining_seconds", &SimulationProgress::estimated_remaining_seconds)
        .def_readwrite("memory_bytes", &SimulationProgress::memory_bytes)
        .def("to_dict", [](const SimulationProgress& p) {
            py::dict result;
            result["current_time"] = p.current_time;
            result["total_time"] = p.total_time;
            result["progress_percent"] = p.progress_percent;
            result["steps_completed"] = p.steps_completed;
            result["total_steps_estimate"] = p.total_steps_estimate;
            result["newton_iterations"] = p.newton_iterations;
            result["convergence_warning"] = p.convergence_warning;
            result["elapsed_seconds"] = p.elapsed_seconds;
            result["estimated_remaining_seconds"] = p.estimated_remaining_seconds;
            result["memory_bytes"] = p.memory_bytes;
            return result;
        }, "Convert progress to dictionary for JSON serialization.");

    // --- Progress Callback Config ---
    py::class_<ProgressCallbackConfig>(m, "ProgressCallbackConfig",
        "Configuration for progress callback behavior.\n\n"
        "Controls how frequently progress callbacks are invoked during simulation.\n"
        "Callbacks are throttled to avoid excessive overhead while still providing\n"
        "responsive GUI updates.\n\n"
        "Attributes:\n"
        "    min_interval_ms: Minimum milliseconds between callbacks (default 100)\n"
        "    min_steps: Minimum simulation steps between callbacks (default 100)\n"
        "    include_memory: Track memory usage in callbacks (slower if True)")
        .def(py::init<>())
        .def_readwrite("min_interval_ms", &ProgressCallbackConfig::min_interval_ms,
            "Minimum milliseconds between callbacks (default 100).")
        .def_readwrite("min_steps", &ProgressCallbackConfig::min_steps,
            "Minimum simulation steps between callbacks (default 100).")
        .def_readwrite("include_memory", &ProgressCallbackConfig::include_memory,
            "Track memory usage in callbacks (slower if True).");

    // --- Streaming Config ---
    py::class_<StreamingConfig>(m, "StreamingConfig",
        "Configuration for memory-efficient result streaming.\n\n"
        "For long simulations, storing every timestep can consume excessive memory.\n"
        "StreamingConfig provides options to reduce memory usage:\n"
        "- Decimation: Store only every Nth point\n"
        "- Rolling buffer: Keep only the most recent max_points\n\n"
        "Example:\n"
        "    opts = pulsim.SimulationOptions()\n"
        "    opts.streaming_decimation = 10       # Store every 10th point\n"
        "    opts.streaming_rolling_buffer = True # Enable rolling buffer\n"
        "    opts.streaming_max_points = 10000    # Keep last 10000 points\n\n"
        "Attributes:\n"
        "    decimation_factor: Store every Nth point (1 = all points)\n"
        "    use_rolling_buffer: If True, keep only last max_points\n"
        "    max_points: Maximum points when rolling buffer enabled\n"
        "    callback_interval_ms: Callback throttle (0 = every stored point)")
        .def(py::init<>())
        .def_readwrite("decimation_factor", &StreamingConfig::decimation_factor,
            "Store every Nth point (1 = all points, default).")
        .def_readwrite("use_rolling_buffer", &StreamingConfig::use_rolling_buffer,
            "If True, keep only the last max_points.")
        .def_readwrite("max_points", &StreamingConfig::max_points,
            "Maximum points to store when using rolling buffer (default 100000).")
        .def_readwrite("callback_interval_ms", &StreamingConfig::callback_interval_ms,
            "Callback throttle in ms (0 = every stored point).");

    // --- Signal Info ---
    py::class_<SignalInfo>(m, "SignalInfo",
        "Metadata about a signal in simulation results.\n\n"
        "Use this to build GUI plot selectors and label axes correctly.\n\n"
        "Attributes:\n"
        "    name: Signal name (e.g., 'V(out)', 'I(L1)')\n"
        "    type: Signal type ('voltage' or 'current')\n"
        "    unit: Unit string ('V' or 'A')\n"
        "    component: Associated component name (empty for node voltages)\n"
        "    nodes: List of node names involved")
        .def(py::init<>())
        .def_readwrite("name", &SignalInfo::name)
        .def_readwrite("type", &SignalInfo::type)
        .def_readwrite("unit", &SignalInfo::unit)
        .def_readwrite("component", &SignalInfo::component)
        .def_readwrite("nodes", &SignalInfo::nodes)
        .def("to_dict", [](const SignalInfo& s) {
            py::dict result;
            result["name"] = s.name;
            result["type"] = s.type;
            result["unit"] = s.unit;
            result["component"] = s.component;
            result["nodes"] = s.nodes;
            return result;
        }, "Convert to dictionary for JSON serialization.");

    // --- Solver Info ---
    py::class_<SolverInfo>(m, "SolverInfo",
        "Information about solver settings used in simulation.\n\n"
        "Included in SimulationResult to document how the simulation was run.\n\n"
        "Attributes:\n"
        "    method: Integration method name (e.g., 'Trapezoidal', 'BDF2')\n"
        "    abstol: Absolute tolerance used\n"
        "    reltol: Relative tolerance used\n"
        "    adaptive_timestep: Whether adaptive timestep was enabled")
        .def(py::init<>())
        .def_readwrite("method", &SolverInfo::method)
        .def_readwrite("abstol", &SolverInfo::abstol)
        .def_readwrite("reltol", &SolverInfo::reltol)
        .def_readwrite("adaptive_timestep", &SolverInfo::adaptive_timestep);

    // --- Simulation Event Type ---
    py::enum_<SimulationEventType>(m, "SimulationEventType",
        "Types of events that can occur during simulation.\n\n"
        "Types:\n"
        "    SwitchClose: A switch closed\n"
        "    SwitchOpen: A switch opened\n"
        "    Convergence: Newton solver had difficulty converging\n"
        "    TimestepChange: Adaptive timestep changed")
        .value("SwitchClose", SimulationEventType::SwitchClose)
        .value("SwitchOpen", SimulationEventType::SwitchOpen)
        .value("Convergence", SimulationEventType::Convergence)
        .value("TimestepChange", SimulationEventType::TimestepChange)
        .export_values();

    // --- Simulation Event ---
    py::class_<SimulationEvent>(m, "SimulationEvent",
        "A discrete event that occurred during simulation.\n\n"
        "Events include switch state changes, convergence issues, and timestep changes.\n"
        "Use events to build event logs, annotate plots, and debug simulation issues.\n\n"
        "Attributes:\n"
        "    time: Simulation time when event occurred (seconds)\n"
        "    type: Event type (SimulationEventType enum)\n"
        "    component: Component name involved (empty if not applicable)\n"
        "    description: Human-readable event description\n"
        "    value1, value2: Event-specific values (e.g., old/new timestep)")
        .def(py::init<>())
        .def_readonly("time", &SimulationEvent::time)
        .def_readonly("type", &SimulationEvent::type)
        .def_readonly("component", &SimulationEvent::component)
        .def_readonly("description", &SimulationEvent::description)
        .def_readonly("value1", &SimulationEvent::value1)
        .def_readonly("value2", &SimulationEvent::value2)
        .def("to_dict", [](const SimulationEvent& e) {
            py::dict result;
            result["time"] = e.time;
            result["type"] = static_cast<int>(e.type);
            result["component"] = e.component;
            result["description"] = e.description;
            result["value1"] = e.value1;
            result["value2"] = e.value2;
            return result;
        }, "Convert to dictionary for JSON serialization.");

    // --- Simulation Result ---
    py::class_<SimulationResult>(m, "SimulationResult")
        .def(py::init<>())
        .def_readonly("time", &SimulationResult::time)
        .def_readonly("signal_names", &SimulationResult::signal_names)
        .def_readonly("data", &SimulationResult::data)
        .def_readonly("total_time_seconds", &SimulationResult::total_time_seconds)
        .def_readonly("total_steps", &SimulationResult::total_steps)
        .def_readonly("newton_iterations_total", &SimulationResult::newton_iterations_total)
        .def_readonly("final_status", &SimulationResult::final_status)
        .def_readonly("error_message", &SimulationResult::error_message)
        // New enhanced fields
        .def_readonly("signal_info", &SimulationResult::signal_info)
        .def_readonly("solver_info", &SimulationResult::solver_info)
        .def_readonly("average_newton_iterations", &SimulationResult::average_newton_iterations)
        .def_readonly("convergence_failures", &SimulationResult::convergence_failures)
        .def_readonly("timestep_reductions", &SimulationResult::timestep_reductions)
        .def_readonly("peak_memory_bytes", &SimulationResult::peak_memory_bytes)
        .def_readonly("events", &SimulationResult::events)
        .def("num_signals", &SimulationResult::num_signals)
        .def("num_points", &SimulationResult::num_points)
        .def("num_events", &SimulationResult::num_events)
        .def("to_dict", [](const SimulationResult& r) {
            py::dict result;
            result["time"] = r.time;
            result["signal_names"] = r.signal_names;
            result["total_time_seconds"] = r.total_time_seconds;
            result["total_steps"] = r.total_steps;
            result["status"] = static_cast<int>(r.final_status);
            result["average_newton_iterations"] = r.average_newton_iterations;
            result["convergence_failures"] = r.convergence_failures;
            result["timestep_reductions"] = r.timestep_reductions;
            result["peak_memory_bytes"] = r.peak_memory_bytes;

            // Convert data to dict of signal_name -> values
            py::dict signals;
            for (size_t i = 0; i < r.signal_names.size(); ++i) {
                std::vector<Real> values;
                values.reserve(r.data.size());
                for (const auto& state : r.data) {
                    if (static_cast<Index>(i) < state.size()) {
                        values.push_back(state(i));
                    }
                }
                signals[py::str(r.signal_names[i])] = values;
            }
            result["signals"] = signals;
            return result;
        });

    // --- Schematic Position ---
    py::class_<SchematicPosition>(m, "SchematicPosition",
        "Schematic position for GUI layout persistence.\n\n"
        "Stores the visual position and orientation of a component in a schematic\n"
        "editor. This information can be exported/imported via JSON netlists.\n\n"
        "Example:\n"
        "    circuit.add_resistor('R1', 'a', 'b', 1000.0)\n"
        "    pos = pulsim.SchematicPosition(x=100.0, y=50.0, orientation=90)\n"
        "    circuit.set_position('R1', pos)\n\n"
        "    # Export to JSON (positions included)\n"
        "    json_str = pulsim.circuit_to_json(circuit, include_positions=True)\n\n"
        "Attributes:\n"
        "    x: X coordinate in schematic units\n"
        "    y: Y coordinate in schematic units\n"
        "    orientation: Rotation in degrees (0, 90, 180, or 270)\n"
        "    mirrored: True if horizontally mirrored")
        .def(py::init<>())
        .def(py::init([](double x, double y, int orientation, bool mirrored) {
            SchematicPosition pos;
            pos.x = x;
            pos.y = y;
            pos.orientation = orientation;
            pos.mirrored = mirrored;
            return pos;
        }), py::arg("x") = 0.0, py::arg("y") = 0.0,
             py::arg("orientation") = 0, py::arg("mirrored") = false)
        .def_readwrite("x", &SchematicPosition::x)
        .def_readwrite("y", &SchematicPosition::y)
        .def_readwrite("orientation", &SchematicPosition::orientation)
        .def_readwrite("mirrored", &SchematicPosition::mirrored)
        .def("to_dict", [](const SchematicPosition& p) {
            py::dict result;
            result["x"] = p.x;
            result["y"] = p.y;
            result["orientation"] = p.orientation;
            result["mirrored"] = p.mirrored;
            return result;
        }, "Convert to dictionary for JSON serialization.");

    // --- Diagnostic ---
    py::class_<Diagnostic>(m, "Diagnostic",
        "A single diagnostic message with location information.\n\n"
        "Contains all information needed to display the diagnostic in a GUI,\n"
        "including the specific location (component, node, parameter) and\n"
        "related components that should be highlighted.\n\n"
        "Attributes:\n"
        "    severity: DiagnosticSeverity (Error, Warning, Info)\n"
        "    code: DiagnosticCode for programmatic handling\n"
        "    message: Human-readable error message\n"
        "    component_name: Component causing the issue (empty if circuit-level)\n"
        "    node_name: Node involved (empty if component-level)\n"
        "    parameter_name: Parameter name (empty if not parameter-related)\n"
        "    related_components: Components to highlight in GUI")
        .def(py::init<>())
        .def_readonly("severity", &Diagnostic::severity)
        .def_readonly("code", &Diagnostic::code)
        .def_readonly("message", &Diagnostic::message)
        .def_readonly("component_name", &Diagnostic::component_name)
        .def_readonly("node_name", &Diagnostic::node_name)
        .def_readonly("parameter_name", &Diagnostic::parameter_name)
        .def_readonly("related_components", &Diagnostic::related_components)
        .def("to_dict", [](const Diagnostic& d) {
            py::dict result;
            result["severity"] = static_cast<int>(d.severity);
            result["code"] = static_cast<int>(d.code);
            result["message"] = d.message;
            result["component_name"] = d.component_name;
            result["node_name"] = d.node_name;
            result["parameter_name"] = d.parameter_name;
            result["related_components"] = d.related_components;
            return result;
        }, "Convert to dictionary for JSON serialization.");

    // --- Validation Result ---
    py::class_<ValidationResult>(m, "ValidationResult",
        "Result of circuit validation containing all diagnostics.\n\n"
        "Use is_valid to check if simulation can proceed. Even if valid,\n"
        "check warnings for potential issues.\n\n"
        "Example:\n"
        "    result = pulsim.validate_circuit(circuit)\n\n"
        "    if result.is_valid:\n"
        "        if result.has_warnings():\n"
        "            for w in result.warnings():\n"
        "                print(f'Warning: {w.message}')\n"
        "        # Can proceed with simulation\n"
        "    else:\n"
        "        for e in result.errors():\n"
        "            highlight_component(e.component_name)\n"
        "            print(f'Error: {e.message}')\n\n"
        "Attributes:\n"
        "    is_valid: True if no errors (warnings OK)\n"
        "    diagnostics: All diagnostic messages")
        .def(py::init<>())
        .def_readonly("is_valid", &ValidationResult::is_valid)
        .def_readonly("diagnostics", &ValidationResult::diagnostics)
        .def("has_errors", &ValidationResult::has_errors,
            "True if any Error-severity diagnostics.")
        .def("has_warnings", &ValidationResult::has_warnings,
            "True if any Warning-severity diagnostics.")
        .def("errors", &ValidationResult::errors,
            "Get only Error-severity diagnostics.")
        .def("warnings", &ValidationResult::warnings,
            "Get only Warning-severity diagnostics.")
        .def("infos", &ValidationResult::infos,
            "Get only Info-severity diagnostics.");

    // --- Circuit ---
    py::class_<Circuit>(m, "Circuit")
        .def(py::init<>())
        .def("add_resistor", &Circuit::add_resistor,
             py::arg("name"), py::arg("n1"), py::arg("n2"), py::arg("resistance"))
        .def("add_capacitor", &Circuit::add_capacitor,
             py::arg("name"), py::arg("n1"), py::arg("n2"), py::arg("capacitance"), py::arg("ic") = 0.0)
        .def("add_inductor", &Circuit::add_inductor,
             py::arg("name"), py::arg("n1"), py::arg("n2"), py::arg("inductance"), py::arg("ic") = 0.0)
        .def("add_voltage_source", [](Circuit& c, const std::string& name, const std::string& npos,
                                       const std::string& nneg, Real value) {
            c.add_voltage_source(name, npos, nneg, DCWaveform{value});
        }, py::arg("name"), py::arg("npos"), py::arg("nneg"), py::arg("value"))
        .def("add_voltage_source", [](Circuit& c, const std::string& name, const std::string& npos,
                                       const std::string& nneg, const PWMWaveform& pwm) {
            c.add_voltage_source(name, npos, nneg, pwm);
        }, py::arg("name"), py::arg("npos"), py::arg("nneg"), py::arg("pwm"))
        .def("add_current_source", [](Circuit& c, const std::string& name, const std::string& npos,
                                       const std::string& nneg, Real value) {
            c.add_current_source(name, npos, nneg, DCWaveform{value});
        }, py::arg("name"), py::arg("npos"), py::arg("nneg"), py::arg("value"))
        .def("add_diode", &Circuit::add_diode,
             py::arg("name"), py::arg("anode"), py::arg("cathode"), py::arg("params") = DiodeParams{})
        .def("add_switch", &Circuit::add_switch,
             py::arg("name"), py::arg("n1"), py::arg("n2"),
             py::arg("ctrl_pos"), py::arg("ctrl_neg"), py::arg("params") = SwitchParams{})
        .def("add_mosfet", &Circuit::add_mosfet,
             py::arg("name"), py::arg("drain"), py::arg("gate"), py::arg("source"),
             py::arg("params") = MOSFETParams{})
        .def("add_transformer", &Circuit::add_transformer,
             py::arg("name"), py::arg("p1"), py::arg("p2"), py::arg("s1"), py::arg("s2"),
             py::arg("params") = TransformerParams{})
        .def("node_count", &Circuit::node_count)
        .def("branch_count", &Circuit::branch_count)
        .def("total_variables", &Circuit::total_variables)
        .def("node_names", &Circuit::node_names)
        .def("validate", [](const Circuit& c) {
            std::string error;
            bool valid = c.validate(error);
            return py::make_tuple(valid, error);
        })
        .def("validate_detailed", [](const Circuit& c) {
            return validate_circuit(c);
        })
        // Schematic position methods
        .def("set_position", &Circuit::set_position,
             py::arg("component_name"), py::arg("position"))
        .def("get_position", &Circuit::get_position,
             py::arg("component_name"))
        .def("has_position", &Circuit::has_position,
             py::arg("component_name"))
        .def("all_positions", &Circuit::all_positions)
        .def("set_all_positions", &Circuit::set_all_positions,
             py::arg("positions"))
        .def("clear_positions", &Circuit::clear_positions);

    // --- Parser ---
    m.def("parse_netlist_file", [](const std::string& path) {
        auto result = NetlistParser::parse_file(path);
        if (!result) {
            throw std::runtime_error(result.error().to_string());
        }
        return result.value();
    }, py::arg("path"), "Parse a circuit from a JSON netlist file");

    m.def("parse_netlist_string", [](const std::string& content) {
        auto result = NetlistParser::parse_string(content);
        if (!result) {
            throw std::runtime_error(result.error().to_string());
        }
        return result.value();
    }, py::arg("content"), "Parse a circuit from a JSON netlist string");

    m.def("circuit_to_json", [](const Circuit& circuit, bool include_positions) {
        return NetlistParser::to_json(circuit, include_positions);
    }, py::arg("circuit"), py::arg("include_positions") = true,
       "Export a circuit to JSON string");

    // --- Simulator ---
    py::class_<PowerLosses>(m, "PowerLosses")
        .def_readonly("conduction_loss", &PowerLosses::conduction_loss)
        .def_readonly("turn_on_loss", &PowerLosses::turn_on_loss)
        .def_readonly("turn_off_loss", &PowerLosses::turn_off_loss)
        .def_readonly("reverse_recovery_loss", &PowerLosses::reverse_recovery_loss)
        .def("switching_loss", &PowerLosses::switching_loss)
        .def("total_loss", &PowerLosses::total_loss);

    py::class_<SwitchEvent>(m, "SwitchEvent")
        .def_readonly("switch_name", &SwitchEvent::switch_name)
        .def_readonly("time", &SwitchEvent::time)
        .def_readonly("new_state", &SwitchEvent::new_state)
        .def_readonly("voltage", &SwitchEvent::voltage)
        .def_readonly("current", &SwitchEvent::current);

    py::class_<Simulator>(m, "Simulator")
        .def(py::init<const Circuit&, const SimulationOptions&>(),
             py::arg("circuit"), py::arg("options") = SimulationOptions{})
        .def("dc_operating_point", [](Simulator& sim) {
            auto result = sim.dc_operating_point();
            return py::make_tuple(static_cast<int>(result.status), result.iterations);
        })
        .def("run_transient", [](Simulator& sim) {
            return sim.run_transient();
        })
        .def("run_transient_with_callback", [](Simulator& sim, py::function callback) {
            return sim.run_transient([&callback](Real time, const Vector& state) {
                callback(time, state);
            });
        }, py::arg("callback"))
        .def("run_transient_with_progress", [](Simulator& sim,
                                               py::object callback,
                                               py::object event_callback,
                                               SimulationController* control,
                                               py::function progress_callback,
                                               double min_interval_ms,
                                               int min_steps) {
            // Wrap Python callbacks
            SimulationCallback sim_callback = nullptr;
            if (!callback.is_none()) {
                sim_callback = [&callback](Real time, const Vector& state) {
                    callback(time, state);
                };
            }

            EventCallback evt_callback = nullptr;
            if (!event_callback.is_none()) {
                evt_callback = [&event_callback](const SwitchEvent& event) {
                    event_callback(event);
                };
            }

            ProgressCallbackConfig config;
            config.callback = [&progress_callback](const SimulationProgress& p) {
                progress_callback(p);
            };
            config.min_interval_ms = min_interval_ms;
            config.min_steps = min_steps;

            return sim.run_transient_with_progress(sim_callback, evt_callback, control, config);
        }, py::arg("callback") = py::none(),
           py::arg("event_callback") = py::none(),
           py::arg("control") = nullptr,
           py::arg("progress_callback") = py::none(),
           py::arg("min_interval_ms") = 100.0,
           py::arg("min_steps") = 100)
        .def("power_losses", &Simulator::power_losses)
        .def("set_options", &Simulator::set_options);

    // Convenience function
    m.def("simulate", [](const Circuit& circuit, const SimulationOptions& options) {
        return simulate(circuit, options);
    }, py::arg("circuit"), py::arg("options") = SimulationOptions{},
       "Run a transient simulation on the circuit");

    // --- Component Metadata ---
    py::class_<ParameterMetadata>(m, "ParameterMetadata",
        "Metadata for a single component parameter.\n\n"
        "Contains all information needed to create a property editor field,\n"
        "including display name, validation constraints, and units.\n\n"
        "Attributes:\n"
        "    name: Internal name (e.g., 'resistance')\n"
        "    display_name: GUI display name (e.g., 'Resistance')\n"
        "    description: Help text / tooltip\n"
        "    type: ParameterType (Real, Integer, Boolean, Enum, String)\n"
        "    default_value: Default value (None if none)\n"
        "    min_value: Minimum allowed (None if unbounded)\n"
        "    max_value: Maximum allowed (None if unbounded)\n"
        "    unit: Unit string (e.g., 'ohm', 'F', 'H')\n"
        "    enum_values: Valid choices for Enum type\n"
        "    required: True if parameter must be provided")
        .def(py::init<>())
        .def_readonly("name", &ParameterMetadata::name)
        .def_readonly("display_name", &ParameterMetadata::display_name)
        .def_readonly("description", &ParameterMetadata::description)
        .def_readonly("type", &ParameterMetadata::type)
        .def_readonly("default_value", &ParameterMetadata::default_value)
        .def_readonly("min_value", &ParameterMetadata::min_value)
        .def_readonly("max_value", &ParameterMetadata::max_value)
        .def_readonly("unit", &ParameterMetadata::unit)
        .def_readonly("enum_values", &ParameterMetadata::enum_values)
        .def_readonly("required", &ParameterMetadata::required)
        .def("to_dict", [](const ParameterMetadata& p) {
            py::dict result;
            result["name"] = p.name;
            result["display_name"] = p.display_name;
            result["description"] = p.description;
            result["type"] = static_cast<int>(p.type);
            result["default_value"] = p.default_value;
            result["min_value"] = p.min_value;
            result["max_value"] = p.max_value;
            result["unit"] = p.unit;
            result["enum_values"] = p.enum_values;
            result["required"] = p.required;
            return result;
        }, "Convert to dictionary for JSON serialization.");

    py::class_<PinMetadata>(m, "PinMetadata",
        "Metadata for a component pin/terminal.\n\n"
        "Describes a connection point on a component for GUI display\n"
        "and netlist generation.\n\n"
        "Attributes:\n"
        "    name: Pin name (e.g., 'anode', 'drain', 'gate')\n"
        "    description: Description (e.g., 'Positive terminal')")
        .def(py::init<>())
        .def_readonly("name", &PinMetadata::name)
        .def_readonly("description", &PinMetadata::description)
        .def("to_dict", [](const PinMetadata& p) {
            py::dict result;
            result["name"] = p.name;
            result["description"] = p.description;
            return result;
        }, "Convert to dictionary for JSON serialization.");

    py::class_<ComponentMetadata>(m, "ComponentMetadata",
        "Complete metadata for a component type.\n\n"
        "Contains all information needed to display the component in a palette,\n"
        "render the component symbol, create property editor fields, and\n"
        "validate component parameters.\n\n"
        "Attributes:\n"
        "    type: ComponentType enum value\n"
        "    name: Internal name (e.g., 'resistor')\n"
        "    display_name: GUI display name (e.g., 'Resistor')\n"
        "    description: Help text / tooltip\n"
        "    category: Category (e.g., 'Passive', 'Semiconductor', 'Sources')\n"
        "    pins: List of PinMetadata for component terminals\n"
        "    parameters: List of ParameterMetadata for editable parameters\n"
        "    symbol_id: Reference for symbol rendering\n"
        "    has_loss_model: True if power loss calculation supported\n"
        "    has_thermal_model: True if thermal simulation supported")
        .def(py::init<>())
        .def_readonly("type", &ComponentMetadata::type)
        .def_readonly("name", &ComponentMetadata::name)
        .def_readonly("display_name", &ComponentMetadata::display_name)
        .def_readonly("description", &ComponentMetadata::description)
        .def_readonly("category", &ComponentMetadata::category)
        .def_readonly("pins", &ComponentMetadata::pins)
        .def_readonly("parameters", &ComponentMetadata::parameters)
        .def_readonly("symbol_id", &ComponentMetadata::symbol_id)
        .def_readonly("has_loss_model", &ComponentMetadata::has_loss_model)
        .def_readonly("has_thermal_model", &ComponentMetadata::has_thermal_model)
        .def("to_dict", [](const ComponentMetadata& c) {
            py::dict result;
            result["type"] = static_cast<int>(c.type);
            result["name"] = c.name;
            result["display_name"] = c.display_name;
            result["description"] = c.description;
            result["category"] = c.category;
            result["symbol_id"] = c.symbol_id;
            result["has_loss_model"] = c.has_loss_model;
            result["has_thermal_model"] = c.has_thermal_model;

            py::list pins;
            for (const auto& pin : c.pins) {
                py::dict p;
                p["name"] = pin.name;
                p["description"] = pin.description;
                pins.append(p);
            }
            result["pins"] = pins;

            py::list params;
            for (const auto& param : c.parameters) {
                py::dict p;
                p["name"] = param.name;
                p["display_name"] = param.display_name;
                p["description"] = param.description;
                p["type"] = static_cast<int>(param.type);
                p["unit"] = param.unit;
                p["required"] = param.required;
                params.append(p);
            }
            result["parameters"] = params;

            return result;
        });

    py::class_<ComponentRegistry>(m, "ComponentRegistry",
        "Singleton registry providing metadata for all component types.\n\n"
        "Use this to build component palettes organized by category and\n"
        "generate property editors with correct field types and validation.\n\n"
        "Example:\n"
        "    registry = pulsim.ComponentRegistry.instance()\n\n"
        "    # Build component palette by category\n"
        "    for category in registry.all_categories():\n"
        "        for comp_type in registry.types_in_category(category):\n"
        "            meta = registry.get(comp_type)\n"
        "            add_palette_item(meta.display_name, meta.symbol_id)\n\n"
        "    # Build property editor for resistor\n"
        "    meta = registry.get(pulsim.ComponentType.Resistor)\n"
        "    for param in meta.parameters:\n"
        "        create_field(param.display_name, param.unit)")
        .def_static("instance", &ComponentRegistry::instance, py::return_value_policy::reference,
            "Get the singleton registry instance.")
        .def("get", &ComponentRegistry::get, py::arg("type"),
             py::return_value_policy::reference_internal,
            "Get metadata for a specific component type.")
        .def("all_types", &ComponentRegistry::all_types,
            "Get all registered component types.")
        .def("types_in_category", &ComponentRegistry::types_in_category, py::arg("category"),
            "Get component types in a specific category.")
        .def("all_categories", &ComponentRegistry::all_categories,
            "Get all available categories.");

    // Convenience function for validate_circuit
    m.def("validate_circuit", &validate_circuit, py::arg("circuit"),
          "Validate a circuit and return detailed diagnostics.\n\n"
          "Performs comprehensive validation including:\n"
          "- Ground reference check\n"
          "- Voltage source loop detection\n"
          "- Floating node detection\n"
          "- Parameter validation\n"
          "- Duplicate name check\n\n"
          "Args:\n"
          "    circuit: The Circuit to validate\n\n"
          "Returns:\n"
          "    ValidationResult containing all diagnostics");

    m.def("diagnostic_code_description", &diagnostic_code_description, py::arg("code"),
          "Get human-readable description of a diagnostic code.\n\n"
          "Useful for displaying help text or tooltips in GUI.\n\n"
          "Args:\n"
          "    code: The DiagnosticCode to describe\n\n"
          "Returns:\n"
          "    Human-readable description string");

    // --- Thermal Simulation ---
    py::class_<ThermalRCStage>(m, "ThermalRCStage")
        .def(py::init<>())
        .def_readwrite("rth", &ThermalRCStage::rth)
        .def_readwrite("cth", &ThermalRCStage::cth)
        .def("tau", &ThermalRCStage::tau);

    py::class_<FosterNetwork>(m, "FosterNetwork")
        .def(py::init<>())
        .def_readwrite("stages", &FosterNetwork::stages)
        .def("rth_total", &FosterNetwork::rth_total)
        .def("zth", &FosterNetwork::zth, py::arg("t"));

    py::class_<ThermalModel>(m, "ThermalModel")
        .def(py::init<>())
        .def_readwrite("device_name", &ThermalModel::device_name)
        .def_readwrite("type", &ThermalModel::type)
        .def_readwrite("rth_jc", &ThermalModel::rth_jc)
        .def_readwrite("rth_cs", &ThermalModel::rth_cs)
        .def_readwrite("rth_sa", &ThermalModel::rth_sa)
        .def_readwrite("foster", &ThermalModel::foster)
        .def_readwrite("tj_max", &ThermalModel::tj_max)
        .def_readwrite("tj_warn", &ThermalModel::tj_warn)
        .def("rth_ja", &ThermalModel::rth_ja);

    py::class_<ThermalState>(m, "ThermalState")
        .def(py::init<>())
        .def_readonly("device_name", &ThermalState::device_name)
        .def_readonly("tj", &ThermalState::tj)
        .def_readonly("tc", &ThermalState::tc)
        .def_readonly("ts", &ThermalState::ts)
        .def_readonly("power_in", &ThermalState::power_in)
        .def_readonly("tj_peak", &ThermalState::tj_peak)
        .def_readonly("tj_peak_time", &ThermalState::tj_peak_time);

    py::class_<ThermalSimulator::ThermalWarning>(m, "ThermalWarning")
        .def_readonly("device_name", &ThermalSimulator::ThermalWarning::device_name)
        .def_readonly("temperature", &ThermalSimulator::ThermalWarning::temperature)
        .def_readonly("time", &ThermalSimulator::ThermalWarning::time)
        .def_readonly("is_failure", &ThermalSimulator::ThermalWarning::is_failure);

    py::class_<ThermalSimulator>(m, "ThermalSimulator")
        .def(py::init<>())
        .def("add_model", &ThermalSimulator::add_model, py::arg("model"))
        .def("set_ambient", &ThermalSimulator::set_ambient, py::arg("t_amb"))
        .def("ambient", &ThermalSimulator::ambient)
        .def("initialize", &ThermalSimulator::initialize)
        .def("step", &ThermalSimulator::step, py::arg("dt"), py::arg("device_powers"))
        .def("junction_temp", &ThermalSimulator::junction_temp, py::arg("device_name"))
        .def("states", &ThermalSimulator::states)
        .def("warnings", &ThermalSimulator::warnings)
        .def("adjust_rds_on", &ThermalSimulator::adjust_rds_on,
             py::arg("rds_on_25c"), py::arg("tj"), py::arg("tc") = 0.004)
        .def("adjust_vth", &ThermalSimulator::adjust_vth,
             py::arg("vth_25c"), py::arg("tj"), py::arg("tc") = -0.003);

    m.def("create_mosfet_thermal", &create_mosfet_thermal,
          py::arg("name"), py::arg("rth_jc"), py::arg("rth_cs") = 0.5, py::arg("rth_sa") = 1.0,
          "Create a typical MOSFET thermal model with 4-stage Foster network");

    m.def("fit_foster_network", &fit_foster_network,
          py::arg("zth_curve"), py::arg("num_stages") = 4,
          "Fit a Foster network from Zth curve datasheet points");

    // --- Device Library ---
    auto devices_mod = m.def_submodule("devices", "Pre-defined device parameter library");

    // Diodes
    devices_mod.def("diode_1N4007", &devices::diode_1N4007,
        "General purpose rectifier diode 1N4007 (1000V, 1A)");
    devices_mod.def("diode_1N4148", &devices::diode_1N4148,
        "Small signal fast switching diode 1N4148 (100V)");
    devices_mod.def("diode_1N5819", &devices::diode_1N5819,
        "Schottky diode 1N5819 (40V, low forward voltage)");
    devices_mod.def("diode_MUR860", &devices::diode_MUR860,
        "Fast recovery diode MUR860 (600V, 8A, 50ns)");
    devices_mod.def("diode_C3D10065A", &devices::diode_C3D10065A,
        "SiC Schottky diode C3D10065A (650V, zero recovery)");

    // MOSFETs
    devices_mod.def("mosfet_IRF540N", &devices::mosfet_IRF540N,
        "N-channel MOSFET IRF540N (100V, 33A, 44mOhm)");
    devices_mod.def("mosfet_IRFZ44N", &devices::mosfet_IRFZ44N,
        "N-channel MOSFET IRFZ44N (55V, 49A, 17.5mOhm)");
    devices_mod.def("mosfet_IRF9540", &devices::mosfet_IRF9540,
        "P-channel MOSFET IRF9540 (-100V, -23A)");
    devices_mod.def("mosfet_BSC0902NS", &devices::mosfet_BSC0902NS,
        "High-efficiency MOSFET BSC0902NS (30V, 2.1mOhm)");
    devices_mod.def("mosfet_EPC2001C", &devices::mosfet_EPC2001C,
        "GaN FET EPC2001C (100V, 4mOhm, ultra-fast)");

    // IGBTs
    devices_mod.def("igbt_IRG4PC40UD", &devices::igbt_IRG4PC40UD,
        "General purpose IGBT IRG4PC40UD (600V, 40A)");
    devices_mod.def("igbt_IRG4BC30KD", &devices::igbt_IRG4BC30KD,
        "High-speed IGBT IRG4BC30KD (600V, 30A)");
    devices_mod.def("igbt_IKW40N120H3", &devices::igbt_IKW40N120H3,
        "High-voltage IGBT IKW40N120H3 (1200V, 40A)");

    // Switches
    devices_mod.def("switch_ideal", &devices::switch_ideal,
        "Ideal switch (1uOhm on, 1TOhm off)");
    devices_mod.def("switch_relay", &devices::switch_relay,
        "Mechanical relay model (100mOhm contact)");
    devices_mod.def("switch_ssr", &devices::switch_ssr,
        "Solid-state relay model (20mOhm)");

    // Version info
    m.attr("__version__") = "0.1.0";
}
