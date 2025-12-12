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
#include "pulsim/v1/control.hpp"

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
    // Nonlinear Devices
    // =========================================================================

    py::class_<IdealDiode>(v2, "IdealDiode", "Ideal diode with on/off conductance")
        .def(py::init<Real, Real, std::string>(),
             py::arg("g_on") = 1e3, py::arg("g_off") = 1e-9, py::arg("name") = "")
        .def("is_conducting", &IdealDiode::is_conducting)
        .def("name", &IdealDiode::name);

    py::class_<IdealSwitch>(v2, "IdealSwitch", "Controllable ideal switch")
        .def(py::init<Real, Real, bool, std::string>(),
             py::arg("g_on") = 1e6, py::arg("g_off") = 1e-12,
             py::arg("closed") = false, py::arg("name") = "")
        .def("close", &IdealSwitch::close)
        .def("open", &IdealSwitch::open)
        .def("set_state", &IdealSwitch::set_state)
        .def("is_closed", &IdealSwitch::is_closed)
        .def("name", &IdealSwitch::name);

    // MOSFET parameters
    py::class_<MOSFET::Params>(v2, "MOSFETParams", "MOSFET Level 1 parameters")
        .def(py::init<>())
        .def_readwrite("vth", &MOSFET::Params::vth, "Threshold voltage (V)")
        .def_readwrite("kp", &MOSFET::Params::kp, "Transconductance (A/V^2)")
        .def_readwrite("lambda_", &MOSFET::Params::lambda, "Channel-length modulation (1/V)")
        .def_readwrite("g_off", &MOSFET::Params::g_off, "Off-state conductance")
        .def_readwrite("is_nmos", &MOSFET::Params::is_nmos, "True for NMOS, False for PMOS");

    py::class_<MOSFET>(v2, "MOSFET", "MOSFET Level 1 (Shichman-Hodges) model")
        .def(py::init<std::string>(), py::arg("name") = "")
        .def(py::init<MOSFET::Params, std::string>(),
             py::arg("params"), py::arg("name") = "")
        .def(py::init<Real, Real, bool, std::string>(),
             py::arg("vth"), py::arg("kp"), py::arg("is_nmos") = true, py::arg("name") = "")
        .def("params", &MOSFET::params, py::return_value_policy::reference_internal)
        .def("name", &MOSFET::name);

    // IGBT parameters
    py::class_<IGBT::Params>(v2, "IGBTParams", "IGBT parameters")
        .def(py::init<>())
        .def_readwrite("vth", &IGBT::Params::vth, "Gate threshold voltage (V)")
        .def_readwrite("g_on", &IGBT::Params::g_on, "On-state conductance (S)")
        .def_readwrite("g_off", &IGBT::Params::g_off, "Off-state conductance (S)")
        .def_readwrite("v_ce_sat", &IGBT::Params::v_ce_sat, "Collector-emitter saturation (V)");

    py::class_<IGBT>(v2, "IGBT", "Simplified IGBT power device model")
        .def(py::init<std::string>(), py::arg("name") = "")
        .def(py::init<IGBT::Params, std::string>(),
             py::arg("params"), py::arg("name") = "")
        .def(py::init<Real, Real, std::string>(),
             py::arg("vth"), py::arg("g_on") = 1e4, py::arg("name") = "")
        .def("is_conducting", &IGBT::is_conducting)
        .def("params", &IGBT::params, py::return_value_policy::reference_internal)
        .def("name", &IGBT::name);

    // =========================================================================
    // Time-Varying Sources
    // =========================================================================

    py::class_<PWMParams>(v2, "PWMParams", "PWM voltage source parameters")
        .def(py::init<>())
        .def_readwrite("v_high", &PWMParams::v_high, "High voltage level (V)")
        .def_readwrite("v_low", &PWMParams::v_low, "Low voltage level (V)")
        .def_readwrite("frequency", &PWMParams::frequency, "Switching frequency (Hz)")
        .def_readwrite("duty", &PWMParams::duty, "Duty cycle (0-1)")
        .def_readwrite("phase", &PWMParams::phase, "Initial phase (rad)")
        .def_readwrite("dead_time", &PWMParams::dead_time, "Dead time (s)")
        .def_readwrite("rise_time", &PWMParams::rise_time, "Rise time (s)")
        .def_readwrite("fall_time", &PWMParams::fall_time, "Fall time (s)");

    py::class_<SineParams>(v2, "SineParams", "Sine voltage source parameters")
        .def(py::init<>())
        .def_readwrite("amplitude", &SineParams::amplitude, "Peak amplitude (V)")
        .def_readwrite("offset", &SineParams::offset, "DC offset (V)")
        .def_readwrite("frequency", &SineParams::frequency, "Frequency (Hz)")
        .def_readwrite("phase", &SineParams::phase, "Initial phase (rad)");

    py::class_<RampParams>(v2, "RampParams", "Ramp/triangle generator parameters")
        .def(py::init<>())
        .def_readwrite("v_min", &RampParams::v_min, "Minimum voltage")
        .def_readwrite("v_max", &RampParams::v_max, "Maximum voltage")
        .def_readwrite("frequency", &RampParams::frequency, "Frequency (Hz)")
        .def_readwrite("phase", &RampParams::phase, "Initial phase (rad)")
        .def_readwrite("triangle", &RampParams::triangle, "Triangle (true) or sawtooth (false)");

    py::class_<PulseParams>(v2, "PulseParams", "Pulse voltage source parameters")
        .def(py::init<>())
        .def_readwrite("v_initial", &PulseParams::v_initial, "Initial voltage")
        .def_readwrite("v_pulse", &PulseParams::v_pulse, "Pulse voltage")
        .def_readwrite("t_delay", &PulseParams::t_delay, "Delay before pulse (s)")
        .def_readwrite("t_rise", &PulseParams::t_rise, "Rise time (s)")
        .def_readwrite("t_fall", &PulseParams::t_fall, "Fall time (s)")
        .def_readwrite("t_width", &PulseParams::t_width, "Pulse width (s)")
        .def_readwrite("period", &PulseParams::period, "Period (0 = single pulse) (s)");

    py::class_<PWMVoltageSource>(v2, "PWMVoltageSource", "PWM voltage source")
        .def(py::init<const PWMParams&, std::string>(),
             py::arg("params"), py::arg("name") = "")
        .def(py::init<Real, Real, Real, Real, std::string>(),
             py::arg("v_high"), py::arg("v_low"), py::arg("frequency"), py::arg("duty"),
             py::arg("name") = "")
        .def("params", py::overload_cast<>(&PWMVoltageSource::params, py::const_),
             py::return_value_policy::reference_internal)
        .def("frequency", &PWMVoltageSource::frequency)
        .def("period", &PWMVoltageSource::period)
        .def("set_duty", &PWMVoltageSource::set_duty, py::arg("duty"))
        .def("set_duty_callback", &PWMVoltageSource::set_duty_callback, py::arg("callback"))
        .def("clear_duty_callback", &PWMVoltageSource::clear_duty_callback)
        .def("duty_at", &PWMVoltageSource::duty_at, py::arg("t"))
        .def("voltage_at", &PWMVoltageSource::voltage_at, py::arg("t"))
        .def("state_at", &PWMVoltageSource::state_at, py::arg("t"));

    py::class_<SineVoltageSource>(v2, "SineVoltageSource", "Sinusoidal voltage source")
        .def(py::init<const SineParams&, std::string>(),
             py::arg("params"), py::arg("name") = "")
        .def(py::init<Real, Real, Real, std::string>(),
             py::arg("amplitude"), py::arg("frequency"), py::arg("offset") = 0.0,
             py::arg("name") = "")
        .def("params", &SineVoltageSource::params,
             py::return_value_policy::reference_internal)
        .def("voltage_at", &SineVoltageSource::voltage_at, py::arg("t"));

    py::class_<PulseVoltageSource>(v2, "PulseVoltageSource", "Pulse voltage source")
        .def(py::init<const PulseParams&, std::string>(),
             py::arg("params"), py::arg("name") = "")
        .def("params", &PulseVoltageSource::params,
             py::return_value_policy::reference_internal)
        .def("voltage_at", &PulseVoltageSource::voltage_at, py::arg("t"));

    py::class_<RampGenerator>(v2, "RampGenerator", "Ramp/triangle waveform generator")
        .def(py::init<const RampParams&>(),
             py::arg("params") = RampParams{})
        .def(py::init<Real, Real, Real, bool>(),
             py::arg("frequency"), py::arg("v_min") = 0.0, py::arg("v_max") = 1.0,
             py::arg("triangle") = false)
        .def("params", py::overload_cast<>(&RampGenerator::params, py::const_),
             py::return_value_policy::reference_internal)
        .def("frequency", &RampGenerator::frequency)
        .def("period", &RampGenerator::period)
        .def("value_at", &RampGenerator::value_at, py::arg("t"));

    // =========================================================================
    // Control Blocks
    // =========================================================================

    py::class_<PIController>(v2, "PIController", "PI controller with anti-windup")
        .def(py::init<Real, Real, Real, Real>(),
             py::arg("Kp"), py::arg("Ki"),
             py::arg("output_min") = 0.0, py::arg("output_max") = 1.0)
        .def("Kp", &PIController::Kp)
        .def("Ki", &PIController::Ki)
        .def("output_min", &PIController::output_min)
        .def("output_max", &PIController::output_max)
        .def("set_Kp", &PIController::set_Kp, py::arg("kp"))
        .def("set_Ki", &PIController::set_Ki, py::arg("ki"))
        .def("set_output_limits", &PIController::set_output_limits,
             py::arg("min"), py::arg("max"))
        .def("integral", &PIController::integral)
        .def("last_output", &PIController::last_output)
        .def("update", py::overload_cast<Real, Real>(&PIController::update),
             py::arg("error"), py::arg("t"))
        .def("update", py::overload_cast<Real, Real, Real>(&PIController::update),
             py::arg("reference"), py::arg("feedback"), py::arg("t"))
        .def("reset", &PIController::reset)
        .def("set_integral", &PIController::set_integral, py::arg("value"));

    py::class_<PIDController>(v2, "PIDController", "PID controller with anti-windup")
        .def(py::init<Real, Real, Real, Real, Real, Real>(),
             py::arg("Kp"), py::arg("Ki"), py::arg("Kd"),
             py::arg("output_min") = 0.0, py::arg("output_max") = 1.0,
             py::arg("derivative_filter") = 0.1)
        .def("Kp", &PIDController::Kp)
        .def("Ki", &PIDController::Ki)
        .def("Kd", &PIDController::Kd)
        .def("set_gains", &PIDController::set_gains,
             py::arg("kp"), py::arg("ki"), py::arg("kd"))
        .def("set_output_limits", &PIDController::set_output_limits,
             py::arg("min"), py::arg("max"))
        .def("set_derivative_filter", &PIDController::set_derivative_filter, py::arg("alpha"))
        .def("integral", &PIDController::integral)
        .def("last_output", &PIDController::last_output)
        .def("update", py::overload_cast<Real, Real>(&PIDController::update),
             py::arg("error"), py::arg("t"))
        .def("update", py::overload_cast<Real, Real, Real>(&PIDController::update),
             py::arg("reference"), py::arg("feedback"), py::arg("t"))
        .def("reset", &PIDController::reset);

    py::class_<Comparator>(v2, "Comparator", "Comparator with optional hysteresis")
        .def(py::init<Real>(), py::arg("hysteresis") = 0.0)
        .def("hysteresis", &Comparator::hysteresis)
        .def("set_hysteresis", &Comparator::set_hysteresis, py::arg("h"))
        .def("compare", &Comparator::compare,
             py::arg("input"), py::arg("reference"))
        .def("output", &Comparator::output,
             py::arg("input"), py::arg("reference"),
             py::arg("v_high") = 1.0, py::arg("v_low") = 0.0)
        .def("state", &Comparator::state)
        .def("reset", &Comparator::reset);

    py::class_<SampleHold>(v2, "SampleHold", "Sample-and-hold block")
        .def(py::init<Real>(), py::arg("sample_period"))
        .def("period", &SampleHold::period)
        .def("frequency", &SampleHold::frequency)
        .def("set_period", &SampleHold::set_period, py::arg("T"))
        .def("value", &SampleHold::value)
        .def("last_sample_time", &SampleHold::last_sample_time)
        .def("update", &SampleHold::update, py::arg("input"), py::arg("t"))
        .def("sample_now", &SampleHold::sample_now, py::arg("input"), py::arg("t"))
        .def("reset", &SampleHold::reset);

    py::class_<RateLimiter>(v2, "RateLimiter", "Limits rate of change of a signal")
        .def(py::init<Real, Real>(),
             py::arg("rising_rate"), py::arg("falling_rate"))
        .def(py::init<Real>(), py::arg("rate"))
        .def("rising_rate", &RateLimiter::rising_rate)
        .def("falling_rate", &RateLimiter::falling_rate)
        .def("set_rates", &RateLimiter::set_rates,
             py::arg("rising"), py::arg("falling"))
        .def("value", &RateLimiter::value)
        .def("update", &RateLimiter::update, py::arg("input"), py::arg("t"))
        .def("reset", &RateLimiter::reset, py::arg("initial") = 0.0);

    py::class_<MovingAverageFilter>(v2, "MovingAverageFilter", "Exponential moving average filter")
        .def(py::init<Real>(), py::arg("time_constant"))
        .def("time_constant", &MovingAverageFilter::time_constant)
        .def("set_time_constant", &MovingAverageFilter::set_time_constant, py::arg("tau"))
        .def("value", &MovingAverageFilter::value)
        .def("update", &MovingAverageFilter::update, py::arg("input"), py::arg("t"))
        .def("reset", &MovingAverageFilter::reset, py::arg("initial") = 0.0);

    py::class_<HysteresisController>(v2, "HysteresisController", "Bang-bang controller with hysteresis")
        .def(py::init<Real, Real, Real, Real>(),
             py::arg("setpoint"), py::arg("band"),
             py::arg("output_high") = 1.0, py::arg("output_low") = 0.0)
        .def("setpoint", &HysteresisController::setpoint)
        .def("band", &HysteresisController::band)
        .def("set_setpoint", &HysteresisController::set_setpoint, py::arg("sp"))
        .def("set_band", &HysteresisController::set_band, py::arg("b"))
        .def("state", &HysteresisController::state)
        .def("output", &HysteresisController::output)
        .def("update", &HysteresisController::update, py::arg("feedback"))
        .def("reset", &HysteresisController::reset);

    py::class_<LookupTable1D>(v2, "LookupTable1D", "1D lookup table with linear interpolation")
        .def(py::init<>())
        .def(py::init<std::vector<Real>, std::vector<Real>>(),
             py::arg("x"), py::arg("y"))
        .def("x_data", &LookupTable1D::x_data)
        .def("y_data", &LookupTable1D::y_data)
        .def("size", &LookupTable1D::size)
        .def("empty", &LookupTable1D::empty)
        .def("__call__", &LookupTable1D::operator(), py::arg("x"))
        .def("interpolate", &LookupTable1D::interpolate, py::arg("x"));

    // =========================================================================
    // Runtime Circuit Builder (Phase 3)
    // =========================================================================

    py::class_<Circuit>(v2, "Circuit", "Runtime circuit builder for simulation")
        .def(py::init<>())
        // Node management
        .def("add_node", &Circuit::add_node, py::arg("name"),
             "Add a named node and return its index")
        .def("get_node", &Circuit::get_node, py::arg("name"),
             "Get node index by name (-1 for ground)")
        .def_static("ground", &Circuit::ground,
             "Get ground node index")
        .def("num_nodes", &Circuit::num_nodes,
             "Number of non-ground nodes")
        .def("num_branches", &Circuit::num_branches,
             "Number of branch currents (for VS, inductors)")
        .def("system_size", &Circuit::system_size,
             "Total system size (nodes + branches)")
        .def("node_name", &Circuit::node_name, py::arg("index"),
             "Get node name by index")
        .def("node_names", &Circuit::node_names,
             "Get all node names")
        // Device addition
        .def("add_resistor", &Circuit::add_resistor,
             py::arg("name"), py::arg("n1"), py::arg("n2"), py::arg("R"),
             "Add resistor between nodes n1 and n2")
        .def("add_capacitor", &Circuit::add_capacitor,
             py::arg("name"), py::arg("n1"), py::arg("n2"), py::arg("C"),
             py::arg("ic") = 0.0,
             "Add capacitor between nodes n1 and n2")
        .def("add_inductor", &Circuit::add_inductor,
             py::arg("name"), py::arg("n1"), py::arg("n2"), py::arg("L"),
             py::arg("ic") = 0.0,
             "Add inductor between nodes n1 and n2")
        .def("add_voltage_source", &Circuit::add_voltage_source,
             py::arg("name"), py::arg("npos"), py::arg("nneg"), py::arg("V"),
             "Add voltage source from npos to nneg")
        .def("add_current_source", &Circuit::add_current_source,
             py::arg("name"), py::arg("npos"), py::arg("nneg"), py::arg("I"),
             "Add current source from npos to nneg")
        .def("add_diode", &Circuit::add_diode,
             py::arg("name"), py::arg("anode"), py::arg("cathode"),
             py::arg("g_on") = 1e3, py::arg("g_off") = 1e-9,
             "Add ideal diode from anode to cathode")
        .def("add_switch", &Circuit::add_switch,
             py::arg("name"), py::arg("n1"), py::arg("n2"),
             py::arg("closed") = false, py::arg("g_on") = 1e6, py::arg("g_off") = 1e-12,
             "Add controllable switch between n1 and n2")
        .def("add_mosfet", &Circuit::add_mosfet,
             py::arg("name"), py::arg("gate"), py::arg("drain"), py::arg("source"),
             py::arg("params") = MOSFET::Params{},
             "Add MOSFET with gate, drain, source nodes")
        .def("add_igbt", &Circuit::add_igbt,
             py::arg("name"), py::arg("gate"), py::arg("collector"), py::arg("emitter"),
             py::arg("params") = IGBT::Params{},
             "Add IGBT with gate, collector, emitter nodes")
        // Time-varying sources
        .def("add_pwm_voltage_source",
             py::overload_cast<const std::string&, Index, Index, const PWMParams&>(
                 &Circuit::add_pwm_voltage_source),
             py::arg("name"), py::arg("npos"), py::arg("nneg"), py::arg("params"),
             "Add PWM voltage source from npos to nneg")
        .def("add_pwm_voltage_source",
             py::overload_cast<const std::string&, Index, Index, Real, Real, Real, Real>(
                 &Circuit::add_pwm_voltage_source),
             py::arg("name"), py::arg("npos"), py::arg("nneg"),
             py::arg("v_high"), py::arg("v_low"), py::arg("frequency"), py::arg("duty"),
             "Add PWM voltage source with simple parameters")
        .def("add_sine_voltage_source",
             py::overload_cast<const std::string&, Index, Index, const SineParams&>(
                 &Circuit::add_sine_voltage_source),
             py::arg("name"), py::arg("npos"), py::arg("nneg"), py::arg("params"),
             "Add sinusoidal voltage source")
        .def("add_sine_voltage_source",
             py::overload_cast<const std::string&, Index, Index, Real, Real, Real>(
                 &Circuit::add_sine_voltage_source),
             py::arg("name"), py::arg("npos"), py::arg("nneg"),
             py::arg("amplitude"), py::arg("frequency"), py::arg("offset") = 0.0,
             "Add sinusoidal voltage source with simple parameters")
        .def("add_pulse_voltage_source", &Circuit::add_pulse_voltage_source,
             py::arg("name"), py::arg("npos"), py::arg("nneg"), py::arg("params"),
             "Add pulse voltage source")
        // PWM control
        .def("set_pwm_duty", &Circuit::set_pwm_duty,
             py::arg("name"), py::arg("duty"),
             "Set fixed duty cycle for PWM source")
        .def("set_pwm_duty_callback", &Circuit::set_pwm_duty_callback,
             py::arg("name"), py::arg("callback"),
             "Set duty cycle callback for PWM source")
        .def("clear_pwm_duty_callback", &Circuit::clear_pwm_duty_callback,
             py::arg("name"),
             "Clear duty callback, use fixed duty")
        .def("get_pwm_state", &Circuit::get_pwm_state,
             py::arg("name"),
             "Get PWM state (ON/OFF) at current time")
        // Time management
        .def("set_current_time", &Circuit::set_current_time,
             py::arg("t"),
             "Set current simulation time")
        .def("current_time", &Circuit::current_time,
             "Get current simulation time")
        .def("has_time_varying", &Circuit::has_time_varying,
             "Check if circuit has time-varying sources")
        // State
        .def("num_devices", &Circuit::num_devices,
             "Number of devices in circuit")
        .def("set_switch_state", &Circuit::set_switch_state,
             py::arg("name"), py::arg("closed"),
             "Set switch state by name")
        .def("set_timestep", &Circuit::set_timestep, py::arg("dt"),
             "Set timestep for dynamic elements")
        .def("timestep", &Circuit::timestep,
             "Get current timestep")
        .def("has_nonlinear", &Circuit::has_nonlinear,
             "Check if circuit has nonlinear devices")
        // Matrix assembly
        .def("assemble_dc", [](const Circuit& ckt) {
            SparseMatrix G;
            Vector b;
            ckt.assemble_dc(G, b);
            // Convert sparse to dense for easier Python use
            return std::make_tuple(Eigen::MatrixXd(G), b);
        }, "Assemble G matrix and b vector for DC analysis")
        .def("assemble_jacobian", [](const Circuit& ckt, const Vector& x) {
            SparseMatrix J;
            Vector f;
            ckt.assemble_jacobian(J, f, x);
            return std::make_tuple(Eigen::MatrixXd(J), f);
        }, py::arg("x"), "Assemble Jacobian J and residual f for Newton iteration");

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

    // =========================================================================
    // Convergence History & Monitoring
    // =========================================================================

    py::class_<IterationRecord>(v2, "IterationRecord", "Single Newton iteration record")
        .def(py::init<>())
        .def_readonly("iteration", &IterationRecord::iteration)
        .def_readonly("residual_norm", &IterationRecord::residual_norm)
        .def_readonly("max_voltage_error", &IterationRecord::max_voltage_error)
        .def_readonly("max_current_error", &IterationRecord::max_current_error)
        .def_readonly("step_norm", &IterationRecord::step_norm)
        .def_readonly("damping", &IterationRecord::damping)
        .def_readonly("converged", &IterationRecord::converged);

    py::class_<ConvergenceHistory>(v2, "ConvergenceHistory", "Complete convergence history")
        .def(py::init<>())
        .def("size", &ConvergenceHistory::size)
        .def("empty", &ConvergenceHistory::empty)
        .def("__len__", &ConvergenceHistory::size)
        .def("__getitem__", [](const ConvergenceHistory& h, std::size_t i) {
            if (i >= h.size()) throw py::index_error();
            return h[i];
        })
        .def("last", &ConvergenceHistory::last)
        .def("final_status", &ConvergenceHistory::final_status)
        .def("is_stalling", &ConvergenceHistory::is_stalling,
             py::arg("window") = 5, py::arg("threshold") = 0.9)
        .def("is_diverging", &ConvergenceHistory::is_diverging, py::arg("window") = 3)
        .def("convergence_rate", &ConvergenceHistory::convergence_rate);

    py::class_<VariableConvergence>(v2, "VariableConvergence", "Per-variable convergence status")
        .def(py::init<>())
        .def_readonly("index", &VariableConvergence::index)
        .def_readonly("value", &VariableConvergence::value)
        .def_readonly("delta", &VariableConvergence::delta)
        .def_readonly("tolerance", &VariableConvergence::tolerance)
        .def_readonly("normalized_error", &VariableConvergence::normalized_error)
        .def_readonly("converged", &VariableConvergence::converged)
        .def_readonly("is_voltage", &VariableConvergence::is_voltage);

    py::class_<PerVariableConvergence>(v2, "PerVariableConvergence", "Per-variable convergence tracker")
        .def(py::init<>())
        .def("size", &PerVariableConvergence::size)
        .def("empty", &PerVariableConvergence::empty)
        .def("__len__", &PerVariableConvergence::size)
        .def("__getitem__", [](const PerVariableConvergence& p, std::size_t i) {
            if (i >= p.size()) throw py::index_error();
            return p[i];
        })
        .def("all_converged", &PerVariableConvergence::all_converged)
        .def("worst", &PerVariableConvergence::worst)
        .def("max_error", &PerVariableConvergence::max_error)
        .def("non_converged_count", &PerVariableConvergence::non_converged_count);

    py::class_<NewtonResult>(v2, "NewtonResult", "Newton solver result")
        .def(py::init<>())
        .def_readonly("solution", &NewtonResult::solution)
        .def_readonly("status", &NewtonResult::status)
        .def_readonly("iterations", &NewtonResult::iterations)
        .def_readonly("final_residual", &NewtonResult::final_residual)
        .def_readonly("final_weighted_error", &NewtonResult::final_weighted_error)
        .def_readonly("history", &NewtonResult::history)
        .def_readonly("variable_convergence", &NewtonResult::variable_convergence)
        .def_readonly("error_message", &NewtonResult::error_message)
        .def("success", &NewtonResult::success);

    // =========================================================================
    // Newton Solver Execution (Phase 4)
    // =========================================================================

    // Solve circuit using Newton-Raphson
    v2.def("solve_dc", [](Circuit& circuit, const Vector& x0, const NewtonOptions& opts) {
        // Configure options with circuit info
        NewtonOptions cfg = opts;
        cfg.num_nodes = circuit.num_nodes();
        cfg.num_branches = circuit.num_branches();

        // Create solver
        NewtonRaphsonSolver<SparseLUPolicy> solver(cfg);

        // System function for Newton
        auto system_func = [&circuit](const Vector& x, Vector& f, SparseMatrix& J) {
            circuit.assemble_jacobian(J, f, x);
        };

        return solver.solve(x0, system_func);
    }, py::arg("circuit"), py::arg("x0"), py::arg("options") = NewtonOptions(),
    R"doc(
    Solve circuit DC operating point using Newton-Raphson.

    Args:
        circuit: Circuit object with devices
        x0: Initial guess vector (size = num_nodes + num_branches)
        options: Newton solver options

    Returns:
        NewtonResult with solution and convergence info
    )doc");

    // Convenience function with automatic initial guess
    v2.def("solve_dc", [](Circuit& circuit, const NewtonOptions& opts) {
        // Create zero initial guess
        Vector x0 = Vector::Zero(circuit.system_size());

        // Configure options with circuit info
        NewtonOptions cfg = opts;
        cfg.num_nodes = circuit.num_nodes();
        cfg.num_branches = circuit.num_branches();

        // Create solver
        NewtonRaphsonSolver<SparseLUPolicy> solver(cfg);

        // System function for Newton
        auto system_func = [&circuit](const Vector& x, Vector& f, SparseMatrix& J) {
            circuit.assemble_jacobian(J, f, x);
        };

        return solver.solve(x0, system_func);
    }, py::arg("circuit"), py::arg("options") = NewtonOptions(),
    R"doc(
    Solve circuit DC operating point using Newton-Raphson with zero initial guess.

    Args:
        circuit: Circuit object with devices
        options: Newton solver options

    Returns:
        NewtonResult with solution and convergence info
    )doc");

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
    // DC Analysis Function (Phase 5)
    // =========================================================================

    // High-level DC analysis with automatic strategy selection
    v2.def("dc_operating_point", [](Circuit& circuit, const DCConvergenceConfig& config) {
        // For DC analysis, use very large timestep so inductor 2L/dt -> 0 (short circuit)
        // and capacitor 2C/dt -> 0 (open circuit)
        circuit.set_timestep(1e6);  // 1 million seconds -> effectively DC

        // Create system function
        auto system_func = [&circuit](const Vector& x, Vector& f, SparseMatrix& J) {
            circuit.assemble_jacobian(J, f, x);
        };

        // Initial guess
        Vector x0 = Vector::Zero(circuit.system_size());

        // Create and run DC solver
        DCConvergenceSolver<SparseLUPolicy> solver(config);
        return solver.solve(x0, circuit.num_nodes(), circuit.num_branches(),
                           system_func, nullptr);
    }, py::arg("circuit"), py::arg("config") = DCConvergenceConfig(),
    R"doc(
    Compute DC operating point with automatic convergence aids.

    This function tries multiple strategies to find the DC solution:
    - Direct Newton solve
    - Gmin stepping
    - Pseudo-transient continuation
    - Random restarts

    Args:
        circuit: Circuit object with devices
        config: DC convergence configuration (strategy, tolerances, etc.)

    Returns:
        DCAnalysisResult with solution and convergence info
    )doc");

    // =========================================================================
    // Transient Simulation (Phase 6)
    // =========================================================================

    // Transient simulation result structure
    py::class_<std::tuple<std::vector<Real>, std::vector<Vector>, bool, std::string>>(v2, "TransientResult",
        "Transient simulation result")
        .def_property_readonly("time", [](const std::tuple<std::vector<Real>, std::vector<Vector>, bool, std::string>& r) {
            return std::get<0>(r);
        })
        .def_property_readonly("states", [](const std::tuple<std::vector<Real>, std::vector<Vector>, bool, std::string>& r) {
            return std::get<1>(r);
        })
        .def_property_readonly("success", [](const std::tuple<std::vector<Real>, std::vector<Vector>, bool, std::string>& r) {
            return std::get<2>(r);
        })
        .def_property_readonly("message", [](const std::tuple<std::vector<Real>, std::vector<Vector>, bool, std::string>& r) {
            return std::get<3>(r);
        });

    // Transient simulation function
    v2.def("run_transient", [](Circuit& circuit, Real t_start, Real t_stop, Real dt,
                                const Vector& x0, const NewtonOptions& newton_opts) {
        std::vector<Real> times;
        std::vector<Vector> states;
        bool success = true;
        std::string message = "Transient completed";

        // Set timestep for dynamic elements
        circuit.set_timestep(dt);

        // Configure Newton solver
        NewtonOptions opts = newton_opts;
        opts.num_nodes = circuit.num_nodes();
        opts.num_branches = circuit.num_branches();
        NewtonRaphsonSolver<SparseLUPolicy> solver(opts);

        // System function
        auto system_func = [&circuit](const Vector& x, Vector& f, SparseMatrix& J) {
            circuit.assemble_jacobian(J, f, x);
        };

        // Initial state
        Vector x = x0;

        // Set initial time for time-varying sources
        circuit.set_current_time(t_start);

        // Initialize dynamic element history from initial condition (e.g., DC op point)
        // Use initialize=true to set i_prev=0 for capacitors (DC steady state)
        circuit.update_history(x, true);

        // Store initial state
        times.push_back(t_start);
        states.push_back(x);

        // Time stepping
        Real t = t_start;
        int step = 0;
        const int max_steps = static_cast<int>((t_stop - t_start) / dt) + 1;

        while (t < t_stop && step < max_steps) {
            // Advance time first (for time-varying sources)
            t += dt;
            circuit.set_current_time(t);

            // Solve at current time
            auto result = solver.solve(x, system_func);

            if (!result.success()) {
                success = false;
                message = "Newton failed at t=" + std::to_string(t) + ": " + result.error_message;
                break;
            }

            // Update solution
            x = result.solution;

            // Update dynamic element history
            circuit.update_history(x);

            // Store state
            step++;
            times.push_back(t);
            states.push_back(x);
        }

        return std::make_tuple(times, states, success, message);
    }, py::arg("circuit"), py::arg("t_start"), py::arg("t_stop"), py::arg("dt"),
       py::arg("x0"), py::arg("newton_options") = NewtonOptions(),
    R"doc(
    Run transient simulation with fixed timestep.

    Args:
        circuit: Circuit object with devices
        t_start: Start time (s)
        t_stop: Stop time (s)
        dt: Fixed timestep (s)
        x0: Initial state vector (e.g., from DC operating point)
        newton_options: Newton solver options

    Returns:
        Tuple of (times, states, success, message)
    )doc");

    // Convenience function with zero initial state
    v2.def("run_transient", [](Circuit& circuit, Real t_start, Real t_stop, Real dt,
                                const NewtonOptions& newton_opts) {
        Vector x0 = Vector::Zero(circuit.system_size());
        // Forward to the full version using py::cpp_function
        std::vector<Real> times;
        std::vector<Vector> states;
        bool success = true;
        std::string message = "Transient completed";

        circuit.set_timestep(dt);

        NewtonOptions opts = newton_opts;
        opts.num_nodes = circuit.num_nodes();
        opts.num_branches = circuit.num_branches();
        NewtonRaphsonSolver<SparseLUPolicy> solver(opts);

        auto system_func = [&circuit](const Vector& x, Vector& f, SparseMatrix& J) {
            circuit.assemble_jacobian(J, f, x);
        };

        Vector x = x0;

        // Set initial time for time-varying sources
        circuit.set_current_time(t_start);

        // Initialize dynamic element history (zero IC case)
        circuit.update_history(x, true);
        times.push_back(t_start);
        states.push_back(x);

        Real t = t_start;
        int step = 0;
        const int max_steps = static_cast<int>((t_stop - t_start) / dt) + 1;

        while (t < t_stop && step < max_steps) {
            // Advance time first (for time-varying sources)
            t += dt;
            circuit.set_current_time(t);

            auto result = solver.solve(x, system_func);
            if (!result.success()) {
                success = false;
                message = "Newton failed at t=" + std::to_string(t) + ": " + result.error_message;
                break;
            }
            x = result.solution;
            circuit.update_history(x);  // Normal update after each step
            step++;
            times.push_back(t);
            states.push_back(x);
        }

        return std::make_tuple(times, states, success, message);
    }, py::arg("circuit"), py::arg("t_start"), py::arg("t_stop"), py::arg("dt"),
       py::arg("newton_options") = NewtonOptions(),
    "Run transient with zero initial state");

    // =========================================================================
    // Robust Transient Simulation with Convergence Aids
    // =========================================================================

    // Robust transient with Gmin fallback and timestep refinement
    v2.def("run_transient_robust", [](Circuit& circuit, Real t_start, Real t_stop, Real dt,
                                       const Vector& x0, const NewtonOptions& newton_opts,
                                       bool use_gmin_fallback, const GminConfig& gmin_config,
                                       int max_dt_reductions, Real dt_reduction_factor) {
        std::vector<Real> times;
        std::vector<Vector> states;
        bool success = true;
        std::string message = "Transient completed";
        int gmin_fallback_count = 0;
        int dt_reduction_count = 0;

        // Set timestep for dynamic elements
        Real current_dt = dt;
        circuit.set_timestep(current_dt);

        // Configure Newton solver
        NewtonOptions opts = newton_opts;
        opts.num_nodes = circuit.num_nodes();
        opts.num_branches = circuit.num_branches();
        NewtonRaphsonSolver<SparseLUPolicy> solver(opts);

        // System function
        auto system_func = [&circuit](const Vector& x, Vector& f, SparseMatrix& J) {
            circuit.assemble_jacobian(J, f, x);
        };

        // Initial state
        Vector x = x0;

        // Set initial time
        circuit.set_current_time(t_start);
        circuit.update_history(x, true);

        // Store initial state
        times.push_back(t_start);
        states.push_back(x);

        Real t = t_start;
        int step = 0;
        const int max_steps = static_cast<int>((t_stop - t_start) / dt * 10) + 1; // Allow for refinement

        while (t < t_stop && step < max_steps) {
            // Advance time
            Real t_next = t + current_dt;
            circuit.set_current_time(t_next);

            // Try normal Newton solve first
            auto result = solver.solve(x, system_func);

            if (!result.success() && use_gmin_fallback) {
                // Fallback 1: Try Gmin stepping
                GminStepping gmin_stepper(gmin_config);

                auto gmin_solve = [&](const Vector& x_start) -> NewtonResult {
                    Real current_gmin = gmin_stepper.current_gmin();
                    Index num_nodes = circuit.num_nodes();

                    auto modified_func = [&](const Vector& x_inner, Vector& f, SparseMatrix& J) {
                        circuit.assemble_jacobian(J, f, x_inner);
                        // Add Gmin to node diagonals only
                        for (Index i = 0; i < num_nodes; ++i) {
                            J.coeffRef(i, i) += current_gmin;
                        }
                    };
                    return solver.solve(x_start, modified_func);
                };

                result = gmin_stepper.execute(x, circuit.num_nodes(), gmin_solve);

                if (result.success()) {
                    gmin_fallback_count++;
                }
            }

            if (!result.success() && max_dt_reductions > 0) {
                // Fallback 2: Try with smaller timestep
                int reductions = 0;
                Real temp_dt = current_dt;
                Vector x_sub = x;
                Real t_sub = t;
                bool sub_success = true;

                while (!result.success() && reductions < max_dt_reductions) {
                    temp_dt *= dt_reduction_factor;
                    circuit.set_timestep(temp_dt);
                    reductions++;

                    // Try sub-stepping from t to t_next
                    x_sub = x;
                    t_sub = t;
                    sub_success = true;

                    while (t_sub < t_next - temp_dt * 0.5) {
                        t_sub += temp_dt;
                        circuit.set_current_time(t_sub);
                        result = solver.solve(x_sub, system_func);

                        if (!result.success()) {
                            sub_success = false;
                            break;
                        }
                        x_sub = result.solution;
                        circuit.update_history(x_sub);
                    }

                    if (sub_success) {
                        result.solution = x_sub;
                        result.status = SolverStatus::Success;
                        dt_reduction_count++;
                        break;
                    }
                }

                // Restore original timestep
                circuit.set_timestep(current_dt);
            }

            if (!result.success()) {
                success = false;
                message = "Newton failed at t=" + std::to_string(t_next) +
                          " after Gmin fallback and " + std::to_string(max_dt_reductions) +
                          " timestep reductions. Error: " + result.error_message;
                break;
            }

            // Update solution
            x = result.solution;
            circuit.update_history(x);

            // Store state
            t = t_next;
            step++;
            times.push_back(t);
            states.push_back(x);
        }

        if (success) {
            message = "Transient completed. Gmin fallbacks: " + std::to_string(gmin_fallback_count) +
                      ", timestep reductions: " + std::to_string(dt_reduction_count);
        }

        return std::make_tuple(times, states, success, message);
    }, py::arg("circuit"), py::arg("t_start"), py::arg("t_stop"), py::arg("dt"),
       py::arg("x0"), py::arg("newton_options") = NewtonOptions(),
       py::arg("use_gmin_fallback") = true, py::arg("gmin_config") = GminConfig(),
       py::arg("max_dt_reductions") = 3, py::arg("dt_reduction_factor") = 0.5,
    R"doc(
    Run transient simulation with convergence aids.

    When Newton fails at a timestep, this function tries:
    1. Gmin stepping (adds small conductances to help convergence)
    2. Timestep reduction (uses smaller substeps)

    Args:
        circuit: Circuit object with devices
        t_start: Start time (s)
        t_stop: Stop time (s)
        dt: Base timestep (s)
        x0: Initial state vector
        newton_options: Newton solver options
        use_gmin_fallback: Enable Gmin stepping fallback (default True)
        gmin_config: Gmin stepping configuration
        max_dt_reductions: Max number of timestep reductions to try (default 3)
        dt_reduction_factor: Factor to reduce dt each attempt (default 0.5)

    Returns:
        Tuple of (times, states, success, message)
    )doc");

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
           // Backwards-compatible alias expected by tests
           .def("damping", &RLCAnalytical::damping_type)
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

    // =========================================================================
    // Thermal Simulation Module
    // =========================================================================

    py::class_<FosterStage>(v2, "FosterStage",
        "Single stage of a Foster thermal network (parallel RC)")
        .def(py::init<Real, Real>(), py::arg("Rth"), py::arg("tau"))
        .def_readwrite("Rth", &FosterStage::Rth, "Thermal resistance (K/W)")
        .def_readwrite("tau", &FosterStage::tau, "Time constant (s)")
        .def("Cth", &FosterStage::Cth, "Compute thermal capacitance from Rth and tau")
        .def("Zth", &FosterStage::Zth, py::arg("t"),
             "Thermal impedance at time t for step power")
        .def("delta_T", &FosterStage::delta_T, py::arg("P"), py::arg("t"),
             "Temperature rise for constant power P at time t");

    py::class_<FosterNetwork>(v2, "FosterNetwork",
        R"doc(Foster thermal network representation.

        Total: Zth(t) = sum_i { Rth_i * (1 - exp(-t/tau_i)) }

        Typical use: datasheet provides (Rth_i, tau_i) pairs from Zth curve fitting.

        Example:
            # Create from Rth and tau lists
            network = FosterNetwork([0.5, 1.0, 2.0], [0.001, 0.01, 0.1], "MOSFET")

            # Or from stages
            stages = [FosterStage(0.5, 0.001), FosterStage(1.0, 0.01)]
            network = FosterNetwork(stages, "IGBT")
        )doc")
        .def(py::init<>())
        .def(py::init<std::vector<FosterStage>, std::string>(),
             py::arg("stages"), py::arg("name") = "")
        .def(py::init<const std::vector<Real>&, const std::vector<Real>&, const std::string&>(),
             py::arg("Rth_values"), py::arg("tau_values"), py::arg("name") = "",
             "Create from Rth and tau vectors")
        .def("add_stage", &FosterNetwork::add_stage,
             py::arg("Rth"), py::arg("tau"), "Add a stage")
        .def("num_stages", &FosterNetwork::num_stages, "Number of stages")
        .def("stage", &FosterNetwork::stage, py::arg("i"),
             "Get stage by index", py::return_value_policy::reference_internal)
        .def("total_Rth", &FosterNetwork::total_Rth,
             "Total thermal resistance (steady-state)")
        .def("Zth", &FosterNetwork::Zth, py::arg("t"),
             "Thermal impedance at time t")
        .def("delta_T", &FosterNetwork::delta_T, py::arg("P"), py::arg("t"),
             "Temperature rise for constant power P at time t")
        .def("delta_T_ss", &FosterNetwork::delta_T_ss, py::arg("P"),
             "Steady-state temperature rise for power P")
        .def("Zth_curve", &FosterNetwork::Zth_curve,
             py::arg("t_start"), py::arg("t_end"), py::arg("num_points"),
             "Generate Zth(t) curve")
        .def("name", &FosterNetwork::name)
        .def("stages", &FosterNetwork::stages, "Get all stages");

    py::class_<CauerStage>(v2, "CauerStage",
        "Single stage of a Cauer thermal network (series R, shunt C)")
        .def(py::init<Real, Real>(), py::arg("Rth"), py::arg("Cth"))
        .def_readwrite("Rth", &CauerStage::Rth, "Thermal resistance (K/W)")
        .def_readwrite("Cth", &CauerStage::Cth, "Thermal capacitance (J/K)")
        .def("tau", &CauerStage::tau, "Time constant of this layer");

    py::class_<CauerNetwork>(v2, "CauerNetwork",
        R"doc(Cauer thermal network representation (ladder network).

        Physically meaningful: each stage represents a thermal layer
        (e.g., junction-to-case, case-to-heatsink, heatsink-to-ambient).
        )doc")
        .def(py::init<>())
        .def(py::init<std::vector<CauerStage>, std::string>(),
             py::arg("stages"), py::arg("name") = "")
        .def(py::init<const std::vector<Real>&, const std::vector<Real>&, const std::string&>(),
             py::arg("Rth_values"), py::arg("Cth_values"), py::arg("name") = "",
             "Create from Rth and Cth vectors")
        .def("add_stage", &CauerNetwork::add_stage,
             py::arg("Rth"), py::arg("Cth"), "Add a stage (layer)")
        .def("num_stages", &CauerNetwork::num_stages, "Number of stages")
        .def("stage", &CauerNetwork::stage, py::arg("i"),
             "Get stage by index (0 = junction side)",
             py::return_value_policy::reference_internal)
        .def("total_Rth", &CauerNetwork::total_Rth,
             "Total thermal resistance (steady-state)")
        .def("total_Cth", &CauerNetwork::total_Cth,
             "Total thermal capacitance")
        .def("delta_T_ss", &CauerNetwork::delta_T_ss, py::arg("P"),
             "Steady-state temperature rise for power P")
        .def("name", &CauerNetwork::name)
        .def("stages", &CauerNetwork::stages, "Get all stages");

    py::class_<ThermalSimulator>(v2, "ThermalSimulator",
        R"doc(Thermal simulator using Foster network model.

        Simulates junction temperature transients given power loss waveform.

        Example:
            # Create thermal network
            network = FosterNetwork([0.5, 1.0], [0.001, 0.1], "MOSFET")

            # Create simulator with 25C ambient
            sim = ThermalSimulator(network, 25.0)

            # Step with 100W for 1ms
            sim.step(100.0, 0.001)
            print(f"Tj = {sim.Tj()}C")

            # Or simulate power waveform
            times = [0.0, 0.001, 0.002, 0.003]
            powers = [0.0, 100.0, 100.0, 0.0]
            temps = sim.simulate(times, powers)
        )doc")
        .def(py::init<const FosterNetwork&, Real>(),
             py::arg("network"), py::arg("T_ambient") = 25.0,
             "Create simulator with Foster network and ambient temperature")
        .def("reset", &ThermalSimulator::reset, "Reset to ambient temperature")
        .def("set_ambient", &ThermalSimulator::set_ambient, py::arg("T_amb"),
             "Set ambient temperature")
        .def("ambient", &ThermalSimulator::ambient, "Get ambient temperature")
        .def("junction_temperature", &ThermalSimulator::junction_temperature,
             "Get current junction temperature")
        .def("Tj", &ThermalSimulator::Tj, "Alias for junction_temperature")
        .def("time", &ThermalSimulator::time, "Get current simulation time")
        .def("network", &ThermalSimulator::network, "Get thermal network",
             py::return_value_policy::reference_internal)
        .def("step", &ThermalSimulator::step, py::arg("power"), py::arg("dt"),
             "Step simulation with constant power for duration dt")
        .def("steady_state_temperature", &ThermalSimulator::steady_state_temperature,
             py::arg("power"), "Compute steady-state temperature for given power")
        .def("simulate", &ThermalSimulator::simulate,
             py::arg("times"), py::arg("powers"),
             "Run simulation for power waveform, returns temperature waveform")
        .def("Zth_curve", &ThermalSimulator::Zth_curve,
             py::arg("t_end"), py::arg("num_points"), py::arg("P_step") = 1.0,
             "Compute Zth(t) step response curve")
        .def("stage_temperatures", &ThermalSimulator::stage_temperatures,
             "Get per-stage temperature rises");

    py::class_<ThermalLimitMonitor>(v2, "ThermalLimitMonitor",
        "Monitors junction temperature against limits")
        .def(py::init<Real, Real>(),
             py::arg("T_warning") = 125.0, py::arg("T_max") = 150.0,
             "Create monitor with temperature limits")
        .def("check", &ThermalLimitMonitor::check, py::arg("Tj"),
             "Check temperature and return status: 0=OK, 1=warning, 2=exceeded")
        .def("is_ok", &ThermalLimitMonitor::is_ok, py::arg("Tj"),
             "Check if temperature is OK")
        .def("is_warning", &ThermalLimitMonitor::is_warning, py::arg("Tj"),
             "Check if in warning zone")
        .def("is_exceeded", &ThermalLimitMonitor::is_exceeded, py::arg("Tj"),
             "Check if maximum exceeded")
        .def("T_warning", &ThermalLimitMonitor::T_warning, "Get warning threshold")
        .def("T_max", &ThermalLimitMonitor::T_max, "Get maximum threshold")
        .def("set_limits", &ThermalLimitMonitor::set_limits,
             py::arg("T_warn"), py::arg("T_max"), "Set thresholds");

    py::class_<ThermalResult>(v2, "ThermalResult", "Result of thermal simulation")
        .def(py::init<>())
        .def_readwrite("times", &ThermalResult::times, "Time points")
        .def_readwrite("temperatures", &ThermalResult::temperatures, "Junction temperatures")
        .def_readwrite("powers", &ThermalResult::powers, "Power loss at each time")
        .def_readwrite("T_max", &ThermalResult::T_max, "Peak temperature")
        .def_readwrite("T_avg", &ThermalResult::T_avg, "Average temperature")
        .def_readwrite("t_max", &ThermalResult::t_max, "Time of peak temperature")
        .def_readwrite("exceeded_limit", &ThermalResult::exceeded_limit,
             "True if T_max was exceeded")
        .def_readwrite("message", &ThermalResult::message)
        .def("compute_stats", &ThermalResult::compute_stats,
             "Compute statistics from waveforms");

    // Factory functions for thermal networks
    v2.def("create_mosfet_thermal_model", &create_mosfet_thermal_model,
           py::arg("Rth_jc"), py::arg("Rth_cs"), py::arg("Rth_sa"),
           py::arg("name") = "",
           R"doc(Create a typical 3-stage Foster network for MOSFET.

           Args:
               Rth_jc: Junction-to-case thermal resistance (K/W)
               Rth_cs: Case-to-sink thermal resistance (K/W)
               Rth_sa: Sink-to-ambient thermal resistance (K/W)
               name: Optional name for the network
           )doc");

    v2.def("create_from_datasheet_4param", &create_from_datasheet_4param,
           py::arg("R1"), py::arg("tau1"),
           py::arg("R2"), py::arg("tau2"),
           py::arg("R3"), py::arg("tau3"),
           py::arg("R4"), py::arg("tau4"),
           py::arg("name") = "",
           "Create Foster network from 4-parameter datasheet model");

    v2.def("create_simple_thermal_model", &create_simple_thermal_model,
           py::arg("Rth_ja"), py::arg("tau") = 1.0, py::arg("name") = "",
           "Create simple single-stage thermal model");

    // =========================================================================
    // Power Loss Calculation Module
    // =========================================================================

    py::class_<MOSFETLossParams>(v2, "MOSFETLossParams",
        "MOSFET loss model parameters for conduction and switching losses")
        .def(py::init<>())
        .def_readwrite("Rds_on", &MOSFETLossParams::Rds_on,
             "On-state resistance at 25C (Ω)")
        .def_readwrite("Rds_on_tc", &MOSFETLossParams::Rds_on_tc,
             "Temperature coefficient (Ω/K)")
        .def_readwrite("Qg", &MOSFETLossParams::Qg,
             "Total gate charge (C)")
        .def_readwrite("Eon_25C", &MOSFETLossParams::Eon_25C,
             "Turn-on energy at 25C (J)")
        .def_readwrite("Eoff_25C", &MOSFETLossParams::Eoff_25C,
             "Turn-off energy at 25C (J)")
        .def_readwrite("I_ref", &MOSFETLossParams::I_ref,
             "Reference current for Esw (A)")
        .def_readwrite("V_ref", &MOSFETLossParams::V_ref,
             "Reference voltage for Esw (V)")
        .def_readwrite("T_ref", &MOSFETLossParams::T_ref,
             "Reference temperature (C)")
        .def_readwrite("Esw_tc", &MOSFETLossParams::Esw_tc,
             "Switching energy temp coefficient (1/K)")
        .def("Rds_on_at_T", &MOSFETLossParams::Rds_on_at_T, py::arg("T"),
             "Calculate Rds_on at temperature T");

    py::class_<IGBTLossParams>(v2, "IGBTLossParams",
        "IGBT loss model parameters")
        .def(py::init<>())
        .def_readwrite("Vce_sat", &IGBTLossParams::Vce_sat,
             "Collector-emitter saturation voltage (V)")
        .def_readwrite("Rce", &IGBTLossParams::Rce,
             "Collector-emitter resistance (Ω)")
        .def_readwrite("Vce_tc", &IGBTLossParams::Vce_tc,
             "Vce temperature coefficient (V/K)")
        .def_readwrite("Eon_25C", &IGBTLossParams::Eon_25C,
             "Turn-on energy at 25C (J)")
        .def_readwrite("Eoff_25C", &IGBTLossParams::Eoff_25C,
             "Turn-off energy at 25C (J)")
        .def_readwrite("I_ref", &IGBTLossParams::I_ref,
             "Reference current (A)")
        .def_readwrite("V_ref", &IGBTLossParams::V_ref,
             "Reference voltage (V)")
        .def_readwrite("T_ref", &IGBTLossParams::T_ref,
             "Reference temperature (C)")
        .def_readwrite("Esw_tc", &IGBTLossParams::Esw_tc,
             "Switching energy temp coefficient (1/K)")
        .def("Vce_sat_at_T", &IGBTLossParams::Vce_sat_at_T, py::arg("T"),
             "Calculate Vce_sat at temperature T");

    py::class_<DiodeLossParams>(v2, "DiodeLossParams",
        "Diode loss model parameters")
        .def(py::init<>())
        .def_readwrite("Vf", &DiodeLossParams::Vf,
             "Forward voltage at 25C (V)")
        .def_readwrite("Rd", &DiodeLossParams::Rd,
             "Dynamic resistance (Ω)")
        .def_readwrite("Vf_tc", &DiodeLossParams::Vf_tc,
             "Vf temperature coefficient (V/K)")
        .def_readwrite("Qrr", &DiodeLossParams::Qrr,
             "Reverse recovery charge (C)")
        .def_readwrite("trr", &DiodeLossParams::trr,
             "Reverse recovery time (s)")
        .def_readwrite("Irr_factor", &DiodeLossParams::Irr_factor,
             "Irr as fraction of If")
        .def_readwrite("Err_factor", &DiodeLossParams::Err_factor,
             "Err factor")
        .def_readwrite("T_ref", &DiodeLossParams::T_ref,
             "Reference temperature (C)")
        .def("Vf_at_T", &DiodeLossParams::Vf_at_T, py::arg("T"),
             "Calculate Vf at temperature T")
        .def("Err", &DiodeLossParams::Err,
             py::arg("If"), py::arg("Vr"), py::arg("T"),
             "Calculate reverse recovery energy");

    // Conduction loss functions
    py::class_<ConductionLoss>(v2, "ConductionLoss",
        "Static methods for conduction loss calculation")
        .def_static("resistor", &ConductionLoss::resistor,
             py::arg("I"), py::arg("R"),
             "Resistor conduction loss: P = I² * R")
        .def_static("mosfet", &ConductionLoss::mosfet,
             py::arg("I"), py::arg("params"), py::arg("T"),
             "MOSFET conduction loss: P = I² * Rds_on(T)")
        .def_static("igbt", &ConductionLoss::igbt,
             py::arg("I"), py::arg("params"), py::arg("T"),
             "IGBT conduction loss: P = Vce_sat * I + Rce * I²")
        .def_static("diode", &ConductionLoss::diode,
             py::arg("I"), py::arg("params"), py::arg("T"),
             "Diode conduction loss: P = Vf * I + Rd * I²");

    // Switching loss functions
    py::class_<SwitchingLoss>(v2, "SwitchingLoss",
        "Static methods for switching loss calculation")
        .def_static("mosfet_Eon", &SwitchingLoss::mosfet_Eon,
             py::arg("I"), py::arg("V"), py::arg("T"), py::arg("params"),
             "MOSFET turn-on energy")
        .def_static("mosfet_Eoff", &SwitchingLoss::mosfet_Eoff,
             py::arg("I"), py::arg("V"), py::arg("T"), py::arg("params"),
             "MOSFET turn-off energy")
        .def_static("mosfet_total", &SwitchingLoss::mosfet_total,
             py::arg("I"), py::arg("V"), py::arg("T"), py::arg("params"),
             "MOSFET total switching energy per cycle")
        .def_static("mosfet_power", &SwitchingLoss::mosfet_power,
             py::arg("I"), py::arg("V"), py::arg("T"), py::arg("f_sw"), py::arg("params"),
             "MOSFET switching power at frequency f_sw")
        .def_static("igbt_Eon", &SwitchingLoss::igbt_Eon,
             py::arg("I"), py::arg("V"), py::arg("T"), py::arg("params"),
             "IGBT turn-on energy")
        .def_static("igbt_Eoff", &SwitchingLoss::igbt_Eoff,
             py::arg("I"), py::arg("V"), py::arg("T"), py::arg("params"),
             "IGBT turn-off energy")
        .def_static("igbt_total", &SwitchingLoss::igbt_total,
             py::arg("I"), py::arg("V"), py::arg("T"), py::arg("params"),
             "IGBT total switching energy per cycle")
        .def_static("igbt_power", &SwitchingLoss::igbt_power,
             py::arg("I"), py::arg("V"), py::arg("T"), py::arg("f_sw"), py::arg("params"),
             "IGBT switching power at frequency f_sw")
        .def_static("diode_Err", &SwitchingLoss::diode_Err,
             py::arg("If"), py::arg("Vr"), py::arg("T"), py::arg("params"),
             "Diode reverse recovery energy")
        .def_static("diode_power", &SwitchingLoss::diode_power,
             py::arg("If"), py::arg("Vr"), py::arg("T"), py::arg("f_sw"), py::arg("params"),
             "Diode reverse recovery power at frequency f_sw");

    py::class_<LossBreakdown>(v2, "LossBreakdown",
        "Breakdown of losses by type")
        .def(py::init<>())
        .def_readwrite("conduction", &LossBreakdown::conduction, "Conduction loss (W)")
        .def_readwrite("turn_on", &LossBreakdown::turn_on, "Turn-on switching loss (W)")
        .def_readwrite("turn_off", &LossBreakdown::turn_off, "Turn-off switching loss (W)")
        .def_readwrite("reverse_recovery", &LossBreakdown::reverse_recovery,
             "Diode reverse recovery loss (W)")
        .def("total", &LossBreakdown::total, "Total loss")
        .def("switching", &LossBreakdown::switching, "Total switching loss");

    py::class_<LossAccumulator>(v2, "LossAccumulator",
        "Accumulates losses over time for a device")
        .def(py::init<>())
        .def("reset", &LossAccumulator::reset, "Reset accumulated energy")
        .def("add_sample", &LossAccumulator::add_sample,
             py::arg("P_cond"), py::arg("dt"),
             "Add instantaneous power sample")
        .def("add_switching_event", &LossAccumulator::add_switching_event,
             py::arg("E_sw"), "Add switching event energy")
        .def("total_energy", &LossAccumulator::total_energy,
             "Get total accumulated energy (J)")
        .def("conduction_energy", &LossAccumulator::conduction_energy,
             "Get conduction energy (J)")
        .def("switching_energy", &LossAccumulator::switching_energy,
             "Get switching energy (J)")
        .def("average_power", &LossAccumulator::average_power,
             "Get average power (W)")
        .def("average_conduction_power", &LossAccumulator::average_conduction_power,
             "Get average conduction power (W)")
        .def("average_switching_power", &LossAccumulator::average_switching_power,
             "Get average switching power (W)")
        .def("duration", &LossAccumulator::duration, "Get simulation duration")
        .def("num_samples", &LossAccumulator::num_samples, "Get number of samples");

    py::class_<EfficiencyCalculator>(v2, "EfficiencyCalculator",
        "Calculate converter efficiency")
        .def_static("from_power", &EfficiencyCalculator::from_power,
             py::arg("P_in"), py::arg("P_out"),
             "Calculate efficiency from input/output power")
        .def_static("from_losses", &EfficiencyCalculator::from_losses,
             py::arg("P_out"), py::arg("P_loss"),
             "Calculate efficiency from output power and losses")
        .def_static("losses_from_efficiency", &EfficiencyCalculator::losses_from_efficiency,
             py::arg("eta"), py::arg("P_out"),
             "Calculate losses from efficiency and output power")
        .def_static("input_power", &EfficiencyCalculator::input_power,
             py::arg("eta"), py::arg("P_out"),
             "Calculate input power from efficiency and output power");

    py::class_<LossResult>(v2, "LossResult", "Complete loss analysis result")
        .def(py::init<>())
        .def_readwrite("device_name", &LossResult::device_name)
        .def_readwrite("breakdown", &LossResult::breakdown)
        .def_readwrite("total_energy", &LossResult::total_energy)
        .def_readwrite("average_power", &LossResult::average_power)
        .def_readwrite("peak_power", &LossResult::peak_power)
        .def_readwrite("rms_current", &LossResult::rms_current)
        .def_readwrite("avg_current", &LossResult::avg_current)
        .def_readwrite("efficiency_contribution", &LossResult::efficiency_contribution)
        .def_readwrite("power_waveform", &LossResult::power_waveform)
        .def_readwrite("times", &LossResult::times)
        .def("compute_stats", &LossResult::compute_stats);

    py::class_<SystemLossSummary>(v2, "SystemLossSummary",
        "System-wide loss summary")
        .def(py::init<>())
        .def_readwrite("device_losses", &SystemLossSummary::device_losses)
        .def_readwrite("total_loss", &SystemLossSummary::total_loss)
        .def_readwrite("total_conduction", &SystemLossSummary::total_conduction)
        .def_readwrite("total_switching", &SystemLossSummary::total_switching)
        .def_readwrite("input_power", &SystemLossSummary::input_power)
        .def_readwrite("output_power", &SystemLossSummary::output_power)
        .def_readwrite("efficiency", &SystemLossSummary::efficiency)
        .def("compute_totals", &SystemLossSummary::compute_totals);

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
