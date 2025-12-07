#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include "pulsim/types.hpp"
#include "pulsim/circuit.hpp"
#include "pulsim/parser.hpp"
#include "pulsim/simulation.hpp"
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
        .def_readwrite("output_signals", &SimulationOptions::output_signals);

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
        .def("to_dict", [](const SimulationResult& r) {
            py::dict result;
            result["time"] = r.time;
            result["signal_names"] = r.signal_names;
            result["total_time_seconds"] = r.total_time_seconds;
            result["total_steps"] = r.total_steps;
            result["status"] = static_cast<int>(r.final_status);

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
        });

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
        .def("power_losses", &Simulator::power_losses)
        .def("set_options", &Simulator::set_options);

    // Convenience function
    m.def("simulate", [](const Circuit& circuit, const SimulationOptions& options) {
        return simulate(circuit, options);
    }, py::arg("circuit"), py::arg("options") = SimulationOptions{},
       "Run a transient simulation on the circuit");

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
