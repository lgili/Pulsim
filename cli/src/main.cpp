#include <CLI/CLI.hpp>
#include <pulsim/pulsim.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <future>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <filesystem>

#ifdef PULSIM_WITH_GRPC
#include "pulsim/api/grpc/server_config.hpp"
#include "pulsim/api/grpc/session_manager.hpp"
#include "pulsim/api/grpc/simulator.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#endif

#ifdef PULSIM_WITH_HDF5
#include <H5Cpp.h>
#endif

#ifdef PULSIM_WITH_PARQUET
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#endif

using namespace pulsim;
using json = nlohmann::json;
namespace fs = std::filesystem;

// Output format enumeration
enum class OutputFormat {
    CSV,
    HDF5,
    Parquet
};

// Sentinel values to detect if CLI option was explicitly provided
constexpr double CLI_SENTINEL = -1e99;
constexpr int CLI_SENTINEL_INT = -1;

// Configuration file structure
struct PulsimConfig {
    // Server settings
    std::string server_address = "0.0.0.0:50051";
    bool server_reflection = true;
    bool server_metrics = true;
    int server_max_sessions = 64;

    // Simulation defaults
    double default_tstop = 1e-3;
    double default_dt = 1e-6;
    double default_dtmax = 1e-5;
    double default_abstol = 1e-9;
    double default_reltol = 1e-6;
    int default_maxiter = 50;

    // Sweep defaults
    int sweep_threads = 0;  // 0 = auto

    // Output defaults
    std::string output_format = "csv";

    static PulsimConfig load(const std::string& path);
    static PulsimConfig load_default();
};

PulsimConfig PulsimConfig::load(const std::string& path) {
    PulsimConfig config;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    json j;
    file >> j;

    if (j.contains("server")) {
        auto& srv = j["server"];
        if (srv.contains("address")) config.server_address = srv["address"];
        if (srv.contains("reflection")) config.server_reflection = srv["reflection"];
        if (srv.contains("metrics")) config.server_metrics = srv["metrics"];
        if (srv.contains("max_sessions")) config.server_max_sessions = srv["max_sessions"];
    }

    if (j.contains("simulation")) {
        auto& sim = j["simulation"];
        if (sim.contains("tstop")) config.default_tstop = sim["tstop"];
        if (sim.contains("dt")) config.default_dt = sim["dt"];
        if (sim.contains("dtmax")) config.default_dtmax = sim["dtmax"];
        if (sim.contains("abstol")) config.default_abstol = sim["abstol"];
        if (sim.contains("reltol")) config.default_reltol = sim["reltol"];
        if (sim.contains("maxiter")) config.default_maxiter = sim["maxiter"];
    }

    if (j.contains("sweep")) {
        auto& swp = j["sweep"];
        if (swp.contains("threads")) config.sweep_threads = swp["threads"];
    }

    if (j.contains("output")) {
        auto& out = j["output"];
        if (out.contains("format")) config.output_format = out["format"];
    }

    return config;
}

PulsimConfig PulsimConfig::load_default() {
    // Try standard locations
    std::vector<std::string> paths = {
        "./pulsim.json",
        "./pulsim.config.json",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/.config/pulsim/config.json",
        "/etc/pulsim/config.json"
    };

    for (const auto& path : paths) {
        if (!path.empty() && fs::exists(path)) {
            try {
                return load(path);
            } catch (...) {
                // Ignore and try next
            }
        }
    }

    return PulsimConfig{};
}

OutputFormat parse_output_format(const std::string& filename, const std::string& format_hint) {
    std::string ext = fs::path(filename).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (!format_hint.empty()) {
        if (format_hint == "csv") return OutputFormat::CSV;
        if (format_hint == "hdf5" || format_hint == "h5") return OutputFormat::HDF5;
        if (format_hint == "parquet" || format_hint == "pq") return OutputFormat::Parquet;
    }

    if (ext == ".h5" || ext == ".hdf5") return OutputFormat::HDF5;
    if (ext == ".parquet" || ext == ".pq") return OutputFormat::Parquet;

    return OutputFormat::CSV;
}

void write_csv(const SimulationResult& result, const std::string& filename) {
    std::ostream* out = &std::cout;
    std::ofstream file;

    if (!filename.empty() && filename != "-") {
        file.open(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open output file: " + filename);
        }
        out = &file;
    }

    // Header
    *out << "time";
    for (const auto& name : result.signal_names) {
        *out << "," << name;
    }
    *out << "\n";

    // Data
    *out << std::scientific << std::setprecision(9);
    for (size_t i = 0; i < result.time.size(); ++i) {
        *out << result.time[i];
        for (Index j = 0; j < result.data[i].size(); ++j) {
            *out << "," << result.data[i](j);
        }
        *out << "\n";
    }
}

#ifdef PULSIM_WITH_HDF5
void write_hdf5(const SimulationResult& result, const std::string& filename) {
    H5::H5File file(filename, H5F_ACC_TRUNC);

    // Create time dataset
    hsize_t dims[1] = {result.time.size()};
    H5::DataSpace dataspace(1, dims);

    H5::DataSet time_ds = file.createDataSet("time", H5::PredType::NATIVE_DOUBLE, dataspace);
    time_ds.write(result.time.data(), H5::PredType::NATIVE_DOUBLE);

    // Create signals group
    H5::Group signals = file.createGroup("/signals");

    for (size_t s = 0; s < result.signal_names.size(); ++s) {
        std::vector<double> signal_data(result.time.size());
        for (size_t i = 0; i < result.time.size(); ++i) {
            signal_data[i] = result.data[i](s);
        }

        H5::DataSet ds = signals.createDataSet(result.signal_names[s],
                                                H5::PredType::NATIVE_DOUBLE, dataspace);
        ds.write(signal_data.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Add metadata
    H5::Group meta = file.createGroup("/metadata");
    hsize_t scalar_dims[1] = {1};
    H5::DataSpace scalar_space(1, scalar_dims);

    double total_time = result.total_time_seconds;
    H5::DataSet time_meta = meta.createDataSet("simulation_time_seconds",
                                                H5::PredType::NATIVE_DOUBLE, scalar_space);
    time_meta.write(&total_time, H5::PredType::NATIVE_DOUBLE);

    int steps = static_cast<int>(result.total_steps);
    H5::DataSet steps_meta = meta.createDataSet("total_steps",
                                                 H5::PredType::NATIVE_INT, scalar_space);
    steps_meta.write(&steps, H5::PredType::NATIVE_INT);
}
#endif

#ifdef PULSIM_WITH_PARQUET
void write_parquet(const SimulationResult& result, const std::string& filename) {
    // Build schema
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.push_back(arrow::field("time", arrow::float64()));
    for (const auto& name : result.signal_names) {
        fields.push_back(arrow::field(name, arrow::float64()));
    }
    auto schema = arrow::schema(fields);

    // Build arrays
    arrow::DoubleBuilder time_builder;
    PARQUET_THROW_NOT_OK(time_builder.AppendValues(result.time));
    std::shared_ptr<arrow::Array> time_array;
    PARQUET_THROW_NOT_OK(time_builder.Finish(&time_array));

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.push_back(time_array);

    for (size_t s = 0; s < result.signal_names.size(); ++s) {
        arrow::DoubleBuilder builder;
        for (size_t i = 0; i < result.time.size(); ++i) {
            PARQUET_THROW_NOT_OK(builder.Append(result.data[i](s)));
        }
        std::shared_ptr<arrow::Array> arr;
        PARQUET_THROW_NOT_OK(builder.Finish(&arr));
        arrays.push_back(arr);
    }

    auto table = arrow::Table::Make(schema, arrays);

    // Write to file
    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    PARQUET_ASSIGN_OR_THROW(outfile, arrow::io::FileOutputStream::Open(filename));
    PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                                     outfile, 1024*1024));
}
#endif

void write_output(const SimulationResult& result, const std::string& filename,
                  OutputFormat format) {
    switch (format) {
        case OutputFormat::CSV:
            write_csv(result, filename);
            break;
        case OutputFormat::HDF5:
#ifdef PULSIM_WITH_HDF5
            write_hdf5(result, filename);
#else
            throw std::runtime_error("HDF5 support not compiled in. Rebuild with -DPULSIM_WITH_HDF5=ON");
#endif
            break;
        case OutputFormat::Parquet:
#ifdef PULSIM_WITH_PARQUET
            write_parquet(result, filename);
#else
            throw std::runtime_error("Parquet support not compiled in. Rebuild with -DPULSIM_WITH_PARQUET=ON");
#endif
            break;
    }
}

void print_progress(Real time, Real tstop) {
    int percent = static_cast<int>(100.0 * time / tstop);
    std::cerr << "\rProgress: " << percent << "% (t=" << std::scientific
              << std::setprecision(3) << time << "s)" << std::flush;
}

int cmd_run(const std::string& netlist_file, const std::string& output_file,
            const std::string& format_hint,
            double cli_tstop, double cli_dt, double cli_dtmax, double cli_tstart,
            double cli_abstol, double cli_reltol, int cli_maxiter,
            bool verbose, bool quiet) {
    try {
        // Parse netlist
        if (!quiet) {
            std::cerr << "Reading netlist: " << netlist_file << std::endl;
        }

        auto parse_result = NetlistParser::parse_file(netlist_file);
        if (!parse_result) {
            std::cerr << "Error: " << parse_result.error().to_string() << std::endl;
            return 1;
        }

        const Circuit& circuit = *parse_result;

        // Parse simulation options from JSON file first
        auto json_opts_result = NetlistParser::parse_simulation_options(netlist_file);
        SimulationOptions opts = json_opts_result ? *json_opts_result : SimulationOptions{};

        // Apply CLI overrides only if explicitly provided (not sentinel values)
        if (cli_tstart != CLI_SENTINEL) opts.tstart = cli_tstart;
        if (cli_tstop != CLI_SENTINEL) opts.tstop = cli_tstop;
        if (cli_dt != CLI_SENTINEL) opts.dt = cli_dt;
        if (cli_dtmax != CLI_SENTINEL) opts.dtmax = cli_dtmax;
        if (cli_abstol != CLI_SENTINEL) opts.abstol = cli_abstol;
        if (cli_reltol != CLI_SENTINEL) opts.reltol = cli_reltol;
        if (cli_maxiter != CLI_SENTINEL_INT) opts.max_newton_iterations = cli_maxiter;

        if (verbose) {
            std::cerr << "Circuit loaded:" << std::endl;
            std::cerr << "  Nodes: " << circuit.node_count() << std::endl;
            std::cerr << "  Components: " << circuit.components().size() << std::endl;
            std::cerr << "  Variables: " << circuit.total_variables() << std::endl;
        }

        // Run simulation
        if (!quiet) {
            std::cerr << "Running transient simulation..." << std::endl;
            std::cerr << "  tstart: " << opts.tstart << "s" << std::endl;
            std::cerr << "  tstop: " << opts.tstop << "s" << std::endl;
            std::cerr << "  dt: " << opts.dt << "s" << std::endl;
            std::cerr << "  dtmax: " << opts.dtmax << "s" << std::endl;
            std::cerr << "  use_ic: " << (opts.use_ic ? "true" : "false") << std::endl;
        }

        Simulator sim(circuit, opts);

        SimulationResult result;
        if (!quiet) {
            result = sim.run_transient([&opts](Real time, const Vector&) {
                print_progress(time, opts.tstop);
            });
            std::cerr << std::endl;  // Newline after progress
        } else {
            result = sim.run_transient();
        }

        if (result.final_status != SolverStatus::Success) {
            std::cerr << "Simulation failed: " << result.error_message << std::endl;
            return 1;
        }

        // Output results
        if (!quiet) {
            std::cerr << "Simulation completed:" << std::endl;
            std::cerr << "  Total steps: " << result.total_steps << std::endl;
            std::cerr << "  Newton iterations: " << result.newton_iterations_total << std::endl;
            std::cerr << "  Wall time: " << std::fixed << std::setprecision(3)
                      << result.total_time_seconds << "s" << std::endl;
        }

        OutputFormat format = parse_output_format(output_file, format_hint);

        if (!output_file.empty()) {
            if (!quiet) {
                std::cerr << "Writing results to: " << output_file << std::endl;
            }
            write_output(result, output_file, format);
        } else {
            // Write to stdout (CSV only)
            write_csv(result, "");
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

int cmd_validate(const std::string& netlist_file, bool verbose) {
    try {
        auto parse_result = NetlistParser::parse_file(netlist_file);
        if (!parse_result) {
            std::cerr << "Error: " << parse_result.error().to_string() << std::endl;
            return 1;
        }

        const Circuit& circuit = *parse_result;

        std::string error;
        if (!circuit.validate(error)) {
            std::cerr << "Validation failed: " << error << std::endl;
            return 2;
        }

        if (verbose) {
            std::cout << "Netlist is valid." << std::endl;
            std::cout << "  Nodes: " << circuit.node_count() << std::endl;
            std::cout << "  Branches: " << circuit.branch_count() << std::endl;
            std::cout << "  Components: " << circuit.components().size() << std::endl;
            std::cout << "  Variables: " << circuit.total_variables() << std::endl;

            std::cout << "\nSignals:" << std::endl;
            for (Index i = 0; i < circuit.total_variables(); ++i) {
                std::cout << "  [" << i << "] " << circuit.signal_name(i) << std::endl;
            }
        } else {
            std::cout << "OK" << std::endl;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

int cmd_info_circuit(const std::string& netlist_file) {
    try {
        auto parse_result = NetlistParser::parse_file(netlist_file);
        if (!parse_result) {
            std::cerr << "Error: " << parse_result.error().to_string() << std::endl;
            return 1;
        }

        const Circuit& circuit = *parse_result;

        std::cout << "Circuit: " << netlist_file << std::endl;
        std::cout << "\nTopology:" << std::endl;
        std::cout << "  Nodes: " << circuit.node_count() << std::endl;
        std::cout << "  Branches: " << circuit.branch_count() << std::endl;
        std::cout << "  Total variables: " << circuit.total_variables() << std::endl;

        std::cout << "\nComponents (" << circuit.components().size() << "):" << std::endl;
        for (const auto& comp : circuit.components()) {
            std::cout << "  " << comp.name() << ": ";
            switch (comp.type()) {
                case ComponentType::Resistor: std::cout << "Resistor"; break;
                case ComponentType::Capacitor: std::cout << "Capacitor"; break;
                case ComponentType::Inductor: std::cout << "Inductor"; break;
                case ComponentType::VoltageSource: std::cout << "Voltage Source"; break;
                case ComponentType::CurrentSource: std::cout << "Current Source"; break;
                case ComponentType::Diode: std::cout << "Diode"; break;
                case ComponentType::Switch: std::cout << "Switch"; break;
                case ComponentType::MOSFET: std::cout << "MOSFET"; break;
                case ComponentType::IGBT: std::cout << "IGBT"; break;
                case ComponentType::Transformer: std::cout << "Transformer"; break;
                default: std::cout << "Unknown"; break;
            }
            std::cout << " (";
            for (size_t i = 0; i < comp.nodes().size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << comp.nodes()[i];
            }
            std::cout << ")" << std::endl;
        }

        std::cout << "\nNodes:" << std::endl;
        for (const auto& name : circuit.node_names()) {
            std::cout << "  " << name << " -> index " << circuit.node_index(name) << std::endl;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

void print_device_info(const std::string& device_type) {
    std::string type = device_type;
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    if (type == "resistor" || type == "r") {
        std::cout << "Resistor (R)\n";
        std::cout << "============\n\n";
        std::cout << "A linear resistor implementing Ohm's law: V = I * R\n\n";
        std::cout << "JSON Format:\n";
        std::cout << "  {\n";
        std::cout << "    \"name\": \"R1\",\n";
        std::cout << "    \"type\": \"R\",\n";
        std::cout << "    \"n1\": \"node_pos\",\n";
        std::cout << "    \"n2\": \"node_neg\",\n";
        std::cout << "    \"value\": 1000.0     // Resistance in Ohms\n";
        std::cout << "  }\n\n";
        std::cout << "Parameters:\n";
        std::cout << "  value: Resistance value in Ohms (required)\n";
    }
    else if (type == "capacitor" || type == "c") {
        std::cout << "Capacitor (C)\n";
        std::cout << "=============\n\n";
        std::cout << "A linear capacitor: I = C * dV/dt\n\n";
        std::cout << "JSON Format:\n";
        std::cout << "  {\n";
        std::cout << "    \"name\": \"C1\",\n";
        std::cout << "    \"type\": \"C\",\n";
        std::cout << "    \"n1\": \"node_pos\",\n";
        std::cout << "    \"n2\": \"node_neg\",\n";
        std::cout << "    \"value\": 1e-6,      // Capacitance in Farads\n";
        std::cout << "    \"ic\": 0.0           // Initial voltage (optional)\n";
        std::cout << "  }\n\n";
        std::cout << "Parameters:\n";
        std::cout << "  value: Capacitance in Farads (required)\n";
        std::cout << "  ic: Initial voltage across capacitor (optional, default: 0)\n";
    }
    else if (type == "inductor" || type == "l") {
        std::cout << "Inductor (L)\n";
        std::cout << "============\n\n";
        std::cout << "A linear inductor: V = L * dI/dt\n\n";
        std::cout << "JSON Format:\n";
        std::cout << "  {\n";
        std::cout << "    \"name\": \"L1\",\n";
        std::cout << "    \"type\": \"L\",\n";
        std::cout << "    \"n1\": \"node_pos\",\n";
        std::cout << "    \"n2\": \"node_neg\",\n";
        std::cout << "    \"value\": 1e-3,      // Inductance in Henries\n";
        std::cout << "    \"ic\": 0.0           // Initial current (optional)\n";
        std::cout << "  }\n\n";
        std::cout << "Parameters:\n";
        std::cout << "  value: Inductance in Henries (required)\n";
        std::cout << "  ic: Initial current through inductor (optional, default: 0)\n";
    }
    else if (type == "voltage" || type == "v" || type == "vsource") {
        std::cout << "Voltage Source (V)\n";
        std::cout << "==================\n\n";
        std::cout << "Independent voltage source with various waveform types.\n\n";
        std::cout << "JSON Format (DC):\n";
        std::cout << "  {\n";
        std::cout << "    \"name\": \"V1\",\n";
        std::cout << "    \"type\": \"V\",\n";
        std::cout << "    \"n1\": \"node_pos\",\n";
        std::cout << "    \"n2\": \"node_neg\",\n";
        std::cout << "    \"waveform\": 5.0     // DC voltage in Volts\n";
        std::cout << "  }\n\n";
        std::cout << "JSON Format (Pulse):\n";
        std::cout << "  {\n";
        std::cout << "    \"name\": \"V1\",\n";
        std::cout << "    \"type\": \"V\",\n";
        std::cout << "    \"n1\": \"node_pos\",\n";
        std::cout << "    \"n2\": \"node_neg\",\n";
        std::cout << "    \"waveform\": {\n";
        std::cout << "      \"type\": \"pulse\",\n";
        std::cout << "      \"v1\": 0.0,        // Low voltage\n";
        std::cout << "      \"v2\": 5.0,        // High voltage\n";
        std::cout << "      \"td\": 0.0,        // Delay\n";
        std::cout << "      \"tr\": 1e-9,       // Rise time\n";
        std::cout << "      \"tf\": 1e-9,       // Fall time\n";
        std::cout << "      \"pw\": 1e-6,       // Pulse width\n";
        std::cout << "      \"per\": 2e-6       // Period\n";
        std::cout << "    }\n";
        std::cout << "  }\n\n";
        std::cout << "Waveform Types:\n";
        std::cout << "  - DC: Constant voltage (number or {\"type\": \"dc\", \"value\": ...})\n";
        std::cout << "  - Pulse: {\"type\": \"pulse\", v1, v2, td, tr, tf, pw, per}\n";
        std::cout << "  - Sine: {\"type\": \"sine\", vo, va, freq, td, theta, phi}\n";
        std::cout << "  - PWL: {\"type\": \"pwl\", \"points\": [[t1,v1], [t2,v2], ...]}\n";
        std::cout << "  - PWM: {\"type\": \"pwm\", v1, v2, freq, duty, dead_time}\n";
    }
    else if (type == "diode" || type == "d") {
        std::cout << "Diode (D)\n";
        std::cout << "=========\n\n";
        std::cout << "Semiconductor diode with Shockley equation and optional capacitances.\n\n";
        std::cout << "JSON Format:\n";
        std::cout << "  {\n";
        std::cout << "    \"name\": \"D1\",\n";
        std::cout << "    \"type\": \"D\",\n";
        std::cout << "    \"n1\": \"anode\",\n";
        std::cout << "    \"n2\": \"cathode\",\n";
        std::cout << "    \"params\": {\n";
        std::cout << "      \"Is\": 1e-14,      // Saturation current (A)\n";
        std::cout << "      \"n\": 1.0,         // Emission coefficient\n";
        std::cout << "      \"Vt\": 0.026,      // Thermal voltage (V)\n";
        std::cout << "      \"Vf\": 0.7,        // Forward voltage drop\n";
        std::cout << "      \"Ron\": 0.01,      // On-resistance (Ohms)\n";
        std::cout << "      \"Roff\": 1e9,      // Off-resistance (Ohms)\n";
        std::cout << "      \"Cj0\": 0.0,       // Zero-bias junction capacitance (F)\n";
        std::cout << "      \"Vj\": 0.7,        // Junction potential (V)\n";
        std::cout << "      \"M\": 0.5,         // Grading coefficient\n";
        std::cout << "      \"tt\": 0.0,        // Transit time (s)\n";
        std::cout << "      \"Qrr\": 0.0        // Reverse recovery charge (C)\n";
        std::cout << "    }\n";
        std::cout << "  }\n";
    }
    else if (type == "mosfet" || type == "m") {
        std::cout << "MOSFET (M)\n";
        std::cout << "==========\n\n";
        std::cout << "Power MOSFET with Level 1 model and parasitic capacitances.\n\n";
        std::cout << "JSON Format:\n";
        std::cout << "  {\n";
        std::cout << "    \"name\": \"M1\",\n";
        std::cout << "    \"type\": \"M\",\n";
        std::cout << "    \"nd\": \"drain\",\n";
        std::cout << "    \"ng\": \"gate\",\n";
        std::cout << "    \"ns\": \"source\",\n";
        std::cout << "    \"params\": {\n";
        std::cout << "      \"type\": \"nmos\",     // \"nmos\" or \"pmos\"\n";
        std::cout << "      \"Vth\": 2.0,         // Threshold voltage (V)\n";
        std::cout << "      \"Kp\": 20.0,         // Transconductance (A/V^2)\n";
        std::cout << "      \"Rds_on\": 0.01,     // On-resistance (Ohms)\n";
        std::cout << "      \"Rds_off\": 1e9,     // Off-resistance (Ohms)\n";
        std::cout << "      \"lambda\": 0.0,      // Channel-length modulation\n";
        std::cout << "      \"Cgs\": 0.0,         // Gate-source capacitance (F)\n";
        std::cout << "      \"Cgd\": 0.0,         // Gate-drain capacitance (F)\n";
        std::cout << "      \"Cds\": 0.0,         // Drain-source capacitance (F)\n";
        std::cout << "      \"body_diode\": true  // Enable body diode\n";
        std::cout << "    }\n";
        std::cout << "  }\n";
    }
    else if (type == "igbt" || type == "q") {
        std::cout << "IGBT (Q)\n";
        std::cout << "========\n\n";
        std::cout << "Insulated Gate Bipolar Transistor.\n\n";
        std::cout << "JSON Format:\n";
        std::cout << "  {\n";
        std::cout << "    \"name\": \"Q1\",\n";
        std::cout << "    \"type\": \"IGBT\",\n";
        std::cout << "    \"nc\": \"collector\",\n";
        std::cout << "    \"ng\": \"gate\",\n";
        std::cout << "    \"ne\": \"emitter\",\n";
        std::cout << "    \"params\": {\n";
        std::cout << "      \"Vce_sat\": 2.0,     // Saturation voltage (V)\n";
        std::cout << "      \"Vge_th\": 5.0,      // Gate threshold (V)\n";
        std::cout << "      \"Ic_max\": 100.0,    // Max collector current (A)\n";
        std::cout << "      \"Rce_on\": 0.01,     // On-resistance (Ohms)\n";
        std::cout << "      \"Cies\": 0.0,        // Input capacitance (F)\n";
        std::cout << "      \"Coes\": 0.0,        // Output capacitance (F)\n";
        std::cout << "      \"Cres\": 0.0         // Reverse transfer capacitance (F)\n";
        std::cout << "    }\n";
        std::cout << "  }\n";
    }
    else if (type == "transformer" || type == "x") {
        std::cout << "Transformer (X)\n";
        std::cout << "===============\n\n";
        std::cout << "Two-winding transformer with magnetizing inductance and leakage.\n\n";
        std::cout << "JSON Format:\n";
        std::cout << "  {\n";
        std::cout << "    \"name\": \"T1\",\n";
        std::cout << "    \"type\": \"Transformer\",\n";
        std::cout << "    \"np1\": \"pri_pos\",\n";
        std::cout << "    \"np2\": \"pri_neg\",\n";
        std::cout << "    \"ns1\": \"sec_pos\",\n";
        std::cout << "    \"ns2\": \"sec_neg\",\n";
        std::cout << "    \"params\": {\n";
        std::cout << "      \"n\": 1.0,           // Turns ratio (Np/Ns)\n";
        std::cout << "      \"Lm\": 1e-3,         // Magnetizing inductance (H)\n";
        std::cout << "      \"Llk_pri\": 0.0,     // Primary leakage (H)\n";
        std::cout << "      \"Llk_sec\": 0.0,     // Secondary leakage (H)\n";
        std::cout << "      \"Rp\": 0.0,          // Primary resistance (Ohms)\n";
        std::cout << "      \"Rs\": 0.0           // Secondary resistance (Ohms)\n";
        std::cout << "    }\n";
        std::cout << "  }\n";
    }
    else if (type == "switch" || type == "s") {
        std::cout << "Switch (S)\n";
        std::cout << "==========\n\n";
        std::cout << "Ideal voltage-controlled switch.\n\n";
        std::cout << "JSON Format:\n";
        std::cout << "  {\n";
        std::cout << "    \"name\": \"S1\",\n";
        std::cout << "    \"type\": \"S\",\n";
        std::cout << "    \"n1\": \"node_pos\",\n";
        std::cout << "    \"n2\": \"node_neg\",\n";
        std::cout << "    \"nctrl\": \"control\",\n";
        std::cout << "    \"params\": {\n";
        std::cout << "      \"Vth\": 0.5,         // Threshold voltage (V)\n";
        std::cout << "      \"Ron\": 0.001,       // On-resistance (Ohms)\n";
        std::cout << "      \"Roff\": 1e9         // Off-resistance (Ohms)\n";
        std::cout << "    }\n";
        std::cout << "  }\n";
    }
    else if (type == "list" || type == "all") {
        std::cout << "Available Device Types:\n";
        std::cout << "=======================\n\n";
        std::cout << "Basic Components:\n";
        std::cout << "  R, Resistor      - Linear resistor\n";
        std::cout << "  C, Capacitor     - Linear capacitor\n";
        std::cout << "  L, Inductor      - Linear inductor\n";
        std::cout << "  V, VSource       - Independent voltage source\n";
        std::cout << "  I, ISource       - Independent current source\n\n";
        std::cout << "Semiconductor Devices:\n";
        std::cout << "  D, Diode         - PN junction diode\n";
        std::cout << "  M, MOSFET        - Power MOSFET\n";
        std::cout << "  Q, IGBT          - Insulated Gate Bipolar Transistor\n\n";
        std::cout << "Switches:\n";
        std::cout << "  S, Switch        - Ideal voltage-controlled switch\n\n";
        std::cout << "Magnetics:\n";
        std::cout << "  X, Transformer   - Two-winding transformer\n\n";
        std::cout << "Use 'pulsim info --device <type>' for detailed information.\n";
    }
    else {
        std::cerr << "Unknown device type: " << device_type << std::endl;
        std::cerr << "Use 'pulsim info --device list' to see available types.\n";
    }
}

int cmd_info(const std::string& netlist_file, const std::string& device_type) {
    if (!device_type.empty()) {
        print_device_info(device_type);
        return 0;
    }

    if (!netlist_file.empty()) {
        return cmd_info_circuit(netlist_file);
    }

    // If neither specified, show device list
    print_device_info("list");
    return 0;
}

// Sweep parameter structure
struct SweepParam {
    std::string component;
    std::string parameter;
    double start;
    double stop;
    int steps;
    bool logarithmic = false;
};

struct SweepResult {
    std::vector<double> param_values;
    std::vector<SimulationResult> results;
};

int cmd_sweep(const std::string& netlist_file, const std::string& output_file,
              const std::string& format_hint,
              const std::vector<std::string>& sweep_specs,
              int num_threads, bool verbose, bool quiet) {
    try {
        if (sweep_specs.empty()) {
            std::cerr << "Error: No sweep parameters specified\n";
            std::cerr << "Usage: --param 'component.param:start:stop:steps[:log]'\n";
            return 1;
        }

        // Parse sweep specifications
        std::vector<SweepParam> params;
        for (const auto& spec : sweep_specs) {
            SweepParam p;
            std::istringstream iss(spec);
            std::string token;

            // Parse component.parameter
            std::getline(iss, token, ':');
            auto dot = token.find('.');
            if (dot == std::string::npos) {
                std::cerr << "Error: Invalid sweep spec '" << spec << "'. Expected 'component.param:...'\n";
                return 1;
            }
            p.component = token.substr(0, dot);
            p.parameter = token.substr(dot + 1);

            // Parse start:stop:steps
            std::getline(iss, token, ':');
            p.start = std::stod(token);
            std::getline(iss, token, ':');
            p.stop = std::stod(token);
            std::getline(iss, token, ':');
            p.steps = std::stoi(token);

            // Optional logarithmic flag
            if (std::getline(iss, token, ':')) {
                p.logarithmic = (token == "log" || token == "1" || token == "true");
            }

            params.push_back(p);
        }

        // Only support single parameter sweep for now
        if (params.size() > 1) {
            std::cerr << "Error: Multi-parameter sweeps not yet supported\n";
            return 1;
        }

        const auto& param = params[0];

        // Read base netlist
        std::ifstream file(netlist_file);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open netlist: " << netlist_file << std::endl;
            return 1;
        }
        json base_netlist;
        file >> base_netlist;

        // Generate parameter values
        std::vector<double> values;
        for (int i = 0; i <= param.steps; ++i) {
            double t = static_cast<double>(i) / param.steps;
            double val;
            if (param.logarithmic) {
                val = param.start * std::pow(param.stop / param.start, t);
            } else {
                val = param.start + t * (param.stop - param.start);
            }
            values.push_back(val);
        }

        // Determine thread count
        if (num_threads <= 0) {
            num_threads = static_cast<int>(std::thread::hardware_concurrency());
            if (num_threads <= 0) num_threads = 4;
        }

        if (!quiet) {
            std::cerr << "Sweep: " << param.component << "." << param.parameter << std::endl;
            std::cerr << "  Range: " << param.start << " to " << param.stop << std::endl;
            std::cerr << "  Steps: " << param.steps + 1 << std::endl;
            std::cerr << "  Threads: " << num_threads << std::endl;
        }

        // Run simulations in parallel
        std::vector<SimulationResult> results(values.size());
        std::atomic<int> completed{0};
        std::mutex output_mutex;

        auto worker = [&](int start_idx, int end_idx) {
            for (int i = start_idx; i < end_idx; ++i) {
                // Create modified netlist
                json netlist = base_netlist;

                // Find and modify component
                bool found = false;
                for (auto& comp : netlist["components"]) {
                    if (comp["name"] == param.component) {
                        if (param.parameter == "value") {
                            comp["value"] = values[i];
                        } else if (comp.contains("params")) {
                            comp["params"][param.parameter] = values[i];
                        } else {
                            comp[param.parameter] = values[i];
                        }
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cerr << "Warning: Component '" << param.component << "' not found\n";
                    continue;
                }

                // Parse and simulate
                auto parse_result = NetlistParser::parse_string(netlist.dump());
                if (!parse_result) {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cerr << "Error parsing sweep " << i << ": "
                              << parse_result.error().to_string() << std::endl;
                    continue;
                }

                auto opts_result = NetlistParser::parse_options(netlist.dump());
                SimulationOptions opts = opts_result ? *opts_result : SimulationOptions{};

                Simulator sim(*parse_result, opts);
                results[i] = sim.run_transient();

                int done = ++completed;
                if (!quiet) {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cerr << "\rProgress: " << done << "/" << values.size() << std::flush;
                }
            }
        };

        // Launch threads
        std::vector<std::thread> threads;
        int chunk = static_cast<int>(values.size()) / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk;
            int end = (t == num_threads - 1) ? static_cast<int>(values.size()) : (t + 1) * chunk;
            threads.emplace_back(worker, start, end);
        }

        for (auto& t : threads) {
            t.join();
        }

        if (!quiet) {
            std::cerr << std::endl;
        }

        // Write results summary
        if (!output_file.empty()) {
            std::ofstream out(output_file);
            out << "# Sweep Results\n";
            out << "# Parameter: " << param.component << "." << param.parameter << "\n";
            out << param.parameter;

            // Find signal names from first successful result
            const SimulationResult* first_valid = nullptr;
            for (const auto& r : results) {
                if (r.final_status == SolverStatus::Success && !r.signal_names.empty()) {
                    first_valid = &r;
                    break;
                }
            }

            if (first_valid) {
                for (const auto& sig : first_valid->signal_names) {
                    out << "," << sig << "_final";
                }
            }
            out << ",status\n";

            out << std::scientific << std::setprecision(9);
            for (size_t i = 0; i < values.size(); ++i) {
                out << values[i];
                if (results[i].final_status == SolverStatus::Success && !results[i].data.empty()) {
                    const auto& final_data = results[i].data.back();
                    for (Index j = 0; j < final_data.size(); ++j) {
                        out << "," << final_data(j);
                    }
                    out << ",success";
                } else {
                    out << ",failed";
                }
                out << "\n";
            }

            if (!quiet) {
                std::cerr << "Results written to: " << output_file << std::endl;
            }
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

#ifdef PULSIM_WITH_GRPC
namespace pulsim::api::grpc {
std::pair<std::unique_ptr<::grpc::Server>, std::unique_ptr<::pulsim::api::v1::SimulatorService::Service>>
build_server(SessionManager& manager, const ServerConfig& config);
}

int cmd_serve(const std::string& address, bool reflection, bool metrics,
              int max_sessions, bool verbose, bool quiet) {
    try {
        using namespace pulsim::api::grpc;

        ServerConfig config;
        config.listen_address = address;
        config.enable_reflection = reflection;
        config.enable_metrics = metrics;
        config.max_sessions = static_cast<std::size_t>(max_sessions);

        SessionManager manager(config);

        auto built = build_server(manager, config);
        auto service = std::move(built.second);
        auto& server = built.first;

        if (!quiet) {
            std::cout << "Pulsim gRPC server listening on " << address << std::endl;
            if (reflection) {
                std::cout << "  Reflection: enabled" << std::endl;
            }
            if (metrics) {
                std::cout << "  Metrics: enabled (port " << config.metrics_port << ")" << std::endl;
            }
            std::cout << "  Max sessions: " << max_sessions << std::endl;
        }

        server->Wait();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
#endif

int main(int argc, char** argv) {
    CLI::App app{"Pulsim - High-performance circuit simulator"};
    app.set_version_flag("-V,--version", "Pulsim 0.1.0");

    // Global options
    bool verbose = false;
    bool quiet = false;
    std::string config_file;
    app.add_flag("-v,--verbose", verbose, "Verbose output");
    app.add_flag("-q,--quiet", quiet, "Quiet mode (errors only)");
    app.add_option("-c,--config", config_file, "Configuration file (JSON)")
        ->check(CLI::ExistingFile);

    // Run command
    auto* run_cmd = app.add_subcommand("run", "Run transient simulation");
    std::string netlist_file;
    std::string output_file;
    std::string format_hint;
    double cli_tstop = CLI_SENTINEL;
    double cli_dt = CLI_SENTINEL;
    double cli_dtmax = CLI_SENTINEL;
    double cli_tstart = CLI_SENTINEL;
    double cli_abstol = CLI_SENTINEL;
    double cli_reltol = CLI_SENTINEL;
    int cli_maxiter = CLI_SENTINEL_INT;

    run_cmd->add_option("netlist", netlist_file, "Netlist file (JSON format)")
        ->required()
        ->check(CLI::ExistingFile);
    run_cmd->add_option("-o,--output", output_file, "Output file (CSV, HDF5, or Parquet)");
    run_cmd->add_option("-f,--format", format_hint, "Output format (csv, hdf5, parquet)")
        ->check(CLI::IsMember({"csv", "hdf5", "h5", "parquet", "pq"}));
    run_cmd->add_option("--tstop", cli_tstop, "Stop time (overrides JSON)");
    run_cmd->add_option("--dt", cli_dt, "Initial time step (overrides JSON)");
    run_cmd->add_option("--dtmax", cli_dtmax, "Maximum time step (overrides JSON)");
    run_cmd->add_option("--tstart", cli_tstart, "Start time (overrides JSON)");
    run_cmd->add_option("--abstol", cli_abstol, "Absolute tolerance (overrides JSON)");
    run_cmd->add_option("--reltol", cli_reltol, "Relative tolerance (overrides JSON)");
    run_cmd->add_option("--maxiter", cli_maxiter, "Max Newton iterations (overrides JSON)");

    run_cmd->callback([&]() {
        std::exit(cmd_run(netlist_file, output_file, format_hint,
                          cli_tstop, cli_dt, cli_dtmax, cli_tstart,
                          cli_abstol, cli_reltol, cli_maxiter,
                          verbose, quiet));
    });

    // Validate command
    auto* validate_cmd = app.add_subcommand("validate", "Validate netlist file");
    std::string validate_file;
    validate_cmd->add_option("netlist", validate_file, "Netlist file (JSON format)")
        ->required()
        ->check(CLI::ExistingFile);
    validate_cmd->callback([&]() {
        std::exit(cmd_validate(validate_file, verbose));
    });

    // Info command (enhanced)
    auto* info_cmd = app.add_subcommand("info", "Show circuit or device information");
    std::string info_file;
    std::string device_type;
    info_cmd->add_option("netlist", info_file, "Netlist file (JSON format)")
        ->check(CLI::ExistingFile);
    info_cmd->add_option("-d,--device", device_type,
        "Device type to show documentation (e.g., mosfet, diode, list)");
    info_cmd->callback([&]() {
        std::exit(cmd_info(info_file, device_type));
    });

    // Sweep command
    auto* sweep_cmd = app.add_subcommand("sweep", "Run parameter sweep");
    std::string sweep_netlist;
    std::string sweep_output;
    std::string sweep_format;
    std::vector<std::string> sweep_params;
    int sweep_threads = 0;

    sweep_cmd->add_option("netlist", sweep_netlist, "Netlist file (JSON format)")
        ->required()
        ->check(CLI::ExistingFile);
    sweep_cmd->add_option("-o,--output", sweep_output, "Output file for sweep results");
    sweep_cmd->add_option("-f,--format", sweep_format, "Output format")
        ->check(CLI::IsMember({"csv", "hdf5", "parquet"}));
    sweep_cmd->add_option("-p,--param", sweep_params,
        "Sweep parameter (format: component.param:start:stop:steps[:log])")
        ->required();
    sweep_cmd->add_option("-j,--jobs", sweep_threads,
        "Number of parallel jobs (0=auto)");

    sweep_cmd->callback([&]() {
        std::exit(cmd_sweep(sweep_netlist, sweep_output, sweep_format,
                            sweep_params, sweep_threads, verbose, quiet));
    });

#ifdef PULSIM_WITH_GRPC
    // Serve command
    auto* serve_cmd = app.add_subcommand("serve", "Start gRPC API server");
    std::string serve_address = "0.0.0.0:50051";
    bool serve_reflection = true;
    bool serve_metrics = true;
    int serve_max_sessions = 64;

    serve_cmd->add_option("-a,--address", serve_address,
        "Listen address (host:port)");
    serve_cmd->add_flag("--reflection,!--no-reflection", serve_reflection,
        "Enable gRPC reflection");
    serve_cmd->add_flag("--metrics,!--no-metrics", serve_metrics,
        "Enable Prometheus metrics");
    serve_cmd->add_option("--max-sessions", serve_max_sessions,
        "Maximum concurrent sessions");

    serve_cmd->callback([&]() {
        std::exit(cmd_serve(serve_address, serve_reflection, serve_metrics,
                            serve_max_sessions, verbose, quiet));
    });
#endif

    app.require_subcommand(1);

    CLI11_PARSE(app, argc, argv);

    return 0;
}
