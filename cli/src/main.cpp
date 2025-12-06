#include <CLI/CLI.hpp>
#include <spicelab/spicelab.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace spicelab;

void write_csv(const SimulationResult& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open output file: " + filename);
    }

    // Header
    file << "time";
    for (const auto& name : result.signal_names) {
        file << "," << name;
    }
    file << "\n";

    // Data
    file << std::scientific << std::setprecision(9);
    for (size_t i = 0; i < result.time.size(); ++i) {
        file << result.time[i];
        for (Index j = 0; j < result.data[i].size(); ++j) {
            file << "," << result.data[i](j);
        }
        file << "\n";
    }
}

void print_progress(Real time, Real tstop) {
    int percent = static_cast<int>(100.0 * time / tstop);
    std::cerr << "\rProgress: " << percent << "% (t=" << std::scientific
              << std::setprecision(3) << time << "s)" << std::flush;
}

// Sentinel values to detect if CLI option was explicitly provided
constexpr double CLI_SENTINEL = -1e99;
constexpr int CLI_SENTINEL_INT = -1;

int cmd_run(const std::string& netlist_file, const std::string& output_file,
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
        // Note: use_ic always comes from JSON only

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

        if (!output_file.empty()) {
            if (!quiet) {
                std::cerr << "Writing results to: " << output_file << std::endl;
            }
            write_csv(result, output_file);
        } else {
            // Write to stdout
            write_csv(result, "/dev/stdout");
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

int cmd_info(const std::string& netlist_file) {
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

int main(int argc, char** argv) {
    CLI::App app{"SpiceLab - High-performance circuit simulator"};
    app.set_version_flag("-V,--version", "SpiceLab 0.1.0");

    // Global options
    bool verbose = false;
    bool quiet = false;
    app.add_flag("-v,--verbose", verbose, "Verbose output");
    app.add_flag("-q,--quiet", quiet, "Quiet mode (errors only)");

    // Run command
    auto* run_cmd = app.add_subcommand("run", "Run transient simulation");
    std::string netlist_file;
    std::string output_file;
    // Use sentinel values so we can detect if CLI option was explicitly provided
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
    run_cmd->add_option("-o,--output", output_file, "Output file (CSV)");
    run_cmd->add_option("--tstop", cli_tstop, "Stop time (overrides JSON)");
    run_cmd->add_option("--dt", cli_dt, "Initial time step (overrides JSON)");
    run_cmd->add_option("--dtmax", cli_dtmax, "Maximum time step (overrides JSON)");
    run_cmd->add_option("--tstart", cli_tstart, "Start time (overrides JSON)");
    run_cmd->add_option("--abstol", cli_abstol, "Absolute tolerance (overrides JSON)");
    run_cmd->add_option("--reltol", cli_reltol, "Relative tolerance (overrides JSON)");
    run_cmd->add_option("--maxiter", cli_maxiter, "Max Newton iterations (overrides JSON)");

    run_cmd->callback([&]() {
        std::exit(cmd_run(netlist_file, output_file, cli_tstop, cli_dt, cli_dtmax,
                          cli_tstart, cli_abstol, cli_reltol, cli_maxiter,
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

    // Info command
    auto* info_cmd = app.add_subcommand("info", "Show circuit information");
    std::string info_file;
    info_cmd->add_option("netlist", info_file, "Netlist file (JSON format)")
        ->required()
        ->check(CLI::ExistingFile);
    info_cmd->callback([&]() {
        std::exit(cmd_info(info_file));
    });

    app.require_subcommand(1);

    CLI11_PARSE(app, argc, argv);

    return 0;
}
