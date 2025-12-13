#include "pulsim/parser.hpp"
#include <fstream>
#include <sstream>
#include <cctype>
#include <cmath>

namespace pulsim {

using json = nlohmann::json;

Real NetlistParser::parse_value_with_suffix(const std::string& str) {
    if (str.empty()) {
        throw std::invalid_argument("Empty value string");
    }

    // Find where the numeric part ends
    size_t suffix_start = str.size();
    for (size_t i = 0; i < str.size(); ++i) {
        char c = str[i];
        if (!std::isdigit(c) && c != '.' && c != '-' && c != '+' && c != 'e' && c != 'E') {
            suffix_start = i;
            break;
        }
    }

    Real value = std::stod(str.substr(0, suffix_start));

    if (suffix_start < str.size()) {
        std::string suffix = str.substr(suffix_start);
        // Convert to lowercase for comparison
        for (auto& c : suffix) c = static_cast<char>(std::tolower(c));

        // Handle SI prefixes
        if (suffix == "f" || suffix == "femto") value *= 1e-15;
        else if (suffix == "p" || suffix == "pico") value *= 1e-12;
        else if (suffix == "n" || suffix == "nano") value *= 1e-9;
        else if (suffix == "u" || suffix == "micro") value *= 1e-6;
        else if (suffix == "m" || suffix == "milli") value *= 1e-3;
        else if (suffix == "k" || suffix == "kilo") value *= 1e3;
        else if (suffix == "meg" || suffix == "mega") value *= 1e6;
        else if (suffix == "g" || suffix == "giga") value *= 1e9;
        else if (suffix == "t" || suffix == "tera") value *= 1e12;
        // Ignore unrecognized suffixes (could be units like "ohm", "F", "H")
    }

    return value;
}

ParseResult<Waveform> NetlistParser::parse_waveform(const json& j) {
    if (j.is_number()) {
        // Simple DC value
        return Waveform{DCWaveform{j.get<Real>()}};
    }

    if (j.is_string()) {
        // DC value with suffix
        return Waveform{DCWaveform{parse_value_with_suffix(j.get<std::string>())}};
    }

    if (!j.is_object()) {
        return ParseError{"Waveform must be a number, string, or object"};
    }

    std::string type = j.value("type", "dc");

    if (type == "dc") {
        Real value = 0.0;
        if (j.contains("value")) {
            if (j["value"].is_string()) {
                value = parse_value_with_suffix(j["value"].get<std::string>());
            } else {
                value = j["value"].get<Real>();
            }
        }
        return Waveform{DCWaveform{value}};
    }

    if (type == "pulse") {
        PulseWaveform pulse;
        pulse.v1 = j.value("v1", 0.0);
        pulse.v2 = j.value("v2", 1.0);
        pulse.td = j.value("td", 0.0);
        pulse.tr = j.value("tr", 1e-9);
        pulse.tf = j.value("tf", 1e-9);
        pulse.pw = j.value("pw", 0.5e-3);
        pulse.period = j.value("period", 1e-3);
        return Waveform{pulse};
    }

    if (type == "sin" || type == "sine") {
        SineWaveform sine;
        sine.offset = j.value("offset", 0.0);
        sine.amplitude = j.value("amplitude", 1.0);
        sine.frequency = j.value("frequency", 1000.0);
        sine.delay = j.value("delay", 0.0);
        sine.damping = j.value("damping", 0.0);
        return Waveform{sine};
    }

    if (type == "pwl") {
        PWLWaveform pwl;
        if (!j.contains("points") || !j["points"].is_array()) {
            return ParseError{"PWL waveform requires 'points' array"};
        }
        for (const auto& pt : j["points"]) {
            if (!pt.is_array() || pt.size() != 2) {
                return ParseError{"PWL points must be [time, value] pairs"};
            }
            pwl.points.emplace_back(pt[0].get<Real>(), pt[1].get<Real>());
        }
        return Waveform{pwl};
    }

    if (type == "pwm") {
        PWMWaveform pwm;
        pwm.v_off = j.value("v_off", 0.0);
        pwm.v_on = j.value("v_on", 5.0);
        pwm.frequency = j.value("frequency", 10e3);
        pwm.duty = j.value("duty", 0.5);
        pwm.dead_time = j.value("dead_time", 0.0);
        pwm.phase = j.value("phase", 0.0);
        pwm.complementary = j.value("complementary", false);
        return Waveform{pwm};
    }

    return ParseError{"Unknown waveform type: " + type};
}

ParseResult<Circuit> NetlistParser::parse_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return ParseError{"Cannot open file: " + path.string()};
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return parse_string(buffer.str());
}

ParseResult<Circuit> NetlistParser::parse_string(const std::string& content) {
    return parse_json(content);
}

ParseResult<Circuit> NetlistParser::parse_json(const std::string& content) {
    json j;
    try {
        j = json::parse(content);
    } catch (const json::parse_error& e) {
        return ParseError{
            "JSON parse error: " + std::string(e.what()),
            static_cast<int>(e.byte)
        };
    }

    Circuit circuit;

    // Parse components
    if (!j.contains("components") || !j["components"].is_array()) {
        return ParseError{"Missing or invalid 'components' array"};
    }

    for (const auto& comp : j["components"]) {
        if (!comp.is_object()) {
            return ParseError{"Component must be an object"};
        }

        std::string type = comp.value("type", "");
        std::string name = comp.value("name", "");

        if (type.empty() || name.empty()) {
            return ParseError{"Component must have 'type' and 'name'"};
        }

        auto get_node = [&comp](const std::string& key) -> std::string {
            if (comp.contains(key)) {
                if (comp[key].is_string()) {
                    return comp[key].get<std::string>();
                } else if (comp[key].is_number()) {
                    return std::to_string(comp[key].get<int>());
                }
            }
            return "";
        };

        try {
            if (type == "resistor" || type == "R") {
                std::string n1 = get_node("n1");
                std::string n2 = get_node("n2");
                if (n1.empty() || n2.empty()) {
                    return ParseError{"Resistor " + name + " requires 'n1' and 'n2'"};
                }
                Real value = 0.0;
                if (comp["value"].is_string()) {
                    value = parse_value_with_suffix(comp["value"].get<std::string>());
                } else {
                    value = comp["value"].get<Real>();
                }
                circuit.add_resistor(name, n1, n2, value);
            }
            else if (type == "capacitor" || type == "C") {
                std::string n1 = get_node("n1");
                std::string n2 = get_node("n2");
                if (n1.empty() || n2.empty()) {
                    return ParseError{"Capacitor " + name + " requires 'n1' and 'n2'"};
                }
                Real value = 0.0;
                if (comp["value"].is_string()) {
                    value = parse_value_with_suffix(comp["value"].get<std::string>());
                } else {
                    value = comp["value"].get<Real>();
                }
                Real ic = comp.value("ic", 0.0);
                circuit.add_capacitor(name, n1, n2, value, ic);
            }
            else if (type == "inductor" || type == "L") {
                std::string n1 = get_node("n1");
                std::string n2 = get_node("n2");
                if (n1.empty() || n2.empty()) {
                    return ParseError{"Inductor " + name + " requires 'n1' and 'n2'"};
                }
                Real value = 0.0;
                if (comp["value"].is_string()) {
                    value = parse_value_with_suffix(comp["value"].get<std::string>());
                } else {
                    value = comp["value"].get<Real>();
                }
                Real ic = comp.value("ic", 0.0);
                circuit.add_inductor(name, n1, n2, value, ic);
            }
            else if (type == "voltage_source" || type == "vsource" || type == "V") {
                std::string npos = get_node("npos");
                std::string nneg = get_node("nneg");
                if (npos.empty()) npos = get_node("n1");
                if (nneg.empty()) nneg = get_node("n2");
                if (npos.empty() || nneg.empty()) {
                    return ParseError{"Voltage source " + name + " requires nodes"};
                }
                auto waveform_result = parse_waveform(comp.value("waveform", json(0.0)));
                if (!waveform_result) {
                    return waveform_result.error();
                }
                circuit.add_voltage_source(name, npos, nneg, *waveform_result);
            }
            else if (type == "current_source" || type == "isource" || type == "I") {
                std::string npos = get_node("npos");
                std::string nneg = get_node("nneg");
                if (npos.empty()) npos = get_node("n1");
                if (nneg.empty()) nneg = get_node("n2");
                if (npos.empty() || nneg.empty()) {
                    return ParseError{"Current source " + name + " requires nodes"};
                }
                auto waveform_result = parse_waveform(comp.value("waveform", json(0.0)));
                if (!waveform_result) {
                    return waveform_result.error();
                }
                circuit.add_current_source(name, npos, nneg, *waveform_result);
            }
            else if (type == "diode" || type == "D") {
                std::string anode = get_node("anode");
                std::string cathode = get_node("cathode");
                if (anode.empty()) anode = get_node("n1");
                if (cathode.empty()) cathode = get_node("n2");
                if (anode.empty() || cathode.empty()) {
                    return ParseError{"Diode " + name + " requires 'anode' and 'cathode'"};
                }
                DiodeParams params;
                params.is = comp.value("is", 1e-14);
                params.n = comp.value("n", 1.0);
                params.ideal = comp.value("ideal", true);
                circuit.add_diode(name, anode, cathode, params);
            }
            else if (type == "switch" || type == "S") {
                std::string n1 = get_node("n1");
                std::string n2 = get_node("n2");
                std::string ctrl_pos = get_node("ctrl_pos");
                std::string ctrl_neg = get_node("ctrl_neg");
                if (ctrl_pos.empty()) ctrl_pos = get_node("ctrl");
                if (ctrl_neg.empty()) ctrl_neg = "0";  // Default control negative to ground
                if (n1.empty() || n2.empty() || ctrl_pos.empty()) {
                    return ParseError{"Switch " + name + " requires 'n1', 'n2', and 'ctrl_pos'"};
                }
                SwitchParams params;
                params.ron = comp.value("ron", 1e-3);
                params.roff = comp.value("roff", 1e9);
                params.vth = comp.value("vth", 0.5);
                params.initial_state = comp.value("initial_state", false);
                circuit.add_switch(name, n1, n2, ctrl_pos, ctrl_neg, params);
            }
            else if (type == "mosfet" || type == "nmos" || type == "pmos" || type == "M") {
                std::string drain = get_node("drain");
                std::string gate = get_node("gate");
                std::string source = get_node("source");
                if (drain.empty()) drain = get_node("d");
                if (gate.empty()) gate = get_node("g");
                if (source.empty()) source = get_node("s");
                if (drain.empty() || gate.empty() || source.empty()) {
                    return ParseError{"MOSFET " + name + " requires 'drain', 'gate', and 'source'"};
                }
                MOSFETParams params;
                if (type == "pmos") {
                    params.type = MOSFETType::PMOS;
                } else {
                    params.type = comp.value("pmos", false) ? MOSFETType::PMOS : MOSFETType::NMOS;
                }
                params.vth = comp.value("vth", 2.0);
                params.kp = comp.value("kp", 20e-6);
                params.lambda = comp.value("lambda", 0.0);
                params.w = comp.value("w", 100e-6);
                params.l = comp.value("l", 10e-6);
                params.body_diode = comp.value("body_diode", false);
                params.is_body = comp.value("is_body", 1e-14);
                params.n_body = comp.value("n_body", 1.0);
                params.cgs = comp.value("cgs", 0.0);
                params.cgd = comp.value("cgd", 0.0);
                params.cds = comp.value("cds", 0.0);
                params.rds_on = comp.value("rds_on", 0.0);
                params.rds_off = comp.value("rds_off", 1e9);
                circuit.add_mosfet(name, drain, gate, source, params);
            }
            else if (type == "transformer" || type == "T") {
                std::string p1 = get_node("p1");
                std::string p2 = get_node("p2");
                std::string s1 = get_node("s1");
                std::string s2 = get_node("s2");
                if (p1.empty() || p2.empty() || s1.empty() || s2.empty()) {
                    return ParseError{"Transformer " + name + " requires 'p1', 'p2', 's1', 's2'"};
                }
                TransformerParams params;
                params.turns_ratio = comp.value("turns_ratio", 1.0);
                params.lm = comp.value("lm", 0.0);
                params.ll1 = comp.value("ll1", 0.0);
                params.ll2 = comp.value("ll2", 0.0);
                circuit.add_transformer(name, p1, p2, s1, s2, params);
            }
            else {
                return ParseError{"Unknown component type: " + type};
            }

            // Parse schematic position if present
            if (comp.contains("position") && comp["position"].is_object()) {
                const auto& pos = comp["position"];
                SchematicPosition schematic_pos;
                schematic_pos.x = pos.value("x", 0.0);
                schematic_pos.y = pos.value("y", 0.0);
                schematic_pos.orientation = pos.value("orientation", 0);
                schematic_pos.mirrored = pos.value("mirrored", false);
                circuit.set_position(name, schematic_pos);
            }
        } catch (const std::exception& e) {
            return ParseError{"Error parsing component " + name + ": " + e.what()};
        }
    }

    std::string error;
    if (!circuit.validate(error)) {
        return ParseError{error};
    }

    return circuit;
}

ParseResult<SimulationOptions> NetlistParser::parse_options(const std::string& content) {
    json j;
    try {
        j = json::parse(content);
    } catch (const json::parse_error& e) {
        return ParseError{"JSON parse error: " + std::string(e.what())};
    }

    SimulationOptions opts;

    if (j.contains("simulation")) {
        const auto& sim = j["simulation"];
        opts.tstart = sim.value("tstart", 0.0);
        opts.tstop = sim.value("tstop", 1.0);
        opts.dt = sim.value("dt", 1e-6);
        opts.dtmin = sim.value("dtmin", 1e-15);
        opts.dtmax = sim.value("dtmax", 1e-3);
        opts.abstol = sim.value("abstol", 1e-12);
        opts.reltol = sim.value("reltol", 1e-3);
        opts.max_newton_iterations = sim.value("maxiter", 50);
        opts.use_ic = sim.value("uic", false);

        if (sim.contains("outputs") && sim["outputs"].is_array()) {
            for (const auto& sig : sim["outputs"]) {
                opts.output_signals.push_back(sig.get<std::string>());
            }
        }
    }

    return opts;
}

ParseResult<SimulationOptions> NetlistParser::parse_simulation_options(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return ParseError{"Cannot open file: " + path.string()};
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return parse_options(buffer.str());
}

std::string NetlistParser::to_json(const Circuit& circuit, bool include_positions) {
    json j;
    json components = json::array();

    auto waveform_to_json = [](const Waveform& wf) -> json {
        return std::visit([](auto&& w) -> json {
            using T = std::decay_t<decltype(w)>;
            if constexpr (std::is_same_v<T, DCWaveform>) {
                return w.value;
            } else if constexpr (std::is_same_v<T, PulseWaveform>) {
                return json{
                    {"type", "pulse"},
                    {"v1", w.v1}, {"v2", w.v2},
                    {"td", w.td}, {"tr", w.tr}, {"tf", w.tf},
                    {"pw", w.pw}, {"period", w.period}
                };
            } else if constexpr (std::is_same_v<T, SineWaveform>) {
                return json{
                    {"type", "sin"},
                    {"offset", w.offset}, {"amplitude", w.amplitude},
                    {"frequency", w.frequency}, {"delay", w.delay},
                    {"damping", w.damping}
                };
            } else if constexpr (std::is_same_v<T, PWLWaveform>) {
                json pts = json::array();
                for (const auto& [t, v] : w.points) {
                    pts.push_back(json::array({t, v}));
                }
                return json{{"type", "pwl"}, {"points", pts}};
            } else if constexpr (std::is_same_v<T, PWMWaveform>) {
                return json{
                    {"type", "pwm"},
                    {"v_off", w.v_off}, {"v_on", w.v_on},
                    {"frequency", w.frequency}, {"duty", w.duty},
                    {"dead_time", w.dead_time}, {"phase", w.phase},
                    {"complementary", w.complementary}
                };
            }
            return json{};
        }, wf);
    };

    for (const auto& comp : circuit.components()) {
        json c;
        c["name"] = comp.name();
        const auto& nodes = comp.nodes();

        switch (comp.type()) {
            case ComponentType::Resistor: {
                c["type"] = "resistor";
                c["n1"] = nodes[0];
                c["n2"] = nodes[1];
                const auto& p = std::get<ResistorParams>(comp.params());
                c["value"] = p.resistance;
                break;
            }
            case ComponentType::Capacitor: {
                c["type"] = "capacitor";
                c["n1"] = nodes[0];
                c["n2"] = nodes[1];
                const auto& p = std::get<CapacitorParams>(comp.params());
                c["value"] = p.capacitance;
                if (p.initial_voltage != 0.0) c["ic"] = p.initial_voltage;
                break;
            }
            case ComponentType::Inductor: {
                c["type"] = "inductor";
                c["n1"] = nodes[0];
                c["n2"] = nodes[1];
                const auto& p = std::get<InductorParams>(comp.params());
                c["value"] = p.inductance;
                if (p.initial_current != 0.0) c["ic"] = p.initial_current;
                break;
            }
            case ComponentType::VoltageSource: {
                c["type"] = "voltage_source";
                c["npos"] = nodes[0];
                c["nneg"] = nodes[1];
                const auto& p = std::get<VoltageSourceParams>(comp.params());
                c["waveform"] = waveform_to_json(p.waveform);
                break;
            }
            case ComponentType::CurrentSource: {
                c["type"] = "current_source";
                c["npos"] = nodes[0];
                c["nneg"] = nodes[1];
                const auto& p = std::get<CurrentSourceParams>(comp.params());
                c["waveform"] = waveform_to_json(p.waveform);
                break;
            }
            case ComponentType::Diode: {
                c["type"] = "diode";
                c["anode"] = nodes[0];
                c["cathode"] = nodes[1];
                const auto& p = std::get<DiodeParams>(comp.params());
                c["is"] = p.is;
                c["n"] = p.n;
                c["ideal"] = p.ideal;
                break;
            }
            case ComponentType::Switch: {
                c["type"] = "switch";
                c["n1"] = nodes[0];
                c["n2"] = nodes[1];
                c["ctrl_pos"] = nodes[2];
                c["ctrl_neg"] = nodes[3];
                const auto& p = std::get<SwitchParams>(comp.params());
                c["ron"] = p.ron;
                c["roff"] = p.roff;
                c["vth"] = p.vth;
                c["initial_state"] = p.initial_state;
                break;
            }
            case ComponentType::MOSFET: {
                c["type"] = "mosfet";
                c["drain"] = nodes[0];
                c["gate"] = nodes[1];
                c["source"] = nodes[2];
                const auto& p = std::get<MOSFETParams>(comp.params());
                c["pmos"] = (p.type == MOSFETType::PMOS);
                c["vth"] = p.vth;
                c["rds_on"] = p.rds_on;
                break;
            }
            case ComponentType::IGBT: {
                c["type"] = "igbt";
                c["collector"] = nodes[0];
                c["gate"] = nodes[1];
                c["emitter"] = nodes[2];
                const auto& p = std::get<IGBTParams>(comp.params());
                c["vth"] = p.vth;
                c["vce_sat"] = p.vce_sat;
                c["rce_on"] = p.rce_on;
                break;
            }
            case ComponentType::Transformer: {
                c["type"] = "transformer";
                c["p1"] = nodes[0];
                c["p2"] = nodes[1];
                c["s1"] = nodes[2];
                c["s2"] = nodes[3];
                const auto& p = std::get<TransformerParams>(comp.params());
                c["turns_ratio"] = p.turns_ratio;
                c["lm"] = p.lm;
                break;
            }
            default:
                c["type"] = "unknown";
                break;
        }

        // Add position if present and requested
        if (include_positions && circuit.has_position(comp.name())) {
            auto pos = circuit.get_position(comp.name());
            if (pos) {
                c["position"] = json{
                    {"x", pos->x},
                    {"y", pos->y},
                    {"orientation", pos->orientation},
                    {"mirrored", pos->mirrored}
                };
            }
        }

        components.push_back(c);
    }

    j["components"] = components;
    return j.dump(2);  // Pretty print with 2 spaces
}

}  // namespace pulsim
