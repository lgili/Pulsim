#include "pulsim/validation.hpp"
#include "pulsim/circuit.hpp"
#include "pulsim/metadata.hpp"
#include <algorithm>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <cmath>

namespace pulsim {

namespace {

// Union-Find data structure for loop detection
class UnionFind {
public:
    int find(const std::string& node) {
        if (parent_.find(node) == parent_.end()) {
            parent_[node] = node;
            rank_[node] = 0;
        }
        if (parent_[node] != node) {
            parent_[node] = find(parent_[node]);  // Path compression
        }
        return node_to_id(parent_[node]);
    }

    bool unite(const std::string& a, const std::string& b) {
        std::string root_a = find_root(a);
        std::string root_b = find_root(b);

        if (root_a == root_b) return false;  // Already in same set (loop found)

        // Union by rank
        if (rank_[root_a] < rank_[root_b]) {
            parent_[root_a] = root_b;
        } else if (rank_[root_a] > rank_[root_b]) {
            parent_[root_b] = root_a;
        } else {
            parent_[root_b] = root_a;
            rank_[root_a]++;
        }
        return true;
    }

    bool connected(const std::string& a, const std::string& b) {
        return find_root(a) == find_root(b);
    }

private:
    std::string find_root(const std::string& node) {
        if (parent_.find(node) == parent_.end()) {
            parent_[node] = node;
            rank_[node] = 0;
        }
        if (parent_[node] != node) {
            parent_[node] = find_root(parent_[node]);
        }
        return parent_[node];
    }

    int node_to_id(const std::string& node) {
        static std::unordered_map<std::string, int> ids;
        static int next_id = 0;
        auto it = ids.find(node);
        if (it == ids.end()) {
            ids[node] = next_id++;
        }
        return ids[node];
    }

    std::unordered_map<std::string, std::string> parent_;
    std::unordered_map<std::string, int> rank_;
};

// Get resistance value from component (for short circuit detection)
Real get_component_resistance(const Component& comp) {
    switch (comp.type()) {
        case ComponentType::Resistor: {
            auto& params = std::get<ResistorParams>(comp.params());
            return params.resistance;
        }
        case ComponentType::Switch: {
            auto& params = std::get<SwitchParams>(comp.params());
            return params.initial_state ? params.ron : params.roff;
        }
        case ComponentType::MOSFET: {
            auto& params = std::get<MOSFETParams>(comp.params());
            return params.rds_on > 0 ? params.rds_on : 1e6;  // Conservative default
        }
        case ComponentType::IGBT: {
            auto& params = std::get<IGBTParams>(comp.params());
            return params.rce_on;
        }
        default:
            return -1.0;  // Not a resistive element
    }
}

// Validate component parameters using the registry
void validate_component_parameters(const Component& comp, ValidationResult& result) {
    const auto& registry = ComponentRegistry::instance();
    const auto* meta = registry.get(comp.type());
    if (!meta) return;

    // Validate each parameter based on component type
    const auto& params = comp.params();

    // Check based on parameter type
    std::visit([&](const auto& p) {
        using T = std::decay_t<decltype(p)>;

        if constexpr (std::is_same_v<T, ResistorParams>) {
            std::string error;
            if (!registry.validate_parameter(comp.type(), "resistance", p.resistance, &error)) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER, error, comp.name());
            }
            if (p.resistance <= 0) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER,
                    "Resistance must be positive", comp.name());
            }
        }
        else if constexpr (std::is_same_v<T, CapacitorParams>) {
            std::string error;
            if (!registry.validate_parameter(comp.type(), "capacitance", p.capacitance, &error)) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER, error, comp.name());
            }
            if (p.capacitance <= 0) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER,
                    "Capacitance must be positive", comp.name());
            }
        }
        else if constexpr (std::is_same_v<T, InductorParams>) {
            std::string error;
            if (!registry.validate_parameter(comp.type(), "inductance", p.inductance, &error)) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER, error, comp.name());
            }
            if (p.inductance <= 0) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER,
                    "Inductance must be positive", comp.name());
            }
        }
        else if constexpr (std::is_same_v<T, DiodeParams>) {
            if (p.is <= 0) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER,
                    "Saturation current (Is) must be positive", comp.name());
            }
            if (p.n <= 0) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER,
                    "Ideality factor (n) must be positive", comp.name());
            }
        }
        else if constexpr (std::is_same_v<T, SwitchParams>) {
            if (p.ron <= 0) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER,
                    "On-resistance (Ron) must be positive", comp.name());
            }
            if (p.roff <= 0) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER,
                    "Off-resistance (Roff) must be positive", comp.name());
            }
            if (p.ron >= p.roff) {
                result.add_warning(DiagnosticCode::E_INVALID_PARAMETER,
                    "On-resistance should be less than off-resistance", comp.name());
            }
        }
        else if constexpr (std::is_same_v<T, MOSFETParams>) {
            if (p.vth < 0 && p.type == MOSFETType::NMOS) {
                result.add_warning(DiagnosticCode::E_INVALID_PARAMETER,
                    "NMOS threshold voltage is typically positive", comp.name());
            }
            if (p.kp <= 0) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER,
                    "Transconductance (Kp) must be positive", comp.name());
            }
            if (p.w <= 0 || p.l <= 0) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER,
                    "Channel width and length must be positive", comp.name());
            }
        }
        else if constexpr (std::is_same_v<T, IGBTParams>) {
            if (p.vth <= 0) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER,
                    "Gate threshold voltage must be positive", comp.name());
            }
            if (p.vce_sat < 0) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER,
                    "Saturation voltage cannot be negative", comp.name());
            }
        }
        else if constexpr (std::is_same_v<T, TransformerParams>) {
            if (p.turns_ratio <= 0) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER,
                    "Turns ratio must be positive", comp.name());
            }
            if (p.lm < 0 || p.ll1 < 0 || p.ll2 < 0) {
                result.add_error(DiagnosticCode::E_INVALID_PARAMETER,
                    "Inductances cannot be negative", comp.name());
            }
        }
    }, params);
}

}  // anonymous namespace

bool ValidationResult::has_errors() const {
    return std::any_of(diagnostics.begin(), diagnostics.end(),
        [](const Diagnostic& d) { return d.severity == DiagnosticSeverity::Error; });
}

bool ValidationResult::has_warnings() const {
    return std::any_of(diagnostics.begin(), diagnostics.end(),
        [](const Diagnostic& d) { return d.severity == DiagnosticSeverity::Warning; });
}

std::vector<Diagnostic> ValidationResult::errors() const {
    std::vector<Diagnostic> result;
    std::copy_if(diagnostics.begin(), diagnostics.end(), std::back_inserter(result),
        [](const Diagnostic& d) { return d.severity == DiagnosticSeverity::Error; });
    return result;
}

std::vector<Diagnostic> ValidationResult::warnings() const {
    std::vector<Diagnostic> result;
    std::copy_if(diagnostics.begin(), diagnostics.end(), std::back_inserter(result),
        [](const Diagnostic& d) { return d.severity == DiagnosticSeverity::Warning; });
    return result;
}

std::vector<Diagnostic> ValidationResult::infos() const {
    std::vector<Diagnostic> result;
    std::copy_if(diagnostics.begin(), diagnostics.end(), std::back_inserter(result),
        [](const Diagnostic& d) { return d.severity == DiagnosticSeverity::Info; });
    return result;
}

void ValidationResult::add_error(DiagnosticCode code, const std::string& message,
                                 const std::string& component, const std::string& node) {
    is_valid = false;
    Diagnostic d;
    d.severity = DiagnosticSeverity::Error;
    d.code = code;
    d.message = message;
    d.component_name = component;
    d.node_name = node;
    diagnostics.push_back(std::move(d));
}

void ValidationResult::add_warning(DiagnosticCode code, const std::string& message,
                                   const std::string& component, const std::string& node) {
    Diagnostic d;
    d.severity = DiagnosticSeverity::Warning;
    d.code = code;
    d.message = message;
    d.component_name = component;
    d.node_name = node;
    diagnostics.push_back(std::move(d));
}

void ValidationResult::add_info(DiagnosticCode code, const std::string& message,
                                const std::string& component, const std::string& node) {
    Diagnostic d;
    d.severity = DiagnosticSeverity::Info;
    d.code = code;
    d.message = message;
    d.component_name = component;
    d.node_name = node;
    diagnostics.push_back(std::move(d));
}

ValidationResult validate_circuit(const Circuit& circuit) {
    ValidationResult result;

    const auto& components = circuit.components();

    // Check for empty circuit
    if (components.empty()) {
        result.add_error(DiagnosticCode::E_NO_COMPONENTS,
                        "Circuit has no components");
        return result;
    }

    // Collect all nodes and check for ground
    std::unordered_set<std::string> all_nodes;
    bool has_ground = false;

    // Count connections per node for floating node detection
    std::unordered_map<std::string, int> node_connections;
    std::unordered_map<std::string, std::vector<std::string>> node_to_components;

    // Check for duplicate component names
    std::unordered_set<std::string> component_names;

    // Tracking for loop detection
    std::vector<const Component*> voltage_sources;
    std::vector<const Component*> inductors;

    for (const auto& comp : components) {
        // Check duplicate names
        if (component_names.count(comp.name())) {
            result.add_error(DiagnosticCode::E_DUPLICATE_NAME,
                            "Duplicate component name: " + comp.name(),
                            comp.name());
        }
        component_names.insert(comp.name());

        // Collect nodes
        for (const auto& node : comp.nodes()) {
            all_nodes.insert(node);
            node_connections[node]++;
            node_to_components[node].push_back(comp.name());

            if (node == "0" || node == "GND" || node == "gnd") {
                has_ground = true;
            }
        }

        // Track voltage sources and inductors for loop detection
        if (comp.type() == ComponentType::VoltageSource) {
            voltage_sources.push_back(&comp);
        }
        if (comp.type() == ComponentType::Inductor) {
            inductors.push_back(&comp);
        }

        // Check for ideal switches (info)
        if (comp.type() == ComponentType::Switch) {
            result.add_info(DiagnosticCode::I_IDEAL_SWITCH,
                          "Using ideal switch model for " + comp.name(),
                          comp.name());
        }

        // Validate component parameters (Task 5.11)
        validate_component_parameters(comp, result);
    }

    // Check for ground node
    if (!has_ground) {
        result.add_error(DiagnosticCode::E_NO_GROUND,
                        "Circuit has no ground node (0 or GND)");
    }

    // Check for floating nodes (nodes with only one connection)
    for (const auto& [node, count] : node_connections) {
        if (count == 1 && node != "0" && node != "GND" && node != "gnd") {
            result.add_warning(DiagnosticCode::W_FLOATING_NODE,
                             "Node '" + node + "' has only one connection (floating)",
                             "", node);
        }
    }

    // Check for voltage source loops (simplified: two voltage sources sharing both nodes)
    if (voltage_sources.size() >= 2) {
        for (size_t i = 0; i < voltage_sources.size(); i++) {
            for (size_t j = i + 1; j < voltage_sources.size(); j++) {
                const auto* vs1 = voltage_sources[i];
                const auto* vs2 = voltage_sources[j];

                const auto& n1 = vs1->nodes();
                const auto& n2 = vs2->nodes();
                if (n1.size() >= 2 && n2.size() >= 2) {
                    // Check if they share the same nodes (potential loop)
                    bool share_both = (n1[0] == n2[0] && n1[1] == n2[1]) ||
                                     (n1[0] == n2[1] && n1[1] == n2[0]);
                    if (share_both) {
                        Diagnostic d;
                        d.severity = DiagnosticSeverity::Error;
                        d.code = DiagnosticCode::E_VOLTAGE_SOURCE_LOOP;
                        d.message = "Voltage sources " + vs1->name() + " and " +
                                   vs2->name() + " form a loop (parallel connection)";
                        d.related_components = {vs1->name(), vs2->name()};
                        result.diagnostics.push_back(std::move(d));
                        result.is_valid = false;
                    }
                }
            }
        }
    }

    // Task 5.9: Check for inductor/voltage source loops
    // Inductors and voltage sources in a loop without resistance cause algebraic problems
    // Use Union-Find to detect if inductors and voltage sources form a connected component
    if (!voltage_sources.empty() && !inductors.empty()) {
        UnionFind uf;

        // Build connectivity graph using only V sources and inductors
        for (const auto* vs : voltage_sources) {
            const auto& nodes = vs->nodes();
            if (nodes.size() >= 2) {
                uf.unite(nodes[0], nodes[1]);
            }
        }
        for (const auto* ind : inductors) {
            const auto& nodes = ind->nodes();
            if (nodes.size() >= 2) {
                // Check if adding this inductor creates a loop with V sources
                if (uf.connected(nodes[0], nodes[1])) {
                    // This inductor connects two nodes already connected through V sources/inductors
                    Diagnostic d;
                    d.severity = DiagnosticSeverity::Error;
                    d.code = DiagnosticCode::E_INDUCTOR_LOOP;
                    d.message = "Inductor " + ind->name() +
                               " forms a loop with voltage sources/inductors (no series resistance)";
                    d.component_name = ind->name();
                    // Find related voltage sources in the same component
                    for (const auto* vs : voltage_sources) {
                        const auto& vs_nodes = vs->nodes();
                        if (vs_nodes.size() >= 2 &&
                            (uf.connected(vs_nodes[0], nodes[0]) || uf.connected(vs_nodes[0], nodes[1]))) {
                            d.related_components.push_back(vs->name());
                        }
                    }
                    d.related_components.push_back(ind->name());
                    result.diagnostics.push_back(std::move(d));
                    result.is_valid = false;
                }
                uf.unite(nodes[0], nodes[1]);
            }
        }
    }

    // Task 5.10: Check for short circuits (very low resistance paths between nodes)
    // Look for resistors with extremely low values that might indicate shorts
    const Real SHORT_CIRCUIT_THRESHOLD = 1e-6;  // 1 micro-ohm

    for (const auto& comp : components) {
        Real resistance = get_component_resistance(comp);
        if (resistance > 0 && resistance < SHORT_CIRCUIT_THRESHOLD) {
            Diagnostic d;
            d.severity = DiagnosticSeverity::Warning;
            d.code = DiagnosticCode::W_SHORT_CIRCUIT;
            d.message = "Component " + comp.name() + " has very low resistance (" +
                       std::to_string(resistance) + " ohm) - potential short circuit";
            d.component_name = comp.name();
            result.diagnostics.push_back(std::move(d));
        }
    }

    // Also check for direct connections between voltage sources and ground with only low-R path
    // Build adjacency list for graph traversal
    std::unordered_map<std::string, std::vector<std::pair<std::string, Real>>> adj;
    for (const auto& comp : components) {
        const auto& nodes = comp.nodes();
        if (nodes.size() >= 2 && comp.type() == ComponentType::Resistor) {
            Real r = get_component_resistance(comp);
            if (r > 0) {
                adj[nodes[0]].push_back({nodes[1], r});
                adj[nodes[1]].push_back({nodes[0], r});
            }
        }
    }

    // For each voltage source, check if there's a very low resistance path to its other terminal
    for (const auto* vs : voltage_sources) {
        const auto& nodes = vs->nodes();
        if (nodes.size() >= 2) {
            // Simple BFS to find minimum resistance path
            std::unordered_map<std::string, Real> min_resistance;
            std::queue<std::string> q;
            q.push(nodes[0]);
            min_resistance[nodes[0]] = 0;

            while (!q.empty()) {
                std::string curr = q.front();
                q.pop();

                for (const auto& [next, r] : adj[curr]) {
                    Real new_r = min_resistance[curr] + r;
                    if (min_resistance.find(next) == min_resistance.end() ||
                        new_r < min_resistance[next]) {
                        min_resistance[next] = new_r;
                        q.push(next);
                    }
                }
            }

            // Check if there's a path with very low resistance
            auto it = min_resistance.find(nodes[1]);
            if (it != min_resistance.end() && it->second < SHORT_CIRCUIT_THRESHOLD) {
                Diagnostic d;
                d.severity = DiagnosticSeverity::Warning;
                d.code = DiagnosticCode::W_SHORT_CIRCUIT;
                d.message = "Very low resistance path (" + std::to_string(it->second) +
                           " ohm) across voltage source " + vs->name();
                d.component_name = vs->name();
                result.diagnostics.push_back(std::move(d));
            }
        }
    }

    return result;
}

std::string diagnostic_code_description(DiagnosticCode code) {
    switch (code) {
        case DiagnosticCode::E_NO_GROUND:
            return "No ground reference node found in circuit";
        case DiagnosticCode::E_VOLTAGE_SOURCE_LOOP:
            return "Voltage sources connected in a loop";
        case DiagnosticCode::E_INDUCTOR_LOOP:
            return "Inductors form a loop with voltage sources";
        case DiagnosticCode::E_NO_DC_PATH:
            return "Node has no DC path to ground";
        case DiagnosticCode::E_INVALID_PARAMETER:
            return "Component parameter out of valid range";
        case DiagnosticCode::E_UNKNOWN_NODE:
            return "Referenced node does not exist";
        case DiagnosticCode::E_DUPLICATE_NAME:
            return "Duplicate component name";
        case DiagnosticCode::E_NO_COMPONENTS:
            return "Circuit has no components";
        case DiagnosticCode::W_FLOATING_NODE:
            return "Node with single connection may be floating";
        case DiagnosticCode::W_SHORT_CIRCUIT:
            return "Very low impedance path detected";
        case DiagnosticCode::W_HIGH_VOLTAGE:
            return "Unusually high voltage may occur";
        case DiagnosticCode::W_MISSING_IC:
            return "No initial condition specified for energy storage element";
        case DiagnosticCode::W_LARGE_TIMESTEP:
            return "Timestep may be too large for circuit dynamics";
        case DiagnosticCode::I_IDEAL_SWITCH:
            return "Using ideal switch model";
        case DiagnosticCode::I_NO_LOSS_MODEL:
            return "Loss model not available for this component";
        case DiagnosticCode::I_PARALLEL_SOURCES:
            return "Parallel voltage sources detected";
        default:
            return "Unknown diagnostic code";
    }
}

}  // namespace pulsim
