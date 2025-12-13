#pragma once

#include "pulsim/types.hpp"
#include <memory>
#include <optional>
#include <variant>

namespace pulsim {

// Forward declarations
class Component;

// Source waveform types
struct DCWaveform {
    Real value;
};

struct PulseWaveform {
    Real v1;      // Initial value
    Real v2;      // Pulsed value
    Real td;      // Delay time
    Real tr;      // Rise time
    Real tf;      // Fall time
    Real pw;      // Pulse width
    Real period;  // Period
};

struct SineWaveform {
    Real offset;     // DC offset
    Real amplitude;  // Peak amplitude
    Real frequency;  // Frequency in Hz
    Real delay;      // Delay time
    Real damping;    // Damping factor (theta)
};

struct PWLWaveform {
    std::vector<std::pair<Real, Real>> points;  // (time, value) pairs
};

// PWM waveform with dead-time support for power electronics
struct PWMWaveform {
    Real v_off = 0.0;       // Off-state voltage (typically 0V)
    Real v_on = 5.0;        // On-state voltage (gate drive voltage)
    Real frequency = 10e3;  // Switching frequency (Hz)
    Real duty = 0.5;        // Duty cycle (0.0 to 1.0)
    Real dead_time = 0.0;   // Dead-time in seconds (inserted at both edges)
    Real phase = 0.0;       // Phase offset (0.0 to 1.0, fraction of period)
    bool complementary = false;  // If true, output is inverted (for low-side drive)

    // Derived values
    Real period() const { return 1.0 / frequency; }
    Real t_on() const { return period() * duty; }  // On-time before dead-time adjustment
};

using Waveform = std::variant<DCWaveform, PulseWaveform, SineWaveform, PWLWaveform, PWMWaveform>;

/**
 * @brief Schematic position for GUI layout persistence.
 *
 * Stores the visual position and orientation of a component in a schematic
 * editor. This information is stored in the Circuit and can be exported/imported
 * via JSON netlists.
 *
 * @section position_usage Usage Example
 * @code
 * Circuit circuit;
 * circuit.add_resistor("R1", "a", "b", 1000.0);
 *
 * // Set position after placing component in GUI
 * SchematicPosition pos;
 * pos.x = 100.0;
 * pos.y = 50.0;
 * pos.orientation = 90;  // Rotated 90 degrees
 * circuit.set_position("R1", pos);
 *
 * // Export to JSON (positions included)
 * std::string json = NetlistParser::to_json(circuit, true);
 *
 * // Later, import and retrieve positions
 * Circuit loaded = NetlistParser::parse_string(json).value();
 * auto pos = loaded.get_position("R1");  // Returns optional
 * @endcode
 */
struct SchematicPosition {
    double x = 0.0;           ///< X coordinate in schematic units
    double y = 0.0;           ///< Y coordinate in schematic units
    int orientation = 0;      ///< Rotation: 0, 90, 180, or 270 degrees
    bool mirrored = false;    ///< True if horizontally mirrored
};

// Component parameters
struct ResistorParams {
    Real resistance;
    Real tc1 = 0.0;  // Temperature coefficient 1
    Real tc2 = 0.0;  // Temperature coefficient 2
};

struct CapacitorParams {
    Real capacitance;
    Real initial_voltage = 0.0;
};

struct InductorParams {
    Real inductance;
    Real initial_current = 0.0;
};

struct VoltageSourceParams {
    Waveform waveform;
    Real internal_resistance = 0.0;
};

struct CurrentSourceParams {
    Waveform waveform;
};

struct DiodeParams {
    Real is = 1e-14;   // Saturation current
    Real n = 1.0;      // Ideality factor
    Real rs = 0.0;     // Series resistance
    Real vt = 0.026;   // Thermal voltage (kT/q at 300K)
    bool ideal = true; // Use ideal model if true

    // Junction capacitance parameters
    Real cj0 = 0.0;    // Zero-bias junction capacitance (F)
    Real vj = 0.7;     // Junction potential (V)
    Real m = 0.5;      // Grading coefficient (0.33 for linearly graded, 0.5 for abrupt)
    Real tt = 0.0;     // Transit time (s) for diffusion capacitance
    Real bv = 100.0;   // Reverse breakdown voltage (V)
    Real ibv = 1e-10;  // Current at breakdown voltage (A)
};

struct SwitchParams {
    Real ron = 1e-3;   // On resistance
    Real roff = 1e9;   // Off resistance
    Real vth = 0.5;    // Threshold voltage
    bool initial_state = false;  // false = open, true = closed
};

// MOSFET type
enum class MOSFETType { NMOS, PMOS };

struct MOSFETParams {
    MOSFETType type = MOSFETType::NMOS;

    // Level 1 (Shichman-Hodges) parameters
    Real vth = 2.0;      // Threshold voltage (V)
    Real kp = 20e-6;     // Transconductance parameter (A/V²)
    Real lambda = 0.0;   // Channel-length modulation (1/V)
    Real w = 100e-6;     // Channel width (m)
    Real l = 10e-6;      // Channel length (m)

    // Body diode parameters (optional)
    bool body_diode = false;
    Real is_body = 1e-14;  // Body diode saturation current
    Real n_body = 1.0;     // Body diode ideality factor

    // Parasitic capacitances (optional)
    Real cgs = 0.0;  // Gate-source capacitance
    Real cgd = 0.0;  // Gate-drain capacitance
    Real cds = 0.0;  // Drain-source capacitance

    // On/Off resistance for ideal mode
    Real rds_on = 0.0;  // If > 0, use simple switch model
    Real rds_off = 1e9;

    // Computed Kp' = Kp * W/L
    Real kp_effective() const { return kp * w / l; }
};

// IGBT simplified model
// Models IGBT as a voltage-controlled switch with on-state voltage drop
struct IGBTParams {
    Real vth = 5.0;      // Gate threshold voltage (V)
    Real vce_sat = 2.0;  // Collector-emitter saturation voltage (V)
    Real rce_on = 0.01;  // On-state resistance (Ohms)
    Real rce_off = 1e9;  // Off-state resistance (Ohms)

    // Tail current parameters (for turn-off transient)
    Real tf = 0.0;       // Fall time (s), 0 = ideal
    Real tr = 0.0;       // Rise time (s), 0 = ideal

    // Input capacitance (gate)
    Real cies = 0.0;     // Input capacitance (F)

    // Anti-parallel diode (freewheeling)
    bool body_diode = true;
    Real is_diode = 1e-12;   // Diode saturation current
    Real n_diode = 1.0;      // Diode ideality factor
    Real vf_diode = 0.7;     // Diode forward voltage (simplified)
};

struct TransformerParams {
    Real turns_ratio = 1.0;   // N1:N2 (primary to secondary)
    Real lm = 1e-3;           // Magnetizing inductance (H), 0 = ideal
    Real ll1 = 0.0;           // Primary leakage inductance
    Real ll2 = 0.0;           // Secondary leakage inductance
};

using ComponentParams = std::variant<
    ResistorParams,
    CapacitorParams,
    InductorParams,
    VoltageSourceParams,
    CurrentSourceParams,
    DiodeParams,
    SwitchParams,
    MOSFETParams,
    IGBTParams,
    TransformerParams
>;

// Component representation
class Component {
public:
    Component(std::string name, ComponentType type,
              std::vector<NodeId> nodes, ComponentParams params)
        : name_(std::move(name))
        , type_(type)
        , nodes_(std::move(nodes))
        , params_(std::move(params)) {}

    const std::string& name() const { return name_; }
    ComponentType type() const { return type_; }
    const std::vector<NodeId>& nodes() const { return nodes_; }
    const ComponentParams& params() const { return params_; }

    // For components that add branch currents (V sources, inductors)
    bool has_branch_current() const {
        return type_ == ComponentType::VoltageSource ||
               type_ == ComponentType::Inductor;
    }

private:
    std::string name_;
    ComponentType type_;
    std::vector<NodeId> nodes_;
    ComponentParams params_;
};

// Circuit representation (Internal Representation - IR)
class Circuit {
public:
    Circuit() = default;

    // Circuit name/title
    const std::string& name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }

    // Add components
    void add_resistor(const std::string& name, const NodeId& n1, const NodeId& n2, Real resistance);
    void add_capacitor(const std::string& name, const NodeId& n1, const NodeId& n2, Real capacitance, Real ic = 0.0);
    void add_inductor(const std::string& name, const NodeId& n1, const NodeId& n2, Real inductance, Real ic = 0.0);
    void add_voltage_source(const std::string& name, const NodeId& npos, const NodeId& nneg, const Waveform& waveform);
    void add_current_source(const std::string& name, const NodeId& npos, const NodeId& nneg, const Waveform& waveform);
    void add_diode(const std::string& name, const NodeId& anode, const NodeId& cathode, const DiodeParams& params = {});
    void add_switch(const std::string& name, const NodeId& n1, const NodeId& n2,
                    const NodeId& ctrl_pos, const NodeId& ctrl_neg, const SwitchParams& params = {});
    void add_mosfet(const std::string& name, const NodeId& drain, const NodeId& gate,
                    const NodeId& source, const MOSFETParams& params = {});
    void add_igbt(const std::string& name, const NodeId& collector, const NodeId& gate,
                  const NodeId& emitter, const IGBTParams& params = {});
    void add_transformer(const std::string& name, const NodeId& p1, const NodeId& p2,
                        const NodeId& s1, const NodeId& s2, const TransformerParams& params = {});

    // Access components
    const std::vector<Component>& components() const { return components_; }
    const Component* find_component(const std::string& name) const;

    // Node management
    Index node_count() const { return static_cast<Index>(node_map_.size()); }
    Index branch_count() const { return branch_count_; }
    Index total_variables() const { return node_count() + branch_count_; }

    // Map node name to index (excluding ground)
    Index node_index(const NodeId& node) const;
    bool is_ground(const NodeId& node) const;

    // Get node name from index
    const NodeId& node_name(Index index) const;

    // Get all node names
    std::vector<NodeId> node_names() const;

    // Signal name for output (e.g., "V(out)", "I(L1)")
    std::string signal_name(Index index) const;

    // Validation
    bool validate(std::string& error_message) const;

    // Schematic position management (for GUI)
    void set_position(const std::string& component_name, const SchematicPosition& pos);
    std::optional<SchematicPosition> get_position(const std::string& component_name) const;
    bool has_position(const std::string& component_name) const;
    std::unordered_map<std::string, SchematicPosition> all_positions() const;
    void set_all_positions(const std::unordered_map<std::string, SchematicPosition>& positions);
    void clear_positions();

private:
    void ensure_node(const NodeId& node);
    void add_component(Component component);

    std::string name_;
    std::vector<Component> components_;
    std::unordered_map<NodeId, Index> node_map_;
    std::vector<NodeId> node_names_;  // Reverse mapping
    Index branch_count_ = 0;
    std::unordered_map<std::string, Index> branch_map_;  // Component name -> branch index
    std::unordered_map<std::string, SchematicPosition> positions_;  // Component positions for GUI
};

}  // namespace pulsim
