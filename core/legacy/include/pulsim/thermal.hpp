#pragma once

#include "pulsim/types.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace pulsim {

// Thermal RC stage (for Foster or Cauer network)
struct ThermalRCStage {
    Real rth;   // Thermal resistance (K/W)
    Real cth;   // Thermal capacitance (J/K), or 0 for steady-state
    Real tau() const { return rth * cth; }  // Time constant
};

// Foster network: parallel RC stages in series
// Zth(t) = sum_i( Rth_i * (1 - exp(-t/tau_i)) )
struct FosterNetwork {
    std::vector<ThermalRCStage> stages;

    // Compute steady-state thermal resistance
    Real rth_total() const {
        Real sum = 0.0;
        for (const auto& s : stages) sum += s.rth;
        return sum;
    }

    // Compute transient thermal impedance at time t
    Real zth(Real t) const {
        Real sum = 0.0;
        for (const auto& s : stages) {
            if (s.cth > 0) {
                sum += s.rth * (1.0 - std::exp(-t / s.tau()));
            } else {
                sum += s.rth;
            }
        }
        return sum;
    }
};

// Cauer network: ladder of series R with shunt C
// More physically meaningful (represents material layers)
struct CauerNetwork {
    std::vector<ThermalRCStage> stages;

    Real rth_total() const {
        Real sum = 0.0;
        for (const auto& s : stages) sum += s.rth;
        return sum;
    }
};

// Thermal network type
enum class ThermalNetworkType {
    Foster,
    Cauer,
    Simple  // Single Rth (steady-state only)
};

// Thermal model for a device
struct ThermalModel {
    std::string device_name;
    ThermalNetworkType type = ThermalNetworkType::Simple;

    // Simple model: single Rth_ja
    Real rth_jc = 0.0;   // Junction-to-case (K/W)
    Real rth_cs = 0.0;   // Case-to-sink (K/W)
    Real rth_sa = 0.0;   // Sink-to-ambient (K/W)

    // Foster/Cauer network (junction to case)
    FosterNetwork foster;
    CauerNetwork cauer;

    // Temperature limits
    Real tj_max = 175.0;  // Maximum junction temperature (C)
    Real tj_warn = 150.0; // Warning threshold (C)

    // Temperature coefficients (for parameter updates)
    Real tc_rds = 0.004;  // Rds_on temp coefficient (typical for Si MOSFET)
    Real tc_vth = -0.003; // Vth temp coefficient

    // Compute total junction-to-ambient thermal resistance
    Real rth_ja() const {
        Real rth_jc_eff = 0.0;
        if (type == ThermalNetworkType::Foster) {
            rth_jc_eff = foster.rth_total();
        } else if (type == ThermalNetworkType::Cauer) {
            rth_jc_eff = cauer.rth_total();
        } else {
            rth_jc_eff = rth_jc;
        }
        return rth_jc_eff + rth_cs + rth_sa;
    }
};

// Thermal state for a device during simulation
struct ThermalState {
    std::string device_name;
    Real tj = 25.0;       // Current junction temperature (C)
    Real tc = 25.0;       // Current case temperature (C)
    Real ts = 25.0;       // Current sink temperature (C)
    Real power_in = 0.0;  // Current power dissipation (W)

    // Foster network state (temperature rise at each stage)
    std::vector<Real> foster_temps;

    // Peak tracking
    Real tj_peak = 25.0;
    Real tj_peak_time = 0.0;

    bool exceeded_warning = false;
    bool exceeded_max = false;
};

// Thermal simulation engine
class ThermalSimulator {
public:
    ThermalSimulator() = default;

    // Add thermal model for a device
    void add_model(const ThermalModel& model);

    // Set ambient temperature (can be time-varying)
    void set_ambient(Real t_amb) { t_ambient_ = t_amb; }
    Real ambient() const { return t_ambient_; }

    // Initialize thermal states
    void initialize();

    // Update thermal network for one timestep
    // Returns max junction temperature
    Real step(Real dt, const std::unordered_map<std::string, Real>& device_powers);

    // Get current junction temperature for a device
    Real junction_temp(const std::string& device_name) const;

    // Get all thermal states
    const std::vector<ThermalState>& states() const { return states_; }

    // Get thermal warnings
    struct ThermalWarning {
        std::string device_name;
        Real temperature;
        Real time;
        bool is_failure;  // true if exceeded max
    };
    const std::vector<ThermalWarning>& warnings() const { return warnings_; }

    // Compute temperature-adjusted parameter
    Real adjust_rds_on(Real rds_on_25c, Real tj, Real tc = 0.004) const;
    Real adjust_vth(Real vth_25c, Real tj, Real tc = -0.003) const;

private:
    std::vector<ThermalModel> models_;
    std::vector<ThermalState> states_;
    std::vector<ThermalWarning> warnings_;
    Real t_ambient_ = 25.0;
    Real current_time_ = 0.0;

    ThermalState* find_state(const std::string& device_name);
    const ThermalModel* find_model(const std::string& device_name) const;
};

// Helper: Create Foster network from datasheet Zth curve
// Points are (time_s, zth_value) pairs
FosterNetwork fit_foster_network(
    const std::vector<std::pair<Real, Real>>& zth_curve,
    int num_stages = 4);

// Helper: Create typical MOSFET thermal model
ThermalModel create_mosfet_thermal(
    const std::string& name,
    Real rth_jc,
    Real rth_cs = 0.5,
    Real rth_sa = 1.0);

}  // namespace pulsim
