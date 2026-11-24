// SPDX-License-Identifier: MIT
//
// Pulsim — add-in-kernel-event-losses, group 2: the loss port of a PWL
// switching device, the reading rule, the event record.
//
//   DeviceLossSpec     what CircuitBuilder::set_device_losses stores
//   commutation_point  THE reading rule (every engine and the periodic
//                      orbit of dsed/period_thermal.hpp call this one)
//   LossEvent          one commutation as recorded: (t, kind, v, i, Tj, E)
//   DeviceLossRecord   a device's events and energy totals in a result
//   CompensatedSum     Neumaier summation for the energy ledgers
//
// No graph or pool here (the pool stores the specs); the run-time tracker
// that reads devices out of a state vector is solver/loss_tracker.hpp.

#pragma once

#include "pulsim/thermal/loss_table.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace pulsim::thermal {

/// Where a device's conduction heat comes from (design "Conduction heat").
enum class ConductionHeat {
    Electrical,   ///< the device's own electrical dissipation per step
    Table,        ///< V_on(|i|, Tj)·|i| from the on-state table
};

/// One device's loss port. Names are resolved when a run starts.
/// Exactly one temperature source: `junction` (a thermal node of the
/// circuit, closed loop) or a finite `tj` (°C, open loop). With `tj`,
/// `heat_node` optionally names the node the heat is deposited into
/// (empty: the energies are recorded, nothing is deposited).
struct DeviceLossSpec {
    std::string device;
    std::string junction;
    std::string heat_node;
    double tj = std::numeric_limits<double>::quiet_NaN();
    std::shared_ptr<const LossTable3D> e_on;
    std::shared_ptr<const LossTable3D> e_off;
    std::shared_ptr<const ConductionTable> v_on;
    ConductionHeat conduction = ConductionHeat::Electrical;

    [[nodiscard]] bool closed_loop() const noexcept { return !junction.empty(); }
    [[nodiscard]] const std::string& deposit_node() const noexcept {
        return closed_loop() ? junction : heat_node;
    }
};

/// (|v|, |i|) a commutation is charged at.
struct CommutationPoint {
    double v = 0.0;
    double i = 0.0;
};

/// The reading rule of add-in-kernel-event-losses (design "Reading
/// rule"): a turn-on reads the blocking voltage BEFORE the commutation
/// and the current AFTER it; a turn-off the voltage after and the
/// current before. `before` is the last accepted state with the old
/// mask, `after` the first one with the new mask.
[[nodiscard]] inline CommutationPoint commutation_point(
    bool turn_on, double v_before, double i_before, double v_after,
    double i_after) noexcept {
    return turn_on ? CommutationPoint{std::abs(v_before), std::abs(i_after)}
                   : CommutationPoint{std::abs(v_after), std::abs(i_before)};
}

/// Neumaier compensated sum: the ledgers of event and conduction energy
/// stay within a few ulp of the exact sum over a long run.
struct CompensatedSum {
    double s = 0.0;
    double c = 0.0;
    void add(double x) noexcept {
        const double t = s + x;
        c += std::abs(s) >= std::abs(x) ? (s - t) + x : (x - t) + s;
        s = t;
    }
    [[nodiscard]] double value() const noexcept { return s + c; }
};

struct LossEvent {
    double t = 0.0;     ///< start of the step the new mask was solved over
    bool turn_on = false;
    double v = 0.0;     ///< |v| read by the rule
    double i = 0.0;     ///< |i| read by the rule
    double tj = 0.0;    ///< junction temperature the table was read at, °C
    double e = 0.0;     ///< table energy, J (0 without a table for the kind)
};

/// A device's record in a result.
struct DeviceLossRecord {
    std::string device;
    std::string node;              ///< deposit node ("" = record only)
    std::vector<LossEvent> events;
    double E_on = 0.0;
    double E_off = 0.0;
    double E_cond_electrical = 0.0;  ///< ∫v·i over the run (always)
    double E_cond_table = 0.0;       ///< ∫V_on(|i|,Tj)|i| (with a V_on table)
    double E_impulse = 0.0;          ///< impulsive re-initialisation energy booked to it
    std::size_t n_on = 0;
    std::size_t n_off = 0;
    bool has_e_on = false;           ///< whether an E_on table was configured
    bool has_e_off = false;          ///< whether an E_off table was configured
};

}  // namespace pulsim::thermal
