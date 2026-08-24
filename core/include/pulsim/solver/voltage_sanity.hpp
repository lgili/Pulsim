#pragma once

// =============================================================================
// Pulsim — is this trace physically plausible?
// =============================================================================
//
// v2.0 Phase 2.
//
// THE FAILURE. An inductor whose conduction path opens produces, in
// an idealized model, an unbounded voltage: v = L·di/dt with di/dt
// forced to -i/dt in one step. Pulsim reports it, finitely and
// without comment:
//
//     Vin(48 V) — L(1 mH) — S ——| gnd,  S opening at 10 kHz
//     max |v(sw)| = 2.9e+06 V
//
// 2.9 megavolts on a 48 V circuit, no warning, no error, `isfinite`
// true throughout. Nothing catches it: the inductor freeze and clamp
// guards watch the CURRENT, and the current here stays at a
// believable 14 A. It is the voltage that leaves physics.
//
// No real circuit does this. The switch avalanches, or the parasitic
// capacitance across it resonates, or the designer fitted a snubber.
// A model that omits all three is not wrong to produce a large
// number — it is wrong to let the user read it as a measurement.
//
// WHAT THIS DOES, AND WHAT IT DOES NOT. It compares the largest node
// voltage the run produced against the largest voltage any
// independent source in the circuit can produce, and reports a node
// that exceeds it by more than `factor`. It does NOT alter the
// result, insert a snubber, or pick a clamp: the value of a snubber
// is a modelling decision that belongs to whoever knows the design's
// stand-off voltage, and substituting one silently is the failure
// this whole phase has been removing. Naming the node is the part
// the simulator can do honestly.
//
// The default factor is 100. A boost converter legitimately exceeds
// its input, a flyback legitimately multiplies by its turns ratio,
// and a resonant tank legitimately rings — 100x accommodates all of
// them and still catches the 60000x above by a wide margin.

#include "pulsim/models/pulse_voltage_source.hpp"
#include "pulsim/models/pwm_voltage_source.hpp"
#include "pulsim/models/sine_voltage_source.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
#include <format>
#include <string>

namespace pulsim::solver {

/// A node whose voltage left the range the circuit's own sources can
/// account for.
struct ImplausibleVoltage {
    Index node = kInvalidIndex;
    Real peak = Real{0};          ///< largest |v| the run reached there
    Real source_scale = Real{0};  ///< largest |V| any source can make
    Real t_peak = Real{0};
};

/// The largest voltage magnitude any INDEPENDENT source in the
/// circuit can produce. Dependent sources are excluded: a VCVS's
/// output is a function of the circuit, so folding its gain in here
/// would define the bound in terms of the thing being checked.
[[nodiscard]] inline Real max_source_magnitude(
    const topology::Graph& graph, const pwl::DevicePool& pool) {
    using SK = pwl::DevicePool::StoredKind;
    Real v_max = Real{0};
    auto consider = [&v_max](Real v) {
        const Real a = v < Real{0} ? -v : v;
        if (a > v_max) {
            v_max = a;
        }
    };
    for (Index b = 0; b < graph.num_branches(); ++b) {
        if (graph.branch(b).kind != topology::BranchKind::Source ||
            !pool.is_registered(b)) {
            continue;
        }
        switch (pool.kind_of(b)) {
        case SK::VoltageSource:
            consider(pool.voltage_source_params(b).V);
            break;
        case SK::SineVoltageSource: {
            const auto& p = pool.sine_voltage_source_params(b);
            consider(std::abs(p.v_dc) + std::abs(p.v_amplitude));
            break;
        }
        case SK::PulseVoltageSource: {
            const auto& p = pool.pulse_voltage_source_params(b);
            consider(p.v_initial);
            consider(p.v_pulsed);
            break;
        }
        case SK::PWMVoltageSource: {
            const auto& p = pool.pwm_voltage_source_params(b);
            consider(p.v_high);
            consider(p.v_low);
            break;
        }
        default:
            break;
        }
    }
    return v_max;
}

/// Scan a finished run for a node voltage the circuit's sources
/// cannot account for. Returns `node == kInvalidIndex` when the
/// trace is plausible, which is the overwhelmingly common case.
///
/// One pass over the node block of the recorded samples. Nothing is
/// modified.
[[nodiscard]] inline ImplausibleVoltage find_implausible_voltage(
    const topology::Graph& graph,
    const pwl::DevicePool& pool,
    const SimulationResult& result,
    Real factor = Real{100}) {
    ImplausibleVoltage out;
    if (!(factor > Real{0}) || result.num_steps() == 0) {
        return out;
    }
    const Real scale = max_source_magnitude(graph, pool);
    if (!(scale > Real{0})) {
        // A current-source-only circuit has no voltage scale to
        // compare against. Say nothing rather than guess one.
        return out;
    }
    const Real bound = factor * scale;
    const Index n_nodes = graph.num_nodes();

    for (Size k = 0; k < result.num_steps(); ++k) {
        const auto x = result.states[k];
        const Index n = std::min<Index>(n_nodes,
                                         static_cast<Index>(x.size()));
        for (Index i = 0; i < n; ++i) {
            const Real a = std::abs(x[i]);
            if (a > bound && a > out.peak) {
                out.node = i;
                out.peak = a;
                out.t_peak = result.times[k];
            }
        }
    }
    if (out.node != kInvalidIndex) {
        out.source_scale = scale;
    }
    return out;
}

/// Human-readable form of the finding, naming the node.
[[nodiscard]] inline std::string describe(
    const topology::Graph& graph,
    const pwl::DevicePool& pool,
    const ImplausibleVoltage& v) {
    if (v.node == kInvalidIndex) {
        return {};
    }
    return std::format(
        "{} reached {:.3e} V at t = {:.6g} s, but the largest "
        "voltage any source in this circuit can produce is {:.4g} V "
        "— a factor of {:.0f}. Nothing in the solver is wrong: an "
        "inductor whose conduction path opens really does produce an "
        "unbounded voltage in an idealized model. No real circuit "
        "does, because the switch avalanches or its parasitic "
        "capacitance rings or the designer fitted a snubber. Add "
        "whichever of those your design has (a resistor or RC across "
        "the switching device is the usual one), or read this number "
        "as the model telling you it is missing one.",
        pwl::row_label(graph, pool, v.node), v.peak,
        v.t_peak, v.source_scale, v.peak / v.source_scale);
}

}  // namespace pulsim::solver
