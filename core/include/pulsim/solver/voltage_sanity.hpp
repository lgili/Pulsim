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
// The default factor is 100, and it is a judgement, not a law —
// which is why it is a parameter. A boost converter and an ordinary
// flyback sit well inside it and a 60000x runaway sits well outside.
// A HIGH-VOLTAGE flyback (a 12 V input driving a 10 kV output, an
// ignition or CRT-style design) legitimately exceeds it and will
// trip the check; raise `factor` for those. Reporting a false alarm
// a user can dismiss is the safer error here than staying silent
// about a megavolt, but it IS an error and it should be said out
// loud rather than discovered.

#include "pulsim/models/pulse_voltage_source.hpp"
#include "pulsim/models/pwm_voltage_source.hpp"
#include "pulsim/models/sine_voltage_source.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
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
    /// F-Z-8: the peak is a restart-rule cut's right-limit row (i·R_off across
    /// the cutting device for its unresolved L/R_off mode), not a node voltage
    /// of the waveform: the record that cut it (empty: an ordinary finding).
    /// cut_state is the record's first state of its own kind (an inductor for
    /// 'cut_set', a capacitor for 'capacitor_loop'), cut_state_kind 'L' / 'C',
    /// or empty / 0 when no listed state is of that kind (the message then
    /// names neither a current nor a voltage). cut_size = 2·dW/|λ| of a
    /// single collapsing state: the cut current (L) or the voltage jump (C).
    std::string cut_device, cut_state, cut_kind;
    char cut_state_kind = 0;
    Real cut_size = Real{0}, cut_energy = Real{0};
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
    // Phase 4 C.4: an ideal transformer legitimately multiplies the
    // source voltage by its turns ratio — a 1:10 step-up producing
    // 10× the source is the circuit working, not a solver failure.
    // Unlike a VCVS's gain (an approximation knob, so the check
    // declines below), n is a physical ratio, so widen the scale by
    // it. Cascades compound; step-DOWN ratios do not narrow it.
    for (Index b = 0; b < graph.num_branches(); ++b) {
        if (graph.branch(b).kind == topology::BranchKind::Source &&
            pool.is_registered(b) &&
            pool.kind_of(b) == SK::IdealTransformer) {
            const Real n = pool.ideal_transformer_params(b).n;
            if (n > Real{1}) v_max *= n;
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
    // A DEPENDENT source defeats the bound. `add_op_amp_ideal` is a
    // VCVS at gain 1e5, and its output is a function of the circuit
    // rather than of any parameter here — so the bound would be set
    // by the input sources alone and an ordinary op-amp circuit
    // would trip it. Folding the gain in instead would define the
    // bound in terms of the thing being checked. Neither is honest,
    // so the check declines to have an opinion.
    for (Index b = 0; b < graph.num_branches(); ++b) {
        if (graph.branch(b).kind == topology::BranchKind::Source &&
            pool.is_registered(b) &&
            (pool.kind_of(b) == pwl::DevicePool::StoredKind::VCVS ||
             pool.kind_of(b) == pwl::DevicePool::StoredKind::VCCS ||
             pool.kind_of(b) == pwl::DevicePool::StoredKind::CCCS ||
             pool.kind_of(b) == pwl::DevicePool::StoredKind::CCVS ||
             pool.kind_of(b) == pwl::DevicePool::StoredKind::Gyrator)) {
            return out;
        }
    }

    const Real scale = max_source_magnitude(graph, pool);
    if (!(scale > Real{0})) {
        // Nothing to compare against: a circuit driven only by
        // current sources, by a user `b_extra_fn`, by a closed-loop
        // controller, or by initial conditions alone. Say nothing
        // rather than invent a scale.
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
            // `!(a <= bound)` rather than `a > bound`: NaN compares
            // false against BOTH, and a trace full of NaN read as
            // "plausible" would be the exact failure a check named
            // for leaving physics exists to catch.
            if (!(a <= bound) && !(a <= out.peak)) {
                out.node = i;
                out.peak = a;
                out.t_peak = result.times[k];
            }
        }
    }
    // The right limits kept beside the samples (an event's state one ulp
    // after it): a cut's spike lives there only (F-5-10). A NaN row is an
    // inner event without a right limit, not a value.
    if (result.has_event_limits && result.event_limit_width > 0) {
        const Size w = result.event_limit_width;
        const Index n = std::min<Index>(n_nodes, static_cast<Index>(w));
        // F-Z-8: a row at an impulse record's instant, at that record's own
        // devices' terminals, is the conductance-form model's right limit
        // before the collapse the record reports (i·R_off for ~L/R_off):
        // judged apart and, when it alone trips, reported as that record
        // (device, state, energy), never as a node voltage
        ImplausibleVoltage cut;
        const ReinitRecord* cut_rec = nullptr;
        auto any_at = [&](Real te) {
            for (const auto& q : result.reinit_records) {
                if (q.t == te) return true;
            }
            return false;
        };
        auto touches = [&](const ReinitRecord& q, Index node) {
            for (Index b = 0; b < graph.num_branches(); ++b) {
                if ((graph.branch(b).from == node || graph.branch(b).to == node) &&
                    std::find(q.devices.begin(), q.devices.end(),
                              graph.branch_name(b)) != q.devices.end()) {
                    return true;
                }
            }
            return false;
        };
        // several records can share an instant (one per dependent group, and
        // a fast-mode record): the one whose devices touch the node, a cut
        // set first (a spike i·R_off is a cut's, not a charge sharing's)
        auto record_for = [&](Real te, Index node) -> const ReinitRecord* {
            const ReinitRecord* hit = nullptr;
            for (const auto& q : result.reinit_records) {
                if (q.t != te || !touches(q, node)) continue;
                if (q.kind == "cut_set") return &q;
                if (hit == nullptr) hit = &q;
            }
            return hit;
        };
        auto scan = [&](const std::vector<Real>& rows, auto&& time_of) {
            for (Size j = 0; (j + 1) * w <= rows.size(); ++j) {
                const Real* r = rows.data() + j * w;
                if (std::isnan(r[0])) continue;
                const Real te = time_of(j);
                const bool has = any_at(te);
                for (Index i = 0; i < n; ++i) {
                    const Real a = std::abs(r[i]);
                    if (a <= bound) continue;   // NaN goes on (!(NaN <= bound))
                    // only the record's own devices' terminals are its
                    const ReinitRecord* q = has ? record_for(te, i) : nullptr;
                    const bool at_cut = q != nullptr;
                    ImplausibleVoltage& o = at_cut ? cut : out;
                    if (!(a <= o.peak)) {
                        o.node = i;
                        o.peak = a;
                        o.t_peak = te;
                        if (at_cut) cut_rec = q;
                    }
                }
            }
        };
        scan(result.right_limit_states, [&](Size j) {
            return j < result.right_limit_index.size() &&
                           result.right_limit_index[j] < result.times.size()
                       ? result.times[result.right_limit_index[j]] : Real{0};
        });
        scan(result.inner_event_right, [&](Size j) {
            return j < result.inner_event_times.size() ? result.inner_event_times[j] : Real{0};
        });
        if (out.node == kInvalidIndex && cut_rec != nullptr) {
            out = cut;
            out.cut_device = cut_rec->devices.empty() ? cut_rec->cause : cut_rec->devices.front();
            out.cut_kind = cut_rec->kind;
            out.cut_energy = cut_rec->dW;
            // the states list every state the jump moved, in entry order (a
            // divider capacitor before the winding whose current was cut):
            // name the first of the record's own kind, never another kind's
            using SK = pwl::DevicePool::StoredKind;
            const char want = cut_rec->kind == "cut_set" ? 'L'
                              : cut_rec->kind == "capacitor_loop" ? 'C' : 0;
            bool linear = false;
            for (const auto& s : cut_rec->states) {
                for (Index b = 0; want != 0 && out.cut_state.empty() && b < graph.num_branches(); ++b) {
                    if (graph.branch_name(b) != s || !pool.is_registered(b)) continue;
                    const SK k = pool.kind_of(b);
                    if ((want == 'L' && (k == SK::Inductor || k == SK::SaturableInductor)) ||
                        (want == 'C' && (k == SK::Capacitor || k == SK::NonlinearCapacitor))) {
                        out.cut_state = s;
                        out.cut_state_kind = want;
                        linear = k == SK::Inductor || k == SK::Capacitor;
                    }
                }
            }
            // λ = L·i (C·ΔV) and dW = ½·L·i² (½·C·ΔV²): i (ΔV) = 2·dW/λ
            // (one linear collapsing state only: dW of several is not one L·i²/2)
            out.cut_size = linear && cut_rec->lambda != Real{0} && cut_rec->states.size() == 1
                ? std::abs(Real{2} * cut_rec->dW / cut_rec->lambda) : Real{0};
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
    // add-pwl-igbt-and-thyristor 3.6: a gated device (IGBT with reverse
    // blocking, GTO, thyristor) at the node conducts one way only and not
    // at all with its gate off; the usual cause is a missing freewheel path.
    std::string gated;
    for (Index b = 0; b < graph.num_branches(); ++b) {
        const auto& br = graph.branch(b);
        if (!pool.is_registered(b)
            || pool.kind_of(b) != pwl::DevicePool::StoredKind::Diode
            || pool.diode_params(b).gate == models::SwitchedDiode::Gate::None
            || (br.from != v.node && br.to != v.node)) {
            continue;
        }
        gated += std::format("{}'{}'", gated.empty() ? "" : ", ",
                             graph.branch_name(b));
    }
    if (!gated.empty()) {
        gated = std::format(
            " The gated device(s) {} at this node conduct one way only and "
            "not at all with the gate off: an inductive current through "
            "them needs a freewheel path (add_diode(..., V_f=...) across "
            "the device or the load).", gated);
    }
    if (!v.cut_device.empty() && v.cut_state_kind == 'L') {
        return std::format(
            "'{}' cut the {} of '{}' at t = {:.6g} s in an "
            "instant: the restart rule booked its energy {}{:.4g} J once, as an "
            "impulse record (result.reinit_records, kind 'cut_set'). The right "
            "limit stored there is the model's own before that collapse: the "
            "current held across the switch's off-conductance, {:.3e} V at {} "
            "for the unresolved L/R_off mode (the circuit's sources reach "
            "{:.4g} V) — an artefact of the ideal cut, not a sample of the "
            "waveform. A real circuit gives the current a path (a freewheel "
            "diode, a snubber, the switch's capacitance or avalanche): add "
            "whichever your design has.",
            v.cut_device,
            v.cut_size > Real{0} ? std::format("{:.4g} A current", v.cut_size)
                                 : std::string("current"),
            v.cut_state, v.t_peak,
            v.cut_size > Real{0} ? "½·L·i² = " : "",   // one state: dW is its own
            v.cut_energy, v.peak, pwl::row_label(graph, pool, v.node), v.source_scale);
    }
    if (!v.cut_device.empty() && v.cut_state_kind == 'C') {
        return std::format(
            "'{}' closed a capacitor loop on '{}' at t = {:.6g} s: its voltage "
            "jumped{} by charge sharing in an instant, and the restart rule booked "
            "the energy {:.4g} J once, as an impulse record "
            "(result.reinit_records, kind 'capacitor_loop'). The right limit "
            "stored there reads {:.3e} V at {} (the circuit's sources reach "
            "{:.4g} V): charge sharing does not explain that, so check the "
            "circuit at that instant. A real circuit limits the sharing current "
            "(the switch's on-resistance, the loop's inductance).",
            v.cut_device, v.cut_state, v.t_peak,
            v.cut_size > Real{0} ? std::format(" by {:.4g} V", v.cut_size) : std::string{},
            v.cut_energy, v.peak, pwl::row_label(graph, pool, v.node), v.source_scale);
    }
    if (!v.cut_device.empty()) {   // the collapsed state's kind is not told
        return std::format(
            "'{}' switched at t = {:.6g} s and the restart rule booked {:.4g} J "
            "once, as an impulse record (result.reinit_records, kind '{}'). The "
            "right limit stored there reads {:.3e} V at {} (the circuit's "
            "sources reach {:.4g} V); check the circuit at that instant.",
            v.cut_device, v.t_peak, v.cut_energy, v.cut_kind,
            v.peak, pwl::row_label(graph, pool, v.node), v.source_scale);
    }
    return gated.empty() ? std::format(
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
        v.t_peak, v.source_scale, v.peak / v.source_scale)
        : std::format(
        "{} reached {:.3e} V at t = {:.6g} s, but the largest "
        "voltage any source in this circuit can produce is {:.4g} V "
        "— a factor of {:.0f}.{}",
        pwl::row_label(graph, pool, v.node), v.peak,
        v.t_peak, v.source_scale, v.peak / v.source_scale, gated);
}

}  // namespace pulsim::solver
