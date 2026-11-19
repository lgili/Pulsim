#pragma once

// One place that turns a Python `switch_fn` argument into the kernel's
// mask function plus, when the object knows them, its exact gate edges.
// Used by every engine binding (run_transient, run_transient_trbdf2 and
// the DSED schedulers), so a schedule type one engine recognises is
// recognised by all: NativeMultiMaskPwm used to be an opaque callable on
// trbdf2, sampled every h/4, and a 1 µs window in a 1 ms period was
// never seen (0 of 40 gate events).

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <functional>

#include "pulsim/blockchain/chain.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/sources/native_pwm_switch_fn.hpp"
#include "pulsim/sources/switch_schedule.hpp"
#include "pulsim/solver/trbdf2_transient.hpp"
#include "pulsim/topology/switch_state.hpp"

namespace pulsim_bindings_common {

struct SwitchFnAdapter {
    std::function<pulsim::topology::SwitchStateMask(pulsim::Real)> fn;
    /// Empty when the object has no analytic edge oracle.
    std::function<pulsim::Real(pulsim::Real)> next_edge;
};

/// `SwitchSchedule`, `NativePwm2Switch` and `NativeMultiMaskPwm` run in
/// C++ (no GIL) and hand over their edges; any other callable goes
/// through pybind11's std::function caster (which takes the GIL per call)
/// and has no edges here.
inline SwitchFnAdapter adapt_switch_fn(const pybind11::object& obj) {
    namespace py = pybind11;
    using pulsim::Real;
    using pulsim::sources::NativeMultiMaskPwm;
    using pulsim::sources::NativePwm2Switch;
    using pulsim::sources::SwitchSchedule;
    SwitchFnAdapter a;
    if (py::isinstance<SwitchSchedule>(obj)) {
        const auto sched = obj.cast<SwitchSchedule>();
        a.fn = sched;
        if (sched.has_edges()) {
            a.next_edge = [sched](Real t) { return sched.next_edge_after(t); };
        }
    } else if (py::isinstance<NativePwm2Switch>(obj)) {
        // Raw pointers only: the caller keeps `obj` alive for the run,
        // and copying a py::object inside a GIL-released kernel is unsafe.
        auto* p = obj.cast<NativePwm2Switch*>();
        a.fn = [p](Real t) { return (*p)(t); };
        a.next_edge = [p](Real t) { return p->next_edge_after(t); };
    } else if (py::isinstance<NativeMultiMaskPwm>(obj)) {
        auto* p = obj.cast<NativeMultiMaskPwm*>();
        a.fn = [p](Real t) { return (*p)(t); };
        a.next_edge = [p](Real t) { return p->next_edge_after(t); };
    } else if (py::isinstance<pulsim::blockchain::ChainPwmGate>(obj)) {
        // A control graph's PWM gate: C++; its edges under the duties
        // held now (trbdf2 re-queries after a tick, pwl after every
        // observer call).
        auto* p = obj.cast<pulsim::blockchain::ChainPwmGate*>();
        a.fn = [p](Real t) { return (*p)(t); };
        a.next_edge = [p](Real t) { return p->next_edge_after(t); };
    } else if (py::isinstance<pulsim::blockchain::ChainSignalGate>(obj)) {
        // A discrete gate: edges at the graph's publication instants.
        auto* p = obj.cast<pulsim::blockchain::ChainSignalGate*>();
        a.fn = [p](Real t) { return (*p)(t); };
        a.next_edge = [p](Real t) { return p->next_edge_after(t); };
    } else {
        a.fn = obj.cast<std::function<
            pulsim::topology::SwitchStateMask(Real)>>();
    }
    return a;
}

/// Same, refusing every mask whose width is not `num_switches`.
inline SwitchFnAdapter adapt_switch_fn(const pybind11::object& obj,
                                       pulsim::Size num_switches) {
    SwitchFnAdapter a = adapt_switch_fn(obj);
    a.fn = [fn = std::move(a.fn), num_switches](pulsim::Real t) {
        auto m = fn(t);
        pulsim::topology::require_mask_width(m, num_switches);
        return m;
    };
    return a;
}

/// A b_extra_fn argument as (fn, next_jump_after). A control graph's
/// CxxChainBExtra runs in C++ (no GIL per call); any other callable goes
/// through pybind's functional caster, its optional next_jump_after too.
inline void resolve_b_extra(const pybind11::object& obj,
                            pulsim::solver::BExtraFn& fn,
                            std::function<pulsim::Real(pulsim::Real)>& next_jump) {
    if (obj.is_none()) return;
    if (pybind11::isinstance<pulsim::blockchain::ChainBExtra>(obj)) {
        const auto& c = obj.cast<const pulsim::blockchain::ChainBExtra&>();
        fn = c.fn;
        next_jump = [ch = c.chain](pulsim::Real t) {
            return ch->next_jump_after(t);
        };
        return;
    }
    fn = obj.cast<pulsim::solver::BExtraFn>();
    if (pybind11::hasattr(obj, "next_jump_after")) {
        next_jump = obj.attr("next_jump_after")
                        .cast<std::function<pulsim::Real(pulsim::Real)>>();
    }
}

/// A (t, x) step / tick callable: a control graph's CxxChainTick runs in
/// C++ (no GIL per call); any other callable through pybind's caster.
inline std::function<void(pulsim::Real, const pulsim::Vector&)> step_fn(
    const pybind11::object& obj) {
    if (obj.is_none()) return {};
    if (pybind11::isinstance<pulsim::blockchain::ChainTick>(obj)) {
        return obj.cast<pulsim::blockchain::ChainTick&>().fn;
    }
    return obj.cast<std::function<void(pulsim::Real, const pulsim::Vector&)>>();
}

/// The variable-step driver's counters as the dict simulate() reads
/// (`result.trbdf2_stats`); one list for every entry point.
inline pybind11::dict trbdf2_stats_dict(const pulsim::solver::TrBdf2Stats& st) {
    pybind11::dict d;
    d["n_accept"]       = st.n_accept;
    d["n_reject"]       = st.n_reject;
    d["n_gate_events"]  = st.n_gate_events;
    d["n_diode_events"] = st.n_diode_events;
    d["n_solves"]       = st.n_solves;
    d["n_chatter_breaks"] = st.n_chatter_breaks;
    d["n_forced_accepts"] = st.n_forced_accepts;
    d["n_ctrl_ticks"]     = st.n_ctrl_ticks;
    d["control_tick_counts"] = st.ctrl_tick_counts;
    d["n_newton_retries"] = st.n_newton_retries;
    d["gate_pulse_min"]   = st.gate_pulse_min;
    d["n_gate_undersampled"] = st.n_gate_undersampled;
    d["gate_sample_spacing_max"] = st.gate_sample_spacing_max;
    d["n_held_diode_redecided"] = st.n_held_diode_redecided;
    d["probe_floor_used"] = st.probe_floor_used;
    d["n_gate_boundary_settles"] = st.n_gate_boundary_settles;
    d["h_max_basis"] = st.h_max_basis;
    d["n_source_breakpoints"] = st.n_source_breakpoints;
    d["n_derivative_refreshes"] = st.n_derivative_refreshes;
    d["n_refresh_probes"] = st.n_refresh_probes;
    d["h_post_event_min"] = st.h_post_event_min;
    d["n_probe_newton_retries"] = st.n_probe_newton_retries;
    d["n_newton_start_retries"] = st.n_newton_start_retries;
    d["n_sdirk_steps"] = st.n_sdirk_steps;
    d["n_reinit_steps"] = st.n_reinit_steps;
    d["n_filtered_estimates"] = st.n_filtered_estimates;
    d["n_interp_floor_landings"] = st.n_interp_floor_landings;
    d["n_roundoff_floor_accepts"] = st.n_roundoff_floor_accepts;
    if (!st.forced_trace.empty()) d["forced_trace"] = st.forced_trace;
    d["n_noise_held_decisions"] = st.n_noise_held_decisions;
    d["n_probe_noise_retakes"] = st.n_probe_noise_retakes;
    d["n_factor_builds"] = st.n_factor_builds;
    d["n_factor_refactors"] = st.n_factor_refactors;
    d["n_factor_hits"] = st.n_factor_hits;
    d["n_conditioning_rejects"] = st.n_conditioning_rejects;
    d["h_conditioning_floor"] = st.h_conditioning_floor;
    return d;
}

}  // namespace pulsim_bindings_common
