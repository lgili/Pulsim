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

#include "pulsim/numeric/types.hpp"
#include "pulsim/sources/native_pwm_switch_fn.hpp"
#include "pulsim/sources/switch_schedule.hpp"
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

}  // namespace pulsim_bindings_common
