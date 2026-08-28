// =============================================================================
// Pulsim — pybind11 bindings for the DSED schedulers (Bridge.10)
// =============================================================================
//
// Wraps the three C++ scheduler templates (PEDSimulator [DOPRI5],
// PEDSimulatorBDF2, PEDSimulatorAuto) so Python callers go through the
// native inner loop instead of the pure-Python port. The Python-side
// `system` and `switch_fn` are still invoked via callbacks (the C++
// scheduler holds a `py::object` and calls `.attr("A_matrix")()` etc.),
// so RHS / b_vector queries still pay the GIL acquire on every step —
// but the scheduler control flow, step controllers, Hermite interpolation,
// LU solves, and event predicates all run natively.
//
// Captured speedup on buck CCM (5 ms, 100 kHz, see
// notes/DSED_BRIDGE_DESIGN.md §11):
//
//   * Python scheduler   : ~60 µs / step (1007 steps, 61 ms wall)
//   * C++ scheduler here : ~10–15 µs / step (4–6× speedup)
//   * Fully-native (no callbacks) : ~5 µs / step — see follow-up
//
// MaskT handling: we wrap Python masks in a thin `PyMask` struct
// (struct so ADL finds our `mode_id_of(PyMask)` overload inside
// `scheduler_auto.hpp`).
//
// Result types: PEDResult / PEDResultAuto are returned as Python
// dataclass-like dicts so callers don't need a custom .py shim.

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/dsed/builder_adapter.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/dsed/scheduler.hpp"
#include "pulsim/dsed/scheduler_bdf2.hpp"
#include "pulsim/dsed/scheduler_auto.hpp"
#include "pulsim/dsed/step_controller.hpp"
#include "pulsim/dsed/stiffness_detector.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/sources/native_pwm_switch_fn.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace pulsim_dsed_bindings {

using pulsim::Real;
using pulsim::Vector;
using pulsim::DenseMatrix;

// ---------------------------------------------------------------------------
// PyMask: thin wrapper around py::object so ADL finds our `mode_id_of`
// overload when the C++ scheduler templates expand.
// ---------------------------------------------------------------------------
struct PyMask {
    py::object obj;

    bool operator==(const PyMask& other) const {
        if (obj.is(other.obj)) return true;
        try {
            return obj.equal(other.obj);
        } catch (...) {
            return false;
        }
    }
    bool operator!=(const PyMask& other) const {
        return !(*this == other);
    }
};

// Phase-0 fix #5: the old free-function here truncated py::hash
// (64-bit Py_hash_t) to int32 for PEDSimulatorAuto's per-mode
// integrator cache — two distinct masks colliding in 32 bits
// silently reused the wrong cached λ_max. PySystem now owns an
// injective interning `mode_id()` (full-hash bucketed, equality-
// checked) that `resolve_mode_id` prefers; no stateless hash
// shim remains.

// ---------------------------------------------------------------------------
// PySystem: adapter that holds a Python `system` object and forwards
// `A_matrix()`, `b_vector(t)`, `rhs(t, x)`, `current_mask()`, `set_mask(m)`
// through pybind11. Matches the `HasLTIPerMode` concept.
// ---------------------------------------------------------------------------
class PySystem {
public:
    explicit PySystem(py::object obj)
        : obj_(std::move(obj)),
          A_attr_(obj_.attr("A_matrix")),
          b_attr_(obj_.attr("b_vector")),
          rhs_attr_(obj_.attr("rhs")),
          mask_attr_(obj_.attr("current_mask")),
          set_mask_attr_(obj_.attr("set_mask")) {}

    // Returns DenseMatrix by value (Eigen copy is cheap for small n_state).
    DenseMatrix A_matrix() const {
        return A_attr_().cast<DenseMatrix>();
    }

    Vector b_vector(Real t) const {
        return b_attr_(t).cast<Vector>();
    }

    Vector rhs(Real t, const Vector& x) const {
        return rhs_attr_(t, x).cast<Vector>();
    }

    PyMask current_mask() const {
        return PyMask{mask_attr_()};
    }

    void set_mask(const PyMask& m) {
        set_mask_attr_(m.obj);
    }

    /// Injective, dense mode id for `m` (Phase-0 fix #5). Buckets by
    /// the FULL 64-bit Python hash, then equality-checks within the
    /// bucket, so distinct masks can never alias an id. Preferred
    /// over the (removed) truncating hash shim by `resolve_mode_id`.
    [[nodiscard]] int mode_id(const PyMask& m) const {
        std::size_t h;
        try {
            h = static_cast<std::size_t>(py::hash(m.obj));
        } catch (...) {
            // Unhashable mask object — fall back to identity. Two
            // views of the same underlying object still match via
            // the equality scan below when hashing is available;
            // identity is the last resort.
            h = reinterpret_cast<std::uintptr_t>(m.obj.ptr());
        }
        auto& bucket = mode_id_buckets_[h];
        for (const auto& [seen, id] : bucket) {
            if (seen == m) return id;
        }
        const int id = next_mode_id_++;
        bucket.emplace_back(m, id);
        return id;
    }

private:
    py::object obj_;
    // Pre-resolved attributes to skip the attr-lookup cost per call.
    py::object A_attr_;
    py::object b_attr_;
    py::object rhs_attr_;
    py::object mask_attr_;
    py::object set_mask_attr_;
    // Phase-0 fix #5: full-hash → [(mask, dense id)] intern table.
    mutable std::unordered_map<std::size_t,
        std::vector<std::pair<PyMask, int>>> mode_id_buckets_;
    mutable int next_mode_id_ = 0;
};

// ---------------------------------------------------------------------------
// PySwitchFn: adapter that holds a Python `switch_fn` callable and an
// optional `next_edge_after(t)` method. Returns PyMask from operator().
// ---------------------------------------------------------------------------
class PySwitchFn {
public:
    explicit PySwitchFn(py::object fn)
        : fn_(std::move(fn)) {
        if (py::hasattr(fn_, "next_edge_after")) {
            nea_attr_ = fn_.attr("next_edge_after");
            has_nea_ = true;
        }
    }

    PyMask operator()(Real t) const {
        return PyMask{fn_(t)};
    }

    Real next_edge_after(Real t) const {
        if (has_nea_) {
            return nea_attr_(t).cast<Real>();
        }
        return std::numeric_limits<Real>::infinity();
    }

private:
    py::object fn_;
    py::object nea_attr_;
    bool has_nea_ = false;
};

// ---------------------------------------------------------------------------
// PEDResult / PEDResultAuto → Python dict converters.
//
// We return a dict so callers can index by name; the existing
// _PEDSimulationResult wrapper in `python/pulsim/_dsed_dispatch.py`
// already handles arbitrary attribute access.
// ---------------------------------------------------------------------------

template <class MaskT>
py::dict ped_result_to_dict(
    const pulsim::dsed::PEDResult<MaskT>& res) {
    // Convert times + states to numpy arrays
    const auto n_steps = static_cast<py::ssize_t>(res.times.size());
    const auto n_state = res.states.empty()
        ? py::ssize_t{0}
        : static_cast<py::ssize_t>(res.states[0].size());

    py::array_t<double> times_arr(n_steps);
    py::array_t<double> states_arr({n_steps, n_state});
    auto* times_ptr = times_arr.mutable_data();
    auto* states_ptr = states_arr.mutable_data();
    for (py::ssize_t i = 0; i < n_steps; ++i) {
        times_ptr[i] = res.times[i];
        for (py::ssize_t j = 0; j < n_state; ++j) {
            states_ptr[i * n_state + j] = res.states[i][j];
        }
    }

    py::dict d;
    d["times"] = std::move(times_arr);
    d["states"] = std::move(states_arr);
    d["n_accept"] = res.n_accept;
    d["n_reject"] = res.n_reject;
    d["n_events"] = res.n_events;
    d["cpu_time_seconds"] = res.cpu_time_seconds;
    // Event log: skip the MaskT objects (they're py::object wrappers
    // around Python objects that we just hand back).
    py::list events;
    for (const auto& e : res.event_log) {
        py::dict ed;
        ed["t"] = e.t;
        ed["name"] = e.name;
        ed["predicate_type"] = static_cast<int>(e.type);
        // Cast MaskT — PyMask carries a raw py::object; the
        // SwitchStateMask path goes through pybind11 auto-cast since
        // the type is bound in `bindings.cpp`.
        if constexpr (std::is_same_v<MaskT, PyMask>) {
            ed["old_mask"] = e.old_mask.obj;
            ed["new_mask"] = e.new_mask.obj;
        } else {
            ed["old_mask"] = py::cast(e.old_mask);
            ed["new_mask"] = py::cast(e.new_mask);
        }
        events.append(ed);
    }
    // Patch the (non-PyMask) overload's mask serialisation
    if constexpr (!std::is_same_v<MaskT, PyMask>) {
        // Empty `event_log` already populated above without mask
        // entries; nothing extra to do here.
    }
    d["event_log"] = events;
    return d;
}

template <class MaskT>
py::dict ped_result_auto_to_dict(
    const pulsim::dsed::PEDResultAuto<MaskT>& res) {
    const auto n_steps = static_cast<py::ssize_t>(res.times.size());
    const auto n_state = res.states.empty()
        ? py::ssize_t{0}
        : static_cast<py::ssize_t>(res.states[0].size());

    py::array_t<double> times_arr(n_steps);
    py::array_t<double> states_arr({n_steps, n_state});
    auto* times_ptr = times_arr.mutable_data();
    auto* states_ptr = states_arr.mutable_data();
    for (py::ssize_t i = 0; i < n_steps; ++i) {
        times_ptr[i] = res.times[i];
        for (py::ssize_t j = 0; j < n_state; ++j) {
            states_ptr[i * n_state + j] = res.states[i][j];
        }
    }

    py::dict d;
    d["times"] = std::move(times_arr);
    d["states"] = std::move(states_arr);
    d["n_accept"] = res.n_accept;
    d["n_reject"] = res.n_reject;
    d["n_events"] = res.n_events;
    d["n_rk45_steps"] = res.n_rk45_steps;
    d["n_bdf2_steps"] = res.n_bdf2_steps;
    d["cpu_time_seconds"] = res.cpu_time_seconds;
    py::list events;
    for (const auto& e : res.event_log) {
        py::dict ed;
        ed["t"] = e.t;
        ed["name"] = e.name;
        ed["predicate_type"] = static_cast<int>(e.type);
        ed["integrator_used"] = static_cast<int>(e.integrator_used);
        if constexpr (std::is_same_v<MaskT, PyMask>) {
            ed["old_mask"] = e.old_mask.obj;
            ed["new_mask"] = e.new_mask.obj;
        } else {
            ed["old_mask"] = py::cast(e.old_mask);
            ed["new_mask"] = py::cast(e.new_mask);
        }
        events.append(ed);
    }
    d["event_log"] = events;
    return d;
}

// ---------------------------------------------------------------------------
// Module init — registers three free functions:
//   run_ped_native(system, switch_fn, x0, t_end, rtol, atol, dt_init, dt_max)
//   run_bdf2_native(system, switch_fn, x0, t_end, h_fixed)
//   run_auto_native(system, switch_fn, x0, t_end, rtol, atol, dt_init,
//                    dt_max, h_bdf2, stiffness_threshold)
// ---------------------------------------------------------------------------
void init_module(py::module_& m) {
    using pulsim::dsed::PEDSimulator;
    using pulsim::dsed::PEDSimulatorBDF2;
    using pulsim::dsed::PEDSimulatorAuto;
    using pulsim::dsed::PIController;
    using pulsim::dsed::EventPredictor;
    using pulsim::dsed::StiffnessDetector;

    m.def("run_ped_native",
        [](py::object system_obj,
           py::object switch_fn_obj,
           Vector x0,
           Real t_end,
           Real rtol,
           Real atol,
           Real dt_init,
           Real dt_max,
           std::size_t store_every) {
            PySystem sys(std::move(system_obj));
            PySwitchFn sf(std::move(switch_fn_obj));
            PIController ctrl(rtol, atol);
            EventPredictor pred;
            PEDSimulator<PySystem, PySwitchFn> sim(
                sys, std::move(sf), std::move(ctrl),
                std::move(pred), dt_init, dt_max,
                store_every);
            auto res = sim.simulate(x0, t_end);
            return ped_result_to_dict(res);
        },
        py::arg("system"),
        py::arg("switch_fn"),
        py::arg("x0"),
        py::arg("t_end"),
        py::arg("rtol")       = Real{1e-6},
        py::arg("atol")       = Real{1e-9},
        py::arg("dt_init")    = Real{1e-9},
        py::arg("dt_max")     = Real{1e-5},
        py::arg("store_every") = std::size_t{1},
        "Native PEDSimulator (DOPRI5 + adaptive PI + event scan) "
        "running the C++ scheduler inner loop. `system` and "
        "`switch_fn` are still invoked via Python callbacks per step "
        "(GIL overhead), but the scheduler control flow runs natively. "
        "Returns a dict with times, states, n_accept, n_reject, "
        "n_events, cpu_time_seconds, event_log.");

    m.def("run_bdf2_native",
        [](py::object system_obj,
           py::object switch_fn_obj,
           Vector x0,
           Real t_end,
           Real h_fixed,
           std::size_t store_every) {
            PySystem sys(std::move(system_obj));
            PySwitchFn sf(std::move(switch_fn_obj));
            PEDSimulatorBDF2<PySystem, PySwitchFn> sim(
                sys, std::move(sf), h_fixed, store_every);
            auto res = sim.simulate(x0, t_end);
            return ped_result_to_dict(res);
        },
        py::arg("system"),
        py::arg("switch_fn"),
        py::arg("x0"),
        py::arg("t_end"),
        py::arg("h_fixed")    = Real{1e-6},
        py::arg("store_every") = std::size_t{1},
        "Native PEDSimulatorBDF2 (implicit BDF2 + Crank-Nicolson "
        "bootstrap) running the C++ scheduler inner loop.");

    m.def("run_auto_native",
        [](py::object system_obj,
           py::object switch_fn_obj,
           Vector x0,
           Real t_end,
           Real rtol,
           Real atol,
           Real dt_init,
           Real dt_max,
           Real h_bdf2,
           Real stiffness_threshold,
           std::size_t store_every) {
            PySystem sys(std::move(system_obj));
            PySwitchFn sf(std::move(switch_fn_obj));
            PIController ctrl(rtol, atol);
            StiffnessDetector det(stiffness_threshold);
            PEDSimulatorAuto<PySystem, PySwitchFn> sim(
                sys, std::move(sf), std::move(ctrl),
                std::move(det), dt_init, dt_max, h_bdf2,
                store_every);
            auto res = sim.simulate(x0, t_end);
            return ped_result_auto_to_dict(res);
        },
        py::arg("system"),
        py::arg("switch_fn"),
        py::arg("x0"),
        py::arg("t_end"),
        py::arg("rtol")                = Real{1e-6},
        py::arg("atol")                = Real{1e-9},
        py::arg("dt_init")             = Real{1e-9},
        py::arg("dt_max")              = Real{1e-5},
        py::arg("h_bdf2")              = Real{1e-6},
        py::arg("stiffness_threshold") = Real{10.0},
        py::arg("store_every")          = std::size_t{1},
        "Native PEDSimulatorAuto (per-mode RK45↔BDF2 dispatch via "
        "stiffness detector) running the C++ scheduler inner loop. "
        "Returns a dict that additionally includes n_rk45_steps + "
        "n_bdf2_steps for each mode-segment.");

    // -------------------------------------------------------------------
    // Bridge.11 — NativeCircuitBuilderAdapter + native-typed schedulers.
    //
    // The schedulers above (`run_ped_native` / `run_bdf2_native` /
    // `run_auto_native`) take a generic `system` Python object and
    // pay GIL roundtrips on every A_matrix / b_vector / rhs call (~22
    // µs/step on buck CCM). The variants below take a native C++
    // `NativeCircuitBuilderAdapter` and a SwitchStateMask-returning
    // switch_fn; the scheduler inner loop calls C++ → C++ for RHS
    // (no GIL, no marshalling), only paying the GIL cost at gate
    // edges when the Python switch_fn is invoked.
    //
    // Expected speedup vs Bridge.10: ~5-10× per-step → buck CCM 5 ms
    // down from ~22 ms wall-clock to ~2-5 ms.
    // -------------------------------------------------------------------

    using pulsim::dsed::NativeCircuitBuilderAdapter;
    using SSM = pulsim::topology::SwitchStateMask;
    using NativeAdapter = NativeCircuitBuilderAdapter<SSM>;

    // --- PySwitchFnSSM: switch_fn adapter with Bridge.12 native fast path ---
    //
    // The C++ scheduler template (`operator()(t)` + `next_edge_after(t)`)
    // is satisfied by THIS class. At construction we check whether the
    // user's `py::object` is actually a pybind11-bound NativePwm2Switch
    // or NativeMultiMaskPwm — if so, we store a C++ pointer to the
    // underlying object and call its native methods directly. Otherwise
    // (Python class with __call__ / next_edge_after) we fall back to
    // the GIL-acquiring path.
    //
    // The py::object `fn_` keeps the underlying native instance alive
    // as long as PySwitchFnSSM is alive (pybind11 ref-count).
    struct PySwitchFnSSM {
        explicit PySwitchFnSSM(py::object fn)
            : fn_{std::move(fn)} {
            // Try to detect the native PWM types first. py::cast<T*>
            // returns the C++ pointer if the py::object holds an
            // instance of the bound type; throws py::cast_error
            // otherwise.
            try {
                native_pwm2_ =
                    fn_.cast<pulsim::sources::NativePwm2Switch*>();
            } catch (const py::cast_error&) {
                native_pwm2_ = nullptr;
            }
            if (!native_pwm2_) {
                try {
                    native_multi_ =
                        fn_.cast<pulsim::sources::NativeMultiMaskPwm*>();
                } catch (const py::cast_error&) {
                    native_multi_ = nullptr;
                }
            }
            if (!native_pwm2_ && !native_multi_) {
                // Fallback path — generic Python callable. Pre-resolve
                // `next_edge_after` attribute (if any) to avoid the
                // hasattr cost per call.
                if (py::hasattr(fn_, "next_edge_after")) {
                    nea_attr_ = fn_.attr("next_edge_after");
                    has_nea_  = true;
                }
            }
        }

        [[nodiscard]] SSM operator()(Real t) const {
            // Bridge.12 native fast path — pure C++, no GIL touch.
            if (native_pwm2_) return (*native_pwm2_)(t);
            if (native_multi_) return (*native_multi_)(t);
            // Generic Python fallback (Bridge.11 path).
            py::gil_scoped_acquire gil;
            return fn_(t).cast<SSM>();
        }

        [[nodiscard]] Real next_edge_after(Real t) const {
            if (native_pwm2_)  return native_pwm2_->next_edge_after(t);
            if (native_multi_) return native_multi_->next_edge_after(t);
            py::gil_scoped_acquire gil;
            if (has_nea_) {
                return nea_attr_(t).cast<Real>();
            }
            return std::numeric_limits<Real>::infinity();
        }

        py::object fn_;
        pulsim::sources::NativePwm2Switch*   native_pwm2_  = nullptr;
        pulsim::sources::NativeMultiMaskPwm* native_multi_ = nullptr;
        py::object nea_attr_;
        bool       has_nea_ = false;
    };

    py::class_<NativeAdapter>(m, "_NativeCircuitBuilderAdapter",
        "Bridge.11 — native C++ CircuitBuilderAdapter. Wraps a "
        "(CircuitBuilder, PwlStateSpaceCache) pair so the C++ "
        "schedulers can call A_matrix / b_vector / rhs without GIL "
        "roundtrips. Time-varying source overlay (sine/PWM/pulse) "
        "is computed natively via the same compute_*_b_extra "
        "helpers run_transient uses; an optional Python b_extra_fn "
        "callback adds onto that result.")
        .def(py::init([](
                const pulsim::builder::CircuitBuilder& builder,
                pulsim::pwl::PwlStateSpaceCache& cache,
                py::object b_extra_fn) {
                std::function<Vector(Real)> cpp_b_extra;
                if (!b_extra_fn.is_none()) {
                    // Wrap the Python callable into std::function. The
                    // GIL cost only matters if the user actually
                    // provides a callback — most circuits use the
                    // auto-detected sine/PWM/pulse path.
                    cpp_b_extra = [cb = std::move(b_extra_fn)](Real t) {
                        py::gil_scoped_acquire gil;
                        return cb(t).cast<Vector>();
                    };
                }
                return std::make_unique<NativeAdapter>(
                    builder.graph(), builder.pool(), cache,
                    std::move(cpp_b_extra));
            }),
            py::arg("builder"),
            py::arg("cache"),
            py::arg("b_extra_fn") = py::none())
        .def("set_mask", &NativeAdapter::set_mask,
            "Set the active mask. Subsequent A_matrix / b_vector / "
            "rhs queries resolve against this mask (cached lazily).")
        .def("current_mask",
            [](const NativeAdapter& self) { return self.current_mask(); },
            "Return the currently-active mask.")
        .def("A_matrix",
            [](const NativeAdapter& self) {
                // Return DenseMatrix by value (Eigen copy is cheap for
                // small n_state). Used by Python tests / diagnostics.
                return self.A_matrix();
            },
            "Continuous-time A matrix for the active mask.")
        .def("b_vector",
            &NativeAdapter::b_vector,
            py::arg("t"),
            "Continuous-time b vector at time `t` (DC + sine + PWM + "
            "pulse + user-supplied overlay).")
        .def("rhs",
            &NativeAdapter::rhs,
            py::arg("t"), py::arg("x"),
            "Continuous-time RHS: dx/dt = A·x + b(t).")
        .def_property_readonly("n_state",
            &NativeAdapter::n_state,
            "Number of continuous-time state variables for the "
            "currently-active mask. Triggers a mask resolution if "
            "not already cached.")
        .def_property_readonly("has_dynamic_sources",
            &NativeAdapter::has_dynamic_sources,
            "Whether any time-varying source is present. Determined "
            "once at construction by probing compute_*_b_extra.")
        .def_property_readonly("num_cached_masks",
            &NativeAdapter::num_cached_masks,
            "Number of distinct masks resolved (and thus cached) "
            "so far. Useful for tests + diagnostics.");

    m.def("run_ped_native_builder",
        [](NativeAdapter& adapter,
           py::object switch_fn_obj,
           Vector x0,
           Real t_end,
           Real rtol,
           Real atol,
           Real dt_init,
           Real dt_max,
           std::size_t store_every) {
            PySwitchFnSSM sf(std::move(switch_fn_obj));
            PIController ctrl(rtol, atol);
            EventPredictor pred;

            // ---- v2.0 Phase 3: auto-derive diode predicates ------
            //
            // The census is the SAME walk run_transient does
            // (DiodeEventState), so the two engines cannot disagree
            // about which switches are diodes. For each diode, two
            // predicates on v_D reconstructed through the recovery
            // map: turn-ON armed while the bit is OFF
            // (g = v_D − V_th), turn-OFF armed while ON
            // (g = v_D — the i_D = g_on·v_D zero-cross). Firing
            // flips the bit inside the scheduler; the zero-time
            // cascade settles instantaneous consequences (a buck's
            // freewheel diode at gate-off).
            std::vector<pulsim::dsed::SchedulerDiode> sched_diodes;
            pulsim::topology::SwitchStateMask diode_owned{0};
            {
                pulsim::pwl::DiodeEventState census(
                    adapter.graph(), adapter.pool());
                if (census.num_diodes() > 0) {
                    diode_owned = census.diode_owned_bits();
                    for (const auto& d : census.entries()) {
                        std::string label =
                            pulsim::pwl::branch_label(
                                adapter.graph(), d.branch_id);
                        const NativeAdapter* ad = &adapter;
                        const pulsim::Index from = d.from;
                        const pulsim::Index to = d.to;
                        const Real vth = d.V_th;
                        pred.add_predicate(pulsim::dsed::EventPredicate{
                            .name = label + " turns ON",
                            .fn = [ad, from, to, vth](
                                      Real t, const Vector& x) {
                                return ad->node_pair_voltage(
                                           from, to, t, x) - vth;
                            },
                            .type = pulsim::dsed::PredicateType::DiodeOn,
                            .priority = pulsim::dsed::
                                default_priority(
                                    pulsim::dsed::PredicateType::DiodeOn),
                            .aux_index =
                                static_cast<int>(d.switch_idx),
                            .required_bit = 0,
                        });
                        pred.add_predicate(pulsim::dsed::EventPredicate{
                            .name = label + " turns OFF",
                            .fn = [ad, from, to](
                                      Real t, const Vector& x) {
                                return ad->node_pair_voltage(
                                           from, to, t, x);
                            },
                            .type = pulsim::dsed::PredicateType::DiodeOff,
                            .priority = pulsim::dsed::
                                default_priority(
                                    pulsim::dsed::PredicateType::DiodeOff),
                            .aux_index =
                                static_cast<int>(d.switch_idx),
                            .required_bit = 1,
                        });
                        sched_diodes.push_back(
                            pulsim::dsed::SchedulerDiode{
                                d.switch_idx, d.from, d.to,
                                d.g_on, d.g_off, d.V_th,
                                std::move(label)});
                    }
                }
            }

            PEDSimulator<NativeAdapter, PySwitchFnSSM> sim(
                adapter, std::move(sf), std::move(ctrl),
                std::move(pred), dt_init, dt_max, store_every);
            if (!sched_diodes.empty()) {
                const auto n_sw = static_cast<pulsim::Size>(
                    adapter.graph().num_switches());
                sim.enable_diode_commutation(
                    diode_owned,
                    pulsim::topology::SwitchStateMask(n_sw),
                    std::move(sched_diodes));
            }
            // Release the GIL during simulate(); the C++ scheduler
            // only re-acquires it at gate edges to call switch_fn.
            py::gil_scoped_release release;
            auto res = sim.simulate(x0, t_end);
            py::gil_scoped_acquire acquire;
            return ped_result_to_dict(res);
        },
        py::arg("adapter"),
        py::arg("switch_fn"),
        py::arg("x0"),
        py::arg("t_end"),
        py::arg("rtol")        = Real{1e-6},
        py::arg("atol")        = Real{1e-9},
        py::arg("dt_init")     = Real{1e-9},
        py::arg("dt_max")      = Real{1e-5},
        py::arg("store_every") = std::size_t{1},
        "Bridge.11 — native PEDSimulator (DOPRI5) over a "
        "NativeCircuitBuilderAdapter. RHS / b_vector / A_matrix all "
        "execute in C++ without GIL roundtrips; the only Python "
        "callback is `switch_fn` at gate edges (rare). v2.0 Phase 3: "
        "PWL diodes are commutated by auto-derived event predicates "
        "on the reconstructed branch voltage.");

    m.def("run_bdf2_native_builder",
        [](NativeAdapter& adapter,
           py::object switch_fn_obj,
           Vector x0,
           Real t_end,
           Real h_fixed,
           std::size_t store_every) {
            PySwitchFnSSM sf(std::move(switch_fn_obj));
            PEDSimulatorBDF2<NativeAdapter, PySwitchFnSSM> sim(
                adapter, std::move(sf), h_fixed, store_every);
            py::gil_scoped_release release;
            auto res = sim.simulate(x0, t_end);
            py::gil_scoped_acquire acquire;
            return ped_result_to_dict(res);
        },
        py::arg("adapter"),
        py::arg("switch_fn"),
        py::arg("x0"),
        py::arg("t_end"),
        py::arg("h_fixed")     = Real{1e-6},
        py::arg("store_every") = std::size_t{1},
        "Bridge.11 — native PEDSimulatorBDF2 over a "
        "NativeCircuitBuilderAdapter (implicit BDF2 + CN bootstrap).");

    m.def("run_auto_native_builder",
        [](NativeAdapter& adapter,
           py::object switch_fn_obj,
           Vector x0,
           Real t_end,
           Real rtol,
           Real atol,
           Real dt_init,
           Real dt_max,
           Real h_bdf2,
           Real stiffness_threshold,
           std::size_t store_every) {
            PySwitchFnSSM sf(std::move(switch_fn_obj));
            PIController ctrl(rtol, atol);
            StiffnessDetector det(stiffness_threshold);
            PEDSimulatorAuto<NativeAdapter, PySwitchFnSSM> sim(
                adapter, std::move(sf), std::move(ctrl),
                std::move(det), dt_init, dt_max, h_bdf2,
                store_every);
            py::gil_scoped_release release;
            auto res = sim.simulate(x0, t_end);
            py::gil_scoped_acquire acquire;
            return ped_result_auto_to_dict(res);
        },
        py::arg("adapter"),
        py::arg("switch_fn"),
        py::arg("x0"),
        py::arg("t_end"),
        py::arg("rtol")                = Real{1e-6},
        py::arg("atol")                = Real{1e-9},
        py::arg("dt_init")             = Real{1e-9},
        py::arg("dt_max")              = Real{1e-5},
        py::arg("h_bdf2")              = Real{1e-6},
        py::arg("stiffness_threshold") = Real{10.0},
        py::arg("store_every")          = std::size_t{1},
        "Bridge.11 — native PEDSimulatorAuto over a "
        "NativeCircuitBuilderAdapter (per-mode RK45↔BDF2 dispatch).");
}

}  // namespace pulsim_dsed_bindings
