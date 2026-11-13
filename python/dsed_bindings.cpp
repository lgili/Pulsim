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
#include "pulsim/dsed/time_eps.hpp"
#include "pulsim/sources/native_pwm_switch_fn.hpp"
#include "pulsim/sources/switch_schedule.hpp"
#include "pulsim/topology/switch_state.hpp"
#include "switch_fn_adapter.hpp"

#include <cstdint>
#include <optional>
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
// Gate-edge search for a switch_fn with no `next_edge_after`.
//
// The schedulers fall back to landing every dt_max/10 (the "poll") when
// the switch_fn cannot name its next edge, and used to take whatever
// mask they found there: every gate edge moved onto the poll grid. At
// the default dt_max = 1e-5 that is a 1 us grid, which deleted a 100 ns
// dead time outright (a synchronous buck's body diodes never conducted)
// and quantised the duty (D = 0.25 at 100 kHz ran as 0.30: 14.28 V
// against 11.49 V).
//
// The search below samples the mask at the end of the poll window and,
// when it changed, bisects (mask evaluations only, no solves) down to
// the time resolution of `time_margin`, returning the first instant
// that already carries the new mask. A pulse that begins AND ends
// inside one poll window is still invisible to it; the dispatch warns
// about that.
// ---------------------------------------------------------------------------
template <class MaskFn>
class EdgeSearch {
public:
    void set_poll(Real poll) noexcept { poll_ = poll; }

    template <class Eval>
    [[nodiscard]] Real next_edge(Real t, Eval&& eval) const {
        if (!(poll_ > Real{0})) {
            return std::numeric_limits<Real>::infinity();
        }
        // The mask is a function of t: the first change after any
        // t' in [from_, edge_) is still edge_.
        if (valid_ && t >= from_ && t < edge_) return edge_;
        // The scheduler sampled the mask just past its last event;
        // match it so an edge-inclusive-left switch_fn is not seen
        // as changing again at t itself.
        // The window end is probed where the scheduler samples after
        // landing, advance_past(hi): landing there next (the usual
        // case), that probe IS m0, one Python call per window less.
        // (Probing AT hi instead broke the common `(t % T) < D * T`
        // switch_fn, whose fmod reads the wrong side at k * T.)
        const MaskFn m0 = (last_ && t == last_hi_)
            ? *last_ : eval(pulsim::dsed::advance_past(t));
        Real lo = t;
        Real hi = t + poll_;
        last_hi_ = hi;
        last_ = eval(pulsim::dsed::advance_past(hi));
        // No change in the window: the plain poll boundary, uncached,
        // exactly as before the search existed (a constant switch_fn,
        // e.g. the default all-closed mask, keeps its step grid).
        if (*last_ == m0) return hi;
        // An edge ON the poll boundary (a PWM whose edges sit on the
        // poll grid) costs one evaluation instead of ~50: the search
        // made the 100 kHz buck of test_dsed_without_scipy 3.5x
        // slower (19 -> 68 ms) without it.
        if (!(eval(hi - pulsim::dsed::time_margin(hi)) == m0)) {
            for (int it = 0; it < 200; ++it) {
                if (hi - lo <= pulsim::dsed::time_margin(hi)) break;
                const Real mid = lo + Real{0.5} * (hi - lo);
                if (!(mid > lo && mid < hi)) break;
                if (eval(mid) == m0) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
        }
        from_ = t;
        edge_ = hi;
        valid_ = true;
        return hi;
    }

private:
    Real poll_ = Real{0};
    mutable std::optional<MaskFn> last_;
    mutable Real last_hi_ = Real{0};
    mutable bool valid_ = false;
    mutable Real from_ = Real{0};
    mutable Real edge_ = Real{0};
};

// ---------------------------------------------------------------------------
// PySwitchFn: adapter that holds a Python `switch_fn` callable and an
// optional `next_edge_after(t)` method. Returns PyMask from operator().
// Without `next_edge_after`, edges are bisected inside each poll
// window once `set_poll` was called (see EdgeSearch).
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
        return search_.next_edge(
            t, [this](Real tau) { return (*this)(tau); });
    }

    void set_poll(Real poll) noexcept { search_.set_poll(poll); }

private:
    py::object fn_;
    py::object nea_attr_;
    bool has_nea_ = false;
    EdgeSearch<PyMask> search_;
};

/// The poll window the schedulers use for a switch_fn without an edge
/// oracle (scheduler.hpp / scheduler_auto.hpp / scheduler_bdf2.hpp:
/// one tenth of their step ceiling).
[[nodiscard]] inline Real poll_window(Real step_ceiling) noexcept {
    return step_ceiling * Real{0.1};
}

// ---------------------------------------------------------------------------
// PEDResult / PEDResultAuto → Python dict converters.
//
// We return a dict so callers can index by name; the existing
// _PEDSimulationResult wrapper in `python/pulsim/_dsed_dispatch.py`
// already handles arbitrary attribute access.
// ---------------------------------------------------------------------------

/// The mask each stored sample is reconstructed and recorded under:
/// the STAMPED mask, i.e. the one in force over the step that ended
/// at that sample (PEDResult::step_masks). Falls back to the
/// post-event sample_masks for a result built without step masks.
template <class ResultT>
const auto& stamped_masks_of(const ResultT& res) {
    return res.step_masks.size() == res.times.size() ? res.step_masks
                                                     : res.sample_masks;
}

/// v2.0 Phase 3 item 5 — reconstruct the FULL MNA trajectory from
/// the reduced samples through the adapter's per-mask recovery map.
/// Samples are grouped by mask so a run that visits M modes pays M
/// mask resolutions, not one per sample. Returns an empty array if
/// the result carries no per-sample masks (a non-builder path).
///
/// Each sample is reconstructed through its STAMPED mask (the mask of
/// the step that produced it), from the state that step ended in. At
/// a sample where an event fired that is the pre-event (left-limit)
/// state, the same convention the kernel engines use: a TR-BDF2
/// landing sample is a solve under the mask from before the flip, and
/// the new mode first appears on the next sample. Recording the
/// post-event mask instead booked one whole step per event to the
/// wrong mode (a DCM boost diode's duty read 0.200 against a true
/// 0.178 at the default dt_max).
template <class AdapterT, class ResultT>
py::array_t<double> reconstruct_full_states(
    AdapterT& adapter, const ResultT& res) {
    const auto n = res.times.size();
    const auto& masks = stamped_masks_of(res);
    if (n == 0 || masks.size() != n) {
        return py::array_t<double>(py::ssize_t{0});
    }
    const bool have_pre =
        res.step_masks.size() == n
        && res.pre_event_index.size() == res.pre_event_states.size();
    std::size_t next_pre = 0;
    auto state_of = [&](std::size_t i) -> const pulsim::Vector& {
        if (have_pre && next_pre < res.pre_event_index.size()
            && res.pre_event_index[next_pre] == i) {
            return res.pre_event_states[next_pre++];
        }
        return res.states[i];
    };
    // Probe the width from the first sample.
    adapter.set_mask(masks[0]);
    const pulsim::Vector first =
        adapter.recover_full(res.times[0], state_of(0));
    const auto n_mna = static_cast<py::ssize_t>(first.size());
    py::array_t<double> out(
        {static_cast<py::ssize_t>(n), n_mna});
    auto* ptr = out.mutable_data();
    for (py::ssize_t j = 0; j < n_mna; ++j) {
        ptr[j] = first[j];
    }
    for (std::size_t i = 1; i < n; ++i) {
        if (!(masks[i] == masks[i - 1])) {
            adapter.set_mask(masks[i]);
        }
        double* row = ptr + static_cast<py::ssize_t>(i) * n_mna;
        if constexpr (requires {
                          adapter.recover_full_into(
                              Real{0}, state_of(i),
                              Eigen::Map<pulsim::Vector>(row, n_mna));
                      }) {
            adapter.recover_full_into(
                res.times[i], state_of(i),
                Eigen::Map<pulsim::Vector>(row, n_mna));
        } else {
            const pulsim::Vector full =
                adapter.recover_full(res.times[i], state_of(i));
            for (py::ssize_t j = 0; j < n_mna; ++j) {
                row[j] = full[j];
            }
        }
    }
    return out;
}

/// The RIGHT limit of every sample where an event changed the state
/// or the mask: the full MNA state rebuilt through the POST-event mask
/// (sample_masks[k]) from the post-event integrator state (states[k]).
/// It is where the NEXT step starts. Written into `d` as
/// `right_limit_index` (sample indices, increasing),
/// `right_limit_states` ((m, n_mna)) and `right_limit_switch_states`
/// ((m, n_switches) bool).
///
/// Why: a stored DSED event sample is the LEFT limit (the state the
/// step that ended there produced, reconstruct_full_states), so a
/// switched branch's current jumps between that sample and the start
/// of the next step. A trapezoid drawn from the left limit spread a
/// freewheel diode's 0 -> I_L turn-on jump as a ramp over the whole
/// following step (1 us by default): a DCM boost diode's average
/// current read 0.0234 A against 0.0535 A. Integrating each step from
/// the right limit of its start sample to the left limit of its end
/// sample closes that charge exactly.
template <class AdapterT, class ResultT>
void reconstruct_right_limits(AdapterT& adapter, const ResultT& res,
                              py::dict& d) {
    const auto n = res.times.size();
    std::vector<std::size_t> idx;
    using MaskT = typename decltype(res.sample_masks)::value_type;
    if constexpr (!std::is_same_v<MaskT, PyMask>) {
        if (n > 0 && res.sample_masks.size() == n
            && res.step_masks.size() == n) {
            std::size_t next_pre = 0;
            for (std::size_t i = 0; i < n; ++i) {
                bool fired = false;
                while (next_pre < res.pre_event_index.size()
                       && res.pre_event_index[next_pre] < i) {
                    ++next_pre;
                }
                // A fired event that left the state untouched (a
                // same-mask gate edge is dropped inside fire_event_,
                // and a source-only circuit "fires" its t_end edge on
                // every step) has right limit == left limit: skip it,
                // or the record doubles the state storage for nothing.
                if (next_pre < res.pre_event_index.size()
                    && res.pre_event_index[next_pre] == i
                    && next_pre < res.pre_event_states.size()) {
                    const auto& xp = res.pre_event_states[next_pre];
                    const auto& xs = res.states[i];
                    fired = xp.size() != xs.size()
                            || (xp.array() != xs.array()).any();
                }
                if (fired || !(res.step_masks[i] == res.sample_masks[i])) {
                    idx.push_back(i);
                }
            }
        }
    }
    const auto m = static_cast<py::ssize_t>(idx.size());
    py::array_t<std::int64_t> idx_arr(m);
    auto* ip = idx_arr.mutable_data();
    for (py::ssize_t j = 0; j < m; ++j) {
        ip[j] = static_cast<std::int64_t>(idx[static_cast<std::size_t>(j)]);
    }
    d["right_limit_index"] = std::move(idx_arr);
    if (m == 0) {
        d["right_limit_states"] =
            py::array_t<double>(std::vector<py::ssize_t>{0, 0});
        d["right_limit_switch_states"] =
            py::array_t<bool>(std::vector<py::ssize_t>{0, 0});
        return;
    }
    if constexpr (!std::is_same_v<MaskT, PyMask>) {
        const auto& m0 = res.sample_masks[idx[0]];
        adapter.set_mask(m0);
        const pulsim::Vector first =
            adapter.recover_full(res.times[idx[0]], res.states[idx[0]]);
        const auto n_mna = static_cast<py::ssize_t>(first.size());
        const auto n_sw = static_cast<py::ssize_t>(m0.size());
        py::array_t<double> st(std::vector<py::ssize_t>{m, n_mna});
        py::array_t<bool> sw(std::vector<py::ssize_t>{m, n_sw});
        auto* sp = st.mutable_data();
        auto* wp = sw.mutable_data();
        for (py::ssize_t j = 0; j < m; ++j) {
            const auto i = idx[static_cast<std::size_t>(j)];
            const auto& mask = res.sample_masks[i];
            if (j > 0) {
                adapter.set_mask(mask);
            }
            const pulsim::Vector full = (j == 0)
                ? first
                : adapter.recover_full(res.times[i], res.states[i]);
            for (py::ssize_t c = 0; c < n_mna; ++c) {
                sp[j * n_mna + c] = full[c];
            }
            for (py::ssize_t c = 0; c < n_sw; ++c) {
                wp[j * n_sw + c] = mask.get(static_cast<pulsim::Size>(c));
            }
        }
        d["right_limit_states"] = std::move(st);
        d["right_limit_switch_states"] = std::move(sw);
    }
}

/// Per-sample switch / diode mask as an (n_samples, n_switches) bool
/// array — the same record `SimulationResult.switch_states` carries
/// on the pwl / trbdf2 engines: row k is the STAMPED mask, the one in
/// force over (t[k-1], t[k]]. Every DSED sample's full MNA state is
/// reconstructed THROUGH that mask (reconstruct_full_states), so
/// i = g(mask)·v holds at every sample by construction. Only the
/// native SwitchStateMask path has a typed mask; the PyMask path (no
/// PWL diodes, reduced states only) returns an empty array.
template <class ResultT>
py::array_t<bool> sample_masks_to_array(const ResultT& res) {
    using MaskT = typename decltype(res.sample_masks)::value_type;
    if constexpr (std::is_same_v<MaskT, PyMask>) {
        return py::array_t<bool>(std::vector<py::ssize_t>{0, 0});
    } else {
        const auto& masks = stamped_masks_of(res);
        const auto n = static_cast<py::ssize_t>(masks.size());
        if (n == 0 || masks.size() != res.times.size()) {
            return py::array_t<bool>(std::vector<py::ssize_t>{0, 0});
        }
        const auto n_sw =
            static_cast<py::ssize_t>(masks[0].size());
        py::array_t<bool> out(std::vector<py::ssize_t>{n, n_sw});
        auto* ptr = out.mutable_data();
        for (py::ssize_t i = 0; i < n; ++i) {
            const auto& m = masks[static_cast<std::size_t>(i)];
            for (py::ssize_t j = 0; j < n_sw; ++j) {
                ptr[i * n_sw + j] = m.get(static_cast<pulsim::Size>(j));
            }
        }
        return out;
    }
}

/// Event times and predicate types as arrays (cheap, always present).
template <class ResultT>
void event_arrays(const ResultT& res, py::dict& d) {
    const auto m = static_cast<py::ssize_t>(res.event_log.size());
    py::array_t<double> et(m);
    py::array_t<int> ty(m);
    for (py::ssize_t i = 0; i < m; ++i) {
        const auto& e = res.event_log[static_cast<std::size_t>(i)];
        et.mutable_data()[i] = e.t;
        ty.mutable_data()[i] = static_cast<int>(e.type);
    }
    d["event_times"] = std::move(et);
    d["event_types"] = std::move(ty);
}

template <class MaskT>
py::dict ped_result_to_dict(
    const pulsim::dsed::PEDResult<MaskT>& res, bool event_log = true) {
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
    d["switch_states"] = sample_masks_to_array(res);
    d["n_accept"] = res.n_accept;
    d["n_reject"] = res.n_reject;
    d["n_events"] = res.n_events;
    d["n_exact_steps"] = res.n_exact_steps;
    d["n_predicate_evals"] = res.n_predicate_evals;
    d["cpu_time_seconds"] = res.cpu_time_seconds;
    event_arrays(res, d);
    // Event log: skip the MaskT objects (they're py::object wrappers
    // around Python objects that we just hand back). event_log=False
    // (the simulate() path) skips the per-event dicts and their two
    // mask objects each: a 2000-period buck has 4000 events.
    py::list events;
    for (const auto& e : res.event_log) {
        if (!event_log) break;
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
    const pulsim::dsed::PEDResultAuto<MaskT>& res, bool event_log = true) {
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
    d["switch_states"] = sample_masks_to_array(res);
    d["n_accept"] = res.n_accept;
    d["n_reject"] = res.n_reject;
    d["n_events"] = res.n_events;
    d["n_rk45_steps"] = res.n_rk45_steps;
    d["n_bdf2_steps"] = res.n_bdf2_steps;
    d["n_exact_steps"] = res.n_exact_steps;
    d["cpu_time_seconds"] = res.cpu_time_seconds;
    event_arrays(res, d);
    py::list events;
    for (const auto& e : res.event_log) {
        if (!event_log) break;
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
/// D10 — the exact propagator's per-mask tier record and counters.
template <class AdapterT>
py::dict exact_report_dict(const AdapterT& adapter,
                           std::size_t n_predicate_evals) {
    py::list masks;
    for (const auto& r : adapter.exact_report()) {
        py::list bits;
        for (pulsim::Size i = 0; i < r.mask.size(); ++i) {
            bits.append(static_cast<bool>(r.mask.get(i)));
        }
        py::dict row;
        row["mask"] = bits;
        row["tier"] = r.tier;
        row["kappa"] = r.kappa;
        row["norm_Ah"] = r.norm_Ah;
        row["reason"] = r.reason;
        masks.append(row);
    }
    py::dict d;
    d["masks"] = masks;
    d["n_eig"] = adapter.n_eig();
    d["n_pade"] = adapter.n_pade();
    d["n_phi_cache_hits"] = adapter.n_pade_hits();
    d["n_predicate_evals"] = n_predicate_evals;
    return d;
}

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
            sf.set_poll(poll_window(dt_max));
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
            sf.set_poll(poll_window(h_fixed));
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
            sf.set_poll(poll_window(dt_max));
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
            // The native schedules (SwitchSchedule, NativePwm2Switch,
            // NativeMultiMaskPwm) are recognised by the adapter every
            // engine binding shares; they run without the GIL.
            if (py::isinstance<pulsim::sources::SwitchSchedule>(fn_)
                || py::isinstance<pulsim::sources::NativePwm2Switch>(fn_)
                || py::isinstance<pulsim::sources::NativeMultiMaskPwm>(fn_)) {
                native_ = pulsim_bindings_common::adapt_switch_fn(fn_);
                is_native_ = true;
            } else if (py::hasattr(fn_, "next_edge_after")) {
                // Generic Python callable: pre-resolve the attribute
                // to avoid the hasattr cost per call.
                nea_attr_ = fn_.attr("next_edge_after");
                has_nea_  = true;
            }
        }

        [[nodiscard]] SSM operator()(Real t) const {
            if (is_native_) return native_.fn(t);
            // Generic Python fallback (Bridge.11 path).
            py::gil_scoped_acquire gil;
            return fn_(t).cast<SSM>();
        }

        [[nodiscard]] Real next_edge_after(Real t) const {
            if (is_native_ && native_.next_edge) return native_.next_edge(t);
            if (has_nea_) {
                py::gil_scoped_acquire gil;
                return nea_attr_(t).cast<Real>();
            }
            // No oracle: bisect inside the poll window (EdgeSearch).
            return search_.next_edge(
                t, [this](Real tau) { return (*this)(tau); });
        }

        void set_poll(Real poll) noexcept { search_.set_poll(poll); }

        py::object fn_;
        pulsim_bindings_common::SwitchFnAdapter native_;
        bool       is_native_ = false;
        py::object nea_attr_;
        bool       has_nea_ = false;
        EdgeSearch<SSM> search_;
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
        .def("reduced_initial_state",
            [](const NativeAdapter& ad,
               const pulsim::builder::CircuitBuilder& b) {
                return ad.reduced_initial_state(b.initial_conditions());
            },
            py::arg("builder"),
            "add-exact-propagator-strategy D5: the reduced state of the "
            "current mask from a full-MNA initial vector "
            "(the builder's c0= / i0= / set_initial), and the largest capacitor-voltage "
            "adjustment the charge-conserving projection made (0 when the "
            "declared values are consistent).")
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
           std::size_t store_every,
           bool enable_event_projection,
           bool event_log) {
            PySwitchFnSSM sf(std::move(switch_fn_obj));
            sf.set_poll(poll_window(dt_max));
            PIController ctrl(rtol, atol);
            EventPredictor pred;

            // ---- v2.0 Phase 3: auto-derive diode predicates ------
            //
            // The census is the SAME walk run_transient does
            // (DiodeEventState), so the two engines cannot disagree
            // about which switches are diodes. For each diode, two
            // predicates on v_D reconstructed through the recovery
            // map: turn-ON armed while the bit is OFF
            // (g = v_D − V_on), turn-OFF armed while ON
            // (g = v_D − V_f — the i_D = g_on·(v_D − V_f) zero-cross),
            // both SwitchedDiode::crossing_signal. Firing
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
                        const auto prm = d.params();
                        pred.add_predicate(pulsim::dsed::EventPredicate{
                            .name = label + " turns ON",
                            .fn = [ad, from, to, prm](
                                      Real t, const Vector& x) {
                                return pulsim::models::SwitchedDiode::
                                    crossing_signal(
                                        false,
                                        ad->node_pair_voltage(
                                            from, to, t, x),
                                        prm);
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
                            .fn = [ad, from, to, prm](
                                      Real t, const Vector& x) {
                                return pulsim::models::SwitchedDiode::
                                    crossing_signal(
                                        true,
                                        ad->node_pair_voltage(
                                            from, to, t, x),
                                        prm);
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
                                d.g_on, d.g_off, d.V_th, d.V_f,
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
            sim.set_event_projection(enable_event_projection);
            // Release the GIL during simulate(); the C++ scheduler
            // only re-acquires it at gate edges to call switch_fn.
            py::gil_scoped_release release;
            auto res = sim.simulate(x0, t_end);
            py::gil_scoped_acquire acquire;
            auto d = ped_result_to_dict(res, event_log);
            d["exact"] = exact_report_dict(adapter, res.n_predicate_evals);
            // v2.0 Phase 3 item 5 — the unified result: full MNA
            // trajectory reconstructed per recorded sample.
            d["states_full"] = reconstruct_full_states(adapter, res);
            reconstruct_right_limits(adapter, res, d);
            return d;
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
        py::arg("enable_event_projection") = true,
        py::arg("event_log") = true,
        "Bridge.11 — native PEDSimulator (DOPRI5) over a "
        "NativeCircuitBuilderAdapter. RHS / b_vector / A_matrix all "
        "execute in C++ without GIL roundtrips; the only Python "
        "callback is `switch_fn` at gate edges (rare). v2.0 Phase 3: "
        "PWL diodes are commutated by auto-derived event predicates "
        "on the reconstructed branch voltage, and every event "
        "projects the state onto the new mode's slow manifold "
        "(consistent reinitialization; pass "
        "enable_event_projection=False for the raw pre-projection "
        "behaviour).");

    m.def("run_bdf2_native_builder",
        [](NativeAdapter& adapter,
           py::object switch_fn_obj,
           Vector x0,
           Real t_end,
           Real h_fixed,
           std::size_t store_every,
           bool event_log) {
            PySwitchFnSSM sf(std::move(switch_fn_obj));
            sf.set_poll(poll_window(h_fixed));
            PEDSimulatorBDF2<NativeAdapter, PySwitchFnSSM> sim(
                adapter, std::move(sf), h_fixed, store_every);
            py::gil_scoped_release release;
            auto res = sim.simulate(x0, t_end);
            py::gil_scoped_acquire acquire;
            auto d = ped_result_to_dict(res, event_log);
            d["states_full"] = reconstruct_full_states(adapter, res);
            reconstruct_right_limits(adapter, res, d);
            return d;
        },
        py::arg("adapter"),
        py::arg("switch_fn"),
        py::arg("x0"),
        py::arg("t_end"),
        py::arg("h_fixed")     = Real{1e-6},
        py::arg("store_every") = std::size_t{1},
        py::arg("event_log") = true,
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
           std::size_t store_every,
           bool event_log) {
            PySwitchFnSSM sf(std::move(switch_fn_obj));
            sf.set_poll(poll_window(dt_max));
            PIController ctrl(rtol, atol);
            StiffnessDetector det(stiffness_threshold);
            PEDSimulatorAuto<NativeAdapter, PySwitchFnSSM> sim(
                adapter, std::move(sf), std::move(ctrl),
                std::move(det), dt_init, dt_max, h_bdf2,
                store_every);
            py::gil_scoped_release release;
            auto res = sim.simulate(x0, t_end);
            py::gil_scoped_acquire acquire;
            auto d = ped_result_auto_to_dict(res, event_log);
            d["exact"] = exact_report_dict(adapter, 0);
            d["states_full"] = reconstruct_full_states(adapter, res);
            reconstruct_right_limits(adapter, res, d);
            return d;
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
        py::arg("event_log") = true,
        "Bridge.11 — native PEDSimulatorAuto over a "
        "NativeCircuitBuilderAdapter (per-mode RK45↔BDF2 dispatch).");
}

}  // namespace pulsim_dsed_bindings
