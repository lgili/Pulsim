#pragma once

// =============================================================================
// Pulsim — the kernel-built Newton device refresh
// =============================================================================
//
// add-unified-device-interface step 2. Both transient drivers used to build
// the Newton refresh as a chain of five `std::function` wrappers, written
// twice, each capturing the previous one and its own copy of the step size.
// Two defects came from that shape: a wrapper that captured the RAW base
// callback silently deleted every stamp before it (owner note
// nonlinear-refresh-wrapper-chain), and TR-BDF2 probe solves that did not set
// the captured step divided by zero (trbdf2-probe-solves-set-the-device-step).
//
// Here the refresh is ONE function: the static stamps (diode / MOSFET / IGBT,
// or the caller's deprecated override), then one pass per stateful history
// in the fixed `StatefulHistories` order, all reading ONE step context. A
// solve sets the context; no stamp captures a step of its own.

#include "pulsim/device/device_path.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/interface_stamps.hpp"
#include "pulsim/pwl/nonlinear_refresh_igbt_tail.hpp"
#include "pulsim/pwl/nonlinear_refresh_lauritzen_diode.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/pwl/nonlinear_refresh_nonlinear_capacitor.hpp"
#include "pulsim/pwl/nonlinear_refresh_pmsm_mna.hpp"
#include "pulsim/pwl/nonlinear_refresh_saturable_inductor.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/pwl/stateful_histories.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"

#include <algorithm>
#include <concepts>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace pulsim::pwl {

/// The step every device stamp of the next solve integrates over: the
/// interval `h` it advances and the rule (`stage`) of its time formula.
/// Owned by the driver, set before every solve, read by reference.
struct StepCtx {
    Real h = Real{0};
    TrBdf2Stage stage = TrBdf2Stage::Trapezoidal;
};

[[nodiscard]] inline CossStage coss_stage_of(TrBdf2Stage s) noexcept {
    switch (s) {
    case TrBdf2Stage::BackwardEuler: return CossStage::BackwardEuler;
    case TrBdf2Stage::Bdf2Stage2:    return CossStage::Bdf2Stage2;
    default:                         return CossStage::Trapezoidal;
    }
}

/// The static Newton stamps through their `Device` laws (DECISIONS #23):
/// the same kinds as `make_combined_diode_mosfet_refresh`, in ONE graph
/// pass (one kind lookup per branch, not one per kind). Two devices on a
/// shared node therefore add in graph order rather than kind order: the
/// paths differ there by a rounding of the sum, inside the 1e-12 gate.
[[nodiscard]] inline NonlinearRefreshFn make_interface_static_refresh() {
    return [](const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
              const topology::Graph& graph, const DevicePool& pool) -> Real {
        if (J_nl.rows() > 0) sparse::zero_keep_pattern(J_nl);
        if (f_nl.size() > 0) f_nl.setZero();
        using SK = DevicePool::StoredKind;
        namespace L = device::laws;
        Real m = Real{0};
        for (Index b = 0; b < graph.num_branches(); ++b) {
            const auto& br = graph.branch(b);
            if (br.kind != topology::BranchKind::Nonlinear) continue;
            switch (pool.kind_of(br.id)) {
            case SK::NonlinearDiode:
                m = std::max(m, iface::stamp_one_static<L::IdealDiode>(
                    x, J_nl, f_nl, std::array<Index, 2>{br.from, br.to},
                    L::IdealDiode::from(pool.nonlinear_diode_params(br.id))));
                break;
            case SK::ShockleyDiode:
                m = std::max(m, iface::stamp_one_static<L::ShockleyDiode>(
                    x, J_nl, f_nl, std::array<Index, 2>{br.from, br.to},
                    L::ShockleyDiode::from(pool.shockley_diode_params(br.id))));
                break;
            case SK::IgbtLevel1:
                m = std::max(m, iface::stamp_one_static<L::IgbtLevel1>(
                    x, J_nl, f_nl, detail::igbt_level1_terminals(pool, br),
                    L::IgbtLevel1::from(pool.igbt_level1_params(br.id))));
                break;
            case SK::MosfetLevel1:
                m = std::max(m, iface::stamp_one_static<L::MosfetLevel1>(
                    x, J_nl, f_nl, detail::mosfet_level1_terminals(pool, br),
                    L::MosfetLevel1::from(pool.mosfet_level1_params(br.id))));
                break;
            default: break;
            }
        }
        return m;
    };
}

[[nodiscard]] inline NonlinearRefreshFn standard_static_refresh(
    device::DevicePath path) {
    return path == device::DevicePath::Interface
        ? make_interface_static_refresh()
        : make_combined_diode_mosfet_refresh();
}

/// The static Newton stamps: the caller's override when one is given
/// (deprecated; refused on the interface path, which a hand callback
/// would silently bypass), else the kernel's standard chain on `path`
/// when the pool has a static Newton device, else none.
[[nodiscard]] inline NonlinearRefreshFn static_device_refresh(
    const DevicePool& pool, const NonlinearRefreshFn& override_fn,
    device::DevicePath path = device::DevicePath::Legacy) {
    if (override_fn && path == device::DevicePath::Interface) {
        throw std::invalid_argument(
            "device_path='interface' with a caller-supplied nonlinear "
            "refresh: the callback would replace the device laws. Drop "
            "the callback or run with device_path='legacy'.");
    }
    if (override_fn || !pool.requires_static_refresh()) return override_fn;
    return standard_static_refresh(path);
}

/// The refresh of a DC operating point (no stateful pass: DC has no
/// step). `enable` unset means "decide from the pool": the standard chain
/// whenever the pool has any Newton device; false solves them as open
/// circuits (the caller asked for it).
[[nodiscard]] inline NonlinearRefreshFn dc_device_refresh(
    const DevicePool& pool, std::optional<bool> enable,
    device::DevicePath path = device::DevicePath::Legacy) {
    const bool want =
        enable.has_value() ? *enable : pool.has_nonlinear_devices();
    if (!want) return {};
    return standard_static_refresh(path);
}

// The Newton stamp of each stateful kind, reading the step ONLY from `c`.
// `inc`: the engine's form of the saturable flux residual (closed-form
// increment on TR-BDF2, direct on the fixed engine), a run constant.
inline Real stamp_kind(NonlinearCapacitorHistory& k, const Vector& x,
                       sparse::Matrix& J, Vector& f, const StepCtx& c,
                       [[maybe_unused]] bool inc) {
    return refresh_nonlinear_capacitors(x, J, f, k, c.h,
                                        coss_stage_of(c.stage),
                                        &k.q_gamma());
}
inline Real stamp_kind(LauritzenDiodeHistory& k, const Vector& x,
                       sparse::Matrix& J, Vector& f, const StepCtx& c,
                       [[maybe_unused]] bool inc) {
    return refresh_lauritzen_diodes(x, J, f, k, c.h, c.stage);
}
inline Real stamp_kind(PmsmMnaHistory& k, const Vector& x,
                       sparse::Matrix& J, Vector& f, const StepCtx& c,
                       [[maybe_unused]] bool inc) {
    return refresh_pmsm_mna(x, J, f, k, c.h, c.stage);
}
inline Real stamp_kind(IgbtTailHistory& k, const Vector& x,
                       sparse::Matrix& J, Vector& f, const StepCtx& c,
                       [[maybe_unused]] bool inc) {
    return refresh_igbt_tails(x, J, f, k, c.h, c.stage);
}
inline Real stamp_kind(SaturableInductorHistory& k, const Vector& x,
                       sparse::Matrix& J, Vector& f, const StepCtx& c,
                       [[maybe_unused]] bool inc) {
    Real m = Real{0};
    for (const auto& e : k.entries()) {
        m = std::max(m, stamp_saturable_inductor(e, x, J, f, c.h, c.stage,
                                                 inc));
    }
    return m;
}

namespace detail {
template <class K>
concept HasStamp = requires(K& k, const Vector& x, sparse::Matrix& J,
                            Vector& f, const StepCtx& c, bool inc) {
    { stamp_kind(k, x, J, f, c, inc) } -> std::same_as<Real>;
};
template <class List> struct AllStamp;
template <class... Ks> struct AllStamp<device::TypeList<Ks...>> {
    static constexpr bool value = (HasStamp<Ks> && ...);
};
}  // namespace detail
static_assert(detail::AllStamp<StatefulKinds>::value,
              "every StatefulKinds entry needs a stamp_kind overload");

/// The whole Newton refresh: `base` (from `static_device_refresh`), then
/// every stateful kind in list order. Empty when there is nothing to
/// stamp, so a linear circuit keeps its linear path. `hs` and `ctx` are
/// captured by reference and must outlive the returned function.
[[nodiscard]] inline NonlinearRefreshFn build_device_refresh(
    NonlinearRefreshFn base, StatefulHistories& hs, const StepCtx& ctx,
    bool increment_form) {
    if (!hs.any()) return base;
    if (hs.interface()) {
        // DECISIONS #23: every stateful kind through its law (set_path
        // refused the run if one is not ported). Static dispatch.
        return [base = std::move(base), &hs, &ctx, increment_form](
                   const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
                   const topology::Graph& g, const DevicePool& pool) -> Real {
            Real m = Real{0};
            if (base) {
                m = base(x, J_nl, f_nl, g, pool);
            } else {
                sparse::zero_keep_pattern(J_nl);
                f_nl.setZero();
            }
            hs.for_each([&](auto& k) {
                using K = std::remove_cvref_t<decltype(k)>;
                if constexpr (iface::kPorted<K>) {
                    if (!k.empty()) {
                        m = std::max(m, iface::stamp(k, x, J_nl, f_nl, ctx.h,
                                                     ctx.stage, increment_form));
                    }
                }
            });
            return m;
        };
    }
    return [base = std::move(base), &hs, &ctx, increment_form](
               const Vector& x, sparse::Matrix& J_nl, Vector& f_nl,
               const topology::Graph& g, const DevicePool& pool) -> Real {
        Real m = Real{0};
        if (base) {
            m = base(x, J_nl, f_nl, g, pool);
        } else {
            sparse::zero_keep_pattern(J_nl);
            f_nl.setZero();
        }
        hs.for_each([&](auto& k) {
            if (!k.empty()) {
                m = std::max(m, stamp_kind(k, x, J_nl, f_nl, ctx,
                                           increment_form));
            }
        });
        return m;
    };
}

}  // namespace pulsim::pwl
