#pragma once

// =============================================================================
// Pulsim — what a probe / recorder reads from a device (step 9)
// =============================================================================
//
// add-unified-device-interface step 9, for add-probes-and-recorders. Every
// ported Newton kind's pin currents, power and outputs come from its LAW at
// the committed state — read-only functions of (x, the histories); nothing
// here writes state or needs a loop edit.
//
//   i_pin[r]  = f_r(x) + q̇_r          current INTO the device at pin r
//                                     (q̇_r: the committed rate, pins with q)
//   power     = Σ_r v_r·i_pin[r]       absorbed power (pin voltages from x)
//   f, q      = the law's values on its internal and local rows (the
//               device's state outputs: v and −λ of a saturable inductor,
//               −f and q_M of a Lauritzen diode, the IGBT tail's Q, ...)
//
// One branch can have TWO readings: an IGBT with a tail gives its static
// i_ss (`for_each_static_reading`, kind "IgbtLevel1") and the tail's
// delta (`for_each_stateful_reading`, kind "igbt_tail"), same `branch_id`;
// a recorder SUMS the readings of a branch.
//
// During a run the readings need the stepper's live `StatefulHistories`;
// the ONE call the stepper makes is specified in DECISIONS #31
// (`StepHooks::device_state(t, x, histories)` after `commit`).

#include "pulsim/device/device.hpp"
#include "pulsim/device/device_state.hpp"
#include "pulsim/pwl/interface_stamps.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/pwl/stateful_histories.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"

#include <array>
#include <cstddef>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace pulsim::pwl {

struct DeviceReading {
    static constexpr std::size_t kMaxPins = 4, kMaxRows = 2;
    std::string_view kind;         ///< state-block name, or the kind-table name
    std::size_t index = 0;         ///< instance within its kind
    Index branch_id = -1;
    std::size_t n_pins = 0;
    std::array<Real, kMaxPins> i_pin{};
    Real power = Real{0};
    std::size_t n_rows = 0;        ///< internal + local rows
    std::array<Real, kMaxRows> f{}, q{};
};

namespace readings_detail {
template <class Law, std::size_t R>
DeviceReading make(std::string_view kind, std::size_t idx, Index branch,
                   const device::aux_t<Law>& aux,
                   const typename Law::template Params<Real>& p,
                   const std::array<Real, R>& xl, const std::array<Real, R>& qdot) {
    device::Out<Real, R> o;
    device::eval<Law>(aux, p, xl, o);
    DeviceReading d;
    d.kind = kind;
    d.index = idx;
    d.branch_id = branch;
    d.n_pins = Law::n_pins;
    for (std::size_t r = 0; r < Law::n_pins; ++r) {
        d.i_pin[r] = o.f[r] + qdot[r];
        d.power += xl[r] * d.i_pin[r];   // xl[r]: the pin voltage (ground 0)
    }
    d.n_rows = R - Law::n_pins;
    for (std::size_t r = Law::n_pins; r < R; ++r) {
        d.f[r - Law::n_pins] = o.f[r];
        d.q[r - Law::n_pins] = o.q[r];
    }
    return d;
}
}  // namespace readings_detail

/// Visit every instance of the PORTED stateful kinds at the committed
/// state x: `fn(const DeviceReading&)`.
template <class Fn>
void for_each_stateful_reading(const StatefulHistories& hs, const Vector& x, Fn&& fn) {
    namespace rd = readings_detail;
    const auto name = [](device::StateBlock b) {
        return device::kStateBlocks[static_cast<std::size_t>(b)].name;
    };
    {
        const auto& k = hs.get<NonlinearCapacitorHistory>();
        std::size_t i = 0;
        for (const auto& e : k.entries()) {
            const std::array<Index, 2> at{e.from, e.to};
            fn(rd::make<iface::CossLaw, 2>(
                name(k.kBlock), i++, e.branch_id, iface::CossLaw::aux_from(e.params),
                iface::CossLaw::from(e.params), iface::gather<2>(x, at, 2),
                {e.i_prev, -e.i_prev}));
        }
    }
    {
        const auto& k = hs.get<SaturableInductorHistory>();
        std::size_t i = 0;
        for (const auto& e : k.entries()) {
            const std::array<Index, 3> at{e.from, e.to, e.branch_var_id};
            fn(rd::make<iface::SatLaw, 3>(
                name(k.kBlock), i++, e.branch_id,
                iface::sat_aux(e, TrBdf2Stage::Trapezoidal), iface::SatLaw::from(e.params),
                iface::gather<3>(x, at, 2), {}));
        }
    }
    const auto local = [&](const auto& k) {
        using H = std::remove_cvref_t<decltype(k)>;
        using Law = typename iface::LocalLawOf<H>::type;
        std::size_t i = 0;
        for (const auto& e : k.entries()) {
            const auto at = iface::pins_of(e);
            constexpr std::size_t NP = std::tuple_size_v<decltype(at)>;
            std::array<Real, NP + 1> xl{};
            const auto v = iface::gather<NP>(x, at, NP);
            for (std::size_t r = 0; r < NP; ++r) xl[r] = v[r];
            xl[NP] = iface::q_of(e);   // the committed stored charge
            fn(rd::make<Law, NP + 1>(name(k.kBlock), i++, e.branch_id, {},
                                     Law::from(e.params), xl, {}));
        }
    };
    local(hs.get<LauritzenDiodeHistory>());
    local(hs.get<IgbtTailHistory>());
}

namespace readings_detail {
/// The static Newton reading of branch `b` (if it is one): `fn(reading)`;
/// `n` counts the instances per kind (the reading's `index`).
template <class Fn>
void static_branch(const topology::Graph& graph, const DevicePool& pool, const Vector& x,
                   Index b, std::size_t (&n)[5], Fn&& fn) {
    namespace L = device::laws;
    using SK = DevicePool::StoredKind;
    const auto& br = graph.branch(b);
    if (br.kind != topology::BranchKind::Nonlinear || !pool.is_registered(br.id)) return;
    const auto sk = pool.kind_of(br.id);
    const auto two = [&](auto law, const auto& prm, std::size_t slot) {
        using Law = decltype(law);
        const std::array<Index, 2> at{br.from, br.to};
        fn(make<Law, 2>(stored_kind_name(sk), n[slot]++, br.id, {}, prm,
                        iface::gather<2>(x, at, 2), {}));
    };
    const auto three = [&](auto law, const auto& prm, const std::array<Index, 3>& at,
                           std::size_t slot) {
        using Law = decltype(law);
        fn(make<Law, 3>(stored_kind_name(sk), n[slot]++, br.id, {}, prm,
                        iface::gather<3>(x, at, 3), {}));
    };
    switch (sk) {
    case SK::NonlinearDiode:
        two(L::IdealDiode{}, L::IdealDiode::from(pool.nonlinear_diode_params(br.id)), 0);
        break;
    case SK::ShockleyDiode:
        two(L::ShockleyDiode{}, L::ShockleyDiode::from(pool.shockley_diode_params(br.id)), 1);
        break;
    case SK::ExcessLoss:
        two(L::ExcessLoss{}, L::ExcessLoss::from(pool.excess_loss_params(br.id)), 4);
        break;
    case SK::MosfetLevel1:
        three(L::MosfetLevel1{}, L::MosfetLevel1::from(pool.mosfet_level1_params(br.id)),
              detail::mosfet_level1_terminals(pool, br), 2);
        break;
    case SK::IgbtLevel1:
        three(L::IgbtLevel1{}, L::IgbtLevel1::from(pool.igbt_level1_params(br.id)),
              detail::igbt_level1_terminals(pool, br), 3);
        break;
    default: break;
    }
}
}  // namespace readings_detail

/// Visit every static Newton branch (smooth / Shockley diode, MOSFET and
/// IGBT level 1) at x: each law is its model's body, the one both device
/// paths stamp (#44).
template <class Fn>
void for_each_static_reading(const topology::Graph& graph, const DevicePool& pool,
                             const Vector& x, Fn&& fn) {
    std::size_t n[5] = {0, 0, 0, 0, 0};
    for (Index b = 0; b < graph.num_branches(); ++b) {
        readings_detail::static_branch(graph, pool, x, b, n, fn);
    }
}

/// The static Newton reading of the one branch `b` at x (the recorders'
/// per-step read: no walk over the other branches). False when `b` is not
/// a static Newton branch; `out.index` is 0.
[[nodiscard]] inline bool static_reading(const topology::Graph& graph, const DevicePool& pool,
                                         const Vector& x, Index b, DeviceReading& out) {
    std::size_t n[5] = {0, 0, 0, 0, 0};
    bool seen = false;
    readings_detail::static_branch(graph, pool, x, b, n, [&](const DeviceReading& d) {
        out = d;
        seen = true;
    });
    return seen;
}

}  // namespace pulsim::pwl
