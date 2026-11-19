#pragma once

// =============================================================================
// Pulsim — the stateful-device kinds of a run, as one object
// =============================================================================
//
// add-unified-device-interface steps 1 and 4. The Coss, Lauritzen, PMSM,
// IGBT-tail and saturable-inductor histories are the kinds of one list,
// `StatefulKinds`, each modelling `device::StatefulKind`
// (device/kind.hpp). The two transient drivers used to call each by name
// (99 call sites), so a sixth kind or a new commit path had to be added in
// every place and was silently missing where it was not. Every driver
// operation is now one fold over the list, in its fixed order (also the
// stamp order of the Newton refresh). The kinds' arithmetic is unchanged.
//
// The in-run rollback is ONE flat vector of reals: every real each kind
// carries from step to step, its stage scratch included (`for_each_state`;
// the Coss kind-level `q_gamma` is not: `capture_gamma` rewrites it before
// every read), so
// restoring it is bit-identical to restoring the entries. The snapshot of a
// run (`save` / `load`) is the physical subset, `to_flat`, in the block
// order of `device::kStateBlocks`.

#include "pulsim/device/device_state.hpp"
#include "pulsim/device/kind.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/history_state.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/external_device_history.hpp"
#include "pulsim/pwl/igbt_tail_history.hpp"
#include "pulsim/pwl/interface_stamps.hpp"
#include "pulsim/pwl/lauritzen_diode_history.hpp"
#include "pulsim/pwl/nonlinear_capacitor_history.hpp"
#include "pulsim/pwl/pmsm_mna_history.hpp"
#include "pulsim/pwl/saturable_inductor_history.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

namespace pulsim::pwl {

/// THE list of stateful Newton kinds, in fold (and stamp) order. A new
/// kind is appended here (design.md, "How to add a device kind").
using StatefulKinds = device::TypeList<
    NonlinearCapacitorHistory, LauritzenDiodeHistory, PmsmMnaHistory,
    IgbtTailHistory, SaturableInductorHistory, ExternalDeviceHistory>;

using StatefulCheck = device::detail::ListCheck<StatefulKinds>;
static_assert(StatefulCheck::kinds,
              "every StatefulKinds entry must model device::StatefulKind");
static_assert(StatefulCheck::reals,
              "a kind's kFlatPerDevice must equal its block's "
              "reals_per_instance in device::kStateBlocks");
static_assert(StatefulCheck::newton,
              "a stateful kind's kStoredKind must be a Newton row of "
              "kKindCaps (engines without a Newton solve refuse it)");
static_assert(StatefulCheck::blocks_claimed_once(),
              "every device::StateBlock but LinearCompanion must be "
              "claimed by exactly one stateful kind");

class StatefulHistories {
public:
    /// A rollback point: every real of every kind, stage scratch included.
    using Snapshot = std::vector<Real>;

    template <class K> [[nodiscard]] K& get() noexcept {
        return std::get<K>(kinds_);
    }
    template <class K> [[nodiscard]] const K& get() const noexcept {
        return std::get<K>(kinds_);
    }

    /// Visit the kinds in list order.
    template <class F> void for_each(F&& f) {
        std::apply([&](auto&... k) { (f(k), ...); }, kinds_);
    }
    template <class F> void for_each(F&& f) const {
        std::apply([&](const auto&... k) { (f(k), ...); }, kinds_);
    }

    void init(const topology::Graph& graph, const DevicePool& pool) {
        for_each([&](auto& k) { k.init(graph, pool); });
    }
    [[nodiscard]] bool any() const noexcept {
        bool a = false;
        for_each([&](const auto& k) { a = a || !k.empty(); });
        return a;
    }
    void seed_from_dc_op(const Vector& x) {
        for_each([&](auto& k) { k.seed_from_dc_op(x); });
    }

    /// Commit the just-solved step of `h` under `stage` (the rule its
    /// solve stamped). `Bdf2Stage2` needs `capture_gamma` at the stage
    /// point of the same step first.
    void commit(const Vector& x, Real h, TrBdf2Stage stage) {
        for_each([&](auto& k) {
            using K = std::remove_cvref_t<decltype(k)>;
            if constexpr (iface::kPorted<K>) {
                if (interface()) { iface::commit(k, x, h, stage); return; }
            }
            k.commit(x, h, stage);
        });
    }
    void capture_gamma(const Vector& x_gamma) {
        for_each([&](auto& k) {
            using K = std::remove_cvref_t<decltype(k)>;
            if constexpr (iface::kPorted<K>) {
                if (interface()) { iface::capture_gamma(k, x_gamma); return; }
            }
            k.capture_gamma(x_gamma);
        });
    }

    // ---- the device path (DECISIONS #23) ----
    /// Choose the path for this run. `Interface` refuses, by name, a
    /// circuit holding a Newton kind whose law is not on the `Device`
    /// contract yet (a run is never mixed).
    void set_path(device::DevicePath p, const topology::Graph& graph,
                  const DevicePool& pool, const char* engine) {
        path_ = p;
        if (p != device::DevicePath::Interface) return;
        std::vector<std::string> names;
        auto add = [&names](std::string_view name) {
            if (std::find(names.begin(), names.end(), name) == names.end()) {
                names.emplace_back(name);
            }
        };
        for_each([&](const auto& k) {
            using K = std::remove_cvref_t<decltype(k)>;
            if (!iface::kPorted<K> && !k.empty()) {
                add(device::kStateBlocks[static_cast<std::size_t>(K::kBlock)].name);
            }
        });
        for (Index b = 0; b < graph.num_branches(); ++b) {
            const Index id = graph.branch(b).id;
            if (!pool.is_registered(id)) continue;
            const auto sk = pool.kind_of(id);
            if (kind_caps(sk).cls == SolveClass::NewtonStatic
                && !iface::static_kind_ported(sk)) {
                add(stored_kind_name(sk));
            }
            if (kind_caps(sk).cls == SolveClass::NewtonStateful) {
                bool covered = false;
                for_each([&](const auto& k) {
                    using K = std::remove_cvref_t<decltype(k)>;
                    if (iface::kPorted<K> && iface::history_covers<K>(sk)) covered = true;
                });
                if (!covered) add(stored_kind_name(sk));
            }
        }
        if (!names.empty()) {
            std::string missing;
            for (const auto& n : names) missing += (missing.empty() ? "" : ", ") + n;
            throw std::invalid_argument(
                std::string(engine) + "(device_path='interface'): not on the "
                "device interface yet: " + missing
                + ". Run this circuit with device_path='legacy'.");
        }
    }
    [[nodiscard]] device::DevicePath path() const noexcept { return path_; }
    [[nodiscard]] bool interface() const noexcept {
        return path_ == device::DevicePath::Interface;
    }
    // ---- the stage form (βh, ψ) of integrator='esdirk43' ----
    // Every kind's stage takes an arbitrary base: one q row per state
    // (q_n, q̇_n); a stage is backward Euler of βh from q_n = ψ. The rows
    // run in fold order, kind after kind.
    [[nodiscard]] std::size_t stage_rows() const {
        std::size_t n = 0;
        for_each([&n](const auto& k) { n += k.stage_rows(); });
        return n;
    }
    void stage_start(Real hh, Real* q0, Real* hqd0) const {
        for_each([&](const auto& k) {
            k.stage_start(hh, q0, hqd0);
            q0 += k.stage_rows();
            hqd0 += k.stage_rows();
        });
    }
    /// Base ψ of every row, and ψ − q_n(step start) built from increments.
    void stage_base(const Real* psi, const Real* dpsi) {
        for_each([&](auto& k) {
            k.stage_base(psi, dpsi);
            psi += k.stage_rows();
            dpsi += k.stage_rows();
        });
    }
    /// d = q(x) − ψ per row (ψ: the bases just set; the charge kinds read
    /// the K0/K1 of the stage's last stamp).
    void stage_increment(const Vector& x, const Real* psi, Real* d) const {
        for_each([&](const auto& k) {
            k.stage_increment(x, psi, d);
            psi += k.stage_rows();
            d += k.stage_rows();
        });
    }
    /// Commit the stage just solved from the bases set (backward Euler of
    /// h_e); `h_step` is the whole step (an external model's accept()).
    void stage_commit(const Vector& x, Real h_e, Real h_step) {
        for_each([&](auto& k) {
            using K = std::remove_cvref_t<decltype(k)>;
            if constexpr (std::is_same_v<K, ExternalDeviceHistory>) {
                k.commit(x, h_e, TrBdf2Stage::BackwardEuler, h_step);
            } else {
                if constexpr (iface::kPorted<K>) {
                    if (interface()) {
                        iface::commit(k, x, h_e, TrBdf2Stage::BackwardEuler);
                        return;
                    }
                }
                // the charge kinds commit through K0/K1: those of THIS base
                // and h_e (a rollback restored the pre-step pair)
                if constexpr (requires { k.begin_step(h_e, TrBdf2Stage::BackwardEuler); }) {
                    k.begin_step(h_e, TrBdf2Stage::BackwardEuler);
                }
                k.commit(x, h_e, TrBdf2Stage::BackwardEuler);
            }
        });
    }
    /// TR-BDF2's error estimate, between the BDF2 solve and the commit:
    /// per q row, q − q_n at the γ point (the stage-1 capture) and at the
    /// step end (the BDF2 stage's own law, as `commit` will compute it).
    void trbdf2_increments(const Vector& x1, Real* dqg, Real* dq1) const {
        for_each([&](const auto& k) {
            if (k.stage_rows() == 0) return;
            k.trbdf2_increments(x1, dqg, dq1);
            dqg += k.stage_rows();
            dq1 += k.stage_rows();
        });
    }
    /// Per q row: atol_q / atol, the device's own conversion from the
    /// electrical unknown its charge or flux follows (a capacitance, an
    /// inductance, a time constant); 1 where a law gives none.
    void q_scales(Real* s) const {
        for_each([&](const auto& k) {
            const std::size_t n = k.stage_rows();
            if (n == 0) return;
            k.q_scales(s);
            for (std::size_t r = 0; r < n; ++r) {
                if (!(s[r] > Real{0}) || !std::isfinite(s[r])) s[r] = Real{1};
            }
            s += n;
        });
    }

    /// After the stages: every scratch of the stage form back to rest.
    void stage_clear() {
        for_each([](auto& k) { k.stage_clear(); });
    }

    /// SDIRK restart: pseudo previous state x0 + c·(x1 − x0).
    void extrapolate_previous(const Vector& x1, Real c) {
        for_each([&](auto& k) { k.extrapolate_previous(x1, c); });
    }
    /// TR-BDF2 post-event refresh of the derivative slots from a
    /// trapezoidal probe of step `h`; the states are not touched.
    void refresh_derivatives(const Vector& x_probe, Real h) {
        for_each([&](auto& k) { k.refresh_derivatives(x_probe, h); });
    }

    /// Into a preallocated rollback point (its size is the run's, so only
    /// the first call of a run allocates).
    void snapshot_into(Snapshot& s) const {
        s.resize(reals_in_rollback());
        std::size_t p = 0;
        for_each([&](const auto& k) {
            using K = std::remove_cvref_t<decltype(k)>;
            for (const auto& e : k.entries()) {
                K::for_each_state(e, [&](const auto& v) {
                    s[p++] = static_cast<Real>(v);
                });
            }
        });
    }
    /// Number of reals `snapshot_into` writes for this run.
    [[nodiscard]] std::size_t reals_in_rollback() const {
        std::size_t n = 0;
        for_each([&n](const auto& k) {
            using K = std::remove_cvref_t<decltype(k)>;
            for (const auto& e : k.entries()) {
                K::for_each_state(e, [&n](const auto&) { ++n; });
            }
        });
        return n;
    }
    void restore(const Snapshot& s) {
        // Rare (a rejected step), so the size is checked every time: a
        // stale or never-taken rollback point must not be read past its end.
        if (s.size() != reals_in_rollback()) {
            throw std::logic_error(
                "StatefulHistories::restore: the rollback point holds "
                + std::to_string(s.size()) + " reals, this run carries "
                + std::to_string(reals_in_rollback()));
        }
        std::size_t p = 0;
        for_each([&](auto& k) {
            using K = std::remove_cvref_t<decltype(k)>;
            for (auto& e : k.mutable_entries()) {
                K::for_each_state(e, [&](auto& v) {
                    v = static_cast<std::remove_cvref_t<decltype(v)>>(s[p++]);
                });
            }
        });
    }

    /// The device-state layout of a run: `history` (the linear
    /// companions) and one block per kind.
    [[nodiscard]] device::DeviceStateLayout layout(
        const HistoryState& history) const {
        device::DeviceStateLayout l;
        l.count[static_cast<std::size_t>(device::StateBlock::LinearCompanion)] =
            static_cast<std::uint32_t>(history.num_dynamic_branches());
        for_each([&l](const auto& k) {
            l.count[static_cast<std::size_t>(k.kBlock)] =
                static_cast<std::uint32_t>(k.entries().size());
        });
        return l;
    }

    /// The whole device state into `s` (dev_state, dev_layout, signature).
    void save(const HistoryState& history, solver::SolverSnapshot& s) const {
        const auto l = layout(history);
        s.dev_layout.assign(l.count.begin(), l.count.end());
        s.layout_signature = l.signature();
        // layout order (device::kStateBlocks); the first block's vector
        // is reused, so a run end allocates as little as the 2.0 fields
        s.dev_state = history.to_flat();
        s.dev_state.reserve(l.size());
        for_each_in_block_order([&s](const auto& k) {
            const auto v = k.to_flat();
            s.dev_state.insert(s.dev_state.end(), v.begin(), v.end());
        });
        s.legacy = {};
    }

    /// Resume: refuses a layout that is not this circuit's, naming the
    /// first block that differs; a legacy payload is converted field by
    /// field, each kind refusing a size mismatch by name.
    void load(const solver::SolverSnapshot& s, HistoryState& history,
              const char* engine) {
        if (s.is_legacy() && !s.dev_state.empty()) {
            // a hand-filled dev_state with no layout would be ignored
            throw std::invalid_argument(
                std::string(engine) + "(resume_from=...): the snapshot "
                "has a dev_state of " + std::to_string(s.dev_state.size())
                + " values but no dev_layout, so they cannot be placed. "
                "Take dev_state and dev_layout from the same snapshot.");
        }
        if (!s.is_legacy()) {
            const auto mine = layout(history);
            const auto theirs = s.layout();
            const std::string diff = mine.first_difference(theirs);
            // A snapshot from a build with fewer blocks (appended later)
            // loads with the missing blocks empty.
            if (!diff.empty() || s.dev_layout.size() > device::kNumStateBlocks
                || s.layout_signature != theirs.signature(s.dev_layout.size())
                || s.dev_state.size() != theirs.size()) {
                throw std::invalid_argument(
                    std::string(engine) + "(resume_from=...): the snapshot "
                    "belongs to a different circuit: "
                    + (diff.empty()
                           ? std::string("its device-state layout is corrupt "
                                         "(signature or size)")
                           : diff + " (this circuit, then the snapshot)")
                    + ". Resuming from it would start the devices from "
                      "wrong state.");
            }
        }
        history.from_flat(s.view(device::StateBlock::LinearCompanion));
        for_each_in_block_order(
            [&s](auto& k) { k.from_flat(s.view(k.kBlock)); });
        // increment-form reference = the snapshot's x
        get<PmsmMnaHistory>().rebase_reference(s.x);
    }

private:
    StatefulKinds::Tuple kinds_;
    device::DevicePath path_ = device::DevicePath::Legacy;

    /// The kinds in snapshot block order (device::kStateBlocks), which is
    /// not the fold order.
    template <class F> void for_each_in_block_order(F&& f) const {
        for (std::size_t b = 1; b < device::kNumStateBlocks; ++b) {
            for_each([&](const auto& k) {
                if (static_cast<std::size_t>(k.kBlock) == b) f(k);
            });
        }
    }
    template <class F> void for_each_in_block_order(F&& f) {
        for (std::size_t b = 1; b < device::kNumStateBlocks; ++b) {
            for_each([&](auto& k) {
                if (static_cast<std::size_t>(k.kBlock) == b) f(k);
            });
        }
    }
};

}  // namespace pulsim::pwl
