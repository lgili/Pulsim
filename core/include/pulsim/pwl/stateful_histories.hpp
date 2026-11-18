#pragma once

// =============================================================================
// Pulsim — the five stateful-device histories as one object
// =============================================================================
//
// add-unified-device-interface step 1. The Coss, Lauritzen, PMSM, IGBT-tail
// and saturable-inductor histories share one de-facto API; the two transient
// drivers used to call each by name (99 call sites), so a sixth kind or a new
// commit path had to be added in every place and was silently missing where
// it was not. Every driver operation is now one call here that visits all
// five in one fixed order. The histories' arithmetic is unchanged.
//
// The per-kind argument conventions (Coss takes a bool for backward Euler
// and has its own BDF2 commit; the saturable inductor takes no h) are
// mapped here, once, from (h, TrBdf2Stage).

#include "pulsim/device/device_state.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/history_state.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/igbt_tail_history.hpp"
#include "pulsim/pwl/lauritzen_diode_history.hpp"
#include "pulsim/pwl/nonlinear_capacitor_history.hpp"
#include "pulsim/pwl/pmsm_mna_history.hpp"
#include "pulsim/pwl/saturable_inductor_history.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/topology/graph.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::pwl {

class StatefulHistories {
public:
    NonlinearCapacitorHistory coss;
    LauritzenDiodeHistory laur;
    PmsmMnaHistory pmsm;
    IgbtTailHistory tail;
    SaturableInductorHistory sat;

    /// A rollback point: every history's entries.
    struct Snapshot {
        std::vector<NonlinearCapacitorHistory::Entry> coss;
        std::vector<LauritzenDiodeHistory::Entry> laur;
        std::vector<PmsmMnaHistory::Entry> pmsm;
        std::vector<IgbtTailHistory::Entry> tail;
        std::vector<SaturableInductorHistory::Entry> sat;
    };

    /// Visit the five histories in the one fixed order (also the stamp
    /// order of the Newton refresh).
    template <class F> void for_each(F&& f) {
        f(coss); f(laur); f(pmsm); f(tail); f(sat);
    }
    template <class F> void for_each(F&& f) const {
        f(coss); f(laur); f(pmsm); f(tail); f(sat);
    }

    void init(const topology::Graph& graph, const DevicePool& pool) {
        for_each([&](auto& h) { h.init(graph, pool); });
    }
    [[nodiscard]] bool any() const noexcept {
        bool a = false;
        for_each([&](const auto& h) { a = a || !h.empty(); });
        return a;
    }
    void seed_from_dc_op(const Vector& x) {
        for_each([&](auto& h) { h.seed_from_dc_op(x); });
    }

    /// Commit the just-solved step of `h` under `stage` (the rule its
    /// solve stamped). `Bdf2Stage2` needs `capture_gamma` at the stage
    /// point of the same step first.
    void commit(const Vector& x, Real h, TrBdf2Stage stage) {
        if (stage == TrBdf2Stage::Bdf2Stage2) {
            coss.commit_bdf2_stage2(x, h);
        } else {
            coss.update_from_state(x, h,
                                   stage == TrBdf2Stage::BackwardEuler);
        }
        laur.update_from_state(x, h, stage);
        pmsm.update_from_state(x, h, stage);
        tail.update_from_state(x, h, stage);
        sat.update_from_state(x, stage);
    }
    void capture_gamma(const Vector& x_gamma) {
        for_each([&](auto& h) { h.capture_gamma(x_gamma); });
    }
    /// SDIRK restart: pseudo previous state x0 + c·(x1 − x0).
    void extrapolate_previous(const Vector& x1, Real c) {
        for_each([&](auto& h) { h.extrapolate_previous(x1, c); });
    }
    /// TR-BDF2 post-event refresh of the derivative slots from a
    /// trapezoidal probe of step `h`; the states are not touched.
    void refresh_derivatives(const Vector& x_probe, Real h) {
        coss.refresh_derivatives(x_probe, h);
        laur.refresh_derivatives(x_probe);
        pmsm.refresh_derivatives(x_probe);
        tail.refresh_derivatives(x_probe);
        sat.refresh_derivatives(x_probe);
    }

    /// Into a preallocated rollback point (assign keeps the capacity).
    void snapshot_into(Snapshot& s) const {
        s.coss.assign(coss.entries().begin(), coss.entries().end());
        s.laur.assign(laur.entries().begin(), laur.entries().end());
        s.pmsm.assign(pmsm.entries().begin(), pmsm.entries().end());
        s.tail.assign(tail.entries().begin(), tail.entries().end());
        s.sat.assign(sat.entries().begin(), sat.entries().end());
    }
    void restore(const Snapshot& s) {
        coss.restore(s.coss);
        laur.restore(s.laur);
        pmsm.restore(s.pmsm);
        tail.restore(s.tail);
        sat.restore(s.sat);
    }

    /// The device-state layout of a run: `history` (the linear
    /// companions) and these five.
    [[nodiscard]] device::DeviceStateLayout layout(
        const HistoryState& history) const {
        using device::StateBlock;
        device::DeviceStateLayout l;
        auto put = [&l](StateBlock b, std::size_t n) {
            l.count[static_cast<std::size_t>(b)] = static_cast<std::uint32_t>(n);
        };
        put(StateBlock::LinearCompanion, history.num_dynamic_branches());
        put(StateBlock::SaturableInductor, sat.entries().size());
        put(StateBlock::NonlinearCapacitor, coss.entries().size());
        put(StateBlock::LauritzenDiode, laur.entries().size());
        put(StateBlock::IgbtTail, tail.entries().size());
        put(StateBlock::PmsmMna, pmsm.entries().size());
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
        auto append = [&s](const std::vector<Real>& v) {
            s.dev_state.insert(s.dev_state.end(), v.begin(), v.end());
        };
        append(sat.to_flat());
        append(coss.to_flat());
        append(laur.to_flat());
        append(tail.to_flat());
        append(pmsm.to_flat());
        s.legacy = {};
    }

    /// Resume: refuses a layout that is not this circuit's, naming the
    /// first block that differs; a legacy payload is converted field by
    /// field, each history refusing a size mismatch by name.
    void load(const solver::SolverSnapshot& s, HistoryState& history,
              const char* engine) {
        using device::StateBlock;
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
            if (!diff.empty() || s.dev_layout.size() != device::kNumStateBlocks
                || s.layout_signature != theirs.signature()
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
        history.from_flat(s.view(StateBlock::LinearCompanion));
        sat.from_flat(s.view(StateBlock::SaturableInductor));
        coss.from_flat(s.view(StateBlock::NonlinearCapacitor));
        laur.from_flat(s.view(StateBlock::LauritzenDiode));
        pmsm.from_flat(s.view(StateBlock::PmsmMna));
        pmsm.rebase_reference(s.x);   // increment-form reference = x
        tail.from_flat(s.view(StateBlock::IgbtTail));
    }
};

}  // namespace pulsim::pwl
