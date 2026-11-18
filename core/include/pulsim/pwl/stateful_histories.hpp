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

#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/igbt_tail_history.hpp"
#include "pulsim/pwl/lauritzen_diode_history.hpp"
#include "pulsim/pwl/nonlinear_capacitor_history.hpp"
#include "pulsim/pwl/pmsm_mna_history.hpp"
#include "pulsim/pwl/saturable_inductor_history.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/topology/graph.hpp"

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

    /// The run-snapshot fields (each history's own to_flat layout).
    void to_snapshot(solver::SolverSnapshot& s) const {
        s.saturable_history = sat.to_flat();
        s.coss_history = coss.to_flat();
        s.lauritzen_history = laur.to_flat();
        s.igbt_tail_history = tail.to_flat();
        s.pmsm_history = pmsm.to_flat();
    }
    /// Resume: each from_flat refuses a size mismatch by name.
    void from_snapshot(const solver::SolverSnapshot& s) {
        sat.from_flat(s.saturable_history);
        coss.from_flat(s.coss_history);
        laur.from_flat(s.lauritzen_history);
        pmsm.from_flat(s.pmsm_history);
        tail.from_flat(s.igbt_tail_history);
    }
};

}  // namespace pulsim::pwl
