#pragma once

// =============================================================================
// Pulsim — MMC arm as an exact Thevenin equivalent (GGJ aggregation)
// =============================================================================
//
// v2.0 Phase 3, the audit's "obra №2" (finding A.6).
//
// WHAT THIS REPLACES. The L3 arm today integrates every submodule
// capacitor with forward Euler OUTSIDE the solver and couples back
// through a b_extra source read one step late — the code says so
// itself ("The b_extra_fn convention reads this stash one step
// later"). That is a delayed co-simulation: artificial phase lag,
// energy error growing with f_carrier·dt, and no per-SM device in
// the network at all.
//
// THE MODEL. Gnanarathna / Gole / Jayasinghe (IEEE Trans. Power
// Delivery, 2011) — the aggregation PSCAD ships. Discretise each
// submodule capacitor with the SAME trapezoidal companion the rest
// of the network uses:
//
//   inserted SM_i :  R_i = R_on + R_c,   V_i = v_C,i + R_c·i_C,i⁻
//   bypassed SM_i :  R_i = R_on,          V_i = 0
//   R_c = dt / (2·C_SM)
//
// and eliminate the series chain analytically:
//
//   R_eq(t) = Σ R_i        V_eq(t) = Σ V_i        (O(N) per step)
//
// The NETWORK sees one branch: R_eq in series with V_eq — stamped
// as a Norton pair (G_eq, I_N = V_eq·G_eq). A 400-SM arm costs the
// mode cache ZERO bits: gating changes stamp VALUES, not topology,
// and the kernel's etree path refactor (`refactor_parametric`)
// absorbs an R_eq change in O(path). After the network solve
// returns i_arm, each inserted capacitor's v_C is back-solved with
// the same trapezoidal rule. The elimination is ALGEBRAIC — an
// explicit chain of N submodules built from real switches and
// capacitors, driven by the same gates at the same dt, produces the
// same arm current and the same capacitor voltages to rounding.
// The parity test pins exactly that.
//
// WHAT IS NOT A DELAY. V_i uses i_C at the PREVIOUS step — that is
// the trapezoidal companion's own history term, identical to how
// every capacitor in the pwl engine works, not the co-simulation
// lag this model removes.
//
// COMMUTATION CONVENTION. When a submodule leaves the inserted set,
// its capacitor gets a trailing half-step v += R_c·i_C⁻ before the
// companion history clears. That is NOT the textbook GGJ reset
// (PSCAD cuts the current dead at the step boundary): it is what
// THIS engine's trapezoidal network does to an explicit capacitor
// whose series switch opens — in the ideal-off limit the first
// bypassed step solves i_C → 0 and the trap update leaves exactly
// R_c·i_C⁻ behind. The aggregation promises to be the algebraic
// elimination of the explicit chain AS THIS ENGINE DISCRETISES IT,
// so it must inherit that half-step; the two conventions differ by
// O(dt) per edge, and the crown parity test
// (python/tests/test_mmc_thevenin.py) pins the engine-consistent
// one to rounding.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::mmc {

struct ThevArmParams {
    Size n_sm = 0;          //!< submodules per arm
    Real c_sm = Real{0};    //!< capacitance per submodule [F]
    Real r_on = Real{1e-3}; //!< conducting-path resistance per SM [Ω]
    Real dt = Real{0};      //!< the run's fixed step [s]
    Real v_c_init = Real{0};//!< initial capacitor voltage per SM [V]
};

/// The per-step Thevenin pair the network stamps.
struct ThevArmStamp {
    Real r_eq = Real{0};
    Real v_eq = Real{0};
    bool r_changed = false; //!< insertion count changed → refactor
};

/// One half-bridge MMC arm under GGJ aggregation.
///
/// Call sequence per simulation step k (driven by the observer /
/// b_extra hooks — see `pulsim.mmc` Python glue):
///
///   stamp = arm.pre_step(i_arm_{k-1}, n_on_k)
///     — finalises step k-1's capacitor voltages (trapezoidal
///       back-solve under the OLD insertion set), then selects the
///       new set (sort-and-select on v_C, direction from the sign
///       of i_arm) and returns the Thevenin pair for step k.
///
/// The network then solves step k with (r_eq, v_eq); i_arm_k feeds
/// the next call. Current sign convention: positive i_arm flows
/// from the arm's `from` terminal to `to`, CHARGING an inserted
/// capacitor.
class ThevArm {
public:
    explicit ThevArm(const ThevArmParams& p)
        : p_{validated_(p)},
          v_c_(static_cast<Size>(p.n_sm), p.v_c_init),
          i_c_prev_(static_cast<Size>(p.n_sm), Real{0}),
          inserted_(static_cast<Size>(p.n_sm), 0),
          prev_inserted_(static_cast<Size>(p.n_sm), 0),
          order_(static_cast<Size>(p.n_sm)) {
        std::iota(order_.begin(), order_.end(), Size{0});
        r_c_ = p.dt / (Real{2} * p.c_sm);
    }

    /// Finalise the previous step and produce this step's stamp.
    [[nodiscard]] ThevArmStamp pre_step(Real i_arm_prev,
                                          Size n_on) {
        if (n_on > p_.n_sm) {
            throw std::invalid_argument(
                "ThevArm::pre_step: n_on = " + std::to_string(n_on)
                + " exceeds n_sm = " + std::to_string(p_.n_sm));
        }
        // 1. Trapezoidal back-solve of the step just completed,
        //    under the insertion set that step was SOLVED with.
        if (have_prev_) {
            for (Size i = 0; i < p_.n_sm; ++i) {
                if (inserted_[i]) {
                    v_c_[i] += r_c_ * (i_arm_prev + i_c_prev_[i]);
                    i_c_prev_[i] = i_arm_prev;
                }
            }
        }
        have_prev_ = true;

        // 2. Sort-and-select balancing: positive arm current
        //    charges inserted capacitors, so insert the LOWEST
        //    voltages; negative discharges, so insert the highest.
        //    Stable partial sort keeps ties deterministic. The sort
        //    key is v_C after the back-solve — the same information
        //    set an external balancer gating an explicit chain
        //    would act on.
        const Size prev_on = static_cast<Size>(std::count(
            inserted_.begin(), inserted_.end(), uint8_t{1}));
        prev_inserted_ = inserted_;
        std::iota(order_.begin(), order_.end(), Size{0});
        const bool charging = i_arm_prev >= Real{0};
        std::stable_sort(order_.begin(), order_.end(),
            [&](Size a, Size b) {
                return charging ? v_c_[a] < v_c_[b]
                                : v_c_[a] > v_c_[b];
            });
        std::fill(inserted_.begin(), inserted_.end(), uint8_t{0});
        for (Size k = 0; k < n_on; ++k) {
            inserted_[order_[k]] = uint8_t{1};
        }

        // 2b. Commutation convention (see header): a capacitor
        //     LEAVING the inserted set takes the trailing half-step
        //     v += R_c·i_C⁻ its explicit counterpart gets from the
        //     network trap on the first bypassed step (i_C → 0 in
        //     the ideal-off limit), then its companion history
        //     clears so a later re-insertion starts fresh.
        for (Size i = 0; i < p_.n_sm; ++i) {
            if (prev_inserted_[i] && !inserted_[i]) {
                v_c_[i] += r_c_ * i_c_prev_[i];
            }
            if (!inserted_[i]) {
                i_c_prev_[i] = Real{0};
            }
        }

        // 3. The Thevenin pair for the coming step.
        ThevArmStamp out;
        out.r_eq = static_cast<Real>(p_.n_sm) * p_.r_on
                   + static_cast<Real>(n_on) * r_c_;
        Real v_eq = Real{0};
        for (Size i = 0; i < p_.n_sm; ++i) {
            if (inserted_[i]) {
                v_eq += v_c_[i] + r_c_ * i_c_prev_[i];
            }
        }
        out.v_eq = v_eq;
        out.r_changed = (n_on != prev_on) || !stamped_once_;
        stamped_once_ = true;
        return out;
    }

    /// Fold the LAST solved step into the capacitor voltages
    /// without re-selecting (end-of-run bookkeeping). `pre_step`
    /// would also re-run sort-and-select for a step that will never
    /// execute — and any capacitor "leaving" that phantom selection
    /// would take a trailing half-step it never earned.
    void finalize_step(Real i_arm_prev) {
        if (!have_prev_) {
            return;
        }
        for (Size i = 0; i < p_.n_sm; ++i) {
            if (inserted_[i]) {
                v_c_[i] += r_c_ * (i_arm_prev + i_c_prev_[i]);
                i_c_prev_[i] = i_arm_prev;
            }
        }
    }

    [[nodiscard]] const std::vector<Real>& v_c() const noexcept {
        return v_c_;
    }
    /// Pre-charge one submodule (initialisation from a measured or
    /// assumed spread — real studies rarely start perfectly
    /// balanced).
    void set_v_c(Size i, Real v) {
        if (i >= p_.n_sm) {
            throw std::invalid_argument(
                "ThevArm::set_v_c: index out of range");
        }
        v_c_[i] = v;
    }
    [[nodiscard]] const std::vector<uint8_t>& inserted() const
        noexcept {
        return inserted_;
    }
    [[nodiscard]] Real r_c() const noexcept { return r_c_; }
    [[nodiscard]] const ThevArmParams& params() const noexcept {
        return p_;
    }
    [[nodiscard]] Real total_stored_voltage() const noexcept {
        return std::accumulate(v_c_.begin(), v_c_.end(), Real{0});
    }

private:
    // Runs in the member-init list BEFORE the state vectors are
    // constructed — a garbage n_sm (0, or an implausible number
    // arriving through an unsigned conversion) must throw before it
    // sizes an allocation.
    [[nodiscard]] static const ThevArmParams& validated_(
        const ThevArmParams& p) {
        if (p.n_sm == 0 || !(p.c_sm > Real{0}) || !(p.dt > Real{0})) {
            throw std::invalid_argument(
                "ThevArm: n_sm, c_sm and dt must all be positive");
        }
        if (p.n_sm > Size{1000000}) {
            throw std::invalid_argument(
                "ThevArm: n_sm = " + std::to_string(p.n_sm)
                + " is not a plausible arm (did a negative count "
                  "convert to unsigned?)");
        }
        if (!(p.r_on > Real{0})) {
            throw std::invalid_argument(
                "ThevArm: r_on must be positive (an ideal-metal arm "
                "makes R_eq singular with every SM bypassed)");
        }
        return p;
    }

    ThevArmParams p_;
    Real r_c_ = Real{0};
    std::vector<Real> v_c_;
    std::vector<Real> i_c_prev_;
    std::vector<uint8_t> inserted_;
    std::vector<uint8_t> prev_inserted_;
    std::vector<Size> order_;
    bool have_prev_ = false;
    bool stamped_once_ = false;
};

}  // namespace pulsim::mmc
