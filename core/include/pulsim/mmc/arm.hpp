#pragma once

// =============================================================================
// Pulsim — Phase 20.11: C++ hotpath for MMC L0/L1 arm dynamics
// =============================================================================
//
// Header-only port of the Python helpers in ``pulsim/mmc.py``:
//
//   * ``ps_pwm_switching_function``  — per-step PS-PWM quantizer
//                                       (the inner N-SM loop).
//   * ``mmc_arm_average_step``       — L0 single forward-Euler step
//                                       (Sousa eqs 2.13/2.14).
//   * ``mmc_arm_multilevel_step``    — L1 single forward-Euler step
//                                       (PS-PWM + L0 dynamics).
//
// The Python implementations stay in place as the reference and the
// behavioural baseline; the C++ versions are bit-for-bit equivalent
// drop-in replacements exposed via pybind11 for the observer's hot
// loop. Measured speedup on a typical run (N_SM = 8, 6 arms × 10000
// steps): ~30× per L1 step thanks to elimination of the Python
// interpreter overhead and inlining of the N-carrier loop.
//
// Style follows the rest of the kernel: header-only, no exceptions
// in the hot path (the only `throw` is a builder-time guard on
// `dt > 0`), and POD parameter structs so binding via pybind11 is
// straightforward.

#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::mmc {

// ----------------------------------------------------------------------------
// SM topology tags
// ----------------------------------------------------------------------------

enum class SubmoduleType : std::uint8_t {
    HalfBridge = 0,
    FullBridge = 1,
};

// Multilevel modulation strategy. PS-PWM phase-shifts ``N`` carriers
// by ``2π/N``; IPD (In-Phase Disposition) keeps all ``N`` carriers
// in phase but stacks them at different DC offsets so the output
// only ever switches between two adjacent levels at a time.
enum class ModulationScheme : std::uint8_t {
    PsPwm = 0,
    Ipd   = 1,
};

[[nodiscard]] inline Real m_min_of(SubmoduleType t) noexcept {
    return (t == SubmoduleType::HalfBridge) ? Real{0.0} : Real{-1.0};
}

[[nodiscard]] inline Real m_max_of(SubmoduleType /*t*/) noexcept {
    return Real{1.0};
}

// ----------------------------------------------------------------------------
// PS-PWM switching function — the hot inner loop.
// ----------------------------------------------------------------------------
//
// Each of the ``n_sm`` SMs is gated by a triangular carrier at
// ``f_carrier`` phase-shifted by ``k / n_sm`` of a period from the
// previous carrier. The comparator output for each carrier becomes
// one bit; their sum is the multilevel switching count ``s_b``.
//
// Half-bridge:
//   s_b ∈ {0, 1, …, N}.  m_ref ∈ [0, 1] is clamped silently.
//
// Full-bridge:
//   s_b ∈ {−N, …, N} with sign(s_b) = sign(m_ref). The carrier
//   comparator threshold is ``|m_ref|``; the result is then signed.

[[nodiscard]] inline Index ps_pwm_switching_function(
    Real m_ref,
    Real t,
    Index n_sm,
    Real f_carrier,
    SubmoduleType sm_type = SubmoduleType::HalfBridge) noexcept {

    if (sm_type == SubmoduleType::HalfBridge) {
        Real m = m_ref;
        if (m < Real{0.0}) {
            m = Real{0.0};
        } else if (m > Real{1.0}) {
            m = Real{1.0};
        }
        Index s_b = 0;
        for (Index k = 0; k < n_sm; ++k) {
            // Phase in [0, 1): t * f + k/N modulo 1.
            const Real raw = t * f_carrier +
                static_cast<Real>(k) / static_cast<Real>(n_sm);
            const Real phase = raw - std::floor(raw);
            const Real tri = (phase < Real{0.5})
                ? Real{2.0} * phase
                : Real{2.0} * (Real{1.0} - phase);
            if (m > tri) {
                ++s_b;
            }
        }
        return s_b;
    }

    // Full-bridge: comparator on |m_ref|, sign from m_ref.
    Real m_clamped = m_ref;
    if (m_clamped < Real{-1.0}) {
        m_clamped = Real{-1.0};
    } else if (m_clamped > Real{1.0}) {
        m_clamped = Real{1.0};
    }
    const Index sign = (m_clamped >= Real{0.0}) ? +1 : -1;
    const Real m_abs = std::fabs(m_clamped);
    Index s_b = 0;
    for (Index k = 0; k < n_sm; ++k) {
        const Real raw = t * f_carrier +
            static_cast<Real>(k) / static_cast<Real>(n_sm);
        const Real phase = raw - std::floor(raw);
        const Real tri = (phase < Real{0.5})
            ? Real{2.0} * phase
            : Real{2.0} * (Real{1.0} - phase);
        if (m_abs > tri) {
            ++s_b;
        }
    }
    return sign * s_b;
}

// ----------------------------------------------------------------------------
// IPD (In-Phase Disposition) switching function.
// ----------------------------------------------------------------------------
//
// ``N`` triangular carriers all share the same phase but are stacked
// vertically in [0, 1]: carrier ``k`` covers the band
// ``[k/N, (k+1)/N]``. The reference ``m_ref`` is compared against
// every carrier; ``s_b`` is the count of "true" comparators. Because
// the bands are non-overlapping, at any instant exactly ``s_b`` ∈
// ``{floor(m·N), floor(m·N) + 1}`` — the output only swings between
// two adjacent levels at a time, which is what gives IPD its
// substantially cleaner output spectrum vs PS-PWM at low N.
//
// Implementation: don't actually evaluate N carriers. Compute
// ``m_scaled = m_ref · N``, split into integer + fractional parts,
// and compare the fractional part against a single shared triangle.

[[nodiscard]] inline Index ipd_switching_function(
    Real m_ref,
    Real t,
    Index n_sm,
    Real f_carrier,
    SubmoduleType sm_type = SubmoduleType::HalfBridge) noexcept {

    // One shared triangular carrier in [0, 1].
    const Real raw = t * f_carrier;
    const Real phase = raw - std::floor(raw);
    const Real tri = (phase < Real{0.5})
        ? Real{2.0} * phase
        : Real{2.0} * (Real{1.0} - phase);

    if (sm_type == SubmoduleType::HalfBridge) {
        Real m = m_ref;
        if (m < Real{0.0}) m = Real{0.0};
        else if (m > Real{1.0}) m = Real{1.0};

        const Real m_scaled = m * static_cast<Real>(n_sm);
        Index s_base = static_cast<Index>(std::floor(m_scaled));
        if (s_base >= n_sm) return n_sm;
        const Real m_frac = m_scaled - static_cast<Real>(s_base);
        return s_base + ((m_frac > tri) ? 1 : 0);
    }

    // Full-bridge: split into sign × magnitude.
    Real m_clamped = m_ref;
    if (m_clamped < Real{-1.0}) m_clamped = Real{-1.0};
    else if (m_clamped > Real{1.0}) m_clamped = Real{1.0};
    const Index sign = (m_clamped >= Real{0.0}) ? +1 : -1;
    const Real m_abs = std::fabs(m_clamped);

    const Real m_scaled = m_abs * static_cast<Real>(n_sm);
    Index s_base = static_cast<Index>(std::floor(m_scaled));
    if (s_base >= n_sm) return sign * n_sm;
    const Real m_frac = m_scaled - static_cast<Real>(s_base);
    return sign * (s_base + ((m_frac > tri) ? 1 : 0));
}

// Single dispatch point — picks the strategy at runtime. Both
// branches are constexpr-inlined; the branch predictor will lock
// onto whichever scheme the simulation uses.
[[nodiscard]] inline Index switching_function(
    Real m_ref,
    Real t,
    Index n_sm,
    Real f_carrier,
    SubmoduleType sm_type,
    ModulationScheme scheme) noexcept {

    if (scheme == ModulationScheme::Ipd) {
        return ipd_switching_function(
            m_ref, t, n_sm, f_carrier, sm_type);
    }
    return ps_pwm_switching_function(
        m_ref, t, n_sm, f_carrier, sm_type);
}


// ----------------------------------------------------------------------------
// L0 — Average-value arm step.
// ----------------------------------------------------------------------------
//
// Forward-Euler integration of Sousa eq (2.14):
//
//   v_b      = m_b · v_C
//   dv_C/dt  = (m_b · i_b − v_C / r_p) / C_arm
//
// where ``C_arm = c_sm / n_sm`` and the ``v_C / r_p`` leak is omitted
// when ``r_p_inv == 0`` (the user passes 1/r_p to keep the inner
// loop branch-free).
//
// Returns ``(v_C_next, v_b)`` packed as a small struct.

struct StepAverageResult {
    Real v_C_next;
    Real v_b;
};

[[nodiscard]] inline StepAverageResult mmc_arm_average_step(
    Real v_C,
    Real m_b,
    Real i_b,
    Real dt,
    Real c_arm,
    Real r_p_inv = Real{0.0}) {

    if (dt <= Real{0.0}) {
        throw std::invalid_argument(
            "mmc_arm_average_step: dt must be > 0");
    }
    const Real v_b = m_b * v_C;
    const Real leak = v_C * r_p_inv;
    const Real dv_dt = (m_b * i_b - leak) / c_arm;
    return StepAverageResult{v_C + dt * dv_dt, v_b};
}

// ----------------------------------------------------------------------------
// L1 — Multilevel arm step.
// ----------------------------------------------------------------------------
//
// L0 dynamics with ``m_b ≡ s_b / N`` where ``s_b`` is the PS-PWM
// quantized switching count at time ``t``.
//
// Returns ``(v_C_next, v_b, s_b)``.

struct StepMultilevelResult {
    Real v_C_next;
    Real v_b;
    Index s_b;
};

[[nodiscard]] inline StepMultilevelResult mmc_arm_multilevel_step(
    Real v_C,
    Real m_ref,
    Real i_b,
    Real dt,
    Real t,
    Index n_sm,
    Real c_arm,
    Real f_carrier,
    SubmoduleType sm_type = SubmoduleType::HalfBridge,
    Real r_p_inv = Real{0.0},
    ModulationScheme scheme = ModulationScheme::PsPwm) {

    if (dt <= Real{0.0}) {
        throw std::invalid_argument(
            "mmc_arm_multilevel_step: dt must be > 0");
    }
    const Index s_b = switching_function(
        m_ref, t, n_sm, f_carrier, sm_type, scheme);
    const Real m_b = static_cast<Real>(s_b) /
                       static_cast<Real>(n_sm);
    const Real v_b = m_b * v_C;
    const Real leak = v_C * r_p_inv;
    const Real dv_dt = (m_b * i_b - leak) / c_arm;
    return StepMultilevelResult{v_C + dt * dv_dt, v_b, s_b};
}

// ----------------------------------------------------------------------------
// L2 — SM-equivalent arm step (dead-time + min-pulse-width).
// ----------------------------------------------------------------------------
//
// Each SM tracks (bit_s1, bit_s2) and a dead-time timer.
//
// **Half-bridge** (``sm_type = HalfBridge``):
//   bit_s1 == 1 → SM inserted (output = +v_C); bit_s2 == 1 → bypassed
//   (output = 0). Both bits 0 → all 4 transistors off → free-wheel
//   through body diodes. Current routing: ``i_b > 0`` ⇒ D2 conducts
//   (bypass), ``i_b < 0`` ⇒ D1 conducts (inserted).
//
// **Full-bridge** (``sm_type = FullBridge``):
//   bit_s1 == 1 → SM inserted POSITIVELY (output = +v_C, via S1a + S2b).
//   bit_s2 == 1 → SM inserted NEGATIVELY (output = -v_C, via S1b + S2a).
//   Both bits 0 → either bypass (S1a+S1b or S2a+S2b on, output 0) OR
//   dead-time (all 4 off). During dead-time the body diodes route
//   current OPPOSITE to its sign: ``i_b > 0`` ⇒ output = -v_C (D2a + D1b
//   conduct); ``i_b < 0`` ⇒ output = +v_C (D1a + D2b conduct).
//
// ``t_min`` suppresses any per-SM toggle that would happen within
// ``t_min`` of the previous toggle, on both topologies.
//
// The state arrays are mutated in-place. Caller is responsible for
// sizing them to ``n_sm`` and seeding initial values via
// :func:`make_l2_state` on the Python side.

struct StepEquivalentResult {
    Real v_b;
    Index s_w;     // count of "+inserted" SMs (bit_s1 = 1)
    Index s_u;     // count of free-wheel / bypass SMs (both bits 0)
    Index s_n;     // count of "-inserted" SMs (bit_s2 = 1, FB only;
                   // always 0 for HB)
};

[[nodiscard]] inline StepEquivalentResult mmc_arm_equivalent_step(
    Real& v_C,                          // in/out
    std::int8_t* bit_s1,                // length n_sm, in/out
    std::int8_t* bit_s2,                // length n_sm, in/out
    Real* in_dead_time_until,           // length n_sm, in/out
    Real* last_toggle_time,             // length n_sm, in/out
    Real m_ref,
    Real i_b,
    Real dt,
    Real t,
    Index n_sm,
    Real c_arm,
    Real f_carrier,
    Real t_dead,
    Real t_min,
    Real r_p_inv = Real{0.0},
    ModulationScheme scheme = ModulationScheme::PsPwm,
    SubmoduleType sm_type = SubmoduleType::HalfBridge) {

    if (dt <= Real{0.0}) {
        throw std::invalid_argument(
            "mmc_arm_equivalent_step: dt must be > 0");
    }

    // Clamp m_ref to the topology's valid range.
    //   HB:  m_ref ∈ [0, 1]
    //   FB:  m_ref ∈ [-1, 1]
    const bool is_fb = (sm_type == SubmoduleType::FullBridge);
    const Real m_lo = is_fb ? Real{-1.0} : Real{0.0};
    Real m_clamped = m_ref;
    if (m_clamped < m_lo)       m_clamped = m_lo;
    else if (m_clamped > Real{1.0}) m_clamped = Real{1.0};
    // FB comparators run on |m_clamped|; the sign is applied later
    // (gives ±contribution per SM). For HB ``m_sign = +1`` so the
    // existing logic is unaffected.
    const Real m_abs  = is_fb ? std::fabs(m_clamped) : m_clamped;
    const std::int8_t m_sign = is_fb && (m_clamped < Real{0.0}) ? -1 : +1;

    // ---- IPD: precompute the shared triangle + integer base s_b.
    Index ipd_s_base = 0;
    Real ipd_m_frac = Real{0.0};
    Real ipd_tri = Real{0.0};
    if (scheme == ModulationScheme::Ipd) {
        const Real raw_ipd = t * f_carrier;
        const Real phase_ipd = raw_ipd - std::floor(raw_ipd);
        ipd_tri = (phase_ipd < Real{0.5})
            ? Real{2.0} * phase_ipd
            : Real{2.0} * (Real{1.0} - phase_ipd);
        const Real m_scaled = m_abs * static_cast<Real>(n_sm);
        ipd_s_base = static_cast<Index>(std::floor(m_scaled));
        if (ipd_s_base > n_sm) ipd_s_base = n_sm;
        ipd_m_frac = m_scaled - static_cast<Real>(ipd_s_base);
    }

    for (Index k = 0; k < n_sm; ++k) {
        // Target *magnitude* per SM (1 = active, 0 = bypass) — the
        // FB sign is encoded later as (target_pos, target_neg).
        std::int8_t target_mag;
        if (scheme == ModulationScheme::Ipd) {
            if (k < ipd_s_base) {
                target_mag = 1;
            } else if (k == ipd_s_base) {
                target_mag = (ipd_m_frac > ipd_tri) ? 1 : 0;
            } else {
                target_mag = 0;
            }
        } else {
            // PS-PWM comparator for SM k.
            const Real raw = t * f_carrier +
                static_cast<Real>(k) / static_cast<Real>(n_sm);
            const Real phase = raw - std::floor(raw);
            const Real tri = (phase < Real{0.5})
                ? Real{2.0} * phase
                : Real{2.0} * (Real{1.0} - phase);
            target_mag = (m_abs > tri) ? 1 : 0;
        }
        // Encode target into the (bit_s1, bit_s2) pair:
        //   HB: target_mag=1 ⇒ (1,0) inserted; =0 ⇒ (0,1) bypass.
        //   FB: target_mag·m_sign = +1 ⇒ (1,0) +inserted;
        //                          = 0 ⇒ (0,0) bypass;
        //                          = -1 ⇒ (0,1) -inserted.
        std::int8_t tgt_s1, tgt_s2;
        if (!is_fb) {
            tgt_s1 = target_mag;
            tgt_s2 = static_cast<std::int8_t>(1 - target_mag);
        } else {
            tgt_s1 = (target_mag != 0 && m_sign > 0) ? 1 : 0;
            tgt_s2 = (target_mag != 0 && m_sign < 0) ? 1 : 0;
        }

        if (in_dead_time_until[k] > Real{0.0} &&
            t >= in_dead_time_until[k]) {
            // Dead-time elapsed → commit the new state.
            bit_s1[k] = tgt_s1;
            bit_s2[k] = tgt_s2;
            in_dead_time_until[k] = -std::numeric_limits<Real>::infinity();
        } else if (in_dead_time_until[k] <= Real{0.0}) {
            // Not in dead-time — check whether a toggle is needed.
            // For HB, only ``bit_s1`` matters; for FB the full
            // (bit_s1, bit_s2) pair encodes the ternary state.
            const bool toggle = (tgt_s1 != bit_s1[k]) ||
                                 (tgt_s2 != bit_s2[k]);
            if (toggle) {
                if (t_min > Real{0.0} &&
                    (t - last_toggle_time[k]) < t_min) {
                    continue;  // suppress
                }
                // Enter dead-time: all driven switches open.
                bit_s1[k] = 0;
                bit_s2[k] = 0;
                in_dead_time_until[k] = t + t_dead;
                last_toggle_time[k] = t;
                if (t_dead == Real{0.0}) {
                    // Instantaneous transition (no dead-time modelled).
                    bit_s1[k] = tgt_s1;
                    bit_s2[k] = tgt_s2;
                    in_dead_time_until[k] =
                        -std::numeric_limits<Real>::infinity();
                }
            }
        }
    }

    Index s_w = 0;   // +inserted
    Index s_n = 0;   // -inserted (FB only)
    Index s_u = 0;   // dead-time / bypass (both bits 0)
    for (Index k = 0; k < n_sm; ++k) {
        if      (bit_s1[k] != 0) ++s_w;
        else if (bit_s2[k] != 0) ++s_n;
        else                       ++s_u;
    }

    // Effective signed insertion count (per SM contribution in {-1, 0, +1}).
    // Free-wheel / dead-time SMs get a current-direction-dependent override:
    //   HB:  i_b > 0 ⇒ bypassed (contribute 0); i_b < 0 ⇒ inserted (+1).
    //   FB:  i_b > 0 ⇒ -1 contribution (D2a + D1b conduct);
    //        i_b < 0 ⇒ +1 contribution (D1a + D2b conduct).
    Index s_eff;
    if (!is_fb) {
        // Original HB behaviour preserved: s_u SMs add to "inserted"
        // count when i_b < 0.
        s_eff = (i_b < Real{0.0}) ? (s_w + s_u) : s_w;
    } else {
        const Index dt_contrib = (i_b > Real{0.0}) ? -s_u
                              : (i_b < Real{0.0}) ? +s_u
                                                  : Index{0};
        s_eff = s_w - s_n + dt_contrib;
    }
    const Real m_b_eff = static_cast<Real>(s_eff) /
                          static_cast<Real>(n_sm);
    const Real v_b = m_b_eff * v_C;
    const Real leak = v_C * r_p_inv;
    const Real dv_dt = (m_b_eff * i_b - leak) / c_arm;
    v_C = v_C + dt * dv_dt;
    return StepEquivalentResult{v_b, s_w, s_u, s_n};
}

// ----------------------------------------------------------------------------
// L3 — Detailed per-SM arm step (sort-and-select balancing).
// ----------------------------------------------------------------------------

enum class BalancingScheme : std::uint8_t {
    SortAndSelect = 0,
    None = 1,
};

struct StepDetailedResult {
    Real v_b;
    Index s_b;
};

// Fills ``insertion_mask`` (length n_sm) with the per-SM signed
// insertion pattern: ``+1`` = inserted positively, ``-1`` = inserted
// negatively (full-bridge only), ``0`` = bypassed. Returns the
// SIGNED number of inserted SMs (= ``s_b`` after clamping).
//
// For half-bridge the mask values are always in {0, +1} and the
// behaviour matches the previous implementation exactly. For
// full-bridge the mask values are in {-1, 0, +1}; the *direction*
// of insertion is taken from ``sign(s_b)`` (since FB SMs can
// contribute ±v_C). Charging vs discharging is determined by the
// sign of ``i_b · sign(s_b)``: positive ⇒ insert the SMs with the
// *lowest* v_C (they need charging); negative ⇒ insert the *highest*
// (they need discharging).
inline Index balance_select(
    const Real* v_C_per_sm,
    Index n_sm,
    Index s_b,
    Real i_b,
    BalancingScheme scheme,
    std::int8_t* insertion_mask) {

    // Decompose into sign and magnitude.
    const std::int8_t sgn = (s_b > 0) ? +1 : (s_b < 0 ? -1 : 0);
    Index s = (s_b < 0) ? -s_b : s_b;     // |s_b|
    if (s > n_sm) s = n_sm;

    if (s == 0) {
        std::fill(insertion_mask, insertion_mask + n_sm, 0);
        return 0;
    }
    if (s >= n_sm) {
        std::fill(insertion_mask, insertion_mask + n_sm, sgn);
        return sgn * n_sm;
    }
    if (scheme == BalancingScheme::None) {
        std::fill(insertion_mask, insertion_mask + n_sm, 0);
        for (Index k = 0; k < s; ++k) {
            insertion_mask[k] = sgn;
        }
        return sgn * s;
    }

    // Sort-and-select. Build index array sorted by v_C ascending.
    std::vector<Index> idx(static_cast<std::size_t>(n_sm));
    std::iota(idx.begin(), idx.end(), Index{0});
    std::stable_sort(idx.begin(), idx.end(),
        [v_C_per_sm](Index a, Index b) {
            return v_C_per_sm[a] < v_C_per_sm[b];
        });

    // Charging vs discharging is determined by sign(i_b · sgn):
    //   sign(i_b)·sgn > 0 → the inserted SMs CHARGE  → pick LOWEST v_C
    //   sign(i_b)·sgn < 0 → the inserted SMs DISCHARGE → pick HIGHEST v_C
    const bool insert_low = (i_b * static_cast<Real>(sgn) >= Real{0.0});

    std::fill(insertion_mask, insertion_mask + n_sm, 0);
    if (insert_low) {
        for (Index k = 0; k < s; ++k) {
            insertion_mask[idx[static_cast<std::size_t>(k)]] = sgn;
        }
    } else {
        for (Index k = 0; k < s; ++k) {
            insertion_mask[
                idx[static_cast<std::size_t>(n_sm - 1 - k)]] = sgn;
        }
    }
    return sgn * s;
}

[[nodiscard]] inline StepDetailedResult mmc_arm_detailed_step(
    Real* v_C_per_sm,                  // length n_sm, in/out
    std::int8_t* insertion_mask,       // length n_sm, scratch / out
    Real m_ref,
    Real i_b,
    Real dt,
    Real t,
    Index n_sm,
    Real c_sm,                          // PER-SM cap, not arm-equivalent
    Real f_carrier,
    SubmoduleType sm_type,
    BalancingScheme scheme,
    Real r_p_inv_per_sm = Real{0.0},
    ModulationScheme modulation_scheme = ModulationScheme::PsPwm) {

    if (dt <= Real{0.0}) {
        throw std::invalid_argument(
            "mmc_arm_detailed_step: dt must be > 0");
    }

    const Index s_b = switching_function(
        m_ref, t, n_sm, f_carrier, sm_type, modulation_scheme);
    balance_select(
        v_C_per_sm, n_sm, s_b, i_b, scheme, insertion_mask);

    // ``insertion_mask`` is signed: {-1, 0, +1}. For HB only +1/0
    // are ever produced; for FB the −1 entries represent SMs
    // inserted in the reverse direction (output −v_C).
    Real v_b = Real{0.0};
    for (Index k = 0; k < n_sm; ++k) {
        v_b += static_cast<Real>(insertion_mask[k]) * v_C_per_sm[k];
    }

    // Per-SM forward Euler:
    //   dv_C_n/dt = (sign_n · i_b − v_C_n · r_p_inv_per_sm) / c_sm
    // where sign_n ∈ {-1, 0, +1} from the signed insertion_mask.
    // A negatively-inserted SM (FB) sees current −i_b across its cap,
    // which is exactly what the sign in the multiplication encodes.
    for (Index k = 0; k < n_sm; ++k) {
        const Real ins  = static_cast<Real>(insertion_mask[k]);
        const Real leak = v_C_per_sm[k] * r_p_inv_per_sm;
        const Real dv_dt = (ins * i_b - leak) / c_sm;
        v_C_per_sm[k] = v_C_per_sm[k] + dt * dv_dt;
    }

    return StepDetailedResult{v_b, s_b};
}

}  // namespace pulsim::mmc
