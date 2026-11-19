#pragma once

// =============================================================================
// Pulsim — the opaque-callable gate adapter (add-unified-event-queue 3.1)
// =============================================================================
//
// A gate given as an opaque function of time is seen only where the
// engine calls it. `SampledGate` holds what the variable-step engine
// learns from those calls and decides with it, moved unchanged out of
// `run_transient_trbdf2`:
//
//   * the per-bit toggle history (pulse widths, the last three toggle
//     times per direction, the rise times at controller ticks);
//   * the step cap: kCapPulses x the narrowest RECENT pulse, lifted until
//     just before a regularly predicted toggle (`pulse_cap`), and the
//     periodic-gate cap of p/20 (`period_cap`);
//   * the sampling of one step (kGateSamples points, or w/2 apart once a
//     recent pulse w is known, plus a golden-ratio walking point) and the
//     bisection of the first sub-interval whose end differs (`locate`);
//   * the confession counters: `n_gate_undersampled` (a pulse closed
//     narrower than the spacing its bit was sampled at while it was open)
//     and `gate_pulse_min`.
//
// `resolution()` is the declared one: the spacing of kGateSamples samples
// over the longest step, h_max / kGateSamples (design.md: "h_max/4 on the
// variable-step engine"). A pulse is undersampled against the spacing its
// bit actually had, which is never coarser.
//
// A one-shot pulse no sample hits cannot be seen by construction.
// Header-only; every member the step path calls is inlined.

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/switch_state.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define PULSIM_SAMPLED_GATE_INLINE [[gnu::always_inline]] inline
#else
#define PULSIM_SAMPLED_GATE_INLINE inline
#endif

namespace pulsim::events {

class SampledGate {
public:
    using Mask = topology::SwitchStateMask;
    static constexpr int kGateSamples = 4;
    static constexpr Real kCapPulses = 64;   // step cap in recent narrowest pulses

    /// `first`: the gate at the run's start (t_start + h_min/2);
    /// `n_undersampled`: the engine's counter this adapter increments;
    /// `learn`: false when the gate declares its edges (an oracle): then
    /// only the plain kGateSamples sampling runs.
    SampledGate(Size n_sw, const Mask& diode_owned, Mask first, Real h_min,
                Real h_max, Real span_ref, bool learn, Size& n_undersampled)
        : learn_(learn),
          h_min_(h_min),
          h_max_(h_max),
          n_sw_(n_sw),
          diode_owned_(diode_owned),
          gate_last_toggle_(n_sw, Real{0}),
          gate_w_last_(2 * n_sw, Real{0}),
          gate_dir_t_(6 * n_sw, -std::numeric_limits<Real>::infinity()),
          gate_toggled_(n_sw, false),
          gate_prev_(std::move(first)),
          // A "pulse" shorter than this is a floating-point glitch of the
          // user's phase comparison, not a pulse a step could resolve.
          gate_pulse_floor_(std::max(Real{16} * h_min, span_ref * Real{1e-9})),
          gate_spacing_open_(n_sw, Real{0}),
          gate_rise_t_(3 * n_sw, -std::numeric_limits<Real>::infinity()),
          gate_rise_prev_(gate_prev_),
          n_undersampled_(n_undersampled) {
        for (Size i = 0; i < n_sw; ++i) {
            if (!diode_owned.get(i)) {
                gate_bits_.push_back(i);
            }
        }
    }

    [[nodiscard]] bool learns() const noexcept { return learn_; }
    /// The declared sampling resolution (see the file comment).
    [[nodiscard]] Real resolution() const noexcept {
        return h_max_ / static_cast<Real>(kGateSamples);
    }
    [[nodiscard]] Size n_gate_undersampled() const noexcept { return n_undersampled_; }
    /// The narrowest closed pulse seen (+inf: none).
    [[nodiscard]] Real gate_pulse_min() const noexcept { return gate_pulse_min_; }
    [[nodiscard]] Real gate_spacing_max() const noexcept { return gate_spacing_max_; }
    /// The gate bits that are not diode-owned.
    [[nodiscard]] const std::vector<Size>& gate_bits() const noexcept { return gate_bits_; }
    /// The gate as last learned (the run's start sample before any toggle).
    [[nodiscard]] const Mask& last_gate() const noexcept { return gate_prev_; }
    /// The periodic-gate step cap, p/20 (+inf: none learned).
    [[nodiscard]] Real period_cap() const noexcept { return gate_period_cap_; }
    /// A controller tick fired at `t` (a trailing-edge PWM rises on it).
    PULSIM_SAMPLED_GATE_INLINE void note_tick(Real t) noexcept { last_tick_t_ = t; }

    /// The gate `sw` seen at `at_t`: records each bit's toggle.
    PULSIM_SAMPLED_GATE_INLINE void learn_toggles(const Mask& sw, Real at_t) {
        if (sw == gate_prev_) {
            return;
        }
        for (Size i = 0; i < n_sw_; ++i) {
            if (diode_owned_.get(i) || sw.get(i) == gate_prev_.get(i)) {
                continue;
            }
            if (gate_toggled_[i]) {
                const Real w = at_t - gate_last_toggle_[i];
                if (w > gate_pulse_floor_) {
                    gate_w_last_[2 * i + (gate_prev_.get(i) ? 1 : 0)] = w;
                }
                if (w > gate_pulse_floor_ && w < gate_pulse_min_) {
                    gate_pulse_min_ = w;
                }
                // 1 % margin: a pulse as wide as the spacing cannot fit
                // between two samples, and the widths carry the
                // bisection's h_min of noise. Every pulse, not only a new
                // minimum: the cap lapses on a quiet bit (burst mode), so
                // a pulse after the pause can be undersampled again.
                if (w > gate_pulse_floor_
                    && gate_spacing_open_[i] > Real{1.01} * w) {
                    ++n_undersampled_;
                }
                gate_spacing_open_[i] = Real{0};
            }
            gate_toggled_[i] = true;
            gate_last_toggle_[i] = at_t;
            Real* dt3 = &gate_dir_t_[6 * i + (sw.get(i) ? 3 : 0)];
            dt3[0] = dt3[1];
            dt3[1] = dt3[2];
            dt3[2] = at_t;
        }
        gate_prev_ = sw;
    }

    // A periodic gate is a periodic source too: the default ceiling gives
    // every PWM / pulse / sine source 20 steps per period, and a gate from
    // switch_fn (a closed loop's, declared edges or not) got span/1000. On
    // ll15 (closed-loop boost, 5 ms) that was one 4 us step per 5 us
    // period, each at its local tolerance and all errors of one sign:
    // I(L) 0.066 A off after 23 periods. A rise counts only at a controller
    // tick (a trailing-edge closed-loop PWM rises on its tick); once a gate
    // bit has so risen three times a constant period p apart, a step is
    // capped at p/20 (default ceiling only; the floors stay those of
    // h_max). An open-loop gate keeps its fewer steps per period (the CCM
    // buck's <= 8 per period is a pinned decision) unless its rises happen
    // to fall on ticks (then it is capped at most at T_tick/20). The cap is
    // a run-wide minimum (a short periodic episode keeps it) and is
    // relearned after a resume (the rise times are not in the snapshot).
    PULSIM_SAMPLED_GATE_INLINE void learn_period(const Mask& sw, Real at_t) {
        if (sw == gate_rise_prev_) return;
        for (const Size i : gate_bits_) {
            if (!sw.get(i) || gate_rise_prev_.get(i)
                || !(std::abs(at_t - last_tick_t_) <= Real{4} * h_min_)) {
                continue;
            }
            Real* r3 = &gate_rise_t_[3 * i];
            r3[0] = r3[1];
            r3[1] = r3[2];
            r3[2] = at_t;
            const Real p1 = r3[2] - r3[1], p0 = r3[1] - r3[0];
            // p1 >= 20 pulse floors: rises a few h_min apart (glitches of
            // a phase comparison) are no period to cap the run by.
            if (std::isfinite(p0) && p1 >= Real{20} * gate_pulse_floor_
                && std::abs(p1 - p0) <= Real{1e-6} * p1 + Real{4} * h_min_) {
                gate_period_cap_ = std::min(gate_period_cap_, p1 / Real{20});
            }
        }
        gate_rise_prev_ = sw;
    }

    // The step cap: kCapPulses x the narrowest RECENT pulse. Per bit, the last
    // high and low widths count while the bit keeps toggling (quiet for at
    // most twice its last period; a quiet bit keeps the run's narrowest
    // pulse as its cap); a running minimum over the run held a
    // closed loop's 0.28 ns start-up pulse as the cap for ever (519,783
    // steps in 0.3 ms of cl_boost_pi with an opaque gate). A phase whose
    // ending toggle comes at a regular period (a trailing-edge PWM's rise
    // at kT) is not capped until just before the predicted toggle: the
    // same run took 58,514 steps with the cap over the whole long phase.
    // Returns the step limit (infinite: none). Also sets the narrowest
    // recent pulse of a bit whose cap was lifted (the samples inside a
    // lifted step stay half of it apart: a gate that gets faster, a
    // skipped pulse resuming, puts pulses where none is predicted).
    PULSIM_SAMPLED_GATE_INLINE Real pulse_cap(Real at_t) {
        Real lim = std::numeric_limits<Real>::infinity();
        gate_sample_w_ = std::numeric_limits<Real>::infinity();
        for (const Size i : gate_bits_) {
            const Real wh = gate_w_last_[2 * i + 1];
            const Real wl = gate_w_last_[2 * i];
            if (!gate_toggled_[i] || (wh <= Real{0} && wl <= Real{0})) {
                continue;
            }
            if (at_t - gate_last_toggle_[i] > Real{2} * (wh + wl)) {
                // Quiet: the narrowest pulse of the run, as before. Pulses
                // resuming after a pause (burst mode, a soft-start pulse
                // then PWM) were stepped over with the cap lapsed: 4 of 99
                // 10 ns pulses, every 1 ns one.
                if (std::isfinite(gate_pulse_min_)) {
                    lim = std::min(lim, Real{2} * gate_pulse_min_);
                }
                continue;
            }
            Real w = std::numeric_limits<Real>::infinity();
            if (wh > Real{0}) w = std::min(w, wh);
            if (wl > Real{0}) w = std::min(w, wl);
            // A pulse a step holds must show at a sample. A pulse ends
            // with a toggle in each direction once per period; when one
            // direction is
            // regular, a step that ends before its next predicted toggle
            // cannot hold a pulse that ends there, and any other toggle it
            // holds is seen at its end and located.
            Real reg = std::numeric_limits<Real>::infinity();
            bool any_reg = false;
            for (const Size dir : {Size{0}, Size{3}}) {
                const Real* d3 = &gate_dir_t_[6 * i + dir];
                const Real p1 = d3[2] - d3[1], p0 = d3[1] - d3[0];
                if (!std::isfinite(p0) || p1 <= Real{0}
                    || std::abs(p1 - p0) > Real{1e-6} * p1 + Real{4} * h_min_) {
                    continue;
                }
                any_reg = true;
                const Real margin = Real{2} * w + Real{1e-3} * p1;
                reg = std::min(reg, d3[2] + p1 - margin - at_t);
            }
            // kCapPulses * w with the gate sampled every w/2 inside the
            // step (gate_sample_w): the same sampling as 2 w with 4 samples,
            // in 32x fewer solves. cl_boost_pi's start-up pulses (0.28 ns,
            // then 0.585 ns, before the rises prove periodic) cost 26,490
            // steps in two periods at 2 w.
            lim = std::min(lim, any_reg ? std::max(kCapPulses * w, reg)
                                        : kCapPulses * w);
            gate_sample_w_ = std::min(gate_sample_w_, w);
        }
        return lim;
    }

    /// Samples the gate over (t, t + h] and, when a sample differs from
    /// `sw_now`, bisects the first such sub-interval to h_min. Returns
    /// true with `hi` the located edge's post-edge end; false when every
    /// sample equals `sw_now` (the step's mask IS `sw_now`: the midpoint
    /// was one of the samples).
    template <class SwitchFn>
    PULSIM_SAMPLED_GATE_INLINE bool locate(SwitchFn&& switch_fn, const Mask& sw_now,
                                           Real t, Real h, Real& hi_out) {
        const Real h_min = h_min_;
        // Samples per step: 4, or half the recent narrowest pulse
        // apart in a lifted step (switch_fn calls only, no solves).
        // Once a recent pulse width w is known the samples are w/2
        // apart (DECISIONS #30), and a step shorter than w/2 needs
        // only its end: the floor of 4 is for gates not yet seen.
        // It cost the CCM buck 3 Python switch_fn calls per step.
        int n_gs = kGateSamples;
        if (learn_ && std::isfinite(gate_sample_w_)
            && gate_sample_w_ > Real{0}) {
            n_gs = static_cast<int>(std::clamp(
                std::ceil(Real{2} * h / gate_sample_w_),
                Real{1}, Real{4096}));
        }
        if (learn_) {
            const Real spacing =
                h / static_cast<Real>(n_gs);
            gate_spacing_max_ =
                std::max(gate_spacing_max_, spacing);
            for (const Size i : gate_bits_) {
                gate_spacing_open_[i] =
                    std::max(gate_spacing_open_[i], spacing);
            }
        }
        // h/4, h/2, 3h/4, h, plus one sample at a fraction
        // that walks the golden-ratio sequence from step to
        // step: fixed fractions alone PHASE-LOCK to a PWM when
        // h is a multiple of its period (h_max = span/1000 =
        // 20 µs against 10 µs), and then never see a pulse
        // that starts after the run has reached h_max.
        gate_jitter_ += Real{0.6180339887498949};
        gate_jitter_ -= std::floor(gate_jitter_);
        std::vector<Real>& frac = gate_frac_;
        frac.resize(static_cast<std::size_t>(n_gs) + 1);
        for (int k = 1; k <= n_gs; ++k) {
            frac[static_cast<std::size_t>(k - 1)] =
                static_cast<Real>(k) / static_cast<Real>(n_gs);
        }
        frac[static_cast<std::size_t>(n_gs)] =
            std::clamp(gate_jitter_, Real{0.01}, Real{0.99});
        std::sort(frac.begin(), frac.end());
        Real lo = t, hi = t + h;
        bool edge_found = false;
        for (const Real f : frac) {
            const Real tk = f >= Real{1} ? t + h : t + h * f;
            if (!(switch_fn(tk) == sw_now)) {
                hi = tk;
                edge_found = true;
                break;
            }
            lo = tk;
        }
        if (!edge_found) {
            return false;
        }
        // A periodic bit predicts its next toggle: two calls
        // 4 h_min either side of it bound the edge, and the
        // bisection reads every midpoint outside the bounds
        // from them instead of calling switch_fn (~25 calls ->
        // ~5 per edge). Same result: the bisection already
        // assumes one change of the mask in (lo, hi], under
        // which a point the gate was verified at answers for
        // every midpoint on its side.
        Real known_lo = lo, known_hi = hi;
        if (learn_) {
            Real te = std::numeric_limits<Real>::infinity();
            for (const Size i : gate_bits_) {
                const Real* d3 =
                    &gate_dir_t_[6 * i + (sw_now.get(i) ? 0 : 3)];
                const Real p1 = d3[2] - d3[1], p0 = d3[1] - d3[0];
                if (std::isfinite(p0) && p1 > Real{0}
                    && std::abs(p1 - p0)
                           <= Real{1e-6} * p1 + Real{4} * h_min
                    && d3[2] + p1 > lo) {
                    te = std::min(te, d3[2] + p1);
                }
            }
            if (te > lo && te < hi) {
                for (const Real tv : {te - Real{4} * h_min,
                                      te + Real{4} * h_min}) {
                    if (!(tv > known_lo && tv < known_hi)) continue;
                    if (switch_fn(tv) == sw_now) {
                        known_lo = tv;
                    } else {
                        known_hi = tv;
                    }
                }
            }
        }
        for (int it = 0; it < 60 && (hi - lo) > h_min;
             ++it) {
            const Real mid = Real{0.5} * (lo + hi);
            const bool same = mid <= known_lo ? true
                : mid >= known_hi ? false
                : (switch_fn(mid) == sw_now);
            if (same) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hi_out = hi;
        return true;
    }

private:
    bool learn_;
    Real h_min_;
    Real h_max_;
    Size n_sw_;
    Mask diode_owned_;
    std::vector<Real> gate_last_toggle_;
    // The last closed high / low pulse width of each gate bit (0 = none).
    std::vector<Real> gate_w_last_;
    // The last three toggle times of each bit per direction (fall, rise).
    std::vector<Real> gate_dir_t_;
    std::vector<bool> gate_toggled_;
    Mask gate_prev_;
    Real gate_pulse_min_ = std::numeric_limits<Real>::infinity();
    Real gate_pulse_floor_;
    Real gate_spacing_max_ = Real{0};
    // Per gate bit: the largest sampling spacing used since that bit's
    // PREVIOUS seen toggle (or the run start, before its first one),
    // i.e. while the pulse now closing was open or while pulses could
    // have passed unseen. A pulse is "undersampled" only against THAT
    // spacing: compared with the run-wide maximum, every smooth SPWM
    // whose pulses narrow gradually (each one landed, at a spacing
    // already capped by the previous pulse) was flagged — 22 times
    // over 0.2 s of a 10 kHz three-phase VSI whose 16000 edges were
    // all landed. Not reset at a bit's FIRST toggle: a PWM that starts
    // late is found by the walking sample only after some pulses went
    // by at the idle spacing, and those are what the warning confesses.
    std::vector<Real> gate_spacing_open_;
    std::vector<Size> gate_bits_;
    Real gate_jitter_ = Real{0};
    Real last_tick_t_ = -std::numeric_limits<Real>::infinity();
    std::vector<Real> gate_rise_t_;
    Mask gate_rise_prev_;
    Real gate_period_cap_ = std::numeric_limits<Real>::infinity();
    Real gate_sample_w_ = std::numeric_limits<Real>::infinity();
    std::vector<Real> gate_frac_;
    Size& n_undersampled_;
};

}  // namespace pulsim::events
