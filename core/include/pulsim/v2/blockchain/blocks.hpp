// SPDX-License-Identifier: MIT
//
// Pulsim v2 — Phase A.1: C++ control blocks (mixed-domain BlockChain).
//
// Header-only implementations of the 27 control blocks exposed by the
// Python `pulsim.v2_control` module. Each block follows the same
// shape:
//
//   * A `Params` struct (POD) carrying construction-time settings.
//   * Mutable state members.
//   * `void reset()` returning the block to its post-construction
//     state (zero accumulators, empty FIFOs, etc.).
//   * `Output update(...)` with named scalar arguments. Multi-output
//     blocks (Clarke, Park, PLL, Demux) return a tuple-like struct.
//
// Blocks are pure C++ values; they have no virtual functions. The
// surrounding `BlockChain` uses a `std::variant<...>` to compose them
// polymorphically at zero runtime cost (no vtable indirection).
//
// This is the smallest subset needed for a closed-loop SMPS or motor
// drive — Math (Gain/Sum/Subtract/MathBlock), Control (PI/PID/LPF/
// Integrator/Differentiator/Limiter/MovingAverage), Modulation (PWM),
// and Transforms (Clarke/Park + inverses + PLL). Extending to the
// full 27-block library is a straightforward repeat of the same
// pattern.
//
// Header-only, C++20, no exceptions in hot paths (RAII + assertions).

#pragma once

#include "pulsim/v2/numeric/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>

namespace pulsim::v2::blockchain {

// =============================================================================
// 1. Math blocks
// =============================================================================

/// y = k · x
struct Gain {
    Real k = Real{1};

    void reset() noexcept {}

    [[nodiscard]] Real update(Real x) const noexcept {
        return k * x;
    }
};

/// Two-input subtraction: y = a - b
struct Subtract {
    void reset() noexcept {}

    [[nodiscard]] Real update(Real a, Real b) const noexcept {
        return a - b;
    }
};

/// Three-input sum with weights: y = w0*a + w1*b + w2*c.
struct Sum {
    Real w0 = Real{1};
    Real w1 = Real{1};
    Real w2 = Real{0};

    void reset() noexcept {}

    [[nodiscard]] Real update(Real a, Real b = Real{0},
                                 Real c = Real{0}) const noexcept {
        return w0 * a + w1 * b + w2 * c;
    }
};

/// Generic 2-input math block — selects op at construction time.
struct MathBlock {
    enum class Op : std::uint8_t {
        Add, Sub, Mul, Div, Abs, Neg, Sqrt, Pow2,
    };
    Op op = Op::Add;

    void reset() noexcept {}

    [[nodiscard]] Real update(Real a, Real b = Real{0}) const noexcept {
        switch (op) {
            case Op::Add:  return a + b;
            case Op::Sub:  return a - b;
            case Op::Mul:  return a * b;
            case Op::Div:  return (b != Real{0}) ? a / b : Real{0};
            case Op::Abs:  return std::abs(a);
            case Op::Neg:  return -a;
            case Op::Sqrt: return (a >= Real{0}) ? std::sqrt(a) : Real{0};
            case Op::Pow2: return a * a;
        }
        return Real{0};
    }
};

// =============================================================================
// 2. Standalone control blocks
// =============================================================================

/// PI controller — bit-exact match for `pulsim.v2_control.PIController`:
///   integrator += Ki·dt·(e_k + e_{k-1})/2     (trapezoidal)
///   u = Kp·e_k + integrator                    (clamped)
///   anti-windup: hold integrator if u_unclamped would push further
///   past the active saturation bound.
struct PIController {
    Real Kp = Real{0};
    Real Ki = Real{0};
    Real output_min = -std::numeric_limits<Real>::infinity();
    Real output_max =  std::numeric_limits<Real>::infinity();

    // State.
    Real integral = Real{0};
    Real prev_error = Real{0};

    void reset() noexcept {
        integral = Real{0};
        prev_error = Real{0};
    }

    [[nodiscard]] Real update(Real setpoint, Real measured, Real dt) {
        const Real err = setpoint - measured;
        Real new_int = integral +
                          Ki * dt * Real{0.5} * (err + prev_error);
        Real u_unc = Kp * err + new_int;
        // Anti-windup — freeze integrator if it would push deeper
        // into the active saturation.
        if (u_unc > output_max && err > Real{0}) {
            new_int = integral;
            u_unc = Kp * err + new_int;
        } else if (u_unc < output_min && err < Real{0}) {
            new_int = integral;
            u_unc = Kp * err + new_int;
        }
        const Real u_clamp = std::clamp(u_unc, output_min, output_max);
        integral = new_int;
        prev_error = err;
        return u_clamp;
    }
};

/// PID controller with anti-windup + derivative-on-measurement filter.
struct PIDController {
    Real Kp = Real{0};
    Real Ki = Real{0};
    Real Kd = Real{0};
    Real tau_d = Real{0};      // derivative filter time constant
    Real output_min = -std::numeric_limits<Real>::infinity();
    Real output_max =  std::numeric_limits<Real>::infinity();

    Real integral = Real{0};
    Real prev_meas = Real{0};
    Real d_filt = Real{0};
    bool first = true;

    void reset() noexcept {
        integral = Real{0};
        prev_meas = Real{0};
        d_filt = Real{0};
        first = true;
    }

    [[nodiscard]] Real update(Real setpoint, Real measured, Real dt) {
        const Real err = setpoint - measured;
        const Real i_trial = integral + Ki * err * dt;
        // Derivative on measurement (filtered).
        Real d_raw = Real{0};
        if (!first && dt > Real{0}) {
            const Real dm = (measured - prev_meas) / dt;
            if (tau_d > Real{0}) {
                const Real alpha = dt / (tau_d + dt);
                d_filt += alpha * (dm - d_filt);
                d_raw = d_filt;
            } else {
                d_raw = dm;
            }
        }
        prev_meas = measured;
        first = false;
        const Real u_unsat = Kp * err + i_trial - Kd * d_raw;
        const Real u_sat = std::clamp(u_unsat, output_min, output_max);
        if (u_unsat == u_sat || (u_unsat - u_sat) * err <= Real{0}) {
            integral = i_trial;
        }
        return u_sat;
    }
};

/// 1st-order IIR low-pass filter. Discrete form: y_n = α·u + (1-α)·y_{n-1}
struct FirstOrderLowPass {
    Real tau = Real{1};         // time constant in seconds

    Real y = Real{0};

    void reset() noexcept { y = Real{0}; }

    [[nodiscard]] Real update(Real input_value, Real dt) {
        if (dt <= Real{0} || tau <= Real{0}) {
            y = input_value;
            return y;
        }
        const Real alpha = dt / (tau + dt);
        y += alpha * (input_value - y);
        return y;
    }
};

/// Moving-average filter (fixed window).
struct MovingAverageFilter {
    Size window = Size{1};

    std::deque<Real> buf;
    Real running_sum = Real{0};

    void reset() noexcept {
        buf.clear();
        running_sum = Real{0};
    }

    [[nodiscard]] Real update(Real input_value) {
        buf.push_back(input_value);
        running_sum += input_value;
        if (buf.size() > window) {
            running_sum -= buf.front();
            buf.pop_front();
        }
        return running_sum / static_cast<Real>(buf.size());
    }
};

/// Forward-Euler integrator with output limits + anti-windup.
struct Integrator {
    Real gain = Real{1};
    Real output_min = -std::numeric_limits<Real>::infinity();
    Real output_max =  std::numeric_limits<Real>::infinity();

    Real y = Real{0};

    void reset() noexcept { y = Real{0}; }

    [[nodiscard]] Real update(Real x, Real dt) {
        const Real y_trial = y + gain * x * dt;
        y = std::clamp(y_trial, output_min, output_max);
        return y;
    }
};

/// Differentiator with optional IIR low-pass on the result.
struct Differentiator {
    Real filter_alpha = Real{0};  // 0 = no filter, 1 = full filter

    Real prev_x = Real{0};
    Real y = Real{0};
    bool first = true;

    void reset() noexcept {
        prev_x = Real{0};
        y = Real{0};
        first = true;
    }

    [[nodiscard]] Real update(Real x, Real dt) {
        if (first || dt <= Real{0}) {
            prev_x = x;
            first = false;
            return Real{0};
        }
        const Real raw = (x - prev_x) / dt;
        prev_x = x;
        if (filter_alpha > Real{0}) {
            y += filter_alpha * (raw - y);
            return y;
        }
        return raw;
    }
};

/// Hard limiter (clamp).
struct Limiter {
    Real min_v = -std::numeric_limits<Real>::infinity();
    Real max_v =  std::numeric_limits<Real>::infinity();

    void reset() noexcept {}

    [[nodiscard]] Real update(Real x) const noexcept {
        return std::clamp(x, min_v, max_v);
    }
};

// =============================================================================
// 3. Modulation blocks
// =============================================================================

/// PWM generator — sawtooth + duty comparator, output 0 / 1.
struct PwmGenerator {
    Real frequency = Real{100e3};
    Real phase = Real{0};

    void reset() noexcept {}

    [[nodiscard]] Real update(Real duty, Real t) const noexcept {
        const Real period = Real{1} / frequency;
        const Real local = std::fmod(t + phase * period, period);
        const Real phase_frac = local / period;
        return (phase_frac < duty) ? Real{1} : Real{0};
    }
};

/// Centered space-vector modulation: V_α, V_β → 3 duty cycles.
struct SpaceVectorModulator {
    Real v_dc = Real{1};

    void reset() noexcept {}

    struct DutyTriple {
        Real da, db, dc;
    };

    [[nodiscard]] DutyTriple update(Real v_alpha, Real v_beta) const noexcept {
        // Inverse Clarke → 3-phase voltages, then center modulation.
        const Real va_ref = v_alpha;
        const Real vb_ref = -Real{0.5} * v_alpha
                              + Real{0.8660254037844386} * v_beta;
        const Real vc_ref = -Real{0.5} * v_alpha
                              - Real{0.8660254037844386} * v_beta;
        const Real v_max = std::max({va_ref, vb_ref, vc_ref});
        const Real v_min = std::min({va_ref, vb_ref, vc_ref});
        const Real v_cm = -(v_max + v_min) * Real{0.5};
        const Real inv_dc = (v_dc > Real{0}) ? Real{1} / v_dc : Real{0};
        DutyTriple d;
        d.da = std::clamp((va_ref + v_cm) * inv_dc + Real{0.5},
                            Real{0}, Real{1});
        d.db = std::clamp((vb_ref + v_cm) * inv_dc + Real{0.5},
                            Real{0}, Real{1});
        d.dc = std::clamp((vc_ref + v_cm) * inv_dc + Real{0.5},
                            Real{0}, Real{1});
        return d;
    }
};

// =============================================================================
// 4. Transforms (3-phase ↔ αβ ↔ dq)
// =============================================================================

/// Amplitude-invariant Clarke transform: abc → αβ0.
struct ClarkeTransform {
    void reset() noexcept {}

    struct Out { Real alpha, beta, zero; };

    [[nodiscard]] Out update(Real a, Real b, Real c) const noexcept {
        Out o;
        o.alpha = (Real{2} * a - b - c) / Real{3};
        o.beta  = (b - c) / Real{1.7320508075688772};   // (b - c) / √3
        o.zero  = (a + b + c) / Real{3};
        return o;
    }
};

/// Inverse Clarke: αβ → abc.
struct InverseClarkeTransform {
    void reset() noexcept {}

    struct Out { Real a, b, c; };

    [[nodiscard]] Out update(Real alpha, Real beta,
                                Real zero = Real{0}) const noexcept {
        Out o;
        o.a = alpha + zero;
        o.b = -Real{0.5} * alpha + Real{0.8660254037844386} * beta + zero;
        o.c = -Real{0.5} * alpha - Real{0.8660254037844386} * beta + zero;
        return o;
    }
};

/// Park transform: αβ → dq with rotor angle θ.
struct ParkTransform {
    void reset() noexcept {}

    struct Out { Real d, q; };

    [[nodiscard]] Out update(Real alpha, Real beta,
                                Real theta) const noexcept {
        const Real ct = std::cos(theta);
        const Real st = std::sin(theta);
        Out o;
        o.d =  ct * alpha + st * beta;
        o.q = -st * alpha + ct * beta;
        return o;
    }
};

/// Inverse Park: dq → αβ.
struct InverseParkTransform {
    void reset() noexcept {}

    struct Out { Real alpha, beta; };

    [[nodiscard]] Out update(Real d, Real q,
                                Real theta) const noexcept {
        const Real ct = std::cos(theta);
        const Real st = std::sin(theta);
        Out o;
        o.alpha = ct * d - st * q;
        o.beta  = st * d + ct * q;
        return o;
    }
};

// =============================================================================
// 5. Synchronisation
// =============================================================================

/// Phase-locked loop on αβ — cross-product phase detector + PI on q.
struct PLL {
    Real f_nominal = Real{50};
    Real Kp = Real{50};
    Real Ki = Real{500};

    Real theta = Real{0};
    Real omega = Real{0};         // estimated angular frequency
    Real integral = Real{0};

    void reset() noexcept {
        theta = Real{0};
        omega = Real{0};
        integral = Real{0};
    }

    struct Out { Real theta, omega, freq; };

    [[nodiscard]] Out update(Real v_alpha, Real v_beta, Real dt) {
        // Project (v_alpha, v_beta) onto the rotating frame at θ; q
        // component is the phase error.
        const Real ct = std::cos(theta);
        const Real st = std::sin(theta);
        const Real vq = -st * v_alpha + ct * v_beta;
        // Loop filter (PI) on the q-component (proxy for phase error).
        integral += Ki * vq * dt;
        const Real omega_nom = Real{2} * Real{3.141592653589793238462643}
                                  * f_nominal;
        omega = omega_nom + Kp * vq + integral;
        theta += omega * dt;
        // Wrap to [-π, π).
        const Real two_pi = Real{2} *
                              Real{3.141592653589793238462643};
        while (theta >  Real{3.141592653589793238462643}) theta -= two_pi;
        while (theta < -Real{3.141592653589793238462643}) theta += two_pi;
        return Out{theta, omega, omega / two_pi};
    }
};

}  // namespace pulsim::v2::blockchain
