/* pulsim block laws: ONE expression per control-graph block kind.
 *
 * C99 and C++: the kernel's graph blocks (blocks.hpp, ported_blocks.hpp)
 * call these functions, and pulsim.export_c copies this file verbatim
 * into the C it generates, so an exported graph runs the same operations
 * as the kernel (bit for bit when both are compiled without FMA
 * contraction: -ffp-contract=off). Each law keeps the operand order and
 * the min/max argument order of its pulsim.control Python class.
 * No allocation, no I/O; state lives in the caller's structs. */
#ifndef PULSIM_BLOCK_LAWS_H
#define PULSIM_BLOCK_LAWS_H

#include <math.h>
#include <string.h>

#ifdef __cplusplus
#define PL_INLINE inline
#elif defined(__GNUC__) || defined(__clang__)
#define PL_INLINE static inline __attribute__((unused))
#else
#define PL_INLINE static inline
#endif

/* The real-valued laws are written once on `pl_real`: double in C. In
 * C++ they are templates on the scalar: a double call instantiates the C
 * law exactly, and the control graph's tick Jacobian
 * (blockchain/tick_jacobian.hpp) instantiates them on the AD scalar.
 * Every comparison that picks a branch of a law is written PL_LT / PL_GT /
 * PL_LE / PL_GE / PL_NE, and PL_KINK marks a kink with no branch (|x|):
 * in C and on a double they are the plain comparison (PL_KINK nothing);
 * on another scalar they also report (a, b, the side taken, the law) to
 * the thread's pl_kink_sink: how a derivative says which side of a kink
 * it took, and refuses one taken ON the kink. */
#ifdef __cplusplus
#include <type_traits>
template <class S>
struct pl_kink_sink {
    virtual ~pl_kink_sink() = default;
    virtual void note(const S& a, const S& b, bool side, const char* law) = 0;
    static pl_kink_sink*& current() noexcept {
        static thread_local pl_kink_sink* s = nullptr;
        return s;
    }
};
template <class A, class B>
inline bool pl_cmp_note(const A& a, const B& b, bool side, const char* law) {
    if constexpr (!(std::is_arithmetic_v<A> && std::is_arithmetic_v<B>)) {
        using S = std::conditional_t<std::is_arithmetic_v<A>, B, A>;
        if (pl_kink_sink<S>* k = pl_kink_sink<S>::current()) k->note(S(a), S(b), side, law);
    }
    return side;
}
#define PL_TPL template <class pl_real>
#define PL_LT(a, b) pl_cmp_note((a), (b), (a) < (b), __func__)
#define PL_GT(a, b) pl_cmp_note((a), (b), (a) > (b), __func__)
#define PL_LE(a, b) pl_cmp_note((a), (b), (a) <= (b), __func__)
#define PL_GE(a, b) pl_cmp_note((a), (b), (a) >= (b), __func__)
#define PL_NE(a, b) pl_cmp_note((a), (b), (a) != (b), __func__)
#define PL_KINK(a, b) ((void)pl_cmp_note((a), (b), (a) < (b), __func__))
#else
#define PL_TPL
#define pl_real double
#define PL_LT(a, b) ((a) < (b))
#define PL_GT(a, b) ((a) > (b))
#define PL_LE(a, b) ((a) <= (b))
#define PL_GE(a, b) ((a) >= (b))
#define PL_NE(a, b) ((a) != (b))
#define PL_KINK(a, b) ((void)0)
#endif

/* Python's min(a, b) / max(a, b): the first argument unless the second
 * is strictly smaller / larger. */
#ifdef __cplusplus
template <class A, class B>
inline auto pl_min(const A& a, const B& b) -> decltype(a + b) {
    using R = decltype(a + b);
    return PL_LT(b, a) ? R(b) : R(a);
}
template <class A, class B>
inline auto pl_max(const A& a, const B& b) -> decltype(a + b) {
    using R = decltype(a + b);
    return PL_GT(b, a) ? R(b) : R(a);
}
/* max(lo, min(hi, u)) */
template <class L, class H, class U>
inline auto pl_clamp(const L& lo, const H& hi, const U& u) -> decltype(lo + hi + u) {
    using R = decltype(lo + hi + u);
    return R(pl_max(lo, pl_min(hi, u)));
}
#else
PL_INLINE double pl_min(double a, double b) { return b < a ? b : a; }
PL_INLINE double pl_max(double a, double b) { return b > a ? b : a; }
/* max(lo, min(hi, u)) */
PL_INLINE double pl_clamp(double lo, double hi, double u) {
    return pl_max(lo, pl_min(hi, u));
}
#endif

/* cos and sin through volatile pointers: a compiler fuses cos(x) and
 * sin(x) into one sincos call, which differs from the two by an ulp on
 * some inputs (4 of 10^4 Park inputs on macOS arm64); Python calls
 * both, so the laws do too. The pointers name these wrappers, not the
 * library's cos / sin: taking the address of a standard library
 * function is unspecified in C++ (and cos may be a macro or overloaded). */
PL_INLINE double pl_cos_call(double x) { return cos(x); }
PL_INLINE double pl_sin_call(double x) { return sin(x); }
PL_INLINE double pl_cos(double x) {
    double (*volatile f)(double) = pl_cos_call;
    return f(x);
}
PL_INLINE double pl_sin(double x) {
    double (*volatile f)(double) = pl_sin_call;
    return f(x);
}
#ifdef __cplusplus
/* another scalar (the AD one): its own cos / sin, found by ADL */
template <class T> inline T pl_cos(const T& x) { return cos(x); }
template <class T> inline T pl_sin(const T& x) { return sin(x); }
#endif

PL_TPL PL_INLINE pl_real pl_gain(pl_real k, pl_real x) { return k * x; }
PL_TPL PL_INLINE pl_real pl_subtract(pl_real a, pl_real b) { return a - b; }

/* sum(w_i * x_i), left to right from 0 */
PL_TPL PL_INLINE pl_real pl_sum(const pl_real* w, const pl_real* x, int n) {
    pl_real acc = 0.0;
    int i;
    for (i = 0; i < n; ++i) {
        const pl_real term = w[i] * x[i];
        acc = acc + term;
    }
    return acc;
}

/* PIController: trapezoidal integral, conditional anti-windup. */
PL_TPL PL_INLINE pl_real pl_pi(pl_real Kp, pl_real Ki, pl_real lo, pl_real hi,
                       pl_real* integral, pl_real* prev_error,
                       pl_real setpoint, pl_real measured, pl_real dt) {
    const pl_real err = setpoint - measured;
    pl_real new_int = *integral + Ki * dt * 0.5 * (err + *prev_error);
    pl_real u = Kp * err + new_int;
    if (PL_GT(u, hi) && PL_GT(err, 0.0)) {
        new_int = *integral;
        u = Kp * err + new_int;
    } else if (PL_LT(u, lo) && PL_LT(err, 0.0)) {
        new_int = *integral;
        u = Kp * err + new_int;
    }
    *integral = new_int;
    *prev_error = err;
    return pl_clamp(lo, hi, u);
}

/* PIDController: trapezoidal I, filtered D on the error. */
PL_TPL PL_INLINE pl_real pl_pid(pl_real Kp, pl_real Ki, pl_real Kd, pl_real alpha,
                        pl_real lo, pl_real hi, pl_real* integ, pl_real* deriv,
                        pl_real* prev_error, pl_real setpoint, pl_real measured,
                        pl_real dt) {
    const pl_real error = setpoint - measured;
    pl_real new_int = *integ + Ki * dt * 0.5 * (error + *prev_error);
    const pl_real d_raw = PL_GT(dt, 0.0) ? (error - *prev_error) / dt : (pl_real)0.0;
    const pl_real da = alpha * d_raw;
    const pl_real db = (1.0 - alpha) * *deriv;
    const pl_real new_deriv = da + db;
    pl_real u = Kp * error + new_int + Kd * new_deriv;
    if (PL_GT(u, hi) && PL_GT(error, 0.0)) {
        new_int = *integ;
        u = Kp * error + new_int + Kd * new_deriv;
    } else if (PL_LT(u, lo) && PL_LT(error, 0.0)) {
        new_int = *integ;
        u = Kp * error + new_int + Kd * new_deriv;
    }
    *integ = new_int;
    *deriv = new_deriv;
    *prev_error = error;
    return pl_clamp(lo, hi, u);
}

/* Integrator: forward Euler; the state holds when it would pass a limit. */
PL_TPL PL_INLINE pl_real pl_integrator(pl_real gain, pl_real lo, pl_real hi,
                               pl_real* state, pl_real x, pl_real dt) {
    pl_real nw = *state + gain * dt * x;
    if (PL_GT(nw, hi) && PL_GT(x, 0.0)) nw = *state;
    else if (PL_LT(nw, lo) && PL_LT(x, 0.0)) nw = *state;
    nw = pl_clamp(lo, hi, nw);
    *state = nw;
    return nw;
}

PL_TPL PL_INLINE pl_real pl_differentiator(pl_real alpha, pl_real* state,
                                   pl_real* prev_x, pl_real x, pl_real dt) {
    const pl_real d_raw = PL_GT(dt, 0.0) ? (x - *prev_x) / dt : (pl_real)0.0;
    const pl_real a = alpha * d_raw;
    const pl_real b = (1.0 - alpha) * *state;
    *state = a + b;
    *prev_x = x;
    return *state;
}

/* MovingAverageFilter (an exponential average). */
PL_TPL PL_INLINE pl_real pl_ema(pl_real alpha, pl_real* state, pl_real x) {
    const pl_real a = alpha * *state;
    const pl_real b = (1.0 - alpha) * x;
    *state = a + b;
    return *state;
}

PL_TPL PL_INLINE pl_real pl_low_pass(pl_real tau, pl_real* state, pl_real x, pl_real dt) {
    if (PL_LE(tau, 0.0)) {
        *state = x;
    } else {
        const pl_real alpha = dt / (tau + dt);
        *state += alpha * (x - *state);
    }
    return *state;
}

PL_TPL PL_INLINE pl_real pl_limiter(pl_real lo, pl_real hi, pl_real x) {
    return pl_clamp(lo, hi, x);
}

/* Comparator (Schmitt trigger); *state is 0 or 1. */
PL_TPL PL_INLINE pl_real pl_comparator(pl_real threshold, pl_real hysteresis,
                               pl_real high, pl_real low, pl_real* state,
                               pl_real x) {
    const pl_real half_band = 0.5 * hysteresis;
    if (*state != 0.0) {
        if (PL_LT(x, threshold - half_band)) *state = 0.0;
    } else {
        if (PL_GT(x, threshold + half_band)) *state = 1.0;
    }
    return *state != 0.0 ? high : low;
}

PL_TPL PL_INLINE pl_real pl_rate_limiter(pl_real up, pl_real down, pl_real* state,
                                 pl_real target, pl_real dt) {
    pl_real delta = target - *state;
    const pl_real max_up = up * dt;
    const pl_real max_down = down * dt;
    if (PL_GT(delta, max_up)) delta = max_up;
    else if (PL_LT(delta, -max_down)) delta = -max_down;
    *state += delta;
    return *state;
}

PL_TPL PL_INLINE pl_real pl_sample_hold(pl_real period, pl_real* state, pl_real* last_t,
                                pl_real x, pl_real t) {
    if (PL_GE(t - *last_t, period)) {
        *state = x;
        *last_t = t;
    }
    return *state;
}

/* TransferFunction: xh / yh newest first, *nx <= nn, *ny <= max(1, nd);
 * the buffers hold nn and max(1, nd) doubles. */
PL_TPL PL_INLINE pl_real pl_transfer_function(const pl_real* num, int nn,
                                      const pl_real* den, int nd,
                                      pl_real* xh, int* nx, pl_real* yh,
                                      int* ny, pl_real x) {
    pl_real num_part = 0.0, den_part = 0.0, y;
    const int ycap = nd > 1 ? nd : 1;
    int i;
    if (nn > 0) {
        const int keep = *nx < nn ? *nx : nn - 1;
        memmove(xh + 1, xh, (size_t)keep * sizeof(*xh));
        xh[0] = x;
        *nx = keep + 1;
    }
    for (i = 0; i < nn; ++i) {
        const pl_real term = num[i] * (i < *nx ? xh[i] : 0.0);
        num_part = num_part + term;
    }
    for (i = 0; i < nd; ++i) {
        const pl_real term = den[i] * (i < *ny ? yh[i] : 0.0);
        den_part = den_part + term;
    }
    y = num_part - den_part;
    {
        const int keep = *ny < ycap ? *ny : ycap - 1;
        memmove(yh + 1, yh, (size_t)keep * sizeof(*yh));
        yh[0] = y;
        *ny = keep + 1;
    }
    return y;
}

/* StateMachine: mode 0 toggle, 1 level, 2 SR latch. */
PL_TPL PL_INLINE pl_real pl_state_machine(int mode, pl_real threshold, pl_real high,
                                  pl_real low, pl_real* state,
                                  pl_real* trigger_prev, pl_real trigger,
                                  pl_real set_, pl_real reset_) {
    if (mode == 0) {
        if (PL_LE(*trigger_prev, threshold) && PL_GT(trigger, threshold)) {
            *state = *state == high ? low : high;
        }
        *trigger_prev = trigger;
    } else if (mode == 1) {
        *state = PL_GT(trigger, threshold) ? high : low;
    } else {
        if (PL_GT(set_, threshold)) *state = high;
        else if (PL_GT(reset_, threshold)) *state = low;
    }
    return *state;
}

PL_TPL PL_INLINE pl_real pl_op_amp(pl_real gain, pl_real rail_min, pl_real rail_max,
                           pl_real in_pos, pl_real in_neg) {
    const pl_real u = gain * (in_pos - in_neg);
    return pl_max(rail_min, pl_min(rail_max, u));
}

PL_TPL PL_INLINE pl_real pl_lookup_1d(const pl_real* xs, const pl_real* ys, int n,
                              pl_real x) {
    int i;
    if (n == 0) return 0.0;
    if (PL_LE(x, xs[0])) return ys[0];
    if (PL_GE(x, xs[n - 1])) return ys[n - 1];
    for (i = 0; i + 1 < n; ++i) {
        if (PL_LE(xs[i], x) && PL_LT(x, xs[i + 1])) {
            const pl_real t = (x - xs[i]) / (xs[i + 1] - xs[i]);
            const pl_real dy = t * (ys[i + 1] - ys[i]);
            return ys[i] + dy;
        }
    }
    return ys[n - 1];
}

/* MathBlock: op 0 add 1 sub 2 mul 3 div 4 abs 5 neg 6 sqrt 7 pow2. */
PL_TPL PL_INLINE pl_real pl_math(int op, pl_real a, pl_real b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return PL_NE(b, 0.0) ? a / b : (pl_real)0.0;
        case 4: PL_KINK(a, 0.0); return fabs(a);
        case 5: return -a;
        case 6: return PL_GE(a, 0.0) ? sqrt(a) : (pl_real)0.0;
        default: return a * a;
    }
}

/* SignalMux: max(0, min(n - 1, int(selector))) for a finite selector
 * (the kernel refuses a non-finite one before calling). */
PL_INLINE int pl_mux_index(int n, double selector) {
    const double sel = trunc(selector);
    const double nm1 = (double)(n - 1);
    const double idx = sel >= nm1 ? nm1 : (sel >= 0.0 ? sel : 0.0);
    return (int)idx;
}

/* PwmGenerator / pulsim.control.pwm_on: 1 while the carrier position is
 * below duty (grid-exact snap of 64 ulp). */
PL_INLINE double pl_pwm(double frequency, double phase, double duty,
                        double t) {
    double T, x, tol, u;
    if (!(frequency > 0.0)) return 0.0;
    T = 1.0 / frequency;
    if (!isfinite(T) || T == 0.0) return 0.0;
    x = (t - phase) / T;
    tol = 64.0 * 2.220446049250313e-16 *
          pl_max(1.0, (fabs(t) + fabs(phase)) / T);
    u = x - floor(x);
    if (u >= 1.0 - tol) u = 0.0;
    return u < duty - tol ? 1.0 : 0.0;
}

/* SpaceVectorModulator: min/max injection, duties clamped to [0, 1]. */
PL_TPL PL_INLINE void pl_svm(pl_real v_dc, pl_real v_alpha, pl_real v_beta,
                      pl_real* d_a, pl_real* d_b, pl_real* d_c) {
    const pl_real v3 = sqrt(3.0);
    const pl_real u_a = v_alpha;
    const pl_real u_b = (-v_alpha + v3 * v_beta) / 2.0;
    const pl_real u_c = (-v_alpha - v3 * v_beta) / 2.0;
    const pl_real u_max = pl_max(pl_max(u_a, u_b), u_c);
    const pl_real u_min = pl_min(pl_min(u_a, u_b), u_c);
    const pl_real u_zero = 0.5 * (u_max + u_min);
    const pl_real vdc = PL_GT(v_dc, 0.0) ? v_dc : (pl_real)1.0;
    *d_a = pl_max(0.0, pl_min(1.0, 0.5 + (u_a - u_zero) / vdc));
    *d_b = pl_max(0.0, pl_min(1.0, 0.5 + (u_b - u_zero) / vdc));
    *d_c = pl_max(0.0, pl_min(1.0, 0.5 + (u_c - u_zero) / vdc));
}

PL_TPL PL_INLINE void pl_clarke(pl_real a, pl_real b, pl_real c, pl_real* alpha,
                         pl_real* beta, pl_real* zero) {
    const pl_real v3 = sqrt(3.0);
    *alpha = (2.0 / 3.0) * (a - 0.5 * b - 0.5 * c);
    *beta = (2.0 / 3.0) * (v3 / 2.0) * (b - c);
    *zero = (1.0 / 3.0) * (a + b + c);
}

PL_TPL PL_INLINE void pl_inverse_clarke(pl_real alpha, pl_real beta, pl_real zero,
                                 pl_real* a, pl_real* b, pl_real* c) {
    const pl_real v3 = sqrt(3.0);
    *a = alpha + zero;
    *b = -0.5 * alpha + (v3 / 2.0) * beta + zero;
    *c = -0.5 * alpha - (v3 / 2.0) * beta + zero;
}

PL_TPL PL_INLINE void pl_park(pl_real alpha, pl_real beta, pl_real theta, pl_real* d,
                       pl_real* q) {
    const pl_real c = pl_cos(theta), s = pl_sin(theta);
    *d = alpha * c + beta * s;
    *q = -alpha * s + beta * c;
}

PL_TPL PL_INLINE void pl_inverse_park(pl_real d, pl_real q, pl_real theta,
                               pl_real* alpha, pl_real* beta) {
    const pl_real c = pl_cos(theta), s = pl_sin(theta);
    *alpha = d * c - q * s;
    *beta = d * s + q * c;
}

/* PLL: cross-product detector, PI on q, angle wrapped to [0, 2*pi). */
PL_TPL PL_INLINE void pl_pll(pl_real f_nominal, pl_real Kp, pl_real Ki, pl_real* theta,
                      pl_real* omega, pl_real* integ, pl_real v_alpha,
                      pl_real v_beta, pl_real dt) {
    const pl_real two_pi = 2.0 * 3.141592653589793;
    const pl_real c = pl_cos(*theta), s = pl_sin(*theta);
    const pl_real q = -v_alpha * s + v_beta * c;
    pl_real domega;
    *integ += Ki * dt * q;
    domega = Kp * q + *integ;
    *omega = two_pi * f_nominal + domega;
    *theta += *omega * dt;
    PL_KINK(fmod(*theta, two_pi), 0.0);   /* the wrap: theta jumps by 2 pi */
    *theta = fmod(*theta, two_pi);
    if (PL_LT(*theta, 0.0)) *theta += two_pi;
}

#endif /* PULSIM_BLOCK_LAWS_H */
