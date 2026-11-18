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

/* Python's min(a, b) / max(a, b): the first argument unless the second
 * is strictly smaller / larger. */
PL_INLINE double pl_min(double a, double b) { return b < a ? b : a; }
PL_INLINE double pl_max(double a, double b) { return b > a ? b : a; }
/* max(lo, min(hi, u)) */
PL_INLINE double pl_clamp(double lo, double hi, double u) {
    return pl_max(lo, pl_min(hi, u));
}

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

PL_INLINE double pl_gain(double k, double x) { return k * x; }
PL_INLINE double pl_subtract(double a, double b) { return a - b; }

/* sum(w_i * x_i), left to right from 0 */
PL_INLINE double pl_sum(const double* w, const double* x, int n) {
    double acc = 0.0;
    int i;
    for (i = 0; i < n; ++i) {
        const double term = w[i] * x[i];
        acc = acc + term;
    }
    return acc;
}

/* PIController: trapezoidal integral, conditional anti-windup. */
PL_INLINE double pl_pi(double Kp, double Ki, double lo, double hi,
                       double* integral, double* prev_error,
                       double setpoint, double measured, double dt) {
    const double err = setpoint - measured;
    double new_int = *integral + Ki * dt * 0.5 * (err + *prev_error);
    double u = Kp * err + new_int;
    if (u > hi && err > 0.0) {
        new_int = *integral;
        u = Kp * err + new_int;
    } else if (u < lo && err < 0.0) {
        new_int = *integral;
        u = Kp * err + new_int;
    }
    *integral = new_int;
    *prev_error = err;
    return pl_clamp(lo, hi, u);
}

/* PIDController: trapezoidal I, filtered D on the error. */
PL_INLINE double pl_pid(double Kp, double Ki, double Kd, double alpha,
                        double lo, double hi, double* integ, double* deriv,
                        double* prev_error, double setpoint, double measured,
                        double dt) {
    const double error = setpoint - measured;
    double new_int = *integ + Ki * dt * 0.5 * (error + *prev_error);
    const double d_raw = dt > 0.0 ? (error - *prev_error) / dt : 0.0;
    const double da = alpha * d_raw;
    const double db = (1.0 - alpha) * *deriv;
    const double new_deriv = da + db;
    double u = Kp * error + new_int + Kd * new_deriv;
    if (u > hi && error > 0.0) {
        new_int = *integ;
        u = Kp * error + new_int + Kd * new_deriv;
    } else if (u < lo && error < 0.0) {
        new_int = *integ;
        u = Kp * error + new_int + Kd * new_deriv;
    }
    *integ = new_int;
    *deriv = new_deriv;
    *prev_error = error;
    return pl_clamp(lo, hi, u);
}

/* Integrator: forward Euler; the state holds when it would pass a limit. */
PL_INLINE double pl_integrator(double gain, double lo, double hi,
                               double* state, double x, double dt) {
    double nw = *state + gain * dt * x;
    if (nw > hi && x > 0.0) nw = *state;
    else if (nw < lo && x < 0.0) nw = *state;
    nw = pl_clamp(lo, hi, nw);
    *state = nw;
    return nw;
}

PL_INLINE double pl_differentiator(double alpha, double* state,
                                   double* prev_x, double x, double dt) {
    const double d_raw = dt > 0.0 ? (x - *prev_x) / dt : 0.0;
    const double a = alpha * d_raw;
    const double b = (1.0 - alpha) * *state;
    *state = a + b;
    *prev_x = x;
    return *state;
}

/* MovingAverageFilter (an exponential average). */
PL_INLINE double pl_ema(double alpha, double* state, double x) {
    const double a = alpha * *state;
    const double b = (1.0 - alpha) * x;
    *state = a + b;
    return *state;
}

PL_INLINE double pl_low_pass(double tau, double* state, double x, double dt) {
    if (tau <= 0.0) {
        *state = x;
    } else {
        const double alpha = dt / (tau + dt);
        *state += alpha * (x - *state);
    }
    return *state;
}

PL_INLINE double pl_limiter(double lo, double hi, double x) {
    return pl_clamp(lo, hi, x);
}

/* Comparator (Schmitt trigger); *state is 0 or 1. */
PL_INLINE double pl_comparator(double threshold, double hysteresis,
                               double high, double low, double* state,
                               double x) {
    const double half_band = 0.5 * hysteresis;
    if (*state != 0.0) {
        if (x < threshold - half_band) *state = 0.0;
    } else {
        if (x > threshold + half_band) *state = 1.0;
    }
    return *state != 0.0 ? high : low;
}

PL_INLINE double pl_rate_limiter(double up, double down, double* state,
                                 double target, double dt) {
    double delta = target - *state;
    const double max_up = up * dt;
    const double max_down = down * dt;
    if (delta > max_up) delta = max_up;
    else if (delta < -max_down) delta = -max_down;
    *state += delta;
    return *state;
}

PL_INLINE double pl_sample_hold(double period, double* state, double* last_t,
                                double x, double t) {
    if (t - *last_t >= period) {
        *state = x;
        *last_t = t;
    }
    return *state;
}

/* TransferFunction: xh / yh newest first, *nx <= nn, *ny <= max(1, nd);
 * the buffers hold nn and max(1, nd) doubles. */
PL_INLINE double pl_transfer_function(const double* num, int nn,
                                      const double* den, int nd,
                                      double* xh, int* nx, double* yh,
                                      int* ny, double x) {
    double num_part = 0.0, den_part = 0.0, y;
    const int ycap = nd > 1 ? nd : 1;
    int i;
    if (nn > 0) {
        const int keep = *nx < nn ? *nx : nn - 1;
        memmove(xh + 1, xh, (size_t)keep * sizeof(double));
        xh[0] = x;
        *nx = keep + 1;
    }
    for (i = 0; i < nn; ++i) {
        const double term = num[i] * (i < *nx ? xh[i] : 0.0);
        num_part = num_part + term;
    }
    for (i = 0; i < nd; ++i) {
        const double term = den[i] * (i < *ny ? yh[i] : 0.0);
        den_part = den_part + term;
    }
    y = num_part - den_part;
    {
        const int keep = *ny < ycap ? *ny : ycap - 1;
        memmove(yh + 1, yh, (size_t)keep * sizeof(double));
        yh[0] = y;
        *ny = keep + 1;
    }
    return y;
}

/* StateMachine: mode 0 toggle, 1 level, 2 SR latch. */
PL_INLINE double pl_state_machine(int mode, double threshold, double high,
                                  double low, double* state,
                                  double* trigger_prev, double trigger,
                                  double set_, double reset_) {
    if (mode == 0) {
        if (*trigger_prev <= threshold && trigger > threshold) {
            *state = *state == high ? low : high;
        }
        *trigger_prev = trigger;
    } else if (mode == 1) {
        *state = trigger > threshold ? high : low;
    } else {
        if (set_ > threshold) *state = high;
        else if (reset_ > threshold) *state = low;
    }
    return *state;
}

PL_INLINE double pl_op_amp(double gain, double rail_min, double rail_max,
                           double in_pos, double in_neg) {
    const double u = gain * (in_pos - in_neg);
    return pl_max(rail_min, pl_min(rail_max, u));
}

PL_INLINE double pl_lookup_1d(const double* xs, const double* ys, int n,
                              double x) {
    int i;
    if (n == 0) return 0.0;
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    for (i = 0; i + 1 < n; ++i) {
        if (xs[i] <= x && x < xs[i + 1]) {
            const double t = (x - xs[i]) / (xs[i + 1] - xs[i]);
            const double dy = t * (ys[i + 1] - ys[i]);
            return ys[i] + dy;
        }
    }
    return ys[n - 1];
}

/* MathBlock: op 0 add 1 sub 2 mul 3 div 4 abs 5 neg 6 sqrt 7 pow2. */
PL_INLINE double pl_math(int op, double a, double b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return b != 0.0 ? a / b : 0.0;
        case 4: return fabs(a);
        case 5: return -a;
        case 6: return a >= 0.0 ? sqrt(a) : 0.0;
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
PL_INLINE void pl_svm(double v_dc, double v_alpha, double v_beta,
                      double* d_a, double* d_b, double* d_c) {
    const double v3 = sqrt(3.0);
    const double u_a = v_alpha;
    const double u_b = (-v_alpha + v3 * v_beta) / 2.0;
    const double u_c = (-v_alpha - v3 * v_beta) / 2.0;
    const double u_max = pl_max(pl_max(u_a, u_b), u_c);
    const double u_min = pl_min(pl_min(u_a, u_b), u_c);
    const double u_zero = 0.5 * (u_max + u_min);
    const double vdc = v_dc > 0.0 ? v_dc : 1.0;
    *d_a = pl_max(0.0, pl_min(1.0, 0.5 + (u_a - u_zero) / vdc));
    *d_b = pl_max(0.0, pl_min(1.0, 0.5 + (u_b - u_zero) / vdc));
    *d_c = pl_max(0.0, pl_min(1.0, 0.5 + (u_c - u_zero) / vdc));
}

PL_INLINE void pl_clarke(double a, double b, double c, double* alpha,
                         double* beta, double* zero) {
    const double v3 = sqrt(3.0);
    *alpha = (2.0 / 3.0) * (a - 0.5 * b - 0.5 * c);
    *beta = (2.0 / 3.0) * (v3 / 2.0) * (b - c);
    *zero = (1.0 / 3.0) * (a + b + c);
}

PL_INLINE void pl_inverse_clarke(double alpha, double beta, double zero,
                                 double* a, double* b, double* c) {
    const double v3 = sqrt(3.0);
    *a = alpha + zero;
    *b = -0.5 * alpha + (v3 / 2.0) * beta + zero;
    *c = -0.5 * alpha - (v3 / 2.0) * beta + zero;
}

PL_INLINE void pl_park(double alpha, double beta, double theta, double* d,
                       double* q) {
    const double c = pl_cos(theta), s = pl_sin(theta);
    *d = alpha * c + beta * s;
    *q = -alpha * s + beta * c;
}

PL_INLINE void pl_inverse_park(double d, double q, double theta,
                               double* alpha, double* beta) {
    const double c = pl_cos(theta), s = pl_sin(theta);
    *alpha = d * c - q * s;
    *beta = d * s + q * c;
}

/* PLL: cross-product detector, PI on q, angle wrapped to [0, 2*pi). */
PL_INLINE void pl_pll(double f_nominal, double Kp, double Ki, double* theta,
                      double* omega, double* integ, double v_alpha,
                      double v_beta, double dt) {
    const double two_pi = 2.0 * 3.141592653589793;
    const double c = pl_cos(*theta), s = pl_sin(*theta);
    const double q = -v_alpha * s + v_beta * c;
    double domega;
    *integ += Ki * dt * q;
    domega = Kp * q + *integ;
    *omega = two_pi * f_nominal + domega;
    *theta += *omega * dt;
    *theta = fmod(*theta, two_pi);
    if (*theta < 0.0) *theta += two_pi;
}

#endif /* PULSIM_BLOCK_LAWS_H */
