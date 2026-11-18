// SPDX-License-Identifier: MIT
//
// Pulsim — control blocks ported from `pulsim.control` to the control
// graph (add-compiled-control-graph, task 5). Each one reproduces its
// Python class operation for operation (same expressions, same order),
// so the two are bit-identical; python/tests/test_control_graph_blocks.py
// checks it on 10^4 random inputs per block.
//
// Direct feedthrough: every input of these blocks is direct feedthrough
// except DelayBlock's (its output is the input of `samples` evaluations
// ago, so it breaks an algebraic loop). DelayBlock is two-phase: `step`
// writes the held output, `update` (after every block has run) shifts
// in this evaluation's input.

#pragma once

#include "pulsim/blockchain/block_adapters.hpp"
#include "pulsim/blockchain/chain.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::blockchain {

/// y[k] = x[k - samples]; the first `samples` outputs are
/// `initial_value`. Mirrors `pulsim.control.DelayBlock`.
struct DelayBlock {
    Size samples = 1;
    Real initial_value = Real{0};
    std::deque<Real> buf;

    void reset() { buf.assign(samples, initial_value); }

    [[nodiscard]] Real output() {
        if (buf.size() != samples) reset();
        return buf.front();
    }

    void push(Real x) {
        if (buf.size() != samples) reset();
        buf.pop_front();
        buf.push_back(x);
    }
};

inline StepResetPair make_delay_step(BlockChain& chain,
                                     std::shared_ptr<DelayBlock> blk,
                                     InputRef x, std::string out_channel) {
    if (blk->samples < 1) {
        throw ControlGraphBuildError(
            "DelayBlock: samples must be >= 1 (got " +
            std::to_string(blk->samples) + ")");
    }
    const InputRef xi = chain.in(std::move(x), /*feedthrough=*/false);
    const Index o = chain.out(out_channel);
    StepResetPair p;
    p.step = [blk, o](ChainContext& ctx) { ctx.put(o, blk->output()); };
    p.update = [blk, xi](ChainContext& ctx) { blk->push(resolve(xi, ctx)); };
    p.reset = [blk]() { blk->reset(); };
    return p;
}


// =============================================================================
// Python-exact ports (task 5.1). Every expression keeps the Python
// operand order (no FMA: the build passes -ffp-contract=off); min/max
// keep Python's argument order (min(a, b) returns b only if b < a).
// =============================================================================

namespace py_exact {
[[nodiscard]] inline Real min(Real a, Real b) noexcept { return b < a ? b : a; }
[[nodiscard]] inline Real max(Real a, Real b) noexcept { return b > a ? b : a; }
}  // namespace py_exact

/// `pulsim.control.Comparator` (Schmitt trigger).
struct PyComparator {
    Real threshold = 0, hysteresis = 0, output_high = 1, output_low = 0;
    bool state = false;
    void reset() { state = false; }
    Real update(Real input_value) {
        const Real half_band = Real{0.5} * hysteresis;
        if (state) {
            if (input_value < threshold - half_band) state = false;
        } else {
            if (input_value > threshold + half_band) state = true;
        }
        return state ? output_high : output_low;
    }
};

/// `pulsim.control.RateLimiter`.
struct PyRateLimiter {
    Real slew_up = 0, slew_down = 0, state = 0;
    void reset() { state = 0; }
    Real update(Real target, Real dt) {
        Real delta = target - state;
        const Real max_up = slew_up * dt;
        const Real max_down = slew_down * dt;
        if (delta > max_up) delta = max_up;
        else if (delta < -max_down) delta = -max_down;
        state += delta;
        return state;
    }
};

/// `pulsim.control.SampleHold`.
struct PySampleHold {
    Real sample_period = 0, state = 0;
    Real last_t = -std::numeric_limits<Real>::infinity();
    void reset() { state = 0; last_t = -std::numeric_limits<Real>::infinity(); }
    Real update(Real input_value, Real t) {
        if (t - last_t >= sample_period) { state = input_value; last_t = t; }
        return state;
    }
};

/// `pulsim.control.TransferFunction` (history lists, newest first).
struct PyTransferFunction {
    std::vector<Real> num, den;
    std::deque<Real> xh, yh;
    void reset() { xh.clear(); yh.clear(); }
    Real update(Real x) {
        xh.push_front(x);
        if (xh.size() > num.size()) xh.resize(num.size());
        Real num_part = 0;
        for (Size i = 0; i < num.size(); ++i) {
            const Real term = num[i] * (i < xh.size() ? xh[i] : Real{0});
            num_part = num_part + term;
        }
        Real den_part = 0;
        for (Size i = 0; i < den.size(); ++i) {
            const Real term = den[i] * (i < yh.size() ? yh[i] : Real{0});
            den_part = den_part + term;
        }
        const Real y = num_part - den_part;
        yh.push_front(y);
        if (yh.size() > den.size()) yh.resize(std::max<Size>(1, den.size()));
        return y;
    }
};

/// `pulsim.control.StateMachine`; mode 0 toggle, 1 level, 2 sr_latch.
struct PyStateMachine {
    int mode = 1;
    Real threshold = 0.5, high_value = 1, low_value = 0, state = 0;
    Real trigger_prev = 0;
    void reset() { state = 0; trigger_prev = 0; }
    Real update(Real trigger, Real set_, Real reset_) {
        if (mode == 0) {
            if (trigger_prev <= threshold && trigger > threshold) {
                state = state == high_value ? low_value : high_value;
            }
            trigger_prev = trigger;
        } else if (mode == 1) {
            state = trigger > threshold ? high_value : low_value;
        } else {
            if (set_ > threshold) state = high_value;
            else if (reset_ > threshold) state = low_value;
        }
        return state;
    }
};

/// `pulsim.control.OpAmp`.
struct PyOpAmp {
    Real gain = 1e5, rail_min = 0, rail_max = 0;
    Real update(Real in_pos, Real in_neg) const {
        const Real u = gain * (in_pos - in_neg);
        return py_exact::max(rail_min, py_exact::min(rail_max, u));
    }
};

/// `pulsim.control.LookupTable1D`.
struct PyLookupTable1D {
    std::vector<Real> xs, ys;
    Real update(Real x) const {
        if (xs.empty()) return 0;
        if (x <= xs.front()) return ys.front();
        if (x >= xs.back()) return ys.back();
        for (Size i = 0; i + 1 < xs.size(); ++i) {
            if (xs[i] <= x && x < xs[i + 1]) {
                const Real t = (x - xs[i]) / (xs[i + 1] - xs[i]);
                const Real dy = t * (ys[i + 1] - ys[i]);
                return ys[i] + dy;
            }
        }
        return ys.back();
    }
};

/// `pulsim.control.MathBlock`; op 0 add 1 sub 2 mul 3 div 4 abs 5 neg
/// 6 sqrt 7 pow2.
struct PyMathBlock {
    int op = 0;
    Real update(Real a, Real b) const {
        switch (op) {
            case 0: return a + b;
            case 1: return a - b;
            case 2: return a * b;
            case 3: return b != Real{0} ? a / b : Real{0};
            case 4: return std::abs(a);
            case 5: return -a;
            case 6: return a >= Real{0} ? std::sqrt(a) : Real{0};
            default: return a * a;
        }
    }
};

/// `pulsim.control.MovingAverageFilter` (an EMA, despite the name).
struct PyEma {
    Real alpha = 0.9, state = 0;
    void reset() { state = 0; }
    Real update(Real x) {
        const Real a = alpha * state;
        const Real b = (Real{1} - alpha) * x;
        state = a + b;
        return state;
    }
};

/// `pulsim.control.Differentiator`.
struct PyDifferentiator {
    Real alpha = 0.2, state = 0, prev_x = 0;
    void reset() { state = 0; prev_x = 0; }
    Real update(Real x, Real dt) {
        const Real d_raw = dt > Real{0} ? (x - prev_x) / dt : Real{0};
        const Real a = alpha * d_raw;
        const Real b = (Real{1} - alpha) * state;
        state = a + b;
        prev_x = x;
        return state;
    }
};

/// `pulsim.control.Integrator` (forward Euler, conditional anti-windup:
/// the state HOLDS when the update would push past a limit).
struct PyIntegrator {
    Real gain = 1, state = 0;
    Real output_min = -std::numeric_limits<Real>::infinity();
    Real output_max = std::numeric_limits<Real>::infinity();
    void reset() { state = 0; }
    Real update(Real x, Real dt) {
        Real nw = state + gain * dt * x;
        if (nw > output_max && x > Real{0}) nw = state;
        else if (nw < output_min && x < Real{0}) nw = state;
        nw = py_exact::max(output_min, py_exact::min(output_max, nw));
        state = nw;
        return nw;
    }
};

/// `pulsim.control.PIDController` (trapezoidal I, filtered D on error).
struct PyPID {
    Real Kp = 1, Ki = 0, Kd = 0, alpha = 0.1;
    Real output_min = -std::numeric_limits<Real>::infinity();
    Real output_max = std::numeric_limits<Real>::infinity();
    Real integ = 0, deriv = 0, prev_error = 0;
    void reset() { integ = 0; deriv = 0; prev_error = 0; }
    Real update(Real setpoint, Real measured, Real dt) {
        const Real error = setpoint - measured;
        Real new_int = integ + Ki * dt * Real{0.5} * (error + prev_error);
        const Real d_raw = dt > Real{0} ? (error - prev_error) / dt : Real{0};
        const Real da = alpha * d_raw;
        const Real db = (Real{1} - alpha) * deriv;
        const Real new_deriv = da + db;
        Real u = Kp * error + new_int + Kd * new_deriv;
        if (u > output_max && error > Real{0}) {
            new_int = integ;
            u = Kp * error + new_int + Kd * new_deriv;
        } else if (u < output_min && error < Real{0}) {
            new_int = integ;
            u = Kp * error + new_int + Kd * new_deriv;
        }
        const Real u_clamped = py_exact::max(output_min, py_exact::min(output_max, u));
        integ = new_int;
        deriv = new_deriv;
        prev_error = error;
        return u_clamped;
    }
};

/// `pulsim.control.FirstOrderLowPass` (with its initial state).
struct PyLowPass {
    Real tau = 0, state = 0;
    void reset() { state = 0; }
    Real update(Real input_value, Real dt) {
        if (tau <= 0) {
            state = input_value;
        } else {
            const Real alpha = dt / (tau + dt);
            state += alpha * (input_value - state);
        }
        return state;
    }
};

/// `pulsim.control.PwmGenerator` + `pwm_on` (phase in seconds; the
/// carrier starts at t = +phase).
struct PyPwm {
    Real frequency = 0, phase = 0;
    void reset() {}
    Real update(Real duty, Real t) const {
        const Real T = frequency > 0 ? Real{1} / frequency
                                     : std::numeric_limits<Real>::infinity();
        if (!std::isfinite(T) || T == 0) return 0;
        const Real x = (t - phase) / T;
        const Real tol = Real{64} * Real{2.220446049250313e-16} *
                         py_exact::max(Real{1}, (std::abs(t) + std::abs(phase)) / T);
        Real u = x - std::floor(x);
        if (u >= Real{1} - tol) u = 0;
        return u < duty - tol ? Real{1} : Real{0};
    }
};

/// Block parameters by name (lists of reals; a scalar is a 1-list).
using PortedParams = std::map<std::string, std::vector<Real>>;

namespace detail {
inline Real param(const PortedParams& p, const std::string& kind,
                  const std::string& name) {
    auto it = p.find(name);
    if (it == p.end() || it->second.size() != 1) {
        throw ControlGraphBuildError(kind + ": parameter '" + name +
                                     "' missing or not a scalar");
    }
    return it->second.front();
}
inline std::vector<Real> list(const PortedParams& p, const std::string& name) {
    auto it = p.find(name);
    return it == p.end() ? std::vector<Real>{} : it->second;
}
inline void arity(const std::string& kind, Size got, Size want_in,
                  Size outs, Size want_out) {
    if (got != want_in || outs != want_out) {
        throw ControlGraphBuildError(
            kind + ": expects " + std::to_string(want_in) + " input(s) and " +
            std::to_string(want_out) + " output(s), got " +
            std::to_string(got) + " and " + std::to_string(outs));
    }
}
}  // namespace detail

/// Build a ported block of `kind` reading `inputs` (in the order the
/// kind documents) and writing `outputs`. Unknown kind: build error.
inline StepResetPair make_ported_step(BlockChain& chain, const std::string& kind,
                                      const PortedParams& p,
                                      std::vector<InputRef> inputs,
                                      const std::vector<std::string>& outputs) {
    using detail::param;
    const Size ni = inputs.size(), no = outputs.size();
    std::vector<InputRef> in;
    in.reserve(ni);
    for (auto& r : inputs) in.push_back(chain.in(std::move(r)));
    std::vector<Index> out;
    out.reserve(no);
    for (const auto& o : outputs) out.push_back(chain.out(o));
    StepResetPair r;
    auto unary = [&](auto blk) {   // one input, one output, stateful
        detail::arity(kind, ni, 1, no, 1);
        r.step = [blk, x = in[0], o = out[0]](ChainContext& c) {
            c.put(o, blk->update(resolve(x, c)));
        };
    };
    auto binary = [&](auto blk) {
        detail::arity(kind, ni, 2, no, 1);
        r.step = [blk, a = in[0], b = in[1], o = out[0]](ChainContext& c) {
            const Real va = resolve(a, c);
            c.put(o, blk->update(va, resolve(b, c)));
        };
    };
    if (kind == "comparator") {
        auto b = std::make_shared<PyComparator>();
        b->threshold = param(p, kind, "threshold");
        b->hysteresis = param(p, kind, "hysteresis");
        b->output_high = param(p, kind, "output_high");
        b->output_low = param(p, kind, "output_low");
        b->state = param(p, kind, "state") != Real{0};
        unary(b); r.reset = [b] { b->reset(); };
    } else if (kind == "rate_limiter") {
        auto b = std::make_shared<PyRateLimiter>();
        b->slew_up = param(p, kind, "slew_up");
        b->slew_down = param(p, kind, "slew_down");
        b->state = param(p, kind, "state");
        binary(b); r.reset = [b] { b->reset(); };
    } else if (kind == "sample_hold") {
        auto b = std::make_shared<PySampleHold>();
        b->sample_period = param(p, kind, "sample_period");
        b->state = param(p, kind, "state");
        b->last_t = param(p, kind, "last_t");
        binary(b); r.reset = [b] { b->reset(); };
    } else if (kind == "transfer_function") {
        auto b = std::make_shared<PyTransferFunction>();
        b->num = detail::list(p, "num");
        b->den = detail::list(p, "den");
        for (Real v : detail::list(p, "x_hist")) b->xh.push_back(v);
        for (Real v : detail::list(p, "y_hist")) b->yh.push_back(v);
        unary(b); r.reset = [b] { b->reset(); };
    } else if (kind == "state_machine") {
        auto b = std::make_shared<PyStateMachine>();
        b->mode = static_cast<int>(param(p, kind, "mode"));
        b->threshold = param(p, kind, "threshold");
        b->high_value = param(p, kind, "high_value");
        b->low_value = param(p, kind, "low_value");
        b->state = param(p, kind, "state");
        b->trigger_prev = param(p, kind, "trigger_prev");
        detail::arity(kind, ni, 3, no, 1);
        r.step = [b, t = in[0], s = in[1], q = in[2], o = out[0]](ChainContext& c) {
            const Real vt = resolve(t, c), vs = resolve(s, c);
            c.put(o, b->update(vt, vs, resolve(q, c)));
        };
        r.reset = [b] { b->reset(); };
    } else if (kind == "op_amp") {
        auto b = std::make_shared<PyOpAmp>();
        b->gain = param(p, kind, "gain");
        b->rail_min = param(p, kind, "rail_min");
        b->rail_max = param(p, kind, "rail_max");
        binary(b); r.reset = [] {};
    } else if (kind == "lookup_table_1d") {
        auto b = std::make_shared<PyLookupTable1D>();
        b->xs = detail::list(p, "xs");
        b->ys = detail::list(p, "ys");
        if (b->xs.size() != b->ys.size()) {
            throw ControlGraphBuildError("lookup_table_1d: xs and ys differ in length");
        }
        unary(b); r.reset = [] {};
    } else if (kind == "math") {
        auto b = std::make_shared<PyMathBlock>();
        b->op = static_cast<int>(param(p, kind, "op"));
        binary(b); r.reset = [] {};
    } else if (kind == "first_order_low_pass") {
        auto b = std::make_shared<PyLowPass>();
        b->tau = param(p, kind, "tau");
        b->state = param(p, kind, "state");
        binary(b); r.reset = [b] { b->reset(); };
    } else if (kind == "pwm") {
        auto b = std::make_shared<PyPwm>();
        b->frequency = param(p, kind, "frequency");
        b->phase = param(p, kind, "phase");
        binary(b); r.reset = [] {};
    } else if (kind == "ema") {
        auto b = std::make_shared<PyEma>();
        b->alpha = param(p, kind, "alpha");
        b->state = param(p, kind, "state");
        unary(b); r.reset = [b] { b->reset(); };
    } else if (kind == "integrator") {
        auto b = std::make_shared<PyIntegrator>();
        b->gain = param(p, kind, "gain");
        b->output_min = param(p, kind, "output_min");
        b->output_max = param(p, kind, "output_max");
        b->state = param(p, kind, "state");
        binary(b); r.reset = [b] { b->reset(); };
    } else if (kind == "differentiator") {
        auto b = std::make_shared<PyDifferentiator>();
        b->alpha = param(p, kind, "alpha");
        b->state = param(p, kind, "state");
        b->prev_x = param(p, kind, "prev_x");
        binary(b); r.reset = [b] { b->reset(); };
    } else if (kind == "pid") {
        auto b = std::make_shared<PyPID>();
        b->Kp = param(p, kind, "Kp"); b->Ki = param(p, kind, "Ki");
        b->Kd = param(p, kind, "Kd"); b->alpha = param(p, kind, "alpha");
        b->output_min = param(p, kind, "output_min");
        b->output_max = param(p, kind, "output_max");
        b->integ = param(p, kind, "integrator_state");
        b->deriv = param(p, kind, "derivative_state");
        b->prev_error = param(p, kind, "prev_error");
        detail::arity(kind, ni, 3, no, 1);
        r.step = [b, s = in[0], m = in[1], d = in[2], o = out[0]](ChainContext& c) {
            const Real vs = resolve(s, c), vm = resolve(m, c);
            c.put(o, b->update(vs, vm, resolve(d, c)));
        };
        r.reset = [b] { b->reset(); };
    } else if (kind == "sum") {
        // y = sum(w_i * x_i), Python's left-to-right sum from 0.
        auto w = detail::list(p, "weights");
        detail::arity(kind, ni, w.size(), no, 1);
        r.step = [w, in, o = out[0]](ChainContext& c) {
            Real acc = 0;
            for (Size i = 0; i < w.size(); ++i) {
                const Real term = w[i] * resolve(in[i], c);
                acc = acc + term;
            }
            c.put(o, acc);
        };
        r.reset = [] {};
    } else if (kind == "signal_mux") {
        // inputs: selector, then the N candidates.
        if (ni < 2 || no != 1) {
            throw ControlGraphBuildError("signal_mux: needs a selector and >= 1 input");
        }
        r.step = [in, o = out[0]](ChainContext& c) {
            // max(0, min(n - 1, int(selector))), int() truncating
            const auto n = static_cast<Real>(in.size() - 1);
            const Real raw = resolve(in[0], c);
            if (!std::isfinite(raw)) {   // Python's int() raises here
                throw std::domain_error("signal_mux: selector is not finite");
            }
            const Real sel = std::trunc(raw);
            const Real idx = sel >= n - 1 ? n - 1 : (sel >= Real{0} ? sel : Real{0});
            c.put(o, resolve(in[static_cast<Size>(idx) + 1], c));
        };
        r.reset = [] {};
    } else if (kind == "signal_demux") {
        detail::arity(kind, ni, 1, no, static_cast<Size>(param(p, kind, "n_outputs")));
        r.step = [x = in[0], out](ChainContext& c) {
            const Real v = resolve(x, c);
            for (Index o : out) c.put(o, v);
        };
        r.reset = [] {};
    } else {
        throw ControlGraphBuildError("control graph: unknown block kind '" + kind + "'");
    }
    return r;
}

}  // namespace pulsim::blockchain
