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
#include "pulsim/blockchain/block_laws.h"
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
    chain.capture(blk);
    return p;
}


// =============================================================================
// Python-exact ports (task 5.1). Every expression keeps the Python
// operand order (no FMA: the build passes -ffp-contract=off); min/max
// keep Python's argument order (min(a, b) returns b only if b < a).
// =============================================================================

// py_exact::min / max / clamp are in blocks.hpp (shared with PI, Limiter).

/// `pulsim.control.Comparator` (Schmitt trigger).
struct PyComparator {
    Real threshold = 0, hysteresis = 0, output_high = 1, output_low = 0;
    bool state = false;
    void reset() { state = false; }
    Real update(Real input_value) {
        Real st = state ? Real{1} : Real{0};
        const Real y = pl_comparator(threshold, hysteresis, output_high,
                                     output_low, &st, input_value);
        state = st != Real{0};
        return y;
    }
};

/// `pulsim.control.RateLimiter`.
struct PyRateLimiter {
    Real slew_up = 0, slew_down = 0, state = 0;
    void reset() { state = 0; }
    Real update(Real target, Real dt) {
        return pl_rate_limiter(slew_up, slew_down, &state, target, dt);
    }
};

/// `pulsim.control.SampleHold`.
struct PySampleHold {
    Real sample_period = 0, state = 0;
    Real last_t = -std::numeric_limits<Real>::infinity();
    void reset() { state = 0; last_t = -std::numeric_limits<Real>::infinity(); }
    Real update(Real input_value, Real t) {
        return pl_sample_hold(sample_period, &state, &last_t, input_value, t);
    }
};

/// `pulsim.control.TransferFunction` (history lists, newest first).
struct PyTransferFunction {
    std::vector<Real> num, den;
    std::vector<Real> xh, yh;   // newest first; capacity num / max(1, den)
    int nx = 0, ny = 0;
    void reset() { nx = 0; ny = 0; }
    Real update(Real x) {
        xh.resize(num.size());
        yh.resize(std::max<Size>(1, den.size()));
        return pl_transfer_function(num.data(), static_cast<int>(num.size()),
                                    den.data(), static_cast<int>(den.size()),
                                    xh.data(), &nx, yh.data(), &ny, x);
    }
};

/// `pulsim.control.StateMachine`; mode 0 toggle, 1 level, 2 sr_latch.
struct PyStateMachine {
    int mode = 1;
    Real threshold = 0.5, high_value = 1, low_value = 0, state = 0;
    Real trigger_prev = 0;
    void reset() { state = 0; trigger_prev = 0; }
    Real update(Real trigger, Real set_, Real reset_) {
        return pl_state_machine(mode, threshold, high_value, low_value, &state,
                                &trigger_prev, trigger, set_, reset_);
    }
};

/// `pulsim.control.OpAmp`.
struct PyOpAmp {
    Real gain = 1e5, rail_min = 0, rail_max = 0;
    Real update(Real in_pos, Real in_neg) const {
        return pl_op_amp(gain, rail_min, rail_max, in_pos, in_neg);
    }
};

/// `pulsim.control.LookupTable1D`.
struct PyLookupTable1D {
    std::vector<Real> xs, ys;
    Real update(Real x) const {
        return pl_lookup_1d(xs.data(), ys.data(), static_cast<int>(xs.size()), x);
    }
};

/// `pulsim.control.MathBlock`; op 0 add 1 sub 2 mul 3 div 4 abs 5 neg
/// 6 sqrt 7 pow2.
struct PyMathBlock {
    int op = 0;
    Real update(Real a, Real b) const { return pl_math(op, a, b); }
};

/// `pulsim.control.MovingAverageFilter` (an EMA, despite the name).
struct PyEma {
    Real alpha = 0.9, state = 0;
    void reset() { state = 0; }
    Real update(Real x) { return pl_ema(alpha, &state, x); }
};

/// `pulsim.control.Differentiator`.
struct PyDifferentiator {
    Real alpha = 0.2, state = 0, prev_x = 0;
    void reset() { state = 0; prev_x = 0; }
    Real update(Real x, Real dt) {
        return pl_differentiator(alpha, &state, &prev_x, x, dt);
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
        return pl_integrator(gain, output_min, output_max, &state, x, dt);
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
        return pl_pid(Kp, Ki, Kd, alpha, output_min, output_max, &integ,
                      &deriv, &prev_error, setpoint, measured, dt);
    }
};

/// `pulsim.control.FirstOrderLowPass` (with its initial state).
struct PyLowPass {
    Real tau = 0, state = 0;
    void reset() { state = 0; }
    Real update(Real input_value, Real dt) {
        return pl_low_pass(tau, &state, input_value, dt);
    }
};

/// `pulsim.control.PwmGenerator` + `pwm_on` (phase in seconds; the
/// carrier starts at t = +phase).
struct PyPwm {
    Real frequency = 0, phase = 0;
    void reset() {}
    Real update(Real duty, Real t) const {
        return pl_pwm(frequency, phase, duty, t);
    }
};

/// `pulsim.control.PLL` (angle in [0, 2*pi), Python's initial state).
struct PyPll {
    Real f_nominal = 60, Kp = 100, Ki = 1000;
    Real theta = 0, omega = 0, integ = 0;
    void reset() { theta = 0; omega = 2.0 * 3.141592653589793 * f_nominal; integ = 0; }
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
        unary(b); r.reset = [b] { b->reset(); }; chain.capture(b);
    } else if (kind == "rate_limiter") {
        auto b = std::make_shared<PyRateLimiter>();
        b->slew_up = param(p, kind, "slew_up");
        b->slew_down = param(p, kind, "slew_down");
        b->state = param(p, kind, "state");
        binary(b); r.reset = [b] { b->reset(); }; chain.capture(b);
    } else if (kind == "sample_hold") {
        auto b = std::make_shared<PySampleHold>();
        b->sample_period = param(p, kind, "sample_period");
        b->state = param(p, kind, "state");
        b->last_t = param(p, kind, "last_t");
        binary(b); r.reset = [b] { b->reset(); }; chain.capture(b);
    } else if (kind == "transfer_function") {
        auto b = std::make_shared<PyTransferFunction>();
        b->num = detail::list(p, "num");
        b->den = detail::list(p, "den");
        b->xh = detail::list(p, "x_hist");
        b->yh = detail::list(p, "y_hist");
        b->nx = static_cast<int>(b->xh.size());
        b->ny = static_cast<int>(b->yh.size());
        // capacity before the first update (update() resizes to it)
        b->xh.resize(std::max(b->xh.size(), b->num.size()));
        b->yh.resize(std::max<Size>(b->yh.size(), std::max<Size>(1, b->den.size())));
        unary(b); r.reset = [b] { b->reset(); }; chain.capture(b);
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
        r.reset = [b] { b->reset(); }; chain.capture(b);
    } else if (kind == "op_amp") {
        auto b = std::make_shared<PyOpAmp>();
        b->gain = param(p, kind, "gain");
        b->rail_min = param(p, kind, "rail_min");
        b->rail_max = param(p, kind, "rail_max");
        binary(b); r.reset = {};   // stateless
    } else if (kind == "lookup_table_1d") {
        auto b = std::make_shared<PyLookupTable1D>();
        b->xs = detail::list(p, "xs");
        b->ys = detail::list(p, "ys");
        if (b->xs.size() != b->ys.size()) {
            throw ControlGraphBuildError("lookup_table_1d: xs and ys differ in length");
        }
        unary(b); r.reset = {};   // stateless
    } else if (kind == "math") {
        auto b = std::make_shared<PyMathBlock>();
        b->op = static_cast<int>(param(p, kind, "op"));
        binary(b); r.reset = {};   // stateless
    } else if (kind == "first_order_low_pass") {
        auto b = std::make_shared<PyLowPass>();
        b->tau = param(p, kind, "tau");
        b->state = param(p, kind, "state");
        binary(b); r.reset = [b] { b->reset(); }; chain.capture(b);
    } else if (kind == "pwm") {
        auto b = std::make_shared<PyPwm>();
        b->frequency = param(p, kind, "frequency");
        b->phase = param(p, kind, "phase");
        binary(b); r.reset = {};   // stateless
    } else if (kind == "ema") {
        auto b = std::make_shared<PyEma>();
        b->alpha = param(p, kind, "alpha");
        b->state = param(p, kind, "state");
        unary(b); r.reset = [b] { b->reset(); }; chain.capture(b);
    } else if (kind == "integrator") {
        auto b = std::make_shared<PyIntegrator>();
        b->gain = param(p, kind, "gain");
        b->output_min = param(p, kind, "output_min");
        b->output_max = param(p, kind, "output_max");
        b->state = param(p, kind, "state");
        binary(b); r.reset = [b] { b->reset(); }; chain.capture(b);
    } else if (kind == "differentiator") {
        auto b = std::make_shared<PyDifferentiator>();
        b->alpha = param(p, kind, "alpha");
        b->state = param(p, kind, "state");
        b->prev_x = param(p, kind, "prev_x");
        binary(b); r.reset = [b] { b->reset(); }; chain.capture(b);
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
        r.reset = [b] { b->reset(); }; chain.capture(b);
    } else if (kind == "sum") {
        // y = sum(w_i * x_i), Python's left-to-right sum from 0.
        auto w = detail::list(p, "weights");
        detail::arity(kind, ni, w.size(), no, 1);
        r.step = [w, in, o = out[0]](ChainContext& c) {
            std::vector<Real> x(in.size());
            for (Size i = 0; i < in.size(); ++i) x[i] = resolve(in[i], c);
            c.put(o, pl_sum(w.data(), x.data(), static_cast<int>(w.size())));
        };
        r.reset = {};   // stateless
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
            const int idx = pl_mux_index(static_cast<int>(n), raw);
            c.put(o, resolve(in[static_cast<Size>(idx) + 1], c));
        };
        r.reset = {};   // stateless
    } else if (kind == "svm") {
        // pulsim.control.SpaceVectorModulator (pl_svm; the kernel
        // add_svm uses the same law since G29).
        detail::arity(kind, ni, 2, no, 3);
        const Real v_dc = param(p, kind, "v_dc");
        r.step = [v_dc, in, out](ChainContext& c) {
            const Real va = resolve(in[0], c);
            Real d[3];
            pl_svm(v_dc, va, resolve(in[1], c), &d[0], &d[1], &d[2]);
            for (Size i = 0; i < 3; ++i) c.put(out[i], d[i]);
        };
        r.reset = {};   // stateless
    } else if (kind == "clarke" || kind == "inverse_clarke" ||
               kind == "park" || kind == "inverse_park") {
        // pulsim.control.{,Inverse}ClarkeTransform / {,Inverse}ParkTransform
        const bool clarke = kind == "clarke" || kind == "inverse_clarke";
        const bool inv = kind == "inverse_clarke" || kind == "inverse_park";
        detail::arity(kind, ni, 3, no, clarke ? 3 : 2);
        r.step = [clarke, inv, in, out](ChainContext& c) {
            const Real a = resolve(in[0], c);
            const Real b = resolve(in[1], c);
            const Real z = resolve(in[2], c);
            Real y[3] = {0, 0, 0};
            if (clarke && !inv) pl_clarke(a, b, z, &y[0], &y[1], &y[2]);
            else if (clarke) pl_inverse_clarke(a, b, z, &y[0], &y[1], &y[2]);
            else if (!inv) pl_park(a, b, z, &y[0], &y[1]);
            else pl_inverse_park(a, b, z, &y[0], &y[1]);
            for (Size i = 0; i < out.size(); ++i) c.put(out[i], y[i]);
        };
        r.reset = {};   // stateless
    } else if (kind == "pll") {
        // pulsim.control.PLL: outputs (theta_hat, omega_hat[, freq]),
        // freq = omega_hat / (2*pi) for the three-output form.
        if (ni != 3 || (no != 2 && no != 3)) {
            throw ControlGraphBuildError("pll: expects 3 inputs and 2 or 3 outputs");
        }
        auto b = std::make_shared<PyPll>();
        b->f_nominal = param(p, kind, "f_nominal");
        b->Kp = param(p, kind, "Kp"); b->Ki = param(p, kind, "Ki");
        b->theta = param(p, kind, "theta_hat");
        b->omega = param(p, kind, "omega_hat");
        b->integ = param(p, kind, "integ");
        r.step = [b, in, out](ChainContext& c) {
            const Real va = resolve(in[0], c), vb = resolve(in[1], c);
            pl_pll(b->f_nominal, b->Kp, b->Ki, &b->theta, &b->omega, &b->integ,
                   va, vb, resolve(in[2], c));
            c.put(out[0], b->theta);
            c.put(out[1], b->omega);
            if (out.size() == 3) c.put(out[2], b->omega / (2.0 * 3.141592653589793));
        };
        r.reset = [b] { b->reset(); }; chain.capture(b);
    } else if (kind == "signal_demux") {
        detail::arity(kind, ni, 1, no, static_cast<Size>(param(p, kind, "n_outputs")));
        r.step = [x = in[0], out](ChainContext& c) {
            const Real v = resolve(x, c);
            for (Index o : out) c.put(o, v);
        };
        r.reset = {};   // stateless
    } else {
        throw ControlGraphBuildError("control graph: unknown block kind '" + kind + "'");
    }
    return r;
}

}  // namespace pulsim::blockchain
