// SPDX-License-Identifier: MIT
//
// Pulsim — BlockChain adapter factories.
//
// One `make_<block>_step()` factory per POD block in `blocks.hpp`.
// Each factory returns a `BlockStepFn` (closure) that:
//   1. Resolves the block's inputs from the ChainContext.
//   2. Calls the POD's `update(...)` method.
//   3. Writes the result(s) to the output channel(s).
//
// The POD block is held by `std::shared_ptr` so the closure can
// outlive the call stack that built it. The closure also holds a
// matching reset closure (registered alongside the step closure in
// `BlockChain`) so `chain.reset()` resets every block in one pass.

#pragma once

#include "pulsim/blockchain/blocks.hpp"
#include "pulsim/blockchain/chain.hpp"

#include <memory>
#include <string>
#include <utility>

namespace pulsim::blockchain {

// =============================================================================
// Helper — small wrapper that returns both the step + reset closures
// =============================================================================

struct StepResetPair {
    BlockStepFn step;
    BlockResetFn reset;
};


// =============================================================================
// Math blocks
// =============================================================================

inline StepResetPair make_gain_step(std::shared_ptr<Gain> blk,
                                        InputRef x,
                                        std::string out_channel) {
    auto step = [blk, x = std::move(x), out_channel = std::move(out_channel)]
                   (ChainContext& ctx) {
        ctx.channels[out_channel] = blk->update(resolve(x, ctx));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_subtract_step(std::shared_ptr<Subtract> blk,
                                            InputRef a, InputRef b,
                                            std::string out_channel) {
    auto step = [blk, a = std::move(a), b = std::move(b),
                    out_channel = std::move(out_channel)]
                   (ChainContext& ctx) {
        ctx.channels[out_channel] =
            blk->update(resolve(a, ctx), resolve(b, ctx));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_sum_step(std::shared_ptr<Sum> blk,
                                       InputRef a, InputRef b, InputRef c,
                                       std::string out_channel) {
    auto step = [blk, a = std::move(a), b = std::move(b),
                    c = std::move(c),
                    out_channel = std::move(out_channel)]
                   (ChainContext& ctx) {
        ctx.channels[out_channel] = blk->update(
            resolve(a, ctx), resolve(b, ctx), resolve(c, ctx));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_math_block_step(std::shared_ptr<MathBlock> blk,
                                              InputRef a, InputRef b,
                                              std::string out_channel) {
    auto step = [blk, a = std::move(a), b = std::move(b),
                    out_channel = std::move(out_channel)]
                   (ChainContext& ctx) {
        ctx.channels[out_channel] =
            blk->update(resolve(a, ctx), resolve(b, ctx));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}


// =============================================================================
// Control blocks
// =============================================================================

inline StepResetPair make_pi_controller_step(
    std::shared_ptr<PIController> blk,
    InputRef setpoint, InputRef measured, InputRef dt_ref,
    std::string out_channel) {

    auto step = [blk, setpoint = std::move(setpoint),
                    measured = std::move(measured),
                    dt_ref = std::move(dt_ref),
                    out_channel = std::move(out_channel)]
                   (ChainContext& ctx) {
        ctx.channels[out_channel] = blk->update(
            resolve(setpoint, ctx),
            resolve(measured, ctx),
            resolve(dt_ref, ctx));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_pid_controller_step(
    std::shared_ptr<PIDController> blk,
    InputRef setpoint, InputRef measured, InputRef dt_ref,
    std::string out_channel) {

    auto step = [blk, setpoint = std::move(setpoint),
                    measured = std::move(measured),
                    dt_ref = std::move(dt_ref),
                    out_channel = std::move(out_channel)]
                   (ChainContext& ctx) {
        ctx.channels[out_channel] = blk->update(
            resolve(setpoint, ctx),
            resolve(measured, ctx),
            resolve(dt_ref, ctx));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_first_order_lpf_step(
    std::shared_ptr<FirstOrderLowPass> blk,
    InputRef input_value, InputRef dt_ref,
    std::string out_channel) {

    auto step = [blk, input_value = std::move(input_value),
                    dt_ref = std::move(dt_ref),
                    out_channel = std::move(out_channel)]
                   (ChainContext& ctx) {
        ctx.channels[out_channel] = blk->update(
            resolve(input_value, ctx), resolve(dt_ref, ctx));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_integrator_step(
    std::shared_ptr<Integrator> blk,
    InputRef x, InputRef dt_ref,
    std::string out_channel) {

    auto step = [blk, x = std::move(x), dt_ref = std::move(dt_ref),
                    out_channel = std::move(out_channel)]
                   (ChainContext& ctx) {
        ctx.channels[out_channel] = blk->update(
            resolve(x, ctx), resolve(dt_ref, ctx));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_differentiator_step(
    std::shared_ptr<Differentiator> blk,
    InputRef x, InputRef dt_ref,
    std::string out_channel) {

    auto step = [blk, x = std::move(x), dt_ref = std::move(dt_ref),
                    out_channel = std::move(out_channel)]
                   (ChainContext& ctx) {
        ctx.channels[out_channel] = blk->update(
            resolve(x, ctx), resolve(dt_ref, ctx));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_limiter_step(std::shared_ptr<Limiter> blk,
                                           InputRef x,
                                           std::string out_channel) {
    auto step = [blk, x = std::move(x),
                    out_channel = std::move(out_channel)]
                   (ChainContext& ctx) {
        ctx.channels[out_channel] = blk->update(resolve(x, ctx));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_moving_average_step(
    std::shared_ptr<MovingAverageFilter> blk,
    InputRef x,
    std::string out_channel) {

    auto step = [blk, x = std::move(x),
                    out_channel = std::move(out_channel)]
                   (ChainContext& ctx) {
        ctx.channels[out_channel] = blk->update(resolve(x, ctx));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}


// =============================================================================
// Modulation
// =============================================================================

inline StepResetPair make_pwm_generator_step(
    std::shared_ptr<PwmGenerator> blk,
    InputRef duty, InputRef t_ref,
    std::string out_channel) {

    auto step = [blk, duty = std::move(duty), t_ref = std::move(t_ref),
                    out_channel = std::move(out_channel)]
                   (ChainContext& ctx) {
        ctx.channels[out_channel] = blk->update(
            resolve(duty, ctx), resolve(t_ref, ctx));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_svm_step(std::shared_ptr<SpaceVectorModulator> blk,
                                       InputRef v_alpha, InputRef v_beta,
                                       std::string da_channel,
                                       std::string db_channel,
                                       std::string dc_channel) {
    auto step = [blk, v_alpha = std::move(v_alpha),
                    v_beta = std::move(v_beta),
                    da_channel = std::move(da_channel),
                    db_channel = std::move(db_channel),
                    dc_channel = std::move(dc_channel)]
                   (ChainContext& ctx) {
        auto o = blk->update(resolve(v_alpha, ctx),
                                  resolve(v_beta, ctx));
        ctx.channels[da_channel] = o.da;
        ctx.channels[db_channel] = o.db;
        ctx.channels[dc_channel] = o.dc;
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}


// =============================================================================
// Transforms — multi-output
// =============================================================================

inline StepResetPair make_clarke_step(std::shared_ptr<ClarkeTransform> blk,
                                          InputRef a, InputRef b, InputRef c,
                                          std::string alpha_channel,
                                          std::string beta_channel,
                                          std::string zero_channel) {
    auto step = [blk, a = std::move(a), b = std::move(b),
                    c = std::move(c),
                    alpha_channel = std::move(alpha_channel),
                    beta_channel = std::move(beta_channel),
                    zero_channel = std::move(zero_channel)]
                   (ChainContext& ctx) {
        auto o = blk->update(resolve(a, ctx), resolve(b, ctx),
                                 resolve(c, ctx));
        ctx.channels[alpha_channel] = o.alpha;
        ctx.channels[beta_channel]  = o.beta;
        ctx.channels[zero_channel]  = o.zero;
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_inverse_clarke_step(
    std::shared_ptr<InverseClarkeTransform> blk,
    InputRef alpha, InputRef beta, InputRef zero,
    std::string a_channel, std::string b_channel, std::string c_channel) {
    auto step = [blk, alpha = std::move(alpha), beta = std::move(beta),
                    zero = std::move(zero),
                    a_channel = std::move(a_channel),
                    b_channel = std::move(b_channel),
                    c_channel = std::move(c_channel)]
                   (ChainContext& ctx) {
        auto o = blk->update(resolve(alpha, ctx), resolve(beta, ctx),
                                 resolve(zero, ctx));
        ctx.channels[a_channel] = o.a;
        ctx.channels[b_channel] = o.b;
        ctx.channels[c_channel] = o.c;
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_park_step(std::shared_ptr<ParkTransform> blk,
                                        InputRef alpha, InputRef beta,
                                        InputRef theta,
                                        std::string d_channel,
                                        std::string q_channel) {
    auto step = [blk, alpha = std::move(alpha), beta = std::move(beta),
                    theta = std::move(theta),
                    d_channel = std::move(d_channel),
                    q_channel = std::move(q_channel)]
                   (ChainContext& ctx) {
        auto o = blk->update(resolve(alpha, ctx), resolve(beta, ctx),
                                 resolve(theta, ctx));
        ctx.channels[d_channel] = o.d;
        ctx.channels[q_channel] = o.q;
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_inverse_park_step(
    std::shared_ptr<InverseParkTransform> blk,
    InputRef d, InputRef q, InputRef theta,
    std::string alpha_channel, std::string beta_channel) {
    auto step = [blk, d = std::move(d), q = std::move(q),
                    theta = std::move(theta),
                    alpha_channel = std::move(alpha_channel),
                    beta_channel = std::move(beta_channel)]
                   (ChainContext& ctx) {
        auto o = blk->update(resolve(d, ctx), resolve(q, ctx),
                                 resolve(theta, ctx));
        ctx.channels[alpha_channel] = o.alpha;
        ctx.channels[beta_channel]  = o.beta;
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}


// =============================================================================
// Synchronization
// =============================================================================

inline StepResetPair make_pll_step(std::shared_ptr<PLL> blk,
                                       InputRef v_alpha, InputRef v_beta,
                                       InputRef dt_ref,
                                       std::string theta_channel,
                                       std::string omega_channel,
                                       std::string freq_channel) {
    auto step = [blk, v_alpha = std::move(v_alpha),
                    v_beta = std::move(v_beta),
                    dt_ref = std::move(dt_ref),
                    theta_channel = std::move(theta_channel),
                    omega_channel = std::move(omega_channel),
                    freq_channel = std::move(freq_channel)]
                   (ChainContext& ctx) {
        auto o = blk->update(resolve(v_alpha, ctx), resolve(v_beta, ctx),
                                 resolve(dt_ref, ctx));
        ctx.channels[theta_channel] = o.theta;
        ctx.channels[omega_channel] = o.omega;
        ctx.channels[freq_channel]  = o.freq;
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

}  // namespace pulsim::blockchain
