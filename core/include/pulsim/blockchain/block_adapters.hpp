// SPDX-License-Identifier: MIT
//
// Pulsim — BlockChain adapter factories.
//
// One `make_<block>_step()` factory per POD block in `blocks.hpp`.
// Each factory resolves the block's inputs and outputs to slots on
// `chain` (recording its reads and writes for the schedule) and
// returns a `BlockStepFn` (closure) that resolves the inputs, calls
// the POD's `update(...)` and writes the result(s) to the output slots.
// Every input of these blocks is direct feedthrough: `update` computes
// the output from this evaluation's input (the stateful ones advance
// their state in the same call).
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
    BlockStepFn update{};   // phase-2 state transition (non-feedthrough blocks)
};


// =============================================================================
// Math blocks
// =============================================================================

inline StepResetPair make_gain_step(BlockChain& chain, std::shared_ptr<Gain> blk,
                                        InputRef x,
                                        std::string out_channel) {
    auto step = [blk, x = chain.in(std::move(x)), out_channel = chain.out(out_channel)]
                   (ChainContext& ctx) {
        ctx.put(out_channel, blk->update(resolve(x, ctx)));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_subtract_step(BlockChain& chain, std::shared_ptr<Subtract> blk,
                                            InputRef a, InputRef b,
                                            std::string out_channel) {
    auto step = [blk, a = chain.in(std::move(a)), b = chain.in(std::move(b)),
                    out_channel = chain.out(out_channel)]
                   (ChainContext& ctx) {
        ctx.put(out_channel, blk->update(resolve(a, ctx), resolve(b, ctx)));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_sum_step(BlockChain& chain, std::shared_ptr<Sum> blk,
                                       InputRef a, InputRef b, InputRef c,
                                       std::string out_channel) {
    auto step = [blk, a = chain.in(std::move(a)), b = chain.in(std::move(b)),
                    c = chain.in(std::move(c)),
                    out_channel = chain.out(out_channel)]
                   (ChainContext& ctx) {
        ctx.put(out_channel, blk->update( resolve(a, ctx), resolve(b, ctx), resolve(c, ctx)));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_math_block_step(BlockChain& chain, std::shared_ptr<MathBlock> blk,
                                              InputRef a, InputRef b,
                                              std::string out_channel) {
    auto step = [blk, a = chain.in(std::move(a)), b = chain.in(std::move(b)),
                    out_channel = chain.out(out_channel)]
                   (ChainContext& ctx) {
        ctx.put(out_channel, blk->update(resolve(a, ctx), resolve(b, ctx)));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}


// =============================================================================
// Control blocks
// =============================================================================

inline StepResetPair make_pi_controller_step(BlockChain& chain,
    std::shared_ptr<PIController> blk,
    InputRef setpoint, InputRef measured, InputRef dt_ref,
    std::string out_channel) {

    auto step = [blk, setpoint = chain.in(std::move(setpoint)),
                    measured = chain.in(std::move(measured)),
                    dt_ref = chain.in(std::move(dt_ref)),
                    out_channel = chain.out(out_channel)]
                   (ChainContext& ctx) {
        ctx.put(out_channel, blk->update( resolve(setpoint, ctx), resolve(measured, ctx), resolve(dt_ref, ctx)));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_pid_controller_step(BlockChain& chain,
    std::shared_ptr<PIDController> blk,
    InputRef setpoint, InputRef measured, InputRef dt_ref,
    std::string out_channel) {

    auto step = [blk, setpoint = chain.in(std::move(setpoint)),
                    measured = chain.in(std::move(measured)),
                    dt_ref = chain.in(std::move(dt_ref)),
                    out_channel = chain.out(out_channel)]
                   (ChainContext& ctx) {
        ctx.put(out_channel, blk->update( resolve(setpoint, ctx), resolve(measured, ctx), resolve(dt_ref, ctx)));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_first_order_lpf_step(BlockChain& chain,
    std::shared_ptr<FirstOrderLowPass> blk,
    InputRef input_value, InputRef dt_ref,
    std::string out_channel) {

    auto step = [blk, input_value = chain.in(std::move(input_value)),
                    dt_ref = chain.in(std::move(dt_ref)),
                    out_channel = chain.out(out_channel)]
                   (ChainContext& ctx) {
        ctx.put(out_channel, blk->update( resolve(input_value, ctx), resolve(dt_ref, ctx)));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_integrator_step(BlockChain& chain,
    std::shared_ptr<Integrator> blk,
    InputRef x, InputRef dt_ref,
    std::string out_channel) {

    auto step = [blk, x = chain.in(std::move(x)), dt_ref = chain.in(std::move(dt_ref)),
                    out_channel = chain.out(out_channel)]
                   (ChainContext& ctx) {
        ctx.put(out_channel, blk->update( resolve(x, ctx), resolve(dt_ref, ctx)));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_differentiator_step(BlockChain& chain,
    std::shared_ptr<Differentiator> blk,
    InputRef x, InputRef dt_ref,
    std::string out_channel) {

    auto step = [blk, x = chain.in(std::move(x)), dt_ref = chain.in(std::move(dt_ref)),
                    out_channel = chain.out(out_channel)]
                   (ChainContext& ctx) {
        ctx.put(out_channel, blk->update( resolve(x, ctx), resolve(dt_ref, ctx)));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_limiter_step(BlockChain& chain, std::shared_ptr<Limiter> blk,
                                           InputRef x,
                                           std::string out_channel) {
    auto step = [blk, x = chain.in(std::move(x)),
                    out_channel = chain.out(out_channel)]
                   (ChainContext& ctx) {
        ctx.put(out_channel, blk->update(resolve(x, ctx)));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_moving_average_step(BlockChain& chain,
    std::shared_ptr<MovingAverageFilter> blk,
    InputRef x,
    std::string out_channel) {

    auto step = [blk, x = chain.in(std::move(x)),
                    out_channel = chain.out(out_channel)]
                   (ChainContext& ctx) {
        ctx.put(out_channel, blk->update(resolve(x, ctx)));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}


// =============================================================================
// Modulation
// =============================================================================

inline StepResetPair make_pwm_generator_step(BlockChain& chain,
    std::shared_ptr<PwmGenerator> blk,
    InputRef duty, InputRef t_ref,
    std::string out_channel) {

    auto step = [blk, duty = chain.in(std::move(duty)), t_ref = chain.in(std::move(t_ref)),
                    out_channel = chain.out(out_channel)]
                   (ChainContext& ctx) {
        ctx.put(out_channel, blk->update( resolve(duty, ctx), resolve(t_ref, ctx)));
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_svm_step(BlockChain& chain, std::shared_ptr<SpaceVectorModulator> blk,
                                       InputRef v_alpha, InputRef v_beta,
                                       std::string da_channel,
                                       std::string db_channel,
                                       std::string dc_channel) {
    auto step = [blk, v_alpha = chain.in(std::move(v_alpha)),
                    v_beta = chain.in(std::move(v_beta)),
                    da_channel = chain.out(da_channel),
                    db_channel = chain.out(db_channel),
                    dc_channel = chain.out(dc_channel)]
                   (ChainContext& ctx) {
        auto o = blk->update(resolve(v_alpha, ctx),
                                  resolve(v_beta, ctx));
        ctx.put(da_channel, o.da);
        ctx.put(db_channel, o.db);
        ctx.put(dc_channel, o.dc);
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}


// =============================================================================
// Transforms — multi-output
// =============================================================================

inline StepResetPair make_clarke_step(BlockChain& chain, std::shared_ptr<ClarkeTransform> blk,
                                          InputRef a, InputRef b, InputRef c,
                                          std::string alpha_channel,
                                          std::string beta_channel,
                                          std::string zero_channel) {
    auto step = [blk, a = chain.in(std::move(a)), b = chain.in(std::move(b)),
                    c = chain.in(std::move(c)),
                    alpha_channel = chain.out(alpha_channel),
                    beta_channel = chain.out(beta_channel),
                    zero_channel = chain.out(zero_channel)]
                   (ChainContext& ctx) {
        auto o = blk->update(resolve(a, ctx), resolve(b, ctx),
                                 resolve(c, ctx));
        ctx.put(alpha_channel, o.alpha);
        ctx.put(beta_channel, o.beta);
        ctx.put(zero_channel, o.zero);
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_inverse_clarke_step(BlockChain& chain,
    std::shared_ptr<InverseClarkeTransform> blk,
    InputRef alpha, InputRef beta, InputRef zero,
    std::string a_channel, std::string b_channel, std::string c_channel) {
    auto step = [blk, alpha = chain.in(std::move(alpha)), beta = chain.in(std::move(beta)),
                    zero = chain.in(std::move(zero)),
                    a_channel = chain.out(a_channel),
                    b_channel = chain.out(b_channel),
                    c_channel = chain.out(c_channel)]
                   (ChainContext& ctx) {
        auto o = blk->update(resolve(alpha, ctx), resolve(beta, ctx),
                                 resolve(zero, ctx));
        ctx.put(a_channel, o.a);
        ctx.put(b_channel, o.b);
        ctx.put(c_channel, o.c);
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_park_step(BlockChain& chain, std::shared_ptr<ParkTransform> blk,
                                        InputRef alpha, InputRef beta,
                                        InputRef theta,
                                        std::string d_channel,
                                        std::string q_channel) {
    auto step = [blk, alpha = chain.in(std::move(alpha)), beta = chain.in(std::move(beta)),
                    theta = chain.in(std::move(theta)),
                    d_channel = chain.out(d_channel),
                    q_channel = chain.out(q_channel)]
                   (ChainContext& ctx) {
        auto o = blk->update(resolve(alpha, ctx), resolve(beta, ctx),
                                 resolve(theta, ctx));
        ctx.put(d_channel, o.d);
        ctx.put(q_channel, o.q);
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

inline StepResetPair make_inverse_park_step(BlockChain& chain,
    std::shared_ptr<InverseParkTransform> blk,
    InputRef d, InputRef q, InputRef theta,
    std::string alpha_channel, std::string beta_channel) {
    auto step = [blk, d = chain.in(std::move(d)), q = chain.in(std::move(q)),
                    theta = chain.in(std::move(theta)),
                    alpha_channel = chain.out(alpha_channel),
                    beta_channel = chain.out(beta_channel)]
                   (ChainContext& ctx) {
        auto o = blk->update(resolve(d, ctx), resolve(q, ctx),
                                 resolve(theta, ctx));
        ctx.put(alpha_channel, o.alpha);
        ctx.put(beta_channel, o.beta);
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}


// =============================================================================
// Synchronization
// =============================================================================

inline StepResetPair make_pll_step(BlockChain& chain, std::shared_ptr<PLL> blk,
                                       InputRef v_alpha, InputRef v_beta,
                                       InputRef dt_ref,
                                       std::string theta_channel,
                                       std::string omega_channel,
                                       std::string freq_channel) {
    auto step = [blk, v_alpha = chain.in(std::move(v_alpha)),
                    v_beta = chain.in(std::move(v_beta)),
                    dt_ref = chain.in(std::move(dt_ref)),
                    theta_channel = chain.out(theta_channel),
                    omega_channel = chain.out(omega_channel),
                    freq_channel = chain.out(freq_channel)]
                   (ChainContext& ctx) {
        auto o = blk->update(resolve(v_alpha, ctx), resolve(v_beta, ctx),
                                 resolve(dt_ref, ctx));
        ctx.put(theta_channel, o.theta);
        ctx.put(omega_channel, o.omega);
        ctx.put(freq_channel, o.freq);
    };
    auto reset = [blk]() { blk->reset(); };
    return {std::move(step), std::move(reset)};
}

}  // namespace pulsim::blockchain
