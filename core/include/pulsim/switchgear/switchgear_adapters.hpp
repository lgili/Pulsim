// SPDX-License-Identifier: MIT
//
// Pulsim v2 — Phase C.4 / C++ port: protection-device chain blocks.
//
// Two stateful blocks that live entirely in C++:
//
//   * `add_thyristor_to_chain(chain, ...)` — gate-triggered latching
//     switch with current-zero turn-off. Reads a gate signal (could
//     be a Channel from a Comparator block, or a Const for "always
//     trigger"), reads the conduction current (Node InputRef →
//     inductor branch current), and writes 1.0 / 0.0 to
//     `output_channel` (high while latched ON). The chain's
//     `make_chain_switch_fn` maps the channel to a switch bit.
//
//   * `add_fuse_to_chain(chain, ...)` — thermal I²t fuse. Integrates
//     i²·dt; opens irreversibly when the integral exceeds the
//     threshold. Writes 1.0 (intact) / 0.0 (blown) to
//     `output_channel`.
//
// Both blocks store their mutable state in a shared_ptr so the
// chain's reset() can zero them.

#pragma once

#include "pulsim/blockchain/chain.hpp"
#include "pulsim/numeric/types.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <utility>

namespace pulsim::switchgear {

using blockchain::BlockChain;
using blockchain::ChainContext;
using blockchain::InputRef;

// =============================================================================
// Thyristor — gate-triggered latch with current-zero turn-off
// =============================================================================

struct ThyristorState {
    bool latched_on = false;
    Real i_holding = Real{0};   // current below which we turn OFF

    void reset() noexcept { latched_on = false; }
};

inline void add_thyristor_to_chain(
    BlockChain& chain,
    InputRef gate_ref,
    InputRef current_ref,
    std::string output_channel,
    Real i_holding = Real{0}) {

    auto state = std::make_shared<ThyristorState>();
    state->i_holding = i_holding;

    auto step = [state, gate_ref = std::move(gate_ref),
                    current_ref = std::move(current_ref),
                    output_channel = std::move(output_channel)]
                   (ChainContext& ctx) {
        // Gate trigger asserted (gate signal > 0.5)?
        const Real gate = blockchain::resolve(gate_ref, ctx);
        const Real i = blockchain::resolve(current_ref, ctx);
        if (state->latched_on) {
            if (std::abs(i) < state->i_holding + Real{1e-12}) {
                state->latched_on = false;
            }
        } else {
            if (gate > Real{0.5}) {
                state->latched_on = true;
            }
        }
        ctx.channels[output_channel] =
            state->latched_on ? Real{1} : Real{0};
    };
    auto reset = [state]() { state->reset(); };
    chain.add(std::move(step), std::move(reset));
}


// =============================================================================
// Fuse — thermal I²t with irreversible open
// =============================================================================

struct FuseState {
    bool blown = false;
    Real i2t = Real{0};
    Real t_last = Real{-1};

    void reset() noexcept {
        blown = false;
        i2t = Real{0};
        t_last = Real{-1};
    }
};

inline void add_fuse_to_chain(
    BlockChain& chain,
    InputRef current_ref,
    Real i2t_threshold,
    std::string output_channel,
    bool initial_intact = true) {

    auto state = std::make_shared<FuseState>();
    state->blown = !initial_intact;

    auto step = [state, current_ref = std::move(current_ref),
                    i2t_threshold,
                    output_channel = std::move(output_channel)]
                   (ChainContext& ctx) {
        if (!state->blown) {
            if (state->t_last >= Real{0}) {
                const Real dt = ctx.t - state->t_last;
                if (dt > Real{0}) {
                    const Real i = blockchain::resolve(current_ref, ctx);
                    state->i2t += i * i * dt;
                    if (state->i2t >= i2t_threshold) {
                        state->blown = true;
                    }
                }
            }
            state->t_last = ctx.t;
        }
        ctx.channels[output_channel] = state->blown ? Real{0} : Real{1};
    };
    auto reset = [state]() { state->reset(); };
    chain.add(std::move(step), std::move(reset));
}

}  // namespace pulsim::switchgear
