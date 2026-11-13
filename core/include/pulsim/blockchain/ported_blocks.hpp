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

#include <deque>
#include <memory>
#include <string>
#include <utility>

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
            "DelayBlock: samples must be >= 1 (got 0)");
    }
    const InputRef xi = chain.in(std::move(x), /*feedthrough=*/false);
    const Index o = chain.out(out_channel);
    StepResetPair p;
    p.step = [blk, o](ChainContext& ctx) { ctx.put(o, blk->output()); };
    p.update = [blk, xi](ChainContext& ctx) { blk->push(resolve(xi, ctx)); };
    p.reset = [blk]() { blk->reset(); };
    return p;
}

}  // namespace pulsim::blockchain
