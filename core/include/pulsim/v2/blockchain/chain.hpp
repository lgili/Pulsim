// SPDX-License-Identifier: MIT
//
// Pulsim v2 — Phase A.1 stage 2 / C++ port: BlockChain executor.
//
// This is the kernel-side BlockChain: a topologically-ordered list
// of mixed-domain control blocks evaluated per simulation step. The
// matching Python wrapper (`pulsim.v2.MixedDomainBlockChain`) used
// to do this work in pure Python via a callback the kernel invoked
// every step — for a 1 M-step simulation that's ~5-10 s of Python
// overhead. Doing it in C++ eliminates that overhead entirely; the
// Python wrapper becomes a builder API only.
//
// Architecture:
//
//   * `Channels` — std::unordered_map<std::string, Real>. Named
//     scalar "wires" between blocks. Each block reads N channels and
//     writes 1 or more.
//   * `InputRef` — describes WHERE to read an input from:
//       - Const(v)           : baked-in scalar
//       - Channel("name")    : look up `channels[name]`
//       - Node("name")       : look up `x[node_idx]` (resolved at bind time)
//       - Time / Dt          : `ctx.t` or `ctx.dt`
//   * `ChainContext` — per-step shared state: channels, t, dt, x,
//     node-name → state-index resolver.
//   * `BlockChain::step(ctx)` — runs every block in insertion order.
//   * `BlockChain::make_step_observer(...)` — returns the
//     `std::function<void(Real, const Vector&)>` lambda the kernel's
//     `run_transient` expects.
//
// The block-step function is a `std::function<void(ChainContext&)>`
// that captures the block's POD state + the input/output channel
// names by closure. Factory functions in `block_adapters.hpp` build
// these closures around the existing POD blocks from `blocks.hpp`.

#pragma once

#include "pulsim/v2/blockchain/blocks.hpp"
#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pulsim::v2::blockchain {

// =============================================================================
// Channels
// =============================================================================

using Channels = std::unordered_map<std::string, Real>;


// =============================================================================
// InputRef — how to resolve one block-input each step
// =============================================================================

struct InputRef {
    enum class Kind { Const, Channel, Node, Time, Dt };
    Kind kind = Kind::Const;
    Real const_value = Real{0};
    std::string name;       // channel name or node name
    Index node_idx = -1;    // resolved by BlockChain::bind()

    static InputRef from_const(Real v)            {
        InputRef r; r.kind = Kind::Const; r.const_value = v; return r;
    }
    static InputRef from_channel(std::string n)  {
        InputRef r; r.kind = Kind::Channel; r.name = std::move(n);
        return r;
    }
    static InputRef from_node(std::string n)     {
        InputRef r; r.kind = Kind::Node; r.name = std::move(n); return r;
    }
    static InputRef from_time()                   {
        InputRef r; r.kind = Kind::Time; return r;
    }
    static InputRef from_dt()                     {
        InputRef r; r.kind = Kind::Dt; return r;
    }
};


// =============================================================================
// ChainContext — per-step shared state
// =============================================================================

struct ChainContext {
    Channels channels;
    Real t = Real{0};
    Real dt = Real{0};
    const Vector* x = nullptr;
    // Per-step b_extra contributions, keyed by state-vector index.
    // Motors and other "source-side" blocks set entries here; the
    // chain's `make_b_extra_fn(state_size)` emits a Vector from
    // this map on every b_extra_fn(t) call.
    std::unordered_map<Index, Real>* b_extra = nullptr;
};

[[nodiscard]] inline Real resolve(const InputRef& ref,
                                      const ChainContext& ctx) noexcept {
    switch (ref.kind) {
        case InputRef::Kind::Const:   return ref.const_value;
        case InputRef::Kind::Channel: {
            auto it = ctx.channels.find(ref.name);
            return (it == ctx.channels.end()) ? Real{0} : it->second;
        }
        case InputRef::Kind::Node:
            if (ctx.x && ref.node_idx >= 0 &&
                  ref.node_idx < static_cast<Index>(ctx.x->size())) {
                return (*ctx.x)[ref.node_idx];
            }
            return Real{0};
        case InputRef::Kind::Time:    return ctx.t;
        case InputRef::Kind::Dt:      return ctx.dt;
    }
    return Real{0};
}


// =============================================================================
// BlockStepFn — type-erased step function
// =============================================================================

using BlockStepFn = std::function<void(ChainContext&)>;
using BlockResetFn = std::function<void()>;


// =============================================================================
// BlockChain — ordered list of block-step closures
// =============================================================================

class BlockChain {
public:
    BlockChain() = default;

    /// Add a block to the chain. Takes a step closure (does compute +
    /// channel write) and an optional reset closure.
    void add(BlockStepFn step,
                BlockResetFn reset_fn = {}) {
        blocks_.push_back(std::move(step));
        if (reset_fn) {
            resets_.push_back(std::move(reset_fn));
        }
    }

    /// Resolve every InputRef of kind Node to a state-vector index.
    /// Must be called once before `step()` if any block uses
    /// `InputRef::Kind::Node` inputs.
    void bind_nodes(const std::function<Index(const std::string&)>&
                        node_id_of) {
        // Iterate over registered node refs and look up.
        for (auto& [name, ref_ptrs] : node_refs_) {
            Index idx = node_id_of(name);
            if (idx < 0) {
                throw std::runtime_error(
                    "BlockChain::bind_nodes: unknown node '" +
                    name + "'");
            }
            for (InputRef* p : ref_ptrs) {
                p->node_idx = idx;
            }
        }
    }

    /// Reset all block states, clear channels, clear b_extra map.
    void reset() {
        for (auto& r : resets_) {
            r();
        }
        ctx_.channels.clear();
        b_extra_map_.clear();
    }

    /// Evaluate every block in insertion order.
    void step(Real t, const Vector* x, Real dt) {
        ctx_.t = t;
        ctx_.dt = dt;
        ctx_.x = x;
        ctx_.b_extra = &b_extra_map_;
        for (auto& b : blocks_) {
            b(ctx_);
        }
    }

    /// Return a Vector of length `state_size` carrying the current
    /// b_extra contributions. Used by the chain's `make_b_extra_fn`.
    [[nodiscard]] Vector get_b_extra(Size state_size) const {
        Vector result =
            Vector::Zero(static_cast<Index>(state_size));
        for (auto& [idx, value] : b_extra_map_) {
            if (idx >= 0 && idx < static_cast<Index>(state_size)) {
                result[idx] = value;
            }
        }
        return result;
    }

    /// Build the b_extra_fn callback for `simulate(...)` / `run_transient`.
    /// Use together with `make_step_observer(dt)` when the chain
    /// has motor blocks or other sources that need to inject
    /// back-EMF / current values per step.
    [[nodiscard]] std::function<Vector(Real)>
    make_b_extra_fn(Size state_size) {
        return [this, state_size](Real /*t*/) {
            return this->get_b_extra(state_size);
        };
    }

    /// Read the latest value on a channel (debugging / replay).
    [[nodiscard]] Real get_channel(const std::string& name) const {
        auto it = ctx_.channels.find(name);
        return (it == ctx_.channels.end()) ? Real{0} : it->second;
    }

    /// Direct mutable access to channels (e.g. user can prime a
    /// channel before the simulation starts).
    Channels& channels() noexcept { return ctx_.channels; }
    const Channels& channels() const noexcept { return ctx_.channels; }

    /// Register an `InputRef::Kind::Node` reference so `bind_nodes`
    /// can patch in the state-vector index. Block factories call this
    /// when they construct an InputRef of kind Node.
    void register_node_ref(InputRef* ref) {
        if (ref->kind == InputRef::Kind::Node) {
            node_refs_[ref->name].push_back(ref);
        }
    }

    /// Number of blocks in the chain (diagnostic).
    [[nodiscard]] std::size_t size() const noexcept {
        return blocks_.size();
    }

    /// Build the `step_observer` callback that the kernel's
    /// `run_transient(..., step_observer=...)` expects. The lambda
    /// captures a pointer to this chain so the chain must outlive
    /// the simulation.
    [[nodiscard]] std::function<void(Real, const Vector&)>
    make_step_observer(Real dt) {
        return [this, dt](Real t, const Vector& x) {
            this->step(t, &x, dt);
        };
    }

private:
    std::vector<BlockStepFn> blocks_;
    std::vector<BlockResetFn> resets_;
    ChainContext ctx_;
    // Inputs of kind Node, grouped by name, so bind_nodes() can
    // patch all of them in one pass.
    std::unordered_map<std::string, std::vector<InputRef*>> node_refs_;
    // Per-step b_extra contributions written by motor blocks /
    // source-injecting devices. Cleared between simulations via
    // reset() but PERSISTS within a simulation (a motor block
    // updates "its" entry every step; entries not touched keep
    // their last value).
    std::unordered_map<Index, Real> b_extra_map_;
};


// =============================================================================
// SwitchFnFactory — build a `switch_fn(t)` from chain channels
// =============================================================================

/// Build a `switch_fn(t)` that drives `num_switches` from the chain's
/// channels. `channel_to_switch[i]` is a (channel_name, switch_idx)
/// pair: the switch is ON when the channel value > 0.5.
[[nodiscard]] inline std::function<
    topology::SwitchStateMask(Real)>
make_chain_switch_fn(BlockChain& chain,
                          Size num_switches,
                          const std::vector<std::pair<std::string, Index>>&
                              channel_to_switch) {
    auto pairs = channel_to_switch;  // owned copy
    auto* chain_ptr = &chain;
    return [chain_ptr, num_switches, pairs](Real /*t*/) {
        topology::SwitchStateMask mask{num_switches};
        for (const auto& [ch, idx] : pairs) {
            auto it = chain_ptr->channels().find(ch);
            if (it != chain_ptr->channels().end() &&
                  it->second > Real{0.5}) {
                mask.set(static_cast<Size>(idx), true);
            }
        }
        return mask;
    };
}

}  // namespace pulsim::v2::blockchain
