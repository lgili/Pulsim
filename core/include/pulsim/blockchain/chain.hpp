// SPDX-License-Identifier: MIT
//
// Pulsim — the compiled control graph (add-compiled-control-graph),
// built inside the existing BlockChain.
//
//   * Signals are integer slots in one flat array (`ChainContext::sig`).
//     Channel names exist only while the graph is built (`slot()`), so
//     the run loop never hashes a string.
//   * Each block records which slots it reads (and whether the read is
//     direct feedthrough) and which it writes. `compile()` orders the
//     blocks by Kahn's algorithm on the direct-feedthrough graph, ties
//     broken by insertion index, so a chain already in topological
//     order keeps its order and any insertion order gives the same
//     schedule. A strongly connected component (Tarjan) of that graph
//     is an algebraic loop: the build fails naming its blocks.
//   * Execution is two-phase: every block's `step` (its output) in
//     schedule order, then every `update` (the state transition of a
//     block whose output does not depend on its input this instant,
//     e.g. a delay).
//
// Block factories (`block_adapters.hpp` and the motor / thermal /
// switchgear / observer adapters) resolve their inputs with `in()` and
// their outputs with `out()` before building the closure, then call
// `add()`; `in()` / `out()` record the block's reads and writes.

#pragma once

#include "pulsim/blockchain/blocks.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pulsim::blockchain {

/// Build-time error of a control graph (algebraic loop, a slot written
/// by two blocks, an unresolved node).
struct ControlGraphBuildError : std::runtime_error {
    using std::runtime_error::runtime_error;
};


// =============================================================================
// InputRef — how to resolve one block input
// =============================================================================

struct InputRef {
    enum class Kind { Const, Channel, Node, Time, Dt };
    Kind kind = Kind::Const;
    Real const_value = Real{0};
    std::string name;       // channel or node name (build time only)
    Index node_idx = -1;    // state-vector index of a Node
    Index slot = -1;        // signal slot of a Channel, set by BlockChain::in

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
// ChainContext — per-evaluation shared state
// =============================================================================

struct ChainContext {
    std::vector<Real> sig;          // one slot per signal
    Real t = Real{0};
    Real dt = Real{0};
    const Vector* x = nullptr;
    // b_extra contributions keyed by state-vector index (motor and
    // thermal blocks); `make_b_extra_fn` turns them into a Vector.
    std::unordered_map<Index, Real>* b_extra = nullptr;

    /// The slot `s` (an output slot may be -1: not wired, not written).
    Real& at(Index s) noexcept { return sig[static_cast<Size>(s)]; }
    void put(Index s, Real v) noexcept {
        if (s >= 0) sig[static_cast<Size>(s)] = v;
    }
};

[[nodiscard]] inline Real resolve(const InputRef& ref,
                                      const ChainContext& ctx) noexcept {
    switch (ref.kind) {
        case InputRef::Kind::Const:   return ref.const_value;
        case InputRef::Kind::Channel: return ctx.sig[static_cast<Size>(ref.slot)];
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


using BlockStepFn = std::function<void(ChainContext&)>;
using BlockResetFn = std::function<void()>;


// =============================================================================
// BlockChain — the control graph
// =============================================================================

class BlockChain {
public:
    struct Read { Index slot; bool feedthrough; };

    struct Block {
        std::string name;
        std::string kind;
        BlockStepFn step;       // phase 1: compute the outputs
        BlockStepFn update;     // phase 2: state transition (may be empty)
        BlockResetFn reset;
        std::vector<Read> reads;
        std::vector<Index> writes;
    };

    BlockChain() = default;

    // ---- building -------------------------------------------------------

    /// Slot of channel `name`, allocated on first mention (reads 0
    /// until a block or `set_channel` writes it). "" → -1 (no slot).
    Index slot(const std::string& name) {
        if (name.empty()) return -1;
        auto it = slot_of_.find(name);
        if (it != slot_of_.end()) return it->second;
        const Index s = static_cast<Index>(slot_names_.size());
        slot_of_.emplace(name, s);
        slot_names_.push_back(name);
        ctx_.sig.push_back(Real{0});
        return s;
    }

    /// Resolve an input of the block being built and record the read.
    /// `feedthrough` = the block's output depends on this input in the
    /// same evaluation (an ordering edge); a delay's input is not.
    InputRef in(InputRef r, bool feedthrough = true) {
        if (r.kind == InputRef::Kind::Channel) {
            r.slot = slot(r.name);
            pending_.reads.push_back({r.slot, feedthrough});
        } else if (r.kind == InputRef::Kind::Node && r.node_idx < 0) {
            if (!node_id_of_) {
                throw ControlGraphBuildError(
                    "BlockChain: node input '" + r.name + "' is not "
                    "resolved; pass a pre-resolved node index or call "
                    "bind_nodes(node_id_of) before adding the block");
            }
            r.node_idx = node_id_of_(r.name);
            if (r.node_idx < 0) {
                throw ControlGraphBuildError(
                    "BlockChain: unknown node '" + r.name + "'");
            }
        }
        return r;
    }

    /// Read a channel by name inside a block being built (adapters that
    /// read channels without an InputRef).
    Index in_slot(const std::string& name, bool feedthrough = true) {
        const Index s = slot(name);
        if (s >= 0) pending_.reads.push_back({s, feedthrough});
        return s;
    }

    /// Slot an output of the block being built writes. "" → -1.
    Index out(const std::string& name) {
        const Index s = slot(name);
        if (s >= 0) pending_.writes.push_back(s);
        return s;
    }

    /// Finish the block being built. `step` writes its outputs (phase
    /// 1); `update`, when given, is its state transition (phase 2).
    void add(BlockStepFn step, BlockResetFn reset_fn = {},
             BlockStepFn update = {}) {
        Block b = std::move(pending_);
        pending_ = Block{};
        b.step = std::move(step);
        b.update = std::move(update);
        b.reset = std::move(reset_fn);
        if (b.name.empty()) b.name = "block" + std::to_string(blocks_.size());
        blocks_.push_back(std::move(b));
        compiled_ = false;
    }

    /// Name / kind of the most recently added block (error messages,
    /// the reorder report).
    void name_last(const std::string& name, const std::string& kind = {}) {
        if (blocks_.empty()) return;
        blocks_.back().name = name;
        if (!kind.empty()) blocks_.back().kind = kind;
    }

    /// Resolver for node inputs given by name (`InputRef::from_node`).
    /// Applies to blocks added after this call.
    void bind_nodes(std::function<Index(const std::string&)> node_id_of) {
        node_id_of_ = std::move(node_id_of);
    }

    // ---- schedule -------------------------------------------------------

    /// Order the blocks; throws ControlGraphBuildError on an algebraic
    /// loop or a slot with two writers. Idempotent.
    void compile() {
        if (compiled_) return;
        const Size n = blocks_.size();
        std::vector<Index> writer(slot_names_.size(), -1);
        for (Size i = 0; i < n; ++i) {
            for (Index s : blocks_[i].writes) {
                auto& w = writer[static_cast<Size>(s)];
                if (w >= 0 && w != static_cast<Index>(i)) {
                    throw ControlGraphBuildError(
                        "control graph: signal '" + slot_names_[static_cast<Size>(s)] +
                        "' is written by two blocks, '" +
                        blocks_[static_cast<Size>(w)].name + "' and '" +
                        blocks_[i].name + "'");
                }
                w = static_cast<Index>(i);
            }
        }
        // Direct-feedthrough edges producer -> consumer.
        std::vector<std::vector<Size>> succ(n);
        std::vector<Size> indeg(n, 0);
        reorders_.clear();
        for (Size i = 0; i < n; ++i) {
            for (const Read& r : blocks_[i].reads) {
                if (!r.feedthrough) continue;
                const Index w = writer[static_cast<Size>(r.slot)];
                if (w < 0) continue;            // never written: reads its held value
                const auto ws = static_cast<Size>(w);
                if (ws == i) {
                    throw ControlGraphBuildError(
                        "control graph: algebraic loop: block '" +
                        blocks_[i].name + "' reads its own output '" +
                        slot_names_[static_cast<Size>(r.slot)] +
                        "' with direct feedthrough. Insert a DelayBlock "
                        "in the loop.");
                }
                succ[ws].push_back(i);
                ++indeg[i];
                if (ws > i) {
                    reorders_.push_back(
                        "'" + blocks_[i].name + "' reads '" +
                        slot_names_[static_cast<Size>(r.slot)] +
                        "' written by '" + blocks_[ws].name +
                        "', inserted after it");
                }
            }
        }
        // Kahn, smallest insertion index first.
        std::priority_queue<Size, std::vector<Size>, std::greater<>> ready;
        for (Size i = 0; i < n; ++i) if (indeg[i] == 0) ready.push(i);
        order_.clear();
        while (!ready.empty()) {
            const Size i = ready.top();
            ready.pop();
            order_.push_back(i);
            for (Size j : succ[i]) if (--indeg[j] == 0) ready.push(j);
        }
        if (order_.size() != n) {
            throw ControlGraphBuildError(loop_message(succ));
        }
        has_update_ = std::any_of(blocks_.begin(), blocks_.end(),
            [](const Block& b) { return static_cast<bool>(b.update); });
        compiled_ = true;
    }

    /// Consumer/producer pairs whose schedule order differs from their
    /// insertion order (each a one-step delay in the insertion-order
    /// chain). Empty until compile().
    [[nodiscard]] const std::vector<std::string>& reorders() const noexcept {
        return reorders_;
    }

    /// Block names in schedule order (compiles).
    [[nodiscard]] std::vector<std::string> schedule() {
        compile();
        std::vector<std::string> out;
        for (Size i : order_) out.push_back(blocks_[i].name);
        return out;
    }

    // ---- running --------------------------------------------------------

    /// Reset all block states, zero the signals, clear b_extra and the
    /// recorded histories.
    void reset() {
        for (auto& b : blocks_) if (b.reset) b.reset();
        std::fill(ctx_.sig.begin(), ctx_.sig.end(), Real{0});
        b_extra_map_.clear();
        clear_recordings();
        n_evals_ = 0;
        n_block_evals_ = 0;
    }

    /// Evaluate the graph once at (t, x) with block step `dt`.
    void step(Real t, const Vector* x, Real dt) {
        compile();
        ctx_.t = t;
        ctx_.dt = dt;
        ctx_.x = x;
        ctx_.b_extra = &b_extra_map_;
        for (Size i : order_) blocks_[i].step(ctx_);
        if (has_update_) {
            for (Size i : order_) {
                if (blocks_[i].update) blocks_[i].update(ctx_);
            }
        }
        ++n_evals_;
        n_block_evals_ += static_cast<std::uint64_t>(order_.size());
        if (!recordings_.empty()) {
            recording_times_.push_back(t);
            for (auto& rec : recordings_) {
                rec.history.push_back(ctx_.sig[static_cast<Size>(rec.slot)]);
            }
        }
    }

    [[nodiscard]] std::uint64_t n_evals() const noexcept { return n_evals_; }
    [[nodiscard]] std::uint64_t n_block_evals() const noexcept {
        return n_block_evals_;
    }

    // ---- recording ------------------------------------------------------

    void record_channel(const std::string& name, std::size_t reserve_n = 0) {
        const Index s = slot(name);
        for (auto& rec : recordings_) {
            if (rec.slot == s) return;
        }
        recordings_.push_back({s, {}});
        if (reserve_n > 0) recordings_.back().history.reserve(reserve_n);
    }

    [[nodiscard]] const std::vector<Real>& get_channel_history(
        const std::string& name) const {
        static const std::vector<Real> empty{};
        auto it = slot_of_.find(name);
        if (it == slot_of_.end()) return empty;
        for (const auto& rec : recordings_) {
            if (rec.slot == it->second) return rec.history;
        }
        return empty;
    }

    [[nodiscard]] const std::vector<Real>& get_recording_times() const noexcept {
        return recording_times_;
    }

    void clear_recordings() noexcept {
        for (auto& rec : recordings_) rec.history.clear();
        recording_times_.clear();
    }

    // ---- sinks ----------------------------------------------------------

    [[nodiscard]] Vector get_b_extra(Size state_size) const {
        Vector result = Vector::Zero(static_cast<Index>(state_size));
        for (auto& [idx, value] : b_extra_map_) {
            if (idx >= 0 && idx < static_cast<Index>(state_size)) {
                result[idx] = value;
            }
        }
        return result;
    }

    [[nodiscard]] std::function<Vector(Real)>
    make_b_extra_fn(Size state_size) {
        return [this, state_size](Real /*t*/) {
            return this->get_b_extra(state_size);
        };
    }

    /// Latest value of a channel (0 for a name never mentioned).
    [[nodiscard]] Real get_channel(const std::string& name) const {
        auto it = slot_of_.find(name);
        return it == slot_of_.end() ? Real{0}
                                    : ctx_.sig[static_cast<Size>(it->second)];
    }

    void set_channel(const std::string& name, Real value) {
        ctx_.sig[static_cast<Size>(slot(name))] = value;
    }

    /// Value of slot `s` (sinks resolved to a slot at build time).
    [[nodiscard]] Real signal(Index s) const noexcept {
        return ctx_.sig[static_cast<Size>(s)];
    }

    [[nodiscard]] std::size_t size() const noexcept { return blocks_.size(); }
    [[nodiscard]] const std::vector<Block>& blocks() const noexcept {
        return blocks_;
    }
    [[nodiscard]] const std::vector<std::string>& slot_names() const noexcept {
        return slot_names_;
    }

    /// The `step_observer` the kernel's run_transient expects: one
    /// evaluation per call. The chain must outlive the simulation.
    [[nodiscard]] std::function<void(Real, const Vector&)>
    make_step_observer(Real dt) {
        compile();
        return [this, dt](Real t, const Vector& x) {
            this->step(t, &x, dt);
        };
    }

private:
    std::string loop_message(const std::vector<std::vector<Size>>& succ) const {
        // Tarjan SCC; report the first component with > 1 block.
        const Size n = blocks_.size();
        std::vector<Index> idx(n, -1), low(n, 0);
        std::vector<bool> on(n, false);
        std::vector<Size> st;
        Index counter = 0;
        std::vector<std::vector<Size>> comps;
        std::function<void(Size)> visit = [&](Size v) {
            idx[v] = low[v] = counter++;
            st.push_back(v);
            on[v] = true;
            for (Size w : succ[v]) {
                if (idx[w] < 0) {
                    visit(w);
                    low[v] = std::min(low[v], low[w]);
                } else if (on[w]) {
                    low[v] = std::min(low[v], idx[w]);
                }
            }
            if (low[v] == idx[v]) {
                std::vector<Size> comp;
                Size w;
                do {
                    w = st.back();
                    st.pop_back();
                    on[w] = false;
                    comp.push_back(w);
                } while (w != v);
                if (comp.size() > 1) comps.push_back(std::move(comp));
            }
        };
        for (Size v = 0; v < n; ++v) if (idx[v] < 0) visit(v);
        std::string msg = "control graph: algebraic loop through blocks ";
        if (!comps.empty()) {
            auto comp = comps.front();
            std::sort(comp.begin(), comp.end());
            for (Size k = 0; k < comp.size(); ++k) {
                msg += (k ? ", '" : "'") + blocks_[comp[k]].name + "'";
            }
        }
        msg += " (each output depends on its input in the same instant). "
               "Insert a DelayBlock in the loop.";
        return msg;
    }

    struct Recording { Index slot; std::vector<Real> history; };

    std::vector<Block> blocks_;
    Block pending_;
    std::vector<Size> order_;
    std::vector<std::string> reorders_;
    bool compiled_ = false;
    bool has_update_ = false;
    std::unordered_map<std::string, Index> slot_of_;
    std::vector<std::string> slot_names_;
    std::function<Index(const std::string&)> node_id_of_;
    ChainContext ctx_;
    std::unordered_map<Index, Real> b_extra_map_;
    std::vector<Recording> recordings_;
    std::vector<Real> recording_times_;
    std::uint64_t n_evals_ = 0;
    std::uint64_t n_block_evals_ = 0;
};


// =============================================================================
// make_chain_switch_fn — a switch_fn(t) from chain signals
// =============================================================================

/// Switch `idx` is ON when its signal is > 0.5. Signals are resolved to
/// slots now (a name no block writes reads 0).
[[nodiscard]] inline std::function<topology::SwitchStateMask(Real)>
make_chain_switch_fn(BlockChain& chain, Size num_switches,
                     const std::vector<std::pair<std::string, Index>>&
                         channel_to_switch) {
    std::vector<std::pair<Index, Index>> pairs;
    for (const auto& [ch, idx] : channel_to_switch) {
        pairs.emplace_back(chain.slot(ch), idx);
    }
    auto* chain_ptr = &chain;
    return [chain_ptr, num_switches, pairs](Real /*t*/) {
        topology::SwitchStateMask mask{num_switches};
        for (const auto& [s, idx] : pairs) {
            if (chain_ptr->signal(s) > Real{0.5}) {
                mask.set(static_cast<Size>(idx), true);
            }
        }
        return mask;
    };
}

}  // namespace pulsim::blockchain
