#pragma once

// =============================================================================
// Pulsim — circuit IR: the immutable `Circuit`
// =============================================================================
//
// add-immutable-circuit-ir. `CircuitBuilder` stays the authoring API;
// `builder.build()` returns a `Circuit`: a handle (one shared pointer)
// to immutable data, cheap to copy and safe to read from any thread.
//
//   CircuitBuilder ──build()──► Circuit ──lowered()──► Graph + DevicePool
//     (mutable)                 (immutable,             (what the engines
//                                shared const data)      consume, as today)
//
// The lowered form is a clone of the builder taken at `build()`, so the
// graph, the pool, every unknown index and the switch order are the
// builder's own, in its insertion order: lowering cannot reorder what it
// never rebuilt. The IR tables (`nodes()`, `devices()`, initial
// conditions, aliases) are read from it once, at construction.
//
// Nothing here is mutable after construction. A run never receives the
// shared lowered builder: `to_builder()` hands out a private clone.

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/ir/fields.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/symbolic_plan.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace pulsim::ir {

struct NodeId   { std::uint32_t v; };
struct DeviceId { std::uint32_t v; };

/// A field value: a literal, or a named parameter (`parameters()[id]`).
/// Group 7 widens the parameter reference to an expression.
using Value = std::variant<Real, ParamId>;

/// A node of the lowered graph; `NodeId{i}` is unknown `i`.
struct Node { std::string name; };

/// One value field of a device: its index in the kind's row of
/// `kKindFields` and its value.
struct Field {
    std::uint16_t index;
    Value value;
};

/// One lowered branch (`DeviceId{i}` is branch `i`): its name (empty
/// for the anonymous branches some composites add), its kind (index in
/// `pwl::kKindCaps`), its pins (from, to, then any control / gate nodes
/// the pool keeps for it; `kGround` = ground) and its value fields.
struct Device {
    std::string name;
    std::uint16_t kind;
    std::vector<Index> pins;
    std::vector<Field> fields;
};

/// The recorded initial condition of one device (`c0=` / `i0=` /
/// `set_initial`), stored per device, not as a write into the state
/// vector.
struct InitialCondition {
    DeviceId device;
    Real value;
};

/// A regularising tie the preflight added to a run's circuit: a
/// resistor `name` from node `node` to ground.
struct Tie {
    std::string name;
    Index node;
    Real resistance;
};

struct Alias {
    std::string name;
    bool node;           // node alias, else a branch alias
    std::string target;  // canonical name
};

class Circuit {
public:
    /// Snapshot `b` (a clone; `b` is not touched).
    [[nodiscard]] static Circuit from_builder(const builder::CircuitBuilder& b) {
        return Circuit{b.clone()};
    }

    [[nodiscard]] std::span<const Node> nodes() const noexcept { return d_->nodes; }
    [[nodiscard]] std::span<const Device> devices() const noexcept { return d_->devices; }
    [[nodiscard]] std::span<const InitialCondition> initial_conditions() const noexcept {
        return d_->ics;
    }
    [[nodiscard]] std::span<const Alias> aliases() const noexcept { return d_->aliases; }
    [[nodiscard]] std::span<const Parameter> parameters() const noexcept {
        return d_->lowered.parameters();
    }

    /// The number a field value stands for.
    [[nodiscard]] Real evaluate(const Value& v) const {
        if (const auto* x = std::get_if<Real>(&v)) return *x;
        return d_->lowered.parameters()[std::get<ParamId>(v).v].value;
    }

    /// Topology, kinds and names (what a symbolic analysis depends on,
    /// and identity).
    [[nodiscard]] std::uint64_t structure_hash() const noexcept { return d_->structure_hash; }
    /// Structure plus every value: fields, couplings, tables, initial
    /// conditions and aliases.
    [[nodiscard]] std::uint64_t value_hash() const noexcept { return d_->value_hash; }

    /// The lowered form: today's `Graph` + `DevicePool`, read-only.
    [[nodiscard]] const builder::CircuitBuilder& lowered() const noexcept { return d_->lowered; }
    [[nodiscard]] const topology::Graph& graph() const noexcept { return d_->lowered.graph(); }
    [[nodiscard]] const pwl::DevicePool& pool() const noexcept { return d_->lowered.pool(); }

    /// A private, mutable copy of the lowered builder: what a run
    /// consumes (the engines take `CircuitBuilder&`).
    [[nodiscard]] builder::CircuitBuilder to_builder() const { return d_->lowered.clone(); }

    /// The circuit with parameter values replaced (`{{"Lbase", 2e-3}}`);
    /// every field bound to a parameter follows it. Bounds of the
    /// parameter and of each driven field are checked; the receiver is
    /// unchanged and the structure (and its symbolic plan) is shared.
    [[nodiscard]] Circuit with_params(
        std::span<const std::pair<std::string_view, Real>> values) const {
        auto b = d_->lowered.clone();
        for (const auto& [name, v] : values) b.set_param(name, v);
        return Circuit{std::move(b), d_->plan};
    }

    /// The circuit with one field set to a literal
    /// (`with_value("R1.R", 2.0)`), equal to rebuilding with that literal.
    [[nodiscard]] Circuit with_value(std::string_view device_dot_field,
                                     Real value) const {
        auto b = d_->lowered.clone();
        b.set_value_("with_value", device_dot_field, value);
        return Circuit{std::move(b), d_->plan};
    }

    /// The circuit with resistors `ties` added (node to ground), each as
    /// the preflight adds one. The receiver is unchanged.
    [[nodiscard]] Circuit with_ties(std::span<const Tie> ties) const {
        auto b = d_->lowered.clone();
        for (const auto& t : ties) {
            if (t.node < 0 || t.node >= b.graph().num_nodes()) {
                throw std::invalid_argument(std::format(
                    "with_ties: tie \"{}\" names node {}, which this "
                    "circuit does not have", t.name, t.node));
            }
            b.add_resistor(t.name, b.graph().node(t.node).name, "gnd",
                           t.resistance);
        }
        return Circuit{std::move(b)};
    }

    /// What the topology preflight does to a run: the circuit it runs
    /// (this one when nothing needed a tie, else a new circuit with the
    /// ties the preflight inserted), the report, and the ties.
    struct Regularized;
    [[nodiscard]] Regularized with_preflight(const pwl::PreflightOptions& opts) const;

    /// The symbolic plan every circuit of this structure shares
    /// (`with_params` / `with_value` derivations). A run's cache adopts
    /// it (`PwlStateSpaceCache::share_symbolic`); it is a cache of
    /// derived data, never part of the circuit's value.
    [[nodiscard]] std::shared_ptr<pwl::SymbolicPlan> symbolic_plan() const noexcept {
        return d_->plan;
    }

    /// Same shared data.
    [[nodiscard]] bool same_data(const Circuit& o) const noexcept { return d_ == o.d_; }

private:
    struct Data {
        builder::CircuitBuilder lowered;
        std::vector<Node> nodes;
        std::vector<Device> devices;
        std::vector<InitialCondition> ics;
        std::vector<Alias> aliases;
        std::uint64_t structure_hash = 0;
        std::uint64_t value_hash = 0;
        std::shared_ptr<pwl::SymbolicPlan> plan;
    };

    explicit Circuit(builder::CircuitBuilder&& b,
                     std::shared_ptr<pwl::SymbolicPlan> plan = nullptr)
        : d_{make_data_(std::move(b), std::move(plan))} {}

    // FNV-1a 64 over 8-byte words.
    struct Hash {
        std::uint64_t h = 0xcbf29ce484222325ULL;
        void word(std::uint64_t x) noexcept {
            for (int k = 0; k < 8; ++k) {
                h ^= (x >> (8 * k)) & 0xFFu;
                h *= 0x100000001b3ULL;
            }
        }
        void real(Real x) noexcept { word(std::bit_cast<std::uint64_t>(x)); }
        void str(std::string_view s) noexcept {
            word(s.size());
            for (const char c : s) {
                h ^= static_cast<unsigned char>(c);
                h *= 0x100000001b3ULL;
            }
        }
        void idx(Index i) noexcept { word(static_cast<std::uint64_t>(i)); }
    };

    /// Control / gate nodes (or the sensed branch) the pool keeps beside
    /// a branch's from / to.
    static void extra_pins_(const pwl::DevicePool& pool, Index b,
                            pwl::DevicePool::StoredKind k,
                            std::vector<Index>& pins) {
        using K = pwl::DevicePool::StoredKind;
        switch (k) {
            case K::MosfetLevel1: pins.push_back(pool.mosfet_level1_gate_node(b)); break;
            case K::IgbtLevel1:   pins.push_back(pool.igbt_level1_gate_node(b)); break;
            case K::VCVS: {
                const auto [p, n] = pool.vcvs_input_nodes(b);
                pins.push_back(p); pins.push_back(n); break;
            }
            case K::IdealTransformer: {
                const auto [p, n] = pool.ideal_transformer_primary_nodes(b);
                pins.push_back(p); pins.push_back(n); break;
            }
            case K::VCCS: case K::Gyrator: {
                const auto [p, n] = pool.control_nodes(b);
                pins.push_back(p); pins.push_back(n); break;
            }
            default: break;
        }
    }

    static std::shared_ptr<const Data> make_data_(
        builder::CircuitBuilder&& b, std::shared_ptr<pwl::SymbolicPlan> plan) {
        auto d = std::make_shared<Data>(Data{std::move(b), {}, {}, {}, {}, 0, 0,
            plan ? std::move(plan) : std::make_shared<pwl::SymbolicPlan>()});
        const auto& g = d->lowered.graph();
        const auto& pool = d->lowered.pool();
        // Fill the graph's lazy id cache now: after construction nothing
        // writes it, so threads sharing the circuit never race on it
        // (with_ties / with_preflight add branches, which clear it).
        (void)g.id();

        Hash s;  // structure
        s.idx(g.num_nodes());
        d->nodes.reserve(static_cast<std::size_t>(g.num_nodes()));
        for (Index i = 0; i < g.num_nodes(); ++i) {
            d->nodes.push_back(Node{g.node(i).name});
            s.str(g.node(i).name);
        }
        Hash v;  // values (structure folded in at the end)
        s.idx(g.num_branches());
        d->devices.reserve(static_cast<std::size_t>(g.num_branches()));
        for (Index bi = 0; bi < g.num_branches(); ++bi) {
            const auto& br = g.branch(bi);
            const auto kind = pool.kind_of(bi);
            Device dev{std::string{g.branch_name(bi)},
                       static_cast<std::uint16_t>(kind), {br.from, br.to}, {}};
            extra_pins_(pool, bi, kind, dev.pins);
            if (kind == pwl::DevicePool::StoredKind::CCCS ||
                kind == pwl::DevicePool::StoredKind::CCVS) {
                s.idx(pool.control_branch(bi));  // sensed branch
            }
            s.str(dev.name);
            s.word(static_cast<std::uint64_t>(br.kind));
            s.word(dev.kind);
            s.word(dev.pins.size());
            for (const Index p : dev.pins) s.idx(p);

            const auto& params = pool.params_of(bi);
            const auto row = fields_of(kind);
            for (std::size_t f = 0; f < row.size(); ++f) {
                const Real x = row[f].get(params);
                dev.fields.push_back(Field{static_cast<std::uint16_t>(f), Value{x}});
                v.real(x);
            }
            // Values outside the scalar rows. Tables are shared immutable
            // objects, identified by address (decision 4 in DECISIONS.md).
            std::visit([&](const auto& p) {
                using P = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<P, models::PwlCurrentSource::Params>) {
                    v.word(p.w.t.size());
                    for (const Real t : p.w.t) v.real(t);
                    for (const Real y : p.w.v) v.real(y);
                    v.word(p.w.periodic ? 1u : 0u);
                } else if constexpr (std::is_same_v<P, models::SaturableInductor::Params>) {
                    v.word(reinterpret_cast<std::uintptr_t>(p.table.get()));
                    v.word(reinterpret_cast<std::uintptr_t>(p.ja.get()));
                } else if constexpr (std::is_same_v<P, models::NonlinearCapacitor::Params>) {
                    v.word(static_cast<std::uint64_t>(p.law));
                    v.word(reinterpret_cast<std::uintptr_t>(p.table.get()));
                }
            }, params);
            d->devices.push_back(std::move(dev));
        }
        for (const auto& c : pool.transformer_couplings()) {
            s.idx(c.primary_branch_id);
            s.idx(c.secondary_branch_id);
            v.real(c.params.L_p);
            v.real(c.params.L_s);
            v.real(c.params.k);
        }
        for (const auto& m : pool.pmsm_mna_machines()) {
            s.idx(m.omega_node);
            s.idx(m.theta_node);
        }
        // Which fields a parameter drives is structure; parameter values
        // are values.
        const auto& params = d->lowered.parameters();
        for (const auto& p : params) {
            v.str(p.name); v.real(p.value); v.str(p.unit);
            v.word(p.lo.has_value()); v.real(p.lo.value_or(0));
            v.word(p.hi.has_value()); v.real(p.hi.value_or(0));
        }
        auto binds = std::vector(d->lowered.param_bindings().begin(),
                                 d->lowered.param_bindings().end());
        std::ranges::sort(binds, {}, [](const auto& x) {
            return std::pair{x.branch, x.field};
        });
        for (const auto& bd : binds) {
            s.idx(bd.branch); s.word(bd.field); s.str(params[bd.param].name);
            for (auto& f : d->devices[static_cast<std::size_t>(bd.branch)].fields) {
                if (f.index == bd.field) f.value = ParamId{bd.param};
            }
        }

        for (const auto& [bid, val] : d->lowered.initial_conditions()) {
            d->ics.push_back(InitialCondition{DeviceId{static_cast<std::uint32_t>(bid)}, val});
        }
        std::ranges::sort(d->ics, {}, [](const InitialCondition& ic) { return ic.device.v; });
        for (const auto& ic : d->ics) { v.word(ic.device.v); v.real(ic.value); }

        using AK = builder::CircuitBuilder::AliasKind;
        for (const auto& [name, t] : d->lowered.aliases()) {
            d->aliases.push_back(Alias{name, t.kind == AK::Node, t.target});
        }
        std::ranges::sort(d->aliases, {}, &Alias::name);
        for (const auto& a : d->aliases) { v.str(a.name); v.word(a.node); v.str(a.target); }

        d->structure_hash = s.h;
        v.word(s.h);
        d->value_hash = v.h;
        return d;
    }

    std::shared_ptr<const Data> d_;
};

struct Circuit::Regularized {
    Circuit circuit;
    pwl::PreflightReport report;
    std::vector<Tie> ties;
};

inline Circuit::Regularized Circuit::with_preflight(
    const pwl::PreflightOptions& opts) const {
    // Pure analysis first: the common, well-posed circuit costs no copy
    // and keeps its identity (and its shared symbolic plan).
    auto found = pwl::analyze_preflight(graph(), pool(), opts);
    if (found.empty() || !opts.auto_regularize) {
        return {*this, std::move(found), {}};
    }
    // The builder's own fixed-point pass, applied to a clone: the run's
    // circuit is exactly what the in-place preflight produced before.
    auto b = d_->lowered.clone();
    const Index before = b.graph().num_branches();
    auto report = b.run_preflight(opts);
    std::vector<Tie> ties;
    for (Index i = before; i < b.graph().num_branches(); ++i) {
        const auto& br = b.graph().branch(i);
        ties.push_back(Tie{std::string{b.graph().branch_name(i)}, br.from,
                           opts.tie_resistance});
    }
    return {Circuit{std::move(b)}, std::move(report), std::move(ties)};
}

}  // namespace pulsim::ir

namespace pulsim::builder {
inline ir::Circuit CircuitBuilder::build() const {
    return ir::Circuit::from_builder(*this);
}
}  // namespace pulsim::builder
