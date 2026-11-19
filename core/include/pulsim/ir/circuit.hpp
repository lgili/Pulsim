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
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace pulsim::ir {

struct NodeId   { std::uint32_t v; };

/// What a pin of a device kind connects to (group 9).
enum class PinRole : std::uint8_t {
    Generic,      // any domain, the same as the other pins of its group
    Electrical,   // an electrical node only
    Speed,        // a mechanical node, across quantity rad/s (PMSM omega)
    Angle,        // a mechanical node, across quantity rad (PMSM theta)
};
struct DeviceId { std::uint32_t v; };

/// A field value: a literal, or a named parameter (`parameters()[id]`).
/// Group 7 widens the parameter reference to an expression.
using Value = std::variant<Real, ParamId>;

/// A node of the lowered graph; `NodeId{i}` is unknown `i`. `domain`
/// is declared or implied by a typed pin (electrical otherwise); `unit`
/// is the across unit a probe on it reports.
struct Node {
    std::string name;
    Domain domain = Domain::Electrical;
    std::string_view unit = "V";
};

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
        auto c = b.clone();
        c.cur_scope_ = 0;          // a scope left open is the author's state,
        c.scope_stack_.clear();    // not the circuit's
        (void)resolve_domains_(c); // a cross-domain connection is refused here
        return Circuit{std::move(c)};
    }

    /// The domain of every node (index = unknown), as `nodes()` reports
    /// it, and the across unit of each. Throws `std::invalid_argument`
    /// naming the device, the pin and the node of a connection across
    /// domains.
    struct Domains {
        std::vector<Domain> domain;
        std::vector<std::string_view> unit;
    };
    [[nodiscard]] static Domains resolve_domains_(const builder::CircuitBuilder& b) {
        const auto& g = b.graph();
        const auto& pool = b.pool();
        const auto n = static_cast<std::size_t>(g.num_nodes());
        std::vector<int> dom(n, -1);           // -1: not decided yet
        std::vector<std::string_view> unit(n, std::string_view{});
        std::vector<std::string> why(n);       // what decided it
        for (const auto& [i, d] : b.declared_domains()) {
            dom[static_cast<std::size_t>(i)] = static_cast<int>(d);
            why[static_cast<std::size_t>(i)] = "declared";
        }
        auto node_name = [&](Index i) { return std::string{g.node(i).name}; };
        auto refuse = [&](std::string_view dev, std::string_view pin, Index nd,
                          std::string_view wants, std::string_view has) {
            throw std::invalid_argument(std::format(
                "device \"{}\": pin {} ({}) is connected to node \"{}\", which "
                "is {}", dev, pin, wants, node_name(nd), has));
        };
        // Typed pins first: they imply a domain (and a unit).
        auto typed = [&](std::string_view dev, std::string_view pin, Index nd,
                         Domain d, std::string_view u) {
            if (nd < 0) return;
            const auto k = static_cast<std::size_t>(nd);
            if (dom[k] >= 0 && dom[k] != static_cast<int>(d)) {
                refuse(dev, pin, nd, domain_name(d),
                       std::format("{} ({})", domain_name(static_cast<Domain>(dom[k])), why[k]));
            }
            dom[k] = static_cast<int>(d);
            if (why[k].empty()) why[k] = std::format("pin {} of \"{}\"", pin, dev);
            if (!u.empty()) unit[k] = u;
        };
        // Magnetic nodes (winding / permeance pins, add-permeance-magnetic-
        // domain): MMF in A. A winding joins an electrical and a magnetic
        // pair by design (a gyrator); a semiconductor pin there is refused.
        for (const Index nd : pool.flux_nodes()) {
            typed("magnetic network", "mmf", nd, Domain::Magnetic, "A");
        }
        for (const auto& m : pool.pmsm_mna_machines()) {
            const auto dev = g.branch_name(m.branch_id[0]);
            typed(dev, "omega", m.omega_node, Domain::Mechanical, "rad/s");
            typed(dev, "theta", m.theta_node, Domain::Mechanical, "rad");
        }
        for (const auto& m : pool.induction_machines()) {   // add-mna-native-machines
            typed(g.branch_name(m.branch_id[0]), "omega", m.omega_node,
                  Domain::Mechanical, "rad/s");
        }
        std::vector<Index> pins;
        for (Index bi = 0; bi < g.num_branches(); ++bi) {
            const auto kind = pool.kind_of(bi);
            if (!electrical_only_(kind) || saturable_permeance_(pool, g, bi, kind)) continue;
            const auto& br = g.branch(bi);
            pins.assign({br.from, br.to});
            extra_pins_(pool, bi, kind, pins);
            for (std::size_t p = 0; p < pins.size(); ++p) {
                typed(g.branch_name(bi), pin_name_(kind, p), pins[p], Domain::Electrical, {});
            }
        }
        // Generic pins: the two pins of a pair share a domain, which spreads
        // along generic devices from a declared or typed node (a resistor
        // from a thermal junction to a new case node makes the case thermal).
        // Two decided domains meeting in one pair are refused; a group no
        // declaration or typed pin reaches is electrical.
        std::vector<std::size_t> up(n);
        for (std::size_t k = 0; k < n; ++k) up[k] = k;
        auto root = [&](std::size_t k) {
            while (up[k] != k) k = up[k] = up[up[k]];
            return k;
        };
        for (Index bi = 0; bi < g.num_branches(); ++bi) {
            const auto kind = pool.kind_of(bi);
            if (electrical_only_(kind) && !saturable_permeance_(pool, g, bi, kind)) continue;
            const auto& br = g.branch(bi);
            pins.assign({br.from, br.to});
            extra_pins_(pool, bi, kind, pins);
            for (std::size_t lo = 0; lo + 1 < pins.size(); lo += 2) {   // groups are pin pairs
                const Index a = pins[lo], c = pins[lo + 1];
                if (a < 0 || c < 0) continue;
                const auto ra = root(static_cast<std::size_t>(a));
                const auto rc = root(static_cast<std::size_t>(c));
                if (ra == rc) continue;
                if (dom[ra] >= 0 && dom[rc] >= 0 && dom[ra] != dom[rc]) {
                    refuse(g.branch_name(bi), pin_name_(kind, lo + 1), c,
                           std::format("{}, as its pin {} on \"{}\"",
                                       domain_name(static_cast<Domain>(dom[ra])),
                                       pin_name_(kind, lo), node_name(a)),
                           std::format("{} ({})", domain_name(static_cast<Domain>(dom[rc])),
                                       why[rc]));
                }
                up[rc] = ra;
                if (dom[ra] < 0) { dom[ra] = dom[rc]; why[ra] = why[rc]; }
            }
        }
        for (std::size_t k = 0; k < n; ++k) {
            const int d = dom[root(k)];
            dom[k] = d < 0 ? 0 : d;
        }
        Domains out;
        out.domain.reserve(n);
        out.unit.reserve(n);
        for (std::size_t k = 0; k < n; ++k) {
            const auto d = static_cast<Domain>(dom[k]);
            out.domain.push_back(d);
            out.unit.push_back(!unit[k].empty() ? unit[k] : default_unit(d));
        }
        return out;
    }

    /// The across unit of a domain (V, degC — the thermal helpers write
    /// Celsius —, rad/s; a PMSM angle pin states rad).
    [[nodiscard]] static constexpr std::string_view default_unit(Domain d) noexcept {
        switch (d) {
            case Domain::Electrical: return "V";
            case Domain::Thermal:    return "degC";
            case Domain::Mechanical: return "rad/s";
            case Domain::Magnetic:   return "A";
        }
        return "V";
    }

    [[nodiscard]] std::span<const Node> nodes() const { return tables_().nodes; }
    /// Domain and across unit of node `name` (ground: electrical, V).
    [[nodiscard]] const Node& node_info(std::string_view name) const {
        static const Node kGround{"gnd", Domain::Electrical, "V"};
        const Index i = d_->lowered.node_id_of(name);
        return i < 0 ? kGround : tables_().nodes[static_cast<std::size_t>(i)];
    }
    [[nodiscard]] std::span<const Device> devices() const { return tables_().devices; }
    [[nodiscard]] std::span<const InitialCondition> initial_conditions() const {
        return tables_().ics;
    }
    [[nodiscard]] std::span<const Alias> aliases() const { return tables_().aliases; }
    [[nodiscard]] std::span<const ControlDecl> controls() const noexcept {
        return d_->lowered.controls();
    }
    [[nodiscard]] std::span<const Parameter> parameters() const noexcept {
        return d_->lowered.parameters();
    }

    /// The scope tree (group 8): scope `i > 0` is `scopes()[i - 1]`.
    [[nodiscard]] std::span<const Scope> scopes() const noexcept {
        return d_->lowered.scopes();
    }
    /// The scope a device, node or parameter named `name` belongs to.
    [[nodiscard]] std::uint32_t scope_of(std::string_view name) const {
        return d_->lowered.scope_of(name);
    }

    /// The number a field value stands for.
    [[nodiscard]] Real evaluate(const Value& v) const {
        if (const auto* x = std::get_if<Real>(&v)) return *x;
        return d_->lowered.parameters()[std::get<ParamId>(v).v].value;
    }

    /// Topology, kinds and names (what a symbolic analysis depends on,
    /// and identity).
    [[nodiscard]] std::uint64_t structure_hash() const { return tables_().structure_hash; }
    /// Structure plus every value: fields, couplings, tables, initial
    /// conditions and aliases.
    [[nodiscard]] std::uint64_t value_hash() const { return tables_().value_hash; }

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
        b.set_params(values);
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

    /// The circuit with control declarations `decls` appended (C blocks,
    /// graph-driven sources). Controls never change the matrix structure,
    /// so the symbolic plan is shared; the receiver is unchanged.
    [[nodiscard]] Circuit with_controls(std::span<const ControlDecl> decls) const {
        auto b = d_->lowered.clone();
        for (const auto& c : decls) b.add_control(c);
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

    /// Name the domain of each preflight finding (its subnet's nodes):
    /// a floating thermal network is a thermal-domain finding.
    void label_domains_(pwl::PreflightReport& r) const {
        if (r.findings.empty()) return;
        const auto& ns = nodes();
        for (auto& f : r.findings) {
            if (f.component.empty()) continue;
            const auto d = ns[static_cast<std::size_t>(f.component.front())].domain;
            f.domain = std::string{domain_name(d)};
            if (d != Domain::Electrical) f.detail = std::format("{}-domain: {}", f.domain, f.detail);
        }
    }

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
    /// What is read off the lowered builder: filled on first use (once,
    /// thread-safe), so a derivation (`with_value` / `with_params`, a
    /// sweep point) pays only for its clone until someone asks.
    struct Tables {
        std::vector<Node> nodes;
        std::vector<Device> devices;
        std::vector<InitialCondition> ics;
        std::vector<Alias> aliases;
        std::uint64_t structure_hash = 0;
        std::uint64_t value_hash = 0;
    };
    struct Data {
        Data(builder::CircuitBuilder&& b, std::shared_ptr<pwl::SymbolicPlan> p)
            : lowered{std::move(b)}, plan{std::move(p)} {}
        builder::CircuitBuilder lowered;
        std::shared_ptr<pwl::SymbolicPlan> plan;
        mutable std::once_flag once;
        mutable Tables t;
    };

    [[nodiscard]] const Tables& tables_() const {
        std::call_once(d_->once, [this] {   // a throw leaves t untouched for the retry
            Tables t;
            fill_tables_(*d_, t);
            d_->t = std::move(t);
        });
        return d_->t;
    }

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

    /// Kinds whose every pin is electrical (semiconductors, nonlinear
    /// magnetics and charges, the PMSM's phases); the others are
    /// domain-generic, pins grouped in pairs (from/to, then each control
    /// pair), each pair within one domain.
    [[nodiscard]] static constexpr bool electrical_only_(pwl::DevicePool::StoredKind k) noexcept {
        using K = pwl::DevicePool::StoredKind;
        switch (k) {
            case K::Switch: case K::Diode: case K::NonlinearDiode:
            case K::MosfetLevel1: case K::IgbtLevel1: case K::SaturableInductor:
            case K::NonlinearCapacitor: case K::ShockleyDiode: case K::LauritzenDiode:
            case K::PmsmMna:
                return true;
            default:
                return false;
        }
    }
    /// A saturable permeance is a nonlinear capacitor on magnetic nodes
    /// (add-permeance-magnetic-domain): domain-generic, not electrical.
    [[nodiscard]] static bool saturable_permeance_(const pwl::DevicePool& pool,
                                                   const topology::Graph& g, Index bi,
                                                   pwl::DevicePool::StoredKind k) noexcept {
        const auto& br = g.branch(bi);
        return k == pwl::DevicePool::StoredKind::NonlinearCapacitor
            && (pool.is_flux_node(br.from) || pool.is_flux_node(br.to));
    }
    [[nodiscard]] static constexpr std::string_view pin_name_(pwl::DevicePool::StoredKind k,
                                                              std::size_t p) noexcept {
        using K = pwl::DevicePool::StoredKind;
        if (k == K::MosfetLevel1) return std::array<std::string_view, 3>{"drain", "source", "gate"}[p];
        if (k == K::IgbtLevel1) return std::array<std::string_view, 3>{"collector", "emitter", "gate"}[p];
        if (p < 4) return std::array<std::string_view, 4>{"from", "to", "ctrl_p", "ctrl_n"}[p];
        return "?";
    }

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
        auto d = std::make_shared<Data>(std::move(b),
            plan ? std::move(plan) : std::make_shared<pwl::SymbolicPlan>());
        // Fill the graph's lazy id cache now: after construction nothing
        // writes it, so threads sharing the circuit never race on it
        // (with_ties / with_preflight add branches, which clear it).
        (void)d->lowered.graph().id();
        return d;
    }

    static void fill_tables_(const Data& dd, Tables& out) {
        auto* d = &out;
        const auto& g = dd.lowered.graph();
        const auto& pool = dd.lowered.pool();
        const auto doms = resolve_domains_(dd.lowered);

        Hash s;  // structure
        s.idx(g.num_nodes());
        d->nodes.reserve(static_cast<std::size_t>(g.num_nodes()));
        for (Index i = 0; i < g.num_nodes(); ++i) {
            const auto k = static_cast<std::size_t>(i);
            d->nodes.push_back(Node{g.node(i).name, doms.domain[k], doms.unit[k]});
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
            // Branch form (its own current unknown) is structure, its R_on
            // a value; a conductance-form circuit hashes as before.
            if (pool.is_branch_form(bi)) {
                s.word(0x6272616e6368ULL);   // "branch"
                // R_on = 0 (a 0 V source) and g_off = 0 (open) change the
                // stamp's role: structure. Their values are values.
                const Real R_on = pool.branch_R_on(bi);
                Real g_off = 0;
                std::visit([&](const auto& prm) {
                    if constexpr (requires { prm.g_off; }) g_off = prm.g_off;
                }, pool.params_of(bi));
                s.word((R_on == Real{0} ? 1u : 0u) | (g_off == Real{0} ? 2u : 0u));
                v.real(R_on);
            }

            const auto& params = pool.params_of(bi);
            const auto row = fields_of(kind);
            for (std::size_t f = 0; f < row.size(); ++f) {
                const Real x = row[f].get(params);
                dev.fields.push_back(Field{static_cast<std::uint16_t>(f), Value{x}});
                v.real(x);
            }
            // Values outside the scalar rows, tables by CONTENT (decision
            // 16: an address differs per process and per round trip).
            auto reals = [&](const std::vector<Real>& xs) {
                v.word(xs.size());
                for (const Real x : xs) v.real(x);
            };
            std::visit([&](const auto& p) {
                using P = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<P, pwl::DevicePool::ExternalDeviceParams>) {
                    // the model by name and library path, the values by content
                    for (const char ch : p.inst->model().name()) v.word(static_cast<unsigned char>(ch));
                    for (const char ch : p.inst->model().library_path()) v.word(static_cast<unsigned char>(ch));
                    reals(p.inst->params());
                } else if constexpr (std::is_same_v<P, models::PwlCurrentSource::Params>) {

                    v.word(p.w.t.size());
                    for (const Real t : p.w.t) v.real(t);
                    for (const Real y : p.w.v) v.real(y);
                    v.word(p.w.periodic ? 1u : 0u);
                } else if constexpr (std::is_same_v<P, models::SaturableInductor::Params>) {
                    v.word(p.table ? 1u : 0u);
                    if (p.table) {
                        reals(p.table->i_knots()); reals(p.table->lambda_knots());
                        reals(p.table->slope_knots()); v.real(p.table->L_residual());
                    }
                    v.word(p.ja ? 1u : 0u);
                    if (const auto* j = p.ja.get()) {
                        for (const Real x : {j->N, j->Ae, j->le, j->lg, j->ja.Ms, j->ja.a,
                                             j->ja.alpha, j->ja.c, j->ja.k,
                                             j->substep_frac, j->M0}) v.real(x);
                        v.word(static_cast<std::uint64_t>(j->substeps_min));
                    }
                } else if constexpr (std::is_same_v<P, models::SwitchedDiode::Params>) {
                    // A gated diode's mode is structure, I_H a value;
                    // an ungated diode hashes as before.
                    if (p.gate != models::SwitchedDiode::Gate::None) {
                        s.word(0x6761746564ULL + static_cast<std::uint64_t>(p.gate));
                        v.real(p.I_H);
                    }
                } else if constexpr (std::is_same_v<P, models::NonlinearCapacitor::Params>) {
                    v.word(static_cast<std::uint64_t>(p.law));
                    v.word(p.table ? 1u : 0u);
                    if (p.table) {
                        reals(p.table->v_knots()); reals(p.table->q_knots());
                        reals(p.table->c_knots());
                    }
                } else if constexpr (std::is_same_v<P, models::PmsmMna::Params>) {
                    if (p.harmonics) {
                        for (const auto& t : *p.harmonics) {
                            v.real(t.h); v.real(t.psi); v.real(t.phi);
                        }
                    }
                    if (p.flux_map) {   // only then: existing hashes unchanged
                        s.word(0x666c75786d6170ULL);
                        reals(p.flux_map->kd); reals(p.flux_map->kq);
                        reals(p.flux_map->coef);
                    }
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
            if (m.omega0 != Real{0} || m.theta0 != Real{0}) {
                v.word(static_cast<std::uint64_t>(m.branch_id[0] + 1));
                v.real(m.omega0); v.real(m.theta0);
            }
        }
        for (const auto& m : pool.induction_machines()) {
            s.idx(m.omega_node);
            s.idx(m.rotor_node[0]);
            s.idx(m.rotor_node[1]);
            if (m.omega0 != Real{0}) {
                v.word(static_cast<std::uint64_t>(m.branch_id[0] + 1));
                v.real(m.omega0);
            }
        }
        // Which fields a parameter drives is structure; parameter values
        // are values.
        // Declared domains are structure, hashed only when one is not
        // electrical (an untagged circuit keeps its hashes); implied ones
        // follow from the kinds already hashed.
        {
            std::vector<std::pair<Index, ir::Domain>> tags;
            for (const auto& [i, dm] : dd.lowered.declared_domains()) {
                if (dm != Domain::Electrical) tags.emplace_back(i, dm);
            }
            if (!tags.empty()) {
                std::ranges::sort(tags);
                s.word(0x646f6d61696eULL);   // "domain", then the count
                s.word(tags.size());
                for (const auto& [i, dm] : tags) { s.idx(i); s.word(static_cast<std::uint64_t>(dm)); }
            }
        }
        // The scope tree is structure; hashed only when there is one, so
        // a circuit without scopes keeps its hashes.
        if (const auto sc = dd.lowered.scopes(); !sc.empty()) {
            s.word(0x73636f706573ULL);  // "scopes", then the count
            s.word(sc.size());
            for (const auto& x : sc) { s.str(x.path); s.str(x.kind); }
        }
        const auto& params = dd.lowered.parameters();
        for (const auto& p : params) {
            v.str(p.name); v.real(p.value); v.str(p.unit);
            if (p.scope != 0) s.word(p.scope);   // where its expression looks names up
            if (!p.expr.empty()) { s.word(0x65787072ULL); s.str(p.name); s.str(p.expr); }
            v.word(p.lo.has_value()); v.real(p.lo.value_or(0));
            v.word(p.hi.has_value()); v.real(p.hi.value_or(0));
        }
        auto binds = std::vector(dd.lowered.param_bindings().begin(),
                                 dd.lowered.param_bindings().end());
        std::ranges::sort(binds, {}, [](const auto& x) {
            return std::pair{x.branch, x.field};
        });
        for (const auto& bd : binds) {
            s.idx(bd.branch); s.word(bd.field); s.str(params[bd.param].name);
            for (auto& f : d->devices[static_cast<std::size_t>(bd.branch)].fields) {
                if (f.index == bd.field) f.value = ParamId{bd.param};
            }
        }

        for (const auto& [bid, val] : dd.lowered.initial_conditions()) {
            d->ics.push_back(InitialCondition{DeviceId{static_cast<std::uint32_t>(bid)}, val});
        }
        std::ranges::sort(d->ics, {}, [](const InitialCondition& ic) { return ic.device.v; });
        for (const auto& ic : d->ics) { v.word(ic.device.v); v.real(ic.value); }
        // Hashed only when present: a circuit without controls keeps
        // its hashes.
        if (const auto& cs = dd.lowered.controls(); !cs.empty()) {
            s.word(0x636f6e74726f6cULL);  // "control", then the count
            s.word(cs.size());
            for (const auto& c : cs) {
                s.str(c.name); s.str(c.kind); v.word(c.content_hash);
            }
        }

        // add-permeance-magnetic-domain: hashed only when present, so a
        // circuit without magnetic nodes keeps its hashes.
        if (const auto& fn = dd.lowered.pool().flux_nodes(); !fn.empty()) {
            s.word(0x6d61676e65746963ULL);  // "magnetic", then the nodes
            s.word(fn.size());
            for (const Index n : fn) s.idx(n);
        }
        if (const auto& mg = dd.lowered.pool().magnetic_geometry(); !mg.empty()) {
            s.word(mg.size());
            for (const auto& g : mg) { s.idx(g.branch); v.real(g.area); v.real(g.length); }
        }

        // add-in-kernel-event-losses: the loss ports, hashed only when
        // present (a circuit without one keeps its hashes). Names and the
        // heat path are structure; temperatures and tables are values.
        if (const auto& dl = dd.lowered.pool().device_losses(); !dl.empty()) {
            s.word(0x6c6f7373706f7274ULL);  // "lossport", then the count
            s.word(dl.size());
            auto vec = [&v](const std::vector<double>& a) {
                v.word(a.size());
                for (const double x : a) v.real(x);
            };
            for (const auto& x : dl) {
                s.str(x.device); s.str(x.junction); s.str(x.heat_node);
                s.word(static_cast<std::uint64_t>(x.conduction));
                v.real(x.tj);
                for (const auto* e : {x.e_on.get(), x.e_off.get()}) {
                    s.word(e != nullptr);
                    if (e != nullptr) {
                        vec(e->v_axis()); vec(e->i_axis()); vec(e->tj_axis()); vec(e->energy());
                    }
                }
                s.word(x.v_on != nullptr);
                if (x.v_on) { vec(x.v_on->i_axis()); vec(x.v_on->tj_axis()); vec(x.v_on->v_on()); }
            }
        }

        using AK = builder::CircuitBuilder::AliasKind;
        for (const auto& [name, t] : dd.lowered.aliases()) {
            d->aliases.push_back(Alias{name, t.kind == AK::Node, t.target});
        }
        std::ranges::sort(d->aliases, {}, &Alias::name);
        for (const auto& a : d->aliases) { v.str(a.name); v.word(a.node); v.str(a.target); }

        d->structure_hash = s.h;
        v.word(s.h);
        d->value_hash = v.h;
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
    label_domains_(found);
    if (found.empty() || !opts.auto_regularize) {
        return {*this, std::move(found), {}};
    }
    // The builder's own fixed-point pass, applied to a clone: the run's
    // circuit is exactly what the in-place preflight produced before.
    auto b = d_->lowered.clone();
    const Index before = b.graph().num_branches();
    auto report = b.run_preflight(opts);
    label_domains_(report);
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
    pwl::require_magnetic_reference(graph_, pool_);
    return ir::Circuit::from_builder(*this);
}
}  // namespace pulsim::builder
