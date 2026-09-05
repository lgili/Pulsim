#pragma once

// =============================================================================
// Pulsim — Layer 4: topology preflight + auto-regularization
// =============================================================================
//
// v2.0 Phase 2 (B.1), closing audit finding
// `no-topology-preflight-or-auto-shunt` (CRITICAL).
//
// THE PROBLEM. The single most common "it should just work" failure
// in a circuit simulator is a node the user never meant to leave
// unreferenced: an isolated transformer secondary, a divider tap
// hanging off nothing but a capacitor, a sub-circuit fed only by
// current sources. Modified nodal analysis has no equation for such
// a node — its matrix column is empty — so the factorization is
// singular. Until now Pulsim's answer was to fail (with a named
// node, since Phase 1) and to document the fix in prose:
//
//     b.add_resistor("R_iso", "sec_gnd", "gnd", 1e9)
//
// That is the right fix. It should not be the user's job to know it.
//
// THE PASS. Before the cache is built, walk the topology twice:
//
//   1. GALVANIC reachability — union every branch, regardless of
//      kind. A component that never reaches ground is an isolated
//      subnet with no voltage reference at all.
//   2. DC reachability — union only the branches that conduct at
//      DC. Capacitors are open at DC and current sources contribute
//      no conductance, so a node behind only those has no DC path
//      to ground even though it is galvanically connected.
//
// Each offending component gets ONE large tie to ground. Large is
// the whole point: a 1 GΩ resistor gives MNA a reference while
// drawing nanoamps, so the circuit's behaviour is untouched. (A
// small tie would be a galvanic BOND — it would silently rewire the
// user's circuit, which is worse than the failure it replaces.)
//
// AUDITABILITY. Nothing is inserted silently. Every action lands in
// a `PreflightReport` naming the node and the value, which the
// Python layer surfaces once as a warning and attaches to the
// result. A user who wants none of this can turn it off and get the
// old named error instead.
//
// Switches are treated as conducting in BOTH states on purpose: the
// assembler always stamps a conductance (`g_on` or `g_off`), so a
// node behind an open switch is not structurally floating — it just
// pivots small. That case belongs to gmin, not here.
//
// INDUCTORS conduct for this pass, matching `dc_assemble`, which
// stamps them as shorts. Note there is a THIRD floating class this
// pass deliberately does NOT auto-fix: the legacy static build
// (`dt = 0`) omits inductor companion stamps entirely, so a node
// reachable only through an inductor is structurally floating
// THERE. Inserting a resistor would be the wrong fix — the right
// one is to build with `dt > 0`, which is exactly what the existing
// named error says. Papering over it with a tie would hide a real
// modelling mistake behind a resistor the user never asked for.

// LAYERING: this lives in Layer 4 (pwl), not Layer 1 (topology),
// because deciding whether a branch conducts requires the DEVICE
// kind — a PassiveLinear branch is a resistor, an inductor or a
// capacitor, and only the pool knows which. Layer 1 is compile-time
// restricted to Layer 0, so a topology/ home would have been a
// layering violation.

#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

#include <format>
#include <string>
#include <vector>

namespace pulsim::pwl {

/// Why a node needed a reference tie.
enum class PreflightIssue {
    IsolatedSubnet,   ///< no path to ground through ANY branch
    NoDcPathToGround, ///< connected, but only through capacitors /
                      ///< current sources — open at DC
};

/// One thing the preflight pass found, and what it did about it.
struct PreflightFinding {
    PreflightIssue issue;
    Index anchor_node = kInvalidIndex;   ///< node that received the tie
    std::vector<Index> component;        ///< every node in the subnet
    Real inserted_resistance = Real{0};  ///< 0 when reporting only
    std::string detail;                  ///< human-facing sentence

    [[nodiscard]] bool was_fixed() const noexcept {
        return inserted_resistance > Real{0};
    }
};

/// What the pass found and did. Empty `findings` means the topology
/// was already well-posed — the overwhelmingly common case, and the
/// one that must cost nothing.
struct PreflightReport {
    std::vector<PreflightFinding> findings;

    [[nodiscard]] bool empty() const noexcept {
        return findings.empty();
    }
    [[nodiscard]] Size num_fixed() const noexcept {
        Size n = 0;
        for (const auto& f : findings) if (f.was_fixed()) ++n;
        return n;
    }

    /// One-paragraph summary suitable for a warning. Empty when
    /// there is nothing to say.
    [[nodiscard]] std::string summary() const {
        if (findings.empty()) return {};
        std::string out = std::format(
            "Pulsim preflight: {} topology issue(s) found, {} "
            "auto-regularized.", findings.size(), num_fixed());
        for (const auto& f : findings) {
            out += "\n  * " + f.detail;
        }
        return out;
    }
};

/// Options for the pass. Defaults are deliberately conservative:
/// a tie big enough to be electrically invisible.
struct PreflightOptions {
    /// Insert the ties, rather than only reporting them.
    bool auto_regularize = true;

    /// Resistance of an inserted reference tie [Ω]. 1 GΩ draws
    /// 12 nA at 12 V — below any meaningful current in power
    /// electronics, while giving MNA the rank it needs.
    Real tie_resistance = Real{1e9};

    /// Prefix for generated device names, so an inserted element is
    /// obvious in any downstream listing.
    std::string name_prefix = "R_auto_iso_";
};

namespace detail {

/// Path-compressed union-find over node ids, with ground (kGround)
/// represented by the extra slot `num_nodes`.
class GroundUnionFind {
public:
    explicit GroundUnionFind(Index num_nodes)
        : ground_slot_{num_nodes},
          parent_(static_cast<Size>(num_nodes) + 1) {
        for (Size i = 0; i < parent_.size(); ++i) {
            parent_[i] = static_cast<Index>(i);
        }
    }

    [[nodiscard]] Index slot_of(Index node) const noexcept {
        return node == kGround ? ground_slot_ : node;
    }

    Index find(Index x) {
        while (parent_[static_cast<Size>(x)] != x) {
            parent_[static_cast<Size>(x)] =
                parent_[static_cast<Size>(parent_[static_cast<Size>(x)])];
            x = parent_[static_cast<Size>(x)];
        }
        return x;
    }

    void unite(Index a, Index b) {
        const Index ra = find(a);
        const Index rb = find(b);
        if (ra != rb) parent_[static_cast<Size>(ra)] = rb;
    }

    /// True when `node` shares a component with ground.
    bool reaches_ground(Index node) {
        return find(slot_of(node)) == find(ground_slot_);
    }

    [[nodiscard]] Index ground_slot() const noexcept { return ground_slot_; }

private:
    Index ground_slot_;
    std::vector<Index> parent_;
};

/// Does this branch carry current at DC?
///
/// Three kinds cannot give a node a DC reference:
///   * capacitors — `dc_assemble` skips them ("open circuit at DC");
///   * ideal current sources — matrix-free, they only write the RHS;
///   * NONLINEAR branches — `dc_assemble` skips these too, stamping
///     them as OPEN CIRCUITS (its own header says so). An earlier
///     version of this predicate claimed they "stamp their
///     linearization" and returned true, which made the pass miss
///     the very case it exists to catch: a node touching only a
///     nonlinear device and a capacitor sails through preflight and
///     then fails at DC. (The transient path stamps only a 1e-12
///     diagonal, and only for a saturable inductor's branch
///     variable — not a node reference either.)
/// Everything else conducts: resistors and inductors, sources
/// (which impose a branch relation), and switches, which always
/// stamp `g_on` or `g_off`.
[[nodiscard]] inline bool conducts_at_dc(const DevicePool& pool,
                                          const topology::Branch& branch) {
    using SK = DevicePool::StoredKind;
    // A branch with no pool entry cannot be classified. `kind_of`
    // THROWS in that case, and a preflight pass must never turn a
    // diagnostic into a second exception — so treat it as
    // conducting, the conservative answer: we may miss a floating
    // node, but we never invent one.
    if (!pool.is_registered(branch.id)) {
        return true;
    }
    switch (branch.kind) {
        case topology::BranchKind::PassiveLinear:
            return pool.kind_of(branch.id) != SK::Capacitor;
        case topology::BranchKind::Source:
            return pool.kind_of(branch.id) != SK::CurrentSource;
        case topology::BranchKind::Nonlinear:
            // A saturable inductor is a short at DC (v = dλ/dt = 0),
            // exactly like a linear one; the other Newton devices
            // are stamped open by the DC assembly and stay so here.
            return pool.kind_of(branch.id) == SK::SaturableInductor;
        case topology::BranchKind::Switch:
            return true;
    }
    return true;
}

/// Group nodes into components under `edge_ok`, then return one
/// representative node per component that does NOT reach ground.
/// The representative is the lowest node id, so the choice is
/// deterministic and the report is reproducible.
template <class EdgePredicate>
[[nodiscard]] inline std::vector<std::vector<Index>>
components_without_ground(const topology::Graph& graph, EdgePredicate edge_ok) {
    GroundUnionFind uf{graph.num_nodes()};
    for (Index b = 0; b < graph.num_branches(); ++b) {
        const auto& br = graph.branch(b);
        if (!edge_ok(br)) continue;
        uf.unite(uf.slot_of(br.from), uf.slot_of(br.to));
    }

    // Bucket nodes by representative, skipping anything grounded.
    std::vector<std::vector<Index>> out;
    std::vector<Index> rep_of_bucket;
    for (Index n = 0; n < graph.num_nodes(); ++n) {
        if (uf.reaches_ground(n)) continue;
        const Index r = uf.find(n);
        Size bucket = out.size();
        for (Size i = 0; i < rep_of_bucket.size(); ++i) {
            if (rep_of_bucket[i] == r) { bucket = i; break; }
        }
        if (bucket == out.size()) {
            out.emplace_back();
            rep_of_bucket.push_back(r);
        }
        out[bucket].push_back(n);
    }
    return out;
}

}  // namespace detail

/// Describe a node for a message: its name when it has one, else its
/// id. (Ground never appears here — it is the reference.)
[[nodiscard]] inline std::string node_label(const topology::Graph& graph, Index n) {
    if (n < 0 || n >= graph.num_nodes()) {
        return std::format("node #{}", n);
    }
    const auto& name = graph.node(n).name;
    return name.empty() ? std::format("node #{}", n)
                        : std::format("node '{}'", name);
}

/// Analyse `graph` / `pool` and report what would need regularizing.
/// Pure — inserts nothing. `analyze_preflight` is what a caller uses
/// to inspect a circuit; `CircuitBuilder::run_preflight` applies it.
[[nodiscard]] inline PreflightReport analyze_preflight(
    const topology::Graph& graph,
    const DevicePool& pool,
    const PreflightOptions& opts = {}) {
    PreflightReport report;

    // ---- 1. Galvanically isolated subnets ------------------------
    const auto isolated = detail::components_without_ground(
        graph, [](const topology::Branch&) { return true; });

    for (const auto& comp : isolated) {
        if (comp.empty()) continue;
        PreflightFinding f;
        f.issue     = PreflightIssue::IsolatedSubnet;
        f.component = comp;
        f.anchor_node = comp.front();
        const std::string scope =
            comp.size() == 1
                ? node_label(graph, f.anchor_node)
                : std::format("the {}-node subnet containing {}",
                               comp.size(),
                               node_label(graph, f.anchor_node));
        f.detail = std::format(
            "{} has no connection to ground through any device — a "
            "galvanically isolated subnet (a transformer secondary, "
            "say) has no voltage reference, so its MNA equations are "
            "singular. Tying one of its nodes to ground through a "
            "high-value resistor ({:g} Ω) supplies the reference "
            "without bonding the nets.",
            scope, opts.tie_resistance);
        report.findings.push_back(std::move(f));
    }

    // ---- 2. Connected, but with no DC path to ground -------------
    //
    // Note this pass runs on the graph AS GIVEN. A caller that
    // APPLIES the galvanic ties must re-analyze afterwards rather
    // than filtering these findings against the galvanic ones:
    // a galvanic finding covers a whole island but earns it only
    // ONE tie, so DC-floating sub-blocks INSIDE that island are
    // still floating after it lands. Suppressing them by component
    // membership (the first version of this code) left them
    // singular while the report claimed they were fixed — see
    // `CircuitBuilder::run_preflight`, which iterates to a fixed
    // point instead.
    const auto dc_floating = detail::components_without_ground(
        graph, [&pool](const topology::Branch& br) {
            return detail::conducts_at_dc(pool, br);
        });

    // A component that is ALSO galvanically isolated is already
    // described by finding 1 at the same anchor; reporting the same
    // anchor twice would be noise. Compare ANCHORS, not membership.
    auto anchor_taken = [&](Index node) {
        for (const auto& f : report.findings) {
            if (f.anchor_node == node) return true;
        }
        return false;
    };

    for (const auto& comp : dc_floating) {
        if (comp.empty() || anchor_taken(comp.front())) continue;
        PreflightFinding f;
        f.issue     = PreflightIssue::NoDcPathToGround;
        f.component = comp;
        f.anchor_node = comp.front();
        f.detail = std::format(
            "{} has no DC path to ground — it is reachable only "
            "through capacitors (open at DC), current sources (no "
            "conductance) or nonlinear devices (which the DC "
            "assembly stamps as open circuits), so the DC operating "
            "point and any static (dt = 0) build are singular there. "
            "A high-value bleeder to ground ({:g} Ω) fixes it without "
            "loading the node.",
            node_label(graph, f.anchor_node), opts.tie_resistance);
        report.findings.push_back(std::move(f));
    }

    return report;
}

}  // namespace pulsim::pwl
