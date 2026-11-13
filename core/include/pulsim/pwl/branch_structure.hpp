#pragma once

// =============================================================================
// add-branch-form-ideal-switch design D3: the per-mode structural pass.
// =============================================================================
//
// With branch-form switches / diodes an ideal open element (g_off = 0) is no
// path and an ideal closed one (R_on = 0) is a 0 V source. For each mode
// (switch mask), on its first visit, union-find passes over the branches
// that connect in that mode find:
//
//   * an island with no path to ground whose boundary is open ideal elements
//     -> UnsupportedError(engine, 'floating_node', <element>, 'Switch'|'Diode')
//     (a boundary with an OFF diode is the gauge of
//     add-complementarity-mode-selection D10, not implemented on the
//     implicit route yet: refused by name with the same feature);
//   * a loop of closed ideal elements and voltage sources
//     -> UnsupportedError(engine, 'ideal_voltage_loop', <element>, kind);
//   * transient only: a loop of capacitors closed by ideal elements and a
//     cut-set of inductors opened by ideal elements: DEPENDENT STATES
//     (design D4), returned to the caller, which re-initialises or refuses.
//
// DC (`dynamic = false`): capacitors open, inductors conducting (a short),
// as in pwl/preflight.hpp; no dependent-state pass. Circuits without an
// ideal branch-form element skip the pass entirely (no cost, bit-identical).

#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/unsupported.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <format>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace pulsim::pwl {

/// One dependent-state structure of a mode (design D4).
struct DependentStructure {
    enum class Kind { CutSet, Loop } kind;
    /// Ideal elements that create it (open ones for a cut-set, the closing
    /// closed one for a loop), by branch id.
    std::vector<Index> elements;
    /// The states it constrains: inductor branch ids (cut-set) or capacitor
    /// branch ids (loop).
    std::vector<Index> states;
    /// Cut-set: +1 when the inductor's current (from -> to) LEAVES the node
    /// set, -1 when it enters; loop: +1 when the capacitor's (from - to)
    /// voltage is traversed forward along the loop, -1 backward. The
    /// constraint is sum(sign_j * x_j) = d (sources), see `rhs_*`.
    std::vector<int> signs;
    /// Loop: voltage sources on the loop (branch id, sign); cut-set:
    /// current sources across the cut (branch id, sign).
    std::vector<std::pair<Index, int>> sources;
};

struct ModeStructure {
    std::vector<DependentStructure> dependent;
    /// ON ideal diodes that close a loop of voltage sources and closed
    /// ideal elements (not refused here: the engine turns them OFF, the
    /// only consistent state for a diode across a stiff loop, and lets the
    /// diode law re-decide; a diode that comes back is refused there).
    std::vector<Index> loop_diodes;
};

class BranchStructureChecker {
public:
    BranchStructureChecker(const topology::Graph& graph,
                           const DevicePool& pool, std::string engine)
        : graph_{graph}, pool_{pool}, engine_{std::move(engine)} {
        for (Index b = 0; b < graph.num_branches(); ++b) {
            if (!pool.is_registered(b) || !pool.is_branch_form(b)) continue;
            const Real g_off =
                pool.kind_of(b) == DevicePool::StoredKind::Diode
                    ? pool.diode_params(b).g_off
                    : pool.switch_g_off(b);
            if (g_off == Real{0} || pool.branch_R_on(b) == Real{0}) {
                active_ = true;
            }
        }
        if (!active_) return;
        // The pass and the D4 record model linear L and C only: a Newton
        // device's state would be missed in a cut-set or loop (review H3).
        using K = DevicePool::StoredKind;
        for (Index b = 0; b < graph.num_branches(); ++b) {
            if (!pool.is_registered(b)) continue;
            const auto k = pool.kind_of(b);
            if (k != K::SaturableInductor && k != K::NonlinearCapacitor &&
                k != K::PmsmMna) {
                continue;
            }
            throw UnsupportedError(
                engine_, "dependent_states", std::string(graph.branch_name(b)),
                std::string(stored_kind_name(k)),
                std::format(
                    "engine '{}': '{}' ({}) carries a nonlinear state, and the "
                    "circuit has an ideal branch-form element (R_on = 0 or "
                    "g_off = 0); the structural pass cannot tell whether a "
                    "mode makes that state dependent. Give every branch-form "
                    "element R_on > 0 and g_off > 0.",
                    engine_, graph.branch_name(b), stored_kind_name(k)));
        }
    }

    /// True when the circuit has an ideal (R_on = 0 or g_off = 0)
    /// branch-form element; otherwise `check` is never needed.
    [[nodiscard]] bool active() const noexcept { return active_; }

    /// Number of structural passes run (one per distinct mode and regime).
    [[nodiscard]] Size checks_run() const noexcept { return checks_; }

    /// The structure of `mask` (cached per mode). Throws UnsupportedError
    /// on a floating island or an ideal voltage loop.
    const ModeStructure& check(const topology::SwitchStateMask& mask,
                               bool dynamic) {
        if (dynamic && last_ != nullptr && last_mask_ == mask) return *last_;
        auto& memo = dynamic ? dyn_ : dc_;
        const std::string key = mask.to_string();
        auto it = memo.find(key);
        if (it == memo.end()) {
            ++checks_;
            it = memo.emplace(key, analyse_(mask, dynamic)).first;
        }
        if (dynamic) { last_mask_ = mask; last_ = &it->second; }
        return it->second;
    }

private:
    enum class Role { Open, Conduct, ZeroImp, Ideal, Cap, Ind, Isrc };

    struct UF {
        std::vector<Index> p;
        explicit UF(Index n) : p(static_cast<std::size_t>(n)) {
            std::iota(p.begin(), p.end(), Index{0});
        }
        Index find(Index a) {
            while (p[a] != a) { p[a] = p[p[a]]; a = p[a]; }
            return a;
        }
        bool unite(Index a, Index b) {
            a = find(a); b = find(b);
            if (a == b) return false;
            p[a] = b;
            return true;
        }
    };

    [[nodiscard]] Index node_(Index n) const noexcept {
        return n < 0 ? graph_.num_nodes() : n;   // ground = last slot
    }

    // The role of branch `b` in this mode. `closed` only for switches.
    Role role_(Index b, bool closed, bool dynamic) const {
        using K = DevicePool::StoredKind;
        const auto k = pool_.kind_of(b);
        switch (k) {
        case K::Resistor: return Role::Conduct;
        case K::Capacitor: return dynamic ? Role::Cap : Role::Open;
        case K::Inductor: return dynamic ? Role::Ind : Role::ZeroImp;
        case K::VoltageSource: case K::PWMVoltageSource:
        case K::SineVoltageSource: case K::PulseVoltageSource:
        case K::VCVS: case K::CCVS: case K::IdealTransformer:
            return Role::ZeroImp;
        case K::CurrentSource: case K::SineCurrentSource:
        case K::PulseCurrentSource: case K::PwlCurrentSource:
            return Role::Isrc;
        case K::VCCS: case K::CCCS: case K::Gyrator:
            return Role::Open;
        case K::Switch: case K::Diode: {
            if (!pool_.is_branch_form(b)) return Role::Conduct;
            const Real g_off = k == K::Diode ? pool_.diode_params(b).g_off
                                             : pool_.switch_g_off(b);
            if (closed) {
                return pool_.branch_R_on(b) == Real{0} ? Role::Ideal
                                                      : Role::Conduct;
            }
            return g_off == Real{0} ? Role::Open : Role::Conduct;
        }
        default:
            return Role::Conduct;   // Newton devices stamp both ends
        }
    }

    [[nodiscard]] std::string kind_name_(Index b) const {
        return std::string(stored_kind_name(pool_.kind_of(b)));
    }

    ModeStructure analyse_(const topology::SwitchStateMask& mask,
                           bool dynamic) {
        const Index n = graph_.num_nodes() + 1;
        const Index gnd = graph_.num_nodes();
        struct E { Index b, a, c; Role r; };
        std::vector<E> edges;
        Size sw = 0;
        for (Index b = 0; b < graph_.num_branches(); ++b) {
            const auto& br = graph_.branch(b);
            const bool is_sw = br.kind == topology::BranchKind::Switch;
            const bool closed = is_sw && mask.get(sw);
            if (is_sw) ++sw;
            if (!pool_.is_registered(b)) continue;
            edges.push_back({b, node_(br.from), node_(br.to),
                             role_(b, closed, dynamic)});
            // The primary is tied through v_p = v_s / n (connectivity, as in preflight; review M4).
            if (pool_.kind_of(b) == DevicePool::StoredKind::IdealTransformer) {
                const auto [pf, pt] = pool_.ideal_transformer_primary_nodes(b);
                edges.push_back({b, node_(pf), node_(pt), Role::Conduct});
            }
        }
        const std::string mode = mask.to_string();
        ModeStructure out;

        // (1) loops of closed ideal elements and zero-impedance branches.
        {
            UF uf(n);
            for (const auto& e : edges) {
                if (e.r == Role::ZeroImp) uf.unite(e.a, e.c);
            }
            // Switches first, so a loop that a diode closes blames the diode.
            const auto is_diode = [&](Index b) {
                return pool_.kind_of(b) == DevicePool::StoredKind::Diode;
            };
            for (int pass = 0; pass < 2; ++pass)
            for (const auto& e : edges) {
                if (e.r != Role::Ideal || is_diode(e.b) != (pass == 1)) continue;
                if (uf.unite(e.a, e.c)) continue;
                if (pass == 1) {
                    out.loop_diodes.push_back(e.b);
                    continue;
                }
                throw UnsupportedError(
                    engine_, "ideal_voltage_loop",
                    std::string(graph_.branch_name(e.b)), kind_name_(e.b),
                    std::format(
                        "engine '{}': in mode {} the closed ideal element '{}' "
                        "(R_on = 0) closes a loop of voltage sources and "
                        "closed ideal switches{}: its current is undefined. "
                        "Give it R_on > 0 (a shoot-through gate pattern is "
                        "the usual cause).",
                        engine_, mode, graph_.branch_name(e.b),
                        dynamic ? "" : " (inductors are shorts at DC)"));
            }
        }
        // (2) islands with no path to ground behind open ideal elements.
        {
            UF uf(n);
            for (const auto& e : edges) {
                if (e.r != Role::Open && e.r != Role::Isrc) uf.unite(e.a, e.c);
            }
            std::unordered_map<Index, std::vector<Index>> boundary;
            for (const auto& e : edges) {
                if (e.r != Role::Open || !pool_.is_branch_form(e.b)) continue;
                for (Index v : {e.a, e.c}) {
                    const Index r = uf.find(v);
                    if (r != uf.find(gnd)) boundary[r].push_back(e.b);
                }
            }
            for (auto& [root, elems] : boundary) {
                std::string nodes;
                for (Index v = 0; v < gnd; ++v) {
                    if (uf.find(v) != root) continue;
                    if (!nodes.empty()) nodes += ", ";
                    nodes += graph_.node(v).name.empty()
                        ? std::format("n{}", v) : graph_.node(v).name;
                }
                std::sort(elems.begin(), elems.end());
                elems.erase(std::unique(elems.begin(), elems.end()),
                            elems.end());
                std::string names;
                bool diode = false;
                for (Index b : elems) {
                    if (!names.empty()) names += ", ";
                    names += graph_.branch_name(b);
                    diode = diode ||
                        pool_.kind_of(b) == DevicePool::StoredKind::Diode;
                }
                const Index first = elems.front();
                throw UnsupportedError(
                    engine_, "floating_node", names,
                    diode ? "Diode" : "Switch",
                    std::format(
                        "engine '{}': in mode {} the node(s) {} have no path "
                        "to ground: every element on their boundary is open "
                        "with g_off = 0 ({}). {}Give one of them g_off > 0, "
                        "or connect a component from the node(s) to ground.",
                        engine_, mode, nodes, names,
                        diode ? "An island bounded by OFF diodes is the "
                                "gauge of the complementarity design (D10), "
                                "not implemented on this engine yet. "
                              : ""));
                (void)first;
            }
        }
        if (!dynamic) return out;

        // (3) capacitor loops closed by ideal elements (dependent states).
        {
            // Spanning forest of zero-impedance branches and capacitors,
            // kept as adjacency so the loop's branches can be listed.
            UF uf(n);
            std::vector<std::vector<std::pair<Index, Index>>> adj(
                static_cast<std::size_t>(n));   // (neighbour, edge index)
            auto add_tree = [&](std::size_t ei) {
                const auto& e = edges[ei];
                adj[e.a].push_back({e.c, static_cast<Index>(ei)});
                adj[e.c].push_back({e.a, static_cast<Index>(ei)});
            };
            for (std::size_t i = 0; i < edges.size(); ++i) {
                const auto& e = edges[i];
                if ((e.r == Role::ZeroImp || e.r == Role::Cap) &&
                    uf.unite(e.a, e.c)) {
                    add_tree(i);
                }
            }
            for (std::size_t i = 0; i < edges.size(); ++i) {
                const auto& e = edges[i];
                if (e.r != Role::Ideal) continue;
                if (uf.unite(e.a, e.c)) { add_tree(i); continue; }
                // Path c -> a in the forest: the loop is e (a -> c) then
                // the path back from c to a.
                std::vector<Index> prev(static_cast<std::size_t>(n), -2);
                std::vector<Index> via(static_cast<std::size_t>(n), -1);
                std::vector<Index> q{e.c};
                prev[e.c] = -1;
                for (std::size_t h = 0; h < q.size(); ++h) {
                    for (auto [w, ei] : adj[q[h]]) {
                        if (prev[w] != -2) continue;
                        prev[w] = q[h]; via[w] = ei; q.push_back(w);
                    }
                }
                DependentStructure d{DependentStructure::Kind::Loop,
                                     {e.b}, {}, {}, {}};
                for (Index v = e.a; v != e.c && prev[v] >= -1; v = prev[v]) {
                    const auto& t = edges[static_cast<std::size_t>(via[v])];
                    // Traversed from prev[v] to v: forward when t.a == prev[v].
                    const int s = t.a == prev[v] ? 1 : -1;
                    if (t.r == Role::Cap) {
                        d.states.push_back(t.b); d.signs.push_back(s);
                    } else if (t.r == Role::Ideal) {
                        d.elements.push_back(t.b);   // review M2
                    } else {
                        d.sources.push_back({t.b, s});
                    }
                }
                if (!d.states.empty()) out.dependent.push_back(std::move(d));
            }
        }
        // (4) inductor cut-sets opened by ideal elements.
        {
            UF uf(n);
            for (const auto& e : edges) {
                if (e.r != Role::Open && e.r != Role::Isrc && e.r != Role::Ind) {
                    uf.unite(e.a, e.c);
                }
            }
            std::map<Index, DependentStructure> cuts;   // ordered: stable records
            for (const auto& e : edges) {
                if (e.r != Role::Open && e.r != Role::Ind && e.r != Role::Isrc) {
                    continue;
                }
                const Index ra = uf.find(e.a), rc = uf.find(e.c);
                if (ra == rc) continue;
                for (auto [r, s] : {std::pair{ra, 1}, std::pair{rc, -1}}) {
                    if (r == uf.find(gnd)) continue;
                    auto& d = cuts.try_emplace(
                        r, DependentStructure{DependentStructure::Kind::CutSet,
                                              {}, {}, {}, {}}).first->second;
                    if (e.r == Role::Ind) {
                        d.states.push_back(e.b); d.signs.push_back(s);
                    } else if (e.r == Role::Isrc) {
                        d.sources.push_back({e.b, s});
                    } else if (pool_.is_branch_form(e.b)) {
                        d.elements.push_back(e.b);
                    }
                }
            }
            for (auto& [r, d] : cuts) {
                if (!d.elements.empty() && !d.states.empty()) {
                    out.dependent.push_back(std::move(d));
                }
            }
        }
        return out;
    }

    const topology::Graph& graph_;
    const DevicePool& pool_;
    std::string engine_;
    bool active_ = false;
    Size checks_ = 0;
    std::unordered_map<std::string, ModeStructure> dyn_, dc_;
    topology::SwitchStateMask last_mask_;
    const ModeStructure* last_ = nullptr;   // node-stable (unordered_map)
};

}  // namespace pulsim::pwl
