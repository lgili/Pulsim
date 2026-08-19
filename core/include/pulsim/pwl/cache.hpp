#pragma once

// =============================================================================
// Pulsim — Layer 4: PwlStateSpaceCache (the PLECS-killer)
// =============================================================================
//
// `pulsim-v2-pwl-state-space-cache` Phase 4.
//
// THE architectural pivot. For each switch combination Layer 1
// enumerates, pre-build the per-segment MNA matrix (Layer 3
// stampers + Layer 2 device models) and pre-factorise via Layer 0's
// DirectSolver. Per-step hot loop: hash-map lookup + triangular
// solve. NO assemble, NO factorize, NO Newton iteration per step.
//
// Expected speedup vs v1: 10-50× on PE workloads. This is THE
// architectural reason v2 will compete with PLECS.
//
// V0 scope: static-only circuits (Resistor + VoltageSource +
// Switch). Capacitor/Inductor (trapezoidal companion + history) and
// nonlinear devices (per-segment Newton) are V1+ extensions.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/dictionary.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/segment.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"
#include "pulsim/topology/enumerator.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <expected>
#include <format>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsim::pwl {

/// Telemetry counters for the rank-1 cache update path
/// (openspec/changes/add-pwl-rank1-update, extended by
/// openspec/changes/add-generalised-path-refactor Part A).
/// Monotonically incremented from inside
/// `PwlStateSpaceCache::solve_rank1`. Use
/// `PwlStateSpaceCache::metrics()` to snapshot.
///
/// * `rank1_hits`            — fast-path hits where the mask either did not
///                             change (b_constant refresh + triangular solve
///                             only) OR changed by exactly one bit AND the
///                             single-bit `partial_refactor` path engaged.
///                             v1.3.0 semantics preserved.
/// * `multi_bit_rank1_hits`  — (v1.4.0+) calls where the mask changed by ≥ 2
///                             bits AND the multi-bit path-union
///                             `partial_refactor` path engaged successfully.
///                             Previously these landed in `full_refactor_hits`
///                             (the v1.3.0 unconditional multi-bit fallback);
///                             v1.4.0 routes them through path-based update
///                             when the union path is short enough.
/// * `full_refactor_hits`    — fresh `factorize()` because the path-based
///                             update would be no cheaper than a full
///                             factorisation (`path_length / n >
///                             MAX_PATH_LENGTH_RATIO`), or the backend does
///                             not implement `partial_refactor`, or the
///                             first-encounter initial factor.
/// * `fallbacks`             — `partial_refactor` was attempted and returned
///                             `false` (numerical singularity / pivot fault
///                             in the path). Distinct from `full_refactor_hits`
///                             which is the "didn't even try" bucket.
///
/// Invariant per
/// `openspec/changes/add-generalised-path-refactor/specs/pwl-rank1-update`:
///     `rank1_hits + multi_bit_rank1_hits + full_refactor_hits + fallbacks
///      == total solve_rank1 calls`.
struct CacheMetrics {
    std::uint64_t rank1_hits            = 0;
    std::uint64_t multi_bit_rank1_hits  = 0;
    std::uint64_t full_refactor_hits    = 0;
    std::uint64_t fallbacks             = 0;

    // ---- v2.0 Phase 1 — LRU segment cache + event-dt solver ----
    // Invariant: `event_hits + event_refactors + event_builds ==
    // total solve_at calls with dt != primary dt`.
    std::uint64_t segment_evictions     = 0;  // lazy LRU evictions
    std::uint64_t event_hits            = 0;  // solve_at: same (mask, dt)
    std::uint64_t event_refactors       = 0;  // solve_at: dt changed,
                                              // in-place numeric refactor
    std::uint64_t event_builds          = 0;  // solve_at: fresh entry
};

/// Result of `PwlStateSpaceCache::refactor_parametric` (v1.4.0,
/// `openspec/changes/add-generalised-path-refactor` Part B).
///
/// Per the spec invariant:
///   `path_refactor_hits + fallback_hits == masks_processed`.
///
/// * `masks_processed`     — total number of cached masks the call
///                           visited. For `Mode::AllActive` this is
///                           `segments_.size()`; for
///                           `Mode::CurrentOnly` it is at most 1 (the
///                           rank-1 sliding solver's current mask
///                           if rank-1 has been used).
/// * `path_refactor_hits`  — segments where `partial_refactor` was
///                           called AND returned `true`. The headline
///                           speedup bucket.
/// * `fallback_hits`       — segments where the path was either too
///                           long (`affected_cols/n > MAX_PATH_LENGTH_RATIO`),
///                           the backend doesn't support
///                           `partial_refactor`, or `partial_refactor`
///                           was attempted and returned `false`
///                           (pivot fault). In every case the
///                           segment was rebuilt via fresh
///                           `factorize(new_J)`.
/// * `wall_time_us`        — total wall-clock spent in the call
///                           (microseconds). Includes re-stamping +
///                           path-walks + factorizations.
struct ParametricRefactorResult {
    std::size_t masks_processed    = 0;
    std::size_t path_refactor_hits = 0;
    std::size_t fallback_hits      = 0;
    double      wall_time_us       = 0.0;
};

/// Selection mode for `PwlStateSpaceCache::refactor_parametric`.
///
/// `AllActive` (default) — process every cached `(mask, segment)`
/// entry. The right choice for sweep / Monte Carlo workloads where
/// downstream `simulate()` calls will visit multiple masks across
/// the transient.
///
/// `CurrentOnly` — process just the most-recent rank-1 sliding
/// solver's mask (if rank-1 has been used). Useful for hot-loop
/// parameter perturbations (e.g. a closed-loop control study that
/// trims one gain live) where the user knows only the active mask
/// matters. A no-op (returns `masks_processed == 0`) when the cache
/// has not been used via `solve_rank1`.
enum class ParametricRefactorMode {
    AllActive,
    CurrentOnly,
};

/// Single parameter update: which device (by `branch_id`, obtained
/// via `CircuitBuilder::branch_id_of(name)`) and its new value.
/// The pool's mutator dispatch (`update_resistor_R`,
/// `update_inductor_L`, `update_capacitor_C`,
/// `update_voltage_source_V`) is selected from the stored device
/// kind on the corresponding branch.
struct ParametricUpdate {
    Index branch_id;
    Real  new_value;
};

/// Structured error returned by the non-throwing `try_*` cache API
/// (Layer 4 V4 ergonomics, C++23). Lets tooling and the Python
/// frontend report singular-matrix failures with the offending mask
/// and dt attached, rather than parsing a string out of `what()`.
///
/// The throwing API (`build`, `lookup`) is unchanged and still
/// the canonical entry point — `try_*` is purely additive for
/// callers that prefer expected-style flow.
struct CacheError {
    enum class Kind {
        StructurallySingular,
        NumericallySingular,
        MaskNotBuilt,
    };
    Kind                       kind;
    topology::SwitchStateMask  mask;
    Real                       dt = Real{0};   // 0 for static-only

    [[nodiscard]] std::string what() const {
        const char* k =
            kind == Kind::StructurallySingular ? "structurally singular"
          : kind == Kind::NumericallySingular  ? "numerically singular"
                                                : "no segment built";
        return std::format(
            "PwlStateSpaceCache: {} for mask {} (dt={})",
            k, mask.to_string(), dt);
    }
};

class PwlStateSpaceCache {
public:
    /// v1.4.0: pool stored as a non-const reference so
    /// `refactor_parametric` can mutate it in place. All read paths
    /// remain unchanged; existing callers that pass a non-const
    /// `DevicePool` keep compiling. A `const DevicePool&` argument
    /// no longer binds — callers that need read-only safety should
    /// `std::cref(pool)` upstream, but in practice every Layer 1-9
    /// consumer passes a freshly built non-const pool.
    PwlStateSpaceCache(const topology::Graph& graph,
                        DevicePool& pool) noexcept
        : graph_{graph}, pool_{pool} {}

    /// Build all 2^N segments and pre-factorize each (V1 dt-aware).
    ///
    /// `dt > 0` enables dynamic-device stamping (Capacitor /
    /// Inductor trap companion). `dt = 0` (or the no-arg overload)
    /// skips dynamic devices and produces a V0-identical static
    /// build.
    ///
    /// Calling build(dt) twice with different dt CLEARs and
    /// rebuilds every factor (the matrix is dt-dependent).
    ///
    /// Time:   O(2^N · (assemble + analyze + factorize))
    /// Memory: O(2^N · (nnz(J) + nnz(L) + nnz(U)))
    void build(Real dt) {
        dt_ = dt;
        lazy_mode_ = false;
        segments_.clear();
        event_entries_.clear();
        const Size num_switches = graph_.num_switches();
        for (auto mask :
             topology::enumerate_switch_states(num_switches)) {
            build_one_segment(mask);
        }
    }

    /// V0 backwards-compat overload — static-only build (dt = 0).
    /// Caps and Inductors are silently skipped.
    void build() { build(Real{0}); }

    /// Lazy build (Layer 4 V6). Stores dt but defers segment
    /// factorisation. Each `solve(mask, ...)` call builds the
    /// requested mask's factor the first time it's seen, then
    /// caches it. Useful when only a few of the 2^N possible
    /// switch states are visited (typical for fixed-duty PWM
    /// converters).
    void build_lazy(Real dt) {
        dt_ = dt;
        lazy_mode_ = true;
        segments_.clear();
        event_entries_.clear();
    }

    /// Currently-built dt. Returns 0 for static-only builds.
    /// Layer 5's run_transient checks this against opts.dt and
    /// throws on mismatch.
    [[nodiscard]] Real dt() const noexcept { return dt_; }

    /// O(1) segment lookup. In eager mode, throws if the mask
    /// wasn't pre-built. In lazy mode (`build_lazy(dt)`), builds
    /// the segment on demand and caches it before returning.
    [[nodiscard]] const PwlSegment& lookup(
        const topology::SwitchStateMask& mask) const {
        auto r = try_lookup(mask);
        if (!r) {
            // Exception type carries meaning across the pybind
            // boundary (adversarial-review finding PY-2): a
            // SINGULAR matrix is a circuit problem and must stay a
            // RuntimeError — with lazy building it now surfaces on
            // first VISIT of the mask, mid-run, and out_of_range
            // would arrive in Python as a baffling IndexError.
            // out_of_range is reserved for the eager-mode
            // "mask not pre-built" bookkeeping case.
            if (r.error().kind == CacheError::Kind::MaskNotBuilt) {
                throw std::out_of_range(r.error().what());
            }
            throw std::runtime_error(r.error().what());
        }
        return **r;
    }

    /// Non-throwing companion to `lookup` (Layer 4 V4 ergonomics).
    /// Returns a pointer to the segment on success, or a
    /// `CacheError` carrying the offending mask + dt on failure.
    ///
    /// Lazy mode triggers a build on first access — if THAT
    /// build hits a singular matrix, the returned error reports
    /// the singularity kind directly.
    [[nodiscard]] std::expected<const PwlSegment*, CacheError>
    try_lookup(const topology::SwitchStateMask& mask) const {
        const auto it = segments_.find(mask);
        if (it != segments_.end()) {
            // LRU touch (v2.0 Phase 1) — mutable tick, no
            // structural mutation: outstanding references stay
            // valid across cache HITS.
            it->second.lru_tick = ++lru_clock_;
            return &it->second;
        }
        if (lazy_mode_) {
            auto built =
                const_cast<PwlStateSpaceCache*>(this)
                    ->try_build_one_segment(mask);
            if (!built) {
                return std::unexpected(built.error());
            }
            return &segments_.find(mask)->second;
        }
        return std::unexpected(CacheError{
            .kind = CacheError::Kind::MaskNotBuilt,
            .mask = mask,
            .dt   = dt_,
        });
    }

    /// HOT LOOP entry point. Look up the segment for `mask` and
    /// solve `J · x = -(b_constant + b_extra)` via the cached
    /// factor. `b_extra` lets Layer 5 inject time-varying source
    /// values or history terms in V1; in V0 it's typically zero.
    ///
    /// Newton's convention is solve(J, -f, dx). For Layer 4's
    /// static-only V0 path the residual at convergence is
    /// f = J·x + b_total = 0, so the solution is x = -(L^-1 ·
    /// (U^-1 · b_total)) = solver.solve(-b_total). We compute that
    /// directly: rhs = -(b_constant + b_extra), then solve.
    void solve(const topology::SwitchStateMask& mask,
                const Vector& b_extra,
                Vector& x) const {
        const PwlSegment& seg = lookup(mask);
        Vector rhs = -(seg.b_constant + b_extra);
        seg.solver->solve(rhs, x);
    }

    // -------------------------------------------------------------
    // DSED bridge — continuous-time LTI state-space extraction
    // (notes/DSED_BRIDGE_DESIGN.md — currently a stub; implementation
    // is Phase 5.1 of the bridge work).
    // -------------------------------------------------------------

    /// Result of `compute_lti_state_space(mask)` — the per-mask
    /// continuous-time state-space `dx_state/dt = A · x_state + b`.
    ///
    /// For use by ``pulsim.dsed.PEDSimulatorAuto`` (and the Python
    /// adapter in ``python/pulsim/dsed/_builder_bridge.py``).
    ///
    /// The state vector convention is
    /// `x_state = [v_C_1, ..., v_C_nc, i_L_1, ..., i_L_nl]` —
    /// `state_row_indices[i]` gives the FULL-MNA row where state `i`
    /// lives, and `state_is_cap[i]` discriminates caps from inductors.
    ///
    /// **Time-varying source overlay (Phase 5.2 / Bridge.6):**
    /// ``b_projection`` is the linear map from full-MNA `b` to the
    /// reduced state-space `b` (i.e. `b_state(t) = b_constant +
    /// b_projection · b_extra_mna(t)`). The Python adapter uses this
    /// to overlay sine / PWM / pulse source values per PED step,
    /// matching ``run_transient``'s b_extra pipeline.
    struct ContinuousLTI {
        DenseMatrix A;                         // n_state × n_state
        Vector b_constant;                      // n_state — DC sources
        std::vector<Index> state_row_indices;   // MNA row of each state
        std::vector<bool> state_is_cap;          // True=cap, False=inductor
        DenseMatrix b_projection;               // n_state × n_mna
                                                // Linear projection from
                                                // full-MNA b to state b
    };

    /// Extract continuous-time `(A, b)` for the given switch mask.
    ///
    /// v2.0 Phase 1: uses the assembler's EXACT (G, C, b) split —
    ///   1. `assemble_segment_split` → G_static, C (1/dt coeffs), b
    ///   2. `M_dyn = C / 2` (J(h) = G + (2/h)·M_dyn — algebraic,
    ///      replacing the two-assembly finite-difference recovery
    ///      of ``notes/DSED_BRIDGE_DESIGN.md`` §2)
    ///   4. Identify state rows via `pool.kind_of` + `branch_var_id_for_inductor`
    ///   5. Schur-complement the algebraic vars out → dense `A, b`
    ///
    /// **STATUS:** stub — throws `std::logic_error` until Phase 5.1
    /// of the bridge work lands. The pybind11 binding is wired so
    /// callers (Python `CircuitBuilderAdapter`) can catch the error
    /// and emit a clear `NotImplementedError` upstream.
    ///
    /// **Restrictions when implemented:**
    /// * LTI circuits only — nonlinear devices (diodes, MOSFETs,
    ///   IGBTs, saturable inductors) require per-operating-point
    ///   linearization not modeled by the PED engine; calling this
    ///   on a nonlinear topology will throw.
    /// * `h_a`, `h_b` are legacy recovery step sizes, retained for
    ///   API compatibility only — the exact split (v2.0 Phase 1)
    ///   ignores them.
    [[nodiscard]] ContinuousLTI
    compute_lti_state_space(
        const topology::SwitchStateMask& mask,
        Real h_a = Real{1e-6},
        Real h_b = Real{5e-7}) const {
        // ---------------------------------------------------------
        // Steps 1-2 (v2.0 Phase 1): EXACT (G, C, b) split from the
        // assembler — no more two-assembly finite-difference
        // recovery (the old path subtracted J(h_a) − J(h_b), losing
        // precision to cancellation whenever the static conductance
        // was small against the companion terms; audit finding A.5).
        // With J(h) = G + (1/h)·C = G_static + (2/h)·M_dyn, the
        // recovery is algebraic: G_st = G, M_dyn = C / 2.
        // `h_a` / `h_b` are retained in the signature for API
        // compatibility but no longer participate.
        // ---------------------------------------------------------
        (void)h_a;
        (void)h_b;
        sparse::Matrix G_split, C_split;
        Vector b_a;
        assemble_segment_split(graph_, pool_, mask,
                                G_split, C_split, b_a);

        const int n_mna = static_cast<int>(G_split.rows());

        DenseMatrix G_st  = DenseMatrix(G_split);  // sparse → dense
        DenseMatrix M_dyn = DenseMatrix(C_split) * Real{0.5};

        // ---------------------------------------------------------
        // Step 3: walk pool to identify caps + inductors. Caps may
        //         be GROUNDED (one terminal == kGround) or FLOATING
        //         (both terminals non-ground). Floating caps are
        //         re-oriented in Step 3b and handled via a
        //         congruence transform in Step 3c (Phase 5.1b —
        //         see notes/DSED_BRIDGE_DESIGN.md §5.1b).
        // ---------------------------------------------------------
        struct CapInfo {
            Index from;       // raw branch.from (may be kGround)
            Index to;         // raw branch.to   (may be kGround)
            Real C;
            // Re-oriented in Step 3b: pos = state-bearing terminal,
            // neg = closer to anchor (may be the synthetic ground).
            // After Step 3b, pos is always a real MNA index.
            Index pos = kInvalidIndex;
            Index neg = kInvalidIndex;
        };
        struct IndInfo {
            Index branch_var;
            Real L;
            Index from;       // raw branch.from (for cycle detection)
            Index to;         // raw branch.to
        };
        std::vector<CapInfo> caps;
        std::vector<IndInfo> inds;
        for (Index bid = 0; bid < graph_.num_branches(); ++bid) {
            const auto& branch = graph_.branch(bid);
            if (branch.kind != topology::BranchKind::PassiveLinear) {
                continue;
            }
            const auto k = pool_.kind_of(branch.id);
            if (k == DevicePool::StoredKind::Capacitor) {
                if (branch.from == kGround && branch.to == kGround) {
                    throw std::runtime_error(
                        "compute_lti_state_space: capacitor has "
                        "both terminals on ground (short circuit).");
                }
                caps.push_back({
                    branch.from, branch.to,
                    pool_.capacitor_params(branch.id).C,
                });
            } else if (k == DevicePool::StoredKind::Inductor) {
                const Index bv = pool_.branch_var_id_for_inductor(
                    branch.id, graph_);
                inds.push_back({
                    bv,
                    pool_.inductor_params(branch.id).L,
                    branch.from,
                    branch.to,
                });
            }
        }

        // -----------------------------------------------------------------
        // Inductor cycle detection (Bridge.9). Inductors form a graph
        // over MNA nodes; a CYCLE in this graph creates a KVL constraint
        // on di_L/dt that makes the per-mode A matrix singular (the
        // Schur complement on G_aa fails because the inductor branch-
        // var rows of G_ss are linearly dependent). The dual problem
        // for caps (floating + parallel) has a coordinate-change fix
        // (Phase 5.1b); for inductors the same fix exists in theory
        // but is much rarer in real PE circuits — we reject the
        // pathological case with a clear error and a workaround
        // pointer (merge into a single equivalent inductor).
        //
        // Detection: union-find over MNA nodes + ground sentinel,
        // union by inductor edges, detect cycle when both endpoints
        // are already in the same component.
        // -----------------------------------------------------------------
        if (!inds.empty()) {
            const Index kGroundIdx_ind = static_cast<Index>(n_mna);
            const Index kNodeCount_ind = kGroundIdx_ind + 1;
            auto map_idx_ind = [&](Index n) {
                return n == kGround ? kGroundIdx_ind : n;
            };
            std::vector<Index> uf_ind(kNodeCount_ind);
            std::iota(uf_ind.begin(), uf_ind.end(), Index{0});
            auto find_root_ind = [&](Index x) {
                while (uf_ind[x] != x) {
                    uf_ind[x] = uf_ind[uf_ind[x]];
                    x = uf_ind[x];
                }
                return x;
            };
            for (const auto& l : inds) {
                Index a = find_root_ind(map_idx_ind(l.from));
                Index b = find_root_ind(map_idx_ind(l.to));
                if (a == b) {
                    throw std::runtime_error(
                        "compute_lti_state_space: inductors form a "
                        "cycle (parallel inductors or longer all-"
                        "inductor loops). The KVL around the loop "
                        "creates a linear dependency on di_L/dt that "
                        "makes the LTI state-space singular. Merge "
                        "the looping inductors into a single equivalent "
                        "inductor upstream (parallel inductors: "
                        "L_eq = (L1·L2)/(L1+L2); series in the same "
                        "loop with mutual coupling: use Pulsim's "
                        "transformer/coupled-inductor API).");
                }
                uf_ind[a] = b;
            }
        }

        const int n_cap = static_cast<int>(caps.size());
        const int n_ind = static_cast<int>(inds.size());
        const int n_state = n_cap + n_ind;
        if (n_state == 0) {
            throw std::runtime_error(
                "compute_lti_state_space: circuit has no "
                "dynamic elements (caps or inductors); the "
                "PED engine integrates state equations, so a "
                "purely static circuit has no state to evolve.");
        }

        // Declared OUTSIDE the n_cap > 0 block so it stays in scope
        // for the B-projection construction in Step 7. Defaults to
        // identity (no congruence) for grounded-only or no-cap cases.
        DenseMatrix T = DenseMatrix::Identity(n_mna, n_mna);
        bool t_is_identity = true;

        // ---------------------------------------------------------
        // Step 3b: orient each cap via BFS over the cap-edge graph.
        //
        // We model caps as undirected edges between MNA nodes (with
        // a synthetic node `n_mna` representing the ground rail).
        // For each connected component of cap edges we pick an
        // *anchor*:
        //   - If the component touches ground, anchor = synthetic
        //     ground (every node in the component carries a cap
        //     state).
        //   - Otherwise, anchor = lowest-index MNA node in the
        //     component (the anchor stays algebraic; all other
        //     nodes carry a cap state).
        //
        // BFS from the anchor orients each tree edge as
        //   pos = farther-from-anchor, neg = closer-to-anchor.
        // A cap that doesn't end up as a tree edge means parallel
        // caps formed a cycle — that's deferred (merge upstream).
        // ---------------------------------------------------------
        if (n_cap > 0) {
            const Index kGroundIdx = static_cast<Index>(n_mna);
            const Index kNodeCount = kGroundIdx + 1;

            auto map_idx = [&](Index n) {
                return n == kGround ? kGroundIdx : n;
            };

            // Union-find over [0, n_mna] to identify components
            std::vector<Index> uf(kNodeCount);
            std::iota(uf.begin(), uf.end(), Index{0});
            auto find_root = [&](Index x) {
                while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
                return x;
            };
            auto unite = [&](Index a, Index b) {
                a = find_root(a);
                b = find_root(b);
                if (a != b) uf[a] = b;
            };
            for (const auto& c : caps) {
                unite(map_idx(c.from), map_idx(c.to));
            }

            // Anchor per component: prefer ground; otherwise lowest
            // MNA index that's present in the component.
            std::vector<Index> root_to_anchor(kNodeCount, kInvalidIndex);
            // First pass: stamp ground anchors.
            {
                Index r = find_root(kGroundIdx);
                root_to_anchor[r] = kGroundIdx;
            }
            // Second pass: stamp lowest-MNA-index for non-grounded
            // components (skip if already ground-anchored).
            for (Index i = 0; i < static_cast<Index>(n_mna); ++i) {
                Index r = find_root(i);
                if (root_to_anchor[r] == kGroundIdx) continue;
                if (root_to_anchor[r] == kInvalidIndex
                    || i < root_to_anchor[r]) {
                    root_to_anchor[r] = i;
                }
            }

            // Cap-edge adjacency over MNA + ground sentinel
            std::vector<std::vector<std::pair<Index, int>>> adj(kNodeCount);
            for (int k = 0; k < n_cap; ++k) {
                const Index a = map_idx(caps[k].from);
                const Index b = map_idx(caps[k].to);
                adj[a].push_back({b, k});
                adj[b].push_back({a, k});
            }

            // BFS from each anchor, recording tree edges (cap indices)
            // in BFS-discovery order.
            std::vector<int> bfs_cap_order;
            bfs_cap_order.reserve(n_cap);
            std::vector<char> visited(kNodeCount, char{0});
            for (Index r = 0; r < kNodeCount; ++r) {
                if (find_root(r) != r) continue;          // not a root
                Index anchor = root_to_anchor[r];
                if (anchor == kInvalidIndex) continue;    // empty comp
                if (visited[anchor]) continue;            // already done
                std::queue<Index> q;
                q.push(anchor);
                visited[anchor] = char{1};
                while (!q.empty()) {
                    Index u = q.front(); q.pop();
                    for (auto [v, cap_k] : adj[u]) {
                        if (visited[v]) continue;
                        visited[v] = char{1};
                        // Orient: pos = farther-from-anchor (v),
                        //         neg = closer-to-anchor   (u)
                        caps[cap_k].pos = v;
                        caps[cap_k].neg = u;
                        bfs_cap_order.push_back(cap_k);
                        q.push(v);
                    }
                }
            }

            // Any cap left unoriented = parallel-cap cycle. Defer.
            for (int k = 0; k < n_cap; ++k) {
                if (caps[k].pos == kInvalidIndex) {
                    throw std::runtime_error(
                        "compute_lti_state_space: capacitors form a "
                        "cycle (parallel caps share both terminals) — "
                        "merge them into a single equivalent cap "
                        "upstream. Phase 5.1b TODO.");
                }
                // The anchor (kGroundIdx OR a real MNA node) can be
                // a `neg` value. The `pos` is always a real MNA row.
                if (caps[k].pos == kGroundIdx) {
                    // Shouldn't happen: BFS from anchor visits non-
                    // anchor nodes as `v`; `v == kGroundIdx` is only
                    // possible if a non-ground anchor had a cap to
                    // ground, but then ground would have been in the
                    // component and already anchored. Defensive guard.
                    throw std::logic_error(
                        "compute_lti_state_space: internal BFS "
                        "produced kGroundIdx as a cap `pos`.");
                }
            }

            // -----------------------------------------------------
            // Step 3c: build T (sparse), apply congruence transform.
            //
            //   M_new = T^T · M_dyn · T
            //   G_new = T^T · G_st  · T
            //   b_new = T^T · b_a
            //
            // T is constructed by processing tree edges in *reverse*
            // BFS order and doing T.col(neg) += T.col(pos). After
            // this, T col `anchor` carries 1's at every node in the
            // grounded component (subtree under anchor), so the
            // anchor's row of M_new is zero — algebraic.
            // -----------------------------------------------------
            // Build T in place (start identity, was declared above
            // for outer-scope access).
            for (auto it = bfs_cap_order.rbegin();
                 it != bfs_cap_order.rend(); ++it) {
                const auto& c = caps[*it];
                // If `neg` is the synthetic ground (grounded cap on
                // a leaf of the tree), there's no column to update.
                if (c.neg == kGroundIdx) continue;
                T.col(c.neg) += T.col(c.pos);
            }

            // Skip the congruence apply if T is exactly identity
            // (no floating caps → grounded-only circuit → T = I).
            // This is the buck-CCM regression fast path.
            t_is_identity = T.isIdentity();
            if (!t_is_identity) {
                DenseMatrix Tt = T.transpose();
                DenseMatrix M_tmp = Tt * M_dyn * T;
                DenseMatrix G_tmp = Tt * G_st  * T;
                Vector      b_tmp = Tt * b_a;
                M_dyn = std::move(M_tmp);
                G_st  = std::move(G_tmp);
                b_a   = std::move(b_tmp);
            }
        }

        // ---------------------------------------------------------
        // Step 4: build the state-row index list (caps first, then
        // inductors) and the algebraic-row index list.
        //
        // For each cap, the state row is the `pos` MNA index (where
        // T put the +C diagonal of M_new). For each inductor, the
        // state row is its branch_var (T leaves inductor rows alone).
        // ---------------------------------------------------------
        std::vector<Index> state_rows;
        std::vector<bool> state_is_cap;
        state_rows.reserve(n_state);
        state_is_cap.reserve(n_state);
        for (const auto& c : caps) {
            state_rows.push_back(c.pos);
            state_is_cap.push_back(true);
        }
        for (const auto& i : inds) {
            state_rows.push_back(i.branch_var);
            state_is_cap.push_back(false);
        }

        std::vector<bool> is_state(n_mna, false);
        for (Index r : state_rows) is_state[r] = true;
        std::vector<Index> alg_rows;
        for (int r = 0; r < n_mna; ++r) {
            if (!is_state[r]) alg_rows.push_back(r);
        }
        const int n_alg = static_cast<int>(alg_rows.size());

        // ---------------------------------------------------------
        // Step 5: partition G_st and b_a into [SS, SA, AS, AA]
        // ---------------------------------------------------------
        DenseMatrix G_ss(n_state, n_state);
        DenseMatrix G_sa(n_state, n_alg);
        DenseMatrix G_as(n_alg,   n_state);
        DenseMatrix G_aa(n_alg,   n_alg);
        Vector b_s(n_state);
        Vector b_a_v(n_alg);
        for (int i = 0; i < n_state; ++i) {
            for (int j = 0; j < n_state; ++j) {
                G_ss(i, j) = G_st(state_rows[i], state_rows[j]);
            }
            for (int j = 0; j < n_alg; ++j) {
                G_sa(i, j) = G_st(state_rows[i], alg_rows[j]);
            }
            b_s(i) = b_a(state_rows[i]);
        }
        for (int i = 0; i < n_alg; ++i) {
            for (int j = 0; j < n_state; ++j) {
                G_as(i, j) = G_st(alg_rows[i], state_rows[j]);
            }
            for (int j = 0; j < n_alg; ++j) {
                G_aa(i, j) = G_st(alg_rows[i], alg_rows[j]);
            }
            b_a_v(i) = b_a(alg_rows[i]);
        }

        // ---------------------------------------------------------
        // Step 6: Schur-complement out the algebraic vars.
        //   M_ss · dx_s/dt = -G_ss · x_s - G_sa · x_a - b_s
        //   G_aa · x_a = -G_as · x_s - b_a_v
        //   → x_a = -G_aa^{-1} · (G_as · x_s + b_a_v)
        //   → M_ss · dx_s/dt = (G_sa · G_aa^{-1} · G_as - G_ss) · x_s
        //                     + (G_sa · G_aa^{-1} · b_a_v - b_s)
        //
        // We also build the projection matrix B (n_state × n_mna)
        // such that for any time-varying b_extra in original MNA
        // coords, b_reduced(t) = B · b_extra(t). This lets the
        // Python adapter overlay sine/PWM/pulse contributions per
        // PED step without re-stamping (Phase 5.2 / Bridge.6).
        //
        //   B = M_ss^{-1} · (G_sa · G_aa^{-1} · S_alg - S_state) · T^T
        //
        // where S_state (n_state × n_mna) and S_alg (n_alg × n_mna)
        // are selector matrices picking the state and algebraic rows.
        // ---------------------------------------------------------
        DenseMatrix Schur_G(n_state, n_state);
        Vector Schur_b(n_state);

        // Build selector matrices (sparse-ish: one 1 per row)
        DenseMatrix S_state = DenseMatrix::Zero(n_state, n_mna);
        for (int i = 0; i < n_state; ++i) {
            S_state(i, state_rows[i]) = Real{1};
        }
        DenseMatrix S_alg = DenseMatrix::Zero(n_alg, n_mna);
        for (int i = 0; i < n_alg; ++i) {
            S_alg(i, alg_rows[i]) = Real{1};
        }
        DenseMatrix B(n_state, n_mna);

        if (n_alg > 0) {
            Eigen::FullPivLU<DenseMatrix> lu_aa(G_aa);
            if (!lu_aa.isInvertible()) {
                throw std::runtime_error(
                    "compute_lti_state_space: algebraic block "
                    "G_aa is singular; circuit topology does "
                    "not have a unique algebraic solution for "
                    "this switch mask.");
            }
            DenseMatrix Ginv_Gas = lu_aa.solve(G_as);
            Vector      Ginv_ba  = lu_aa.solve(b_a_v);
            Schur_G = G_sa * Ginv_Gas - G_ss;
            Schur_b = G_sa * Ginv_ba  - b_s;

            // B uses the SAME factorisation: G_aa^{-1} · S_alg.
            DenseMatrix Ginv_Salg = lu_aa.solve(S_alg);
            B = G_sa * Ginv_Salg - S_state;
        } else {
            Schur_G = -G_ss;
            Schur_b = -b_s;
            B = -S_state;
        }

        // Apply T^T on the right (the input b is in ORIGINAL coords;
        // the extractor internally transformed b_a via T^T already).
        if (!t_is_identity) {
            B = B * T.transpose();
        }

        // ---------------------------------------------------------
        // Step 7: apply M_ss^{-1}. M_ss is diagonal: +C for caps,
        // -L for inductors. Apply per-row scaling to A, b_reduced,
        // AND B (since b_reduced and B both go through M_ss^{-1}).
        // ---------------------------------------------------------
        DenseMatrix A = Schur_G;
        Vector b_reduced = Schur_b;
        for (int i = 0; i < n_cap; ++i) {
            const Real inv = Real{1} / caps[i].C;
            A.row(i) *= inv;
            b_reduced(i) *= inv;
            B.row(i) *= inv;
        }
        for (int i = 0; i < n_ind; ++i) {
            const Real inv = Real{-1} / inds[i].L;
            A.row(n_cap + i) *= inv;
            b_reduced(n_cap + i) *= inv;
            B.row(n_cap + i) *= inv;
        }

        return ContinuousLTI{
            std::move(A),
            std::move(b_reduced),
            std::move(state_rows),
            std::move(state_is_cap),
            std::move(B),
        };
    }

    /// Number of segments currently cached. In eager mode this
    /// is `2^N` after `build()`; in lazy mode it grows as
    /// `solve(mask, ...)` calls populate the cache.
    [[nodiscard]] Size num_built_segments() const noexcept {
        return segments_.size();
    }

    [[nodiscard]] Size num_segments() const noexcept {
        return segments_.size();
    }

    /// Multi-dt solve (Layer 4 V7; REWORKED in v2.0 Phase 1 —
    /// audit finding `alt-dt-cache-unbounded-factorization`).
    /// Solves with a dt that MAY be different from the primary
    /// `this->dt()`.
    ///
    /// The v1.x implementation kept a map keyed by EXACT Real dt
    /// of full per-mask segment maps: every event-interpolated dt
    /// from sub-step bisection (a continuous value, distinct for
    /// essentially every commutation) permanently retained a full
    /// analyze+factorize'd segment — unbounded memory AND two full
    /// symbolic analyses per event.
    ///
    /// Now: a small LRU (≤ `kMaxEventEntries` masks) of reusable
    /// EVENT SOLVERS. Each entry stores the mask's (G, C, b) split
    /// (`J(dt) = G + (1/dt)·C`, see `assemble_segment_split`) and a
    /// solver analyzed ONCE — a dt change is a value re-combine +
    /// numeric `factorize()` on the shared pattern (the sparsity
    /// pattern is dt-invariant). No dt-keyed storage exists at all.
    ///
    /// When `dt == this->dt()`, delegates to the primary
    /// `solve(mask, b_extra, x)` (same fast path, bit-identical).
    void solve_at(const topology::SwitchStateMask& mask,
                   Real dt,
                   const Vector& b_extra,
                   Vector& x) const {
        if (dt == dt_) {
            solve(mask, b_extra, x);
            return;
        }
        const std::uint64_t tick = ++lru_clock_;
        auto it = event_entries_.find(mask);
        if (it == event_entries_.end()) {
            // Fresh mask: make room first (erase invalidates
            // references, but so does the emplace below — callers
            // may not hold entry references across solve_at calls,
            // same contract as the primary flat_map).
            if (event_entries_.size() >=
                static_cast<std::size_t>(kMaxEventEntries)) {
                evict_lru_event_entry_();
            }
            EventEntry e;
            assemble_segment_split(graph_, pool_, mask,
                                    e.G, e.C, e.b_constant);
            e.solver = sparse::make_default_solver(
                pool_.state_size(graph_), sparse::Backend::Auto);
            const sparse::Matrix J = combine_event_J_(e, mask, dt);
            if (!e.solver->analyze(J)) {
                throw std::runtime_error(std::format(
                    "PwlStateSpaceCache::solve_at: analyze failed "
                    "(structurally singular) for mask {} (dt={})",
                    mask.to_string(), dt));
            }
            if (!e.solver->factorize(J)) {
                throw std::runtime_error(std::format(
                    "PwlStateSpaceCache::solve_at: factorize failed "
                    "(numerically singular) for mask {} (dt={})",
                    mask.to_string(), dt));
            }
            e.current_dt = dt;
            e.lru_tick   = tick;
            it = event_entries_.emplace(mask, std::move(e)).first;
            event_builds_.fetch_add(1, std::memory_order_relaxed);
        } else if (it->second.current_dt != dt) {
            // Known mask, new dt: numeric refactor on the SAME
            // symbolic analysis (pattern is dt-invariant).
            EventEntry& e = it->second;
            const sparse::Matrix J = combine_event_J_(e, mask, dt);
            if (!e.solver->factorize(J)) {
                // The solver destroys its previous factor up-front,
                // so the entry can no longer serve ANY dt — erase it
                // before throwing (adversarial-review finding
                // EVT-STALE-FACTOR: keeping it resident with the old
                // current_dt made a later solve_at at the previously
                // WORKING dt hit the reuse branch and die with
                // logic_error instead of transparently rebuilding).
                event_entries_.erase(it);
                throw std::runtime_error(std::format(
                    "PwlStateSpaceCache::solve_at: refactorize "
                    "failed (numerically singular) for mask {} "
                    "(dt={})", mask.to_string(), dt));
            }
            e.current_dt = dt;
            e.lru_tick   = tick;
            event_refactors_.fetch_add(1, std::memory_order_relaxed);
        } else {
            it->second.lru_tick = tick;
            event_hits_.fetch_add(1, std::memory_order_relaxed);
        }
        const EventEntry& e = it->second;
        Vector rhs = -(e.b_constant + b_extra);
        e.solver->solve(rhs, x);
    }

    /// Maximum number of event-solver entries `solve_at` keeps
    /// resident (LRU beyond this). Sub-step event correction
    /// alternates between the pre- and post-commutation masks, so
    /// a handful of slots covers real converters with headroom.
    static constexpr Size kMaxEventEntries = 8;

    /// Number of event-solver entries currently resident
    /// (≤ `kMaxEventEntries`).
    [[nodiscard]] Size num_event_entries() const noexcept {
        return event_entries_.size();
    }

    /// Number of DISTINCT dt values currently loaded across the
    /// event-solver entries. (v2.0 Phase 1: an entry holds ONE
    /// factor at its most recent dt — dt values are no longer
    /// accumulated, so this is bounded by `kMaxEventEntries`.)
    [[nodiscard]] Size num_alt_dt_values() const noexcept {
        std::set<Real> dts;
        for (const auto& [mask, e] : event_entries_) {
            dts.insert(e.current_dt);
        }
        return static_cast<Size>(dts.size());
    }

    /// Number of event-solver entries whose CURRENT factor is at
    /// the given dt (0 if none).
    [[nodiscard]] Size num_alt_segments_at(Real dt) const noexcept {
        Size n = 0;
        for (const auto& [mask, e] : event_entries_) {
            if (e.current_dt == dt) ++n;
        }
        return n;
    }

    // -------------------------------------------------------------------------
    // Layer 4 V8 — rank-1 cache update fast path
    // (openspec/changes/add-pwl-rank1-update)
    // -------------------------------------------------------------------------
    //
    // `solve_rank1` maintains its OWN sliding factorisation that is updated
    // incrementally across consecutive calls: on a single-bit Gray-code mask
    // flip AND a partial-refactor-capable backend (e.g. KLU), it calls
    // `DirectSolver::partial_refactor` to update the factor in O(path)
    // instead of O(nnz·log n) full refactorisation. Multi-bit flips,
    // unsupported backends, numerical singularities, and first-encounter
    // masks all fall back transparently to the same `factorize` cost the
    // existing `solve` path would pay.
    //
    // The sliding solver is INDEPENDENT of the per-mask `segments_` map
    // populated by `build` / `build_lazy` / `solve`. The two paths are
    // orthogonal:
    //   * `solve(mask)`       — best for sequences that revisit a small set
    //                           of masks (cache hit on revisit).
    //   * `solve_rank1(mask)` — best for sequences that march through many
    //                           single-bit-adjacent masks (2^N too large for
    //                           the per-mask cache to help).
    //
    // Per the proposal:
    //   * Backend without `partial_refactor` support → silent fallback to
    //     full re-factor; `metrics().fallbacks` increments.
    //   * `partial_refactor` returning false (numerical singularity) →
    //     silent fallback to full re-factor; `metrics().fallbacks`
    //     increments.
    //   * Multi-bit flip OR first call → full re-factor; the existing
    //     full path; `metrics().full_refactor_hits` increments.
    //   * Single-bit flip with partial-refactor success →
    //     `metrics().rank1_hits` increments.

    /// Rank-1 sliding-solver entry point. Builds J for `mask`, then either
    /// reuses, partial-refactors, or full-refactors the cached factor as
    /// the mask diff vs the previous call dictates. Falls back
    /// transparently on any unsupported / failed partial-refactor.
    ///
    /// Requires `build(dt)` or `build_lazy(dt)` to have been called first
    /// (to set `dt_`). Throws `std::runtime_error` on initial-factorize
    /// numerical singularity.
    void solve_rank1(const topology::SwitchStateMask& mask,
                      const Vector& b_extra,
                      Vector& x) const {
        sparse::Matrix J;
        Vector b;
        assemble_segment(graph_, pool_, mask, dt_, J, b);
        sparse::compress_in_place(J);

        if (!rank1_initialized_) {
            // First call: fresh analyze + factorize on a new solver
            // picked by the factory. Backend hint defaults to
            // Backend::Auto (which currently falls through to
            // SparseLuSolver — the Pulsim native LU lands in a
            // follow-up, see openspec/changes/replace-klu-with-pulsim-sparse-lu/).
            // The caller may override via set_rank1_backend(...).
            rank1_solver_ = sparse::make_default_solver(
                pool_.state_size(graph_), rank1_backend_hint_);
            if (!rank1_solver_->analyze(J)) {
                throw std::runtime_error(std::format(
                    "PwlStateSpaceCache::solve_rank1: initial analyze "
                    "failed (structurally singular) for mask {}",
                    mask.to_string()));
            }
            if (!rank1_solver_->factorize(J)) {
                throw std::runtime_error(std::format(
                    "PwlStateSpaceCache::solve_rank1: initial factorize "
                    "failed (numerically singular) for mask {} (dt={})",
                    mask.to_string(), dt_));
            }
            rank1_b_constant_ = b;
            rank1_mask_       = mask;
            rank1_initialized_ = true;
            full_refactor_hits_.fetch_add(1, std::memory_order_relaxed);
        } else if (mask == rank1_mask_) {
            // No mask change — solver still valid, just refresh b_constant
            // in case time-varying sources updated their baseline.
            rank1_b_constant_ = b;
            rank1_hits_.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Mask changed: decide partial vs full refactor.
            //
            // v1.4.0 (openspec/changes/add-generalised-path-refactor Part A):
            // multi-bit flips are no longer unconditionally routed to
            // full factorize. Instead we compute the affected columns for
            // EVERY toggled bit (deduplicated), query the solver for the
            // would-be path length, and:
            //   - if path-length / n ≤ MAX_PATH_LENGTH_RATIO → try
            //     partial_refactor on the multi-column changed set
            //   - else → fall back to full factorize (path-based wouldn't
            //     be cheaper anyway)
            const int pop = static_cast<int>(
                mask.hamming_distance(rank1_mask_));

            const auto changed_cols =
                compute_changed_columns_(rank1_mask_, mask);
            const std::span<const Index> changed_cols_span{
                changed_cols.data(), changed_cols.size()};

            const bool can_try_partial =
                rank1_solver_->supports_partial_refactor();
            // Routing rule per
            // `openspec/changes/add-generalised-path-refactor`
            // §pwl-rank1-update spec:
            //   * pop == 1 (single-bit, v1.3.0 case) — always try
            //     partial_refactor when the backend supports it. The
            //     captured microbench shows 100% success rate on
            //     single-bit Gray-code flips; the path is always
            //     short. No ratio gate needed.
            //   * pop >= 2 (multi-bit, v1.4.0 NEW) — query
            //     partial_refactor_count_path() and skip the
            //     path-based attempt only when the union path covers
            //     > MAX_PATH_LENGTH_RATIO of the columns (where
            //     path-based wouldn't save work vs a fresh factorise).
            bool try_path = can_try_partial;
            if (try_path && pop > 1) {
                const Index n_state = static_cast<Index>(
                    pool_.state_size(graph_));
                const Index path_len =
                    rank1_solver_->partial_refactor_count_path(
                        changed_cols_span);
                const Real path_ratio = n_state > 0
                    ? (static_cast<Real>(path_len) /
                       static_cast<Real>(n_state))
                    : Real{1};
                try_path = (path_ratio <= sparse::MAX_PATH_LENGTH_RATIO);
            }

            bool refactored_via_partial = false;
            if (try_path) {
                refactored_via_partial = rank1_solver_->partial_refactor(
                    J, changed_cols_span);
            }

            if (refactored_via_partial) {
                rank1_b_constant_ = b;
                rank1_mask_       = mask;
                if (pop == 1) {
                    rank1_hits_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    multi_bit_rank1_hits_.fetch_add(
                        1, std::memory_order_relaxed);
                }
            } else {
                // Full refactor on the same symbolic. The sparsity pattern
                // is identical across all switch states (only conductance
                // values change), so analyze() does NOT need to re-run.
                if (!rank1_solver_->factorize(J)) {
                    throw std::runtime_error(std::format(
                        "PwlStateSpaceCache::solve_rank1: full refactor "
                        "fallback failed (numerically singular) for mask "
                        "{} (dt={})",
                        mask.to_string(), dt_));
                }
                rank1_b_constant_ = b;
                rank1_mask_       = mask;
                if (try_path) {
                    // partial_refactor was attempted and returned false
                    // (numerical/pivot fault) — fall back.
                    fallbacks_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    // Either the backend can't do partial OR the path
                    // ratio was above MAX_PATH_LENGTH_RATIO → didn't
                    // even try. Counted under full_refactor_hits, not
                    // fallbacks (the "didn't try" vs "tried and failed"
                    // distinction matters for diagnostics).
                    full_refactor_hits_.fetch_add(
                        1, std::memory_order_relaxed);
                }
            }
        }

        Vector rhs = -(rank1_b_constant_ + b_extra);
        rank1_solver_->solve(rhs, x);
    }

    /// Override the backend `solve_rank1` uses for its sliding solver.
    /// By default `solve_rank1` calls
    /// `make_default_solver(state_size, Backend::Auto)`. Call this
    /// before the first `solve_rank1` to force a specific backend —
    /// e.g. `Backend::Pulsim` for benchmarks that want the path-based
    /// partial-refactor path (once
    /// openspec/changes/replace-klu-with-pulsim-sparse-lu/ Sections 2-5
    /// land), or `Backend::Eigen` to force the Eigen::SparseLU
    /// fallback path for measurement.
    ///
    /// No-op once `solve_rank1` has been called — the solver has
    /// already been constructed.
    void set_rank1_backend(sparse::Backend hint) noexcept {
        if (!rank1_initialized_) {
            rank1_backend_hint_ = hint;
        }
    }

    /// Snapshot the rank-1 telemetry counters. Read-only from outside;
    /// monotonic across the cache's lifetime.
    [[nodiscard]] CacheMetrics metrics() const noexcept {
        return {
            .rank1_hits = rank1_hits_.load(
                std::memory_order_relaxed),
            .multi_bit_rank1_hits = multi_bit_rank1_hits_.load(
                std::memory_order_relaxed),
            .full_refactor_hits = full_refactor_hits_.load(
                std::memory_order_relaxed),
            .fallbacks = fallbacks_.load(
                std::memory_order_relaxed),
            .segment_evictions = segment_evictions_.load(
                std::memory_order_relaxed),
            .event_hits = event_hits_.load(
                std::memory_order_relaxed),
            .event_refactors = event_refactors_.load(
                std::memory_order_relaxed),
            .event_builds = event_builds_.load(
                std::memory_order_relaxed),
        };
    }

    // -------------------------------------------------------------
    // v2.0 Phase 1 — lazy segment cache byte budget (audit finding
    // `no-mode-cache-eviction`).
    //
    // In LAZY mode, once the estimated resident bytes of the
    // per-mask segment cache would exceed the budget, the least-
    // recently-solved segment is evicted before a new one is
    // inserted. Re-visiting an evicted mask transparently rebuilds
    // it (one assemble + factorize — the lazy path's normal first-
    // visit cost). Eviction happens ONLY when a new segment is
    // built, never on a cache hit. Reference contract: callers must
    // not hold `lookup()` references across cache-MUTATING calls —
    // a lazy insert reallocates under the flat_map Dictionary
    // config, and eviction additionally erases under either config
    // — and every in-repo caller (run_transient's Newton path
    // included) consumes its reference before the next cache call.
    //
    // Note: the ≤ 8 event-solver entries (`solve_at`) are OUTSIDE
    // this budget — they are bounded by count, not bytes.
    //
    // Eager `build()` mode never evicts: pre-building all 2^N
    // factors is an explicit caller decision, and the eager
    // `lookup()` contract ("throws if the mask wasn't pre-built")
    // cannot survive eviction.
    // -------------------------------------------------------------

    /// Default lazy-cache budget: 1 GiB — far above any realistic
    /// visited-mode working set (hundreds of large-circuit factors),
    /// while bounding a week-long mode-random-walk run.
    static constexpr std::size_t kDefaultSegmentBudgetBytes =
        std::size_t{1} << 30;

    /// Set the lazy segment cache budget in bytes. `0` disables
    /// eviction entirely (the pre-v2.0 unbounded behaviour).
    void set_segment_budget_bytes(std::size_t bytes) noexcept {
        segment_budget_bytes_ = bytes;
    }

    [[nodiscard]] std::size_t segment_budget_bytes() const noexcept {
        return segment_budget_bytes_;
    }

    /// Estimated resident bytes of the per-mask segment cache
    /// (assembled J + b + numeric LU factors per segment). An
    /// ESTIMATE for budget arithmetic, not an allocator audit.
    [[nodiscard]] std::size_t segment_cache_bytes() const noexcept {
        std::size_t total = 0;
        for (const auto& [mask, seg] : segments_) {
            total += seg.approx_bytes();
        }
        return total;
    }

    // -------------------------------------------------------------------------
    // v1.4.0 Part B — parametric refactor for sweep / Monte Carlo workloads
    // -------------------------------------------------------------------------
    //
    // Pushes a batch of `(branch_id, new_value)` updates into the pool,
    // then walks every active mask in `segments_` (or just the
    // rank-1 sliding solver's mask, per `mode`) and re-stamps + path-
    // refactors each cached segment at the minimum column set.
    //
    // The key win vs the legacy "build_lazy + sweep" pattern: we
    // **reuse the symbolic factorization** (`analyze`) and the
    // **majority of L+U entries** across sweep points. Only the
    // columns of J that depend on the changed parameters need re-
    // elimination, and only via the etree path of those columns.
    //
    // Result invariant: `path_refactor_hits + fallback_hits ==
    //                    masks_processed`.
    //
    // Numerical contract: post-refactor `solve(mask, ...)` matches a
    // fresh `analyze + factorize + solve` on the updated pool within
    // 1e-10 (the same tolerance the multi-bit path-union gate
    // enforces).
    //
    // Pre-conditions:
    //   * `build(...)` or `build_lazy(...)` has been called.
    //   * Every `updates[i].branch_id` is a registered device whose
    //     kind matches the `update_*` mutator semantics (Resistor,
    //     Inductor, Capacitor, VoltageSource). Other kinds silently
    //     fall back to full-rebuild for every affected mask.
    //   * Topology is unchanged (graph node/branch structure intact).
    //     Updating values of a switch is not supported; switches are
    //     toggled via the mask, not via parametric refactor.
    [[nodiscard]] ParametricRefactorResult refactor_parametric(
        std::span<const ParametricUpdate> updates,
        ParametricRefactorMode mode = ParametricRefactorMode::AllActive) {
        ParametricRefactorResult result;
        const auto t0 = std::chrono::steady_clock::now();

        // v2.0 Phase 1: parameter updates invalidate the event
        // entries' (G, C, b) snapshots. Drop them — the next
        // solve_at rebuilds from the updated pool. (The v1.x aux
        // cache silently kept STALE factors here.)
        event_entries_.clear();

        if (updates.empty()) {
            // No-op. Still report wall_time for telemetry callers.
            const auto t1 = std::chrono::steady_clock::now();
            result.wall_time_us =
                std::chrono::duration<double, std::micro>(t1 - t0).count();
            return result;
        }

        // 1. Push the updates into the pool. Each `update_*` mutator
        //    dispatches on the stored device kind; mismatched kinds
        //    throw std::out_of_range which propagates to the caller.
        for (const auto& upd : updates) {
            apply_parametric_update_(upd);
        }

        // 2. Build the deduplicated union of columns affected by the
        //    parameter changes. An empty affected-cols set on any
        //    branch (e.g. VoltageSource — RHS-only) means that
        //    branch's change doesn't touch J; only b_constant needs
        //    re-stamping, which happens during the per-segment
        //    re-assemble step inside refactor_one_parametric_.
        std::set<Index> affected_cols_set;
        for (const auto& upd : updates) {
            const auto cols =
                pool_.columns_affected_by_branch(upd.branch_id, graph_);
            for (Index c : cols) affected_cols_set.insert(c);
        }

        const std::vector<Index> affected_cols(
            affected_cols_set.begin(), affected_cols_set.end());
        const std::span<const Index> affected_cols_span{
            affected_cols.data(), affected_cols.size()};

        // 3. Pick which segments to process.
        std::vector<topology::SwitchStateMask> masks_to_process;
        if (mode == ParametricRefactorMode::CurrentOnly) {
            if (rank1_initialized_) {
                masks_to_process.push_back(rank1_mask_);
            }
        } else {
            // AllActive — every cached primary segment AND, if the
            // rank-1 sliding solver is in use, its mask too.
            masks_to_process.reserve(segments_.size() +
                                      (rank1_initialized_ ? 1 : 0));
            for (const auto& [m, _seg] : segments_) {
                masks_to_process.push_back(m);
            }
            if (rank1_initialized_) {
                // Avoid double-processing if the rank-1 mask is also
                // in segments_ (e.g. the user did a mixed
                // solve + solve_rank1 workflow).
                if (std::find(masks_to_process.begin(),
                              masks_to_process.end(),
                              rank1_mask_) == masks_to_process.end()) {
                    masks_to_process.push_back(rank1_mask_);
                }
            }
        }

        // 4. For each selected mask, re-assemble J + b at the new
        //    pool values, then path-refactor (or fall back to full
        //    factorize when the path covers > MAX_PATH_LENGTH_RATIO
        //    of n, or when the backend doesn't support partial_refactor).
        for (const auto& mask : masks_to_process) {
            ++result.masks_processed;
            if (refactor_one_parametric_(mask, affected_cols_span)) {
                ++result.path_refactor_hits;
            } else {
                ++result.fallback_hits;
            }
        }

        const auto t1 = std::chrono::steady_clock::now();
        result.wall_time_us =
            std::chrono::duration<double, std::micro>(t1 - t0).count();
        return result;
    }

    /// Convenience overload for the very common one-parameter case:
    /// `cache.refactor_parametric(branch_id, new_value)`.
    [[nodiscard]] ParametricRefactorResult refactor_parametric(
        Index branch_id, Real new_value,
        ParametricRefactorMode mode = ParametricRefactorMode::AllActive) {
        ParametricUpdate one{branch_id, new_value};
        return refactor_parametric(
            std::span<const ParametricUpdate>(&one, 1), mode);
    }

private:
    /// Non-throwing build of a single segment. Returns the
    /// segment by move on success, or a `CacheError` carrying
    /// the singularity kind + mask + dt on failure.
    [[nodiscard]] std::expected<PwlSegment, CacheError>
    try_make_segment(const topology::SwitchStateMask& mask, Real dt) {
        sparse::Matrix J;
        Vector b;
        assemble_segment(graph_, pool_, mask, dt, J, b);
        sparse::compress_in_place(J);

        // v1.4.0: each segment's solver is the Pulsim in-house LU
        // (`Backend::Auto` → Pulsim since v1.3.0). This gives
        // `refactor_parametric` access to `partial_refactor` on the
        // per-mask cached factor; the legacy two-arg
        // `make_default_solver(state_size, Backend::Auto)` call is
        // bit-identical to the v1.3.0 result on real-scalar SPD
        // matrices. The no-arg `make_default_solver()` overload
        // (Eigen baseline) is now kept only for the showcase /
        // sanity-check tests that pin the Eigen behaviour for
        // paper-comparison purposes.
        auto solver = sparse::make_default_solver(
            pool_.state_size(graph_), sparse::Backend::Auto);
        if (!solver->analyze(J)) {
            return std::unexpected(CacheError{
                .kind = CacheError::Kind::StructurallySingular,
                .mask = mask,
                .dt   = dt,
            });
        }
        if (!solver->factorize(J)) {
            return std::unexpected(CacheError{
                .kind = CacheError::Kind::NumericallySingular,
                .mask = mask,
                .dt   = dt,
            });
        }

        PwlSegment seg;
        seg.J = std::move(J);
        seg.b_constant = std::move(b);
        seg.solver = std::move(solver);
        seg.state_size = pool_.state_size(graph_);
        return seg;
    }

    /// Throwing wrapper — the existing kernel hot path. Delegates
    /// to `try_make_segment` and translates the typed error into
    /// a `runtime_error` whose `what()` matches the V0 format
    /// callers (and tests) already grep for.
    [[nodiscard]] PwlSegment make_segment(
        const topology::SwitchStateMask& mask, Real dt) {
        auto r = try_make_segment(mask, dt);
        if (!r) {
            throw std::runtime_error(r.error().what());
        }
        return std::move(*r);
    }

    /// Non-throwing single-segment insert at the current dt.
    /// Used by `try_lookup`'s lazy-mode path.
    [[nodiscard]] std::expected<void, CacheError>
    try_build_one_segment(const topology::SwitchStateMask& mask) {
        auto seg = try_make_segment(mask, dt_);
        if (!seg) {
            return std::unexpected(seg.error());
        }
        // v2.0 Phase 1: LRU eviction BEFORE the insert (lazy mode
        // only — see the budget block above). Reference-safety
        // contract: callers must not hold segment references across
        // cache-mutating calls (a lazy insert reallocates under the
        // flat_map Dictionary config, and eviction erases under
        // either config) — every in-repo caller consumes its
        // `lookup()` reference before the next cache call
        // (run_transient's Newton path included). The incoming
        // segment is ALWAYS inserted even when it alone exceeds the
        // budget — refusing to build would turn a memory knob into
        // a wrong-answer knob.
        if (lazy_mode_ && segment_budget_bytes_ > 0) {
            const std::size_t incoming = seg->approx_bytes();
            std::size_t resident = segment_cache_bytes();
            while (!segments_.empty() &&
                   resident + incoming > segment_budget_bytes_) {
                auto victim = segments_.begin();
                for (auto it = segments_.begin();
                     it != segments_.end(); ++it) {
                    if (it->second.lru_tick <
                        victim->second.lru_tick) {
                        victim = it;
                    }
                }
                resident -= victim->second.approx_bytes();
                segments_.erase(victim);
                segment_evictions_.fetch_add(
                    1, std::memory_order_relaxed);
            }
        }
        seg->lru_tick = ++lru_clock_;
        segments_.emplace(mask, std::move(*seg));
        return {};
    }

    /// Insert a primary-cache segment at the current dt.
    /// Eager `build()` calls in a loop; lazy `lookup()` on
    /// demand.
    void build_one_segment(const topology::SwitchStateMask& mask) {
        segments_.emplace(mask, make_segment(mask, dt_));
    }

    /// Dispatch a single parametric update to the correct pool
    /// mutator based on the stored device kind. v1.4.0 supports
    /// Resistor, Inductor, Capacitor, VoltageSource. Other kinds
    /// raise std::out_of_range — caller is expected to either avoid
    /// the update or rebuild the cache from scratch via
    /// `build_lazy(dt_)`.
    void apply_parametric_update_(const ParametricUpdate& upd) {
        const auto kind = pool_.kind_of(upd.branch_id);
        using SK = DevicePool::StoredKind;
        switch (kind) {
            case SK::Resistor:
                pool_.update_resistor_R(upd.branch_id, upd.new_value);
                break;
            case SK::Inductor:
                pool_.update_inductor_L(upd.branch_id, upd.new_value);
                break;
            case SK::Capacitor:
                pool_.update_capacitor_C(upd.branch_id, upd.new_value);
                break;
            case SK::VoltageSource:
                pool_.update_voltage_source_V(upd.branch_id,
                                                upd.new_value);
                break;
            default:
                throw std::out_of_range(std::format(
                    "PwlStateSpaceCache::refactor_parametric: "
                    "branch {} has device kind {} which is not "
                    "supported by the v1.4.0 parametric refactor "
                    "pipeline (Resistor / Inductor / Capacitor / "
                    "VoltageSource only). Rebuild the cache via "
                    "build_lazy(dt) after updating this device.",
                    upd.branch_id, static_cast<int>(kind)));
        }
    }

    /// Re-assemble + refactor one (mask, segment) for the parametric
    /// case. Updates BOTH the primary segment (if cached) AND the
    /// rank-1 sliding solver (if initialized at this mask) — the
    /// two paths share a logical mask but have independent solver
    /// instances. Returns true if every applicable target took the
    /// path-based update; false if any target fell back to fresh
    /// `factorize()`. Throws std::runtime_error on fresh-factorize
    /// numerical singularity (no recovery — the parameter values are
    /// genuinely problematic).
    [[nodiscard]] bool refactor_one_parametric_(
        const topology::SwitchStateMask& mask,
        std::span<const Index> affected_cols) {
        // We may update BOTH a primary segment AND the rank-1
        // sliding solver if the mask matches both. Re-assemble
        // (J, b) once per target so refactor_solver_'s `std::move`
        // of new_J doesn't fight between paths.
        bool any_target_existed = false;
        bool any_path_succeeded = false;

        auto it = segments_.find(mask);
        if (it != segments_.end()) {
            any_target_existed = true;
            sparse::Matrix new_J;
            Vector new_b;
            assemble_segment(graph_, pool_, mask, dt_, new_J, new_b);
            sparse::compress_in_place(new_J);
            const bool ok = refactor_solver_(
                it->second.solver.get(),
                it->second.J, it->second.b_constant,
                new_J, new_b, affected_cols);
            any_path_succeeded = any_path_succeeded || ok;
        }
        if (rank1_initialized_ && mask == rank1_mask_) {
            any_target_existed = true;
            sparse::Matrix new_J;
            Vector new_b;
            assemble_segment(graph_, pool_, mask, dt_, new_J, new_b);
            sparse::compress_in_place(new_J);
            // The rank-1 sliding solver doesn't cache J (it
            // re-assembles every solve_rank1 call), but its
            // factor + b_constant are stateful and must be
            // refreshed here so any subsequent solve_rank1 sees
            // the post-update pool values.
            sparse::Matrix dummy_old_J;
            const bool ok = refactor_solver_(
                rank1_solver_.get(),
                dummy_old_J,
                rank1_b_constant_,
                new_J, new_b, affected_cols);
            any_path_succeeded = any_path_succeeded || ok;
        }
        if (!any_target_existed) {
            // No cached factor for this mask anywhere. Count as
            // fallback so the result invariant still holds; the
            // next solve(mask) will lazily build a fresh segment
            // from the post-update pool.
            return false;
        }
        return any_path_succeeded;
    }

    /// Common path-vs-factorize logic shared by primary and rank-1
    /// segments. Mutates `seg_J` + `seg_b` to the new values on
    /// return regardless of which path was taken.
    [[nodiscard]] bool refactor_solver_(
        sparse::DirectSolver* solver,
        sparse::Matrix& seg_J, Vector& seg_b,
        sparse::Matrix& new_J, Vector& new_b,
        std::span<const Index> affected_cols) {
        const Index n = static_cast<Index>(pool_.state_size(graph_));
        const bool can_try_partial =
            solver->supports_partial_refactor();

        // Decide whether to attempt the path-based update. Same
        // ratio gate as the multi-bit case: if the union path
        // covers > MAX_PATH_LENGTH_RATIO of n, the path-based
        // update wouldn't be cheaper than fresh factorize.
        bool try_path = can_try_partial && !affected_cols.empty();
        if (try_path) {
            const Index path_len =
                solver->partial_refactor_count_path(affected_cols);
            const Real ratio = n > 0
                ? (static_cast<Real>(path_len) / static_cast<Real>(n))
                : Real{1};
            try_path = (ratio <= sparse::MAX_PATH_LENGTH_RATIO);
        }

        bool refactored_via_partial = false;
        if (try_path) {
            refactored_via_partial =
                solver->partial_refactor(new_J, affected_cols);
        }

        if (!refactored_via_partial) {
            if (!solver->factorize(new_J)) {
                throw std::runtime_error(
                    "PwlStateSpaceCache::refactor_parametric: full "
                    "factorize fallback failed (numerically singular). "
                    "Check the updated parameter values.");
            }
        }

        // Mutate the cached segment's J + b in place so subsequent
        // solve() calls return the post-update result.
        seg_J = std::move(new_J);
        seg_b = std::move(new_b);
        return refactored_via_partial;
    }

    /// Compute the DEDUPED list of MNA column indices that change
    /// between two switch masks. For each switch bit that differs, the
    /// affected columns come from
    /// `DevicePool::columns_affected_by_switch(sw_idx, graph)` (the
    /// switch's two terminal nodes, skipping ground anchors). Multiple
    /// switches sharing a node produce a single deduplicated entry.
    ///
    /// v1.4.0 (openspec/changes/add-generalised-path-refactor Part A):
    /// dedup is now done HERE rather than left to the solver, because
    /// the solver's `partial_refactor_count_path()` query needs a
    /// canonical input to give a meaningful answer to the
    /// `MAX_PATH_LENGTH_RATIO` gate. Dedup at the call site keeps the
    /// solver's API surface narrow (it just walks the set it's given).
    ///
    /// Consumed by `DirectSolver::partial_refactor` overrides that
    /// implement path-based re-elimination. Backends without
    /// `partial_refactor` support return false; `solve_rank1`'s
    /// fallback to full `factorize` engages transparently.
    [[nodiscard]] std::vector<Index> compute_changed_columns_(
        const topology::SwitchStateMask& prev_mask,
        const topology::SwitchStateMask& curr_mask) const {
        if (prev_mask.hamming_distance(curr_mask) == 0) {
            return {};
        }
        // Use a set for in-place dedup. Switches sharing a node (common
        // in half-bridge / full-bridge topologies where both legs
        // anchor to the same midpoint) would otherwise produce
        // duplicate column entries — fine for the solver's path-union
        // (its in_path bitmap dedupes anyway) but bad for the count
        // query which counts unique paths.
        std::set<Index> uniq;
        Size switch_idx = 0;
        for (Index b_id = 0; b_id < graph_.num_branches(); ++b_id) {
            const auto& branch = graph_.branch(b_id);
            if (branch.kind == topology::BranchKind::Switch) {
                if (prev_mask.get(switch_idx) != curr_mask.get(switch_idx)) {
                    if (branch.from >= 0) uniq.insert(branch.from);
                    if (branch.to   >= 0) uniq.insert(branch.to);
                }
                ++switch_idx;
            }
        }
        return std::vector<Index>(uniq.begin(), uniq.end());
    }

    const topology::Graph& graph_;
    DevicePool& pool_;
    numeric::Dictionary<topology::SwitchStateMask, PwlSegment>
        segments_;
    Real dt_ = Real{0};   // 0 = static-only build (V0)
    bool lazy_mode_ = false;

    // -------------------------------------------------------------
    // v2.0 Phase 1 — event-dt solver entries (`solve_at`).
    //
    // One entry per recently-evented mask, LRU-bounded at
    // `kMaxEventEntries`. Each stores the (G, C, b) split so a dt
    // change is `J = G + (1/dt)·C` + numeric factorize on the
    // entry's ONE symbolic analysis — replacing the v1.x
    // exact-Real-dt-keyed map that retained a full analyzed +
    // factorized segment per (mask, dt) pair forever.
    // `mutable` so `solve_at` (const) can populate on demand.
    // -------------------------------------------------------------
    struct EventEntry {
        sparse::Matrix G;          // static stamps for this mask
        sparse::Matrix C;          // 1/dt coefficients (mask-inv.)
        Vector b_constant;          // dt-independent source stamps
        std::unique_ptr<sparse::DirectSolver> solver;  // analyzed once
        Real current_dt = Real{-1};
        std::uint64_t lru_tick = 0;

        EventEntry() = default;
        EventEntry(EventEntry&&) noexcept = default;
        EventEntry& operator=(EventEntry&&) noexcept = default;
        EventEntry(const EventEntry&) = delete;
        EventEntry& operator=(const EventEntry&) = delete;
    };

    /// Recombine an event entry's split at `dt`. `dt > 0` is the
    /// hot path (`G + (1/dt)·C`); `dt <= 0` falls back to the V0
    /// static-only assembly (caps/inductors skipped — matching
    /// what the v1.x aux cache produced for a static solve_at).
    [[nodiscard]] sparse::Matrix combine_event_J_(
        const EventEntry& e,
        const topology::SwitchStateMask& mask,
        Real dt) const {
        if (dt > Real{0}) {
            sparse::Matrix J = e.G + (Real{1} / dt) * e.C;
            return J;
        }
        sparse::Matrix J;
        Vector b_unused;
        assemble_segment(graph_, pool_, mask, Real{0}, J, b_unused);
        sparse::compress_in_place(J);
        return J;
    }

    /// Erase the least-recently-used event entry. Called only when
    /// the table is full and a NEW mask arrives.
    void evict_lru_event_entry_() const {
        if (event_entries_.empty()) return;
        auto victim = event_entries_.begin();
        for (auto it = event_entries_.begin();
             it != event_entries_.end(); ++it) {
            if (it->second.lru_tick < victim->second.lru_tick) {
                victim = it;
            }
        }
        event_entries_.erase(victim);
    }

    mutable numeric::Dictionary<topology::SwitchStateMask,
                                 EventEntry> event_entries_;

    // Monotonic recency clock shared by the event-entry LRU and
    // the lazy segment LRU. `mutable` — bumped from const solve
    // paths.
    mutable std::uint64_t lru_clock_ = 0;

    // Layer 4 V8: sliding solver + state for the rank-1 fast-path
    // (`solve_rank1`). Independent of `segments_`. `mutable` so
    // `solve_rank1` (const) can update them.
    mutable std::unique_ptr<sparse::DirectSolver> rank1_solver_;
    mutable topology::SwitchStateMask rank1_mask_{};
    mutable Vector                     rank1_b_constant_;
    mutable bool                       rank1_initialized_ = false;
    mutable sparse::Backend            rank1_backend_hint_ = sparse::Backend::Auto;

    // Layer 4 V8 telemetry. Atomic so a background thread can sample
    // mid-simulation without locking; const-correct for use inside
    // const `solve_rank1` and const `metrics()`.
    mutable std::atomic<std::uint64_t> rank1_hits_{0};
    mutable std::atomic<std::uint64_t> multi_bit_rank1_hits_{0};
    mutable std::atomic<std::uint64_t> full_refactor_hits_{0};
    mutable std::atomic<std::uint64_t> fallbacks_{0};

    // v2.0 Phase 1 telemetry — same atomic-sampling contract.
    mutable std::atomic<std::uint64_t> segment_evictions_{0};
    mutable std::atomic<std::uint64_t> event_hits_{0};
    mutable std::atomic<std::uint64_t> event_refactors_{0};
    mutable std::atomic<std::uint64_t> event_builds_{0};

    // v2.0 Phase 1 — lazy segment cache byte budget (audit finding
    // `no-mode-cache-eviction`). 0 disables eviction. The default
    // is deliberately generous: far above any realistic per-mode
    // working set, so behaviour is unchanged for normal runs while
    // a mode-random-walking week-long simulation can no longer grow
    // RSS without bound. Applies to LAZY mode only — an eager
    // `build()` is an explicit request to hold all 2^N factors,
    // and evicting one would break the eager `lookup()` contract.
    std::size_t segment_budget_bytes_ = kDefaultSegmentBudgetBytes;
};

}  // namespace pulsim::pwl
