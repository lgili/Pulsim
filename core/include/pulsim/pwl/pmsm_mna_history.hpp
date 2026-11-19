#pragma once

// =============================================================================
// Pulsim — per-step state for the MNA-native PMSM
// =============================================================================
//
// The machine's electrical state is the flux linkage λ_abc, and the
// trapezoidal rule on v_k = R i_k + dλ_k/dt needs, per phase, the
// previous step's (λ_k, i_k, v_k):
//
//     (λ_k,new − λ_k,old) = (h/2)·[(v_k,new − R i_k,new)
//                                  + (v_k,old − R i_k,old)]
//
// λ_old is committed from the SAME law the stamp used (L(θ)·i +
// λ_pm(θ) at the converged point), never re-integrated, so the
// committed flux cannot disagree with the currents the circuit was
// actually solved with. Snapshot/restore exists so a rejected or
// halved step can roll the flux back — the saturable inductor's
// history lacks that and therefore has to refuse dt-halving; this
// one does not have to.
//
// TR-BDF2 SECOND STAGE. Unlike the two charge-state devices, where
// only a coefficient moves, here the two stages have DIFFERENT
// SHAPES. Trapezoidal carries the previous derivative:
//
//     (v − R i) + (dλ/dt)_n − (2/h)·(λ − λ_n) = 0
//
// BDF2 is one-sided and carries none of it:
//
//     (v − R i) − (c1·λ + c2·λ_γ + c3·λ_n)/h = 0
//
// so the previous-derivative term is not merely rescaled, it is
// ABSENT. That is why the derivative is now stored explicitly as
// `dlam_old` rather than reconstructed from (v_old, i_old) at the
// stamp: the two stages read one committed quantity instead of each
// deriving its own, and cannot disagree about what the last step
// did.
//
// INDUCTION MACHINES (add-mna-native-machines group 2) share this
// history: their stator rows are the same flux rows with
// λ_abc = L·i + M·λ_r, a CONSTANT L and the rotor flux λ_r read from
// two internal nodes (1 F capacitors the engine integrates itself).
// An entry with `induction == true` reads `im` instead of `params`;
// every commit / snapshot / stage path below is the PMSM's, so the
// engines need no second plumbing. PMSM entries come first, then the
// induction machines, in insertion order (the flat snapshot relies on
// it).

#include "pulsim/models/induction_machine_mna.hpp"
#include "pulsim/models/pmsm_mna.hpp"
#include "pulsim/device/device_state.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"

#include <array>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::pwl {

class PmsmMnaHistory {
public:
    struct Entry {
        std::array<Index, 3> branch_id{};
        std::array<Index, 3> phase_node{};   //!< a, b, c terminals
        Index neutral = -1;
        std::array<Index, 3> cur_row{};       //!< branch-current unknowns
        Index omega_node = -1;
        Index theta_node = -1;
        models::PmsmMna::Params params;
        std::array<Real, 3> lambda_old{};
        std::array<Real, 3> i_old{};
        std::array<Real, 3> v_old{};          //!< v_phase − v_neutral
        //! dλ/dt at the last committed point (= v_old − R·i_old).
        //! The trapezoidal residual reads it; the BDF2 one does not.
        std::array<Real, 3> dlam_old{};
        //! INCREMENT FORM. The flux row reads λ(x) − λ_old as
        //! Δλ(x; ref) − lam_off: Δλ from (Δi, Δθ, Δλ_r) against the
        //! reference point (i_old, theta_old, lr_old) in closed form
        //! (`delta_lambda`), so (2/h)·(λ − λ_old) carries the round-off
        //! of the increment, not eps·(2/h)·|λ|. `lam_off` is nonzero only
        //! between an SDIRK `extrapolate_previous` and the next commit.
        Real theta_old = 0;                    //!< θ node voltage (PMSM)
        std::array<Real, 2> lr_old{};          //!< rotor-flux nodes (IM)
        std::array<Real, 3> lam_off{};
        //! λ_γ − λ_old in the same form, for the BDF2 stage.
        std::array<Real, 3> dlam_gamma{};
        //! Induction machine: rotor-flux nodes and parameters.
        bool induction = false;
        std::array<Index, 2> rotor_node{-1, -1};
        models::InductionMachineMna::Params im{};
        //! Capacitance to ground on the θ node (PMSM) and the rotor-flux
        //! nodes (IM): the builder's 1 F integrators. The stamp injects
        //! C·rate so the node voltage integrates the rate whatever C a
        //! with_value / YAML edit gave them (C ≠ 1 was silently wrong).
        Real c_theta = 1;
        std::array<Real, 2> c_rotor{1, 1};
        //! IM: the constant stator matrices of `im`, built once per run.
        models::InductionMachineMna::StatorMatrices im_sm{};
        std::string name;   //!< phase-a branch name (flux-map refusals)
    };

    /// Σ C of the linear capacitors between `node` and ground. Any other
    /// charge-storing branch on the node (a capacitor to another node, a
    /// nonlinear capacitor) would integrate the rate with a different
    /// weight: refused by name, as is a node with no capacitor.
    [[nodiscard]] static Real cap_to_ground(const topology::Graph& g,
                                            const DevicePool& pool,
                                            Index node, const char* what) {
        if (node < 0) return Real{1};
        using SK = DevicePool::StoredKind;
        Real c = 0;
        for (Index b = 0; b < g.num_branches(); ++b) {
            if (!pool.is_registered(b)) continue;
            const auto& br = g.branch(b);
            if (br.from != node && br.to != node) continue;
            const auto k = pool.kind_of(b);
            if (k != SK::Capacitor && k != SK::NonlinearCapacitor) continue;
            if (k == SK::Capacitor && (br.from < 0 || br.to < 0)) {
                c += pool.capacitor_params(b).C;
                continue;
            }
            throw std::invalid_argument(
                std::string{what} + " node '" + g.node(node).name
                + "' carries the capacitor '" + std::string{g.branch_name(b)}
                + "' that is not a linear capacitor to ground; the node "
                  "integrates the machine's own state and admits none");
        }
        if (!(c > Real{0})) {
            throw std::invalid_argument(
                std::string{what} + " node '" + g.node(node).name
                + "' has no capacitor to ground to integrate it");
        }
        return c;
    }

    void init(const topology::Graph& graph, const DevicePool& pool) {
        entries_.clear();
        for (const auto& m : pool.pmsm_mna_machines()) {
            Entry e;
            e.branch_id = m.branch_id;
            for (Size k = 0; k < 3; ++k) {
                const auto& br = graph.branch(m.branch_id[k]);
                e.phase_node[k] = br.from;
                e.neutral = br.to;
                e.cur_row[k] =
                    pool.branch_var_id_for_inductor(m.branch_id[k],
                                                    graph);
            }
            e.omega_node = m.omega_node;
            e.theta_node = m.theta_node;
            e.params = m.params;
            e.c_theta = cap_to_ground(graph, pool, m.theta_node,
                                      "PMSM angle");
            // At rest the PM flux is already linked; start λ at
            // λ_pm(θ = 0), not at zero. The increment-form reference
            // (θ_old, i_old) is taken from x at the start, so a θ node
            // IC ≠ 0 does not jump the flux (test_machine_dc_hold).
            e.lambda_old = models::PmsmMna::lambda_pm(
                e.params, Real{0}, 0);
            e.name = std::string{graph.branch_name(m.branch_id[0])};
            if (e.params.flux_map) {   // λ(i = 0, θ = 0) of the map
                e.lambda_old = models::PmsmMna::flux_map_state(
                    e.params, Real{0}, {}).lam;
            }
            entries_.push_back(e);
        }
        for (const auto& m : pool.induction_machines()) {
            Entry e;
            e.induction = true;
            e.branch_id = m.branch_id;
            for (Size k = 0; k < 3; ++k) {
                const auto& br = graph.branch(m.branch_id[k]);
                e.phase_node[k] = br.from;
                e.neutral = br.to;
                e.cur_row[k] =
                    pool.branch_var_id_for_inductor(m.branch_id[k],
                                                    graph);
            }
            e.omega_node = m.omega_node;
            e.rotor_node = m.rotor_node;
            e.im = m.params;
            e.im_sm = models::InductionMachineMna::stator(e.im);
            for (Size r = 0; r < 2; ++r) {
                e.c_rotor[r] = cap_to_ground(graph, pool, m.rotor_node[r],
                                             "induction-machine rotor-flux");
            }
            entries_.push_back(e);   // λ_old = 0: no current, no rotor flux
        }
    }

    [[nodiscard]] static Real r_s(const Entry& e) noexcept {
        return e.induction ? e.im.R_s : e.params.R_s;
    }

    [[nodiscard]] bool empty() const noexcept {
        return entries_.empty();
    }
    [[nodiscard]] const std::vector<Entry>& entries() const noexcept {
        return entries_;
    }

    [[nodiscard]] static Real theta_e_of(const Entry& e,
                                         const Vector& x) noexcept {
        return e.params.pole_pairs
               * stamping::read_node_voltage(x, e.theta_node);
    }

    /// λ from the model's law at currents i, angle node θ (PMSM) and
    /// rotor flux lr (IM). `lambda_at` and `rebase_reference` both use it,
    /// so a commit and a resume evaluate the same expressions.
    [[nodiscard]] static std::array<Real, 3> lambda_of(
        const Entry& e, const std::array<Real, 3>& i, Real theta_node,
        const std::array<Real, 2>& lr) {
        std::array<Real, 3> lam{};
        if (e.induction) {
            for (Size k = 0; k < 3; ++k) {
                Real acc = e.im_sm.M[k][0] * lr[0] + e.im_sm.M[k][1] * lr[1];
                for (Size j = 0; j < 3; ++j) acc += e.im_sm.L[k][j] * i[j];
                lam[k] = acc;
            }
            return lam;
        }
        const Real th = e.params.pole_pairs * theta_node;
        if (e.params.flux_map) {
            return models::PmsmMna::flux_map_state(e.params, th, i).lam;
        }
        const auto ind = models::PmsmMna::inductance(e.params, th);
        const auto lpm = models::PmsmMna::lambda_pm(e.params, th, 0);
        for (Size k = 0; k < 3; ++k) {
            lam[k] = lpm[k];
            for (Size j = 0; j < 3; ++j) lam[k] += ind.L[k][j] * i[j];
        }
        return lam;
    }

    /// λ at the state in `x`, from the model's own law.
    [[nodiscard]] static std::array<Real, 3> lambda_at(
        const Entry& e, const Vector& x) {
        return lambda_of(
            e, {x[e.cur_row[0]], x[e.cur_row[1]], x[e.cur_row[2]]},
            stamping::read_node_voltage(x, e.theta_node),
            {stamping::read_node_voltage(x, e.rotor_node[0]),
             stamping::read_node_voltage(x, e.rotor_node[1])});
    }

    /// λ(x) − λ(reference) in closed form (no `lam_off`). `L_at_x`: the
    /// PMSM's L(θ_e(x)) when the caller has it (the refresh), else null.
    [[nodiscard]] static std::array<Real, 3> delta_lambda(
        const Entry& e, const Vector& x,
        const models::PmsmMna::Mat3* L_at_x = nullptr) {
        std::array<Real, 3> di{};
        for (Size k = 0; k < 3; ++k) di[k] = x[e.cur_row[k]] - e.i_old[k];
        if (e.induction) {
            const auto& sm = e.im_sm;
            const Real d0 =
                stamping::read_node_voltage(x, e.rotor_node[0]) - e.lr_old[0];
            const Real d1 =
                stamping::read_node_voltage(x, e.rotor_node[1]) - e.lr_old[1];
            std::array<Real, 3> out{};
            for (Size k = 0; k < 3; ++k) {
                Real acc = sm.M[k][0] * d0 + sm.M[k][1] * d1;
                for (Size j = 0; j < 3; ++j) acc += sm.L[k][j] * di[j];
                out[k] = acc;
            }
            return out;
        }
        const Real pp = e.params.pole_pairs;
        const Real th = stamping::read_node_voltage(x, e.theta_node);
        const Real th_e = pp * th;
        if (e.params.flux_map) {
            return models::PmsmMna::flux_map_increment(
                e.params, th_e, pp * e.theta_old, pp * (th - e.theta_old),
                e.i_old, di);
        }
        const auto L = L_at_x != nullptr
                           ? *L_at_x
                           : models::PmsmMna::inductance(e.params, th_e).L;
        return models::PmsmMna::lambda_increment(
            e.params, L, th_e, pp * e.theta_old, pp * (th - e.theta_old),
            e.i_old, di);
    }

    /// Snapshot λ at the γ stage point; the BDF2 history term reads
    /// it. Must run between the two stages.
    void capture_gamma(const Vector& x_gamma) {
        for (auto& e : entries_) {
            const auto d = delta_lambda(e, x_gamma);
            for (Size k = 0; k < 3; ++k) e.dlam_gamma[k] = d[k] - e.lam_off[k];
        }
    }

    /// The increment form's reference point = the state in x, offset 0.
    static void set_reference(Entry& e, const Vector& x) {
        for (Size k = 0; k < 3; ++k) e.i_old[k] = x[e.cur_row[k]];
        e.theta_old = stamping::read_node_voltage(x, e.theta_node);
        for (Size r = 0; r < 2; ++r) {
            e.lr_old[r] = stamping::read_node_voltage(x, e.rotor_node[r]);
        }
        e.lam_off = {};
    }

    /// Resume: the flat block carries (λ, i, v); the reference angle and
    /// rotor flux are node voltages of the snapshot's x. The snapshot's
    /// λ_old stays the flux the row integrates from: the offset is
    /// λ_old − λ(i_old, θ(x), λ_r(x)), bitwise 0 for a snapshot taken at
    /// a commit (same law, same inputs) and the stored flux otherwise.
    void rebase_reference(const Vector& x) {
        for (auto& e : entries_) {
            e.theta_old = stamping::read_node_voltage(x, e.theta_node);
            for (Size r = 0; r < 2; ++r) {
                e.lr_old[r] = stamping::read_node_voltage(x, e.rotor_node[r]);
            }
            const auto lam = lambda_of(e, e.i_old, e.theta_old, e.lr_old);
            for (Size k = 0; k < 3; ++k) e.lam_off[k] = e.lambda_old[k] - lam[k];
        }
    }

    /// Commit the converged step: (λ, i, v) at x, with λ from the
    /// model's own law at this θ.
    void update_from_state(
        const Vector& x, Real /*h*/,
        TrBdf2Stage /*stage*/ = TrBdf2Stage::Trapezoidal) {
        for (auto& e : entries_) {
            if (!e.induction && e.params.flux_map) check_flux_map_band_(e, x);
            const auto lam = lambda_at(e, x);
            const Real v_n = stamping::read_node_voltage(x, e.neutral);
            set_reference(e, x);
            for (Size k = 0; k < 3; ++k) {
                const Real i = x[e.cur_row[k]];   // = i_old (set_reference)
                e.lambda_old[k] = lam[k];
                e.v_old[k] =
                    stamping::read_node_voltage(x, e.phase_node[k]) - v_n;
                // dλ/dt from the device equation itself, exact at
                // the converged point under either stage.
                e.dlam_old[k] = e.v_old[k] - r_s(e) * i;
            }
        }
    }

    /// SDIRK restart: pseudo previous flux λ0 + c·(λ1 − λ0), λ1 from
    /// the model's law at the stage-1 solve.
    /// An accepted state of a flux-map machine must lie on the map's grid
    /// plus its validity band (`FluxMapDQ::in_band`): past it the
    /// extension is unmeasured and can lose positive definiteness.
    static void check_flux_map_band_(const Entry& e, const Vector& x) {
        const std::array<Real, 3> i{x[e.cur_row[0]], x[e.cur_row[1]], x[e.cur_row[2]]};
        const auto st = models::PmsmMna::flux_map_state(
            e.params, theta_e_of(e, x), i);
        const auto& fm = *e.params.flux_map;
        if (!fm.in_band(st.i_d, st.i_q)) {
            throw std::invalid_argument(std::format(
                "PMSM '{}': the accepted state (i_d, i_q) = ({}, {}) A left "
                "flux map '{}' (grid i_d [{}, {}], i_q [{}, {}] A plus {} x "
                "the span); extend the table", e.name, st.i_d, st.i_q,
                fm.source, fm.kd.front(), fm.kd.back(), fm.kq.front(),
                fm.kq.back(), models::FluxMapDQ::kBand));
        }
    }

    void extrapolate_previous(const Vector& x1, Real c) {
        for (auto& e : entries_) {
            const auto lam = lambda_at(e, x1);
            const auto d = delta_lambda(e, x1);
            for (Size k = 0; k < 3; ++k) {
                e.lambda_old[k] += c * (lam[k] - e.lambda_old[k]);
                e.lam_off[k] += c * (d[k] - e.lam_off[k]);
            }
        }
    }

    /// TR-BDF2 post-event refresh: the phase voltages and dλ/dt from
    /// the probe; (λ, i) are not touched.
    void refresh_derivatives(const Vector& x_probe) {
        for (auto& e : entries_) {
            const Real v_n =
                stamping::read_node_voltage(x_probe, e.neutral);
            for (Size k = 0; k < 3; ++k) {
                e.v_old[k] =
                    stamping::read_node_voltage(x_probe, e.phase_node[k])
                    - v_n;
                e.dlam_old[k] = e.v_old[k] - r_s(e) * e.i_old[k];
            }
        }
    }

    /// Seed from a warm-start / DC operating point — same fields as
    /// a commit, so the first stamp sees a consistent previous step.
    void seed_from_dc_op(const Vector& x) { update_from_state(x, Real{0}); }

    // ---- device::StatefulKind (device/kind.hpp) ----
    static constexpr device::StateBlock kBlock = device::StateBlock::PmsmMna;
    static constexpr DevicePool::StoredKind kStoredKind =
        DevicePool::StoredKind::PmsmMna;
    static constexpr Size kFlatPerDevice = 9;
    /// Every real an entry carries from step to step, the snapshot's
    /// (`to_flat` order) first, then the stage scratch: the in-run
    /// rollback copies exactly these.
    template <class E, class F> static void for_each_state(E& e, F&& f) {
        f(e.lambda_old[0]);
        f(e.lambda_old[1]);
        f(e.lambda_old[2]);
        f(e.i_old[0]);
        f(e.i_old[1]);
        f(e.i_old[2]);
        f(e.v_old[0]);
        f(e.v_old[1]);
        f(e.v_old[2]);
        f(e.dlam_old[0]);
        f(e.dlam_old[1]);
        f(e.dlam_old[2]);
        // increment-form reference point and offsets
        f(e.theta_old);
        f(e.lr_old[0]);
        f(e.lr_old[1]);
        f(e.lam_off[0]);
        f(e.lam_off[1]);
        f(e.lam_off[2]);
        f(e.dlam_gamma[0]);
        f(e.dlam_gamma[1]);
        f(e.dlam_gamma[2]);
    }
    [[nodiscard]] std::vector<Entry>& mutable_entries() noexcept {
        return entries_;
    }
    void commit(const Vector& x, Real h, TrBdf2Stage stage) {
        update_from_state(x, h, stage);
    }
    /// The slot refresh reads the probe's voltages, not its step.
    void refresh_derivatives(const Vector& x_probe, Real /*h*/) {
        refresh_derivatives(x_probe);
    }

    /// Flat (λ, i, v) triples per machine. `dlam_old` is not
    /// carried — it is a function of the (v, i) that are — and
    /// `dlam_gamma` lives only between the two stages of one step,
    /// while a snapshot is always taken at a step boundary.
    [[nodiscard]] std::vector<Real> to_flat() const {
        std::vector<Real> out;
        out.reserve(entries_.size() * 9);
        for (const auto& e : entries_) {
            for (Size k = 0; k < 3; ++k) out.push_back(e.lambda_old[k]);
            for (Size k = 0; k < 3; ++k) out.push_back(e.i_old[k]);
            for (Size k = 0; k < 3; ++k) out.push_back(e.v_old[k]);
        }
        return out;
    }
    void from_flat(const std::vector<Real>& flat) {
        if (flat.size() != entries_.size() * 9) {
            throw std::invalid_argument(
                "PmsmMnaHistory::from_flat: expected "
                + std::to_string(entries_.size() * 9) + " values but got "
                + std::to_string(flat.size()));
        }
        Size p = 0;
        for (auto& e : entries_) {
            for (Size k = 0; k < 3; ++k) e.lambda_old[k] = flat[p++];
            for (Size k = 0; k < 3; ++k) e.i_old[k] = flat[p++];
            for (Size k = 0; k < 3; ++k) e.v_old[k] = flat[p++];
            for (Size k = 0; k < 3; ++k) {
                e.dlam_old[k] = e.v_old[k] - r_s(e) * e.i_old[k];
            }
        }
    }

private:
    std::vector<Entry> entries_;
};

}  // namespace pulsim::pwl
