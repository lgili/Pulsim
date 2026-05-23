#pragma once

// =============================================================================
// Pulsim — Layer 6 V0: CircuitBuilder (high-level circuit construction)
// =============================================================================
//
// `pulsim-v2-builder-api` Phase 1.
//
// User-facing wrapper that hides the two-object `Graph +
// DevicePool` split. The builder accepts string node names,
// SI-unit parameter values, and named devices; internally it
// auto-tracks indices, dispatches to the kernel's `add_branch`
// + `pool.add_*` methods, and exposes `graph()` / `pool()`
// const-refs for `PwlStateSpaceCache` and `run_transient`.
//
// Design constraints:
//   * Pure additive wrapper. NO numerical work.
//   * Header-only (consistent with the rest of v2).
//   * No exceptions on duplicate device names (V0 — V1 may
//     add validation).
//   * "gnd" / "GND" / "0" all map to `Graph::ground()`.
//   * Case-sensitive for non-ground names.

#include "pulsim/models/capacitor.hpp"
#include "pulsim/models/current_source.hpp"
#include "pulsim/models/ideal_diode.hpp"
#include "pulsim/models/inductor.hpp"
#include "pulsim/models/igbt_level1.hpp"
#include "pulsim/models/mosfet_level1.hpp"
#include "pulsim/models/vcvs.hpp"
#include "pulsim/models/pulse_voltage_source.hpp"
#include "pulsim/models/pwm_voltage_source.hpp"
#include "pulsim/models/sine_voltage_source.hpp"
#include "pulsim/models/resistor.hpp"
#include "pulsim/models/transformer.hpp"
#include "pulsim/models/voltage_source.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace pulsim::builder {

class CircuitBuilder {
public:
    CircuitBuilder() = default;

    /// Return the node index for `name`, creating it if not
    /// yet registered. `"gnd"` / `"GND"` / `"0"` all map to
    /// `graph().ground()` without consuming a node slot.
    Index node(std::string name) {
        return resolve_node_(name);
    }

    // -------- Add methods --------------------------------------------------

    CircuitBuilder& add_voltage_source(
        std::string /*name*/, std::string from,
        std::string to, Real V) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::Source);
        pool_.add_voltage_source(
            b_id, models::VoltageSource::Params{V});
        return *this;
    }

    /// Add a PWM voltage source (Layer 2 V4) — a square-
    /// wave switching between `v_high` and `v_low` at the
    /// given `frequency` and `duty` cycle.
    ///
    /// Eliminates the common SMPS pattern of writing a
    /// custom `b_extra_fn(t)` lambda for PWM gate drives.
    /// `run_transient` automatically overlays the PWM
    /// value at each timestep.
    ///
    /// Parameters:
    ///   v_high     [V]  output during ON portion
    ///   v_low      [V]  output during OFF portion
    ///   frequency  [Hz] switching frequency
    ///   duty       [-]  ON-time fraction ∈ [0, 1]
    ///   phase      [s]  start-of-cycle offset (default 0)
    CircuitBuilder& add_pwm_voltage_source(
        std::string /*name*/, std::string from,
        std::string to, Real v_high, Real v_low,
        Real frequency, Real duty,
        Real phase = Real{0}) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::Source);
        pool_.add_pwm_voltage_source(
            b_id, models::PWMVoltageSource::Params{
                .v_high     = v_high,
                .v_low      = v_low,
                .frequency  = frequency,
                .duty       = duty,
                .phase      = phase,
            });
        return *this;
    }

    /// Add a sinusoidal AC voltage source (Layer 2 V11).
    ///   v_dc         [V]   DC offset
    ///   v_amplitude  [V]   peak amplitude of sine wave
    ///   frequency    [Hz]  fundamental frequency
    ///   phase        [rad] phase angle (default 0)
    /// Output: v(t) = v_dc + v_amplitude · sin(2π·f·t + φ).
    CircuitBuilder& add_sine_voltage_source(
        std::string /*name*/, std::string from,
        std::string to,
        Real v_dc, Real v_amplitude,
        Real frequency, Real phase = Real{0}) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::Source);
        pool_.add_sine_voltage_source(
            b_id, models::SineVoltageSource::Params{
                .v_dc        = v_dc,
                .v_amplitude = v_amplitude,
                .frequency   = frequency,
                .phase       = phase,
            });
        return *this;
    }

    /// Add a 3-terminal SH1 MOSFET (Layer 2 V13). The drain-
    /// source path is a `BranchKind::Nonlinear` branch; the
    /// gate is just a node reference (no current flow on
    /// Level 1 — ideal gate). Newton stamps both the drain-
    /// source and drain-gate Jacobian off-diagonals.
    ///
    ///   K           [A/V²] transconductance parameter
    ///   V_T         [V]    threshold voltage
    ///   lambda      [1/V]  channel-length modulation
    ///   kappa       [1/V]  cutoff-region sigmoid sharpness
    CircuitBuilder& add_mosfet_level1(
        std::string /*name*/,
        std::string drain, std::string source,
        std::string gate,
        Real K, Real V_T,
        Real lambda = Real{0.02},
        Real kappa  = Real{15.0},
        bool with_body_diode = false) {
        const Index drain_idx  = resolve_node_(drain);
        const Index source_idx = resolve_node_(source);
        const Index gate_idx   = resolve_node_(gate);
        const Index b_id = graph_.add_branch(
            drain_idx, source_idx,
            topology::BranchKind::Nonlinear);
        pool_.add_mosfet_level1(
            b_id, gate_idx,
            models::MosfetLevel1::Params{
                .K = K, .V_T = V_T,
                .lambda = lambda, .kappa = kappa,
            });
        // V18 ergonomics: optional anti-parallel body diode
        // (real MOSFETs have one physically). Conducts when
        // V_drain < V_source − V_F, clamping V_DS to ~−0.5V.
        // Required to keep Newton out of the spurious-V_DS<0
        // root of the SH1 triode polynomial during transient
        // events (e.g. inductive loads during dead-time).
        if (with_body_diode) {
            const Index body_b = graph_.add_branch(
                source_idx, drain_idx,
                topology::BranchKind::Switch);
            pool_.add_diode(body_b,
                /*g_on=*/Real{1e3},
                /*g_off=*/Real{1e-9},
                /*V_th=*/Real{0.5});
        }
        return *this;
    }

    /// Add a 4-terminal VCVS (Layer 2 V15):
    ///   V(out_pos) − V(out_neg) = gain · (V(in_pos) − V(in_neg))
    /// Use a high gain (e.g. 1e5) plus negative feedback for
    /// an ideal-op-amp approximation.
    CircuitBuilder& add_vcvs(
        std::string /*name*/,
        std::string in_pos,  std::string in_neg,
        std::string out_pos, std::string out_neg,
        Real gain) {
        const Index inp_idx = resolve_node_(in_pos);
        const Index inn_idx = resolve_node_(in_neg);
        const Index outp_idx = resolve_node_(out_pos);
        const Index outn_idx = resolve_node_(out_neg);
        const Index b_id = graph_.add_branch(
            outp_idx, outn_idx,
            topology::BranchKind::Source);
        pool_.add_vcvs(
            b_id, inp_idx, inn_idx,
            models::VCVS::Params{.gain = gain});
        return *this;
    }

    /// Add an IDEAL OP-AMP: high-gain VCVS with single-ended
    /// output (out_neg = gnd). Default gain = 10⁵ (≈ open-
    /// loop typical for compensated devices). Combine with
    /// negative feedback to enforce the "virtual short"
    /// V_in_pos ≈ V_in_neg.
    CircuitBuilder& add_op_amp_ideal(
        std::string /*name*/,
        std::string in_pos, std::string in_neg,
        std::string out,
        Real gain = Real{1e5}) {
        return add_vcvs(/*name=*/"", in_pos, in_neg,
                          out, /*out_neg=*/"gnd", gain);
    }

    /// Add a 3-terminal IGBT Level 1 (Layer 2 V14). Same
    /// architectural pattern as `add_mosfet_level1`. The
    /// collector→emitter is a Nonlinear branch; gate is a
    /// node reference (no gate current — ideal gate).
    ///   V_CE_sat    [V] saturation voltage (knee)
    ///   R_CE_sat    [Ω] on-state slope resistance
    ///   V_T         [V] gate threshold
    ///   kappa       [1/V] cutoff sigmoid sharpness
    CircuitBuilder& add_igbt_level1(
        std::string /*name*/,
        std::string collector, std::string emitter,
        std::string gate,
        Real V_CE_sat = Real{1.5},
        Real R_CE_sat = Real{0.05},
        Real V_T      = Real{5.0},
        Real kappa    = Real{10.0}) {
        const Index c_idx = resolve_node_(collector);
        const Index e_idx = resolve_node_(emitter);
        const Index g_idx = resolve_node_(gate);
        const Index b_id = graph_.add_branch(
            c_idx, e_idx,
            topology::BranchKind::Nonlinear);
        pool_.add_igbt_level1(
            b_id, g_idx,
            models::IgbtLevel1::Params{
                .V_CE_sat = V_CE_sat,
                .R_CE_sat = R_CE_sat,
                .V_T      = V_T,
                .kappa    = kappa,
            });
        return *this;
    }

    /// Add a pulse / step voltage source (Layer 2 V12).
    ///   v_initial    [V] baseline (before & between pulses)
    ///   v_pulsed     [V] level during the pulse window
    ///   t_start      [s] delay before first pulse fires
    ///   pulse_width  [s] duration of each pulse
    ///   period       [s] repetition period; 0 → single-shot
    CircuitBuilder& add_pulse_voltage_source(
        std::string /*name*/, std::string from,
        std::string to,
        Real v_initial, Real v_pulsed,
        Real t_start, Real pulse_width,
        Real period    = Real{0},
        Real rise_time = Real{0},
        Real fall_time = Real{0}) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::Source);
        pool_.add_pulse_voltage_source(
            b_id, models::PulseVoltageSource::Params{
                .v_initial   = v_initial,
                .v_pulsed    = v_pulsed,
                .t_start     = t_start,
                .pulse_width = pulse_width,
                .period      = period,
                .rise_time   = rise_time,
                .fall_time   = fall_time,
            });
        return *this;
    }

    /// Add a constant DC current source (Layer 2 V3).
    /// `I` (amperes) flows FROM `from` TO `to`. Positive `I`
    /// means conventional current direction.
    ///
    /// Unlike voltage sources, current sources do NOT add a
    /// branch-current unknown to the state vector — the
    /// current is fixed at I. Useful for bias currents,
    /// photovoltaic models, dq-frame stator excitation,
    /// Norton equivalents, etc.
    ///
    /// For TIME-VARYING current (e.g. sinusoidal injection),
    /// use this with `I = 0` baseline and modulate via
    /// `b_extra_fn(t)` — same pattern as time-varying
    /// VoltageSource.
    CircuitBuilder& add_current_source(
        std::string /*name*/, std::string from,
        std::string to, Real I) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::Source);
        pool_.add_current_source(
            b_id, models::CurrentSource::Params{I});
        return *this;
    }

    CircuitBuilder& add_resistor(
        std::string /*name*/, std::string from,
        std::string to, Real R_ohms) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::PassiveLinear);
        pool_.add_resistor(
            b_id, models::Resistor::Params{
                .G = Real{1} / R_ohms});
        return *this;
    }

    CircuitBuilder& add_capacitor(
        std::string /*name*/, std::string from,
        std::string to, Real C_farads) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::PassiveLinear);
        pool_.add_capacitor(
            b_id, models::Capacitor::Params{C_farads});
        return *this;
    }

    CircuitBuilder& add_inductor(
        std::string /*name*/, std::string from,
        std::string to, Real L_henries) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::PassiveLinear);
        pool_.add_inductor(
            b_id, models::Inductor::Params{L_henries});
        return *this;
    }

    /// Add a binary switched diode (Layer 5 V2's
    /// `SwitchedDiode`). The branch uses
    /// `BranchKind::Switch` (the diode behaves as a switch
    /// from the topology's perspective).
    CircuitBuilder& add_diode(
        std::string /*name*/, std::string anode,
        std::string cathode, Real g_on, Real g_off,
        Real V_th = Real{0}) {
        const Index a_idx = resolve_node_(anode);
        const Index k_idx = resolve_node_(cathode);
        const Index b_id = graph_.add_branch(
            a_idx, k_idx,
            topology::BranchKind::Switch);
        pool_.add_diode(b_id, g_on, g_off, V_th);
        return *this;
    }

    /// Add a smooth-blend `IdealDiode` (Layer 4 V3's
    /// AD-driven nonlinear model). The branch uses
    /// `BranchKind::Nonlinear`.
    CircuitBuilder& add_nonlinear_diode(
        std::string /*name*/, std::string anode,
        std::string cathode,
        models::IdealDiode::Params params) {
        const Index a_idx = resolve_node_(anode);
        const Index k_idx = resolve_node_(cathode);
        const Index b_id = graph_.add_branch(
            a_idx, k_idx,
            topology::BranchKind::Nonlinear);
        pool_.add_nonlinear_diode(b_id, params);
        return *this;
    }

    /// Add a controlled switch (drives by `switch_fn`
    /// at simulation time).
    CircuitBuilder& add_switch(
        std::string /*name*/, std::string from,
        std::string to, Real g_on, Real g_off) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::Switch);
        pool_.add_switch(b_id, g_on, g_off);
        return *this;
    }

    // -------- Power-device convenience helpers (Layer 2 V1) -----------------
    //
    // SMPS-realistic shorthands. They map to existing
    // primitives (`add_switch`, `add_diode`) but accept
    // user-friendly ohms / volts and pick sensible
    // defaults so a buck/boost/flyback prototype works
    // without parameter tuning.

    /// Add an n-channel power MOSFET as a single
    /// controlled switch (drain → source). No body diode.
    /// Defaults: R_on = 1 mΩ, R_off = 1 GΩ (typical for
    /// modern Si MOSFETs in SMPS applications).
    CircuitBuilder& add_mosfet(
        std::string name, std::string drain,
        std::string source,
        Real R_on  = Real{1e-3},
        Real R_off = Real{1e9}) {
        return add_switch(std::move(name),
                            std::move(drain),
                            std::move(source),
                            Real{1} / R_on,
                            Real{1} / R_off);
    }

    /// Add an n-channel power MOSFET WITH its intrinsic
    /// anti-parallel body diode. Adds two branches:
    ///   1. switch (drain → source) with R_on / R_off
    ///   2. SwitchedDiode (source → drain) with V_F drop
    ///      that conducts during freewheeling intervals
    /// Defaults model a typical Si MOSFET: R_on = 1 mΩ,
    /// R_off = 1 GΩ, body-diode V_F = 0.7 V.
    CircuitBuilder& add_mosfet_with_body_diode(
        std::string name, std::string drain,
        std::string source,
        Real R_on        = Real{1e-3},
        Real R_off       = Real{1e9},
        Real V_F         = Real{0.7},
        Real g_on_diode  = Real{1e3},
        Real g_off_diode = Real{1e-9}) {
        const Index drain_idx  = resolve_node_(drain);
        const Index source_idx = resolve_node_(source);
        // Main switch (drain → source).
        const Index b_switch = graph_.add_branch(
            drain_idx, source_idx,
            topology::BranchKind::Switch);
        pool_.add_switch(b_switch,
                          Real{1} / R_on,
                          Real{1} / R_off);
        // Body diode (source → drain — anti-parallel).
        const Index b_body = graph_.add_branch(
            source_idx, drain_idx,
            topology::BranchKind::Switch);
        pool_.add_diode(b_body, g_on_diode,
                          g_off_diode, V_F);
        (void)name;
        return *this;
    }

    /// Add an IGBT as a single controlled switch
    /// (collector → emitter). No anti-parallel diode —
    /// discrete IGBTs typically don't include one; the
    /// user wires `add_diode(...)` separately if needed.
    /// Defaults: R_on = 10 mΩ, R_off = 1 GΩ (typical for
    /// IGBT modules).
    CircuitBuilder& add_igbt(
        std::string name, std::string collector,
        std::string emitter,
        Real R_on  = Real{10e-3},
        Real R_off = Real{1e9}) {
        return add_switch(std::move(name),
                            std::move(collector),
                            std::move(emitter),
                            Real{1} / R_on,
                            Real{1} / R_off);
    }

    /// Layer 2 V2 — two-winding linear transformer.
    /// Adds two coupled inductor branches:
    ///   * primary  (p_from → p_to)  with L_p
    ///   * secondary (s_from → s_to) with L_s
    ///   * mutual inductance M = k · √(L_p · L_s)
    /// Default k = 1 (ideal coupling, no leakage). For real
    /// transformers use k ∈ [0.9, 0.99] to model leakage.
    /// k = 0 makes the two windings independent (no
    /// transformer action — useful for testing).
    ///
    /// The transformer requires the cache to be built with
    /// dt > 0 (dynamic path); on the static path (dt = 0)
    /// the coupling has no effect and the inductors behave
    /// as open circuits, matching the standalone inductor
    /// model.
    CircuitBuilder& add_transformer(
        std::string name,
        std::string p_from, std::string p_to,
        std::string s_from, std::string s_to,
        Real L_p, Real L_s, Real k = Real{1}) {
        const Index p_from_idx = resolve_node_(p_from);
        const Index p_to_idx   = resolve_node_(p_to);
        const Index s_from_idx = resolve_node_(s_from);
        const Index s_to_idx   = resolve_node_(s_to);

        const Index p_branch = graph_.add_branch(
            p_from_idx, p_to_idx,
            topology::BranchKind::PassiveLinear);
        pool_.add_inductor(
            p_branch, models::Inductor::Params{L_p});

        const Index s_branch = graph_.add_branch(
            s_from_idx, s_to_idx,
            topology::BranchKind::PassiveLinear);
        pool_.add_inductor(
            s_branch, models::Inductor::Params{L_s});

        pool_.add_transformer_coupling(
            p_branch, s_branch,
            models::TwoWindingTransformer::Params{
                .L_p = L_p, .L_s = L_s, .k = k});

        (void)name;
        return *this;
    }

    /// Layer 2 V17: SaturableInductor — nonlinear L(i) device
    /// integrated via Newton refresh. The branch is added as
    /// `BranchKind::Nonlinear`; assemble skips it, and the
    /// Newton refresh stamps the nonlinear trap-rule
    /// constraint per iteration.
    ///
    /// Required call sequence in run_transient:
    ///   * Pass `enable_nonlinear_refresh=True` (or wire
    ///     `make_combined_nonlinear_refresh(history, dt)`
    ///     manually for non-Python callers).
    ///   * History is automatically initialized + updated by
    ///     the solver when the circuit contains saturable
    ///     inductors.
    CircuitBuilder& add_saturable_inductor(
        std::string /*name*/,
        std::string from, std::string to,
        Real L_0, Real I_sat,
        Real L_residual = Real{0}) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = graph_.add_branch(
            from_idx, to_idx,
            topology::BranchKind::Nonlinear);
        pool_.add_saturable_inductor(
            b_id,
            models::SaturableInductor::Params{
                .L_0 = L_0,
                .I_sat = I_sat,
                .L_residual = L_residual,
            });
        return *this;
    }

    /// Layer 2 V16: N-winding transformer (2 ≤ N ≤ 6). Each
    /// winding is added as a regular `Inductor` branch, and
    /// the N·(N−1)/2 pair-wise couplings are registered with
    /// the existing `transformer_couplings_` mechanism — a
    /// multi-winding transformer is mathematically equivalent
    /// to all pair-wise 2-winding couplings between its
    /// windings.
    ///
    /// `windings` is a list of `(from_node, to_node, L_i)`
    /// tuples. `k_ij` is the full N×N coupling matrix (only
    /// the upper triangle is read; diagonal entries are
    /// ignored). Defaults to k_ij = 1 for all off-diagonals
    /// if `k_matrix` is empty.
    struct WindingSpec {
        std::string from;
        std::string to;
        Real L;
    };

    CircuitBuilder& add_multi_winding_transformer(
        std::string /*name*/,
        const std::vector<WindingSpec>& windings,
        const std::vector<std::vector<Real>>& k_matrix = {}) {
        const Size N = windings.size();
        if (N < 2 || N > 6) {
            throw std::invalid_argument(
                "add_multi_winding_transformer: N must be in [2, 6]");
        }
        // Add each winding as a regular inductor branch.
        std::vector<Index> branch_ids;
        branch_ids.reserve(N);
        for (const auto& w : windings) {
            const Index from_idx = resolve_node_(w.from);
            const Index to_idx   = resolve_node_(w.to);
            const Index b = graph_.add_branch(
                from_idx, to_idx,
                topology::BranchKind::PassiveLinear);
            pool_.add_inductor(
                b, models::Inductor::Params{w.L});
            branch_ids.push_back(b);
        }
        // Register all pair-wise 2-winding couplings.
        for (Size i = 0; i < N; ++i) {
            for (Size j = i + 1; j < N; ++j) {
                Real k_ij = Real{1};   // default tight coupling
                if (!k_matrix.empty() &&
                        i < k_matrix.size() &&
                        j < k_matrix[i].size()) {
                    k_ij = k_matrix[i][j];
                }
                pool_.add_transformer_coupling(
                    branch_ids[i], branch_ids[j],
                    models::TwoWindingTransformer::Params{
                        .L_p = windings[i].L,
                        .L_s = windings[j].L,
                        .k   = k_ij});
            }
        }
        return *this;
    }

    // -------- Accessors ----------------------------------------------------

    [[nodiscard]] const topology::Graph& graph() const noexcept {
        return graph_;
    }

    [[nodiscard]] const pwl::DevicePool& pool() const noexcept {
        return pool_;
    }

    [[nodiscard]] Size num_branches() const noexcept {
        return graph_.num_branches();
    }

    /// Look up a previously-registered node by name. Throws
    /// `std::out_of_range` if not found. The "gnd" alias is
    /// handled here too.
    [[nodiscard]] Index node_id_of(
        const std::string& name) const {
        if (is_ground_alias_(name)) {
            return graph_.ground();
        }
        const auto it = node_map_.find(name);
        if (it == node_map_.end()) {
            throw std::out_of_range(
                "CircuitBuilder::node_id_of: node \"" +
                name + "\" was never registered");
        }
        return it->second;
    }

private:
    [[nodiscard]] static bool is_ground_alias_(
        const std::string& name) noexcept {
        return name == "gnd" || name == "GND" ||
               name == "0";
    }

    Index resolve_node_(const std::string& name) {
        if (is_ground_alias_(name)) {
            return graph_.ground();
        }
        const auto it = node_map_.find(name);
        if (it != node_map_.end()) {
            return it->second;
        }
        const Index idx = graph_.add_node(name);
        node_map_.emplace(name, idx);
        return idx;
    }

    topology::Graph                         graph_;
    pwl::DevicePool                         pool_;
    std::unordered_map<std::string, Index>  node_map_;
};

}  // namespace pulsim::builder
