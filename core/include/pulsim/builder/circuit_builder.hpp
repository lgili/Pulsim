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
#include "pulsim/models/gapped_core.hpp"
#include "pulsim/models/jiles_atherton.hpp"
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
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/preflight.hpp"
#include "pulsim/topology/graph.hpp"

#include <array>
#include <cmath>
#include <format>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pulsim::builder {

class CircuitBuilder {
public:
    CircuitBuilder() = default;

    /// Return the node index for `name`, creating it if not
    /// yet registered. `"gnd"` / `"GND"` / `"0"` all map to
    /// `graph().ground()` without consuming a node slot.
    ///
    /// Takes a `std::string_view` so call sites using string
    /// literals (the DSL's common case) skip the per-call
    /// `std::string` allocation; the materialised owned copy
    /// only happens on the first insertion of a given name.
    Index node(std::string_view name) {
        return resolve_node_(name);
    }

    // -------- Add methods --------------------------------------------------

    CircuitBuilder& add_voltage_source(
        std::string_view name, std::string_view from,
        std::string_view to, Real V) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = add_branch_(
            name, from_idx, to_idx,
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
        std::string_view name, std::string_view from,
        std::string_view to, Real v_high, Real v_low,
        Real frequency, Real duty,
        Real phase = Real{0}) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = add_branch_(
            name, from_idx, to_idx,
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
        std::string_view name, std::string_view from,
        std::string_view to,
        Real v_dc, Real v_amplitude,
        Real frequency, Real phase = Real{0}) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = add_branch_(
            name, from_idx, to_idx,
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
        std::string_view name,
        std::string_view drain, std::string_view source,
        std::string_view gate,
        Real K, Real V_T,
        Real lambda = Real{0.02},
        Real kappa  = Real{15.0},
        bool with_body_diode = true) {
        const Index drain_idx  = resolve_node_(drain);
        const Index source_idx = resolve_node_(source);
        const Index gate_idx   = resolve_node_(gate);
        const Index b_id = add_branch_(
            name, drain_idx, source_idx,
            topology::BranchKind::Nonlinear);
        pool_.add_mosfet_level1(
            b_id, gate_idx,
            models::MosfetLevel1::Params{
                .K = K, .V_T = V_T,
                .lambda = lambda, .kappa = kappa,
            });
        // The intrinsic anti-parallel body diode, ON BY DEFAULT
        // (v2.0, audit C.1) — see `add_mosfet` for why. It is no
        // longer a numerical crutch: since the third quadrant was
        // symmetrized there is no spurious V_DS < 0 root for
        // Newton to fall into. It is here because it is part of
        // the device, and because it is what carries the current
        // when the GATE IS OFF, where the channel correctly
        // blocks and an inductive load would otherwise have no
        // path at all. Pass `with_body_diode = false` for an
        // eGaN HEMT, which has no p-n body diode.
        if (with_body_diode) {
            const std::string body_name = std::string{name} + "_body";
            const Index body_b = add_branch_(
                body_name, source_idx, drain_idx,
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
        std::string_view name,
        std::string_view in_pos,  std::string_view in_neg,
        std::string_view out_pos, std::string_view out_neg,
        Real gain) {
        const Index inp_idx = resolve_node_(in_pos);
        const Index inn_idx = resolve_node_(in_neg);
        const Index outp_idx = resolve_node_(out_pos);
        const Index outn_idx = resolve_node_(out_neg);
        const Index b_id = add_branch_(
            name, outp_idx, outn_idx,
            topology::BranchKind::Source);
        pool_.add_vcvs(
            b_id, inp_idx, inn_idx,
            models::VCVS::Params{.gain = gain});
        return *this;
    }

    /// Phase 4 C.4 — IDEAL TRANSFORMER: v_s = n·v_p, i_p = −n·i_s,
    /// n = N_s/N_p. The one magnetics element the kernel lacked.
    /// The secondary is the branch (its current is `i(name)`); the
    /// primary is a pair of node references. It transforms DC — put
    /// a magnetising inductance across the primary if the physical
    /// device has one (the saturable transformer does that for you).
    CircuitBuilder& add_ideal_transformer(
        std::string_view name,
        std::string_view p_from, std::string_view p_to,
        std::string_view s_from, std::string_view s_to,
        Real n) {
        if (!(n > Real{0}) || !std::isfinite(n)) {
            throw std::invalid_argument(std::format(
                "add_ideal_transformer(\"{}\"): turns ratio n = N_s/N_p "
                "must be a positive finite number (got {}). Reverse the "
                "secondary terminals for a polarity flip rather than "
                "passing a negative ratio.", name, n));
        }
        const Index pf = resolve_node_(p_from);
        const Index pt = resolve_node_(p_to);
        const Index sf = resolve_node_(s_from);
        const Index st = resolve_node_(s_to);
        if (pf == pt) {
            throw std::invalid_argument(std::format(
                "add_ideal_transformer(\"{}\"): primary terminals are the "
                "same node '{}' — the primary would see zero voltage and "
                "the secondary would be pinned to zero.", name, p_from));
        }
        if (sf == st) {
            throw std::invalid_argument(std::format(
                "add_ideal_transformer(\"{}\"): secondary terminals are "
                "the same node '{}'.", name, s_from));
        }
        const Index b_id = add_branch_(
            name, sf, st, topology::BranchKind::Source);
        pool_.add_ideal_transformer(
            b_id, pf, pt, models::IdealTransformer::Params{.n = n});
        return *this;
    }

    /// Phase 4 C.4 — a saturable inductor whose law is a TABLE:
    /// (i_k, λ_k) knots for i ≥ 0 from the origin, strictly increasing
    /// in both, extended as an odd function. Monotone-cubic between
    /// knots (L > 0 everywhere), linear beyond the last one. The
    /// device is a `SaturableInductor` in every other respect —
    /// same history, same TR-BDF2 stage, same result accessors.
    CircuitBuilder& add_saturable_inductor_table(
        std::string_view name,
        std::string_view n_pos, std::string_view n_neg,
        std::vector<Real> i_knots, std::vector<Real> lambda_knots) {
        auto table = std::make_shared<const models::FluxTable>(
            std::move(i_knots), std::move(lambda_knots),
            std::format("add_saturable_inductor_table(\"{}\")", name));
        models::SaturableInductor::Params p;
        p.L_0 = table->L_0();
        p.L_residual = table->L_residual();
        p.I_sat = table->i_max();
        p.table = std::move(table);
        return add_saturable_inductor_params_(name, n_pos, n_neg, p);
    }

    /// Phase 4 C.4 — a saturable inductor on a GAPPED CORE, from
    /// geometry: N turns, effective area Ae [m²], mean path le [m],
    /// total gap lg [m], initial μ_r, and B_sat [T] (≡ μ₀·M_s, the
    /// datasheet value). The λ(i) table is generated by sweeping the
    /// core field H through B = μ₀(H + M_s·tanh(H/H₀)) and Ampère's
    /// law, with the exact slope at every knot —
    /// see models/gapped_core.hpp for the derivation and the worked
    /// ETD29 numbers. This is the magnetising branch the saturable
    /// transformer is built on.
    CircuitBuilder& add_gapped_core_inductor(
        std::string_view name,
        std::string_view n_pos, std::string_view n_neg,
        Real N, Real Ae, Real le, Real lg,
        Real mu_r0 = Real{2000}, Real B_sat = Real{0.35},
        Size knots = 128) {
        models::GappedCore::Params c;
        c.N = N; c.Ae = Ae; c.le = le; c.lg = lg;
        c.mu_r0 = mu_r0; c.B_sat = B_sat; c.knots = knots;
        auto table = std::make_shared<const models::FluxTable>(
            models::GappedCore::make_table(
                c, std::format("add_gapped_core_inductor(\"{}\")", name)));
        models::SaturableInductor::Params p;
        p.L_0 = table->L_0();
        p.L_residual = table->L_residual();
        p.I_sat = models::GappedCore::knee_current(c);
        p.table = std::move(table);
        return add_saturable_inductor_params_(name, n_pos, n_neg, p);
    }

    /// Phase 4 C.4 — HYSTERETIC CORE INDUCTOR: Jiles-Atherton INSIDE
    /// the Newton loop. N turns on a core (Ae, le, TOTAL gap lg) of a
    /// material given by its five JA parameters. The magnetisation is
    /// solved at the same time level as the current, with its exact
    /// tangent in the Jacobian; there is no dummy source, no
    /// observer, and no one-step lag.
    ///
    /// The observer it replaces (`add_hysteretic_inductor` +
    /// `make_hysteretic_inductor_observer` in Python) injected
    /// ψ·dM/dt from the PREVIOUS step with its sign inverted — the
    /// magnetisation acted as a negative inductance (current leading
    /// voltage, |I₁| > V/R on a passive branch) — and was unstable
    /// above q = L_M/(dt(R + 2L₀/dt)) ≈ 0.5 (measured: 0.4 → 0.999 A,
    /// 0.6 → 5e4 A, 2.0 → NaN), i.e. for every shipped use.
    ///
    /// `i(name)` is the winding current. The B–H trajectory can be
    /// replayed from it with the Python helpers (`compute_bh_loop`).
    CircuitBuilder& add_hysteretic_core_inductor(
        std::string_view name,
        std::string_view from, std::string_view to,
        Real N, Real Ae, Real le, Real lg,
        const models::JilesAthertonParams& ja,
        int substeps_min = 8, Real M0 = Real{0}) {
        models::JilesAthertonCore::Params c;
        c.N = N; c.Ae = Ae; c.le = le; c.lg = lg; c.ja = ja;
        c.substeps_min = substeps_min; c.M0 = M0;
        models::JilesAthertonCore::validate(
            c, std::format("add_hysteretic_core_inductor(\"{}\")", name));
        models::SaturableInductor::Params p;
        // Reported metadata: the initial (anhysteretic) inductance and
        // the air value; I_sat as the current at H = 3a on the virgin
        // curve.
        const Real chi0 = ja.Ms / (Real{3} * ja.a - ja.alpha * ja.Ms);
        p.L_0 = models::JilesAthertonCore::inductance_of(c, chi0);
        p.L_residual = models::JilesAthertonCore::inductance_of(c, Real{0});
        p.I_sat = models::JilesAthertonCore::current_of(c, Real{3} * ja.a, chi0 * Real{3} * ja.a);
        p.ja = std::make_shared<const models::JilesAthertonCore::Params>(c);
        return add_saturable_inductor_params_(name, from, to, p);
    }

    /// Phase 4 C.4 — SATURABLE TRANSFORMER on a gapped core.
    ///
    ///     p_from ──[ L_leak_p ]──┬──● IDEAL n ●──[ L_leak_s ]── s_from
    ///                          [L_m]
    ///     p_to   ────────────────┴──●         ●──────────────── s_to
    ///
    /// The T-model: per-winding LINEAR leakage inductances, an ideal
    /// transformer with n = N_s/N_p, and ONE magnetising branch —
    /// a gapped-core flux device λ(i) referred to the PRIMARY (N_p
    /// turns on the given geometry) — which is the only nonlinear
    /// element. The core sees the primary voltage after the primary
    /// leakage drop, which is where a core sits physically.
    ///
    /// Why not `add_transformer`: that is a pair of coupled inductors,
    /// linear by construction, with the magnetising inductance folded
    /// into L_p and M. There is no place in it for a core, so nothing
    /// in it can saturate; pushed to 3× B_sat the example flyback
    /// still returned a tidy output voltage. This device runs away
    /// there, because the core does.
    ///
    /// Cross-check against the linear device below saturation: this
    /// is the coupled pair with L_p = L_leak_p + L_m, M = n·L_m,
    /// L_s = n²·L_m + L_leak_s, i.e. k = M/√(L_p L_s).
    ///
    /// Branches created: `name.lp` (if L_leak_p > 0), `name.m` (the
    /// magnetising branch; `i(name.m)` is the magnetising current
    /// referred to the primary), `name` (the ideal secondary;
    /// `i(name)` is the secondary current), `name.ls` (if
    /// L_leak_s > 0). Internal nodes `name.pm` / `name.sm`.
    CircuitBuilder& add_saturable_transformer(
        std::string_view name,
        std::string_view p_from, std::string_view p_to,
        std::string_view s_from, std::string_view s_to,
        Real N_p, Real N_s,
        Real Ae, Real le, Real lg,
        Real mu_r0 = Real{2000}, Real B_sat = Real{0.35},
        Real L_leak_p = Real{0}, Real L_leak_s = Real{0}) {
        if (!(N_p > 0) || !(N_s > 0) || N_p != std::floor(N_p)
            || N_s != std::floor(N_s)) {
            throw std::invalid_argument(std::format(
                "add_saturable_transformer(\"{}\"): N_p and N_s must be "
                "positive integers (got {}, {}); reverse the secondary "
                "terminals for a polarity flip.", name, N_p, N_s));
        }
        if (!(L_leak_p >= 0) || !(L_leak_s >= 0)) {
            throw std::invalid_argument(std::format(
                "add_saturable_transformer(\"{}\"): leakage inductances "
                "must be >= 0 (got {}, {}).", name, L_leak_p, L_leak_s));
        }
        const std::string pm = std::format("{}.pm", name);
        const std::string sm = std::format("{}.sm", name);
        // Primary side: leakage (optional) into the magnetising node.
        std::string core_from = std::string{p_from};
        if (L_leak_p > 0) {
            add_inductor(std::format("{}.lp", name), p_from, pm, L_leak_p);
            core_from = pm;
        }
        add_gapped_core_inductor(std::format("{}.m", name), core_from, p_to,
                                 N_p, Ae, le, lg, mu_r0, B_sat);
        // Secondary side: ideal transformer, then leakage (optional).
        std::string sec_from = std::string{s_from};
        if (L_leak_s > 0) {
            add_inductor(std::format("{}.ls", name), sm, s_from, L_leak_s);
            sec_from = sm;
        }
        add_ideal_transformer(name, core_from, p_to, sec_from, s_to,
                              N_s / N_p);
        return *this;
    }

    /// Add an IDEAL OP-AMP: high-gain VCVS with single-ended
    /// output (out_neg = gnd). Default gain = 10⁵ (≈ open-
    /// loop typical for compensated devices). Combine with
    /// negative feedback to enforce the "virtual short"
    /// V_in_pos ≈ V_in_neg.
    CircuitBuilder& add_op_amp_ideal(
        std::string_view name,
        std::string_view in_pos, std::string_view in_neg,
        std::string_view out,
        Real gain = Real{1e5}) {
        return add_vcvs(name, in_pos, in_neg,
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
        std::string_view name,
        std::string_view collector, std::string_view emitter,
        std::string_view gate,
        Real V_CE_sat = Real{1.5},
        Real R_CE_sat = Real{0.05},
        Real V_T      = Real{5.0},
        Real kappa    = Real{10.0},
        Real v_knee   = Real{0.01},
        bool with_fwd = false,
        Real tau_tail = Real{0},
        Real k_tail   = Real{0}) {
        if (!(R_CE_sat > Real{0})) {
            throw std::invalid_argument(std::format(
                "add_igbt_level1(\"{}\"): R_CE_sat must be > 0 (got {}); a "
                "zero/negative on-state resistance divides by zero in the "
                "IGBT current law and yields NaN.", name, R_CE_sat));
        }
        if (tau_tail < Real{0}) {
            throw std::invalid_argument(std::format(
                "add_igbt_level1(\"{}\"): tau_tail must be >= 0 (got {}); "
                "it is the turn-off tail's time constant, and 0 disables "
                "the tail.", name, tau_tail));
        }
        if (k_tail < Real{0} || k_tail >= Real{1}) {
            throw std::invalid_argument(std::format(
                "add_igbt_level1(\"{}\"): k_tail must be in [0, 1) (got "
                "{}); it is the FRACTION of on-state current that "
                "continues as tail after the channel cuts off, so 1 would "
                "mean the channel carries nothing at all.", name, k_tail));
        }
        if ((tau_tail > Real{0}) != (k_tail > Real{0})) {
            throw std::invalid_argument(std::format(
                "add_igbt_level1(\"{}\"): tau_tail and k_tail must be set "
                "together (got tau_tail={}, k_tail={}). A tail needs both "
                "how long it lasts and how big it starts; setting one "
                "alone models no tail at all, silently.",
                name, tau_tail, k_tail));
        }
        if (!(v_knee > Real{0})) {
            throw std::invalid_argument(std::format(
                "add_igbt_level1(\"{}\"): v_knee must be > 0 (got {}); it is "
                "the width of the collector knee that enforces I_C >= 0, and "
                "at zero the model conducts in reverse — which a "
                "minority-carrier device cannot do.", name, v_knee));
        }
        const Index c_idx = resolve_node_(collector);
        const Index e_idx = resolve_node_(emitter);
        const Index g_idx = resolve_node_(gate);
        const Index b_id = add_branch_(
            name, c_idx, e_idx,
            topology::BranchKind::Nonlinear);
        pool_.add_igbt_level1(
            b_id, g_idx,
            models::IgbtLevel1::Params{
                .V_CE_sat = V_CE_sat,
                .R_CE_sat = R_CE_sat,
                .V_T      = V_T,
                .kappa    = kappa,
                .v_knee   = v_knee,
                .tau_tail = tau_tail,
                .k_tail   = k_tail,
            });
        // Anti-parallel freewheeling diode, co-packaged in every
        // real IGBT module. Now that the transistor correctly
        // refuses reverse current, an inductive load has NO path
        // during freewheeling without this — the solve fails
        // rather than quietly running the current backwards
        // through the transistor, which is the trade this model
        // makes on purpose.
        if (with_fwd) {
            const std::string fwd_name = std::string{name} + "_fwd";
            const Index fwd_b = add_branch_(
                fwd_name, e_idx, c_idx,
                topology::BranchKind::Switch);
            pool_.add_diode(fwd_b,
                /*g_on=*/Real{1e3},
                /*g_off=*/Real{1e-9},
                /*V_th=*/Real{0.5});
        }
        return *this;
    }

    /// Add an EXPONENTIAL (Shockley) diode — Phase 4, audit C.1.
    ///
    ///     i = I_S·(exp(v/(n·V_T)) − 1) + v·G_min
    ///
    /// Unlike `add_diode` (binary PWL) and the smooth-blend
    /// `IdealDiode`, this one does not fix the forward drop. A
    /// real junction's V_F rises ~60 mV per decade of current, so
    /// the same device drops 0.53 V at 1 mA and 0.77 V at 10 A —
    /// and a fixed-V_F model is wrong in OPPOSITE directions at
    /// the two ends, which no single fitted value can remove.
    ///
    ///   I_S    [A] saturation current (1e-12 typical Si)
    ///   n      [-] emission coefficient, 1 to 2
    ///   V_T    [V] thermal voltage kT/q; use
    ///              `ShockleyDiode::thermal_voltage(T)` for a
    ///              junction temperature other than ~300 K
    ///   G_min  [S] parallel conductance, keeps a reverse-biased
    ///              junction's node non-singular
    ///   BV     [V] reverse breakdown magnitude; 0 disables it
    ///
    /// There is deliberately no series-resistance parameter: put
    /// an ordinary resistor in series, which is exact. See the
    /// model header for why the inner iteration that would be
    /// needed here diverges.
    CircuitBuilder& add_shockley_diode(
        std::string_view name,
        std::string_view anode, std::string_view cathode,
        Real I_S   = Real{1e-12},
        Real n     = Real{1.0},
        Real V_T   = Real{0.025852},
        Real G_min = Real{1e-12},
        Real BV    = Real{0}) {
        const models::ShockleyDiode::Params p{
            .I_S = I_S, .n = n, .V_T = V_T,
            .G_min = G_min, .BV = BV,
        };
        models::ShockleyDiode::validate(p);
        const Index a_idx = resolve_node_(anode);
        const Index c_idx = resolve_node_(cathode);
        const Index b_id = add_branch_(
            name, a_idx, c_idx,
            topology::BranchKind::Nonlinear);
        pool_.add_shockley_diode(b_id, p);
        return *this;
    }

    /// Add a Lauritzen-Mattsson diode: an exponential junction
    /// that also STORES CHARGE, so it recovers.
    ///
    /// v2.0 Phase 4, audit C.1. Every other diode here is static
    /// I-V, and a static law cannot recover, because recovery is
    /// stored charge leaving the device. On a double-pulse test
    /// (400 V, 20 A, 50 nH loop) the PWL and Shockley diodes both
    /// commutate 20 A straight to zero with a reverse peak of
    /// exactly 0.00000 A; a real fast-recovery Si part peaks
    /// 15-30 A negative, and that current flows through the
    /// turning-on switch, where it usually dominates turn-on loss.
    ///
    ///   I_S   [A] saturation current
    ///   n     [-] emission coefficient, 1 to 2
    ///   V_T   [V] thermal voltage kT/q
    ///   tau   [s] carrier lifetime — HOW LONG recovery lasts.
    ///             Fast-recovery Si: 10-100 ns. Standard
    ///             rectifier: microseconds.
    ///   T_M   [s] transit time — HOW HARD the reverse peak is.
    ///             Must stay well below `tau`.
    ///   G_min [S] parallel conductance
    ///
    /// The DC curve is unchanged: in steady state this is an
    /// ordinary Shockley junction with I_S scaled by
    /// tau/(tau + T_M). Only the dynamics are new. A Schottky
    /// stores no charge at all — use `add_shockley_diode` for
    /// one rather than driving `tau` toward zero here.
    CircuitBuilder& add_lauritzen_diode(
        std::string_view name,
        std::string_view anode, std::string_view cathode,
        Real I_S   = Real{1e-12},
        Real n     = Real{1.0},
        Real V_T   = Real{0.025852},
        Real tau   = Real{1e-7},
        Real T_M   = Real{1e-8},
        Real G_min = Real{1e-12}) {
        const models::LauritzenDiode::Params p{
            .I_S = I_S, .n = n, .V_T = V_T,
            .tau = tau, .T_M = T_M, .G_min = G_min,
        };
        models::LauritzenDiode::validate(p);
        const Index a_idx = resolve_node_(anode);
        const Index c_idx = resolve_node_(cathode);
        const Index b_id = add_branch_(
            name, a_idx, c_idx,
            topology::BranchKind::Nonlinear);
        pool_.add_lauritzen_diode(b_id, p);
        return *this;
    }

    /// Add an MNA-native PMSM (Phase 4, audit C.3).
    ///
    /// Three stator branches `name_a/_b/_c` from the phase nodes to
    /// `neutral`, each with a branch-current unknown, stamped per
    /// Newton iteration with the FULL θ-dependent inductance matrix
    /// L(θ) = T⁻¹·diag(L_d, L_q, L_0)·T and the PM flux — so an
    /// IPM's d- and q-axis dynamics are exact (the observer-based
    /// `add_pmsm` uses one average L and gets τ_d wrong by +100 %,
    /// τ_q by −33 % on a 1/3 mH machine), and there is no one-step
    /// lag anywhere (the observer's costs −0.37 % of speed at
    /// dt = 5e-5 and −8 % at 5e-4).
    ///
    /// Mechanics are NODES: `omega_node` carries a capacitor J (so
    /// J·dω/dt is trapezoidal, in the same solve) and a resistor
    /// 1/B; `theta_node` carries a 1 F capacitor fed by ω. T_em from
    /// the co-energy of the same L(θ) — the reluctance torque
    /// appears because L(θ) is in the matrix, not by being added on.
    /// A speed-dependent load is simply another element on
    /// `omega_node`; `T_load` here is the constant part.
    ///
    ///   R_s, L_d, L_q [Ω, H]   psi_pm [Wb]   pole_pairs
    ///   J [kg·m²]   B [N·m·s/rad]   T_load [N·m]
    ///   L_0 [H] zero-sequence (0 → dq average; only conditioning)
    ///   omega0 [rad/s], theta0 [rad] initial mechanical state
    CircuitBuilder& add_pmsm_mna(
        std::string_view name,
        std::string_view phase_a, std::string_view phase_b,
        std::string_view phase_c, std::string_view neutral,
        std::string_view omega_node, std::string_view theta_node,
        Real R_s, Real L_d, Real L_q, Real psi_pm, Real pole_pairs,
        Real J, Real B = Real{0}, Real T_load = Real{0},
        Real L_0 = Real{0},
        Real omega0 = Real{0}, Real theta0 = Real{0}) {
        if (!(J > Real{0})) {
            throw std::invalid_argument(std::format(
                "add_pmsm_mna(\"{}\"): J must be > 0 (got {}); it is the "
                "capacitor on the omega node, and a zero inertia leaves "
                "that node with no dynamic and no pivot.", name, J));
        }
        if (B < Real{0}) {
            throw std::invalid_argument(std::format(
                "add_pmsm_mna(\"{}\"): B must be >= 0 (got {}).", name, B));
        }
        const models::PmsmMna::Params p{
            .R_s = R_s, .L_d = L_d, .L_q = L_q, .L_0 = L_0,
            .psi_pm = psi_pm, .pole_pairs = pole_pairs,
            .T_load = T_load,
        };
        models::PmsmMna::validate(p);
        const Index n_idx = resolve_node_(neutral);
        const std::array<std::string_view, 3> ph{phase_a, phase_b,
                                                 phase_c};
        std::array<Index, 3> ids{};
        for (Size k = 0; k < 3; ++k) {
            const std::string bname =
                std::string{name} + "_" + "abc"[k];
            ids[k] = add_branch_(bname, resolve_node_(ph[k]), n_idx,
                                 topology::BranchKind::Nonlinear);
        }
        const Index om_idx = resolve_node_(omega_node);
        const Index th_idx = resolve_node_(theta_node);
        pool_.add_pmsm_mna(ids, om_idx, th_idx, p);
        // Mechanical circuit: ordinary linear elements, so the
        // mechanics inherit the trapezoidal companion.
        add_capacitor(std::string{name} + "_J", omega_node, "gnd", J,
                      omega0);
        if (B > Real{0}) {
            add_resistor(std::string{name} + "_B", omega_node, "gnd",
                         Real{1} / B);
        }
        add_capacitor(std::string{name} + "_theta", theta_node, "gnd",
                      Real{1}, theta0);
        return *this;
    }

    /// Add a pulse / step voltage source (Layer 2 V12).
    ///   v_initial    [V] baseline (before & between pulses)
    ///   v_pulsed     [V] level during the pulse window
    ///   t_start      [s] delay before first pulse fires
    ///   pulse_width  [s] duration of each pulse
    ///   period       [s] repetition period; 0 → single-shot
    CircuitBuilder& add_pulse_voltage_source(
        std::string_view name, std::string_view from,
        std::string_view to,
        Real v_initial, Real v_pulsed,
        Real t_start, Real pulse_width,
        Real period    = Real{0},
        Real rise_time = Real{0},
        Real fall_time = Real{0}) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = add_branch_(
            name, from_idx, to_idx,
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
        std::string_view name, std::string_view from,
        std::string_view to, Real I) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = add_branch_(
            name, from_idx, to_idx,
            topology::BranchKind::Source);
        pool_.add_current_source(
            b_id, models::CurrentSource::Params{I});
        return *this;
    }

    CircuitBuilder& add_resistor(
        std::string_view name, std::string_view from,
        std::string_view to, Real R_ohms) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = add_branch_(
            name, from_idx, to_idx,
            topology::BranchKind::PassiveLinear);
        pool_.add_resistor(
            b_id, models::Resistor::Params{
                .G = Real{1} / R_ohms});
        return *this;
    }

    // -------------------------------------------------------------
    // v2.0 Phase 2 (B.1) — topology preflight + auto-regularization
    // -------------------------------------------------------------
    //
    // Closes audit finding `no-topology-preflight-or-auto-shunt`.
    // Runs the connectivity sweep and, unless the caller opts out,
    // gives every unreferenced subnet a high-value tie to ground —
    // the fix `docs/gotchas.md` has been asking users to type by
    // hand.
    //
    // Must run BEFORE a `PwlStateSpaceCache` is constructed: the
    // cache holds `const Graph&` / `DevicePool&` references, and
    // this appends branches to both. Appending a BRANCH (never a
    // node) leaves `state_size` and every branch-variable index
    // untouched, so nothing downstream shifts; the graph's
    // structural id correctly changes, since the topology did.
    //
    // Idempotent: once a subnet has its tie it reaches ground, so a
    // second call finds nothing.
    [[nodiscard]] pwl::PreflightReport run_preflight(
        const pwl::PreflightOptions& opts = {}) {
        if (!(opts.tie_resistance > Real{0}) ||
            !std::isfinite(opts.tie_resistance)) {
            throw std::invalid_argument(std::format(
                "run_preflight: tie_resistance must be a positive, "
                "finite value (got {}). It becomes a conductance "
                "1/R, so 0 would stamp an infinity and quietly turn "
                "the whole solution into NaN.", opts.tie_resistance));
        }
        if (!opts.auto_regularize) {
            return pwl::analyze_preflight(graph_, pool_, opts);
        }

        // ITERATE TO A FIXED POINT rather than applying one analysis.
        //
        // A galvanic finding covers a whole island but earns it a
        // single tie, so DC-floating sub-blocks INSIDE that island
        // are still floating once it lands — e.g. a current source
        // feeding a resistor chain, where the source conducts
        // galvanically but contributes no conductance at DC. The
        // first version of this code filtered the DC findings
        // against the galvanic ones by component membership and so
        // reported those sub-blocks as fixed while leaving them
        // singular. Re-analysing after each round makes the
        // nesting fall out for free.
        //
        // Terminates: every round ties at least one component to
        // ground, strictly reducing the number of components that
        // do not reach it. The bound is a belt-and-braces guard, not
        // a real limit.
        pwl::PreflightReport all;
        const Size max_rounds =
            static_cast<Size>(graph_.num_nodes()) + 1;
        for (Size round = 0; round < max_rounds; ++round) {
            auto found = pwl::analyze_preflight(graph_, pool_, opts);
            if (found.empty()) break;
            for (auto& f : found.findings) {
                const std::string node_key =
                    graph_.node(f.anchor_node).name.empty()
                        ? std::format("n{}", f.anchor_node)
                        : graph_.node(f.anchor_node).name;
                const std::string tie_name =
                    opts.name_prefix + node_key;
                // Branch first, THEN the pool entry — an
                // unregistered branch would make a re-run of the
                // sweep (and the assembler) throw on `kind_of`.
                const Index b_id = add_branch_(
                    tie_name, f.anchor_node, kGround,
                    topology::BranchKind::PassiveLinear);
                pool_.add_resistor(
                    b_id, models::Resistor::Params{
                        .G = Real{1} / opts.tie_resistance});
                f.inserted_resistance = opts.tie_resistance;
                f.detail += std::format(
                    " Pulsim inserted '{}' ({:g} Ω to ground).",
                    tie_name, opts.tie_resistance);
                all.findings.push_back(std::move(f));
            }
        }
        return all;
    }

    CircuitBuilder& add_capacitor(
        std::string_view name, std::string_view from,
        std::string_view to, Real C_farads,
        std::optional<Real> c0 = std::nullopt) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = add_branch_(
            name, from_idx, to_idx,
            topology::BranchKind::PassiveLinear);
        pool_.add_capacitor(
            b_id, models::Capacitor::Params{C_farads});
        if (c0.has_value()) {
            initial_conditions_.emplace(b_id, *c0);
        }
        return *this;
    }

    CircuitBuilder& add_inductor(
        std::string_view name, std::string_view from,
        std::string_view to, Real L_henries,
        std::optional<Real> i0 = std::nullopt) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = add_branch_(
            name, from_idx, to_idx,
            topology::BranchKind::PassiveLinear);
        pool_.add_inductor(
            b_id, models::Inductor::Params{L_henries});
        if (i0.has_value()) {
            initial_conditions_.emplace(b_id, *i0);
        }
        return *this;
    }

    /// Record (or override) an initial condition for a named
    /// capacitor or inductor branch — useful when the IC isn't
    /// known at `add_capacitor` / `add_inductor` time (e.g., set
    /// from a GUI's right-click menu after the device is dropped).
    ///
    /// Throws `std::out_of_range` if `name` doesn't match a
    /// registered branch. Does NOT validate the kind — passing a
    /// resistor's name is accepted but silently ignored at
    /// :meth:`initial_state` synthesis time.
    void set_initial(std::string_view name, Real value) {
        const Index b_id = branch_id_of(name);  // throws if unknown
        initial_conditions_[b_id] = value;
    }

    /// Synthesise the flat initial-state vector from recorded ICs.
    /// Returns a `Vector` of size
    /// `num_nodes + num_state_branches` (the same shape
    /// `pulsim.simulate(initial_state=…)` expects). Capacitor ICs
    /// land on the *node* voltage column of the capacitor's
    /// positive terminal; inductor and source ICs land on their
    /// pool-mapped current state-vector slot.
    [[nodiscard]] pulsim::Vector initial_state() const {
        // State size: node voltages + every state-branch current.
        // We don't have a single API for "state size" — count by
        // probing the pool for each branch.
        const Index n_nodes = graph_.num_nodes();
        Index n_state_extras = 0;
        for (Index b = 0; b < graph_.num_branches(); ++b) {
            if (pool_.is_inductor(b) || pool_.is_voltage_source(b)) {
                ++n_state_extras;
            }
        }
        pulsim::Vector x0 = pulsim::Vector::Zero(
            static_cast<int>(n_nodes + n_state_extras));
        for (const auto& [b_id, value] : initial_conditions_) {
            if (b_id < 0 || b_id >= graph_.num_branches()) continue;
            // Inductor / source: write into the pool-mapped state
            // column.
            if (pool_.is_inductor(b_id)) {
                const Index col = pool_.branch_var_id_for_inductor(
                    b_id, graph_);
                if (col >= 0 && col < x0.size()) {
                    x0(col) = value;
                }
            } else if (pool_.is_voltage_source(b_id)) {
                const Index col = pool_.branch_var_id_for_source(
                    b_id, graph_);
                if (col >= 0 && col < x0.size()) {
                    x0(col) = value;
                }
            } else if (pool_.is_capacitor(b_id)) {
                // Cap IC → node voltage of the positive terminal.
                const auto& br = graph_.branch(b_id);
                if (br.from >= 0 && br.from < n_nodes) {
                    x0(br.from) = value;
                }
            }
            // Resistors and other kinds: silently ignored.
        }
        return x0;
    }

    /// Human-readable names for every entry of the state vector the
    /// kernel solves. Returns a ``std::vector<std::string>`` of size
    /// ``pool.state_size(graph)`` — same layout the kernel uses:
    ///
    ///   * indices ``[0, num_nodes)``           → ``"V(<node_name>)"``
    ///   * indices ``[num_nodes, ..)``          → either
    ///     ``"Is(<branch_name>)"`` for voltage-source currents OR
    ///     ``"I(<branch_name>)"`` for inductor currents, placed at
    ///     the slot the pool returns from ``branch_var_id_for_*``.
    ///
    /// Unnamed nodes (rare) are reported as ``"V(n<id>)"``; unnamed
    /// branches as ``"<kind>(b<id>)"``. Useful primarily for the
    /// live scope: ``pulsim.LiveScope`` / PulsimGUI use these as
    /// human labels and to resolve ``state_idx`` for new signals
    /// registered against the running stream.
    [[nodiscard]] std::vector<std::string> state_var_names() const {
        const Index n_nodes = graph_.num_nodes();
        const std::size_t n_state =
            static_cast<std::size_t>(pool_.state_size(graph_));
        std::vector<std::string> names(n_state);
        // Node voltage slots: 0 .. n_nodes - 1.
        for (Index i = 0; i < n_nodes && static_cast<std::size_t>(i) < n_state;
              ++i) {
            const auto& nm = graph_.node(i).name;
            names[static_cast<std::size_t>(i)] =
                nm.empty()
                    ? std::string("V(n") + std::to_string(i) + ")"
                    : std::string("V(") + nm + ")";
        }
        // Branch state slots — pool tells us the column.
        for (Index b = 0; b < graph_.num_branches(); ++b) {
            std::string bname{name_of(b)};
            if (bname.empty()) {
                bname = std::string("b") + std::to_string(b);
            }
            Index col = -1;
            std::string prefix;
            if (pool_.is_voltage_source(b)) {
                col = pool_.branch_var_id_for_source(b, graph_);
                prefix = "Is(";
            } else if (pool_.is_inductor(b)) {
                col = pool_.branch_var_id_for_inductor(b, graph_);
                prefix = "I(";
            } else {
                continue;
            }
            if (col >= 0 && static_cast<std::size_t>(col) < n_state) {
                names[static_cast<std::size_t>(col)] =
                    prefix + bname + ")";
            }
        }
        // Sanity: any still-empty slot gets a generic label so the
        // GUI never sees an empty string (would look broken).
        for (std::size_t i = 0; i < n_state; ++i) {
            if (names[i].empty()) {
                names[i] = std::string("x[") + std::to_string(i) + "]";
            }
        }
        return names;
    }

    /// Returns the recorded `{branch_id: value}` IC map (read-only
    /// view; rarely useful from Python — :meth:`initial_state` is
    /// the consumer).
    [[nodiscard]] const std::unordered_map<Index, Real>&
    initial_conditions() const noexcept {
        return initial_conditions_;
    }

    /// Add a binary switched diode (Layer 5 V2's
    /// `SwitchedDiode`). The branch uses
    /// `BranchKind::Switch` (the diode behaves as a switch
    /// from the topology's perspective).
    CircuitBuilder& add_diode(
        std::string_view name, std::string_view anode,
        std::string_view cathode, Real g_on, Real g_off,
        Real V_th = Real{0}) {
        const Index a_idx = resolve_node_(anode);
        const Index k_idx = resolve_node_(cathode);
        const Index b_id = add_branch_(
            name, a_idx, k_idx,
            topology::BranchKind::Switch);
        pool_.add_diode(b_id, g_on, g_off, V_th);
        return *this;
    }

    /// Add a smooth-blend `IdealDiode` (Layer 4 V3's
    /// AD-driven nonlinear model). The branch uses
    /// `BranchKind::Nonlinear`.
    CircuitBuilder& add_nonlinear_diode(
        std::string_view name, std::string_view anode,
        std::string_view cathode,
        models::IdealDiode::Params params) {
        if (!(params.R_d > Real{0})) {
            throw std::invalid_argument(std::format(
                "add_nonlinear_diode(\"{}\"): R_d must be > 0 (got {}); a "
                "zero/negative slope resistance divides by zero in the diode "
                "current law and yields NaN.", name, params.R_d));
        }
        const Index a_idx = resolve_node_(anode);
        const Index k_idx = resolve_node_(cathode);
        const Index b_id = add_branch_(
            name, a_idx, k_idx,
            topology::BranchKind::Nonlinear);
        pool_.add_nonlinear_diode(b_id, params);
        return *this;
    }

    /// Add a controlled switch (drives by `switch_fn`
    /// at simulation time).
    CircuitBuilder& add_switch(
        std::string_view name, std::string_view from,
        std::string_view to, Real g_on, Real g_off) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = add_branch_(
            name, from_idx, to_idx,
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

    /// Add an n-channel power MOSFET: a controlled switch
    /// (drain → source) plus its INTRINSIC anti-parallel body
    /// diode (source → drain), which is on by DEFAULT.
    ///
    /// v2.0 (audit C.1, "diodo de corpo intrínseco por padrão").
    /// The body diode is not an accessory — it is part of the
    /// device, formed by the same p-n junction that makes the
    /// transistor, and no vertical power MOSFET exists without
    /// one. Leaving it out by default was wrong twice over:
    ///
    ///  * PHYSICALLY. A gate-off MOSFET in an inductive path is
    ///    R_off = 1 GΩ, so the freewheeling current has nowhere
    ///    to go and the node runs away. That is not a modelling
    ///    subtlety, it is the normal state of the low-side
    ///    device in every synchronous converter during dead
    ///    time.
    ///  * BY REVEALED PREFERENCE. This repository's own call
    ///    sites voted 54 to 17 for
    ///    `add_mosfet_with_body_diode` over bare `add_mosfet`.
    ///    A default that is overridden three times out of four
    ///    is the wrong default.
    ///
    /// The cost is nil: the PWL cache factors only the switch
    /// states a run actually visits, so an extra branch that
    /// never changes state is one more stamp, not another power
    /// of two. Measured on chains of 2–8 MOSFETs: 1.00x.
    ///
    /// Pass `body_diode = false` for a device that genuinely has
    /// none — an eGaN HEMT is the real case, since it conducts
    /// in reverse through the channel rather than through a p-n
    /// junction — or when you are modelling a bare switch.
    ///
    /// Defaults: R_on = 1 mΩ, R_off = 1 GΩ, body V_F = 0.7 V
    /// (typical for modern Si MOSFETs in SMPS applications).
    CircuitBuilder& add_mosfet(
        std::string_view name, std::string_view drain,
        std::string_view source,
        Real R_on  = Real{1e-3},
        Real R_off = Real{1e9},
        bool body_diode = true,
        Real V_F        = Real{0.7}) {
        if (!body_diode) {
            return add_switch(name, drain, source,
                                Real{1} / R_on,
                                Real{1} / R_off);
        }
        return add_mosfet_with_body_diode(
            name, drain, source, R_on, R_off, V_F);
    }

    /// Add an n-channel power MOSFET WITH its intrinsic
    /// anti-parallel body diode. Adds two branches:
    ///   1. switch (drain → source) with R_on / R_off
    ///   2. SwitchedDiode (source → drain) with V_F drop
    ///      that conducts during freewheeling intervals
    /// Defaults model a typical Si MOSFET: R_on = 1 mΩ,
    /// R_off = 1 GΩ, body-diode V_F = 0.7 V.
    CircuitBuilder& add_mosfet_with_body_diode(
        std::string_view name, std::string_view drain,
        std::string_view source,
        Real R_on        = Real{1e-3},
        Real R_off       = Real{1e9},
        Real V_F         = Real{0.7},
        Real g_on_diode  = Real{1e3},
        Real g_off_diode = Real{1e-9}) {
        const Index drain_idx  = resolve_node_(drain);
        const Index source_idx = resolve_node_(source);
        // Main switch (drain → source).
        const Index b_switch = add_branch_(
            name, drain_idx, source_idx,
            topology::BranchKind::Switch);
        pool_.add_switch(b_switch,
                          Real{1} / R_on,
                          Real{1} / R_off);
        // Body diode (source → drain — anti-parallel).
        const std::string body_name = std::string{name} + "_body";
        const Index b_body = add_branch_(
            body_name, source_idx, drain_idx,
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
        std::string_view name, std::string_view collector,
        std::string_view emitter,
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
        std::string_view name,
        std::string_view p_from, std::string_view p_to,
        std::string_view s_from, std::string_view s_to,
        Real L_p, Real L_s, Real k = Real{1}) {
        if (!(L_p > Real{0}) || !(L_s > Real{0})) {
            throw std::invalid_argument(std::format(
                "add_transformer(\"{}\"): L_p and L_s must be > 0 (got "
                "L_p={}, L_s={}); the mutual term M = k·sqrt(L_p·L_s) is NaN "
                "for non-positive inductances.", name, L_p, L_s));
        }
        const Index p_from_idx = resolve_node_(p_from);
        const Index p_to_idx   = resolve_node_(p_to);
        const Index s_from_idx = resolve_node_(s_from);
        const Index s_to_idx   = resolve_node_(s_to);

        const std::string p_name = std::string{name} + ".p";
        const Index p_branch = add_branch_(
            p_name, p_from_idx, p_to_idx,
            topology::BranchKind::PassiveLinear);
        pool_.add_inductor(
            p_branch, models::Inductor::Params{L_p});

        const std::string s_name = std::string{name} + ".s";
        const Index s_branch = add_branch_(
            s_name, s_from_idx, s_to_idx,
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
    ///   * Pass `enable_nonlinear_refresh=True`. (This used to
    ///     name `make_combined_nonlinear_refresh(history, dt)` as
    ///     a manual alternative for non-Python callers; that
    ///     function had no call sites and has been deleted — the
    ///     one implementation is `pwl::stamp_saturable_inductor`,
    ///     which both engines call.)
    ///   * History is automatically initialized + updated by
    ///     the solver when the circuit contains saturable
    ///     inductors.
    /// Phase 4 C.1 — a charge-based nonlinear capacitor, i.e. a
    /// MOSFET's Coss: C(v) = C0 / (1 + v/V0)^m.
    ///
    /// What decides ZVS is the CHARGE Q(V) = int C dv, not the
    /// small-signal C the datasheet quotes at the operating
    /// point. For C0 = 2 nF, V0 = 25 V, m = 0.5 at 400 V those
    /// differ by 1.61x, which is the difference between a dead
    /// time that reads as clean ZVS and one that leaves the node
    /// at 209 V when the switch turns on.
    CircuitBuilder& add_nonlinear_capacitor(
        std::string_view name,
        std::string_view from, std::string_view to,
        Real C0, Real V0, Real m = Real{0.5},
        Real v_floor = Real{-0.9}) {
        models::NonlinearCapacitor::Params params{
            .C0 = C0, .V0 = V0, .m = m, .v_floor = v_floor};
        try {
            models::NonlinearCapacitor::validate(params);
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(std::format(
                "add_nonlinear_capacitor(\"{}\"): {}", name,
                e.what()));
        }
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = add_branch_(
            name, from_idx, to_idx,
            topology::BranchKind::Nonlinear);
        pool_.add_nonlinear_capacitor(b_id, params);
        return *this;
    }

    CircuitBuilder& add_saturable_inductor(
        std::string_view name,
        std::string_view from, std::string_view to,
        Real L_0, Real I_sat,
        Real L_residual = Real{0}) {
        if (!(I_sat > Real{0})) {
            throw std::invalid_argument(std::format(
                "add_saturable_inductor(\"{}\"): I_sat must be > 0 (got {}); "
                "the saturation ratio i_L/I_sat divides by zero otherwise and "
                "yields NaN.", name, I_sat));
        }
        models::SaturableInductor::Params p;
        p.L_0 = L_0;
        p.I_sat = I_sat;
        p.L_residual = L_residual;
        return add_saturable_inductor_params_(name, from, to, p);
    }

    /// Shared tail of the three saturable-inductor entry points
    /// (analytic atan, explicit table, gapped core): one Nonlinear
    /// branch on the inductor numbering, whatever the law.
    CircuitBuilder& add_saturable_inductor_params_(
        std::string_view name,
        std::string_view from, std::string_view to,
        const models::SaturableInductor::Params& p) {
        const Index from_idx = resolve_node_(from);
        const Index to_idx   = resolve_node_(to);
        const Index b_id = add_branch_(
            name, from_idx, to_idx,
            topology::BranchKind::Nonlinear);
        pool_.add_saturable_inductor(b_id, p);
        return *this;
    }

    /// Couple two EXISTING linear inductors by name with coefficient
    /// k, exactly as `add_transformer` couples the pair it creates.
    /// This is the method `pulsim.add_flyback` had always called
    /// behind `hasattr(builder, "add_inductor_coupling")` — and it
    /// did not exist, so the factory built two UNCOUPLED inductors
    /// and its test, which counts branches, passed. Both branches
    /// must be plain (linear) inductors: the coupling lives in the
    /// linear assembly and a saturable inductor never reaches it —
    /// its Jacobian would be coupled and its history would not.
    CircuitBuilder& add_inductor_coupling(
        std::string_view name_a, std::string_view name_b, Real k) {
        if (!(k >= Real{0}) || !(k <= Real{1})) {
            throw std::invalid_argument(std::format(
                "add_inductor_coupling(\"{}\", \"{}\"): k must be in "
                "[0, 1] (got {}); reverse one winding for a negative "
                "coupling.", name_a, name_b, k));
        }
        const Index a = branch_id_of(name_a);
        const Index b = branch_id_of(name_b);
        if (a < 0 || b < 0) {
            throw std::invalid_argument(std::format(
                "add_inductor_coupling(\"{}\", \"{}\"): no branch named "
                "'{}'.", name_a, name_b, a < 0 ? name_a : name_b));
        }
        if (a == b) {
            throw std::invalid_argument(std::format(
                "add_inductor_coupling(\"{}\"): an inductor cannot be "
                "coupled to itself.", name_a));
        }
        for (const auto [id, nm] : {std::pair{a, name_a}, std::pair{b, name_b}}) {
            if (!pool_.is_registered(id) ||
                pool_.kind_of(id) != pwl::DevicePool::StoredKind::Inductor) {
                throw std::invalid_argument(std::format(
                    "add_inductor_coupling: '{}' is not a linear inductor. "
                    "Only plain inductors can be magnetically coupled "
                    "through this API; for a saturating core use the "
                    "saturable transformer.", nm));
            }
        }
        pool_.add_transformer_coupling(
            a, b,
            models::TwoWindingTransformer::Params{
                .L_p = pool_.inductor_params(a).L,
                .L_s = pool_.inductor_params(b).L,
                .k = k});
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
        std::string_view name,
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
        Size winding_idx = 0;
        for (const auto& w : windings) {
            if (!(w.L > Real{0})) {
                throw std::invalid_argument(std::format(
                    "add_multi_winding_transformer(\"{}\"): every winding "
                    "inductance must be > 0 (got {}); pair-wise couplings use "
                    "sqrt(L_i·L_j), which is NaN for non-positive L.",
                    name, w.L));
            }
            const Index from_idx = resolve_node_(w.from);
            const Index to_idx   = resolve_node_(w.to);
            const std::string winding_name =
                std::string{name} + ".w" + std::to_string(winding_idx++);
            const Index b = add_branch_(
                winding_name, from_idx, to_idx,
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

    /// Non-const pool accessor (v1.4.0). Required by
    /// `PwlStateSpaceCache::refactor_parametric` which mutates the
    /// pool's per-device values to drive a parameter sweep / Monte
    /// Carlo. Existing const-`pool()` consumers (the vast majority)
    /// are unaffected; this overload is picked only when the caller
    /// has a non-const builder reference.
    [[nodiscard]] pwl::DevicePool& pool() noexcept {
        return pool_;
    }

    [[nodiscard]] Size num_branches() const noexcept {
        return graph_.num_branches();
    }

    /// User-supplied name for `branch_id`. Set automatically by every
    /// `add_*` call; consumed by `schematic_rendering` to label
    /// components in the rendered SVG. Returns an empty view if the
    /// branch was never registered.
    [[nodiscard]] std::string_view name_of(Index branch_id) const {
        const auto it = branch_names_.find(branch_id);
        if (it == branch_names_.end()) return std::string_view{};
        return it->second;
    }

    /// Inverse of `name_of`: lookup the branch_id for a user-supplied
    /// component name (the first argument of every `add_*` call).
    /// Used by v1.4.0's parametric refactor pipeline
    /// (`PwlStateSpaceCache::refactor_parametric`) to translate
    /// user-facing param strings like `"L_out"` into the branch_id
    /// the pool's `update_*` mutators consume.
    ///
    /// Throws `std::out_of_range` if `name` was never registered.
    /// O(num_branches) per call; cache the result if calling in a
    /// hot loop.
    [[nodiscard]] Index branch_id_of(std::string_view name) const {
        // O(1) through the name index (v2.0). The scan below is
        // kept only as the alias/miss path.
        const auto hit = branch_index_.find(name);
        if (hit != branch_index_.end()) {
            return hit->second;
        }
        for (const auto& [b_id, n] : branch_names_) {
            if (n == name) return b_id;
        }
        // Alias fallback (Node aliases not allowed here; only
        // Branch-kind aliases resolve to a branch id).
        const auto a_it = aliases_.find(std::string{name});
        if (a_it != aliases_.end() &&
            a_it->second.kind == AliasKind::Branch) {
            return branch_id_of(a_it->second.target);
        }
        throw std::out_of_range(std::format(
            "CircuitBuilder::branch_id_of: component name \"{}\" "
            "was never registered", name));
    }

    /// Bit position of `name` in the `SwitchStateMask` produced by
    /// `switch_fn(t)`. Counts only `BranchKind::Switch` branches, in
    /// builder-call order (which matches Layer 4's enumeration).
    ///
    /// PulsimGUI's compat shim used to reconstruct this map by hand
    /// (`pending_gate_signals`) — exposing it here lets the GUI drop
    /// ~150 lines of bookkeeping.
    ///
    /// Throws `std::out_of_range` if `name` isn't registered, or if
    /// `name` refers to a non-switching branch (passive / source /
    /// nonlinear).
    [[nodiscard]] Index switch_index_of(std::string_view name) const {
        const Index b_id = branch_id_of(name);  // throws if unknown
        const auto& br = graph_.branch(b_id);
        if (br.kind != topology::BranchKind::Switch) {
            throw std::out_of_range(std::format(
                "CircuitBuilder::switch_index_of: branch \"{}\" is "
                "not a switching device (kind={})",
                name, branch_kind_name_(br.kind)));
        }
        // Count Switch-kind branches with id < b_id.
        Index idx = 0;
        for (Index i = 0; i < b_id; ++i) {
            if (graph_.branch(i).kind == topology::BranchKind::Switch) {
                ++idx;
            }
        }
        return idx;
    }

    /// Lightweight POD describing one registered device. Returned by
    /// `devices()` — useful for GUI introspection and the
    /// "enumerate every component" pattern that PulsimGUI hits often.
    struct DeviceInfo {
        std::string              name;
        std::string              kind;       // BranchKind name
        std::vector<std::string> terminals;  // node names (from, to)
    };

    /// Enumerate every device the builder has accepted, in `add_*`
    /// call order. Anonymous branches (no name) are skipped — they
    /// only come from internal helpers (snubber RC expansion,
    /// transformer parallel branches, …) that the caller didn't name.
    [[nodiscard]] std::vector<DeviceInfo> devices() const {
        std::vector<DeviceInfo> out;
        out.reserve(static_cast<std::size_t>(graph_.num_branches()));
        for (Index b_id = 0; b_id < graph_.num_branches(); ++b_id) {
            const auto name_view = name_of(b_id);
            if (name_view.empty()) continue;
            const auto& br = graph_.branch(b_id);
            DeviceInfo info;
            info.name = std::string(name_view);
            info.kind = branch_kind_name_(br.kind);
            info.terminals = {
                node_name_or_ground_(br.from),
                node_name_or_ground_(br.to),
            };
            out.push_back(std::move(info));
        }
        return out;
    }

    /// Look up a previously-registered node by name. Throws
    /// `std::out_of_range` if not found. The "gnd" alias is
    /// handled here too. Any caller-registered alias from
    /// :meth:`set_alias` resolves transparently.
    [[nodiscard]] Index node_id_of(std::string_view name) const {
        if (is_ground_alias_(name)) {
            return graph_.ground();
        }
        const auto it = node_map_.find(name);
        if (it != node_map_.end()) {
            return it->second;
        }
        // Fall back to alias resolution before raising.
        const auto a_it = aliases_.find(std::string{name});
        if (a_it != aliases_.end() &&
            a_it->second.kind == AliasKind::Node) {
            return node_id_of(a_it->second.target);
        }
        throw std::out_of_range(std::format(
            "CircuitBuilder::node_id_of: node \"{}\" was never "
            "registered",
            name));
    }

    /// `set_alias` / `aliases` / `AliasKind` — add-python-builder-
    /// ergonomics (v1.5). Lets the GUI attach human-readable node
    /// or branch names to round-trip through pulsim files without
    /// maintaining a parallel registry.
    enum class AliasKind : std::uint8_t { Node, Branch };
    struct AliasTarget {
        AliasKind   kind;
        std::string target;  // canonical name in node_map_ or
                              // branch_names_.
    };

    /// Register a human-readable alias for an existing node or
    /// branch. Exactly one of ``node`` / ``branch`` must be
    /// non-empty. The alias name SHALL NOT collide with an existing
    /// canonical name. Empty inputs raise ``std::invalid_argument``.
    void set_alias(std::string_view human_name,
                   std::optional<std::string_view> node,
                   std::optional<std::string_view> branch) {
        if (human_name.empty()) {
            throw std::invalid_argument(
                "set_alias: human_name must be non-empty.");
        }
        const bool has_node = node.has_value() && !node->empty();
        const bool has_branch = branch.has_value() && !branch->empty();
        if (has_node == has_branch) {
            throw std::invalid_argument(
                "set_alias: pass exactly one of node= or branch=.");
        }
        // Collision check against canonical names.
        if (node_map_.contains(std::string{human_name})) {
            throw std::invalid_argument(std::format(
                "set_alias: \"{}\" is already a canonical node name.",
                human_name));
        }
        // Branch-name canonical check via the existing helper.
        try {
            (void)branch_id_of(human_name);
            throw std::invalid_argument(std::format(
                "set_alias: \"{}\" is already a canonical branch / "
                "device name.",
                human_name));
        } catch (const std::out_of_range&) {
            // Good: not a registered branch name.
        }
        AliasKind kind = has_node ? AliasKind::Node : AliasKind::Branch;
        std::string target = has_node ? std::string{*node}
                                       : std::string{*branch};
        aliases_[std::string{human_name}] = AliasTarget{kind,
                                                         std::move(target)};
    }

    /// Read-only access to the registered alias map. Bound to
    /// Python as ``{human_name: (kind, target)}``.
    [[nodiscard]] const std::unordered_map<std::string, AliasTarget>&
    aliases() const noexcept {
        return aliases_;
    }

private:
    [[nodiscard]] static constexpr bool is_ground_alias_(
        std::string_view name) noexcept {
        return name == "gnd" || name == "GND" || name == "0";
    }

    /// Human-readable BranchKind label — used in error messages
    /// (`switch_index_of` rejection) and in `devices()[i].kind`.
    /// Matches the lowercase enum-name convention the GUI expects.
    [[nodiscard]] static constexpr std::string_view branch_kind_name_(
        topology::BranchKind k) noexcept {
        switch (k) {
            case topology::BranchKind::PassiveLinear: return "passive";
            case topology::BranchKind::Source:        return "source";
            case topology::BranchKind::Switch:        return "switch";
            case topology::BranchKind::Nonlinear:     return "nonlinear";
        }
        return "unknown";
    }

    /// Node name lookup that handles the ground sentinel (-1) by
    /// returning "gnd". Used by `devices()` to build the terminals
    /// list. The graph stores user-supplied names in `Node::name`.
    [[nodiscard]] std::string node_name_or_ground_(Index id) const {
        if (id == topology::Graph::ground()) return "gnd";
        if (id < 0 || id >= graph_.num_nodes()) return "";
        return graph_.node(id).name;
    }

    Index resolve_node_(std::string_view name) {
        if (is_ground_alias_(name)) {
            return graph_.ground();
        }
        // Heterogeneous find — no allocation when the node was
        // already registered (the common case for nets touched
        // by multiple devices).
        const auto it = node_map_.find(name);
        if (it != node_map_.end()) {
            return it->second;
        }
        // First insertion: materialise the owned string once.
        std::string owned{name};
        const Index idx = graph_.add_node(owned);
        node_map_.emplace(std::move(owned), idx);
        return idx;
    }

    /// Transparent hash + equal so `node_map_.find(string_view)`
    /// works without materialising a `std::string` per lookup.
    struct NodeKeyHash {
        using is_transparent = void;
        std::size_t operator()(std::string_view sv) const noexcept {
            return std::hash<std::string_view>{}(sv);
        }
        std::size_t operator()(const std::string& s) const noexcept {
            return std::hash<std::string_view>{}(s);
        }
        std::size_t operator()(const char* c) const noexcept {
            return std::hash<std::string_view>{}(c);
        }
    };

    /// Helper used by every `add_*` method: forwards to
    /// `graph_.add_branch`, then registers the user-supplied
    /// component name (if any) keyed on the new branch id.
    ///
    /// v2.0 Phase 1: the name is ALSO pushed into the Graph itself
    /// (audit finding `kernel-has-no-name-context-for-errors`).
    /// Before, names lived only here and never crossed into the
    /// kernel, so every solver diagnostic could speak only in
    /// integer ids and mask bitstrings; now a singular factorization
    /// or a stalled Newton can name the offending device.
    Index add_branch_(std::string_view component_name,
                      Index from, Index to,
                      topology::BranchKind kind) {
        // v2.0 (audit A.7): a DUPLICATE component name is an
        // error, not a shrug. It used to be accepted silently and
        // `branch_id_of` returned the FIRST match, so the second
        // device was unreachable by name for the rest of the run —
        // no error, no warning, and every name-based accessor,
        // trace and diagnostic pointed at the wrong branch.
        // Subsystem instancing makes this urgent: one typo'd path
        // would collide a hundred devices in silence.
        if (!component_name.empty()) {
            const auto dup = branch_index_.find(component_name);
            if (dup != branch_index_.end()) {
                throw std::invalid_argument(std::format(
                        "CircuitBuilder: component name \"{}\" is "
                        "already used by branch {}. Names must be "
                        "unique — `branch_id_of` and every "
                        "name-based accessor, trace and diagnostic "
                        "resolve by name, so a second device with "
                        "this one would be unreachable and the "
                        "first would answer for it. (Building "
                        "repeated cells? `define_subsystem` scopes "
                        "names for you: leg_a/sm17/D2.)",
                        component_name, dup->second));
            }
        }
        const Index b_id = graph_.add_branch(from, to, kind);
        if (!component_name.empty()) {
            branch_names_.emplace(b_id, std::string{component_name});
            branch_index_.emplace(std::string{component_name}, b_id);
            graph_.set_branch_name(b_id, std::string{component_name});
        }
        return b_id;
    }

    topology::Graph                         graph_;
    pwl::DevicePool                         pool_;
    std::unordered_map<std::string, Index,
                        NodeKeyHash, std::equal_to<>> node_map_;
    numeric::Dictionary<Index, std::string> branch_names_;
    /// name -> branch id. Serves BOTH the duplicate check and
    /// `branch_id_of`, which was an O(num_branches) linear scan:
    /// building a 400-cell design was quadratic in the name
    /// lookups alone (measured 8.6 ms for 2401 branches with a
    /// scan-based duplicate check, growing 7x for 4x the cells).
    std::unordered_map<std::string, Index,
                        NodeKeyHash, std::equal_to<>> branch_index_;
    // add-python-builder-ergonomics: per-branch IC storage, consumed
    // by `initial_state()`. Keyed by branch_id so `set_initial` and
    // the IC-aware overloads of add_capacitor / add_inductor share
    // the same backing dict.
    std::unordered_map<Index, Real>         initial_conditions_;
    // Alias storage. Mutable through `set_alias`; consulted by the
    // canonical lookups (`node_id_of`, `branch_id_of`).
    std::unordered_map<std::string, AliasTarget> aliases_;
};

}  // namespace pulsim::builder
