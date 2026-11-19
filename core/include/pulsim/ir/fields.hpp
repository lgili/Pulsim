#pragma once

// =============================================================================
// Pulsim — circuit IR: the value fields of every device kind
// =============================================================================
//
// add-immutable-circuit-ir, groups 1 and 6. One row per `DevicePool`
// kind (the order of `DevicePool::StoredKind` and of `kKindCaps`); each
// row lists the kind's scalar value fields: the name `with_value` takes
// ("R1.R"), an optional authoring alias (the builder argument name when
// it differs, "R_ohms"), the bound a new value must satisfy, and a
// getter / setter on the pool's parameter variant.
//
// A field is a value the builder stores verbatim (or, for R / R_on /
// R_off, as its reciprocal exactly as the builder computes it), so
// setting it equals rebuilding with the new literal, bit for bit. The
// table is generic over kinds: a new kind adds a row, and the
// `static_assert` below fails until it does.

#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <string_view>
#include <variant>

namespace pulsim::ir {

/// What a field's value must satisfy (checked by `with_value`).
enum class Bound : std::uint8_t {
    Finite,        // any finite number
    Positive,      // > 0
    NonNegative,   // >= 0
    UnitInterval,  // [0, 1]
};

[[nodiscard]] constexpr std::string_view bound_text(Bound b) noexcept {
    switch (b) {
        case Bound::Finite:       return "a finite number";
        case Bound::Positive:     return "a finite number > 0";
        case Bound::NonNegative:  return "a finite number >= 0";
        case Bound::UnitInterval: return "a number in [0, 1]";
    }
    return "?";
}

[[nodiscard]] inline bool within(Bound b, Real v) noexcept {
    if (!std::isfinite(v)) return false;
    switch (b) {
        case Bound::Finite:       return true;
        case Bound::Positive:     return v > Real{0};
        case Bound::NonNegative:  return v >= Real{0};
        case Bound::UnitInterval: return v >= Real{0} && v <= Real{1};
    }
    return false;
}

using ParamsVariant = pwl::DevicePool::ParamsVariant;

/// A named parameter (group 6): fields bound to it take its value, and
/// `Circuit::with_params` re-values them. `lo` / `hi` are its own
/// bounds, checked beside the bound of every field it drives.
struct Parameter {
    std::string name;
    Real value;
    std::string unit;
    std::optional<Real> lo;
    std::optional<Real> hi;
    std::string expr{};   // group 7: defined by this expression (empty: a literal)
    std::uint32_t scope = 0;  // group 8: the scope it was declared in (0 = root)
};

/// The physical domain of a node (group 9). All domains are solved in
/// one MNA system; the tag drives the connection check, the preflight's
/// per-domain findings and the unit a probe reports. `Magnetic` is the
/// permeance domain (add-permeance-magnetic-domain): the pool's flux
/// nodes, MMF in A.
enum class Domain : std::uint8_t { Electrical, Thermal, Mechanical, Magnetic };

inline constexpr std::array<std::string_view, 4> kDomainNames{
    "electrical", "thermal", "mechanical", "magnetic"};

[[nodiscard]] constexpr std::string_view domain_name(Domain d) noexcept {
    return kDomainNames[static_cast<std::size_t>(d)];
}

/// `"thermal"` -> Domain::Thermal; nullopt for an unknown name.
[[nodiscard]] constexpr std::optional<Domain> domain_from_name(std::string_view n) noexcept {
    for (std::size_t i = 0; i < kDomainNames.size(); ++i) {
        if (kDomainNames[i] == n) return static_cast<Domain>(i);
    }
    return std::nullopt;
}

/// A scope (group 8): one instance of a subsystem, or a path segment
/// above one. Id 0 is the root (implicit); scope `i > 0` is
/// `scopes()[i - 1]`. Members are found by name: an element belongs to
/// the deepest scope whose `path + "/"` prefixes its flattened name.
struct Scope {
    std::string name;         // its own segment ("sm17")
    std::uint32_t parent;     // 0 = the root
    std::string kind;         // the subsystem definition ("" = a plain path segment)
    std::string path;         // "leg_a/sm17"
};

struct ParamId { std::uint32_t v; };

/// A control declaration (group 3): a block the circuit carries but the
/// IR does not model — a C block (`kind == "c_block"`), a source driven
/// by a control graph (`"graph_driven"`). Opaque: `handle` belongs to the
/// binding that made it (a Python object for these two) and is passed
/// through to the run; the IR hashes `name` and `kind` (structure) and
/// `content_hash` (value), which the declaring side computes stably.
struct ControlDecl {
    std::string name;
    std::string kind;
    std::uint64_t content_hash = 0;
    std::shared_ptr<const void> handle;
};

struct FieldSpec {
    std::string_view name;
    std::string_view alias;  // authoring argument name, or empty
    Bound bound;
    Real (*get)(const ParamsVariant&);
    void (*set)(ParamsVariant&, Real);
    bool derived = false;    // computed from a stored member (R = 1/G): not serialised
    // add-unified-device-interface group 12: the documentation of the row
    // (the device reference page is generated from it) and whether
    // `components()` lists it (under `key()`): exactly the scalars it
    // listed before the table drove it, so its output is unchanged (a
    // pinned digest, test_circuit_ir.py); `Circuit.devices()` has them all.
    std::string_view unit{};
    std::string_view description{};
    bool listed = false;

    [[nodiscard]] constexpr bool named(std::string_view n) const noexcept {
        return n == name || (!alias.empty() && n == alias);
    }
    /// The authoring name: the builder argument (`alias`) when it differs.
    [[nodiscard]] constexpr std::string_view key() const noexcept {
        return alias.empty() ? name : alias;
    }
    [[nodiscard]] constexpr FieldSpec doc(std::string_view u, std::string_view d) const {
        FieldSpec f = *this;
        f.unit = u;
        f.description = d;
        return f;
    }
    [[nodiscard]] constexpr FieldSpec listed_in_components() const {
        FieldSpec f = *this;
        f.listed = true;
        return f;
    }
};

namespace detail {

template <class P, auto M>
[[nodiscard]] constexpr FieldSpec direct(std::string_view name, Bound b,
                                         std::string_view alias = {}) {
    return FieldSpec{
        name, alias, b,
        [](const ParamsVariant& e) -> Real { return std::get<P>(e).*M; },
        [](ParamsVariant& e, Real v) { std::get<P>(e).*M = v; }};
}

/// A field one struct deep (the waveform current sources wrap the
/// voltage-source waveform as `.w`).
template <class P, auto Outer, auto Inner>
[[nodiscard]] constexpr FieldSpec nested(std::string_view name, Bound b) {
    return FieldSpec{
        name, {}, b,
        [](const ParamsVariant& e) -> Real {
            return (std::get<P>(e).*Outer).*Inner;
        },
        [](ParamsVariant& e, Real v) { (std::get<P>(e).*Outer).*Inner = v; }};
}

/// A resistance stored as its conductance: the builder writes
/// `1 / R`, and so does the setter, so the stored bits equal a rebuild.
template <class P, auto M>
[[nodiscard]] constexpr FieldSpec reciprocal(std::string_view name,
                                             std::string_view alias = {}) {
    return FieldSpec{
        name, alias, Bound::Positive,
        [](const ParamsVariant& e) -> Real {
            return Real{1} / (std::get<P>(e).*M);
        },
        [](ParamsVariant& e, Real v) { std::get<P>(e).*M = Real{1} / v; },
        true};
}

namespace m = ::pulsim::models;
using Sw = pwl::DevicePool::SwitchParams;
using B = Bound;

inline constexpr std::array kResistor{
    reciprocal<m::Resistor::Params, &m::Resistor::Params::G>("R", "R_ohms").doc("Ohm", "resistance (stored as G = 1/R)").listed_in_components(),
    direct<m::Resistor::Params, &m::Resistor::Params::G>("G", B::Positive).doc("S", "conductance 1/R")};
inline constexpr std::array kVoltageSource{
    direct<m::VoltageSource::Params, &m::VoltageSource::Params::V>("V", B::Finite).doc("V", "source voltage").listed_in_components()};
inline constexpr std::array kSwitch{
    direct<Sw, &Sw::g_on>("g_on", B::Positive).doc("S", "closed-state conductance").listed_in_components(),
    direct<Sw, &Sw::g_off>("g_off", B::Positive).doc("S", "open-state conductance").listed_in_components(),
    reciprocal<Sw, &Sw::g_on>("R_on").doc("Ohm", "closed-state resistance 1/g_on"),
    reciprocal<Sw, &Sw::g_off>("R_off").doc("Ohm", "open-state resistance 1/g_off")};
inline constexpr std::array kCapacitor{
    direct<m::Capacitor::Params, &m::Capacitor::Params::C>("C", B::Positive, "C_farads").doc("F", "capacitance").listed_in_components()};
inline constexpr std::array kInductor{
    direct<m::Inductor::Params, &m::Inductor::Params::L>("L", B::Positive, "L_henries").doc("H", "inductance").listed_in_components()};
using SD = m::SwitchedDiode::Params;
inline constexpr std::array kDiode{
    direct<SD, &SD::g_on>("g_on", B::Positive).doc("S", "on-state conductance").listed_in_components(),
    direct<SD, &SD::g_off>("g_off", B::Positive).doc("S", "off-state conductance").listed_in_components(),
    direct<SD, &SD::V_th>("V_th", B::Finite).doc("V", "commutation threshold (not a drop)").listed_in_components(),
    direct<SD, &SD::V_f>("V_f", B::NonNegative).doc("V", "forward conduction drop").listed_in_components()};
using ID = m::IdealDiode::Params;
inline constexpr std::array kNonlinearDiode{
    direct<ID, &ID::V_F0>("V_F0", B::Finite).doc("V", "knee voltage of the sigmoid"),
    direct<ID, &ID::R_d>("R_d", B::Positive).doc("Ohm", "forward slope resistance"),
    direct<ID, &ID::G_off>("G_off", B::NonNegative).doc("S", "reverse leakage conductance"),
    direct<ID, &ID::kappa>("kappa", B::Positive).doc("1/V", "sigmoid sharpness")};
inline constexpr std::array kCurrentSource{
    direct<m::CurrentSource::Params, &m::CurrentSource::Params::I>("I", B::Finite).doc("A", "source current, out of the from node").listed_in_components()};
using PW = m::PWMVoltageSource::Params;
inline constexpr std::array kPwm{
    direct<PW, &PW::v_high>("v_high", B::Finite).doc("V", "high level"),
    direct<PW, &PW::v_low>("v_low", B::Finite).doc("V", "low level"),
    direct<PW, &PW::frequency>("frequency", B::Positive).doc("Hz", "switching frequency"),
    direct<PW, &PW::duty>("duty", B::UnitInterval).doc("1", "duty cycle"),
    direct<PW, &PW::phase>("phase", B::Finite).doc("rad", "carrier phase")};
using SV = m::SineVoltageSource::Params;
inline constexpr std::array kSine{
    direct<SV, &SV::v_dc>("v_dc", B::Finite).doc("V", "DC offset"),
    direct<SV, &SV::v_amplitude>("v_amplitude", B::Finite).doc("V", "amplitude"),
    direct<SV, &SV::frequency>("frequency", B::Positive).doc("Hz", "frequency").listed_in_components(),
    direct<SV, &SV::phase>("phase", B::Finite).doc("rad", "phase")};
using PV = m::PulseVoltageSource::Params;
inline constexpr std::array kPulse{
    direct<PV, &PV::v_initial>("v_initial", B::Finite).doc("V", "initial level"),
    direct<PV, &PV::v_pulsed>("v_pulsed", B::Finite).doc("V", "pulsed level"),
    direct<PV, &PV::t_start>("t_start", B::NonNegative).doc("s", "delay before the first edge"),
    direct<PV, &PV::pulse_width>("pulse_width", B::NonNegative).doc("s", "time at the pulsed level"),
    direct<PV, &PV::period>("period", B::NonNegative).doc("s", "period (0: one pulse)"),
    direct<PV, &PV::rise_time>("rise_time", B::NonNegative).doc("s", "rise time"),
    direct<PV, &PV::fall_time>("fall_time", B::NonNegative).doc("s", "fall time")};
using ML = m::MosfetLevel1::Params;
inline constexpr std::array kMosfetL1{
    direct<ML, &ML::K>("K", B::Positive).doc("A/V^2", "transconductance parameter"),
    direct<ML, &ML::V_T>("V_T", B::Finite).doc("V", "gate threshold"),
    direct<ML, &ML::lambda>("lambda", B::NonNegative, "lambda_").doc("1/V", "channel-length modulation"),
    direct<ML, &ML::kappa>("kappa", B::Positive).doc("1/V", "sigmoid sharpness")};
using IG = m::IgbtLevel1::Params;
inline constexpr std::array kIgbtL1{
    direct<IG, &IG::V_CE_sat>("V_CE_sat", B::Finite).doc("V", "saturation voltage"),
    direct<IG, &IG::R_CE_sat>("R_CE_sat", B::Positive).doc("Ohm", "on-state slope resistance"),
    direct<IG, &IG::V_T>("V_T", B::Finite).doc("V", "gate threshold"),
    direct<IG, &IG::kappa>("kappa", B::Positive).doc("1/V", "sigmoid sharpness"),
    direct<IG, &IG::v_knee>("v_knee", B::Positive).doc("V", "smooth-max knee width"),
    direct<IG, &IG::tau_tail>("tau_tail", B::NonNegative).doc("s", "tail-current time constant (0: no tail)"),
    direct<IG, &IG::k_tail>("k_tail", B::UnitInterval).doc("1", "tail fraction of the turn-off current")};
inline constexpr std::array kVcvs{
    direct<m::VCVS::Params, &m::VCVS::Params::gain>("gain", B::Finite).doc("V/V", "voltage gain")};
using SI = m::SaturableInductor::Params;
inline constexpr std::array kSaturable{
    direct<SI, &SI::L_0>("L_0", B::Positive).doc("H", "unsaturated inductance"),
    direct<SI, &SI::I_sat>("I_sat", B::Positive).doc("A", "saturation current"),
    direct<SI, &SI::L_residual>("L_residual", B::NonNegative).doc("H", "saturated floor inductance (0: none)")};
using NC = m::NonlinearCapacitor::Params;
inline constexpr std::array kNonlinearCap{
    direct<NC, &NC::C0>("C0", B::Positive).doc("F", "junction law: zero-bias capacitance"),
    direct<NC, &NC::V0>("V0", B::Positive).doc("V", "junction law: built-in potential (tanh law: voltage scale)"),
    direct<NC, &NC::m>("m", B::Positive).doc("1", "junction law: grading coefficient"),
    direct<NC, &NC::v_floor>("v_floor", B::Finite).doc("1", "junction law: clamp on v/V0 (> -1)"),
    direct<NC, &NC::cgd_max>("cgd_max", B::NonNegative).doc("F", "vdmos/tanh law: maximum capacitance"),
    direct<NC, &NC::cgd_min>("cgd_min", B::NonNegative).doc("F", "vdmos/tanh law: minimum capacitance"),
    direct<NC, &NC::a>("a", B::NonNegative).doc("1/V", "vdmos law: transition slope")};
using SH = m::ShockleyDiode::Params;
inline constexpr std::array kShockley{
    direct<SH, &SH::I_S>("I_S", B::Positive).doc("A", "saturation current"),
    direct<SH, &SH::n>("n", B::Positive).doc("1", "emission coefficient"),
    direct<SH, &SH::V_T>("V_T", B::Positive).doc("V", "thermal voltage kT/q"),
    direct<SH, &SH::G_min>("G_min", B::NonNegative).doc("S", "parallel conductance"),
    direct<SH, &SH::i_lim>("i_lim", B::Positive).doc("A", "current at which the exponential turns linear"),
    direct<SH, &SH::BV>("BV", B::NonNegative).doc("V", "reverse breakdown voltage (0: none)")};
using EX = m::ExcessLoss::Params;
inline constexpr std::array kExcessLoss{
    direct<EX, &EX::c>("c", B::Positive).doc("A/V^0.5", "excess-loss coefficient: F = c*sign(u)*|u|^0.5 for |u| >> eps").listed_in_components(),
    direct<EX, &EX::eps>("eps", B::Positive).doc("V", "flux rate below which the law is linear (slope c/sqrt(eps))").listed_in_components()};
using LD = m::LauritzenDiode::Params;
inline constexpr std::array kLauritzen{
    direct<LD, &LD::I_S>("I_S", B::Positive).doc("A", "saturation current"),
    direct<LD, &LD::n>("n", B::Positive).doc("1", "emission coefficient"),
    direct<LD, &LD::V_T>("V_T", B::Positive).doc("V", "thermal voltage kT/q"),
    direct<LD, &LD::tau>("tau", B::Positive).doc("s", "carrier lifetime"),
    direct<LD, &LD::T_M>("T_M", B::Positive).doc("s", "transit time (< tau)"),
    direct<LD, &LD::G_min>("G_min", B::NonNegative).doc("S", "parallel conductance"),
    direct<LD, &LD::i_lim>("i_lim", B::Positive).doc("A", "current at which the exponential turns linear")};
using PM = m::PmsmMna::Params;
inline constexpr std::array kPmsm{
    direct<PM, &PM::R_s>("R_s", B::Positive).doc("Ohm", "stator resistance"),
    direct<PM, &PM::L_d>("L_d", B::Positive).doc("H", "d-axis inductance"),
    direct<PM, &PM::L_q>("L_q", B::Positive).doc("H", "q-axis inductance"),
    direct<PM, &PM::L_0>("L_0", B::NonNegative).doc("H", "zero-sequence inductance (0: the dq average)"),
    direct<PM, &PM::psi_pm>("psi_pm", B::NonNegative).doc("Wb", "permanent-magnet flux linkage"),
    direct<PM, &PM::pole_pairs>("pole_pairs", B::Positive).doc("1", "pole pairs"),
    direct<PM, &PM::T_load>("T_load", B::Finite).doc("N*m", "constant load torque")};
using IM = m::InductionMachineMna::Params;
inline constexpr std::array kInductionMachine{
    direct<IM, &IM::R_s>("R_s", B::NonNegative).doc("Ohm", "stator resistance"),
    direct<IM, &IM::L_ls>("L_ls", B::Positive).doc("H", "stator leakage inductance"),
    direct<IM, &IM::R_r>("R_r", B::Positive).doc("Ohm", "rotor resistance, referred"),
    direct<IM, &IM::L_lr>("L_lr", B::Positive).doc("H", "rotor leakage inductance, referred"),
    direct<IM, &IM::L_m>("L_m", B::Positive).doc("H", "magnetising inductance"),
    direct<IM, &IM::pole_pairs>("pole_pairs", B::Positive).doc("1", "pole pairs"),
    direct<IM, &IM::T_load>("T_load", B::Finite).doc("N*m", "constant load torque")};
inline constexpr std::array kIdealXfmr{
    direct<m::IdealTransformer::Params, &m::IdealTransformer::Params::n>("n", B::Positive).doc("1", "turns ratio").listed_in_components()};
inline constexpr std::array kVccs{
    direct<m::VCCS::Params, &m::VCCS::Params::gm>("gm", B::Finite).doc("S", "transconductance").listed_in_components()};
inline constexpr std::array kCccs{
    direct<m::CCCS::Params, &m::CCCS::Params::gain>("gain", B::Finite).doc("A/A", "current gain").listed_in_components()};
inline constexpr std::array kCcvs{
    direct<m::CCVS::Params, &m::CCVS::Params::rm>("rm", B::Finite).doc("Ohm", "transresistance").listed_in_components()};
inline constexpr std::array kGyrator{
    direct<m::Gyrator::Params, &m::Gyrator::Params::g>("g", B::Finite).doc("S", "gyration conductance").listed_in_components()};
using SC = m::SineCurrentSource::Params;
inline constexpr std::array kSineI{
    nested<SC, &SC::w, &SV::v_dc>("i_dc", B::Finite).doc("A", "DC offset").listed_in_components(),
    nested<SC, &SC::w, &SV::v_amplitude>("i_amplitude", B::Finite).doc("A", "amplitude").listed_in_components(),
    nested<SC, &SC::w, &SV::frequency>("frequency", B::Positive).doc("Hz", "frequency").listed_in_components(),
    nested<SC, &SC::w, &SV::phase>("phase", B::Finite).doc("rad", "phase").listed_in_components()};
using PC = m::PulseCurrentSource::Params;
inline constexpr std::array kPulseI{
    nested<PC, &PC::w, &PV::v_initial>("i_initial", B::Finite).doc("A", "initial level").listed_in_components(),
    nested<PC, &PC::w, &PV::v_pulsed>("i_pulsed", B::Finite).doc("A", "pulsed level").listed_in_components(),
    nested<PC, &PC::w, &PV::t_start>("t_start", B::NonNegative).doc("s", "delay before the first edge").listed_in_components(),
    nested<PC, &PC::w, &PV::pulse_width>("pulse_width", B::NonNegative).doc("s", "time at the pulsed level").listed_in_components(),
    nested<PC, &PC::w, &PV::period>("period", B::NonNegative).doc("s", "period (0: one pulse)").listed_in_components(),
    nested<PC, &PC::w, &PV::rise_time>("rise_time", B::NonNegative).doc("s", "rise time").listed_in_components(),
    nested<PC, &PC::w, &PV::fall_time>("fall_time", B::NonNegative).doc("s", "fall time").listed_in_components()};
// The PWL current source's knots are a table, not a scalar field: it has
// no row entries (a parameter cannot bind to it; `with_value` names it).
inline constexpr std::array<FieldSpec, 0> kPwlI{};
// An external device's parameters live in its model's field table (run-time
// names), not in scalar rows; its values are hashed by content (circuit.hpp).
inline constexpr std::array<FieldSpec, 0> kExternal{};

}  // namespace detail

/// Field rows indexed by `DevicePool::StoredKind`.
inline constexpr std::array<std::span<const FieldSpec>, pwl::DevicePool::kNumKinds>
    kKindFields{
        detail::kResistor,     detail::kVoltageSource, detail::kSwitch,
        detail::kCapacitor,    detail::kInductor,      detail::kDiode,
        detail::kNonlinearDiode, detail::kCurrentSource, detail::kPwm,
        detail::kSine,         detail::kPulse,         detail::kMosfetL1,
        detail::kIgbtL1,       detail::kVcvs,          detail::kSaturable,
        detail::kNonlinearCap, detail::kShockley,      detail::kLauritzen,
        detail::kPmsm,         detail::kIdealXfmr,     detail::kVccs,
        detail::kCccs,         detail::kCcvs,          detail::kGyrator,
        detail::kSineI,        detail::kPulseI,        detail::kPwlI,
        detail::kInductionMachine, detail::kExternal, detail::kExcessLoss};
static_assert(kKindFields.size() == pwl::kKindCaps.size(),
              "every device kind needs a row in ir::kKindFields");

/// Every row carries its unit and description (the reference page and
/// the builder's error messages read them).
[[nodiscard]] consteval bool every_field_documented() {
    for (const auto row : kKindFields) {
        for (const auto& f : row) {
            if (f.unit.empty() || f.description.empty()) return false;
        }
    }
    return true;
}
static_assert(every_field_documented(), "an ir::kKindFields row lacks .doc(unit, description)");

[[nodiscard]] inline std::span<const FieldSpec> fields_of(
    pwl::DevicePool::StoredKind k) noexcept {
    return kKindFields[static_cast<std::size_t>(k)];
}

/// What a device's `add_*` checks beyond one field's bound, applied to
/// its values after one field changed (`field`): the model's own
/// cross-field `validate`, and a refusal for a field the device's law
/// does not read (the new value would be silently ignored). Throws
/// `std::invalid_argument`.
inline void check_device(const ParamsVariant& v, std::string_view field) {
    namespace m = ::pulsim::models;
    std::visit([&](const auto& p) {
        using P = std::decay_t<decltype(p)>;
        auto unread = [&](std::string_view law) {
            throw std::invalid_argument(std::string{"its "} + std::string{law} +
                " law does not read " + std::string{field} +
                "; rebuild the device with the new value");
        };
        if constexpr (std::is_same_v<P, m::NonlinearCapacitor::Params>) {
            using L = m::ChargeLawKind;
            const bool vdmos_field = field == "cgd_max" || field == "cgd_min" || field == "a";
            if (p.law == L::Table) unread("table");
            if (p.law == L::VdmosCgd && !vdmos_field) unread("vdmos_cgd");
            if (p.law == L::Tanh && field != "cgd_max" && field != "cgd_min" && field != "V0") {
                unread("tanh");
            }
            if (p.law == L::Junction && vdmos_field) unread("junction");
            m::NonlinearCapacitor::validate(p);
        } else if constexpr (std::is_same_v<P, m::SaturableInductor::Params>) {
            if (p.table) unread("table");
            if (p.ja) unread("Jiles-Atherton");
        } else if constexpr (std::is_same_v<P, m::LauritzenDiode::Params>) {
            m::LauritzenDiode::validate(p);
        } else if constexpr (std::is_same_v<P, m::ShockleyDiode::Params>) {
            m::ShockleyDiode::validate(p);
        } else if constexpr (std::is_same_v<P, m::ExcessLoss::Params>) {
            m::ExcessLoss::validate(p);
        } else if constexpr (std::is_same_v<P, m::PmsmMna::Params>) {
            if (p.harmonics && field == "psi_pm") unread("PM-harmonic");
            if (p.flux_map && (field == "psi_pm" || field == "L_d" || field == "L_q")) {
                unread("flux-map");
            }
            m::PmsmMna::validate(p);
        } else if constexpr (std::is_same_v<P, m::InductionMachineMna::Params>) {
            m::InductionMachineMna::validate(p);
        } else if constexpr (std::is_same_v<P, m::IgbtLevel1::Params>) {
            // As add_igbt_level1: k_tail < 1, and a tail needs both.
            if (!(p.k_tail < Real{1})) {
                throw std::invalid_argument("k_tail must be < 1 (1 means "
                    "the channel carries nothing at all)");
            }
            if ((p.tau_tail > Real{0}) != (p.k_tail > Real{0})) {
                throw std::invalid_argument("tau_tail and k_tail must be "
                    "set together (one alone means no tail, silently); "
                    "rebuild the device with both");
            }
        }
    }, v);
}

[[nodiscard]] inline const FieldSpec* find_field(
    pwl::DevicePool::StoredKind k, std::string_view name) noexcept {
    for (const auto& f : fields_of(k)) {
        if (f.named(name)) return &f;
    }
    return nullptr;
}

}  // namespace pulsim::ir
