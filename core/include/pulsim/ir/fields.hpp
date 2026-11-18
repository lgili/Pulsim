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
};

struct ParamId { std::uint32_t v; };

struct FieldSpec {
    std::string_view name;
    std::string_view alias;  // authoring argument name, or empty
    Bound bound;
    Real (*get)(const ParamsVariant&);
    void (*set)(ParamsVariant&, Real);

    [[nodiscard]] constexpr bool named(std::string_view n) const noexcept {
        return n == name || (!alias.empty() && n == alias);
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
        [](ParamsVariant& e, Real v) { std::get<P>(e).*M = Real{1} / v; }};
}

namespace m = ::pulsim::models;
using Sw = pwl::DevicePool::SwitchParams;
using B = Bound;

inline constexpr std::array kResistor{
    reciprocal<m::Resistor::Params, &m::Resistor::Params::G>("R", "R_ohms"),
    direct<m::Resistor::Params, &m::Resistor::Params::G>("G", B::Positive)};
inline constexpr std::array kVoltageSource{
    direct<m::VoltageSource::Params, &m::VoltageSource::Params::V>("V", B::Finite)};
inline constexpr std::array kSwitch{
    direct<Sw, &Sw::g_on>("g_on", B::Positive),
    direct<Sw, &Sw::g_off>("g_off", B::Positive),
    reciprocal<Sw, &Sw::g_on>("R_on"),
    reciprocal<Sw, &Sw::g_off>("R_off")};
inline constexpr std::array kCapacitor{
    direct<m::Capacitor::Params, &m::Capacitor::Params::C>("C", B::Positive, "C_farads")};
inline constexpr std::array kInductor{
    direct<m::Inductor::Params, &m::Inductor::Params::L>("L", B::Positive, "L_henries")};
using SD = m::SwitchedDiode::Params;
inline constexpr std::array kDiode{
    direct<SD, &SD::g_on>("g_on", B::Positive),
    direct<SD, &SD::g_off>("g_off", B::Positive),
    direct<SD, &SD::V_th>("V_th", B::Finite),
    direct<SD, &SD::V_f>("V_f", B::NonNegative)};
using ID = m::IdealDiode::Params;
inline constexpr std::array kNonlinearDiode{
    direct<ID, &ID::V_F0>("V_F0", B::Finite),
    direct<ID, &ID::R_d>("R_d", B::Positive),
    direct<ID, &ID::G_off>("G_off", B::NonNegative),
    direct<ID, &ID::kappa>("kappa", B::Positive)};
inline constexpr std::array kCurrentSource{
    direct<m::CurrentSource::Params, &m::CurrentSource::Params::I>("I", B::Finite)};
using PW = m::PWMVoltageSource::Params;
inline constexpr std::array kPwm{
    direct<PW, &PW::v_high>("v_high", B::Finite),
    direct<PW, &PW::v_low>("v_low", B::Finite),
    direct<PW, &PW::frequency>("frequency", B::Positive),
    direct<PW, &PW::duty>("duty", B::UnitInterval),
    direct<PW, &PW::phase>("phase", B::Finite)};
using SV = m::SineVoltageSource::Params;
inline constexpr std::array kSine{
    direct<SV, &SV::v_dc>("v_dc", B::Finite),
    direct<SV, &SV::v_amplitude>("v_amplitude", B::Finite),
    direct<SV, &SV::frequency>("frequency", B::Positive),
    direct<SV, &SV::phase>("phase", B::Finite)};
using PV = m::PulseVoltageSource::Params;
inline constexpr std::array kPulse{
    direct<PV, &PV::v_initial>("v_initial", B::Finite),
    direct<PV, &PV::v_pulsed>("v_pulsed", B::Finite),
    direct<PV, &PV::t_start>("t_start", B::NonNegative),
    direct<PV, &PV::pulse_width>("pulse_width", B::NonNegative),
    direct<PV, &PV::period>("period", B::NonNegative),
    direct<PV, &PV::rise_time>("rise_time", B::NonNegative),
    direct<PV, &PV::fall_time>("fall_time", B::NonNegative)};
using ML = m::MosfetLevel1::Params;
inline constexpr std::array kMosfetL1{
    direct<ML, &ML::K>("K", B::Positive),
    direct<ML, &ML::V_T>("V_T", B::Finite),
    direct<ML, &ML::lambda>("lambda", B::NonNegative, "lambda_"),
    direct<ML, &ML::kappa>("kappa", B::Positive)};
using IG = m::IgbtLevel1::Params;
inline constexpr std::array kIgbtL1{
    direct<IG, &IG::V_CE_sat>("V_CE_sat", B::Finite),
    direct<IG, &IG::R_CE_sat>("R_CE_sat", B::Positive),
    direct<IG, &IG::V_T>("V_T", B::Finite),
    direct<IG, &IG::kappa>("kappa", B::Positive),
    direct<IG, &IG::v_knee>("v_knee", B::Positive),
    direct<IG, &IG::tau_tail>("tau_tail", B::NonNegative),
    direct<IG, &IG::k_tail>("k_tail", B::UnitInterval)};
inline constexpr std::array kVcvs{
    direct<m::VCVS::Params, &m::VCVS::Params::gain>("gain", B::Finite)};
using SI = m::SaturableInductor::Params;
inline constexpr std::array kSaturable{
    direct<SI, &SI::L_0>("L_0", B::Positive),
    direct<SI, &SI::I_sat>("I_sat", B::Positive),
    direct<SI, &SI::L_residual>("L_residual", B::NonNegative)};
using NC = m::NonlinearCapacitor::Params;
inline constexpr std::array kNonlinearCap{
    direct<NC, &NC::C0>("C0", B::Positive),
    direct<NC, &NC::V0>("V0", B::Positive),
    direct<NC, &NC::m>("m", B::Positive),
    direct<NC, &NC::v_floor>("v_floor", B::Finite),
    direct<NC, &NC::cgd_max>("cgd_max", B::NonNegative),
    direct<NC, &NC::cgd_min>("cgd_min", B::NonNegative),
    direct<NC, &NC::a>("a", B::NonNegative)};
using SH = m::ShockleyDiode::Params;
inline constexpr std::array kShockley{
    direct<SH, &SH::I_S>("I_S", B::Positive),
    direct<SH, &SH::n>("n", B::Positive),
    direct<SH, &SH::V_T>("V_T", B::Positive),
    direct<SH, &SH::G_min>("G_min", B::NonNegative),
    direct<SH, &SH::i_lim>("i_lim", B::Positive),
    direct<SH, &SH::BV>("BV", B::NonNegative)};
using LD = m::LauritzenDiode::Params;
inline constexpr std::array kLauritzen{
    direct<LD, &LD::I_S>("I_S", B::Positive),
    direct<LD, &LD::n>("n", B::Positive),
    direct<LD, &LD::V_T>("V_T", B::Positive),
    direct<LD, &LD::tau>("tau", B::Positive),
    direct<LD, &LD::T_M>("T_M", B::Positive),
    direct<LD, &LD::G_min>("G_min", B::NonNegative),
    direct<LD, &LD::i_lim>("i_lim", B::Positive)};
using PM = m::PmsmMna::Params;
inline constexpr std::array kPmsm{
    direct<PM, &PM::R_s>("R_s", B::Positive),
    direct<PM, &PM::L_d>("L_d", B::Positive),
    direct<PM, &PM::L_q>("L_q", B::Positive),
    direct<PM, &PM::L_0>("L_0", B::NonNegative),
    direct<PM, &PM::psi_pm>("psi_pm", B::NonNegative),
    direct<PM, &PM::pole_pairs>("pole_pairs", B::Positive),
    direct<PM, &PM::T_load>("T_load", B::Finite)};
inline constexpr std::array kIdealXfmr{
    direct<m::IdealTransformer::Params, &m::IdealTransformer::Params::n>("n", B::Positive)};
inline constexpr std::array kVccs{
    direct<m::VCCS::Params, &m::VCCS::Params::gm>("gm", B::Finite)};
inline constexpr std::array kCccs{
    direct<m::CCCS::Params, &m::CCCS::Params::gain>("gain", B::Finite)};
inline constexpr std::array kCcvs{
    direct<m::CCVS::Params, &m::CCVS::Params::rm>("rm", B::Finite)};
inline constexpr std::array kGyrator{
    direct<m::Gyrator::Params, &m::Gyrator::Params::g>("g", B::Finite)};
using SC = m::SineCurrentSource::Params;
inline constexpr std::array kSineI{
    nested<SC, &SC::w, &SV::v_dc>("i_dc", B::Finite),
    nested<SC, &SC::w, &SV::v_amplitude>("i_amplitude", B::Finite),
    nested<SC, &SC::w, &SV::frequency>("frequency", B::Positive),
    nested<SC, &SC::w, &SV::phase>("phase", B::Finite)};
using PC = m::PulseCurrentSource::Params;
inline constexpr std::array kPulseI{
    nested<PC, &PC::w, &PV::v_initial>("i_initial", B::Finite),
    nested<PC, &PC::w, &PV::v_pulsed>("i_pulsed", B::Finite),
    nested<PC, &PC::w, &PV::t_start>("t_start", B::NonNegative),
    nested<PC, &PC::w, &PV::pulse_width>("pulse_width", B::NonNegative),
    nested<PC, &PC::w, &PV::period>("period", B::NonNegative),
    nested<PC, &PC::w, &PV::rise_time>("rise_time", B::NonNegative),
    nested<PC, &PC::w, &PV::fall_time>("fall_time", B::NonNegative)};
// The PWL current source's knots are a table, not a scalar field: it has
// no row entries (a parameter cannot bind to it; `with_value` names it).
inline constexpr std::array<FieldSpec, 0> kPwlI{};

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
        detail::kSineI,        detail::kPulseI,        detail::kPwlI};
static_assert(kKindFields.size() == pwl::kKindCaps.size(),
              "every device kind needs a row in ir::kKindFields");

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
            if (p.law == L::Junction && vdmos_field) unread("junction");
            m::NonlinearCapacitor::validate(p);
        } else if constexpr (std::is_same_v<P, m::SaturableInductor::Params>) {
            if (p.table) unread("table");
            if (p.ja) unread("Jiles-Atherton");
        } else if constexpr (std::is_same_v<P, m::LauritzenDiode::Params>) {
            m::LauritzenDiode::validate(p);
        } else if constexpr (std::is_same_v<P, m::ShockleyDiode::Params>) {
            m::ShockleyDiode::validate(p);
        } else if constexpr (std::is_same_v<P, m::PmsmMna::Params>) {
            m::PmsmMna::validate(p);
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
