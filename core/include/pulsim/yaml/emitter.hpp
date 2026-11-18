#pragma once

// add-immutable-circuit-ir group 4: the YAML round trip of a Circuit.
//
// `to_yaml(c)` writes the LOWERED circuit: its nodes in unknown order,
// one entry per registry kind instance in branch order (a PMSM's three
// phase branches as one entry), then the side records the pool and the
// builder keep (couplings, branch-form order, parameters and their
// bindings, initial conditions, aliases, power-MOSFET records, control
// declarations). `from_yaml(text)` replays it into a builder in the same
// order, so every index, counter and hash comes back equal.
//
// Schema: the scalar keys of a kind are its non-derived rows of
// `ir::kKindFields` (the stored members, so a reciprocal such as R = 1/G
// is written as G and no division happens on the way back); the kind's
// YAML name is `kKindYaml`, checked against `pwl::kKindCaps` at compile
// time. Defaults (the value of a value-initialised `Params`) are
// omitted, bit for bit. Numbers are written shortest-round-trip
// (`std::to_chars`: at most 17 significant digits, locale-free) and read
// with `std::from_chars`, so `to_yaml` is a fixed point after one round.
//
// Not serialisable, and refused naming the device: a C block (its step
// function is a Python callable) and an unnamed device another entry
// must refer to.

#include "pulsim/ir/circuit.hpp"
#include "pulsim/models/charge_table.hpp"
#include "pulsim/models/flux_table.hpp"
#include "pulsim/models/jiles_atherton.hpp"
#include "pulsim/models/power_mosfet.hpp"
#include "pulsim/numeric/parse_real.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace pulsim::ir {

/// YAML `kind:` of each registry kind, in `DevicePool::StoredKind` order.
inline constexpr std::array<std::string_view, pwl::DevicePool::kNumKinds> kKindYaml{
    "resistor", "voltage_source", "switch", "capacitor", "inductor",
    "diode", "nonlinear_diode", "current_source", "pwm_voltage_source",
    "sine_voltage_source", "pulse_voltage_source", "mosfet_level1",
    "igbt_level1", "vcvs", "saturable_inductor", "nonlinear_capacitor",
    "shockley_diode", "lauritzen_diode", "pmsm_mna", "ideal_transformer",
    "vccs", "cccs", "ccvs", "gyrator", "sine_current_source",
    "pulse_current_source", "pwl_current_source"};

namespace yaml_detail {

/// "PWMVoltageSource" matches "pwm_voltage_source": equal once case and
/// underscores are ignored.
[[nodiscard]] constexpr bool same_kind_name(std::string_view caps,
                                            std::string_view yaml) noexcept {
    std::size_t i = 0, j = 0;
    auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c; };
    while (true) {
        while (j < yaml.size() && yaml[j] == '_') ++j;
        if (i == caps.size() || j == yaml.size()) return i == caps.size() && j == yaml.size();
        if (lower(caps[i]) != yaml[j]) return false;
        ++i; ++j;
    }
}

[[nodiscard]] consteval bool every_kind_named() {
    for (std::size_t k = 0; k < kKindYaml.size(); ++k) {
        if (!same_kind_name(pwl::kKindCaps[k].name, kKindYaml[k])) return false;
    }
    return true;
}
static_assert(every_kind_named(),
              "ir::kKindYaml needs one YAML name per registry kind, in kKindCaps order");

using K = pwl::DevicePool::StoredKind;
namespace m = ::pulsim::models;

/// The graph role each kind's builder call gives its branch; written
/// only when a branch differs.
[[nodiscard]] inline topology::BranchKind default_role(K k) noexcept {
    using BK = topology::BranchKind;
    switch (k) {
        case K::Resistor: case K::Capacitor: case K::Inductor:
            return BK::PassiveLinear;
        case K::Switch: case K::Diode:
            return BK::Switch;
        case K::NonlinearDiode: case K::MosfetLevel1: case K::IgbtLevel1:
        case K::SaturableInductor: case K::NonlinearCapacitor:
        case K::ShockleyDiode: case K::LauritzenDiode: case K::PmsmMna:
            return BK::Nonlinear;
        default:
            return BK::Source;
    }
}
inline constexpr std::array<std::string_view, 4> kRoles{
    "passive", "source", "switch", "nonlinear"};

/// Number of pins of a kind's entry (from, to, then its control pins).
[[nodiscard]] inline std::size_t pin_count(K k) noexcept {
    switch (k) {
        case K::MosfetLevel1: case K::IgbtLevel1: return 3;
        case K::VCVS: case K::IdealTransformer: case K::VCCS: case K::Gyrator:
            return 4;
        default: return 2;
    }
}

[[nodiscard]] inline ParamsVariant make_params(std::size_t k) {
    ParamsVariant out;
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        ((k == I ? (void)out.template emplace<I>() : void()), ...);
    }(std::make_index_sequence<std::variant_size_v<ParamsVariant>>{});
    return out;
}

// ---------------------------------------------------------------- writing

[[nodiscard]] inline std::string num(Real x) {
    if (std::isnan(x)) return ".nan";
    if (std::isinf(x)) return x > 0 ? ".inf" : "-.inf";
    std::array<char, 32> buf{};
    const auto r = std::to_chars(buf.data(), buf.data() + buf.size(), x);
    return std::string(buf.data(), r.ptr);
}

[[nodiscard]] inline std::string quote(std::string_view s) {
    std::string o = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 0x20) {
            o += std::format("\\x{:02x}", static_cast<unsigned>(static_cast<unsigned char>(c)));
        } else {
            o += c;
        }
    }
    return o + '"';
}

[[nodiscard]] inline std::string nums(const std::vector<Real>& v) {
    std::string o = "[";
    for (std::size_t i = 0; i < v.size(); ++i) o += (i ? ", " : "") + num(v[i]);
    return o + "]";
}

/// A non-finite number met while writing key `key` (to_yaml names the
/// entry): NaN / inf cannot round-trip as a value of a circuit.
struct NonFinite { std::string key, token; };

/// The non-finite token in `v` (a number or a list of numbers), or "".
[[nodiscard]] inline std::string_view non_finite_in(std::string_view v) noexcept {
    if (v.empty() || v.front() == '"' || v.front() == '{' ||
        v.find('"') != std::string_view::npos) return {};
    for (const std::string_view t : {std::string_view{"-.inf"}, std::string_view{".inf"},
                                     std::string_view{".nan"}}) {
        if (v.find(t) != std::string_view::npos) return t;
    }
    return {};
}

/// A flow map built key by key; values are already YAML text.
struct Map {
    std::string s;
    Map& add(std::string_view k, std::string v) {
        if (const auto t = non_finite_in(v); !t.empty()) {
            throw NonFinite{std::string{k}, std::string{t}};
        }
        s += s.empty() ? "{" : ", ";
        s += k; s += ": "; s += v;
        return *this;
    }
    [[nodiscard]] std::string str() const { return s.empty() ? "{}" : s + "}"; }
};

[[nodiscard]] inline bool same_bits(Real a, Real b) noexcept {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

/// The non-derived rows of `row` whose value differs from the default.
/// `skip` names a field written elsewhere (a branch-form g_on = 1/R_on).
inline void put_fields(Map& m, const ParamsVariant& v, std::span<const FieldSpec> row,
                       std::string_view skip = {}) {
    const ParamsVariant d = make_params(v.index());
    for (const auto& f : row) {
        if (f.derived || (!skip.empty() && f.name == skip)) continue;
        const Real x = f.get(v);
        if (!same_bits(x, f.get(d))) m.add(f.name, num(x));
    }
}

template <class P>
[[nodiscard]] inline std::string sub_fields(const P& p, K kind) {
    Map m;
    put_fields(m, ParamsVariant{p}, fields_of(kind));
    return m.str();
}

[[nodiscard]] inline std::string flux_table(const m::FluxTable& t) {
    return Map{}.add("i", nums(t.i_knots())).add("lambda", nums(t.lambda_knots()))
        .add("slope", nums(t.slope_knots())).add("L_tail", num(t.L_residual())).str();
}

[[nodiscard]] inline std::string charge_table(const m::ChargeTable& t) {
    return Map{}.add("v", nums(t.v_knots())).add("q", nums(t.q_knots()))
        .add("c", nums(t.c_knots())).str();
}

[[nodiscard]] inline std::string ja_core(const m::JilesAthertonCore::Params& p) {
    return Map{}.add("N", num(p.N)).add("Ae", num(p.Ae)).add("le", num(p.le))
        .add("lg", num(p.lg)).add("Ms", num(p.ja.Ms)).add("a", num(p.ja.a))
        .add("alpha", num(p.ja.alpha)).add("c", num(p.ja.c)).add("k", num(p.ja.k))
        .add("substeps_min", std::to_string(p.substeps_min))
        .add("substep_frac", num(p.substep_frac)).add("M0", num(p.M0)).str();
}

inline constexpr std::array<std::string_view, 3> kLaws{"junction", "table", "vdmos_cgd"};

[[nodiscard]] inline std::string capacitance(const m::PowerMosfet::Capacitance& c) {
    if (const Real* x = std::get_if<Real>(&c)) return num(*x);
    const auto& q = std::get<m::NonlinearCapacitor::Params>(c);
    Map mm;
    put_fields(mm, ParamsVariant{q}, fields_of(K::NonlinearCapacitor));
    if (q.law != m::ChargeLawKind::Junction) {
        mm.add("law", std::string{kLaws[static_cast<std::size_t>(q.law)]});
    }
    if (q.table) mm.add("table", charge_table(*q.table));
    return mm.str();
}

}  // namespace yaml_detail

namespace yaml_detail {
[[nodiscard]] inline std::string to_yaml_body_(const Circuit& c, std::string& where);
}  // namespace yaml_detail

/// The circuit as YAML (see the file comment). Throws
/// `std::invalid_argument` naming what cannot be written, a NaN or an
/// infinity among them (`to_yaml: device "R1": "G" is .nan ...`).
[[nodiscard]] inline std::string to_yaml(const Circuit& c) {
    std::string where = "the document";
    try {
        return yaml_detail::to_yaml_body_(c, where);
    } catch (const yaml_detail::NonFinite& e) {
        throw std::invalid_argument(std::format(
            "to_yaml: {}: \"{}\" is {}; a non-finite value cannot be written "
            "(set a finite one)", where, e.key, e.token));
    }
}

[[nodiscard]] inline std::string yaml_detail::to_yaml_body_(const Circuit& c,
                                                            std::string& where) {
    using namespace yaml_detail;
    const auto& g = c.graph();
    const auto& pool = c.pool();
    const auto& b = c.lowered();
    auto node = [&](Index i) {
        return quote(i == topology::Graph::ground() ? std::string_view{"gnd"}
                                                    : std::string_view{g.node(i).name});
    };
    auto bname = [&](Index bi, std::string_view who) {
        const auto n = g.branch_name(bi);
        if (n.empty()) {
            throw std::invalid_argument(std::format(
                "to_yaml: {} refers to branch {}, which has no name", who, bi));
        }
        return quote(n);
    };

    // A node declared non-electrical is written {name, domain}.
    auto node_entry = [&](Index i) {
        const auto& dd = b.declared_domains();
        const auto it = dd.find(i);
        if (it == dd.end() || it->second == Domain::Electrical) return node(i);
        return yaml_detail::Map{}.add("name", node(i))
            .add("domain", std::string{domain_name(it->second)}).str();
    };
    std::string o = "pulsim_circuit: 1\nnodes: [";
    for (Index i = 0; i < g.num_nodes(); ++i) o += (i ? ", " : "") + node_entry(i);
    o += "]\ndevices:\n";

    const auto& machines = pool.pmsm_mna_machines();
    const auto devs = c.devices();
    for (Index bi = 0; bi < g.num_branches(); ++bi) {
        const auto kind = pool.kind_of(bi);
        const auto& dev = devs[static_cast<std::size_t>(bi)];
        const auto& prm = pool.params_of(bi);
        where = std::format("device \"{}\"", dev.name);
        Map e;
        e.add("kind", std::string{kKindYaml[static_cast<std::size_t>(kind)]});
        if (kind == K::PmsmMna) {
            const auto mach = std::ranges::find_if(machines, [&](const auto& mc) {
                return std::ranges::find(mc.branch_id, bi) != mc.branch_id.end();
            });
            if (mach->branch_id[0] != bi) continue;  // written with phase a
            for (std::size_t ph = 1; ph < 3; ++ph) {
                if (mach->branch_id[ph] != bi + static_cast<Index>(ph)) {
                    throw std::invalid_argument(std::format(
                        "to_yaml: PMSM \"{}\": its phase branches are not "
                        "consecutive", dev.name));
                }
            }
            std::string phases = "[";
            for (std::size_t ph = 0; ph < 3; ++ph) {
                const auto& br = g.branch(mach->branch_id[ph]);
                phases += (ph ? ", " : "") + Map{}
                    .add("name", quote(g.branch_name(mach->branch_id[ph])))
                    .add("pins", "[" + node(br.from) + ", " + node(br.to) + "]").str();
            }
            e.add("phases", phases + "]");
            e.add("omega", node(mach->omega_node)).add("theta", node(mach->theta_node));
            put_fields(e, prm, fields_of(kind));
            o += "  - " + e.str() + "\n";
            continue;
        }
        e.add("name", quote(dev.name));
        std::string pins = "[";
        for (std::size_t p = 0; p < dev.pins.size(); ++p) {
            pins += (p ? ", " : "") + node(dev.pins[p]);
        }
        e.add("pins", pins + "]");
        if (const auto role = g.branch(bi).kind; role != default_role(kind)) {
            e.add("role", std::string{kRoles[static_cast<std::size_t>(role)]});
        }
        if (kind == K::CCCS || kind == K::CCVS) {
            e.add("control", bname(pool.control_branch(bi), dev.name));
        }
        // a branch-form g_on is 1/R_on (inf when ideal): branch_form carries R_on
        put_fields(e, prm, fields_of(kind), pool.is_branch_form(bi) ? "g_on" : "");
        std::visit([&](const auto& p) {
            using P = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<P, m::SaturableInductor::Params>) {
                if (p.table) e.add("table", flux_table(*p.table));
                if (p.ja) e.add("ja", ja_core(*p.ja));
            } else if constexpr (std::is_same_v<P, m::NonlinearCapacitor::Params>) {
                if (p.law != m::ChargeLawKind::Junction) {
                    e.add("law", std::string{kLaws[static_cast<std::size_t>(p.law)]});
                }
                if (p.table) e.add("table", charge_table(*p.table));
            } else if constexpr (std::is_same_v<P, m::PwlCurrentSource::Params>) {
                e.add("t", nums(p.w.t)).add("i", nums(p.w.v));
                if (p.w.periodic) e.add("periodic", "true");
            }
        }, prm);
        o += "  - " + e.str() + "\n";
    }

    if (const auto& cs = pool.transformer_couplings(); !cs.empty()) {
        o += "couplings:\n";
        for (const auto& cp : cs) {
            where = "a coupling";
            o += "  - " + Map{}.add("primary", bname(cp.primary_branch_id, "a coupling"))
                .add("secondary", bname(cp.secondary_branch_id, "a coupling"))
                .add("L_p", num(cp.params.L_p)).add("L_s", num(cp.params.L_s))
                .add("k", num(cp.params.k)).str() + "\n";
        }
    }
    std::vector<std::pair<Index, Index>> forms;  // (state index, branch)
    for (Index bi = 0; bi < g.num_branches(); ++bi) {
        if (pool.is_branch_form(bi)) forms.emplace_back(pool.branch_var_id_for_switch(bi, g), bi);
    }
    if (!forms.empty()) {
        std::ranges::sort(forms);
        o += "branch_form:\n";
        for (const auto& [_, bi] : forms) {
            where = "a branch form";
            o += "  - " + Map{}.add("device", bname(bi, "branch form"))
                .add("R_on", num(pool.branch_R_on(bi))).str() + "\n";
        }
    }
    if (const auto sc = b.scopes(); !sc.empty()) {
        o += "scopes:\n";   // parents before children (creation order)
        for (const auto& x : sc) {
            Map mm;
            mm.add("path", quote(x.path));
            if (!x.kind.empty()) mm.add("kind", quote(x.kind));
            o += "  - " + mm.str() + "\n";
        }
    }
    if (const auto ps = b.parameters(); !ps.empty()) {
        o += "parameters:\n";
        for (const auto& p : ps) {
            where = std::format("parameter \"{}\"", p.name);
            Map mm;
            mm.add("name", quote(p.name));
            if (p.scope != 0) mm.add("scope", quote(b.scope_path(p.scope)));
            if (p.expr.empty()) mm.add("value", num(p.value));
            else mm.add("expr", quote(p.expr));
            if (!p.unit.empty()) mm.add("unit", quote(p.unit));
            if (p.lo) mm.add("lo", num(*p.lo));
            if (p.hi) mm.add("hi", num(*p.hi));
            o += "  - " + mm.str() + "\n";
        }
    }
    if (const auto bs = b.param_bindings(); !bs.empty()) {
        o += "bindings:\n";
        for (const auto& bd : bs) {
            where = "a binding";
            const auto kind = pool.kind_of(bd.branch);
            o += "  - " + Map{}.add("param", quote(b.parameters()[bd.param].name))
                .add("field", quote(std::format("{}.{}", g.branch_name(bd.branch),
                                                fields_of(kind)[bd.field].name)))
                .str() + "\n";
        }
    }
    if (const auto ics = c.initial_conditions(); !ics.empty()) {
        o += "initial_conditions:\n";
        for (const auto& ic : ics) {
            where = "an initial condition";
            o += "  - " + Map{}.add("device", bname(static_cast<Index>(ic.device.v), "an initial condition"))
                .add("value", num(ic.value)).str() + "\n";
        }
    }
    if (const auto as = c.aliases(); !as.empty()) {
        o += "aliases:\n";
        for (const auto& a : as) {
            where = "an alias";
            o += "  - " + Map{}.add("name", quote(a.name))
                .add(a.node ? "node" : "branch", quote(a.target)).str() + "\n";
        }
    }
    std::vector<Index> channels;
    for (Index bi = 0; bi < g.num_branches(); ++bi) {
        if (b.power_mosfet_record(bi)) channels.push_back(bi);
    }
    if (!channels.empty()) {
        o += "power_mosfets:\n";
        for (const Index ch : channels) {
            const auto& r = *b.power_mosfet_record(ch);
            where = std::format("power MOSFET \"{}\"", g.branch_name(ch));
            const auto& p = r.params;
            std::string members = "[";
            for (std::size_t i = 0; i < r.sub_elements.size(); ++i) {
                members += (i ? ", " : "") + quote(r.sub_elements[i]);
            }
            Map mm;
            mm.add("name", quote(r.name)).add("channel", bname(ch, r.name))
                .add("drain", quote(r.drain)).add("source", quote(r.source))
                .add("gate", quote(r.gate)).add("members", members + "]")
                .add("K_eff", num(r.K_eff)).add("V_T_eff", num(r.V_T_eff))
                .add("R_d_eff", num(r.R_d_eff))
                .add("K", num(p.K)).add("V_T", num(p.V_T)).add("lambda", num(p.lambda))
                .add("kappa", num(p.kappa)).add("R_d", num(p.R_d)).add("C_gs", num(p.C_gs))
                .add("C_gd", capacitance(p.C_gd)).add("C_ds", capacitance(p.C_ds))
                .add("R_g", num(p.R_g)).add("L_s", num(p.L_s))
                .add("body", std::to_string(static_cast<int>(p.body)))
                .add("shockley", sub_fields(p.shockley, K::ShockleyDiode))
                .add("lauritzen", sub_fields(p.lauritzen, K::LauritzenDiode))
                .add("T_j", num(p.T_j)).add("rds_tc1", num(p.rds_tc1))
                .add("rds_tc2", num(p.rds_tc2)).add("vt_tc", num(p.vt_tc));
            o += "  - " + mm.str() + "\n";
        }
    }
    if (const auto cs = c.controls(); !cs.empty()) {
        o += "controls:\n";
        for (const auto& ct : cs) {
            where = "a control";
            if (ct.handle) {
                throw std::invalid_argument(std::format(
                    "to_yaml: control \"{}\" ({}) carries a Python callable, "
                    "which YAML cannot hold; build the circuit without it and "
                    "attach it to the run", ct.name, ct.kind));
            }
            o += "  - " + Map{}.add("name", quote(ct.name)).add("kind", quote(ct.kind))
                .add("hash", std::format("\"{:016x}\"", ct.content_hash)).str() + "\n";
        }
    }
    return o;
}

// ---------------------------------------------------------------- reading

struct YamlIo {
    using K = pwl::DevicePool::StoredKind;

    [[noreturn]] static void fail(std::string_view who, std::string_view what) {
        throw std::invalid_argument(std::format("from_yaml: {}: {}", who, what));
    }

    static std::string num_text_(Real x) { return yaml_detail::num(x); }

    static Real real(const YAML::Node& n, std::string_view who, std::string_view key) {
        if (!n || !n.IsScalar()) fail(who, std::format("\"{}\" must be a number", key));
        const std::string& s = n.Scalar();
        Real x{};
        if (!numeric::parse_real(s.data(), s.data() + s.size(), x)) {
            fail(who, std::format("\"{}\" is not a number: \"{}\"", key, s));
        }
        if (!std::isfinite(x)) {   // to_yaml never writes a non-finite value
            fail(who, std::format("\"{}\" must be a finite number, got \"{}\"", key, s));
        }
        return x;
    }
    /// An integer in [lo, hi] (a float cast to int is UB out of range).
    static int integer(const YAML::Node& n, std::string_view who, std::string_view key,
                       int lo, int hi) {
        const std::string s = text(n, who, key);
        int x = 0;
        const auto r = std::from_chars(s.data(), s.data() + s.size(), x);
        if (r.ec != std::errc{} || r.ptr != s.data() + s.size() || x < lo || x > hi) {
            fail(who, std::format("\"{}\" must be an integer in [{}, {}], got \"{}\"",
                                  key, lo, hi, s));
        }
        return x;
    }
    static std::string text(const YAML::Node& n, std::string_view who, std::string_view key) {
        if (!n || !n.IsScalar()) fail(who, std::format("\"{}\" must be a string", key));
        return n.Scalar();
    }
    static std::vector<Real> reals(const YAML::Node& n, std::string_view who,
                                   std::string_view key) {
        if (!n || !n.IsSequence()) fail(who, std::format("\"{}\" must be a list", key));
        std::vector<Real> v;
        for (const auto& x : n) v.push_back(real(x, who, key));
        return v;
    }
    /// Refuse a key outside `allowed` (and the kind's field names).
    static void keys(const YAML::Node& n, std::string_view who,
                     std::initializer_list<std::string_view> allowed,
                     std::span<const FieldSpec> row = {}) {
        if (!n.IsMap()) fail(who, "expected a mapping");
        for (const auto& kv : n) {
            const auto k = kv.first.Scalar();
            const bool ok = std::ranges::find(allowed, std::string_view{k}) != allowed.end() ||
                std::ranges::any_of(row, [&](const FieldSpec& f) {
                    return !f.derived && f.name == k;
                });
            if (!ok) fail(who, std::format("unknown key \"{}\"", k));
        }
    }
    static void set_fields(ParamsVariant& v, const YAML::Node& n, std::string_view who,
                           std::span<const FieldSpec> row) {
        for (const auto& f : row) {
            if (!f.derived && n[std::string{f.name}]) {
                f.set(v, real(n[std::string{f.name}], who, f.name));
            }
        }
    }
    static models::NonlinearCapacitor::Params charge(const YAML::Node& n,
                                                     std::string_view who) {
        ParamsVariant v = yaml_detail::make_params(static_cast<std::size_t>(K::NonlinearCapacitor));
        set_fields(v, n, who, fields_of(K::NonlinearCapacitor));
        auto p = std::get<models::NonlinearCapacitor::Params>(v);
        if (const auto l = n["law"]) {
            const auto s = text(l, who, "law");
            const auto it = std::ranges::find(yaml_detail::kLaws, std::string_view{s});
            if (it == yaml_detail::kLaws.end()) fail(who, std::format("unknown law \"{}\"", s));
            p.law = static_cast<models::ChargeLawKind>(it - yaml_detail::kLaws.begin());
        }
        if (const auto t = n["table"]) {
            keys(t, who, {"v", "q", "c"});
            p.table = std::make_shared<const models::ChargeTable>(
                reals(t["v"], who, "table.v"), reals(t["q"], who, "table.q"),
                reals(t["c"], who, "table.c"), std::string{who});
        }
        return p;
    }

    struct Pending { YAML::Node n; K kind; std::string name; std::vector<Index> branches; };

    /// Pass 1 of one device entry: its branches (names, pins, roles).
    template <class Pin>
    static Pending begin_entry_(builder::CircuitBuilder& b, const YAML::Node& d, Pin& pin) {
            if (!d.IsMap()) fail("a device", "expected a mapping");
            const auto ks = text(d["kind"], "a device", "kind");
            const auto it = std::ranges::find(kKindYaml, std::string_view{ks});
            if (it == kKindYaml.end()) fail(ks, "unknown kind");
            const auto kind = static_cast<K>(it - kKindYaml.begin());
            Pending p{d, kind, {}, {}};
            if (kind == K::PmsmMna) {
                const auto ph = d["phases"];
                p.name = ph && ph.IsSequence() && ph.size() == 3
                    ? text(ph[0]["name"], "pmsm_mna", "name") : "pmsm_mna";
                if (!ph || !ph.IsSequence() || ph.size() != 3) fail(p.name, "needs three phases");
                for (const auto& x : ph) {
                    if (!x.IsMap()) fail(p.name, "each phase is {name, pins}");
                    const auto nm = text(x["name"], p.name, "name");
                    const auto pins = x["pins"];
                    if (!pins || !pins.IsSequence() || pins.size() != 2) fail(nm, "needs two pins");
                    p.branches.push_back(b.add_branch_(nm, pin(pins[0], nm), pin(pins[1], nm),
                                                       topology::BranchKind::Nonlinear));
                }
            } else {
                p.name = text(d["name"], ks, "name");
                const auto pins = d["pins"];
                const auto want = yaml_detail::pin_count(kind);
                if (!pins || !pins.IsSequence() || pins.size() != want) {
                    fail(p.name, std::format("a {} needs {} pins", ks, want));
                }
                auto role = yaml_detail::default_role(kind);
                if (const auto r = d["role"]) {
                    const auto rs = text(r, p.name, "role");
                    const auto ri = std::ranges::find(yaml_detail::kRoles, std::string_view{rs});
                    if (ri == yaml_detail::kRoles.end()) fail(p.name, std::format("unknown role \"{}\"", rs));
                    role = static_cast<topology::BranchKind>(ri - yaml_detail::kRoles.begin());
                }
                p.branches.push_back(b.add_branch_(p.name, pin(pins[0], p.name),
                                                   pin(pins[1], p.name), role));
            }
            return p;
    }

    static void check_bounds_(builder::CircuitBuilder& b, const Pending& pd) {
        const Index bi = pd.branches.front();
        const bool form = b.pool_.is_branch_form(bi);
        for (const auto& f : fields_of(pd.kind)) {
            if (f.derived || !pd.n[std::string{f.name}]) continue;
            if (form && (f.name == "g_on" || f.name == "g_off")) continue;
            const Real x = f.get(b.pool_.params_of(bi));
            if (!within(f.bound, x)) {
                fail(pd.name, std::format("{} must be {} (got {})", f.name,
                                          bound_text(f.bound), num_text_(x)));
            }
        }
        if (form) {
            Real g_off = 0;
            std::visit([&](const auto& prm) {
                if constexpr (requires { prm.g_off; }) g_off = prm.g_off;
            }, b.pool_.params_of(bi));
            try {
                builder::CircuitBuilder::require_branch_values_(
                    "from_yaml", pd.name, b.pool_.branch_R_on(bi), g_off);
            } catch (const std::invalid_argument& e) {
                fail(pd.name, e.what());
            }
        }
    }

    /// One entry of this dialect (`{kind, name, pins, <fields>}`) read into
    /// `b` with its pins created by name: the authoring loader's reader
    /// for a registry kind it has no authoring type for, so every kind
    /// the emitter writes is a kind `yaml::load` reads (one table,
    /// `kKindYaml` / `kKindFields`, checked at compile time).
    static void read_entry(builder::CircuitBuilder& b, const YAML::Node& d) {
        auto pin = [&](const YAML::Node& n, std::string_view who) {
            return b.resolve_node_(text(n, who, "pins"));
        };
        const auto p = begin_entry_(b, d, pin);
        add_device_(b, p.n, p.kind, p.name, p.branches, pin);
        check_bounds_(b, p);
    }

    static Circuit read(std::string_view text_in, bool bounds = true) {
        const YAML::Node root = [&] {
            try {
                return YAML::Load(std::string{text_in});
            } catch (const YAML::Exception& e) {
                fail("document", e.what());
            }
        }();
        if (!root.IsMap()) fail("document", "expected a mapping");
        keys(root, "document", {"pulsim_circuit", "nodes", "devices", "couplings",
                                "branch_form", "scopes", "parameters", "bindings",
                                "initial_conditions", "aliases", "power_mosfets",
                                "controls"});
        if (text(root["pulsim_circuit"], "document", "pulsim_circuit") != "1") {
            fail("document", "pulsim_circuit must be 1");
        }
        for (const auto* k : {"nodes", "devices", "couplings", "branch_form", "scopes", "parameters",
                              "bindings", "initial_conditions", "aliases",
                              "power_mosfets", "controls"}) {
            if (root[k] && !root[k].IsSequence()) fail(k, "must be a list");
        }
        builder::CircuitBuilder b;
        if (const auto ns = root["nodes"]) {
            for (const auto& n : ns) {
                const bool tagged = n.IsMap();
                if (tagged) keys(n, "nodes", {"name", "domain"});
                const auto name = text(tagged ? n["name"] : n, "nodes", "nodes");
                const Index created = b.resolve_node_(name);   // sequenced (MSVC)
                if (created != b.graph().num_nodes() - 1) {
                    fail("nodes", std::format("\"{}\" is listed twice or names ground", name));
                }
                if (tagged) {
                    const auto dn = text(n["domain"], name, "domain");
                    const auto d = domain_from_name(dn);
                    if (!d) fail(name, std::format("unknown domain \"{}\"", dn));
                    try {
                        (void)b.node(name, *d);
                    } catch (const std::invalid_argument& e) {
                        fail(name, e.what());
                    }
                }
            }
        }
        const auto n_nodes = b.graph().num_nodes();
        auto pin = [&](const YAML::Node& n, std::string_view who) {
            const Index i = b.resolve_node_(text(n, who, "pins"));
            if (b.graph().num_nodes() != n_nodes) {
                fail(who, std::format("pin \"{}\" is not in nodes", n.Scalar()));
            }
            return i;
        };

        // Pass 1: every branch, so a sensed branch named later resolves.
        std::vector<Pending> pend;
        for (const auto& d : root["devices"]) pend.push_back(begin_entry_(b, d, pin));
        // Pass 2: the pool, in branch order (its counters follow it).
        for (const auto& p : pend) add_device_(b, p.n, p.kind, p.name, p.branches, pin);

        for (const auto& cp : root["couplings"]) {
            keys(cp, "a coupling", {"primary", "secondary", "L_p", "L_s", "k"});
            const auto pr = text(cp["primary"], "a coupling", "primary");
            b.pool_.add_transformer_coupling(
                b.branch_id_of(pr), b.branch_id_of(text(cp["secondary"], pr, "secondary")),
                models::TwoWindingTransformer::Params{
                    real(cp["L_p"], pr, "L_p"), real(cp["L_s"], pr, "L_s"),
                    real(cp["k"], pr, "k")});
        }
        for (const auto& f : root["branch_form"]) {
            keys(f, "branch_form", {"device", "R_on"});
            const auto nm = text(f["device"], "branch_form", "device");
            const auto bi = b.branch_id_of(nm);
            const auto fk = b.pool_.kind_of(bi);
            if (fk != K::Switch && fk != K::Diode) fail(nm, "branch form is for a switch or a diode");
            const Real R_on = real(f["R_on"], nm, "R_on");
            b.pool_.set_branch_form(bi, R_on);
            for (const auto& fs : fields_of(fk)) {   // g_on is not written: 1/R_on, as add_* sets it
                if (fs.name == "g_on") fs.set(b.pool_.params_of_mut(bi), builder::CircuitBuilder::branch_g_on_(R_on));
            }
        }
        // The authoring bounds (`ir::kKindFields`), as with_value applies
        // them, on every value the document writes (an omitted value is
        // the model's default). A branch-form switch or diode has its own
        // rule (R_on, g_off >= 0; g_on = 1/R_on). Not every add_* checks
        // them (add_resistor takes R < 0), so a pickle, whose text this
        // library wrote, reads with bounds = false.
        if (bounds) for (const auto& pd : pend) check_bounds_(b, pd);
        for (const auto& x : root["scopes"]) {
            keys(x, "a scope", {"path", "kind"});
            const auto path = text(x["path"], "a scope", "path");
            const auto kind = x["kind"] ? text(x["kind"], path, "kind") : std::string{};
            const auto slash = path.rfind('/');
            if (b.scope_index_.contains(path)) fail(path, "scope listed twice");
            if (slash != std::string::npos && !b.scope_index_.contains(path.substr(0, slash))) {
                fail(path, "a scope must follow its parent");
            }
            (void)b.open_scope(path, kind);   // creates exactly this scope
            b.close_scope();
        }
        for (const auto& p : root["parameters"]) {
            keys(p, "a parameter", {"name", "scope", "value", "expr", "unit", "lo", "hi"});
            auto nm = text(p["name"], "a parameter", "name");
            std::optional<Real> lo, hi;
            if (p["lo"]) lo = real(p["lo"], nm, "lo");
            if (p["hi"]) hi = real(p["hi"], nm, "hi");
            const auto unit = p["unit"] ? text(p["unit"], nm, "unit") : std::string{};
            if (p["scope"]) {   // declared inside it: name = scope path + "/" + local
                const auto sp = text(p["scope"], nm, "scope");
                const auto it = b.scope_index_.find(sp);
                if (it == b.scope_index_.end()) fail(nm, std::format("unknown scope \"{}\"", sp));
                if (!nm.starts_with(sp + "/")) fail(nm, std::format("not a name in scope \"{}\"", sp));
                nm = nm.substr(sp.size() + 1);
                b.cur_scope_ = it->second;
            }
            if (p["expr"]) b.param_expr(nm, text(p["expr"], nm, "expr"), unit, lo, hi);
            else b.param(nm, real(p["value"], nm, "value"), unit, lo, hi);
            b.cur_scope_ = 0;
        }
        for (const auto& bd : root["bindings"]) {
            keys(bd, "a binding", {"param", "field"});
            b.bind_param(text(bd["field"], "a binding", "field"),
                         text(bd["param"], "a binding", "param"));
        }
        for (const auto& ic : root["initial_conditions"]) {
            keys(ic, "an initial condition", {"device", "value"});
            const auto nm = text(ic["device"], "an initial condition", "device");
            b.set_initial(nm, real(ic["value"], nm, "value"));
        }
        for (const auto& a : root["aliases"]) {
            keys(a, "an alias", {"name", "node", "branch"});
            const auto nm = text(a["name"], "an alias", "name");
            std::optional<std::string> nd, br;
            if (a["node"]) nd = text(a["node"], nm, "node");
            if (a["branch"]) br = text(a["branch"], nm, "branch");
            b.set_alias(nm, nd ? std::optional<std::string_view>{*nd} : std::nullopt,
                        br ? std::optional<std::string_view>{*br} : std::nullopt);
        }
        for (const auto& pm : root["power_mosfets"]) read_power_mosfet_(b, pm);
        for (const auto& ct : root["controls"]) {
            keys(ct, "a control", {"name", "kind", "hash"});
            const auto nm = text(ct["name"], "a control", "name");
            const auto hs = text(ct["hash"], nm, "hash");
            std::uint64_t h = 0;
            const auto r = std::from_chars(hs.data(), hs.data() + hs.size(), h, 16);
            if (r.ec != std::errc{} || r.ptr != hs.data() + hs.size()) {
                fail(nm, std::format("hash \"{}\" is not hexadecimal", hs));
            }
            b.add_control(ControlDecl{nm, text(ct["kind"], nm, "kind"), h, nullptr});
        }
        return Circuit::from_builder(b);
    }

private:
    template <class Pin>
    static void add_device_(builder::CircuitBuilder& b, const YAML::Node& n, K kind,
                            const std::string& who, const std::vector<Index>& br,
                            Pin& pin) {
        namespace m = ::pulsim::models;
        const auto row = fields_of(kind);
        keys(n, who, {"kind", "name", "pins", "role", "control", "phases", "omega", "theta",
                      "table", "ja", "law", "t", "i", "periodic"}, row);
        ParamsVariant v = yaml_detail::make_params(static_cast<std::size_t>(kind));
        set_fields(v, n, who, row);
        auto& pool = b.pool_;
        const Index bi = br.front();
        auto xpin = [&](std::size_t k) { return pin(n["pins"][k], who); };
        std::visit([&](auto& p) {
            using P = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<P, m::Resistor::Params>) pool.add_resistor(bi, p);
            else if constexpr (std::is_same_v<P, m::VoltageSource::Params>) pool.add_voltage_source(bi, p);
            else if constexpr (std::is_same_v<P, pwl::DevicePool::SwitchParams>) pool.add_switch(bi, p.g_on, p.g_off);
            else if constexpr (std::is_same_v<P, m::Capacitor::Params>) pool.add_capacitor(bi, p);
            else if constexpr (std::is_same_v<P, m::Inductor::Params>) pool.add_inductor(bi, p);
            else if constexpr (std::is_same_v<P, m::SwitchedDiode::Params>) pool.add_diode(bi, p.g_on, p.g_off, p.V_th, p.V_f);
            else if constexpr (std::is_same_v<P, m::IdealDiode::Params>) pool.add_nonlinear_diode(bi, p);
            else if constexpr (std::is_same_v<P, m::CurrentSource::Params>) pool.add_current_source(bi, p);
            else if constexpr (std::is_same_v<P, m::PWMVoltageSource::Params>) pool.add_pwm_voltage_source(bi, p);
            else if constexpr (std::is_same_v<P, m::SineVoltageSource::Params>) pool.add_sine_voltage_source(bi, p);
            else if constexpr (std::is_same_v<P, m::PulseVoltageSource::Params>) pool.add_pulse_voltage_source(bi, p);
            else if constexpr (std::is_same_v<P, m::MosfetLevel1::Params>) pool.add_mosfet_level1(bi, xpin(2), p);
            else if constexpr (std::is_same_v<P, m::IgbtLevel1::Params>) pool.add_igbt_level1(bi, xpin(2), p);
            else if constexpr (std::is_same_v<P, m::VCVS::Params>) pool.add_vcvs(bi, xpin(2), xpin(3), p);
            else if constexpr (std::is_same_v<P, m::SaturableInductor::Params>) {
                if (const auto t = n["table"]) {
                    keys(t, who, {"i", "lambda", "slope", "L_tail"});
                    p.table = std::make_shared<const m::FluxTable>(
                        reals(t["i"], who, "table.i"), reals(t["lambda"], who, "table.lambda"),
                        reals(t["slope"], who, "table.slope"), real(t["L_tail"], who, "table.L_tail"),
                        who);
                }
                if (const auto j = n["ja"]) {
                    keys(j, who, {"N", "Ae", "le", "lg", "Ms", "a", "alpha", "c", "k",
                                  "substeps_min", "substep_frac", "M0"});
                    m::JilesAthertonCore::Params q;
                    q.N = real(j["N"], who, "ja.N"); q.Ae = real(j["Ae"], who, "ja.Ae");
                    q.le = real(j["le"], who, "ja.le"); q.lg = real(j["lg"], who, "ja.lg");
                    q.ja.Ms = real(j["Ms"], who, "ja.Ms"); q.ja.a = real(j["a"], who, "ja.a");
                    q.ja.alpha = real(j["alpha"], who, "ja.alpha");
                    q.ja.c = real(j["c"], who, "ja.c"); q.ja.k = real(j["k"], who, "ja.k");
                    q.substeps_min = integer(j["substeps_min"], who, "ja.substeps_min", 1, 1 << 20);
                    q.substep_frac = real(j["substep_frac"], who, "ja.substep_frac");
                    q.M0 = real(j["M0"], who, "ja.M0");
                    p.ja = std::make_shared<const m::JilesAthertonCore::Params>(q);
                }
                pool.add_saturable_inductor(bi, p);
            } else if constexpr (std::is_same_v<P, m::NonlinearCapacitor::Params>) {
                pool.add_nonlinear_capacitor(bi, charge(n, who));
            } else if constexpr (std::is_same_v<P, m::ShockleyDiode::Params>) pool.add_shockley_diode(bi, p);
            else if constexpr (std::is_same_v<P, m::LauritzenDiode::Params>) pool.add_lauritzen_diode(bi, p);
            else if constexpr (std::is_same_v<P, m::PmsmMna::Params>) {
                pool.add_pmsm_mna({br[0], br[1], br[2]}, pin(n["omega"], who),
                                  pin(n["theta"], who), p);
            } else if constexpr (std::is_same_v<P, m::IdealTransformer::Params>) pool.add_ideal_transformer(bi, xpin(2), xpin(3), p);
            else if constexpr (std::is_same_v<P, m::VCCS::Params>) pool.add_vccs(bi, xpin(2), xpin(3), p);
            else if constexpr (std::is_same_v<P, m::CCCS::Params>) pool.add_cccs(bi, b.branch_id_of(text(n["control"], who, "control")), p);
            else if constexpr (std::is_same_v<P, m::CCVS::Params>) pool.add_ccvs(bi, b.branch_id_of(text(n["control"], who, "control")), p);
            else if constexpr (std::is_same_v<P, m::Gyrator::Params>) pool.add_gyrator(bi, xpin(2), xpin(3), p);
            else if constexpr (std::is_same_v<P, m::SineCurrentSource::Params>) pool.add_sine_current_source(bi, p);
            else if constexpr (std::is_same_v<P, m::PulseCurrentSource::Params>) pool.add_pulse_current_source(bi, p);
            else if constexpr (std::is_same_v<P, m::PwlCurrentSource::Params>) {
                p.w.t = reals(n["t"], who, "t");
                p.w.v = reals(n["i"], who, "i");
                p.w.periodic = n["periodic"] && text(n["periodic"], who, "periodic") == "true";
                m::PwlWaveform::validate(p.w, who);
                pool.add_pwl_current_source(bi, p);
            } else {
                static_assert(sizeof(P) == 0, "from_yaml: a registry kind without a reader");
            }
        }, v);
    }

    static void read_power_mosfet_(builder::CircuitBuilder& b, const YAML::Node& n) {
        namespace m = ::pulsim::models;
        keys(n, "a power MOSFET", {"name", "channel", "drain", "source", "gate", "members",
                                   "K_eff", "V_T_eff", "R_d_eff", "K", "V_T", "lambda",
                                   "kappa", "R_d", "C_gs", "C_gd", "C_ds", "R_g", "L_s",
                                   "body", "shockley", "lauritzen", "T_j", "rds_tc1",
                                   "rds_tc2", "vt_tc"});
        m::PowerMosfet::Record r;
        r.name = text(n["name"], "a power MOSFET", "name");
        const auto& w = r.name;
        r.drain = text(n["drain"], w, "drain");
        r.source = text(n["source"], w, "source");
        r.gate = text(n["gate"], w, "gate");
        for (const auto& x : n["members"]) r.sub_elements.push_back(text(x, w, "members"));
        r.K_eff = real(n["K_eff"], w, "K_eff");
        r.V_T_eff = real(n["V_T_eff"], w, "V_T_eff");
        r.R_d_eff = real(n["R_d_eff"], w, "R_d_eff");
        auto& p = r.params;
        p.K = real(n["K"], w, "K"); p.V_T = real(n["V_T"], w, "V_T");
        p.lambda = real(n["lambda"], w, "lambda"); p.kappa = real(n["kappa"], w, "kappa");
        p.R_d = real(n["R_d"], w, "R_d"); p.C_gs = real(n["C_gs"], w, "C_gs");
        auto cap = [&](const YAML::Node& c, std::string_view key) -> m::PowerMosfet::Capacitance {
            if (c && c.IsMap()) return charge(c, std::format("{}.{}", w, key));
            return real(c, w, key);
        };
        p.C_gd = cap(n["C_gd"], "C_gd");
        p.C_ds = cap(n["C_ds"], "C_ds");
        p.R_g = real(n["R_g"], w, "R_g"); p.L_s = real(n["L_s"], w, "L_s");
        p.body = static_cast<m::PowerMosfet::BodyDiode>(integer(n["body"], w, "body", 0, 2));
        if (!n["shockley"] || !n["lauritzen"]) fail(w, "needs \"shockley\" and \"lauritzen\"");
        ParamsVariant sh{p.shockley};
        keys(n["shockley"], w, {}, fields_of(K::ShockleyDiode));
        set_fields(sh, n["shockley"], w, fields_of(K::ShockleyDiode));
        p.shockley = std::get<m::ShockleyDiode::Params>(sh);
        ParamsVariant la{p.lauritzen};
        keys(n["lauritzen"], w, {}, fields_of(K::LauritzenDiode));
        set_fields(la, n["lauritzen"], w, fields_of(K::LauritzenDiode));
        p.lauritzen = std::get<m::LauritzenDiode::Params>(la);
        p.T_j = real(n["T_j"], w, "T_j"); p.rds_tc1 = real(n["rds_tc1"], w, "rds_tc1");
        p.rds_tc2 = real(n["rds_tc2"], w, "rds_tc2"); p.vt_tc = real(n["vt_tc"], w, "vt_tc");
        const Index ch = b.branch_id_of(text(n["channel"], w, "channel"));
        b.power_mosfets_.emplace(ch, std::move(r));
    }
};

/// Inverse of `to_yaml`. Throws `std::invalid_argument` naming the
/// device and the key on a malformed document, or on a value outside
/// its bound unless `bounds` is false (pickle: text this library wrote).
[[nodiscard]] inline Circuit from_yaml(std::string_view text, bool bounds = true) {
    try {
        return YamlIo::read(text, bounds);
    } catch (const YAML::Exception& e) {  // a shape the checks above missed
        throw std::invalid_argument(std::format("from_yaml: malformed document: {}", e.what()));
    } catch (const std::out_of_range& e) {  // branch_id_of: an unknown device name
        throw std::invalid_argument(std::format("from_yaml: {}", e.what()));
    }
}

}  // namespace pulsim::ir
