#pragma once

// add-probes-and-recorders: a probe is one scalar expression over READINGS
// ("atoms"): `v(node)`, `v(a,b)`, `i(device)`, `p(device)`, written with
// the operators and functions of the circuit IR's parameter expressions
// (ir/expr.hpp), e.g. `v(dp) - v(dn)`, `v(out) * i(L1)`, `abs(i(D1))`.
//
// The atoms are cut out lexically (`v(` / `i(` / `p(` at an identifier
// boundary up to the matching `)`) and replaced by placeholders, and the
// rest is compiled by `ir::Expr` itself: there is one expression parser.
// Atom text is normalised (blanks removed), so `v( a , b )` and `v(a,b)`
// are the same reading. The caller resolves each atom to a value per
// sample (a state column, a node-voltage difference, a device law under
// the recorded mode) and passes the values in `atoms()` order.

#include "pulsim/ir/expr.hpp"
#include "pulsim/numeric/types.hpp"

#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pulsim::record {

class Probe {
public:
    /// Compile `text`; throws std::invalid_argument naming the probe on a
    /// syntax error, an unknown function, a bare name that is not a
    /// reading, or an unbalanced reading.
    [[nodiscard]] static Probe compile(std::string_view text) {
        Probe p;
        p.text_ = std::string{text};
        std::string rewritten;
        rewritten.reserve(text.size());
        const auto who = std::format("probe \"{}\"", text);
        if (text.find("__a") != std::string_view::npos) {
            throw std::invalid_argument(std::format("{}: \"__a\" is reserved", who));
        }
        for (std::size_t i = 0; i < text.size();) {
            const char c = text[i];
            const bool boundary = i == 0 || !ident(text[i - 1]);
            if (boundary && (c == 'v' || c == 'i' || c == 'p')) {
                std::size_t j = i + 1;
                while (j < text.size() && (text[j] == ' ' || text[j] == '\t')) ++j;
                if (j < text.size() && text[j] == '(') {
                    const auto close = text.find(')', j);
                    if (close == std::string_view::npos) {
                        throw std::invalid_argument(std::format(
                            "{}: reading \"{}(\" has no ')'", who, c));
                    }
                    std::string atom{c};
                    atom += '(';
                    bool gap = false;  // a blank since the last kept character
                    for (std::size_t k = j + 1; k < close; ++k) {
                        const char ch = text[k];
                        if (ch == '(') {
                            throw std::invalid_argument(std::format(
                                "{}: a reading takes names, not expressions: \"{}\"",
                                who, text.substr(i, close + 1 - i)));
                        }
                        if (ch == ' ' || ch == '\t') { gap = true; continue; }
                        if (gap && ch != ',' && atom.back() != ',' && atom.back() != '(') {
                            throw std::invalid_argument(std::format(
                                "{}: a blank inside a name in \"{}\"", who,
                                text.substr(i, close + 1 - i)));
                        }
                        gap = false;
                        atom += ch;
                    }
                    atom += ')';
                    if (atom.size() == 3) {
                        throw std::invalid_argument(std::format(
                            "{}: empty reading \"{}()\"", who, c));
                    }
                    std::uint32_t k = 0;
                    while (k < p.atoms_.size() && p.atoms_[k] != atom) ++k;
                    if (k == p.atoms_.size()) p.atoms_.push_back(atom);
                    rewritten += std::format(" __a{} ", k);
                    i = close + 1;
                    continue;
                }
            }
            rewritten += c;
            ++i;
        }
        // a probe that IS one reading (`v(n)`) is read directly, not
        // through the expression (item 3: the per-step cost)
        p.plain_ = p.atoms_.size() == 1 &&
                   rewritten.find_first_not_of(' ') == rewritten.find("__a0") &&
                   rewritten.find_last_not_of(' ') == rewritten.find("__a0") + 3;
        p.expr_ = ir::Expr::compile(
            rewritten,
            [&](std::string_view n) -> std::optional<std::uint32_t> {
                if (n.size() < 4 || n.substr(0, 3) != "__a") return std::nullopt;
                std::uint32_t k = 0;
                for (const char d : n.substr(3)) {
                    if (d < '0' || d > '9') return std::nullopt;
                    k = k * 10 + static_cast<std::uint32_t>(d - '0');
                }
                if (k >= p.atoms_.size()) return std::nullopt;
                return k;
            },
            who);
        return p;
    }

    [[nodiscard]] const std::string& text() const noexcept { return text_; }
    /// The readings, normalised, in first-use order.
    [[nodiscard]] std::span<const std::string> atoms() const noexcept { return atoms_; }
    /// The probe's value from its readings (in `atoms()` order).
    [[nodiscard]] Real eval(std::span<const Real> atom_values) const {
        return plain_ ? atom_values[0] : expr_.eval(atom_values);
    }
    /// The probe is one reading, unchanged.
    [[nodiscard]] bool plain() const noexcept { return plain_; }
    /// The compiled expression (Param k = atom k; also for a plain probe).
    [[nodiscard]] std::span<const ir::Expr::Instr> program() const noexcept {
        return expr_.program();
    }

private:
    static bool ident(char c) noexcept {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' ||
               (c >= '0' && c <= '9') || c == '.';
    }
    std::string text_;
    std::vector<std::string> atoms_;
    ir::Expr expr_;
    bool plain_ = false;
};

}  // namespace pulsim::record
