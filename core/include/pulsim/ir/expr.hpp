#pragma once

// add-immutable-circuit-ir group 7: parameter expressions.
//
// Grammar (recursive descent, compiled to a postfix program):
//   expr    := term (('+' | '-') term)*
//   term    := unary (('*' | '/') unary)*
//   unary   := ('+' | '-') unary | power
//   power   := primary ('^' unary)?          right-associative; -2^2 = -4
//   primary := number | name | name '(' expr (',' expr)* ')' | '(' expr ')'
// Numbers are C-locale decimals (`1e-3`, `.5`); names are identifiers
// resolved to parameter indices when compiled (`pi` is a constant).
// Functions: abs sqrt exp log log10 sin cos tan atan sinh cosh tanh
// (one argument), min max pow atan2 (two). The evaluator runs the
// program on doubles in the order written, so a result is bit-identical
// to the same arithmetic written out by hand.

#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pulsim::ir {

class Expr {
public:
    enum class Op : std::uint8_t {
        Const, Param, Neg, Add, Sub, Mul, Div, Pow,
        Abs, Sqrt, Exp, Log, Log10, Sin, Cos, Tan, Atan, Sinh, Cosh, Tanh,
        Min, Max, Atan2,
    };
    struct Instr {
        Op op;
        Real value = 0;          // Const
        std::uint32_t index = 0; // Param
    };

    /// Name -> parameter index, or nullopt for an unknown name.
    using Resolve = std::function<std::optional<std::uint32_t>(std::string_view)>;

    /// Compile `text`; `who` names the owner in errors ("parameter
    /// \"L2\"", "device \"L1\" field \"L\"").
    [[nodiscard]] static Expr compile(std::string_view text, const Resolve& resolve,
                                      std::string_view who) {
        Parser p{text, resolve, who, {}, 0};
        p.skip();
        if (p.at_end()) p.fail("is empty");
        p.expr();
        p.skip();
        if (!p.at_end()) p.fail(std::format("unexpected '{}'", text[p.pos]));
        Expr e;
        e.text_ = std::string{text};
        e.prog_ = std::move(p.out);
        return e;
    }

    [[nodiscard]] Real eval(std::span<const Real> params) const {
        Real st[64];
        std::size_t sp = 0;
        std::vector<Real> big;  // a program deeper than 64 (generated text)
        Real* s = st;
        if (depth_() > 64) { big.resize(depth_()); s = big.data(); }
        for (const auto& in : prog_) {
            switch (in.op) {
                case Op::Const: s[sp++] = in.value; break;
                case Op::Param: s[sp++] = params[in.index]; break;
                case Op::Neg: s[sp - 1] = -s[sp - 1]; break;
                case Op::Add: --sp; s[sp - 1] = s[sp - 1] + s[sp]; break;
                case Op::Sub: --sp; s[sp - 1] = s[sp - 1] - s[sp]; break;
                case Op::Mul: --sp; s[sp - 1] = s[sp - 1] * s[sp]; break;
                case Op::Div: --sp; s[sp - 1] = s[sp - 1] / s[sp]; break;
                case Op::Pow: --sp; s[sp - 1] = std::pow(s[sp - 1], s[sp]); break;
                case Op::Min: --sp; s[sp - 1] = std::fmin(s[sp - 1], s[sp]); break;
                case Op::Max: --sp; s[sp - 1] = std::fmax(s[sp - 1], s[sp]); break;
                case Op::Atan2: --sp; s[sp - 1] = std::atan2(s[sp - 1], s[sp]); break;
                case Op::Abs: s[sp - 1] = std::fabs(s[sp - 1]); break;
                case Op::Sqrt: s[sp - 1] = std::sqrt(s[sp - 1]); break;
                case Op::Exp: s[sp - 1] = std::exp(s[sp - 1]); break;
                case Op::Log: s[sp - 1] = std::log(s[sp - 1]); break;
                case Op::Log10: s[sp - 1] = std::log10(s[sp - 1]); break;
                case Op::Sin: s[sp - 1] = std::sin(s[sp - 1]); break;
                case Op::Cos: s[sp - 1] = std::cos(s[sp - 1]); break;
                case Op::Tan: s[sp - 1] = std::tan(s[sp - 1]); break;
                case Op::Atan: s[sp - 1] = std::atan(s[sp - 1]); break;
                case Op::Sinh: s[sp - 1] = std::sinh(s[sp - 1]); break;
                case Op::Cosh: s[sp - 1] = std::cosh(s[sp - 1]); break;
                case Op::Tanh: s[sp - 1] = std::tanh(s[sp - 1]); break;
            }
        }
        return s[0];
    }

    [[nodiscard]] const std::string& text() const noexcept { return text_; }
    [[nodiscard]] std::span<const Instr> program() const noexcept { return prog_; }
    /// Parameter indices the expression reads.
    [[nodiscard]] std::vector<std::uint32_t> inputs() const {
        std::vector<std::uint32_t> v;
        for (const auto& in : prog_) if (in.op == Op::Param) v.push_back(in.index);
        return v;
    }

    /// Identifiers in `text` other than functions and `pi` (for ordering
    /// definitions before they are compiled). Throws on a syntax error.
    [[nodiscard]] static std::vector<std::string> names_in(std::string_view text,
                                                           std::string_view who) {
        std::vector<std::string> names;
        (void)compile(text, [&](std::string_view n) -> std::optional<std::uint32_t> {
            names.emplace_back(n);
            return 0;
        }, who);
        return names;
    }

private:
    std::string text_;
    std::vector<Instr> prog_;

    [[nodiscard]] std::size_t depth_() const noexcept {
        std::size_t d = 0, mx = 0;
        for (const auto& in : prog_) {
            switch (in.op) {
                case Op::Const: case Op::Param: mx = std::max(mx, ++d); break;
                case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Pow:
                case Op::Min: case Op::Max: case Op::Atan2: --d; break;
                default: break;
            }
        }
        return mx;
    }

    struct Fn { std::string_view name; Op op; int arity; };
    static constexpr Fn kFns[] = {
        {"abs", Op::Abs, 1}, {"sqrt", Op::Sqrt, 1}, {"exp", Op::Exp, 1},
        {"log", Op::Log, 1}, {"log10", Op::Log10, 1}, {"sin", Op::Sin, 1},
        {"cos", Op::Cos, 1}, {"tan", Op::Tan, 1}, {"atan", Op::Atan, 1},
        {"sinh", Op::Sinh, 1}, {"cosh", Op::Cosh, 1}, {"tanh", Op::Tanh, 1},
        {"min", Op::Min, 2}, {"max", Op::Max, 2}, {"pow", Op::Pow, 2},
        {"atan2", Op::Atan2, 2},
    };

    struct Parser {
        std::string_view t;
        const Resolve& resolve;
        std::string_view who;
        std::vector<Instr> out;
        std::size_t pos;
        int depth = 0;

        [[noreturn]] void fail(std::string_view what) const {
            throw std::invalid_argument(std::format(
                "{}: expression \"{}\" at column {}: {}", who, t, pos + 1, what));
        }
        [[nodiscard]] bool at_end() const noexcept { return pos >= t.size(); }
        void skip() noexcept {
            while (pos < t.size() && (t[pos] == ' ' || t[pos] == '\t')) ++pos;
        }
        bool eat(char c) {
            skip();
            if (pos < t.size() && t[pos] == c) { ++pos; return true; }
            return false;
        }
        void emit(Op op) { out.push_back(Instr{op}); }

        void expr() {
            if (++depth > 200) fail("nests deeper than 200");
            term();
            while (true) {
                if (eat('+')) { term(); emit(Op::Add); }
                else if (eat('-')) { term(); emit(Op::Sub); }
                else break;
            }
            --depth;
        }
        void term() {
            unary();
            while (true) {
                if (eat('*')) { unary(); emit(Op::Mul); }
                else if (eat('/')) { unary(); emit(Op::Div); }
                else break;
            }
        }
        void unary() {
            if (++depth > 200) fail("nests deeper than 200");
            if (eat('-')) { unary(); emit(Op::Neg); }
            else if (eat('+')) { unary(); }
            else power();
            --depth;
        }
        void power() {
            primary();
            if (eat('^')) { unary(); emit(Op::Pow); }
        }
        static bool ident_start(char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
        }
        static bool ident(char c) { return ident_start(c) || (c >= '0' && c <= '9'); }

        void primary() {
            skip();
            if (at_end()) fail("ends early");
            const char c = t[pos];
            if (c == '(') {
                ++pos;
                expr();
                if (!eat(')')) fail("expected ')'");
                return;
            }
            if ((c >= '0' && c <= '9') || c == '.') {
                std::size_t e = pos;
                while (e < t.size() && ((t[e] >= '0' && t[e] <= '9') || t[e] == '.')) ++e;
                if (e < t.size() && (t[e] == 'e' || t[e] == 'E')) {
                    std::size_t f = e + 1;
                    if (f < t.size() && (t[f] == '+' || t[f] == '-')) ++f;
                    if (f < t.size() && t[f] >= '0' && t[f] <= '9') {
                        e = f;
                        while (e < t.size() && t[e] >= '0' && t[e] <= '9') ++e;
                    }
                }
                Real v{};
                const auto r = std::from_chars(t.data() + pos, t.data() + e, v);
                if (r.ec != std::errc{} || r.ptr != t.data() + e) {
                    fail(std::format("bad number \"{}\"", t.substr(pos, e - pos)));
                }
                if (e < t.size() && ident(t[e])) {
                    fail(std::format("bad number \"{}\" (no unit suffixes)",
                                     t.substr(pos, e + 1 - pos)));
                }
                pos = e;
                out.push_back(Instr{Op::Const, v});
                return;
            }
            if (!ident_start(c)) fail(std::format("unexpected '{}'", c));
            std::size_t e = pos;
            while (e < t.size() && ident(t[e])) ++e;
            const auto name = t.substr(pos, e - pos);
            const auto at = pos;
            pos = e;
            if (eat('(')) {
                const Fn* fn = nullptr;
                for (const auto& f : kFns) if (f.name == name) fn = &f;
                if (fn == nullptr) { pos = at; fail(std::format("unknown function \"{}\"", name)); }
                expr();
                for (int k = 1; k < fn->arity; ++k) {
                    if (!eat(',')) fail(std::format("{}() takes {} arguments", name, fn->arity));
                    expr();
                }
                if (!eat(')')) fail(std::format("{}() takes {} argument{}", name,
                                                fn->arity, fn->arity == 1 ? "" : "s"));
                emit(fn->op);
                return;
            }
            const auto idx = resolve(name);  // a parameter named pi wins
            if (!idx && name == "pi") { out.push_back(Instr{Op::Const, std::numbers::pi_v<Real>}); return; }
            if (!idx) { pos = at; fail(std::format("unknown name \"{}\"", name)); }
            out.push_back(Instr{Op::Param, 0, *idx});
        }
    };
};

/// Order definitions so each comes after the names it reads; refuses a
/// cycle naming its members. `deps[i]` are the indices definition `i`
/// reads (indices into the same list). Returns the evaluation order.
[[nodiscard]] inline std::vector<std::size_t> dependency_order(
    std::span<const std::string> names, std::span<const std::vector<std::size_t>> deps) {
    const std::size_t n = names.size();
    std::vector<int> state(n, 0);  // 0 new, 1 on stack, 2 done
    std::vector<std::size_t> order;
    // Iterative depth-first search: a chain as long as the file cannot
    // overflow the call stack. Frame = (definition, next dependency).
    std::vector<std::pair<std::size_t, std::size_t>> stack;
    for (std::size_t root = 0; root < n; ++root) {
        if (state[root] != 0) continue;
        stack.emplace_back(root, 0);
        state[root] = 1;
        while (!stack.empty()) {
            auto& [i, k] = stack.back();
            if (k == deps[i].size()) {
                state[i] = 2;
                order.push_back(i);
                stack.pop_back();
                continue;
            }
            const std::size_t d = deps[i][k++];
            if (state[d] == 2) continue;
            if (state[d] == 1) {
                std::string cyc;
                auto it = std::find_if(stack.begin(), stack.end(),
                                       [&](const auto& f) { return f.first == d; });
                for (; it != stack.end(); ++it) cyc += names[it->first] + " -> ";
                throw std::invalid_argument(std::format(
                    "parameter \"{}\": its expression depends on itself ({}{})",
                    names[d], cyc, names[d]));
            }
            state[d] = 1;
            stack.emplace_back(d, 0);
        }
    }
    return order;
}

}  // namespace pulsim::ir
