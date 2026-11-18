"""add-immutable-circuit-ir group 7: parameter expressions.

The C++ evaluator (``ir/expr.hpp``) against a Python reference on 10^4
random expressions of exact operations (bit for bit) and 10^4 libm calls
(within LIBM_ULPS ulp); a circuit written with expressions against the same
circuit written with the evaluated literals (bit for bit); errors name
the parameter or device, the field and the text; ``to_yaml`` writes the
text, not the value.
"""
from __future__ import annotations

import math
import random
import struct
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim import _pulsim as k

NAMES = ("a", "b", "c", "Lbase", "x_1")
# IEEE-exact operations: the C++ evaluator and Python round each the same
# way on every platform, so these are compared bit for bit.
EXACT_UNARY = {"abs": abs, "sqrt": math.sqrt}
EXACT_BINARY = {"min": min, "max": max}
# libm functions: glibc, macOS and MSVC differ in the last ulp or two, so
# these are compared to LIBM_ULPS ulp of the reference, on arguments the
# exact generator makes (so the argument itself is bit-identical).
LIBM = {"exp": math.exp, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "atan": math.atan, "sinh": math.sinh, "cosh": math.cosh,
        "tanh": math.tanh}
LIBM_ULPS = 4


def _gen(rng, env, depth):
    """A random expression of exact operations as (text, value); every
    sub-value is finite."""
    if depth == 0 or rng.random() < 0.25:
        if rng.random() < 0.5:
            n = rng.choice(NAMES)
            return n, env[n]
        x = rng.choice((rng.uniform(-10, 10), rng.uniform(0, 1),
                        float(rng.randint(0, 9)), rng.uniform(1e-6, 1e-3)))
        return repr(x), x
    r = rng.random()
    if r < 0.6:
        op = rng.choice("+-*/")
        (ta, a), (tb, b) = _gen(rng, env, depth - 1), _gen(rng, env, depth - 1)
        if op == "/" and b == 0.0:
            return ta, a
        v = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else 0}[op]
        return f"({ta} {op} {tb})", v
    if r < 0.7:
        ta, a = _gen(rng, env, depth - 1)
        return f"-{ta}" if not ta.startswith("-") else f"-({ta})", -a
    if r < 0.85:
        f = rng.choice(list(EXACT_UNARY))
        ta, a = _gen(rng, env, depth - 1)
        if f == "sqrt":
            return f"sqrt(abs({ta}) + 1)", math.sqrt(abs(a) + 1)
        return f"{f}({ta})", EXACT_UNARY[f](a)
    f = rng.choice(list(EXACT_BINARY))
    (ta, a), (tb, b) = _gen(rng, env, depth - 1), _gen(rng, env, depth - 1)
    return f"{f}({ta}, {tb})", EXACT_BINARY[f](a, b)


def _env(rng):
    return {n: rng.uniform(-5, 5) for n in NAMES}


def test_fuzz_exact_operations_bit_for_bit():
    rng = random.Random(20261118)
    for _ in range(10_000):
        env = _env(rng)
        text, want = _gen(rng, env, rng.randint(1, 6))
        if not math.isfinite(want):
            continue
        got = k._eval_expr(text, list(env), list(env.values()))
        # the same bits, the sign of a zero included
        assert struct.pack("<d", got) == struct.pack("<d", want), (text, got, want)


def _libm_case(rng, env):
    """One libm call (or ^, which is pow) on exact arguments, kept in the
    function's real domain and away from overflow."""
    (ta, a), (tb, b) = _gen(rng, env, 2), _gen(rng, env, 2)
    f = rng.choice([*LIBM, "pow", "^", "atan2"])
    if f in ("log", "log10"):
        return f"{f}(abs({ta}) + 1)", LIBM[f](abs(a) + 1)
    if f in ("exp", "sinh", "cosh"):
        return f"{f}({ta} / (abs({ta}) + 1))", LIBM[f](a / (abs(a) + 1))
    if f in LIBM:
        return f"{f}({ta})", LIBM[f](a)
    if f == "pow":
        return f"pow(abs({ta}) + 1, {tb} / (abs({tb}) + 1))", \
            math.pow(abs(a) + 1, b / (abs(b) + 1))
    if f == "^":
        e = rng.choice((2.0, 0.5, 3.0, -1.0))
        return f"(abs({ta}) + 0.5)^{e!r}", math.pow(abs(a) + 0.5, e)
    return f"atan2({ta}, {tb})", math.atan2(a, b)


def test_min_max_keep_the_first_operand_on_a_tie():
    # Python's rule, so a signed zero does not depend on the libm's fmin
    for text, want in (("min(0, -0)", 0.0), ("min(-0, 0)", -0.0),
                       ("max(0, -0)", 0.0), ("max(-0, 0)", -0.0)):
        assert struct.pack("<d", k._eval_expr(text, [], [])) == struct.pack("<d", want), text


def test_fuzz_libm_functions_within_a_few_ulp():
    rng = random.Random(20261119)
    for _ in range(10_000):
        env = _env(rng)
        text, want = _libm_case(rng, env)
        if not math.isfinite(want):
            continue
        got = k._eval_expr(text, list(env), list(env.values()))
        assert abs(got - want) <= LIBM_ULPS * math.ulp(want), (text, got, want)


@pytest.mark.parametrize(("text", "match"), [
    ("2*", r"ends early"),
    ("2 + q", r'unknown name "q"'),
    ("foo(1)", r'unknown function "foo"'),
    ("max(1)", r"max\(\) takes 2 arguments"),
    ("(1", r"expected '\)'"),
    ("10k", r'bad number "10k" \(no unit suffixes\)'),
    ("1 2", r"unexpected '2'"),
])
def test_a_syntax_error_names_the_text_and_column(text, match):
    with pytest.raises(ValueError, match=r'test: expression ".*" at column \d+: ' + match):
        k._eval_expr(text, ["a"], [1.0], "test")


def test_precedence_matches_python():
    for text, want in (("-2^2", -(2.0 ** 2)), ("2^3^2", 2.0 ** 3 ** 2),
                       ("2^-1", 2.0 ** -1), ("1 - 2 - 3", 1 - 2 - 3),
                       ("8 / 4 / 2", 8 / 4 / 2), ("2 * pi", 2 * math.pi)):
        assert k._eval_expr(text, [], []) == want, text


def _buck(lit):
    b = p.CircuitBuilder()
    if lit:
        Lb, L2, R2 = 1e-3, 2e-3 * 1.5, 10.0 / 4
    else:
        Lb = b.param("Lbase", 1e-3, "H")
        L2 = b.param("L2", "2*Lbase*1.5", "H")
        b.param("Vin", 10.0)
        R2 = b.param("Rl", "Vin/4")
    b.add_voltage_source("V1", "in", "gnd", 10.0)
    b.add_resistor("R1", "in", "a", 1.0)
    b.add_inductor("L1", "a", "o", L2)
    b.add_capacitor("C1", "o", "gnd", 1e-6)
    b.add_resistor("RL", "o", "gnd", R2)
    b.add_inductor("L0", "o", "gnd", Lb)
    return b


def test_a_circuit_with_expressions_equals_the_literal_circuit():
    e, lit = _buck(False).build(), _buck(True).build()
    assert e.get_value("L1.L") == 2e-3 * 1.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra = p.simulate(e, 2e-4, dt=1e-7, engine="pwl")
        rb = p.simulate(lit, 2e-4, dt=1e-7, engine="pwl")
    np.testing.assert_array_equal(np.asarray(ra.states), np.asarray(rb.states))


def test_with_params_reevaluates_and_a_derived_parameter_is_refused():
    c = _buck(False).build()
    c2 = c.with_params({"Lbase": 2e-3, "Vin": 20.0})
    assert c2.get_value("L1.L") == 2 * 2e-3 * 1.5
    assert c2.get_value("RL.R") == 20.0 / 4
    assert c2.get_value("L0.L") == 2e-3
    assert c2.structure_hash() == c.structure_hash()
    with pytest.raises(ValueError, match=r'"L2" is defined by the expression "2\*Lbase\*1.5"'):
        c.with_params({"L2": 1.0})


def test_errors_name_the_parameter_and_the_text():
    b = p.CircuitBuilder()
    b.param("a", 0.0)
    with pytest.raises(ValueError, match=r'param\("z"\): expression "1/a" evaluates to inf'):
        b.param("z", "1/a")
    with pytest.raises(ValueError, match=r'param\("y"\): expression "a \+ nope" at column 5: unknown name "nope"'):
        b.param("y", "a + nope")
    b.param("w", "a + 1")
    b.add_voltage_source("V1", "in", "gnd", 1.0)
    b.add_resistor("R1", "in", "gnd", b.param("r", "w * 2"))
    with pytest.raises(ValueError, match=r'parameter "r" = 0: field R of device "R1" must be a finite number > 0'):
        b.set_param("a", -1.0)


def test_to_yaml_writes_the_text_and_round_trips():
    c = _buck(False).build()
    y = c.to_yaml()
    assert '{name: "L2", expr: "2*Lbase*1.5", unit: "H"}' in y
    c2 = p.Circuit.from_yaml(y)
    assert (c2.value_hash(), c2.structure_hash()) == (c.value_hash(), c.structure_hash())
    assert c2.to_yaml() == y
    assert c2.with_params({"Lbase": 2e-3}).get_value("L1.L") == 2 * 2e-3 * 1.5


def test_yaml_parameters_and_string_fields():
    text = """
circuit:
  parameters:
    - {name: L2, value: "2*Lbase"}
    - {name: Lbase, value: 1.0e-3, unit: H}
  devices:
    - {type: voltage_source, name: V1, from: in, to: gnd, V: 10.0}
    - {type: resistor, name: R1, from: in, to: o, R: "4*Lbase*1000"}
    - {type: inductor, name: L1, from: o, to: gnd, L: L2}
simulation: {t_end: 1.0e-4, dt: 1.0e-7}
"""
    lc = p.load_yaml_string(text)
    b = lc.builder
    assert b.get_value("R1.R") == 4 * 1e-3 * 1000
    assert b.get_value("L1.L") == 2e-3
    c = b.build().with_params({"Lbase": 2e-3})
    assert c.get_value("R1.R") == 4 * 2e-3 * 1000
    assert c.get_value("L1.L") == 4e-3


@pytest.mark.parametrize(("params", "dev", "match"), [
    ('[{name: a, value: "b"}, {name: b, value: "a"}]', "", r"depends on itself \(a -> b -> a\)"),
    ("[{name: a, value: 1.0}]", '{type: resistor, name: R1, from: in, to: gnd, R: "a + q"}',
     r'device "R1" field "R": expression "a \+ q" at column 5: unknown name "q"'),
    ("[{name: a, value: 1.0, bogus: 1}]", "", r"parameter 'a': unknown key 'bogus'"),
])
def test_yaml_expression_errors(params, dev, match):
    devs = "\n    - " + dev if dev else ""
    text = f"""
circuit:
  parameters: {params}
  devices:
    - {{type: voltage_source, name: V1, from: in, to: gnd, V: 1.0}}{devs}
"""
    with pytest.raises(Exception, match=match):
        p.load_yaml_string(text)


SPICE = """* rc with parameters
.param Rb=1k Cb=1u
.param R2={2*rb}
V1 in 0 10
R1 in out {R2}
C1 out 0 {Cb/2}
R3 out 0 {sqrt(Rb)*Rb}
.end
"""


def test_spice_params_stay_parameters_and_braces_evaluate():
    b = p.spice_to_builder(SPICE)
    assert [q.name for q in b.parameters()][:3] == ["Rb", "Cb", "R2"]
    c = b.build()
    assert c.get_value("R1.R") == 2 * 1e3
    assert c.get_value("C1.C") == 1e-6 / 2
    assert c.get_value("R3.R") == math.sqrt(1e3) * 1e3
    c2 = c.with_params({"Rb": 4e3})
    assert c2.get_value("R1.R") == 2 * 4e3
    assert c2.get_value("R3.R") == math.sqrt(4e3) * 4e3
    assert c2.get_value("C1.C") == 1e-6 / 2


def test_spice_bad_expression_names_the_value():
    with pytest.raises(ValueError, match=r"SPICE value \{2\*\}"):
        p.spice_to_builder("* x\nV1 a 0 1\nR1 a 0 {2*}\n.end\n")


def test_with_params_order_does_not_matter():
    # Review: one set_param per name checked D = Ton / T against its
    # bound on half-updated values (0.8 / 1e-5 > 1 with Ton set first).
    b = p.CircuitBuilder()
    b.param("T", 1e-5)
    b.param("Ton", 4e-6)
    D = b.param("D", "Ton/T")
    b.add_pwm_voltage_source("Vp", "a", "gnd", 1.0, 0.0, 1e5, D)
    b.add_resistor("R", "a", "gnd", 1.0)
    c = b.build()
    for order in ({"Ton": 8e-6, "T": 2e-5}, {"T": 2e-5, "Ton": 8e-6}):
        assert c.with_params(order).get_value("Vp.duty") == 8e-6 / 2e-5


def test_a_parameter_named_pi_shadows_the_constant():
    b = p.CircuitBuilder()
    b.param("pi", 3.0)
    assert b.param("x", "2*pi").value == 6.0
    assert k._eval_expr("2*pi", [], []) == 2 * math.pi


def test_a_long_parameter_chain_orders_without_recursion():
    n = 20_000
    ps = ",".join(f'{{name: p{i}, value: "p{i - 1}"}}' for i in range(n, 0, -1))
    text = f"""
circuit:
  parameters: [{ps}, {{name: p0, value: 1.0}}]
  devices:
    - {{type: voltage_source, name: V1, from: in, to: gnd, V: 1.0}}
    - {{type: resistor, name: R1, from: in, to: gnd, R: p{n}}}
"""
    assert p.load_yaml_string(text).builder.get_value("R1.R") == 1.0


def test_a_yaml_number_stays_a_literal():
    lc = p.load_yaml_string("""
circuit:
  devices:
    - {type: voltage_source, name: V1, from: in, to: gnd, V: +5}
    - {type: resistor, name: R1, from: in, to: gnd, R: 1.0e3}
""")
    assert list(lc.builder.parameters()) == []
    assert lc.builder.get_value("V1.V") == 5.0


def test_eval_expr_checks_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        k._eval_expr("b", ["a", "b"], [1.0])
