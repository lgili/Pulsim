"""The shipped type stub ``pulsim/_pulsim.pyi`` must match the built extension.

pybind11 callables have no ``inspect.signature``; IDEs, type checkers and
AI assistants read argument names and order from the ``.pyi``. A stale
stub therefore teaches a wrong call with full confidence. These tests are
dependency-free (no pybind11-stubgen): they compare the stub, parsed with
``ast``, against the live module -- names, parameter names, parameter
order and which parameters have a default (taken from the signature line
pybind11 writes at the top of every ``__doc__``).

The byte-for-byte check lives in ``scripts/generate_stubs.py --check``
(pinned stubgen, one CI job). The mypy checks at the bottom run only
where mypy is installed (the ``dev`` extra).
"""

from __future__ import annotations

import ast
import inspect
import io
import os
import re
import subprocess
import sys
import tokenize
import types
from pathlib import Path

import pytest

pulsim = pytest.importorskip("pulsim")
native = pytest.importorskip("pulsim._pulsim")

REGEN = "python scripts/generate_stubs.py"
REPO_ROOT = Path(__file__).resolve().parents[2]
STUB_PATH = Path(pulsim.__file__).with_name("_pulsim.pyi")


@pytest.fixture(scope="module")
def stub_src() -> str:
    assert STUB_PATH.is_file(), (
        f"{STUB_PATH} is missing next to the imported pulsim package; "
        f"run {REGEN} (and make sure the build copies *.pyi)")
    return STUB_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Stub index
# ---------------------------------------------------------------------------
Params = tuple[tuple[str, bool], ...]  # ((name, has_default), ...)


def _stub_params(fn: ast.FunctionDef) -> Params:
    a = fn.args
    positional = [*a.posonlyargs, *a.args]
    n_req = len(positional) - len(a.defaults)
    out = [(p.arg, i >= n_req) for i, p in enumerate(positional)]
    if a.vararg is not None:
        out.append(("*" + a.vararg.arg, False))
    out += [(p.arg, d is not None) for p, d in zip(a.kwonlyargs, a.kw_defaults)]
    if a.kwarg is not None:
        out.append(("**" + a.kwarg.arg, False))
    return tuple(out)


def _decorators(fn: ast.FunctionDef) -> set[str]:
    return {ast.unparse(d) for d in fn.decorator_list}


class _StubClass:
    def __init__(self, node: ast.ClassDef) -> None:
        self.functions: dict[str, list[ast.FunctionDef]] = {}
        self.attributes: set[str] = set()   # AnnAssign / Assign: writable
        self.properties: set[str] = set()
        self.setters: set[str] = set()
        self.classes: set[str] = set()
        for m in node.body:
            if isinstance(m, ast.FunctionDef):
                decs = _decorators(m)
                if "property" in decs:
                    self.properties.add(m.name)
                elif f"{m.name}.setter" in decs:
                    self.setters.add(m.name)
                else:
                    self.functions.setdefault(m.name, []).append(m)
            elif isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name):
                self.attributes.add(m.target.id)
            elif isinstance(m, ast.Assign):
                self.attributes.update(
                    t.id for t in m.targets if isinstance(t, ast.Name))
            elif isinstance(m, ast.ClassDef):
                self.classes.add(m.name)

    def names(self) -> set[str]:
        return (set(self.functions) | self.attributes | self.properties
                | self.classes)


class _StubIndex:
    def __init__(self, src: str) -> None:
        tree = ast.parse(src)
        self.functions: dict[str, list[ast.FunctionDef]] = {}
        self.classes: dict[str, _StubClass] = {}
        self.names: set[str] = set()
        for n in tree.body:
            if isinstance(n, ast.FunctionDef):
                self.functions.setdefault(n.name, []).append(n)
                self.names.add(n.name)
            elif isinstance(n, ast.ClassDef):
                self.classes[n.name] = _StubClass(n)
                self.names.add(n.name)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                self.names.add(n.target.id)
            elif isinstance(n, ast.Assign):
                self.names.update(
                    t.id for t in n.targets if isinstance(t, ast.Name))


# ---------------------------------------------------------------------------
# Runtime side
# ---------------------------------------------------------------------------
def _is_dunder(name: str) -> bool:
    # ``_pybind11_conduit_v1_`` (pybind11 3) is pybind11's cross-module
    # interop hook on every class, not pulsim API: treated like a dunder.
    return (name.startswith("__") and name.endswith("__")) \
        or name.startswith("_pybind11_")


def _is_pybind_callable(obj: object) -> bool:
    return type(obj).__name__ in ("builtin_function_or_method", "instancemethod")


def _runtime_classes() -> dict[str, type]:
    return {n: o for n in dir(native) if not _is_dunder(n)
            for o in [getattr(native, n)] if isinstance(o, type)}


def _split_top_level(text: str) -> list[str]:
    """Split on commas outside brackets, ``<...>`` reprs and strings."""
    parts, depth, quote, start, i = [], 0, None, 0, 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "([{<":
            depth += 1
        elif ch in ")]}>" and not (ch == ">" and text[i - 1] == "-"):
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
        i += 1
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def _params_from_doc_signature(sig: str, name: str) -> Params:
    """``name(a: int, b: float = 1.0) -> X`` -> ``(("a", False), ("b", True))``.

    Not parsed with ``ast``: pybind11 signatures may legally contain C++
    names and ``<object at 0x...>`` defaults, which are not Python.
    """
    assert sig.startswith(name + "("), sig
    depth = 0
    for i in range(len(name), len(sig)):
        if sig[i] in "([{":
            depth += 1
        elif sig[i] in ")]}":
            depth -= 1
            if depth == 0:
                inner = sig[len(name) + 1:i]
                break
    else:  # pragma: no cover - malformed pybind doc
        raise AssertionError(f"unbalanced signature: {sig}")
    out = []
    for p in _split_top_level(inner):
        if p in ("*", "/"):
            continue
        head = p.split("=", 1)[0] if "=" in p else p
        pname = head.split(":", 1)[0].strip()
        out.append((pname, "=" in p))
    return tuple(out)


def _runtime_overloads(obj: object, name: str) -> set[Params] | None:
    """Parameter lists of a pybind callable, one per overload."""
    doc = getattr(obj, "__doc__", None) or ""
    lines = doc.splitlines()
    if not lines or not lines[0].startswith(name + "("):
        return None
    if len(lines) > 1 and lines[1].strip() == "Overloaded function.":
        sigs = [re.sub(r"^\d+\. ", "", ln) for ln in lines
                if re.match(rf"^\d+\. {re.escape(name)}\(", ln)]
        return {_params_from_doc_signature(s, name) for s in sigs}
    return {_params_from_doc_signature(lines[0], name)}


def _python_params(fn: types.FunctionType) -> tuple[str, ...]:
    out = []
    for p in inspect.signature(fn).parameters.values():
        prefix = {p.VAR_POSITIONAL: "*", p.VAR_KEYWORD: "**"}.get(p.kind, "")
        out.append(prefix + p.name)
    return tuple(out)


def _instance_attributes() -> set[str]:
    """Public attributes that ``pulsim.simulate`` sets on its result.

    Not discoverable on the class, so run the smallest circuit through
    both engine paths (fixed ``dt`` and variable-step ``auto``).
    """
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("V1", "a", "0", 1.0)
    b.add_resistor("R1", "a", "c", 1e3)
    b.add_capacitor("C1", "c", "0", 1e-6)
    names: set[str] = set()
    for kw in ({}, {"dt": 1e-6}):
        res = pulsim.simulate(b, t_end=2e-5, **kw)
        names.update(k for k in vars(res) if not k.startswith("_"))
    return names


# ---------------------------------------------------------------------------
# Checks (pure functions of the stub text, so the mutation tests can feed
# them a corrupted stub)
# ---------------------------------------------------------------------------
def coverage_problems(stub_src: str, instance_attrs: set[str]) -> list[str]:
    idx = _StubIndex(stub_src)
    problems = []
    for n in dir(native):
        if not _is_dunder(n) and n not in idx.names:
            problems.append(f"pulsim._pulsim.{n}")
    for cname, cls in _runtime_classes().items():
        scls = idx.classes.get(cname)
        if scls is None:
            continue  # already reported above
        declared = scls.names() | scls.setters
        for k, v in vars(cls).items():
            # Dunders and pybind internals (``__entries``) are only
            # checked when they are bound callables (``__init__``, ...).
            if k.startswith("__") and not _is_pybind_callable(v):
                continue
            if k.startswith("_pybind11_"):
                continue
            if k not in declared:
                problems.append(f"{cname}.{k}")
    sr = idx.classes.get("SimulationResult")
    for attr in sorted(instance_attrs):
        if sr is None or attr not in sr.names():
            problems.append(f"SimulationResult.{attr} (set by pulsim.simulate)")
    return problems


def _fmt(params: Params) -> str:
    return "(" + ", ".join(n + ("=..." if d else "") for n, d in params) + ")"


def signature_problems(stub_src: str) -> list[str]:
    idx = _StubIndex(stub_src)
    problems: list[str] = []

    def compare(qual: str, obj: object, name: str,
                stub_defs: list[ast.FunctionDef] | None) -> None:
        if isinstance(obj, staticmethod):
            obj = obj.__func__
        if isinstance(obj, types.FunctionType):
            if not stub_defs:
                return  # coverage_problems reports the absence
            runtime = _python_params(obj)
            for d in stub_defs:
                stub_names = tuple(n for n, _ in _stub_params(d))
                if stub_names != runtime:
                    problems.append(
                        f"{qual}: Python layer takes {runtime}, stub declares "
                        f"{stub_names} (edit scripts/stubs_overlay.pyi)")
            return
        if not _is_pybind_callable(obj):
            return
        runtime = _runtime_overloads(obj, name)
        if runtime is None or not stub_defs:
            return
        stub = {_stub_params(d) for d in stub_defs}
        if runtime != stub:
            problems.append(
                f"{qual}: binding {sorted(map(_fmt, runtime))} "
                f"!= stub {sorted(map(_fmt, stub))}")

    for n in dir(native):
        if _is_dunder(n):
            continue
        obj = getattr(native, n)
        if not isinstance(obj, type):
            compare(n, obj, n, idx.functions.get(n))
    for cname, cls in _runtime_classes().items():
        scls = idx.classes.get(cname)
        if scls is None:
            continue
        for k, v in vars(cls).items():
            qual = f"{cname}.{k}"
            if k.startswith("__") and not _is_pybind_callable(v):
                continue  # __members__ & co: enum machinery, not API
            if isinstance(v, property) and (k in scls.properties or k in scls.attributes):
                writable_rt = v.fset is not None
                writable_stub = k in scls.setters or k in scls.attributes
                if writable_rt != writable_stub:
                    problems.append(
                        f"{qual}: property is {'read-write' if writable_rt else 'read-only'}"
                        f" at runtime but {'read-write' if writable_stub else 'read-only'}"
                        " in the stub")
                continue
            compare(qual, v, k, scls.functions.get(k))
    return problems


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _code_tokens(src: str) -> str:
    toks = tokenize.generate_tokens(io.StringIO(src).readline)
    return " ".join(t.string for t in toks
                    if t.type not in (tokenize.STRING, tokenize.COMMENT))


def test_stub_is_valid_python_without_generator_artifacts(stub_src):
    ast.parse(stub_src)
    code = _code_tokens(stub_src)
    assert "pybind11_stubgen" not in code, (
        "the stub imports pybind11_stubgen (not a runtime dependency); "
        f"run {REGEN}")
    assert not re.search(r"0x[0-9a-fA-F]{6,}", code), (
        f"the stub contains an object address; run {REGEN}")
    assert "::" not in code, f"the stub contains a C++ name; run {REGEN}"


def test_py_typed_marker_ships():
    assert Path(pulsim.__file__).with_name("py.typed").is_file()


@pytest.fixture(scope="module")
def instance_attrs() -> set[str]:
    return _instance_attributes()


def test_stub_declares_every_runtime_member(stub_src, instance_attrs):
    # Sanity: the attributes the guide relies on are really there.
    assert {"engine_used", "engine_route_reason"} <= instance_attrs
    problems = coverage_problems(stub_src, instance_attrs)
    assert not problems, (
        "pulsim/_pulsim.pyi is missing names the built module has:\n  "
        + "\n  ".join(problems) + f"\nRegenerate with: {REGEN}")


def test_stub_signatures_match_bindings(stub_src):
    problems = signature_problems(stub_src)
    assert not problems, (
        "pulsim/_pulsim.pyi disagrees with the built module "
        "(parameter names / order / defaults):\n  "
        + "\n  ".join(problems) + f"\nRegenerate with: {REGEN}")


def test_signature_check_actually_reads_the_bindings():
    # Guard against a parser that silently returns nothing.
    sig = _runtime_overloads(native.CircuitBuilder.add_sine_voltage_source,
                             "add_sine_voltage_source")
    assert sig == {(("self", False), ("name", False), ("n_pos", False),
                    ("n_neg", False), ("v_dc", False), ("v_amplitude", False),
                    ("frequency", False), ("phase", True))}


# --- the checks are not vacuous (task 3.4) ---------------------------------
def _mutate(stub_src: str, cls: str, transform) -> str:
    tree = ast.parse(stub_src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            transform(node)
            return ast.unparse(tree)
    raise AssertionError(f"class {cls} not in stub")


def test_coverage_check_catches_a_missing_method(stub_src, instance_attrs):
    def drop(node: ast.ClassDef) -> None:
        node.body = [m for m in node.body
                     if getattr(m, "name", None) != "add_sine_voltage_source"]

    broken = _mutate(stub_src, "CircuitBuilder", drop)
    problems = coverage_problems(broken, instance_attrs)
    assert problems == ["CircuitBuilder.add_sine_voltage_source"]


def test_coverage_check_catches_a_missing_overlay_member(stub_src, instance_attrs):
    def drop(node: ast.ClassDef) -> None:
        node.body = [m for m in node.body
                     if not (isinstance(m, ast.AnnAssign)
                             and getattr(m.target, "id", "") == "engine_used")]

    broken = _mutate(stub_src, "SimulationResult", drop)
    problems = coverage_problems(broken, instance_attrs)
    assert problems == ["SimulationResult.engine_used (set by pulsim.simulate)"]


def test_signature_check_catches_swapped_parameters(stub_src):
    def swap(node: ast.ClassDef) -> None:
        for m in node.body:
            if isinstance(m, ast.FunctionDef) and m.name == "add_sine_voltage_source":
                names = [a.arg for a in m.args.args]
                i, j = names.index("v_dc"), names.index("v_amplitude")
                m.args.args[i], m.args.args[j] = m.args.args[j], m.args.args[i]

    broken = _mutate(stub_src, "CircuitBuilder", swap)
    problems = signature_problems(broken)
    assert len(problems) == 1
    assert problems[0].startswith("CircuitBuilder.add_sine_voltage_source:")
    # Both orders are reported.
    assert "v_dc, v_amplitude" in problems[0]
    assert "v_amplitude, v_dc" in problems[0]


def test_signature_check_catches_a_dropped_default(stub_src):
    def drop_default(node: ast.ClassDef) -> None:
        for m in node.body:
            if isinstance(m, ast.FunctionDef) and m.name == "add_diode":
                m.args.defaults = []

    problems = signature_problems(_mutate(stub_src, "CircuitBuilder", drop_default))
    assert [p.split(":")[0] for p in problems] == ["CircuitBuilder.add_diode"]


def test_signature_check_catches_a_drifted_overlay(stub_src):
    def rename(node: ast.ClassDef) -> None:
        for m in node.body:
            if isinstance(m, ast.FunctionDef) and m.name == "power":
                m.args.args[1].arg = "name"

    problems = signature_problems(_mutate(stub_src, "SimulationResult", rename))
    assert len(problems) == 1 and "SimulationResult.power" in problems[0]


# --- mypy (task 3.5); skipped where mypy is not installed -------------------
@pytest.fixture(scope="module")
def mypy_cache(tmp_path_factory) -> Path:
    # One cache for the module: the first run analyses the pulsim
    # package once, the others reuse it.
    return tmp_path_factory.mktemp("mypy_cache")


def _mypy(tmp_path: Path, cache: Path, *targets: str) -> subprocess.CompletedProcess[str]:
    pytest.importorskip("mypy", reason="mypy not installed (dev extra)")
    env = dict(os.environ)
    # Point mypy at the directory that CONTAINS the imported pulsim, so
    # it type-checks exactly what this test run imported. An installed
    # wheel is found through the interpreter's site-packages instead
    # (mypy refuses site-packages on MYPYPATH).
    pkg_parent = Path(pulsim.__file__).resolve().parents[1]
    if pkg_parent.name not in ("site-packages", "dist-packages"):
        env["MYPYPATH"] = str(pkg_parent)
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--follow-imports=silent",
         "--ignore-missing-imports", "--no-error-summary",
         "--cache-dir", str(cache), *targets],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)


def test_mypy_accepts_the_stub(tmp_path, mypy_cache):
    # By module name: a site-packages file path makes mypy treat
    # site-packages as a source root (typing_extensions "shadows" errors).
    proc = _mypy(tmp_path, mypy_cache, "-m", "pulsim._pulsim")
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_mypy_rejects_a_guessed_keyword(tmp_path, mypy_cache):
    (tmp_path / "guess.py").write_text(
        "import pulsim as p\n"
        "b = p.CircuitBuilder()\n"
        'b.add_sine_voltage_source("V", "a", "gnd", amplitude=10.0, frequency=50.0)\n')
    proc = _mypy(tmp_path, mypy_cache, "guess.py")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert 'Unexpected keyword argument "amplitude"' in proc.stdout
    assert "add_sine_voltage_source" in proc.stdout


def _doc_snippets() -> list[Path]:
    docs = [REPO_ROOT / "docs" / "getting-started.md",
            Path(pulsim.__file__).with_name("AI_GUIDE.md")]
    return [d for d in docs if d.is_file()]


def test_mypy_finds_no_pulsim_error_in_the_documented_snippets(tmp_path, mypy_cache):
    docs = _doc_snippets()
    if not docs:
        pytest.skip("no documentation snippets found (installed without the repo)")
    # Names whose appearance in a mypy message means the error is about
    # the pulsim API rather than the snippet's own plumbing.
    pulsim_names = {n for n in dir(pulsim) if not n.startswith("_")}
    pulsim_names |= set(_StubIndex(STUB_PATH.read_text(encoding="utf-8")).classes)
    targets = []
    for doc in docs:
        blocks = re.findall(r"(?ms)^```python\n(.*?)^```",
                            doc.read_text(encoding="utf-8"))
        target = tmp_path / (re.sub(r"\W", "_", doc.stem) + ".py")
        target.write_text("\n".join(blocks), encoding="utf-8")
        targets.append(target.name)
    proc = _mypy(tmp_path, mypy_cache, *targets)
    errors = [ln for ln in proc.stdout.splitlines() if ": error:" in ln]
    about_pulsim = [
        ln for ln in errors
        if "pulsim" in ln or any(f'"{n}"' in ln for n in pulsim_names)]
    assert not about_pulsim, "\n".join(about_pulsim)
