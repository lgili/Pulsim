"""The usage guide shipped in the package (``pulsim/AI_GUIDE.md``).

The guide exists because assistants write Pulsim 1.x calls with full
confidence. It is only useful if it cannot rot, so this module checks
what can be checked without running it (the code blocks themselves run
in ``test_doc_snippets.py``):

* the accessor ``pulsim.ai_guide()`` and ``python -m pulsim.guide``;
* the size budget (the guide is meant to be loaded whole);
* every inline ``p.<name>`` / ``pulsim.<name>`` / ``b.<method>`` /
  ``CircuitBuilder.<method>`` / ``res.<attr>`` reference resolves;
* the quick-reference signatures match the installed package;
* the unknown-attribute hint of ``pulsim.__getattr__`` names the guide;
* in-package docstrings do not contradict the guide.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import inspect
import io
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pulsim = pytest.importorskip("pulsim")

GUIDE = Path(pulsim.__file__).with_name("AI_GUIDE.md")
STUB = Path(pulsim.__file__).with_name("_pulsim.pyi")
MAX_BYTES = 32 * 1024


@pytest.fixture(scope="module")
def guide_text() -> str:
    assert GUIDE.is_file(), f"{GUIDE} is missing from the imported package"
    return GUIDE.read_text(encoding="utf-8")


def _subprocess_env() -> dict[str, str]:
    """Make a child interpreter import the same pulsim as this session."""
    env = dict(os.environ)
    parent = str(Path(pulsim.__file__).parents[1])  # not resolved: keep symlinks
    if Path(parent).name not in ("site-packages", "dist-packages"):
        env["PYTHONPATH"] = parent + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_cli(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-W", "error", "-m", "pulsim.guide", *args],
        env=_subprocess_env(), capture_output=True, timeout=120)


# ---------------------------------------------------------------------------
# Accessor and CLI
# ---------------------------------------------------------------------------
def test_accessor_returns_version_header_and_the_file(guide_text):
    text = pulsim.ai_guide()
    header, _, body = text.partition("\n")
    assert pulsim.__version__ in header
    assert str(GUIDE) in header or GUIDE.name in header
    assert body == guide_text
    assert "ai_guide" in pulsim.__all__


def test_cli_prints_the_guide_without_warnings():
    proc = _run_cli()
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert proc.stderr == b""
    out = proc.stdout.decode("utf-8").replace("\r\n", "\n")
    assert out == pulsim.ai_guide() + "\n"


def test_cli_path_mode_names_the_guide_file(guide_text):
    proc = _run_cli("--path")
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    lines = proc.stdout.decode("utf-8").splitlines()
    assert len(lines) == 1
    assert Path(lines[0]).read_text(encoding="utf-8") == guide_text


def test_importing_pulsim_does_not_import_the_cli_module():
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, pulsim; print('pulsim.guide' in sys.modules)"],
        env=_subprocess_env(), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_guide_fits_the_budget_and_is_ascii(guide_text):
    size = GUIDE.stat().st_size
    assert size <= MAX_BYTES, f"AI_GUIDE.md is {size} bytes (> {MAX_BYTES})"
    # ASCII keeps `python -m pulsim.guide` safe on any console encoding.
    bad = [(n, ln) for n, ln in enumerate(guide_text.splitlines(), 1)
           if not ln.isascii()]
    assert not bad, bad[:5]


# ---------------------------------------------------------------------------
# Inline names resolve
# ---------------------------------------------------------------------------
_REF = re.compile(
    r"(?<![\w.])(?P<obj>p|pulsim|b|CircuitBuilder|res)\.(?P<name>[A-Za-z_]\w*)(?![\w*])")
_RESULT_ATTRS = {"engine_used", "engine_route_reason"}  # set by simulate()


def _unresolved(text: str) -> list[str]:
    bad = []
    for m in _REF.finditer(text):
        obj, name = m.group("obj"), m.group("name")
        if obj in ("p", "pulsim"):
            ok = (hasattr(pulsim, name)
                  or importlib.util.find_spec(f"pulsim.{name}") is not None)
        elif obj in ("b", "CircuitBuilder"):
            ok = hasattr(pulsim.CircuitBuilder, name)
        else:
            ok = hasattr(pulsim.SimulationResult, name) or name in _RESULT_ATTRS
        if not ok:
            line = text.count("\n", 0, m.start()) + 1
            bad.append(f"AI_GUIDE.md:{line}: {m.group(0)}")
    return bad


def test_every_inline_reference_resolves(guide_text):
    bad = _unresolved(guide_text)
    assert not bad, "names the installed pulsim does not have:\n" + "\n".join(bad)


def test_inline_references_resolve_without_optional_extras():
    # CI installs ".[dev]", which has neither pyqtgraph nor matplotlib;
    # pulsim exports e.g. LiveScope only when pyqtgraph imports. Block
    # both in a child interpreter so this check does not depend on what
    # the local Python happens to have.
    code = (
        "import sys\n"
        "sys.modules['pyqtgraph'] = None\n"
        "sys.modules['matplotlib'] = None\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import test_ai_guide as t\n"
        "print('\\n'.join(t._unresolved(t.GUIDE.read_text(encoding='utf-8'))))\n")
    proc = subprocess.run([sys.executable, "-c", code], env=_subprocess_env(),
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", (
        "names that need an optional extra:\n" + proc.stdout)


def test_name_check_catches_an_invented_name(guide_text):
    edited = guide_text + "\nUse `p.add_buck_converter(...)` and `b.add_mosfett`.\n"
    bad = _unresolved(edited)
    assert any("add_buck_converter" in x for x in bad)
    assert any("add_mosfett" in x for x in bad)


def test_no_retired_1x_symbol_in_code(guide_text):
    fences = re.findall(r"(?ms)^```\w*\n(.*?)^```", guide_text)
    hits = []
    for code in fences:
        for m in re.finditer(r"\b(?:p|pulsim)\.(\w+)", code):
            if m.group(1) in pulsim._V1_SYMBOL_HINTS and not hasattr(pulsim, m.group(1)):
                hits.append(m.group(0))
    assert not hits, hits


# ---------------------------------------------------------------------------
# Quick-reference signatures match the package
# ---------------------------------------------------------------------------
Param = tuple[str, str, object]   # (name, kind, default or _NO)
_NO = object()


def _ast_params(args: ast.arguments, drop_self: bool) -> list[Param]:
    def default(node):
        if node is None:
            return _NO
        try:
            return ast.literal_eval(node)
        except ValueError:
            return ast.unparse(node)

    pos = [*args.posonlyargs, *args.args]
    pads = [None] * (len(pos) - len(args.defaults)) + list(args.defaults)
    out: list[Param] = [(a.arg, "pos", default(d)) for a, d in zip(pos, pads)]
    if args.vararg:
        out.append((args.vararg.arg, "*args", _NO))
    out += [(a.arg, "kw", default(d)) for a, d in zip(args.kwonlyargs, args.kw_defaults)]
    if args.kwarg:
        out.append((args.kwarg.arg, "**kw", _NO))
    if drop_self and out and out[0][0] == "self":
        out = out[1:]
    return out


def _inspect_params(fn) -> list[Param]:
    kinds = {inspect.Parameter.POSITIONAL_ONLY: "pos",
             inspect.Parameter.POSITIONAL_OR_KEYWORD: "pos",
             inspect.Parameter.VAR_POSITIONAL: "*args",
             inspect.Parameter.KEYWORD_ONLY: "kw",
             inspect.Parameter.VAR_KEYWORD: "**kw"}
    out = []
    for prm in inspect.signature(fn).parameters.values():
        d = _NO if prm.default is inspect.Parameter.empty else prm.default
        out.append((prm.name, kinds[prm.kind], d))
    if out and out[0][0] == "self":
        out = out[1:]
    return out


@pytest.fixture(scope="module")
def stub_index() -> dict[str, list[list[Param]]]:
    """``"Class.method"`` / ``"function"`` -> parameter lists (overloads)."""
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    index: dict[str, list[list[Param]]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            index.setdefault(node.name, []).append(_ast_params(node.args, False))
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    index.setdefault(f"{node.name}.{sub.name}", []).append(
                        _ast_params(sub.args, True))
    return index


def _runtime_params(obj: str, name: str, stub_index) -> list[list[Param]]:
    if obj == "b":
        return stub_index[f"CircuitBuilder.{name}"]
    if obj == "res":
        return [_inspect_params(getattr(pulsim.SimulationResult, name))]
    target = getattr(pulsim, name)
    if isinstance(target, type):
        return stub_index[f"{name}.__init__"]
    try:
        return [_inspect_params(target)]
    except ValueError:  # pybind builtin: the stub has the signature
        return stub_index[name]


def _quickref(text: str) -> list[tuple[int, str]]:
    m = re.search(r"(?ms)^## 4\. .*?^```text\n(.*?)^```", text)
    assert m, "quick-reference block (section 4, ```text fence) not found"
    first = text.count("\n", 0, m.start(1)) + 1
    return [(first + k, ln) for k, ln in enumerate(m.group(1).splitlines()) if ln.strip()]


def signature_problems(text: str, stub_index) -> list[str]:
    problems = []
    for line_no, line in _quickref(text):
        m = re.fullmatch(r"(p|b|res)\.(\w+)\((.*)\)", line.strip())
        if not m:
            problems.append(f"line {line_no}: not of the form obj.name(params): {line}")
            continue
        obj, name, params = m.groups()
        partial = params.rstrip().endswith("...")
        if partial:
            params = params.rstrip()[:-3].rstrip().rstrip(",")
        listed = _ast_params(ast.parse(f"def f({params}): pass").body[0].args, False)
        candidates = _runtime_params(obj, name, stub_index)
        ok = any((real[:len(listed)] if partial else real) == listed
                 for real in candidates)
        if not ok:
            problems.append(f"line {line_no}: {obj}.{name}: guide lists {listed}, "
                            f"the package has {candidates}")
    return problems


def test_quick_reference_matches_the_package(guide_text, stub_index):
    problems = signature_problems(guide_text, stub_index)
    assert not problems, "\n".join(problems)


def test_signature_check_catches_swapped_and_wrong_defaults(guide_text, stub_index):
    swapped = guide_text.replace(
        "b.add_sine_voltage_source(name, n_pos, n_neg, v_dc, v_amplitude,",
        "b.add_sine_voltage_source(name, n_pos, n_neg, v_amplitude, v_dc,")
    assert swapped != guide_text
    assert any("add_sine_voltage_source" in p for p in signature_problems(swapped, stub_index))
    wrong = guide_text.replace("b.add_diode(name, anode, cathode, g_on, g_off, V_th=0.0)",
                               "b.add_diode(name, anode, cathode, g_on, g_off, V_th=0.7)")
    assert wrong != guide_text
    assert any("add_diode" in p for p in signature_problems(wrong, stub_index))


# ---------------------------------------------------------------------------
# The rest of the package points to, and agrees with, the guide
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["Simulator", "NoSuchThing"])
def test_unknown_attribute_hint_names_the_guide(name):
    with pytest.raises(AttributeError, match=r"python -m pulsim\.guide"):
        getattr(pulsim, name)


def _param_paragraph(doc: str, param: str) -> str:
    lines = doc.splitlines()
    for k, ln in enumerate(lines):
        names = [s.strip() for s in ln.split(",")]
        if param in names and ln == ln.lstrip():
            body = []
            for nxt in lines[k + 1:]:
                if nxt and nxt == nxt.lstrip():
                    break
                body.append(nxt)
            return " ".join(" ".join(body).split())
    raise AssertionError(f"parameter {param!r} not documented")


@pytest.mark.parametrize("func,param", [
    ("add_bridge_rectifier", "V_th"),
    ("add_three_phase_vsi", "V_f"),
    ("add_buck", "V_th"),
    ("add_boost", "V_th"),
    ("add_flyback", "V_th"),
])
def test_diode_threshold_parameters_are_not_called_a_drop(func, param):
    text = _param_paragraph(inspect.getdoc(getattr(pulsim, func)), param)
    assert "threshold" in text.lower(), text
    assert not re.search(r"(?<!not a )forward (drop|voltage)", text, re.I), text


def test_catalogue_does_not_recommend_a_micro_ohm_reference():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pulsim.catalog()
    out = buf.getvalue()
    assert not re.search(r"(µ|u)\s*(Ω|ohm)", out, re.I), "micro-ohm advice in catalog()"
    entry = re.search(r"add_resistor.*?(?=\n\s*\n|\n\s*add_)", out, re.S)
    assert entry and re.search(r"large|1 ?G|1e9", entry.group(0), re.I), out[:2000]


# ---------------------------------------------------------------------------
# docs/llms.txt
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
LLMS = REPO_ROOT / "docs" / "llms.txt"
_RAW = "https://raw.githubusercontent.com/lgili/Pulsim/main/"


def test_llms_txt_structure_and_links():
    if not LLMS.is_file():
        pytest.skip("docs/llms.txt not found (running without the repository)")
    text = LLMS.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert re.match(r"# \S", lines[0]), "line 1 must be an H1"
    first = next(ln for ln in lines[1:] if ln.strip())
    assert first.startswith("> "), "the H1 must be followed by a blockquote summary"
    assert "python -m pulsim.guide" in first
    sections = [ln[3:] for ln in lines if ln.startswith("## ")]
    assert sections[:1] == ["Start here"], sections
    links = re.findall(r"\[([^\]]+)\]\(([^)\s]+)\)", text)
    assert links and links[0][1].endswith("python/pulsim/AI_GUIDE.md"), links[:1]
    missing = []
    for _, target in links:
        if target.startswith(_RAW):
            path = REPO_ROOT / target[len(_RAW):]
        elif re.match(r"https?://", target):
            continue
        else:
            path = LLMS.parent / target
        if not path.is_file():
            missing.append(target)
    assert not missing, f"llms.txt links to files that do not exist: {missing}"
