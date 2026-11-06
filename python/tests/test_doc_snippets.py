"""Execute the ``python`` blocks of the user-facing documentation.

Docs that are not executed rot: until 2.0's tutorials were fixed by hand,
three of the six raised ``TypeError`` on their first helper call, and one
read ``examples/buck.yaml``, which a pip install does not ship. This module
runs every block, in order, the way a reader pastes them.

Conventions for doc authors
---------------------------

Which files
    ``python/pulsim/AI_GUIDE.md``, ``README.md``, ``docs/getting-started.md``
    and ``docs/tutorials/*.md`` (see ``DOC_FILES``). One test per file.

Which blocks
    A fence that starts at column 0 with the info string exactly
    ``python``. ``bash``, ``text``, ``pycon``, ``yaml`` and bare fences
    never run. A fence inside a blockquote (``> ```python``) or indented
    under a list item is not collected either: use those for fragments
    that are not meant to run on their own.

How they run
    All blocks of one file run in document order in ONE shared namespace
    (later blocks see earlier names), with a fresh temporary directory as
    the working directory (so the docs must not depend on a clone of the
    repository, and ``save=`` writes nowhere) and ``MPLBACKEND=Agg``. A
    failure names ``file:line`` of the failing fence; tracebacks carry
    the markdown line numbers.

Expected output
    When a ``python`` fence is followed by a bare or ``text`` fence, with
    only blank lines or ONE prose line ending in ``:`` between them (e.g.
    ``Expected output:``), the block's stdout must match that fence.
    Whitespace is normalised, words must match exactly, and each number
    may differ by one unit of its last printed digit (or 1e-6 relative,
    whichever is larger).

Annotations
    An HTML comment on the line directly above a fence (invisible on
    GitHub and in MkDocs):

    * ``<!-- doctest: requires=matplotlib[,pandas] -->`` -- if a module
      is missing, this block and the REST OF THE FILE are skipped and the
      test reports a skip naming the module.
    * ``<!-- doctest: skip reason="..." -->`` -- an illustrative block
      that cannot run as written. The reason is mandatory.
    * ``<!-- doctest: no-output-check -->`` -- run, but do not compare
      against the following fence.
    * ``<!-- doctest: xfail reason="..." -->`` -- the block documents a
      check that FAILS on this version because of a known open defect.
      It must raise ``AssertionError`` (any other exception is a broken
      block); when it stops raising the test fails, so the annotation is
      removed as soon as the defect is fixed. The reason is mandatory and
      should name the fix. Later blocks still run.

    Any other ``doctest:`` annotation fails the test.

Runtime
    The whole module warns above ``BUDGET_WARN_S`` and fails above
    ``BUDGET_FAIL_S`` of snippet wall time.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import linecache
import os
import re
import shlex
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytest.importorskip("pulsim")

pytestmark = pytest.mark.doc_snippets

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_FILES = [
    REPO_ROOT / "python" / "pulsim" / "AI_GUIDE.md",
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "getting-started.md",
    *sorted((REPO_ROOT / "docs" / "tutorials").glob("*.md")),
]
BUDGET_WARN_S = 30.0
BUDGET_FAIL_S = 60.0
_ELAPSED: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
@dataclass
class Fence:
    info: str
    line: int          # 1-based line of the opening ``` fence
    body: str
    annotation: str | None
    end_line: int      # 1-based line of the closing fence


@dataclass
class Block:
    line: int
    code: str
    expected: str | None = None
    requires: list[str] = field(default_factory=list)
    skip_reason: str | None = None
    xfail_reason: str | None = None


_ANNOTATION = re.compile(r"^<!--\s*doctest:(.*?)-->\s*$")
_OPEN = re.compile(r"^```([^\s`]*)\s*$")


def _fences(text: str) -> list[Fence]:
    lines = text.splitlines()
    out: list[Fence] = []
    i = 0
    while i < len(lines):
        m = _OPEN.match(lines[i])
        if not m:
            i += 1
            continue
        start = i
        j = i + 1
        while j < len(lines) and lines[j].rstrip() != "```":
            j += 1
        if j >= len(lines):
            raise ValueError(f"line {start + 1}: unclosed code fence")
        ann = None
        if start > 0:
            am = _ANNOTATION.match(lines[start - 1].strip())
            if am:
                ann = am.group(1).strip()
        out.append(Fence(m.group(1), start + 1,
                         "\n".join(lines[start + 1:j]) + "\n", ann, j + 1))
        i = j + 1
    return out


def _parse_annotation(ann: str, where: str, block: Block) -> None:
    head, _, rest = ann.partition(" ")
    if head.startswith("requires="):
        mods = [m.strip() for m in head[len("requires="):].split(",")]
        if not all(mods) or rest.strip():
            raise ValueError(f"{where}: malformed annotation 'doctest: {ann}'")
        block.requires = mods
        return
    if head in ("skip", "xfail"):
        try:
            kv = dict(tok.split("=", 1) for tok in shlex.split(rest))
        except ValueError:
            kv = {}
        reason = kv.get("reason", "").strip()
        if not reason or set(kv) != {"reason"}:
            raise ValueError(
                f"{where}: 'doctest: {head}' needs a non-empty "
                f'reason="..." (got {ann!r})')
        if head == "skip":
            block.skip_reason = reason
        else:
            block.xfail_reason = reason
        return
    if ann == "no-output-check":
        block.expected = None
        return
    raise ValueError(f"{where}: unknown annotation 'doctest: {ann}'")


def collect_blocks(text: str, name: str = "<doc>") -> list[Block]:
    """The runnable blocks of a markdown text, with their expectations."""
    lines = text.splitlines()
    fences = _fences(text)
    blocks: list[Block] = []
    for k, f in enumerate(fences):
        if f.info != "python":
            if f.annotation is not None:
                raise ValueError(
                    f"{name}:{f.line}: 'doctest:' annotation on a "
                    f"{f.info or 'bare'} fence, which never runs")
            continue
        block = Block(f.line, f.body)
        nxt = fences[k + 1] if k + 1 < len(fences) else None
        if nxt is not None and nxt.info in ("", "text"):
            between = [ln.strip() for ln in lines[f.end_line:nxt.line - 1]]
            prose = [ln for ln in between if ln]
            if not prose or (len(prose) == 1 and prose[0].endswith(":")):
                block.expected = nxt.body
        if f.annotation is not None:
            _parse_annotation(f.annotation, f"{name}:{f.line}", block)
        blocks.append(block)
    return blocks


# ---------------------------------------------------------------------------
# Output comparison
# ---------------------------------------------------------------------------
_NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _unit(token: str) -> float:
    mant, _, exp = token.lower().partition("e")
    decimals = len(mant.split(".", 1)[1]) if "." in mant else 0
    return 10.0 ** (-decimals + (int(exp) if exp else 0))


def output_mismatch(got: str, expected: str) -> str | None:
    """None if ``got`` matches ``expected`` under the documented rules."""
    g = " ".join(got.split())
    e = " ".join(expected.split())
    g_parts, e_parts = _NUM.split(g), _NUM.split(e)
    g_nums, e_nums = _NUM.findall(g), _NUM.findall(e)
    if g_parts != e_parts or len(g_nums) != len(e_nums):
        return f"text differs:\n  got:      {g!r}\n  expected: {e!r}"
    for gn, en in zip(g_nums, e_nums):
        tol = max(_unit(en), _unit(gn), 1e-6 * abs(float(en)))
        if abs(float(gn) - float(en)) > tol * (1 + 1e-9):
            return (f"number {gn} where the doc says {en}:\n"
                    f"  got:      {g!r}\n  expected: {e!r}")
    return None


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
class SnippetFailure(Exception):
    pass


@dataclass
class RunReport:
    ran: int = 0
    skipped_blocks: list[str] = field(default_factory=list)
    skip_rest: str | None = None
    xfailed: list[str] = field(default_factory=list)


def run_markdown(text: str, name: str, filename: str, cwd: Path) -> RunReport:
    """Run the blocks of ``text`` in order; raise SnippetFailure on error.

    ``filename`` is used for tracebacks; when it is a real file the
    traceback shows the markdown source lines.
    """
    try:
        blocks = collect_blocks(text, name)
    except ValueError as exc:
        raise SnippetFailure(str(exc)) from None
    ns: dict = {"__name__": "__main__"}
    report = RunReport()
    old_cwd = os.getcwd()
    os.chdir(cwd)
    try:
        for block in blocks:
            where = f"{name}:{block.line}"
            missing = [m for m in block.requires
                       if importlib.util.find_spec(m) is None]
            if missing:
                report.skip_rest = (f"{where}: requires {', '.join(missing)}, "
                                    f"not installed; this block and the rest "
                                    f"of the file were not run")
                return report
            if block.skip_reason is not None:
                report.skipped_blocks.append(f"{where}: {block.skip_reason}")
                continue
            # Pad so that tracebacks report the markdown line numbers.
            code = "\n" * block.line + block.code
            buf = io.StringIO()
            try:
                compiled = compile(code, filename, "exec")
                with contextlib.redirect_stdout(buf):
                    exec(compiled, ns)
            except Exception as exc:
                if (block.xfail_reason is not None
                        and isinstance(exc, AssertionError)):
                    report.ran += 1
                    report.xfailed.append(f"{where}: {exc}")
                    continue
                tb = "".join(traceback.format_exception(exc))
                raise SnippetFailure(
                    f"{where}: the block raised "
                    f"{type(exc).__name__}: {exc}\n\n{tb}") from None
            finally:
                _close_figures()
            report.ran += 1
            if block.xfail_reason is not None:
                raise SnippetFailure(
                    f"{where}: the block is annotated 'doctest: xfail' "
                    f"({block.xfail_reason}) but ran without error. The "
                    f"defect it documents is fixed: remove the annotation.")
            if block.expected is not None:
                diff = output_mismatch(buf.getvalue(), block.expected)
                if diff:
                    raise SnippetFailure(
                        f"{where}: printed output does not match the "
                        f"expected-output fence that follows it -- {diff}")
    finally:
        os.chdir(old_cwd)
    return report


def _close_figures() -> None:
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        plt.close("all")


@pytest.fixture
def agg_backend(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    mpl = sys.modules.get("matplotlib")
    if mpl is not None:
        mpl.use("Agg", force=True)


@pytest.mark.parametrize(
    "doc", [d for d in DOC_FILES if d.is_file()],
    ids=lambda d: str(d.relative_to(REPO_ROOT)))
def test_doc_snippets_run(doc: Path, tmp_path, agg_backend):
    rel = str(doc.relative_to(REPO_ROOT))
    linecache.checkcache(str(doc))
    t0 = time.perf_counter()
    failure = None
    try:
        with warnings.catch_warnings():
            # Snippets may legitimately warn (that is part of what some
            # of them teach); a warning is not a doc failure.
            warnings.simplefilter("ignore")
            report = run_markdown(doc.read_text(encoding="utf-8"), rel,
                                  str(doc), tmp_path)
    except SnippetFailure as exc:
        failure = str(exc)
    finally:
        _ELAPSED[rel] = time.perf_counter() - t0
    if failure is not None:
        pytest.fail(failure, pytrace=False)
    if report.skip_rest is not None:
        pytest.skip(report.skip_rest)


def test_doc_snippets_runtime_budget():
    """Every executed doc together must stay cheap enough for every CI leg."""
    if not _ELAPSED:
        pytest.skip("no doc snippets ran in this session")
    total = sum(_ELAPSED.values())
    detail = ", ".join(f"{k} {v:.1f}s" for k, v in
                       sorted(_ELAPSED.items(), key=lambda kv: -kv[1]))
    assert total <= BUDGET_FAIL_S, (
        f"doc snippets took {total:.1f}s (> {BUDGET_FAIL_S:.0f}s): {detail}")
    if total > BUDGET_WARN_S:
        warnings.warn(f"doc snippets took {total:.1f}s "
                      f"(> {BUDGET_WARN_S:.0f}s): {detail}", stacklevel=1)


# ---------------------------------------------------------------------------
# Self-tests of the runner
# ---------------------------------------------------------------------------
def _run(md: str, tmp_path: Path) -> RunReport:
    return run_markdown(md, "fixture.md", "fixture.md", tmp_path)


FENCE = "```"


def test_runner_reports_the_line_of_a_broken_call(tmp_path):
    md = (f"# T\n\n{FENCE}python\nimport pulsim as p\nb = p.CircuitBuilder()\n"
          f"{FENCE}\n\ntext\n\n{FENCE}python\n"
          f'b.add_resistor("R", "a", "gnd", resistance=1.0)\n{FENCE}\n')
    with pytest.raises(SnippetFailure, match=r"fixture\.md:10: .*TypeError"):
        _run(md, tmp_path)


def test_runner_shares_one_namespace_and_skips_other_fences(tmp_path):
    md = (f"{FENCE}python\nx = 2\n{FENCE}\n\n{FENCE}bash\nexit 1\n{FENCE}\n\n"
          f"{FENCE}\nnot python\n{FENCE}\n\n> {FENCE}python\n> raise SystemExit\n"
          f"> {FENCE}\n\n{FENCE}python\nassert x == 2\n{FENCE}\n")
    assert _run(md, tmp_path).ran == 2


def test_runner_catches_a_stale_expected_value(tmp_path):
    md = (f'{FENCE}python\nprint(f"v = {{3.151:.3f}} V")\n{FENCE}\n\n'
          f"Expected output:\n{FENCE}\nv = 3.161 V\n{FENCE}\n")
    with pytest.raises(SnippetFailure, match=r"3\.151 where the doc says 3\.161"):
        _run(md, tmp_path)


def test_runner_accepts_a_last_digit_difference(tmp_path):
    md = (f"{FENCE}python\nprint('6.320 V at', 12)\n{FENCE}\n\n"
          f"{FENCE}text\n6.319 V  at 12\n{FENCE}\n")
    assert _run(md, tmp_path).ran == 1
    assert output_mismatch("6.321", "6.319") is not None
    assert output_mismatch("pwl", "trbdf2") is not None


def test_runner_ignores_a_fence_after_more_than_one_prose_line(tmp_path):
    md = (f"{FENCE}python\nprint(1)\n{FENCE}\n\nSome prose.\nMore prose:\n\n"
          f"{FENCE}\n2\n{FENCE}\n")
    assert _run(md, tmp_path).ran == 1


def test_runner_requires_a_reason_to_skip(tmp_path):
    md = f"<!-- doctest: skip -->\n{FENCE}python\nprint(1)\n{FENCE}\n"
    with pytest.raises(SnippetFailure, match="needs a non-empty reason"):
        _run(md, tmp_path)
    md = f'<!-- doctest: skip reason="fragment" -->\n{FENCE}python\nraise\n{FENCE}\n'
    assert _run(md, tmp_path).skipped_blocks == ["fixture.md:2: fragment"]


def test_runner_rejects_an_unknown_annotation(tmp_path):
    md = f"<!-- doctest: sometimes -->\n{FENCE}python\nprint(1)\n{FENCE}\n"
    with pytest.raises(SnippetFailure, match="unknown annotation"):
        _run(md, tmp_path)


def test_runner_skips_the_rest_of_the_file_for_a_missing_module(tmp_path):
    md = (f"{FENCE}python\nx = 1\n{FENCE}\n\n"
          f"<!-- doctest: requires=no_such_module_xyz -->\n{FENCE}python\n"
          f"import no_such_module_xyz\n{FENCE}\n\n{FENCE}python\nraise\n{FENCE}\n")
    report = _run(md, tmp_path)
    assert report.ran == 1
    assert "no_such_module_xyz" in report.skip_rest


def test_runner_xfail_must_fail_and_then_continues(tmp_path):
    md = (f'<!-- doctest: xfail reason="known defect" -->\n{FENCE}python\n'
          f"assert False\n{FENCE}\n\n{FENCE}python\nprint('after')\n{FENCE}\n")
    report = _run(md, tmp_path)
    assert report.ran == 2 and len(report.xfailed) == 1
    md = (f'<!-- doctest: xfail reason="known defect" -->\n{FENCE}python\n'
          f"undefined_name\n{FENCE}\n")
    with pytest.raises(SnippetFailure, match="NameError"):
        _run(md, tmp_path)
    md = (f'<!-- doctest: xfail reason="known defect" -->\n{FENCE}python\n'
          f"assert True\n{FENCE}\n")
    with pytest.raises(SnippetFailure, match="remove the annotation"):
        _run(md, tmp_path)


def test_runner_uses_a_scratch_working_directory(tmp_path):
    md = f"{FENCE}python\nimport os\nopen('out.txt', 'w').write('x')\n{FENCE}\n"
    _run(md, tmp_path)
    assert (tmp_path / "out.txt").is_file()
