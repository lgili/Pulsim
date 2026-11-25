"""`near` and `far` are macros in the Windows headers (they expand to nothing),
so a C++ variable with either name compiles everywhere but under MSVC. This
check fails on every platform instead of only in the Windows CI jobs."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_CODE = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"', re.S)
_NAME = re.compile(r"\b(near|far)\b")


def _hits(text):
    code = _CODE.sub(lambda m: " " * len(m.group()), text)   # keep offsets
    return [(code.count("\n", 0, m.start()) + 1, m.group()) for m in _NAME.finditer(code)]


def test_the_check_sees_a_declaration_and_skips_comments_and_strings():
    assert _hits('const bool far = a ? b : c;  // far away\nf("so far");') == [(1, "far")]


def test_no_cpp_identifier_is_a_windows_macro():
    # what the extension's translation units include (they pull in <windows.h>
    # through Python.h); the C++ test binaries are not built with those headers
    files = [*ROOT.glob("core/include/**/*.h*"), *ROOT.glob("python/*.[ch]pp")]
    assert files
    bad = [f"{f.relative_to(ROOT)}:{line}: {name}"
           for f in files for line, name in _hits(f.read_text(encoding="utf-8"))]
    assert not bad, bad
