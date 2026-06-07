"""Phase 2/3 tests for the C block: compiled C/C++ shared libraries
(``lib=``, via ctypes) and inline auto-compiled source (``code=``).

Skipped when no C compiler is available."""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

import pulsim as p
from pulsim.c_block import _find_compiler

pytestmark = pytest.mark.skipif(
    _find_compiler("c") is None, reason="no C compiler available")


def _buck_in_out(vin=2.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", vin)
    b.add_resistor("Rin", "in", "gnd", 1e3)
    b.add_resistor("Rout", "out", "gnd", 1e3)
    return b


def _v_out(res):
    return float(np.asarray(res.v("out"), dtype=float)[-1])


def test_inline_c_matches_python():
    """Inline C gain block gives the same V(out)=3·V(in)=6 as Python."""
    b = _buck_in_out()
    p.add_c_block(b, inputs=[("v", "in")], outputs=[("v", "out", "gnd")],
                  dt=1e-4, code="out[0] = 3.0 * in[0];", lang="c")
    res = p.simulate(b, t_end=2e-3, dt=2e-6)
    assert abs(_v_out(res) - 6.0) < 1e-6


def test_inline_cpp_block():
    if _find_compiler("cpp") is None:
        pytest.skip("no C++ compiler available")
    b = _buck_in_out()
    p.add_c_block(b, inputs=[("v", "in")], outputs=[("v", "out", "gnd")],
                  dt=1e-4, code="out[0] = std::sqrt(in[0]*in[0]) * 3.0;",
                  lang="cpp")
    res = p.simulate(b, t_end=2e-3, dt=2e-6)
    assert abs(_v_out(res) - 6.0) < 1e-6


def test_lib_path_loads_compiled_so(tmp_path):
    """Compile a .so out-of-band and drive the block via ``lib=``."""
    comp = _find_compiler("c")
    src = tmp_path / "gain.c"
    src.write_text(
        "void pulsim_cblock_step(const double* in, int n_in, double* out,\n"
        "        int n_out, double t, double dt, void** state) {\n"
        "    out[0] = 3.0 * in[0];\n"
        "}\n")
    so = tmp_path / "gain.so"
    subprocess.run([comp, "-shared", "-fPIC", "-O2", str(src),
                    "-o", str(so)], check=True, capture_output=True, text=True)

    b = _buck_in_out()
    p.add_c_block(b, inputs=[("v", "in")], outputs=[("v", "out", "gnd")],
                  dt=1e-4, lib=str(so))
    res = p.simulate(b, t_end=2e-3, dt=2e-6)
    assert abs(_v_out(res) - 6.0) < 1e-6


def test_native_state_init_term(tmp_path):
    """init/term + opaque *state: a C integrator accumulates across steps."""
    comp = _find_compiler("c")
    src = tmp_path / "integ.c"
    src.write_text(
        "#include <stdlib.h>\n"
        "void* pulsim_cblock_init(int n_in, int n_out) {\n"
        "    return calloc(1, sizeof(double)); }\n"
        "void pulsim_cblock_term(void* s) { free(s); }\n"
        "void pulsim_cblock_step(const double* in, int n_in, double* out,\n"
        "        int n_out, double t, double dt, void** state) {\n"
        "    double* acc = (double*)(*state);\n"
        "    *acc += in[0] * dt;\n"
        "    out[0] = *acc;\n"
        "}\n")
    so = tmp_path / "integ.so"
    subprocess.run([comp, "-shared", "-fPIC", "-O2", str(src),
                    "-o", str(so)], check=True, capture_output=True, text=True)

    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 1.0)
    b.add_resistor("Rin", "in", "gnd", 1e3)
    b.add_resistor("Rout", "out", "gnd", 1e3)
    h = p.add_c_block(b, inputs=[("v", "in")],
                      outputs=[("v", "out", "gnd")],
                      dt=1e-4, lib=str(so))
    t_end = 5e-3
    res = p.simulate(b, t_end=t_end, dt=2e-6)
    # acc ≈ ∫1·dt = t_end = 5e-3 (held at the block rate).
    assert abs(_v_out(res) - t_end) < 2e-4
    assert h.teardown is not None


def test_inline_cache_reuse():
    """Identical inline source compiles once and is reused from cache."""
    from pulsim.c_block import _compile_inline
    so1 = _compile_inline("out[0] = in[0];", "c", "pulsim_cblock_step")
    so2 = _compile_inline("out[0] = in[0];", "c", "pulsim_cblock_step")
    assert so1 == so2 and so1.exists()


def test_exactly_one_source_required():
    b = _buck_in_out()
    with pytest.raises(ValueError, match="exactly one"):
        p.add_c_block(b, inputs=[], outputs=[("v", "out", "gnd")], dt=1e-4)


def test_yaml_c_blocks_list_of_dicts():
    """wire_c_blocks_from_yaml from a Python list (no PyYAML needed)."""
    b = _buck_in_out()
    hs = p.wire_c_blocks_from_yaml(b, [{
        "inputs": [["v", "in"]], "outputs": [["v", "out", "gnd"]],
        "dt": 1e-4, "lang": "c", "code": "out[0] = 3.0*in[0];"}])
    assert len(hs) == 1
    res = p.simulate(b, t_end=2e-3, dt=2e-6)
    assert abs(_v_out(res) - 6.0) < 1e-6


def test_yaml_c_blocks_yaml_string():
    """wire_c_blocks_from_yaml from a YAML string (needs PyYAML)."""
    pytest.importorskip("yaml")
    b = _buck_in_out()
    spec = """
- inputs:  [["v", "in"]]
  outputs: [["v", "out", "gnd"]]
  dt: 1.0e-4
  lang: c
  code: "out[0] = 3.0 * in[0];"
"""
    p.wire_c_blocks_from_yaml(b, spec)
    res = p.simulate(b, t_end=2e-3, dt=2e-6)
    assert abs(_v_out(res) - 6.0) < 1e-6


def test_yaml_c_block_unknown_field_rejected():
    b = _buck_in_out()
    with pytest.raises(ValueError, match="unknown field"):
        p.wire_c_blocks_from_yaml(b, [{
            "inputs": [], "outputs": [("v", "out", "gnd")], "dt": 1e-4,
            "code": "out[0]=1;", "lang": "c", "bogus": 42}])
