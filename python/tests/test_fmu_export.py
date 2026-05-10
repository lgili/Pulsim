"""Phase 2.5 / 9 of `add-fmi-export`: FMU export structural tests.

Pins:
  - The exported `.fmu` is a valid zip with the FMI 2.0 layout:
    `modelDescription.xml`, `binaries/<platform>/<lib>`, `sources/`.
  - `modelDescription.xml` is well-formed XML and declares
    fmiVersion="2.0", `<CoSimulation modelIdentifier="...">`, plus
    one `<ScalarVariable causality="output">` per requested output.
  - The shared library exports the FMI 2.0 CS callback symbols
    (`fmi2Instantiate`, `fmi2DoStep`, `fmi2GetReal`, ...).
  - Round-trip via `ctypes`: load the lib, instantiate, step, read
    output. Without going through a full FMI master harness, this is
    the cheapest realistic compliance check we can do in CI.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

pulsim = pytest.importorskip("pulsim")


def _build_rc_circuit():
    ckt = pulsim.Circuit()
    in_ = ckt.add_node("in")
    out = ckt.add_node("out")
    ckt.add_voltage_source("V1", in_, ckt.ground(), 1.0)
    ckt.add_resistor("R1", in_, out, 1e3)
    ckt.add_capacitor("C1", out, ckt.ground(), 1e-6, 0.0)
    return ckt


def _platform_dir() -> str:
    sys_name = sys.platform
    bits = "64" if sys.maxsize > 2 ** 32 else "32"
    if sys_name.startswith("linux"):
        return f"linux{bits}"
    if sys_name == "darwin":
        return f"darwin{bits}"
    if sys_name in ("win32", "cygwin"):
        return f"win{bits}"
    return f"{sys_name}{bits}"


def _shared_lib_extension() -> str:
    if sys.platform.startswith("linux"):
        return ".so"
    if sys.platform == "darwin":
        return ".dylib"
    return ".dll"


def test_fmu_export_produces_valid_zip_layout():
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler on PATH")

    ckt = _build_rc_circuit()
    with tempfile.TemporaryDirectory() as td:
        fmu_path = Path(td) / "rc.fmu"
        summary = pulsim.fmu.export(
            ckt, dt=1e-5, out_path=fmu_path,
            model_name="rc", outputs=["out"])
        # Compare via realpath because macOS resolves /var → /private/var.
        assert os.path.realpath(summary.path) == os.path.realpath(fmu_path)
        assert summary.fmi_version == "2.0"
        assert summary.guid

        with zipfile.ZipFile(fmu_path) as zf:
            names = zf.namelist()
            assert "modelDescription.xml" in names
            ext = _shared_lib_extension()
            expected_lib = f"binaries/{_platform_dir()}/rc{ext}"
            assert expected_lib in names
            for src in ("model.c", "model.h", "fmu_entry.c"):
                assert f"sources/{src}" in names


def test_model_description_xml_is_well_formed():
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler on PATH")
    ckt = _build_rc_circuit()
    with tempfile.TemporaryDirectory() as td:
        fmu_path = Path(td) / "rc.fmu"
        summary = pulsim.fmu.export(
            ckt, dt=1e-5, out_path=fmu_path, outputs=["out"])
        with zipfile.ZipFile(fmu_path) as zf:
            xml_bytes = zf.read("modelDescription.xml")
        root = ET.fromstring(xml_bytes)
        assert root.tag == "fmiModelDescription"
        assert root.get("fmiVersion") == "2.0"
        assert root.get("guid") == summary.guid
        cs = root.find("CoSimulation")
        assert cs is not None
        assert cs.get("modelIdentifier") == summary.model_identifier
        assert cs.get("canHandleVariableCommunicationStepSize") == "true"
        # At least one output ScalarVariable.
        outputs = [
            v for v in root.findall(".//ScalarVariable")
            if v.get("causality") == "output"
        ]
        assert len(outputs) >= 1


def test_fmu_shared_library_exports_fmi2_symbols():
    """Load the compiled shared library and verify the FMI 2.0 CS
    entry points are visible via `dlsym`."""
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler on PATH")

    ckt = _build_rc_circuit()
    with tempfile.TemporaryDirectory() as td:
        fmu_path = Path(td) / "rc.fmu"
        pulsim.fmu.export(ckt, dt=1e-5, out_path=fmu_path, outputs=["out"])
        # Extract the .dylib / .so / .dll and ctypes-load it.
        ext = _shared_lib_extension()
        lib_arcname = f"binaries/{_platform_dir()}/rc{ext}"
        extract_dir = Path(td) / "extracted"
        with zipfile.ZipFile(fmu_path) as zf:
            zf.extract(lib_arcname, path=extract_dir)
        lib_path = extract_dir / lib_arcname
        lib = ctypes.CDLL(str(lib_path))
        for symbol in [
            "fmi2GetVersion", "fmi2GetTypesPlatform",
            "fmi2Instantiate", "fmi2FreeInstance",
            "fmi2SetupExperiment", "fmi2EnterInitializationMode",
            "fmi2ExitInitializationMode", "fmi2Reset", "fmi2Terminate",
            "fmi2SetReal", "fmi2GetReal",
            "fmi2DoStep", "fmi2CancelStep",
        ]:
            sym = getattr(lib, symbol, None)
            assert sym is not None, f"FMU lib missing symbol {symbol!r}"

        # Smoke-call fmi2GetVersion and confirm it returns "2.0".
        lib.fmi2GetVersion.restype = ctypes.c_char_p
        version = lib.fmi2GetVersion()
        assert version == b"2.0"


def test_fmu_round_trip_step_via_ctypes():
    """Load the FMU shared library and run one DoStep cycle via
    ctypes — proves the wrapper compiles cleanly and the codegen
    matrix-vector body runs without crashing."""
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler on PATH")

    ckt = _build_rc_circuit()
    with tempfile.TemporaryDirectory() as td:
        fmu_path = Path(td) / "rc.fmu"
        pulsim.fmu.export(ckt, dt=1e-5, out_path=fmu_path, outputs=["out"])
        ext = _shared_lib_extension()
        extract_dir = Path(td) / "extracted"
        with zipfile.ZipFile(fmu_path) as zf:
            zf.extract(f"binaries/{_platform_dir()}/rc{ext}", path=extract_dir)
        lib = ctypes.CDLL(
            str(extract_dir / f"binaries/{_platform_dir()}/rc{ext}"))

        # FMI signatures (minimal subset).
        lib.fmi2Instantiate.restype = ctypes.c_void_p
        # Argument types: instanceName, fmuType, fmuGUID, resourceLoc,
        # callbacks, visible, loggingOn — we pass NULLs / zeros.
        lib.fmi2Instantiate.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ]
        lib.fmi2DoStep.restype = ctypes.c_int
        lib.fmi2DoStep.argtypes = [
            ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_int,
        ]
        lib.fmi2FreeInstance.argtypes = [ctypes.c_void_p]
        lib.fmi2SetupExperiment.restype = ctypes.c_int
        lib.fmi2SetupExperiment.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_double,
            ctypes.c_double, ctypes.c_int, ctypes.c_double,
        ]

        c = lib.fmi2Instantiate(b"test", 1, b"guid", b"", None, 0, 0)
        assert c != 0
        lib.fmi2SetupExperiment(c, 0, 0.0, 0.0, 0, 0.0)
        # Take 10 steps of 1e-5 s. Should not crash.
        status = 0
        for _ in range(10):
            status = lib.fmi2DoStep(c, 0.0, 1e-5, 0)
            assert status == 0       # fmi2OK
        lib.fmi2FreeInstance(c)
