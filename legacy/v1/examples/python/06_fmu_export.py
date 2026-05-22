"""FMI 2.0 Co-Simulation FMU export.

Wraps the codegen-emitted state space inside an FMI 2.0 CS callback layer
(`fmi2Instantiate`, `fmi2DoStep`, ...) and zips the bundle into a `.fmu`
file. Drop the resulting `.fmu` into any FMI master that speaks 2.0
(OMSimulator, FMPy, Dymola, Simulink Co-Simulation block, ANSYS Twin
Builder).

This script:
  1. Builds an RC circuit and exports it as ``rc.fmu``.
  2. Inspects the resulting zip layout (`modelDescription.xml`,
     `binaries/<platform>/`, `sources/`).
  3. Loads the shared library via ``ctypes`` and confirms the FMI 2.0 CS
     callback symbols are exported.

Run::

    python 06_fmu_export.py

See also: docs/fmi-export.md
"""

from __future__ import annotations

import ctypes
import shutil
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pulsim


FMI2_CS_SYMBOLS = (
    "fmi2GetVersion",
    "fmi2GetTypesPlatform",
    "fmi2Instantiate",
    "fmi2FreeInstance",
    "fmi2Reset",
    "fmi2Terminate",
    "fmi2SetupExperiment",
    "fmi2EnterInitializationMode",
    "fmi2ExitInitializationMode",
    "fmi2SetReal",
    "fmi2GetReal",
    "fmi2DoStep",
    "fmi2CancelStep",
)


def _platform_dir() -> str:
    bits = "64" if sys.maxsize > 2 ** 32 else "32"
    if sys.platform.startswith("linux"):
        return f"linux{bits}"
    if sys.platform == "darwin":
        return f"darwin{bits}"
    if sys.platform in ("win32", "cygwin"):
        return f"win{bits}"
    return f"{sys.platform}{bits}"


def _shared_lib_extension() -> str:
    if sys.platform.startswith("linux"):
        return ".so"
    if sys.platform == "darwin":
        return ".dylib"
    return ".dll"


def main() -> int:
    if shutil.which("cc") is None and shutil.which("gcc") is None:
        print("no gcc/cc on PATH — FMU export needs a C compiler. "
              "Install Xcode CLT or gcc and re-run.")
        return 0

    ckt = pulsim.Circuit()
    in_ = ckt.add_node("in")
    out = ckt.add_node("out")
    ckt.add_voltage_source("V1", in_, ckt.ground(), 1.0)
    ckt.add_resistor("R1", in_, out, 1e3)
    ckt.add_capacitor("C1", out, ckt.ground(), 1e-6, 0.0)

    work = Path(tempfile.mkdtemp(prefix="pulsim_fmu_"))
    fmu_path = work / "rc.fmu"
    summary = pulsim.fmu.export(
        ckt,
        dt=1e-5,
        out_path=fmu_path,
        model_name="rc",
        outputs=["out"],
    )
    print(f"Exported FMU:")
    print(f"  path:    {summary.path}")
    print(f"  GUID:    {summary.guid}")
    print(f"  FMI ver: {summary.fmi_version}")
    print(f"  states:  {summary.state_size}    "
          f"inputs: {summary.input_size}    "
          f"outputs: {summary.output_size}")

    # ----- inspect zip layout -----
    print("\nArchive contents:")
    with zipfile.ZipFile(fmu_path) as zf:
        names = zf.namelist()
        for n in sorted(names):
            print(f"  {n}")
        ext = _shared_lib_extension()
        expected_lib = f"binaries/{_platform_dir()}/rc{ext}"
        assert expected_lib in names, f"missing {expected_lib}"
        for src in ("model.c", "model.h", "fmu_entry.c"):
            assert f"sources/{src}" in names

        # ----- modelDescription.xml -----
        xml_bytes = zf.read("modelDescription.xml")
        zf.extract(expected_lib, path=work / "extracted")
    root = ET.fromstring(xml_bytes)
    cs = root.find("CoSimulation")
    print(f"\nmodelDescription.xml CoSimulation block:")
    print(f"  modelIdentifier  = {cs.attrib['modelIdentifier']!r}")
    print(f"  canHandleVarH    = {cs.attrib.get('canHandleVariableCommunicationStepSize')}")
    print(f"  canBeInstantiatedOnlyOncePerProcess = "
          f"{cs.attrib.get('canBeInstantiatedOnlyOncePerProcess')}")

    # ----- ctypes round-trip: load the lib, confirm FMI symbols -----
    extracted_lib = work / "extracted" / expected_lib
    print(f"\nLoading shared library: {extracted_lib}")
    lib = ctypes.CDLL(str(extracted_lib))
    missing = [s for s in FMI2_CS_SYMBOLS if not hasattr(lib, s)]
    print(f"  FMI 2.0 CS symbols exported: "
          f"{len(FMI2_CS_SYMBOLS) - len(missing)} / {len(FMI2_CS_SYMBOLS)}")
    if missing:
        print(f"  ! missing: {missing}")
        return 2
    print("  ✓ all required FMI 2.0 CS callbacks exported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
