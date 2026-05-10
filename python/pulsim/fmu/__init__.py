"""FMI 2.0 Co-Simulation FMU export.

`add-fmi-export` Phase 1+2: takes a Pulsim Circuit + a fixed step dt
and produces a self-contained `.fmu` file (zip of `modelDescription.xml`
+ shared library + sources) that runs in any FMI 2.0 CS-compliant
master (OMSimulator, Dymola, Simulink, FMPy).

Usage::

    import pulsim
    out = pulsim.fmu.export(
        ckt,
        dt=1e-6,
        out_path="gen/buck.fmu",
        model_name="buck",
        inputs=[],          # external inputs (FMI value refs); empty
                            # for a self-contained slave
        outputs=["vout"],   # named circuit nodes to expose
    )
    print(out.path)

The exported FMU is co-simulation only (Phase 3 Model-Exchange is
deferred); each `fmi2DoStep` call advances the internal state by the
master-supplied step size by integer multiples of the codegen-fixed
internal dt. The internal solver is the discrete state-space machinery
shipped in `add-realtime-code-generation`.
"""

from .exporter import (
    FmuExportSummary,
    export,
)

__all__ = [
    "FmuExportSummary",
    "export",
]
