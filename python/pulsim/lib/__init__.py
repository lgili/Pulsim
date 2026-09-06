"""``pulsim.lib`` — the parts library (audit C.5).

    >>> q = pulsim.lib.mosfet("C3M0065090J")
    >>> spec = q.switch_spec(Tj=125.0)      # for losses.device_loss_summary
    >>> part = pulsim.lib.import_plecs_xml("~/Downloads/IKW40N120T2.xml")

Three things this exists to fix, all measured before it was written:

* **Six parts, re-typed by hand.** The shipped device YAMLs under
  ``devices/`` were consumed by nothing in the package — a user
  wanting a MOSFET's switching energies transcribed the same dict by
  hand three calls in a row. They are package data now
  (``pulsim/lib/data``) behind one lookup — and, on inspection, they
  are SYNTHETIC: every switching-energy table is a straight line
  through the origin, and four of six headers name the wrong package.
  They stay as illustrations, marked ``provenance.method: synthetic``,
  and :func:`part` refuses to resolve them by number unless asked
  (``allow_synthetic=True``): a lookup of a real part number must not
  succeed and lie.
* **Provenance.** A digitised datasheet curve is the manufacturer's
  copyright, and a number with no origin cannot be checked. Every
  part file carries a ``provenance`` block naming where each number
  came from and how; a part without one is refused by name.
* **The PLECS route.** Infineon, Wolfspeed, onsemi, ROHM and others
  publish PLECS thermal description files (``*.xml``) for thousands
  of devices — switching-energy tables on (I, V, Tj) grids, on-state
  voltage tables, Foster/Cauer thermal chains. :func:`import_plecs_xml`
  reads the file the user downloaded. Nothing is redistributed, and
  the tables are the vendor's own rather than a transcription.

The SPICE importer's silent ``.MODEL`` discard is the other half of
C.5 and lives in :mod:`pulsim.spice_import`.
"""

from __future__ import annotations

from .parts import (
    Part,
    PartNotFound,
    ProvenanceError,
    Provenance,
    ThermalChain,
    core,
    diode,
    igbt,
    list_parts,
    load_part_file,
    mosfet,
    part,
    part_from_mapping,
    save_part,
    search,
    search_paths,
)
from .plecs_xml import PlecsImportError, import_plecs_xml

__all__ = [
    "Part",
    "PartNotFound",
    "ProvenanceError",
    "Provenance",
    "ThermalChain",
    "PlecsImportError",
    "core",
    "diode",
    "igbt",
    "import_plecs_xml",
    "list_parts",
    "load_part_file",
    "mosfet",
    "part",
    "part_from_mapping",
    "save_part",
    "search",
    "search_paths",
]
