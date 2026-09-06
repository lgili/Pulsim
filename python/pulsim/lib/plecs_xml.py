"""Import a PLECS thermal description file (``*.xml``) as a Part.

Infineon, Wolfspeed, onsemi, ROHM and others publish these for their
devices; PLECS's own editor writes them. The file the user downloaded
is read in place — nothing is redistributed — and its tables are the
vendor's, not a transcription.

What the format carries and how it is mapped (the PLECS manual,
ch. 4 "Thermal Modeling", documents the semantics, units and sign
conventions; the element names are those of the files in the wild):

* ``<Package class= vendor= partnumber= version=>`` — identity. The
  class is one of ``Diode``, ``IGBT``, ``MOSFET``, ``IGBT with
  Diode``, ``MOSFET with Diode``; anything else is refused by name
  (a thyristor's or a package description's tables are not a
  switch's).
* ``<SemiconductorData type=>`` with ``<TurnOnLoss>``, ``<TurnOffLoss>``
  (energy on current × voltage × temperature) and ``<ConductionLoss>``
  (on-state voltage drop on current × temperature). Each carries a
  ``<ComputationMethod>``: a lookup table, a formula, or a table
  modified by a formula.
* ``<ThermalModel>`` with one ``<Branch type="Foster"|"Cauer">`` of
  ``<RTauElement R= Tau=>`` or ``<RCElement R= C=>`` stages.
* ``<Variables>`` — the intrinsic v, i, T with the vendor's Min/Max
  limits (carried as ``Part.limits``), plus custom variables that
  formulas refer to.

**Sign conventions** (manual p. 199–200). A switch's losses sit in the
positive-voltage, positive-current quadrant. A Diode's switching
losses are specified in the NEGATIVE-voltage, positive-current
quadrant; a "... with Diode" description folds the diode into the
same tables as the switch — its switching losses in the
positive-voltage, NEGATIVE-current quadrant and its conduction drop
at negative current. Handing such axes to a lookup that takes
|V| and |I| returns 0 J for every diode recovery event with no error,
so the importer selects the quadrant by class and folds the axes to
their magnitudes; the Part then holds ``Erec`` / ``v_on_diode`` for
the diode half of a "with Diode" file.

What is REFUSED by name, because reading it wrong would be silent:

* a computation method that is not a plain lookup table — a formula
  is an expression PLECS evaluates at simulation time (gate
  resistance scaling, for instance), and dropping it would ship the
  vendor's table with the vendor's correction missing. Pass
  ``ignore_formula=True`` to take the table anyway — unless the
  formula refers to a declared variable or a custom table, in which
  case the correction depends on a knob the file expects you to set
  and the table is refused regardless; the provenance note records
  any formula that was dropped;
* a table whose value count does not match its axes, a non-monotone
  or duplicated axis (a descending axis is accepted and reversed);
* a class/type mismatch, more than one ``<SemiconductorData>`` (a
  package description with several chips), more than one conduction
  definition (gate-dependent files) unless ``conduction_index`` picks
  one, more than one thermal branch, a thermal element that does not
  match its branch type, or anything else in ``<ThermalModel>``.
"""

from __future__ import annotations

import datetime as _dt
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..loss_tables import LossTable
from .parts import Part, Provenance, ThermalChain


class PlecsImportError(ValueError):
    """The file could not be read without guessing."""


_TABLE_METHODS = {"table only", "lookup table", "table"}
_FORMULA_METHODS = {"formula only", "formula"}
_MIXED_METHODS = {"table and formula", "lookup table and formula"}
# PLECS class string (lower-cased) -> (pulsim class, carries a diode half)
_CLASS_MAP = {
    "diode": ("diode", False),
    "igbt": ("igbt", False),
    "mosfet": ("mosfet", False),
    "igbt with diode": ("igbt", True),
    "mosfet with diode": ("mosfet", True),
}


def _floats(text: Optional[str], what: str) -> np.ndarray:
    if text is None or not text.strip():
        raise PlecsImportError(f"{what}: empty")
    toks = text.replace(",", " ").split()
    try:
        arr = np.asarray([float(t) for t in toks], dtype=float)
    except ValueError as e:
        raise PlecsImportError(f"{what}: non-numeric value in {text[:60]!r}") from e
    if not np.all(np.isfinite(arr)):
        raise PlecsImportError(f"{what}: non-finite value.")
    return arr


def _axis(el: ET.Element, tag: str, what: str) -> Tuple[np.ndarray, np.ndarray]:
    """An axis as an ascending array plus the permutation that sorts
    the file's order into it. Descending is accepted (and reversed);
    duplicates or a non-monotone axis are refused."""
    ax = el.find(tag)
    if ax is None:
        raise PlecsImportError(f"{what}: no <{tag}>.")
    raw = _floats(ax.text, f"{what}/{tag}")
    if raw.size > 1:
        d = np.diff(raw)
        if np.any(d == 0):
            raise PlecsImportError(f"{what}/{tag}: duplicated value in {raw.tolist()}.")
        if not (np.all(d > 0) or np.all(d < 0)):
            raise PlecsImportError(f"{what}/{tag}: not monotone: {raw.tolist()}.")
    order = np.argsort(raw, kind="stable")
    return raw[order], order


def _method(el: ET.Element, what: str, ignore_formula: bool, notes: List[str],
            variables: Dict[str, Dict[str, Any]]) -> None:
    m = el.find("ComputationMethod")
    method = (m.text or "").strip().lower() if m is not None else "table only"
    if method in _TABLE_METHODS:
        return
    formula = el.find("Formula")
    ftext = (formula.text or "").strip() if formula is not None else "(none given)"
    if method in _FORMULA_METHODS:
        raise PlecsImportError(
            f"{what}: ComputationMethod is {method!r} — the loss is an "
            f"expression ({ftext!r}) that PLECS evaluates at run time, with "
            "no table to fall back on. This importer does not evaluate "
            "formulas; use the vendor's table-only file or transcribe the "
            "values with provenance.")
    if method in _MIXED_METHODS:
        if not ignore_formula:
            raise PlecsImportError(
                f"{what}: ComputationMethod is {method!r}: the table is "
                f"modified at run time by the formula {ftext!r} (gate-"
                "resistance scaling, typically). Taking the table alone "
                "would ship the vendor's numbers with the vendor's "
                "correction missing. Pass ignore_formula=True to take the "
                "raw table; the Part's provenance note will record the "
                "formula that was dropped.")
        used = sorted(n for n in variables
                      if n not in ("v", "i", "T") and re.search(rf"\b{re.escape(n)}\b", ftext))
        if used or "lookup(" in ftext.replace(" ", ""):
            defaults = {n: variables[n].get("default") for n in used}
            raise PlecsImportError(
                f"{what}: the formula {ftext!r} depends on "
                + (f"the declared variables {defaults} (name: default)" if used else "a custom table")
                + ". The table alone is the vendor's number at whatever setting "
                "the table was taken at, and the correction the file expects "
                "you to apply cannot be dropped knowingly — ignore_formula "
                "does not cover it. Use a table-only file, or transcribe the "
                "values at your setting with provenance.")
        notes.append(f"{what}: formula {ftext!r} ignored (ignore_formula=True)")
        return
    raise PlecsImportError(
        f"{what}: unknown ComputationMethod {method!r}; this importer knows "
        "'Table only', 'Formula only' and 'Table and formula'.")


def _energy_raw(el: ET.Element, what: str, ignore_formula: bool, notes: List[str],
                variables: Dict[str, Dict[str, Any]]):
    """The signed (v, i, tj, E[v, i, tj]) grid as the file states it,
    axes sorted ascending."""
    _method(el, what, ignore_formula, notes, variables)
    i, i_ord = _axis(el, "CurrentAxis", what)
    v, v_ord = _axis(el, "VoltageAxis", what)
    tj, t_ord = _axis(el, "TemperatureAxis", what)
    en = el.find("Energy")
    if en is None:
        raise PlecsImportError(f"{what}: no <Energy> element.")
    try:
        scale = float(en.get("scale", "1"))
    except ValueError as e:
        raise PlecsImportError(f"{what}: Energy scale={en.get('scale')!r} is not a number.") from e
    for child in en:
        if child.tag != "Temperature":
            raise PlecsImportError(
                f"{what}: <Energy> holds a <{child.tag}> element; expected only "
                "<Temperature> blocks (one per TemperatureAxis entry).")
    temps = en.findall("Temperature")
    if len(temps) != tj.size:
        raise PlecsImportError(
            f"{what}: {len(temps)} <Temperature> blocks but the TemperatureAxis "
            f"has {tj.size} entries.")
    energy = np.zeros((v.size, i.size, tj.size))
    for kt_file, tblock in enumerate(temps):
        kt = int(np.where(t_ord == kt_file)[0][0])
        for child in tblock:
            if child.tag != "Voltage":
                raise PlecsImportError(
                    f"{what}: Temperature[{kt_file}] holds a <{child.tag}> "
                    "element; expected only <Voltage> rows.")
        rows = tblock.findall("Voltage")
        if len(rows) != v.size:
            raise PlecsImportError(
                f"{what}: temperature block {kt_file} has {len(rows)} <Voltage> "
                f"rows but the VoltageAxis has {v.size} entries.")
        for kv_file, vrow in enumerate(rows):
            kv = int(np.where(v_ord == kv_file)[0][0])
            vals = _floats(vrow.text, f"{what}/Temperature[{kt_file}]/Voltage[{kv_file}]")
            if vals.size != i.size:
                raise PlecsImportError(
                    f"{what}: Temperature[{kt_file}]/Voltage[{kv_file}] has "
                    f"{vals.size} values but the CurrentAxis has {i.size} entries.")
            energy[kv, :, kt] = vals[i_ord] * scale
    if np.any(energy < 0):
        raise PlecsImportError(f"{what}: negative energy in the table.")
    return v, i, tj, energy


def _quadrant(v, i, tj, energy, v_sign: int, i_sign: int, what: str) -> LossTable:
    """The (v_sign, i_sign) quadrant of a signed table, axes folded to
    their magnitudes and sorted ascending. Zero belongs to both halves."""
    vm = (v >= 0) if v_sign > 0 else (v <= 0)
    im = (i >= 0) if i_sign > 0 else (i <= 0)
    if not vm.any() or not im.any():
        raise PlecsImportError(
            f"{what}: no points in the {'+' if v_sign > 0 else '−'}V/"
            f"{'+' if i_sign > 0 else '−'}I quadrant (voltage axis "
            f"{v.tolist()}, current axis {i.tolist()}).")
    sub = energy[np.ix_(vm, im, np.ones(tj.size, dtype=bool))]
    va = np.abs(v[vm])
    ia = np.abs(i[im])
    vo = np.argsort(va, kind="stable")
    io = np.argsort(ia, kind="stable")
    return LossTable(v_axis=va[vo], i_axis=ia[io], tj_axis=tj,
                     energy=sub[np.ix_(vo, io, np.arange(tj.size))])


def _conduction_raw(el: ET.Element, what: str, ignore_formula: bool, notes: List[str],
                    variables: Dict[str, Dict[str, Any]]):
    _method(el, what, ignore_formula, notes, variables)
    i, i_ord = _axis(el, "CurrentAxis", what)
    tj, t_ord = _axis(el, "TemperatureAxis", what)
    vd = el.find("VoltageDrop")
    if vd is None:
        raise PlecsImportError(f"{what}: no <VoltageDrop> element.")
    try:
        scale = float(vd.get("scale", "1"))
    except ValueError as e:
        raise PlecsImportError(f"{what}: VoltageDrop scale={vd.get('scale')!r} is not a number.") from e
    for child in vd:
        if child.tag != "Temperature":
            raise PlecsImportError(
                f"{what}: <VoltageDrop> holds a <{child.tag}>; expected only "
                "<Temperature> rows.")
    temps = vd.findall("Temperature")
    if len(temps) != tj.size:
        raise PlecsImportError(
            f"{what}: {len(temps)} <Temperature> rows but the TemperatureAxis "
            f"has {tj.size} entries.")
    v = np.zeros((i.size, tj.size))
    for kt_file, row in enumerate(temps):
        kt = int(np.where(t_ord == kt_file)[0][0])
        vals = _floats(row.text, f"{what}/Temperature[{kt_file}]")
        if vals.size != i.size:
            raise PlecsImportError(
                f"{what}: Temperature[{kt_file}] has {vals.size} values but the "
                f"CurrentAxis has {i.size} entries.")
        v[:, kt] = vals[i_ord] * scale
    return i, tj, v


def _conduction_half(i, tj, v, i_sign: int, what: str, notes: List[str]):
    """One current half of an on-state table, folded to |i| and |v|."""
    im = (i >= 0) if i_sign > 0 else (i <= 0)
    if not im.any():
        raise PlecsImportError(
            f"{what}: no {'positive' if i_sign > 0 else 'negative'}-current "
            f"points in the on-state table (current axis {i.tolist()}).")
    ia = np.abs(i[im])
    sub = v[im, :]
    nz = sub[np.abs(ia) > 0, :]
    if nz.size and np.any(nz > 0) and np.any(nz < 0):
        raise PlecsImportError(
            f"{what}: the {'positive' if i_sign > 0 else 'negative'}-current "
            "half of the on-state table has drops of both signs; the "
            "polarity cannot be folded.")
    if nz.size and np.any(nz < 0):
        notes.append(f"{what}: on-state drop folded from negative values")
    sub = np.abs(sub)
    order = np.argsort(ia, kind="stable")
    ia, sub = ia[order], sub[order, :]
    pos = ia > 0
    if pos.sum() >= 2 and np.any(np.diff(sub[pos, :], axis=0) < -1e-12):
        notes.append(f"{what}: on-state drop not monotone in current")
    return ia, tj, sub


def _thermal(el: ET.Element, what: str) -> Optional[ThermalChain]:
    for child in el:
        if child.tag != "Branch":
            raise PlecsImportError(
                f"{what}: holds a <{child.tag}> element; this importer takes one "
                "<Branch> (Foster or Cauer). An impedance matrix or a "
                "state-space block belongs to a package description, which is "
                "not a single device's chain.")
    branches = el.findall("Branch")
    if not branches:
        return None
    if len(branches) > 1:
        raise PlecsImportError(
            f"{what}: {len(branches)} <Branch> chains; this importer takes "
            "exactly one junction-to-case chain.")
    br = branches[0]
    kind = (br.get("type") or "").strip().lower()

    def attr(e: ET.Element, name: str, k: int) -> float:
        val = e.get(name)
        if val is None:
            raise PlecsImportError(f"{what}: stage {k} has no {name!r} attribute.")
        try:
            return float(val)
        except ValueError as ex:
            raise PlecsImportError(
                f"{what}: stage {k} {name}={val!r} is not a bare number (a unit "
                "suffix is not accepted).") from ex

    if kind not in ("foster", "cauer"):
        raise PlecsImportError(f"{what}: Branch type {kind!r}; expected Foster or Cauer.")
    want = "RTauElement" if kind == "foster" else "RCElement"
    for child in br:
        if child.tag != want:
            raise PlecsImportError(
                f"{what}: a {kind.capitalize()} branch holds a <{child.tag}>; "
                f"expected only <{want}> stages.")
    els = br.findall(want)
    if not els:
        return None
    R = tuple(attr(e, "R", k) for k, e in enumerate(els))
    second = tuple(attr(e, "Tau" if kind == "foster" else "C", k) for k, e in enumerate(els))
    if any(r <= 0 for r in R) or any(x <= 0 for x in second):
        raise PlecsImportError(f"{what}: thermal stages must be positive (R={R}, {second}).")
    return ThermalChain(kind, R, second)


def _variables(root: ET.Element, what: str) -> Dict[str, Dict[str, Any]]:
    """``<Variables>`` in either serialisation: child elements
    (Name|Variable, Prompt|Description, Default|Value, Min, Max) or
    attributes (name, description, default, min, max)."""
    out: Dict[str, Dict[str, Any]] = {}
    vs = root.find("Variables")
    if vs is None:
        return out
    for k, v in enumerate(vs):
        if v.tag != "Variable":
            raise PlecsImportError(f"{what}/Variables: holds a <{v.tag}>; expected <Variable>.")

        def get(*names: str) -> Optional[str]:
            for n in names:
                a = v.get(n)
                if a is not None:
                    return a.strip()
                a = v.get(n.lower())
                if a is not None:
                    return a.strip()
                t = v.findtext(n)
                if t is not None:
                    return t.strip()
            return None

        name = get("Name", "Variable")
        if not name:
            raise PlecsImportError(f"{what}/Variables: entry {k} has no name.")
        rec: Dict[str, Any] = {"prompt": get("Prompt", "Description") or ""}
        for key, names in (("default", ("Default", "Value")), ("min", ("Min",)), ("max", ("Max",))):
            s = get(*names)
            if s is None or s == "":
                rec[key] = None
                continue
            try:
                rec[key] = float(s)
            except ValueError as e:
                raise PlecsImportError(
                    f"{what}/Variables: {name}.{key} = {s!r} is not a number.") from e
        out[name] = rec
    return out


def import_plecs_xml(path, *, ignore_formula: bool = False,
                     conduction_index: Optional[int] = None) -> Part:
    """Read a PLECS thermal description file into a :class:`Part`.

    Parameters
    ----------
    path
        The ``.xml`` file the user downloaded from the vendor.
    ignore_formula
        Take the lookup table of a "table and formula" loss even
        though the formula is not applied — only when the formula
        refers to no variable or custom table. Off by default; the
        refusal names the formula.
    conduction_index
        Which ``<ConductionLoss>`` to take when the file carries more
        than one (gate-dependent descriptions); refused otherwise.
    """
    p = Path(path).expanduser()
    try:
        root = ET.parse(p).getroot()
    except ET.ParseError as e:
        raise PlecsImportError(f"{p}: not well-formed XML: {e}") from e
    if root.tag != "Package":
        raise PlecsImportError(
            f"{p}: root element is <{root.tag}>, expected <Package> — is this "
            "a PLECS thermal description file?")
    cls_raw = (root.get("class") or "").strip()
    mapped = _CLASS_MAP.get(cls_raw.lower())
    if mapped is None:
        raise PlecsImportError(
            f"{p}: Package class {cls_raw!r}; this importer knows Diode, IGBT, "
            "MOSFET, 'IGBT with Diode' and 'MOSFET with Diode'. A thyristor, "
            "GTO, TRIAC or package description is not a switch's table set.")
    cls, with_diode = mapped
    vendor = (root.get("vendor") or "unknown").strip()
    number = (root.get("partnumber") or p.stem).strip()
    notes: List[str] = [f"PLECS thermal description v{root.get('version', '?')}, class {cls_raw!r}"]

    variables = _variables(root, p.name)
    limits: Dict[str, Tuple[float, float]] = {}
    for n in ("v", "i", "T"):
        rec = variables.get(n)
        if rec and rec.get("min") is not None and rec.get("max") is not None:
            limits[n] = (float(rec["min"]), float(rec["max"]))
    custom = {n: r.get("default") for n, r in variables.items() if n not in ("v", "i", "T")}
    if custom:
        notes.append(f"variables declared: {custom} (name: default)")

    sds = root.findall("SemiconductorData")
    if len(sds) != 1:
        raise PlecsImportError(
            f"{p}: {len(sds)} <SemiconductorData> elements; a single-device "
            "file has exactly one (a package description with several chips "
            "is not imported — export each device on its own).")
    sd = sds[0]
    sd_type = (sd.get("type") or "").strip()
    if sd_type and sd_type.lower() != cls_raw.lower():
        raise PlecsImportError(
            f"{p}: Package class {cls_raw!r} but SemiconductorData type "
            f"{sd_type!r}; the file disagrees with itself.")

    loss_tables: Dict[str, LossTable] = {}
    for tag, key in (("TurnOnLoss", "Eon"), ("TurnOffLoss", "Eoff")):
        els = sd.findall(tag)
        if not els:
            continue
        if len(els) > 1:
            raise PlecsImportError(f"{p}: {len(els)} <{tag}> tables; expected one.")
        what = f"{p.name}/{tag}"
        v, i, tj, E = _energy_raw(els[0], what, ignore_formula, notes, variables)
        if cls == "diode":
            # Diode switching losses: negative voltage, positive current
            # (manual p. 199). A file written with positive voltages is
            # taken as it is.
            v_sign = -1 if np.any(v < 0) else 1
            if v_sign < 0:
                notes.append(f"{what}: voltage axis folded from the −V/+I quadrant")
            loss_tables["Erec" if key == "Eoff" else "Eon_diode"] = _quadrant(v, i, tj, E, v_sign, 1, what)
            continue
        loss_tables[key] = _quadrant(v, i, tj, E, 1, 1, what)
        if with_diode:
            # The diode half: positive voltage, negative current (manual p. 200).
            if not np.any(i < 0):
                raise PlecsImportError(
                    f"{what}: class {cls_raw!r} but the current axis {i.tolist()} "
                    "has no negative half where the diode's losses would be.")
            loss_tables["Erec" if key == "Eoff" else "Eon_diode"] = _quadrant(v, i, tj, E, 1, -1, what)
            notes.append(f"{what}: diode half split from the +V/−I quadrant")
        elif np.any(i < 0) and np.any(E[:, i < 0, :] != 0):
            raise PlecsImportError(
                f"{what}: class {cls_raw!r} (no diode) but the negative-current "
                "half of the table is not zero; if this is a 'with Diode' "
                "description the class is wrong, and the diode's losses would "
                "be dropped.")
    if "Eon_diode" in loss_tables and not np.any(loss_tables["Eon_diode"].energy):
        del loss_tables["Eon_diode"]

    conduction: Dict[str, Any] = {}
    cls_els = sd.findall("ConductionLoss")
    if len(cls_els) > 1 and conduction_index is None:
        raise PlecsImportError(
            f"{p}: {len(cls_els)} <ConductionLoss> definitions (a gate-dependent "
            "description carries one per gate state); pass conduction_index= "
            "to say which one applies to your run.")
    if cls_els:
        idx = 0 if conduction_index is None else int(conduction_index)
        if idx < 0 or idx >= len(cls_els):
            raise PlecsImportError(f"{p}: conduction_index={idx} but there are {len(cls_els)} definitions.")
        what = f"{p.name}/ConductionLoss[{idx}]"
        i, tj, v = _conduction_raw(cls_els[idx], what, ignore_formula, notes, variables)
        if cls == "diode":
            i_sign = 1 if np.any(i > 0) else -1
            conduction["v_on"] = _conduction_half(i, tj, v, i_sign, what, notes)
        else:
            conduction["v_on"] = _conduction_half(i, tj, v, 1, what, notes)
            if with_diode:
                if not np.any(i < 0):
                    raise PlecsImportError(
                        f"{what}: class {cls_raw!r} but no negative-current half "
                        "for the diode's on-state drop.")
                conduction["v_on_diode"] = _conduction_half(i, tj, v, -1, what, notes)

    chain = None
    tms = root.findall("ThermalModel")
    if len(tms) > 1:
        raise PlecsImportError(f"{p}: {len(tms)} <ThermalModel> elements; expected one.")
    if tms:
        chain = _thermal(tms[0], f"{p.name}/ThermalModel")

    comment = (root.findtext("Comment") or "").strip()
    params: Dict[str, Any] = {"plecs_class": cls_raw}
    if comment:
        params["comment"] = comment
    for c in root.findall("Constants/Constant"):
        n = c.get("name") or (c.findtext("Name") or "").strip()
        val = c.get("value") or (c.findtext("Value") or "").strip()
        if n:
            try:
                params[f"const_{n}"] = float(val)
            except ValueError:
                params[f"const_{n}"] = val

    if not loss_tables and not conduction:
        raise PlecsImportError(
            f"{p}: no loss tables and no conduction table were found in "
            "<SemiconductorData>; nothing to import.")
    prov = Provenance(
        source="plecs_xml",
        source_ref=str(p),
        retrieved=_dt.date.today().isoformat(),
        method="imported_plecs_xml",
        note="; ".join(notes),
    )
    return Part(cls=cls, vendor=vendor, number=number, provenance=prov,
                params=params, loss_tables=loss_tables, conduction=conduction,
                thermal_chain=chain, limits=limits, path=p)
