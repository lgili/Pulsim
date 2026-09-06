"""Part records, the YAML schema, provenance, and the registry.

A part file is YAML with a fixed shape. The schema is enforced by
hand here rather than through a JSON-schema library: every refusal
names the field and says what would have gone wrong, which a
generic validator cannot.

    class: mosfet | igbt | diode | core
    vendor: Wolfspeed
    part: C3M0065090J
    provenance:                      # REQUIRED — see Provenance
      source: datasheet
      source_ref: "C3M0065090J rev 2018-04, Fig. 12 and 13"
      retrieved: "2026-05-14"
      method: transcribed            # see Provenance for the list
      note: "..."
    # electrical facts (class-dependent)
    V_th_25c: 2.6
    R_ds_on_25c: 0.065
    ...
    # curves and tables
    Coss: [{V: 0.0, C: 1.8e-9}, ...]
    Eon: {x: [I...], y: [V...], values: [...], Tj: 25.0}   # 2-D at ONE Tj
    Eon_table: {v: [...], i: [...], tj: [...], energy: [...]} # 3-D
    v_on: {i: [...], tj: [...], v: [...]}                     # on-state drop
    thermal: {foster: [{R: ..., tau: ...}, ...]} | {cauer: [{R: ..., C: ...}, ...]}
    limits: {v: [0, 1200], i: [0, 80], T: [-40, 175]}

Numbers written the YAML-1.1 way (``4e-3`` with no dot or sign in
the exponent) load as strings under PyYAML; the loader coerces them,
so a part file does not silently lose a parameter to a type.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from ..loss_tables import LossTable

_CLASSES = ("mosfet", "igbt", "diode", "core")
_METHODS = ("transcribed", "imported_plecs_xml", "measured", "vendor_model", "synthetic")


class PartNotFound(LookupError):
    """No part file matched the requested name."""


class ProvenanceError(ValueError):
    """A part file has no usable provenance block."""


# =============================================================================
# Provenance
# =============================================================================

@dataclass(frozen=True)
class Provenance:
    """Where a part's numbers came from.

    ``method`` says how the numbers got into the file:

    * ``transcribed`` — read off a datasheet by a person. The curves
      and tables are then a reproduction of the manufacturer's
      plotted data, which is their copyright; ``note`` should say
      what was transcribed, from which figure and revision.
      :meth:`Part.notice` returns the attribution line.
    * ``imported_plecs_xml`` — produced by :func:`import_plecs_xml`
      from a file the user downloaded. ``source_ref`` names the file
      (or its URL); nothing is redistributed.
    * ``measured`` — the user's own bench data.
    * ``vendor_model`` — parameters from a vendor-published model file
      (a SPICE ``.lib``), with ``source_ref`` naming it.
    * ``synthetic`` — shapes written for an example, with no vendor
      source. The parts Pulsim ships are all of this kind: their
      switching-energy tables are straight lines through the origin,
      which no datasheet curve is. A synthetic part is listed, but
      :func:`part` refuses to resolve it by number unless
      ``allow_synthetic=True`` — a lookup of a real part number must
      not succeed and lie.
    """
    source: str
    source_ref: str
    retrieved: str
    method: str
    note: str = ""

    @staticmethod
    def from_mapping(m: Mapping[str, Any], what: str) -> "Provenance":
        if not isinstance(m, Mapping):
            raise ProvenanceError(
                f"{what}: 'provenance' must be a mapping with source, "
                f"source_ref, retrieved and method; got {type(m).__name__}.")
        missing = [k for k in ("source", "source_ref", "retrieved", "method")
                   if not str(m.get(k, "")).strip()]
        if missing:
            raise ProvenanceError(
                f"{what}: provenance is missing {missing}. A part file has to "
                "say where every number came from and how — a digitised "
                "datasheet curve is the manufacturer's copyright, and a "
                "value with no origin cannot be checked against anything. "
                "Fields: source (datasheet | plecs_xml | measurement | "
                "spice_lib | synthetic), source_ref (revision / file / URL), "
                "retrieved (YYYY-MM-DD), method (transcribed | "
                "imported_plecs_xml | measured | vendor_model | synthetic), "
                "note.")
        method = str(m["method"]).strip()
        if method not in _METHODS:
            raise ProvenanceError(
                f"{what}: provenance.method = {method!r}; expected one of "
                f"{_METHODS}.")
        return Provenance(source=str(m["source"]).strip(),
                          source_ref=str(m["source_ref"]).strip(),
                          retrieved=str(m["retrieved"]).strip(),
                          method=method,
                          note=str(m.get("note", "")).strip())


# =============================================================================
# Thermal chain
# =============================================================================

@dataclass(frozen=True)
class ThermalChain:
    """A Foster or Cauer junction-to-case chain, as the vendor gave it.

    ``kind`` is ``"foster"`` (R [K/W], tau [s] per stage) or
    ``"cauer"`` (R [K/W], C [J/K] per stage), in the order the stages
    appear in the file — junction first.
    """
    kind: str
    R: tuple
    second: tuple   # tau for foster, C for cauer

    def foster_stages(self):
        """The stages as :class:`pulsim.thermal.FosterStage` objects.
        Refuses a Cauer chain rather than converting it: the
        Foster↔Cauer conversion is not unique for chains with similar
        time constants, and a silently converted chain would carry a
        shape the vendor never published."""
        if self.kind != "foster":
            raise ValueError(
                "ThermalChain.foster_stages(): this chain is Cauer. Use "
                "cauer_stages() with add_cauer_thermal_network(); a "
                "Cauer→Foster conversion is not unique and is not done "
                "for you.")
        from ..thermal import FosterStage
        return [FosterStage(R_th_K_per_W=r, tau_s=t) for r, t in zip(self.R, self.second)]

    def cauer_stages(self):
        if self.kind != "cauer":
            raise ValueError(
                "ThermalChain.cauer_stages(): this chain is Foster. Use "
                "foster_stages() with add_foster_network().")
        from ..thermal import CauerStage
        return [CauerStage(R_th_K_per_W=r, C_th_J_per_K=c) for r, c in zip(self.R, self.second)]

    @property
    def R_th_total(self) -> float:
        return float(sum(self.R))


# =============================================================================
# Part
# =============================================================================

def _as_2d_table(name: str, d: Mapping[str, Any], what: str):
    """A datasheet-style 2-D table {x: I, y: V, values: row-major over
    (y, x)} at ONE junction temperature."""
    for k in ("x", "y", "values"):
        if k not in d:
            raise ValueError(f"{what}: {name} needs x, y and values.")
    x = np.asarray(d["x"], dtype=float)
    y = np.asarray(d["y"], dtype=float)
    vals = np.asarray(d["values"], dtype=float)
    if vals.size != x.size * y.size:
        raise ValueError(
            f"{what}: {name}.values has {vals.size} entries but x×y is "
            f"{x.size}×{y.size} = {x.size * y.size}.")
    return x, y, vals.reshape(y.size, x.size)


def _scalar_tj(Tj) -> Optional[float]:
    """The Tj a caller passed, as one number when it is one (or an
    array with one distinct value); None for a callable or a spread."""
    if callable(Tj):
        return None
    arr = np.atleast_1d(np.asarray(Tj, dtype=float))
    if arr.size == 0:
        return None
    if np.ptp(arr) > 1e-9:
        return None
    return float(arr.flat[0])


def _fit_on_state(i: np.ndarray, tj: np.ndarray, v: np.ndarray, Tj: float,
                  what: str) -> Tuple[float, float, float, float]:
    """Fit ``v_on = V0 + r·i`` over the positive-current part of an
    on-state table at ``Tj``. Returns (V0, r, max residual, Tj used).
    Refuses a temperature outside the table's axis rather than
    extrapolating an on-state drop."""
    if tj.size == 1:
        col = v[:, 0]
        tj_used = float(tj[0])
        if abs(Tj - tj_used) > 1.0:
            raise ValueError(
                f"{what}: the on-state table is at ONE temperature "
                f"({tj_used} °C) and you asked for Tj = {Tj}; the temperature "
                "dependence is not in the table.")
    else:
        if Tj < tj[0] - 1e-9 or Tj > tj[-1] + 1e-9:
            raise ValueError(
                f"{what}: Tj = {Tj} is outside the table's temperature axis "
                f"[{tj[0]}, {tj[-1]}] °C; an on-state drop is not extrapolated.")
        col = np.asarray([np.interp(Tj, tj, v[k, :]) for k in range(i.size)])
        tj_used = float(Tj)
    mask = i > 0
    if int(mask.sum()) < 2:
        raise ValueError(f"{what}: fewer than two positive-current points to fit.")
    A = np.vstack([np.ones(int(mask.sum())), i[mask]]).T
    sol = np.linalg.lstsq(A, col[mask], rcond=None)[0]
    V0, r = float(sol[0]), float(sol[1])
    resid = float(np.max(np.abs(col[mask] - (V0 + r * i[mask]))))
    return V0, r, resid, tj_used


@dataclass
class Part:
    """One device from the library.

    Attributes hold the raw, validated numbers; the methods turn them
    into what the rest of Pulsim consumes. Every method that would
    otherwise guess refuses by name instead.
    """
    cls: str
    vendor: str
    number: str
    provenance: Provenance
    params: Dict[str, Any] = field(default_factory=dict)
    curves: Dict[str, np.ndarray] = field(default_factory=dict)      # name -> (n, 2) array
    tables_2d: Dict[str, Any] = field(default_factory=dict)          # name -> (x, y, values, Tj)
    loss_tables: Dict[str, LossTable] = field(default_factory=dict)  # "Eon"/"Eoff"/"Erec"/"Eon_diode" -> 3-D
    conduction: Dict[str, Any] = field(default_factory=dict)         # "v_on"/"v_on_diode" -> (i, tj, v[i, tj])
    thermal_chain: Optional[ThermalChain] = None
    limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # "v"/"i"/"T" -> (min, max)
    path: Optional[Path] = None

    # -- identity ----------------------------------------------------
    @property
    def name(self) -> str:
        return self.number

    @property
    def is_synthetic(self) -> bool:
        return self.provenance.method == "synthetic"

    def __repr__(self) -> str:
        return (f"Part({self.cls} {self.vendor} {self.number}, "
                f"{self.provenance.method} from {self.provenance.source_ref!r})")

    def notice(self) -> str:
        """The attribution / caveat line for this part. Nothing prints
        it for you except the synthetic warning in :func:`part`."""
        if self.is_synthetic:
            return (f"pulsim.lib: {self.vendor} {self.number} is a SYNTHETIC "
                    "illustration — its curves and tables are shapes written "
                    f"for examples, not {self.vendor}'s data. Do not act on a "
                    "loss figure computed from it.")
        if self.provenance.method == "transcribed":
            return (f"pulsim.lib: {self.vendor} {self.number} — values "
                    f"transcribed from {self.provenance.source_ref}; the "
                    f"curves and tables are {self.vendor}'s datasheet data, "
                    "reproduced for interoperability.")
        return ""

    # -- what the simulator consumes ---------------------------------
    def _check_single_tj(self, table: LossTable, key: str, Tj, allow: bool, what: str) -> None:
        """A one-point temperature axis answers every Tj with the same
        number; LossTable's interpolation weight is 0 there. Refuse an
        off-point query unless the caller opts in."""
        if table.tj_axis.size != 1:
            return
        t0 = float(table.tj_axis[0])
        s = _scalar_tj(Tj)
        if s is not None and abs(s - t0) <= 1.0:
            return
        msg = (f"{what}: {key} is tabulated at ONE junction temperature "
               f"({t0} °C) and you asked for Tj = {Tj!r}; the temperature "
               "dependence is NOT in these numbers (reading a 25 °C table "
               "at 125 °C was the −35 % switching-loss error of audit C.1). "
               "Pass allow_tj_mismatch=True to use the table knowingly, or "
               "import a file with a temperature axis.")
        if not allow:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=3)

    def _rr_curve(self, table: LossTable, key: str, V_ref: Optional[float], Tj,
                  allow: bool, what: str) -> Dict[str, Any]:
        """Reduce a 3-D recovery-energy table to the ``E_rr_curve`` +
        ``V_R_ref`` pair the diode loss panel takes, at one voltage
        and one temperature."""
        self._check_single_tj(table, key, Tj, allow, what)
        s = _scalar_tj(Tj)
        if s is None:
            raise ValueError(
                f"{what}: a diode's recovery curve is reduced at ONE Tj; pass "
                "a scalar Tj (the loss panel has no 3-D diode table).")
        if V_ref is None:
            if table.v_axis.size == 1:
                V_ref = float(table.v_axis[0])
            else:
                raise ValueError(
                    f"{what}: {key} is tabulated at voltages "
                    f"{table.v_axis.tolist()} — pass V_ref= (the reverse "
                    "voltage the diode blocks in this run).")
        if V_ref < table.v_axis[0] - 1e-9 or V_ref > table.v_axis[-1] + 1e-9:
            raise ValueError(
                f"{what}: V_ref = {V_ref} is outside the {key} voltage axis "
                f"[{table.v_axis[0]}, {table.v_axis[-1]}] V.")
        if table.tj_axis.size > 1 and (s < table.tj_axis[0] - 1e-9 or s > table.tj_axis[-1] + 1e-9):
            raise ValueError(
                f"{what}: Tj = {s} is outside the {key} temperature axis "
                f"[{table.tj_axis[0]}, {table.tj_axis[-1]}] °C.")
        curve = [(float(ii), float(table(float(V_ref), float(ii), s))) for ii in table.i_axis]
        return {"E_rr_curve": curve, "V_R_ref": float(V_ref), "Tj": s}

    def switch_spec(self, *, Tj=None, V_ref: Optional[float] = None,
                    allow_tj_mismatch: bool = False) -> Dict[str, Any]:
        """The per-device spec dict :func:`pulsim.losses.device_loss_summary`
        takes.

        A switch with 3-D tables (V, I, Tj) passes them through as
        ``E_on_table`` / ``E_off_table`` and needs ``Tj`` (a scalar,
        an array on the result grid, or a callable). A 2-D datasheet
        table taken at ONE temperature is passed as ``E_on_curve`` /
        ``E_off_curve`` at ``V_ref`` — and only at the ``Tj`` it was
        taken at, unless ``allow_tj_mismatch=True``: a 25 °C table
        read at 125 °C was the −35 % switching-loss error audit C.1
        measured, and a warning is not enough to stop it. The spec
        records ``Tj_table`` either way.

        A diode's recovery table is reduced to ``E_rr_curve`` +
        ``V_R_ref`` at the stated ``V_ref`` and ``Tj``.
        """
        what = f"{self!r}.switch_spec()"
        spec: Dict[str, Any] = {}
        if self.loss_tables:
            if Tj is None:
                raise ValueError(
                    f"{what}: the part carries loss tables indexed by junction "
                    "temperature — pass Tj= (scalar, array on the result grid, "
                    "or callable).")
            if self.cls == "diode":
                if "Erec" not in self.loss_tables:
                    raise ValueError(f"{what}: no recovery-energy (Erec) table.")
                return self._rr_curve(self.loss_tables["Erec"], "Erec", V_ref, Tj,
                                      allow_tj_mismatch, what)
            for key, out in (("Eon", "E_on_table"), ("Eoff", "E_off_table")):
                if key in self.loss_tables:
                    self._check_single_tj(self.loss_tables[key], key, Tj, allow_tj_mismatch, what)
                    spec[out] = self.loss_tables[key]
            if not spec:
                raise ValueError(f"{what}: no Eon/Eoff tables (has {sorted(self.loss_tables)}).")
            spec["Tj"] = Tj
            return spec
        if not self.tables_2d:
            raise ValueError(
                f"{what}: this part has no switching-energy data (no "
                "Eon/Eoff/Erec). Import the vendor's PLECS thermal file or "
                "add a transcribed table with its provenance.")
        tj_table = self._tables_tj()
        if Tj is None:
            raise ValueError(
                f"{what}: the part's energy tables were taken at ONE junction "
                f"temperature ({tj_table} °C) and have no Tj axis. State the Tj "
                "you will run at (Tj=…) so the mismatch is on record; reading "
                "a 25 °C table at 125 °C was a −35 % switching-loss error on a "
                "600 V IGBT (audit C.1).")
        s = _scalar_tj(Tj)
        if s is None or abs(s - tj_table) > 1.0:
            msg = (f"{what}: Eon/Eoff/Erec are {tj_table} °C tables and you asked "
                   f"for Tj = {Tj!r}; the temperature dependence is NOT in these "
                   "numbers. Import the vendor's PLECS file for a (V, I, Tj) "
                   "table, or pass allow_tj_mismatch=True to use the "
                   f"{tj_table} °C numbers knowingly.")
            if not allow_tj_mismatch:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)
        keys = ((("Erec", "E_rr_curve"),) if self.cls == "diode"
                else (("Eon", "E_on_curve"), ("Eoff", "E_off_curve")))
        for key, out_key in keys:
            if key not in self.tables_2d:
                continue
            x, y, vals, _ = self.tables_2d[key]
            if V_ref is None:
                if y.size != 1:
                    raise ValueError(
                        f"{what}: {key} is tabulated at voltages {y.tolist()} — "
                        "pass V_ref= to pick the row the run's bus voltage "
                        "corresponds to (the spec scales linearly from it).")
                row = 0
            else:
                idx = np.argmin(np.abs(y - float(V_ref)))
                if abs(y[idx] - float(V_ref)) > 1e-9 * max(1.0, abs(V_ref)):
                    raise ValueError(
                        f"{what}: {key} has no row at V_ref = {V_ref}; tabulated "
                        f"voltages are {y.tolist()}.")
                row = int(idx)
            spec[out_key] = [(float(xi), float(vi)) for xi, vi in zip(x, vals[row])]
            spec["V_R_ref" if self.cls == "diode" else "V_ref"] = float(y[row])
        if not spec:
            raise ValueError(f"{what}: no usable table for a {self.cls} "
                             f"(has {sorted(self.tables_2d)}).")
        spec["Tj_table"] = tj_table
        return spec

    def diode_spec(self, *, Tj=None, V_ref: Optional[float] = None,
                   allow_tj_mismatch: bool = False) -> Dict[str, Any]:
        """The recovery spec of the body / anti-parallel diode of a
        "with Diode" part (PLECS files fold it into the negative-current
        half of the switch's tables; the importer splits it out)."""
        what = f"{self!r}.diode_spec()"
        if "Erec" not in self.loss_tables:
            raise ValueError(
                f"{what}: this part carries no diode recovery table; a plain "
                "MOSFET/IGBT description has none (PLECS puts the diode in the "
                "'... with Diode' class).")
        if Tj is None:
            raise ValueError(f"{what}: pass Tj= (scalar).")
        return self._rr_curve(self.loss_tables["Erec"], "Erec", V_ref, Tj, allow_tj_mismatch, what)

    def _tables_tj(self) -> float:
        tjs = {round(float(t[3]), 3) for t in self.tables_2d.values()}
        return float(next(iter(tjs))) if len(tjs) == 1 else float("nan")

    def conduction_spec(self, *, Tj: Optional[float] = None, diode: bool = False) -> Dict[str, Any]:
        """On-state parameters for the PWL switch models.

        With an on-state table (``v_on`` from a PLECS file) the drop is
        fitted as ``V0 + r·i`` over the positive currents at ``Tj``;
        the fit's largest residual is returned as ``fit_residual_V``
        and a residual above 5 % of the drop is warned about — the PWL
        models are offset-plus-slope and a strongly curved table does
        not reduce to that without loss. Without a table, the scalar
        ratings in the file are used.
        """
        what = f"{self!r}.conduction_spec()"
        key = "v_on_diode" if diode else "v_on"
        if key in self.conduction:
            if Tj is None:
                raise ValueError(f"{what}: pass Tj= to read the on-state table.")
            i, tj, v = self.conduction[key]
            V0, r, resid, tj_used = _fit_on_state(i, tj, v, float(Tj), what)
            vmax = float(np.max(np.abs(v)))
            if vmax > 0 and resid > 0.05 * vmax:
                warnings.warn(
                    f"{what}: the on-state table at {tj_used} °C is not an "
                    f"offset-plus-slope line; the fit misses it by up to "
                    f"{resid:.3g} V (drop up to {vmax:.3g} V).", stacklevel=2)
            out = {"Tj": tj_used, "fit_residual_V": resid}
            if diode or self.cls == "diode":
                out.update({"V_F": V0, "R_s": r})
            elif self.cls == "igbt":
                out.update({"V_CE_sat": V0, "R_CE_sat": r})
            else:
                out.update({"R_on": r, "V_on_0": V0})
            return out
        if diode:
            raise ValueError(f"{what}: no diode on-state table in this part.")
        p = self.params
        if self.cls == "mosfet":
            if "R_ds_on_25c" not in p:
                raise ValueError(f"{what}: no R_ds_on_25c and no on-state table.")
            return {"R_on": float(p["R_ds_on_25c"]),
                    "R_on_temp_coef": float(p.get("R_ds_on_temp_coef", 0.0)),
                    "V_F": float(p.get("V_sd", 0.7))}
        if self.cls == "igbt":
            if "V_ce_sat_25c" not in p:
                raise ValueError(f"{what}: no V_ce_sat_25c and no on-state table.")
            return {"V_CE_sat": float(p["V_ce_sat_25c"]),
                    "R_CE_sat": float(p.get("R_ce_sat", 0.0))}
        if self.cls == "diode":
            if "V_f_25c" not in p:
                raise ValueError(f"{what}: no V_f_25c and no on-state table.")
            return {"V_F": float(p["V_f_25c"]), "R_s": float(p.get("R_s", 0.0))}
        raise ValueError(f"{what}: conduction_spec() is for semiconductors.")

    def thermal(self) -> ThermalChain:
        if self.thermal_chain is None:
            where = f" ({self.path})" if self.path else ""
            raise ValueError(
                f"{self!r}: no thermal model in the part file{where} — no "
                "Foster/Cauer chain and no Zth curve. Two routes: import the "
                "vendor's PLECS thermal file (its <ThermalModel> carries the "
                "chain) with pulsim.lib.import_plecs_xml(), or transcribe the "
                "datasheet's Zth(t) and fit it with "
                "pulsim.thermal.fit_foster_from_zth(), then add it under "
                "'thermal:' with its provenance.")
        return self.thermal_chain

    def capacitance(self, which: str) -> np.ndarray:
        """A C(V) curve (Coss/Ciss/Crss) as an (n, 2) array."""
        if which not in self.curves:
            raise ValueError(f"{self!r}: no {which} curve.")
        return self.curves[which]


# =============================================================================
# YAML → Part
# =============================================================================

_NUM_RE = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")


def _coerce_numbers(x: Any) -> Any:
    """PyYAML follows YAML 1.1, where ``4e-3`` is a string; coerce
    every such string so a parameter is not silently lost to a type."""
    if isinstance(x, str):
        s = x.strip()
        if _NUM_RE.match(s):
            try:
                return float(s)
            except ValueError:  # pragma: no cover
                return x
        return x
    if isinstance(x, Mapping):
        return {k: _coerce_numbers(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_coerce_numbers(v) for v in x]
    return x


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pulsim.lib reads part files with PyYAML, which is not installed; "
            "run `pip install pulsim[lib]` (or `pip install pyyaml`).") from e
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, Mapping):
        raise ValueError(f"{path}: not a mapping at top level")
    return _coerce_numbers(doc)


def part_from_mapping(doc: Mapping[str, Any], what: str, path: Optional[Path] = None) -> Part:
    doc = _coerce_numbers(doc)
    cls = str(doc.get("class", "")).strip().lower()
    if cls not in _CLASSES:
        raise ValueError(f"{what}: class = {doc.get('class')!r}; expected one of {_CLASSES}.")
    vendor = str(doc.get("vendor", "")).strip()
    number = str(doc.get("part", doc.get("material", ""))).strip()
    if not vendor or not number:
        raise ValueError(f"{what}: vendor and part (or material) are required.")
    if "provenance" not in doc:
        raise ProvenanceError(
            f"{what}: no 'provenance' block. Every part says where its "
            "numbers came from — see pulsim.lib.Provenance.")
    prov = Provenance.from_mapping(doc["provenance"], what)

    skip = {"class", "vendor", "part", "material", "provenance", "thermal",
            "Coss", "Ciss", "Crss", "Eon", "Eoff", "Erec", "Eon_table", "Eoff_table",
            "Erec_table", "Eon_diode_table", "bh_curve", "v_on", "v_on_diode", "limits"}
    params = {k: v for k, v in doc.items() if k not in skip}

    curves: Dict[str, np.ndarray] = {}
    for c in ("Coss", "Ciss", "Crss"):
        if c in doc:
            pts = doc[c]
            arr = np.asarray([[float(pt["V"]), float(pt["C"])] for pt in pts], dtype=float)
            if arr.ndim != 2 or arr.shape[0] < 2:
                raise ValueError(f"{what}: {c} needs at least two {{V, C}} points.")
            if np.any(np.diff(arr[:, 0]) <= 0):
                raise ValueError(f"{what}: {c} voltages must be strictly increasing.")
            curves[c] = arr
    if "bh_curve" in doc:
        pts = doc["bh_curve"]
        curves["BH"] = np.asarray([[float(pt["H"]), float(pt["B"])] for pt in pts], dtype=float)

    tables_2d: Dict[str, Any] = {}
    for t in ("Eon", "Eoff", "Erec"):
        if t in doc:
            x, y, vals = _as_2d_table(t, doc[t], what)
            tj = float(doc[t].get("Tj", doc.get("Tj_tables", 25.0)))
            if np.any(vals < 0):
                raise ValueError(f"{what}: {t} has a negative energy.")
            tables_2d[t] = (x, y, vals, tj)

    loss_tables: Dict[str, LossTable] = {}
    for t in ("Eon", "Eoff", "Erec", "Eon_diode"):
        key = f"{t}_table"
        if key in doc:
            d = doc[key]
            v = np.asarray(d["v"], dtype=float)
            i = np.asarray(d["i"], dtype=float)
            tj = np.asarray(d["tj"], dtype=float)
            e = np.asarray(d["energy"], dtype=float)
            if e.size != v.size * i.size * tj.size:
                raise ValueError(
                    f"{what}: {key}.energy has {e.size} entries but v×i×tj is "
                    f"{v.size}×{i.size}×{tj.size}.")
            loss_tables[t] = LossTable(v_axis=v, i_axis=i, tj_axis=tj,
                                       energy=e.reshape(v.size, i.size, tj.size))

    conduction: Dict[str, Any] = {}
    for key in ("v_on", "v_on_diode"):
        if key in doc:
            d = doc[key]
            i = np.asarray(d["i"], dtype=float)
            tj = np.asarray(d["tj"], dtype=float)
            v = np.asarray(d["v"], dtype=float)
            if v.size != i.size * tj.size:
                raise ValueError(f"{what}: {key}.v has {v.size} entries but i×tj is {i.size}×{tj.size}.")
            conduction[key] = (i, tj, v.reshape(i.size, tj.size))

    chain: Optional[ThermalChain] = None
    if "thermal" in doc and doc["thermal"]:
        th = doc["thermal"]
        if "foster" in th:
            st = th["foster"]
            chain = ThermalChain("foster", tuple(float(s["R"]) for s in st),
                                 tuple(float(s["tau"]) for s in st))
        elif "cauer" in th:
            st = th["cauer"]
            chain = ThermalChain("cauer", tuple(float(s["R"]) for s in st),
                                 tuple(float(s["C"]) for s in st))
        else:
            raise ValueError(f"{what}: thermal needs a 'foster' or 'cauer' list.")
        if any(r <= 0 for r in chain.R) or any(x <= 0 for x in chain.second):
            raise ValueError(f"{what}: thermal chain entries must be positive.")

    limits: Dict[str, Tuple[float, float]] = {}
    for k, rng in (doc.get("limits") or {}).items():
        try:
            lo, hi = float(rng[0]), float(rng[1])
        except (TypeError, ValueError, IndexError) as e:
            raise ValueError(f"{what}: limits.{k} must be [min, max].") from e
        limits[str(k)] = (lo, hi)

    return Part(cls=cls, vendor=vendor, number=number, provenance=prov,
                params=params, curves=curves, tables_2d=tables_2d,
                loss_tables=loss_tables, conduction=conduction,
                thermal_chain=chain, limits=limits, path=path)


def load_part_file(path) -> Part:
    """Load one part file (YAML) and validate it."""
    p = Path(path).expanduser()
    return part_from_mapping(_load_yaml(p), str(p), path=p)


# =============================================================================
# Registry
# =============================================================================

_DATA_DIR = Path(__file__).resolve().parent / "data"
_SHADOW_WARNED: set = set()


def search_paths(extra: Optional[Iterable[Any]] = None) -> List[Path]:
    """Where parts are looked for, first match wins:

    1. every directory passed as ``extra`` (a project's own parts),
    2. every directory in ``PULSIM_PARTS_PATH`` (``:``-separated),
    3. the parts shipped with Pulsim (``pulsim/lib/data``).

    The current working directory is deliberately NOT searched: a
    stray ``parts/`` next to a script silently changing a simulation's
    losses is the kind of default this library exists to remove. A
    user file that shadows a shipped one is announced once, naming
    both files.
    """
    out: List[Path] = []
    for d in (extra or []):
        pd = Path(d).expanduser()
        if not pd.is_dir():
            raise ValueError(f"pulsim.lib: parts directory {pd} does not exist.")
        out.append(pd)
    env = os.environ.get("PULSIM_PARTS_PATH", "")
    for d in env.split(os.pathsep):
        d = d.strip()
        if d and Path(d).expanduser().is_dir():
            out.append(Path(d).expanduser())
    out.append(_DATA_DIR)
    return out


def _iter_part_files(paths: Iterable[Path]):
    first: Dict[str, Path] = {}
    for root in paths:
        for f in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
            key = f.stem.lower()
            if key in first:
                if key not in _SHADOW_WARNED and first[key] != f:
                    _SHADOW_WARNED.add(key)
                    warnings.warn(
                        f"pulsim.lib: {first[key]} shadows {f} for part "
                        f"{f.stem!r}; the first one on the search path wins.",
                        stacklevel=4)
                continue
            first[key] = f
            yield f


def list_parts(cls: Optional[str] = None, *, paths: Optional[Iterable[Any]] = None) -> List[Part]:
    """Every loadable part on the search path, synthetic ones included
    (they are listed, not hidden — :func:`part` is what refuses them).
    Files that fail validation are skipped with a warning naming them."""
    out = []
    for f in _iter_part_files(search_paths(paths)):
        try:
            pt = load_part_file(f)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"pulsim.lib: skipping {f}: {e}", stacklevel=2)
            continue
        if cls is None or pt.cls == cls:
            out.append(pt)
    return out


def part(number: str, *, cls: Optional[str] = None, vendor: Optional[str] = None,
         allow_synthetic: bool = False, paths: Optional[Iterable[Any]] = None) -> Part:
    """Look a part up by number (case-insensitive, exact). ``cls`` and
    ``vendor`` narrow the match.

    A part whose provenance is ``synthetic`` is refused by name unless
    ``allow_synthetic=True``, and then warned about: the shipped
    illustrations carry real part numbers, and a lookup that succeeds
    on one and returns invented losses would be a wrong answer with
    no signal.
    """
    want = number.strip().lower()
    sp = search_paths(paths)
    candidates = []
    synthetic = []
    for f in _iter_part_files(sp):
        if f.stem.lower() != want:
            continue
        try:
            pt = load_part_file(f)
        except ProvenanceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"pulsim.lib: {f} matched {number!r} but failed to load: {e}") from e
        if cls and pt.cls != cls:
            continue
        if vendor and pt.vendor.lower() != vendor.lower():
            continue
        if pt.is_synthetic and not allow_synthetic:
            synthetic.append(pt)
            continue
        candidates.append(pt)
    if not candidates:
        if synthetic:
            pt = synthetic[0]
            raise PartNotFound(
                f"pulsim.lib: the only {number!r} on the search path is a "
                f"SYNTHETIC illustration ({pt.path}): its tables are shapes "
                f"written for examples, not {pt.vendor}'s published data (the "
                "switching energies are straight lines through the origin). "
                "For real numbers import the vendor's PLECS thermal file with "
                "pulsim.lib.import_plecs_xml(path), or transcribe the datasheet "
                "with provenance into a directory on PULSIM_PARTS_PATH. Pass "
                "allow_synthetic=True to use the illustration knowingly.")
        near = [p.stem for p in _iter_part_files(sp)
                if want[:5] and p.stem.lower().startswith(want[:5])]
        hint = f" Did you mean {near[:5]}?" if near else ""
        raise PartNotFound(
            f"pulsim.lib: no part named {number!r}"
            + (f" of class {cls!r}" if cls else "")
            + (f" from {vendor!r}" if vendor else "")
            + f" on the search path {[str(p) for p in sp]}.{hint} "
            "Import the vendor's PLECS thermal file with "
            "pulsim.lib.import_plecs_xml(path) and save it with save_part().")
    pt = candidates[0]
    if pt.is_synthetic:
        warnings.warn(pt.notice(), stacklevel=2)
    return pt


def mosfet(number: str, **kw) -> Part:
    return part(number, cls="mosfet", **kw)


def igbt(number: str, **kw) -> Part:
    return part(number, cls="igbt", **kw)


def diode(number: str, **kw) -> Part:
    return part(number, cls="diode", **kw)


def core(material: str, **kw) -> Part:
    return part(material, cls="core", **kw)


def search(*, cls: Optional[str] = None, vendor: Optional[str] = None,
           v_max_min: Optional[float] = None, include_synthetic: bool = False,
           paths: Optional[Iterable[Any]] = None) -> List[Part]:
    """Filter the library by class, vendor and minimum voltage rating.
    Synthetic illustrations are left out unless asked for."""
    out = []
    for pt in list_parts(cls, paths=paths):
        if pt.is_synthetic and not include_synthetic:
            continue
        if vendor and pt.vendor.lower() != vendor.lower():
            continue
        if v_max_min is not None:
            vm = pt.params.get("V_ds_max", pt.params.get("V_ces_max", pt.params.get("V_rrm")))
            if vm is None or float(vm) < v_max_min:
                continue
        out.append(pt)
    return out


def save_part(part_: Part, path) -> Path:
    """Write a Part back to YAML (used for imported PLECS files)."""
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pulsim.lib writes part files with PyYAML, which is not installed; "
            "run `pip install pulsim[lib]`.") from e
    doc: Dict[str, Any] = {
        "class": part_.cls, "vendor": part_.vendor, "part": part_.number,
        "provenance": {
            "source": part_.provenance.source,
            "source_ref": part_.provenance.source_ref,
            "retrieved": part_.provenance.retrieved,
            "method": part_.provenance.method,
            "note": part_.provenance.note,
        },
    }
    doc.update({k: v for k, v in part_.params.items()})
    for name, tab in part_.loss_tables.items():
        doc[f"{name}_table"] = {
            "v": tab.v_axis.tolist(), "i": tab.i_axis.tolist(),
            "tj": tab.tj_axis.tolist(), "energy": tab.energy.ravel().tolist()}
    for name, (i, tj, v) in part_.conduction.items():
        doc[name] = {"i": i.tolist(), "tj": tj.tolist(), "v": v.ravel().tolist()}
    if part_.thermal_chain is not None:
        ch = part_.thermal_chain
        if ch.kind == "foster":
            doc["thermal"] = {"foster": [{"R": r, "tau": t} for r, t in zip(ch.R, ch.second)]}
        else:
            doc["thermal"] = {"cauer": [{"R": r, "C": c} for r, c in zip(ch.R, ch.second)]}
    if part_.limits:
        doc["limits"] = {k: [lo, hi] for k, (lo, hi) in part_.limits.items()}
    for name, arr in part_.curves.items():
        if name == "BH":
            doc["bh_curve"] = [{"H": float(h), "B": float(b)} for h, b in arr]
        else:
            doc[name] = [{"V": float(v), "C": float(c)} for v, c in arr]
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return p
