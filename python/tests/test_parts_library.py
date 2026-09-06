"""The parts library — audit C.5.

Before this, the six device YAMLs under ``devices/`` were consumed by
nothing in the package (verified by grep). Read closely, they are
synthetic: every switching-energy table is exactly E = c·I and
proportional in V, and four of six headers name the wrong package. A
registry that resolved the real part number 'C3M0065090J' to those
tables would succeed and lie, so the shipped parts are marked
``synthetic`` and refused by number unless asked for. Every part
carries provenance, PLECS files are imported with their sign
conventions honoured, and a one-temperature table refuses an
off-temperature query.
"""

from __future__ import annotations

import textwrap
import warnings

import numpy as np
import pytest

import pulsim as p

lib = p.lib


def _mosfet(number="C3M0065090J"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lib.mosfet(number, allow_synthetic=True)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def test_shipped_parts_are_synthetic_and_not_resolvable_by_number():
    parts = lib.list_parts()
    assert len(parts) >= 10
    assert all(pt.is_synthetic for pt in parts)
    assert all(pt.provenance.source_ref and pt.provenance.retrieved for pt in parts)
    with pytest.raises(lib.PartNotFound, match="SYNTHETIC"):
        lib.mosfet("C3M0065090J")
    with pytest.warns(UserWarning, match="SYNTHETIC"):
        q = lib.mosfet("C3M0065090J", allow_synthetic=True)
    assert q.vendor == "Wolfspeed" and q.cls == "mosfet"
    assert q.params["R_ds_on_25c"] == pytest.approx(0.065)
    assert q.params["R_ds_on_temp_coef"] == pytest.approx(4e-3)   # YAML-1.1 '4e-3' coerced
    assert "Coss" in q.curves and q.curves["Coss"].shape[1] == 2
    assert lib.search(cls="mosfet") == []                           # synthetic left out
    assert len(lib.search(cls="mosfet", include_synthetic=True)) >= 3


def test_lookup_is_case_insensitive_and_class_scoped():
    assert _mosfet("c3m0065090j").number == "C3M0065090J"
    with pytest.raises(lib.PartNotFound, match="C3M00"):
        lib.mosfet("C3M0065090D", allow_synthetic=True)      # sibling not shipped: hint
    with pytest.raises(lib.PartNotFound, match="search path"):
        lib.igbt("C3M0065090J", allow_synthetic=True)         # wrong class
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert lib.diode("C4D20120D", allow_synthetic=True).cls == "diode"
        assert "BH" in lib.core("N87", allow_synthetic=True).curves


def test_a_part_without_provenance_is_refused_by_name(tmp_path):
    f = tmp_path / "NOPROV.yaml"
    f.write_text(textwrap.dedent("""
        class: mosfet
        vendor: Acme
        part: NOPROV
        R_ds_on_25c: 0.01
    """))
    with pytest.raises(lib.ProvenanceError, match="provenance"):
        lib.load_part_file(f)
    f.write_text(textwrap.dedent("""
        class: mosfet
        vendor: Acme
        part: NOPROV
        provenance: {source: datasheet, source_ref: "rev 1", retrieved: "2026-09-06", method: guessed}
        R_ds_on_25c: 0.01
    """))
    with pytest.raises(lib.ProvenanceError, match="method"):
        lib.load_part_file(f)


def test_user_parts_directory_overrides_the_shipped_one_and_says_so(tmp_path, monkeypatch):
    d = tmp_path / "mine"
    d.mkdir()
    (d / "C3M0065090J.yaml").write_text(textwrap.dedent("""
        class: mosfet
        vendor: Wolfspeed
        part: C3M0065090J
        provenance: {source: measurement, source_ref: "bench 2026-09-01", retrieved: "2026-09-06", method: measured}
        R_ds_on_25c: 0.070
    """))
    monkeypatch.setenv("PULSIM_PARTS_PATH", str(d))
    lib.parts._SHADOW_WARNED.clear()
    with pytest.warns(UserWarning, match="shadows"):
        q = lib.mosfet("C3M0065090J")                 # not synthetic: no opt-in needed
    assert q.params["R_ds_on_25c"] == pytest.approx(0.070)
    assert q.provenance.method == "measured"


def test_the_working_directory_is_not_searched(tmp_path, monkeypatch):
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "CWDPART.yaml").write_text(textwrap.dedent("""
        class: diode
        vendor: Acme
        part: CWDPART
        provenance: {source: measurement, source_ref: "x", retrieved: "2026-09-06", method: measured}
        V_f_25c: 1.0
    """))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PULSIM_PARTS_PATH", raising=False)
    with pytest.raises(lib.PartNotFound):
        lib.diode("CWDPART")
    assert lib.diode("CWDPART", paths=[tmp_path / "parts"]).params["V_f_25c"] == 1.0


# --------------------------------------------------------------------------
# what the simulator consumes
# --------------------------------------------------------------------------

def test_switch_spec_from_a_single_temperature_table_refuses_off_temperature():
    q = _mosfet()                             # Eon/Eoff at 25 C, rows at 400/600 V
    with pytest.raises(ValueError, match="ONE junction temperature"):
        q.switch_spec()
    with pytest.raises(ValueError, match="V_ref"):
        q.switch_spec(Tj=25.0)                # two voltage rows: pick one
    with pytest.raises(ValueError, match="temperature dependence is NOT"):
        q.switch_spec(Tj=125.0, V_ref=600.0)
    with pytest.warns(UserWarning, match="temperature dependence is NOT"):
        spec = q.switch_spec(Tj=125.0, V_ref=600.0, allow_tj_mismatch=True)
    assert spec["V_ref"] == 600.0 and spec["Tj_table"] == 25.0
    spec = q.switch_spec(Tj=25.0, V_ref=600.0)
    assert spec["E_on_curve"][0] == (10.0, 75e-6)
    assert spec["E_off_curve"][-1] == (35.0, 105e-6)


def test_switch_spec_feeds_the_loss_summary():
    spec = _mosfet().switch_spec(Tj=25.0, V_ref=600.0)
    from pulsim.losses import _switch_switching_loss
    times = np.linspace(0.0, 1e-3, 11)
    closed = np.zeros(11, dtype=bool)
    closed[5:] = True
    out = _switch_switching_loss(closed, times, np.full(11, 600.0), np.full(11, 20.0), spec)
    assert out["E_sw_on_total"] == pytest.approx(150e-6, rel=1e-6)   # the 600 V row at 20 A


def test_thermal_is_refused_when_the_part_has_none_naming_the_file_and_the_routes():
    q = _mosfet()
    with pytest.raises(ValueError, match="C3M0065090J.yaml.*import_plecs_xml.*fit_foster_from_zth"):
        q.thermal()


# --------------------------------------------------------------------------
# PLECS thermal description files
# --------------------------------------------------------------------------

_IGBT = """<?xml version="1.0" encoding="UTF-8"?>
<Package class="IGBT" vendor="Acme" partnumber="AK40N120" version="1.2">
  <Variables>
    <Variable><Name>v</Name><Prompt>Blocking voltage</Prompt><Default>600</Default><Min>0</Min><Max>1200</Max></Variable>
    <Variable><Name>i</Name><Prompt>On-state current</Prompt><Default>20</Default><Min>0</Min><Max>80</Max></Variable>
    <Variable><Name>T</Name><Prompt>Junction temperature</Prompt><Default>25</Default><Min>-40</Min><Max>175</Max></Variable>
    <Variable name="Rg" description="Gate resistance" default="10" min="1" max="50"/>
  </Variables>
  <SemiconductorData type="IGBT">
    <TurnOnLoss>
      <ComputationMethod>Table only</ComputationMethod>
      <CurrentAxis>0 20 40</CurrentAxis>
      <VoltageAxis>0 300 600</VoltageAxis>
      <TemperatureAxis>25 125</TemperatureAxis>
      <Energy scale="0.001">
        <Temperature>
          <Voltage>0 0 0</Voltage>
          <Voltage>0 1.0 2.0</Voltage>
          <Voltage>0 2.0 4.0</Voltage>
        </Temperature>
        <Temperature>
          <Voltage>0 0 0</Voltage>
          <Voltage>0 1.5 3.0</Voltage>
          <Voltage>0 3.0 6.0</Voltage>
        </Temperature>
      </Energy>
    </TurnOnLoss>
    <TurnOffLoss>
      <ComputationMethod>Table only</ComputationMethod>
      <CurrentAxis>0 20 40</CurrentAxis>
      <VoltageAxis>0 300 600</VoltageAxis>
      <TemperatureAxis>25 125</TemperatureAxis>
      <Energy scale="0.001">
        <Temperature>
          <Voltage>0 0 0</Voltage><Voltage>0 0.5 1.0</Voltage><Voltage>0 1.0 2.0</Voltage>
        </Temperature>
        <Temperature>
          <Voltage>0 0 0</Voltage><Voltage>0 0.8 1.6</Voltage><Voltage>0 1.6 3.2</Voltage>
        </Temperature>
      </Energy>
    </TurnOffLoss>
    <ConductionLoss>
      <ComputationMethod>Table only</ComputationMethod>
      <CurrentAxis>0 20 40</CurrentAxis>
      <TemperatureAxis>25 125</TemperatureAxis>
      <VoltageDrop scale="1">
        <Temperature>0 1.6 2.0</Temperature>
        <Temperature>0 1.8 2.4</Temperature>
      </VoltageDrop>
    </ConductionLoss>
  </SemiconductorData>
  <ThermalModel>
    <Branch type="Foster">
      <RTauElement R="0.1" Tau="0.001"/>
      <RTauElement R="0.2" Tau="0.02"/>
    </Branch>
  </ThermalModel>
  <Comment>synthetic fixture</Comment>
</Package>
"""

# A diode file the way the manual says vendors write it: switching
# losses in the NEGATIVE-voltage / positive-current quadrant, the
# turn-on loss a single zero point.
_DIODE = """<?xml version="1.0" encoding="UTF-8"?>
<Package class="Diode" vendor="Acme" partnumber="AD30S120" version="1.0">
  <SemiconductorData type="Diode">
    <TurnOnLoss>
      <ComputationMethod>Table only</ComputationMethod>
      <CurrentAxis>0</CurrentAxis>
      <VoltageAxis>0</VoltageAxis>
      <TemperatureAxis>25</TemperatureAxis>
      <Energy scale="1"><Temperature><Voltage>0</Voltage></Temperature></Energy>
    </TurnOnLoss>
    <TurnOffLoss>
      <ComputationMethod>Table only</ComputationMethod>
      <CurrentAxis>0 10 20</CurrentAxis>
      <VoltageAxis>-600 -300 0</VoltageAxis>
      <TemperatureAxis>25 125</TemperatureAxis>
      <Energy scale="0.001">
        <Temperature>
          <Voltage>0 0.30 0.50</Voltage>
          <Voltage>0 0.15 0.25</Voltage>
          <Voltage>0 0 0</Voltage>
        </Temperature>
        <Temperature>
          <Voltage>0 0.60 1.00</Voltage>
          <Voltage>0 0.30 0.50</Voltage>
          <Voltage>0 0 0</Voltage>
        </Temperature>
      </Energy>
    </TurnOffLoss>
    <ConductionLoss>
      <ComputationMethod>Table only</ComputationMethod>
      <CurrentAxis>0 10 20</CurrentAxis>
      <TemperatureAxis>25 125</TemperatureAxis>
      <VoltageDrop scale="1">
        <Temperature>0 1.4 1.7</Temperature>
        <Temperature>0 1.3 1.7</Temperature>
      </VoltageDrop>
    </ConductionLoss>
  </SemiconductorData>
</Package>
"""

# "IGBT with Diode": the diode folded into the negative-current half
# of the same tables (switching losses in +V/−I, on-state drop at −I).
_WITH_DIODE = """<?xml version="1.0" encoding="UTF-8"?>
<Package class="IGBT with Diode" vendor="Acme" partnumber="AK40N120D" version="1.2">
  <SemiconductorData type="IGBT with Diode">
    <TurnOnLoss>
      <ComputationMethod>Table only</ComputationMethod>
      <CurrentAxis>-40 -20 0 20 40</CurrentAxis>
      <VoltageAxis>0 300 600</VoltageAxis>
      <TemperatureAxis>25 125</TemperatureAxis>
      <Energy scale="0.001">
        <Temperature>
          <Voltage>0 0 0 0 0</Voltage>
          <Voltage>0.4 0.2 0 1.0 2.0</Voltage>
          <Voltage>0.8 0.4 0 2.0 4.0</Voltage>
        </Temperature>
        <Temperature>
          <Voltage>0 0 0 0 0</Voltage>
          <Voltage>0.6 0.3 0 1.5 3.0</Voltage>
          <Voltage>1.2 0.6 0 3.0 6.0</Voltage>
        </Temperature>
      </Energy>
    </TurnOnLoss>
    <TurnOffLoss>
      <ComputationMethod>Table only</ComputationMethod>
      <CurrentAxis>-40 -20 0 20 40</CurrentAxis>
      <VoltageAxis>0 300 600</VoltageAxis>
      <TemperatureAxis>25 125</TemperatureAxis>
      <Energy scale="0.001">
        <Temperature>
          <Voltage>0 0 0 0 0</Voltage>
          <Voltage>0.5 0.25 0 0.5 1.0</Voltage>
          <Voltage>1.0 0.5 0 1.0 2.0</Voltage>
        </Temperature>
        <Temperature>
          <Voltage>0 0 0 0 0</Voltage>
          <Voltage>0.8 0.4 0 0.8 1.6</Voltage>
          <Voltage>1.6 0.8 0 1.6 3.2</Voltage>
        </Temperature>
      </Energy>
    </TurnOffLoss>
    <ConductionLoss>
      <ComputationMethod>Table only</ComputationMethod>
      <CurrentAxis>-40 -20 0 20 40</CurrentAxis>
      <TemperatureAxis>25 125</TemperatureAxis>
      <VoltageDrop scale="1">
        <Temperature>-1.9 -1.5 0 1.6 2.0</Temperature>
        <Temperature>-2.1 -1.6 0 1.8 2.4</Temperature>
      </VoltageDrop>
    </ConductionLoss>
  </SemiconductorData>
</Package>
"""


def test_plecs_xml_imports_tables_limits_and_thermal_chain(tmp_path):
    f = tmp_path / "AK40N120.xml"
    f.write_text(_IGBT)
    part = lib.import_plecs_xml(f)
    assert part.cls == "igbt" and part.vendor == "Acme" and part.number == "AK40N120"
    assert part.provenance.method == "imported_plecs_xml"
    assert part.limits == {"v": (0.0, 1200.0), "i": (0.0, 80.0), "T": (-40.0, 175.0)}
    assert "'Rg': 10.0" in part.provenance.note
    eon = part.loss_tables["Eon"]
    assert eon(600.0, 40.0, 125.0) == pytest.approx(6.0e-3)     # scale 0.001 applied
    assert eon(300.0, 20.0, 25.0) == pytest.approx(1.0e-3)
    spec = part.switch_spec(Tj=125.0)
    assert spec["E_on_table"] is eon and spec["Tj"] == 125.0
    i, tj, v = part.conduction["v_on"]
    assert v[2, 1] == pytest.approx(2.4)
    cs = part.conduction_spec(Tj=125.0)
    assert cs["V_CE_sat"] == pytest.approx(1.2) and cs["R_CE_sat"] == pytest.approx(0.03)
    with pytest.raises(ValueError, match="outside"):
        part.conduction_spec(Tj=150.0)
    ch = part.thermal()
    assert ch.kind == "foster" and ch.R_th_total == pytest.approx(0.3)
    assert ch.foster_stages()[1].tau_s == pytest.approx(0.02)
    with pytest.raises(ValueError, match="is Foster"):
        ch.cauer_stages()


def test_plecs_diode_file_negative_voltage_axis_is_folded_not_zeroed(tmp_path):
    f = tmp_path / "AD30S120.xml"
    f.write_text(_DIODE)
    part = lib.import_plecs_xml(f)
    assert part.cls == "diode"
    assert "Eon_diode" not in part.loss_tables            # the zero point is dropped
    erec = part.loss_tables["Erec"]
    assert erec.v_axis.tolist() == [0.0, 300.0, 600.0]
    assert erec(600.0, 20.0, 125.0) == pytest.approx(1.0e-3)   # would be 0 J unfolded
    assert erec(300.0, 10.0, 25.0) == pytest.approx(0.15e-3)
    spec = part.switch_spec(Tj=125.0, V_ref=600.0)
    assert spec["V_R_ref"] == 600.0
    assert spec["E_rr_curve"] == [(0.0, 0.0), (10.0, pytest.approx(0.6e-3)), (20.0, pytest.approx(1.0e-3))]
    with pytest.raises(ValueError, match="V_ref"):
        part.switch_spec(Tj=125.0)
    cs = part.conduction_spec(Tj=25.0)
    assert cs["V_F"] == pytest.approx(1.1) and cs["R_s"] == pytest.approx(0.03)


def test_plecs_with_diode_file_is_split_into_switch_and_diode_halves(tmp_path):
    f = tmp_path / "AK40N120D.xml"
    f.write_text(_WITH_DIODE)
    part = lib.import_plecs_xml(f)
    assert part.loss_tables["Eon"](600.0, 40.0, 25.0) == pytest.approx(4.0e-3)
    assert part.loss_tables["Erec"](600.0, 40.0, 25.0) == pytest.approx(1.0e-3)
    assert part.loss_tables["Eon_diode"](600.0, 40.0, 25.0) == pytest.approx(0.8e-3)
    assert part.diode_spec(Tj=125.0, V_ref=600.0)["E_rr_curve"][-1] == (40.0, pytest.approx(1.6e-3))
    i, tj, v = part.conduction["v_on_diode"]
    assert i.tolist() == [0.0, 20.0, 40.0] and v[2, 0] == pytest.approx(1.9)
    assert part.conduction_spec(Tj=25.0, diode=True)["V_F"] == pytest.approx(1.1)
    # The same table under a plain "IGBT" class is refused: the diode
    # half would be dropped.
    g = tmp_path / "wrong.xml"
    g.write_text(_WITH_DIODE.replace('class="IGBT with Diode"', 'class="IGBT"').replace('type="IGBT with Diode"', 'type="IGBT"'))
    with pytest.raises(lib.PlecsImportError, match="negative-current half"):
        lib.import_plecs_xml(g)


def test_plecs_xml_round_trips_through_a_part_file(tmp_path):
    f = tmp_path / "AK40N120.xml"
    f.write_text(_IGBT)
    part = lib.import_plecs_xml(f)
    out = lib.save_part(part, tmp_path / "parts" / "AK40N120.yaml")
    back = lib.load_part_file(out)
    assert back.loss_tables["Eoff"](600.0, 40.0, 125.0) == pytest.approx(3.2e-3)
    assert back.thermal().R == part.thermal().R
    assert back.limits == part.limits
    assert lib.igbt("AK40N120", paths=[tmp_path / "parts"]).number == "AK40N120"


def test_plecs_single_temperature_table_refuses_off_temperature(tmp_path):
    f = tmp_path / "one.xml"
    f.write_text(_IGBT.replace("<TemperatureAxis>25 125</TemperatureAxis>", "<TemperatureAxis>25</TemperatureAxis>")
                 .replace("""        <Temperature>
          <Voltage>0 0 0</Voltage>
          <Voltage>0 1.5 3.0</Voltage>
          <Voltage>0 3.0 6.0</Voltage>
        </Temperature>
""", "").replace("""        <Temperature>
          <Voltage>0 0 0</Voltage><Voltage>0 0.8 1.6</Voltage><Voltage>0 1.6 3.2</Voltage>
        </Temperature>
""", "").replace("        <Temperature>0 1.8 2.4</Temperature>\n", ""))
    part = lib.import_plecs_xml(f)
    assert part.switch_spec(Tj=25.0)["Tj"] == 25.0
    with pytest.raises(ValueError, match="ONE junction temperature"):
        part.switch_spec(Tj=125.0)
    with pytest.warns(UserWarning, match="ONE junction temperature"):
        part.switch_spec(Tj=125.0, allow_tj_mismatch=True)


def test_plecs_xml_refuses_formulas_by_name(tmp_path):
    f = tmp_path / "F.xml"
    f.write_text(_IGBT.replace("<ComputationMethod>Table only</ComputationMethod>",
                               "<ComputationMethod>Table and formula</ComputationMethod><Formula>E*1.1</Formula>", 1))
    with pytest.raises(lib.PlecsImportError, match="E\\*1.1"):
        lib.import_plecs_xml(f)
    part = lib.import_plecs_xml(f, ignore_formula=True)
    assert "E*1.1" in part.provenance.note
    # A formula that depends on a declared variable cannot be dropped knowingly.
    g = tmp_path / "G.xml"
    g.write_text(_IGBT.replace("<ComputationMethod>Table only</ComputationMethod>",
                               "<ComputationMethod>Table and formula</ComputationMethod><Formula>E*Rg/10</Formula>", 1))
    with pytest.raises(lib.PlecsImportError, match="Rg"):
        lib.import_plecs_xml(g, ignore_formula=True)


@pytest.mark.parametrize("old,new,frag", [
    ("<Voltage>0 2.0 4.0</Voltage>", "<Voltage>0 2.0</Voltage>", "CurrentAxis has 3"),
    ('<Package class="IGBT"', '<Package class="Triac"', "Triac"),
    ('type="IGBT"', 'type="MOSFET"', "disagrees"),
    ('<RTauElement R="0.2" Tau="0.02"/>', '<RCElement R="0.2" C="0.02"/>', "RCElement"),
    ('<RTauElement R="0.2" Tau="0.02"/>', '<RTauElement R="0.2" Tau="20ms"/>', "bare number"),
    ("<VoltageAxis>0 300 600</VoltageAxis>", "<VoltageAxis>0 300 300</VoltageAxis>", "duplicated"),
    ("<ThermalModel>", "<ThermalModel><ImpedanceMatrix/>", "ImpedanceMatrix"),
])
def test_plecs_xml_refuses_what_it_cannot_read_without_guessing(tmp_path, old, new, frag):
    f = tmp_path / "bad.xml"
    f.write_text(_IGBT.replace(old, new, 1))
    with pytest.raises(lib.PlecsImportError, match=frag):
        lib.import_plecs_xml(f)


def test_plecs_two_conduction_definitions_need_a_choice(tmp_path):
    cl = _IGBT[_IGBT.index("    <ConductionLoss>"):_IGBT.index("    </ConductionLoss>") + len("    </ConductionLoss>\n")]
    f = tmp_path / "two.xml"
    f.write_text(_IGBT.replace(cl, cl + cl.replace("0 1.6 2.0", "0 1.0 1.2")))
    with pytest.raises(lib.PlecsImportError, match="conduction_index"):
        lib.import_plecs_xml(f)
    part = lib.import_plecs_xml(f, conduction_index=1)
    assert part.conduction["v_on"][2][1, 0] == pytest.approx(1.0)


def test_plecs_descending_axis_is_reversed_not_misread(tmp_path):
    f = tmp_path / "desc.xml"
    f.write_text(_IGBT.replace("<VoltageAxis>0 300 600</VoltageAxis>", "<VoltageAxis>600 300 0</VoltageAxis>", 1)
                 .replace("""          <Voltage>0 0 0</Voltage>
          <Voltage>0 1.0 2.0</Voltage>
          <Voltage>0 2.0 4.0</Voltage>""", """          <Voltage>0 2.0 4.0</Voltage>
          <Voltage>0 1.0 2.0</Voltage>
          <Voltage>0 0 0</Voltage>""")
                 .replace("""          <Voltage>0 0 0</Voltage>
          <Voltage>0 1.5 3.0</Voltage>
          <Voltage>0 3.0 6.0</Voltage>""", """          <Voltage>0 3.0 6.0</Voltage>
          <Voltage>0 1.5 3.0</Voltage>
          <Voltage>0 0 0</Voltage>"""))
    part = lib.import_plecs_xml(f)
    assert part.loss_tables["Eon"](600.0, 40.0, 125.0) == pytest.approx(6.0e-3)
