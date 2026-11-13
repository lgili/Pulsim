"""``add_nonlinear_diode`` is retired (fix-nonlinear-diode-passivity).

Its smooth-blend law ``logistic(k(v - V_F0))·(v - V_F0)/R_d`` is negative
for every ``0 < v < V_F0``: at the default ``IdealDiodeParams`` it pushes
-1.3923 A out of a forward-biased anode at +0.636 V. The law is kept (the
hostile-circuit fixtures rely on it), but every public entry point now
refuses it unless the caller passes the explicit opt-in, and the SPICE
importer maps LTspice's ``Ron/Roff/Vfwd`` diode to the PWL diode with a
forward drop.
"""

from __future__ import annotations

import pathlib
import re
import warnings

import numpy as np
import pytest

import pulsim as p


def _sim(b, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, **kw)


# ---------------------------------------------------------------------
# the refusal
# ---------------------------------------------------------------------

def test_the_builder_refuses_the_retired_device_and_names_the_replacements():
    b = p.CircuitBuilder()
    with pytest.raises(p.RetiredDeviceError) as e:
        b.add_nonlinear_diode("Dx", "a", "gnd", p.IdealDiodeParams())
    msg = str(e.value)
    assert isinstance(e.value, ValueError)
    for needle in ("Dx", "-1.39 A", "+0.636 V", "add_diode", "V_f=",
                   "add_shockley_diode", "allow_non_passive=True"):
        assert needle in msg, needle
    assert b.num_branches == 0          # nothing was stamped


def _forced(v, **kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", v)
    b.add_nonlinear_diode("D", "a", "gnd", p.IdealDiodeParams(), **kw)
    return b


def test_the_opt_in_still_builds_the_legacy_law():
    # The law is unchanged: the forced 0.636 V point still sources power.
    r = _sim(_forced(0.636, allow_non_passive=True),
             t_end=1e-5, dt=1e-6, engine="pwl")
    # r.i("V") is the source's MNA branch current: minus the diode current.
    i_diode = -float(np.asarray(r.i("V"))[-1])
    assert i_diode == pytest.approx(-1.3923, abs=1e-3)
    x = np.asarray(p.compute_dc_op(_forced(0.636, allow_non_passive=True)))
    assert x[-1] == pytest.approx(1.3923, abs=1e-3)


# ---------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------

_YAML = """
circuit:
  devices:
    - type: nonlinear_diode
      name: Dn
      anode: a
      cathode: gnd{extra}
"""


def test_yaml_refuses_the_retired_device_without_the_opt_in():
    with pytest.raises(RuntimeError, match=r"Dn.*retired.*allow_non_passive"):
        p.load_yaml_string(_YAML.format(extra=""))


def test_yaml_builds_it_with_the_opt_in():
    lc = p.load_yaml_string(_YAML.format(extra="\n      allow_non_passive: true"))
    kinds = {c["name"]: c["kind"] for c in lc.builder.components()}
    assert kinds == {"Dn": "nonlinear_diode"}


# ---------------------------------------------------------------------
# importer: LTspice Ron/Roff/Vfwd -> PWL diode with V_f
# ---------------------------------------------------------------------

_NET = """D1 a 0 DX{area}
.MODEL DX D(Ron=0.05 Roff=1Meg Vfwd=0.7)
.END
"""


def _component(b, name):
    return next(c for c in b.components() if c["name"] == name)


def test_imported_pwl_diode_is_a_switched_diode_with_a_forward_drop():
    b = p.spice_to_builder(_NET.format(area=""))
    c = _component(b, "D1")
    assert c["kind"] == "diode"
    prm = c["params"]
    assert prm["g_on"] == pytest.approx(20.0, rel=1e-12)
    assert prm["g_off"] == pytest.approx(1e-6, rel=1e-12)
    assert prm["V_f"] == 0.7
    assert prm["V_th"] == 0.0
    # 10 A forward through the diode. Added through the builder: the
    # importer's own `I` source sign is a separate defect.
    b.add_current_source("I1", "a", "gnd", 10.0)
    r = _sim(b, t_end=2e-3, dt=1e-6, engine="pwl")
    assert float(np.asarray(r.v("a"))[-1]) == pytest.approx(1.2, rel=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = next(e for e in p.device_loss_summary(b, r, t_from=1e-3)
                 if e["name"] == "D1")
    assert e["P_avg"] == pytest.approx(12.0, rel=1e-9)


def test_imported_pwl_diode_area_scales_both_conductances():
    prm = _component(p.spice_to_builder(_NET.format(area=" 2")), "D1")["params"]
    assert prm["g_on"] == pytest.approx(40.0, rel=1e-12)
    assert prm["g_off"] == pytest.approx(2e-6, rel=1e-12)


# ---------------------------------------------------------------------
# no library path reaches the device without the opt-in
# ---------------------------------------------------------------------

def test_no_library_module_calls_the_retired_builder_method():
    root = pathlib.Path(p.__file__).resolve().parent
    callers = [
        str(f.relative_to(root)) for f in root.rglob("*.py")
        if f.name != "_builder_ergonomics.py"
        and re.search(r"\.add_nonlinear_diode\s*\(", f.read_text("utf-8"))
    ]
    assert callers == []
