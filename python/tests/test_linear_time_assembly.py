"""update-linear-time-assembly tier 1: counter regression (no wall clock).

One sparse structure and one symbolic analysis per assembly regime,
whatever the number of switch states a run visits.
"""

import warnings

import pulsim


def _vsi():
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", 600.0)
    v = pulsim.add_three_phase_vsi(b, "INV", vdc_pos="dcp", vdc_neg="gnd",
                                   out_a="a", out_b="b", out_c="c")
    for ph in "abc":
        b.add_resistor(f"R{ph}", ph, f"m{ph}", 10.0)
        b.add_inductor(f"L{ph}", f"m{ph}", "n", 5e-3)
    legs = pulsim.ThreePhaseLegIndices(*v.switch_indices)
    fn = pulsim.make_three_phase_spwm_fn(10e3, 50.0, 0.8, legs,
                                         b.graph.num_switches, 1e-6)
    return b, fn


def test_vsi_fixed_engine_one_structure_one_analysis():
    b, fn = _vsi()
    caches = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pulsim.simulate(b, t_end=5e-3, dt=1e-7, engine="pwl", switch_fn=fn,
                        on_cache=caches.append)
    (cache,) = caches
    m = cache.metrics()
    assert cache.num_built_segments() > 20  # many masks visited
    assert m.structure_builds <= 2
    assert m.symbolic_analyses <= 2
    assert m.symbolic_fallbacks == 0

