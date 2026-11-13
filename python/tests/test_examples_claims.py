"""The examples tell the truth: each one runs and meets its stated numbers.

The 2026-11 examples audit found closed loops judged mid start-up
(run_buck_closed_loop.py printed 16.47 V for a 12 V target), a phase
margin printed "by construction" (60° claimed, 37.9° real), a flyback
wired as a forward converter, milliohm ground ties shorting sources,
switch bits counted by hand, and stored notebook outputs from removed
APIs. This module pins what the audit established, so it cannot rot:

* ``test_script_claims`` runs every script under ``examples/scripts``
  headlessly and checks each number in ``SCRIPTS`` (a regex on stdout,
  a target and a tolerance). Scripts with no row must still run clean.
* ``test_notebook_claims`` executes each notebook in ``NOTEBOOKS``
  (nbclient) and checks the same way.
* ``test_stored_notebook_outputs_are_current`` is cheap and always on:
  the outputs stored in those notebooks must carry the same numbers,
  so a refreshed notebook cannot silently go stale.

The first two are slow (minutes): they run only with
``PULSIM_RUN_EXAMPLES=1`` (marker ``examples``). No wall-clock checks.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "examples" / "scripts"
RUN_SLOW = os.environ.get("PULSIM_RUN_EXAMPLES", "0") == "1"
needs_tree = pytest.mark.skipif(not SCRIPTS_DIR.is_dir(),
                                reason="examples/ is not shipped with a pip install")


def near(target, tol):
    return (target - tol, target + tol)


def below(bound):
    return (float("-inf"), bound)


def above(bound):
    return (bound, float("inf"))


F = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"

# script -> [(regex with ONE group, (lo, hi))]. Reference for each value
# in the script's own docstring (analytic formula or settled setpoint).
SCRIPTS = {
    "run_buck_closed_loop.py": [
        (r"pre-step .*?mean=" + F, near(12.0, 0.05)),
        (r"post-step .*?mean=" + F, near(14.0, 0.05)),
        (r"PI updates: (\d+)", near(2001, 0)),
    ],
    "run_auto_tune_buck.py": [
        (r"predicted: crossover \[400\] Hz, PM = " + F, near(60.0, 0.5)),
        (r"measured: crossover \[\d+\] Hz, PM = " + F, near(60.0, 3.0)),
        (r"pre-step .*?mean=" + F, near(12.0, 0.05)),
        (r"post-step .*?mean=" + F, near(14.0, 0.05)),
    ],
    "run_boost_closed_loop.py": [
        (r"pre-step.*?mean\s*=\s*" + F, near(20.0, 0.1)),
        (r"post-step.*?mean\s*=\s*" + F, near(28.0, 0.1)),
    ],
    "run_boost_saturable_closed_loop.py": [
        (r"pre-step.*?mean = " + F, near(18.0, 0.1)),
        (r"post-step.*?mean = " + F, near(22.0, 0.1)),
    ],
    "run_flyback_closed_loop.py": [
        (r"pre-step.*?mean\s*=\s*" + F, near(20.0, 0.1)),
        (r"post-step.*?mean\s*=\s*" + F, near(26.0, 0.15)),
        (r"max V_sw = " + F, below(400.0)),
    ],
    "run_buck_closed_loop_chain.py": [
        (r"pre-step.*?mean\s*=\s*" + F, near(12.0, 0.05)),
        (r"post-step.*?mean\s*=\s*" + F, near(14.0, 0.05)),
    ],
    "run_three_phase_dq_closed_loop.py": [
        (r"i_d\s+pre-step.*?= " + F, near(5.0, 0.1)),
        (r"i_q\s+post-step.*?= " + F, near(-3.0, 0.1)),
    ],
    "run_mmc_3phase_dq_closed_loop.py": [
        (r"i_d\s+pre-step.*?= " + F, near(5.0, 0.1)),
        (r"i_q\s+post-step.*?= " + F, near(-3.0, 0.1)),
    ],
    "run_pfc_boost_closed_loop.py": [
        (r"V_bus mean \(late\):\s+" + F, near(200.0, 1.0)),
        (r"PF = P/\(V_rms·I_rms\): " + F, above(0.98)),
    ],
    "run_im_ifoc_closed_loop.py": [
        (r"i_d\s+= " + F, near(8.0, 0.05)),
        (r"i_q\s+= " + F, near(6.0, 0.05)),
        # 1.5·pp·(Lm²/Lr)·i_d·i_q = 11.607 Nm; 12.9 with the old
        # forward-Euler rotor flux.
        (r"T_em\s+= " + F, near(11.607, 0.12)),
    ],
    "run_im_sensorless_mras.py": [(r"error .*\(" + F + r"%\)", near(0.0, 2.0))],
    "run_im_vf_open_loop.py": [(r"ω_final\s+= " + F, near(157.08, 0.1))],
    "run_foc_motor_drive.py": [(r"i_q = " + F, near(5.0, 0.05))],
    "run_fast_block_pi_buck.py": [
        (r"v_out \(last 2 ms\) = " + F, near(12.0, 0.05)),
        (F + r" V pp", below(0.05)),
    ],
    "run_dc_motor_speed.py": [
        (r"ω mean\s+= " + F, near(100.0, 0.1)),
        (r"i_armature mean = " + F, near(2.02, 0.01)),
    ],
    "run_ac_sweep_buck.py": [(r"rms \|dB error\|: " + F, below(0.5))],
    "debug_buck_bode_offset.py": [
        (r"ANALYTICAL with applied duty.*?ratio " + F + "x", near(1.0, 0.01))],
    "run_ac_sweep_rc.py": [
        (r"max \|dB error\|: " + F, below(0.5)),
        (r"max \|deg error\|: " + F, below(2.0)),
    ],
    "run_mna_sweep_rc.py": [(r"max \|mag err\|\s+= " + F, below(0.5))],
    "run_rlc_step_response.py": [
        (r"max \|V_C - analytic\| = " + F + " mV", below(50.0))],
    "run_three_phase_diode_rectifier.py": [
        (r"V_dc mean ≈ " + F, near(171.2, 1.5))],
    "run_3phase_rectifier_grid.py": [
        (r"V_dc mean \(after 40 ms\): " + F, near(547.0, 5.0))],
    "run_ldo_with_opamp.py": [(r"V_out DC ≈ " + F, near(7.190, 0.01))],
    "run_common_source_amplifier.py": [
        (r"V_drain DC ≈ " + F, near(4.545, 0.03)),
        (r"Av \(sim\)\s+≈ " + F, near(-9.92, 0.1)),
    ],
    "run_pwm_chopper_realistic_mosfet.py": [
        (r"V_drain mean:\s+" + F, near(5.33, 0.02))],
    "run_buck.py": [(r"V_out DC ≈ " + F, near(12.0, 0.05))],
    "run_buck_with_helper.py": [(r"V_out steady-state: " + F, near(5.0, 0.02))],
    "run_three_phase_inverter.py": [
        (r"i_a peak \(last cycle\): " + F, near(24.3, 1.0))],
    "run_boost_realistic_igbt.py": [(r"V_out DC ≈ " + F, near(19.62, 0.1))],
    "run_boost_saturable_inductor.py": [(r"V_out DC ≈ " + F, near(19.62, 0.1))],
    "run_sweep_buck_lc.py": [(r"max \|ripple .* - 1\|: " + F + " %", below(10.0))],
    "run_flyback.py": [(r"V_out DC ≈ " + F, near(19.85, 0.2))],
    "run_psfb_hysteretic_inloop.py": [(r"v_out_avg = " + F, near(6.88, 0.1))],
    "run_bdf1_vs_trap.py": [(r"trap = " + F + " V", near(1.9969, 0.005))],
    "run_adaptive_rl_settle.py": [(r"max err / I_final: " + F + "%", below(0.5))],
    "run_adaptive_rk_speedup.py": [(r"agree to " + F, below(1e-5))],
    "run_thermal_buck.py": [(r"T_j rise:\s+" + F + " K", near(15.20, 0.05))],
    "run_device_loss_to_thermal.py": [(r"actual T_j_peak: " + F, near(54.78, 0.1))],
    "run_mmc_l0_vs_l1.py": [(r"max \|L1 v_C - L0 v_C\| = " + F, below(1.0))],
    "run_mmc_l3_balancing.py": [(r"final spread \(balanced\) = " + F, below(0.5))],
    "run_pmsm_smo_sensorless.py": [(r"peak " + F + "°", below(10.0))],
}

# Interactive (pyqtgraph window, no headless mode): never run here.
INTERACTIVE = {"run_live_scope_buck.py"}

# notebook (repo-relative) -> [(regex on its text outputs, (lo, hi))]
NOTEBOOKS = {
    "examples/notebooks/02_buck_converter.ipynb": [
        (r"V_out mean\s+=\s+" + F, near(4.994, 0.025))],
    "examples/notebooks/04_parameter_sweeps.ipynb": [
        (r"Simulated: R = " + F, near(59.86, 0.3))],
    "examples/notebooks/07_convergence.ipynb": [
        (r"auto\s+: V\(n2\) = " + F, near(8.3333, 1e-4)),
        # diverged to ~1e84 V before the PTC shift moved to the KCL rows
        (r"pseudo_trans : V\(n2\) = " + F, near(8.3333, 1e-4))],
    # The adaptive driver must read the same buck as the fixed step at the
    # same dt (it read 7.9619 V against 8.0431 V before resume_from).
    "examples/notebooks/24_buck_mosfet_diode_pwm_fixed_vs_variable.ipynb": [
        (r"mode: fixed[\s\S]*?vout_mean_tail: " + F, near(8.0431, 0.004)),
        (r"mode: variable[\s\S]*?vout_mean_tail: " + F, near(8.0431, 0.004))],
    "examples/notebooks/12_spice_parser.ipynb": [
        (r"v_dc_out = " + F + " V", near(9.29, 0.05))],
    "examples/notebooks/14_api_reference.ipynb": [
        (r"V_out \(mean 32-40 ms\) = " + F, near(12.0, 0.06))],
    "examples/notebooks/15_new_components_catalog.ipynb": [
        (r"Total OK: (\d+) / 20", near(20, 0))],
    "examples/notebooks/buck_converter_design.ipynb": [
        (r"V_out médio: " + F + "V", near(11.9205, 0.06))],
    "examples/notebooks/boost_converter_design.ipynb": [
        (r"Vout médio: " + F + "V", near(23.953, 0.12))],
    "examples/notebooks/boost_pfc_closed_loop.ipynb": [
        (r"PF:\s+" + F, above(0.99)),
        (r"V_dc média:\s+" + F, near(99.48, 1.0)),
    ],
    "examples/notebooks/flyback_converter_design.ipynb": [
        (r"Vout médio: " + F + "V", near(47.93, 0.48))],
    "examples/notebooks/forward_converter_design.ipynb": [
        (r"Vout medio:\s+" + F, near(11.970, 0.06))],
    "examples/pulsim_tutorial.ipynb": [
        (r"V_out avg " + F, near(25.80, 0.13))],
    # Project validations: pinned to the audited run (each within its
    # stated tolerance of the analytic value printed beside it).
    "projects/converters/buck/00_buck_pulsim_validation.ipynb": [
        (r"Pulsim  V_o \(mean over last [\d.]+ ms\): " + F, near(11.9951, 0.012))],
    "projects/converters/boost/00_boost_pulsim_validation.ipynb": [
        (r"Pulsim  V_o \(mean over last [\d.]+ ms\): " + F, near(23.9874, 0.024))],
    "projects/converters/buck_boost/00_buck_boost_pulsim_validation.ipynb": [
        (r"Pulsim  \|V_o\| \(mean over last [\d.]+ ms\): " + F, near(11.9915, 0.012))],
    "projects/converters/forward/00_forward_pulsim_validation.ipynb": [
        (r"Pulsim  V_o \(time average, last 1 ms\): " + F, near(4.5201, 0.0045))],
    "projects/converters/flyback/00_flyback_pulsim_validation.ipynb": [
        (r"k = 0.99    : V_o = " + F, near(11.0795, 0.011))],
    "projects/converters/half_bridge/00_half_bridge_pulsim_validation.ipynb": [
        (r"Pulsim  V_o \(time average, last [\d.]+ ms\): " + F, near(4.6873, 0.0047))],
    "projects/converters/boost_pfc/00_boost_pfc_pulsim_validation.ipynb": [
        (r"Max envelope error\s+: " + F + "%", below(3.0))],
    "projects/inverters/vsi_3phase/00_vsi_pulsim_validation.ipynb": [
        (r"v_ab fundamental peak \(Pulsim, DFT\): " + F, near(324.86, 0.33))],
    "projects/inverters/npc_3phase/00_npc_pulsim_validation.ipynb": [
        (r"NPC 3-level v_ab THD\s+:\s+" + F, near(0.83, 0.05))],
    "projects/inverters/mmc/00_mmc_pulsim_validation.ipynb": [
        (r"v_ac fundamental peak \(Pulsim, carrier-filtered\): " + F, near(156.95, 0.16))],
    # m3c: the L1 plant carries the Tab 16 load (R + jωL); fundamentals,
    # not staircase peaks (2026-11-13 follow-up of the audit).
    "projects/inverters/m3c_3phase/05_m3c_long_simulation.ipynb": [
        (r"V_ab fundamental: " + F + " kV", near(15.56, 0.08)),
        (r"I_a fundamental: " + F + " A", near(149.69, 0.75)),
        (r"P_in - P_out = " + F + " %", near(0.0, 0.5)),
        (r"Média geral: " + F + " kV", near(24.0, 0.05)),
        (r"Maior desvio instantâneo na janela: " + F, near(10.6, 1.0)),
    ],
    "projects/inverters/m3c_3phase/06_m3c_dbpc.ipynb": [
        (r"45 Hz\s+: \|i_a1\| =\s+" + F, near(100.0, 1.5)),
        (r"0\.5 Hz\s+: \|i_a1\| =\s+" + F, near(100.0, 1.5)),
        (r"DBPC:  \|i_a1\| = .*THD = " + F, near(2.87, 0.3)),
        (r"PI/dq: \|i_a1\| = .*THD = " + F, near(6.28, 0.6)),
    ],
    "projects/inverters/m3c_3phase/07_m3c_motor_ramp.ipynb": [
        (F + r" kV @ t = ", near(11.02, 0.3)),
        (r"f_out nesse instante: " + F, below(5.0)),
        (r"cap mean = " + F + " V", near(24000, 24)),
    ],
    # 24002 -> 23954 V: the plant gained L_load and the DBPC its
    # reference timing (precharge release 1.349 -> 1.372 s).
    "projects/inverters/m3c_3phase/08_m3c_precharge_start.ipynb": [
        (r"V_mean:\s+" + F, near(23954, 24)),
        (r"Released to RUNNING:\s+" + F, near(1.372, 0.02)),
    ],
    "projects/inverters/pfc_vsi_drive/00_pfc_vsi_drive_pulsim_validation.ipynb": [
        (r"OP_2.2\s+P_total\s+" + F, near(57.79, 0.058))],
}


def _values(text, regex):
    return [float(m.group(1)) for m in re.finditer(regex, text)]


def _check(text, checks, where):
    for regex, (lo, hi) in checks:
        vals = _values(text, regex)
        assert vals, f"{where}: no match for {regex!r}"
        assert lo <= vals[0] <= hi, f"{where}: {regex!r} -> {vals[0]} not in [{lo}, {hi}]"


def _env():
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "python"), env.get("PYTHONPATH", "")])
    return env


ALL_SCRIPTS = sorted(p.name for p in SCRIPTS_DIR.glob("*.py")) \
    if SCRIPTS_DIR.is_dir() else []


def test_table_names_existing_examples():
    for name in SCRIPTS:
        assert (SCRIPTS_DIR / name).is_file(), name
    for nb in NOTEBOOKS:
        assert (ROOT / nb).is_file(), nb


@needs_tree
@pytest.mark.parametrize("name", [n for n in ALL_SCRIPTS if n not in INTERACTIVE])
@pytest.mark.examples
@pytest.mark.skipif(not RUN_SLOW, reason="set PULSIM_RUN_EXAMPLES=1")
def test_script_claims(name, tmp_path):
    probe = subprocess.run([sys.executable, "-c", "import pulsim; print(pulsim.__file__)"],
                           env=_env(), cwd=tmp_path, capture_output=True, text=True)
    assert probe.stdout.strip().startswith(str(ROOT)), probe.stdout
    r = subprocess.run([sys.executable, str(SCRIPTS_DIR / name)], env=_env(),
                       cwd=tmp_path, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    for bad in ("Traceback", "RetiredDeviceError", "ALL CLOSED"):
        assert bad not in r.stderr, f"{name}: {bad} in stderr:\n{r.stderr[-3000:]}"
    _check(r.stdout, SCRIPTS.get(name, []), name)


def _outputs_text(nb):
    out = []
    for cell in nb["cells"]:
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream":
                out.append("".join(o.get("text", "")))
            elif "data" in o and "text/plain" in o["data"]:
                out.append("".join(o["data"]["text/plain"]))
            elif o.get("output_type") == "error":
                out.append("Traceback " + o.get("ename", ""))
    return "\n".join(out)


@needs_tree
@pytest.mark.parametrize("nb_path", sorted(NOTEBOOKS))
def test_stored_notebook_outputs_are_current(nb_path):
    nb = json.loads((ROOT / nb_path).read_text(encoding="utf-8"))
    text = _outputs_text(nb)
    assert "Traceback" not in text, f"{nb_path}: a stored output is an error"
    _check(text, NOTEBOOKS[nb_path], nb_path + " (stored outputs)")


@needs_tree
@pytest.mark.parametrize("nb_path", sorted(NOTEBOOKS))
@pytest.mark.examples
@pytest.mark.skipif(not RUN_SLOW, reason="set PULSIM_RUN_EXAMPLES=1")
def test_notebook_claims(nb_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    path = ROOT / nb_path
    nb = nbformat.read(path, as_version=4)
    os.environ.update(MPLBACKEND="Agg",
                      PYTHONPATH=os.pathsep.join([str(ROOT / "python"),
                                                  os.environ.get("PYTHONPATH", "")]))
    nbclient.NotebookClient(nb, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}}).execute()
    _check(_outputs_text(nb), NOTEBOOKS[nb_path], nb_path)
