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
    # Was 75.09 / 165.4 / 24.43 W from the primary current (which carries
    # the reflected load) on a core whose B was 0.42 T against the
    # circuit's 23 mT. Now i_m on ETD49, N = 3: only ferrite matches.
    "run_psfb_ja_loss_analysis.py": [
        (r"P_out \(last period\)\s+= " + F, near(23.6, 0.3)),
        (r"circuit B peak, N = 3\s+= " + F, near(0.123, 0.002)),
        (r"ferrite_n87\s+[\d.]+ T\s+" + F, near(0.83, 0.02)),
        (r"ferrite_n87\s+[\d.]+ T\s+[\d.]+\s+" + F + " W", near(18.05, 0.2)),
        (r"permalloy\s+[\d.]+ T\s+" + F, above(3.0)),
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
        (r"V_out médio: " + F + "V", near(11.9205, 0.06)),
        # Diode conduction is V_T0·I_avg + Rd·I_rms² (was V_T0·I_avg alone:
        # 2.812 W, η 92.05 %); the resistive term at I_rms = 4.35 A adds 0.38 W.
        (r"A, RMS: [\d.]+A\nPerda de condução: " + F + "W", near(3.190, 0.003)),
        (r"EFICIÊNCIA: " + F + "%", near(91.52, 0.01))],
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
    # Integration (feat/core-integration): the MOSFET body-diode V_F is now a
    # conduction drop, not a turn-on threshold; old value reproduced exactly
    # with the old semantics.
    # 4.5201 -> 4.5195.
    "projects/converters/forward/00_forward_pulsim_validation.ipynb": [
        (r"Pulsim  V_o \(time average, last 1 ms\): " + F, near(4.5195, 0.0045))],
    "projects/converters/flyback/00_flyback_pulsim_validation.ipynb": [
        (r"k = 0.99    : V_o = " + F, near(11.0787, 0.011))],  # 11.0795 -> 11.0787, body-diode V_F (above)
    "projects/converters/half_bridge/00_half_bridge_pulsim_validation.ipynb": [
        (r"Pulsim  V_o \(time average, last [\d.]+ ms\): " + F, near(4.6812, 0.0047))],  # 4.6873 -> 4.6812, body-diode V_F (above)
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
    # mmc_3phase: i_a used to read an arm current (stale state index):
    # THD 81.7 -> 1.715 %. The v_C ripple is the arm energy balance with
    # the simulated 2f circulating current (185.5 V vs simulated 184.8 V).
    "projects/inverters/mmc_3phase/01_mmc_validation_gean.ipynb": [
        (r"i_a peak \[A\]\s+" + F, near(26.64, 0.27)),
        (r"THD\(i_a\) \[%\]\s+" + F, near(1.715, 0.1)),
        (r"RMS\(CA\(i_circ,a\)\) \[A\]\s+" + F, near(7.54, 0.15)),
        (r"v_C pkpk médio \[V\]\s+" + F, near(184.8, 3.7)),
        (r"sem i_circ\s+:\s+" + F, near(291.4, 0.3)),
        (r"com a 2ω simulada :\s+" + F, near(185.5, 3.7)),
        (r"mínimo sobre 2ω\s+:\s+" + F, near(178.2, 0.2)),
    ],
    "projects/inverters/mmc_3phase/02_mmc_closed_loop_dq.ipynb": [
        (r"i_d post-step \(150-200 ms\) \[A\]\s+" + F, near(15.0, 0.15))],
    # +energy loop i_d_pre -0.587 -> 2.001 A: the v_C low-pass started at
    # 0 V and wound the outer integrator to a -2.6 A bias.
    "projects/inverters/mmc_3phase/03_mmc_advanced_control.ipynb": [
        (r"\+ energy loop\s+" + F, near(2.001, 0.02))],
    "projects/inverters/mmc_3phase/04_mmc_igbt_level1.ipynb": [
        (r"Tese-calibrado\s+(\d+\.\d\d)\s", near(26.64, 0.27))],
    "projects/inverters/mmc_3phase/05_mmc_baseline_validation.ipynb": [
        (r"Tier 2[\s\S]*?PASSED: (\d+)/3", near(3, 0)),
        (r"THD L0=" + F, near(1.747, 0.1)),
    ],
    # 57.79 -> 74.27 W: no 470 uF on rect_p any more (line PF 0.62 ->
    # 0.997) and the bridge loss counts all four diodes on the simulated
    # line current; L001 copper now within 1 % of PSIM at OP 2.3.
    # 74.27 -> 74.18 W and P_ohm_L001 2.838 -> 2.835 W: body-diode V_F is a
    # conduction drop after the integration merge (reproduced exactly).
    "projects/inverters/pfc_vsi_drive/00_pfc_vsi_drive_pulsim_validation.ipynb": [
        (r"OP_2.2\s+P_total\s+" + F, near(74.18, 0.074)),
        (r"OP_2.3\s+P_ohm_L001\s+" + F, near(2.835, 0.028)),
        (r"OP_2.3\s+eta_inverter\s+" + F, near(0.956, 0.001))],
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


# ---------------------------------------------------------------------------
# YAML examples with no driver script: each row loads the file, runs it the
# way its header says, and measures one number against its closed form.
# ---------------------------------------------------------------------------

def _yaml_run(name, switch_fn=None, t_end=None):
    import numpy as np
    import pulsim as p
    lc = p.load_yaml_file(str(ROOT / "examples" / name))
    b = lc.builder
    sw = switch_fn(p, b) if switch_fn else None
    r = (p.simulate(lc, switch_fn=sw) if t_end is None else
         p.simulate(b, t_end=t_end, dt=lc.options.dt, switch_fn=sw))
    t, X = np.asarray(r.times), np.asarray(r.states)
    assert np.all(np.isfinite(X)), name
    return b, r, t, X


def _pwm(name, f=20e3, d=0.5):
    return lambda p, b: p.make_pwm_switch_fn(
        frequency=f, duty=d, switch_idx=b.switch_index_of(name),
        num_switches=b.graph.num_switches)


def _interleaved(p, b, T=50e-6):
    i0, i1, n = b.switch_index_of("S0"), b.switch_index_of("S1"), b.graph.num_switches

    def sw(t):
        m = p.SwitchStateMask(n)
        m.set(i0, (t % T) < 0.5 * T)
        m.set(i1, ((t + 0.5 * T) % T) < 0.5 * T)
        return m
    return sw


def _dead_time_pair(p, b):
    return p.make_dead_time_pwm_pair_fn(
        frequency=20e3, duty=0.5, hs_switch_idx=b.switch_index_of("S_hi"),
        ls_switch_idx=b.switch_index_of("S_lo"),
        num_switches=b.graph.num_switches, dead_time=1e-6)


def _dcm_buck(D, L, R, T):
    import math
    K = 2 * L / (R * T)
    return 2 / (1 + math.sqrt(1 + 4 * K / D**2))


def _yaml_value(key):
    import math
    import numpy as np
    import pulsim as p
    if key == "rc_circuit.yaml":            # 5 V, 1 kΩ, 1 µF at t_end
        b, r, t, X = _yaml_run(key)
        return X[-1, b.node_id_of("out")], 5 * (1 - math.exp(-t[-1] / 1e-3)), 1e-4
    if key == "rl_circuit.yaml":            # 10 V, 100 Ω, 10 mH at t_end
        b, r, t, X = _yaml_run(key)
        return np.asarray(r.i("L1"))[-1], 0.1 * (1 - math.exp(-t[-1] / 1e-4)), 1e-5
    if key == "rlc_circuit.yaml":           # settles to the 10 V source
        b, r, t, X = _yaml_run(key, t_end=20e-3)
        return X[-1, b.node_id_of("out")], 10.0, 0.05
    if key == "rlc_step_response.yaml":     # header: peak 18.5 V (ζ = 0.05)
        b, r, t, X = _yaml_run(key)
        z = 0.05
        return (X[:, b.node_id_of("vc")].max(),
                10 * (1 + math.exp(-math.pi * z / math.sqrt(1 - z * z))), 0.01)
    if key == "buck_converter.yaml":        # header: DCM, 25.8 V
        b, r, t, X = _yaml_run(key, switch_fn=_pwm("S1"), t_end=20e-3)
        v = p.time_average(t, X[:, b.node_id_of("out")], t_from=18e-3)
        return v, 48 * _dcm_buck(0.5, 100e-6, 10.0, 50e-6), 0.01 * 25.8
    if key == "stress_buck_switching.yaml":  # header: 2-phase DCM, 28.2 V
        b, r, t, X = _yaml_run(key, switch_fn=_interleaved)
        v = p.time_average(t, X[:, b.node_id_of("out")], t_from=8e-3)
        return v, 48 * _dcm_buck(0.5, 150e-6, 20.0, 50e-6), 0.01 * 28.2
    if key == "half_bridge_pwm.yaml":       # ±50 V square into 10 Ω + 1 mH
        b, r, t, X = _yaml_run(key, switch_fn=_dead_time_pair)
        i = np.asarray(r.i("Lload"))[t >= t[-1] - 50e-6]
        return 0.5 * np.ptp(i), 5 * math.tanh(25e-6 / 2e-4), 0.02 * 0.622
    # The rest state no number: they must run, finite.
    _yaml_run(key)
    return 0.0, 0.0, 0.0


YAMLS = ["rc_circuit.yaml", "rl_circuit.yaml", "rlc_circuit.yaml",
         "rlc_step_response.yaml", "buck_converter.yaml",
         "stress_buck_switching.yaml", "half_bridge_pwm.yaml", "rc_pulse.yaml",
         "stress_large_rc_ladder.yaml", "stress_large_resistive_ladder.yaml",
         "hysteretic_inductor_demo.yaml", "induction_motor_dol.yaml"]


@needs_tree
@pytest.mark.parametrize("name", YAMLS)
@pytest.mark.examples
@pytest.mark.skipif(not RUN_SLOW, reason="set PULSIM_RUN_EXAMPLES=1")
def test_yaml_claims(name):
    measured, expected, tol = _yaml_value(name)
    assert abs(measured - expected) <= tol, f"{name}: {measured} vs {expected} ± {tol}"


@needs_tree
def test_v1_yaml_is_refused():
    import pulsim as p
    with pytest.raises(Exception, match="(?i)v1|schema|legacy"):
        p.load_yaml_file(str(ROOT / "examples" / "refrigerator_compressor_cc.yaml"))
