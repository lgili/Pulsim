"""Cross-validation of pulsim M3C against Gili (2024) thesis.

Phase 22.10. Runs the L1 plant at the Tab. 16 operating point
(parameters identical to the thesis HIL/OPAL-RT experiment), computes
key metrics, and reports whether each is within the thesis-claimed
range.

The thesis Cap 7 (Figs 87-122) presents HIL waveforms but does NOT
give explicit numerical THD or step-response times. We compare:

  * Output fundamental current at each Tab. 16 frequency
    (5, 30, 45, 50, 55 Hz) against Ohm's law expectation.
  * Output THD at 45 Hz (thesis baseline) — should be low (≤ 25 %)
    after the multilevel modulation.
  * Input current at 50 Hz under power transfer (i_d_in > 0).
  * Capacitor voltage balance — mean should stay near
    N · v_cap_nominal = 24 kV with bounded spread.

References:
  * Gili (2024) Tab. 16 (HIL parameters)
  * Gili (2024) Figs 87-98 (steady-state waveforms)
  * Gili (2024) Figs 99-104 (power step responses)
  * Gili (2024) Figs 105-109 (cap voltage tracking)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from m3c_3phase_model import (
    M3cParams,
    abc_to_dq,
    build_l1_plant,
    predict_i_out_peak,
    run_l1_cost_loop,
    run_l1_dq_full_closed_loop_with_cap_loop,
    thd,
)


@dataclass
class M3cValidationMetrics:
    """Summary of one validation run at a Tab. 16 operating point."""

    f_out: float
    i_a_out_peak: float
    i_a_out_peak_predicted: float
    i_a_out_rms: float
    thd_pct: float
    i_a_in_peak: float
    v_caps_mean: float
    v_caps_spread: float
    n_distinct_configs: int

    @property
    def peak_rel_err(self) -> float:
        return abs(
            self.i_a_out_peak - self.i_a_out_peak_predicted
        ) / self.i_a_out_peak_predicted


def run_validation_at_freq(
    f_out: float,
    *,
    m_v: float = 1.0,
    t_end: float = 250e-3,
    dt: float = 25e-6,
    closed_loop: bool = False,
) -> M3cValidationMetrics:
    """Run the M3C L1 plant at the given output frequency and return
    quantitative metrics.

    Parameters
    ----------
    f_out
        Output frequency [Hz]. Tab. 16 lists 5, 30, 45, 50, 55 Hz.
    m_v
        Output modulation index (default 1.0 for full-amplitude
        Ohm's-law-match baseline).
    t_end, dt
        Simulation horizon and integration step.
    closed_loop
        If False (default) use the Phase 22.6 cost-function loop
        (open-loop SVM); if True use the Phase 22.9 dq + cap-outer
        loop with i_d_out_ref = predicted peak.
    """
    params = M3cParams(f_out=f_out, m_v=m_v)
    plant = build_l1_plant(params)

    if closed_loop:
        i_d_ref = predict_i_out_peak(params)
        result, ctrl, _, _, _ = run_l1_dq_full_closed_loop_with_cap_loop(
            plant, params,
            i_d_in_ref=0.0,
            i_d_out_ref=i_d_ref, i_q_out_ref=0.0,
            t_end=t_end, dt=dt,
        )
    else:
        result, ctrl = run_l1_cost_loop(plant, params, t_end=t_end, dt=dt)

    # Window for steady-state metrics (last 3 fundamental periods).
    fs = 1.0 / dt
    n_per_period = int(round((1.0 / f_out) * fs))
    if n_per_period * 3 > len(result.i_a_out):
        # Fall back to last half of run.
        ia = result.i_a_out[len(result.i_a_out) // 2:]
    else:
        ia = result.i_a_out[-3 * n_per_period:]
    ib = result.i_b_out[-len(ia):]
    ic = result.i_c_out[-len(ia):]
    ia_in = result.i_a_in[-len(ia):]

    # Compute fundamental from FFT.
    spec = np.fft.rfft(ia)
    k1 = int(round(f_out * len(ia) / fs))
    fund_pk = 2.0 * float(np.abs(spec[k1])) / len(ia)

    return M3cValidationMetrics(
        f_out=f_out,
        i_a_out_peak=fund_pk,
        i_a_out_peak_predicted=predict_i_out_peak(params),
        i_a_out_rms=float(np.sqrt(np.mean(ia ** 2))),
        thd_pct=thd(ia, fs, f_out),
        i_a_in_peak=float(np.max(np.abs(ia_in))),
        v_caps_mean=float(np.mean(ctrl.v_caps_module)),
        v_caps_spread=float(
            max(ctrl.v_caps_module) - min(ctrl.v_caps_module)
        ),
        n_distinct_configs=len(
            set(c.grid for c in ctrl.chosen_configs)
        ),
    )


def run_tab16_sweep(
    frequencies: Iterable[float] = (5.0, 30.0, 45.0, 50.0, 55.0),
    **kwargs,
) -> dict[float, M3cValidationMetrics]:
    """Run the full Tab. 16 frequency sweep.

    Returns a dict ``{f_out: metrics}`` ready for table display."""
    return {f: run_validation_at_freq(f, **kwargs) for f in frequencies}


def format_validation_table(
    results: dict[float, M3cValidationMetrics],
) -> str:
    """Pretty-print the validation results as a markdown-style table."""
    header = (
        "| f_out [Hz] | i_a_pk [A] | predicted [A] | rel-err [%] |"
        " THD [%] | v_caps mean [V] | v_caps spread [V] |"
    )
    sep = (
        "|-----------:|-----------:|--------------:|------------:|"
        "--------:|----------------:|------------------:|"
    )
    rows = [header, sep]
    for f, m in sorted(results.items()):
        rows.append(
            f"| {f:9.1f} | {m.i_a_out_peak:9.2f} | "
            f"{m.i_a_out_peak_predicted:12.2f} | "
            f"{m.peak_rel_err*100:10.2f} | {m.thd_pct:6.2f} | "
            f"{m.v_caps_mean:14.0f} | {m.v_caps_spread:16.0f} |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    print("Running M3C validation sweep (Tab. 16 frequencies)...")
    print()
    results = run_tab16_sweep()
    print(format_validation_table(results))
