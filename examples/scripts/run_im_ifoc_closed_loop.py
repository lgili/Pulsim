#!/usr/bin/env python3
"""IM indirect FOC closed-loop with MixedDomainBlockChain and torque-step load.

Closed-loop speed + current Indirect Field-Oriented Control (IFOC) on the
4 kW squirrel-cage :class:`pulsim.InductionMotor` built with
:class:`pulsim.MixedDomainBlockChain`:

  * **Pre-magnetisation** (``0 .. t_mag``): ``omega_ref = 0``, ``i_d_ref``
    builds rotor flux cleanly along the d-axis (``theta_e = 0``).
  * **Speed ramp & tracking** (``t_mag .. t_step``): speed PI regulates
    rotor speed to ``omega_ref`` under initial load ``t_load_init``.
  * **Torque-step disturbance** (``t >= t_step``): mechanical load steps
    ``t_load_init -> t_load_step``. The speed loop raises ``i_q_ref``,
    electromagnetic torque rises to match ``T_load + B * omega_m``, and
    rotor speed recovers to ``omega_ref``.

Control pipeline inside ``MixedDomainBlockChain``::

    omega_ref (LookupTable1D) ──► PI_speed ──► i_q_ref ──► IfocSlip ──► theta_e, omega_e
    i_a, i_b, i_c ──► Clarke ──► Park(theta_e) ──► i_d, i_q
    (i_d_ref, i_d) ──► PI_d ──┐
                               ├──► DqDecouple ──► InvPark ──► InvClarke ──► drive_source(Va, Vb, Vc)
    (i_q_ref, i_q) ──► PI_q ──┘
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import pulsim as p


# ---------- 4 kW reference machine -----------------------------------------

P_RATED = 4_000.0
V_LL = 400.0
F_NOM = 50.0
PP = 2

# ---------- Controller & scenario defaults ---------------------------------

I_D_REF = 8.0
I_Q_MAX = 20.0
V_LIMIT = math.sqrt(2.0) * V_LL / math.sqrt(3.0)

OMEGA_REF_DEFAULT = 100.0
T_MAG_DEFAULT = 0.15
T_STEP_DEFAULT = 0.70
T_END_DEFAULT = 1.20
DT_DEFAULT = 50e-6

T_LOAD_INIT_DEFAULT = 2.0
T_LOAD_STEP_DEFAULT = 12.0
B_VISCOUS_DEFAULT = 0.02

# Current-loop bandwidth 100 Hz; speed-loop natural frequency 30 rad/s.
OMEGA_CL_CURRENT = 2.0 * math.pi * 100.0
OMEGA_CL_SPEED = 30.0
ZETA_SPEED = 0.8


class _IfocSlipIntegrator:
    """Compute IFOC electrical angle and frequency from ``i_q_ref``."""

    def __init__(self, *, pole_pairs: int, t_r: float, i_d_ref: float) -> None:
        self.pp = pole_pairs
        self.t_r = t_r
        self.i_d_ref = i_d_ref
        self.theta_slip = 0.0

    def reset(self) -> None:
        self.theta_slip = 0.0

    def update(
        self,
        *,
        i_q_ref: float,
        theta_m: float,
        omega_m: float,
        dt: float,
    ) -> tuple[float, float]:
        omega_sl = i_q_ref / (self.t_r * self.i_d_ref) if self.i_d_ref > 1e-6 else 0.0
        self.theta_slip += omega_sl * dt
        theta_e = self.pp * theta_m + self.theta_slip
        omega_e = self.pp * omega_m + omega_sl
        return (theta_e, omega_e)


class _DqDecoupler:
    """Add dq cross-coupling feedforward terms and clamp to ``[-v_limit, v_limit]``."""

    def __init__(self, *, l_s: float, l_sigma_s: float, v_limit: float) -> None:
        self.l_s = l_s
        self.l_sigma_s = l_sigma_s
        self.v_limit = v_limit

    def reset(self) -> None:
        pass

    def update(
        self,
        *,
        v_d_pi: float,
        v_q_pi: float,
        omega_e: float,
        i_d: float,
        i_q: float,
    ) -> tuple[float, float]:
        v_d = max(
            -self.v_limit,
            min(self.v_limit, v_d_pi - omega_e * self.l_sigma_s * i_q),
        )
        v_q = max(
            -self.v_limit,
            min(self.v_limit, v_q_pi + omega_e * self.l_s * i_d),
        )
        return (v_d, v_q)


def run_ifoc_torque_step(
    *,
    t_end: float = T_END_DEFAULT,
    dt: float = DT_DEFAULT,
    omega_ref: float = OMEGA_REF_DEFAULT,
    t_mag: float = T_MAG_DEFAULT,
    t_step: float = T_STEP_DEFAULT,
    t_load_init: float = T_LOAD_INIT_DEFAULT,
    t_load_step: float = T_LOAD_STEP_DEFAULT,
    b_viscous: float = B_VISCOUS_DEFAULT,
    i_d_ref: float = I_D_REF,
    i_q_max: float = I_Q_MAX,
    log_dt: float = 1e-3,
) -> dict[str, Any]:
    """Run closed-loop IFOC of ``InductionMotor`` with a load-torque step.

    Returns a dictionary with sampled waveforms and steady-state / disturbance
    metrics.
    """
    t_wall_0 = time.perf_counter()

    b = p.CircuitBuilder()
    kw = p.im_parameters_from_nameplate(
        P_rated_W=P_RATED, V_LL_V=V_LL, f_Hz=F_NOM, pole_pairs=PP,
    )
    diag = kw.pop("_diagnostics")

    motor = p.add_induction_motor(
        b,
        name="IM",
        phase_nodes=("a", "b", "c"),
        neutral_node="n",
        T_load_fn=lambda t, _w: t_load_step if t >= t_step else t_load_init,
        B=b_viscous,
        **kw,
    )
    b.add_resistor("R_leak", "n", "gnd", 1e6)

    t_r = kw["L_r"] / kw["R_r"]
    l_s = kw["L_s"]
    l_m = kw["L_m"]
    r_s = kw["R_s"]
    sigma = 1.0 - (l_m * l_m) / (l_s * kw["L_r"])
    l_sigma_s = sigma * l_s

    kp_i = l_sigma_s * OMEGA_CL_CURRENT
    ki_i = kp_i * r_s / l_sigma_s

    kt_eff = 1.5 * PP * (l_m * l_m / kw["L_r"]) * i_d_ref
    kp_w = (2.0 * ZETA_SPEED * OMEGA_CL_SPEED * kw["J"]) / kt_eff
    ki_w = (OMEGA_CL_SPEED * OMEGA_CL_SPEED * kw["J"]) / kt_eff

    chain = p.MixedDomainBlockChain()
    chain.drive_source(b, "v_a", ("v", "a", "gnd"))
    chain.drive_source(b, "v_b", ("v", "b", "gnd"))
    chain.drive_source(b, "v_c", ("v", "c", "gnd"))

    t_ramp_end = min(t_mag + 0.10, t_step - 0.05)
    chain.add(
        "omega_ref",
        p.LookupTable1D(
            xs=(0.0, t_mag, t_ramp_end, t_end),
            ys=(0.0, 0.0, omega_ref, omega_ref),
        ),
        inputs=dict(x="time"),
        output="omega_ref",
    )
    chain.add(
        "pi_speed",
        p.PIController(
            Kp=kp_w, Ki=ki_w, output_min=-i_q_max, output_max=i_q_max,
        ),
        inputs=dict(setpoint="channel:omega_ref", measured="channel:omega_m", dt="dt"),
        output="i_q_ref",
    )
    chain.add(
        "ifoc_angle",
        p.PythonBlock(_IfocSlipIntegrator(pole_pairs=PP, t_r=t_r, i_d_ref=i_d_ref)),
        inputs=dict(
            i_q_ref="channel:i_q_ref",
            theta_m="channel:theta_m",
            omega_m="channel:omega_m",
            dt="dt",
        ),
        output=("theta_e", "omega_e"),
    )
    chain.add(
        "clarke",
        p.ClarkeTransform(),
        inputs=dict(a="channel:i_a", b="channel:i_b", c="channel:i_c"),
        output=("i_alpha", "i_beta", "i_zero"),
    )
    chain.add(
        "park",
        p.ParkTransform(),
        inputs=dict(
            alpha="channel:i_alpha",
            beta="channel:i_beta",
            theta="channel:theta_e",
        ),
        output=("i_d", "i_q"),
    )
    chain.add(
        "pi_d",
        p.PIController(
            Kp=kp_i, Ki=ki_i, output_min=-V_LIMIT, output_max=V_LIMIT,
        ),
        inputs=dict(setpoint=i_d_ref, measured="channel:i_d", dt="dt"),
        output="v_d_pi",
    )
    chain.add(
        "pi_q",
        p.PIController(
            Kp=kp_i, Ki=ki_i, output_min=-V_LIMIT, output_max=V_LIMIT,
        ),
        inputs=dict(setpoint="channel:i_q_ref", measured="channel:i_q", dt="dt"),
        output="v_q_pi",
    )
    chain.add(
        "decouple",
        p.PythonBlock(_DqDecoupler(l_s=l_s, l_sigma_s=l_sigma_s, v_limit=V_LIMIT)),
        inputs=dict(
            v_d_pi="channel:v_d_pi",
            v_q_pi="channel:v_q_pi",
            omega_e="channel:omega_e",
            i_d="channel:i_d",
            i_q="channel:i_q",
        ),
        output=("v_d", "v_q"),
    )
    chain.add(
        "ipark",
        p.InverseParkTransform(),
        inputs=dict(
            d="channel:v_d", q="channel:v_q", theta="channel:theta_e",
        ),
        output=("v_alpha", "v_beta"),
    )
    chain.add(
        "iclarke",
        p.InverseClarkeTransform(),
        inputs=dict(alpha="channel:v_alpha", beta="channel:v_beta", zero=0.0),
        output=("v_a", "v_b", "v_c"),
    )

    chain_obs = chain.make_step_observer(b, dt=dt)
    set_ch = chain._cxx_chain.set_channel
    get_ch = chain._cxx_chain.get_channel
    im_obs, im_b_extra = p.make_induction_motor_observer(b, motor, dt=dt)

    i_a_idx = b.pool.branch_var_id_for_inductor(motor.phase_branch_ids[0], b.graph)
    i_b_idx = b.pool.branch_var_id_for_inductor(motor.phase_branch_ids[1], b.graph)
    i_c_idx = b.pool.branch_var_id_for_inductor(motor.phase_branch_ids[2], b.graph)

    log_t: list[float] = []
    log_omega: list[float] = []
    log_omega_ref: list[float] = []
    log_tem: list[float] = []
    log_tload: list[float] = []
    log_psi: list[float] = []
    log_id: list[float] = []
    log_iq: list[float] = []

    def combined_observer(t: float, x) -> None:
        im_obs(t, x)
        set_ch("i_a", float(x[i_a_idx]))
        set_ch("i_b", float(x[i_b_idx]))
        set_ch("i_c", float(x[i_c_idx]))
        set_ch("omega_m", motor.mech.omega_rad_s)
        set_ch("theta_m", motor.mech.theta_rad)
        chain_obs(t, x)

        if not log_t or t - log_t[-1] >= log_dt:
            log_t.append(t)
            log_omega.append(motor.mech.omega_rad_s)
            log_omega_ref.append(float(get_ch("omega_ref")))
            log_tem.append(motor.last_T_em_Nm)
            log_tload.append(t_load_step if t >= t_step else t_load_init)
            log_psi.append(math.hypot(motor.psi_r_alpha_Wb, motor.psi_r_beta_Wb))
            log_id.append(float(get_ch("i_d")))
            log_iq.append(float(get_ch("i_q")))

    combined_observer._b_extra_fn = chain_obs._b_extra_fn  # type: ignore[attr-defined]

    res = p.simulate(
        b,
        t_end=t_end,
        dt=dt,
        switch_fn=lambda _t: p.SwitchStateMask(0),
        step_observer=combined_observer,
        b_extra_fn=im_b_extra,
    )

    wall_time_s = time.perf_counter() - t_wall_0

    t_arr = np.asarray(log_t)
    omega_arr = np.asarray(log_omega)
    omega_ref_arr = np.asarray(log_omega_ref)
    tem_arr = np.asarray(log_tem)
    tload_arr = np.asarray(log_tload)
    psi_arr = np.asarray(log_psi)
    id_arr = np.asarray(log_id)
    iq_arr = np.asarray(log_iq)

    pre_mask = (t_arr >= t_step - 0.10) & (t_arr < t_step)
    dip_mask = (t_arr >= t_step) & (t_arr <= min(t_end, t_step + 0.25))
    post_mask = t_arr >= t_end - 0.10

    pre_omega = float(np.mean(omega_arr[pre_mask]))
    post_omega = float(np.mean(omega_arr[post_mask]))
    dip_min_omega = float(np.min(omega_arr[dip_mask]))
    pre_tem = float(np.mean(tem_arr[pre_mask]))
    post_tem = float(np.mean(tem_arr[post_mask]))

    return {
        "traces": {
            "t": t_arr,
            "omega_m": omega_arr,
            "omega_ref": omega_ref_arr,
            "T_em": tem_arr,
            "T_load": tload_arr,
            "psi_r": psi_arr,
            "i_d": id_arr,
            "i_q": iq_arr,
        },
        "metrics": {
            "omega_ref": omega_ref,
            "omega_sync": float(diag["omega_sync_rad_s"]),
            "t_r_s": float(t_r),
            "sigma": float(sigma),
            "l_sigma_s_H": float(l_sigma_s),
            "kp_i": float(kp_i),
            "ki_i": float(ki_i),
            "kp_w": float(kp_w),
            "ki_w": float(ki_w),
            "pre_step_omega_mean": pre_omega,
            "dip_min_omega": dip_min_omega,
            "post_step_omega_mean": post_omega,
            "recovery_error_rad_s": abs(post_omega - omega_ref),
            "pre_step_torque_mean": pre_tem,
            "post_step_torque_mean": post_tem,
            "expected_pre_torque": t_load_init + b_viscous * omega_ref,
            "expected_post_torque": t_load_step + b_viscous * omega_ref,
            "psi_r_ss": float(np.mean(psi_arr[post_mask])),
            "i_d_ss": float(np.mean(id_arr[post_mask])),
            "i_q_pre": float(np.mean(iq_arr[pre_mask])),
            "i_q_post": float(np.mean(iq_arr[post_mask])),
            "steps": int(res.num_steps()),
            "wall_time_s": float(wall_time_s),
        },
    }


def _save_plot(traces: dict[str, np.ndarray], save_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_ms = traces["t"] * 1e3
    fig, (ax_w, ax_t, ax_i) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax_w.plot(t_ms, traces["omega_m"], label="omega_m [rad/s]", lw=1.2)
    ax_w.plot(t_ms, traces["omega_ref"], "--", label="omega_ref [rad/s]", lw=1.0)
    ax_w.set_ylabel("Speed [rad/s]")
    ax_w.grid(alpha=0.3)
    ax_w.legend(loc="lower right")
    ax_w.set_title("Induction Motor IFOC (MixedDomainBlockChain) — Load Torque Step")

    ax_t.plot(t_ms, traces["T_em"], label="T_em [Nm]", lw=1.2)
    ax_t.plot(t_ms, traces["T_load"], "--", label="T_load [Nm]", lw=1.0)
    ax_t.set_ylabel("Torque [Nm]")
    ax_t.grid(alpha=0.3)
    ax_t.legend(loc="upper left")

    ax_i.plot(t_ms, traces["i_d"], label="i_d [A]", lw=1.1)
    ax_i.plot(t_ms, traces["i_q"], label="i_q [A]", lw=1.1)
    ax_i.set_xlabel("Time [ms]")
    ax_i.set_ylabel("dq currents [A]")
    ax_i.grid(alpha=0.3)
    ax_i.legend(loc="upper left")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        nargs="?",
        const=str(Path(__file__).resolve().parent / "output" / "im_ifoc_torque_step.png"),
        default=None,
        help="Optional output PNG path (never calls plt.show()).",
    )
    args = parser.parse_args(argv)

    out = run_ifoc_torque_step()
    tr = out["traces"]
    m = out["metrics"]

    print("IM IFOC closed-loop with MixedDomainBlockChain (torque-step disturbance):")
    print(f"  Plant: 4 kW / 400 V / 50 Hz / 4-pole IM (Tr={m['t_r_s']*1e3:.2f} ms, sigma={m['sigma']:.3f})")
    print(f"  Current PI : Kp={m['kp_i']:.2f}, Ki={m['ki_i']:.1f}")
    print(f"  Speed   PI : Kp={m['kp_w']:.3f}, Ki={m['ki_w']:.2f}")
    print(f"  Pre-step   : omega_m={m['pre_step_omega_mean']:.2f} rad/s (ref {m['omega_ref']:.1f}), "
          f"T_em={m['pre_step_torque_mean']:.2f} Nm (exp {m['expected_pre_torque']:.2f}), "
          f"i_q={m['i_q_pre']:.2f} A")
    print(f"  Step dip   : min omega_m={m['dip_min_omega']:.2f} rad/s")
    print(f"  Post-step  : omega_m={m['post_step_omega_mean']:.2f} rad/s (err {m['recovery_error_rad_s']:.3f}), "
          f"T_em={m['post_step_torque_mean']:.2f} Nm (exp {m['expected_post_torque']:.2f}), "
          f"i_q={m['i_q_post']:.2f} A")
    print(f"  Flux / i_d : |psi_r|={m['psi_r_ss']:.3f} Wb, i_d={m['i_d_ss']:.3f} A")
    print(f"  Steps      : {m['steps']:,} in {m['wall_time_s']:.3f} s")

    # Verify closed-loop tracking and load-disturbance rejection.
    assert abs(m["pre_step_omega_mean"] - m["omega_ref"]) < 1.0
    assert tr["T_load"][-1] > tr["T_load"][0]
    assert m["post_step_torque_mean"] > m["pre_step_torque_mean"] + 5.0
    assert m["dip_min_omega"] < m["pre_step_omega_mean"]
    assert m["recovery_error_rad_s"] < 1.0

    if args.save is not None:
        save_path = Path(args.save)
        _save_plot(tr, save_path)
        print(f"  Plot saved : {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
