"""Pulsim — sensorless rotor observers (Phase 2.3).

Plug-in :class:`MixedDomainBlockChain` blocks that estimate the rotor
electrical angle and mechanical speed of an AC machine from stator
voltage commands and current measurements alone — no shaft encoder
required.

Two algorithms ship in Phase 2.3:

* :class:`SlidingModeObserver` — for **PMSM** (and other synchronous
  machines with permanent-magnet back-EMF). Uses a stator-current
  sliding-mode observer + first-order low-pass filter on the
  equivalent-control output + a small PLL to extract `θ̂_e` and
  `ω̂_e` from the back-EMF estimate. Works well from ~10 % rated
  speed up; below that the back-EMF magnitude becomes too small
  for reliable observation and the surfaced `low_speed_flag`
  output should be acted upon by the caller.

* :class:`FluxMRASObserver` — for **induction motor** (see
  :class:`pulsim.InductionMotor`). Compares the voltage-model
  rotor-flux estimate to the current-model estimate and drives
  the rotor-speed adaptation loop. Effective from ~10 % rated
  speed up.

Both classes follow the standard ``BlockChain`` block interface
(``.reset()``, ``.update(*, ...) → tuple``) so they wire in via:

    chain.add("smo", p.SlidingModeObserver(Rs=..., Ls=...),
               inputs=dict(v_alpha="channel:v_alpha",
                            v_beta="channel:v_beta",
                            i_alpha="channel:i_alpha",
                            i_beta="channel:i_beta", dt="dt"),
               output=("theta_hat", "omega_hat",
                       "e_hat_alpha", "e_hat_beta",
                       "low_speed_flag"))

These observers are intentionally pure-Python and live in the
control layer — the simulator kernel does not need to know about
them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


__all__ = [
    "SlidingModeObserver",
    "FluxMRASObserver",
]


# =============================================================================
# Sliding-Mode Observer (SMO) for PMSM
# =============================================================================

@dataclass
class SlidingModeObserver:
    """Sliding-mode rotor observer for a PMSM.

    Algorithm (standard textbook form — see Utkin's *Sliding Modes in
    Control and Optimization* §10, or the Texas Instruments InstaSPIN
    application note SPRABQ8):

      1. **Stator-current observer** in α-β:
         dî_α/dt = (1/Ls)·(v_α − Rs·î_α − z_α)
         dî_β/dt = (1/Ls)·(v_β − Rs·î_β − z_β)
         where ``z = K_sl · sat(î − i, ε)`` is the sliding-mode
         injection (``sat`` = saturating sign-with-boundary-layer
         that suppresses chattering).
      2. **Equivalent-control LPF** — when the observer is "on the
         sliding surface" the time-average of `z` equals the
         back-EMF; a first-order low-pass filter at ``ω_lpf``
         extracts the average:
         ê_α ← LPF(z_α), ê_β ← LPF(z_β).
      3. **Angle PLL** — drive the q-axis projection of the
         back-EMF estimate to zero. With the PMSM convention
         ``e_α = −E_peak · sin(θ_e), e_β = +E_peak · cos(θ_e)``,
         the q-axis projection of ``(−e_α, e_β)`` onto the
         observer's frame is ``e_α · sin(θ̂) + e_β · cos(θ̂)``.
         A small PI loop drives that signal to zero and outputs
         ``θ̂_e`` and ``ω̂_e``.

    Tuning rules of thumb:
      * ``K_sl`` ≳ peak back-EMF magnitude (volts). Higher gain
        = faster lock but more chattering.
      * ``omega_lpf`` ≈ 10 × rated electrical frequency.
      * ``Kp_pll`` and ``Ki_pll`` give a PLL bandwidth of about
        ``sqrt(Ki_pll)`` rad/s with damping ``Kp/(2·sqrt(Ki))``.
        The defaults match a ~100 Hz PLL bandwidth, critically
        damped — adequate for 50/60 Hz fundamental motors.

    Failure modes (documented honesty):
      * Below ~10 % rated speed the back-EMF magnitude collapses
        toward `K_sl` and the observer can lose lock. The
        ``low_speed_flag`` output goes True when the back-EMF
        magnitude falls below ``low_speed_threshold``.
      * Wrong ``Rs`` / ``Ls`` parameters produce a constant
        steady-state angle offset proportional to the mismatch.
    """

    Rs: float
    """Per-phase stator resistance [Ω]."""
    Ls: float
    """Per-phase stator inductance [H] (synchronous, Ld=Lq assumed)."""
    K_sl: float = 50.0
    """Sliding-mode injection gain [V]. Set ≳ peak back-EMF."""
    omega_lpf: float = 2.0 * math.pi * 500.0
    """Low-pass-filter cutoff [rad/s] for the equivalent-control
    signal. Default 500 Hz."""
    Kp_pll: float = 200.0
    """PLL proportional gain."""
    Ki_pll: float = 1.0e4
    """PLL integral gain."""
    f_init_hz: float = 50.0
    """Initial frequency guess for the PLL [Hz]. Set close to
    expected nominal electrical frequency for faster lock."""
    epsilon: float = 0.5
    """Boundary-layer width for the saturating-sign function
    (in current-error units, A). Reduces chattering at the cost
    of a steady-state angle ripple proportional to ``epsilon /
    K_sl``."""
    low_speed_threshold: float = 0.05
    """Back-EMF magnitude (V) below which the observer raises the
    ``low_speed_flag`` output. Default 0.05 V — appropriate for
    a small machine; scale with rated peak EMF."""

    # Live state — initialized inside __post_init__ / reset.
    i_hat_alpha: float = field(default=0.0, init=False, repr=False)
    i_hat_beta:  float = field(default=0.0, init=False, repr=False)
    e_hat_alpha: float = field(default=0.0, init=False, repr=False)
    e_hat_beta:  float = field(default=0.0, init=False, repr=False)
    theta_hat:   float = field(default=0.0, init=False, repr=False)
    omega_hat:   float = field(default=0.0, init=False, repr=False)
    _pll_integ:  float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.i_hat_alpha = 0.0
        self.i_hat_beta = 0.0
        self.e_hat_alpha = 0.0
        self.e_hat_beta = 0.0
        self.theta_hat = 0.0
        self.omega_hat = 2.0 * math.pi * self.f_init_hz
        self._pll_integ = 0.0

    def update(self, *,
                 v_alpha: float, v_beta: float,
                 i_alpha: float, i_beta: float,
                 dt: float) -> tuple:
        """Run one observer step. Returns
        ``(theta_hat, omega_hat, e_hat_alpha, e_hat_beta,
        low_speed_flag)``."""
        # ---- 1. Sliding-mode current observer ---------------------------
        err_a = self.i_hat_alpha - i_alpha
        err_b = self.i_hat_beta  - i_beta
        # Saturating sign (boundary-layer width = epsilon) — replaces the
        # discontinuous sign() with a smooth ramp that asymptotes to ±1
        # outside the layer. Eliminates chattering at the cost of finite
        # gain inside the layer.
        z_a = self.K_sl * math.tanh(err_a / self.epsilon)
        z_b = self.K_sl * math.tanh(err_b / self.epsilon)

        di_hat_a = (v_alpha - self.Rs * self.i_hat_alpha - z_a) / self.Ls
        di_hat_b = (v_beta  - self.Rs * self.i_hat_beta  - z_b) / self.Ls
        self.i_hat_alpha += di_hat_a * dt
        self.i_hat_beta  += di_hat_b * dt

        # ---- 2. Low-pass equivalent-control extraction ------------------
        # First-order LPF discretized by forward-Euler: ê ← ê + α·(z − ê).
        # α = ω_lpf · dt, clamped to [0, 1] for stability with coarse dt.
        alpha_lpf = max(0.0, min(1.0, self.omega_lpf * dt))
        self.e_hat_alpha += alpha_lpf * (z_a - self.e_hat_alpha)
        self.e_hat_beta  += alpha_lpf * (z_b - self.e_hat_beta)

        # ---- 3. PLL — drive the d-axis projection of the back-EMF
        # estimate to zero. PMSM back-EMF convention (matched to
        # ``pulsim.add_pmsm``'s observer):
        #     e_α = −E_peak · sin(θ_e)
        #     e_β = +E_peak · cos(θ_e)
        # The back-EMF vector therefore points at angle (θ_e + π/2) —
        # it's the time derivative of the PM flux, leading the rotor
        # by 90° electrical.  The observer's d-axis is aligned with
        # the rotor flux (i.e. with θ_hat); the d-component of the
        # back-EMF vector is:
        #     d̂ =  e_α · cos(θ̂) + e_β · sin(θ̂)
        #       = −E · sin(θ_e − θ̂)         (using the identities)
        # so the *negative* of this projection equals E·sin(θ_e − θ̂)
        # — a clean Goodwin-style phase detector whose zero crossing
        # gives θ̂ = θ_e and whose sign drives the PLL in the
        # right direction.
        c, s = math.cos(self.theta_hat), math.sin(self.theta_hat)
        e_phase = -(self.e_hat_alpha * c + self.e_hat_beta * s)
        self._pll_integ += self.Ki_pll * dt * e_phase
        delta_omega = self.Kp_pll * e_phase + self._pll_integ
        self.omega_hat = 2.0 * math.pi * self.f_init_hz + delta_omega
        self.theta_hat += self.omega_hat * dt
        # Wrap to [0, 2π).
        self.theta_hat = math.fmod(self.theta_hat, 2.0 * math.pi)
        if self.theta_hat < 0.0:
            self.theta_hat += 2.0 * math.pi

        # ---- 4. Low-speed diagnostic ------------------------------------
        e_mag = math.hypot(self.e_hat_alpha, self.e_hat_beta)
        low_speed_flag = 1.0 if e_mag < self.low_speed_threshold else 0.0

        return (self.theta_hat, self.omega_hat,
                  self.e_hat_alpha, self.e_hat_beta,
                  low_speed_flag)


# =============================================================================
# Flux-MRAS Observer for Induction Motor
# =============================================================================

@dataclass
class FluxMRASObserver:
    """Model-Reference Adaptive System (MRAS) speed observer for the
    squirrel-cage induction motor.

    Algorithm (the *rotor-flux* MRAS variant — Schauder 1992):

      1. **Reference model** — voltage-driven rotor-flux estimate
         (does NOT use the speed, hence "reference"):
           ψ_rα^ref = ∫(v_α − Rs·i_α − σ·Ls·di_α/dt) · (Lr/Lm) dt
           ψ_rβ^ref = ∫(v_β − Rs·i_β − σ·Ls·di_β/dt) · (Lr/Lm) dt

      2. **Adaptive model** — current-driven rotor-flux estimate
         (uses the speed estimate ω̂):
           dψ_rα^adj/dt = (Lm/Tr)·i_α − ψ_rα^adj/Tr − ω̂·ψ_rβ^adj
           dψ_rβ^adj/dt = (Lm/Tr)·i_β − ψ_rβ^adj/Tr + ω̂·ψ_rα^adj

      3. **Adaptation loop** — Lyapunov-derived speed update from
         the cross-product mismatch:
           ε = ψ_rβ^ref · ψ_rα^adj − ψ_rα^ref · ψ_rβ^adj
           ω̂ ← ω̂ + (Kp_mras · ε + Ki_mras · ∫ε)

    The MRAS works in the speed range where the voltage-model
    integration is well-conditioned — typically above ~10 % of
    rated speed. Below that, the integrator drifts because the
    stator drop (Rs·is) becomes comparable to the integrand and
    small parameter errors get integrated indefinitely.

    Parameters mirror :class:`InductionMotor`. Pass ``L_sigma_s =
    sigma * L_s`` precomputed (or use the convenience constructor
    :meth:`from_motor`).
    """

    Rs: float
    """Stator resistance [Ω]."""
    Ls: float
    """Stator self-inductance [H]."""
    Lr: float
    """Rotor self-inductance referred to stator [H]."""
    Lm: float
    """Mutual inductance referred to stator [H]."""
    Rr: float = 0.0
    """Rotor resistance referred to stator [Ω]. **Required for
    accurate speed estimation** — the rotor time constant
    ``Tr = Lr / Rr`` drives the adaptive-model dynamics. Default
    0.0 triggers the legacy heuristic ``Rr ≈ 0.5·Rs`` (works only
    for textbook-NEMA-B machines; arbitrary errors otherwise).
    Use :meth:`from_motor` to wire ``Rr`` from an existing
    :class:`pulsim.InductionMotor` handle by construction."""
    voltage_model_hpf_omega: float = 5.0
    """Cut-off [rad/s] of the leaky integrator that replaces the
    pure ∫ in the voltage-driven reference model. A pure integrator
    of ``(v − Rs·i − σLs·di/dt)`` drifts indefinitely under any DC
    offset; the standard Tajima-Hori (1993) workaround replaces it
    with a low-cutoff first-order LPF that high-pass-filters the
    drift while passing the fundamental. Default 5 rad/s ≈ 0.8 Hz,
    well below 50/60 Hz mains. Set to 0 to disable (pure
    integrator)."""
    Kp_mras: float = 50.0
    """MRAS proportional gain — drives the cross-product error to
    zero. Higher = faster lock, but risk of oscillation."""
    Ki_mras: float = 1.0e3
    """MRAS integral gain — eliminates steady-state speed error."""

    # Live state.
    psi_r_alpha_ref: float = field(default=0.0, init=False, repr=False)
    psi_r_beta_ref:  float = field(default=0.0, init=False, repr=False)
    psi_r_alpha_adj: float = field(default=0.0, init=False, repr=False)
    psi_r_beta_adj:  float = field(default=0.0, init=False, repr=False)
    omega_hat:       float = field(default=0.0, init=False, repr=False)
    _mras_integ:     float = field(default=0.0, init=False, repr=False)
    _i_alpha_prev:   float = field(default=0.0, init=False, repr=False)
    _i_beta_prev:    float = field(default=0.0, init=False, repr=False)

    @classmethod
    def from_motor(cls, motor, *, Kp_mras: float = 50.0,
                       Ki_mras: float = 1e3) -> "FluxMRASObserver":
        """Convenience constructor — pulls Rs / Ls / Lr / Lm / **Rr**
        from an existing :class:`pulsim.InductionMotor` handle. This
        keeps the observer's rotor time constant in lock-step with
        the plant's by construction."""
        return cls(
            Rs=motor.R_s_ohm,
            Ls=motor.L_s_H,
            Lr=motor.L_r_H,
            Lm=motor.L_m_H,
            Rr=motor.R_r_ohm,
            Kp_mras=Kp_mras,
            Ki_mras=Ki_mras,
        )

    def reset(self) -> None:
        self.psi_r_alpha_ref = 0.0
        self.psi_r_beta_ref = 0.0
        self.psi_r_alpha_adj = 0.0
        self.psi_r_beta_adj = 0.0
        self.omega_hat = 0.0
        self._mras_integ = 0.0
        self._i_alpha_prev = 0.0
        self._i_beta_prev = 0.0

    def update(self, *,
                 v_alpha: float, v_beta: float,
                 i_alpha: float, i_beta: float,
                 dt: float) -> tuple:
        """Run one observer step. Returns
        ``(omega_hat, psi_r_alpha_adj, psi_r_beta_adj)``."""
        sigma = 1.0 - (self.Lm ** 2) / (self.Ls * self.Lr)
        L_sigma_s = sigma * self.Ls
        # Rotor time constant: Tr = Lr / Rr. When Rr wasn't supplied
        # explicitly (Rr <= 0) fall back to the rough heuristic
        # Rr ≈ 0.5·Rs (NEMA-B-machine ballpark) — accurate only for
        # textbook designs. Real machines need Rr from a measured
        # equivalent-circuit; ``from_motor`` wires it in by construction.
        Rr_eff = self.Rr if self.Rr > 0.0 else (0.5 * self.Rs)
        Tr = self.Lr / max(Rr_eff, 1e-9)

        # ---- Reference model (voltage-driven) ---------------------------
        # Numerical derivative of i_alpha, i_beta (forward-difference).
        di_a = (i_alpha - self._i_alpha_prev) / dt if dt > 0 else 0.0
        di_b = (i_beta  - self._i_beta_prev)  / dt if dt > 0 else 0.0
        self._i_alpha_prev = i_alpha
        self._i_beta_prev  = i_beta
        # Stator-frame back-EMF reconstruction.
        e_alpha = v_alpha - self.Rs * i_alpha - L_sigma_s * di_a
        e_beta  = v_beta  - self.Rs * i_beta  - L_sigma_s * di_b
        # Tajima-Hori (1993) leaky-integrator replacement for the
        # pure ∫: dψ/dt = (Lr/Lm)·e − ω_hp · ψ. The cutoff ω_hp
        # acts as a high-pass at low frequency (rejecting integrator
        # drift) while passing the fundamental. Disabled when ω_hp
        # = 0 (pure integrator, original Schauder form).
        scale = self.Lr / self.Lm
        leak = self.voltage_model_hpf_omega
        self.psi_r_alpha_ref += (
            scale * e_alpha - leak * self.psi_r_alpha_ref) * dt
        self.psi_r_beta_ref  += (
            scale * e_beta  - leak * self.psi_r_beta_ref) * dt

        # ---- Adaptive model (current-driven) ----------------------------
        # dψ_rα^adj/dt = (Lm/Tr)·i_α − ψ_rα^adj/Tr − ω̂·ψ_rβ^adj
        inv_Tr = 1.0 / Tr
        dpsi_a = (self.Lm * inv_Tr * i_alpha
                     - inv_Tr * self.psi_r_alpha_adj
                     - self.omega_hat * self.psi_r_beta_adj)
        dpsi_b = (self.Lm * inv_Tr * i_beta
                     - inv_Tr * self.psi_r_beta_adj
                     + self.omega_hat * self.psi_r_alpha_adj)
        self.psi_r_alpha_adj += dpsi_a * dt
        self.psi_r_beta_adj  += dpsi_b * dt

        # ---- Adaptation loop -------------------------------------------
        # Schauder cross-product error: ε = ψ̂ × ψ (z-component).
        # With ψ̂ = adaptive, ψ = reference, the PI controller on
        # ε drives ω̂ to the value that aligns ψ̂ with ψ.
        eps = (self.psi_r_beta_ref  * self.psi_r_alpha_adj
                  - self.psi_r_alpha_ref * self.psi_r_beta_adj)
        self._mras_integ += self.Ki_mras * dt * eps
        self.omega_hat = self.Kp_mras * eps + self._mras_integ

        return (self.omega_hat,
                  self.psi_r_alpha_adj, self.psi_r_beta_adj)
