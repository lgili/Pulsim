# Sensorless Motor Observers

Pulsim v1.5 ships two **sensorless** rotor observers — algorithms that
estimate the rotor electrical angle and mechanical speed from stator
voltages and currents alone, without any shaft encoder. Both follow the
standard `MixedDomainBlockChain` block interface (`.reset()` + `.update(*,
...) → tuple`).

## Why sensorless?

Industrial AC drives almost universally avoid the shaft encoder:
- One less cable, one less point of failure.
- No mechanical alignment / hall-sensor placement.
- Cheaper BOM, smaller footprint.
- A sealed shaft (no encoder feed-through) handles harsh environments.

Modern variable-frequency drives use sensorless control above ~10 % rated
speed; the low-speed window is bridged by V/f-ramp startup or
high-frequency-injection (HFI) variants — both queued for future
Pulsim releases.

## Available observers

### `SlidingModeObserver` — for PMSM

Algorithm (Utkin §10 / TI InstaSPIN SPRABQ8):

1. **Sliding-mode current observer** in α-β:
   $$\frac{d\hat{i}_\alpha}{dt} = \frac{1}{L_s}\left(v_\alpha - R_s\hat{i}_\alpha - z_\alpha\right)$$
   $z = K_{sl}\cdot\text{sat}(\hat{i} - i, \varepsilon)$ — saturating-sign injection.

2. **Equivalent-control LPF** extracts the back-EMF estimate $\hat{e}$ from $z$.

3. **Angle PLL** drives the d-axis projection of $\hat{e}$ to zero;
   outputs $\hat{\theta}_e$ and $\hat{\omega}_e$ directly (the d-component
   $\hat{e}_\alpha\cos\hat{\theta} + \hat{e}_\beta\sin\hat{\theta}$ equals
   $-E\sin(\theta_e - \hat{\theta})$, a clean Goodwin-style phase detector).

### `FluxMRASObserver` — for induction motor

Algorithm (Schauder 1992, with Tajima-Hori 1993 leaky-integrator
enhancement):

1. **Reference model** — voltage-driven rotor-flux estimate
   $\psi_r^{\text{ref}}$ via $\int(v - R_s i - \sigma L_s\,di/dt)\,(L_r/L_m)\,dt$.
   Pulsim replaces the pure integrator with a low-cutoff leaky integrator
   (default cutoff `voltage_model_hpf_omega = 5 rad/s` ≈ 0.8 Hz) to reject
   the DC drift that would otherwise blow up under any input bias.

2. **Adaptive model** — current-driven (uses $\hat{\omega}$).

3. **Adaptation loop** — Lyapunov-derived PI on the cross-product
   $\epsilon = \psi_\beta^{\text{ref}}\hat{\psi}_\alpha - \psi_\alpha^{\text{ref}}\hat{\psi}_\beta$.

## Quick start — PMSM SMO

```python
import pulsim as p

smo = p.SlidingModeObserver(
    Rs=0.5, Ls=200e-6,
    K_sl=20.0,           # > peak back-EMF voltage
    f_init_hz=50.0,      # initial PLL guess (close to nominal helps)
    omega_lpf=2*math.pi*2000.0,
)

# Each step:
theta_hat, omega_hat, e_a, e_b, low_speed_flag = smo.update(
    v_alpha=v_a, v_beta=v_b,
    i_alpha=i_a, i_beta=i_b,
    dt=DT,
)
```

End-to-end runnable example: `examples/scripts/run_pmsm_smo_sensorless.py`.
Synthetic 200 rad/s mechanical (800 rad/s electrical) trace; SMO locks
the angle within **4.2° peak / 4.2° RMS** after a 50 ms lock-in window,
$\hat{\omega}$ matches $\omega_{e,\text{true}}$ to 7 sig figs.

## Quick start — IM MRAS

```python
import pulsim as p

# Either explicit:
mras = p.FluxMRASObserver(
    Rs=2.46, Ls=0.0948, Lr=0.0948, Lm=0.0874, Rr=1.23,
    voltage_model_hpf_omega=5.0,
    Kp_mras=10.0, Ki_mras=500.0,
)

# Or pulled from an existing motor handle:
mras = p.FluxMRASObserver.from_motor(motor)

omega_hat, psi_alpha_adj, psi_beta_adj = mras.update(
    v_alpha=v_a, v_beta=v_b,
    i_alpha=i_a, i_beta=i_b,
    dt=DT,
)
```

## Tuning guide

### SMO (PMSM)

| Knob | Default | When to adjust |
|---|---|---|
| `K_sl` | 50 V | Set to ~ 2 × peak back-EMF. Higher = faster lock, more chatter. |
| `omega_lpf` | 500 Hz | ~10 × rated electrical frequency. Higher = less LPF delay, more noise pass-through. |
| `Kp_pll` | 200 | PI on the phase error. Increase if lock-in is too slow. |
| `Ki_pll` | 1e4 | Integral gain — PLL bandwidth ≈ √Ki. |
| `epsilon` | 0.5 A | Boundary-layer width for `tanh()`. Smaller = closer to bang-bang sign. |
| `f_init_hz` | 50 Hz | Initial PLL frequency guess. Match to nominal motor frequency for faster lock. |
| `low_speed_threshold` | 0.05 V | Back-EMF magnitude under which the `low_speed_flag` output goes True. |

### MRAS (IM)

| Knob | Default | When to adjust |
|---|---|---|
| `Rr` | 0.0 → falls back to `0.5·Rs` heuristic | **Strongly recommended** to set explicitly via `from_motor()`. Wrong `Rr` → proportional speed offset. |
| `voltage_model_hpf_omega` | 5 rad/s | Set to 0 for pure integrator (original Schauder). Increase if voltage-model drift dominates; decrease for cleaner low-frequency tracking. |
| `Kp_mras`, `Ki_mras` | 50, 1e3 | PI gains for the cross-product → $\hat{\omega}$ loop. Lyapunov-derived; retune per-machine if oscillation. |

## Failure modes (honest catalog)

- **SMO at very low speed** — back-EMF magnitude collapses toward `K_sl`,
  observer can lose lock. Use the `low_speed_flag` output to hand off to
  a V/f startup ramp (industry standard).
- **MRAS at low speed** — voltage-model integration drifts because
  $R_s i_s$ becomes comparable to the integrand; the leaky-integrator
  mitigates but does not eliminate this.
- **MRAS bootstrap with $\hat{\omega} = 0$** — the adaptive-model
  rotor-flux magnitude stays near zero until $\hat{\omega}$ is in the
  neighbourhood of $\omega_e^{\text{true}}$, so the cross-product error
  is dominated by amplitude mismatch instead of phase. Practical
  implementations bootstrap with a V/f-ramp open-loop start, then
  hand over to the closed-loop MRAS after ~10 % rated speed.
- **Parameter mismatch (Rs, Lr, Lm, Rr)** — all observers are
  parametric. Plant-parameter drift (temperature, magnetic
  saturation) shows up as a steady-state speed/angle offset. PSIM /
  PLECS workarounds include real-time parameter adaptation; not yet
  shipped in Pulsim.

## See also

- [Induction Motors](motors-induction.md)
- `python/pulsim/observers.py` — source.
- `openspec/changes/add-sensorless-motor-observers/` — proposal + spec.
