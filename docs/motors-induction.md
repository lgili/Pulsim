# Induction Motors

Pulsim ships a three-phase squirrel-cage **induction motor** (IM) as part of
the v1.5 Phase 2 physics-parity release. The model is a 5th-order dq
stationary-frame representation that plugs into the same observer pattern
already used by the `PMSM` and `BLDC` motors.

## Quick start

```python
import pulsim as p

# 4 kW / 400 V / 50 Hz / 4-pole nameplate → starting parameter set.
kw = p.im_parameters_from_nameplate(P_rated_W=4_000.0,
                                          V_LL_V=400.0, f_Hz=50.0,
                                          pole_pairs=2)
diag = kw.pop("_diagnostics")
print(f"Synchronous speed: {diag['omega_sync_rad_s']:.2f} rad/s")

b = p.CircuitBuilder()
# ... 3-φ source + R_leak on the star point ...

motor = p.add_induction_motor(
    b, name="IM",
    phase_nodes=("a", "b", "c"),
    neutral_node="n",
    T_load=0.0,
    **kw,
)

obs, b_extra = p.make_induction_motor_observer(b, motor, dt=DT)
res = p.simulate(b, t_end=1.5, dt=DT,
                  switch_fn=lambda t: p.SwitchStateMask(0),
                  step_observer=obs, b_extra_fn=b_extra)

print(f"Final mechanical speed: {motor.mech.omega_rad_s:.2f} rad/s")
print(f"Final rotor flux: |ψ_r| = {math.hypot(motor.psi_r_alpha_Wb, motor.psi_r_beta_Wb):.3f} Wb")
```

End-to-end runnable example: `examples/scripts/run_im_direct_online_start.py`.
On a 4 kW machine alimentada por 230 V RMS @ 50 Hz the script reaches the
synchronous speed of **1500 rpm at 0 % slip in ~1.5 s sim / 0.2 s wall-clock**
(Apple M-series).

## Model

Stationary-frame αβ rotor-flux state equations (Krause §6.6):

$$
\begin{aligned}
\frac{d\psi_{r\alpha}}{dt} &= -\frac{1}{T_r}\psi_{r\alpha} + \frac{L_m}{T_r}i_{s\alpha} - \omega_e\psi_{r\beta} \\
\frac{d\psi_{r\beta}}{dt}  &= -\frac{1}{T_r}\psi_{r\beta} + \frac{L_m}{T_r}i_{s\beta} + \omega_e\psi_{r\alpha} \\
T_e &= \frac{3}{2}p\frac{L_m}{L_r}\left(i_{s\beta}\psi_{r\alpha} - i_{s\alpha}\psi_{r\beta}\right)
\end{aligned}
$$

where $T_r = L_r/R_r$ is the rotor time constant, $\omega_e = p \cdot \omega_m$
is the electrical speed, and $p$ is the number of pole pairs.

## Topology decomposition

Each phase contributes $R_s + \sigma L_s$ in series with a modulated
"back-EMF" source between the phase terminal and the star neutral, where
$\sigma = 1 - L_m^2/(L_s L_r)$ is the **leakage factor** and $\sigma L_s$
is the **stator transient inductance**. The mutual-coupling dynamics live
inside the observer (the $L_m/L_r \cdot d\psi_r/dt$ term injected via
`b_extra`); the MNA matrix sees only the linear $\sigma L_s$.

This split lets us reuse the existing linear-inductor path in the kernel
without adding a coupled-inductor device — the same architectural trick
the PMSM / BLDC observers use for back-EMF injection.

## Parameter identification

`im_parameters_from_nameplate(P_rated_W, V_LL_V, f_Hz, ...)` derives a
starting `(R_s, L_s, R_r, L_r, L_m, J)` set from the motor nameplate using
NEMA-B rules of thumb. The split is approximate (R_s : R_r : X_σs : X_σr :
X_m ≈ 0.10 : 0.05 : 0.08 : 0.08 : 0.95 of the rated per-phase impedance);
precise design requires a locked-rotor + no-load test or a manufacturer-
provided equivalent-circuit.

The returned dict also carries `_diagnostics` with the synchronous speed,
rated torque, rated current, and per-phase impedance for sanity checks.

## Operating modes covered

| Mode | What works | Showcase |
|---|---|---|
| **DOL start** (direct-on-line) | ✅ — drives by 3 sine sources at nominal V/f | `run_im_direct_online_start.py` |
| **V/f open-loop ramp** | ⚠️ partial — DOL serves as a proxy; explicit V/f ramp showcase deferred | n/a |
| **IFOC closed-loop** | ⚠️ planned for v1.5.2 — needs the BlockChain wiring with Park/Clarke/PI/SVM | n/a |
| **Sensorless MRAS speed observer** | ⚠️ observer present (`FluxMRASObserver`) with `Rr` + leaky-integrator parameters; bootstrap for low-speed operation needs more tuning | n/a |

## Known limitations (v1.5)

1. **No C++ port** — the observer is pure-Python. Performance is fine at
   v2's typical sub-µs `dt` (the IM observer adds ~5 µs / step on M-series),
   but a port is queued for v1.6.0 if profiling shows a hotspot.

2. **No YAML support** — the `device_type: induction_motor` parser entry
   is queued for v1.6.0. Use the Python API meanwhile.

3. **Synchronous-speed singularity** — at exactly slip = 0 the rotor flux
   dynamics decay with no driving term. In practice this is fine because
   any load forces a non-zero slip, but the helper's `s_floor = 1e-6`
   clamp is documented in the docstring.

## See also

- [Motors — Sensorless Observers](motors-sensorless.md)
- `python/pulsim/motors.py` — source.
- `openspec/changes/add-induction-motor-squirrel-cage/` — proposal + spec.
