# Tutorial 10 — From datasheet curves to a double-pulse test

**Goal.** Turn the curves of a power-MOSFET datasheet into an
`add_power_mosfet` device with the fitters of `pulsim.datasheet`, then
measure its switching energies with `pulsim.double_pulse_test`.

**Concepts introduced:**
- `fit_mosfet_channel`, `fit_rdson_temperature`, `fit_charge_table`,
  `power_mosfet_from_datasheet` — each returns its residual and refuses
  a fit above its limit.
- `double_pulse_test` — the clamped inductive double pulse, energies
  integrated on the result's own time grid.
- Fitted data versus held-out data: a curve the parameters were fitted
  to cannot validate them.

**The curves below are SYNTHETIC.** They are generated from known
parameters so that the round trip can be checked; they are not a real
part. A real part's curves are digitised from its datasheet (record the
document, revision and figure number), and its datasheet switching
energies stay out of the fit: they are the held-out check.

## 1. Curves

```python
import numpy as np
import pulsim as p
from pulsim import datasheet as ds

# Transfer curve in saturation (V_DS = 20 V): I_D = K (V_GS - V_T)^2.
v_gs = np.linspace(4.5, 10.0, 12)
i_d = 1.43 * (v_gs - 3.0) ** 2

# Capacitances at V_GS = 0 against V_DS, as printed on a datasheet.
v_ds = np.array([0.0, 1.0, 10.0, 100.0, 400.0])
C_iss = np.column_stack([v_ds, [3e-9, 2.6e-9, 2.2e-9, 2.05e-9, 2.02e-9]])
C_oss = np.column_stack([v_ds, [2e-9, 1.4e-9, 5e-10, 1.5e-10, 8e-11]])
C_rss = np.column_stack([v_ds, [1e-9, 6e-10, 2e-10, 5e-11, 2e-11]])

# Normalised R_DS(on) against junction temperature.
T_j = np.array([-40.0, 25.0, 75.0, 125.0, 150.0])
r_norm = 1 + 5e-3 * (T_j - 25) + 2e-5 * (T_j - 25) ** 2
```

## 2. Fit

```python
ch = ds.fit_mosfet_channel(v_gs, i_d, v_ds=20.0)
print(f"K = {ch['K']:.4g} A/V^2, V_T = {ch['V_T']:.4g} V, residual {ch['residual']:.1e}")

out = ds.power_mosfet_from_datasheet(
    transfer={"v_gs": v_gs, "i_d": i_d, "v_ds": 20.0},
    R_ds_on=0.1, V_gs_spec=10.0,
    C_iss=C_iss, C_oss=C_oss, C_rss=C_rss, R_g=2.0,
    rds_temperature={"T_j": T_j, "r_norm": r_norm})
kw = out["kwargs"]
print({k: round(float(v), 9) for k, v in out["residuals"].items()})
assert abs(kw["K"] - 1.43) < 1e-6 and abs(kw["rds_tc1"] - 5e-3) < 1e-9
```

`C_gd` and `C_ds` come back as charge-table laws (`ChargeLaw.table`):
the capacitor is stamped in charge form, so charge is conserved whatever
`C(v)` does. `R_d` is what R_DS(on) leaves after the channel's own
`1/(2K(V_GS − V_T))`; a negative value is refused (the transfer curve
and R_DS(on) disagree).

## 3. Double pulse

```python
r = p.double_pulse_test(kw, V_dd=400.0, I_load=10.0, R_g_ext=10.0,
                        V_drive=12.0, L_load=500e-6, engine="trbdf2")
print(f"E_on {r.E_on * 1e6:.2f} uJ, E_off {r.E_off * 1e6:.2f} uJ, "
      f"plateau {r.t_plateau * 1e9:.1f} ns at {r.V_plateau:.2f} V")
assert r.E_on > 0 and r.E_off > 0
```

Use `engine="trbdf2"`: the gate transient lasts tens of nanoseconds, and
the fixed-step `pwl` engine needs a step that resolves it (`dt=2e-10`
for a 60 ns plateau; the resolution diagnostic warns otherwise).

## 4. What this does and does not show

- The fitters reproduce their own input curves to the printed residuals.
  That is a fit, not a validation.
- A part is labelled `validated` only when `Part.acceptance_check()`
  reproduces the datasheet E_on / E_off (data held out of the fit) at
  the datasheet test condition within the part's declared tolerance
  (at most 25 %). No shipped part carries that label today: the shipped
  part files are synthetic illustrations and are refused by part number
  unless `allow_synthetic=True`.
- The Level-1 channel has one constant `K`; a real part's
  transconductance rolls off at high current, which can move E_on by
  more than the tolerance (see `docs/known-limitations.md`).
