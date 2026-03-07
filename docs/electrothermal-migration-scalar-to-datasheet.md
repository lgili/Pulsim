# Electrothermal Migration: Scalar to Datasheet

This guide shows how to migrate a component from scalar switching-loss setup to datasheet-grade loss + staged thermal modeling.

## 1) Scalar Baseline (Current)

```yaml
components:
  - type: mosfet
    name: M1
    nodes: [gate, vin, sw]
    vth: 3.0
    kp: 0.35
    g_off: 1e-8
    loss:
      eon: 2.0e-6
      eoff: 3.0e-6
      err: 0.0
    thermal:
      enabled: true
      rth: 0.8
      cth: 0.03
      temp_init: 25
      temp_ref: 25
      alpha: 0.004
```

Use this when you only have single representative switching energies.

## 2) Datasheet Loss Axes

Replace scalar loss with `loss.model: datasheet` and explicit axes:

```yaml
loss:
  model: datasheet
  axes:
    current: [0, 5, 10]
    voltage: [0, 20, 40]
    temperature: [25, 75, 125]
  eon:  [ ... N entries ... ]
  eoff: [ ... N entries ... ]
  err:  [ ... N entries ... ]
```

Rules:

- axes must be finite and strictly increasing.
- table length must be `len(I) * len(V) * len(T)`.
- all energy values must be finite and `>= 0`.

## 3) Upgrade Thermal Network (Optional)

If you need multi-time-constant behavior:

```yaml
thermal:
  enabled: true
  network: foster   # single_rc | foster | cauer
  rth_stages: [0.2, 0.4, 0.3]
  cth_stages: [0.005, 0.03, 0.2]
  temp_init: 25
  temp_ref: 25
  alpha: 0.004
```

If multiple devices share one heatsink:

```yaml
thermal:
  enabled: true
  rth: 0.7
  cth: 0.02
  shared_sink_id: hs_main
  shared_sink_rth: 0.35
  shared_sink_cth: 0.10
```

All members of the same `shared_sink_id` must use the same sink `rth/cth`.

## 4) Validation Checklist

After migration, confirm:

- parser has no errors (`parser.errors == []`).
- canonical channels exist (`T(M1)`, `Pcond(M1)`, `Psw_on(M1)`, `Psw_off(M1)`, `Prr(M1)`, `Ploss(M1)`).
- `len(channel) == len(result.time)` for each channel.
- summary consistency holds:
  - `final_temperature == last(T(M1))`
  - `peak_temperature == max(T(M1))`
  - `average_temperature == mean(T(M1))`

## 5) Common Migration Pitfalls

- Using non-monotonic axes.
- Wrong table length for datasheet arrays.
- Setting `shared_sink_rth/shared_sink_cth` without `shared_sink_id`.
- Mixing different sink values inside one shared sink id.
- Reconstructing losses/temperature in GUI instead of using backend canonical channels.
