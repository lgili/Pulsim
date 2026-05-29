# PSIM + PLECS demo libraries — Pulsim cross-validation catalog

**Purpose.** PSIM (Powersim/Altair) and PLECS (Plexim) each ship a
curated library of demo circuits. Rebuilding those same canonical
circuits in Pulsim and comparing waveforms/KPIs gives us two independent
references to validate the kernel against. This doc catalogs what the
two vendors ship and ranks the highest-value circuits to build first.

> Sourcing note: PLECS publishes a clean per-circuit demo index
> (docs.plexim.com/plecs-demos) — names below are exact. PSIM has no
> equivalent public per-file manifest; its category list + several named
> circuits are corroborated from the PSIM User's Guide and the official
> Tutorial List PDF, but some PSIM filenames are *representative* of the
> installed `examples\` folder rather than guaranteed 1:1. See Sources.

---

## Recommended validation subset (build these first)

Ordered easy→hard. Priority = appears in **both** vendors' demo sets
(strongest cross-validation). "Ready?" = current Pulsim capability.

| # | Circuit | Both vendors | Exercises | Pulsim ready? |
|---|---|:--:|---|---|
| 1 | RC / RL / RLC + AC frequency response | ✅ | MNA + AC sweep | yes (transient), partial (AC sweep) |
| 2 | Single-phase diode rectifier (R/RL/cap load) | ✅ | Diode PWL, ripple/THD | **yes** |
| 3 | Three-phase 6-pulse diode bridge | ✅ | 3-ph sources + diode commutation | **yes** |
| 4 | Three-phase thyristor bridge, α-control | ✅ | SCR + line-synced firing | partial |
| 5 | Buck (open-loop) | ✅ | PWM switching + CCM ripple | **yes** |
| 6 | Boost (open-loop) | ✅ | PWM switching | **yes** |
| 7 | Buck-boost / Ćuk / SEPIC | ✅ | Multi-reactive DC-DC | **yes** |
| 8 | Buck w/ closed-loop voltage control (PID) | ✅ | Voltage loop + regulation | partial |
| 9 | Flyback (isolated) | ✅ | Transformer/coupled-L + switch | partial (magnetics) |
| 10 | Forward converter | ✅ | Transformer reset | partial |
| 11 | Single-phase H-bridge VSI (SPWM) | ✅ | Full bridge + SPWM + THD | **yes** |
| 12 | Three-phase 2-level VSI (SPWM + SVPWM) | ✅ | 3-ph bridge + modulation | yes (SPWM), needs-work (SVPWM) |
| 13 | Three-level NPC inverter | ✅ | Multilevel + neutral balance | partial |
| 14 | Resonant LLC (half-bridge) | ✅ | Resonant tank + variable-f + ZVS | needs-work |
| 15 | Phase-shift full-bridge / Dual Active Bridge | ✅ | Phase-shift mod + transformer | needs-work |
| 16 | Boost PFC (single-phase) | ✅ | Current-shaping + PF/THD | needs-work |
| 17 | Induction motor V/f drive (SPWM VSI) | ✅ | IM model + V/f | partial |
| 18 | PMSM FOC drive (SVPWM) | ✅ | dq FOC + SVPWM + machine | needs-work |
| 19 | BLDC drive, hall-sensor 6-step | ✅ | BLDC + commutation | partial |
| 20 | Single-phase grid-tie PV inverter + MPPT | ✅ | PV source + MPPT + PLL | needs-work |

**Bonus (single-vendor, high-value):** MMC / HVDC link (PLECS "HVDC
Transmission System with MMCs"; PSIM MMC application area) — Pulsim
already has MMC support, so this is a differentiating validation. Partial.

**Feature roadmap implied by the "needs-work" cluster:** SVPWM
generator, AC/loop-gain sweep harness, peak-current-mode comparator +
slope comp, PV nonlinear source + MPPT algorithms, resonant-tank /
variable-frequency control, phase-shift modulation.

---

## PSIM demo library

PSIM ships examples in its installed `examples\` folder (category
subfolders) plus online Tutorials + Application Notes (AN001–AN008).

### Rectifiers
| Demo | Description | Pulsim ready? |
|---|---|---|
| Single-phase diode rectifier | 1-ph full bridge, RL/cap load | yes |
| Three-phase diode rectifier | 6-pulse uncontrolled bridge | yes |
| Single/three-phase thyristor (controlled) rectifier | α-controlled SCR bridge, Vac zero-cross sync | partial |
| Cycloconverter | AC-AC direct conversion | needs-work |
| Boost PFC | 1-ph boost power-factor correction | partial |

### DC-DC
| Demo | Description | Pulsim ready? |
|---|---|---|
| 1-quadrant chopper (`chop.sch`) | Basic step-down chopper | yes |
| Buck (open-loop) | MNA + PWM | yes |
| UC3842 buck | Closed-loop, PWM-IC current-mode | needs-work |
| Flyback | Isolated, transformer | partial |
| Forward / push-pull / half-/full-bridge | Isolated DC-DC family | partial |
| Phase-shift full-bridge | ZVS PSFB + sync rect | needs-work |
| Resonant LLC | LLC, variable-freq (AN003/004/005/007/008) | needs-work |
| ZVS / ZCS resonant cells | Soft-switching blocks | partial |

### Inverters
| Demo | Description | Pulsim ready? |
|---|---|---|
| Three-phase VSI (LUT PWM) | PWM patterns by modulation index | partial |
| Space-vector PWM 3-ph inverter | SVPWM | needs-work |
| Multi-level inverter | NPC / cascaded | partial |
| MMC | Modular multilevel (grid) | yes |

### Motor drives
| Demo | Description | Pulsim ready? |
|---|---|---|
| DC motor drive (chopper-fed) | DC machine + chopper speed control | partial |
| Induction motor V/f (open-loop) | IM + SPWM VSI + diode-bridge bus | partial |
| PMSM FOC (AN001/002) | dq FOC + efficiency map | needs-work |
| PMSM sensorless (InstaSPIN/DRV83xx) | Observer-based sensorless | needs-work |
| BLDC w/ hall sensor | BDCM 6-step commutation | partial |
| Switched reluctance | SRM nonlinear inductance | needs-work |

### Grid / renewable
| Demo | Description | Pulsim ready? |
|---|---|---|
| Solar module + MPPT | PV array + MPPT control | needs-work |
| Wind turbine | Turbine block + generator + converter | needs-work |
| Grid-connected / microgrid (PV+battery) | Grid-tie inverter + PLL | needs-work |
| (Enhanced) PLL | Grid synchronization | needs-work |

### Magnetics / thermal / device models
| Demo | Description | Pulsim ready? |
|---|---|---|
| IGBT & MOSFET loss calc (Thermal Module) | Cond+sw loss from device DB | partial |
| Inductor loss / DB | Core+winding loss | partial |
| Diode w/ reverse recovery | Q_rr modeling | partial |
| IGBT Level-2 | Tail current | needs-work |
| SiC / GaN device loss | Wide-bandgap switching | needs-work |
| Li-ion battery / ultracap | Energy-storage models | needs-work |

---

## PLECS demo library

Exact names from docs.plexim.com/plecs-demos + plexim.com application-examples.

### Basic topologies
| Demo | Description | Pulsim ready? |
|---|---|---|
| Frequency Response of Passive Circuit | RLC AC/freq response | partial |
| Low Pass Filter Circuits | RC/RL/LC filters | yes |
| Operational Amplifier Circuits | Ideal/non-ideal op-amp | partial |
| Diode Rectifier / Three-Phase Diode Bridge Rectifier | 1-/3-ph uncontrolled | yes |
| Thyristor Chopper Circuit | SCR chopper | partial |
| Boost Converter | Open-loop boost | yes |
| Buck Converter with Thermal Model | Buck + junction-temp network | partial |
| Buck-Boost / Ćuk / SEPIC / Watkins-Johnson | DC-DC topologies | yes |
| Forward / Flyback Converter | Isolated + transformer | partial |
| Resonant Half-/Full-Bridge SLR | Series-resonant | partial |
| H-Bridge Inverter Circuit | 1-ph H-bridge VSI | yes |
| Cascaded Multilevel Inverter | Cascaded H-bridge | partial |

### Power supplies (control / PFC / resonant / 3-ph)
| Demo | Description | Pulsim ready? |
|---|---|---|
| Buck w/ Voltage / Cascaded / Peak-Current / Digital Controls | Control-loop variants | partial → needs-work |
| Buck w/ Loop Gain Analysis | AC loop-gain Bode | needs-work |
| Buck in Boundary Conduction Mode | BCM | partial |
| (Multi-phase) Synchronous Buck | Sync-rect + interleave | partial |
| Flyback w/ Analog Controls / Magnetics | Analog loop / reluctance core | partial → needs-work |
| Ćuk / PSFB w/ Integrated Magnetics | Coupled magnetics | needs-work |
| Dual Active Bridge | Phase-shift, SiC, thermal+magnetic | needs-work |
| LLC Variable Frequency Resonant | Freq control, ZVS | needs-work |
| Flying Capacitor DC-DC | Flying-cap multilevel | partial |
| (Bridgeless / Totem-Pole / 3-Level) Boost PFC | PFC variants | needs-work |
| Vienna / Swiss Rectifier | 3-ph PFC | needs-work |
| Single/Three-Phase Thyristor Converter (2-/6-pulse) | Line-commutated SCR | partial |
| Neutral-Point Clamped / T-Type Converter | 3-level VSI | partial |
| Three-Phase VSI / VSI with Pre-Charge | 2-level 3-ph | yes / partial |
| Z-Source / Current-Source Inverter | Impedance-source | needs-work |
| Single-Phase Active Filter | Shunt APF | needs-work |
| Single-Phase Battery Charger / Two-Stage LED Driver | AC/DC+DC/DC | needs-work |

### Motor drives
| Demo | Description | Pulsim ready? |
|---|---|---|
| DC Motor Drive w/ Armature Chopper | DC machine + chopper | partial |
| Brushless DC Machine | BLDC trapezoidal | partial |
| Permanent-Magnet Synchronous Machine | PMSM FOC | needs-work |
| Lookup-Table-Based PMSM | FEA/LUT saturable | needs-work |
| Direct Flux Vector Control | Salient PMSM, DFVC | needs-work |
| Induction Machine w/ DTC | IM direct torque control | needs-work |
| Switched Reluctance (6-4/8-6/10-8) | SRM | needs-work |
| Synchronous Generator + Rectifier | SG + rectifier | needs-work |

### Power generation (renewables)
| Demo | Description | Pulsim ready? |
|---|---|---|
| Single-/Three-Phase PV Inverter (+ Partial Shading) | Grid-tie PV + MPPT | needs-work |
| Photovoltaic String Model | Nonlinear I-V | needs-work |
| DFIG Wind Turbine / PMSG Windpower | Variable-speed wind, B2B | needs-work |

### Power distribution
| Demo | Description | Pulsim ready? |
|---|---|---|
| HVDC Transmission System with MMCs | Dual-MMC HVDC link | partial (Pulsim has MMC) |
| STATCOM Cascaded H-Bridge | MV STATCOM, cell-cap balancing | needs-work |
| Microgrid in Island Operation | Droop-controlled generators | needs-work |

### Controls / analysis tools
| Demo | Description | Pulsim ready? |
|---|---|---|
| Buck with Analysis Tools | Steady-state / impulse-response | needs-work |
| Buck with Parameter Sweep | Parametric sweep | partial |
| Space Vector Control of Boost / 3-ph Boost Rectifier | SVM + 3-ph rectifier | needs-work |
| Inverter with C-Script PWM Modulator | Custom modulator hook | partial |
| Rainflow Counting & Lifetime Prediction | Thermal cycle counting | needs-work |

---

## Sources

**PLECS**
- Application Examples index — https://www.plexim.com/support/examples
- Demo Models docs — https://docs.plexim.com/plecs-demos/ (Basic Topologies, Power Supplies, Motor Drives, Power Generation, Electronics, Automotive, Automation)

**PSIM**
- PSIM Tutorial List (official PDF) — https://2023.help.altair.com/psim-tut/tutorials/PSIM%20Tutorial%20List.pdf
- Altair PSIM applications — https://altair.com/psim-applications
- PSIM User Manual V9.0.2 — https://www.myway.co.jp/products/psim/dlfiles/pdf/PSIM_User_Manual_V9.0.2.pdf
- PSIM 9.0.3 Application Examples (examples-folder category list)
- Phase-shift full-bridge PSIM example (Altair Community)
