# Pulsim Device Model Reference

This document provides detailed information about all device models supported by Pulsim, including their equations, parameters, and usage guidelines.

## Table of Contents

1. [Passive Components](#passive-components)
2. [Sources](#sources)
3. [Semiconductor Devices](#semiconductor-devices)
4. [Switching Devices](#switching-devices)
5. [Magnetic Components](#magnetic-components)
6. [Thermal Models](#thermal-models)
7. [Device Library](#device-library)

---

## Passive Components

### Resistor

**Symbol:** R
**Equation:** V = I × R

The resistor is modeled as a linear conductance G = 1/R in the MNA matrix.

**MNA Stamp:**
```
     n+    n-
n+ [ +G   -G  ]
n- [ -G   +G  ]
```

**Parameters:**
| Parameter | Symbol | Unit | Range | Description |
|-----------|--------|------|-------|-------------|
| value | R | Ω | > 0 | Resistance |

---

### Capacitor

**Symbol:** C
**Equation:** I = C × dV/dt

Pulsim uses the companion model approach for time integration.

**Backward Euler Companion Model:**
```
I_eq = (C/dt) × V_new - (C/dt) × V_old
G_eq = C/dt
```

**Trapezoidal Companion Model:**
```
I_eq = (2C/dt) × V_new - (2C/dt) × V_old - I_old
G_eq = 2C/dt
```

**Parameters:**
| Parameter | Symbol | Unit | Range | Description |
|-----------|--------|------|-------|-------------|
| value | C | F | > 0 | Capacitance |
| ic | V₀ | V | any | Initial voltage |

---

### Inductor

**Symbol:** L
**Equation:** V = L × dI/dt

**Backward Euler Companion Model:**
```
V_eq = (L/dt) × I_new - (L/dt) × I_old
R_eq = L/dt
```

**Trapezoidal Companion Model:**
```
V_eq = (2L/dt) × I_new - (2L/dt) × I_old + V_old
R_eq = 2L/dt
```

**Parameters:**
| Parameter | Symbol | Unit | Range | Description |
|-----------|--------|------|-------|-------------|
| value | L | H | > 0 | Inductance |
| ic | I₀ | A | any | Initial current |

---

## Sources

### DC Voltage Source

**Symbol:** V
**Equation:** V = V_dc

Implemented using a voltage source MNA stamp with an additional equation row.

**MNA Stamp:**
```
      n+    n-    I_v
n+  [  0    0    +1  ]
n-  [  0    0    -1  ]
I_v [ +1   -1     0  ] = V_dc
```

---

### Pulse Voltage Source

**Waveform Definition:**
```
        V2  ____________________
           /|                  |\
          / |                  | \
         /  |                  |  \
    V1 _/   |                  |   \_____
       |    |                  |   |
       td   tr      pw         tf
       |<-------- period ---------->|
```

**Parameters:**
| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| v1 | V₁ | V | Low voltage level |
| v2 | V₂ | V | High voltage level |
| td | t_d | s | Delay time |
| tr | t_r | s | Rise time |
| tf | t_f | s | Fall time |
| pw | t_pw | s | Pulse width |
| period | T | s | Period |

---

### Sinusoidal Voltage Source

**Equation:**
```
V(t) = V_o + V_a × sin(2π × f × (t - t_d) + φ) × e^(-θ(t-t_d))
```

**Parameters:**
| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| vo | V_o | V | DC offset |
| va | V_a | V | Amplitude |
| freq | f | Hz | Frequency |
| td | t_d | s | Delay |
| theta | θ | 1/s | Damping factor |
| phi | φ | deg | Phase |

---

### PWM Source

Generates a pulse-width modulated signal for switching control.

**Parameters:**
| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| frequency | f_sw | Hz | Switching frequency |
| duty | D | - | Duty cycle (0-1) |
| phase | φ | deg | Phase offset |
| dead_time | t_dead | s | Dead time between transitions |
| inverted | - | bool | Invert output |

**Dead Time Handling:**

For complementary switches (e.g., high-side and low-side in half-bridge):
```
S_high: |----ON----|     |----ON----|
S_low:       |----ON----|     |----ON----|
        ^dead^      ^dead^
```

---

## Semiconductor Devices

### Diode - Ideal Model

**Symbol:** D
**Model:** ideal

**Equations:**
```
OFF state (V_ak < 0):  I = 0, G = G_off = 1e-9
ON state (V_ak >= 0):  I = V_ak / R_on, G = 1/R_on
```

**Parameters:**
| Parameter | Symbol | Unit | Default | Description |
|-----------|--------|------|---------|-------------|
| ron | R_on | Ω | 0.001 | On-state resistance |
| roff | R_off | Ω | 1e9 | Off-state resistance |

---

### Diode - Shockley Model

**Symbol:** D
**Model:** shockley

**DC Equation:**
```
I_d = I_s × (e^(V_d/(n×V_t)) - 1)

where V_t = kT/q ≈ 26mV at 300K
```

**Junction Capacitance:**
```
C_j = C_jo / (1 - V_d/V_j)^m   for V_d < FC×V_j
C_j = C_jo × (1-FC)^(-1-m) × (1-FC(1+m) + m×V_d/V_j)   for V_d >= FC×V_j
```

**Diffusion Capacitance:**
```
C_d = τ_t × dI_d/dV_d = τ_t × I_s/(n×V_t) × e^(V_d/(n×V_t))
```

**Parameters:**
| Parameter | Symbol | Unit | Default | Description |
|-----------|--------|------|---------|-------------|
| is | I_s | A | 1e-14 | Saturation current |
| n | n | - | 1.0 | Emission coefficient |
| rs | R_s | Ω | 0 | Series resistance |
| cjo | C_jo | F | 0 | Zero-bias junction capacitance |
| vj | V_j | V | 0.7 | Junction potential |
| m | m | - | 0.5 | Grading coefficient |
| tt | τ_t | s | 0 | Transit time |
| bv | BV | V | ∞ | Breakdown voltage |
| ibv | I_BV | A | 1e-10 | Breakdown knee current |
| fc | FC | - | 0.5 | Forward-bias capacitance coefficient |

**Linearization for Newton-Raphson:**
```
I_d ≈ I_d0 + G_d × (V_d - V_d0)
G_d = dI_d/dV_d = I_s/(n×V_t) × e^(V_d0/(n×V_t))
```

---

### MOSFET - Level 1 Model

**Symbol:** M (NMOS/PMOS)

**Operating Regions:**

**Cutoff (V_gs < V_th):**
```
I_ds = 0
```

**Linear/Triode (V_gs >= V_th, V_ds < V_gs - V_th):**
```
I_ds = K_p × (W/L) × [(V_gs - V_th) × V_ds - V_ds²/2] × (1 + λ × V_ds)
```

**Saturation (V_gs >= V_th, V_ds >= V_gs - V_th):**
```
I_ds = (K_p/2) × (W/L) × (V_gs - V_th)² × (1 + λ × V_ds)
```

**Parameters:**
| Parameter | Symbol | Unit | Default | Description |
|-----------|--------|------|---------|-------------|
| vth | V_th | V | 2.0 | Threshold voltage |
| kp | K_p | A/V² | 20e-6 | Transconductance parameter |
| lambda | λ | 1/V | 0.01 | Channel-length modulation |
| w | W | m | 100e-6 | Channel width |
| l | L | m | 10e-6 | Channel length |

**Capacitances:**
| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| cgs | C_gs | F | Gate-source capacitance |
| cgd | C_gd | F | Gate-drain (Miller) capacitance |
| cds | C_ds | F | Drain-source capacitance |

**Jacobian Elements for Newton-Raphson:**
```
∂I_ds/∂V_gs = g_m = K_p × (W/L) × V_ds                    (linear)
∂I_ds/∂V_gs = g_m = K_p × (W/L) × (V_gs - V_th)           (saturation)

∂I_ds/∂V_ds = g_ds = K_p × (W/L) × (V_gs - V_th - V_ds)   (linear)
∂I_ds/∂V_ds = g_ds = λ × I_ds                              (saturation)
```

---

### Power MOSFET - Simplified Model

For power electronics simulation, a simplified model using datasheet parameters is more practical.

**On-State:**
```
V_ds = I_ds × R_ds(on)(T_j)
```

**Temperature Dependence:**
```
R_ds(on)(T_j) = R_ds(on)(25°C) × [1 + TC1×(T_j-25) + TC2×(T_j-25)²]
```

**Typical TC values for Si MOSFETs:**
- TC1 ≈ 0.005 to 0.01 /°C
- TC2 ≈ 0 (often neglected)

**Body Diode:**

The intrinsic body diode is modeled as a fast-recovery diode:
```
I_d = I_s × (e^(V_d/(n×V_t)) - 1)
```

With reverse recovery:
```
Q_rr = I_F × t_rr / 2
E_rr = Q_rr × V_R
```

**Parameters:**
| Parameter | Symbol | Unit | Default | Description |
|-----------|--------|------|---------|-------------|
| rds_on | R_ds(on) | Ω | - | On-state drain-source resistance |
| vth | V_th | V | 3.0 | Gate threshold voltage |
| ciss | C_iss | F | - | Input capacitance (C_gs + C_gd) |
| coss | C_oss | F | - | Output capacitance (C_gd + C_ds) |
| crss | C_rss | F | - | Reverse transfer capacitance (C_gd) |
| qg | Q_g | C | - | Total gate charge |
| tc1 | TC1 | 1/°C | 0.007 | Temperature coefficient 1 |
| tc2 | TC2 | 1/°C² | 0 | Temperature coefficient 2 |

---

### IGBT - Simplified Model

**On-State Voltage:**
```
V_ce(sat) = V_ce0 + R_ce × I_c
```

**Parameters:**
| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| vce_sat | V_ce0 | V | On-state voltage at zero current |
| rce | R_ce | Ω | On-state resistance |
| vth | V_th | V | Gate threshold voltage |
| td_on | t_d(on) | s | Turn-on delay time |
| tr | t_r | s | Current rise time |
| td_off | t_d(off) | s | Turn-off delay time |
| tf | t_f | s | Current fall time |
| cies | C_ies | F | Input capacitance |
| coes | C_oes | F | Output capacitance |
| cres | C_res | F | Reverse transfer capacitance |

---

## Switching Devices

### Ideal Switch

**Model:**
```
OFF: R = R_off (very high, ~1e9 Ω)
ON:  R = R_on (very low, ~1 mΩ)
```

**Control Logic:**
```
V_control > V_th → ON
V_control <= V_th → OFF
```

**Event Handling:**

When a switch changes state:
1. Event detected at crossing time
2. Simulation steps back to crossing
3. Matrix is reformulated
4. Integration restarts with new topology

---

## Magnetic Components

### Ideal Transformer

**Equations:**
```
V_1 = n × V_2
I_1 = -I_2 / n

where n = N_1/N_2 (turns ratio)
```

**MNA Implementation:**

The ideal transformer is implemented using coupled voltage and current sources.

---

### Transformer with Parasitics

**Equivalent Circuit:**
```
    R_p    L_lk_p        n:1        L_lk_s    R_s
o---/\/\/---LLLL---o    | |    o---LLLL---/\/\/---o
                       | |
                    L_m ||
                       | |
o----------------------o    o---------------------o
```

**Parameters:**
| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| turns_ratio | n | - | Primary to secondary turns ratio |
| lm | L_m | H | Magnetizing inductance |
| llk_pri | L_lk,p | H | Primary leakage inductance |
| llk_sec | L_lk,s | H | Secondary leakage inductance |
| rp | R_p | Ω | Primary winding resistance |
| rs | R_s | Ω | Secondary winding resistance |

---

## Thermal Models

### Foster Network

Pulsim uses the Foster RC network representation for thermal modeling:

```
       R_th1        R_th2        R_th3
P_loss --/\/\/--+--/\/\/--+--/\/\/--+-- T_ambient
               |         |         |
              ===C1     ===C2     ===C3
               |         |         |
              GND       GND       GND
```

**Time Constants:**
```
τ_i = R_th_i × C_th_i
```

**Transient Response:**
```
Z_th(t) = Σ R_th_i × (1 - e^(-t/τ_i))
```

**Steady-State:**
```
R_th_jc = Σ R_th_i
```

**Parameters:**
| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| rth | R_th | K/W | Array of thermal resistances |
| tau | τ | s | Array of time constants |

### Temperature-Dependent Parameters

**MOSFET R_ds(on):**
```
R_ds(on)(T) = R_ds(on)(25°C) × (T/300)^α

where α ≈ 2.0 to 2.5 for silicon
```

**Diode Forward Voltage:**
```
V_f(T) = V_f(25°C) - α_vf × (T - 25)

where α_vf ≈ 2 mV/°C
```

---

## Device Library

Pulsim includes a library of pre-defined device models based on manufacturer datasheets.

### MOSFETs

| Model | V_ds | R_ds(on) | I_d | Package |
|-------|------|----------|-----|---------|
| IRF540N | 100V | 44mΩ | 33A | TO-220 |
| IRFP460 | 500V | 270mΩ | 20A | TO-247 |
| IPP200N25N3 | 250V | 20mΩ | 64A | TO-220 |
| STW48NM60N | 600V | 70mΩ | 44A | TO-247 |

### IGBTs

| Model | V_ce | V_ce(sat) | I_c | Package |
|-------|------|-----------|-----|---------|
| IRG4PH50UD | 1200V | 2.0V | 45A | TO-247 |
| IKW40N120H3 | 1200V | 1.7V | 40A | TO-247 |

### Diodes

| Model | V_r | V_f | I_f | t_rr |
|-------|-----|-----|-----|------|
| 1N4148 | 100V | 0.7V | 200mA | 4ns |
| MUR860 | 600V | 1.0V | 8A | 60ns |
| STTH30R06 | 600V | 0.95V | 30A | 35ns |

### Using Library Models

```json
{
  "type": "MOSFET",
  "name": "Q1",
  "nodes": ["d", "g", "s", "0"],
  "model": "IRF540N"
}
```

Or override specific parameters:

```json
{
  "type": "MOSFET",
  "name": "Q1",
  "nodes": ["d", "g", "s", "0"],
  "model": "IRF540N",
  "rds_on": 0.05
}
```

---

## Model Selection Guidelines

### For Fast Switching Analysis
- Use simplified models with datasheet parameters
- Include gate charge model for timing
- Enable adaptive timestep near switching events

### For Loss Calculation
- Use models with accurate on-state characteristics
- Include temperature dependence for thermal coupling
- Use switching energy lookup tables

### For EMI Analysis
- Include all parasitic capacitances
- Use short timesteps (< 1/10 of rise time)
- Model PCB parasitics as lumped elements

### For Thermal Analysis
- Use Foster models with 3-4 time constants
- Enable bidirectional thermal-electrical coupling
- Run for sufficient time to reach thermal steady-state
