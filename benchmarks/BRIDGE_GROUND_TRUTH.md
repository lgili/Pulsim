# Floating-bus diode bridge: exact reference

Circuit: `python/tests/test_trbdf2_rounding_fuzz.py::_bridge` (also
`test_trbdf2_error_control.py`), the one of DECISIONS #19
(add-branch-form-ideal-switch) and KNOWN_LIMITATIONS "TR-BDF2 on a bridge
whose DC side floats on a tie resistor".

| Element | Value |
|---|---|
| Vac (s-gnd) | 325 sin(2π·50 t) V |
| Rs, Ls | 0.5 Ω, 1 mH |
| D1 ac1→dp, D2 gnd→dp, D3 dn→ac1, D4 dn→gnd | g_on 1e3 S, g_off 1e-9 S, V_th = V_f = 0 |
| C, RL (dp-dn) | 470 µF, 100 Ω |
| Rtie (dn-gnd) | 1 MΩ |
| Initial state, t_end | zero (`simulate` default), 40 ms |

Diode rule (`models/switched_diode.hpp`): ON → OFF iff g_on·v ≤ 0,
OFF → ON iff v ≥ 0. A mode is consistent iff every ON diode has v > 0 and
every OFF diode has v < 0.

Script: `benchmarks/tools/bridge_ground_truth.py`. It propagates each mode
exactly (eigen-expansion with the sine as two extra states) in mpmath at
50 digits. It bisects every commutation to 1e-38 s and checks all 16 modes
at every event. Run it with
`uv run --with mpmath --with numpy python benchmarks/tools/bridge_ground_truth.py --all`.
One run takes 4 s.

## Verdict

- 30–40 ms time-weighted average of dp − dn: **311.343881428885841697 V**.
  All of these runs agree on these 21 digits: 30, 50 and 60 digits of
  working precision, and 1e-15 perturbations of the element values.
- The mode after every event is **unique** in exact arithmetic: exactly 1
  of 16 modes is forward-consistent at each of the 11 commutations.
  Nothing is ambiguous, so no alternative branch exists to follow.
- The model is well posed. With V_th = V_f = 0 the diode's i(v) law is
  continuous and strictly monotone, so the network is a Lipschitz ODE in
  (iL, vC), and its solution is unique.
- **pwl is right** to every digit it prints. dsed/exact is wrong by 0.027 V; trbdf2
  by 2.5e-3 V (its step-ceiling error at rtol 2e-5). Their errors are engine errors, not an
  ambiguity in the model.

| Engine (reported in DECISIONS #19) | 30–40 ms bus average [V] | Error vs truth [V] | Relative |
|---|---|---|---|
| exact (this script) | 311.3438814289 | 0 | 0 |
| pwl, dt 1e-6 and 2.5e-7 | 311.343881 | −4.3e-7 (below its last printed digit) | 1.4e-9 |
| dsed / exact | 311.316685 | −0.027196 | −8.7e-5 |
| trbdf2, both forms, rtol 2e-5, window clipped at 30 ms (every fuzz run) | 311.341410 | −2.47e-3 | −7.9e-6 |
| trbdf2, both forms, rtol 1e-7 | 311.343690 | −1.91e-4 | −6.1e-7 |

The trbdf2 "311.154 V" first reported was the average of the samples
t ≥ 30 ms over 10 ms: its first sample is at 30.006 ms (a metric error,
not the engine; DECISIONS #19).

## Events (exact, 50 digits)

| # | t [s] | Change | Forward-consistent modes |
|---|---|---|---|
| 0 | 0 | D1+ D4+ | 1 of 16 (all 16 statically: every v = 0) |
| 1 | 0.004453708135472069740740 | D1− D4− (tie) | 1 |
| 2 | 0.010005784513217728980424 | D3+ | 1 |
| 3 | 0.013234043039166346336001 | D2+ | 1 |
| 4 | 0.016037029154868641651128 | D2− | 1 |
| 5 | 0.019993816884401272433491 | D3− | 1 |
| 6 | 0.023578074673711604264767 | D1+ D4+ (tie) | 1 |
| 7 | 0.026164880107693963425638 | D1− D4− (tie) | 1 |
| 8 | 0.030006040329416432127840 | D3+ | 1 |
| 9 | 0.033448621701173689951408 | D2+ | 1 |
| 10 | 0.036121739554460249850922 | D2− | 1 |
| 11 | 0.039993904164045823079728 | D3− | 1 |

- **The D1/D4 tie is structural.** In the mode (D1, D4 ON; D2, D3 OFF),
  KCL gives (g_on + g_off)(v_D1 − v_D4) = v_D4/R_tie. So v_D1 and v_D4
  reach zero at the same instant for any element values (computed:
  both −1.197e-38 V at event 1). A 1e-15 perturbation cannot break this
  tie.
- At the tie the only consistent mode turns both diodes off together.
  The two one-diode turn-offs are both inconsistent. An engine that turns
  off one diode at a time must therefore come back to the pair, which
  matches the pwl chatter of D1/D4.
- **The bus rides on the line.** Between events 2 and 5 (and 8 and 11),
  D3 is ON and carries only the rail's leakage (µA). dn follows ac1
  through the negative half cycle, and the window's common-mode average
  is −49.6037455 V.
- **float64 cannot resolve some events.** For D3's turn-on at the line
  zero crossing, v_D3 = i/g_on stays below the float64 round-off of
  ~300 V node voltages for a long time. In the independent float64 check
  (below), the post-event mode becomes resolvable only after 1e-10 s at
  D3+, 1e-12 s at D3−, and 1e-8 s at D1+D4+ (event 6). Every other event
  resolves after 1e-14 s. These are the decisions that rounding makes in
  the engines.

## Diode losses, 30–40 ms average (exact)

| Diode | Total [W] | While ON [W] | ON time in window [s] |
|---|---|---|---|
| D1 | 9.7217977502e-5 | 0 | 0 |
| D2 | 0.053414538298 | 0.053393605537 | 0.0026731178533 |
| D3 | 0.053395632031 | 0.053395632030 | 0.0099878638346 |
| D4 | 5.1879950745e-5 | 0 | 0 |

The 30–40 ms window holds only a negative half cycle: D1 and D4 dissipate
only g_off·v².

## Sanity checks

| Check | Result |
|---|---|
| (b) Working precision 30 / 60 vs 50 digits | bus changes by 1.2e-26 V / −3.7e-47 V |
| (c) Energy balance over 30–40 ms: (source − R and diode dissipation − ΔE_stored) / source | −1.85e-47 (50 digits) |
| (a) Grounded variant (Rtie = 1 mΩ): exact vs independent float64 solver | 295.44501219826 vs 295.4450122106 V (Δ 1.2e-8 V); 6 commutations in both after the t < 1e-30 s start-up flips; energy balance 4.1e-52 |
| Independent float64 solver on the floating bridge | 311.3438869829 V (Δ +5.6e-6 V; float64 resolves the rail events late, see above); 11 commutations in both |
| Sensitivity, 1e-15 relative on every element (4 sign patterns) | bus moves −3.3e-13 to +3.1e-13 V |
| Sensitivity, 1e-9 relative (2 sign patterns) | bus moves −2.9e-7 and −3.3e-7 V |

- The independent float64 solver (`--scipy`) shares no code with the
  exact path. It assembles a full 4×4 MNA with the capacitor as a voltage
  source and propagates with `scipy.linalg.expm`. It finds roots with
  brentq and chooses the post-event mode by a finite nudge rather than by
  Taylor signs.
- The bus average is a smooth, linear function of the element values:
  dV/dp·p ≈ 300 V. A 1e-15 change moves it by 3e-13 V. There is no
  bifurcation (trbdf2's 0.19 V spread was a metric error, see above).
- **Start-up artifact.** At t = 0 the state is all zero and every diode
  voltage is exactly 0. At 30 and 60 digits (and in some perturbed runs),
  the Taylor mode test admits a few extra flips before t = 1.5e-17 s
  (30 digits) or 6e-33 s (60 digits). These flips carry no charge: the
  bus is unchanged to 1e-26 V, and every event after t = 1e-16 s is
  identical. The script therefore counts "commutations" (11) apart from
  "root hits".

## Not computed

- Why dsed lands on 311.3167 V. No engine was run here (no build was
  allowed). trbdf2 was measured afterwards on the TR-BDF2 track: the
  exact event sequence, 311.341410 V at rtol 2e-5 (DECISIONS #19 of
  add-branch-form-ideal-switch).
- Where the pwl run's D1/D4 chatter at 4.66 ms comes from. The exact
  turn-off is at 4.4537 ms.
- The branch form (R_on = 1e-3 Ω) gives g_on = 1/R_on = 1e3 S, so its
  model is identical, and the same truth applies (measured: the same
  311.341410 V as the conductance form).
