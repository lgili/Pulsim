#!/usr/bin/env python3
"""Live oscilloscope on a closed-loop buck — interactive demo.

Opens a pyqtgraph window showing V_out, V_in, and the inductor
current in real time while the closed-loop buck simulation runs in
a worker thread. The user can:
  * Toggle individual signals via the checkboxes.
  * Click STOP to halt the kernel — partial trace is preserved.
  * Zoom / pan with the mouse (standard pyqtgraph controls).

This is the integration test for the streaming infrastructure: the
sim is intentionally LONG (5 s, 50 M steps at 100 ns dt) so the user
gets a feel for "the scope updates while it runs, I can stop early
when I see what I need".
"""

from __future__ import annotations

import threading

import pulsim as p


V_IN     = 24.0
V_REF    = 12.0
F_PWM    = 100e3
DT       = 1.0e-7
T_END    = 5.0       # 5 seconds — 50 M nominal steps; long enough


def build_plant() -> p.CircuitBuilder:
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_IN)
    b.add_mosfet_with_body_diode("Q1", "vin", "sw",
                                    R_on=1e-3, R_off=1e9, V_F=0.7)
    b.add_diode("D_FW", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    # Track the inductor's branch id so the scope can read its current.
    ind_id = b.graph.num_branches
    b.add_inductor("L1", "sw", "vout", 100e-6)
    b.add_capacitor("Cout", "vout", "gnd", 47e-6)
    b.add_resistor("R_load", "vout", "gnd", 5.0)
    return b, ind_id


def main() -> None:
    b, ind_id = build_plant()

    # Closed-loop controller chain.
    chain = p.MixedDomainBlockChain()
    chain.add("lpf", p.FirstOrderLowPass(tau=500e-6),
                inputs=dict(input_value="vout", dt="dt"),
                output="v_filt")
    chain.add("err", p.Subtract(),
                inputs=dict(a=V_REF, b="channel:v_filt"),
                output="error")
    chain.add("pi", p.PIController(Kp=0.020, Ki=150.0,
                                       output_min=0.05,
                                       output_max=0.95),
                inputs=dict(setpoint=V_REF,
                              measured="channel:v_filt", dt="dt"),
                output="duty")
    chain.add("pwm", p.PwmGenerator(frequency=F_PWM),
                inputs=dict(duty="channel:duty", t="time"),
                output="gate")
    chain_observer = chain.make_step_observer(b, dt=DT,
                                                    use_kernel=True)
    sw_fn = chain.make_pwm_switch_fn("gate",
                                          num_switches=b.graph.num_switches,
                                          switch_idx=0)

    # Streaming setup. With dt=100 ns, downsample=100 → 1 sample
    # every 10 µs sent to the GUI (100 kHz visual rate — way more
    # than the human eye can use). batch_size=50 ⇒ flush every 5 ms
    # of sim time. Very smooth even at high sim throughput.
    stream = p.LiveStream(batch_size=50, max_queue=400,
                              downsample=100)

    # Compose the step_observer: the chain's observer + the stream's.
    def combined_observer(t, x):
        chain_observer(t, x)
        stream.step_observer(t, x)

    def run_sim():
        try:
            return p.simulate(
                b, t_end=T_END, dt=DT,
                switch_fn=sw_fn,
                step_observer=combined_observer,
                should_continue=stream.should_continue,
                max_event_iterations=8,
            )
        finally:
            stream.flush_pending()

    print(f"  Buck V_in={V_IN}V → V_out target {V_REF}V "
          f"(closed-loop PI)")
    print(f"  Simulation: dt={DT*1e9:.0f} ns, t_end={T_END} s "
          f"({int(T_END/DT)} steps nominal)")
    print("  Live scope opening — close the window or click STOP "
          "to halt.")

    sim_thread = threading.Thread(target=run_sim, daemon=True)
    sim_thread.start()

    # Build + start the scope (blocks until the window closes).
    # Multi-panel layout: voltages on top, current in the middle,
    # control signals at the bottom — shared X axis, independent Y.
    scope = p.LiveScope(b, stream,
                            panels=["Voltage", "Current", "Control"],
                            window_seconds=2.0,
                            update_hz=30.0,
                            title="Pulsim — Buck CL live")
    scope.add_node_voltage("vin",  panel="Voltage", color="#4e79a7")
    scope.add_node_voltage("vout", panel="Voltage", color="#f28e2b")
    scope.add_chain_channel(chain, "v_filt", panel="Voltage",
                                 label="v_filt", unit="V",
                                 color="#9467bd")
    scope.add_branch_current(ind_id, panel="Current",
                                  label="I(L1)", color="#59a14f")
    scope.add_chain_channel(chain, "duty", panel="Control",
                                 label="duty", color="#e15759",
                                 fmt="{:+6.3f}")

    scope.start()

    # Ensure the worker has stopped if user closed the window.
    stream.stop()
    sim_thread.join(timeout=2.0)
    print(f"\n  Done.")
    print(f"    samples received: {stream.n_steps_received}")
    print(f"    batches emitted:  {stream.n_batches_emitted}")
    print(f"    batches dropped:  {stream.n_batches_dropped}")


if __name__ == "__main__":
    main()
