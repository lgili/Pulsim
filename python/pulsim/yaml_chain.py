"""YAML `chain:` section wiring for composite devices (Phase 2 close).

The C++ YAML loader (`load_yaml_string` / `load_yaml_file`) builds the
circuit topology and returns a `LoadedCircuit(builder, options)`. This
module adds the second step that was deferred from the loader because
chain wiring lives in a separate kernel module (`CxxBlockChain`):

    loaded = pulsim.load_yaml_file("im_dol.yaml")
    chain = pulsim.wire_chain_from_yaml(loaded, '''
    - type: induction_motor
      device: IM
      J: 0.01
      B: 0.0
      T_load: 0.0
      omega_channel: omega
      theta_channel: theta
      torque_channel: torque
    ''')

The chain spec resolves branch indices via `builder.branch_index_of(name)`
using the deterministic names the YAML loader created (e.g.,
``IM_Lsig_a``, ``IM_E_a``, ``L_core_L0``, ``L_core_V_M``).

Supported chain block types:
    - induction_motor
    - hysteretic_inductor
    - sliding_mode_observer
    - flux_mras_observer
    - (extend by adding new entries to `_BLOCK_HANDLERS`)
"""
from __future__ import annotations

from typing import Any, Mapping

import yaml as _yaml  # PyYAML — already a runtime dep of pulsim

from . import _pulsim as _k


__all__ = ["wire_chain_from_yaml"]


def _resolve_im_indices(builder, device_name: str):
    """Resolve the 6 state-vector indices the IM C++ adapter needs
    from the deterministic branch names created by the YAML loader."""
    graph = builder.graph
    pool = builder.pool
    phase_inductor_idx = [
        pool.branch_var_id_for_inductor(
            builder.branch_id_of(f"{device_name}_Lsig_{p}"), graph)
        for p in ("a", "b", "c")
    ]
    bemf_source_idx = [
        pool.branch_var_id_for_source(
            builder.branch_id_of(f"{device_name}_E_{p}"), graph)
        for p in ("a", "b", "c")
    ]
    return phase_inductor_idx, bemf_source_idx


def _resolve_hysteretic_indices(builder, device_name: str):
    """Resolve the 2 state-vector indices the JA adapter needs."""
    graph = builder.graph
    pool = builder.pool
    inductor_idx = pool.branch_var_id_for_inductor(
        builder.branch_id_of(f"{device_name}_L0"), graph)
    bemf_idx = pool.branch_var_id_for_source(
        builder.branch_id_of(f"{device_name}_V_M"), graph)
    return inductor_idx, bemf_idx


def _add_induction_motor_block(chain, builder, spec: Mapping[str, Any]):
    """Wire an `add_induction_motor` call from a chain-spec dict."""
    device_name = str(spec["device"])
    phase_idx, bemf_idx = _resolve_im_indices(builder, device_name)

    # Pull motor parameters back from the original YAML device? We
    # require the user to re-state R_s / L_s / R_r / L_r / L_m /
    # pole_pairs in the chain spec — the loader doesn't keep them in
    # an indexed structure that we can re-query. This duplication
    # would go away once the chain section becomes a true compiler
    # pass that reads both topology and chain in one shot (v1.6).
    chain.add_induction_motor(
        J=float(spec.get("J", 1e-6)),
        B=float(spec.get("B", 0.0)),
        T_load=float(spec.get("T_load", 0.0)),
        R_s=float(spec["R_s"]),
        L_s=float(spec["L_s"]),
        R_r=float(spec["R_r"]),
        L_r=float(spec["L_r"]),
        L_m=float(spec["L_m"]),
        pole_pairs=int(spec["pole_pairs"]),
        phase_inductor_idx=phase_idx,
        bemf_source_idx=bemf_idx,
        omega_channel=str(spec.get("omega_channel", "omega")),
        theta_channel=str(spec.get("theta_channel", "theta")),
        psi_alpha_channel=str(spec.get("psi_alpha_channel", "")),
        psi_beta_channel=str(spec.get("psi_beta_channel", "")),
        torque_channel=str(spec.get("torque_channel", "")),
        slip_channel=str(spec.get("slip_channel", "")),
    )


def _add_hysteretic_inductor_block(chain, builder, spec: Mapping[str, Any]):
    """Wire an `add_hysteretic_inductor` call from a chain-spec dict."""
    device_name = str(spec["device"])
    inductor_idx, bemf_idx = _resolve_hysteretic_indices(builder, device_name)
    chain.add_hysteretic_inductor(
        Ms=float(spec["Ms"]),
        a=float(spec["a"]),
        alpha=float(spec["alpha"]),
        c=float(spec["c"]),
        k=float(spec["k"]),
        N_turns=int(spec["N_turns"]),
        l_m=float(spec["l_m"]),
        A_core=float(spec["A_core"]),
        inductor_branch_var_idx=inductor_idx,
        bemf_source_idx=bemf_idx,
        M_channel=str(spec.get("M_channel", "")),
        B_channel=str(spec.get("B_channel", "")),
        H_channel=str(spec.get("H_channel", "")),
        vm_channel=str(spec.get("vm_channel", "")),
    )


def _add_sliding_mode_observer_block(chain, builder, spec: Mapping[str, Any]):
    """Wire a PMSM Sliding-Mode Observer (Phase 2.3 C++ port).

    Pure channel-based — no topology indices needed. The block reads
    ``v_alpha/v_beta/i_alpha/i_beta`` from the chain channel bus and
    writes ``theta_hat`` / ``omega_hat`` (and optional back-EMF /
    low-speed-flag channels)."""
    del builder  # unused: SMO is pure channel I/O
    kwargs: dict[str, Any] = {
        "Rs": float(spec["Rs"]),
        "Ls": float(spec["Ls"]),
        "v_alpha_channel": str(spec.get("v_alpha_channel", "v_alpha")),
        "v_beta_channel":  str(spec.get("v_beta_channel",  "v_beta")),
        "i_alpha_channel": str(spec.get("i_alpha_channel", "i_alpha")),
        "i_beta_channel":  str(spec.get("i_beta_channel",  "i_beta")),
        "theta_hat_channel": str(spec.get("theta_hat_channel", "theta_hat")),
        "omega_hat_channel": str(spec.get("omega_hat_channel", "omega_hat")),
    }
    for opt in ("K_sl", "omega_lpf", "Kp_pll", "Ki_pll",
                "f_init_hz", "epsilon", "low_speed_threshold"):
        if opt in spec:
            kwargs[opt] = float(spec[opt])
    for opt in ("e_alpha_hat_channel", "e_beta_hat_channel",
                "low_speed_flag_channel"):
        if opt in spec:
            kwargs[opt] = str(spec[opt])
    chain.add_sliding_mode_observer(**kwargs)


def _add_flux_mras_observer_block(chain, builder, spec: Mapping[str, Any]):
    """Wire an IM Flux-MRAS Observer (Phase 2.3 C++ port).

    Pure channel-based — no topology indices needed. Reads
    ``v_alpha/v_beta/i_alpha/i_beta`` and writes ``omega_hat`` (and
    optional ψ_adj / ψ_ref channels for telemetry)."""
    del builder  # unused: MRAS is pure channel I/O
    kwargs: dict[str, Any] = {
        "Rs": float(spec["Rs"]),
        "Ls": float(spec["Ls"]),
        "Lr": float(spec["Lr"]),
        "Lm": float(spec["Lm"]),
        "v_alpha_channel": str(spec.get("v_alpha_channel", "v_alpha")),
        "v_beta_channel":  str(spec.get("v_beta_channel",  "v_beta")),
        "i_alpha_channel": str(spec.get("i_alpha_channel", "i_alpha")),
        "i_beta_channel":  str(spec.get("i_beta_channel",  "i_beta")),
        "omega_hat_channel": str(spec.get("omega_hat_channel", "omega_hat")),
    }
    for opt in ("Rr", "voltage_model_hpf_omega",
                "Kp_mras", "Ki_mras", "eps_norm_floor"):
        if opt in spec:
            kwargs[opt] = float(spec[opt])
    if "normalise_eps" in spec:
        kwargs["normalise_eps"] = bool(spec["normalise_eps"])
    for opt in ("psi_adj_alpha_channel", "psi_adj_beta_channel",
                "psi_ref_alpha_channel", "psi_ref_beta_channel"):
        if opt in spec:
            kwargs[opt] = str(spec[opt])
    chain.add_flux_mras_observer(**kwargs)


_BLOCK_HANDLERS = {
    "induction_motor": _add_induction_motor_block,
    "hysteretic_inductor": _add_hysteretic_inductor_block,
    "sliding_mode_observer": _add_sliding_mode_observer_block,
    "flux_mras_observer": _add_flux_mras_observer_block,
}


def wire_chain_from_yaml(loaded, chain_spec):
    """Wire a `CxxBlockChain` from a YAML or Python chain spec.

    Parameters
    ----------
    loaded : LoadedCircuit
        Return value of :func:`pulsim.load_yaml_string` /
        :func:`pulsim.load_yaml_file`.
    chain_spec : str | list[dict]
        YAML string (parsed via PyYAML) OR a Python list of dicts,
        each with at least a ``type`` field.

    Returns
    -------
    CxxBlockChain
        Populated chain ready to use via
        ``chain.make_step_observer(dt)`` + ``chain.make_b_extra_fn(N)``.

    Raises
    ------
    KeyError
        If a block type is not supported or a referenced device name
        was not found in the loaded builder.
    ValueError
        If the chain spec is malformed.
    """
    if isinstance(chain_spec, str):
        chain_blocks = _yaml.safe_load(chain_spec)
    else:
        chain_blocks = chain_spec
    if chain_blocks is None:
        return _k.CxxBlockChain()
    if not isinstance(chain_blocks, list):
        raise ValueError(
            "chain spec must be a YAML list (or Python list of dicts); "
            f"got {type(chain_blocks).__name__}")

    chain = _k.CxxBlockChain()
    for i, block in enumerate(chain_blocks):
        if not isinstance(block, dict):
            raise ValueError(
                f"chain block #{i} must be a mapping; got "
                f"{type(block).__name__}")
        block_type = block.get("type")
        if block_type is None:
            raise ValueError(
                f"chain block #{i} missing required field 'type'")
        handler = _BLOCK_HANDLERS.get(str(block_type))
        if handler is None:
            raise KeyError(
                f"chain block #{i} has unknown type {block_type!r}; "
                f"supported: {sorted(_BLOCK_HANDLERS)}")
        handler(chain, loaded.builder, block)
    return chain
