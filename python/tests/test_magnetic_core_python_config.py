from __future__ import annotations

import pulsim as ps


def test_magnetic_core_config_maps_to_numeric_params_and_metadata() -> None:
    config = ps.MagneticCoreConfig(
        enabled=True,
        model="hysteresis",
        loss_policy="loss_summary",
        saturation_current=2.0,
        saturation_inductance=200e-6,
        saturation_exponent=2.5,
        core_loss_k=0.4,
        core_loss_alpha=2.2,
        core_loss_freq_coeff=1e-4,
        i_equiv_init=0.1,
        hysteresis_band=0.05,
        hysteresis_strength=0.25,
        hysteresis_loss_coeff=0.3,
        hysteresis_state_init=1.0,
    )

    numeric, metadata = ps.apply_magnetic_core_config({}, {}, config)
    assert numeric["magnetic_core_enabled"] == 1.0
    assert numeric["core_loss_k"] == 0.4
    assert numeric["core_loss_alpha"] == 2.2
    assert numeric["core_loss_freq_coeff"] == 1e-4
    assert numeric["magnetic_i_equiv_init"] == 0.1
    assert numeric["saturation_current"] == 2.0
    assert numeric["saturation_inductance"] == 200e-6
    assert numeric["saturation_exponent"] == 2.5
    assert numeric["hysteresis_band"] == 0.05
    assert numeric["hysteresis_strength"] == 0.25
    assert numeric["hysteresis_loss_coeff"] == 0.3
    assert numeric["hysteresis_state_init"] == 1.0

    assert metadata["magnetic_core_model"] == "hysteresis"
    assert metadata["magnetic_core_loss_policy"] == "loss_summary"


def test_magnetic_core_config_invalid_model_raises() -> None:
    try:
        ps.MagneticCoreConfig(model="preisach").validate()
    except ps.MagneticCoreConfigError as exc:
        assert exc.code == "magnetic_core_model_unsupported"
        assert exc.field == "magnetic_core.model"
    else:
        raise AssertionError("Expected MagneticCoreConfigError")


def test_magnetic_core_config_invalid_loss_policy_raises() -> None:
    try:
        ps.MagneticCoreConfig(loss_policy="bad").validate()
    except ps.MagneticCoreConfigError as exc:
        assert exc.code == "magnetic_core_config_invalid"
        assert exc.field == "magnetic_core.loss_policy"
    else:
        raise AssertionError("Expected MagneticCoreConfigError")


def test_magnetic_core_config_negative_values_raise() -> None:
    try:
        ps.MagneticCoreConfig(core_loss_k=-0.1).validate()
    except ps.MagneticCoreConfigError as exc:
        assert exc.code == "magnetic_core_parameter_out_of_range"
        assert "core_loss_k" in exc.field
    else:
        raise AssertionError("Expected MagneticCoreConfigError")
