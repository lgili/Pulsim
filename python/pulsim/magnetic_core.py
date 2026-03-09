"""Magnetic-core configuration helpers for Pulsim virtual components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = [
    "MagneticCoreConfig",
    "MagneticCoreConfigError",
    "apply_magnetic_core_config",
]


class MagneticCoreConfigError(ValueError):
    """Raised when a magnetic-core configuration is invalid.

    Attributes
    ----------
    code : str
        Stable machine-readable error code.
    field : str
        Field path that failed validation.
    """

    def __init__(self, message: str, *, code: str, field: str) -> None:
        super().__init__(message)
        self.code = code
        self.field = field


def _normalize_token(value: str) -> str:
    return "".join(ch.lower() for ch in value.strip() if ch.isalnum() or ch in ("_", "-"))


def _validate_finite(name: str, value: float) -> None:
    if not (value == value and value not in (float("inf"), float("-inf"))):
        raise MagneticCoreConfigError(
            f"{name} must be finite",
            code="magnetic_core_parameter_out_of_range",
            field=name,
        )


def _validate_nonnegative(name: str, value: float) -> None:
    _validate_finite(name, value)
    if value < 0.0:
        raise MagneticCoreConfigError(
            f"{name} must be >= 0",
            code="magnetic_core_parameter_out_of_range",
            field=name,
        )


def _validate_positive(name: str, value: float) -> None:
    _validate_finite(name, value)
    if value <= 0.0:
        raise MagneticCoreConfigError(
            f"{name} must be > 0",
            code="magnetic_core_parameter_out_of_range",
            field=name,
        )


@dataclass(frozen=True)
class MagneticCoreConfig:
    """Typed nonlinear magnetic-core configuration for virtual components.

    Use :func:`apply_magnetic_core_config` to merge into the ``numeric_params``
    and ``metadata`` dictionaries passed to
    :meth:`pulsim.Circuit.add_virtual_component`.
    """

    enabled: bool = True
    model: str = "saturation"
    loss_policy: str = "telemetry_only"

    saturation_current: float | None = None
    saturation_inductance: float | None = None
    saturation_exponent: float | None = None

    core_loss_k: float = 0.0
    core_loss_alpha: float = 2.0
    core_loss_freq_coeff: float = 0.0
    i_equiv_init: float = 0.0

    hysteresis_band: float | None = None
    hysteresis_strength: float = 0.15
    hysteresis_loss_coeff: float = 0.2
    hysteresis_state_init: float = 1.0

    def normalized_model(self) -> str:
        token = _normalize_token(self.model or "")
        return token.replace("-", "_")

    def normalized_loss_policy(self) -> str:
        token = _normalize_token(self.loss_policy or "")
        token = token.replace("-", "_")
        if token in {"telemetryonly", "telemetry_only"}:
            return "telemetry_only"
        if token in {
            "losssummary",
            "loss_summary",
            "summary",
            "include_in_loss_summary",
            "includeinlosssummary",
        }:
            return "loss_summary"
        return token

    def validate(self) -> None:
        model = self.normalized_model()
        if model not in {"saturation", "hysteresis"}:
            raise MagneticCoreConfigError(
                f"Unsupported magnetic_core.model '{self.model}'",
                code="magnetic_core_model_unsupported",
                field="magnetic_core.model",
            )

        policy = self.normalized_loss_policy()
        if policy not in {"telemetry_only", "loss_summary"}:
            raise MagneticCoreConfigError(
                f"Unsupported magnetic_core.loss_policy '{self.loss_policy}'",
                code="magnetic_core_config_invalid",
                field="magnetic_core.loss_policy",
            )

        if self.saturation_current is not None:
            _validate_positive("magnetic_core.saturation_current", float(self.saturation_current))
        if self.saturation_inductance is not None:
            _validate_positive("magnetic_core.saturation_inductance", float(self.saturation_inductance))
        if self.saturation_exponent is not None:
            _validate_finite("magnetic_core.saturation_exponent", float(self.saturation_exponent))
            if float(self.saturation_exponent) < 1.0:
                raise MagneticCoreConfigError(
                    "magnetic_core.saturation_exponent must be >= 1",
                    code="magnetic_core_parameter_out_of_range",
                    field="magnetic_core.saturation_exponent",
                )

        _validate_nonnegative("magnetic_core.core_loss_k", float(self.core_loss_k))
        _validate_nonnegative("magnetic_core.core_loss_alpha", float(self.core_loss_alpha))
        _validate_nonnegative("magnetic_core.core_loss_freq_coeff", float(self.core_loss_freq_coeff))
        _validate_nonnegative("magnetic_core.i_equiv_init", float(self.i_equiv_init))

        if self.hysteresis_band is not None:
            _validate_nonnegative("magnetic_core.hysteresis_band", float(self.hysteresis_band))
        _validate_finite("magnetic_core.hysteresis_strength", float(self.hysteresis_strength))
        if not (0.0 <= float(self.hysteresis_strength) <= 1.0):
            raise MagneticCoreConfigError(
                "magnetic_core.hysteresis_strength must be in [0, 1]",
                code="magnetic_core_parameter_out_of_range",
                field="magnetic_core.hysteresis_strength",
            )
        _validate_nonnegative("magnetic_core.hysteresis_loss_coeff", float(self.hysteresis_loss_coeff))
        _validate_finite("magnetic_core.hysteresis_state_init", float(self.hysteresis_state_init))

    def to_numeric_params(self) -> Dict[str, float]:
        """Return numeric parameters compatible with ``Circuit.add_virtual_component``."""
        self.validate()
        params: Dict[str, float] = {}
        params["magnetic_core_enabled"] = 1.0 if self.enabled else 0.0
        params["core_loss_k"] = float(self.core_loss_k) if self.enabled else 0.0
        params["core_loss_alpha"] = float(self.core_loss_alpha)
        params["core_loss_freq_coeff"] = float(self.core_loss_freq_coeff)
        params["magnetic_i_equiv_init"] = float(self.i_equiv_init)
        if self.saturation_current is not None:
            params["saturation_current"] = float(self.saturation_current)
        if self.saturation_inductance is not None:
            params["saturation_inductance"] = float(self.saturation_inductance)
        if self.saturation_exponent is not None:
            params["saturation_exponent"] = float(self.saturation_exponent)
        if self.hysteresis_band is not None:
            params["hysteresis_band"] = float(self.hysteresis_band)
        params["hysteresis_strength"] = float(self.hysteresis_strength)
        params["hysteresis_loss_coeff"] = float(self.hysteresis_loss_coeff)
        params["hysteresis_state_init"] = float(self.hysteresis_state_init)
        return params

    def to_metadata(self) -> Dict[str, str]:
        """Return metadata compatible with ``Circuit.add_virtual_component``."""
        self.validate()
        metadata = {
            "magnetic_core_model": self.normalized_model(),
            "magnetic_core_loss_policy": self.normalized_loss_policy(),
        }
        return metadata


def apply_magnetic_core_config(
    numeric_params: Dict[str, float],
    metadata: Dict[str, str],
    config: MagneticCoreConfig,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Merge a magnetic-core config into component parameter dictionaries.

    Returns new dictionaries; the inputs are not mutated.
    """
    params = dict(numeric_params)
    params.update(config.to_numeric_params())
    meta = dict(metadata)
    meta.update(config.to_metadata())
    return params, meta
