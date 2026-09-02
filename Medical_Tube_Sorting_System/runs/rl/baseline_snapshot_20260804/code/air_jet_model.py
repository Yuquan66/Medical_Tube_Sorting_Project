"""Distance-aware pneumatic jet model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AirJetConfig:
    """Parameters for the simplified air plume."""

    pulse_duration: float = 0.12
    reference_impulse_x: float = 0.161
    reference_impulse_z: float = 0.115
    distance_softening: float = 0.20
    nozzle_radius: float = 0.06
    spread_rate: float = 0.12
    minimum_distance_factor: float = 0.35
    maximum_distance_factor: float = 1.75
    valve_exponent: float = 2.0


@dataclass(frozen=True)
class AirJetSample:
    """One force sample from the air plume."""

    force: np.ndarray
    distance: float
    axial_distance: float
    radial_offset: float
    plume_radius: float
    distance_factor: float
    radial_factor: float
    attenuation: float


def sample_air_jet(
    tube_position: np.ndarray,
    nozzle_position: np.ndarray,
    valve_opening: float,
    reference_distance: float,
    config: AirJetConfig,
) -> AirJetSample:
    """Calculate the current pneumatic force."""
    tube_position = np.asarray(tube_position, dtype=float)
    nozzle_position = np.asarray(nozzle_position, dtype=float)
    relative = tube_position - nozzle_position

    distance = float(np.linalg.norm(relative))
    axial_distance = max(0.0, float(relative[0]))
    radial_offset = float(np.linalg.norm(relative[1:]))
    plume_radius = max(
        config.nozzle_radius,
        config.nozzle_radius + config.spread_rate * axial_distance,
    )

    softened_reference = (
        max(float(reference_distance), 1e-6)
        + config.distance_softening
    )
    softened_distance = distance + config.distance_softening
    distance_factor = float(
        np.clip(
            (softened_reference / max(softened_distance, 1e-6)) ** 2,
            config.minimum_distance_factor,
            config.maximum_distance_factor,
        )
    )
    radial_factor = float(
        np.exp(-2.0 * (radial_offset / plume_radius) ** 2)
    )
    attenuation = distance_factor * radial_factor

    opening = float(np.clip(valve_opening, 0.0, 1.0))
    valve_factor = opening ** config.valve_exponent
    reference_force = np.array(
        [
            config.reference_impulse_x / config.pulse_duration,
            0.0,
            config.reference_impulse_z / config.pulse_duration,
        ],
        dtype=float,
    )
    force = reference_force * valve_factor * attenuation

    return AirJetSample(
        force=force,
        distance=distance,
        axial_distance=axial_distance,
        radial_offset=radial_offset,
        plume_radius=plume_radius,
        distance_factor=distance_factor,
        radial_factor=radial_factor,
        attenuation=attenuation,
    )
