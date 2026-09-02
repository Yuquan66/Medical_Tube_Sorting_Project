"""Shared environment for pneumatic tube sorting."""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from air_jet_model import AirJetConfig, sample_air_jet
from tube_specs import (
    JET_REFERENCE_IMPULSE_X_VALUES,
    JET_REFERENCE_IMPULSE_Z_VALUES,
    TUBE_CLASSES,
    TUBE_RADII_AT_UNIT_SCALE,
    TUBE_SPECS,
    TUBE_URDF_FILENAMES,
)


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".runtime" / "matplotlib"))
URDF_DIR = BASE_DIR / "urdfs"

DISCRETE_VALVE_LEVELS = (0.65, 0.70, 0.80, 0.90, 1.00)

TUBE_URDF_PATHS = [
    URDF_DIR / "tubes" / filename
    for filename in TUBE_URDF_FILENAMES
]

HORIZONTAL_CONVEYOR_URDF = (
    URDF_DIR / "horizontal_conveyor_complete.urdf"
)
AIR_JET_URDF = URDF_DIR / "air_jet.urdf"
BIN_URDF = URDF_DIR / "bin.urdf"
BIN_FUNNEL_URDF = URDF_DIR / "bin_funnel.urdf"


@dataclass(frozen=True)
class SortingConfig:
    """Frozen phase-2 baseline plus RL decision/reward settings."""

    time_step: float = 1.0 / 240.0
    solver_iterations: int = 80
    physics_substeps: int = 2
    control_frame_skip: int = 8  # Run the controller at 30 Hz.

    system_x: float = -2.014984
    system_y: float = -6.364140
    system_z: float = 9.124548
    system_yaw: float = 1.570796

    horizontal_belt_speed: float = 1.0
    tube_lateral_offset: float = 0.35
    tube_scale: float = 1.0

    bin_spacing: float = 2.0
    jet_to_bin_lead: float = 0.50
    jet_trigger_lead: float = 0.02
    jet_trigger_half_width: float = 0.25
    preferred_trigger_offset: float = -0.08
    timing_reward_sigma: float = 0.08
    jet_impulse_x: float = JET_REFERENCE_IMPULSE_X_VALUES[0]
    jet_impulse_z: float = JET_REFERENCE_IMPULSE_Z_VALUES[0]
    jet_impulse_x_by_jet: tuple[float, ...] = (
        JET_REFERENCE_IMPULSE_X_VALUES
    )
    jet_impulse_z_by_jet: tuple[float, ...] = (
        JET_REFERENCE_IMPULSE_Z_VALUES
    )
    jet_pulse_duration: float = 0.12
    jet_distance_softening: float = 0.20
    jet_nozzle_radius: float = 0.06
    jet_spread_rate: float = 0.12
    jet_minimum_distance_factor: float = 0.35
    jet_maximum_distance_factor: float = 1.75
    jet_valve_exponent: float = 2.0
    valve_deadband: float = 0.60
    result_timeout_seconds: float = 3.0
    bin_capture_depth: float = 0.25
    bin_drop_below_belt: float = 0.15
    bin_confirmation_seconds: float = 0.05

    bin_friction: float = 0.20
    end_of_sorting_y: float = 9.20
    decision_start_distance: float = 0.30
    max_episode_decisions: int = 650

    correct_sort_reward: float = 50.0
    correct_jet_reward: float = 15.0
    failure_penalty: float = 15.0
    misfire_penalty: float = 1.0
    wrong_nozzle_penalty: float = 5.0
    missed_opportunity_penalty: float = 5.0
    correct_wait_reward: float = 0.05
    delay_penalty: float = 0.002
    energy_penalty: float = 0.02
    correct_jet_intensity_weight: float = 0.50
    actuation_intensity_penalty: float = 0.0


class ConveyorSortingEnv(gym.Env):
    """Single-tube pneumatic sorting task shared by all RL algorithms."""

    metadata = {
        "render_modes": [None, "human"],
        "render_fps": 30,
    }

    def __init__(
        self,
        *,
        action_mode: str = "discrete",
        render_mode: str | None = None,
        recognition_error_probability: float = 0.0,
        config: SortingConfig | None = None,
    ) -> None:
        super().__init__()
        if action_mode not in {"discrete", "continuous"}:
            raise ValueError(
                "action_mode must be 'discrete' or 'continuous'."
            )
        if render_mode not in {None, "human"}:
            raise ValueError("render_mode must be None or 'human'.")
        if not 0.0 <= recognition_error_probability < 1.0:
            raise ValueError(
                "recognition_error_probability must be in [0, 1)."
            )

        self.action_mode = action_mode
        self.render_mode = render_mode
        self.recognition_error_probability = (
            recognition_error_probability
        )
        self.config = config or SortingConfig()

        self._validate_assets()

        if self.action_mode == "discrete":
            self.action_space = spaces.Discrete(
                1 + len(DISCRETE_VALVE_LEVELS)
            )
        else:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            )

        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        connection_mode = p.GUI if render_mode == "human" else p.DIRECT
        self.physics_client = p.connect(connection_mode)
        if self.physics_client < 0:
            raise RuntimeError("Could not connect to PyBullet.")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.system_position = np.array(
            [
                self.config.system_x,
                self.config.system_y,
                self.config.system_z,
            ],
            dtype=float,
        )
        self.system_orientation = p.getQuaternionFromEuler(
            [0.0, 0.0, self.config.system_yaw]
        )
        if (
            len(self.config.jet_impulse_x_by_jet) != len(TUBE_CLASSES)
            or len(self.config.jet_impulse_z_by_jet) != len(TUBE_CLASSES)
        ):
            raise ValueError("Each jet impulse list must contain five values.")
        self.air_jet_configs = tuple(
            AirJetConfig(
                pulse_duration=self.config.jet_pulse_duration,
                reference_impulse_x=(
                    self.config.jet_impulse_x_by_jet[index]
                ),
                reference_impulse_z=(
                    self.config.jet_impulse_z_by_jet[index]
                ),
                distance_softening=self.config.jet_distance_softening,
                nozzle_radius=self.config.jet_nozzle_radius,
                spread_rate=self.config.jet_spread_rate,
                minimum_distance_factor=(
                    self.config.jet_minimum_distance_factor
                ),
                maximum_distance_factor=(
                    self.config.jet_maximum_distance_factor
                ),
                valve_exponent=self.config.jet_valve_exponent,
            )
            for index in range(len(TUBE_CLASSES))
        )
        self.tube_radii = np.array(
            TUBE_RADII_AT_UNIT_SCALE,
            dtype=float,
        )

        self.tube_id: int | None = None
        self.tube_ids: list[int] = []
        self.tube_parking_positions: list[tuple[float, float, float]] = []
        self.true_type: int | None = None
        self.detected_type: int | None = None
        self.tube_mass = TUBE_SPECS[0].total_mass_kg
        self.tube_radius = 0.1
        self.lateral_offset = 0.0
        self.tube_axis_direction = 1.0
        self.transport_orientation = (0.0, 0.0, 0.0, 1.0)
        self.horizontal_y = self.config.system_y + 0.02
        self.measurement_x = self.config.system_x
        self.measurement_y = self.horizontal_y
        self.measurement_physics_step = 0
        self.fired = False
        self.jet_commanded = False
        self.fired_jet_index: int | None = None
        self.fired_intensity = 0.0
        self.jet_command_physics_step: int | None = None
        self.jet_command_x: float | None = None
        self.jet_command_y_offset: float | None = None
        self.fire_physics_step: int | None = None
        self.jet_distance: float | None = None
        self.jet_radial_offset: float | None = None
        self.jet_attenuation: float | None = None
        self.jet_peak_force = 0.0
        self.jet_accumulated_impulse = 0.0
        self.bin_candidate_index: int | None = None
        self.bin_candidate_steps = 0
        self.physics_steps = 0
        self.decision_steps = 0
        self.cumulative_valve_opening = 0.0
        self.misfire_count = 0
        self.outcome: str | None = None
        self._episode_done = False

        self.horizontal_conveyor_id: int | None = None
        self.jet_positions: list[float] = []
        self.jet_nozzle_positions: list[tuple[float, float, float]] = []
        self.jet_reference_distance = 1.0
        self.bin_aabbs: list[Any] = [None] * len(TUBE_CLASSES)
        self.bin_entry_x = -0.39
        self.jet_trigger_x_min = -3.64

        self._build_static_scene()

    def _validate_assets(self) -> None:
        required = [
            HORIZONTAL_CONVEYOR_URDF,
            AIR_JET_URDF,
            BIN_URDF,
            BIN_FUNNEL_URDF,
            *TUBE_URDF_PATHS,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing RL environment assets:\n" + "\n".join(missing)
            )

    def _build_static_scene(self) -> None:
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(self.config.time_step)
        p.setPhysicsEngineParameter(
            numSolverIterations=self.config.solver_iterations,
            numSubSteps=self.config.physics_substeps,
        )
        p.loadURDF("plane.urdf")

        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=24.0,
                cameraYaw=45.0,
                cameraPitch=-35.0,
                cameraTargetPosition=[0.0, -1.0, 8.0],
            )

        self.horizontal_conveyor_id = p.loadURDF(
            str(HORIZONTAL_CONVEYOR_URDF),
            basePosition=self.system_position.tolist(),
            baseOrientation=self.system_orientation,
            useFixedBase=True,
        )
        p.changeVisualShape(
            self.horizontal_conveyor_id,
            -1,
            rgbaColor=[0.35, 0.75, 0.35, 1.0],
        )

        self.jet_positions = []
        self.jet_nozzle_positions = []
        self.bin_aabbs = [None] * len(TUBE_CLASSES)

        for index in range(len(TUBE_CLASSES)):
            bin_local_offset = [
                self.config.bin_spacing * index,
                0.0,
                0.0,
            ]
            bin_world_position, bin_world_orientation = p.multiplyTransforms(
                self.system_position.tolist(),
                self.system_orientation,
                bin_local_offset,
                [0.0, 0.0, 0.0, 1.0],
            )
            jet_local_offset = [
                (
                    self.config.bin_spacing * index
                    - self.config.jet_to_bin_lead
                ),
                0.0,
                0.0,
            ]
            jet_world_position, jet_world_orientation = p.multiplyTransforms(
                self.system_position.tolist(),
                self.system_orientation,
                jet_local_offset,
                [0.0, 0.0, 0.0, 1.0],
            )

            jet_id = p.loadURDF(
                str(AIR_JET_URDF),
                basePosition=jet_world_position,
                baseOrientation=jet_world_orientation,
                useFixedBase=True,
            )
            p.changeVisualShape(
                jet_id,
                -1,
                rgbaColor=[0.63, 0.63, 0.63, 1.0],
            )
            jet_aabb = p.getAABB(jet_id, -1)
            jet_center_y = (
                jet_aabb[0][1] + jet_aabb[1][1]
            ) * 0.5
            jet_center_z = (
                jet_aabb[0][2] + jet_aabb[1][2]
            ) * 0.5
            self.jet_positions.append(float(jet_center_y))
            self.jet_nozzle_positions.append(
                (
                    float(jet_aabb[1][0]),
                    float(jet_center_y),
                    float(jet_center_z),
                )
            )
            if index == 0:
                self.jet_trigger_x_min = jet_aabb[1][0] + 0.02
                self.jet_reference_distance = abs(
                    self.config.system_x - jet_aabb[1][0]
                )

            bin_path = BIN_URDF if index == 0 else BIN_FUNNEL_URDF
            bin_id = p.loadURDF(
                str(bin_path),
                basePosition=bin_world_position,
                baseOrientation=bin_world_orientation,
                useFixedBase=True,
            )
            p.changeVisualShape(
                bin_id,
                -1,
                rgbaColor=[0.63, 0.63, 0.63, 1.0],
            )
            p.changeDynamics(
                bin_id,
                -1,
                lateralFriction=self.config.bin_friction,
                restitution=0.0,
            )
            self.bin_aabbs[index] = p.getAABB(bin_id, -1)
            if index == 0:
                self.bin_entry_x = (
                    self.bin_aabbs[index][0][0] - 0.02
                )

        self._build_tube_pool()

    def _set_tube_collision(self, body_id: int, enabled: bool) -> None:
        """Enable or disable collisions for one pooled tube."""
        group = 1 if enabled else 0
        mask = 1 if enabled else 0
        for link_index in range(-1, p.getNumJoints(body_id)):
            p.setCollisionFilterGroupMask(
                body_id,
                link_index,
                collisionFilterGroup=group,
                collisionFilterMask=mask,
            )

    def _build_tube_pool(self) -> None:
        """Load each tube once and park it outside the scene."""
        self.tube_ids = []
        self.tube_parking_positions = []
        for tube_type, tube_path in enumerate(TUBE_URDF_PATHS):
            parking_position = (
                self.config.system_x,
                self.config.system_y - 30.0,
                -20.0 - 2.0 * tube_type,
            )
            body_id = p.loadURDF(
                str(tube_path),
                basePosition=parking_position,
                globalScaling=self.config.tube_scale,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
            )
            self._set_tube_collision(body_id, False)
            self.tube_ids.append(body_id)
            self.tube_parking_positions.append(parking_position)

            loaded_mass = sum(
                p.getDynamicsInfo(body_id, link_index)[0]
                for link_index in range(
                    -1,
                    p.getNumJoints(body_id),
                )
            )
            expected_mass = TUBE_SPECS[tube_type].total_mass_kg
            if not np.isclose(loaded_mass, expected_mass, atol=1e-9):
                raise RuntimeError(
                    f"Tube class {tube_type} mass is "
                    f"{loaded_mass:.6f} kg; expected "
                    f"{expected_mass:.6f} kg from its URDF."
                )

    def _park_active_tube(self) -> None:
        """Move the active tube back to its collision-free parking point."""
        if self.tube_id is None or self.true_type is None:
            return
        self._set_tube_collision(self.tube_id, False)
        p.resetBasePositionAndOrientation(
            self.tube_id,
            self.tube_parking_positions[int(self.true_type)],
            [0.0, 0.0, 0.0, 1.0],
        )
        p.resetBaseVelocity(
            self.tube_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
        )

    def _sample_detected_type(self, true_type: int) -> int:
        if (
            self.np_random.random()
            >= self.recognition_error_probability
        ):
            return true_type
        alternatives = [
            class_id
            for class_id in TUBE_CLASSES
            if class_id != true_type
        ]
        return int(self.np_random.choice(alternatives))

    def _spawn_tube(
        self,
        options: dict[str, Any] | None = None,
    ) -> None:
        options = options or {}
        requested_type = options.get("tube_type")
        if requested_type is None:
            self.true_type = int(self.np_random.integers(0, 5))
        else:
            self.true_type = int(requested_type)
            if self.true_type not in TUBE_CLASSES:
                raise ValueError("tube_type must be between 0 and 4.")
        self.detected_type = self._sample_detected_type(self.true_type)
        self.tube_radius = float(
            self.tube_radii[self.true_type]
            * self.config.tube_scale
            + 0.004
        )
        requested_offset = options.get("lateral_offset")
        if requested_offset is None:
            self.lateral_offset = float(
                self.np_random.uniform(
                    -self.config.tube_lateral_offset,
                    self.config.tube_lateral_offset,
                )
            )
        else:
            self.lateral_offset = float(requested_offset)
            if abs(self.lateral_offset) > self.config.tube_lateral_offset:
                raise ValueError(
                    "lateral_offset is outside the configured range."
                )
        target_jet_y = self.jet_positions[int(self.detected_type)]
        self.horizontal_y = (
            target_jet_y - self.config.decision_start_distance
        )
        self.measurement_x = (
            self.config.system_x + self.lateral_offset
        )
        self.measurement_y = self.horizontal_y
        self.measurement_physics_step = self.physics_steps

        requested_direction = options.get("axis_direction")
        if requested_direction is None:
            self.tube_axis_direction = float(
                self.np_random.choice([-1.0, 1.0])
            )
        else:
            self.tube_axis_direction = float(requested_direction)
            if self.tube_axis_direction not in {-1.0, 1.0}:
                raise ValueError("axis_direction must be -1 or 1.")
        self.transport_orientation = p.getQuaternionFromEuler(
            [
                0.0,
                self.tube_axis_direction * np.pi / 2.0,
                0.0,
            ]
        )

        start_position = [
            self.config.system_x + self.lateral_offset,
            self.horizontal_y,
            self.config.system_z + self.tube_radius,
        ]
        self.tube_id = self.tube_ids[int(self.true_type)]
        p.resetBasePositionAndOrientation(
            self.tube_id,
            start_position,
            self.transport_orientation,
        )
        p.resetBaseVelocity(
            self.tube_id,
            linearVelocity=[0.0, self.config.horizontal_belt_speed, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
        )
        self._set_tube_collision(self.tube_id, True)
        self.tube_mass = sum(
            p.getDynamicsInfo(self.tube_id, link_index)[0]
            for link_index in range(
                -1,
                p.getNumJoints(self.tube_id),
            )
        )
        expected_mass = TUBE_SPECS[self.true_type].total_mass_kg
        if not np.isclose(self.tube_mass, expected_mass, atol=1e-9):
            raise RuntimeError(
                f"Tube class {self.true_type} mass is "
                f"{self.tube_mass:.6f} kg; expected "
                f"{expected_mass:.6f} kg from its URDF."
            )

    def _get_obs(self) -> np.ndarray:
        position = self._estimated_position()
        x_normalized = float(
            np.clip(
                (
                    position[0] - self.config.system_x
                )
                / self.config.tube_lateral_offset,
                -1.0,
                1.0,
            )
        )
        target_jet_y = self.jet_positions[int(self.detected_type)]
        y_normalized = float(
            np.clip(position[1] - target_jet_y, -1.0, 1.0)
        )
        class_normalized = float(
            (int(self.detected_type) - 2.0) / 2.0
        )
        return np.array(
            [x_normalized, y_normalized, class_normalized],
            dtype=np.float32,
        )

    def _tube_position(self) -> tuple[float, float, float]:
        if self.tube_id is None:
            return (
                self.config.system_x,
                self.horizontal_y,
                self.config.system_z,
            )
        position, _ = p.getBasePositionAndOrientation(self.tube_id)
        return tuple(float(value) for value in position)

    def _estimated_position(self) -> tuple[float, float, float]:
        elapsed_steps = max(
            0,
            self.physics_steps - self.measurement_physics_step,
        )
        estimated_y = (
            self.measurement_y
            + self.config.horizontal_belt_speed
            * elapsed_steps
            * self.config.time_step
        )
        return (
            float(self.measurement_x),
            float(estimated_y),
            float(self.config.system_z + self.tube_radius),
        )

    def _decode_action(
        self,
        action: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return physical valve openings and raw normalized commands."""
        if self.action_mode == "discrete":
            action_index = int(np.asarray(action).item())
            if not self.action_space.contains(action_index):
                raise ValueError(f"Invalid discrete action: {action_index}")
            openings = np.zeros(5, dtype=np.float32)
            if action_index > 0:
                level_index = action_index - 1
                openings[int(self.detected_type)] = (
                    DISCRETE_VALVE_LEVELS[level_index]
                )
            return openings, openings.copy()

        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.shape != (1,):
            raise ValueError(
                f"Continuous action must have shape (1,), got {raw.shape}."
            )
        raw = np.clip(raw, -1.0, 1.0)
        openings = np.zeros(5, dtype=np.float32)
        openings[int(self.detected_type)] = float(
            (raw[0] + 1.0) * 0.5
        )
        return openings, raw

    def _current_local_jet(
        self,
        position: tuple[float, float, float],
    ) -> int | None:
        for jet_index, jet_y in enumerate(self.jet_positions):
            if (
                abs(position[1] - jet_y)
                < self.config.jet_trigger_half_width
                and self.jet_trigger_x_min
                < position[0]
                < self.bin_entry_x
                and 8.80 < position[2] < 9.50
            ):
                return jet_index
        return None

    def _schedule_air_jet(
        self,
        jet_index: int,
        intensity: float,
    ) -> None:
        """Schedule one valve pulse."""
        self.jet_commanded = True
        self.fired_jet_index = jet_index
        self.fired_intensity = float(intensity)
        self.jet_command_physics_step = self.physics_steps
        estimated_position = self._estimated_position()
        self.jet_command_x = float(estimated_position[0])
        self.jet_command_y_offset = float(
            estimated_position[1] - self.jet_positions[jet_index]
        )

    def _start_air_jet(self) -> None:
        """Start one air pulse."""
        self.fired = True
        self.fire_physics_step = self.physics_steps

        if self.render_mode == "human":
            position = self._tube_position()
            p.addUserDebugLine(
                list(
                    self.jet_nozzle_positions[
                        self.fired_jet_index
                    ]
                ),
                list(position),
                [1.0, 0.0, 0.0],
                lineWidth=2.0,
                lifeTime=self.config.jet_pulse_duration,
            )

    def _update_air_jet_force(self) -> None:
        """Apply the active distance-aware force."""
        if not self.jet_commanded or self.fired_jet_index is None:
            return

        position = self._tube_position()
        if not self.fired:
            self._start_air_jet()

        elapsed = (
            self.physics_steps - self.fire_physics_step
        ) * self.config.time_step
        if elapsed >= self.config.jet_pulse_duration:
            return

        sample = sample_air_jet(
            np.asarray(position, dtype=float),
            np.asarray(
                self.jet_nozzle_positions[self.fired_jet_index],
                dtype=float,
            ),
            self.fired_intensity,
            self.jet_reference_distance,
            self.air_jet_configs[self.fired_jet_index],
        )
        p.applyExternalForce(
            self.tube_id,
            -1,
            sample.force.tolist(),
            list(position),
            p.WORLD_FRAME,
        )
        force_magnitude = float(np.linalg.norm(sample.force))
        self.jet_peak_force = max(
            self.jet_peak_force,
            force_magnitude,
        )
        self.jet_accumulated_impulse += (
            force_magnitude * self.config.time_step
        )
        if self.jet_distance is None:
            self.jet_distance = sample.distance
            self.jet_radial_offset = sample.radial_offset
            self.jet_attenuation = sample.attenuation

    def _actuation_reward(
        self,
        valve_openings: np.ndarray,
    ) -> float:
        reward = -self.config.delay_penalty
        if self.jet_commanded:
            return reward

        reward -= (
            self.config.energy_penalty
            * float(np.sum(valve_openings))
        )
        self.cumulative_valve_opening += float(
            np.sum(valve_openings)
        )

        commanded = (
            valve_openings > self.config.valve_deadband
        )
        estimated_position = self._estimated_position()
        local_jet = self._current_local_jet(estimated_position)
        timing_score = 0.0
        if local_jet is not None:
            trigger_offset = (
                estimated_position[1] - self.jet_positions[local_jet]
            )
            timing_error = (
                trigger_offset - self.config.preferred_trigger_offset
            )
            timing_score = float(
                np.exp(
                    -0.5
                    * (
                        timing_error / self.config.timing_reward_sigma
                    )
                    ** 2
                )
            )

        for jet_index, is_commanded in enumerate(commanded):
            if not is_commanded:
                continue
            if jet_index != self.true_type:
                self.misfire_count += 1
                reward -= (
                    self.config.wrong_nozzle_penalty
                    * float(valve_openings[jet_index])
                )
            elif jet_index != local_jet:
                reward -= (
                    self.config.misfire_penalty
                    * float(valve_openings[jet_index])
                )

        if (
            local_jet == self.true_type
            and not commanded[local_jet]
        ):
            if timing_score >= 0.5:
                reward -= (
                    self.config.missed_opportunity_penalty
                    * timing_score
                )
            elif not np.any(commanded):
                reward += self.config.correct_wait_reward

        if (
            not self.jet_commanded
            and local_jet is not None
            and commanded[local_jet]
        ):
            intensity = float(valve_openings[local_jet])
            if local_jet == self.true_type:
                timing_reward = (
                    self.config.correct_jet_reward * timing_score
                    - self.config.misfire_penalty
                    * (1.0 - timing_score)
                )
                reward += timing_reward * (
                    1.0
                    - self.config.correct_jet_intensity_weight
                    + self.config.correct_jet_intensity_weight
                    * intensity
                )
            else:
                self.misfire_count += 1
                reward -= (
                    self.config.misfire_penalty * intensity
                )
            reward -= (
                self.config.actuation_intensity_penalty * intensity
            )
            self._schedule_air_jet(local_jet, intensity)

        return reward

    def _move_one_physics_step(self) -> None:
        if not self.fired:
            self.horizontal_y += (
                self.config.horizontal_belt_speed
                * self.config.time_step
            )
            position = [
                self.config.system_x + self.lateral_offset,
                self.horizontal_y,
                self.config.system_z + self.tube_radius,
            ]
            p.resetBasePositionAndOrientation(
                self.tube_id,
                position,
                self.transport_orientation,
            )
            p.resetBaseVelocity(
                self.tube_id,
                linearVelocity=[
                    0.0,
                    self.config.horizontal_belt_speed,
                    0.0,
                ],
                angularVelocity=[0.0, 0.0, 0.0],
            )

        self._update_air_jet_force()
        p.stepSimulation()
        self.physics_steps += 1

    def _terminal_outcome(self) -> str | None:
        position = self._tube_position()
        if self.fired:
            collection_bin_index = None
            for bin_index, aabb in enumerate(self.bin_aabbs):
                capture_ceiling = min(
                    aabb[1][2] - 0.05,
                    (
                        self.config.system_z
                        - self.config.bin_drop_below_belt
                    ),
                )
                inside_bin = (
                    aabb[0][0] + self.config.bin_capture_depth
                    < position[0]
                    < aabb[1][0] - 0.10
                    and aabb[0][1] + 0.08
                    < position[1]
                    < aabb[1][1] - 0.08
                    and aabb[0][2] + 0.05
                    < position[2]
                    < capture_ceiling
                )
                if inside_bin:
                    collection_bin_index = bin_index
                    break

            if collection_bin_index is None:
                self.bin_candidate_index = None
                self.bin_candidate_steps = 0
            elif self.bin_candidate_index == collection_bin_index:
                self.bin_candidate_steps += 1
            else:
                self.bin_candidate_index = collection_bin_index
                self.bin_candidate_steps = 1

            required_steps = max(
                1,
                int(
                    round(
                        self.config.bin_confirmation_seconds
                        / self.config.time_step
                    )
                ),
            )
            if self.bin_candidate_steps >= required_steps:
                return (
                    "correct_bin"
                    if collection_bin_index == self.true_type
                    else "wrong_bin"
                )

            if (
                self.fire_physics_step is not None
                and (
                    self.physics_steps - self.fire_physics_step
                )
                * self.config.time_step
                > self.config.result_timeout_seconds
            ):
                return "missed_bin"
        elif position[1] > (
            self.jet_positions[int(self.detected_type)]
            + self.config.jet_trigger_half_width
        ):
            return "missed_nozzle"

        return None

    def _terminal_reward(self, outcome: str) -> float:
        if outcome == "correct_bin":
            return self.config.correct_sort_reward
        return -self.config.failure_penalty

    def _info(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome or "",
            "true_type": int(self.true_type),
            "true_class": TUBE_CLASSES[int(self.true_type)],
            "tube_material": TUBE_SPECS[int(self.true_type)].body_material,
            "tube_mass": self.tube_mass,
            "nominal_residual_mass": (
                TUBE_SPECS[int(self.true_type)].residual_mass_kg
            ),
            "detected_type": int(self.detected_type),
            "detected_class": TUBE_CLASSES[int(self.detected_type)],
            "tube_axis_direction": self.tube_axis_direction,
            "lateral_offset": self.lateral_offset,
            "fired_jet": (
                self.fired_jet_index + 1
                if self.fired_jet_index is not None
                else None
            ),
            "fired_intensity": self.fired_intensity,
            "jet_command_x": self.jet_command_x,
            "jet_command_y_offset": self.jet_command_y_offset,
            "jet_distance": self.jet_distance,
            "jet_radial_offset": self.jet_radial_offset,
            "jet_attenuation": self.jet_attenuation,
            "jet_peak_force": self.jet_peak_force,
            "jet_accumulated_impulse": (
                self.jet_accumulated_impulse
            ),
            "misfire_count": self.misfire_count,
            "cumulative_valve_opening": (
                self.cumulative_valve_opening
            ),
            "decision_steps": self.decision_steps,
            "physics_steps": self.physics_steps,
            "position": self._tube_position(),
            "estimated_position": self._estimated_position(),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._park_active_tube()

        self.tube_id = None
        self.fired = False
        self.jet_commanded = False
        self.fired_jet_index = None
        self.fired_intensity = 0.0
        self.jet_command_physics_step = None
        self.jet_command_x = None
        self.jet_command_y_offset = None
        self.fire_physics_step = None
        self.jet_distance = None
        self.jet_radial_offset = None
        self.jet_attenuation = None
        self.jet_peak_force = 0.0
        self.jet_accumulated_impulse = 0.0
        self.bin_candidate_index = None
        self.bin_candidate_steps = 0
        self.physics_steps = 0
        self.decision_steps = 0
        self.cumulative_valve_opening = 0.0
        self.misfire_count = 0
        self.outcome = None
        self._episode_done = False
        self._spawn_tube(options)
        return self._get_obs(), self._info()

    def step(
        self,
        action: Any,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        if self._episode_done:
            raise RuntimeError(
                "step() called after episode completion; call reset()."
            )

        valve_openings, _ = self._decode_action(action)
        reward = self._actuation_reward(valve_openings)
        self.decision_steps += 1

        terminated = False
        truncated = False
        physics_step_budget = self.config.control_frame_skip
        if self.jet_commanded:
            physics_step_budget = int(
                np.ceil(
                    (
                        self.config.result_timeout_seconds
                        + self.config.time_step
                    )
                    / self.config.time_step
                )
            )
        for _ in range(physics_step_budget):
            self._move_one_physics_step()
            outcome = self._terminal_outcome()
            if outcome is not None:
                self.outcome = outcome
                reward += self._terminal_reward(outcome)
                terminated = True
                self._episode_done = True
                break

        if (
            not terminated
            and self.decision_steps
            >= self.config.max_episode_decisions
        ):
            self.outcome = "truncated"
            reward -= self.config.failure_penalty
            truncated = True
            self._episode_done = True

        if self.render_mode == "human":
            time.sleep(
                self.config.time_step
                * self.config.control_frame_skip
            )

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            self._info(),
        )

    def render(self) -> None:
        # PyBullet handles GUI rendering.
        return None

    def close(self) -> None:
        if self.physics_client >= 0 and p.isConnected(
            self.physics_client
        ):
            p.disconnect(self.physics_client)
        self.physics_client = -1


def evaluate_model(
    model: Any,
    *,
    action_mode: str,
    episodes: int,
    seed: int,
    output_csv: Path,
    recognition_error_probability: float = 0.0,
    config: SortingConfig | None = None,
) -> dict[str, Any]:
    """Evaluate one trained model and write one terminal row per episode."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    env = ConveyorSortingEnv(
        action_mode=action_mode,
        render_mode=None,
        recognition_error_probability=(
            recognition_error_probability
        ),
        config=config,
    )
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            episode_seed = seed * 100_000 + episode
            observation, reset_info = env.reset(seed=episode_seed)
            terminated = False
            truncated = False
            episode_return = 0.0
            final_info = reset_info

            while not (terminated or truncated):
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    final_info,
                ) = env.step(action)
                episode_return += float(reward)

            rows.append(
                {
                    "episode": episode + 1,
                    "seed": episode_seed,
                    "true_type": final_info["true_type"],
                    "true_class": final_info["true_class"],
                    "detected_type": final_info["detected_type"],
                    "detected_class": final_info["detected_class"],
                    "outcome": final_info["outcome"],
                    "success": int(
                        final_info["outcome"] == "correct_bin"
                    ),
                    "fired_jet": final_info["fired_jet"],
                    "fired_intensity": (
                        final_info["fired_intensity"]
                    ),
                    "lateral_offset": (
                        final_info["lateral_offset"]
                    ),
                    "tube_axis_direction": (
                        final_info["tube_axis_direction"]
                    ),
                    "jet_command_x": (
                        final_info["jet_command_x"]
                    ),
                    "jet_command_y_offset": (
                        final_info["jet_command_y_offset"]
                    ),
                    "jet_distance": final_info["jet_distance"],
                    "jet_radial_offset": (
                        final_info["jet_radial_offset"]
                    ),
                    "jet_attenuation": (
                        final_info["jet_attenuation"]
                    ),
                    "jet_peak_force": (
                        final_info["jet_peak_force"]
                    ),
                    "jet_accumulated_impulse": (
                        final_info["jet_accumulated_impulse"]
                    ),
                    "misfire_count": (
                        final_info["misfire_count"]
                    ),
                    "cumulative_valve_opening": (
                        final_info["cumulative_valve_opening"]
                    ),
                    "decision_steps": (
                        final_info["decision_steps"]
                    ),
                    "episode_return": episode_return,
                }
            )
    finally:
        env.close()

    fields = list(rows[0]) if rows else []
    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    outcome_counts: dict[str, int] = {}
    class_results: dict[str, dict[str, int | float]] = {}
    for row in rows:
        outcome = str(row["outcome"])
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        class_name = str(row["true_class"])
        if class_name not in class_results:
            class_results[class_name] = {
                "episodes": 0,
                "correct": 0,
                "success_rate": 0.0,
            }
        class_results[class_name]["episodes"] = (
            int(class_results[class_name]["episodes"]) + 1
        )
        class_results[class_name]["correct"] = (
            int(class_results[class_name]["correct"])
            + int(row["success"])
        )
    for values in class_results.values():
        values["success_rate"] = (
            int(values["correct"]) / int(values["episodes"])
        )
    success_rate = (
        sum(int(row["success"]) for row in rows) / len(rows)
        if rows
        else 0.0
    )
    summary = {
        "episodes": len(rows),
        "success_rate": success_rate,
        "outcome_counts": outcome_counts,
        "class_results": class_results,
        "mean_return": (
            float(np.mean([row["episode_return"] for row in rows]))
            if rows
            else 0.0
        ),
        "mean_misfires": (
            float(np.mean([row["misfire_count"] for row in rows]))
            if rows
            else 0.0
        ),
        "mean_fired_intensity": (
            float(np.mean([row["fired_intensity"] for row in rows]))
            if rows
            else 0.0
        ),
        "mean_jet_accumulated_impulse": (
            float(
                np.mean(
                    [row["jet_accumulated_impulse"] for row in rows]
                )
            )
            if rows
            else 0.0
        ),
        "mean_jet_peak_force": (
            float(np.mean([row["jet_peak_force"] for row in rows]))
            if rows
            else 0.0
        ),
        "mean_cumulative_valve_command": (
            float(
                np.mean(
                    [
                        row["cumulative_valve_opening"]
                        for row in rows
                    ]
                )
            )
            if rows
            else 0.0
        ),
        "mean_valve_command_per_decision": (
            float(
                np.mean(
                    [
                        row["cumulative_valve_opening"]
                        / max(int(row["decision_steps"]), 1)
                        for row in rows
                    ]
                )
            )
            if rows
            else 0.0
        ),
        "results_csv": str(output_csv.resolve()),
    }
    summary_path = output_csv.with_name(
        f"{output_csv.stem}_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def save_environment_config(
    path: Path,
    config: SortingConfig | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(config or SortingConfig()), indent=2),
        encoding="utf-8",
    )
    return path


def plot_policy_heatmap(
    model: Any,
    *,
    action_mode: str,
    output_png: Path,
) -> Path:
    """Plot the learned state-action map in the report's y/class format."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    y_values = np.linspace(-1.0, 1.0, 241, dtype=np.float32)
    class_values = np.linspace(-1.0, 1.0, 5, dtype=np.float32)

    if action_mode == "discrete":
        action_grid = np.zeros(
            (len(class_values), len(y_values)),
            dtype=np.int32,
        )
        for class_index, class_value in enumerate(class_values):
            for y_index, y_value in enumerate(y_values):
                observation = np.array(
                    [0.0, y_value, class_value],
                    dtype=np.float32,
                )
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )
                action_index = int(np.asarray(action).item())
                action_grid[class_index, y_index] = action_index

        colors = [
            "#111111",
            "#d62728",
            "#1f77b4",
            "#2ca02c",
            "#9467bd",
            "#ff7f0e",
        ]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), cmap.N)
        figure, axis = plt.subplots(figsize=(10, 4.8))
        image = axis.imshow(
            action_grid,
            extent=[-1.0, 1.0, -1.25, 1.25],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        colorbar = figure.colorbar(
            image,
            ax=axis,
            ticks=np.arange(6),
        )
        colorbar.ax.set_yticklabels(
            ["Off", "65%", "70%", "80%", "90%", "100%"]
        )
        axis.set_title("Discrete Valve Policy Heatmap")
        axis.set_xlabel("Normalized distance to target jet")
        axis.set_ylabel("Normalized detected class")
        figure.tight_layout()
    else:
        intensity_grid = np.zeros(
            (len(class_values), len(y_values)),
            dtype=np.float32,
        )
        for class_index, class_value in enumerate(class_values):
            for y_index, y_value in enumerate(y_values):
                observation = np.array(
                    [0.0, y_value, class_value],
                    dtype=np.float32,
                )
                raw_action, _ = model.predict(
                    observation,
                    deterministic=True,
                )
                raw_action = np.asarray(
                    raw_action,
                    dtype=np.float32,
                ).reshape(1)
                opening = np.clip(
                    (raw_action[0] + 1.0) * 0.5,
                    0.0,
                    1.0,
                )
                intensity_grid[class_index, y_index] = float(opening)

        figure, axis = plt.subplots(figsize=(10, 4.8))
        intensity = axis.imshow(
            intensity_grid,
            extent=[-1.0, 1.0, -1.25, 1.25],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            interpolation="bilinear",
            vmin=0.0,
            vmax=1.0,
        )
        figure.colorbar(
            intensity,
            ax=axis,
            label="Target valve opening",
        )
        axis.set_title("Continuous Valve Policy Heatmap")
        axis.set_xlabel("Normalized distance to target jet")
        axis.set_ylabel("Normalized detected class")
        figure.tight_layout()

    figure.savefig(output_png, dpi=180)
    plt.close(figure)
    return output_png


def plot_policy_position_maps(
    model: Any,
    *,
    action_mode: str,
    output_png: Path,
) -> Path:
    """Plot lateral position and nozzle distance for every tube class."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    x_values = np.linspace(-1.0, 1.0, 101, dtype=np.float32)
    y_values = np.linspace(-1.0, 1.0, 121, dtype=np.float32)
    class_values = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    flat_axes = list(axes.flat)
    images = []

    if action_mode == "discrete":
        cmap = ListedColormap(
            [
                "#111111",
                "#d62728",
                "#1f77b4",
                "#2ca02c",
                "#9467bd",
                "#ff7f0e",
            ]
        )
        norm = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), cmap.N)
    else:
        cmap = "viridis"
        norm = None

    for class_index, class_value in enumerate(class_values):
        value_grid = np.zeros(
            (len(y_values), len(x_values)),
            dtype=np.float32,
        )
        for y_index, y_value in enumerate(y_values):
            for x_index, x_value in enumerate(x_values):
                observation = np.array(
                    [x_value, y_value, class_value],
                    dtype=np.float32,
                )
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )
                if action_mode == "discrete":
                    value = int(np.asarray(action).item())
                else:
                    raw_action = float(
                        np.asarray(action, dtype=np.float32).reshape(1)[0]
                    )
                    value = float(
                        np.clip((raw_action + 1.0) * 0.5, 0.0, 1.0)
                    )
                value_grid[y_index, x_index] = value

        axis = flat_axes[class_index]
        image = axis.imshow(
            value_grid,
            extent=[-1.0, 1.0, -1.0, 1.0],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation=(
                "nearest" if action_mode == "discrete" else "bilinear"
            ),
            vmin=None if action_mode == "discrete" else 0.0,
            vmax=None if action_mode == "discrete" else 1.0,
        )
        images.append(image)
        axis.set_title(TUBE_CLASSES[class_index])
        axis.set_xlabel("Normalized lateral position")
        axis.set_ylabel("Normalized distance to target nozzle")

    flat_axes[-1].set_visible(False)
    if action_mode == "discrete":
        colorbar = figure.colorbar(
            images[0],
            ax=flat_axes[:5],
            ticks=np.arange(6),
            shrink=0.9,
        )
        colorbar.ax.set_yticklabels(
            ["Off", "65%", "70%", "80%", "90%", "100%"]
        )
        figure.suptitle("Discrete Position and Timing Policy")
    else:
        figure.colorbar(
            images[0],
            ax=flat_axes[:5],
            label="Target valve opening",
            shrink=0.9,
        )
        figure.suptitle("Continuous Position and Timing Policy")

    figure.savefig(output_png, dpi=180)
    plt.close(figure)
    return output_png


def plot_training_curve(
    monitor_csv: Path,
    output_png: Path,
    *,
    rolling_window: int = 20,
) -> Path | None:
    """Plot episode returns from a Stable-Baselines3 Monitor CSV."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    monitor_csv = Path(monitor_csv)
    if not monitor_csv.exists():
        return None

    returns: list[float] = []
    with monitor_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            if line.strip().startswith("r,"):
                continue
            parts = line.strip().split(",")
            if not parts or not parts[0]:
                continue
            try:
                returns.append(float(parts[0]))
            except ValueError:
                continue

    if not returns:
        return None

    values = np.asarray(returns, dtype=float)
    window = max(1, min(rolling_window, len(values)))
    kernel = np.ones(window, dtype=float) / window
    rolling = np.convolve(values, kernel, mode="valid")
    rolling_x = np.arange(window, len(values) + 1)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(
        np.arange(1, len(values) + 1),
        values,
        color="#9ecae1",
        linewidth=1.0,
        label="Episode return",
    )
    axis.plot(
        rolling_x,
        rolling,
        color="#08519c",
        linewidth=2.0,
        label=f"{window}-episode mean",
    )
    axis.set_xlabel("Episode")
    axis.set_ylabel("Return")
    axis.set_title("Training Reward")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_png, dpi=180)
    plt.close(figure)
    return output_png
