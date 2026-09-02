"""Load trained policies for the main simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DISCRETE_VALVE_LEVELS = (0.65, 0.70, 0.80, 0.90, 1.00)

POLICY_FILENAMES = {
    "dqn": "dqn_sorting_policy.zip",
    "ppo": "ppo_sorting_policy.zip",
    "ppo_discrete": "ppo_discrete_sorting_policy.zip",
    "ppo_continuous": "ppo_continuous_sorting_policy.zip",
    "sac": "sac_sorting_policy.zip",
}

POLICY_FALLBACK_FILENAMES = {
    "ppo_discrete": ("ppo_sorting_policy.zip",),
}

POLICY_ACTION_MODES = {
    "dqn": "discrete",
    "ppo": "discrete",
    "ppo_discrete": "discrete",
    "ppo_continuous": "continuous",
    "sac": "continuous",
}

POLICY_SEARCH_DIRECTORIES = {
    "dqn": (Path("dqn"),),
    "ppo": (Path("ppo"),),
    "ppo_discrete": (Path("ppo") / "discrete", Path("ppo")),
    "ppo_continuous": (Path("ppo") / "continuous", Path("ppo")),
    "sac": (Path("sac"),),
}


def resolve_policy_path(
    base_dir: Path,
    algorithm: str,
    override: str | None = None,
) -> Path | None:
    algorithm = algorithm.lower()
    if algorithm not in POLICY_FILENAMES:
        raise ValueError(f"Unsupported RL algorithm: {algorithm}")

    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = Path(base_dir) / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"RL policy not found: {path}")
        return path

    filenames = (
        POLICY_FILENAMES[algorithm],
        *POLICY_FALLBACK_FILENAMES.get(algorithm, ()),
    )
    best_root = Path(base_dir) / "runs" / "rl" / "best"
    for filename in filenames:
        preferred_path = best_root / filename
        if preferred_path.exists():
            return preferred_path.resolve()

    candidates: list[Path] = []
    rl_root = Path(base_dir) / "runs" / "rl"
    for relative_dir in POLICY_SEARCH_DIRECTORIES[algorithm]:
        search_root = rl_root / relative_dir
        if not search_root.exists():
            continue
        for filename in filenames:
            candidates.extend(search_root.rglob(filename))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def load_policy(algorithm: str, policy_path: Path) -> Any:
    algorithm = algorithm.lower()
    if algorithm == "dqn":
        from stable_baselines3 import DQN

        model_class = DQN
    elif algorithm in {"ppo", "ppo_discrete", "ppo_continuous"}:
        from stable_baselines3 import PPO

        model_class = PPO
    elif algorithm == "sac":
        from stable_baselines3 import SAC

        model_class = SAC
    else:
        raise ValueError(f"Unsupported RL algorithm: {algorithm}")
    model = model_class.load(str(policy_path), device="auto")
    validate_policy(model, algorithm)
    return model


def validate_policy(model: Any, algorithm: str) -> None:
    algorithm = algorithm.lower()
    if algorithm not in POLICY_ACTION_MODES:
        raise ValueError(f"Unsupported RL algorithm: {algorithm}")
    if getattr(model.observation_space, "shape", None) != (3,):
        raise ValueError("The RL policy must use a three-value observation.")
    expected_mode = POLICY_ACTION_MODES[algorithm]
    actual_mode = policy_action_mode(model)
    if actual_mode != expected_mode:
        raise ValueError(
            f"{algorithm} requires a {expected_mode} policy, but the "
            f"loaded model uses {actual_mode} actions."
        )


def policy_action_mode(model: Any) -> str:
    action_space = getattr(model, "action_space", None)
    action_count = getattr(action_space, "n", None)
    if action_count is not None:
        expected_actions = 1 + len(DISCRETE_VALVE_LEVELS)
        if action_count != expected_actions:
            raise ValueError(
                f"The discrete RL policy must have {expected_actions} actions."
            )
        return "discrete"
    if getattr(action_space, "shape", None) == (1,):
        return "continuous"
    raise ValueError(
        "The RL policy must use Discrete(6) or one continuous action."
    )


def build_observation(
    x_position: float,
    y_position: float,
    detected_type: int,
    target_jet_y: float,
) -> np.ndarray:
    x_normalized = float(
        np.clip((x_position + 2.014984) / 0.35, -1.0, 1.0)
    )
    y_normalized = float(
        np.clip(y_position - target_jet_y, -1.0, 1.0)
    )
    class_normalized = float((int(detected_type) - 2.0) / 2.0)
    return np.array(
        [x_normalized, y_normalized, class_normalized],
        dtype=np.float32,
    )


def predict_valve_openings(
    model: Any,
    algorithm: str,
    observation: np.ndarray,
    detected_type: int,
) -> tuple[np.ndarray, int | None]:
    action, _ = model.predict(observation, deterministic=True)
    action_mode = policy_action_mode(model)
    expected_mode = POLICY_ACTION_MODES.get(algorithm.lower())
    if expected_mode is None:
        raise ValueError(f"Unsupported RL algorithm: {algorithm}")
    if action_mode != expected_mode:
        raise ValueError(
            f"{algorithm} expects {expected_mode} actions, but the loaded "
            f"model uses {action_mode} actions."
        )
    if action_mode == "discrete":
        action_index = int(np.asarray(action).item())
        maximum_action = len(DISCRETE_VALVE_LEVELS)
        if action_index < 0 or action_index > maximum_action:
            raise ValueError(f"Invalid discrete policy action: {action_index}")
        openings = np.zeros(5, dtype=np.float32)
        if action_index > 0:
            level_index = action_index - 1
            openings[int(detected_type)] = DISCRETE_VALVE_LEVELS[level_index]
        return openings, action_index

    raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
    if raw_action.shape != (1,):
        raise ValueError(
            f"Continuous action must contain one value: {raw_action.shape}"
        )
    openings = np.zeros(5, dtype=np.float32)
    openings[int(detected_type)] = float(
        np.clip((raw_action[0] + 1.0) * 0.5, 0.0, 1.0)
    )
    return openings, None
