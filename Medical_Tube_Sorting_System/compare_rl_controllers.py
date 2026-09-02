"""Evaluate discrete and continuous RL controller pairs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from rl_policy_runtime import load_policy, resolve_policy_path
from rl_sorting_env import (
    DISCRETE_VALVE_LEVELS,
    TUBE_CLASSES,
    evaluate_model,
)


BASE_DIR = Path(__file__).resolve().parent

CONTROLLERS = (
    {
        "key": "fixed_rule",
        "label": "Fixed rule",
        "comparison_group": "reference",
        "action_mode": "discrete",
    },
    {
        "key": "dqn",
        "label": "DQN",
        "comparison_group": "discrete",
        "action_mode": "discrete",
    },
    {
        "key": "ppo_discrete",
        "label": "PPO-Discrete",
        "comparison_group": "discrete",
        "action_mode": "discrete",
    },
    {
        "key": "ppo_continuous",
        "label": "PPO-Continuous",
        "comparison_group": "continuous",
        "action_mode": "continuous",
    },
    {
        "key": "sac",
        "label": "SAC",
        "comparison_group": "continuous",
        "action_mode": "continuous",
    },
)

OUTCOME_KEYS = (
    "correct_bin",
    "wrong_bin",
    "missed_nozzle",
    "missed_bin",
    "truncated",
)


class FixedRulePolicy:
    """Fire the YOLO-routed nozzle at full opening near its centre."""

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[int, None]:
        del deterministic
        distance = abs(float(observation[1]))
        if distance > 0.08:
            return 0, None
        return len(DISCRETE_VALVE_LEVELS), None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DQN with discrete PPO and continuous PPO with SAC "
            "under common evaluation seeds."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--recognition-error", type=float, default=0.0)
    parser.add_argument("--output-root")
    parser.add_argument("--dqn-policy")
    parser.add_argument("--ppo-discrete-policy")
    parser.add_argument("--ppo-continuous-policy")
    parser.add_argument("--sac-policy")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.seeds:
        raise ValueError("At least one evaluation seed is required.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Evaluation seeds must be unique.")
    if args.episodes_per_seed <= 0:
        raise ValueError("episodes-per-seed must be positive.")
    if not 0.0 <= args.recognition_error < 1.0:
        raise ValueError("recognition-error must be in [0, 1).")


def policy_overrides(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "dqn": args.dqn_policy,
        "ppo_discrete": args.ppo_discrete_policy,
        "ppo_continuous": args.ppo_continuous_policy,
        "sac": args.sac_policy,
    }


def output_directory(args: argparse.Namespace) -> Path:
    if args.output_root:
        root = Path(args.output_root).expanduser()
        if not root.is_absolute():
            root = BASE_DIR / root
        path = root.resolve()
    else:
        run_name = datetime.now().strftime("comparison_%Y%m%d_%H%M%S")
        path = BASE_DIR / "runs" / "rl" / "comparisons" / run_name
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_required_policies(
    overrides: dict[str, str | None],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for controller in CONTROLLERS:
        key = str(controller["key"])
        if key == "fixed_rule":
            continue
        path = resolve_policy_path(BASE_DIR, key, overrides[key])
        if path is None:
            missing.append(key)
        else:
            paths[key] = path
    if missing:
        raise FileNotFoundError(
            "Missing trained policies: "
            + ", ".join(missing)
            + ". Train all four controller configurations first."
        )
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(args: argparse.Namespace) -> Path:
    validate_args(args)
    overrides = policy_overrides(args)
    paths = resolve_required_policies(overrides)
    run_dir = output_directory(args)

    run_config = {
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes_per_seed,
        "recognition_error_probability": args.recognition_error,
        "policies": {
            "fixed_rule": "built-in full-opening timing rule",
            **{key: str(path) for key, path in paths.items()},
        },
    }
    (run_dir / "comparison_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    class_seed_rows: list[dict[str, Any]] = []
    for controller in CONTROLLERS:
        key = str(controller["key"])
        label = str(controller["label"])
        group = str(controller["comparison_group"])
        action_mode = str(controller["action_mode"])
        model = (
            FixedRulePolicy()
            if key == "fixed_rule"
            else load_policy(key, paths[key])
        )
        controller_rows: list[dict[str, Any]] = []

        for seed in args.seeds:
            result_path = run_dir / f"{key}_seed{seed}.csv"
            result = evaluate_model(
                model,
                action_mode=action_mode,
                episodes=args.episodes_per_seed,
                seed=seed,
                output_csv=result_path,
                recognition_error_probability=args.recognition_error,
            )
            row = {
                "comparison_group": group,
                "controller": label,
                "policy_key": key,
                "action_mode": action_mode,
                "seed": seed,
                "episodes": result["episodes"],
                "success_rate": result["success_rate"],
                "mean_return": result["mean_return"],
                "mean_misfires": result["mean_misfires"],
                "mean_fired_intensity": result[
                    "mean_fired_intensity"
                ],
                "mean_jet_accumulated_impulse": result[
                    "mean_jet_accumulated_impulse"
                ],
                "mean_jet_peak_force": result["mean_jet_peak_force"],
                "mean_cumulative_valve_command": result[
                    "mean_cumulative_valve_command"
                ],
                "mean_valve_command_per_decision": result[
                    "mean_valve_command_per_decision"
                ],
                "results_csv": result["results_csv"],
            }
            for outcome in OUTCOME_KEYS:
                row[outcome] = result["outcome_counts"].get(outcome, 0)
            seed_rows.append(row)
            controller_rows.append(row)
            for class_name in TUBE_CLASSES.values():
                class_result = result["class_results"].get(
                    class_name,
                    {"episodes": 0, "correct": 0, "success_rate": 0.0},
                )
                class_seed_rows.append(
                    {
                        "comparison_group": group,
                        "controller": label,
                        "policy_key": key,
                        "seed": seed,
                        "tube_class": class_name,
                        "episodes": class_result["episodes"],
                        "correct": class_result["correct"],
                        "success_rate": class_result["success_rate"],
                    }
                )

        success_rates = [float(row["success_rate"]) for row in controller_rows]
        mean_returns = [float(row["mean_return"]) for row in controller_rows]
        mean_misfires = [float(row["mean_misfires"]) for row in controller_rows]
        mean_fired_intensities = [
            float(row["mean_fired_intensity"])
            for row in controller_rows
        ]
        mean_jet_impulses = [
            float(row["mean_jet_accumulated_impulse"])
            for row in controller_rows
        ]
        mean_peak_forces = [
            float(row["mean_jet_peak_force"])
            for row in controller_rows
        ]
        mean_cumulative_commands = [
            float(row["mean_cumulative_valve_command"])
            for row in controller_rows
        ]
        mean_commands_per_decision = [
            float(row["mean_valve_command_per_decision"])
            for row in controller_rows
        ]
        total_outcomes = {
            outcome: sum(int(row[outcome]) for row in controller_rows)
            for outcome in OUTCOME_KEYS
        }
        summary_rows.append(
            {
                "comparison_group": group,
                "controller": label,
                "policy_key": key,
                "action_mode": action_mode,
                "seed_count": len(controller_rows),
                "episodes_per_seed": args.episodes_per_seed,
                "total_episodes": len(controller_rows)
                * args.episodes_per_seed,
                "mean_success_rate": mean(success_rates),
                "success_rate_sd": (
                    stdev(success_rates) if len(success_rates) > 1 else 0.0
                ),
                "mean_return": mean(mean_returns),
                "mean_misfires": mean(mean_misfires),
                "mean_fired_intensity": mean(mean_fired_intensities),
                "mean_jet_accumulated_impulse": mean(mean_jet_impulses),
                "mean_jet_peak_force": mean(mean_peak_forces),
                "mean_cumulative_valve_command": mean(
                    mean_cumulative_commands
                ),
                "mean_valve_command_per_decision": mean(
                    mean_commands_per_decision
                ),
                **total_outcomes,
                "policy_path": (
                    "built-in full-opening timing rule"
                    if key == "fixed_rule"
                    else str(paths[key])
                ),
            }
        )

    write_csv(run_dir / "comparison_by_seed.csv", seed_rows)
    write_csv(run_dir / "comparison_summary.csv", summary_rows)
    write_csv(
        run_dir / "comparison_by_class_and_seed.csv",
        class_seed_rows,
    )
    class_summary_rows: list[dict[str, Any]] = []
    for controller in CONTROLLERS:
        key = str(controller["key"])
        label = str(controller["label"])
        group = str(controller["comparison_group"])
        for class_name in TUBE_CLASSES.values():
            matching = [
                row
                for row in class_seed_rows
                if row["policy_key"] == key
                and row["tube_class"] == class_name
            ]
            episodes = sum(int(row["episodes"]) for row in matching)
            correct = sum(int(row["correct"]) for row in matching)
            class_summary_rows.append(
                {
                    "comparison_group": group,
                    "controller": label,
                    "policy_key": key,
                    "tube_class": class_name,
                    "episodes": episodes,
                    "correct": correct,
                    "success_rate": (
                        correct / episodes if episodes else 0.0
                    ),
                }
            )
    write_csv(
        run_dir / "comparison_by_class_summary.csv",
        class_summary_rows,
    )
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    print(f"Comparison results: {run_dir}")
    return run_dir


if __name__ == "__main__":
    run_comparison(parse_args())
