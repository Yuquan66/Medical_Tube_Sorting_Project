"""Compare original and improved SAC policies on common test seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from rl_policy_runtime import load_policy
from rl_sorting_env import TUBE_CLASSES, evaluate_model


BASE_DIR = Path(__file__).resolve().parent
OUTCOMES = (
    "correct_bin",
    "wrong_bin",
    "missed_nozzle",
    "missed_bin",
    "truncated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two continuous SAC policies."
    )
    parser.add_argument(
        "--original-policy",
        default=str(
            BASE_DIR
            / "runs"
            / "rl"
            / "sac"
            / "sac_continuous_formal_v2_seed42"
            / "sac_sorting_policy.zip"
        ),
    )
    parser.add_argument("--improved-policy", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        default=str(
            BASE_DIR / "runs" / "rl" / "comparisons" / "sac_improved_v1"
        ),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    policies = {
        "Original SAC": Path(args.original_policy).resolve(),
        "Improved SAC": Path(args.improved_policy).resolve(),
    }
    models = {
        label: load_policy("sac", path)
        for label, path in policies.items()
    }
    seed_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    for label, model in models.items():
        for seed in args.seeds:
            key = label.lower().replace(" ", "_")
            result = evaluate_model(
                model,
                action_mode="continuous",
                episodes=args.episodes_per_seed,
                seed=seed,
                output_csv=output_dir / f"{key}_seed{seed}.csv",
            )
            row = {
                "controller": label,
                "seed": seed,
                "episodes": result["episodes"],
                "success_rate": result["success_rate"],
                "mean_return": result["mean_return"],
                "mean_fired_intensity": result["mean_fired_intensity"],
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
            }
            for outcome in OUTCOMES:
                row[outcome] = result["outcome_counts"].get(outcome, 0)
            seed_rows.append(row)

            for class_name in TUBE_CLASSES.values():
                class_result = result["class_results"].get(
                    class_name,
                    {"episodes": 0, "correct": 0, "success_rate": 0.0},
                )
                class_rows.append(
                    {
                        "controller": label,
                        "seed": seed,
                        "tube_class": class_name,
                        "episodes": class_result["episodes"],
                        "correct": class_result["correct"],
                        "success_rate": class_result["success_rate"],
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    for label in policies:
        rows = [row for row in seed_rows if row["controller"] == label]
        success_rates = [float(row["success_rate"]) for row in rows]
        summary = {
            "controller": label,
            "seed_count": len(rows),
            "episodes_per_seed": args.episodes_per_seed,
            "total_episodes": sum(int(row["episodes"]) for row in rows),
            "mean_success_rate": mean(success_rates),
            "success_rate_sd": stdev(success_rates),
            "mean_return": mean(float(row["mean_return"]) for row in rows),
            "mean_fired_intensity": mean(
                float(row["mean_fired_intensity"]) for row in rows
            ),
            "mean_jet_accumulated_impulse": mean(
                float(row["mean_jet_accumulated_impulse"])
                for row in rows
            ),
            "mean_jet_peak_force": mean(
                float(row["mean_jet_peak_force"]) for row in rows
            ),
            "mean_cumulative_valve_command": mean(
                float(row["mean_cumulative_valve_command"])
                for row in rows
            ),
            "mean_valve_command_per_decision": mean(
                float(row["mean_valve_command_per_decision"])
                for row in rows
            ),
            "policy_path": str(policies[label]),
        }
        for outcome in OUTCOMES:
            summary[outcome] = sum(int(row[outcome]) for row in rows)
        summary_rows.append(summary)

    class_summary_rows: list[dict[str, Any]] = []
    for label in policies:
        for class_name in TUBE_CLASSES.values():
            rows = [
                row
                for row in class_rows
                if row["controller"] == label
                and row["tube_class"] == class_name
            ]
            episodes = sum(int(row["episodes"]) for row in rows)
            correct = sum(int(row["correct"]) for row in rows)
            class_summary_rows.append(
                {
                    "controller": label,
                    "tube_class": class_name,
                    "episodes": episodes,
                    "correct": correct,
                    "success_rate": correct / episodes if episodes else 0.0,
                }
            )

    write_csv(output_dir / "comparison_by_seed.csv", seed_rows)
    write_csv(output_dir / "comparison_summary.csv", summary_rows)
    write_csv(output_dir / "comparison_by_class_and_seed.csv", class_rows)
    write_csv(output_dir / "comparison_by_class_summary.csv", class_summary_rows)
    (output_dir / "comparison_config.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "episodes_per_seed": args.episodes_per_seed,
                "policies": {
                    label: str(path) for label, path in policies.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, indent=2))
    return output_dir


if __name__ == "__main__":
    run(parse_args())
