"""Run a visible 100-tube validation."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from rl_policy_runtime import resolve_policy_path
from tube_specs import TUBE_SPECS


BASE_DIR = Path(__file__).resolve().parent
TUBE_CLASSES = tuple(spec.name for spec in TUBE_SPECS)
TEST_TUBE_COUNT = int(os.environ.get("VALIDATION_TUBE_COUNT", "100"))
if TEST_TUBE_COUNT <= 0:
    raise ValueError("VALIDATION_TUBE_COUNT must be positive.")
RL_CONTROL_MODES = {
    "dqn",
    "ppo_discrete",
    "ppo_continuous",
    "sac",
}


def choose_control_mode():
    """Select the validation control mode."""
    requested_mode = os.environ.get(
        "VALIDATION_CONTROL_MODE",
        "",
    ).strip().lower()
    valid_modes = {"ground_truth", "yolo", *RL_CONTROL_MODES}
    if requested_mode in valid_modes:
        return requested_mode
    print("\n100-tube validation modes:")
    print("1. Ground truth - physical sorting qualification")
    print("2. Strict YOLO - rule control")
    print("3. YOLO + DQN")
    print("4. YOLO + PPO discrete")
    print("5. YOLO + PPO continuous")
    print("6. YOLO + SAC")
    choice = input("Select a mode [1]: ").strip() or "1"
    return {
        "1": "ground_truth",
        "2": "yolo",
        "3": "dqn",
        "4": "ppo_discrete",
        "5": "ppo_continuous",
        "6": "sac",
    }.get(choice)


def classify_trial(row, control_mode):
    """Classify one validation result."""
    if row.get("outcome") == "unfinished":
        return "unfinished"

    expected_class = int(row["evaluation_class_id"])
    detected_value = row.get("detected_class_id", "")
    if control_mode != "ground_truth":
        if detected_value == "" or int(detected_value) != expected_class:
            return "recognition_failure"

    selected_value = row.get("selected_jet", "")
    if selected_value == "" or int(selected_value) != expected_class + 1:
        return "nozzle_selection_failure"

    if row.get("outcome") == "correct_bin":
        return "correct"
    return "physical_failure"


def build_summary_rows(trials, control_mode):
    """Build class and overall summaries."""
    grouped = defaultdict(list)
    for row in trials:
        grouped[row["evaluation_class"]].append(row)

    groups = [("overall", trials)]
    groups.extend(
        (class_name, grouped[class_name])
        for class_name in TUBE_CLASSES
        if grouped[class_name]
    )

    summary = []
    for group_name, rows in groups:
        categories = Counter(
            classify_trial(row, control_mode)
            for row in rows
        )
        total = len(rows)
        correct = categories["correct"]
        summary.append(
            {
                "group": group_name,
                "total": total,
                "correct": correct,
                "recognition_failure": categories[
                    "recognition_failure"
                ],
                "nozzle_selection_failure": categories[
                    "nozzle_selection_failure"
                ],
                "physical_failure": categories["physical_failure"],
                "unfinished": categories["unfinished"],
                "success_rate": (
                    f"{correct / total:.6f}" if total else ""
                ),
            }
        )
    return summary


def write_summary(path, rows):
    """Write validation summary data."""
    fields = [
        "group",
        "total",
        "correct",
        "recognition_failure",
        "nozzle_selection_failure",
        "physical_failure",
        "unfinished",
        "success_rate",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    """Print validation results."""
    print("\n100-tube validation summary")
    print(
        "group | correct/total | recognition | nozzle selection | "
        "physical | unfinished"
    )
    for row in rows:
        print(
            f"{row['group']} | "
            f"{row['correct']}/{row['total']} | "
            f"{row['recognition_failure']} | "
            f"{row['nozzle_selection_failure']} | "
            f"{row['physical_failure']} | "
            f"{row['unfinished']}"
        )


def run_validation():
    """Start one visible validation."""
    control_mode = choose_control_mode()
    if control_mode is None:
        print("Invalid validation mode.")
        return 1
    if control_mode != "ground_truth" and not (BASE_DIR / "best.pt").exists():
        print("best.pt is required for vision-based validation.")
        return 1

    policy_path = None
    if control_mode in RL_CONTROL_MODES:
        try:
            policy_path = resolve_policy_path(
                BASE_DIR,
                control_mode,
                os.environ.get("RL_POLICY_PATH"),
            )
        except Exception as exc:
            print(f"Could not load the requested RL policy: {exc}")
            return 1
        if policy_path is None:
            print(f"No {control_mode} policy was found.")
            return 1
        print(f"RL policy: {policy_path}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = BASE_DIR / "runs" / "validation_100"
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_path = output_dir / f"{run_id}_{control_mode}_trials.csv"
    summary_path = output_dir / f"{run_id}_{control_mode}_summary.csv"

    environment = os.environ.copy()
    for name in (
        "AIR_JET_REFERENCE_IMPULSE_X_BY_JET",
        "AIR_JET_REFERENCE_IMPULSE_Z_BY_JET",
        "CALIBRATION_CLASS_IDS",
        "CALIBRATION_LATERAL_OFFSETS",
        "CALIBRATION_PAIRED_VALVE_OPENINGS",
        "CALIBRATION_TRIGGER_LEADS",
        "CALIBRATION_VALVE_OPENINGS",
        "RULE_TRIGGER_LEAD_DISTANCE_BY_JET",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "AIR_JET_CALIBRATION": "0",
            "CALIBRATION_FAST_START": "0",
            "PYBULLET_GUI": "1",
            "REMOVE_COMPLETED_TUBES": "1",
            "RULE_DISTANCE_COMPENSATION": "0",
            "RESULT_RETENTION_SECONDS": "1.0",
            "SHOW_DETECTION_WINDOW": (
                "1" if control_mode == "yolo" else "0"
            ),
            "SHOW_PYBULLET_GUI": "0",
            "SIMULATION_REALTIME": "1",
            "SIMULATION_RESULTS_CSV": str(trial_path),
            "SPAWN_INTERVAL_STEPS": "1000",
            "TARGET_TUBE_COUNT": str(TEST_TUBE_COUNT),
            "TUBE_CONTROL_MODE": control_mode,
        }
    )
    if policy_path is not None:
        environment["RL_POLICY_PATH"] = str(policy_path)
    if control_mode == "ground_truth":
        environment["TUBE_YOLO_MODEL"] = str(
            BASE_DIR / "validation_no_yolo_model.pt"
        )

    print(
        f"Starting visible {TEST_TUBE_COUNT}-tube "
        f"{control_mode} validation."
    )
    completed = subprocess.run(
        [sys.executable, str(BASE_DIR / "pneumatic_sorting_test.py")],
        cwd=str(BASE_DIR),
        env=environment,
        check=False,
    )
    if not trial_path.exists():
        print("Validation did not create a trial log.")
        return completed.returncode

    with trial_path.open(newline="", encoding="utf-8-sig") as stream:
        trials = list(csv.DictReader(stream))
    if not trials:
        print("Validation log contains no trials.")
        return completed.returncode

    summary_rows = build_summary_rows(trials, control_mode)
    write_summary(summary_path, summary_rows)
    print_summary(summary_rows)
    print(f"\nTrial log: {trial_path}")
    print(f"Summary log: {summary_path}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(run_validation())
