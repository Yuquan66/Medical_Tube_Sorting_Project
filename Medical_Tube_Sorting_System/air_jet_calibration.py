"""Run visible air-jet calibration trials."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tube_specs import TUBE_SPECS


BASE_DIR = Path(__file__).resolve().parent
TUBE_CLASSES = tuple(spec.name for spec in TUBE_SPECS)
LATERAL_OFFSETS = (-0.35, -0.175, 0.0, 0.175, 0.35)
NOMINAL_PRESSURE_KPA = 55.0


def numeric_values(rows, field):
    """Return available numeric values."""
    values = []
    for row in rows:
        value = row.get(field, "")
        if value == "":
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def mean_value(rows, field):
    """Return a formatted mean."""
    values = numeric_values(rows, field)
    if not values:
        return ""
    return f"{sum(values) / len(values):.6f}"


def build_summary_rows(trials):
    """Group calibration results."""
    groups = []
    offset_groups = defaultdict(list)
    class_groups = defaultdict(list)

    for row in trials:
        offset_groups[row["spawn_lateral_offset_m"]].append(row)
        class_groups[row["evaluation_class"]].append(row)

    for offset in sorted(offset_groups, key=float):
        groups.append(("offset_m", offset, offset_groups[offset]))
    for class_name in TUBE_CLASSES:
        if class_name in class_groups:
            groups.append(("tube_class", class_name, class_groups[class_name]))

    summary = []
    for group_type, group_value, rows in groups:
        correct = sum(row.get("outcome") == "correct_bin" for row in rows)
        summary.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "trials": len(rows),
                "correct": correct,
                "success_rate": f"{correct / len(rows):.6f}",
                "mean_distance_m": mean_value(rows, "jet_distance_m"),
                "mean_axial_distance_m": mean_value(
                    rows, "jet_axial_distance_m"
                ),
                "mean_radial_offset_m": mean_value(
                    rows, "jet_radial_offset_m"
                ),
                "mean_attenuation": mean_value(rows, "jet_attenuation"),
                "mean_peak_force_n": mean_value(
                    rows, "jet_peak_force_n"
                ),
                "mean_impulse_ns": mean_value(rows, "jet_impulse_ns"),
                "mean_end_velocity_x_mps": mean_value(
                    rows, "jet_end_velocity_x_mps"
                ),
                "mean_end_velocity_z_mps": mean_value(
                    rows, "jet_end_velocity_z_mps"
                ),
            }
        )
    return summary


def write_summary(path, rows):
    """Write grouped calibration results."""
    fields = [
        "group_type",
        "group_value",
        "trials",
        "correct",
        "success_rate",
        "mean_distance_m",
        "mean_axial_distance_m",
        "mean_radial_offset_m",
        "mean_attenuation",
        "mean_peak_force_n",
        "mean_impulse_ns",
        "mean_end_velocity_x_mps",
        "mean_end_velocity_z_mps",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(trials, summary_rows):
    """Print the calibration result."""
    correct = sum(row.get("outcome") == "correct_bin" for row in trials)
    outcomes = defaultdict(int)
    for row in trials:
        outcomes[row.get("outcome", "unknown")] += 1

    print("\nAir-jet calibration result")
    print(f"Trials completed: {len(trials)}")
    print(f"Correct bins: {correct}")
    print(f"Outcomes: {dict(outcomes)}")
    print("\nOffset summary")
    print("offset (m) | correct/trials | attenuation | force (N) | impulse (N s)")
    for row in summary_rows:
        if row["group_type"] != "offset_m":
            continue
        print(
            f"{float(row['group_value']):>10.3f} | "
            f"{row['correct']:>2}/{row['trials']:<2}          | "
            f"{row['mean_attenuation']:>11} | "
            f"{row['mean_peak_force_n']:>9} | "
            f"{row['mean_impulse_ns']:>13}"
        )


def run_calibration():
    """Start the visible calibration run."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = BASE_DIR / "runs" / "air_jet_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_path = output_dir / f"{run_id}_trials.csv"
    summary_path = output_dir / f"{run_id}_summary.csv"

    environment = os.environ.copy()
    environment.update(
        {
            "AIR_JET_CALIBRATION": "1",
            "CALIBRATION_LATERAL_OFFSETS": ",".join(
                str(value) for value in LATERAL_OFFSETS
            ),
            "NOMINAL_JET_PRESSURE_KPA": str(NOMINAL_PRESSURE_KPA),
            "PYBULLET_GUI": "1",
            "REMOVE_COMPLETED_TUBES": "1",
            "RESULT_RETENTION_SECONDS": "1.0",
            "SHOW_DETECTION_WINDOW": "0",
            "SHOW_PYBULLET_GUI": "0",
            "SIMULATION_REALTIME": "1",
            "SIMULATION_RESULTS_CSV": str(trial_path),
            "TARGET_TUBE_COUNT": str(
                len(TUBE_CLASSES) * len(LATERAL_OFFSETS)
            ),
            "TUBE_CONTROL_MODE": "ground_truth",
            "TUBE_YOLO_MODEL": str(
                BASE_DIR / "calibration_no_yolo_model.pt"
            ),
        }
    )

    print("Starting visible air-jet calibration.")
    print(
        f"Nominal pressure label: {NOMINAL_PRESSURE_KPA:.1f} kPa. "
        "It is not converted directly into force."
    )
    print(
        f"Trials: {len(TUBE_CLASSES)} tube classes x "
        f"{len(LATERAL_OFFSETS)} lateral positions."
    )
    print("Each trial runs alone to prevent tube collisions.")

    completed = subprocess.run(
        [sys.executable, str(BASE_DIR / "pneumatic_sorting_test.py")],
        cwd=str(BASE_DIR),
        env=environment,
        check=False,
    )
    if not trial_path.exists():
        print("Calibration did not create a trial log.")
        return completed.returncode

    with trial_path.open(newline="", encoding="utf-8-sig") as stream:
        trials = list(csv.DictReader(stream))
    if not trials:
        print("Calibration log contains no completed trials.")
        return completed.returncode

    summary_rows = build_summary_rows(trials)
    write_summary(summary_path, summary_rows)
    print_summary(trials, summary_rows)
    print(f"\nTrial log: {trial_path}")
    print(f"Summary log: {summary_path}")
    print("No pneumatic parameter was changed automatically.")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(run_calibration())
