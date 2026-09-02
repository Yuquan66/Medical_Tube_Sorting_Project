"""Plot paired end-to-end validation results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_ORDER = [
    "universal polystyrene tube",
    "universal polypropylene tube",
    "polypropylene centrifuge tube 1",
    "polypropylene centrifuge tube 2",
    "polypropylene lysis tube",
]
CLASS_LABELS = ["PS", "PP", "Centrifuge 1", "Centrifuge 2", "Lysis"]


def parse_args():
    """Read command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("improved", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path):
    """Read one trial file."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def category(row):
    """Return the top-level outcome category."""
    expected = int(row["evaluation_class_id"])
    detected = row.get("detected_class_id", "")
    if detected == "" or int(detected) != expected:
        return "Recognition failure"
    selected = row.get("selected_jet", "")
    if selected == "" or int(selected) != expected + 1:
        return "Nozzle selection failure"
    if row.get("outcome") == "correct_bin":
        return "Correct"
    return "Physical failure"


def summarize(rows):
    """Calculate overall and class metrics."""
    counts = Counter(category(row) for row in rows)
    class_totals = Counter(row["evaluation_class"] for row in rows)
    class_correct = Counter(
        row["evaluation_class"]
        for row in rows
        if category(row) == "Correct"
    )
    return {
        "total": len(rows),
        "correct": counts["Correct"],
        "success_rate": counts["Correct"] / len(rows),
        "recognition_failure": counts["Recognition failure"],
        "nozzle_selection_failure": counts[
            "Nozzle selection failure"
        ],
        "physical_failure": counts["Physical failure"],
        "class_total": dict(class_totals),
        "class_correct": dict(class_correct),
    }


def save_overall_figure(baseline, improved, output_dir):
    """Plot overall results and failure counts."""
    names = ["Baseline", "Improved"]
    rates = [baseline["success_rate"] * 100, improved["success_rate"] * 100]
    colors = ["#7b8da6", "#188977"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bars = axes[0].bar(names, rates, color=colors, width=0.58)
    axes[0].set_ylabel("Successful sorting (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("End-to-end sorting success")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, rates):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.0f}%",
            ha="center",
            fontweight="bold",
        )

    failure_names = [
        "Recognition",
        "Nozzle selection",
        "Physical",
    ]
    baseline_failures = [
        baseline["recognition_failure"],
        baseline["nozzle_selection_failure"],
        baseline["physical_failure"],
    ]
    improved_failures = [
        improved["recognition_failure"],
        improved["nozzle_selection_failure"],
        improved["physical_failure"],
    ]
    x = np.arange(len(failure_names))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        baseline_failures,
        width,
        label="Baseline",
        color=colors[0],
    )
    axes[1].bar(
        x + width / 2,
        improved_failures,
        width,
        label="Improved",
        color=colors[1],
    )
    axes[1].set_xticks(x, failure_names)
    axes[1].set_ylabel("Number of tubes")
    axes[1].set_title("Failure decomposition")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "overall_and_failures.png", dpi=220)
    plt.close(fig)


def save_class_figure(baseline, improved, output_dir):
    """Plot success rate for each class."""
    baseline_rates = []
    improved_rates = []
    for class_name in CLASS_ORDER:
        baseline_rates.append(
            100
            * baseline["class_correct"].get(class_name, 0)
            / baseline["class_total"].get(class_name, 1)
        )
        improved_rates.append(
            100
            * improved["class_correct"].get(class_name, 0)
            / improved["class_total"].get(class_name, 1)
        )

    x = np.arange(len(CLASS_ORDER))
    width = 0.37
    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.bar(
        x - width / 2,
        baseline_rates,
        width,
        label="Baseline",
        color="#7b8da6",
    )
    ax.bar(
        x + width / 2,
        improved_rates,
        width,
        label="Improved",
        color="#188977",
    )
    ax.set_xticks(x, CLASS_LABELS)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Successful sorting (%)")
    ax.set_title("Sorting success by tube class")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "class_success_rate.png", dpi=220)
    plt.close(fig)


def save_paired_figure(baseline_rows, improved_rows, output_dir):
    """Plot paired changes between the two runs."""
    baseline_by_id = {
        int(row["tube_sequence"]): category(row) for row in baseline_rows
    }
    improved_by_id = {
        int(row["tube_sequence"]): category(row) for row in improved_rows
    }
    paired_ids = sorted(set(baseline_by_id) & set(improved_by_id))
    transitions = Counter()
    for tube_id in paired_ids:
        before = baseline_by_id[tube_id] == "Correct"
        after = improved_by_id[tube_id] == "Correct"
        if before and after:
            transitions["Correct in both"] += 1
        elif not before and after:
            transitions["Fixed by improvement"] += 1
        elif before and not after:
            transitions["Regression"] += 1
        else:
            transitions["Failed in both"] += 1

    labels = [
        "Correct in both",
        "Fixed by improvement",
        "Regression",
        "Failed in both",
    ]
    values = [transitions[label] for label in labels]
    colors = ["#188977", "#58b7a6", "#cf5c5c", "#b6bdc8"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Number of paired tubes")
    ax.set_title("Paired outcome changes")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            str(value),
            ha="center",
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "paired_outcome_changes.png", dpi=220)
    plt.close(fig)
    return dict(transitions)


def save_action_figure(rows, output_dir):
    """Plot the physical impulse and result by lateral position."""
    grouped = defaultdict(list)
    for row in rows:
        if row.get("estimated_command_x", "") == "":
            continue
        if category(row) not in {"Correct", "Physical failure"}:
            continue
        grouped[row["evaluation_class"]].append(row)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=False)
    axes = axes.ravel()
    for index, class_name in enumerate(CLASS_ORDER):
        ax = axes[index]
        for row in grouped[class_name]:
            is_correct = category(row) == "Correct"
            ax.scatter(
                float(row["estimated_command_x"]),
                float(row["jet_impulse_ns"]),
                color="#188977" if is_correct else "#cf5c5c",
                marker="o" if is_correct else "x",
                s=34,
                alpha=0.85,
            )
        ax.set_title(CLASS_LABELS[index])
        ax.set_xlabel("Estimated x position")
        ax.set_ylabel("Effective impulse (N s)")
        ax.grid(alpha=0.22)
    axes[-1].axis("off")
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#188977",
            label="Correct", markersize=7,
        ),
        plt.Line2D(
            [0], [0], marker="x", color="#cf5c5c", label="Failed",
            markersize=7, linestyle="None",
        ),
    ]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.93, 0.12))
    fig.suptitle("Improved controller: position, impulse and outcome", y=0.99)
    fig.tight_layout()
    fig.savefig(output_dir / "position_impulse_outcomes.png", dpi=220)
    plt.close(fig)


def save_tables(
    baseline_rows,
    improved_rows,
    baseline_summary,
    improved_summary,
    transitions,
    output_dir,
):
    """Write summary files."""
    table_path = output_dir / "end_to_end_comparison.csv"
    fields = [
        "configuration",
        "total",
        "correct",
        "success_rate",
        "recognition_failure",
        "nozzle_selection_failure",
        "physical_failure",
    ]
    with table_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, summary in (
            ("baseline", baseline_summary),
            ("improved", improved_summary),
        ):
            writer.writerow(
                {
                    "configuration": name,
                    **{field: summary[field] for field in fields[1:]},
                }
            )

    report = {
        "baseline": baseline_summary,
        "improved": improved_summary,
        "paired_transitions": transitions,
        "paired_tube_count": min(len(baseline_rows), len(improved_rows)),
        "absolute_success_rate_change": (
            improved_summary["success_rate"]
            - baseline_summary["success_rate"]
        ),
        "physical_failure_reduction": (
            baseline_summary["physical_failure"]
            - improved_summary["physical_failure"]
        ),
    }
    with (output_dir / "end_to_end_comparison.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, indent=2)


def main():
    """Create all paired validation outputs."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = read_rows(args.baseline)
    improved_rows = read_rows(args.improved)
    baseline_summary = summarize(baseline_rows)
    improved_summary = summarize(improved_rows)
    save_overall_figure(baseline_summary, improved_summary, args.output_dir)
    save_class_figure(baseline_summary, improved_summary, args.output_dir)
    transitions = save_paired_figure(
        baseline_rows, improved_rows, args.output_dir
    )
    save_action_figure(improved_rows, args.output_dir)
    save_tables(
        baseline_rows,
        improved_rows,
        baseline_summary,
        improved_summary,
        transitions,
        args.output_dir,
    )
    print(f"Saved paired validation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
