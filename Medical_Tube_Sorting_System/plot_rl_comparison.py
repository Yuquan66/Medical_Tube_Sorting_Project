"""Create report figures from saved RL evaluation files."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / ".runtime"
RUNTIME_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


CONTROLLERS = (
    ("fixed_rule", "Fixed rule", "#7f7f7f"),
    ("dqn", "DQN", "#4c78a8"),
    ("ppo_discrete", "PPO discrete", "#59a14f"),
    ("ppo_continuous", "PPO continuous", "#f28e2b"),
    ("sac", "SAC", "#e15759"),
)

CLASS_LABELS = {
    "universal polystyrene tube": "Universal PS",
    "universal polypropylene tube": "Universal PP",
    "polypropylene centrifuge tube 1": "PP centrifuge 1",
    "polypropylene centrifuge tube 2": "PP centrifuge 2",
    "polypropylene lysis tube": "PP lysis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot saved RL comparison results."
    )
    parser.add_argument(
        "--comparison-dir",
        default=str(
            BASE_DIR / "runs" / "rl" / "comparisons" / "formal_common_v2"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--original-sac-dir",
        default=str(
            BASE_DIR
            / "runs"
            / "rl"
            / "sac"
            / "sac_continuous_formal_v2_seed42"
        ),
    )
    parser.add_argument(
        "--sac-comparison-dir",
        default=str(
            BASE_DIR
            / "runs"
            / "rl"
            / "comparisons"
            / "sac_improved_v1"
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(figure: Any, path: Path) -> None:
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_success_rate(
    summary_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    by_key = {row["policy_key"]: row for row in summary_rows}
    labels = [item[1] for item in CONTROLLERS]
    colors = [item[2] for item in CONTROLLERS]
    means = [
        float(by_key[item[0]]["mean_success_rate"]) * 100.0
        for item in CONTROLLERS
    ]
    errors = [
        float(by_key[item[0]]["success_rate_sd"]) * 100.0
        for item in CONTROLLERS
    ]

    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    bars = axis.bar(
        labels,
        means,
        yerr=errors,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )
    for bar, value in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 2.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )
    axis.set_ylim(0.0, 108.0)
    axis.set_ylabel("Sorting success rate (%)")
    axis.set_title("Controller Performance Across Five Test Seeds")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_seed_distribution(
    seed_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    positions = np.arange(1, len(CONTROLLERS) + 1)
    values = []
    for key, _, _ in CONTROLLERS:
        rows = [row for row in seed_rows if row["policy_key"] == key]
        rows.sort(key=lambda row: int(row["seed"]))
        values.append(
            [float(row["success_rate"]) * 100.0 for row in rows]
        )
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showmeans=True,
    )
    for patch, controller in zip(boxes["boxes"], CONTROLLERS):
        patch.set_facecolor(controller[2])
        patch.set_alpha(0.45)
    for index, (series, controller) in enumerate(
        zip(values, CONTROLLERS),
        start=1,
    ):
        offsets = np.linspace(-0.12, 0.12, len(series))
        axis.scatter(
            index + offsets,
            series,
            color=controller[2],
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([item[1] for item in CONTROLLERS])
    axis.set_ylabel("Sorting success rate (%)")
    axis.set_title("Success Rate for Test Seeds 42 to 46")
    axis.set_ylim(35.0, 103.0)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_class_success(
    class_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    class_names = list(CLASS_LABELS)
    x_values = np.arange(len(class_names))
    width = 0.16
    figure, axis = plt.subplots(figsize=(12.0, 5.8))
    for index, (key, label, color) in enumerate(CONTROLLERS):
        by_class = {
            row["tube_class"]: float(row["success_rate"]) * 100.0
            for row in class_rows
            if row["policy_key"] == key
        }
        values = [by_class[class_name] for class_name in class_names]
        axis.bar(
            x_values + (index - 2) * width,
            values,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.4,
        )
    axis.set_xticks(x_values)
    axis.set_xticklabels(
        [CLASS_LABELS[class_name] for class_name in class_names]
    )
    axis.set_ylabel("Sorting success rate (%)")
    axis.set_title("Controller Performance by Tube Class")
    axis.set_ylim(0.0, 108.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, loc="lower center")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_outcomes(
    summary_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    by_key = {row["policy_key"]: row for row in summary_rows}
    labels = [item[1] for item in CONTROLLERS]
    outcome_fields = (
        ("correct_bin", "Correct bin", "#59a14f"),
        ("missed_bin", "Missed bin", "#e15759"),
        ("wrong_bin", "Wrong bin", "#f28e2b"),
        ("missed_nozzle", "Missed nozzle", "#4e79a7"),
        ("truncated", "Truncated", "#b07aa1"),
    )
    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    bottom = np.zeros(len(CONTROLLERS), dtype=float)
    for field, label, color in outcome_fields:
        values = np.array(
            [float(by_key[item[0]][field]) for item in CONTROLLERS]
        )
        axis.bar(
            labels,
            values,
            bottom=bottom,
            label=label,
            color=color,
        )
        bottom += values
    axis.set_ylabel("Episodes")
    axis.set_title("Outcome Counts from 500 Test Episodes")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, loc="upper center")
    figure.tight_layout()
    save_figure(figure, output_path)


def read_monitor(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("r", encoding="utf-8") as stream:
        lines = [line for line in stream if not line.startswith("#")]
    rows = list(csv.DictReader(lines))
    lengths = np.array([float(row["l"]) for row in rows], dtype=float)
    rewards = np.array([float(row["r"]) for row in rows], dtype=float)
    return np.cumsum(lengths), rewards


def moving_average(values: np.ndarray, window: int = 200) -> np.ndarray:
    if values.size < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    averaged = np.convolve(values, kernel, mode="valid")
    prefix = np.full(window - 1, np.nan, dtype=float)
    return np.concatenate((prefix, averaged))


def plot_training_curves(output_path: Path) -> None:
    paths = {
        "DQN": BASE_DIR
        / "runs"
        / "rl"
        / "dqn"
        / "dqn_discrete_formal_v5_seed42"
        / "monitor.csv",
        "PPO discrete": BASE_DIR
        / "runs"
        / "rl"
        / "ppo"
        / "discrete"
        / "ppo_discrete_formal_seed42"
        / "monitor.csv",
        "PPO continuous": BASE_DIR
        / "runs"
        / "rl"
        / "ppo"
        / "continuous"
        / "ppo_continuous_formal_seed42"
        / "monitor.csv",
        "SAC": BASE_DIR
        / "runs"
        / "rl"
        / "sac"
        / "sac_continuous_formal_v2_seed42"
        / "monitor.csv",
    }
    colors = {item[1]: item[2] for item in CONTROLLERS}
    figure, axis = plt.subplots(figsize=(10.0, 5.5))
    for label, path in paths.items():
        steps, rewards = read_monitor(path)
        axis.plot(
            steps,
            moving_average(rewards),
            label=label,
            color=colors[label],
            linewidth=1.5,
        )
    axis.set_xlim(0.0, 200_000.0)
    axis.set_xlabel("Training steps")
    axis.set_ylabel("Mean episode return (200-episode window)")
    axis.set_title("Training Curves for the Four RL Controllers")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_lateral_intensity(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 5.6))
    colors = plt.get_cmap("tab10")
    for class_index, class_name in enumerate(CLASS_LABELS):
        class_rows = [row for row in rows if row["true_class"] == class_name]
        x_values = np.array(
            [float(row["lateral_offset"]) / 0.35 for row in class_rows]
        )
        intensities = np.array(
            [float(row["fired_intensity"]) for row in class_rows]
        )
        axis.scatter(
            x_values,
            intensities,
            s=18,
            alpha=0.28,
            color=colors(class_index),
            label=CLASS_LABELS[class_name],
        )
        bins = np.linspace(-1.0, 1.0, 7)
        centres = (bins[:-1] + bins[1:]) / 2.0
        means = []
        for lower, upper in zip(bins[:-1], bins[1:]):
            selected = intensities[
                (x_values >= lower) & (x_values < upper)
            ]
            means.append(float(np.mean(selected)) if selected.size else np.nan)
        axis.plot(
            centres,
            means,
            color=colors(class_index),
            linewidth=2.0,
        )
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(0.55, 1.02)
    axis.set_xlabel("Normalized lateral tube position")
    axis.set_ylabel("Firing intensity")
    axis.set_title("SAC Firing Intensity Against Lateral Position")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_sac_improved_summary(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    labels = [row["controller"] for row in rows]
    colors = ("#e15759", "#59a14f")
    success = [float(row["mean_success_rate"]) * 100.0 for row in rows]
    errors = [float(row["success_rate_sd"]) * 100.0 for row in rows]
    intensities = [float(row["mean_fired_intensity"]) for row in rows]
    impulses = [
        float(row["mean_jet_accumulated_impulse"]) for row in rows
    ]

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.5))
    metrics = (
        (success, errors, "Success rate (%)", (95.0, 101.0)),
        (intensities, None, "Mean firing intensity", (0.85, 0.97)),
        (impulses, None, "Mean effective impulse (N s)", (0.065, 0.082)),
    )
    for axis, (values, error, title, limits) in zip(axes, metrics):
        bars = axis.bar(
            labels,
            values,
            yerr=error,
            capsize=5,
            color=colors,
            edgecolor="black",
        )
        axis.set_title(title)
        axis.set_ylim(*limits)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + (limits[1] - limits[0]) * 0.035,
                f"{value:.4g}",
                ha="center",
            )
    figure.suptitle("Original and Improved SAC Across Five Test Seeds")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_sac_variant_lateral(
    comparison_dir: Path,
    output_path: Path,
) -> None:
    rows_by_controller = {
        "Original SAC": [],
        "Improved SAC": [],
    }
    for label, prefix in (
        ("Original SAC", "original_sac_seed"),
        ("Improved SAC", "improved_sac_seed"),
    ):
        for path in sorted(comparison_dir.glob(f"{prefix}*.csv")):
            rows_by_controller[label].extend(read_csv(path))

    figure, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), sharex=True)
    flat_axes = axes.ravel()
    bins = np.linspace(-1.0, 1.0, 9)
    centres = (bins[:-1] + bins[1:]) / 2.0
    styles = {
        "Original SAC": ("#e15759", "o"),
        "Improved SAC": ("#59a14f", "s"),
    }
    for class_index, class_name in enumerate(CLASS_LABELS):
        axis = flat_axes[class_index]
        for label, rows in rows_by_controller.items():
            class_rows = [row for row in rows if row["true_class"] == class_name]
            x_values = np.array(
                [float(row["lateral_offset"]) / 0.35 for row in class_rows]
            )
            intensities = np.array(
                [float(row["fired_intensity"]) for row in class_rows]
            )
            means = []
            for lower, upper in zip(bins[:-1], bins[1:]):
                selected = intensities[
                    (x_values >= lower) & (x_values < upper)
                ]
                means.append(
                    float(np.mean(selected)) if selected.size else np.nan
                )
            color, marker = styles[label]
            axis.plot(
                centres,
                means,
                color=color,
                marker=marker,
                linewidth=1.8,
                label=label,
            )
        axis.set_title(CLASS_LABELS[class_name])
        axis.set_ylim(0.68, 1.01)
        axis.grid(alpha=0.25)
    flat_axes[-1].set_visible(False)
    for axis in axes[-1, :2]:
        axis.set_xlabel("Normalized lateral position")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean firing intensity")
    flat_axes[0].legend()
    figure.suptitle("SAC Firing Intensity Before and After Reward Improvement")
    figure.tight_layout()
    save_figure(figure, output_path)


def main(args: argparse.Namespace) -> Path:
    comparison_dir = Path(args.comparison_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else comparison_dir / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(comparison_dir / "comparison_summary.csv")
    seed_rows = read_csv(comparison_dir / "comparison_by_seed.csv")
    class_rows = read_csv(
        comparison_dir / "comparison_by_class_summary.csv"
    )
    original_rows = read_csv(Path(args.original_sac_dir) / "evaluation.csv")
    sac_comparison_dir = Path(args.sac_comparison_dir).resolve()
    sac_comparison_rows = read_csv(
        sac_comparison_dir / "comparison_summary.csv"
    )

    figures = {
        "controller_success_rate.png": lambda path: plot_success_rate(
            summary_rows, path
        ),
        "seed_success_distribution.png": lambda path: plot_seed_distribution(
            seed_rows, path
        ),
        "class_success_rate.png": lambda path: plot_class_success(
            class_rows, path
        ),
        "outcome_counts.png": lambda path: plot_outcomes(summary_rows, path),
        "training_curves_comparison.png": plot_training_curves,
        "sac_lateral_intensity.png": lambda path: plot_lateral_intensity(
            original_rows, path
        ),
        "sac_improved_five_seed_comparison.png": lambda path: (
            plot_sac_improved_summary(sac_comparison_rows, path)
        ),
        "sac_improved_lateral_intensity.png": lambda path: (
            plot_sac_variant_lateral(sac_comparison_dir, path)
        ),
    }
    for name, plotter in figures.items():
        plotter(output_dir / name)

    manifest = {
        "comparison_directory": str(comparison_dir),
        "original_sac_directory": str(Path(args.original_sac_dir).resolve()),
        "sac_comparison_directory": str(sac_comparison_dir),
        "figures": [str((output_dir / name).resolve()) for name in figures],
    }
    (output_dir / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return output_dir


if __name__ == "__main__":
    main(parse_args())
