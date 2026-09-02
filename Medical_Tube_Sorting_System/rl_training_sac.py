"""Train SAC for continuous proportional-valve control."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / ".runtime"
RUNTIME_DIR.mkdir(exist_ok=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from rl_sorting_env import (
    ConveyorSortingEnv,
    SortingConfig,
    evaluate_model,
    plot_policy_heatmap,
    plot_policy_position_maps,
    plot_training_curve,
    save_environment_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train SAC with one continuous YOLO-routed valve action."
        )
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--recognition-error",
        type=float,
        default=0.0,
        help=(
            "Probability that the detector class supplied to the policy "
            "differs from the private evaluation class."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--energy-penalty", type=float, default=0.02)
    parser.add_argument(
        "--correct-jet-intensity-weight",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--actuation-intensity-penalty",
        type=float,
        default=0.0,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-root",
        default=str(BASE_DIR / "runs" / "rl" / "sac"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from")
    parser.add_argument("--resume-replay-buffer")
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.timesteps <= 0:
        raise ValueError("timesteps must be positive.")
    if args.eval_episodes <= 0:
        raise ValueError("eval-episodes must be positive.")
    if not 0.0 <= args.recognition_error < 1.0:
        raise ValueError("recognition-error must be in [0, 1).")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive.")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1].")
    if args.buffer_size <= 0 or args.learning_starts < 0:
        raise ValueError(
            "buffer-size must be positive and learning-starts non-negative."
        )
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if not 0.0 < args.tau <= 1.0:
        raise ValueError("tau must be in (0, 1].")
    if args.energy_penalty < 0.0:
        raise ValueError("energy-penalty must be non-negative.")
    if not 0.0 <= args.correct_jet_intensity_weight <= 1.0:
        raise ValueError(
            "correct-jet-intensity-weight must be in [0, 1]."
        )
    if args.actuation_intensity_penalty < 0.0:
        raise ValueError(
            "actuation-intensity-penalty must be non-negative."
        )
    for path in (args.resume_from, args.resume_replay_buffer):
        if path and not Path(path).exists():
            raise FileNotFoundError(f"Resume file not found: {path}")


def make_run_directory(args: argparse.Namespace) -> Path:
    run_name = args.run_name or datetime.now().strftime(
        "sac_%Y%m%d_%H%M%S"
    )
    run_dir = Path(args.output_root).resolve() / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train(args: argparse.Namespace) -> Path:
    validate_args(args)
    run_dir = make_run_directory(args)
    sorting_config = SortingConfig(
        energy_penalty=args.energy_penalty,
        correct_jet_intensity_weight=(
            args.correct_jet_intensity_weight
        ),
        actuation_intensity_penalty=(
            args.actuation_intensity_penalty
        ),
    )

    if not args.skip_env_check:
        check_environment = ConveyorSortingEnv(
            action_mode="continuous",
            recognition_error_probability=args.recognition_error,
            config=sorting_config,
        )
        try:
            check_env(check_environment, warn=True)
        finally:
            check_environment.close()

    monitor_path = run_dir / "monitor.csv"
    raw_environment = ConveyorSortingEnv(
        action_mode="continuous",
        recognition_error_probability=args.recognition_error,
        config=sorting_config,
    )
    environment = Monitor(
        raw_environment,
        filename=str(monitor_path),
    )

    tensorboard_log = (
        str(run_dir / "tensorboard")
        if args.tensorboard
        else None
    )
    if args.resume_from:
        model = SAC.load(
            args.resume_from,
            env=environment,
            device=args.device,
        )
        model.tensorboard_log = tensorboard_log
        if args.resume_replay_buffer:
            model.load_replay_buffer(args.resume_replay_buffer)
    else:
        model = SAC(
            "MlpPolicy",
            environment,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=1,
            gradient_steps=1,
            seed=args.seed,
            device=args.device,
            verbose=0,
            tensorboard_log=tensorboard_log,
        )

    training_settings = {
        "algorithm": "SAC",
        "hardware_assumption": (
            "Five proportional valves are selected by the YOLO class; "
            "SAC continuously controls the selected valve opening."
        ),
        "action_definition": {
            "shape": [1],
            "policy_range": [-1.0, 1.0],
            "physical_mapping": (
                "opening=(action+1)/2; the system applies it to the "
                "YOLO-routed proportional valve"
            ),
        },
        "observation": [
            "normalized_x_position",
            "normalized_y_distance_to_target_jet",
            "normalized_detected_class",
        ],
        "timesteps": args.timesteps,
        "seed": args.seed,
        "recognition_error_probability": args.recognition_error,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "tau": args.tau,
        "energy_penalty": args.energy_penalty,
        "correct_jet_intensity_weight": (
            args.correct_jet_intensity_weight
        ),
        "actuation_intensity_penalty": (
            args.actuation_intensity_penalty
        ),
        "device": args.device,
        "resume_from": args.resume_from,
        "resume_replay_buffer": args.resume_replay_buffer,
    }
    (run_dir / "training_config.json").write_text(
        json.dumps(training_settings, indent=2),
        encoding="utf-8",
    )
    save_environment_config(
        run_dir / "environment_config.json",
        config=sorting_config,
    )

    print(f"SAC run directory: {run_dir}")
    print(
        "Starting continuous SAC training with a YOLO-routed valve..."
    )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=not bool(args.resume_from),
            progress_bar=False,
        )
    finally:
        environment.close()

    model_path = run_dir / "sac_sorting_policy"
    model.save(str(model_path))
    model.save_replay_buffer(str(run_dir / "replay_buffer.pkl"))
    print(f"Saved SAC model: {model_path}.zip")

    if not args.skip_evaluation:
        summary = evaluate_model(
            model,
            action_mode="continuous",
            episodes=args.eval_episodes,
            seed=args.seed,
            output_csv=run_dir / "evaluation.csv",
            recognition_error_probability=args.recognition_error,
            config=sorting_config,
        )
        plot_policy_heatmap(
            model,
            action_mode="continuous",
            output_png=run_dir / "policy_heatmap.png",
        )
        plot_policy_position_maps(
            model,
            action_mode="continuous",
            output_png=run_dir / "policy_position_maps.png",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    plot_training_curve(
        monitor_path,
        run_dir / "training_reward.png",
    )
    return model_path.with_suffix(".zip")


if __name__ == "__main__":
    train(parse_args())
