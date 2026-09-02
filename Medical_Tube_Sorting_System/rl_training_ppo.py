"""Train PPO for discrete or continuous proportional-valve control."""

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

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from rl_sorting_env import (
    ConveyorSortingEnv,
    evaluate_model,
    plot_policy_heatmap,
    plot_policy_position_maps,
    plot_training_curve,
    save_environment_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO for nozzle, strength, and timing control."
        )
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--action-mode",
        choices=("discrete", "continuous"),
        default="discrete",
        help=(
            "Use the six-action valve interface or one continuous "
            "YOLO-routed proportional-valve command."
        ),
    )
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
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-root",
        help=(
            "Run parent directory. The default separates discrete and "
            "continuous PPO runs."
        ),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from")
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
    if args.n_steps <= 0 or args.batch_size <= 0:
        raise ValueError("n-steps and batch-size must be positive.")
    if args.n_steps % args.batch_size != 0:
        raise ValueError(
            "For one environment, n-steps must be divisible by batch-size."
        )
    if args.resume_from and not Path(args.resume_from).exists():
        raise FileNotFoundError(
            f"Resume model not found: {args.resume_from}"
        )


def make_run_directory(args: argparse.Namespace) -> Path:
    run_name = args.run_name or datetime.now().strftime(
        f"ppo_{args.action_mode}_%Y%m%d_%H%M%S"
    )
    output_root = (
        Path(args.output_root)
        if args.output_root
        else BASE_DIR / "runs" / "rl" / "ppo" / args.action_mode
    )
    run_dir = output_root.resolve() / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train(args: argparse.Namespace) -> Path:
    validate_args(args)
    run_dir = make_run_directory(args)

    if not args.skip_env_check:
        check_environment = ConveyorSortingEnv(
            action_mode=args.action_mode,
            recognition_error_probability=args.recognition_error,
        )
        try:
            check_env(check_environment, warn=True)
        finally:
            check_environment.close()

    monitor_path = run_dir / "monitor.csv"
    raw_environment = ConveyorSortingEnv(
        action_mode=args.action_mode,
        recognition_error_probability=args.recognition_error,
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
        model = PPO.load(
            args.resume_from,
            env=environment,
            device=args.device,
        )
        model.tensorboard_log = tensorboard_log
    else:
        model = PPO(
            "MlpPolicy",
            environment,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            seed=args.seed,
            device=args.device,
            verbose=0,
            tensorboard_log=tensorboard_log,
        )

    training_settings = {
        "algorithm": "PPO",
        "action_mode": args.action_mode,
        "action_definition": (
            (
                "0 is off; actions 1-5 select valve openings 0.65, "
                "0.7, 0.8, 0.9, or 1.0 for the YOLO-routed jet."
            )
            if args.action_mode == "discrete"
            else (
                "One policy action in [-1, 1] maps linearly to a [0, 1] "
                "opening for the YOLO-routed proportional valve."
            )
        ),
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
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "device": args.device,
        "resume_from": args.resume_from,
    }
    (run_dir / "training_config.json").write_text(
        json.dumps(training_settings, indent=2),
        encoding="utf-8",
    )
    save_environment_config(run_dir / "environment_config.json")

    print(f"PPO run directory: {run_dir}")
    print(f"Starting {args.action_mode} PPO training...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=not bool(args.resume_from),
            progress_bar=False,
        )
    finally:
        environment.close()

    model_path = run_dir / f"ppo_{args.action_mode}_sorting_policy"
    model.save(str(model_path))
    print(f"Saved PPO model: {model_path}.zip")

    if not args.skip_evaluation:
        summary = evaluate_model(
            model,
            action_mode=args.action_mode,
            episodes=args.eval_episodes,
            seed=args.seed,
            output_csv=run_dir / "evaluation.csv",
            recognition_error_probability=args.recognition_error,
        )
        plot_policy_heatmap(
            model,
            action_mode=args.action_mode,
            output_png=run_dir / "policy_heatmap.png",
        )
        plot_policy_position_maps(
            model,
            action_mode=args.action_mode,
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
