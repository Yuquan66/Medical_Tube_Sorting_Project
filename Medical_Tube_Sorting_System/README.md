# Medical Tube Sorting Simulation

This repository contains the PyBullet simulation, YOLO detector, pneumatic jet model, reinforcement learning controllers, trained models, and experiment results used for the medical plastic tube sorting project.

## Setup

Python 3.10 or later is recommended.

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Run the project

```text
.venv\Scripts\python tube_detection_project.py
```

The menu provides dataset generation, YOLO training, rule based simulation, DQN training, discrete and continuous PPO training, continuous SAC training, trained policy simulation, air jet calibration, 100 tube validation, and controller comparison.

## Main files

- `pneumatic_sorting_test.py`: visible conveyor and pneumatic sorting simulation.
- `object_detection.py`: synthetic YOLO dataset generation.
- `train_yolo.py`: YOLO training and evaluation.
- `rl_sorting_env.py`: reinforcement learning environment.
- `rl_training_dqn.py`: discrete DQN training.
- `rl_training_ppo.py`: discrete or continuous PPO training.
- `rl_training_sac.py`: continuous SAC training.
- `compare_rl_controllers.py`: common seed controller evaluation.
- `compare_sac_variants.py`: original and improved SAC evaluation.

## Data and results

- `datasets/`: the 1,000 image synthetic dataset and 400 image coloured dataset.
- `best.pt`: YOLO model used by the main simulation.
- `runs/rl/`: trained policies, training records, and controller comparisons.
- `runs/validation_100/`: paired 100 tube end to end validation data and figures.
- `runs/air_jet_calibration/`: pneumatic parameter calibration records.

All stored paths are relative to the repository root. Experimental CSV and JSON files contain the recorded results used to generate the comparison figures.
