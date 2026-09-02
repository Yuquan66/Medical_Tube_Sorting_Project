import os
import subprocess
import sys
from pathlib import Path

from rl_policy_runtime import resolve_policy_path


BASE_DIR = Path(__file__).resolve().parent
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

RL_ALGORITHM_LABELS = {
    "dqn": "DQN - Discrete",
    "ppo_discrete": "PPO - Discrete",
    "ppo_continuous": "PPO - Continuous",
    "sac": "SAC - Continuous",
}


def run_script(script_name, *, environment=None, arguments=None):
    command = [sys.executable, str(BASE_DIR / script_name)]
    if arguments:
        command.extend(arguments)
    return subprocess.run(
        command,
        cwd=str(BASE_DIR),
        env=environment,
        check=False,
    )


def run_main_simulation():
    print("\nSimulation control modes:")
    print("1. Auto - YOLO if best.pt exists, otherwise debug control")
    print("2. Strict YOLO - disable jets if best.pt is unavailable")
    print("3. Ground truth - mechanical debug only")
    choice = input("Select a mode [1]: ").strip() or "1"
    modes = {
        "1": "auto",
        "2": "yolo",
        "3": "ground_truth",
    }
    control_mode = modes.get(choice)
    if control_mode is None:
        print("Invalid simulation mode.")
        return

    environment = os.environ.copy()
    environment["TUBE_CONTROL_MODE"] = control_mode
    run_script("pneumatic_sorting_test.py", environment=environment)


def choose_rl_algorithm():
    print("\nRL algorithms:")
    print("1. DQN - Discrete")
    print("2. PPO - Discrete")
    print("3. PPO - Continuous")
    print("4. SAC - Continuous")
    choice = input("Select an algorithm: ").strip()
    return {
        "1": "dqn",
        "2": "ppo_discrete",
        "3": "ppo_continuous",
        "4": "sac",
    }.get(choice)


def train_ppo():
    print("\nPPO action modes:")
    print("1. Discrete - off or one of five valve openings")
    print("2. Continuous - one proportional-valve opening")
    choice = input("Select an action mode [1]: ").strip() or "1"
    action_mode = {
        "1": "discrete",
        "2": "continuous",
    }.get(choice)
    if action_mode is None:
        print("Invalid PPO action mode.")
        return
    run_script(
        "rl_training_ppo.py",
        arguments=["--action-mode", action_mode],
    )


def run_rl_simulation():
    algorithm = choose_rl_algorithm()
    if algorithm is None:
        print("Invalid RL algorithm.")
        return

    if not (BASE_DIR / "best.pt").exists():
        print("best.pt is required for YOLO + RL simulation.")
        return

    try:
        policy_path = resolve_policy_path(BASE_DIR, algorithm)
        algorithm_label = RL_ALGORITHM_LABELS[algorithm]
        if policy_path is not None:
            print(f"Selected {algorithm_label} policy: {policy_path}")
            override = input(
                "Press Enter to use it, or enter another policy path: "
            ).strip()
            if override:
                policy_path = resolve_policy_path(
                    BASE_DIR,
                    algorithm,
                    override,
                )
        else:
            override = input(
                f"No formal {algorithm_label} policy was found. "
                "Enter a policy path, or press Enter to cancel: "
            ).strip()
            if not override:
                return
            policy_path = resolve_policy_path(
                BASE_DIR,
                algorithm,
                override,
            )
    except Exception as exc:
        print(f"Could not select the RL policy: {exc}")
        return

    environment = os.environ.copy()
    environment["TUBE_CONTROL_MODE"] = algorithm
    environment["RL_POLICY_PATH"] = str(policy_path)
    run_script("pneumatic_sorting_test.py", environment=environment)


def train_yolo():
    arguments = []
    if (BASE_DIR / "best.pt").exists():
        replace = input(
            "best.pt already exists. Replace it after training? [y/N]: "
        ).strip().lower()
        if replace != "y":
            print("YOLO training cancelled.")
            return
        arguments.append("--overwrite-best")
    run_script("train_yolo.py", arguments=arguments)


def main_menu():
    while True:
        print("\n" + "=" * 38)
        print(" TUBE DETECTION PROJECT MENU")
        print("=" * 38)
        print("1. Generate YOLO Dataset")
        print("2. Train YOLO Detector")
        print("3. Run Rule-Based Main Simulation")
        print("4. Train DQN")
        print("5. Train PPO - Discrete or Continuous")
        print("6. Train Continuous SAC")
        print("7. Run YOLO + Trained RL Simulation")
        print("8. Run Visual Air-Jet Calibration")
        print("9. Run Visible 100-Tube Validation")
        print("10. Compare Discrete and Continuous RL Pairs")
        print("11. Exit")

        choice = input("\nSelect an option (1-11): ").strip()

        if choice == "1":
            from object_detection import run_data_collection

            run_data_collection()
        elif choice == "2":
            train_yolo()
        elif choice == "3":
            run_main_simulation()
        elif choice == "4":
            run_script("rl_training_dqn.py")
        elif choice == "5":
            train_ppo()
        elif choice == "6":
            run_script("rl_training_sac.py")
        elif choice == "7":
            run_rl_simulation()
        elif choice == "8":
            run_script("air_jet_calibration.py")
        elif choice == "9":
            run_script("run_100_tube_validation.py")
        elif choice == "10":
            run_script("compare_rl_controllers.py")
        elif choice == "11":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main_menu()
