import argparse
import csv
from pathlib import Path

from rl_sorting_env import (
    DISCRETE_VALVE_LEVELS,
    TUBE_CLASSES,
    ConveyorSortingEnv,
)


def parse_float_list(raw_value):
    values = tuple(
        float(value.strip())
        for value in raw_value.split(",")
        if value.strip()
    )
    if not values:
        raise ValueError("At least one value is required.")
    return values


def run_episode(
    env,
    *,
    tube_type,
    trigger_offset,
    valve_level_index,
    lateral_offset,
    axis_direction,
    seed,
):
    observation, info = env.reset(
        seed=seed,
        options={
            "tube_type": tube_type,
            "lateral_offset": lateral_offset,
            "axis_direction": axis_direction,
        },
    )
    terminated = False
    truncated = False
    command_sent = False
    episode_return = 0.0
    while not (terminated or truncated):
        action = 0
        if not command_sent and float(observation[1]) >= trigger_offset:
            action = (
                1
                + valve_level_index
            )
            command_sent = True
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
    position = info["position"]
    return {
        "tube_type": tube_type,
        "tube_class": TUBE_CLASSES[tube_type],
        "tube_mass_kg": info["tube_mass"],
        "trigger_offset": trigger_offset,
        "valve_opening": DISCRETE_VALVE_LEVELS[valve_level_index],
        "lateral_offset": lateral_offset,
        "axis_direction": axis_direction,
        "outcome": info["outcome"],
        "episode_return": episode_return,
        "final_x": position[0],
        "final_y": position[1],
        "final_z": position[2],
        "jet_distance": info["jet_distance"],
        "jet_attenuation": info["jet_attenuation"],
        "jet_impulse": info["jet_accumulated_impulse"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="runs/rl/physics_calibration.csv",
    )
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument(
        "--trigger-offsets",
        default="-0.16,-0.08,0.0,0.08,0.16",
    )
    parser.add_argument("--lateral-offsets", default="0.0")
    parser.add_argument("--axis-directions", default="1.0")
    args = parser.parse_args()

    trigger_offsets = parse_float_list(args.trigger_offsets)
    lateral_offsets = parse_float_list(args.lateral_offsets)
    axis_directions = parse_float_list(args.axis_directions)
    for lateral_offset in lateral_offsets:
        if not -0.35 <= lateral_offset <= 0.35:
            raise ValueError("Lateral offsets must be within -0.35 to 0.35.")
    for axis_direction in axis_directions:
        if axis_direction not in {-1.0, 1.0}:
            raise ValueError("Axis directions must be -1 or 1.")
    valve_indices = tuple(range(len(DISCRETE_VALVE_LEVELS)))
    rows = []
    env = ConveyorSortingEnv(action_mode="discrete")
    try:
        episode = 0
        for tube_type in TUBE_CLASSES:
            for lateral_offset in lateral_offsets:
                for axis_direction in axis_directions:
                    for trigger_offset in trigger_offsets:
                        for valve_index in valve_indices:
                            rows.append(
                                run_episode(
                                    env,
                                    tube_type=tube_type,
                                    trigger_offset=trigger_offset,
                                    valve_level_index=valve_index,
                                    lateral_offset=lateral_offset,
                                    axis_direction=axis_direction,
                                    seed=args.seed + episode,
                                )
                            )
                            episode += 1
    finally:
        env.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for tube_type, tube_class in TUBE_CLASSES.items():
        successes = [
            row
            for row in rows
            if row["tube_type"] == tube_type
            and row["outcome"] == "correct_bin"
        ]
        trial_count = (
            len(trigger_offsets)
            * len(valve_indices)
            * len(lateral_offsets)
            * len(axis_directions)
        )
        print(
            f"{tube_class}: {len(successes)}/"
            f"{trial_count} success"
        )
        for row in successes[:5]:
            print(
                f"  trigger={row['trigger_offset']:+.2f}, "
                f"valve={row['valve_opening']:.2f}"
            )
    print(f"Saved calibration: {output_path.resolve()}")


if __name__ == "__main__":
    main()
