"""Generate YOLO tube images in PyBullet."""

import argparse
import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from tube_specs import (
    TUBE_CLASSES,
    TUBE_RADII_AT_UNIT_SCALE,
    TUBE_URDF_FILENAMES,
)


BASE_DIR = Path(__file__).resolve().parent
URDF_DIR = BASE_DIR / "urdfs"
TUBE_URDF_DIR = URDF_DIR / "tubes"

TUBE_URDF_PATHS = [
    TUBE_URDF_DIR / filename
    for filename in TUBE_URDF_FILENAMES
]

SYSTEM_POSITION = np.array(
    [-2.014984, -6.364140, 9.124548],
    dtype=float,
)
SYSTEM_YAW = 1.570796
HORIZONTAL_SURFACE_Z = float(SYSTEM_POSITION[2])

TUBE_RADII = TUBE_RADII_AT_UNIT_SCALE
TUBE_BELT_CLEARANCE = 0.004

CAMERA_TARGET = np.array(
    [-2.014984, 1.670000, 9.100000],
    dtype=float,
)
CAMERA_DISTANCE = 8.0
CAMERA_YAW = 90.0
CAMERA_PITCH = -89.9
CAMERA_FOV = 60.0

# Recognition area before the first jet.
VISION_ROI_AT_960X540 = (0, 150, 230, 390)

SEGMENTATION_OBJECT_MASK = (1 << 24) - 1


@dataclass
class DatasetConfig:
    output_dir: str
    num_images: int = 1000
    width: int = 960
    height: int = 540
    min_tubes: int = 1
    max_tubes: int = 4
    empty_probability: float = 0.05
    validation_fraction: float = 0.15
    test_fraction: float = 0.10
    seed: int = 42
    gui: bool = False
    full_frame: bool = False
    scene_reset_interval: int = 20
    overwrite: bool = False


def validate_project_assets():
    required_paths = [
        URDF_DIR / "horizontal_conveyor_complete.urdf",
        URDF_DIR / "incline_conveyor.urdf",
        URDF_DIR / "conveyor_belt_shortened.urdf",
        URDF_DIR / "air_jet.urdf",
        URDF_DIR / "bin.urdf",
        URDF_DIR / "bin_funnel.urdf",
        URDF_DIR / "camera.urdf",
        URDF_DIR / "post.urdf",
        *TUBE_URDF_PATHS,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing assets:\n" + "\n".join(missing)
        )


def validate_config(config):
    if config.num_images <= 0:
        raise ValueError("num_images must be positive.")
    if config.width < 64 or config.height < 64:
        raise ValueError("Image width and height must be at least 64.")
    if not 0 <= config.empty_probability < 1:
        raise ValueError("empty_probability must be in [0, 1).")
    if not 0 <= config.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    if not 0 <= config.test_fraction < 1:
        raise ValueError("test_fraction must be in [0, 1).")
    if config.validation_fraction + config.test_fraction >= 1:
        raise ValueError(
            "validation_fraction + test_fraction must be less than 1."
        )
    if not 1 <= config.min_tubes <= config.max_tubes:
        raise ValueError(
            "Tube counts must satisfy 1 <= min_tubes <= max_tubes."
        )
    if config.scene_reset_interval <= 0:
        raise ValueError("scene_reset_interval must be positive.")


def prepare_output_directory(output_dir, overwrite):
    output_dir = output_dir.resolve()
    protected_paths = {
        Path(output_dir.anchor).resolve(),
        BASE_DIR.resolve(),
        Path.home().resolve(),
    }
    if output_dir in protected_paths:
        raise ValueError(f"Unsafe dataset output path: {output_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Dataset directory is not empty: {output_dir}. "
                "Use --overwrite only when you intend to replace it."
            )
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )
        (output_dir / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )
    return output_dir


def load_static_body(path, position, orientation):
    return p.loadURDF(
        str(path),
        basePosition=np.asarray(position, dtype=float).tolist(),
        baseOrientation=orientation,
        useFixedBase=True,
    )


def build_sorting_scene():
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    system_orientation = p.getQuaternionFromEuler(
        [0.0, 0.0, SYSTEM_YAW]
    )
    system_position = SYSTEM_POSITION.tolist()

    conveyor_ids = [
        load_static_body(
            URDF_DIR / "horizontal_conveyor_complete.urdf",
            system_position,
            system_orientation,
        ),
        load_static_body(
            URDF_DIR / "incline_conveyor.urdf",
            system_position,
            system_orientation,
        ),
        load_static_body(
            URDF_DIR / "conveyor_belt_shortened.urdf",
            system_position,
            system_orientation,
        ),
    ]
    for body_id in conveyor_ids:
        p.changeVisualShape(
            body_id,
            -1,
            rgbaColor=[0.627451, 0.627451, 0.627451, 1.0],
        )

    camera_id = load_static_body(
        URDF_DIR / "camera.urdf",
        system_position,
        system_orientation,
    )
    p.changeVisualShape(
        camera_id,
        -1,
        rgbaColor=[0.15, 0.15, 0.18, 1.0],
    )

    post_id = load_static_body(
        URDF_DIR / "post.urdf",
        system_position,
        system_orientation,
    )
    p.changeVisualShape(
        post_id,
        -1,
        rgbaColor=[0.45, 0.45, 0.48, 1.0],
    )

    main_bin_id = load_static_body(
        URDF_DIR / "bin.urdf",
        system_position,
        system_orientation,
    )
    p.changeVisualShape(
        main_bin_id,
        -1,
        rgbaColor=[0.627451, 0.627451, 0.627451, 1.0],
    )

    for jet_index in range(5):
        local_offset = [2.0 * jet_index, 0.0, 0.0]
        world_position, world_orientation = p.multiplyTransforms(
            system_position,
            system_orientation,
            local_offset,
            [0.0, 0.0, 0.0, 1.0],
        )
        air_jet_id = load_static_body(
            URDF_DIR / "air_jet.urdf",
            world_position,
            world_orientation,
        )
        p.changeVisualShape(
            air_jet_id,
            -1,
            rgbaColor=[0.627451, 0.627451, 0.627451, 1.0],
        )

        if jet_index > 0:
            funnel_id = load_static_body(
                URDF_DIR / "bin_funnel.urdf",
                world_position,
                world_orientation,
            )
            p.changeVisualShape(
                funnel_id,
                -1,
                rgbaColor=[0.627451, 0.627451, 0.627451, 1.0],
            )


def make_split_schedule(config, rng):
    num_val = int(round(config.num_images * config.validation_fraction))
    num_test = int(round(config.num_images * config.test_fraction))
    num_train = config.num_images - num_val - num_test
    schedule = (
        ["train"] * num_train
        + ["val"] * num_val
        + ["test"] * num_test
    )
    rng.shuffle(schedule)
    return schedule


def sample_tube_positions(rng, count):
    # Space tubes across the recognition area.
    lower_bound = -5.9
    upper_bound = -2.8
    minimum_gap = 0.65
    free_length = (
        upper_bound
        - lower_bound
        - minimum_gap * (count - 1)
    )
    if free_length < 0.0:
        raise ValueError(
            f"{count} tubes cannot fit in the configured recognition zone."
        )

    offsets = np.sort(rng.uniform(0.0, free_length, size=count))
    positions = (
        lower_bound
        + offsets
        + minimum_gap * np.arange(count)
    )
    rng.shuffle(positions)
    return [float(position) for position in positions]


def spawn_tubes_for_image(rng, image_index, config):
    if rng.random() < config.empty_probability:
        return []

    count = int(rng.integers(config.min_tubes, config.max_tubes + 1))
    y_positions = sample_tube_positions(rng, count)
    primary_class = image_index % len(TUBE_CLASSES)
    class_ids = [primary_class]
    class_ids.extend(
        int(rng.integers(0, len(TUBE_CLASSES)))
        for _ in range(count - 1)
    )
    rng.shuffle(class_ids)

    spawned = []
    for class_id, y_position in zip(class_ids, y_positions):
        lateral_offset = float(rng.uniform(-0.38, 0.38))
        tube_radius = TUBE_RADII[class_id] + TUBE_BELT_CLEARANCE
        position = [
            float(SYSTEM_POSITION[0] + lateral_offset),
            y_position,
            float(HORIZONTAL_SURFACE_Z + tube_radius),
        ]

        direction = float(rng.choice([-1.0, 1.0]))
        across_belt = p.getQuaternionFromEuler(
            [0.0, direction * np.pi / 2.0, 0.0]
        )
        axial_twist = p.getQuaternionFromEuler(
            [0.0, 0.0, float(rng.uniform(0.0, 2.0 * np.pi))]
        )
        _, orientation = p.multiplyTransforms(
            [0.0, 0.0, 0.0],
            across_belt,
            [0.0, 0.0, 0.0],
            axial_twist,
        )

        body_id = p.loadURDF(
            str(TUBE_URDF_PATHS[class_id]),
            basePosition=position,
            baseOrientation=orientation,
            flags=(
                p.URDF_USE_INERTIA_FROM_FILE
                | p.URDF_USE_MATERIAL_COLORS_FROM_MTL
            ),
        )
        spawned.append(
            {
                "body_id": body_id,
                "class_id": class_id,
                "class_name": TUBE_CLASSES[class_id],
            }
        )
    return spawned


def sample_camera_matrices(rng, width, height):
    target = CAMERA_TARGET.copy()
    target[0] += rng.uniform(-0.08, 0.08)
    target[1] += rng.uniform(-0.15, 0.15)
    distance = CAMERA_DISTANCE + rng.uniform(-0.12, 0.12)
    yaw = CAMERA_YAW + rng.uniform(-0.8, 0.8)
    pitch = CAMERA_PITCH + rng.uniform(-0.25, 0.25)
    fov = CAMERA_FOV + rng.uniform(-1.0, 1.0)

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        target.tolist(),
        float(distance),
        float(yaw),
        float(pitch),
        0.0,
        2,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        float(fov),
        width / height,
        0.1,
        100.0,
    )
    return view_matrix, projection_matrix


def segmentation_box(segmentation, body_id, width, height):
    object_ids = np.bitwise_and(
        segmentation.astype(np.int64),
        SEGMENTATION_OBJECT_MASK,
    )
    visible_pixels = np.where(object_ids == int(body_id))
    if visible_pixels[0].size < 20:
        return None

    y_min = int(visible_pixels[0].min())
    y_max = int(visible_pixels[0].max())
    x_min = int(visible_pixels[1].min())
    x_max = int(visible_pixels[1].max())

    box_width = x_max - x_min + 1
    box_height = y_max - y_min + 1
    if box_width < 3 or box_height < 3:
        return None

    x_center = (x_min + x_max + 1) / (2.0 * width)
    y_center = (y_min + y_max + 1) / (2.0 * height)
    width_normalized = box_width / width
    height_normalized = box_height / height
    return (
        x_center,
        y_center,
        width_normalized,
        height_normalized,
        int(visible_pixels[0].size),
    )


def render_image(rng, config, renderer):
    view_matrix, projection_matrix = sample_camera_matrices(
        rng,
        config.width,
        config.height,
    )
    light_direction = [
        float(rng.uniform(-1.0, 1.0)),
        float(rng.uniform(-1.0, 1.0)),
        float(rng.uniform(0.4, 1.0)),
    ]
    image_data = p.getCameraImage(
        config.width,
        config.height,
        view_matrix,
        projection_matrix,
        shadow=1,
        lightDirection=light_direction,
        lightColor=[
            float(rng.uniform(0.88, 1.0)),
            float(rng.uniform(0.88, 1.0)),
            float(rng.uniform(0.88, 1.0)),
        ],
        lightDistance=20.0,
        lightAmbientCoeff=float(rng.uniform(0.35, 0.55)),
        lightDiffuseCoeff=float(rng.uniform(0.55, 0.75)),
        lightSpecularCoeff=float(rng.uniform(0.05, 0.20)),
        renderer=renderer,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    )
    rgb = np.asarray(image_data[2], dtype=np.uint8).reshape(
        config.height,
        config.width,
        4,
    )[:, :, :3]
    segmentation = np.asarray(image_data[4], dtype=np.int64).reshape(
        config.height,
        config.width,
    )
    if not config.full_frame:
        base_x_min, base_y_min, base_x_max, base_y_max = (
            VISION_ROI_AT_960X540
        )
        x_min = int(round(base_x_min * config.width / 960))
        y_min = int(round(base_y_min * config.height / 540))
        x_max = int(round(base_x_max * config.width / 960))
        y_max = int(round(base_y_max * config.height / 540))
        x_min = max(0, min(x_min, config.width - 1))
        y_min = max(0, min(y_min, config.height - 1))
        x_max = max(x_min + 1, min(x_max, config.width))
        y_max = max(y_min + 1, min(y_max, config.height))
        rgb = rgb[y_min:y_max, x_min:x_max]
        segmentation = segmentation[y_min:y_max, x_min:x_max]
    return rgb, segmentation


def write_data_yaml(output_dir):
    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    yaml_lines.extend(
        f"  {class_id}: {class_name}"
        for class_id, class_name in TUBE_CLASSES.items()
    )
    (output_dir / "data.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )


def generate_dataset(config):
    validate_config(config)
    validate_project_assets()
    output_dir = prepare_output_directory(
        Path(config.output_dir),
        config.overwrite,
    )
    rng = np.random.default_rng(config.seed)
    split_schedule = make_split_schedule(config, rng)

    connection_mode = p.GUI if config.gui else p.DIRECT
    client_id = p.connect(connection_mode)
    if client_id < 0:
        raise RuntimeError("Could not connect to PyBullet.")

    renderer = (
        p.ER_BULLET_HARDWARE_OPENGL
        if config.gui
        else p.ER_TINY_RENDERER
    )
    metadata_path = output_dir / "metadata.csv"
    class_counts = {class_id: 0 for class_id in TUBE_CLASSES}
    split_counts = {"train": 0, "val": 0, "test": 0}

    try:
        if config.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        build_sorting_scene()

        with metadata_path.open(
                "w",
                newline="",
                encoding="utf-8"
        ) as metadata_file:
            writer = csv.DictWriter(
                metadata_file,
                fieldnames=[
                    "image",
                    "split",
                    "spawned_tubes",
                    "visible_labels",
                    "visible_pixels",
                ],
            )
            writer.writeheader()

            progress_interval = max(1, config.num_images // 20)
            for image_index, split in enumerate(split_schedule):
            # Reset body IDs between rendered samples.
                if (
                        image_index > 0
                        and image_index % config.scene_reset_interval == 0
                ):
                    build_sorting_scene()

                tubes = spawn_tubes_for_image(
                    rng,
                    image_index,
                    config,
                )
                rgb, segmentation = render_image(
                    rng,
                    config,
                    renderer,
                )

                labels = []
                visible_pixel_total = 0
                rendered_height, rendered_width = rgb.shape[:2]
                for tube in tubes:
                    box = segmentation_box(
                        segmentation,
                        tube["body_id"],
                        rendered_width,
                        rendered_height,
                    )
                    if box is None:
                        continue
                    x_center, y_center, box_width, box_height, pixels = box
                    labels.append(
                        f"{tube['class_id']} "
                        f"{x_center:.6f} {y_center:.6f} "
                        f"{box_width:.6f} {box_height:.6f}"
                    )
                    class_counts[tube["class_id"]] += 1
                    visible_pixel_total += pixels

                stem = f"tube_{image_index:06d}"
                image_relative = Path("images") / split / f"{stem}.jpg"
                label_relative = Path("labels") / split / f"{stem}.txt"
                image_path = output_dir / image_relative
                label_path = output_dir / label_relative

                saved = cv2.imwrite(
                    str(image_path),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                if not saved:
                    raise OSError(f"Could not save image: {image_path}")
                label_path.write_text(
                    "\n".join(labels) + ("\n" if labels else ""),
                    encoding="utf-8",
                )
                writer.writerow(
                    {
                        "image": image_relative.as_posix(),
                        "split": split,
                        "spawned_tubes": len(tubes),
                        "visible_labels": len(labels),
                        "visible_pixels": visible_pixel_total,
                    }
                )
                split_counts[split] += 1

                for tube in tubes:
                    p.removeBody(tube["body_id"])

                completed = image_index + 1
                if (
                        completed % progress_interval == 0
                        or completed == config.num_images
                ):
                    print(
                        f"Generated {completed}/{config.num_images} images"
                    )

        write_data_yaml(output_dir)
        manifest = {
            "config": asdict(config),
            "classes": TUBE_CLASSES,
            "class_label_counts": class_counts,
            "split_image_counts": split_counts,
            "annotation_source": (
                "PyBullet visible-pixel segmentation masks"
            ),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    finally:
        if p.isConnected():
            p.disconnect()

    print(f"Dataset saved to: {output_dir}")
    print(f"Class label counts: {class_counts}")
    print(f"Split image counts: {split_counts}")
    return output_dir


def run_data_collection():
    """Compatibility entry point for the original project menu."""
    dataset_name = input(
        "Dataset output folder "
        "[datasets/tubes_synthetic]: "
    ).strip()
    if not dataset_name:
        dataset_name = "datasets/tubes_synthetic"
    generate_dataset(
        DatasetConfig(
            output_dir=str(BASE_DIR / dataset_name),
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a five-class synthetic tube dataset with exact "
            "visible-pixel YOLO boxes."
        )
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "datasets" / "tubes_synthetic"),
    )
    parser.add_argument("--images", type=int, default=1000)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--min-tubes", type=int, default=1)
    parser.add_argument("--max-tubes", type=int, default=4)
    parser.add_argument("--empty-probability", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help=(
            "Save the complete camera frame instead of the upstream "
            "recognition ROI."
        ),
    )
    parser.add_argument(
        "--scene-reset-interval",
        type=int,
        default=20,
        help=(
            "Rebuild the static scene periodically to keep hardware "
            "segmentation object IDs reliable."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return DatasetConfig(
        output_dir=args.output,
        num_images=args.images,
        width=args.width,
        height=args.height,
        min_tubes=args.min_tubes,
        max_tubes=args.max_tubes,
        empty_probability=args.empty_probability,
        validation_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        gui=args.gui,
        full_frame=args.full_frame,
        scene_reset_interval=args.scene_reset_interval,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    generate_dataset(parse_args())
