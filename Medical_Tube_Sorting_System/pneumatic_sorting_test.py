import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import os
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / ".runtime"
RUNTIME_DIR.mkdir(exist_ok=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))

from ultralytics import YOLO
from air_jet_model import AirJetConfig, sample_air_jet
from rl_policy_runtime import (
    build_observation,
    load_policy,
    predict_valve_openings,
    resolve_policy_path,
)
from tube_specs import (
    JET_REFERENCE_IMPULSE_X_VALUES as SPEC_JET_IMPULSE_X,
    JET_REFERENCE_IMPULSE_Z_VALUES as SPEC_JET_IMPULSE_Z,
    TUBE_CLASSES,
    TUBE_RADII_AT_UNIT_SCALE,
    TUBE_SPECS,
    TUBE_URDF_FILENAMES,
)

# Set project paths and load YOLO.

CUSTOM_MODEL_PATH = Path(
    os.environ.get("TUBE_YOLO_MODEL", BASE_DIR / "best.pt")
)

def normalize_class_name(name):
    """Normalize a YOLO class name."""
    normalised = str(name).strip().lower()
    normalised = normalised.replace("_", " ").replace("-", " ")
    normalised = normalised.replace("polypropylene", "polypropene")
    return " ".join(normalised.split())


def build_model_class_map(model_names):
    """Map model classes to tube classes."""
    if isinstance(model_names, dict):
        model_items = model_names.items()
    else:
        model_items = enumerate(model_names)

    expected = {}
    for spec in TUBE_SPECS:
        names = (spec.name, *spec.yolo_aliases)
        for name in names:
            expected[normalize_class_name(name)] = spec.class_id
    class_map = {}
    for model_class_id, model_class_name in model_items:
        tube_type = expected.get(normalize_class_name(model_class_name))
        if tube_type is not None:
            class_map[int(model_class_id)] = tube_type

    if set(class_map.values()) != set(TUBE_CLASSES):
        return None
    return class_map


model = None
MODEL_CLASS_TO_TUBE_TYPE = None
if CUSTOM_MODEL_PATH.exists():
    try:
        model = YOLO(str(CUSTOM_MODEL_PATH))
        MODEL_CLASS_TO_TUBE_TYPE = build_model_class_map(model.names)
    except Exception as exc:
        print(f"WARNING: Could not load the custom YOLO model: {exc}")
        model = None

    if model is not None and MODEL_CLASS_TO_TUBE_TYPE is not None:
        print(f"Loaded custom YOLO model: {CUSTOM_MODEL_PATH}")
        print(f"YOLO class-to-jet map: {MODEL_CLASS_TO_TUBE_TYPE}")
    elif model is not None:
        print(
            "WARNING: The custom YOLO model must expose all five tube "
            "class names. Vision control is disabled."
        )
        model = None
else:
    print(
        f"WARNING: Custom YOLO weights were not found at "
        f"{CUSTOM_MODEL_PATH}. Custom YOLO control is unavailable."
    )

REQUESTED_CONTROL_MODE = os.environ.get(
    "TUBE_CONTROL_MODE",
    "auto"
).strip().lower()
VALID_CONTROL_MODES = {
    "auto",
    "yolo",
    "ground_truth",
    "dqn",
    "ppo",
    "ppo_discrete",
    "ppo_continuous",
    "sac",
}
if REQUESTED_CONTROL_MODE not in VALID_CONTROL_MODES:
    raise ValueError(
        "TUBE_CONTROL_MODE must be auto, yolo, ground_truth, "
        "dqn, ppo, ppo_discrete, ppo_continuous, or sac."
    )

RL_ALGORITHMS = {
    "dqn",
    "ppo",
    "ppo_discrete",
    "ppo_continuous",
    "sac",
}
RL_POLICY = None
RL_POLICY_PATH = None
if REQUESTED_CONTROL_MODE in RL_ALGORITHMS:
    try:
        RL_POLICY_PATH = resolve_policy_path(
            BASE_DIR,
            REQUESTED_CONTROL_MODE,
            os.environ.get("RL_POLICY_PATH"),
        )
        if RL_POLICY_PATH is not None:
            RL_POLICY = load_policy(
                REQUESTED_CONTROL_MODE,
                RL_POLICY_PATH,
            )
    except Exception as exc:
        print(f"WARNING: Could not load the RL policy: {exc}")
        RL_POLICY = None
        RL_POLICY_PATH = None

if REQUESTED_CONTROL_MODE == "auto":
    ACTIVE_CONTROL_MODE = (
        "yolo" if model is not None else "ground_truth"
    )
elif REQUESTED_CONTROL_MODE == "yolo" and model is None:
    ACTIVE_CONTROL_MODE = "disabled"
elif REQUESTED_CONTROL_MODE in RL_ALGORITHMS:
    ACTIVE_CONTROL_MODE = (
        REQUESTED_CONTROL_MODE
        if model is not None and RL_POLICY is not None
        else "disabled"
    )
else:
    ACTIVE_CONTROL_MODE = REQUESTED_CONTROL_MODE

if ACTIVE_CONTROL_MODE == "yolo":
    print("Control mode: YOLO detections drive the air jets.")
elif ACTIVE_CONTROL_MODE in RL_ALGORITHMS:
    print(
        f"Control mode: YOLO detections feed the "
        f"{ACTIVE_CONTROL_MODE.replace('_', ' ').upper()} policy."
    )
    print(f"RL policy: {RL_POLICY_PATH}")
elif ACTIVE_CONTROL_MODE == "ground_truth":
    print(
        "WARNING: DEBUG CONTROL MODE uses simulation ground truth so the "
        "mechanical sorting path can be tested. This is not YOLO control."
    )
else:
    if REQUESTED_CONTROL_MODE in RL_ALGORITHMS:
        print(
            "WARNING: RL control requires both best.pt and a trained "
            f"{REQUESTED_CONTROL_MODE.upper()} policy. Air jets are disabled."
        )
    else:
        print(
            "WARNING: Strict YOLO control was requested, but no valid custom "
            "model is available. Air jets are disabled."
        )

VISION_INFERENCE_ENABLED = (
    model is not None
    and ACTIVE_CONTROL_MODE != "ground_truth"
)
if ACTIVE_CONTROL_MODE == "ground_truth":
    print("YOLO inference is disabled during mechanical testing.")

if ACTIVE_CONTROL_MODE == "yolo":
    CONTROL_STATUS_TEXT = "YOLO DETECTION DRIVES AIR JETS"
    CONTROL_STATUS_COLOR = (0, 220, 0)
elif ACTIVE_CONTROL_MODE in RL_ALGORITHMS:
    CONTROL_STATUS_TEXT = (
        "YOLO + "
        f"{ACTIVE_CONTROL_MODE.replace('_', ' ').upper()} DRIVES AIR JETS"
    )
    CONTROL_STATUS_COLOR = (255, 180, 0)
elif ACTIVE_CONTROL_MODE == "ground_truth":
    CONTROL_STATUS_TEXT = (
        "DEBUG: GROUND-TRUTH JETS; CAMERA ONLY WITHOUT best.pt"
    )
    CONTROL_STATUS_COLOR = (0, 165, 255)
else:
    CONTROL_STATUS_TEXT = (
        f"{REQUESTED_CONTROL_MODE.upper()}: REQUIRED MODEL MISSING"
    )
    CONTROL_STATUS_COLOR = (0, 0, 255)

# Set model paths.
URDF_DIR = BASE_DIR / "urdfs"
MESH_DIR = BASE_DIR / "meshes"

CONVEYOR_2_URDF = URDF_DIR / "incline_conveyor.urdf"
CONVEYOR_3_URDF = URDF_DIR / "conveyor_belt_shortened.urdf"
CONVEYOR_4_URDF = URDF_DIR / "horizontal_conveyor_complete.urdf"
AIR_JET_URDF = URDF_DIR / "air_jet.urdf"
BIN_URDF = URDF_DIR / "bin.urdf"
BIN_FUNNEL_URDF = URDF_DIR / "bin_funnel.urdf"
CAMERA_URDF = URDF_DIR / "camera.urdf"
POST_URDF = URDF_DIR / "post.urdf"
UR5_MESH = MESH_DIR / "UR5.STL"
TUBE_URDF_PATHS = [
    URDF_DIR / "tubes" / filename
    for filename in TUBE_URDF_FILENAMES
]

# Set up the simulation.
GUI_ENABLED = os.environ.get("PYBULLET_GUI", "1") != "0"
SHOW_PYBULLET_GUI = os.environ.get("SHOW_PYBULLET_GUI", "1") != "0"
SHOW_UR5_MODEL = (
    GUI_ENABLED
    and os.environ.get("SHOW_UR5_MODEL", "1") != "0"
)
REALTIME_PACING = os.environ.get("SIMULATION_REALTIME", "1") != "0"
MAX_SIMULATION_STEPS = int(os.environ.get("MAX_SIMULATION_STEPS", "0"))
AIR_JET_CALIBRATION_MODE = (
    os.environ.get("AIR_JET_CALIBRATION", "0") == "1"
)
CALIBRATION_FAST_START = (
    os.environ.get("CALIBRATION_FAST_START", "1") == "1"
)
CALIBRATION_LATERAL_OFFSETS = tuple(
    float(value.strip())
    for value in os.environ.get(
        "CALIBRATION_LATERAL_OFFSETS",
        "-0.35,-0.175,0.0,0.175,0.35",
    ).split(",")
    if value.strip()
)
if not CALIBRATION_LATERAL_OFFSETS:
    raise ValueError("CALIBRATION_LATERAL_OFFSETS cannot be empty.")
CALIBRATION_CLASS_IDS = tuple(
    int(value.strip())
    for value in os.environ.get(
        "CALIBRATION_CLASS_IDS",
        ",".join(str(index) for index in range(len(TUBE_CLASSES))),
    ).split(",")
    if value.strip()
)
if (
    not CALIBRATION_CLASS_IDS
    or any(
        class_id not in TUBE_CLASSES
        for class_id in CALIBRATION_CLASS_IDS
    )
):
    raise ValueError("CALIBRATION_CLASS_IDS contains an invalid class.")
CALIBRATION_VALVE_OPENINGS_EXPLICIT = (
    "CALIBRATION_VALVE_OPENINGS" in os.environ
)
CALIBRATION_VALVE_OPENINGS = tuple(
    float(value.strip())
    for value in os.environ.get(
        "CALIBRATION_VALVE_OPENINGS",
        "1.0",
    ).split(",")
    if value.strip()
)
if (
    not CALIBRATION_VALVE_OPENINGS
    or any(
        opening < 0.0 or opening > 1.0
        for opening in CALIBRATION_VALVE_OPENINGS
    )
):
    raise ValueError(
        "CALIBRATION_VALVE_OPENINGS must be within 0 and 1."
    )
CALIBRATION_PAIRED_VALVE_OPENINGS = tuple(
    float(value.strip())
    for value in os.environ.get(
        "CALIBRATION_PAIRED_VALVE_OPENINGS",
        "",
    ).split(",")
    if value.strip()
)
if CALIBRATION_PAIRED_VALVE_OPENINGS and (
    len(CALIBRATION_PAIRED_VALVE_OPENINGS)
    != len(CALIBRATION_LATERAL_OFFSETS)
    or any(
        opening < 0.0 or opening > 1.0
        for opening in CALIBRATION_PAIRED_VALVE_OPENINGS
    )
):
    raise ValueError(
        "CALIBRATION_PAIRED_VALVE_OPENINGS must match the x offsets."
    )
CALIBRATION_TRIGGER_LEADS_EXPLICIT = (
    "CALIBRATION_TRIGGER_LEADS" in os.environ
)
CALIBRATION_TRIGGER_LEADS = tuple(
    float(value.strip())
    for value in os.environ.get(
        "CALIBRATION_TRIGGER_LEADS",
        "0.02",
    ).split(",")
    if value.strip()
)
if not CALIBRATION_TRIGGER_LEADS:
    raise ValueError("CALIBRATION_TRIGGER_LEADS cannot be empty.")
default_target_count = (
    len(CALIBRATION_CLASS_IDS)
    * len(CALIBRATION_LATERAL_OFFSETS)
    * (
        1
        if CALIBRATION_PAIRED_VALVE_OPENINGS
        else len(CALIBRATION_VALVE_OPENINGS)
    )
    * len(CALIBRATION_TRIGGER_LEADS)
    if AIR_JET_CALIBRATION_MODE
    else 0
)
TARGET_TUBE_COUNT = int(
    os.environ.get("TARGET_TUBE_COUNT", str(default_target_count))
)
REMOVE_COMPLETED_TUBES = (
    os.environ.get("REMOVE_COMPLETED_TUBES", "1") != "0"
)
RESULT_RETENTION_SECONDS = float(
    os.environ.get("RESULT_RETENTION_SECONDS", "15.0")
)
SPAWN_INTERVAL_STEPS = int(
    os.environ.get("SPAWN_INTERVAL_STEPS", "1000")
)
CALIBRATION_INITIAL_DELAY_STEPS = int(
    os.environ.get("CALIBRATION_INITIAL_DELAY_STEPS", "240")
)
AIR_JET_TRACE_INTERVAL_STEPS = int(
    os.environ.get("AIR_JET_TRACE_INTERVAL_STEPS", "0")
)
SIMULATION_SEED = int(os.environ.get("SIMULATION_SEED", "42"))
simulation_rng = np.random.default_rng(SIMULATION_SEED)

p.connect(p.GUI if GUI_ENABLED else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

TIME_STEP = 1.0 / 240.0

if GUI_ENABLED:
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI,
        1 if SHOW_PYBULLET_GUI else 0
    )
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

p.setGravity(0, 0, -9.81)
p.setTimeStep(TIME_STEP)

p.setPhysicsEngineParameter(
    numSolverIterations=80,
    numSubSteps=2
)

p.loadURDF("plane.urdf")

# Set the debug camera.
p.resetDebugVisualizerCamera(
    cameraDistance=30,
    cameraYaw=45,
    cameraPitch=-35,
    cameraTargetPosition=[0, -2, 8]
)

def load_urdf_if_available(path, *, label, **kwargs):
    """Load an optional URDF."""
    path = Path(path)
    if not path.exists():
        print(f"WARNING: {label} not found: {path}")
        return None
    try:
        body_id = p.loadURDF(str(path), **kwargs)
        print(f"Loaded {label}: {path.name}")
        return body_id
    except Exception as exc:
        print(f"WARNING: Could not load {label}: {exc}")
        return None


def load_visual_mesh_if_available(
        path,
        *,
        label,
        mesh_scale,
        base_position,
        base_orientation,
        rgba_color,
):
    """Load a fixed display mesh without collision."""
    path = Path(path)
    if not path.exists():
        print(f"WARNING: {label} not found: {path}")
        return None
    try:
        visual_id = p.createVisualShape(
            p.GEOM_MESH,
            fileName=str(path),
            meshScale=mesh_scale,
            rgbaColor=rgba_color,
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=base_position,
            baseOrientation=base_orientation,
        )
        print(f"Loaded {label}: {path.name}")
        return body_id
    except Exception as exc:
        print(f"WARNING: Could not load {label}: {exc}")
        return None


# Load URDF models.

SYSTEM_POSITION = [
    -2.014984,
    -6.364140,
    9.124548
]

SYSTEM_YAW = 1.570796

SYSTEM_ORIENTATION = p.getQuaternionFromEuler(
    [0, 0, SYSTEM_YAW]
)
SHORT_CONVEYOR_X_OFFSET = 0.32
SHORT_CONVEYOR_X_MIN = -0.05 + SHORT_CONVEYOR_X_OFFSET
SHORT_CONVEYOR_X_MAX = 1.74 + SHORT_CONVEYOR_X_OFFSET
SHORT_CONVEYOR_POSITION = [
    SYSTEM_POSITION[0] + SHORT_CONVEYOR_X_OFFSET,
    SYSTEM_POSITION[1],
    SYSTEM_POSITION[2],
]
INCLINE_BELT_SPEED = 1.05
TRANSFER_FORWARD_SPEED = 3.2
HORIZONTAL_FORWARD_SPEED = 1.0

INCLINE_PATH_START = np.array([
    -2.014984,
    -17.506180,
    1.817238
], dtype=float)

INCLINE_PATH_END = np.array([
    -2.014984,
    -7.548070,
    10.174038
], dtype=float)

INCLINE_PATH_VECTOR = INCLINE_PATH_END - INCLINE_PATH_START
INCLINE_PATH_LENGTH = float(np.linalg.norm(INCLINE_PATH_VECTOR))
INCLINE_TANGENT = INCLINE_PATH_VECTOR / INCLINE_PATH_LENGTH
INCLINE_NORMAL = np.array(
    [0.0, -INCLINE_TANGENT[2], INCLINE_TANGENT[1]],
    dtype=float
)

INCLINE_COLLISION_OFFSET = 0.055
HORIZONTAL_ENTRY_Y = float(SYSTEM_POSITION[1])
HORIZONTAL_SURFACE_Z = float(SYSTEM_POSITION[2])

TUBE_LATERAL_OFFSET = 0.35
NOMINAL_JET_PRESSURE_KPA = float(
    os.environ.get("NOMINAL_JET_PRESSURE_KPA", "55.0")
)
DEFAULT_AIR_JET_CONFIG = AirJetConfig()
AIR_JET_CONFIG = AirJetConfig(
    pulse_duration=float(
        os.environ.get(
            "AIR_JET_PULSE_DURATION",
            str(DEFAULT_AIR_JET_CONFIG.pulse_duration),
        )
    ),
    reference_impulse_x=float(
        os.environ.get(
            "AIR_JET_REFERENCE_IMPULSE_X",
            str(DEFAULT_AIR_JET_CONFIG.reference_impulse_x),
        )
    ),
    reference_impulse_z=float(
        os.environ.get(
            "AIR_JET_REFERENCE_IMPULSE_Z",
            str(DEFAULT_AIR_JET_CONFIG.reference_impulse_z),
        )
    ),
    distance_softening=float(
        os.environ.get(
            "AIR_JET_DISTANCE_SOFTENING",
            str(DEFAULT_AIR_JET_CONFIG.distance_softening),
        )
    ),
    nozzle_radius=float(
        os.environ.get(
            "AIR_JET_NOZZLE_RADIUS",
            str(DEFAULT_AIR_JET_CONFIG.nozzle_radius),
        )
    ),
    spread_rate=float(
        os.environ.get(
            "AIR_JET_SPREAD_RATE",
            str(DEFAULT_AIR_JET_CONFIG.spread_rate),
        )
    ),
    minimum_distance_factor=float(
        os.environ.get(
            "AIR_JET_MINIMUM_DISTANCE_FACTOR",
            str(DEFAULT_AIR_JET_CONFIG.minimum_distance_factor),
        )
    ),
    maximum_distance_factor=float(
        os.environ.get(
            "AIR_JET_MAXIMUM_DISTANCE_FACTOR",
            str(DEFAULT_AIR_JET_CONFIG.maximum_distance_factor),
        )
    ),
    valve_exponent=float(
        os.environ.get(
            "AIR_JET_VALVE_EXPONENT",
            str(DEFAULT_AIR_JET_CONFIG.valve_exponent),
        )
    ),
)


def parse_jet_parameter_list(name, default_value):
    """Read five nozzle parameter values."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        if isinstance(default_value, (tuple, list)):
            values = tuple(float(value) for value in default_value)
        else:
            values = (float(default_value),) * len(TUBE_CLASSES)
        if len(values) != len(TUBE_CLASSES):
            raise ValueError(f"{name} default must contain five values.")
        return values
    values = tuple(
        float(value.strip())
        for value in raw_value.split(",")
        if value.strip()
    )
    if len(values) != len(TUBE_CLASSES):
        raise ValueError(f"{name} must contain five values.")
    return values


JET_REFERENCE_IMPULSE_X_VALUES = parse_jet_parameter_list(
    "AIR_JET_REFERENCE_IMPULSE_X_BY_JET",
    SPEC_JET_IMPULSE_X,
)
JET_REFERENCE_IMPULSE_Z_VALUES = parse_jet_parameter_list(
    "AIR_JET_REFERENCE_IMPULSE_Z_BY_JET",
    SPEC_JET_IMPULSE_Z,
)
AIR_JET_CONFIGS = tuple(
    replace(
        AIR_JET_CONFIG,
        reference_impulse_x=JET_REFERENCE_IMPULSE_X_VALUES[index],
        reference_impulse_z=JET_REFERENCE_IMPULSE_Z_VALUES[index],
    )
    for index in range(len(TUBE_CLASSES))
)
JET_PULSE_DURATION = AIR_JET_CONFIG.pulse_duration
JET_TO_BIN_LEAD_DISTANCE = float(
    os.environ.get("JET_TO_BIN_LEAD_DISTANCE", "0.50")
)
JET_TO_BIN_LEAD_DISTANCES = parse_jet_parameter_list(
    "JET_TO_BIN_LEAD_DISTANCE_BY_JET",
    JET_TO_BIN_LEAD_DISTANCE,
)
RULE_DISTANCE_COMPENSATION = (
    os.environ.get("RULE_DISTANCE_COMPENSATION", "0") == "1"
)
RULE_TARGET_ATTENUATION_VALUES = parse_jet_parameter_list(
    "RULE_TARGET_ATTENUATION_BY_JET",
    0.8,
)
RULE_VALVE_X_ANCHORS = np.array(
    [-0.35, -0.175, 0.0, 0.175, 0.35],
    dtype=float,
)
RULE_VALVE_OPENING_PROFILES = (
    (0.80, 0.90, 1.00, 1.00, 1.00),
    (0.80, 0.90, 1.00, 1.00, 1.00),
    (0.70, 0.80, 0.90, 0.90, 1.00),
    (0.70, 0.80, 0.90, 0.90, 1.00),
    (0.65, 0.80, 0.90, 1.00, 1.00),
)
RULE_TRIGGER_BY_JET_EXPLICIT = (
    "RULE_TRIGGER_LEAD_DISTANCE_BY_JET" in os.environ
)
RULE_TRIGGER_LEAD_DISTANCES = parse_jet_parameter_list(
    "RULE_TRIGGER_LEAD_DISTANCE_BY_JET",
    (0.02, 0.02, 0.04, 0.04, 0.04),
)
RULE_TRIGGER_LEAD_PROFILES = (
    (0.02, 0.02, 0.02, 0.02, 0.02),
    (0.02, 0.02, 0.02, 0.02, 0.02),
    (0.04, 0.04, 0.04, 0.04, 0.04),
    (0.04, 0.04, 0.04, 0.04, 0.04),
    (0.04, 0.04, 0.04, 0.04, 0.04),
)
RULE_TRIGGER_HALF_WIDTH = float(
    os.environ.get("RULE_TRIGGER_HALF_WIDTH", "0.03")
)
RL_TRIGGER_HALF_WIDTH = float(
    os.environ.get("RL_TRIGGER_HALF_WIDTH", "0.25")
)
SORTING_RESULT_TIMEOUT_SECONDS = 3.0
BIN_CAPTURE_DEPTH = 0.25
BIN_DROP_BELOW_BELT = 0.15
BIN_CONFIRMATION_SECONDS = 0.05
BIN_CONFIRMATION_STEPS = max(
    1,
    int(round(BIN_CONFIRMATION_SECONDS / TIME_STEP))
)
BIN_FRICTION = 0.20
RL_CONTROL_INTERVAL_STEPS = 8
RL_VALVE_DEADBAND_BY_JET = tuple(
    float(value.strip())
    for value in os.environ.get(
        "RL_VALVE_DEADBAND_BY_JET",
        "0.60,0.60,0.60,0.60,0.60",
    ).split(",")
)
if len(RL_VALVE_DEADBAND_BY_JET) != len(TUBE_CLASSES):
    raise ValueError(
        "RL_VALVE_DEADBAND_BY_JET must contain five values."
    )
RL_OBSERVATION_LEAD_SECONDS_BY_JET = tuple(
    float(value.strip())
    for value in os.environ.get(
        "RL_OBSERVATION_LEAD_SECONDS_BY_JET",
        "0.0,0.0,0.02,0.02,0.0",
    ).split(",")
)
if len(RL_OBSERVATION_LEAD_SECONDS_BY_JET) != len(TUBE_CLASSES):
    raise ValueError(
        "RL_OBSERVATION_LEAD_SECONDS_BY_JET must contain five values."
    )


def calculate_rule_valve_opening(position, jet_index):
    """Calculate valve opening from the tube x position."""
    lateral_offset = float(position[0] - SYSTEM_POSITION[0])
    opening = float(
        np.interp(
            lateral_offset,
            RULE_VALVE_X_ANCHORS,
            RULE_VALVE_OPENING_PROFILES[jet_index],
        )
    )
    if RULE_DISTANCE_COMPENSATION:
        unit_sample = sample_air_jet(
            np.asarray(position, dtype=float),
            np.asarray(jet_nozzle_positions[jet_index], dtype=float),
            1.0,
            jet_reference_distance,
            AIR_JET_CONFIGS[jet_index],
        )
        target_attenuation = RULE_TARGET_ATTENUATION_VALUES[jet_index]
        exponent = AIR_JET_CONFIGS[jet_index].valve_exponent
        opening *= (
            target_attenuation / max(unit_sample.attenuation, 1e-6)
        ) ** (1.0 / exponent)
    return float(np.clip(opening, 0.0, 1.0))


def calculate_rule_trigger_lead(position, jet_index):
    """Calculate trigger lead from the tube x position."""
    if RULE_TRIGGER_BY_JET_EXPLICIT:
        return RULE_TRIGGER_LEAD_DISTANCES[jet_index]
    lateral_offset = float(position[0] - SYSTEM_POSITION[0])
    return float(
        np.interp(
            lateral_offset,
            RULE_VALVE_X_ANCHORS,
            RULE_TRIGGER_LEAD_PROFILES[jet_index],
        )
    )


def update_air_jet_force(tube, position):
    """Start and update one pneumatic pulse."""
    if not tube['jet_commanded']:
        return

    jet_index = tube['fired_jet_index']
    if jet_index is None:
        return

    if not tube['fired']:
        tube['fired'] = True
        tube['transport_phase'] = 'ballistic'
        tube['jet_step'] = step_count
        tube['jet_position'] = tuple(float(value) for value in position)
        tube['post_jet_min_x'] = float(position[0])
        tube['post_jet_max_x'] = float(position[0])
        tube['post_jet_min_z'] = float(position[2])
        tube['post_jet_max_z'] = float(position[2])
    elapsed = (step_count - tube['jet_step']) * TIME_STEP
    if elapsed >= JET_PULSE_DURATION:
        if not tube['jet_pulse_complete']:
            end_position, _ = p.getBasePositionAndOrientation(
                tube['base_id']
            )
            end_velocity, _ = p.getBaseVelocity(tube['base_id'])
            tube['jet_end_position'] = tuple(
                float(value) for value in end_position
            )
            tube['jet_end_velocity'] = tuple(
                float(value) for value in end_velocity
            )
            tube['jet_pulse_complete'] = True
        return

    sample = sample_air_jet(
        np.asarray(position, dtype=float),
        np.asarray(jet_nozzle_positions[jet_index], dtype=float),
        tube['jet_intensity'],
        jet_reference_distance,
        AIR_JET_CONFIGS[jet_index],
    )
    p.applyExternalForce(
        tube['base_id'],
        -1,
        sample.force.tolist(),
        list(position),
        p.WORLD_FRAME,
    )

    force_magnitude = float(np.linalg.norm(sample.force))
    tube['jet_peak_force'] = max(
        tube['jet_peak_force'],
        force_magnitude,
    )
    tube['jet_accumulated_impulse'] += (
        force_magnitude * TIME_STEP
    )
    if tube['jet_distance'] is None:
        tube['jet_distance'] = sample.distance
        tube['jet_axial_distance'] = sample.axial_distance
        tube['jet_radial_offset'] = sample.radial_offset
        tube['jet_plume_radius'] = sample.plume_radius
        tube['jet_distance_factor'] = sample.distance_factor
        tube['jet_radial_factor'] = sample.radial_factor
        tube['jet_attenuation'] = sample.attenuation
        start_velocity, _ = p.getBaseVelocity(tube['base_id'])
        tube['jet_start_velocity'] = tuple(
            float(value) for value in start_velocity
        )
        print(
            f"JET_EVENT tube={tube['sequence_id']}, "
            f"jet={jet_index + 1}, "
            f"control={ACTIVE_CONTROL_MODE}, "
            f"distance={sample.distance:.3f}, "
            f"radial={sample.radial_offset:.3f}, "
            f"attenuation={sample.attenuation:.3f}, "
            f"force={force_magnitude:.3f}"
        )
        p.addUserDebugLine(
            jet_nozzle_positions[jet_index],
            list(position),
            lineColorRGB=[1, 0, 0],
            lineWidth=2,
            lifeTime=JET_PULSE_DURATION,
        )


conveyor_id_4 = load_urdf_if_available(
    CONVEYOR_4_URDF,
    label="top conveyor",
    basePosition=SYSTEM_POSITION,
    baseOrientation=SYSTEM_ORIENTATION,
    useFixedBase=True,
)

if conveyor_id_4 is not None:
    p.changeVisualShape(
        conveyor_id_4,
        -1,
        rgbaColor=[0.627451, 0.627451, 0.627451, 1]
    )


# Incline conveyor

conveyor_id_2 = load_urdf_if_available(
    CONVEYOR_2_URDF,
    label="inclined conveyor",
    basePosition=SYSTEM_POSITION,
    baseOrientation=SYSTEM_ORIENTATION,
    useFixedBase=True,
)

if conveyor_id_2 is not None:
    p.changeVisualShape(
        conveyor_id_2,
        -1,
        rgbaColor=[0.35, 0.75, 0.35, 1]
    )


# Conveyor transforms

if conveyor_id_4 is not None:
    horizontal_pos, horizontal_ori = (
        p.getBasePositionAndOrientation(conveyor_id_4)
    )

    print("HORIZONTAL POSITION:", horizontal_pos)
    print("HORIZONTAL ORIENTATION:", horizontal_ori)


if conveyor_id_2 is not None:
    incline_pos, incline_ori = (
        p.getBasePositionAndOrientation(conveyor_id_2)
    )

    print("INCLINE POSITION:", incline_pos)
    print("INCLINE ORIENTATION:", incline_ori)

camera_model_id = load_urdf_if_available(
    CAMERA_URDF,
    label="camera model",
    basePosition=SYSTEM_POSITION,
    baseOrientation=SYSTEM_ORIENTATION,
    useFixedBase=True,
)

if camera_model_id is not None:
    p.changeVisualShape(
        camera_model_id,
        -1,
        rgbaColor=[0.15, 0.15, 0.18, 1]
    )

post_model_id = load_urdf_if_available(
    POST_URDF,
    label="support post",
    basePosition=SYSTEM_POSITION,
    baseOrientation=SYSTEM_ORIENTATION,
    useFixedBase=True,
)

if post_model_id is not None:
    p.changeVisualShape(
        post_model_id,
        -1,
        rgbaColor=[0.45, 0.45, 0.48, 1]
    )

conveyor_id_3 = load_urdf_if_available(
    CONVEYOR_3_URDF,
    label="short conveyor",
    basePosition=SHORT_CONVEYOR_POSITION,
    baseOrientation=SYSTEM_ORIENTATION,
    useFixedBase=True,
)

if conveyor_id_3 is not None:
    p.changeVisualShape(conveyor_id_3, -1, rgbaColor=[0.627451, 0.627451, 0.627451, 1])

cyl_col_id = p.createCollisionShape(
    p.GEOM_CYLINDER,
    radius=0.8,
    height=2,
)
cyl_vis_id = p.createVisualShape(
    p.GEOM_CYLINDER,
    radius=0.8,
    length=2,
    rgbaColor=[0.627451, 0.627451, 0.627451, 1],
)
p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=cyl_col_id,
    baseVisualShapeIndex=cyl_vis_id,
    basePosition=[5.05, 11.37, 1.0],
)

ur5_model_id = None
if SHOW_UR5_MODEL:
    ur5_base_position = np.array([5.05, 11.37, 2.0], dtype=float)
    ur5_mesh_scale = 0.01
    ur5_base_anchor = np.array(
        [-262.1099, -325.58545, 91.235344],
        dtype=float,
    ) * ur5_mesh_scale
    ur5_orientation = p.getQuaternionFromEuler(
        [np.pi / 2.0, 0.0, np.deg2rad(205.0)]
    )
    rotated_anchor = np.array(
        p.rotateVector(ur5_orientation, ur5_base_anchor.tolist()),
        dtype=float,
    )
    ur5_mesh_position = ur5_base_position - rotated_anchor
    ur5_model_id = load_visual_mesh_if_available(
        UR5_MESH,
        label="UR5 robot model",
        mesh_scale=[ur5_mesh_scale] * 3,
        base_position=ur5_mesh_position.tolist(),
        base_orientation=ur5_orientation,
        rgba_color=[0.72, 0.74, 0.78, 1.0],
    )

box_stand_half_extents = [2.1, 1.35, 1]
box_stand_col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_stand_half_extents)
box_stand_vis_id = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=box_stand_half_extents,
    rgbaColor=[0.627451, 0.627451, 0.627451, 1],
)
p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=box_stand_col_id,
    baseVisualShapeIndex=box_stand_vis_id,
    basePosition=[0, 9, 1],
)

BIN_SPACING = 200.0 * 0.01
bin_entry_x = -0.39
bin_instances = [None] * len(TUBE_CLASSES)
bin_aabbs = [None] * len(TUBE_CLASSES)

bin_instance = load_urdf_if_available(
    BIN_URDF,
    label="main bin",
    basePosition=SYSTEM_POSITION,
    baseOrientation=SYSTEM_ORIENTATION,
    useFixedBase=True,
)

if bin_instance is not None:
    bin_instances[0] = bin_instance
    p.changeVisualShape(
        bin_instance,
        -1,
        rgbaColor=[0.627451, 0.627451, 0.627451, 1]
    )
    p.changeDynamics(
        bin_instance,
        -1,
        lateralFriction=BIN_FRICTION,
        restitution=0.0
    )
    bin_aabb = p.getAABB(bin_instance, -1)
    bin_aabbs[0] = bin_aabb
    bin_entry_x = bin_aabb[0][0] - 0.02

JET_SPACING = BIN_SPACING
jet_positions = []
jet_nozzle_positions = []
jet_trigger_x_min = -3.64

JET_START_LOCAL = [
    0.0,
    0.0,
    0.0
]
for i in range(5):
    jet_local_position = [
        (
            JET_START_LOCAL[0]
            + JET_SPACING * i
            - JET_TO_BIN_LEAD_DISTANCES[i]
        ),
        JET_START_LOCAL[1],
        JET_START_LOCAL[2]
    ]

    jet_world_position, jet_world_orientation = p.multiplyTransforms(
        SYSTEM_POSITION,
        SYSTEM_ORIENTATION,
        jet_local_position,
        [0, 0, 0, 1]
    )

    air_jet_instance = load_urdf_if_available(
        AIR_JET_URDF,
        label=f"air jet {i + 1}",
        basePosition=jet_world_position,
        baseOrientation=jet_world_orientation,
        useFixedBase=True,
    )

    if air_jet_instance is not None:
        p.changeVisualShape(
            air_jet_instance,
            -1,
            rgbaColor=[0.627451, 0.627451, 0.627451, 1]
        )
        jet_aabb = p.getAABB(air_jet_instance, -1)
        jet_center = [
            (jet_aabb[0][axis] + jet_aabb[1][axis]) * 0.5
            for axis in range(3)
        ]
        jet_positions.append(jet_center[1])
        jet_nozzle_positions.append([
            jet_aabb[1][0],
            jet_center[1],
            jet_center[2],
        ])

        if i == 0:
            jet_x_source = jet_center[0]
            jet_z_level = jet_center[2]
            jet_trigger_x_min = jet_aabb[1][0] + 0.02
    else:
        jet_positions.append(jet_world_position[1])
        jet_nozzle_positions.append([
            jet_world_position[0],
            jet_world_position[1],
            jet_world_position[2],
        ])

    if i == 0:
        continue

    bin_local_offset = [
        BIN_SPACING * i,
        0.0,
        0.0
    ]

    bin_world_position, bin_world_orientation = p.multiplyTransforms(
        SYSTEM_POSITION,
        SYSTEM_ORIENTATION,
        bin_local_offset,
        [0, 0, 0, 1]
    )

    binf_instance = load_urdf_if_available(
        BIN_FUNNEL_URDF,
        label=f"bin funnel {i + 1}",
        basePosition=bin_world_position,
        baseOrientation=bin_world_orientation,
        useFixedBase=True,
    )

    if binf_instance is not None:
        bin_instances[i] = binf_instance
        p.changeVisualShape(
            binf_instance,
            -1,
            rgbaColor=[0.627451, 0.627451, 0.627451, 1]
        )
        p.changeDynamics(
            binf_instance,
            -1,
            lateralFriction=BIN_FRICTION,
            restitution=0.0
        )
        bin_aabbs[i] = p.getAABB(binf_instance, -1)

jet_reference_distance = abs(
    SYSTEM_POSITION[0] - jet_nozzle_positions[0][0]
)

CAM_WIDTH, CAM_HEIGHT = 960, 540
CAM_TARGET = [
    -2.014984,
    1.670000,
    9.100000
]
# Set the recognition area before jet 1.
VISION_ROI = (0, 150, 230, 390)
VISION_ROI_X_MIN, VISION_ROI_Y_MIN, VISION_ROI_X_MAX, VISION_ROI_Y_MAX = (
    VISION_ROI
)
VISION_ROI_WIDTH = VISION_ROI_X_MAX - VISION_ROI_X_MIN
VISION_ROI_HEIGHT = VISION_ROI_Y_MAX - VISION_ROI_Y_MIN
DETECTION_DISPLAY_SCALE = 2
# Set the camera and inference rates.
VISION_INTERVAL_STEPS = int(
    os.environ.get("VISION_INTERVAL_STEPS", "30")
)
YOLO_CONFIDENCE = float(
    os.environ.get("YOLO_CONFIDENCE", "0.15")
)
YOLO_IMAGE_SIZE = int(
    os.environ.get("YOLO_IMAGE_SIZE", "320")
)
MIN_DETECTION_HITS = int(
    os.environ.get("MIN_DETECTION_HITS", "2")
)
SHOW_DETECTION_WINDOW = (
    GUI_ENABLED
    and os.environ.get("SHOW_DETECTION_WINDOW", "1") != "0"
)
VISION_SYNCHRONOUS = (
    VISION_INFERENCE_ENABLED
    and os.environ.get(
        "VISION_SYNCHRONOUS",
        "1" if not GUI_ENABLED else "0",
    ) != "0"
)
VISION_TRACE_ASSOCIATION = (
    os.environ.get("VISION_TRACE_ASSOCIATION", "0") == "1"
)

view_matrix = p.computeViewMatrixFromYawPitchRoll(
    CAM_TARGET,
    8.0,
    90,
    -89.9,
    0,
    2
)
proj_matrix = p.computeProjectionMatrixFOV(
    60,
    CAM_WIDTH / CAM_HEIGHT,
    0.1,
    100.0
)
view_transform = np.asarray(view_matrix).reshape(4, 4, order="F")
projection_transform = np.asarray(proj_matrix).reshape(4, 4, order="F")
inverse_camera_transform = np.linalg.inv(
    projection_transform @ view_transform
)


def project_world_to_image(position):
    # Project one world point.
    world_point = np.array(
        [position[0], position[1], position[2], 1.0],
        dtype=float
    )
    clip_point = projection_transform @ view_transform @ world_point

    if clip_point[3] <= 0.0:
        return None

    image_point = clip_point[:3] / clip_point[3]
    if np.any(np.abs(image_point[:2]) > 1.0):
        return None

    full_image_position = np.array([
        (image_point[0] + 1.0) * 0.5 * CAM_WIDTH,
        (1.0 - image_point[1]) * 0.5 * CAM_HEIGHT
    ])
    roi_image_position = full_image_position - np.array(
        [VISION_ROI_X_MIN, VISION_ROI_Y_MIN],
        dtype=float
    )
    if (
            roi_image_position[0] < 0.0
            or roi_image_position[0] >= VISION_ROI_WIDTH
            or roi_image_position[1] < 0.0
            or roi_image_position[1] >= VISION_ROI_HEIGHT
    ):
        return None
    return roi_image_position


def image_to_world_on_plane(image_position, plane_z):
    """Project one image point onto the conveyor plane."""
    full_x = float(image_position[0]) + VISION_ROI_X_MIN
    full_y = float(image_position[1]) + VISION_ROI_Y_MIN
    ndc_x = 2.0 * full_x / CAM_WIDTH - 1.0
    ndc_y = 1.0 - 2.0 * full_y / CAM_HEIGHT

    world_points = []
    for ndc_z in (-1.0, 1.0):
        world_h = inverse_camera_transform @ np.array(
            [ndc_x, ndc_y, ndc_z, 1.0],
            dtype=float,
        )
        if abs(world_h[3]) < 1e-9:
            return None
        world_points.append(world_h[:3] / world_h[3])

    ray_start, ray_end = world_points
    ray = ray_end - ray_start
    if abs(ray[2]) < 1e-9:
        return None
    distance = (float(plane_z) - ray_start[2]) / ray[2]
    if distance < 0.0:
        return None
    return ray_start + distance * ray


def add_camera_scene_guide():
    """Show the camera and its detection area."""
    if not GUI_ENABLED or not p.isConnected():
        return

    if camera_model_id is not None:
        camera_aabb = p.getAABB(camera_model_id, -1)
        p.addUserDebugText(
            "CAMERA",
            [
                (camera_aabb[0][0] + camera_aabb[1][0]) * 0.5,
                (camera_aabb[0][1] + camera_aabb[1][1]) * 0.5,
                camera_aabb[1][2] + 0.4,
            ],
            textColorRGB=[1.0, 0.85, 0.0],
            textSize=1.2,
            lifeTime=0,
        )

    image_corners = (
        (0.0, 0.0),
        (float(VISION_ROI_WIDTH), 0.0),
        (float(VISION_ROI_WIDTH), float(VISION_ROI_HEIGHT)),
        (0.0, float(VISION_ROI_HEIGHT)),
    )
    guide_height = HORIZONTAL_SURFACE_Z + 0.04
    world_corners = [
        image_to_world_on_plane(corner, guide_height)
        for corner in image_corners
    ]
    if any(corner is None for corner in world_corners):
        return

    for corner_index in range(len(world_corners)):
        p.addUserDebugLine(
            world_corners[corner_index].tolist(),
            world_corners[(corner_index + 1) % len(world_corners)].tolist(),
            lineColorRGB=[1.0, 0.85, 0.0],
            lineWidth=2,
            lifeTime=0,
        )

    guide_center = np.mean(world_corners, axis=0)
    p.addUserDebugText(
        "YOLO Detection Zone",
        (guide_center + np.array([0.0, 0.0, 0.18])).tolist(),
        textColorRGB=[1.0, 0.85, 0.0],
        textSize=1.0,
        lifeTime=0,
    )


add_camera_scene_guide()


def set_tracking_measurement(tube, position, capture_step, source):
    """Store one upstream camera measurement."""
    if tube['tracking_step'] is not None:
        return
    tube['tracking_source'] = source
    tube['tracking_step'] = int(capture_step)
    tube['tracking_x'] = float(position[0])
    tube['tracking_y'] = float(position[1])


def estimate_tracked_position(tube):
    """Estimate tube position after camera detection."""
    if tube['tracking_step'] is None:
        return None
    elapsed_steps = max(0, step_count - tube['tracking_step'])
    estimated_y = (
        tube['tracking_y']
        + HORIZONTAL_FORWARD_SPEED * elapsed_steps * TIME_STEP
    )
    return np.array(
        [
            tube['tracking_x'],
            estimated_y,
            HORIZONTAL_SURFACE_Z + tube['tube_radius'],
        ],
        dtype=float,
    )


def capture_vision_snapshot():
    """Capture visible tube image positions."""
    snapshot = []
    for tube in tubes:
        if (
            tube['fired']
            or tube['transport_phase'] != 'horizontal'
        ):
            continue

        position, _ = p.getBasePositionAndOrientation(tube['base_id'])
        image_position = project_world_to_image(position)
        if image_position is not None:
            snapshot.append(
                (
                    tube['sequence_id'],
                    image_position,
                    tuple(float(value) for value in position),
                    step_count,
                )
            )
    return snapshot


def update_vision_labels(result, snapshot):
    """Match detections to captured tube positions."""
    if result.boxes is None or len(result.boxes) == 0:
        return

    xyxy = result.boxes.xyxy.cpu().numpy()
    model_class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    active_tubes = {
        tube['sequence_id']: tube
        for tube in tubes
        if not tube['fired']
    }
    visible_tubes = [
        (
            active_tubes[sequence_id],
            image_position,
            world_position,
            capture_step,
        )
        for (
            sequence_id,
            image_position,
            world_position,
            capture_step,
        ) in snapshot
        if sequence_id in active_tubes
    ]

    assigned_bodies = set()
    detection_order = np.argsort(-confidences)
    for detection_index in detection_order:
        model_class_id = int(model_class_ids[detection_index])
        class_id = MODEL_CLASS_TO_TUBE_TYPE.get(model_class_id)
        if class_id is None:
            continue

        x_min, y_min, x_max, y_max = xyxy[detection_index]
        box_center = np.array([
            (x_min + x_max) * 0.5,
            (y_min + y_max) * 0.5
        ])
        margin = max(6.0, 0.1 * max(x_max - x_min, y_max - y_min))

        candidates = []
        for (
            tube,
            image_position,
            world_position,
            capture_step,
        ) in visible_tubes:
            if tube['base_id'] in assigned_bodies:
                continue
            if (
                    x_min - margin <= image_position[0] <= x_max + margin
                    and y_min - margin <= image_position[1] <= y_max + margin
            ):
                distance = float(np.linalg.norm(image_position - box_center))
                candidates.append(
                    (
                        distance,
                        tube,
                        world_position,
                        capture_step,
                    )
                )

        if not candidates:
            continue

        (
            _,
            matched_tube,
            _matched_world_position,
            measurement_step,
        ) = min(candidates, key=lambda item: item[0])
        matched_tube['vision_scores'][class_id] += float(
            confidences[detection_index]
        )
        matched_tube['detected_type'] = int(
            np.argmax(matched_tube['vision_scores'])
        )
        matched_tube['detection_confidence'] = float(
            confidences[detection_index]
        )
        matched_tube['detection_hits'] += 1
        if matched_tube['first_detection_step'] is None:
            matched_tube['first_detection_step'] = measurement_step
        matched_tube['last_detection_step'] = measurement_step
        if matched_tube['detection_hits'] >= MIN_DETECTION_HITS:
            measurement_position = image_to_world_on_plane(
                box_center,
                (
                    HORIZONTAL_SURFACE_Z
                    + TUBE_BELT_CLEARANCES[class_id]
                ),
            )
            if measurement_position is not None:
                tracking_was_empty = (
                    matched_tube['tracking_step'] is None
                )
                set_tracking_measurement(
                    matched_tube,
                    measurement_position,
                    measurement_step,
                    "yolo_bbox",
                )
                if VISION_TRACE_ASSOCIATION and tracking_was_empty:
                    print(
                        f"VISION_ASSOCIATION tube="
                        f"{matched_tube['sequence_id']}, "
                        f"captured=("
                        f"{_matched_world_position[0]:.3f}, "
                        f"{_matched_world_position[1]:.3f}), "
                        f"measured=("
                        f"{measurement_position[0]:.3f}, "
                        f"{measurement_position[1]:.3f})"
                    )
        if (
                matched_tube['detection_hits'] >= MIN_DETECTION_HITS
                and matched_tube['reported_detected_type']
                != matched_tube['detected_type']
        ):
            print(
                f"DETECTION_EVENT tube={matched_tube['sequence_id']}, "
                f"body={matched_tube['base_id']}, "
                f"evaluation_class={matched_tube['class']}, "
                f"detected_class="
                f"{TUBE_CLASSES[matched_tube['detected_type']]}, "
                f"confidence={matched_tube['detection_confidence']:.3f}, "
                f"hits={matched_tube['detection_hits']}"
            )
            matched_tube['reported_detected_type'] = (
                matched_tube['detected_type']
            )
        assigned_bodies.add(matched_tube['base_id'])


tubes = []
step_count = 0
tube_sequence_counter = 0
vision_executor = (
    ThreadPoolExecutor(max_workers=1, thread_name_prefix="tube-yolo")
    if VISION_INFERENCE_ENABLED and not VISION_SYNCHRONOUS
    else None
)
vision_future = None
vision_snapshot = None

RUN_ID = (
    f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    f"_seed{SIMULATION_SEED}"
)
DEFAULT_RESULTS_PATH = (
    BASE_DIR / "runs" / "simulation_results" / f"{RUN_ID}.csv"
)
RESULTS_CSV_PATH = Path(
    os.environ.get("SIMULATION_RESULTS_CSV", DEFAULT_RESULTS_PATH)
)
if not RESULTS_CSV_PATH.is_absolute():
    RESULTS_CSV_PATH = BASE_DIR / RESULTS_CSV_PATH
RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

RESULT_FIELDS = [
    "run_id",
    "tube_sequence",
    "body_id",
    "seed",
    "requested_control_mode",
    "control_mode",
    "rl_policy_path",
    "rl_observation_lead_s",
    "rl_valve_deadband",
    "calibration_mode",
    "nominal_pressure_kpa",
    "rule_distance_compensation",
    "rule_target_attenuation",
    "tube_mass_kg",
    "spawn_lateral_offset_m",
    "tube_axis_direction",
    "reference_impulse_x_ns",
    "reference_impulse_z_ns",
    "pulse_duration_s",
    "jet_to_bin_lead_m",
    "nozzle_radius_m",
    "spread_rate",
    "spawn_step",
    "spawn_time_s",
    "evaluation_class_id",
    "evaluation_class",
    "detected_class_id",
    "detected_class",
    "confidence",
    "detection_hits",
    "first_detection_step",
    "last_detection_step",
    "tracking_source",
    "tracking_step",
    "tracking_x",
    "tracking_y",
    "selected_jet",
    "policy_action",
    "jet_intensity",
    "trigger_lead_m",
    "jet_command_step",
    "jet_command_x",
    "jet_command_y",
    "jet_command_z",
    "estimated_command_x",
    "estimated_command_y",
    "tracking_error_x",
    "tracking_error_y",
    "jet_step",
    "jet_x",
    "jet_y",
    "jet_z",
    "jet_distance_m",
    "jet_axial_distance_m",
    "jet_radial_offset_m",
    "jet_plume_radius_m",
    "jet_distance_factor",
    "jet_radial_factor",
    "jet_attenuation",
    "jet_peak_force_n",
    "jet_impulse_ns",
    "jet_start_velocity_x_mps",
    "jet_start_velocity_y_mps",
    "jet_start_velocity_z_mps",
    "jet_end_x",
    "jet_end_y",
    "jet_end_z",
    "jet_end_velocity_x_mps",
    "jet_end_velocity_y_mps",
    "jet_end_velocity_z_mps",
    "post_jet_min_x",
    "post_jet_max_x",
    "post_jet_min_z",
    "post_jet_max_z",
    "selected_bin_contact_steps",
    "selected_bin_contact_points",
    "first_selected_bin_contact_step",
    "first_selected_bin_contact_x",
    "first_selected_bin_contact_y",
    "first_selected_bin_contact_z",
    "crossed_bin_entry",
    "first_bin_entry_step",
    "first_bin_entry_x",
    "first_bin_entry_y",
    "first_bin_entry_z",
    "inside_selected_bin_xy",
    "below_selected_bin_rim",
    "final_bin",
    "outcome",
    "failure_stage",
    "success",
    "final_step",
    "final_x",
    "final_y",
    "final_z",
]
results_file = RESULTS_CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig"
)
results_writer = csv.DictWriter(results_file, fieldnames=RESULT_FIELDS)
results_writer.writeheader()
results_file.flush()
result_counts = {}
print(f"Simulation result log: {RESULTS_CSV_PATH}")
trajectory_file = None
trajectory_writer = None
TRAJECTORY_CSV_PATH = RESULTS_CSV_PATH.with_name(
    f"{RESULTS_CSV_PATH.stem}_trajectory.csv"
)
if AIR_JET_TRACE_INTERVAL_STEPS > 0:
    trajectory_fields = [
        "run_id",
        "tube_sequence",
        "step",
        "time_s",
        "evaluation_class",
        "selected_jet",
        "x",
        "y",
        "z",
        "velocity_x_mps",
        "velocity_y_mps",
        "velocity_z_mps",
        "selected_bin_contact_points",
        "crossed_bin_entry",
        "inside_selected_bin_xy",
        "below_selected_bin_rim",
        "bin_candidate_steps",
    ]
    trajectory_file = TRAJECTORY_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    )
    trajectory_writer = csv.DictWriter(
        trajectory_file,
        fieldnames=trajectory_fields,
    )
    trajectory_writer.writeheader()
    trajectory_file.flush()
    print(f"Air-jet trajectory log: {TRAJECTORY_CSV_PATH}")
if AIR_JET_CALIBRATION_MODE:
    print(
        "Air-jet calibration: "
        f"{TARGET_TUBE_COUNT} trials, "
        f"classes={CALIBRATION_CLASS_IDS}, "
        f"offsets={CALIBRATION_LATERAL_OFFSETS}, "
        f"openings={CALIBRATION_VALVE_OPENINGS}, "
        f"trigger_leads={CALIBRATION_TRIGGER_LEADS}, "
        f"nominal_pressure={NOMINAL_JET_PRESSURE_KPA:.1f} kPa, "
        f"reference_impulse_x={JET_REFERENCE_IMPULSE_X_VALUES}, "
        f"reference_impulse_z={JET_REFERENCE_IMPULSE_Z_VALUES} N s"
    )


def determine_failure_stage(tube, outcome):
    """Classify the physical result stage."""
    if outcome == "correct_bin":
        return "collected"
    if outcome == "wrong_bin":
        return "wrong_bin"
    if outcome == "unfinished":
        return "unfinished"
    if not tube['fired']:
        return "no_jet_effect"
    if tube['below_selected_bin_rim']:
        return "below_rim_not_confirmed"
    if tube['inside_selected_bin_xy']:
        return "inside_xy_above_rim"
    if tube['selected_bin_contact_steps'] > 0:
        return "funnel_collision"
    if tube['crossed_bin_entry']:
        return "crossed_entry_outside_funnel"
    return "insufficient_lateral_travel"


def record_trajectory_sample(tube, position):
    """Write one post-jet trajectory sample."""
    if (
        trajectory_writer is None
        or not tube['fired']
        or step_count % AIR_JET_TRACE_INTERVAL_STEPS != 0
    ):
        return

    velocity, _ = p.getBaseVelocity(tube['base_id'])
    selected_bin_index = tube['fired_jet_index']
    contact_points = 0
    if (
        selected_bin_index is not None
        and bin_instances[selected_bin_index] is not None
    ):
        contact_points = len(
            p.getContactPoints(
                bodyA=tube['base_id'],
                bodyB=bin_instances[selected_bin_index],
            )
        )
    trajectory_writer.writerow(
        {
            "run_id": RUN_ID,
            "tube_sequence": tube['sequence_id'],
            "step": step_count,
            "time_s": f"{step_count * TIME_STEP:.6f}",
            "evaluation_class": tube['class'],
            "selected_jet": (
                selected_bin_index + 1
                if selected_bin_index is not None
                else ""
            ),
            "x": f"{float(position[0]):.6f}",
            "y": f"{float(position[1]):.6f}",
            "z": f"{float(position[2]):.6f}",
            "velocity_x_mps": f"{float(velocity[0]):.6f}",
            "velocity_y_mps": f"{float(velocity[1]):.6f}",
            "velocity_z_mps": f"{float(velocity[2]):.6f}",
            "selected_bin_contact_points": contact_points,
            "crossed_bin_entry": int(tube['crossed_bin_entry']),
            "inside_selected_bin_xy": int(
                tube['inside_selected_bin_xy']
            ),
            "below_selected_bin_rim": int(
                tube['below_selected_bin_rim']
            ),
            "bin_candidate_steps": tube['bin_candidate_steps'],
        }
    )


def record_tube_result(tube, outcome, final_bin_index=None, position=None):
    """Write a tube result."""
    if tube['result_recorded']:
        return

    if position is None and p.isConnected():
        position, _ = p.getBasePositionAndOrientation(tube['base_id'])
    if position is None:
        position = (float("nan"),) * 3

    detected_type = tube['detected_type']
    detected_name = (
        TUBE_CLASSES[detected_type]
        if detected_type is not None
        else ""
    )
    jet_position = tube['jet_position']
    jet_command_position = tube['jet_command_position']
    estimated_command_position = tube[
        'jet_command_estimated_position'
    ]
    jet_start_velocity = tube['jet_start_velocity']
    jet_end_position = tube['jet_end_position']
    jet_end_velocity = tube['jet_end_velocity']
    first_contact_position = tube['first_selected_bin_contact_position']
    first_bin_entry_position = tube['first_bin_entry_position']
    selected_jet_index = tube['fired_jet_index']
    selected_air_config = (
        AIR_JET_CONFIGS[selected_jet_index]
        if selected_jet_index is not None
        else AIR_JET_CONFIG
    )

    def format_component(values, index):
        if values is None:
            return ""
        return f"{float(values[index]):.6f}"

    row = {
        "run_id": RUN_ID,
        "tube_sequence": tube['sequence_id'],
        "body_id": tube['base_id'],
        "seed": SIMULATION_SEED,
        "requested_control_mode": REQUESTED_CONTROL_MODE,
        "control_mode": ACTIVE_CONTROL_MODE,
        "rl_policy_path": (
            str(RL_POLICY_PATH) if RL_POLICY_PATH is not None else ""
        ),
        "rl_observation_lead_s": (
            (
                f"{RL_OBSERVATION_LEAD_SECONDS_BY_JET[selected_jet_index]:.6f}"
                if selected_jet_index is not None
                else ""
            )
        ),
        "rl_valve_deadband": (
            (
                f"{RL_VALVE_DEADBAND_BY_JET[selected_jet_index]:.6f}"
                if selected_jet_index is not None
                else ""
            )
        ),
        "calibration_mode": int(AIR_JET_CALIBRATION_MODE),
        "nominal_pressure_kpa": f"{NOMINAL_JET_PRESSURE_KPA:.6f}",
        "rule_distance_compensation": int(
            RULE_DISTANCE_COMPENSATION
        ),
        "rule_target_attenuation": (
            f"{RULE_TARGET_ATTENUATION_VALUES[selected_jet_index]:.6f}"
            if selected_jet_index is not None
            else ""
        ),
        "tube_mass_kg": f"{tube['mass']:.6f}",
        "spawn_lateral_offset_m": f"{tube['lateral_offset']:.6f}",
        "tube_axis_direction": f"{tube['tube_axis_direction']:.1f}",
        "reference_impulse_x_ns": (
            f"{selected_air_config.reference_impulse_x:.6f}"
        ),
        "reference_impulse_z_ns": (
            f"{selected_air_config.reference_impulse_z:.6f}"
        ),
        "pulse_duration_s": (
            f"{selected_air_config.pulse_duration:.6f}"
        ),
        "jet_to_bin_lead_m": (
            f"{JET_TO_BIN_LEAD_DISTANCES[selected_jet_index]:.6f}"
            if selected_jet_index is not None
            else ""
        ),
        "nozzle_radius_m": f"{selected_air_config.nozzle_radius:.6f}",
        "spread_rate": f"{selected_air_config.spread_rate:.6f}",
        "spawn_step": tube['spawn_step'],
        "spawn_time_s": f"{tube['spawn_step'] * TIME_STEP:.6f}",
        "evaluation_class_id": tube['type'],
        "evaluation_class": tube['class'],
        "detected_class_id": (
            detected_type if detected_type is not None else ""
        ),
        "detected_class": detected_name,
        "confidence": f"{tube['detection_confidence']:.6f}",
        "detection_hits": tube['detection_hits'],
        "first_detection_step": (
            tube['first_detection_step']
            if tube['first_detection_step'] is not None
            else ""
        ),
        "last_detection_step": (
            tube['last_detection_step']
            if tube['last_detection_step'] is not None
            else ""
        ),
        "tracking_source": tube['tracking_source'] or "",
        "tracking_step": (
            tube['tracking_step']
            if tube['tracking_step'] is not None
            else ""
        ),
        "tracking_x": (
            f"{tube['tracking_x']:.6f}"
            if tube['tracking_x'] is not None
            else ""
        ),
        "tracking_y": (
            f"{tube['tracking_y']:.6f}"
            if tube['tracking_y'] is not None
            else ""
        ),
        "selected_jet": (
            tube['fired_jet_index'] + 1
            if tube['fired_jet_index'] is not None
            else ""
        ),
        "policy_action": tube['policy_action_log'],
        "jet_intensity": f"{tube['jet_intensity']:.6f}",
        "trigger_lead_m": (
            f"{tube['calibration_trigger_lead']:.6f}"
            if tube['calibration_trigger_lead'] is not None
            else (
                f"{tube['trigger_lead']:.6f}"
                if (
                    selected_jet_index is not None
                    and tube['trigger_lead'] is not None
                    and ACTIVE_CONTROL_MODE not in RL_ALGORITHMS
                )
                else ""
            )
        ),
        "jet_command_step": (
            tube['jet_command_step']
            if tube['jet_command_step'] is not None
            else ""
        ),
        "jet_command_x": (
            f"{jet_command_position[0]:.6f}"
            if jet_command_position is not None
            else ""
        ),
        "jet_command_y": (
            f"{jet_command_position[1]:.6f}"
            if jet_command_position is not None
            else ""
        ),
        "jet_command_z": (
            f"{jet_command_position[2]:.6f}"
            if jet_command_position is not None
            else ""
        ),
        "estimated_command_x": format_component(
            estimated_command_position,
            0,
        ),
        "estimated_command_y": format_component(
            estimated_command_position,
            1,
        ),
        "tracking_error_x": (
            f"{jet_command_position[0] - estimated_command_position[0]:.6f}"
            if (
                jet_command_position is not None
                and estimated_command_position is not None
            )
            else ""
        ),
        "tracking_error_y": (
            f"{jet_command_position[1] - estimated_command_position[1]:.6f}"
            if (
                jet_command_position is not None
                and estimated_command_position is not None
            )
            else ""
        ),
        "jet_step": tube['jet_step'] if tube['jet_step'] is not None else "",
        "jet_x": (
            f"{jet_position[0]:.6f}" if jet_position is not None else ""
        ),
        "jet_y": (
            f"{jet_position[1]:.6f}" if jet_position is not None else ""
        ),
        "jet_z": (
            f"{jet_position[2]:.6f}" if jet_position is not None else ""
        ),
        "jet_distance_m": (
            f"{tube['jet_distance']:.6f}"
            if tube['jet_distance'] is not None
            else ""
        ),
        "jet_axial_distance_m": (
            f"{tube['jet_axial_distance']:.6f}"
            if tube['jet_axial_distance'] is not None
            else ""
        ),
        "jet_radial_offset_m": (
            f"{tube['jet_radial_offset']:.6f}"
            if tube['jet_radial_offset'] is not None
            else ""
        ),
        "jet_plume_radius_m": (
            f"{tube['jet_plume_radius']:.6f}"
            if tube['jet_plume_radius'] is not None
            else ""
        ),
        "jet_distance_factor": (
            f"{tube['jet_distance_factor']:.6f}"
            if tube['jet_distance_factor'] is not None
            else ""
        ),
        "jet_radial_factor": (
            f"{tube['jet_radial_factor']:.6f}"
            if tube['jet_radial_factor'] is not None
            else ""
        ),
        "jet_attenuation": (
            f"{tube['jet_attenuation']:.6f}"
            if tube['jet_attenuation'] is not None
            else ""
        ),
        "jet_peak_force_n": f"{tube['jet_peak_force']:.6f}",
        "jet_impulse_ns": (
            f"{tube['jet_accumulated_impulse']:.6f}"
        ),
        "jet_start_velocity_x_mps": format_component(
            jet_start_velocity, 0
        ),
        "jet_start_velocity_y_mps": format_component(
            jet_start_velocity, 1
        ),
        "jet_start_velocity_z_mps": format_component(
            jet_start_velocity, 2
        ),
        "jet_end_x": format_component(jet_end_position, 0),
        "jet_end_y": format_component(jet_end_position, 1),
        "jet_end_z": format_component(jet_end_position, 2),
        "jet_end_velocity_x_mps": format_component(
            jet_end_velocity, 0
        ),
        "jet_end_velocity_y_mps": format_component(
            jet_end_velocity, 1
        ),
        "jet_end_velocity_z_mps": format_component(
            jet_end_velocity, 2
        ),
        "post_jet_min_x": (
            f"{tube['post_jet_min_x']:.6f}"
            if tube['post_jet_min_x'] is not None
            else ""
        ),
        "post_jet_max_x": (
            f"{tube['post_jet_max_x']:.6f}"
            if tube['post_jet_max_x'] is not None
            else ""
        ),
        "post_jet_min_z": (
            f"{tube['post_jet_min_z']:.6f}"
            if tube['post_jet_min_z'] is not None
            else ""
        ),
        "post_jet_max_z": (
            f"{tube['post_jet_max_z']:.6f}"
            if tube['post_jet_max_z'] is not None
            else ""
        ),
        "selected_bin_contact_steps": (
            tube['selected_bin_contact_steps']
        ),
        "selected_bin_contact_points": (
            tube['selected_bin_contact_points']
        ),
        "first_selected_bin_contact_step": (
            tube['first_selected_bin_contact_step']
            if tube['first_selected_bin_contact_step'] is not None
            else ""
        ),
        "first_selected_bin_contact_x": format_component(
            first_contact_position, 0
        ),
        "first_selected_bin_contact_y": format_component(
            first_contact_position, 1
        ),
        "first_selected_bin_contact_z": format_component(
            first_contact_position, 2
        ),
        "crossed_bin_entry": int(tube['crossed_bin_entry']),
        "first_bin_entry_step": (
            tube['first_bin_entry_step']
            if tube['first_bin_entry_step'] is not None
            else ""
        ),
        "first_bin_entry_x": format_component(
            first_bin_entry_position, 0
        ),
        "first_bin_entry_y": format_component(
            first_bin_entry_position, 1
        ),
        "first_bin_entry_z": format_component(
            first_bin_entry_position, 2
        ),
        "inside_selected_bin_xy": int(
            tube['inside_selected_bin_xy']
        ),
        "below_selected_bin_rim": int(
            tube['below_selected_bin_rim']
        ),
        "final_bin": (
            final_bin_index + 1 if final_bin_index is not None else ""
        ),
        "outcome": outcome,
        "failure_stage": determine_failure_stage(tube, outcome),
        "success": (
            "" if outcome == "unfinished"
            else int(outcome == "correct_bin")
        ),
        "final_step": step_count,
        "final_x": f"{float(position[0]):.6f}",
        "final_y": f"{float(position[1]):.6f}",
        "final_z": f"{float(position[2]):.6f}",
    }
    results_writer.writerow(row)
    results_file.flush()
    tube['result_recorded'] = True
    tube['result_outcome'] = outcome
    tube['result_step'] = step_count
    result_counts[outcome] = result_counts.get(outcome, 0) + 1
    if GUI_ENABLED and p.isConnected():
        success = outcome == "correct_bin"
        tube['result_debug_id'] = p.addUserDebugText(
            "SORTED" if success else outcome.upper(),
            [0.0, 0.0, 0.35],
            textColorRGB=[0.0, 0.8, 0.0] if success else [1.0, 0.0, 0.0],
            textSize=1.2,
            lifeTime=RESULT_RETENTION_SECONDS,
            parentObjectUniqueId=tube['base_id'],
            parentLinkIndex=-1
        )
    print(
        f"RESULT_EVENT tube={tube['sequence_id']}, "
        f"outcome={outcome}, expected_bin={tube['type'] + 1}, "
        f"final_bin="
        f"{final_bin_index + 1 if final_bin_index is not None else 'none'}"
    )


def remove_tube(tube):
    """Remove a tube and its result label."""
    debug_id = tube.get('result_debug_id')
    if (
            debug_id is not None
            and GUI_ENABLED
            and p.isConnected()
    ):
        p.removeUserDebugItem(debug_id)
    if p.isConnected():
        p.removeBody(tube['base_id'])
    if tube in tubes:
        tubes.remove(tube)


# Spawn tubes.
TUBE_SCALE = 1
INCLINE_SPAWN_PROGRESS = 0.60

# Tube dimensions
TUBE_BELT_CLEARANCES = [
    radius * TUBE_SCALE + 0.004
    for radius in TUBE_RADII_AT_UNIT_SCALE
]

def spawn_tube():
    global tube_sequence_counter

    tube_sequence_counter += 1
    if AIR_JET_CALIBRATION_MODE:
        trial_index = tube_sequence_counter - 1
        opening_count = (
            1
            if CALIBRATION_PAIRED_VALVE_OPENINGS
            else len(CALIBRATION_VALVE_OPENINGS)
        )
        trigger_count = len(CALIBRATION_TRIGGER_LEADS)
        offset_count = len(CALIBRATION_LATERAL_OFFSETS)
        opening_index = trial_index % opening_count
        trigger_index = (
            trial_index // opening_count
        ) % trigger_count
        offset_index = (
            trial_index // (opening_count * trigger_count)
        ) % offset_count
        class_order_index = (
            trial_index
            // (opening_count * trigger_count * offset_count)
        ) % len(CALIBRATION_CLASS_IDS)
        idx = CALIBRATION_CLASS_IDS[class_order_index]
        lateral_offset = CALIBRATION_LATERAL_OFFSETS[
            offset_index
        ]
        if CALIBRATION_PAIRED_VALVE_OPENINGS:
            calibration_valve_opening = (
                CALIBRATION_PAIRED_VALVE_OPENINGS[offset_index]
            )
        elif CALIBRATION_VALVE_OPENINGS_EXPLICIT:
            calibration_valve_opening = (
                CALIBRATION_VALVE_OPENINGS[opening_index]
            )
        else:
            calibration_valve_opening = None
        calibration_trigger_lead = (
            CALIBRATION_TRIGGER_LEADS[trigger_index]
            if CALIBRATION_TRIGGER_LEADS_EXPLICIT
            else None
        )
    else:
        idx = int(simulation_rng.integers(0, len(TUBE_CLASSES)))
        lateral_offset = simulation_rng.uniform(
            -TUBE_LATERAL_OFFSET,
            TUBE_LATERAL_OFFSET
        )
        calibration_valve_opening = None
        calibration_trigger_lead = None

    tube_radius = TUBE_BELT_CLEARANCES[idx]
    incline_clearance = tube_radius + INCLINE_COLLISION_OFFSET
    start_pos = (
        INCLINE_PATH_START
        + INCLINE_TANGENT * INCLINE_SPAWN_PROGRESS
        + INCLINE_NORMAL * incline_clearance
    )
    start_pos[0] += lateral_offset

    tube_axis_direction = float(
        simulation_rng.choice([-1.0, 1.0])
    )
    across_belt_orientation = p.getQuaternionFromEuler(
        [0.0, tube_axis_direction * np.pi / 2.0, 0.0]
    )
    start_ori = across_belt_orientation

    fast_calibration_start = (
        AIR_JET_CALIBRATION_MODE and CALIBRATION_FAST_START
    )
    if fast_calibration_start:
        start_pos = np.array(
            [
                SYSTEM_POSITION[0] + lateral_offset,
                jet_positions[idx] - 0.55,
                HORIZONTAL_SURFACE_Z + tube_radius,
            ],
            dtype=float,
        )

    tube_id = p.loadURDF(
        str(TUBE_URDF_PATHS[idx]),
        start_pos.tolist(),
        start_ori,
        globalScaling=TUBE_SCALE,
        flags=(
            p.URDF_USE_INERTIA_FROM_FILE
            | p.URDF_USE_MATERIAL_COLORS_FROM_MTL
        ),
    )

    tube_mass = sum(
        p.getDynamicsInfo(tube_id, link_index)[0]
        for link_index in range(-1, p.getNumJoints(tube_id))
    )
    expected_mass = TUBE_SPECS[idx].total_mass_kg
    if not np.isclose(tube_mass, expected_mass, atol=1e-9):
        raise RuntimeError(
            f"Tube class {idx} mass is {tube_mass:.6f} kg; "
            f"expected {expected_mass:.6f} kg from its URDF."
        )
    tube_name = TUBE_CLASSES[idx]

    tubes.append({
        'sequence_id': tube_sequence_counter,
        'base_id': tube_id,
        'cap_id': tube_id,
        'constraint_id': None,
        'type': idx,
        'class': tube_name,
        'detected_type': None,
        'detection_confidence': 0.0,
        'detection_hits': 0,
        'reported_detected_type': None,
        'first_detection_step': None,
        'last_detection_step': None,
        'tracking_source': None,
        'tracking_step': None,
        'tracking_x': None,
        'tracking_y': None,
        'jet_command_estimated_position': None,
        'vision_scores': np.zeros(len(TUBE_CLASSES), dtype=float),
        'jet_commanded': False,
        'jet_command_step': None,
        'jet_command_position': None,
        'fired': False,
        'fired_jet_index': None,
        'jet_intensity': 0.0,
        'calibration_valve_opening': calibration_valve_opening,
        'calibration_trigger_lead': calibration_trigger_lead,
        'trigger_lead': None,
        'jet_step': None,
        'jet_position': None,
        'jet_distance': None,
        'jet_axial_distance': None,
        'jet_radial_offset': None,
        'jet_plume_radius': None,
        'jet_distance_factor': None,
        'jet_radial_factor': None,
        'jet_attenuation': None,
        'jet_peak_force': 0.0,
        'jet_accumulated_impulse': 0.0,
        'jet_start_velocity': None,
        'jet_end_position': None,
        'jet_end_velocity': None,
        'jet_pulse_complete': False,
        'post_jet_min_x': None,
        'post_jet_max_x': None,
        'post_jet_min_z': None,
        'post_jet_max_z': None,
        'selected_bin_contact_steps': 0,
        'selected_bin_contact_points': 0,
        'first_selected_bin_contact_step': None,
        'first_selected_bin_contact_position': None,
        'crossed_bin_entry': False,
        'first_bin_entry_step': None,
        'first_bin_entry_position': None,
        'inside_selected_bin_xy': False,
        'below_selected_bin_rim': False,
        'policy_action_log': "",
        'rl_discrete_action': 5,
        'rl_valve_openings': np.zeros(5, dtype=np.float32),
        'rl_last_decision_step': None,
        'missed_jet_reported': False,
        'result_recorded': False,
        'result_outcome': None,
        'result_step': None,
        'result_debug_id': None,
        'bin_candidate_index': None,
        'bin_candidate_steps': 0,
        'spawn_step': step_count,
        'transport_phase': (
            'horizontal' if fast_calibration_start else 'incline'
        ),
        'incline_progress': INCLINE_SPAWN_PROGRESS,
        'horizontal_y': (
            float(start_pos[1]) if fast_calibration_start else None
        ),
        'lateral_offset': lateral_offset,
        'tube_axis_direction': tube_axis_direction,
        'tube_radius': tube_radius,
        'incline_clearance': incline_clearance,
        'transport_orientation': start_ori,
        'mass': tube_mass
    })
    if fast_calibration_start:
        set_tracking_measurement(
            tubes[-1],
            start_pos,
            step_count,
            "calibration_start",
        )

    print(
        f"Spawned tube sequence={tube_sequence_counter}, body={tube_id}, "
        f"evaluation_class={tube_name}, "
        f"lateral_offset={lateral_offset:.3f} m, "
        f"calibration_opening="
        f"{calibration_valve_opening if calibration_valve_opening is not None else 'none'}, "
        f"calibration_trigger_lead="
        f"{calibration_trigger_lead if calibration_trigger_lead is not None else 'none'}, "
        f"mass={tube_mass:.3f} kg"
    )

# Draw debug guides.
spawn_zone_near = (
    INCLINE_PATH_START
    + INCLINE_TANGENT * 0.20
    + INCLINE_NORMAL * 0.02
)
spawn_zone_far = (
    INCLINE_PATH_START
    + INCLINE_TANGENT * 1.10
    + INCLINE_NORMAL * 0.02
)
corners = [
    [
        INCLINE_PATH_START[0] - 1.0,
        spawn_zone_near[1],
        spawn_zone_near[2]
    ],
    [
        INCLINE_PATH_START[0] - 1.0,
        spawn_zone_far[1],
        spawn_zone_far[2]
    ],
    [
        INCLINE_PATH_START[0] + 1.0,
        spawn_zone_far[1],
        spawn_zone_far[2]
    ],
    [
        INCLINE_PATH_START[0] + 1.0,
        spawn_zone_near[1],
        spawn_zone_near[2]
    ]
]
p.addUserDebugLine(corners[0], corners[1], [1, 0, 0], lineWidth=3) # Bottom
p.addUserDebugLine(corners[1], corners[2], [1, 0, 0], lineWidth=3) # Right
p.addUserDebugLine(corners[2], corners[3], [1, 0, 0], lineWidth=3) # Top
p.addUserDebugLine(corners[3], corners[0], [1, 0, 0], lineWidth=3) # Left
p.addUserDebugText(
    "Tube Spawn Zone",
    (
        (spawn_zone_near + spawn_zone_far) / 2.0
        + INCLINE_NORMAL * 0.35
    ).tolist(),
    [1, 0, 0],
    1.5
)

corners = [
    [SHORT_CONVEYOR_X_MIN, -3.64, 5.00],
    [SHORT_CONVEYOR_X_MIN, 8.66, 5.00],
    [SHORT_CONVEYOR_X_MAX, 8.66, 5.00],
    [SHORT_CONVEYOR_X_MAX, -3.64, 5.00]
]
p.addUserDebugLine(corners[0], corners[1], [0, 0, 1], lineWidth=3) # Bottom
p.addUserDebugLine(corners[1], corners[2], [0, 0, 1], lineWidth=3) # Right
p.addUserDebugLine(corners[2], corners[3], [0, 0, 1], lineWidth=3) # Top
p.addUserDebugLine(corners[3], corners[0], [0, 0, 1], lineWidth=3) # Left

x_min, x_max, y_min, y_max, z_min, z_max = (
    SHORT_CONVEYOR_X_MIN,
    SHORT_CONVEYOR_X_MAX,
    -3.64,
    8.66,
    5.00,
    5.60
)
corners = [
    [x_min, y_min, z_min],
    [x_min, y_max, z_min],
    [x_min, y_max, z_max],
    [x_min, y_min, z_max],
    [x_max, y_min, z_min],
    [x_max, y_max, z_min],
    [x_max, y_max, z_max],
    [x_max, y_min, z_max],
]
# Front face (x_min)
p.addUserDebugLine(corners[0], corners[1], [1,0,0], 3)
p.addUserDebugLine(corners[1], corners[2], [1,0,0], 3)
p.addUserDebugLine(corners[2], corners[3], [1,0,0], 3)
p.addUserDebugLine(corners[3], corners[0], [1,0,0], 3)
# Back face (x_max)
p.addUserDebugLine(corners[4], corners[5], [1,0,0], 3)
p.addUserDebugLine(corners[5], corners[6], [1,0,0], 3)
p.addUserDebugLine(corners[6], corners[7], [1,0,0], 3)
p.addUserDebugLine(corners[7], corners[4], [1,0,0], 3)
# Connect front to back
p.addUserDebugLine(corners[0], corners[4], [1,0,0], 3)
p.addUserDebugLine(corners[1], corners[5], [1,0,0], 3)
p.addUserDebugLine(corners[2], corners[6], [1,0,0], 3)
p.addUserDebugLine(corners[3], corners[7], [1,0,0], 3)


def add_control_status_banner(bgr_frame):
    """Show the active control mode."""
    display_frame = np.asarray(bgr_frame).copy()
    cv2.rectangle(
        display_frame,
        (0, 0),
        (display_frame.shape[1], 44),
        (0, 0, 0),
        thickness=-1
    )
    cv2.putText(
        display_frame,
        CONTROL_STATUS_TEXT,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        CONTROL_STATUS_COLOR,
        2,
        cv2.LINE_AA
    )
    return display_frame


def process_vision_results(results, snapshot):
    """Apply one YOLO result to the tracked tubes."""
    if len(results) == 0:
        return

    update_vision_labels(results[0], snapshot)
    results[0].names = {
        model_class_id: TUBE_CLASSES[tube_type]
        for model_class_id, tube_type
        in MODEL_CLASS_TO_TUBE_TYPE.items()
    }

    if SHOW_DETECTION_WINDOW:
        annotated_frame = results[0].plot(
            line_width=2,
            labels=True,
        )
        annotated_frame = add_control_status_banner(annotated_frame)
        annotated_frame = cv2.resize(
            annotated_frame,
            None,
            fx=DETECTION_DISPLAY_SCALE,
            fy=DETECTION_DISPLAY_SCALE,
            interpolation=cv2.INTER_LINEAR,
        )
        cv2.imshow("Detection Feed", annotated_frame)
        cv2.waitKey(1)


def synchronize_horizontal_guides():
    """Apply the hidden guide position before camera capture."""
    for tube in tubes:
        if (
            tube['fired']
            or tube['transport_phase'] != 'horizontal'
            or tube['horizontal_y'] is None
        ):
            continue
        guided_position = [
            SYSTEM_POSITION[0] + tube['lateral_offset'],
            tube['horizontal_y'],
            HORIZONTAL_SURFACE_Z + tube['tube_radius'],
        ]
        p.resetBasePositionAndOrientation(
            tube['base_id'],
            guided_position,
            tube['transport_orientation'],
        )
        p.resetBaseVelocity(
            tube['base_id'],
            linearVelocity=[0.0, HORIZONTAL_FORWARD_SPEED, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
        )



# Run the simulation.
if GUI_ENABLED:
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    if AIR_JET_CALIBRATION_MODE:
        p.addUserDebugText(
            "AIR-JET CALIBRATION",
            [
                SYSTEM_POSITION[0] - 1.0,
                SYSTEM_POSITION[1] + 4.0,
                SYSTEM_POSITION[2] + 2.0,
            ],
            textColorRGB=[1.0, 0.4, 0.0],
            textSize=1.5,
            lifeTime=0,
        )

next_step_deadline = time.perf_counter()
try:
    while (
            MAX_SIMULATION_STEPS <= 0
            or step_count < MAX_SIMULATION_STEPS
    ):
        if not p.isConnected():
            print("Simulation window was closed.")
            break

        p.stepSimulation()
        step_count += 1
              
        
        target_allows_spawn = (
            TARGET_TUBE_COUNT <= 0
            or tube_sequence_counter < TARGET_TUBE_COUNT
        )
        if AIR_JET_CALIBRATION_MODE:
            spawn_due = (
                step_count >= CALIBRATION_INITIAL_DELAY_STEPS
                and not tubes
            )
        else:
            spawn_due = step_count % SPAWN_INTERVAL_STEPS == 0
        if target_allows_spawn and spawn_due:
            spawn_tube()

        synchronize_horizontal_guides()

    # Read completed detections.
        if vision_future is not None and vision_future.done():
            try:
                results = vision_future.result()
                process_vision_results(
                    results,
                    vision_snapshot or [],
                )
            except Exception as exc:
                print(f"WARNING: YOLO inference failed: {exc}")
            finally:
                vision_future = None
                vision_snapshot = None

        # Render the detection camera.
        if (
                (VISION_INFERENCE_ENABLED or SHOW_DETECTION_WINDOW)
                and vision_future is None
                and step_count % VISION_INTERVAL_STEPS == 0
        ):
            img_data = p.getCameraImage(
                CAM_WIDTH,
                CAM_HEIGHT,
                view_matrix,
                proj_matrix,
                shadow=0,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            full_rgb = np.reshape(
                img_data[2],
                (CAM_HEIGHT, CAM_WIDTH, 4)
            )[:, :, :3].copy()
            rgb = full_rgb[
                VISION_ROI_Y_MIN:VISION_ROI_Y_MAX,
                VISION_ROI_X_MIN:VISION_ROI_X_MAX
            ]
            if VISION_INFERENCE_ENABLED:
                model_frame = cv2.cvtColor(
                    rgb,
                    cv2.COLOR_RGB2BGR
                )
                vision_snapshot = capture_vision_snapshot()
                if vision_snapshot:
                    if VISION_SYNCHRONOUS:
                        try:
                            results = model.predict(
                                model_frame,
                                conf=YOLO_CONFIDENCE,
                                verbose=False,
                                imgsz=YOLO_IMAGE_SIZE,
                            )
                            process_vision_results(
                                results,
                                vision_snapshot,
                            )
                        except Exception as exc:
                            print(
                                f"WARNING: YOLO inference failed: {exc}"
                            )
                        finally:
                            vision_snapshot = None
                    else:
                        vision_future = vision_executor.submit(
                            model.predict,
                            model_frame,
                            conf=YOLO_CONFIDENCE,
                            verbose=False,
                            imgsz=YOLO_IMAGE_SIZE
                        )
                else:
                    vision_snapshot = None
            elif SHOW_DETECTION_WINDOW:
                camera_frame = cv2.cvtColor(
                    rgb,
                    cv2.COLOR_RGB2BGR
                )
                camera_frame = add_control_status_banner(camera_frame)
                camera_frame = cv2.resize(
                    camera_frame,
                    None,
                    fx=DETECTION_DISPLAY_SCALE,
                    fy=DETECTION_DISPLAY_SCALE,
                    interpolation=cv2.INTER_LINEAR
                )
                cv2.imshow(
                    "Detection Feed",
                    camera_frame
                )
                cv2.waitKey(1)

        # Move the tubes.
        for tube in tubes[:]:
            pos, _ = p.getBasePositionAndOrientation(
                tube['base_id']
            )

            cap_pos, _ = p.getBasePositionAndOrientation(
                tube['cap_id']
            )

            if tube['fired']:
                tube['post_jet_min_x'] = min(
                    tube['post_jet_min_x'],
                    float(pos[0]),
                )
                tube['post_jet_max_x'] = max(
                    tube['post_jet_max_x'],
                    float(pos[0]),
                )
                tube['post_jet_min_z'] = min(
                    tube['post_jet_min_z'],
                    float(pos[2]),
                )
                tube['post_jet_max_z'] = max(
                    tube['post_jet_max_z'],
                    float(pos[2]),
                )

                if (
                    not tube['crossed_bin_entry']
                    and pos[0] >= bin_entry_x
                ):
                    tube['crossed_bin_entry'] = True
                    tube['first_bin_entry_step'] = step_count
                    tube['first_bin_entry_position'] = tuple(
                        float(value) for value in pos
                    )

                selected_bin_index = tube['fired_jet_index']
                if selected_bin_index is not None:
                    selected_aabb = bin_aabbs[selected_bin_index]
                    selected_body = bin_instances[selected_bin_index]
                    if selected_aabb is not None:
                        inside_xy = (
                            selected_aabb[0][0] < pos[0] < selected_aabb[1][0]
                            and selected_aabb[0][1]
                            < pos[1]
                            < selected_aabb[1][1]
                        )
                        if inside_xy:
                            tube['inside_selected_bin_xy'] = True
                            if pos[2] < selected_aabb[1][2] - 0.05:
                                tube['below_selected_bin_rim'] = True
                    if selected_body is not None:
                        contacts = p.getContactPoints(
                            bodyA=tube['base_id'],
                            bodyB=selected_body,
                        )
                        if contacts:
                            tube['selected_bin_contact_steps'] += 1
                            tube['selected_bin_contact_points'] += len(
                                contacts
                            )
                            if (
                                tube['first_selected_bin_contact_step']
                                is None
                            ):
                                tube[
                                    'first_selected_bin_contact_step'
                                ] = step_count
                                tube[
                                    'first_selected_bin_contact_position'
                                ] = tuple(float(value) for value in pos)

                record_trajectory_sample(tube, pos)

            # Move failed tubes to the conveyor exit.
            if (
                    tube['result_outcome'] == 'missed_bin'
                    and 8.80 < pos[2] < 9.50
                    and jet_trigger_x_min < pos[0] < bin_entry_x
            ):
                tube['horizontal_y'] = (
                    max(
                        float(pos[1]),
                        float(tube['horizontal_y'] or pos[1])
                    )
                    + HORIZONTAL_FORWARD_SPEED * TIME_STEP
                )
                pos = np.array([
                    float(pos[0]),
                    tube['horizontal_y'],
                    HORIZONTAL_SURFACE_Z + tube['tube_radius']
                ])
                p.resetBasePositionAndOrientation(
                    tube['base_id'],
                    pos.tolist(),
                    tube['transport_orientation']
                )
                p.resetBaseVelocity(
                    tube['base_id'],
                    linearVelocity=[
                        0.0,
                        HORIZONTAL_FORWARD_SPEED,
                        0.0
                    ],
                    angularVelocity=[0.0, 0.0, 0.0]
                )

            # Move on the incline conveyor.
            if (
                    not tube['fired']
                    and tube['transport_phase'] == 'incline'
                    and conveyor_id_2 is not None
            ):
                tube['incline_progress'] = min(
                    tube['incline_progress']
                    + INCLINE_BELT_SPEED * TIME_STEP,
                    INCLINE_PATH_LENGTH
                )

                guided_position = (
                    INCLINE_PATH_START
                    + INCLINE_TANGENT * tube['incline_progress']
                    + INCLINE_NORMAL * tube['incline_clearance']
                )
                guided_position[0] = (
                    INCLINE_PATH_START[0]
                    + tube['lateral_offset']
                )

                p.resetBasePositionAndOrientation(
                    tube['base_id'],
                    guided_position.tolist(),
                    tube['transport_orientation']
                )
                p.resetBaseVelocity(
                    tube['base_id'],
                    linearVelocity=(
                        INCLINE_TANGENT * INCLINE_BELT_SPEED
                    ).tolist(),
                    angularVelocity=[0.0, 0.0, 0.0]
                )
                pos = guided_position

                if tube['incline_progress'] >= INCLINE_PATH_LENGTH:
                    tube['transport_phase'] = 'transfer'
                    p.resetBaseVelocity(
                        tube['base_id'],
                        linearVelocity=[
                            0.0,
                            TRANSFER_FORWARD_SPEED,
                            0.0
                        ],
                        angularVelocity=[0.0, 0.0, 0.0]
                    )

            # Transfer to the horizontal conveyor.
            if (
                    not tube['fired']
                    and tube['transport_phase'] == 'transfer'
                    and conveyor_id_4 is not None
            ):
                horizontal_contacts = p.getContactPoints(
                    bodyA=tube['base_id'],
                    bodyB=conveyor_id_4
                )
                landing_height = (
                    HORIZONTAL_SURFACE_Z
                    + tube['tube_radius']
                )
                reached_belt = (
                    pos[1] > HORIZONTAL_ENTRY_Y - 0.25
                    and pos[2] <= landing_height + 0.25
                )

                if horizontal_contacts or reached_belt:
                    tube['transport_phase'] = 'horizontal'
                        # Keep the current forward position.
                    tube['horizontal_y'] = max(
                        float(pos[1]),
                        HORIZONTAL_ENTRY_Y + 0.02
                    )
                    pos = np.array([
                        SYSTEM_POSITION[0] + tube['lateral_offset'],
                        tube['horizontal_y'],
                        landing_height
                    ])

                    p.resetBasePositionAndOrientation(
                        tube['base_id'],
                        pos.tolist(),
                        tube['transport_orientation']
                    )
                    p.resetBaseVelocity(
                        tube['base_id'],
                        linearVelocity=[
                            0.0,
                            HORIZONTAL_FORWARD_SPEED,
                            0.0
                        ],
                        angularVelocity=[0.0, 0.0, 0.0]
                    )

            # Move on the horizontal conveyor.
            if (
                    not tube['fired']
                    and tube['transport_phase'] == 'horizontal'
                    and conveyor_id_4 is not None
            ):
                tube['horizontal_y'] += (
                    HORIZONTAL_FORWARD_SPEED * TIME_STEP
                )
                pos = np.array([
                    SYSTEM_POSITION[0] + tube['lateral_offset'],
                    tube['horizontal_y'],
                    HORIZONTAL_SURFACE_Z + tube['tube_radius']
                ])

                p.resetBasePositionAndOrientation(
                    tube['base_id'],
                    pos.tolist(),
                    tube['transport_orientation']
                )
                p.resetBaseVelocity(
                    tube['base_id'],
                    linearVelocity=[
                        0.0,
                        HORIZONTAL_FORWARD_SPEED,
                        0.0
                    ],
                    angularVelocity=[0.0, 0.0, 0.0]
                )

            # Confirm tube collection.
            if tube['fired'] and not tube['result_recorded']:
                collection_bin_index = None
                for bin_index, aabb in enumerate(bin_aabbs):
                    if aabb is None:
                        continue
                    capture_ceiling = min(
                        aabb[1][2] - 0.05,
                        HORIZONTAL_SURFACE_Z - BIN_DROP_BELOW_BELT
                    )
                    inside_bin = (
                        aabb[0][0] + BIN_CAPTURE_DEPTH
                        < pos[0]
                        < aabb[1][0] - 0.10
                        and aabb[0][1] + 0.08 < pos[1] < aabb[1][1] - 0.08
                        and aabb[0][2] + 0.05 < pos[2] < capture_ceiling
                    )
                    if inside_bin:
                        collection_bin_index = bin_index
                        break

                if collection_bin_index is None:
                    tube['bin_candidate_index'] = None
                    tube['bin_candidate_steps'] = 0
                elif tube['bin_candidate_index'] == collection_bin_index:
                    tube['bin_candidate_steps'] += 1
                else:
                    tube['bin_candidate_index'] = collection_bin_index
                    tube['bin_candidate_steps'] = 1

                if tube['bin_candidate_steps'] >= BIN_CONFIRMATION_STEPS:
                    bin_index = tube['bin_candidate_index']
                    if bin_index is not None:
                        outcome = (
                            "correct_bin"
                            if bin_index == tube['type']
                            else "wrong_bin"
                        )
                        record_tube_result(
                            tube,
                            outcome,
                            final_bin_index=bin_index,
                            position=pos
                        )

            # Remove completed tubes.
            if (
                    tube['result_recorded']
                    and REMOVE_COMPLETED_TUBES
                    and tube['result_step'] is not None
            ):
                result_age = (
                    step_count - tube['result_step']
                ) * TIME_STEP
                outside_workspace = (
                    pos[2] < 4.0
                    or abs(pos[0]) > 10.0
                    or result_age >= RESULT_RETENTION_SECONDS
                )
                if outside_workspace:
                    remove_tube(tube)
                    continue

            if pos[1] > 9.20:
                if not tube['result_recorded']:
                    outcome = (
                        "missed_bin" if tube['fired'] else "missed_nozzle"
                    )
                    record_tube_result(tube, outcome, position=pos)
                remove_tube(tube)
                continue

            # Move on the short conveyor.
            if (
                    SHORT_CONVEYOR_X_MIN < pos[0] < SHORT_CONVEYOR_X_MAX
                    and -3.64 < pos[1] < 8.66
                    and 5.00 < pos[2] < 5.60
            ):
                _, current_angular_velocity = p.getBaseVelocity(
                    tube['base_id']
                )

                p.resetBaseVelocity(
                    tube['base_id'],
                    linearVelocity=[0.0, 2.0, 0.0],
                    angularVelocity=current_angular_velocity
                )

        # Control the air jets.
        if step_count % 200 == 0:
            for jet_index, y_pos in enumerate(jet_positions):
                trigger_y = (
                    y_pos - RULE_TRIGGER_LEAD_DISTANCES[jet_index]
                )
                min_pt = [
                    jet_trigger_x_min,
                    trigger_y - RULE_TRIGGER_HALF_WIDTH,
                    8.80
                ]

                max_pt = [
                    bin_entry_x,
                    trigger_y + RULE_TRIGGER_HALF_WIDTH,
                    9.50
                ]
                def draw_box(low, high, color):
                    points = [
                        [low[0], low[1], low[2]], [high[0], low[1], low[2]],
                        [high[0], high[1], low[2]], [low[0], high[1], low[2]],
                        [low[0], low[1], high[2]], [high[0], low[1], high[2]],
                        [high[0], high[1], high[2]], [low[0], high[1], high[2]]
                    ]
                    lines = [
                        [0,1], [1,2], [2,3], [3,0], # Bottom
                        [4,5], [5,6], [6,7], [7,4], # Top
                        [0,4], [1,5], [2,6], [3,7]
                    ]
                    for start, end in lines:
                        p.addUserDebugLine(
                            points[start],
                            points[end],
                            color,
                            lineWidth=1,
                            lifeTime=1.0
                        )

                draw_box(min_pt, max_pt, [0, 1, 0])
        for tube in tubes[:]:
            pos, _ = p.getBasePositionAndOrientation(tube['base_id'])
            if tube['result_recorded']:
                continue

            if (
                ACTIVE_CONTROL_MODE == "ground_truth"
                and tube['tracking_step'] is None
                and tube['transport_phase'] == 'horizontal'
                and project_world_to_image(pos) is not None
            ):
                set_tracking_measurement(
                    tube,
                    pos,
                    step_count,
                    "ground_truth_camera",
                )
                tube['first_detection_step'] = step_count
                tube['last_detection_step'] = step_count

            update_air_jet_force(tube, pos)
            rule_target_jet = None
            control_ready = False
            estimated_pos = estimate_tracked_position(tube)

            if ACTIVE_CONTROL_MODE == "yolo":
                rule_target_jet = tube['detected_type']
                if (
                        rule_target_jet is None
                        or tube['detection_hits'] < MIN_DETECTION_HITS
                ):
                    rule_target_jet = None
                control_ready = (
                    rule_target_jet is not None
                    and estimated_pos is not None
                )
            elif ACTIVE_CONTROL_MODE == "ground_truth":
                rule_target_jet = tube['type']
                control_ready = estimated_pos is not None
            elif ACTIVE_CONTROL_MODE in RL_ALGORITHMS:
                detection_ready = (
                    tube['detected_type'] is not None
                    and tube['detection_hits'] >= MIN_DETECTION_HITS
                    and estimated_pos is not None
                )
                if (
                        detection_ready
                        and not tube['jet_commanded']
                        and tube['transport_phase'] == 'horizontal'
                        and step_count % RL_CONTROL_INTERVAL_STEPS == 0
                ):
                    try:
                        observation = build_observation(
                            estimated_pos[0],
                            (
                                estimated_pos[1]
                                + HORIZONTAL_FORWARD_SPEED
                                * RL_OBSERVATION_LEAD_SECONDS_BY_JET[
                                    tube['detected_type']
                                ]
                            ),
                            tube['detected_type'],
                            jet_positions[tube['detected_type']],
                        )
                        openings, action_index = predict_valve_openings(
                            RL_POLICY,
                            ACTIVE_CONTROL_MODE,
                            observation,
                            tube['detected_type'],
                        )
                        tube['rl_valve_openings'] = openings
                        tube['rl_discrete_action'] = (
                            action_index
                            if action_index is not None
                            else -1
                        )
                        if action_index is not None:
                            tube['policy_action_log'] = str(action_index)
                        else:
                            tube['policy_action_log'] = (
                                "["
                                + ",".join(
                                    f"{value:.3f}"
                                    for value in openings
                                )
                                + "]"
                            )
                        tube['rl_last_decision_step'] = step_count
                    except Exception as exc:
                        tube['rl_valve_openings'] = np.zeros(
                            5,
                            dtype=np.float32,
                        )
                        print(
                            f"WARNING: RL prediction failed for tube "
                            f"{tube['sequence_id']}: {exc}"
                        )
                control_ready = (
                    detection_ready
                    and tube['rl_last_decision_step'] is not None
                )

            if control_ready:
                for jet_index, target_jet_y in enumerate(jet_positions):
                    if ACTIVE_CONTROL_MODE in RL_ALGORITHMS:
                        trigger_y = target_jet_y
                        trigger_half_width = RL_TRIGGER_HALF_WIDTH
                    else:
                        trigger_lead = (
                            tube['calibration_trigger_lead']
                            if (
                                AIR_JET_CALIBRATION_MODE
                                and tube['calibration_trigger_lead']
                                is not None
                            )
                            else calculate_rule_trigger_lead(
                                estimated_pos,
                                jet_index,
                            )
                        )
                        trigger_y = target_jet_y - trigger_lead
                        trigger_half_width = RULE_TRIGGER_HALF_WIDTH
                    y_hit = (
                        abs(estimated_pos[1] - trigger_y)
                        < trigger_half_width
                    )

                    x_hit = (
                            jet_trigger_x_min
                            < estimated_pos[0]
                            < bin_entry_x
                    )

                    z_hit = (
                            8.80 < estimated_pos[2] < 9.50
                    )

                    if ACTIVE_CONTROL_MODE in RL_ALGORITHMS:
                        jet_intensity = float(
                            tube['rl_valve_openings'][jet_index]
                        )
                        jet_commanded = (
                            jet_intensity
                            > RL_VALVE_DEADBAND_BY_JET[jet_index]
                        )
                    else:
                        jet_commanded = (
                            rule_target_jet == jet_index
                        )
                        if jet_commanded:
                            if (
                                AIR_JET_CALIBRATION_MODE
                                and tube['calibration_valve_opening']
                                is not None
                            ):
                                jet_intensity = tube[
                                    'calibration_valve_opening'
                                ]
                            else:
                                jet_intensity = calculate_rule_valve_opening(
                                    estimated_pos,
                                    jet_index,
                                )
                        else:
                            jet_intensity = 0.0

                    if (
                            not tube['jet_commanded']
                            and jet_commanded
                            and y_hit
                            and x_hit
                            and z_hit
                    ):
                        tube['jet_commanded'] = True
                        tube['trigger_lead'] = (
                            0.0
                            if ACTIVE_CONTROL_MODE in RL_ALGORITHMS
                            else trigger_lead
                        )
                        tube['fired_jet_index'] = jet_index
                        tube['jet_intensity'] = jet_intensity
                        tube['jet_command_step'] = step_count
                        tube['jet_command_position'] = tuple(
                            float(value) for value in pos
                        )
                        tube['jet_command_estimated_position'] = tuple(
                            float(value) for value in estimated_pos
                        )
                        detected_name = (
                            TUBE_CLASSES[tube['detected_type']]
                            if tube['detected_type'] is not None
                            else "none"
                        )
                        print(
                            f"JET_COMMAND_EVENT tube={tube['sequence_id']}, "
                            f"jet={jet_index + 1}, "
                            f"control={ACTIVE_CONTROL_MODE}, "
                            f"evaluation_class={tube['class']}, "
                            f"detected_class={detected_name}, "
                            f"policy_action={tube['policy_action_log']}, "
                            f"intensity={jet_intensity:.3f}"
                        )

                        break

            # Record missed jet events.
            if estimated_pos is None:
                expected_trigger_end_y = None
            elif ACTIVE_CONTROL_MODE in RL_ALGORITHMS:
                expected_trigger_end_y = (
                    jet_positions[tube['type']] + RL_TRIGGER_HALF_WIDTH
                )
            else:
                expected_trigger_lead = (
                    tube['calibration_trigger_lead']
                    if (
                        AIR_JET_CALIBRATION_MODE
                        and tube['calibration_trigger_lead'] is not None
                    )
                    else calculate_rule_trigger_lead(
                        estimated_pos,
                        tube['type'],
                    )
                )
                expected_trigger_end_y = (
                    jet_positions[tube['type']]
                    - expected_trigger_lead
                    + RULE_TRIGGER_HALF_WIDTH
                )
            if (
                    not tube['jet_commanded']
                    and not tube['missed_jet_reported']
                    and tube['transport_phase'] == 'horizontal'
                    and expected_trigger_end_y is not None
                    and estimated_pos[1] > expected_trigger_end_y
            ):
                detected_name = (
                    TUBE_CLASSES[tube['detected_type']]
                    if tube['detected_type'] is not None
                    else "none"
                )
                tube['missed_jet_reported'] = True
                print(
                    f"MISSED_JET tube={tube['sequence_id']}, "
                    f"body={tube['base_id']}, "
                    f"evaluation_class={tube['class']}, "
                    f"detected_class={detected_name}, "
                    f"hits={tube['detection_hits']}, "
                    f"position=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                )

            if (
                    tube['fired']
                    and not tube['result_recorded']
                    and tube['jet_step'] is not None
                    and (
                        step_count - tube['jet_step']
                        > SORTING_RESULT_TIMEOUT_SECONDS / TIME_STEP
                    )
            ):
                record_tube_result(
                    tube,
                    "missed_bin",
                    position=pos
                )

        if (
                TARGET_TUBE_COUNT > 0
                and tube_sequence_counter >= TARGET_TUBE_COUNT
                and (
                    (
                        REMOVE_COMPLETED_TUBES
                        and not tubes
                    )
                    or (
                        not REMOVE_COMPLETED_TUBES
                        and all(
                            tube['result_recorded']
                            for tube in tubes
                        )
                    )
                )
        ):
            print(
                f"Completed target tube count: {TARGET_TUBE_COUNT}"
            )
            break
                
                                    
        if REALTIME_PACING:
            next_step_deadline += TIME_STEP
            sleep_duration = next_step_deadline - time.perf_counter()
            if sleep_duration > 0.0:
                time.sleep(sleep_duration)
            elif sleep_duration < -0.25:
        # Reset accumulated timing lag.
                next_step_deadline = time.perf_counter()
except KeyboardInterrupt:
    print("Simulation stopped by user.")
except Exception as exc:
    print(f"Simulation stopped because of an error: {exc}")
finally:
    if vision_executor is not None:
        vision_executor.shutdown(wait=True, cancel_futures=True)
    if p.isConnected():
        for tube in tubes:
            if not tube['result_recorded']:
                try:
                    final_position, _ = p.getBasePositionAndOrientation(
                        tube['base_id']
                    )
                except Exception:
                    final_position = None
                record_tube_result(
                    tube,
                    "unfinished",
                    position=final_position
                )
    print(f"Simulation result summary: {result_counts}")
    print(f"Simulation result CSV: {RESULTS_CSV_PATH}")
    results_file.close()
    if trajectory_file is not None:
        trajectory_file.close()
    if p.isConnected():
        p.disconnect()
    cv2.destroyAllWindows()
