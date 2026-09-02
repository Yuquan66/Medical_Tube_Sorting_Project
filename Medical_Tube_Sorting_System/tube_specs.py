from dataclasses import dataclass


@dataclass(frozen=True)
class TubeSpec:
    class_id: int
    name: str
    yolo_aliases: tuple[str, ...]
    urdf_filename: str
    body_material: str
    cap_material: str
    empty_body_mass_kg: float
    cap_mass_kg: float
    residual_mass_kg: float
    body_diameter_mm: float
    cap_diameter_mm: float
    nominal_length_mm: float
    simulation_radius: float
    lateral_friction: float
    cap_lateral_friction: float
    reference_impulse_x: float
    reference_impulse_z: float

    @property
    def body_mass_kg(self) -> float:
        return self.empty_body_mass_kg + self.residual_mass_kg

    @property
    def total_mass_kg(self) -> float:
        return self.body_mass_kg + self.cap_mass_kg


TUBE_SPECS = (
    TubeSpec(
        class_id=0,
        name="universal polystyrene tube",
        yolo_aliases=(
            "polypropene tube 1",
            "polystyrene universal tube",
            "universal tube ps",
        ),
        urdf_filename="polypropene-tube-1.urdf",
        body_material="PS",
        cap_material="HDPE",
        empty_body_mass_kg=0.007,
        cap_mass_kg=0.008,
        residual_mass_kg=0.008,
        body_diameter_mm=17.0,
        cap_diameter_mm=31.0,
        nominal_length_mm=102.0,
        simulation_radius=0.15875,
        lateral_friction=0.40,
        cap_lateral_friction=0.30,
        reference_impulse_x=0.161,
        reference_impulse_z=0.115,
    ),
    TubeSpec(
        class_id=1,
        name="universal polypropylene tube",
        yolo_aliases=(
            "polypropene tube 2",
            "polypropylene universal tube",
            "universal tube pp",
        ),
        urdf_filename="polypropene-tube-2.urdf",
        body_material="PP",
        cap_material="HDPE",
        empty_body_mass_kg=0.005,
        cap_mass_kg=0.008,
        residual_mass_kg=0.005,
        body_diameter_mm=17.0,
        cap_diameter_mm=31.0,
        nominal_length_mm=68.0,
        simulation_radius=0.15875,
        lateral_friction=0.35,
        cap_lateral_friction=0.30,
        reference_impulse_x=0.126,
        reference_impulse_z=0.090,
    ),
    TubeSpec(
        class_id=2,
        name="polypropylene centrifuge tube 1",
        yolo_aliases=(
            "polystyrene tube 1",
            "polypropene centrifuge tube 1",
            "centrifuge tube pp 1",
        ),
        urdf_filename="polypropene-centrifugal-1.urdf",
        body_material="PP",
        cap_material="HDPE",
        empty_body_mass_kg=0.006,
        cap_mass_kg=0.003,
        residual_mass_kg=0.004,
        body_diameter_mm=17.5,
        cap_diameter_mm=22.6,
        nominal_length_mm=124.0,
        simulation_radius=0.11675,
        lateral_friction=0.35,
        cap_lateral_friction=0.30,
        reference_impulse_x=0.091,
        reference_impulse_z=0.065,
    ),
    TubeSpec(
        class_id=3,
        name="polypropylene centrifuge tube 2",
        yolo_aliases=(
            "polystyrene tube 2",
            "polypropene centrifuge tube 2",
            "centrifuge tube pp 2",
        ),
        urdf_filename="polypropene-centrifugal-2.urdf",
        body_material="PP",
        cap_material="HDPE",
        empty_body_mass_kg=0.005,
        cap_mass_kg=0.003,
        residual_mass_kg=0.004,
        body_diameter_mm=17.0,
        cap_diameter_mm=21.4,
        nominal_length_mm=124.0,
        simulation_radius=0.11675,
        lateral_friction=0.35,
        cap_lateral_friction=0.30,
        reference_impulse_x=0.084,
        reference_impulse_z=0.060,
    ),
    TubeSpec(
        class_id=4,
        name="polypropylene lysis tube",
        yolo_aliases=(
            "lysis tube",
            "polypropene lysis tube",
        ),
        urdf_filename="lysis-tube-1.urdf",
        body_material="PP",
        cap_material="HDPE",
        empty_body_mass_kg=0.0035,
        cap_mass_kg=0.0015,
        residual_mass_kg=0.003,
        body_diameter_mm=14.0,
        cap_diameter_mm=16.0,
        nominal_length_mm=122.0,
        simulation_radius=0.08375,
        lateral_friction=0.35,
        cap_lateral_friction=0.30,
        reference_impulse_x=0.056,
        reference_impulse_z=0.045,
    ),
)

TUBE_SPEC_BY_ID = {
    spec.class_id: spec
    for spec in TUBE_SPECS
}
TUBE_CLASSES = {
    spec.class_id: spec.name
    for spec in TUBE_SPECS
}
TUBE_URDF_FILENAMES = tuple(
    spec.urdf_filename
    for spec in TUBE_SPECS
)
TUBE_RADII_AT_UNIT_SCALE = tuple(
    spec.simulation_radius
    for spec in TUBE_SPECS
)
JET_REFERENCE_IMPULSE_X_VALUES = tuple(
    spec.reference_impulse_x
    for spec in TUBE_SPECS
)
JET_REFERENCE_IMPULSE_Z_VALUES = tuple(
    spec.reference_impulse_z
    for spec in TUBE_SPECS
)


def validate_tube_specs() -> None:
    expected_ids = set(range(len(TUBE_SPECS)))
    actual_ids = {spec.class_id for spec in TUBE_SPECS}
    if actual_ids != expected_ids:
        raise ValueError("Tube class IDs must be consecutive.")
    for spec in TUBE_SPECS:
        if spec.total_mass_kg <= 0.0:
            raise ValueError(f"Invalid tube mass for class {spec.class_id}.")
        if spec.simulation_radius <= 0.0:
            raise ValueError(f"Invalid tube radius for class {spec.class_id}.")


validate_tube_specs()
