"""Task configuration classes for MolmoSpaces experiments."""

from pathlib import Path
from typing import TypeAlias

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_config import Config


class BaseMujocoTaskConfig(Config):
    """Base configuration for MuJoCo tasks.

    NOTE:
    If these task config parameters are left to None, they will be sampled by the task sampler.
    If these task config parameters are not None, their value will take precedence over
    any parameters sampled by the task sampler and will remain fixed across all simulation
    tasks sampled by the task sampler.
    """

    task_cls: type | None  # [AbstractMujocoTask]

    # dict of object names to xml locations
    added_objects: dict[str, Path] = {}

    # Object positions (for internal use by eval_task_sampler)
    # dict of object names to world poses
    object_poses: dict[str, list[float]] | None = None

    # Map from task-relevant role (e.g. 'object_name', 'pickup_name',
    # or 'place_name') to the chosen referral expression
    referral_expressions: dict[str, str] = {}

    # Map from task-relevant role (e.g. 'object_name', 'pickup_name',
    # or 'place_name') to prioritized referral expressions.
    # Each represented referral expression is represented as a tuple with
    #  - CLIP score difference between the actual target and the most similar other object in the context,
    #  - CLIP score for the expression and the actual target
    #  - the referral expression/description
    referral_expressions_priority: dict[str, list[tuple[float, float, str]]] = {}

    robot_base_pose: list[float] | None = None  # initial robot base pose, xyz + quat

    # Sensor settings (common to all task types)
    use_sensors: bool = True  # Whether to use the sensor system
    tracked_object_names: list[str] | None = None  # Object names for ObjectPoseSensor
    action_dtype: str = "float32"  # Enforced dtype for all action components


class PickTaskConfig(BaseMujocoTaskConfig):
    """Configuration for Franka move-to-pose task."""

    task_cls: type | None = None  # Will be set by importing module to avoid circular imports

    # Object names
    pickup_obj_start_pose: list[float] | None = None
    pickup_obj_goal_pose: list[float] | None = None
    receptacle_name: str | None = None
    place_target_name: str | None = None
    pickup_obj_name: str | None = None

    # Task parameters
    succ_pos_threshold: float = 0.01  # lower threshold lift height in meters
    # succ_rot_threshold: float = 0.15  # Rotation success threshold in radians

    # Rendering settings
    enable_rendering: bool = True  # Whether to enable environment rendering for visual sensors


class PickAndPlaceTaskConfig(PickTaskConfig):
    place_receptacle_name: str | None = None
    place_receptacle_start_pose: list[float] | None = None
    succ_pos_threshold: float = np.inf  # no position success threshold, we use support instead
    receptacle_supported_weight_frac: float = (
        0.5  # how much of the object weight should be supported by the receptacle
    )
    max_place_receptacle_pos_displacement: float = (
        0.1  # maximum distance the receptacle can be moved
    )
    max_place_receptacle_rot_displacement: float = np.radians(
        45
    )  # maximum rotation the receptacle can be rotated
    carry_forward_rel_pos_threshold: float = 0.005  # meters
    carry_forward_rel_rot_threshold: float = np.radians(10)  # radians


class PickAndPlaceNextToTaskConfig(PickAndPlaceTaskConfig):
    max_place_receptacle_pos_displacement: float = (
        0.05  # maximum distance the receptacle can be moved
    )
    max_place_receptacle_rot_displacement: float = np.radians(45)
    # actually task success
    min_surface_to_surface_gap: float = 0
    max_surface_to_surface_gap: float = 0.05


class PickAndPlaceColorTaskConfig(PickAndPlaceTaskConfig):
    object_colors: dict[str, list[float]] | None = None  # object name -> rgba
    other_receptacle_names: list[str] | None = None
    other_receptacle_start_poses: dict[str, list[float]] | None = None


class PackingTaskConfig(PickAndPlaceTaskConfig):
    pass


class OpeningTaskConfig(PickTaskConfig):
    """Configuration for opening task."""

    # --- Opening-specific task parameters ---
    articulation_object_name: str | None = None  # e.g., "door|2|8_Doorway_Double_7_doorway_door_7"
    joint_name: str | None = None  # e.g., "joint_0"
    joint_index: int | None = 0  # index of the joint to open
    joint_start_position: float | None = None
    joint_goal_position: float | None = None

    # Success criteria
    any_inst_of_category: bool = False  # for open, reward for any instance of category
    task_success_threshold: float = 0.20  # percentage of opening

    # Rendering settings
    enable_rendering: bool = True  # Whether to enable environment rendering for visual sensors


class DoorOpeningTaskConfig(BaseMujocoTaskConfig):
    """Configuration for RBY1 door opening task.

    NOTE:
    If these task config parameters are left to None, they will be sampled by the task sampler.
    If these task config parameters are not None, their value will take precedence over
    any parameters sampled by the task sampler and will remain fixed across all simulation
    tasks sampled by the task sampler.
    """

    task_cls: type = None  # Will be set by importing module to avoid circular imports

    # --- DoorOpening-specific task parameters ---
    door_body_name: str | None = None  # e.g., "door|2|8_Doorway_Double_7_doorway_door_7"
    articulated_joint_range: np.ndarray | None = None
    articulated_joint_reset_state: np.ndarray | None = None
    additional_tcp_rotation_offset_mat: np.ndarray = (
        R.from_euler("Y", -90, degrees=True)
    ).as_matrix()
    additional_tcp_offset_distance: float = 0.03  # optional distance to move tcp closer/farther from the door handle (tune as per gripper / grasping requirements)

    # Reward function
    door_open_reward: float = 1.0

    # Success criteria
    door_openness_threshold: float = 0.67  # percentage of door opening

    # Debug visualizations
    viz_target_ee: bool = True  # Visualize target end-effector pose


class NavToObjTaskConfig(BaseMujocoTaskConfig):
    """Configuration for RBY1 navigation to object task.

    NOTE:
    If these task config parameters are left to None, they will be sampled by the task sampler.
    If these task config parameters are not None, their value will take precedence over
    any parameters sampled by the task sampler and will remain fixed across all simulation
    tasks sampled by the task sampler.
    Uses pickup_obj_name for compatibility with EvalTaskSampler.
    """

    task_cls: type | None = None  # Will be set by importing module to avoid circular imports
    seed: int | None = None

    # Object navigation-specific task parameters (using pickup_obj_* naming for compatibility)
    pickup_obj_name: str | None = None  # Target object instance name
    pickup_obj_candidates: list[str] | None = (
        None  # List of all candidate object instances of this type
    )
    pickup_obj_category: str | None = None  # Semantic category (e.g., "apple")
    pickup_obj_synset: str | None = None  # WordNet synset (e.g., "apple.n.01")

    # For compatibility with EvalTaskSampler (not used in nav tasks, but needed for shared code)
    receptacle_name: str | None = None
    pickup_obj_start_pose: list[float] | None = None

    # Task parameters
    succ_pos_threshold: float = 1.5  # meters  # Success distance threshold in meters

    # Name of the camera (registry name) used for the object-visibility success check.
    # Defaults to the RBY1 head camera; mobile-base embodiments override this.
    visibility_camera_name: str = "head_camera"

    # Optional list of registry camera names for the object-visibility success
    # check. When set, the target counts as visible if it is visible from ANY of
    # these cameras (OR). Falls back to the single ``visibility_camera_name`` when
    # None. Used by mobile-base rigs with a left/right shoulder camera pair.
    visibility_camera_names: list[str] | None = None

    # Minimum fraction of the frame (0..1) the target must occupy in a camera for
    # it to count as "visible" for the success gate. This backstops the sampler's
    # physical-size filter: even a normally-sized object counts as visible only if
    # it presents a large-enough silhouette at the standoff (not a handful of far
    # pixels). 0.0 keeps the historical "any nonzero pixels" behaviour.
    min_visible_fraction: float = 0.0

    # When True, a nav episode only counts as success if the target is BOTH within
    # ``succ_pos_threshold`` AND visible from ``visibility_camera_name``. When False,
    # success is judged on distance alone (the visibility check is skipped). This
    # exists to suppress false-negative failures caused by an uncalibrated /
    # mis-framed nav camera; callers that disable it should warn loudly at launch.
    require_object_visible: bool = True

    # When True, the success distance is measured to the object's SURFACE (centre
    # distance minus the object's in-plane bounding reach) instead of its centre.
    # This avoids penalising large objects, whose closest collision-free base pose
    # can lie beyond ``succ_pos_threshold`` from the centre even when the robot is
    # right at the object. Default False (historical centre-distance behaviour).
    succ_use_surface_distance: bool = False

    # --- Goal-pose-reaching success (pre-grasp semantics) ----------------------
    # When True, nav success is judged on whether the robot REACHED ITS PLANNED
    # PRE-GRASP GOAL POSE, decoupled from object-centre distance. This is the
    # correct criterion for mobile manipulation: for a target on furniture the
    # closest navigable pose is the furniture *edge* (the pre-grasp standoff), so
    # the robot can never get within ``succ_pos_threshold`` of the object centre
    # even though it is perfectly positioned to grasp. The navigation policy
    # publishes the goal pose it committed to via ``set_planned_nav_goal`` and
    # success requires the base to arrive within ``succ_goal_pos_threshold`` and
    # ``succ_goal_yaw_threshold`` of it. A validity guard rejects goals whose
    # surface standoff to the object exceeds ``max_pregrasp_standoff_m`` (a target
    # with no reachable pre-grasp pose is a genuine failure, not a free success).
    succ_use_goal_pose: bool = False
    succ_goal_pos_threshold: float = 0.30  # metres; base->goal planar tolerance
    succ_goal_yaw_threshold: float = float(np.deg2rad(20))  # heading tolerance
    max_pregrasp_standoff_m: float = 1.75  # max object-surface standoff for a valid goal

    # Rendering settings
    enable_rendering: bool = True  # Whether to enable environment rendering for visual sensors


AllTaskConfigs: TypeAlias = (
    BaseMujocoTaskConfig
    | PickTaskConfig
    | PickAndPlaceTaskConfig
    | PickAndPlaceColorTaskConfig
    | PickAndPlaceNextToTaskConfig
    | PackingTaskConfig
    | OpeningTaskConfig
    | DoorOpeningTaskConfig
    | NavToObjTaskConfig
)
