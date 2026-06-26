"""
Example configuration for RBY1 navigation to object data generation using the extracted task sampler.
This shows how the scene randomization functionality from the reference script has been
properly integrated into the modular task sampler architecture.
"""

from __future__ import annotations

import numpy as np

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.camera_configs import (
    CameraSystemConfig,
    FrankaDroidCameraSystem,
    RBY1MjcfCameraSystem,
)
from molmo_spaces.configs.policy_configs import AStarNavToObjPolicyConfig, BasePolicyConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig, MobileFrankaRobotConfig, RBY1Config
from molmo_spaces.configs.task_configs import NavToObjTaskConfig
from molmo_spaces.configs.task_sampler_configs import NavToObjTaskSamplerConfig
from molmo_spaces.tasks.nav_task import NavToObjTask
from molmo_spaces.tasks.nav_task_sampler import NavToObjTaskSampler
from molmo_spaces.utils.profiler_utils import Profiler


class NavToObjBaseConfig(MlSpacesExpConfig):
    """Base configuration for navigation to object data generation tasks."""

    # NOTE: will not work if used directly. Subclass examples in data_generation/configs.py

    # --- Experiment-level config parameters ---
    num_envs: int = 1  # Number of environments to run in each thread
    use_passive_viewer: bool = False  # Launch passive viewer for rendering
    viewer_camera: None = None
    viewer_cam_dict: dict = {
        "distance": 5.0,
        "azimuth": 45.0,
        "elevation": -30.0,
        "lookat": np.array([0.0, 0.0, 0.5]),
    }
    policy_dt_ms: float = 200.0  # policy time step
    ctrl_dt_ms: float = 2.0  # control time step
    sim_dt_ms: float = 2.0  # simulation time step
    task_horizon: int = 500  # Maximum steps per episode to prevent infinite runs
    record_videos: bool = False  # Whether to record videos of episodes

    # --- Data generation settings ---
    num_threads: int = 1  # parallel data generation threads
    profile: bool = True  # Whether to profile the data generation pipeline
    profiler: Profiler | None = None
    output_dir: str | None = None  # Directory to save generated data
    use_wandb: bool = False  # Whether to use Weights & Biases logging
    wandb_name: str | None = None  # Weights & Biases run name
    wandb_project: str | None = None  # Weights & Biases project name

    # --- Task type configuration ---
    task_type: str = "nav_to_obj"  # Task type: nav_to_obj

    # --- ProcTHOR dataset configuration ---
    scene_dataset: str = "procthor-10k"  # Name of the scene dataset to load
    data_split: str = "train"  # Data split to use

    robot_config: BaseRobotConfig | None = None

    # Camera configuration - using new unified camera system
    camera_config: RBY1MjcfCameraSystem = RBY1MjcfCameraSystem()

    # Task sampler configuration (imported from task_sampler_configs.py)
    task_sampler_config: NavToObjTaskSamplerConfig = NavToObjTaskSamplerConfig(
        task_sampler_class=NavToObjTaskSampler
    )

    # Task configuration (imported from task_configs.py)
    task_config: NavToObjTaskConfig = NavToObjTaskConfig(task_cls=NavToObjTask)
    task_config_preset: NavToObjTaskConfig | None = None

    # Policy configuration (imported from policy_configs.py)
    policy_config: BasePolicyConfig = AStarNavToObjPolicyConfig()

    def _init_policy_config(self) -> BasePolicyConfig:
        """Initialize policy config. Override in subclasses for dynamic initialization."""
        return self.policy_config

    def model_post_init(self, __context) -> None:
        """Initialize and validate configuration after Pydantic model initialization"""
        super().model_post_init(__context)

        try:
            self.policy_config = self._init_policy_config()
        except RuntimeError as e:
            # Check if this is a CUDA/GPU-related error
            error_msg = str(e)
            if "NVIDIA" in error_msg or "CUDA" in error_msg or "GPU" in error_msg:
                # No GPU available - this is expected on manager nodes that just coordinate jobs
                # Policy config will be initialized later on worker nodes that have GPUs
                print(
                    f"Warning: Skipping policy config initialization due to missing GPU: {error_msg}"
                )
                self.policy_config = None
            else:
                raise

        # Auto-create profiler instance if profiling is enabled
        if self.profile and self.profiler is None:
            self.profiler = Profiler()

    @property
    def tag(self) -> str:
        return "nav_to_obj_datagen"

    class SavedEpisode(Config):
        camera_config: RBY1MjcfCameraSystem | None = None  # Configuration for cameras and sensors
        robot_config: RBY1Config | None = None  # Configuration for the robot
        task_config: NavToObjTaskConfig | None = None  # Configuration for tasks
        task_cls_str: str | None = None


# Compact "tucked" arm pose for the single-arm Mobile Franka while it navigates.
# Reused from the Franka CAP robot config (a shipped, known-valid pose): the
# shoulder is pulled back and the elbow folded so the end-effector stays close
# to the base footprint and won't clip doorways/furniture during navigation.
# NOTE: validate visually in the spike before large runs; tune if it clips.
MOBILE_FRANKA_RETRACTED_ARM_QPOS: list[float] = [0.0, -1.5, 0.116, -2.45, 0.0, 0.842, 0.965]


class MobileFrankaNavToObjConfig(NavToObjBaseConfig):
    """Navigation-to-object config for the single-arm Mobile Franka.

    Same ``franka_droid`` embodiment as the manipulation pipeline mounted on a
    holonomic mobile base. During navigation the arm is held in a fixed,
    retracted pose (zero arm init noise + the nav policy never commands the arm),
    so the only commanded degrees of freedom are the base (x, y, yaw).
    """

    task_type: str = "nav_to_obj"

    # End the episode the instant the nav goal is reached (object visible within
    # the success distance) instead of running to ``task_horizon``. Without this,
    # a robot that arrives but whose final-orientation waypoint can't be satisfied
    # exactly keeps emitting the same pose and the trajectory accrues a long
    # frozen tail -- noise in the demonstration. With it, demos stop cleanly on
    # success.
    end_on_success: bool = True

    robot_config: BaseRobotConfig = MobileFrankaRobotConfig(
        init_qpos={
            "base": [0.0, 0.0, 0.0],
            "arm": MOBILE_FRANKA_RETRACTED_ARM_QPOS,
            "gripper": [0.00296, 0.00296],  # closed
        },
        # Zero arm noise => the arm starts identically every episode and, since
        # the nav policy only commands the base, it stays fixed throughout.
        init_qpos_noise_range={"arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    )

    # Sensible standalone default; the eiger datagen pipeline overrides this with
    # its own forward-facing nav camera rig via ``config.camera_config = ...``.
    camera_config: CameraSystemConfig = FrankaDroidCameraSystem()

    # The Mobile Franka has no RBY1 "head_camera"; use the forward-facing base
    # nav camera for the object-visibility success check.
    task_config: NavToObjTaskConfig = NavToObjTaskConfig(
        task_cls=NavToObjTask,
        visibility_camera_name="nav_camera",
        # Goal is to end within grasping range (the demo precedes a grasp), so a
        # trajectory only counts as success when the base is close to the object,
        # not at the lenient 1.5m default. Combined with
        # filter_for_successful_trajectories this keeps only grasp-close demos.
        succ_pos_threshold=0.8,
    )

    # Tame the A* policy's recovery behaviour for clean datagen. The brittle
    # distance-rate heuristic, when it fires, backtracks and re-routes the base --
    # the "aimless wandering" seen in otherwise-successful episodes. Setting
    # ``plan_max_retries=0`` makes a genuinely-stuck base terminate instead of
    # re-route (no wandering), while ``plan_fail_after_waypoint_steps=25`` still
    # ends a permanently-stalled episode promptly instead of spinning in place to
    # ``task_horizon``. 25 is well above the few steps needed to clear the dense
    # ~0.25m waypoints, so it won't trip on normal motion. The standoff params pull
    # the robot into grasping range of the object (default stops ~1.1-1.5m out).
    policy_config: BasePolicyConfig = AStarNavToObjPolicyConfig(
        plan_max_retries=0,
        plan_fail_after_waypoint_steps=25,
        nav_goal_distance_threshold=0.25,
        path_min_dist_to_target_center=0.4,
    )

    task_sampler_config: NavToObjTaskSamplerConfig = NavToObjTaskSamplerConfig(
        task_sampler_class=NavToObjTaskSampler,
        pickup_types=None,
        # Base footprint is [0.5, 0.5, 0.58]; ~0.35 keeps a small safety margin.
        robot_safety_radius=0.35,
        robot_object_z_offset=0.1,
        base_pose_sampling_radius_range=(4.0, 20.0),
        face_target=False,
        max_robot_placement_attempts=10,
        filter_for_successful_trajectories=True,
    )

    @property
    def tag(self) -> str:
        return "mobile_franka_nav_to_obj_datagen"

    class SavedEpisode(Config):
        camera_config: CameraSystemConfig | None = None
        robot_config: BaseRobotConfig | None = None
        task_config: NavToObjTaskConfig | None = None
        task_cls_str: str | None = None
