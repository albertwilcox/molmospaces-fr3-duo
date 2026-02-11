"""
Data generation configs for Franka move-to-pose tasks.

These configs subclass from the base_pick_config and are registered
for use in the data generation pipeline.
"""

import math
from pathlib import Path

from molmo_spaces.configs import BasePolicyConfig, BaseRobotConfig
from molmo_spaces.configs.base_open_task_configs import ClosingBaseConfig, OpeningBaseConfig
from molmo_spaces.configs.base_pick_and_place_color_configs import PickAndPlaceColorDataGenConfig

# This is here so that un-pickling benchmarks works
from molmo_spaces.configs.base_pick_and_place_configs import (
    PickAndPlaceDataGenConfig,
)
from molmo_spaces.configs.base_pick_and_place_next_to_configs import PickAndPlaceNextToDataGenConfig
from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import (
    FrankaDroidCameraSystem,
    FrankaEasyRandomizedDroidCameraSystem,
    FrankaGoProD405D455CameraSystem,
    FrankaOmniPurposeCameraSystem,
    FrankaRandomizedD405D455CameraSystem,
    FrankaRandomizedDroidCameraSystem,
)
from molmo_spaces.configs.policy_configs import (
    OpenClosePlannerPolicyConfig,
    PickPlannerPolicyConfig,
)
from molmo_spaces.configs.robot_configs import FloatingRUMRobotConfig, FrankaRobotConfig
from molmo_spaces.configs.task_sampler_configs import (
    OpenTaskSamplerConfig,
    PickAndPlaceNextToTaskSamplerConfig,
    PickAndPlaceTaskSamplerConfig,
    PickTaskSamplerConfig,
    RUMPickTaskSamplerConfig,
)
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.opening_task_samplers import OpenTaskSampler
from molmo_spaces.tasks.pick_and_place_next_to_task_sampler import PickAndPlaceNextToTaskSampler
from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.constants.object_constants import PICK_AND_PLACE_OBJECTS

# Oder of configs should be order the code is executed in
# scenes, robots, camera, task_sampler, policy, output


@register_config("FrankaPickDroidDataGenConfig")
class FrankaPickDroidDataGenConfig(PickBaseConfig):
    """Data generation config for Franka pick task with DROID-style fixed cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_droid_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_droid_datagen"


@register_config("FrankaPickGoProD405D455DataGenConfig")
class FrankaPickGoProD405D455DataGenConfig(PickBaseConfig):
    """Data generation config for Franka pick task with GoPro D405 cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaGoProD405D455CameraSystem = FrankaGoProD405D455CameraSystem()
    num_workers: int = 4
    task_horizon: int = 150
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_go_pro_d405_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_go_pro_d405_datagen"


@register_config("FrankaPickRandomizedDataGenConfig")
class FrankaPickRandomizedDataGenConfig(PickBaseConfig):
    """Data generation config for Franka pick task with randomized exocentric cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_randomized_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_randomized_datagen"


@register_config("RUMPickDataGenConfig")
class RUMPickDataGenConfig(PickBaseConfig):
    scene_dataset: str = "holodeck-objaverse"
    robot_config: FloatingRUMRobotConfig = FloatingRUMRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaRandomizedD405D455CameraSystem(
        img_resolution=(960, 720)
    )
    task_sampler_config: RUMPickTaskSamplerConfig = RUMPickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler, robot_object_z_offset=0
    )
    policy_config: PickPlannerPolicyConfig = PickPlannerPolicyConfig()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "rum_pick_v1"

    @property
    def tag(self) -> str:
        return "rum_pick_datagen"


@register_config("FrankaPickAndPlaceDataGenConfig")
class FrankaPickAndPlaceDataGenConfig(PickAndPlaceDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaRandomizedDroidCameraSystem = FrankaRandomizedDroidCameraSystem()
    policy_dt_ms: float = 66.0  # ~15hz
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_randomized_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_datagen"


@register_config("FrankaPickAndPlaceEasyDataGenConfig")
class FrankaPickAndPlaceEasyDataGenConfig(PickAndPlaceDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaEasyRandomizedDroidCameraSystem = FrankaEasyRandomizedDroidCameraSystem()
    policy_dt_ms: float = 66.0  # ~15hz
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_randomized_easy_v1"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_easy_datagen"


@register_config("FrankaPickAndPlaceDroidDataGenConfig")
class FrankaPickAndPlaceDroidDataGenConfig(PickAndPlaceDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_droid_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_droid_datagen"


@register_config("FrankaPickAndPlaceGoProD405D455DataGenConfig")
class FrankaPickAndPlaceGoProD405D455DataGenConfig(PickAndPlaceDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaGoProD405D455CameraSystem = FrankaGoProD405D455CameraSystem()
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_go_pro_d405_v1"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_go_pro_d405_datagen"


@register_config("FrankaPickAndPlaceNextToDataGenConfig")
class FrankaPickAndPlaceNextToDataGenConfig(PickAndPlaceNextToDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_next_to_randomized_v1"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_next_to_datagen"


@register_config("FrankaPickAndPlaceNextToDroidDataGenConfig")
class FrankaPickAndPlaceNextToDroidDataGenConfig(PickAndPlaceNextToDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_next_to_droid_v1"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_next_to_droid_datagen"


@register_config("FrankaPickAndPlaceColorDataGenConfig")
class FrankaPickAndPlaceColorDataGenConfig(PickAndPlaceColorDataGenConfig):
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_colors_randomized_v1"
    )
    wandb_project: str = "molmo-spaces-data-generation"
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_color_datagen"


@register_config("FrankaPickAndPlaceColorDroidDataGenConfig")
class FrankaPickAndPlaceColorDroidDataGenConfig(PickAndPlaceColorDataGenConfig):
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_colors_droid_randomized_v1"
    )
    wandb_project: str = "molmo-spaces-data-generation"
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_color_droid_datagen"


@register_config("FrankaOpenDataGenConfig")
class FrankaOpenDataGenConfig(OpeningBaseConfig):
    """Data generation config for Franka open task."""

    scene_dataset: str = "ithor"  # Name of the scene dataset to load
    data_split: str = "train"  # Data split to use
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0,  # 0.67 for close task, 0 for open task
    )
    policy_config: BasePolicyConfig = OpenClosePlannerPolicyConfig()
    task_horizon: int | None = 200  # Maximum number of steps per episode (if None, no time limit)
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "open_v1"

    @property
    def tag(self) -> str:
        return "franka_open_datagen"


@register_config("FrankaCloseDataGenConfig")
class FrankaCloseDataGenConfig(ClosingBaseConfig):
    """Data generation config for Franka open task."""

    scene_dataset: str = "ithor"  # Name of the scene dataset to load
    data_split: str = "train"  # Data split to use
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0.5,  # 0.67 for close task, 0 for open task
    )
    task_horizon: int | None = 200  # Maximum number of steps per episode (if None, no time limit)
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "close_v1"

    @property
    def tag(self) -> str:
        return "franka_close_datagen"


@register_config("FrankaPickAndPlaceGoProD405D455DataGenConfigDebug")
class FrankaPickAndPlaceGoProD405D455DataGenConfigDebug(FrankaPickAndPlaceDroidDataGenConfig):
    """Data generation config for Franka pick and place task with GoPro D405 cameras - deterministic version."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaGoProD405D455CameraSystem = FrankaGoProD405D455CameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        samples_per_house=10,
        max_tasks=100,
        # pickup_types=["Mug"],
        house_inds=[2],
    )
    num_workers: int = 1
    task_horizon: int = 100
    use_wandb: bool = False
    log_level: str = "debug"
    filter_for_successful_trajectories: bool = False
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_go_pro_d405_v1_debug"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_go_pro_d405_d455_datagen_debug"


@register_config("FrankaPickOmniCamConfig")
class FrankaPickOmniCamConfig(PickBaseConfig):
    """Data generation config for Franka pick task with Omni-directional cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_omni_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_omni_datagen"


@register_config("FrankaPickAndPlaceOmniCamConfig")
class FrankaPickAndPlaceOmniCamConfig(PickAndPlaceDataGenConfig):
    """Data generation config for Franka pick task with Omni-directional cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_omni_v1"
    log_level: str = "info"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_omnicam_datagen"


# PickAndPlaceNextToDataGenConfig
@register_config("FrankaPickAndPlaceNextToOmniCamConfig")
class FrankaPickAndPlaceNextToOmniCamConfig(PickAndPlaceNextToDataGenConfig):
    """Data generation config for Franka pick task with Omni-directional cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_next_to_omni_v1"
    )
    log_level: str = "info"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_next_to_omnicam_datagen"


# PickAndPlaceColorDataGenConfig
@register_config("FrankaPickAndPlaceColorOmniCamConfig")
class FrankaPickAndPlaceColorOmniCamConfig(PickAndPlaceColorDataGenConfig):
    """Data generation config for Franka pick task with Omni-directional cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_color_omni_v1"
    log_level: str = "info"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_color_omnicam_datagen"


################################################################################
# Benchmark configs
################################################################################


@register_config("FrankaPickDroidMiniBench")
class FrankaPickDroidMiniBench(PickBaseConfig):
    scene_dataset: str = "procthor-10k"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        samples_per_house=40,
        house_inds=list(range(101)),
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_droid_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_minbench"


@register_config("FrankaPickandPlaceMiniBench")
class FrankaPickandPlaceDroidMiniBench(PickAndPlaceDataGenConfig):
    scene_dataset: str = "procthor-10k"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=40,
        house_inds=list(range(101)),
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_droid_v1"

    @property
    def tag(self) -> str:
        return "franka_pickandplace_minbench"


@register_config("FrankaPickDroidBench")
class FrankaPickDroidBench(PickBaseConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        samples_per_house=40,
        house_inds=list(range(101)),
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_obja_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_bench"


@register_config("FrankaPickandPlaceDroidBench")
class FrankaPickandPlaceDroidBench(PickAndPlaceDataGenConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=40,
        house_inds=list(range(101)),
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_obja_v1"

    @property
    def tag(self) -> str:
        return "franka_pickandplace_bench"


@register_config("FrankaPickandPlaceNextToDroidBench")
class FrankaPickandPlaceNextToDroidBench(PickAndPlaceNextToDataGenConfig):
    scene_dataset: str = "procthor-objaverse"
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_next_to_obja_v1"

    @property
    def tag(self) -> str:
        return "franka_pickandplacenextto_bench"


@register_config("FrankaPickandPlaceColorDroidBench")
class FrankaPickandPlaceColorDroidBench(PickAndPlaceColorDataGenConfig):
    """Data generation config for Franka pick task with DROID-style fixed cameras."""

    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_color_obja_v1"
    scene_dataset: str = "procthor-objaverse"

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()

    @property
    def tag(self) -> str:
        return "franka_pickandplacecolor_bench"


@register_config("FrankaOpenHardBench")
class FrankaOpenHardBench(OpeningBaseConfig):
    """Data generation config for Franka open task."""

    scene_dataset: str = "ithor"  # Name of the scene dataset to load
    data_split: str = "val"  # Data split to use
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0,  # 0.67 for close task, 0 for open task
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    policy_config: BasePolicyConfig = OpenClosePlannerPolicyConfig()
    task_horizon: int | None = 200  # Maximum number of steps per episode (if None, no time limit)
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "open_bench"

    @property
    def tag(self) -> str:
        return "franka_open_hard_bench"


@register_config("FrankaCloseHardBench")
class FrankaCloseHardBench(ClosingBaseConfig):
    """Data generation config for Franka open task."""

    scene_dataset: str = "ithor"  # Name of the scene dataset to load
    data_split: str = "val"  # Data split to use
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0.5,  # 0.67 for close task, 0 for open task
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    task_horizon: int | None = 200  # Maximum number of steps per episode (if None, no time limit)
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "close_bench"

    @property
    def tag(self) -> str:
        return "franka_close_hard_bench"


@register_config("FrankaPickHardBench")
class FrankaPickHardBench(PickBaseConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_hard_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_hard_bench"


@register_config("FrankaPickandPlaceHardBench")
class FrankaPickandPlaceHardBench(PickAndPlaceDataGenConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )

    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_hard_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_hard_bench"


@register_config("FrankaPickandPlaceNextToHardBench")
class FrankaPickandPlaceNextToHardBench(PickAndPlaceNextToDataGenConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )

    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceNextToTaskSamplerConfig(
        task_sampler_class=PickAndPlaceNextToTaskSampler,
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_next_to_hard_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_next_to_hard_bench"
