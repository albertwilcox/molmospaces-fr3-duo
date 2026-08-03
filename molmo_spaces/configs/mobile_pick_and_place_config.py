"""Configuration for the CONTINUOUS mobile pick-and-place-with-navigation task.

This composes the two already-working pipelines in the fork:

* Navigation: the Mobile Franka (holonomic base + arm + gripper) driven by the
  A*/pure-pursuit planner (``MobileFrankaNavToObjConfig``).
* Fixed-base manipulation: the pick-and-place planner
  (``PickAndPlaceDataGenConfig``) which grasps an object and drops it in/on a
  receptacle.

The combined expert (:class:`MobilePickAndPlaceStateMachinePolicy`) navigates to
the pickup object, runs ONLY the pick portion parked at the object, navigates to
the receptacle, then runs ONLY the place portion parked at the receptacle. Both
manipulation sub-trajectories are (re)built from the *actual* parked base pose,
because manipulation target poses are world-frame and IK is base-relative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.configs.base_nav_to_obj_config import MOBILE_FRANKA_RETRACTED_ARM_QPOS
from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import CameraSystemConfig, FrankaDroidCameraSystem
from molmo_spaces.configs.policy_configs import (
    AStarNavToObjPolicyConfig,
    BasePolicyConfig,
    PickAndPlacePlannerPolicyConfig,
)
from molmo_spaces.configs.robot_configs import BaseRobotConfig, MobileFrankaRobotConfig
from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
from molmo_spaces.configs.task_sampler_configs import PickAndPlaceTaskSamplerConfig

if TYPE_CHECKING:
    pass


class MobilePickAndPlaceTaskConfig(PickAndPlaceTaskConfig):
    """Task config for mobile pick-and-place.

    Inherits the fixed-base pick-and-place success fields (``pickup_obj_name``,
    ``place_receptacle_name``, support/displacement thresholds) and adds the
    navigation goal-pose success fields used to judge the pre-grasp / pre-place
    standoff the base parks at.
    """

    # --- Navigation standoff (pre-grasp / pre-place) success fields ------------
    # The base can never reach the object/receptacle *centre* (it stops at the
    # furniture edge), so nav arrival is judged on a standoff ring, and — when
    # ``succ_use_goal_pose`` — on reaching the planned pre-grasp goal pose.
    nav_succ_pos_threshold: float = 0.8  # standoff ring radius (m) for goal selection
    visibility_camera_name: str = "nav_camera"
    require_object_visible: bool = False
    succ_use_goal_pose: bool = True
    succ_goal_pos_threshold: float = 0.30  # base->goal planar tolerance (m)
    succ_goal_yaw_threshold: float = float(np.deg2rad(25))
    max_pregrasp_standoff_m: float = 1.75

    # Feasibility-verified base poses recorded by the sampler (7D x,y,z,qw,qx,qy,qz,
    # world frame). ``robot_base_pose`` (inherited) is the grasp-feasible pose near
    # the pickup object; ``place_robot_base_pose`` is the place-feasible pose near
    # the receptacle. The FSM navigates the base to these instead of re-sampling
    # the closest navigable cell, transferring the fixed-base pipeline's
    # feasibility guarantee to the mobile pipeline.
    place_robot_base_pose: list[float] | None = None


class MobilePickAndPlaceTaskSamplerConfig(PickAndPlaceTaskSamplerConfig):
    """Sampler config: reuse the pick-and-place object/receptacle selection but
    place the mobile robot at a *navigable* start pose far from the object."""

    task_sampler_class: type | None = None

    # A mobile base sits on the floor; keep its z fixed here rather than deriving
    # it from the (tabletop) object height as the fixed-base sampler does.
    mobile_base_z: float = 0.1

    # Standoff radius range for the *feasibility* placement near the pickup
    # object (representative of where navigation parks the base for the grasp).
    manip_standoff_radius_range: tuple[float, float] = (0.35, 0.7)

    # Radius range for the navigable START pose (far enough that navigation is
    # genuinely exercised, close enough that the smoke test stays fast/robust).
    nav_start_radius_range: tuple[float, float] = (2.0, 6.0)

    # Robot footprint radius for occupancy-map placement + A* agent radius.
    robot_safety_radius: float = 0.3
    # The nav start pose is far away; visibility of the object from the start is
    # neither required nor desirable, so skip the placement visibility check.
    check_robot_placement_visibility: bool = False
    # Two receptacles preloaded is plenty for the vertical slice.
    num_place_receptacles: int = 2

    # --- Far-apart place receptacle (genuine second navigation segment) -----
    # The fixed-base sampler places the receptacle on the *same* support surface
    # as the pickup object (within ~0.5 m), so the base barely moves between the
    # grasp and the place. For mobile pick-and-place we instead stand the
    # receptacle on the floor a real navigation distance away, so the episode is
    # genuinely navigate -> grasp -> navigate -> place.
    far_place_on_floor: bool = True
    # Distance band (m) from the pickup object to stand the place receptacle.
    far_min_object_to_receptacle_dist: float = 2.0
    far_max_object_to_receptacle_dist: float = 5.0


class MobilePickAndPlacePolicyConfig(BasePolicyConfig):
    """Config for the mobile pick-and-place state-machine expert.

    Carries the two sub-policy configs it composes (navigation + manipulation)
    plus the navigation standoff parameters it drives them with.
    """

    policy_cls: type = None  # set in model_post_init to avoid circular imports
    policy_type: str = "planner"

    # Navigation sub-policy (A* + closed-loop pure-pursuit follower). Tuned like
    # ``MobileFrankaNavToObjConfig`` so the base parks within grasping range.
    nav_policy_config: AStarNavToObjPolicyConfig = AStarNavToObjPolicyConfig(
        plan_max_retries=0,
        plan_fail_after_waypoint_steps=25,
        nav_goal_distance_threshold=0.25,
        path_min_dist_to_target_center=0.4,
        use_pure_pursuit=True,
    )

    # Manipulation sub-policy (pick-and-place planner). Only the pick / place
    # halves of its trajectory are executed, each rebuilt from the parked base.
    manip_policy_config: PickAndPlacePlannerPolicyConfig = PickAndPlacePlannerPolicyConfig()

    # Standoff ring radius the navigation goal sampler aims for (metres).
    nav_standoff_succ_threshold: float = 0.8
    # Camera used by the (default-off) nav goal-visibility gate.
    nav_visibility_camera_name: str = "nav_camera"

    # --- Arm-motion safety clamp (kinematic smoothness) ---------------------
    # The manipulation planner interpolates the TCP target smoothly in task
    # space and solves *stateless* IK per control step. Between consecutive
    # targets the base-locked IK can jump branches (elbow flip), producing
    # single-step joint jumps of tens of rad/s (far past the FR3 hardware limit
    # of ~2.6 rad/s) and occasionally solutions outside the joint limits. We
    # clamp the commanded arm joint deltas to a natural per-step velocity and
    # keep them inside the joint limits so the executed motion is smooth and
    # feasible. Normal manipulation moves the arm well under this cap, so the
    # clamp only smooths the pathological IK-branch-flip spikes.
    arm_smoothing_enabled: bool = True
    # Max commanded arm joint speed (rad/s). Well under every FR3 hardware
    # velocity limit (min 2.62 rad/s) yet above normal planned motion (~0.5).
    arm_max_vel_rad_s: float = 1.5
    # Keep commanded joints this far (rad) inside their position limits.
    arm_pos_limit_margin_rad: float = 0.05

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.mobile_pick_and_place_state_machine_policy import (
                MobilePickAndPlaceStateMachinePolicy,
            )

            self.policy_cls = MobilePickAndPlaceStateMachinePolicy


class MobileFrankaPickAndPlaceConfig(PickBaseConfig):
    """Combined mobile pick-and-place-with-navigation config."""

    task_type: str = "mobile_pick_and_place"
    num_workers: int = 1

    # Mobile Franka: base + arm + gripper. Arm starts in the retracted stow pose
    # (zero noise) so it stays clear of doorways/furniture during navigation; the
    # gripper starts open, ready for the first grasp.
    robot_config: BaseRobotConfig = MobileFrankaRobotConfig(
        init_qpos={
            "base": [0.0, 0.0, 0.0],
            "arm": MOBILE_FRANKA_RETRACTED_ARM_QPOS,
            "gripper": [0.04, 0.04],  # open
        },
        init_qpos_noise_range={"arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    )

    camera_config: CameraSystemConfig = FrankaDroidCameraSystem()

    task_sampler_config: MobilePickAndPlaceTaskSamplerConfig = MobilePickAndPlaceTaskSamplerConfig(
        task_sampler_class=None,  # set in model_post_init
        pickup_types=[],
        samples_per_house=5,
    )

    task_config: MobilePickAndPlaceTaskConfig = MobilePickAndPlaceTaskConfig(task_cls=None)

    policy_config: BasePolicyConfig = MobilePickAndPlacePolicyConfig()

    def model_post_init(self, __context) -> None:
        # Wire the task/sampler classes lazily to avoid import cycles.
        if self.task_sampler_config.task_sampler_class is None:
            from molmo_spaces.tasks.mobile_pick_and_place_task_sampler import (
                MobilePickAndPlaceTaskSampler,
            )

            self.task_sampler_config.task_sampler_class = MobilePickAndPlaceTaskSampler
        if self.task_config.task_cls is None:
            from molmo_spaces.tasks.mobile_pick_and_place_task import MobilePickAndPlaceTask

            self.task_config.task_cls = MobilePickAndPlaceTask
        super().model_post_init(__context)

    @property
    def tag(self) -> str:
        return "mobile_franka_pick_and_place_datagen"

    class SavedEpisode(Config):
        camera_config: CameraSystemConfig | None = None
        robot_config: BaseRobotConfig | None = None
        task_config: MobilePickAndPlaceTaskConfig | None = None
        task_cls_str: str | None = None
