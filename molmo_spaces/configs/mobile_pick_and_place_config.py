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
    # receptacle on a *different elevated surface* (table / counter / shelf) a
    # real navigation distance away, so the episode is genuinely
    # navigate -> grasp -> navigate -> place AND the place target is at a
    # natural, reachable manipulation height (never on the floor).
    far_place_on_elevated_surface: bool = True
    # Legacy: stand the receptacle on the floor far away. Kept for backward
    # compatibility but disabled by default -- floor placements put the place
    # target below the arm's comfortable workspace and produced unreachable
    # (and unnatural) placements.
    far_place_on_floor: bool = False
    # A candidate place surface must have its top at least this far above the
    # floor to count as "elevated" (excludes rugs / floor-level geoms).
    elevated_min_height_m: float = 0.30
    # A candidate place surface must have at least this much flat top area (m^2)
    # so the receptacle actually fits on it.
    elevated_min_surface_area_m2: float = 0.06
    # Distance band (m) from the pickup object to stand the place receptacle.
    # Widened from a tight [2,5] band so far *elevated* surfaces (which are
    # sparser than the floor) are found more often before falling back to the
    # same-surface (close, still elevated) placement.
    far_min_object_to_receptacle_dist: float = 1.5
    far_max_object_to_receptacle_dist: float = 8.0

    # --- Same-room placement filter -----------------------------------------
    # Restrict the place receptacle to a surface in the *same room* as the
    # pickup object. This keeps the task an intra-room mobile pick-and-place
    # (navigate -> grasp -> navigate -> place all within one room), which is the
    # in-scope regime for the project. Room membership is read from the scene
    # object body-name convention ``..._<room_id>`` (the trailing field); a
    # candidate elevated surface is same-room iff its supporting body's room id
    # matches the pickup object's. When enabled, the far-distance band lower
    # bound is relaxed to ``same_room_min_object_to_receptacle_dist`` because
    # intra-room navigation distances are naturally shorter, so we still exercise
    # a genuine (if shorter) place-nav segment without starving candidates.
    same_room_place_only: bool = True
    same_room_min_object_to_receptacle_dist: float = 0.8


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

    # --- Manip-standoff snap collision gate ---------------------------------
    # After navigation the FSM freezes the base and snaps it to a manip standoff
    # that is IK-feasible for the grasp/place. That snap is instantaneous, so if
    # navigation parked the base on the wrong side of a wall (e.g. it could not
    # traverse to the receptacle), the nearest IK-feasible standoff lies across
    # the wall and the base visibly teleports THROUGH it. We reject any standoff
    # whose straight-line snap from the parked pose crosses a wall/obstacle,
    # tested against the scene occupancy map (the same agent-radius-dilated
    # ProcTHOR map place_robot_near samples from). Because a legitimate manip
    # standoff sits right next to the target furniture (map-occupied), we only
    # test the MIDDLE of the snap segment -- samples farther than
    # ``manip_snap_endpoint_margin_m`` from BOTH endpoints -- so furniture
    # adjacency at either end is ignored while a wall in the interior is caught.
    # Snaps shorter than 2x the margin therefore have no interior to test and
    # always pass (they are never wall crossings), so ordinary short standoff
    # repositioning is unaffected. If every candidate crosses a wall the segment
    # fails and is retried/relocated rather than emitting a wall-crossing demo.
    manip_snap_collision_gate_enabled: bool = True
    # Spacing (metres) at which the snap segment interior is sampled.
    manip_snap_collision_sample_spacing_m: float = 0.15
    # Snap-segment samples within this distance (metres) of either endpoint are
    # skipped (furniture is legitimately occupied next to a manip standoff).
    manip_snap_endpoint_margin_m: float = 0.6

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

    # --- Pre-navigation arm retract (stow pose) -----------------------------
    # Before/while navigating, retract the arm to a compact, natural "home" tuck
    # instead of pinning it at whatever (possibly extended, near-joint-limit)
    # pose it happened to be in when the nav segment began. Watching the raw
    # demos, the arm would immediately straighten out toward a joint limit the
    # moment navigation started (an unnatural extended-arm drive). We instead
    # ramp the held setpoint from the measured pose toward this stow config at
    # the arm velocity cap (arm_max_vel_rad_s), so the arm tucks in smoothly and
    # then drives with a retracted arm. The target matches the canonical
    # robocasa mobile-manipulation home pose (per-joint mean over the reference
    # dataset /mnt/disk/robocasa_data, std ~0.02 rad): a compact FR3 tuck.
    nav_arm_retract_enabled: bool = True
    # 7-DOF FR3 stow/home joint config (rad), joint1..joint7 order. Matches the
    # robocasa reference dataset's reset arm configuration.
    nav_arm_stow_qpos: tuple[float, ...] = (
        -0.02,
        -1.04,
        -0.02,
        -2.27,
        0.04,
        1.52,
        0.70,
    )

    # --- Base-motion slew limiter (nav smoothness) --------------------------
    # The pure-pursuit follower emits a base *pose* target a fixed look-ahead
    # ahead of the current pose; the holonomic base position servo then charges
    # toward it at whatever speed it can (~2 m/s). That translation/turn drags
    # the position-held stowed (or grasped) arm hard enough to spike its
    # measured joint velocity past the FR3 hardware limit (a visible, unnatural
    # arm jiggle while driving). We slew-limit the base pose command each nav
    # step: bound the per-step translation and yaw of the *target* relative to
    # the current base pose, and ramp the translation cap up over the first few
    # steps of each nav segment so the base accelerates gently instead of
    # lurching from rest. This keeps genuine navigation intact (the base still
    # follows the same path, just smoothly) while eliminating the arm fling.
    base_slew_enabled: bool = True
    # Max base translation speed during nav (m/s). The holonomic base is a
    # critically-damped position servo (kp=25000, ζ=1) that tracks the
    # slew-capped setpoint within a control step, so this cap is effectively the
    # cruise speed. Raised from 1.1 -> 1.6 (real indoor bases cruise ~1-1.5 m/s)
    # to speed navigation up; the stowed/held arm is separately position-held and
    # velocity-clamped (arm_smoothing) so the faster base does not fling it.
    base_max_speed_m_s: float = 1.6
    # Max base yaw rate during nav (rad/s). Raised 1.5 -> 2.0 to match.
    base_max_yaw_rate_rad_s: float = 2.0
    # Steps over which the translation speed cap ramps from ~0 to the max at the
    # start of each nav segment (gentle acceleration).
    base_accel_ramp_steps: int = 8

    # --- Smooth base approach (nav -> manip standoff) -----------------------
    # When navigation finishes, the manip feasibility search selects a standoff
    # base pose that generally differs from where the base parked and snaps the
    # base there in a single frame -- a base "teleport" in the recorded data. We
    # instead drive the base from the parked pose to the standoff over several
    # slew-limited steps (labelled as navigation, arm stowed), so the recorded
    # base motion stays continuous and under the teleport-detection threshold.
    #
    # DISABLED BY DEFAULT: the parked->standoff gap is small (~0.1-0.4 m) and is
    # NOT the gross "teleport back to the start" the debug videos show (that is
    # the retry checkpoint-restore, which is truncated out of saved data).
    # Integrating physics (mj_step) while the pinned base slides the last ~0.1 m
    # into the manipulation standoff perturbs the scene enough to corrupt the
    # immediately-following grasp (empirically: PICK "Object is not in grasp!"
    # gross misses, dropping end-to-end success from ~8% to ~0% in a 24-house
    # A/B). The single-frame standoff snap the direct path leaves is small and
    # visually imperceptible, so we keep the (grasp-preserving) direct snap.
    base_approach_enabled: bool = False
    # Position/heading tolerance for declaring the approach complete (the final
    # re-pin to the exact standoff is then far below the teleport threshold).
    base_approach_pos_tol_m: float = 0.04
    base_approach_yaw_tol_rad: float = 0.05
    # Hard cap on approach steps so a stuck approach still commits to the manip
    # phase (a rare, small residual snap) rather than looping forever.
    base_approach_max_steps: int = 60

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
