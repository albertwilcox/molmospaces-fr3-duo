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

    # --- Direct-on-furniture place destinations ------------------------------
    # By default the place target is a spawned receptacle object (e.g. a bowl)
    # stood on a far surface, and the pickup object is placed INTO it. To
    # diversify place destinations, with probability ``place_on_furniture_prob``
    # we instead redirect the place target to an existing same-room furniture
    # body (cabinet / dresser / nightstand / small table): the pickup object is
    # then placed directly ON that furniture's top surface -- no receptacle
    # spawned -- and, when the furniture already holds other items, effectively
    # NEXT TO them. This reuses the existing receptacle-support success test and
    # placement planner unchanged (the furniture simply plays the receptacle
    # role). Only furniture whose top-surface footprint is small enough for its
    # centre to remain within arm reach from a standoff is eligible, so
    # placement feasibility (and thus the overall success rate) is preserved;
    # oversized furniture (e.g. a large bed) is skipped and the bowl path is
    # kept for that episode.
    place_on_furniture_prob: float = 0.4
    # Max XY half-extent (m) of an eligible furniture footprint. The base parks
    # ~``manip_standoff_radius`` from the furniture and the object is placed at
    # the furniture centre, so the centre must stay within the arm's reach; a
    # 0.4 m half-extent keeps centre-to-base distance within the Franka envelope.
    furniture_place_max_half_extent_m: float = 0.40

    # --- Broad furniture placement (any flat furniture is a place target) ----
    # When True, treat *any* eligible flat-topped furniture (bed / sofa / table /
    # counter / desk / dresser ...) as a valid place destination, not just small
    # furniture whose CENTRE is reachable. This is safe because the runtime place
    # builder no longer requires the receptacle centre: ``_get_placement_poses``
    # falls back to ``_nearest_reachable_place_pose``, which searches the whole
    # furniture top footprint for the IK-reachable point nearest the base. So a
    # large bed/table only needs *some* reachable point on its top (which the base
    # parks beside), not a reachable centre -- and plopping an object onto a broad
    # flat surface is an EASIER placement than fitting it into a small bowl.
    # Enabling this:
    #   * lifts the small-footprint gate (``furniture_place_max_half_extent_m`` is
    #     ignored for large furniture; a reachable-top-point check is used
    #     instead, mirroring the runtime),
    #   * drops the body-origin-under-surface guard (only needed for the old
    #     centre-drop path), and
    #   * raises the furniture-redirect probability to
    #     ``broad_furniture_place_prob``.
    # Default False preserves the exact prior (small-furniture-only) behaviour.
    prefer_furniture_place: bool = False
    broad_furniture_place_prob: float = 0.75
    # Minimum top-surface XY half-extent (m) for a large furniture body to be an
    # eligible broad place target (excludes thin ledges / chair seats that read as
    # furniture but are not sensible drop surfaces).
    broad_furniture_min_half_extent_m: float = 0.20

    # --- Sample-time place-reachability verification -------------------------
    # The pickup loop already rejects objects with no feasible grasp, but the
    # place receptacle was previously accepted on collision-free placement alone.
    # Genuinely-unreachable receptacles (too high, too deep in a corner, or held
    # at an awkward carried orientation) then wasted whole episodes on the
    # runtime "no reachable base standoff for place" failure. When enabled, the
    # sampler replicates the placement planner's pose construction (grasp
    # orientation inherited, translated to the receptacle top) and the FSM's ring
    # standoff search: it accepts the receptacle only if some (standoff, grasp)
    # pair yields IK-feasible pre-place AND place poses. Acceptance therefore
    # guarantees the runtime PLACE phase can find a feasible standoff, removing
    # the dominant place failure mode. Fail-open: if grasps/metadata are missing
    # the receptacle is kept (never blocks on incomplete data).
    verify_place_reachable: bool = True
    # Cap on candidate carried-grasp orientations probed per receptacle.
    place_reachable_max_grasps: int = 8
    # Number of ring standoff angles probed per radius (radii reuse the pickup
    # standoff band ``manip_standoff_radius_range``).
    place_reachable_standoff_angles: int = 12
    # Vertical clearance (m) added above the receptacle top for the probe place
    # pose (mirrors the planner's small ``place_z_offset``).
    # Number of collision-free receptacle-facing standoffs (sampled via the same
    # ``place_robot_near`` sampler the runtime uses for its place nav-goal hint)
    # that the probe IK-verifies before declaring a receptacle unreachable.
    place_reachable_standoff_tries: int = 24
    place_reachable_z_offset_m: float = 0.05
    # Runtime place-fallback replication (see ``_find_place_standoff``): when the
    # receptacle-CENTRE place pose is IK-unreachable the runtime places at the
    # nearest reachable point on the receptacle top footprint (a grid search,
    # ``_nearest_reachable_place_pose``). The probe mirrors that grid so its
    # reachability verdict matches the runtime -- removing the centre-only
    # false-positive rejections that previously forced ``place_reject_on_unreachable``
    # off. These mirror the policy's ``place_edge_margin_m`` / ``place_search_grid_n``.
    place_reachable_edge_margin_m: float = 0.03
    place_reachable_search_grid_n: int = 5
    # If True, reject a receptacle when the sample-time probe finds no
    # place-feasible standoff, advancing the selection loop to another candidate.
    # The probe now replicates the runtime's place fallback (receptacle-centre
    # place pose, then a nearest-to-base grid search over the receptacle top
    # footprint -- see ``place_reachable_search_grid_n``), so it no longer
    # false-rejects receptacles the runtime could place on off-centre. That makes
    # rejection safe in principle; keep the default False until an A/B confirms
    # the higher-fidelity probe does not regress otherwise-succeeding houses,
    # then flip to True to skip guaranteed-unreachable place targets at sample
    # time (avoiding a wasted full-length PLACE-fail rollout).
    place_reject_on_unreachable: bool = False

    # If True, exclude candidate place surfaces that belong to an *enclosed*
    # container appliance (fridge/oven/microwave/dishwasher/cabinet/drawer etc.)
    # from the far-elevated-surface search, so the spawned place receptacle is
    # never stood on a shelf inside a fridge/cabinet. The mobile base cannot
    # maneuver into an appliance interior to place (and grazing the door swings
    # it shut), which is a frequent PLACE-phase failure. Unlike
    # ``place_reject_on_unreachable`` this only PRUNES the candidate list -- open
    # tables/counters/shelves remain, so it changes the place distribution
    # without ever exhausting a house (no intrinsic-failure risk). Default False
    # keeps the exact prior candidate set.
    place_exclude_enclosed_container_surfaces: bool = False

    # When the place-feasibility probe finds no reachable standoff for the spawned
    # receptacle at its initially sampled spot, RE-SAMPLE the receptacle to a
    # different point on the far elevated surface and re-probe, up to this many
    # times, keeping the first spot from which the arm can place. Because the
    # probe is strictly more conservative than the runtime place, a probe-verified
    # spot is high-confidence for the runtime, so this makes the place target
    # reachable by construction wherever a reachable spot exists -- directly
    # attacking the dominant "no reachable base standoff for place" failure and
    # avoiding a wasted full-length rollout on a guaranteed PLACE failure. Set to
    # 0 to disable re-sampling (single placement attempt, prior behavior).
    #
    # DEFAULT 0 (disabled): empirically the base-locked ring-IK probe is too
    # conservative for the ELEVATED far-surface placements this sampler produces
    # -- it fails to verify spots the runtime can actually place on (it verified
    # none across repeated re-samples on validated houses), so re-sampling found
    # no reachable spot to switch to and only added sample-time overhead before
    # restoring the original placement. Re-enable once the probe replicates the
    # runtime's held-object place-standoff search (higher fidelity), at which
    # point re-sampling can make the place target reachable by construction.
    place_receptacle_resample_tries: int = 0

    # --- Overhead clearance for pickup candidates ------------------------- #
    # Reject pickup candidates that have OTHER scene geometry directly above
    # them within a vertical column: a top-down / approach-from-above grasp
    # would collide with the overhanging object (e.g. an item on a shelf under
    # the next shelf, or a mug tucked under a cabinet lip), which leads to
    # empty-grasp misfires. Enabled by default; disable to restore the prior
    # (clearance-agnostic) candidate pool.
    require_overhead_clearance: bool = True
    # Height (m) of the clear column required directly above a candidate's top
    # face. Roughly the gripper + approach standoff the arm needs from above.
    overhead_clearance_height_m: float = 0.25
    # Horizontal shrink (m) applied to the candidate's XY footprint before
    # testing for overhang, so a geom merely flush beside the object (touching
    # its side, not truly above it) does not disqualify it.
    overhead_clearance_xy_margin_m: float = -0.02
    # Minimum XY overlap fraction of the candidate footprint that an overhead
    # geom must cover to count as blocking (filters slivers / grazing contacts).
    overhead_clearance_min_overlap_frac: float = 0.10


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

    # --- Manipulability / reach-margin base scoring -------------------------
    # The manip base search used to accept the FIRST standoff from which every
    # grasp/place waypoint has *some* base-locked IK solution. "IK exists" is a
    # binary that says nothing about how close that solution sits to the arm's
    # joint limits, and the dominant end-to-end failure ("object not in grasp!")
    # is precisely a grasp executed at the reach/joint-limit boundary (strongly
    # correlated with pickup-object height near the arm's vertical reach limit).
    # Instead of accepting the first feasible standoff we score a bounded set of
    # feasible standoffs by their reach margin -- the minimum fractional distance
    # of the IK solution's arm joints from their nearer position limit across all
    # waypoints -- and commit to the most interior (most robust) one. A small
    # proximity penalty keeps the chosen standoff close to where navigation
    # parked (small, imperceptible base snaps) unless a farther standoff buys a
    # materially larger reach margin. Set enabled=False to restore the legacy
    # accept-first-feasible behaviour.
    manip_reach_scoring_enabled: bool = True
    # Number of all-waypoint-feasible standoffs to score before committing to the
    # best (bounds the extra IK cost of scanning past the first feasible one).
    manip_reach_scored_candidates: int = 6
    # Reach-margin penalty per metre the standoff sits from the parked base, so
    # a closer standoff wins unless a farther one is meaningfully more interior.
    manip_reach_proximity_penalty_per_m: float = 0.15

    # --- Grasp verification + regrasp ---------------------------------------
    # The grasp primitive can report "done" while the gripper closed on empty
    # space (a marginal reach-limit grasp), after which the FSM would happily
    # navigate to the receptacle carrying nothing and only discover the miss at
    # the final success judge -- wasting the whole episode. After the PICK lift
    # completes we verify the pickup object actually rose with the gripper
    # (its world-z increased by at least ``grasp_verify_min_rise_m`` relative to
    # its pre-pick resting height). A miss re-runs the PICK segment, which the
    # existing retry rotates to a fresh standoff/approach angle for a different
    # grasp, instead of proceeding empty-handed.
    grasp_verify_enabled: bool = True
    # Minimum object world-z rise (metres) to consider the object grasped+lifted.
    # Below the smallest lift height (0.05 m) so a successful shallow lift passes.
    grasp_verify_min_rise_m: float = 0.03
    # If the object did not clearly rise (e.g. lift skipped at the reach limit),
    # still accept the grasp when the object is within this distance (metres) of
    # the gripper TCP -- it is in the hand. An empty grasp leaves it ~a lift
    # height away from the raised TCP.
    grasp_verify_max_tcp_dist_m: float = 0.12

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
    # PLACE-entry smooth approach. Unlike PICK, the object is already grasp-locked
    # (rigidly fixed to the gripper) when entering PLACE, so sliding the base to
    # the place standoff over snap-held steps cannot corrupt a grasp. Enabling it
    # eliminates the manip-entry base "teleport" in the observation stream at the
    # NAV_TO_RECEPTACLE -> PLACE transition (e.g. a ~0.2 m single-frame base snap)
    # without the grasp-corruption risk that keeps ``base_approach_enabled`` off.
    base_approach_place_enabled: bool = True
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
