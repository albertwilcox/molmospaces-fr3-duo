"""State-machine expert for CONTINUOUS mobile pick-and-place-with-navigation.

The expert composes the two working pipelines in the fork without duplicating
their logic:

* an A* / pure-pursuit navigation sub-policy (drives only the base), and
* the pick-and-place manipulation planner (drives only the arm + gripper).

Because manipulation target poses are world-frame and IK is solved relative to
the *current* base pose, the manipulation primitives cannot be precomputed. The
FSM therefore builds the pick primitives only AFTER navigation has parked the
base at the object, and the place primitives only AFTER navigation has parked the
base at the receptacle. A dedicated manipulation subclass locks the base during
IK so the arm (never the base) satisfies the reach.

Phases: ``NAV_TO_OBJ`` → ``PICK`` → ``NAV_TO_RECEPTACLE`` → ``PLACE`` → ``DONE``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import mujoco
from mujoco import MjSpec

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.task_configs import NavToObjTaskConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.policy.base_policy import PlannerPolicy
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    GripperAction,
    TCPMoveSegment,
    TCPMoveSequence,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
    PickAndPlacePlannerPolicy,
)
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
from molmo_spaces.utils.grasp_sample import add_grasp_collision_bodies, compute_grasp_pose
from molmo_spaces.utils.pose import pos_quat_to_pose_mat, pose_mat_to_pos_quat

log = logging.getLogger(__name__)

# Phase labels.
NAV_TO_OBJ = "NAV_TO_OBJ"
PICK = "PICK"
NAV_TO_RECEPTACLE = "NAV_TO_RECEPTACLE"
PLACE = "PLACE"
DONE = "DONE"

# Internal control-flow-only phases: the smooth base "approach" that drives the
# base from where navigation parked to the manipulation standoff selected by the
# feasibility search, over several slew-limited steps, instead of teleporting it
# there in one frame (which showed up in the data as a base "teleport"). These
# are reported as their parent navigation phase for subtask labelling (the base
# is still driving toward its target with the arm stowed), so they never appear
# as distinct phases in ``get_all_phases``.
APPROACH_PICK = "APPROACH_PICK"
APPROACH_PLACE = "APPROACH_PLACE"
# Map each approach phase to (parent nav phase for labelling, manip phase to
# enter on arrival).
_APPROACH_PARENT = {APPROACH_PICK: NAV_TO_OBJ, APPROACH_PLACE: NAV_TO_RECEPTACLE}
_APPROACH_MANIP = {APPROACH_PICK: PICK, APPROACH_PLACE: PLACE}

_BASE_MG_ID = "base"

# Base translation (m) above which a ``_snap_base_pose`` teleport is treated as
# a large discontinuity and the arm dof velocities are zeroed to absorb the
# resulting kick (small per-step re-pin drift corrections stay below this).
_LARGE_BASE_SNAP_M = 0.15


class _MobileManipPlannerPolicy(PickAndPlacePlannerPolicy):
    """Pick-and-place planner adapted for a mobile base.

    Two adaptations vs. the fixed-base planner:

    1. IK locks the base — only the arm is unlocked — so the parked base pose is
       treated as fixed (the fixed-base planner would otherwise let IK translate
       the holonomic base to satisfy the reach).
    2. The trajectory is split into ``"pick"`` and ``"place"`` halves, each built
       from the *current* (parked) base pose via :meth:`_compute_trajectory`.
    """

    #: Candidate vertical lift heights (m) tried after grasping, tallest first.
    _LIFT_HEIGHTS: tuple[float, ...] = (0.20, 0.15, 0.10, 0.05)

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        super().__init__(config, task)
        self.phase: str = PICK

    # -- base-locked IK ------------------------------------------------------ #
    def _arm_move_group_ids(self) -> list[str]:
        """Move groups unlocked for manipulation IK: the arm only (never the
        holonomic base, never the gripper)."""
        gripper_mgs = set(self.robot_view.get_gripper_movegroup_ids())
        return [
            mg
            for mg in self.robot_view.move_group_ids()
            if mg not in gripper_mgs and mg != _BASE_MG_ID
        ]

    def _tcp_to_jp_fn(self, mg_id: str, target_pose: np.ndarray) -> dict[str, Any]:
        kinematics = self.task.env.current_robot.kinematics
        arm_mgs = self._arm_move_group_ids()

        jp = kinematics.ik(
            mg_id,
            target_pose,
            arm_mgs,
            self.robot_view.get_qpos_dict(),
            self.robot_view.base.pose,
        )

        action = self.robot_view.get_ctrl_dict()
        if jp is not None:
            self.sequential_ik_failures = 0
            action.update({mg: jp[mg] for mg in arm_mgs})
        else:
            self.sequential_ik_failures += 1
            log.info(
                f"⚠️ IK failed (base-locked), holding position, fails:{self.sequential_ik_failures}"
            )
            if self.sequential_ik_failures >= self.policy_config.max_sequential_ik_failures:
                log.info("❌ Too many sequential IK failures, triggering retry.")
                return self._handle_failure()

        return action

    def check_feasible_ik(self, pose: np.ndarray) -> bool | np.ndarray:
        # Batch pre-filter (grasp selection) stays optimistic; the definitive
        # per-grasp confirmation goes through the scalar (base-locked) path.
        if pose.ndim > 2:
            return super().check_feasible_ik(pose)

        assert pose.shape == (4, 4)
        kinematics = self.task.env.current_robot.kinematics
        jp = kinematics.ik(
            self.active_gripper_mg_id,
            pose,
            self._arm_move_group_ids(),
            self.robot_view.get_qpos_dict(),
            base_pose=self.robot_view.base.pose,
        )
        return jp is not None

    # -- phase-split trajectory --------------------------------------------- #
    def _compute_trajectory(self) -> list[ActionPrimitive]:
        if self.phase == PICK:
            return self._compute_pick_primitives()
        if self.phase == PLACE:
            return self._compute_place_primitives()
        raise ValueError(f"Unexpected manipulation phase: {self.phase}")

    def _compute_pick_primitives(self) -> list[ActionPrimitive]:
        robot_view = self.robot_view
        task_config = self.config.task_config
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj: MlSpacesObject = om.get_object_by_name(task_config.pickup_obj_name)

        self.active_gripper_mg_id = self.select_arm_for_object(pickup_obj.position)
        gripper_mg_id = self.active_gripper_mg_id

        grasp_pose_world = compute_grasp_pose(
            self,
            pickup_obj,
            robot_view,
            check_collision=self.policy_config.filter_colliding_grasps,
            n_collision_checks=self.policy_config.grasp_collision_max_grasps,
            collision_batch_size=self.policy_config.grasp_collision_batch_size,
            check_ik=self.policy_config.filter_feasible_grasps,
            n_ik_checks=self.policy_config.grasp_feasibility_max_grasps,
            ik_batch_size=self.policy_config.grasp_feasibility_batch_size,
            pos_cost_weight=self.policy_config.grasp_pos_cost_weight,
            rot_cost_weight=self.policy_config.grasp_rot_cost_weight,
            vertical_cost_weight=self.policy_config.grasp_vertical_cost_weight,
            com_dist_cost_weight=self.policy_config.grasp_com_dist_cost_weight,
        )

        pregrasp_pose = grasp_pose_world.copy()
        pregrasp_pose[:3, 3] -= self.policy_config.pregrasp_z_offset * pregrasp_pose[:3, 2]

        if not self.check_feasible_ik(grasp_pose_world):
            raise ValueError("IK failed for grasp pose (parked base out of reach)")

        # The parked base can leave the grasp near the arm's reach limit, so the
        # pregrasp stand-off (further from the base along the approach axis) may be
        # infeasible even when the grasp is reachable. Shrink the offset until the
        # pregrasp is reachable; if none works, approach the grasp directly.
        base_offset = self.policy_config.pregrasp_z_offset
        pregrasp_feasible = False
        for frac in (1.0, 0.75, 0.5, 0.25):
            candidate = grasp_pose_world.copy()
            candidate[:3, 3] -= (base_offset * frac) * candidate[:3, 2]
            if self.check_feasible_ik(candidate):
                pregrasp_pose = candidate
                pregrasp_feasible = True
                break
        if not pregrasp_feasible:
            log.warning(
                "[MOBILE PNP FSM] No feasible pregrasp stand-off; approaching grasp directly."
            )
            pregrasp_pose = grasp_pose_world.copy()

        lift_pose = grasp_pose_world.copy()
        for height in self._LIFT_HEIGHTS:
            candidate = grasp_pose_world.copy()
            candidate[2, 3] += height
            if self.check_feasible_ik(candidate):
                lift_pose = candidate
                break
        else:
            log.warning("No feasible lift height; lifting is skipped (grasp height retained).")

        start_ee_pose = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world

        if self.policy_config.check_pick_path_collisions:
            model = self.task.env.current_model
            allowed_root_names = {model.body(model.body_rootid[pickup_obj.object_id]).name}
            hits = self._swept_path_collision_bodies(
                corridors=[
                    (pregrasp_pose, grasp_pose_world),
                    (grasp_pose_world, lift_pose),
                ],
                allowed_root_names=allowed_root_names,
                n_samples=self.policy_config.path_collision_samples_per_segment,
            )
            if hits:
                log.info(f"[MOBILE PNP FSM] PICK arm path sweeps into {sorted(hits)}; rejecting")
                raise ValueError(f"Path collision (pick) with {sorted(hits)}")

        return [
            GripperAction(robot_view, True, 0.0, gripper_mg_id=gripper_mg_id),
            TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                gripper_mg_id=gripper_mg_id,
                move_segments=[
                    TCPMoveSegment(
                        name="pregrasp",
                        start_pose=start_ee_pose,
                        end_pose=pregrasp_pose,
                        speed=self.policy_config.speed_fast,
                    ),
                    TCPMoveSegment(
                        name="grasp",
                        start_pose=pregrasp_pose,
                        end_pose=grasp_pose_world,
                        speed=self.policy_config.speed_slow,
                    ),
                ],
            ),
            GripperAction(
                robot_view,
                False,
                self.policy_config.gripper_close_duration,
                gripper_mg_id=gripper_mg_id,
            ),
            TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                is_holding_object=True,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                gripper_mg_id=gripper_mg_id,
                move_segments=[
                    TCPMoveSegment(
                        name="lift",
                        start_pose=grasp_pose_world,
                        end_pose=lift_pose,
                        speed=self.policy_config.speed_slow,
                    ),
                ],
            ),
        ]

    def _nearest_reachable_place_pose(
        self,
        place_pose: np.ndarray,
        receptacle_center: np.ndarray,
        receptacle_size: np.ndarray,
        pickup_obj_size: np.ndarray,
    ) -> np.ndarray | None:
        """Search the receptacle top surface for the IK-reachable place point
        closest to the base, keeping the object fully on the receptacle.

        Returns a 4x4 place pose (orientation/height preserved from ``place_pose``)
        or ``None`` if no candidate on the surface is reachable.
        """
        # Shrink the searchable footprint by the object's half-extent (plus a small
        # margin) so the placed object stays within the receptacle top.
        margin = self.policy_config.place_edge_margin_m
        half_x = max(receptacle_size[0] / 2 - pickup_obj_size[0] / 2 - margin, 0.0)
        half_y = max(receptacle_size[1] / 2 - pickup_obj_size[1] / 2 - margin, 0.0)
        n = self.policy_config.place_search_grid_n
        xs = np.linspace(-half_x, half_x, n) + receptacle_center[0]
        ys = np.linspace(-half_y, half_y, n) + receptacle_center[1]
        base_xy = self.robot_view.base.pose[:2, 3]

        candidates = []
        for x in xs:
            for y in ys:
                candidates.append((x, y))
        # Nearest-to-base first: most likely inside the arm workspace.
        candidates.sort(key=lambda p: (p[0] - base_xy[0]) ** 2 + (p[1] - base_xy[1]) ** 2)

        for x, y in candidates:
            candidate = place_pose.copy()
            candidate[0, 3] = x
            candidate[1, 3] = y
            if self.check_feasible_ik(candidate):
                log.info(
                    "[MOBILE PNP FSM] Placing at nearest reachable receptacle point "
                    f"({x:.2f}, {y:.2f}) instead of centre "
                    f"({receptacle_center[0]:.2f}, {receptacle_center[1]:.2f})."
                )
                return candidate
        return None

    def _get_placement_poses(
        self,
        grasp_pose_world: np.ndarray,
        pickup_obj: MlSpacesObject,
        place_receptacle: MlSpacesObject,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Base-locked placement pose builder.

        The parked base can leave the receptacle centre close to the arm's reach
        limit, so the raised preplace pose is the first to become infeasible.
        Shrink the preplace stand-off height until it is reachable, and if the
        preplace is unreachable at every height, approach the place pose
        directly. The final ``place`` pose (object centred on the receptacle) is
        still required to be reachable.
        """
        place_receptacle_aabb_center, place_receptacle_aabb_size = body_aabb(
            self.task.env.current_data.model,
            self.task.env.current_data,
            place_receptacle.object_id,
        )
        receptacle_top_z = place_receptacle_aabb_center[2] + place_receptacle_aabb_size[2] / 2
        pickup_obj_aabb_center, pickup_obj_aabb_size = body_aabb(
            self.task.env.current_data.model,
            self.task.env.current_data,
            pickup_obj.object_id,
        )
        pickup_obj_bottom_z = pickup_obj_aabb_center[2] - pickup_obj_aabb_size[2] / 2
        pickup_obj_clearance_offset = max(grasp_pose_world[2, 3] - pickup_obj_bottom_z, 0.0)

        centering = grasp_pose_world[:3, 3] - pickup_obj.position

        place_pose = grasp_pose_world.copy()
        place_pose[:2, 3] = place_receptacle.position[:2]
        place_pose[2, 3] = receptacle_top_z + pickup_obj_clearance_offset
        place_pose[:3, 3] += centering
        if not self.check_feasible_ik(place_pose):
            # The receptacle centre can sit past the arm's reach when the base is
            # parked at the closest navigable cell (large tables/counters). Rather
            # than give up, place at the nearest IK-reachable point on the
            # receptacle top surface: any point within the top footprint (shrunk
            # by the object's half-extent so it still lands on the receptacle)
            # satisfies the "object on receptacle" success predicate. Prefer the
            # candidate closest to the base (most central to the arm workspace).
            place_pose = self._nearest_reachable_place_pose(
                place_pose=place_pose,
                receptacle_center=place_receptacle_aabb_center,
                receptacle_size=place_receptacle_aabb_size,
                pickup_obj_size=pickup_obj_aabb_size,
            )
            if place_pose is None:
                raise ValueError("IK failed for place pose (parked base out of reach)")

        base_z_offset = self.policy_config.place_z_offset
        preplace_pose = place_pose.copy()
        preplace_feasible = False
        for frac in (1.0, 0.75, 0.5, 0.25):
            candidate = place_pose.copy()
            candidate[2, 3] = (
                receptacle_top_z
                + pickup_obj_clearance_offset
                + base_z_offset * frac
            )
            if self.check_feasible_ik(candidate):
                preplace_pose = candidate
                preplace_feasible = True
                break
        if not preplace_feasible:
            log.warning(
                "[MOBILE PNP FSM] No feasible preplace stand-off; "
                "descending onto the receptacle directly."
            )
            preplace_pose = place_pose.copy()

        postplace_pose = place_pose.copy()
        postplace_pose[:3, 3] -= self.policy_config.end_z_offset * postplace_pose[:3, 2]

        return preplace_pose, place_pose, postplace_pose

    def _compute_place_primitives(self) -> list[ActionPrimitive]:
        robot_view = self.robot_view
        task_config = self.config.task_config
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj: MlSpacesObject = om.get_object_by_name(task_config.pickup_obj_name)
        place_receptacle: MlSpacesObject = om.get_object_by_name(
            task_config.place_receptacle_name
        )

        gripper_mg_id = self.active_gripper_mg_id
        # The object is currently held; treat the live end-effector pose as the
        # "grasp pose" so the placement helper centres the object on the
        # receptacle and IK-checks the poses from the parked base.
        current_ee_pose = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world.copy()

        preplace_pose, place_pose, postplace_pose = self._get_placement_poses(
            grasp_pose_world=current_ee_pose,
            pickup_obj=pickup_obj,
            place_receptacle=place_receptacle,
        )

        if self.policy_config.check_place_path_collisions:
            model = self.task.env.current_model
            allowed_root_names = {
                model.body(model.body_rootid[obj.object_id]).name
                for obj in (pickup_obj, place_receptacle)
            }
            hits = self._swept_path_collision_bodies(
                corridors=[
                    (current_ee_pose, preplace_pose),
                    (preplace_pose, place_pose),
                ],
                allowed_root_names=allowed_root_names,
                n_samples=self.policy_config.path_collision_samples_per_segment,
            )
            if hits:
                log.info(f"[MOBILE PNP FSM] PLACE arm path sweeps into {sorted(hits)}; rejecting")
                raise ValueError(f"Path collision (place) with {sorted(hits)}")

        return [
            TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                is_holding_object=True,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                gripper_mg_id=gripper_mg_id,
                move_segments=[
                    TCPMoveSegment(
                        name="preplace",
                        start_pose=current_ee_pose,
                        end_pose=preplace_pose,
                        speed=self.policy_config.speed_fast,
                    ),
                    TCPMoveSegment(
                        name="place",
                        start_pose=preplace_pose,
                        end_pose=place_pose,
                        speed=self.policy_config.speed_slow,
                    ),
                ],
            ),
            GripperAction(
                robot_view,
                True,
                self.policy_config.gripper_open_duration,
                gripper_mg_id=gripper_mg_id,
            ),
            TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                gripper_mg_id=gripper_mg_id,
                move_segments=[
                    TCPMoveSegment(
                        name="retreat",
                        start_pose=place_pose,
                        end_pose=postplace_pose,
                        speed=self.policy_config.speed_fast,
                    ),
                ],
            ),
        ]


class MobilePickAndPlaceStateMachinePolicy(PlannerPolicy):
    """FSM expert: navigate → pick → navigate → place."""

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        super().__init__(config, task)
        self.policy_config = config.policy_config

        # --- Navigation sub-policy ---------------------------------------- #
        nav_policy_config = self.policy_config.nav_policy_config
        nav_task_config = NavToObjTaskConfig(
            task_cls=None,
            succ_pos_threshold=self.policy_config.nav_standoff_succ_threshold,
            visibility_camera_name=self.policy_config.nav_visibility_camera_name,
            require_object_visible=False,
            succ_use_goal_pose=False,
        )
        self._nav_config = config.model_copy(
            update={"policy_config": nav_policy_config, "task_config": nav_task_config}
        )
        self._nav_policy = nav_policy_config.policy_cls(self._nav_config, task)

        # --- Manipulation sub-policy -------------------------------------- #
        manip_policy_config = self.policy_config.manip_policy_config
        # The FSM probes several candidate base standoffs by actually building
        # the grasp/place primitives. Exhaustively IK-checking every non-colliding
        # grasp (~256) costs ~120s for an INFEASIBLE pose (a feasible pose exits
        # early), which makes the candidate search intractable. Cap the check so a
        # bad standoff is rejected in a few seconds; grasps are cost-ordered, so a
        # genuinely feasible pose still finds a grasp within the cap.
        manip_policy_config = manip_policy_config.model_copy(
            update={"grasp_feasibility_max_grasps": 32}
        )
        self._manip_config = config.model_copy(update={"policy_config": manip_policy_config})
        self._manip_policy = _MobileManipPlannerPolicy(self._manip_config, task)

        self._phase: str = NAV_TO_OBJ

        # Base pose to pin during manipulation (set by the base search).
        self._locked_base_pose: np.ndarray | None = None
        # Per-segment nav step counter for the base acceleration ramp; reset on
        # entry to each nav segment so the base eases off gently from rest.
        self._nav_step_in_segment: int = 0
        # Arm joint targets to pin while navigating, captured on entry to each
        # nav segment. The raw actuator ctrl can be uninitialised (near-zero) at
        # episode start, which would command the arm to straighten into a
        # near-limit extended pose the instant the demo begins; pinning the
        # captured stow pose keeps the arm safely tucked instead.
        self._nav_arm_hold: dict[str, np.ndarray] | None = None

        # Grasp lock: keep the held object rigidly fixed to the gripper during
        # transport navigation so it doesn't slip out under base acceleration.
        self._grasp_offset: np.ndarray | None = None
        self._grasp_mg_id: str | None = None
        # Set once the object has been set down at PLACE; a failure afterwards
        # (e.g. a retreat IK hiccup) must NOT retry, which would teleport the
        # already-placed object back into the gripper and undo the success.
        self._place_released: bool = False
        # --- Subtask checkpoint / retry ----------------------------------- #
        # Cache the full MuJoCo state at the start of each nav+manip segment so
        # a failed subtask can restore the last good state and retry (with a
        # freshly-sampled parking pose) instead of discarding the whole episode.
        self._checkpoints: dict[str, tuple[np.ndarray, int]] = {}
        self._retry_counts: dict[str, int] = {}
        self._max_segment_retries = getattr(self.policy_config, "max_segment_retries", 2)

    # Segment = (navigate to a target, then manipulate). A failure anywhere in
    # a segment restores the segment-start checkpoint and re-runs the segment.
    _SEGMENTS = {
        PICK: {
            "nav_phase": NAV_TO_OBJ,
            "target_attr": "pickup_obj_name",
            "override_attr": "robot_base_pose",
        },
        PLACE: {
            "nav_phase": NAV_TO_RECEPTACLE,
            "target_attr": "place_receptacle_name",
            "override_attr": "place_robot_base_pose",
        },
    }

    def _capture_state(self) -> np.ndarray:
        model = self.task.env.current_model
        data = self.task.env.current_data
        size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.empty(size, dtype=np.float64)
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        return state

    def _restore_state(self, state: np.ndarray) -> None:
        model = self.task.env.current_model
        data = self.task.env.current_data
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)

    def _make_checkpoint(self) -> tuple[np.ndarray, int]:
        """Capture the full MuJoCo state plus the current recording length so a
        later restore can both rewind the physics AND discard the recorded frames
        of the failed attempt (see :meth:`_restore_checkpoint`)."""
        return self._capture_state(), len(self.task.observation_cache)

    def _restore_checkpoint(self, ckpt: tuple[np.ndarray, int]) -> None:
        """Restore a checkpoint captured by :meth:`_make_checkpoint`: rewind the
        physics state and truncate the task's per-step recording buffers back to
        the checkpoint length, so the failed subtask attempt (and the teleport
        discontinuity introduced by rewinding) never appears in the saved
        trajectory. The parallel per-step traces the rollout keeps for subtask
        labels re-sync automatically off ``len(observation_cache)``."""
        state, obs_len = ckpt
        self._restore_state(state)
        self._truncate_recording(obs_len)

    def _truncate_recording(self, obs_len: int) -> None:
        """Drop recorded frames past ``obs_len`` from every task cache so the
        saved trajectory contains only the retained (good) prefix. The
        observation/reward/terminal/truncated/success caches include the reset
        frame (length = steps + 1); the action cache does not (length = steps)."""
        task = self.task
        obs_len = max(0, min(obs_len, len(task.observation_cache)))
        del task.observation_cache[obs_len:]
        del task.reward_cache[obs_len:]
        del task.terminal_cache[obs_len:]
        del task.truncated_cache[obs_len:]
        del task.success_cache[obs_len:]
        act_len = max(0, obs_len - 1)
        del task.action_cache[act_len:]

    # ------------------------------------------------------------------ #
    # Arm-motion safety clamp (kinematic smoothness)                     #
    # ------------------------------------------------------------------ #
    def _smooth_arm_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Clamp commanded arm joint targets to a natural per-step velocity and
        keep them inside the joint position limits.

        The manipulation planner solves stateless IK per control step, so between
        consecutive smooth TCP targets the base-locked IK can jump branches
        (elbow flip) and emit single-step joint jumps of tens of rad/s, or return
        solutions outside the joint limits. We clamp each arm move group's
        commanded delta (relative to the *current measured* joint positions) to
        ``arm_max_vel_rad_s * dt`` and clip the result inside the joint limits.
        Normal planned motion is well under the cap, so this only smooths the
        pathological IK-branch-flip spikes without lagging ordinary moves.
        """
        cfg = self.policy_config
        if not getattr(cfg, "arm_smoothing_enabled", True):
            return action

        dt = self.config.policy_dt_ms / 1000.0
        max_step = cfg.arm_max_vel_rad_s * dt
        margin = cfg.arm_pos_limit_margin_rad
        robot_view = self._manip_policy.robot_view

        for mg in self._manip_policy._arm_move_group_ids():
            if mg not in action:
                continue
            cmd = np.asarray(action[mg], dtype=np.float64)
            mgv = robot_view.get_move_group(mg)
            cur = np.asarray(mgv.joint_pos, dtype=np.float64)
            if cmd.shape != cur.shape:
                continue
            # 1) Velocity clamp: bound the per-step change from the measured pose.
            delta = np.clip(cmd - cur, -max_step, max_step)
            cmd = cur + delta
            # 2) Position clamp: keep inside the joint limits (with a margin).
            limits = np.asarray(mgv.joint_pos_limits, dtype=np.float64)
            if limits.shape == (cmd.shape[0], 2):
                lo = limits[:, 0] + margin
                hi = limits[:, 1] - margin
                # Guard against degenerate/inverted ranges after margin.
                valid = hi >= lo
                cmd = np.where(valid, np.clip(cmd, lo, hi), cmd)
            action[mg] = cmd
        return action

    def _held_nav_arm_setpoints(self, robot_view: Any) -> dict[str, np.ndarray]:
        """Arm move-group setpoints to hold during a nav/approach segment.

        Instead of pinning the arm at whatever (possibly extended, near-limit)
        pose it happened to be in when the segment began, ramp the held setpoint
        from the measured pose toward the compact stow/home config
        (``nav_arm_stow_qpos``) at the arm velocity cap (``arm_max_vel_rad_s``).
        The base's position servo then drives with a retracted arm and the ramp
        keeps the retract smooth (no fast/flinging arm motion). Falls back to a
        constant hold of the entry pose when the retract is disabled or the move
        group's DOF does not match the stow config (e.g. a non-7-DOF arm)."""
        cfg = self.policy_config
        arm_mgs = self._manip_policy._arm_move_group_ids()
        if self._nav_arm_hold is None:
            # Seed the ramp from the measured pose so it starts where the arm is.
            self._nav_arm_hold = {
                mg: np.asarray(
                    robot_view.get_move_group(mg).joint_pos, dtype=np.float64
                ).copy()
                for mg in arm_mgs
            }
        if not getattr(cfg, "nav_arm_retract_enabled", True):
            return self._nav_arm_hold

        dt = self.config.policy_dt_ms / 1000.0
        max_step = cfg.arm_max_vel_rad_s * dt
        margin = cfg.arm_pos_limit_margin_rad
        stow = np.asarray(cfg.nav_arm_stow_qpos, dtype=np.float64)
        for mg in arm_mgs:
            cur = self._nav_arm_hold.get(mg)
            if cur is None:
                continue
            if cur.shape != stow.shape:
                # DOF mismatch (not the 7-DOF FR3 arm): hold the entry pose.
                continue
            target = stow.copy()
            mgv = robot_view.get_move_group(mg)
            limits = np.asarray(mgv.joint_pos_limits, dtype=np.float64)
            if limits.shape == (cur.shape[0], 2):
                lo = limits[:, 0] + margin
                hi = limits[:, 1] - margin
                valid = hi >= lo
                target = np.where(valid, np.clip(target, lo, hi), target)
            # Ramp toward the stow config at the per-step velocity cap.
            delta = np.clip(target - cur, -max_step, max_step)
            self._nav_arm_hold[mg] = cur + delta
        return self._nav_arm_hold

    def _slew_base_command(self, base_cmd: np.ndarray) -> np.ndarray:
        """Slew-limit a base pose command ``[x, y, heading]`` so the holonomic
        base accelerates and turns gently during navigation.

        The pure-pursuit follower emits a target a fixed look-ahead ahead of the
        current pose; the base position servo then charges toward it at whatever
        speed it can (~2 m/s), dragging the position-held stowed/held arm hard
        enough to spike its joint velocity past the hardware limit. We bound the
        per-step translation and yaw of the *target* relative to the current base
        pose (converting the m/s and rad/s caps to per-step deltas via the
        control ``dt``), and ramp the translation cap up over the first few steps
        of each nav segment so the base eases off from rest. The base still
        follows the same path, just smoothly, so navigation is unchanged while
        the arm no longer gets flung.
        """
        cfg = self.policy_config
        if not getattr(cfg, "base_slew_enabled", True):
            return base_cmd

        dt = self.config.policy_dt_ms / 1000.0
        base_pose = self.task.env.current_robot.robot_view.base.pose
        cur_xy = np.asarray(base_pose[:2, 3], dtype=np.float64)
        cur_yaw = float(np.arctan2(base_pose[1, 0], base_pose[0, 0]))

        cmd = np.asarray(base_cmd, dtype=np.float64).copy()

        # Ramp the translation cap up from rest over the first few nav steps.
        ramp_steps = max(int(cfg.base_accel_ramp_steps), 1)
        ramp = min(self._nav_step_in_segment + 1, ramp_steps) / ramp_steps
        max_step = cfg.base_max_speed_m_s * dt * ramp
        delta = cmd[:2] - cur_xy
        dist = float(np.linalg.norm(delta))
        if dist > max_step > 0.0:
            cmd[:2] = cur_xy + delta * (max_step / dist)

        # Bound the per-step yaw change (wrap to [-pi, pi] first).
        max_yaw = cfg.base_max_yaw_rate_rad_s * dt
        dyaw = float(np.arctan2(np.sin(cmd[2] - cur_yaw), np.cos(cmd[2] - cur_yaw)))
        dyaw = float(np.clip(dyaw, -max_yaw, max_yaw))
        cmd[2] = cur_yaw + dyaw

        self._nav_step_in_segment += 1
        return cmd

    def _begin_base_approach(self, approach_phase: str, parked_pose: np.ndarray) -> None:
        """Start a smooth base approach toward the searched manip standoff.

        ``_enter_pick``/``_enter_place`` select ``_locked_base_pose`` by snapping
        the base to a feasibility-verified standoff, which teleports it in one
        frame (a base "teleport" in the recorded data). Instead of executing from
        there, we snap the base back to where navigation actually parked and then
        drive it to the standoff over several slew-limited steps, so the recorded
        base motion is continuous and stays under the teleport threshold. The
        manip primitives were already built at the standoff during the search and
        stay valid: the base is re-pinned exactly there the instant we enter the
        manip phase.
        """
        cfg = self.policy_config
        target = self._locked_base_pose.copy()
        carry = self.config.task_config.pickup_obj_name if approach_phase == APPROACH_PLACE else None
        if not getattr(cfg, "base_approach_enabled", True):
            # Approach disabled: leave the base at the standoff (legacy teleport).
            self._phase = _APPROACH_MANIP[approach_phase]
            return
        # Undo the search teleport: put the base back where nav parked (carrying
        # the held object along on PLACE so it isn't dropped) so the approach
        # starts from the real parked pose.
        self._snap_base_pose(parked_pose, carry_object_name=carry)
        self._approach_target = target
        self._approach_carry = carry
        self._approach_steps = 0
        # Re-arm the arm hold so the approach pins the arm at its current stow
        # pose (mirrors a fresh nav segment).
        self._nav_arm_hold = None
        self._nav_step_in_segment = 0
        self._phase = approach_phase
        log.info(
            f"[MOBILE PNP FSM] {_APPROACH_PARENT[approach_phase]} parked at "
            f"({parked_pose[0, 3]:.2f}, {parked_pose[1, 3]:.2f}); smoothly approaching "
            f"standoff ({target[0, 3]:.2f}, {target[1, 3]:.2f})."
        )

    def _step_base_approach(self, observation: Any) -> dict[str, Any] | None:
        """One step of the smooth base approach. Returns an action dict while
        still approaching, or ``None`` once the standoff is reached (the caller
        then enters the manip phase, which re-pins the base at the standoff).

        The base is advanced by *bounded qpos increments* toward the standoff
        (each step's translation/yaw capped to the nav speed limits), exactly as
        the manip phase pins the base, rather than by charging the physics
        position servo at the target. Driving by increments keeps every recorded
        step continuous and under the teleport threshold AND cannot stall when
        the short parked->standoff move grazes the furniture the robot is about
        to manipulate from (the servo would wedge and leave a residual snap)."""
        cfg = self.policy_config
        robot_view = self.task.env.current_robot.robot_view
        gripper_ids = robot_view.get_gripper_movegroup_ids()
        target = self._approach_target

        base_pose = robot_view.base.pose
        cur_xy = np.asarray(base_pose[:2, 3], dtype=np.float64)
        cur_yaw = float(np.arctan2(base_pose[1, 0], base_pose[0, 0]))
        tgt_xy = np.asarray(target[:2, 3], dtype=np.float64)
        tgt_yaw = float(np.arctan2(target[1, 0], target[0, 0]))
        pos_err = float(np.linalg.norm(tgt_xy - cur_xy))
        yaw_err = abs(float(np.arctan2(np.sin(tgt_yaw - cur_yaw), np.cos(tgt_yaw - cur_yaw))))

        pos_tol = getattr(cfg, "base_approach_pos_tol_m", 0.04)
        yaw_tol = getattr(cfg, "base_approach_yaw_tol_rad", 0.05)
        max_steps = int(getattr(cfg, "base_approach_max_steps", 60))
        if (pos_err <= pos_tol and yaw_err <= yaw_tol) or self._approach_steps >= max_steps:
            log.info(
                f"[MOBILE PNP FSM] base approach done after {self._approach_steps} steps "
                f"(pos_err={pos_err:.3f}m, yaw_err={yaw_err:.3f}rad)."
            )
            return None  # arrived (or budget spent): commit to manipulation.

        # Bound this step's translation and yaw to the nav speed caps (ramping
        # the translation up from rest over the first few steps), then snap the
        # base to the resulting intermediate pose.
        dt = self.config.policy_dt_ms / 1000.0
        ramp_steps = max(int(getattr(cfg, "base_accel_ramp_steps", 8)), 1)
        ramp = min(self._approach_steps + 1, ramp_steps) / ramp_steps
        max_step = float(cfg.base_max_speed_m_s) * dt * ramp
        max_yaw = float(cfg.base_max_yaw_rate_rad_s) * dt

        delta = tgt_xy - cur_xy
        dist = float(np.linalg.norm(delta))
        new_xy = tgt_xy if dist <= max_step or dist == 0.0 else cur_xy + delta * (max_step / dist)
        dyaw = float(np.arctan2(np.sin(tgt_yaw - cur_yaw), np.cos(tgt_yaw - cur_yaw)))
        dyaw = float(np.clip(dyaw, -max_yaw, max_yaw))
        new_yaw = cur_yaw + dyaw

        step_pose = np.eye(4)
        c, s = np.cos(new_yaw), np.sin(new_yaw)
        step_pose[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        step_pose[0, 3], step_pose[1, 3] = float(new_xy[0]), float(new_xy[1])
        step_pose[2, 3] = float(target[2, 3])
        # Snap to the intermediate pose (carrying the held object on PLACE) and
        # set the base setpoint so physics holds it there for this step.
        self._snap_base_pose(step_pose, carry_object_name=self._approach_carry)

        hold = robot_view.get_ctrl_dict(["arm"] + gripper_ids)
        for mg, qpos in self._held_nav_arm_setpoints(robot_view).items():
            hold[mg] = qpos
        # Base is pinned via the snap above; do not command it through the action.
        hold.pop(_BASE_MG_ID, None)
        self._approach_steps += 1
        return hold

    def _retry_segment(self, seg: str) -> bool:
        """Restore the segment-start checkpoint and re-run the segment.

        Returns False (episode should go to DONE) when no checkpoint exists or
        the retry budget for this segment is exhausted.
        """
        info = self._SEGMENTS[seg]
        ckpt = self._checkpoints.get(seg)
        used = self._retry_counts.get(seg, 0)
        if ckpt is None or used >= self._max_segment_retries:
            log.warning(
                f"[MOBILE PNP FSM] {seg} segment retry unavailable "
                f"(checkpoint={'yes' if ckpt is not None else 'no'}, used={used}/"
                f"{self._max_segment_retries})"
            )
            return False
        self._retry_counts[seg] = used + 1
        self._restore_checkpoint(ckpt)
        task_cfg = self.config.task_config
        self.task.set_nav_target(getattr(task_cfg, info["target_attr"]))
        # First retry re-drives to the feasibility-verified override; later
        # retries drop it to force a freshly-sampled standoff, adding the
        # variation needed to escape a reproducibly-infeasible parking pose.
        override = getattr(task_cfg, info["override_attr"], None) if used == 0 else None
        self.task.set_nav_goal_override(override)
        self._nav_policy.reset()
        self._nav_step_in_segment = 0
        self._nav_arm_hold = None
        self._phase = info["nav_phase"]
        log.info(
            f"[MOBILE PNP FSM] retrying {seg} segment "
            f"(attempt {used + 1}/{self._max_segment_retries}) from cached checkpoint"
        )
        return True

    @property
    def planners(self) -> dict:
        return {}

    def get_phase(self) -> str:
        # Report the smooth base-approach sub-phases as their parent navigation
        # phase: the base is still driving toward its goal with the arm stowed,
        # so the subtask labeller merges them into the navigation segment (and
        # they must not surface as unknown phases that would be dropped).
        return _APPROACH_PARENT.get(self._phase, self._phase)

    def get_all_phases(self) -> dict[str, int]:
        return {NAV_TO_OBJ: 0, PICK: 1, NAV_TO_RECEPTACLE: 2, PLACE: 3, DONE: 4}

    def reset(self) -> None:
        self._phase = NAV_TO_OBJ
        self._nav_step_in_segment = 0
        self._nav_arm_hold = None
        self._checkpoints = {}
        self._retry_counts = {}
        self._locked_base_pose = None
        # Smooth base-approach sub-phase state (drives the base from the parked
        # nav pose to the manip standoff over several slew-limited steps).
        self._approach_target = None
        self._approach_carry = None
        self._approach_steps = 0
        self._grasp_offset = None
        self._grasp_mg_id = None
        self._place_released = False
        self.task.set_nav_target(self.config.task_config.pickup_obj_name)
        # Drive to the grasp-feasibility-verified base pose the sampler recorded
        # (Avenue A), not the closest navigable cell.
        self.task.set_nav_goal_override(getattr(self.config.task_config, "robot_base_pose", None))
        self._nav_policy.reset()
        # Cache the pristine post-sample state (and recording length) so the
        # whole pick segment (navigate-to-object + pick) can be retried from
        # scratch, with the failed attempt truncated out of the saved data.
        self._checkpoints[PICK] = self._make_checkpoint()
        log.info("[MOBILE PNP FSM] reset → phase NAV_TO_OBJ")

    # ------------------------------------------------------------------ #
    # Phase transitions                                                   #
    # ------------------------------------------------------------------ #
    def _tcp_world_pose(self) -> np.ndarray | None:
        """World pose of the active gripper TCP (leaf frame), or None."""
        mg_id = self._grasp_mg_id
        if mg_id is None:
            return None
        return self.task.env.current_robot.robot_view.get_move_group(mg_id).leaf_frame_to_world

    def _capture_grasp_lock(self) -> None:
        """After a successful pick, record the held object's pose in the TCP
        frame so it can be rigidly re-asserted during transport navigation."""
        mg_id = getattr(self._manip_policy, "active_gripper_mg_id", None)
        if mg_id is None:
            mg_id = self.task.env.current_robot.robot_view.get_gripper_movegroup_ids()[0]
        self._grasp_mg_id = mg_id
        tcp = self._tcp_world_pose()
        obj_pose = self._held_object_pose()
        if tcp is None or obj_pose is None:
            self._grasp_offset = None
            return
        self._grasp_offset = np.linalg.inv(tcp) @ obj_pose
        log.info("[MOBILE PNP FSM] captured grasp lock (object fixed to gripper for transport).")

    def _held_object_pose(self) -> np.ndarray | None:
        model = self.task.env.current_model
        data = self.task.env.current_data
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)
        jnt_id = int(model.body_jntadr[obj.body_id])
        if jnt_id == -1 or model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_FREE:
            return None
        qadr = int(model.jnt_qposadr[jnt_id])
        return pos_quat_to_pose_mat(data.qpos[qadr : qadr + 3], data.qpos[qadr + 3 : qadr + 7])

    def _apply_grasp_lock(self) -> None:
        """Re-assert the held object's pose relative to the current TCP so it
        stays physically between the fingers (preventing inertial slip-out during
        transport). The object remains between the fingers, so the gripper's
        finger gap is preserved and the ``is not in grasp`` check still passes."""
        if self._grasp_offset is None:
            return
        tcp = self._tcp_world_pose()
        if tcp is None:
            return
        model = self.task.env.current_model
        data = self.task.env.current_data
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)
        jnt_id = int(model.body_jntadr[obj.body_id])
        if jnt_id == -1 or model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_FREE:
            return
        qadr = int(model.jnt_qposadr[jnt_id])
        dofadr = int(model.jnt_dofadr[jnt_id])
        pos, quat = pose_mat_to_pos_quat(tcp @ self._grasp_offset)
        data.qpos[qadr : qadr + 3] = pos
        data.qpos[qadr + 3 : qadr + 7] = quat
        data.qvel[dofadr : dofadr + 6] = 0.0

    def _snap_base_pose(self, pose_mat: np.ndarray, carry_object_name: str | None = None) -> None:
        """Snap the (frozen) base to ``pose_mat`` and hold it there.

        Sets base qpos AND the holonomic actuator setpoint (else the position
        servo drives the base back to the stale nav setpoint during manip), then
        forwards so kinematics/IK see the new base pose.

        The base teleport is instantaneous, so anything held only by gripper
        friction (``carry_object_name``) does NOT follow and would be left behind
        (dropped) unless we move it too. When carrying, we rigidly transform the
        held object by the same world SE2 delta as the base so it stays in hand.
        """
        robot_view = self.task.env.current_robot.robot_view
        model = self.task.env.current_model
        data = self.task.env.current_data

        old_base = robot_view.base.pose.copy()
        robot_view.base.pose = pose_mat
        x, y = float(pose_mat[0, 3]), float(pose_mat[1, 3])
        theta = float(np.arctan2(pose_mat[1, 0], pose_mat[0, 0]))
        robot_view.base.ctrl = np.array([x, y, theta])

        # Zero base velocities so re-asserting qpos each step doesn't fight
        # residual momentum (which would jostle the arm / held object).
        base = robot_view.base
        joint_ids = getattr(base, "_joint_ids", None)
        if joint_ids is not None:
            for jid in joint_ids:
                dofadr = int(model.jnt_dofadr[jid])
                data.qvel[dofadr] = 0.0

        # A LARGE base teleport (e.g. the initial manip-entry snap when nav
        # parked far from the searched manip pose) instantaneously displaces the
        # base under the position-held arm, kicking the arm dofs and igniting a
        # physics blow-up (joint velocities/positions past the hardware limits).
        # When the snap is large, also zero the arm dof velocities so the
        # teleport imparts no spurious arm momentum. Small per-step re-pin drift
        # corrections leave the arm velocity untouched so normal manipulation
        # motion is unaffected.
        base_shift = float(np.linalg.norm(pose_mat[:2, 3] - old_base[:2, 3]))
        if base_shift > _LARGE_BASE_SNAP_M:
            for mg in self._manip_policy._arm_move_group_ids():
                mgv = robot_view.get_move_group(mg)
                arm_jids = getattr(mgv, "_joint_ids", None)
                if arm_jids is None:
                    continue
                for jid in arm_jids:
                    dofadr = int(model.jnt_dofadr[jid])
                    data.qvel[dofadr] = 0.0

        if carry_object_name is not None:
            delta = pose_mat @ np.linalg.inv(old_base)
            om = self.task.env.object_managers[self.task.env.current_batch_index]
            obj = om.get_object_by_name(carry_object_name)
            jnt_id = int(model.body_jntadr[obj.body_id])
            if jnt_id != -1 and model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
                qadr = int(model.jnt_qposadr[jnt_id])
                new_obj_pose = delta @ pos_quat_to_pose_mat(
                    data.qpos[qadr : qadr + 3], data.qpos[qadr + 3 : qadr + 7]
                )
                pos, quat = pose_mat_to_pos_quat(new_obj_pose)
                data.qpos[qadr : qadr + 3] = pos
                data.qpos[qadr + 3 : qadr + 7] = quat
                dofadr = int(model.jnt_dofadr[jnt_id])
                data.qvel[dofadr : dofadr + 6] = 0.0

        mujoco.mj_forward(model, data)

    def _snap_path_is_clear(self, target_pose: np.ndarray) -> bool:
        """True if the straight-line base snap from the current (parked) base
        pose to ``target_pose`` does not cross a wall/obstacle.

        The manip-standoff snap is instantaneous, so a candidate whose endpoint
        is IK-feasible can still teleport the base THROUGH a wall when nav parked
        on the far side of it. We test the snap segment against the scene
        occupancy map (agent-radius-dilated ProcTHOR map; True == free). A manip
        standoff legitimately abuts the target furniture (map-occupied), and the
        parked pose may too, so only the segment INTERIOR -- samples farther than
        ``manip_snap_endpoint_margin_m`` from both endpoints -- is tested. Short
        snaps (< 2x the margin) have no interior and always pass; a wall in the
        middle of a long snap is detected and the candidate rejected. Fails open
        (returns True) if the gate is disabled or the map is unavailable, so it
        can never spuriously drop an otherwise valid episode."""
        cfg = self.policy_config
        if not getattr(cfg, "manip_snap_collision_gate_enabled", True):
            return True
        env = self.task.env
        robot_view = env.current_robot.robot_view
        start = robot_view.base.pose
        start_xy = np.asarray(start[:2, 3], dtype=np.float64)
        tgt_xy = np.asarray(target_pose[:2, 3], dtype=np.float64)
        dist = float(np.linalg.norm(tgt_xy - start_xy))
        margin = float(getattr(cfg, "manip_snap_endpoint_margin_m", 0.6))
        if dist <= 2.0 * margin:
            return True  # no interior to test: too short to be a wall crossing.
        try:
            thormap = env.get_thormap()
        except Exception:
            return True  # fail-open: never block when the map is unavailable.
        spacing = max(float(getattr(cfg, "manip_snap_collision_sample_spacing_m", 0.15)), 0.05)
        n = int(np.clip(np.ceil(dist / spacing), 2, 80))
        pts: list[list[float]] = []
        for k in range(1, n):
            t = k / n
            xy = start_xy + t * (tgt_xy - start_xy)
            if np.linalg.norm(xy - start_xy) < margin or np.linalg.norm(xy - tgt_xy) < margin:
                continue
            pts.append([float(xy[0]), float(xy[1]), 0.0])
        if not pts:
            return True
        try:
            free = np.asarray(thormap.check_collision(np.asarray(pts, dtype=np.float64)))
        except Exception:
            return True  # fail-open on any map query error.
        return bool(np.all(free))  # every interior sample must be free space.

    def _manip_base_candidates(
        self, target_name: str, verified_pose_7d: list[float] | None
    ) -> list[np.ndarray]:
        """Candidate base poses for manipulating ``target_name``.

        The sampler's feasibility-verified pose (if any) is tried first, then a
        ring of receptacle/object-facing standoffs, ordered by proximity to the
        current parked base so the snap (and any resulting data discontinuity) is
        as small as possible.
        """
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        target = om.get_object_by_name(target_name)
        target_xy = np.asarray(target.position)[:2]
        robot_view = self.task.env.current_robot.robot_view
        base_pose = robot_view.base.pose
        base_xy = base_pose[:2, 3]
        base_z = float(base_pose[2, 3])

        candidates: list[np.ndarray] = []
        if verified_pose_7d is not None:
            candidates.append(pos_quat_to_pose_mat(np.asarray(verified_pose_7d, dtype=float)))

        # Standoffs near the arm's comfortable reach (closer is generally more
        # reachable). Denser than a minimal probe because the per-step cost is
        # now ~8x lower (get_info memoization + receptacle-heuristic gate), so a
        # wider feasibility search is affordable and directly attacks the
        # dominant "IK failed (base-locked)" PLACE failures: a finer ring makes
        # it far likelier that the strict all-waypoint-reachable pass finds a
        # standoff, instead of falling back to a build-only pose that then aborts
        # mid-execution. Includes a closer 0.34m radius for short-reach targets.
        radii = (0.34, 0.40, 0.46, 0.52)
        n_ang = 12
        ring: list[np.ndarray] = []
        for r in radii:
            for k in range(n_ang):
                a = 2 * np.pi * k / n_ang
                bx = float(target_xy[0] + r * np.cos(a))
                by = float(target_xy[1] + r * np.sin(a))
                theta = float(np.arctan2(target_xy[1] - by, target_xy[0] - bx))
                c, s = np.cos(theta), np.sin(theta)
                m = np.eye(4)
                m[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                m[0, 3], m[1, 3], m[2, 3] = bx, by, base_z
                ring.append(m)
        # Prefer standoffs closest to where navigation already parked the base.
        ring.sort(key=lambda m: (m[0, 3] - base_xy[0]) ** 2 + (m[1, 3] - base_xy[1]) ** 2)
        # Reject any standoff whose instantaneous snap from the parked pose would
        # cross a wall/obstacle (the base "teleport through a wall" artifact):
        # keep only candidates with a fully collision-free snap path. The
        # verified hint is gated the same way -- if nav could not reach it, its
        # snap crosses the wall too and must not be used. Filter lazily (nearest
        # first) and stop once the probe budget of clear standoffs is met.
        candidates = [c for c in candidates if self._snap_path_is_clear(c)]
        budget = 18
        n_ring_kept = 0
        for m in ring:
            if n_ring_kept >= budget:
                break
            if self._snap_path_is_clear(m):
                candidates.append(m)
                n_ring_kept += 1
        return candidates

    def _search_manip_base_pose(
        self, phase: str, target_name: str, verified_pose_7d: list[float] | None
    ) -> bool:
        """Snap the base to candidate standoffs and keep the first from which the
        real manip primitives build (IK-feasible against the live scene).

        This replaces open-loop reliance on the navigation parking pose: because
        the base is frozen during manipulation, we are free to place it at any
        standoff from which the grasp/place is actually reachable, using the true
        held-object orientation rather than a pre-sampled guess.
        """
        self._manip_policy.phase = phase
        # During PLACE the pickup object is held only by gripper friction; carry
        # it along with every base teleport so the search doesn't drop it.
        carry = self.config.task_config.pickup_obj_name if phase == PLACE else None
        candidates = self._manip_base_candidates(target_name, verified_pose_7d)
        # Rotate the candidate order by how many times this segment has already
        # been retried, so each retry commits to a genuinely DIFFERENT standoff
        # (base position + approach angle). Without this, the ring re-sorts by
        # proximity to the restored parked base and re-selects the same first
        # candidate every retry, reproducing the exact grasp that just failed
        # ("Object is not in grasp!"). A fresh approach angle is the cheapest way
        # to convert a marginal base-locked grasp into a secure one.
        attempt = self._retry_counts.get(phase, 0)
        if candidates and attempt:
            k = attempt % len(candidates)
            candidates = candidates[k:] + candidates[:k]
        # A candidate whose grasp pose alone is IK-feasible can still fail mid
        # execution: with the base frozen, an intermediate waypoint (pregrasp
        # standoff, lift) may be out of the arm's base-locked reach, which shows
        # up as a run of "IK failed (base-locked)" aborts. So we prefer the first
        # candidate from which EVERY planned waypoint is base-locked reachable
        # (strict pass) and only fall back to the first that merely builds
        # (lenient pass) when no candidate fully clears — never regressing below
        # the previous accept-first-build behaviour.
        first_built: int | None = None
        first_built_pose: np.ndarray | None = None
        for strict in (True, False):
            for i, base_pose in enumerate(candidates):
                if not strict and first_built is not None and i != first_built:
                    # Lenient pass: we already know the first buildable candidate.
                    continue
                self._snap_base_pose(base_pose, carry_object_name=carry)
                try:
                    self._manip_policy.reset(reset_retries=True)
                except ValueError:
                    continue
                if strict:
                    if first_built is None:
                        first_built = i
                        first_built_pose = base_pose.copy()
                    targets = list(self._manip_policy.target_poses.values())
                    if targets and not all(
                        bool(self._manip_policy.check_feasible_ik(np.asarray(p)))
                        for p in targets
                    ):
                        continue  # some waypoint unreachable base-locked; skip.
                # Pin the (holonomic) base here for the whole manip phase: unlike
                # the bolted fixed-base robot, the mobile base drifts under
                # arm/grasp reaction forces, shifting the grasp enough to miss.
                self._locked_base_pose = base_pose.copy()
                log.info(
                    f"[MOBILE PNP FSM] {phase} base found "
                    f"({'all-waypoint' if strict else 'build-only'} candidate "
                    f"{i}/{len(candidates)}) at "
                    f"({base_pose[0, 3]:.2f}, {base_pose[1, 3]:.2f})."
                )
                return True
            if first_built is None:
                # No candidate even builds: the lenient pass cannot help either.
                break
            if strict and first_built_pose is not None:
                # Re-snap/rebuild at the known-buildable candidate for the lenient
                # accept below (state was left on the last strict-rejected snap).
                self._snap_base_pose(first_built_pose, carry_object_name=carry)
                try:
                    self._manip_policy.reset(reset_retries=True)
                except ValueError:
                    break
                self._locked_base_pose = first_built_pose.copy()
                log.info(
                    f"[MOBILE PNP FSM] {phase} base found (build-only fallback "
                    f"candidate {first_built}/{len(candidates)}) at "
                    f"({first_built_pose[0, 3]:.2f}, {first_built_pose[1, 3]:.2f})."
                )
                return True
        log.warning(
            f"[MOBILE PNP FSM] {phase} build failed: no reachable base standoff for "
            f"'{target_name}' among {len(candidates)} candidates."
        )
        return False

    def _enter_pick(self) -> bool:
        """Build the pick primitives from the *navigated* base pose.

        Navigation already drives the base to the grasp-feasibility-verified
        standoff the sampler recorded (``robot_base_pose``, used as the A* goal
        override), so the parked pose is normally itself grasp-feasible. We
        therefore build in place at the navigated pose -- no ring search, no base
        teleport -- exactly as PLACE does. Only if the parked pose is infeasible
        (e.g. navigation parked short) do we fall back to a few nearby standoffs,
        reached via the wall-gated smooth base approach rather than a raw
        teleport, before finally retrying the segment (fresh navigation)."""
        self._manip_policy.phase = PICK
        current = self.task.env.current_robot.robot_view.base.pose.copy()
        attempt = self._retry_counts.get(PICK, 0)
        if attempt == 0:
            try:
                self._manip_policy.reset(reset_retries=True)
                self._locked_base_pose = current
                log.info(
                    "[MOBILE PNP FSM] NAV_TO_OBJ done → phase PICK "
                    f"(built at navigated pose ({current[0, 3]:.2f}, {current[1, 3]:.2f}))."
                )
                return True
            except ValueError:
                log.info(
                    "[MOBILE PNP FSM] PICK not feasible at navigated pose; "
                    "trying nearby standoffs."
                )
        else:
            # On a retry, the navigated-pose grasp already failed once; go
            # straight to the ring search, which rotates the standoff (a fresh
            # base position + approach angle each retry) to convert a marginal
            # base-locked grasp into a secure one. The snap to a ring standoff is
            # small (radius <=~0.5 m) and wall-gated, never a through-wall jump.
            log.info(
                f"[MOBILE PNP FSM] PICK retry {attempt}: searching a fresh standoff "
                "for a new grasp angle."
            )
        ok = self._search_manip_base_pose(
            PICK,
            self.config.task_config.pickup_obj_name,
            getattr(self.config.task_config, "robot_base_pose", None),
        )
        if ok:
            log.info("[MOBILE PNP FSM] NAV_TO_OBJ done → phase PICK (nearby standoff).")
        return ok

    def _enter_place(self) -> bool:
        """Build the place primitives from the *navigated* base pose.

        Unlike PICK (empty gripper), teleport-searching standoffs during PLACE
        jostles the held object out of the force grasp, so we avoid it: the base
        has already been navigated to the receptacle-facing place hint, and the
        planner's ``_nearest_reachable_place_pose`` fallback handles receptacles
        whose centre sits past the arm's reach. We therefore build in place at the
        current pose. Only if that is infeasible do we try a few nearby standoffs
        (carrying the held object along), accepting the small jostle as a last
        resort before falling back to a segment retry (fresh navigation).
        """
        self._manip_policy.phase = PLACE
        current = self.task.env.current_robot.robot_view.base.pose.copy()
        try:
            self._manip_policy.reset(reset_retries=True)
            self._locked_base_pose = current
            log.info(
                "[MOBILE PNP FSM] NAV_TO_RECEPTACLE done → phase PLACE "
                f"(built at navigated pose ({current[0, 3]:.2f}, {current[1, 3]:.2f}))."
            )
            return True
        except ValueError:
            log.info(
                "[MOBILE PNP FSM] PLACE not feasible at navigated pose; "
                "trying nearby standoffs (carrying object)."
            )
        ok = self._search_manip_base_pose(
            PLACE,
            self.config.task_config.place_receptacle_name,
            getattr(self.config.task_config, "place_robot_base_pose", None),
        )
        if ok:
            log.info("[MOBILE PNP FSM] NAV_TO_RECEPTACLE done → phase PLACE (nearby standoff).")
        return ok

    def _done_action(self) -> dict[str, Any]:
        gripper_ids = self.task.env.current_robot.robot_view.get_gripper_movegroup_ids()
        action = self.task.env.current_robot.robot_view.get_ctrl_dict(["arm"] + gripper_ids)
        action["done"] = True
        return action

    # ------------------------------------------------------------------ #
    # Main step                                                           #
    # ------------------------------------------------------------------ #
    def get_action(self, observation: Any) -> dict[str, Any]:
        # Bounded loop so phase transitions on a single step resolve without
        # emitting stale (e.g. final navigation) commands.
        for _ in range(len(self.get_all_phases()) + 2):
            if self._phase in (NAV_TO_OBJ, NAV_TO_RECEPTACLE):
                # While transporting a grasped object, keep it rigidly fixed to
                # the gripper so base acceleration can't fling it out of the hand.
                if self._phase == NAV_TO_RECEPTACLE:
                    self._apply_grasp_lock()
                nav_action = self._nav_policy.get_action(observation)
                nav_done = nav_action.pop("done", False)
                if not nav_done:
                    robot_view = self.task.env.current_robot.robot_view
                    gripper_ids = robot_view.get_gripper_movegroup_ids()
                    # Actively hold the arm (+ gripper) at their held setpoints
                    # during transit and slew-limit the base pose command so
                    # driving/turning the holonomic base is gentle enough not to
                    # fling the stowed/held arm (which otherwise shows up as
                    # unnaturally fast arm motion during navigation). This applies
                    # to BOTH nav segments; NAV_TO_RECEPTACLE additionally holds
                    # the grasped object rigidly (grasp lock above).
                    hold = robot_view.get_ctrl_dict(["arm"] + gripper_ids)
                    # Pin the arm to a compact stow pose while navigating instead
                    # of the (possibly uninitialised, near-zero) actuator ctrl,
                    # which would otherwise straighten the arm out into a
                    # near-limit extended pose the moment nav starts. The setpoint
                    # ramps smoothly from the entry pose toward the stow config so
                    # the arm retracts naturally, then the position servo actively
                    # holds the tuck and rejects any base-motion-induced drift.
                    for mg, qpos in self._held_nav_arm_setpoints(robot_view).items():
                        hold[mg] = qpos
                    hold[_BASE_MG_ID] = self._slew_base_command(nav_action[_BASE_MG_ID])
                    return hold

                if self._phase == NAV_TO_OBJ:
                    parked_pose = self.task.env.current_robot.robot_view.base.pose.copy()
                    if not self._enter_pick():
                        if not self._retry_segment(PICK):
                            self._phase = DONE
                        continue
                    # Drive smoothly from the parked pose to the searched standoff
                    # instead of teleporting there (sets phase to APPROACH_PICK,
                    # or straight to PICK if approach is disabled).
                    self._begin_base_approach(APPROACH_PICK, parked_pose)
                else:
                    parked_pose = self.task.env.current_robot.robot_view.base.pose.copy()
                    if not self._enter_place():
                        if not self._retry_segment(PLACE):
                            self._phase = DONE
                        continue
                    # Keep the transport grasp lock active through the PLACE
                    # preplace move (arm repositioning above the receptacle while
                    # holding); it is released at the final lowering/open segment
                    # inside the PICK/PLACE branch so the object can be set down.
                    self._begin_base_approach(APPROACH_PLACE, parked_pose)
                continue

            if self._phase in (APPROACH_PICK, APPROACH_PLACE):
                action = self._step_base_approach(observation)
                if action is not None:
                    return action
                # Arrived at the standoff: re-pin the base exactly there and hand
                # off to the manipulation phase built during the search.
                self._snap_base_pose(self._locked_base_pose, carry_object_name=self._approach_carry)
                self._phase = _APPROACH_MANIP[self._phase]
                self._approach_target = None
                self._approach_carry = None
                continue

            if self._phase in (PICK, PLACE):
                # Re-pin the holonomic base each step: it otherwise drifts under
                # arm/grasp reaction forces (the bolted fixed-base robot cannot),
                # shifting the grasp/place enough to miss. Pinning qpos+ctrl every
                # step makes the base effectively rigid for the manip phase.
                if self._locked_base_pose is not None:
                    self._snap_base_pose(self._locked_base_pose)
                # Hold the object rigidly through the PLACE preplace repositioning,
                # then release it at the lowering/open segment so it can be set
                # down. Releasing earlier let it slip out at preplace entry.
                if self._phase == PLACE and self._grasp_offset is not None:
                    if self._manip_policy.get_phase() == "preplace":
                        self._apply_grasp_lock()
                    else:
                        # Reached the lowering/open segment: the object is being
                        # set down. Stop the grasp lock and mark it released so a
                        # later failure won't retry and undo the placement.
                        self._grasp_offset = None
                        self._place_released = True
                try:
                    manip_action = self._manip_policy.get_action(observation)
                except (ValueError, AssertionError) as e:
                    # A mid-execution retry re-plans the base-locked trajectory
                    # and can raise ValueError if the object/base shifted enough
                    # to make it infeasible; primitive-execution invariants can
                    # also trip an AssertionError (e.g. a degenerate 0-length
                    # segment). Fail this phase gracefully (the episode is then
                    # discarded by the datagen worker) instead of crashing.
                    log.warning(
                        f"[MOBILE PNP FSM] {self._phase} aborted mid-execution: "
                        f"{type(e).__name__}: {e}"
                    )
                    # If the object is already placed, never retry: a retry
                    # restores the checkpoint and teleports the placed object back
                    # into the gripper, destroying a successful placement. Finish
                    # and let the task judge success on the placed object.
                    if self._phase == PLACE and self._place_released:
                        log.info(
                            "[MOBILE PNP FSM] PLACE post-release failure ignored "
                            "(object already set down) → phase DONE"
                        )
                        self._phase = DONE
                    elif not self._retry_segment(self._phase):
                        self._phase = DONE
                    continue
                manip_done = manip_action.pop("done", False)
                manip_failed = manip_action.pop("success", None) is False
                # Never command the base during manipulation: drop the key so
                # the task freezes the base at the parked pose.
                manip_action.pop(_BASE_MG_ID, None)

                if not manip_done:
                    return self._smooth_arm_action(manip_action)

                if manip_failed:
                    log.warning(f"[MOBILE PNP FSM] manipulation phase {self._phase} FAILED")
                    if self._phase == PLACE and self._place_released:
                        log.info(
                            "[MOBILE PNP FSM] PLACE post-release failure ignored "
                            "(object already set down) → phase DONE"
                        )
                        self._phase = DONE
                    elif not self._retry_segment(self._phase):
                        self._phase = DONE
                    continue

                if self._phase == PICK:
                    # Pick succeeded: cache the held-object state (and recording
                    # length) so the place segment can be retried without redoing
                    # navigation+pick, with the failed attempt truncated out.
                    self._checkpoints[PLACE] = self._make_checkpoint()
                    # Lock the object to the gripper for the transport nav.
                    self._capture_grasp_lock()
                    self.task.set_nav_target(self.config.task_config.place_receptacle_name)
                    # Drive to the place-feasibility-verified base pose near the
                    # receptacle (Avenue A); None => fall back to goal sampling.
                    self.task.set_nav_goal_override(
                        getattr(self.config.task_config, "place_robot_base_pose", None)
                    )
                    self._nav_policy.reset()
                    self._nav_step_in_segment = 0
                    self._nav_arm_hold = None
                    self._phase = NAV_TO_RECEPTACLE
                    log.info("[MOBILE PNP FSM] PICK done → phase NAV_TO_RECEPTACLE")
                else:
                    self._phase = DONE
                    log.info("[MOBILE PNP FSM] PLACE done → phase DONE")
                continue

            # DONE
            return self._done_action()

        raise RuntimeError("MobilePickAndPlace FSM exceeded transition budget in a single step")

    # ------------------------------------------------------------------ #
    # Scene auxiliary objects (grasp-collision bodies for the planner)    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def add_auxiliary_objects(config: MlSpacesExpConfig, spec: MjSpec) -> None:
        PlannerPolicy.add_auxiliary_objects(config, spec)
        manip_cfg = config.policy_config.manip_policy_config
        if manip_cfg.filter_colliding_grasps:
            add_grasp_collision_bodies(
                spec,
                manip_cfg.grasp_collision_batch_size,
                manip_cfg.grasp_width,
                manip_cfg.grasp_length,
                manip_cfg.grasp_height,
                np.array(manip_cfg.grasp_base_pos),
            )
