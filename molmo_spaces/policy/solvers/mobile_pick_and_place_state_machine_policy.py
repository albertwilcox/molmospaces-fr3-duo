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
from molmo_spaces.utils.pose import pos_quat_to_pose_mat

log = logging.getLogger(__name__)

# Phase labels.
NAV_TO_OBJ = "NAV_TO_OBJ"
PICK = "PICK"
NAV_TO_RECEPTACLE = "NAV_TO_RECEPTACLE"
PLACE = "PLACE"
DONE = "DONE"

_BASE_MG_ID = "base"


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
        self._manip_config = config.model_copy(update={"policy_config": manip_policy_config})
        self._manip_policy = _MobileManipPlannerPolicy(self._manip_config, task)

        self._phase: str = NAV_TO_OBJ

        # --- Subtask checkpoint / retry ----------------------------------- #
        # Cache the full MuJoCo state at the start of each nav+manip segment so
        # a failed subtask can restore the last good state and retry (with a
        # freshly-sampled parking pose) instead of discarding the whole episode.
        self._checkpoints: dict[str, np.ndarray] = {}
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
        self._restore_state(ckpt)
        task_cfg = self.config.task_config
        self.task.set_nav_target(getattr(task_cfg, info["target_attr"]))
        # First retry re-drives to the feasibility-verified override; later
        # retries drop it to force a freshly-sampled standoff, adding the
        # variation needed to escape a reproducibly-infeasible parking pose.
        override = getattr(task_cfg, info["override_attr"], None) if used == 0 else None
        self.task.set_nav_goal_override(override)
        self._nav_policy.reset()
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
        return self._phase

    def get_all_phases(self) -> dict[str, int]:
        return {NAV_TO_OBJ: 0, PICK: 1, NAV_TO_RECEPTACLE: 2, PLACE: 3, DONE: 4}

    def reset(self) -> None:
        self._phase = NAV_TO_OBJ
        self._checkpoints = {}
        self._retry_counts = {}
        self.task.set_nav_target(self.config.task_config.pickup_obj_name)
        # Drive to the grasp-feasibility-verified base pose the sampler recorded
        # (Avenue A), not the closest navigable cell.
        self.task.set_nav_goal_override(getattr(self.config.task_config, "robot_base_pose", None))
        self._nav_policy.reset()
        # Cache the pristine post-sample state so the whole pick segment
        # (navigate-to-object + pick) can be retried from scratch.
        self._checkpoints[PICK] = self._capture_state()
        log.info("[MOBILE PNP FSM] reset → phase NAV_TO_OBJ")

    # ------------------------------------------------------------------ #
    # Phase transitions                                                   #
    # ------------------------------------------------------------------ #
    def _snap_base_pose(self, pose_mat: np.ndarray) -> None:
        """Snap the (frozen) base to ``pose_mat`` and hold it there.

        Sets base qpos AND the holonomic actuator setpoint (else the position
        servo drives the base back to the stale nav setpoint during manip), then
        forwards so kinematics/IK see the new base pose.
        """
        robot_view = self.task.env.current_robot.robot_view
        robot_view.base.pose = pose_mat
        x, y = float(pose_mat[0, 3]), float(pose_mat[1, 3])
        theta = float(np.arctan2(pose_mat[1, 0], pose_mat[0, 0]))
        robot_view.base.ctrl = np.array([x, y, theta])
        mujoco.mj_forward(self.task.env.current_model, self.task.env.current_data)

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

        radii = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
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
        candidates.extend(ring)
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
        candidates = self._manip_base_candidates(target_name, verified_pose_7d)
        for i, base_pose in enumerate(candidates):
            self._snap_base_pose(base_pose)
            try:
                self._manip_policy.reset(reset_retries=True)
            except ValueError:
                continue
            log.info(
                f"[MOBILE PNP FSM] {phase} base found (candidate {i}/{len(candidates)}) at "
                f"({base_pose[0, 3]:.2f}, {base_pose[1, 3]:.2f})."
            )
            return True
        log.warning(
            f"[MOBILE PNP FSM] {phase} build failed: no reachable base standoff for "
            f"'{target_name}' among {len(candidates)} candidates."
        )
        return False

    def _enter_pick(self) -> bool:
        """Park the base at a reachable standoff and build the pick primitives.
        Returns False if the object is unreachable from every candidate."""
        ok = self._search_manip_base_pose(
            PICK,
            self.config.task_config.pickup_obj_name,
            getattr(self.config.task_config, "robot_base_pose", None),
        )
        if ok:
            log.info("[MOBILE PNP FSM] NAV_TO_OBJ done → phase PICK")
        return ok

    def _enter_place(self) -> bool:
        """Park the base at a reachable standoff and build the place primitives."""
        ok = self._search_manip_base_pose(
            PLACE,
            self.config.task_config.place_receptacle_name,
            getattr(self.config.task_config, "place_robot_base_pose", None),
        )
        if ok:
            log.info("[MOBILE PNP FSM] NAV_TO_RECEPTACLE done → phase PLACE")
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
                nav_action = self._nav_policy.get_action(observation)
                nav_done = nav_action.pop("done", False)
                if not nav_done:
                    # Drive only the base; arm stays stowed / holding, gripper
                    # holds its state (absent keys => held stationary).
                    return {_BASE_MG_ID: nav_action[_BASE_MG_ID]}

                if self._phase == NAV_TO_OBJ:
                    if not self._enter_pick():
                        if not self._retry_segment(PICK):
                            self._phase = DONE
                        continue
                    self._phase = PICK
                else:
                    if not self._enter_place():
                        if not self._retry_segment(PLACE):
                            self._phase = DONE
                        continue
                    self._phase = PLACE
                continue

            if self._phase in (PICK, PLACE):
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
                    if not self._retry_segment(self._phase):
                        self._phase = DONE
                    continue
                manip_done = manip_action.pop("done", False)
                manip_failed = manip_action.pop("success", None) is False
                # Never command the base during manipulation: drop the key so
                # the task freezes the base at the parked pose.
                manip_action.pop(_BASE_MG_ID, None)

                if not manip_done:
                    return manip_action

                if manip_failed:
                    log.warning(f"[MOBILE PNP FSM] manipulation phase {self._phase} FAILED")
                    if not self._retry_segment(self._phase):
                        self._phase = DONE
                    continue

                if self._phase == PICK:
                    # Pick succeeded: cache the held-object state so the place
                    # segment can be retried without redoing navigation+pick.
                    self._checkpoints[PLACE] = self._capture_state()
                    self.task.set_nav_target(self.config.task_config.place_receptacle_name)
                    # Drive to the place-feasibility-verified base pose near the
                    # receptacle (Avenue A); None => fall back to goal sampling.
                    self.task.set_nav_goal_override(
                        getattr(self.config.task_config, "place_robot_base_pose", None)
                    )
                    self._nav_policy.reset()
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
