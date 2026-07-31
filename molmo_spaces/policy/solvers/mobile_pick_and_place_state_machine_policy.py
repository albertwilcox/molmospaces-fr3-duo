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

    @property
    def planners(self) -> dict:
        return {}

    def get_phase(self) -> str:
        return self._phase

    def get_all_phases(self) -> dict[str, int]:
        return {NAV_TO_OBJ: 0, PICK: 1, NAV_TO_RECEPTACLE: 2, PLACE: 3, DONE: 4}

    def reset(self) -> None:
        self._phase = NAV_TO_OBJ
        self.task.set_nav_target(self.config.task_config.pickup_obj_name)
        self._nav_policy.reset()
        log.info("[MOBILE PNP FSM] reset → phase NAV_TO_OBJ")

    # ------------------------------------------------------------------ #
    # Phase transitions                                                   #
    # ------------------------------------------------------------------ #
    def _enter_pick(self) -> bool:
        """Build the pick primitives from the parked base. Returns False if the
        object is unreachable from where navigation parked."""
        self._manip_policy.phase = PICK
        try:
            self._manip_policy.reset()
        except ValueError as e:
            log.warning(f"[MOBILE PNP FSM] PICK build failed (unreachable): {e}")
            return False
        log.info("[MOBILE PNP FSM] NAV_TO_OBJ done → phase PICK")
        return True

    def _enter_place(self) -> bool:
        """Build the place primitives from the parked base over the receptacle."""
        self._manip_policy.phase = PLACE
        try:
            self._manip_policy.reset(reset_retries=True)
        except ValueError as e:
            log.warning(f"[MOBILE PNP FSM] PLACE build failed (unreachable): {e}")
            return False
        log.info("[MOBILE PNP FSM] NAV_TO_RECEPTACLE done → phase PLACE")
        return True

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
                        self._phase = DONE
                        continue
                    self._phase = PICK
                else:
                    if not self._enter_place():
                        self._phase = DONE
                        continue
                    self._phase = PLACE
                continue

            if self._phase in (PICK, PLACE):
                manip_action = self._manip_policy.get_action(observation)
                manip_done = manip_action.pop("done", False)
                manip_failed = manip_action.pop("success", None) is False
                # Never command the base during manipulation: drop the key so
                # the task freezes the base at the parked pose.
                manip_action.pop(_BASE_MG_ID, None)

                if not manip_done:
                    return manip_action

                if manip_failed:
                    log.warning(f"[MOBILE PNP FSM] manipulation phase {self._phase} FAILED")
                    self._phase = DONE
                    continue

                if self._phase == PICK:
                    self.task.set_nav_target(self.config.task_config.place_receptacle_name)
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
