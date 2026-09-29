"""Operational-space (Cartesian) end-effector pose controller.

This controller lets a position-actuated arm be driven with **delta end-effector
pose** commands, matching robosuite/robocasa's default ``OSC_POSE`` arm
controller. It exists so a policy trained in RoboCasa (which emits 6-D delta EEF
actions through an ``OSC_POSE`` ``HybridMobileBase`` composite controller) can
drive the MolmoSpaces Franka without swapping the policy to a
joint-space action space.

Command conventions from robosuite's ``OSC_POSE`` (PandaOmron default config):

* The 6-D action is a delta ``[dpos(3), daxisangle(3)]`` expressed in the robot
  **base** frame (``input_ref_frame="base"``), nominally normalized to
  ``[-1, 1]``.
* robosuite scales the (clipped) action affinely to physical units using the
  controller's ``output_max``/``output_min``: position ``±0.05 m`` and rotation
  ``±0.5 rad`` per axis by default. For the symmetric ranges used by robocasa
  this is simply ``dpos = clip(a, -1, 1) * 0.05`` and
  ``drot = clip(a, -1, 1) * 0.5``.
* Goals use either the achieved or previous desired pose, according to
  ``HybridMobileBase`` mode. Rotation deltas premultiply in the base frame.

Because MolmoSpaces arms use **position actuators** (not torque), the scaled
Cartesian goal is realized by solving inverse kinematics for joint-position
targets (via :class:`MlSpacesKinematics`, seeded at the current joints with only
the arm move group unlocked) rather than by an operational-space torque law. The
command scaling matches OSC, but achieved motion and contact dynamics do not
necessarily match. Best-effort IK and bounded joint steps are explicitly tracked.
"""

import logging
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.controllers.abstract import AbstractPositionController
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.robot_views.abstract import MoveGroup, RobotView

log = logging.getLogger(__name__)

# robosuite default ``OSC_POSE`` PandaOmron output limits (see
# ``robosuite/controllers/config/robots/default_pandaomron.json``): position
# ``±0.05 m``, orientation ``±0.5 rad`` per axis, with ``input_max/min = ±1``.
_DEFAULT_POS_SCALE = 0.05
_DEFAULT_ROT_SCALE = 0.5


class OSCPoseController(AbstractPositionController):
    """Cartesian delta-EEF-pose controller for a position-actuated arm.

    Args:
        robot_move_group: The arm move group to control. Its leaf frame is the
            end-effector control point (the gripper grasp site for the Franka).
        move_group_id: The arm move group's ID in the robot view (e.g. ``"arm"``).
        kinematics: A kinematics solver for the robot, used for IK. This uses its
            own scratch MuJoCo model/data, so IK does not perturb the live sim.
        robot_view: The (live) robot view, used to read the base pose and the
            full joint-position dict that seeds IK.
        pos_scale: Per-axis position scaling applied to the clipped action
            (robosuite ``OSC_POSE`` ``output_max`` for position).
        rot_scale: Per-axis axis-angle scaling applied to the clipped action
            (robosuite ``OSC_POSE`` ``output_max`` for orientation).
        input_clip: Clip range applied to the raw action before scaling, matching
            robosuite's ``input_max``/``input_min``.
        ik_damping: Damped-least-squares damping for IK.
        ik_max_iter: Maximum IK Newton iterations.
        ik_eps: IK convergence threshold on the pose error norm.
    """

    def __init__(
        self,
        robot_move_group: MoveGroup,
        move_group_id: str,
        kinematics: MlSpacesKinematics,
        robot_view: RobotView,
        pos_scale: float = _DEFAULT_POS_SCALE,
        rot_scale: float = _DEFAULT_ROT_SCALE,
        input_clip: float = 1.0,
        ik_damping: float = 1e-4,
        ik_max_iter: int = 200,
        ik_eps: float = 1e-4,
    ) -> None:
        super().__init__(robot_move_group)

        self.move_group_id = move_group_id
        self._kinematics = kinematics
        self._robot_view = robot_view

        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.input_clip = input_clip
        self.ik_damping = ik_damping
        self.ik_max_iter = ik_max_iter
        self.ik_eps = ik_eps

        self.ctrl_dim = robot_move_group.n_actuators
        self.ctrl_range = robot_move_group.ctrl_limits

        self._stationary = True
        self._joint_target = self.robot_move_group.joint_pos.copy()
        self.goal_update_mode: Literal["achieved", "desired"] = "achieved"
        self.ik_nonconverged_steps = 0
        self.command_steps = 0
        self.reset()

    @property
    def target(self) -> np.ndarray:
        return self._target

    @target.setter
    def target(self, value: np.ndarray) -> None:
        self._target = value

    @property
    def target_pos(self) -> np.ndarray:
        """The joint-position target realizing the commanded eef pose."""
        return self._joint_target.copy()

    @property
    def stationary(self) -> bool:
        return self._stationary

    def _clip_and_scale(self, eef_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Clip the raw 6-D action to ``[-input_clip, input_clip]`` and scale to
        physical position (m) and axis-angle (rad) deltas."""
        clipped = np.clip(eef_delta, -self.input_clip, self.input_clip)
        dpos = clipped[:3] * self.pos_scale
        drot = clipped[3:6] * self.rot_scale
        return dpos, drot

    def set_target(self, eef_delta: np.ndarray) -> None:
        """Set a delta end-effector pose target (base frame, robosuite layout).

        Args:
            eef_delta: 6-D ``[dpos(3), daxisangle(3)]`` action, nominally in
                ``[-1, 1]``, expressed in the robot base frame.
        """
        eef_delta = np.asarray(eef_delta, dtype=np.float64).reshape(-1)
        assert eef_delta.shape[0] == 6, (
            f"OSCPoseController expects a 6-D [dpos(3), daxisangle(3)] action, got {eef_delta.shape}"
        )
        if not np.isfinite(eef_delta).all():
            raise ValueError("OSC action must be finite")
        stamp = self.robot_move_group.mj_data.time
        if self._last_command_time == stamp and np.array_equal(eef_delta, self.target):
            return
        self._last_command_time = stamp
        self.command_steps += 1

        self._stationary = False
        self.target = eef_delta

        dpos, drot = self._clip_and_scale(eef_delta)

        # Current eef pose in the robot base frame (goal_update_mode="achieved").
        cur_pose_base = (
            self._goal_pose_base
            if self.goal_update_mode == "desired"
            else self.robot_move_group.leaf_frame_to_robot
        )
        cur_pos_base = cur_pose_base[:3, 3]
        cur_ori_base = cur_pose_base[:3, :3]

        goal_pos_base = cur_pos_base + dpos
        rot_delta = R.from_rotvec(drot).as_matrix()
        goal_ori_base = rot_delta @ cur_ori_base

        goal_pose_base = np.eye(4)
        goal_pose_base[:3, :3] = goal_ori_base
        goal_pose_base[:3, 3] = goal_pos_base
        self._goal_pose_base = goal_pose_base.copy()

        q0 = self._robot_view.get_qpos_dict()
        base_pose = self._robot_view.base.pose

        solution = self._kinematics.ik(
            move_group_id=self.move_group_id,
            pose=goal_pose_base,
            unlocked_move_group_ids=[self.move_group_id],
            q0=q0,
            base_pose=base_pose,
            rel_to_base=True,
            eps=self.ik_eps,
            max_iter=self.ik_max_iter,
            damping=self.ik_damping,
            return_best_effort=True,
            max_step_size=0.2,
        )

        if solution is not None:
            joint_target = np.asarray(solution[self.move_group_id], dtype=np.float64)
            if not np.isfinite(joint_target).all():
                raise FloatingPointError("IK returned non-finite joint targets")
            current = self.robot_move_group.joint_pos
            joint_delta = joint_target - current
            largest_step = float(np.max(np.abs(joint_delta)))
            if largest_step > 0.2:
                joint_delta *= 0.2 / largest_step
            joint_target = current + joint_delta
            self._joint_target = np.clip(joint_target, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
            if not self._kinematics.last_ik_converged:
                self.ik_nonconverged_steps += 1
                if self.ik_nonconverged_steps == 1 or self.ik_nonconverged_steps % 100 == 0:
                    log.warning(
                        "[OSCPoseController][%s] best-effort IK: residual=%.5g, nonconverged=%d/%d",
                        self.move_group_id,
                        self._kinematics.last_ik_error,
                        self.ik_nonconverged_steps,
                        self.command_steps,
                    )
        else:
            # With ``return_best_effort=True`` this should not happen (IK returns
            # the partially-converged iterate), but guard defensively: hold the
            # last joint target rather than jumping.
            log.warning(
                "[OSCPoseController][%s] IK returned no solution for goal pose; holding last target.",
                self.move_group_id,
            )

    def set_to_stationary(self) -> None:
        """Hold the arm at its current joint positions."""
        self._stationary = True
        self.target = np.zeros(6)
        self._joint_target = self.robot_move_group.joint_pos.copy()
        self._goal_pose_base = self.robot_move_group.leaf_frame_to_robot.copy()
        self._last_command_time = None

    def compute_ctrl_inputs(self) -> np.ndarray:
        """Return the joint-position targets for the arm's position actuators."""
        return self._joint_target.copy()

    def reset(self) -> None:
        self.set_to_stationary()
