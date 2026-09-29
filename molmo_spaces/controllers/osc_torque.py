"""Task-space impedance with moving-base damping and nullspace posture control."""

from typing import Literal

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.transform import Rotation

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.robots.robot_views.abstract import MoveGroup, RobotView


class TorqueOSCConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    kp: float = Field(default=150.0, gt=0, allow_inf_nan=False)
    nullspace_kp: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    velocity_reference: Literal["tcp", "arm_root"] = "tcp"


def _refresh_control_dynamics(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    # Refresh OSC's kinematics, inertia and bias without rerunning collision,
    # constraint solving, actuation or sensors before the actual physics step.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_tendon(model, data)
    mujoco.mj_makeM(model, data)
    mujoco.mj_comVel(model, data)
    mujoco.mj_passive(model, data)
    mujoco.mj_rne(model, data, 0, data.qfrc_bias)


class TorqueOSCController(Controller):
    def __init__(
        self,
        robot_move_group: MoveGroup,
        robot_view: RobotView,
        config: TorqueOSCConfig | None = None,
    ) -> None:
        super().__init__(robot_move_group)
        self.config = TorqueOSCConfig() if config is None else config
        self._robot_view = robot_view
        self._model = robot_move_group.mj_model
        self._data = robot_move_group.mj_data
        self._dofs = np.asarray(robot_move_group.joint_veladr)
        self._mass = np.zeros((self._model.nv, self._model.nv))
        self._jac_pos = np.zeros((3, self._model.nv))
        self._jac_rot = np.zeros_like(self._jac_pos)
        self._base_jac_pos = np.zeros_like(self._jac_pos)
        self._base_jac_rot = np.zeros_like(self._jac_pos)
        self._velocity_reference_body = (
            robot_move_group.root_body_id
            if self.config.velocity_reference == "arm_root"
            else robot_view.base.root_body_id
        )
        self.goal_update_mode: Literal["achieved", "desired"] = "achieved"
        self.command_steps = 0
        self.torque_saturation_steps = 0
        self.last_torques = np.zeros(len(self._dofs))
        self.reset()

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def stationary(self) -> bool:
        return self._stationary

    def set_target(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Torque OSC requires a finite six-dimensional Cartesian action")
        if self._last_command_time == self._data.time and np.array_equal(target, self._target):
            return
        self._last_command_time = self._data.time
        self._target = target.copy()
        self.command_steps += 1
        self._stationary = False
        reference = (
            self._goal_pose_base
            if self.goal_update_mode == "desired"
            else self.robot_move_group.leaf_frame_to_robot
        )
        delta = np.clip(target, -1, 1)
        goal = reference.copy()
        goal[:3, 3] += 0.05 * delta[:3]
        goal[:3, :3] = Rotation.from_rotvec(0.5 * delta[3:]).as_matrix() @ reference[:3, :3]
        self._goal_pose_base = goal

    def set_to_stationary(self) -> None:
        self._stationary = True
        self._target = np.zeros(6)
        self._goal_pose_base = self.robot_move_group.leaf_frame_to_robot.copy()
        self._last_command_time = None

    def reset(self) -> None:
        self._rest_joints = self.robot_move_group.joint_pos.copy()
        self.set_to_stationary()

    def compute_ctrl_inputs(self) -> np.ndarray:
        model, data = self._model, self._data
        _refresh_control_dynamics(model, data)
        mujoco.mj_jacSite(
            model, data, self._jac_pos, self._jac_rot, self.robot_move_group.leaf_frame_id
        )
        mujoco.mj_jacBody(
            model,
            data,
            self._base_jac_pos,
            self._base_jac_rot,
            self._velocity_reference_body,
        )
        mujoco.mj_fullM(model, self._mass, data.qM)
        mass = self._mass[np.ix_(self._dofs, self._dofs)]
        jac = np.concatenate([self._jac_pos[:, self._dofs], self._jac_rot[:, self._dofs]])
        inverse_mass_jac = np.linalg.solve(mass, jac.T)
        inverse_inertia = jac @ inverse_mass_jac
        inertia = np.linalg.pinv(inverse_inertia)
        inertia_pos = np.linalg.pinv(inverse_inertia[:3, :3])
        inertia_rot = np.linalg.pinv(inverse_inertia[3:, 3:])
        nullspace = np.eye(len(self._dofs)) - inverse_mass_jac @ inertia @ jac

        current = self.robot_move_group.leaf_frame_to_world
        goal = self._robot_view.base.pose @ self._goal_pose_base
        position_error = goal[:3, 3] - current[:3, 3]
        rotation_error = 0.5 * np.cross(current[:3, :3].T, goal[:3, :3].T).sum(axis=0)
        base_linear_velocity = self._base_jac_pos @ data.qvel
        base_angular_velocity = self._base_jac_rot @ data.qvel
        base_point_velocity = base_linear_velocity
        if self.config.velocity_reference == "tcp":
            base_to_eef = current[:3, 3] - data.xpos[self._velocity_reference_body]
            # Native RoboCasa's opt-in legacy mode instead subtracts arm-root velocity.
            base_point_velocity = base_point_velocity + np.cross(base_angular_velocity, base_to_eef)
        relative_linear_velocity = self._jac_pos @ data.qvel - base_point_velocity
        relative_angular_velocity = self._jac_rot @ data.qvel - base_angular_velocity
        kd = 2 * np.sqrt(self.config.kp)
        force = inertia_pos @ (self.config.kp * position_error - kd * relative_linear_velocity)
        moment = inertia_rot @ (self.config.kp * rotation_error - kd * relative_angular_velocity)
        posture = (
            self.config.nullspace_kp * (self._rest_joints - self.robot_move_group.joint_pos)
            - 2 * np.sqrt(self.config.nullspace_kp) * data.qvel[self._dofs]
        )
        # Existing body gravcomp is already applied by MuJoCo, not by the actuator.
        compensation = data.qfrc_bias[self._dofs] - data.qfrc_gravcomp[self._dofs]
        torques = (
            jac.T @ np.concatenate([force, moment]) + compensation + nullspace.T @ mass @ posture
        )
        if not np.isfinite(torques).all():
            raise FloatingPointError("Torque OSC produced non-finite actuator commands")
        limits = self.robot_move_group.ctrl_limits
        clipped = np.clip(torques, limits[:, 0], limits[:, 1])
        self.torque_saturation_steps += int(np.any(clipped != torques))
        self.last_torques = clipped.copy()
        return clipped
