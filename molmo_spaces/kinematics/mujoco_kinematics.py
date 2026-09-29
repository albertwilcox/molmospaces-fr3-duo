"""
General-purpose forward and inverse kinematics solver for robots in MuJoCo.
This solver is not parallelizable and runs on the CPU. While fairly fast,
this is not suitable for large batches.
"""

from typing import Literal, TYPE_CHECKING
import logging

import mujoco
import numpy as np

from molmo_spaces.robots.robot_views.abstract import HoloJointsRobotBaseGroup, RobotView
from molmo_spaces.utils.linalg_utils import (
    inverse_homogeneous_matrix,
    relative_to_global_transform,
    transform_to_twist,
)

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import BaseRobotConfig


log = logging.getLogger(__name__)


class MlSpacesKinematics:
    """A general-purpose kinematics solver for robots in MuJoCo.

    This class provides methods for computing forward kinematics (FK), inverse kinematics (IK),
    and converting between joint velocities and end-effector twists. It works with any robot
    that can be represented in MuJoCo through the RobotView abstraction.

    The class handles both world-frame and robot-base-frame relative poses and twists,
    making it flexible for different use cases.
    """

    def __init__(
        self, robot_config: "BaseRobotConfig", reference_robot_view: RobotView | None = None
    ) -> None:
        """
        Create a kinematics solver for a robot.

        Args:
            robot_config: The robot configuration.
            reference_robot_view: Live robot whose fixed planar-base mounting height
                must match this scratch model.
        """
        spec = mujoco.MjSpec()
        robot_config.robot_cls.add_robot_to_scene(
            robot_config,
            spec,
            prefix=robot_config.robot_namespace,
            pos=[0.0, 0.0, 0.0],
            quat=[1.0, 0.0, 0.0, 0.0],
            strip_meshes=True,
        )

        self._mj_model = spec.compile()
        self._mj_data = mujoco.MjData(self._mj_model)
        self._robot_view = robot_config.robot_view_factory(
            self._mj_data, robot_config.robot_namespace
        )
        if reference_robot_view is not None and isinstance(reference_robot_view.base, HoloJointsRobotBaseGroup):
            # Planar qpos and its pose setter omit fixed body Z. A live/scratch
            # mismatch otherwise adds that height error to every Cartesian command.
            reference_base = reference_robot_view.base
            self._mj_model.body_pos[self._robot_view.base.root_body_id, 2] = (
                reference_base.mj_model.body_pos[reference_base.root_body_id, 2]
            )
        mujoco.mj_forward(self._mj_model, self._mj_data)

    def _constrain_state(self) -> None:
        """Constrain the current state to be within the joint limits.

        This method clips all joint positions to their respective limits. It should be
        called after any operation that might move joints outside their valid ranges.
        """
        for mg_id in self._robot_view.move_group_ids():
            mg = self._robot_view.get_move_group(mg_id)
            mg.joint_pos = np.clip(
                mg.joint_pos, mg.joint_pos_limits[:, 0], mg.joint_pos_limits[:, 1]
            )

    def fk(
        self,
        move_group_qpos: dict[str, np.ndarray],
        base_pose: np.ndarray,
        rel_to_base: bool = False,
    ) -> dict[str, np.ndarray]:
        """Compute forward kinematics for all move groups.

        This method computes the pose of each move group's leaf frame given the joint positions.
        The poses can be returned either relative to the world frame or relative to the robot base.

        Args:
            move_group_qpos: Dictionary mapping move group IDs to their joint positions
            base_pose: 4x4 pose matrix of the robot base.
            rel_to_base: If True, return poses relative to robot base. If False, return poses in world frame.

        Returns:
            Dictionary mapping move group IDs to their 4x4 pose matrices
        """
        self._robot_view.set_qpos_dict(move_group_qpos)
        self._robot_view.base.pose = base_pose
        mujoco.mj_kinematics(self._mj_model, self._mj_data)
        ret = {}
        for move_group_id in self._robot_view.move_group_ids():
            move_group = self._robot_view.get_move_group(move_group_id)
            if rel_to_base:
                ret[move_group_id] = move_group.leaf_frame_to_robot
            else:
                ret[move_group_id] = self._robot_view.base.pose @ move_group.leaf_frame_to_robot
        return ret

    def twist_to_joint_vel(
        self,
        move_group_id: str,
        twist: np.ndarray,
        unlocked_move_group_ids: list[str],
        q0: dict[str, np.ndarray],
        base_pose: np.ndarray,
        twist_frame: Literal["world", "base", "leaf"] = "world",
        damping: float = 1e-12,
    ):
        """Convert an end-effector twist to joint velocities.

        This method computes the joint velocities that would result in the specified
        end-effector twist. It uses damped least squares to handle singularities.

        Args:
            move_group_id: ID of the move group whose end-effector twist is specified
            twist: 6D twist vector (linear and angular velocities)
            unlocked_move_group_ids: List of move group IDs whose joints can move
            q0: Dictionary mapping move group IDs to their current joint positions
            base_pose: 4x4 pose matrix of the robot base.
            twist_frame: Frame of the twist, either in the world frame, the robot base frame, or the leaf frame of the move group.
            damping: Damping factor for the damped least squares solution

        Returns:
            Array of joint velocities that would achieve the desired twist

        Raises:
            ValueError: If q0 keys don't match move group IDs
        """
        if set(q0.keys()) != set(self._robot_view.move_group_ids()):
            raise ValueError(
                f"q0 keys must match move group ids: {set(q0.keys())} != {set(self._robot_view.move_group_ids())}"
            )
        self._robot_view.set_qpos_dict(q0)
        # Mobile q0 includes base joints: apply the requested standoff last.
        self._robot_view.base.pose = base_pose

        # convert twist to world frame
        if twist_frame == "base":
            twist[:3] = self._robot_view.base.pose[:3, :3] @ twist[:3]
            twist[3:] = self._robot_view.base.pose[:3, :3] @ twist[3:]
        elif twist_frame == "leaf":
            move_group = self._robot_view.get_move_group(move_group_id)
            twist[:3] = (
                self._robot_view.base.pose @ move_group.leaf_frame_to_robot[:3, :3] @ twist[:3]
            )
            twist[3:] = (
                self._robot_view.base.pose @ move_group.leaf_frame_to_robot[:3, :3] @ twist[3:]
            )
        else:
            assert twist_frame == "world"

        J = self._robot_view.get_jacobian(move_group_id, unlocked_move_group_ids)
        if (JJT_det := np.linalg.det(J @ J.T)) < 1e-20:
            log.warning(
                f"[MlSpacesKinematics][{self._robot_view.name}] IK Jacobian is rank deficient! det(JJ^T)={JJT_det:.0e}"
            )
        H = J @ J.T + damping * np.eye(J.shape[0])
        q_dot = J.T @ np.linalg.solve(H, twist)
        return q_dot

    def ik(
        self,
        move_group_id: str,
        pose: np.ndarray,
        unlocked_move_group_ids: list[str] | None,
        q0: dict[str, np.ndarray],
        base_pose: np.ndarray,
        rel_to_base: bool = False,
        eps: float = 1e-4,
        max_iter: int = 1000,
        damping: float = 1e-12,
        dt: float = 1.0,
        return_best_effort: bool = False,
        max_step_size: float | None = None,
    ) -> dict[str, np.ndarray] | None:
        """Solve inverse kinematics to reach a target pose.

        This method iteratively solves for joint positions that would place the end-effector
        at the target pose. It uses a damped least squares approach with velocity-based
        updates.

        Args:
            move_group_id: ID of the move group whose end-effector pose is specified
            pose: 4x4 target pose matrix
            unlocked_move_group_ids: List of move group IDs whose joints can move
            q0: Dictionary mapping move group IDs to their initial joint positions
            base_pose: 4x4 pose matrix of the robot base.
            rel_to_base: If True, target pose is relative to robot base. If False, pose is in world frame.
            eps: Convergence threshold for the error norm
            max_iter: Maximum number of iterations
            damping: Damping factor for the damped least squares solution
            dt: Time step for velocity integration
            max_step_size: Optional bound on each Newton update, uniformly rescaled
                to preserve its joint-space direction near singularities.
            return_best_effort: If True, return the best-effort (partially converged)
                joint solution when the error threshold is not met within
                ``max_iter``, instead of ``None``. The returned state is the
                lowest-error finite iterate, including the initial state.
                This is not a torque-controller dynamics model. Callers using ``None`` as
                a reachability test must leave this ``False`` (the default).

        Returns:
            Dictionary mapping move group IDs to their joint positions if successful.
            If IK does not converge: the best-effort partial solution when
            ``return_best_effort`` is True, else ``None``.

        Raises:
            ValueError: If q0 keys don't match move group IDs
        """
        if set(q0.keys()) != set(self._robot_view.move_group_ids()):
            raise ValueError(
                f"q0 keys must match move group ids: {set(q0.keys())} != {set(self._robot_view.move_group_ids())}"
            )
        if max_iter < 1 or eps <= 0 or damping < 0:
            raise ValueError("IK requires max_iter >= 1, eps > 0 and damping >= 0")
        if max_step_size is not None and (not np.isfinite(max_step_size) or max_step_size <= 0):
            raise ValueError("IK max_step_size must be finite and positive")
        if not np.isfinite(pose).all():
            raise ValueError("IK target pose must be finite")

        if unlocked_move_group_ids is None:
            unlocked_move_group_ids = self._robot_view.move_group_ids()

        self._robot_view.set_qpos_dict(q0)
        self._robot_view.base.pose = base_pose
        if rel_to_base:
            pose = self._robot_view.base.pose @ pose

        move_group = self._robot_view.get_move_group(move_group_id)
        best_error = float("inf")
        best_qpos = {key: value.copy() for key, value in q0.items()}
        self.last_ik_converged = False
        for i in range(max_iter):
            mujoco.mj_fwdPosition(self._mj_model, self._mj_data)
            mujoco.mj_sensorPos(self._mj_model, self._mj_data)

            ee_pose = relative_to_global_transform(
                move_group.leaf_frame_to_robot, self._robot_view.base.pose
            )

            err_trf = inverse_homogeneous_matrix(ee_pose) @ pose
            twist_lin, twist_ang = transform_to_twist(err_trf)

            err = np.concatenate([ee_pose[:3, :3] @ twist_lin, ee_pose[:3, :3] @ twist_ang])
            error_norm = float(np.linalg.norm(err))
            if not np.isfinite(error_norm):
                raise FloatingPointError("Non-finite IK pose error")
            if error_norm < best_error:
                best_error = error_norm
                if return_best_effort:
                    best_qpos = {key: value.copy() for key, value in self._robot_view.get_qpos_dict().items()}
            self.last_ik_error = best_error
            if error_norm < eps:
                succ = True
                self.last_ik_converged = True
                break
            elif i == max_iter - 1:
                succ = False
                break

            J: np.ndarray = self._robot_view.get_jacobian(move_group_id, unlocked_move_group_ids)
            if (JJT_det := np.linalg.det(J @ J.T)) < 1e-20:
                log.warning(
                    f"[MlSpacesKinematics][{self._robot_view.name}] IK Jacobian is rank deficient! det(JJ^T)={JJT_det:.0e}"
                )

            H = J @ J.T
            # Damped least squares: add ``damping`` to the diagonal in place
            # rather than allocating ``damping * np.eye(J.shape[0])`` every Newton
            # iteration (this ran ~300k+ times/episode in the IK hot loop).
            H.flat[:: H.shape[0] + 1] += damping
            q_dot = J.T @ np.linalg.solve(H, err)
            dq = q_dot * dt
            if max_step_size is not None:
                largest_step = float(np.max(np.abs(dq), initial=0))
                if largest_step > max_step_size:
                    dq *= max_step_size / largest_step

            j = 0
            for mg_id in unlocked_move_group_ids:
                mg = self._robot_view.get_move_group(mg_id)
                mg.joint_pos = mg.integrate_joint_vel(mg.joint_pos, dq[j : j + mg.vel_dim])
                j += mg.vel_dim
            self._constrain_state()

        if succ:
            return self._robot_view.get_qpos_dict()
        if return_best_effort:
            return best_qpos
        return None
