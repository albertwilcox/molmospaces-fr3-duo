# import mujoco
# import numpy as np
# from mujoco import MjData, MjModel
# from scipy.spatial.transform import Rotation as R

# from molmo_spaces.kinematics.mujoco_kinematics import MujocoKinematics
# from molmo_spaces.robots.robot_views import StretchDexRobotView
# from molmo_spaces.utils.linalg_utils import (
#     inverse_homogeneous_matrix,
#     relative_to_global_transform,
#     transform_to_twist,
# )


# class StretchKinematics(MujocoKinematics):
#     agent: StretchDexRobotView

#     def __init__(
#         self,
#         model: MjModel,
#         data: MjData | None,
#         namespace: str = "robot_0/",
#         fix_mobile_base: bool = False,
#         agent: StretchDexAgent = None,
#     ) -> None:
#         if data is None:
#             data = MjData(model)
#         agent = StretchDexAgent.from_model(model, namespace=namespace)
#         lo, hi = agent.joint_limits
#         super().__init__(model, data, agent, lo, hi)
#         self.fix_mobile_base = fix_mobile_base
#         # diag = np.ones(self.state_space_dim)
#         # diag[0:3] *= 10  # weight planar to be less important
#         # self.W = np.diag(diag)
#         mujoco.mj_forward(self.model, self.data)

#     def _set_qpos(self, state) -> None:  #: Optional[dict, np.ndarray] = None):
#         if isinstance(state, np.ndarray):
#             n = len(state)
#             if n == 3:
#                 state = {"base": state[0:3], "arm": None}
#             elif n == 5:
#                 state = {"base": None, "arm": state}
#             elif n == 8:
#                 state = {"base": state[0:3], "arm": state[3:]}
#             else:
#                 raise ValueError(f"Unknown state length {n}")

#         base_state = state["base"]
#         arm_state = state["arm"]

#         # i = 0
#         if base_state is not None:  # not self.fix_mobile_base:
#             # teleport - q planar is in the world frame
#             position = [base_state[0], base_state[1], 0.0]
#             rotate_rad = base_state[2] if len(base_state) > 2 else 0.0
#             # i += 3  # increment index

#             quaternion = R.from_euler("xyz", [0, 0, rotate_rad], degrees=False).as_quat(
#                 scalar_first=True
#             )
#             # quaternion = R.from_matrix(global_transform[:3, :3]).as_quat(scalar_first=True)

#             agent_jntadr = self.model.body(self.agent.root_id).jntadr[0]
#             agent_qposadr = self.model.jnt_qposadr[agent_jntadr]
#             self.data.qpos[agent_qposadr : agent_qposadr + 3] = position
#             self.data.qpos[agent_qposadr + 3 : agent_qposadr + 7] = quaternion

#         if arm_state is not None:
#             if isinstance(arm_state, np.ndarray) and len(arm_state) == 0:
#                 return
#             # rotate arm joints
#             state_dict = dict(
#                 lift=arm_state[0],
#                 arm_extend=arm_state[1] * 0.25,  # because arm_extend is composition of 4 joints
#                 wrist_yaw=arm_state[2],
#                 wrist_pitch=arm_state[3],
#                 wrist_roll=arm_state[4],
#             )

#             namespace = self.agent.actuator_to_joints.namespace
#             actuator_to_joints_map = self.agent.actuator_to_joints.d
#             for actuator, joints in actuator_to_joints_map.items():
#                 actuator_trn_type = self.model.actuator(namespace + actuator).trntype[0]
#                 joint_target = state_dict.get(actuator)
#                 if joint_target is None:
#                     continue
#                 for joint in joints:
#                     if actuator_trn_type == 0:
#                         qposadr = self.model.jnt(namespace + joint).qposadr[0]
#                         self.data.qpos[qposadr : qposadr + 1] = joint_target
#                     elif actuator_trn_type == 3:
#                         # TODO: get joints from tendon...
#                         joint_names = [
#                             "joint_arm_l0",
#                             "joint_arm_l1",
#                             "joint_arm_l2",
#                             "joint_arm_l3",
#                         ]
#                         assert joints == joint_names

#                         qposadr = self.model.jnt(namespace + joint).qposadr[0]
#                         self.data.qpos[qposadr : qposadr + 1] = joint_target
#                     else:
#                         raise ValueError(f"Unknown actuator turntype {actuator_trn_type}")

#     def _fk(self, qpos: np.ndarray) -> None:
#         super()._fk(qpos)
#         mujoco.mj_tendon(self.model, self.data)
#         mujoco.mj_sensorPos(self.model, self.data)

#     def ik(
#         self,
#         position: np.ndarray,
#         quaternion: np.ndarray,
#         q0: np.ndarray | None = None,
#         sample_q0: bool = False,
#         eps=0.04,
#         max_iter=1000,
#         damping=1e-12,
#         dt=1.0,
#     ):
#         assert q0 is not None or sample_q0, "Must provide an initial guess or allow sampling!"
#         if not self.fix_mobile_base:
#             # For IK, planar joints are in world frame
#             q0[0] = self.agent.position[0]
#             q0[1] = self.agent.position[1]
#             q0[2] = R.from_matrix(self.agent.rotation_matrix[:3, :3]).as_euler(
#                 "xyz", degrees=False
#             )[2]

#         if sample_q0:
#             get_logger().warning("Defaulting to uniform sampling for initial IK guess")

#             index_offset = 0
#             if not self.fix_mobile_base:
#                 index_offset = 3
#                 # TODO: sample base

#             # sample arm
#             q0[index_offset:] = (
#                 np.random.rand(self.state_space_dim - index_offset)
#                 * (self._state_space_high[index_offset:] - self._state_space_low[index_offset:])
#                 + self._state_space_low[index_offset:]
#             )
#         return super().ik(
#             position, quaternion, q0, sample_q0=False, eps=eps, max_iter=max_iter, damping=damping
#         )

#     def ik_pose(
#         self,
#         pose: np.ndarray,
#         q0: np.ndarray | None = None,
#         sample_q0: bool = False,
#         eps=1e-4,  # 0.04
#         max_iter=1000,
#         damping=1e-12,
#         dt=1.0,
#     ):
#         # return super().ik_pose(
#         #    pose, q0=q0, sample_q0=sample_q0, eps=eps, max_iter=max_iter, damping=damping, dt=dt
#         # )
#         target_pose = pose

#         if not self.fix_mobile_base:
#             # For IK, planar joints are in world frame
#             q0[0] = self.agent(self.model, self.data).position[0]
#             q0[1] = self.agent(self.model, self.data).position[1]
#             q0[2] = R.from_matrix(
#                 self.agent(self.model, self.data).rotation_matrix[:3, :3]
#             ).as_euler("xyz", degrees=False)[2]
#             q0[3:] = self.agent(self.model, self.data).joint_pos[3:]

#         if sample_q0:
#             get_logger().warning("Defaulting to uniform sampling for initial IK guess")

#             index_offset = 0
#             if not self.fix_mobile_base:
#                 index_offset = 3
#                 # TODO: sample base

#             # sample arm
#             q0[index_offset:] = (
#                 np.random.rand(self.state_space_dim - index_offset)
#                 * (self._state_space_high[index_offset:] - self._state_space_low[index_offset:])
#                 + self._state_space_low[index_offset:]
#             )

#         _err = np.inf
#         _i_err = 0
#         q = q0.copy()
#         for i in range(max_iter):
#             # compute forward kinematics
#             self._fk(q)
#             mujoco.mj_comPos(self.model, self.data)
#             mujoco.mj_tendon(self.model, self.data)
#             mujoco.mj_sensorPos(self.model, self.data)
#             self.agent(self.model, self.data)
#             ee_pose = relative_to_global_transform(
#                 self.agent.ee_pose_from_base, self.agent.base_pose
#             )

#             # compute error
#             err_trf = inverse_homogeneous_matrix(ee_pose) @ target_pose
#             twist_lin, twist_ang = transform_to_twist(err_trf)

#             err = np.concatenate([ee_pose[:3, :3] @ twist_lin, ee_pose[:3, :3] @ twist_ang])
#             if np.linalg.norm(err) < eps:
#                 succ = True
#                 break
#             elif i == max_iter - 1:
#                 succ = False
#                 break

#             # compute J
#             J = self.agent.ee_jacobian(self.model, self.data)

#             # Weighted damped least squares
#             W = self.W if self.W is not None else np.eye(J.shape[1])
#             H = J @ W @ J.T + damping * np.eye(J.shape[0])
#             q_dot = J.T @ np.linalg.solve(H, err)  # .ravel())

#             # integrate the joint velocities in-place
#             q += q_dot * dt

#             # joint limit
#             q = self._enforce_joint_limits(q)

#         if succ:
#             goal_joint_state = q  # self.agent.joint_pos.copy()
#             return goal_joint_state
#         else:
#             get_logger().debug(
#                 f"Failed to solve IK, completed {i} iterations with remaining error {np.linalg.norm(err)}, q={q}"
#             )
#             return None
