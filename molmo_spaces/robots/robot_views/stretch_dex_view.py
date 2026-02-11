from collections.abc import Mapping
from typing import Any, NoReturn

import cv2
import mujoco
import numpy as np
from matplotlib import pyplot as plt
from mujoco import MjData, MjModel, mjtObj
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.mj_extensions import MjModelBindings
from molmo_spaces.robots.robot_views.abstract import RobotView
from molmo_spaces.utils.linalg_utils import global_to_relative_transform
from molmo_spaces.utils.mj_model_and_data_utils import extract_mj_names


class NamespaceDictWrapper:
    def __init__(self, namespace: str, d: dict[str, Any]) -> None:
        assert isinstance(d, Mapping)
        self.namespace = namespace
        self.d = d

    def __getitem__(self, item: str):
        return self.d[self.namespace + item]

    def __setitem__(self, key: str, value: Any) -> None:
        self.d[self.namespace + key] = value

    def __contains__(self, item: str) -> bool:
        return self.namespace + item in self.d

    def __len__(self) -> int:
        return len(self.d)

    def keys(self):
        return [k[len(self.namespace) :] for k in self.d]


class StretchDexRobotView(RobotView):
    GRIPPER_LEFT_NAME = "rubber_tip_left"
    GRIPPER_RIGHT_NAME = "rubber_tip_right"

    # Default rotation matrix for the wrist when in top-down grasping pose
    """Default rotation matrix for the wrist when in top-down grasping pose.
    This 3x3 matrix represents the orientation of the end-effector for a Stretch3 with DexWrist in top-down pose.
    The wrist is aligned vertically above the target."""
    STRETCH_TOPDOWN_WRIST_ROTATION = np.array(
        [[1.0, 0.02266, 0.000], [0.02256, -0.9960, -0.0861], [0.0021403, 0.0860793, -1.0]]
    )

    """Default joint positions for top-down grasping pose.
    8-dimensional array corresponding to:
    [lift, arm_extend, wrist_yaw, wrist_pitch, wrist_roll, gripper, head_pan, head_tilt]
    The -1.57 (~90 degrees) pitch angle orients the gripper downward.
    The 0.1 lift position lifts the gripper up so the gripper does not touch the base."""
    TOPDOWN_JOINT_POS = np.array([0.1, 0.0, 0.0, -1.57, 0.0, 0.04, 0.0, 0.0])
    INIT_JOINT_POS = np.array([0.0, 0.0, -1.57, 0.1, 0.0, 0.0, -1.57, 0.0, 0.04, 0.0, 0.0])

    def init_joint_pos():
        return StretchDexRobotView.TOPDOWN_JOINT_POS.copy()

    def init_gripper_pos() -> NoReturn:
        raise NotImplementedError

    def __init__(self, model: MjModel, namespace: str = "robot_0/") -> None:
        super().__init__(model)
        self._state_space_dim = 8  # 3 base planar joint + 5 arm joints used for planner
        self.namespace = namespace

        base_id = model.body(self.namespace).id  # + "base_link").id
        self._root_id = model.body_rootid[base_id]

        self._get_gripper_geom_names()
        self._set_joint_names(model)
        self._set_arm_joint_limits(model)
        self._set_actuator_to_joints(model)
        self.reset()

    @classmethod
    def from_model(cls, model, namespace: str = "robot_0/"):
        return StretchDexRobotView(model, namespace)

    @property
    def name(self) -> str:
        return "stretch_dex"

    @property
    def camera_names(self):
        return [
            "d405_rgb",
            "d405_depth",
            "d435i_camera_rgb",
            "d435i_camera_depth",
            "nav_camera_rgb",
        ]

    @property
    def in_view_camera_name(self) -> str:
        return "nav_camera_rgb"

    @property
    def state_space_dim(self):
        return self._state_space_dim

    @state_space_dim.setter
    def state_space_dim(self, dim) -> None:
        self._state_space_dim = dim
        if len(self.state_space_low) == 8 and dim < 8:
            if dim == 5:  # manip only - remove first three joints
                self.state_space_low = self.state_space_low[3:]
                self.state_space_high = self.state_space_high[3:]
            elif dim == 3:  # base only XYTheta
                self.state_space_low = self.state_space_low[0:3]
                self.state_space_high = self.state_space_high[0:3]
            elif dim == 2:  # base only XY
                self.state_space_low = self.state_space_low[0:2]
                self.state_space_high = self.state_space_high[0:2]
            else:
                raise ValueError(f"Invalid state space dimension: {dim}")
        assert len(self.state_space_low) == len(self.state_space_high) == dim, (
            "State space bounds must match dimension"
        )

    @property
    def joint_limits(self):
        return self.state_space_low, self.state_space_high

    @property
    def actuator_to_joints(self):
        # .namespace: str
        # .d: dict
        return self._actuator_to_joints

    @property
    def actuator_ctrl_inputs(self):
        return self._actuator_ctrl_inputs

    @property
    def position(self):
        return self._position

    @property
    def quaternion(self):
        return self._quaternion

    @property
    def has_fallen(self):
        euler = R.from_quat(self.quaternion, scalar_first=True).as_euler("xyz", degrees=False)
        return np.abs(euler[0]) > np.pi / 2.0 or np.abs(euler[1]) > np.pi / 2.0

    @property
    def rotation_matrix(self):
        return self._rotation_matrix

    @property
    def base_pose(self):
        return self._rotation_matrix

    @property
    def linvel(self):
        return self._linvel

    @property
    def angvel(self):
        return self._angvel

    @property
    def head_camera_pose(self):
        return self._head_camera_pose

    @property
    def wrist_camera_pose(self):
        return self._wrist_camera_pose

    @property
    def joints(self):
        raise NotImplementedError
        return self._joints

    @property
    def finger_body_ids(self):
        # self.model.geom(self.left_gripper_geom_name).id
        # self.model.geom(self.right_gripper_geom_name).id
        body_left_id = self.model.geom(self.left_gripper_geom_name).bodyid.item()
        body_right_id = self.model.geom(self.right_gripper_geom_name).bodyid.item()
        finger_tip_id = [body_left_id, body_right_id]
        return finger_tip_id

    @property
    def root_id(self) -> int:
        return self._root_id

    @property
    def joint_pos_names_all(self):
        """in the order ctrl but index meant for joint_pos property"""
        return [
            "lift",
            "arm_extend",
            "wrist_yaw",
            "wrist_pitch",
            "wrist_roll",
            "grip",
            "head_pan",
            "head_tilt",
        ]

    @property
    def joint_pos_names(self):
        """in the order ctrl but index meant for joint_pos property"""
        return ["lift", "arm_extend", "wrist_yaw", "wrist_pitch", "wrist_roll"]

    @property
    def arm_joint_pos(self):
        return self.all_joint_pos

    @property
    def joint_pos(self):
        """for planner. corresponing to state dim"""
        planner_joint_pos = self.all_joint_pos[:-3]
        if self.state_space_dim == 8:
            # base joints in global frame
            base_joints = np.zeros(3)
            base_joints[:2] = self.position[:2]
            base_joints[2] = R.from_matrix(self.rotation_matrix[:3, :3]).as_euler(
                "xyz", degrees=False
            )[2]
            planner_joint_pos = np.insert(planner_joint_pos, 0, base_joints)
        elif self.state_space_dim == 5:
            planner_joint_pos = planner_joint_pos
        elif self.state_space_dim == 3:
            base_joints = np.zeros(3)
            base_joints[:2] = self.position[:2]
            base_joints[2] = R.from_matrix(self.rotation_matrix[:3, :3]).as_euler(
                "xyz", degrees=False
            )[2]
            planner_joint_pos = base_joints
        elif self.state_space_dim == 2:
            planner_joint_pos = self.position[:2]
        elif self.state_space_dim == 7:
            planner_joint_pos = np.insert(planner_joint_pos, 0, self.position[:2])
        else:
            raise ValueError(f"Invalid state space dimension: {self.state_space_dim}")
        return planner_joint_pos

    @property
    def gripper_vel(self) -> NoReturn:
        raise NotImplementedError

    @property
    def all_joint_pos(self):
        """all joints arm, gripper, head"""
        assert all(
            actuator_name in self.actuator_to_joints.d for actuator_name in self.joint_pos_names_all
        ), "Actuator names must be in actuator_to_joints"
        all_actuator_joints = self.get_actuator_joints(self.joint_pos_names_all)
        # maybe add cap to min max bounds?
        all_actuator_joints[self.joint_pos_names_all.index("arm_extend")] = max(
            all_actuator_joints[self.joint_pos_names_all.index("arm_extend")], 0.0
        )  # arm. sometimes gets -0.0000abc but bound is set to 0.
        return all_actuator_joints

    @property
    def grasp_center_from_base(self):
        return dict(pos=self._grasp_center_position, quat=self._grasp_center_quaternion)

    @property
    def ee_pose_from_base(self):
        trf = np.eye(4)
        trf[:3, :3] = R.from_quat(self._grasp_center_quaternion, scalar_first=True).as_matrix()
        trf[:3, 3] = self._grasp_center_position
        return trf

    def ee_jacobian(self, model: MjModel, data: MjData):
        self.update_ee_jacobian(model, data)
        return self._ee_jacobian

    def reset(self) -> None:
        # base
        self._position = None
        self._quaternion = None
        self._rotation_matrix = None
        self._linvel = None
        self._angvel = None
        # arm joints
        self._joints = None
        # gripper
        self._grasp_center_position = None
        self._grasp_center_quaternion = None
        # actuator
        self._actuator_ctrl_inputs = None
        # planner base joint [x,y,theta]
        self._start_base_pose = None

    def __call__(self, model: MjModel, data: MjData):
        assert self.model is model, "[StretchDexRobotView.__call__] Model mismatch"
        self.update(data)
        return self

    # ---- Set Functions ----
    def _set_joint_names(self, model: mujoco.MjModel) -> None:
        joint_names = []
        start_idx = len(self.namespace)
        for i, obj_type in enumerate(model.sensor_objtype):
            if obj_type == mjtObj.mjOBJ_JOINT or obj_type == mjtObj.mjOBJ_TENDON:
                joint_names.append(model.sensor(i).name[start_idx:])
        self.joint_names = joint_names

    def _set_arm_joint_limits(self, model: MjModel) -> None:
        state_space_low = np.array([-50.0] * self._state_space_dim)
        state_space_high = np.array([50.0] * self._state_space_dim)

        # only update arm joints
        i = 3
        for name in [
            "lift",
            "arm_extend",
            "wrist_yaw",
            "wrist_pitch",
            "wrist_roll",
        ]:
            state_space_low[i] = model.actuator(self.namespace + name).ctrlrange[0]
            state_space_high[i] = model.actuator(self.namespace + name).ctrlrange[1]
            i += 1
        self.state_space_low = state_space_low
        self.state_space_high = state_space_high

    def _set_actuator_to_joints(self, model: MjModel) -> None:
        _actuator_to_joints = {}
        start_idx = len(self.namespace)
        actuators_trn_ids = model.actuator_trnid
        actuators_trn_types = model.actuator_trntype
        for i, act_tr in enumerate(actuators_trn_types):
            actuator_name = model.actuator(i).name[start_idx:]
            _actuator_to_joints[actuator_name] = []
            trn_id = actuators_trn_ids[i][0]  # (id, -1)
            if act_tr == 0:  # joint
                trn_name = model.jnt(trn_id).name[start_idx:]
                _actuator_to_joints[actuator_name].append(trn_name)
                if trn_name == "joint_arm_l0":  # NOTE specific to stretch agent
                    _actuator_to_joints[actuator_name].append("joint_arm_l1")
                    _actuator_to_joints[actuator_name].append("joint_arm_l2")
                    _actuator_to_joints[actuator_name].append("joint_arm_l3")
            if act_tr == 3:  # tendon
                trn_name = model.tendon(trn_id).name[start_idx:]
                if trn_name == "extend":
                    # NOTE specific to stretch agent
                    _actuator_to_joints[actuator_name].append("joint_arm_l0")
                    _actuator_to_joints[actuator_name].append("joint_arm_l1")
                    _actuator_to_joints[actuator_name].append("joint_arm_l2")
                    _actuator_to_joints[actuator_name].append("joint_arm_l3")
                # _actuator_to_joints[actuator_name].append(trn_name)
        self._actuator_to_joints = NamespaceDictWrapper(self.namespace, _actuator_to_joints)

    def set_joint_qpos(self, model: MjModel, data: MjData, joint_pos: np.ndarray) -> None:
        assert self.model is model, "[StretchDexRobotView.set_joint_qpos] Model mismatch"
        for actuator, joints in self.actuator_to_joints.d.items():
            if actuator not in self.joint_pos_names_all:
                continue
            n = len(joints)
            for j in joints:
                bodyid = self.model.joint(self.namespace + j).bodyid.item()
                qposadr = self.model.joint(self.namespace + j).qposadr.item()
                dim = self.model.body(bodyid).dofnum.item()
                index = self.joint_pos_names_all.index(actuator)
                joint_target = joint_pos[index] / float(n)
                data.qpos[qposadr : qposadr + dim] = joint_target
            data.ctrl[2:] = joint_pos  # set actuator control inputs

    # ---- Update Functions ----
    def _read_from_sensor(self, sensor_name: str, data: MjData):
        s_adr = self.model.sensor(self.namespace + sensor_name).adr.item()
        s_dim = self.model.sensor(self.namespace + sensor_name).dim.item()
        return data.sensordata[s_adr : s_adr + s_dim].copy()

    def update(self, data: MjData) -> None:
        self._position = self._read_from_sensor("base_position", data)
        self._quaternion = self._read_from_sensor("base_quaternion", data)
        self._linvel = self._read_from_sensor("base_linvel", data)
        self._angvel = self._read_from_sensor("base_angvel", data)
        self._grasp_center_position = self._read_from_sensor("grasp_center_pos_from_base", data)
        self._grasp_center_quaternion = self._read_from_sensor("grasp_center_quat_from_base", data)
        self.update_actuator_ctrl_inputs(data)
        self.update_joints_pos(data)
        self.update_rotation_matrix()
        self.update_camera_poses(data)

    def update_camera_poses(self, data) -> None:
        def pose_from_data(data, body_id):
            pose = np.eye(4)
            pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
            pose[:3, 3] = data.xpos[body_id]
            return pose

        head_camera = self.model.body(self.namespace + "realsense").id
        self._head_camera_pose = pose_from_data(data, head_camera)
        wrist_camera = self.model.body(self.namespace + "d405_cam").id
        self._wrist_camera_pose = pose_from_data(data, wrist_camera)

    def update_rotation_matrix(self) -> None:
        #  update_base_transform(self, data: MjData):
        r = R.from_quat(self.quaternion, scalar_first=True).as_matrix()
        t = self.position
        transform = np.eye(4)
        transform[0:3, 0:3] = r
        transform[0:3, 3] = t
        self._rotation_matrix = transform

    def update_joints_pos(self, data: MjData) -> None:
        _joints = np.zeros(len(self.joint_names))
        for i, joint_name in enumerate(self.joint_names):
            _joints[i] = self._read_from_sensor(joint_name, data).item()
        self._joints = _joints

    def update_actuator_ctrl_inputs(self, data) -> None:
        n = self.model.nu
        ctrl_inputs = np.zeros(n)
        for i in range(n):
            if self.model.actuator(i).name.startswith(self.namespace):
                ctrl_inputs[i] = data.actuator(self.model.actuator(i).name).ctrl[0]
        self._actuator_ctrl_inputs = ctrl_inputs

    def update_ee_jacobian(self, model: MjModel, data: MjData) -> None:
        assert self.model is model, "[StretchDexRobotView.update_ee_jacobian] Model mismatch"
        # Jacobian for all sites
        site_names = [
            "base",
            "lift",
            "link_arm_l0",
            "link_arm_l1",
            "link_arm_l2",
            "link_arm_l3",
            "wrist_yaw",
            "wrist_pitch",
            "wrist_roll",
        ]  # , "link_grasp_center"]
        arm_extend_sites = ["link_arm_l0", "link_arm_l1", "link_arm_l2", "link_arm_l3"]

        # Compute Jacobian for each site
        n_sites = len(site_names)
        J_sites = np.empty((6 * n_sites, self.model.nv))
        for i, site_name in enumerate(site_names):
            site_id = self.model.site(self.namespace + site_name).id
            jacp = J_sites[6 * i : 6 * i + 3]
            jacr = J_sites[6 * i + 3 : 6 * i + 6]
            mujoco.mj_jacSite(self.model, data, jacp, jacr, site_id)
            # void mj_jacSite(const mjModel* m, const mjData* d, mjtNum* jacp, mjtNum* jacr, int site) {
            #    mj_jac(m, d, jacp, jacr, d->site_xpos + 3*site, m->site_bodyid[site]);
            #    }

        # Final jacobian
        J = np.zeros((6, self.state_space_dim))  # grasp center
        offset = 0
        if self.state_space_dim == 8:
            # Map base jacobian - world frame
            # 6 dof (xyz linear. xyz angular)
            base_dof_adr = self.model.body(self.root_id).dofadr[0]
            # but index 0,1,2 is in robot frame...
            J[:, 0] = J_sites[:6, base_dof_adr + 0]  # X Translation
            J[:, 1] = J_sites[:6, base_dof_adr + 1]  # Y Translation

            J[:, 2] = J_sites[:6, base_dof_adr + 5]  # Theta Rotation
            offset = 3

        # Map arm joints jacobian
        for _i, site_name in enumerate(site_names[1:]):
            if site_name in arm_extend_sites:
                if site_name != "link_arm_l0":
                    continue  # already
                for arm_site in arm_extend_sites:
                    # For primsatic arm joint, combine all four joint jacobians
                    index = site_names.index(arm_site)
                    site_body_id = self.model.site(self.namespace + site_name).bodyid[0]
                    joint_id = self.model.body(site_body_id).jntadr[0]
                    joint_dofadr = self.model.jnt(joint_id).dofadr[0]
                    J[:, offset] += J_sites[6 * (index) : 6 * (index + 1), joint_dofadr]
            else:
                index = site_names.index(site_name)
                site_body_id = self.model.site(self.namespace + site_name).bodyid[0]
                joint_id = self.model.body(site_body_id).jntadr[0]
                joint_dofadr = self.model.jnt(joint_id).dofadr[0]
                J[:, offset] = J_sites[6 * (index) : 6 * (index + 1), joint_dofadr]
            offset += 1
        # 6 rows for spatial velocity [dx,dy, dz, wx, wy, wz]
        # self.state_space_dim columns for either planar base [0:2] and/or just arm joints [3:7]
        self._ee_jacobian = J

    # ---- Get Functions ----
    def get_camera_pose_rel_base(self, camera_name: str, model: MjModel, data: MjData):
        assert self.model == model
        self.update(data)
        camera = self.model.camera(self.namespace + camera_name)
        camera_body_id = camera.bodyid[0]

        # Get world position and orientation of camera body from data
        pos = data.xpos[camera_body_id].copy()
        rot = data.xmat[camera_body_id].reshape(3, 3).copy()

        # Combine into transformation matrix
        T_world = np.eye(4)
        T_world[:3, :3] = rot
        T_world[:3, 3] = pos
        return global_to_relative_transform(T_world, self.base_pose)

    def get_actuator_joints(self, actuator_names):
        _actuator_joints = np.zeros(len(actuator_names))
        for i, act_name in enumerate(actuator_names):
            joint_names = self.actuator_to_joints.d[act_name]
            joint_index = [self.joint_names.index(j) for j in joint_names]
            _actuator_joints[i] = np.sum(self._joints[joint_index])
        return _actuator_joints

    def check_collision(self, model: MjModel, data: MjData, grasped_objs: set[int] = None) -> bool:
        assert self.model == model
        # TODO: ignore finger collision with grasped objects
        # check if the agent is in collision with any geoms in the scene
        agent_id = self.root_id
        contacts = data.contact
        for c in contacts:
            if c.exclude != 0:
                continue
            body1 = self.model.geom_bodyid[c.geom1]
            body2 = self.model.geom_bodyid[c.geom2]

            rootbody1 = self.model.body_rootid[body1]
            rootbody2 = self.model.body_rootid[body2]

            if rootbody1 == agent_id or rootbody2 == agent_id:
                # Ignore Collision with gripper tips and target object
                if (
                    self.model.body(body1).name == self.left_gripper_geom_name
                    or self.model.body(body1).name == self.right_gripper_geom_name
                ) or (
                    self.model.body(body2).name == self.left_gripper_geom_name
                    or self.model.body(body2).name == self.right_gripper_geom_name
                ):
                    if grasped_objs is not None and grasped_objs:
                        if (
                            self.model.body(body1).id in grasped_objs
                            or self.model.body(body2).id in grasped_objs
                        ):
                            continue
                    else:
                        continue

                # Ignore Collision with itself
                if rootbody1 == rootbody2 and rootbody1 == agent_id:
                    continue

                # Collision with the floor
                if (
                    self.model.body(rootbody2).name == "floor"
                    or self.model.body(rootbody1).name == "floor"
                    or self.model.body(rootbody2).name == "world"
                    or self.model.body(rootbody1).name == "world"
                ):
                    if (
                        self.model.body(rootbody2).name == self.namespace + "link_left_wheel"
                        or self.model.body(rootbody2).name == self.namespace + "link_right_wheel"
                        or self.model.body(rootbody1).name == self.namespace + "link_left_wheel"
                        or self.model.body(rootbody1).name == self.namespace + "link_right_wheel"
                        or self.model.body(rootbody1).name == self.namespace + "base_link"
                        or self.model.body(rootbody2).name == self.namespace + "base_link"
                        or self.model.body(rootbody1).name == self.namespace
                        or self.model.body(rootbody2).name == self.namespace
                    ):
                        # print("Collision with floor detected", model.body(body1).name, model.body(body2).name)
                        continue
                    # print(
                    #    "Collision with floor detected",
                    #    self.model.body(body1).name,
                    #    self.model.body(body2).name,
                    # )
                    # return False
                print(
                    "Collision detected",
                    self.model.body(body1).name,
                    self.model.body(body2).name,
                )
                return True
        return False

    def _get_gripper_geom_names(self) -> None:
        self.left_gripper_geom_name = self.namespace + self.GRIPPER_LEFT_NAME
        self.right_gripper_geom_name = self.namespace + self.GRIPPER_RIGHT_NAME

    def get_object_picked_up(self, model: MjModelBindings, data: MjData):
        assert self.model == model.model
        # check if the agent is touching any object
        # check if the object is picked up.
        # return that object id or name
        left_id = self.model.geom(self.left_gripper_geom_name).id
        right_id = self.model.geom(self.right_gripper_geom_name).id
        finger_tip_id = [left_id, right_id]

        gripper_object = []
        contacts = data.contact
        for c in contacts:
            if c.geom1 in finger_tip_id and c.geom2 in finger_tip_id:
                continue
            elif c.geom1 in finger_tip_id:
                body_id = self.model.geom_bodyid[c.geom2]
                body_name = self.model.body(body_id).name
                if body_name not in gripper_object:
                    gripper_object.append(body_name)
            elif c.geom2 in finger_tip_id:
                body_id = self.model.geom_bodyid[c.geom1]
                body_name = self.model.body(body_id).name
                if body_name not in gripper_object:
                    gripper_object.append(body_name)
        return gripper_object

    def calculate_head_angles_to_look_at_object(
        self,
        object_transform,
        frame="global",
        base_transform=None,
        camera_transform=None,
        model=None,
        data=None,
    ):
        """
        Calculate the head angles to look at an object

        Args:
            base_transform (_type_): _description_
            camera_transform (_type_): _description_
            object_transform (_type_): _description_
            frame (str, optional): _description_. Defaults to "global".
        """
        if base_transform is None:
            assert model is not None and data is not None
            base_transform = self.__call__(model, data).base_pose
        if camera_transform is None:
            assert model is not None and data is not None
            camera_transform = self.__call__(model, data).head_camera_pose

        camera_transform_from_base = global_to_relative_transform(camera_transform, base_transform)
        object_transform_from_base = global_to_relative_transform(object_transform, base_transform)

        look_at_vector = object_transform_from_base[:3, 3] - camera_transform_from_base[:3, 3]
        look_at_vector = look_at_vector / np.linalg.norm(look_at_vector)

        head_pan = np.arctan2(look_at_vector[1], look_at_vector[0])
        head_pan = np.mod(head_pan + np.pi, 2 * np.pi) - np.pi  # wrap to [-pi, pi]
        head_tilt = np.arctan2(look_at_vector[2], np.linalg.norm(look_at_vector[:2]))

        return head_pan, head_tilt

    def check_target_in_view(
        self,
        target_id: int,
        data: MjData,
        camera_name: str,
        visualize: bool = False,
        device_id: int = None,
    ):
        # NOTE: This  will help double check if the target is in view
        # target_id = self.model.body(target_id).rootid.item()

        # assert self.model.camera()
        if not hasattr(self, "_renderer") or self._renderer is None:
            from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer

            self._renderer = MjOpenGLRenderer(model=self.model, device_id=device_id)

        self._renderer.enable_segmentation_rendering()
        self._renderer.update(data, camera=camera_name)
        seg = self._renderer.render()[..., 2]  # (object ID, object type, body ID)

        # get all children bodyids of the target
        # Check if target is in view
        target_seg = np.isin(
            seg,
            [i for i in range(self.model.nbody) if self.model.body(i).rootid.item() == target_id],
        )

        if visualize:
            self._renderer.disable_segmentation_rendering()
            self._renderer.update(data, camera=camera_name)
            img = self._renderer.render()

            # blend original image with target segment
            overlay = img.copy()
            overlay[target_seg] = [0, 255, 0]  # green
            alpha = 0.5
            highlighted_img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
            highlighted_img = cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB)

            # self.model.body(target_id).name
            plt.imshow(highlighted_img)
            plt.show()
            # cv2.imshow(f"{target_name} in view", highlighted_img)
            # cv2.waitKey(1)
            # cv2.destroyAllWindows()

        return target_seg.any()


### ---------------------  DEPRECATE ---------------------
class _StretchDexRobotView(RobotView):
    # TODO: refactor Agent class

    def __init__(
        self,
        sensor_name2id: dict[str, int],
        sensor_dim_ptr,
        sensor_adr_ptr,
        sensor_objid_ptr,
        sensor_objtype_ptr,
        # actuator_name2id: dict,
        # actuator_trnid_ptr,# transmission id: joint, tendon, site
        # actuator_trntype_ptr, # transmission type: joint, tendon, site
        # actuator_ctrlrange_ptr, # control range for each actuator
        sensordata_ptr=None,
        namespace: str = "",
    ) -> None:  # , model_bindings: MjModel, data: mujoco.MjData):
        """
        sensor_name2id (n): dictionary of sensor name to sensor id (index in sensor_adr_ptr) (fixed)
        sensor_dim_ptr (n): dim for each sensor (fixed)
        sensor_adr_ptr (n): a starting index in sensordata_ptr for each sensor (fixed)
        sensordata_ptr (n * dim per each sensor): pointer to the flattened array of all sensors in the model_bindings (gets updated after each mujoco step)
        sensor_objid_ptr (n): id of sensorized object (id of body, joint, tenodon, etc)
        sensor_objtype_ptr (n): type of sensorized object (xbody, joint, tendon, etc)

        NOTE: I have separate update functions in case env resets during the trajectory. I can update the object state to previous state.
        """
        self.namespace = namespace
        self.sensor_name2id = NamespaceDictWrapper(namespace, sensor_name2id)
        self.sensor_dim_ptr = sensor_dim_ptr
        self.sensor_adr_ptr = sensor_adr_ptr
        self.sensor_objid_ptr = sensor_objid_ptr
        self.sensor_objtype_ptr = sensor_objtype_ptr
        self.sensordata_ptr = sensordata_ptr

        # joints are either 3(joint) or 18(tendon: linear collection of joints)
        indices = np.where((self.sensor_objtype_ptr == 18) | (self.sensor_objtype_ptr == 3))[0]
        self.joint_names = [list(self.sensor_name2id.keys())[i] for i in indices]
        self.n_joints = int(len(self.joint_names))

        self.joint_names_ros = [
            "joint_right_wheel",
            "joint_left_wheel",
            "joint_lift",
            "joint_arm_l3",
            "joint_arm_l2",
            "joint_arm_l1",
            "joint_arm_l0",
            "joint_wrist_yaw",
            "joint_head_pan",
            "joint_head_tilt",
            "joint_wrist_pitch",
            "joint_wrist_roll",
            "joint_gripper_finger_right",
            "joint_gripper_finger_left",
        ]
        self.n_joints_ros = len(self.joint_names_ros)

        self.left_gripper_geom_name = namespace + "rubber_tip_left"
        self.right_gripper_geom_name = namespace + "rubber_tip_right"
        self._grasp_center_site_id = None

        # agent state
        self.reset()
        self.namespace = namespace
        self.state_space_dim = 8  # 3 base planar joint + 5 arm joints used for planner
        self.state_space_low = None
        self.state_space_high = None

    @property
    def name(self) -> str:
        return "stretch"

    @classmethod
    def from_model(
        cls,
        model: MjModel,
        sensor_name2id: dict[str, int],
        sensordata_ptr: str | None = None,
        namespace: str = "robot_0/",
    ):
        stretch = cls(
            sensor_name2id=sensor_name2id,
            sensor_dim_ptr=model.sensor_dim,
            sensor_adr_ptr=model.sensor_adr,
            sensor_objid_ptr=model.sensor_objid,
            sensor_objtype_ptr=model.sensor_objtype,
            sensordata_ptr=sensordata_ptr,
            namespace=namespace,
        )
        stretch._update_arm_joint_limits(model)
        return stretch

    @property
    def joint_limits(self):
        return self.state_space_low, self.state_space_high

    def _update_arm_joint_limits(self, model: MjModel) -> None:
        state_space_low = np.array(
            [-1000.0] * self.state_space_dim
        )  # really large number to avoid clipping for base
        state_space_high = np.array([1000.0] * self.state_space_dim)

        # only update arm joints
        i = 3
        for name in [
            "lift",
            "arm_extend",
            "wrist_yaw",
            "wrist_pitch",
            "wrist_roll",
        ]:
            state_space_low[i] = model.actuator(self.namespace + name).ctrlrange[0]
            state_space_high[i] = model.actuator(self.namespace + name).ctrlrange[1]
            i += 1
        self.state_space_low = state_space_low
        self.state_space_high = state_space_high

    def get_actuator_joints(self, actuator_names):
        _actuator_joints = np.zeros(len(actuator_names))
        for i, act_name in enumerate(actuator_names):
            joint_names = self.actuator_to_joints[act_name]
            joint_index = [self.joint_names.index(j) for j in joint_names]
            _actuator_joints[i] = np.sum(self.joints[joint_index])
        return _actuator_joints

    @property
    def actuator_ctrl_inputs(self):
        return self._actuator_ctrl_inputs

    def update_actuator_ctrl_inputs(self, model, data) -> None:
        (
            _,
            actuator_name2id,
            actuator_id2name,
        ) = extract_mj_names(
            model=model, name_adr=None, num_obj=model.nu, obj_type=mjtObj.mjOBJ_ACTUATOR
        )
        actuator_names = [name for name, id in actuator_name2id.items()]
        self._actuator_ctrl_inputs = np.array(
            [
                data.actuator(name).ctrl[0]
                for name in actuator_names
                if name.startswith(self.namespace)
            ]
        )

    @property
    def actuator_to_joints(self):
        # .namespace: str
        # .d: dict
        return self._a2j

    def _actuator_to_joints(self, model, data) -> None:
        # TODO: update this function with merged model
        (
            _,
            actuator_name2id,
            actuator_id2name,
        ) = extract_mj_names(
            model=model, name_adr=None, num_obj=model.nu, obj_type=mjtObj.mjOBJ_ACTUATOR
        )
        (
            _,
            joint_name2id,
            joint_id2name,
        ) = extract_mj_names(
            model=model, name_adr=None, num_obj=model.njnt, obj_type=mjtObj.mjOBJ_JOINT
        )
        (
            _,
            tendon_name2id,
            tendon_id2name,
        ) = extract_mj_names(
            model=model,
            name_adr=None,
            num_obj=model.ntendon,
            obj_type=mjtObj.mjOBJ_TENDON,
        )

        a2j = {}

        actuators_trn_ids = model.actuator_trnid
        actuators_trn_types = model.actuator_trntype
        for i, act_tr in enumerate(actuators_trn_types):
            actuator_name = actuator_id2name[i]
            a2j[actuator_name] = []
            trn_id = actuators_trn_ids[i][0]  # (id, -1)

            if act_tr == 0:  # joint
                trn_name = joint_id2name[trn_id]
                if trn_name.find(self.namespace) != -1:
                    trn_name = trn_name[len(self.namespace) :]
                a2j[actuator_name].append(trn_name)

                if trn_name == "joint_arm_l0":  # specific to stretch agent
                    a2j[actuator_name].append("joint_arm_l1")
                    a2j[actuator_name].append("joint_arm_l2")
                    a2j[actuator_name].append("joint_arm_l3")

            if act_tr == 3:  # tendon
                trn_name = tendon_id2name[trn_id]
                if trn_name.find(self.namespace) != -1:
                    trn_name = trn_name[len(self.namespace) :]
                a2j[actuator_name].append(trn_name)
                # NOTE: Should I add joints of the object in the tendon group?
                # Might not be necessary
                # tendon_adr = model.tendon_adr[trn_id] # adr of first object
                # tendon_num = model.tendon_num[trn_id] # num of objects
        self._a2j = NamespaceDictWrapper(self.namespace, a2j)

    def reset(self) -> None:
        self._position = None
        self._quaternion = None
        self._rotation_matrix = None
        self._linvel = None
        self._angvel = None
        self._joints = None
        self._joint_state = None
        self._joints_cartesian_from_base = None
        self._gripper_tip_center = None
        self._gripper_tip_center_from_base = None
        self.all_bvhadr = None
        self._a2j = None
        self._actuator_ctrl_inputs = None

    def set_joint_qvel(self, model: MjModelBindings, data: mujoco.MjData) -> None:
        joint_names = [
            "joint_arm_l0",
            "joint_arm_l1",
            "joint_arm_l2",
            "joint_arm_l3",
            "joint_lift",
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll",
            "joint_gripper_slide",
            "joint_gripper_finger_right",
            "rubber_right_y",
            "joint_gripper_finger_left",
            "rubber_left_y",
            "joint_head_pan",
            "joint_head_tilt",
            "joint_left_wheel",
            "joint_right_wheel",
        ]
        joint_names = [self.namespace + joint_name for joint_name in joint_names]
        self.joint_ids = [model.joint_name2id[joint_name] for joint_name in joint_names]
        self.qvel = data.qvel

    def set_sensordata_ptr(self, sensordata_ptr: dict[str, Any]) -> None:
        self.sensordata_ptr = sensordata_ptr

    @property
    def position(self):
        return self._position

    def update_position(self) -> None:
        # base link position in world coordinate
        start_ind = self.sensor_adr_ptr[self.sensor_name2id["base_position"]]
        end_ind = start_ind + self.sensor_dim_ptr[self.sensor_name2id["base_position"]]
        self._position = self.sensordata_ptr[start_ind:end_ind].copy()

    @property
    def quaternion(self):
        return self._quaternion

    def update_quaternion(self) -> None:
        # base link quaternion in world coordinate NOTE: [w, x, y, z]
        start_ind = self.sensor_adr_ptr[self.sensor_name2id["base_quaternion"]]
        end_ind = start_ind + self.sensor_dim_ptr[self.sensor_name2id["base_quaternion"]]
        self._quaternion = self.sensordata_ptr[start_ind:end_ind].copy()

    @property
    def rotation_matrix(self):
        return self._rotation_matrix

    def update_rotation_matrix(self) -> None:
        # base link rotation matrix in world coordinate
        # or use xmat from mjData (9)
        quaternion_xyzw = self.quaternion[[1, 2, 3, 0]]
        r = R.from_quat(quaternion_xyzw).as_matrix()
        t = self.position

        transform = np.eye(4)
        transform[0:3, 0:3] = r
        transform[0:3, 3] = t
        self._rotation_matrix = transform

    @property
    def linvel(self):
        return self._linvel

    def update_linvel(self) -> None:
        start_ind = self.sensor_adr_ptr[self.sensor_name2id["base_linvel"]]
        end_ind = start_ind + self.sensor_dim_ptr[self.sensor_name2id["base_linvel"]]
        self._linvel = self.sensordata_ptr[start_ind:end_ind].copy()

    @property
    def angvel(self):
        return self._angvel

    def update_angvel(self) -> None:
        start_ind = self.sensor_adr_ptr[self.sensor_name2id["base_angvel"]]
        end_ind = start_ind + self.sensor_dim_ptr[self.sensor_name2id["base_angvel"]]
        self._angvel = self.sensordata_ptr[start_ind:end_ind].copy()

    @property
    def joints(self):
        return self._joints

    def update_joints(self) -> None:
        # TODO: arm_extend tracks "joint_arm_l0" which is 1/4 of the entire length.
        # So, we need to multiply by 4 to get the actual length
        _joints = np.zeros(self.n_joints)
        for joint_name in self.joint_names:
            start_ind = self.sensor_adr_ptr[self.sensor_name2id[joint_name]]
            end_ind = start_ind + self.sensor_dim_ptr[self.sensor_name2id[joint_name]]
            _joints[self.joint_names.index(joint_name)] = self.sensordata_ptr[
                start_ind:end_ind
            ].copy()
        self._joints = _joints

    @property
    def joint_state(self):
        return self._joint_state

    def update_joint_state(self) -> None:
        """
        Mujoco joint 0 -> URDF
        lift 0 == 0.5
        arm 0 == 0
        """

        _joint_state = np.zeros(self.n_joints_ros)
        for joint_name in self.joint_names_ros:
            start_ind = self.sensor_adr_ptr[self.sensor_name2id[joint_name]]
            end_ind = start_ind + self.sensor_dim_ptr[self.sensor_name2id[joint_name]]
            _joint_state[self.joint_names_ros.index(joint_name)] = self.sensordata_ptr[
                start_ind:end_ind
            ].copy()

            if joint_name == "joint_lift":
                _joint_state[self.joint_names_ros.index(joint_name)] += 0.5

        self._joint_state = _joint_state

    @property
    def joints_cartesian_from_base(self):
        return self._joints_cartesian_from_base

    def update_joints_cartesian_from_base(self) -> None:
        _joints_cartesian_from_base = np.zeros((self.n_joints, 3))
        for joint_name in self.joint_names:
            joint_name_from_base = joint_name + "_from_base"
            if joint_name_from_base not in self.sensor_name2id:
                continue
            start_ind = self.sensor_adr_ptr[self.sensor_name2id[joint_name_from_base]]
            end_ind = start_ind + self.sensor_dim_ptr[self.sensor_name2id[joint_name_from_base]]
            joint_position = self.sensordata_ptr[start_ind:end_ind].copy()
            _joints_cartesian_from_base[self.joint_names.index(joint_name)] = joint_position
        self._joints_cartesian_from_base = _joints_cartesian_from_base

    @property
    def grasp_center_from_base(self):
        return dict(pos=self._grasp_center_position, quat=self._grasp_center_quat)

    def ee_jacobian(self):
        J = np.zeros((6, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J[:3], J[-3:], self._grasp_center_site_id)
        return J  # J[:, self.qvel_adrs]
        # raise NotImplementedError  # TODO: Implement when switching to mujoco IK

    @property
    def joint_pos(self) -> None:
        return None  # TODO: Implement when switching to mujoco IK

    @property
    def base_pose(self):
        return self._rotation_matrix

    @property
    def ee_pose_from_base(self):
        trf = np.eye(4)
        trf[:3, :3] = R.from_quat(self._grasp_center_quat, scalar_first=True).as_matrix()
        trf[:3, 3] = self._grasp_center_position
        return trf

    def update_grasp_center_from_base(self) -> None:
        start_ind = self.sensor_adr_ptr[self.sensor_name2id["grasp_center_pos_from_base"]]
        end_ind = start_ind + self.sensor_dim_ptr[self.sensor_name2id["grasp_center_pos_from_base"]]
        self._grasp_center_position = self.sensordata_ptr[start_ind:end_ind].copy()

        start_ind = self.sensor_adr_ptr[self.sensor_name2id["grasp_center_quat_from_base"]]
        end_ind = (
            start_ind + self.sensor_dim_ptr[self.sensor_name2id["grasp_center_quat_from_base"]]
        )
        self._grasp_center_quat = self.sensordata_ptr[start_ind:end_ind].copy()

    @property
    def gripper_tip_center(self):
        return self._gripper_tip_center

    def update_gripper_tip_center(self) -> None:
        # gripper tip center position in world coordinate
        left_start_ind = self.sensor_adr_ptr[self.sensor_name2id["rubber_tip_left"]]
        left_end_ind = left_start_ind + self.sensor_dim_ptr[self.sensor_name2id["rubber_tip_left"]]
        left_position = self.sensordata_ptr[left_start_ind:left_end_ind].copy()

        right_start_ind = self.sensor_adr_ptr[self.sensor_name2id["rubber_tip_right"]]
        right_end_ind = (
            right_start_ind + self.sensor_dim_ptr[self.sensor_name2id["rubber_tip_right"]]
        )
        right_position = self.sensordata_ptr[right_start_ind:right_end_ind].copy()

        self._gripper_tip_center = (left_position + right_position) / 2

    @property
    def gripper_tip_center_from_base(self):
        return self._gripper_tip_center_from_base

    def update_gripper_tip_center_from_base(self) -> None:
        # gripper tip center position in world coordinate
        left_start_ind = self.sensor_adr_ptr[self.sensor_name2id["rubber_tip_left_from_base"]]
        left_end_ind = (
            left_start_ind + self.sensor_dim_ptr[self.sensor_name2id["rubber_tip_left_from_base"]]
        )
        left_position = self.sensordata_ptr[left_start_ind:left_end_ind].copy()

        right_start_ind = self.sensor_adr_ptr[self.sensor_name2id["rubber_tip_right_from_base"]]
        right_end_ind = (
            right_start_ind + self.sensor_dim_ptr[self.sensor_name2id["rubber_tip_right_from_base"]]
        )
        right_position = self.sensordata_ptr[right_start_ind:right_end_ind].copy()
        self._gripper_tip_center_from_base = (left_position + right_position) / 2

    #    def __call__(self, model, data):
    #        self.model = model
    #        [self.update]()

    def update(self) -> None:
        # seperate thread
        self.update_position()
        self.update_quaternion()
        self.update_rotation_matrix()
        self.update_linvel()
        self.update_angvel()
        self.update_joints()
        self.update_joints_cartesian_from_base()
        self.update_gripper_tip_center()
        self.update_gripper_tip_center_from_base()

    def __call__(self, model: MjModel, data: mujoco.MjData):
        if self.sensordata_ptr is None:
            self.set_sensordata_ptr(data.sensordata)

        if self._a2j is None:
            self._actuator_to_joints(model, data)

        self.update()
        self.update_actuator_ctrl_inputs(model, data)
        try:
            model.body(self.namespace + "link_grasp_center")
            self.update_grasp_center_from_base()
        except KeyError:
            pass
        return self

    def stopped_moving(self, threshold=1e-6):
        # check if the agent is stopped moving
        self.joint_qvels = self.qvel[
            self.joint_ids
        ]  # np.ndarray... np.ndarray(joint_qvels) but outside of function, will create a instance (self) copy # np.array copies the data to a new memory
        # print("Max joint qvel: ", np.max(self.joint_qvels)) ## Visually 0.001 might be okay
        return bool(np.all(np.abs(self.joint_qvels) < threshold))

        if self.joints is None:
            self.update()
        target_joints = self.joints.copy()
        self.update()
        curr_joints = self.joints

        # check if the agent has stopped moving
        return np.all(np.abs(curr_joints - target_joints) <= 1e-5)

    def check_collision(
        self, model: mujoco.MjModel, data: mujoco.MjData, grasped_objs: set[int]
    ) -> bool:
        # check if the agent is in collision with any geoms in the scene
        agent_id = self.root_id
        contacts = data.contact
        for c in contacts:
            if c.exclude != 0:
                continue
            body1 = model.geom_bodyid[c.geom1]
            body2 = model.geom_bodyid[c.geom2]

            rootbody1 = model.body_rootid[body1]
            rootbody2 = model.body_rootid[body2]

            if rootbody1 == agent_id or rootbody2 == agent_id:
                # Ignore Collision with gripper tips and target object
                if (
                    model.body(body1).name == self.namespace + "rubber_tip_left"
                    or model.body(body1).name == self.namespace + "rubber_tip_right"
                ) or (
                    model.body(body2).name == self.namespace + "rubber_tip_left"
                    or model.body(body2).name == self.namespace + "rubber_tip_right"
                ):
                    continue

                # Ignore Collision with itself
                if False:
                    if model.body(rootbody1).name == model.body(rootbody2).name:
                        continue

                # Collision with the floor
                if model.body(rootbody2).name == "floor" or model.body(rootbody1).name == "floor":
                    if (
                        model.body(rootbody2).name == self.namespace + "link_left_wheel"
                        or model.body(rootbody2).name == self.namespace + "link_right_wheel"
                        or model.body(rootbody1).name == self.namespace + "link_left_wheel"
                        or model.body(rootbody1).name == self.namespace + "link_right_wheel"
                        or model.body(rootbody1).name == self.namespace + "base_link"
                        or model.body(rootbody2).name == self.namespace + "base_link"
                        or model.body(rootbody1).name == self.namespace
                        or model.body(rootbody2).name == self.namespace
                    ):
                        # print("Collision with floor detected", model_bindings.body_id2name[body1], model_bindings.body_id2name[body2])
                        continue
                    # print(
                    #    "Collision with floor detected",
                    #    model_bindings.body_id2name[body1],
                    #    model_bindings.body_id2name[body2],
                    # )
                    # return False
                print(
                    "Collision detected",
                    model.body(body1).name,
                    model.body(body2).name,
                )
                return True
            elif len(grasped_objs & {body1, body2}) == 1:
                # Check for collisions between grasped object and environment
                return True
        return False

    def get_object_picked_up(self, model: MjModelBindings, data: mujoco.MjData):
        # check if the agent is touching any object
        # check if the object is picked up.
        # return that object id or name
        if self.left_gripper_geom_name in model.geom_name2id:
            left_id = model.geom_name2id[self.left_gripper_geom_name]
            right_id = model.geom_name2id[self.right_gripper_geom_name]
            finger_tip_id = [left_id, right_id]
        else:
            # Handles the Stretch 3 xml where the names of things have changed.
            xml = model.xml

            # Find body with name self.namespace + "rubber_tip_left" and find the non-visual geom
            # under this body
            left_tip_body_list = [
                body
                for body in xml.findall(".//body")
                if body.get("name") == self.namespace + "rubber_tip_left"
            ]
            assert len(left_tip_body_list) == 1
            left_tip_body = left_tip_body_list[0]
            left_tip_geom_list = [
                geom
                for geom in left_tip_body.findall(".//geom")
                if "visual" not in geom.get("class")
            ]
            assert len(left_tip_geom_list) == 1

            # Do the same thing as above but with the self.namespace + "rubber_tip_right" body.
            right_tip_body_list = [
                body
                for body in xml.findall(".//body")
                if body.get("name") == self.namespace + "rubber_tip_right"
            ]
            assert len(right_tip_body_list) == 1
            right_tip_body = right_tip_body_list[0]
            right_tip_geom_list = [
                geom
                for geom in right_tip_body.findall(".//geom")
                if "visual" not in geom.get("class")
            ]
            assert len(right_tip_geom_list) == 1

            self.left_gripper_geom_name = left_tip_geom_list[0].get("name")
            self.right_gripper_geom_name = right_tip_geom_list[0].get("name")

            return self.get_object_picked_up(model, data)

        gripper_object = []
        contacts = data.contact
        for c in contacts:
            if c.geom1 in finger_tip_id and c.geom2 in finger_tip_id:
                continue
            elif c.geom1 in finger_tip_id:
                body_id = model.model.geom_bodyid[c.geom2]
                body_name = model.body_id2name[body_id]
                if body_name not in gripper_object:
                    gripper_object.append(body_name)
            elif c.geom2 in finger_tip_id:
                body_id = model.model.geom_bodyid[c.geom1]
                body_name = model.body_id2name[body_id]
                if body_name not in gripper_object:
                    gripper_object.append(body_name)
        return gripper_object

    def has_robot_fallen(self, tilt_threshold=45) -> bool:
        rotation = R.from_quat(self.quaternion[[1, 2, 3, 0]])
        euler_angles = rotation.as_euler("xyz", degrees=True)
        roll, pitch, yaw = euler_angles
        return bool(abs(roll) > tilt_threshold or abs(pitch) > tilt_threshold)
