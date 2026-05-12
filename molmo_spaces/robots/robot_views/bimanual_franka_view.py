"""RobotView for the bimanual Franka (two FR3 arms with Robotiq 2F-85 grippers).

Move groups:
  - base: Mocap body controlling both arms
  - left_arm: Left 7-DOF FR3 arm
  - right_arm: Right 7-DOF FR3 arm
  - left_gripper: Left Robotiq 2F-85 gripper
  - right_gripper: Right Robotiq 2F-85 gripper
"""

from typing import Literal

import mujoco
import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    GripperGroup,
    MocapRobotBaseGroup,
    MoveGroup,
    RobotView,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose, site_pose


class BimanualFrankaBaseGroup(MocapRobotBaseGroup):
    """Mocap base group for the bimanual Franka."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        # Try mocap_base first (when added to scene via add_robot_to_scene),
        # fall back to base (when loaded standalone)
        try:
            body_id: int = mj_data.model.body(f"{namespace}mocap_base").id
        except KeyError:
            body_id = mj_data.model.body(f"{namespace}base").id
        super().__init__(mj_data, body_id)


class BimanualFrankaArmGroup(MoveGroup):
    """7-DOF FR3 arm group for one side of the bimanual Franka."""

    def __init__(
        self,
        mj_data: MjData,
        side: Literal["left", "right"],
        base_group: BimanualFrankaBaseGroup,
        namespace: str = "",
    ) -> None:
        model = mj_data.model
        self._namespace = namespace
        self._side = side
        self._arm_prefix = f"{namespace}{side}_"

        # 7 arm joints
        joint_ids = [
            model.joint(f"{self._arm_prefix}fr3v2_joint{i + 1}").id for i in range(7)
        ]
        # 7 arm actuators (same names as joints)
        act_ids = [
            model.actuator(f"{self._arm_prefix}fr3v2_joint{i + 1}").id for i in range(7)
        ]
        self._arm_root_id = model.body(f"{self._arm_prefix}fr3v2_link0").id
        self._ee_site_id = model.site(f"{namespace}{side}_grasp_site").id
        super().__init__(mj_data, joint_ids, act_ids, self._arm_root_id, base_group)

    @property
    def side(self) -> str:
        return self._side

    @property
    def noop_ctrl(self) -> np.ndarray:
        return self.joint_pos.copy()

    @property
    def leaf_frame_to_world(self) -> np.ndarray:
        return site_pose(self.mj_data, self._ee_site_id)

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._arm_root_id)

    def get_jacobian(self) -> np.ndarray:
        J = np.zeros((6, self.mj_model.nv))
        mujoco.mj_jacSite(self.mj_model, self.mj_data, J[:3], J[3:], self._ee_site_id)
        return J


class BimanualFrankaGripperGroup(GripperGroup):
    """Robotiq 2F-85 gripper group for one side of the bimanual Franka.

    Uses a single actuated knuckle joint with mimic equality constraints
    for the other finger joints.
    """

    def __init__(
        self,
        mj_data: MjData,
        side: Literal["left", "right"],
        base_group: BimanualFrankaBaseGroup,
        namespace: str = "",
    ) -> None:
        model = mj_data.model
        self._namespace = namespace
        self._side = side
        self._gripper_prefix = f"{namespace}{side}_robotiq_85_"

        # The driver joint (left_knuckle) and its mimic (right_knuckle)
        joint_ids = [
            model.joint(f"{self._gripper_prefix}left_knuckle_joint").id,
            model.joint(f"{self._gripper_prefix}right_knuckle_joint").id,
        ]
        # Single actuator controls left_knuckle (right follows via equality constraint)
        act_ids = [
            model.actuator(f"{self._gripper_prefix}left_knuckle_joint").id,
        ]
        root_body_id = model.body(f"{namespace}{side}_robotiq_85_base_link").id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base_group)
        self._ee_site_id = model.site(f"{namespace}{side}_grasp_site").id

        # Finger pad geom IDs for distance computation
        self._finger_1_geom_id = model.geom(f"{namespace}{side}_left_pad").id
        self._finger_2_geom_id = model.geom(f"{namespace}{side}_right_pad").id

    @property
    def side(self) -> str:
        return self._side

    def set_gripper_ctrl_open(self, open: bool) -> None:
        """Set gripper to fully open or closed.

        Robotiq 85 knuckle joint range: 0 (open) to 0.8 (closed).
        """
        self.ctrl = [0.0 if open else 0.8]

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        """(min, max) distance between finger tips."""
        return 0.0, 0.085

    @property
    def inter_finger_dist(self) -> float:
        """Current distance between finger pads."""
        dist = mujoco.mj_geomDistance(
            self.mj_model,
            self.mj_data,
            self._finger_1_geom_id,
            self._finger_2_geom_id,
            0.1,
            None,
        )
        return max(0.0, dist)

    @property
    def leaf_frame_to_world(self) -> np.ndarray:
        return site_pose(self.mj_data, self._ee_site_id)

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world

    def get_jacobian(self) -> np.ndarray:
        J = np.zeros((6, self.mj_model.nv))
        mujoco.mj_jacSite(self.mj_model, self.mj_data, J[:3], J[3:], self._ee_site_id)
        return J


class BimanualFrankaRobotView(RobotView):
    """Robot view for the bimanual Franka (two 7-DOF FR3 arms with Robotiq grippers).

    Move groups:
      - base: Mocap body controlling both arms
      - left_arm: Left 7-DOF FR3 arm
      - right_arm: Right 7-DOF FR3 arm
      - left_gripper: Left Robotiq 2F-85
      - right_gripper: Right Robotiq 2F-85
    """

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = BimanualFrankaBaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "left_arm": BimanualFrankaArmGroup(mj_data, "left", base, namespace=namespace),
            "right_arm": BimanualFrankaArmGroup(mj_data, "right", base, namespace=namespace),
            "left_gripper": BimanualFrankaGripperGroup(mj_data, "left", base, namespace=namespace),
            "right_gripper": BimanualFrankaGripperGroup(
                mj_data, "right", base, namespace=namespace
            ),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}bimanual_franka"

    @property
    def base(self) -> BimanualFrankaBaseGroup:
        return self._move_groups["base"]
