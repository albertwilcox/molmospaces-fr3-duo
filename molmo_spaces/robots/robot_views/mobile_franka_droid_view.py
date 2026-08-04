from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    HoloJointsRobotBaseGroup,
    RobotView,
)
from molmo_spaces.robots.robot_views.franka_droid_view import RobotIQGripperGroup
from molmo_spaces.robots.robot_views.franka_fr3_view import FrankaFR3ArmGroup


class MobileFrankaDroidBaseGroup(HoloJointsRobotBaseGroup):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        world_site_id = model.site(f"{namespace}world").id
        holo_base_site_id = model.site(f"{namespace}base_site").id
        joints = [model.joint(f"{namespace}base_{axis}").id for axis in ["x", "y", "theta"]]
        act = [model.actuator(f"{namespace}base_{axis}_act").id for axis in ["x", "y", "theta"]]
        root_body_id = model.body(f"{namespace}base").id
        super().__init__(mj_data, world_site_id, holo_base_site_id, joints, act, root_body_id)
        # The base yaw is a free (unlimited) hinge, but its position actuator ships
        # with ctrlrange=[-pi, pi]. The holonomic ``ctrl`` setter commands the yaw
        # in a *continuous* frame (curr_yaw + normalize(target - curr_yaw)), which
        # legitimately exceeds +-pi when the base rotates across the +-180 deg
        # boundary. Clipping that setpoint at +-pi pins the base at the boundary:
        # in-place alignment turns whose shortest path crosses +-pi stall there,
        # exhaust the nav align budget, and the residual heading error is then
        # applied as an instantaneous base "teleport" by the manip standoff snap.
        # Widen the actuator range so the base can slew continuously across +-pi.
        # The setter still bounds each command to within +-pi of the current qpos,
        # so this never introduces large/unsafe setpoint jumps.
        theta_act = act[2]
        model.actuator_ctrllimited[theta_act] = 0
        model.actuator_ctrlrange[theta_act] = [-1.0e6, 1.0e6]


class MobileFrankaDroidRobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = MobileFrankaDroidBaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "arm": FrankaFR3ArmGroup(
                mj_data, base, namespace=namespace, grasp_site_name="gripper/grasp_site"
            ),
            "gripper": RobotIQGripperGroup(mj_data, base, namespace=namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}mobile_franka_droid"

    @property
    def base(self) -> MobileFrankaDroidBaseGroup:
        return self._move_groups["base"]
