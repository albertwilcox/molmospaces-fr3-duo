from mujoco import MjData, MjModel

from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.robot_views.rby1_view import RBY1RobotView


class RBY1Kinematics(MlSpacesKinematics):
    def __init__(
        self,
        model: MjModel,
        data: MjData | None = None,
        namespace: str = "",
        holo_base: bool = False,
    ) -> None:
        if data is None:
            data = MjData(model)
        robot_view = RBY1RobotView(data, namespace=namespace, holo_base=holo_base)
        super().__init__(data, robot_view)


if __name__ == "__main__":

    def main() -> None:
        import mujoco
        import numpy as np
        from mujoco.viewer import launch_passive

        np.set_printoptions(linewidth=np.inf)
        import os
        import time

        cwd = os.getcwd()
        os.chdir("assets/robots/rby1")
        model = MjModel.from_xml_path("rby1.xml")
        os.chdir(cwd)

        data = MjData(model)
        mujoco.mj_forward(model, data)

        namespace = ""
        robot_view = RBY1RobotView(data, namespace=namespace)
        right_arm = robot_view.get_move_group("right_arm")
        kinematics = RBY1Kinematics(model, namespace=namespace)

        qp = np.random.uniform(*right_arm.joint_pos_limits.T)
        right_arm.joint_pos = qp
        print(qp)
        mujoco.mj_forward(model, data)

        pose0 = robot_view.base.pose @ right_arm.leaf_frame_to_robot
        pose1 = pose0.copy()
        pose1[2, 3] += 0.2

        groups = [
            "right_arm",
            "torso",
        ]

        with launch_passive(model, data) as viewer:
            viewer.sync()
            i = 0
            while viewer.is_running():
                if i % 2 == 0:
                    ret = kinematics.ik(
                        "right_arm", pose1, groups, robot_view.get_qpos_dict(), robot_view.base.pose
                    )
                else:
                    ret = kinematics.ik(
                        "right_arm", pose0, groups, robot_view.get_qpos_dict(), robot_view.base.pose
                    )
                print(i % 2, ret)
                i += 1
                if ret is not None:
                    robot_view.set_qpos_dict(ret)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(3)

    main()
