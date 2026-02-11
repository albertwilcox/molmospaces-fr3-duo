import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mujoco
from mujoco import MjData, MjSpec, mjtGeom

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.kinematics.franka_kinematics import FrankaKinematics
from molmo_spaces.kinematics.parallel.franka_parallel_kinematics import (
    FrankaParallelKinematics,
)
from molmo_spaces.molmo_spaces_constants import get_robot_path
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig


log = logging.getLogger(__name__)


class FrankaRobot(Robot):
    """Franka robot implementation for the framework."""

    def __init__(
        self,
        mj_data: MjData,
        config: "MlSpacesExpConfig",
    ) -> None:
        super().__init__(mj_data, config)
        self._robot_view = config.robot_config.robot_view_factory(
            mj_data, config.robot_config.robot_namespace
        )
        self._kinematics = FrankaKinematics(
            self.mj_model,
            namespace=config.robot_config.robot_namespace,
            robot_view_factory=config.robot_config.robot_view_factory,
        )

        self._parallel_kinematics = FrankaParallelKinematics(config.robot_config)
        arm_controller_cls = (
            JointPosController
            if config.robot_config.command_mode == {}
            or config.robot_config.command_mode["arm"] == "joint_position"
            else JointRelPosController
        )
        self._controllers = {
            "arm": arm_controller_cls(self._robot_view.get_move_group("arm")),
            "gripper": JointPosController(self._robot_view.get_move_group("gripper")),
        }

    @property
    def namespace(self):
        return self.exp_config.robot_config.robot_namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def parallel_kinematics(self):
        return self._parallel_kinematics

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    @property
    def state_dim(self) -> int:
        return 7  # Franka arm has 7 DOF

    def action_dim(self, move_group_ids: list[str]):
        return sum(self._robot_view.get_move_group(mg_id).n_actuators for mg_id in move_group_ids)

    def get_arm_move_group_ids(self) -> list[str]:
        """Franka has a single arm move group."""
        return ["arm"]

    def update_control(self, action_command_dict: dict[str, Any]) -> None:
        action_command_dict = self._apply_action_noise_and_save_unnoised_cmd_jp(action_command_dict)

        for mg_id, controller in self.controllers.items():
            if mg_id in action_command_dict and action_command_dict[mg_id] is not None:
                controller.set_target(action_command_dict[mg_id])
            elif not controller.stationary:
                controller.set_to_stationary()

    def compute_control(self) -> None:
        for controller in self.controllers.values():
            ctrl_inputs = controller.compute_ctrl_inputs()
            controller.robot_move_group.ctrl = ctrl_inputs

    def set_joint_pos(self, robot_joint_pos_dict) -> None:
        for mg_id, joint_pos in robot_joint_pos_dict.items():
            self._robot_view.get_move_group(mg_id).joint_pos = joint_pos

    def set_world_pose(self, robot_world_pose) -> None:
        self._robot_view.base.pose = robot_world_pose

    def reset(self) -> None:
        for mg_id, default_pos in self.exp_config.robot_config.init_qpos.items():
            if mg_id in self._robot_view.move_group_ids():
                self._robot_view.get_move_group(mg_id).joint_pos = default_pos

    @staticmethod
    def robot_model_root_name() -> str:
        return "fr3_link0"

    @classmethod
    def create_robot_base_material(
        cls,
        robot_config: "FrankaRobotConfig",
        spec: MjSpec,
        prefix: str,
        randomize_base_texture: bool,
    ) -> None:
        texture_dir = get_robot_path(robot_config.name) / "assets" / "base_textures"
        assert texture_dir.is_dir(), f"Texture directory {texture_dir} does not exist"
        texture_path: Path | None = None
        if randomize_base_texture:
            texture_paths = list(texture_dir.glob("*.png"))
            texture_paths.sort(key=lambda x: x.name)
            assert len(texture_paths) > 0, f"No robot base texture paths found in {texture_dir}"
            log.debug(f"Found {len(texture_paths)} robot base texture paths")
            texture_path = random.choice(texture_paths)
        else:
            texture_path = texture_dir / "DarkWood2.png"
            assert texture_path.is_file(), f"Default texture {texture_path} does not exist"

        texture_name = f"{prefix}robot_base_texture"
        spec.add_texture(
            name=texture_name,
            type=mujoco.mjtTexture.mjTEXTURE_CUBE,
            file=str(texture_path),
        )
        log.debug(f"Successfully created texture from {texture_path}")

        material_name = f"{prefix}robot_base_material"
        robot_base_mat = spec.add_material(name=material_name)
        robot_base_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name
        log.debug(f"Successfully created material {material_name}")
        return material_name

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "FrankaRobotConfig",
        spec: MjSpec,
        robot_spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
    ) -> None:
        robot_config = cast("FrankaRobotConfig", robot_config)
        add_base = robot_config.base_size is not None
        pos = pos + [0.0] if len(pos) == 2 else pos

        material_name = cls.create_robot_base_material(
            robot_config, spec, prefix, randomize_textures
        )

        robot_body = spec.worldbody.add_body(
            name=f"{prefix}base",
            pos=pos,
            quat=quat,
            mocap=True,
        )
        if add_base:
            base_height = robot_config.base_size[2]

            # Add base geometry (wooden platform)
            robot_body.add_geom(
                type=mjtGeom.mjGEOM_BOX,
                size=[x / 2 for x in robot_config.base_size],
                pos=[0, 0, base_height / 2],
                material=material_name,
                group=0,  # Visual group
            )
            attach_frame = robot_body.add_frame(pos=[0, 0, base_height])
        else:
            attach_frame = robot_body.add_frame()

        # Attach the robot to the base via the frame
        robot_root_name = cls.robot_model_root_name()
        robot_root = robot_spec.body(robot_root_name)
        if robot_root is None:
            raise ValueError(f"Robot {robot_root_name=} not found in {robot_spec}")
        attach_frame.attach_body(robot_root, prefix, "")
