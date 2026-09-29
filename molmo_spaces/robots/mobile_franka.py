import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
from mujoco import MjData, MjSpec, mjtGeom
from scipy.spatial.transform import Rotation as R

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.controllers.osc_pose import OSCPoseController
from molmo_spaces.controllers.osc_torque import TorqueOSCConfig, TorqueOSCController
from molmo_spaces.controllers.planar_velocity import PlanarVelocityController
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.warp_kinematics import SimpleWarpKinematics
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import MobileFrankaRobotConfig


log = logging.getLogger(__name__)


def calibrate_planar_translation(
    model: mujoco.MjModel, namespace: str, root_body_id: int, reference_mass_kg: float
) -> float:
    """Normalize translation transients while preserving speed and deadband."""
    if not np.isfinite(reference_mass_kg) or reference_mass_kg <= 0:
        raise ValueError("reference_mass_kg must be finite and positive")
    mass = float(model.body_subtreemass[root_body_id])
    scale = mass / reference_mass_kg
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The planar base must have finite positive subtree mass")
    for name in ("base_x", "base_y"):
        actuator = model.actuator(f"{namespace}{name}_act")
        joint = model.joint(f"{namespace}{name}")
        actuator.gainprm[0] = 1000.0 * scale
        actuator.biasprm[2] = -1000.0 * scale
        actuator.forcerange[:] = np.array([-600.0, 600.0]) * scale
        model.dof_frictionloss[joint.dofadr] = 250.0 * scale
    log.info(
        "Planar translation mass calibration: %.6f kg / %.6f kg = %.6f",
        mass, reference_mass_kg, scale,
    )
    return scale


class MobileFrankaRobot(Robot):
    def __init__(
        self,
        mj_data: MjData,
        config: "MlSpacesExpConfig",
    ) -> None:
        super().__init__(mj_data, config)
        self._robot_view = config.robot_config.robot_view_factory(
            mj_data, config.robot_config.robot_namespace
        )
        reference_mass = config.robot_config.base_velocity_reference_mass_kg
        if reference_mass is not None:
            if config.robot_config.command_mode["base"] != "holo_joint_planar_velocity":
                raise ValueError("Base mass calibration requires planar velocity control")
            calibrate_planar_translation(
                mj_data.model, config.robot_config.robot_namespace,
                self._robot_view.base.root_body_id, reference_mass,
            )
        self._kinematics = MlSpacesKinematics(config.robot_config, reference_robot_view=self._robot_view)

        self._parallel_kinematics = SimpleWarpKinematics(config.robot_config)
        arm_command_mode = (
            "joint_position"
            if config.robot_config.command_mode == {}
            else config.robot_config.command_mode["arm"]
        )
        if config.robot_config.torque_velocity_reference != "tcp" and arm_command_mode != "osc_torque":
            raise ValueError("Arm-root velocity-reference calibration requires torque OSC")
        if arm_command_mode == "osc_pose":
            arm_controller: Controller = OSCPoseController(
                self._robot_view.get_move_group("arm"),
                move_group_id="arm",
                kinematics=self._kinematics,
                robot_view=self._robot_view,
            )
        elif arm_command_mode == "osc_torque":
            if config.robot_config.action_noise_config.enabled:
                raise ValueError("Joint-space action noise is incompatible with torque OSC")
            arm_controller = TorqueOSCController(
                self._robot_view.get_move_group("arm"),
                self._robot_view,
                TorqueOSCConfig(velocity_reference=config.robot_config.torque_velocity_reference),
            )
        elif arm_command_mode == "joint_position":
            arm_controller = JointPosController(self._robot_view.get_move_group("arm"))
        else:
            arm_controller = JointRelPosController(self._robot_view.get_move_group("arm"))
        base_controller_cls: type[Controller] = {
            "holo_joint_planar_position": JointPosController,
            "holo_joint_rel_planar_position": JointRelPosController,
            "holo_joint_planar_velocity": PlanarVelocityController,
        }[config.robot_config.command_mode["base"]]
        self._controllers = {
            "base": base_controller_cls(self._robot_view.get_move_group("base")),
            "arm": arm_controller,
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

    def get_arm_move_group_ids(self) -> list[str]:
        return ["arm"]

    def reset(self) -> None:
        for mg_id, default_pos in self.exp_config.robot_config.init_qpos.items():
            if mg_id in self._robot_view.move_group_ids():
                self._robot_view.get_move_group(mg_id).joint_pos = default_pos

    def compute_control(self) -> None:
        for controller in self.controllers.values():
            controls = controller.compute_ctrl_inputs()
            group = controller.robot_move_group
            if isinstance(controller, PlanarVelocityController):
                # Holonomic position setters wrap yaw; velocity commands must not.
                group.mj_data.ctrl[group.actuator_ids] = controls
            else:
                group.ctrl = controls

    @staticmethod
    def robot_model_root_name() -> str:
        return "fr3_link0"

    @classmethod
    def apply_control_overrides(cls, spec: MjSpec, robot_config: "MobileFrankaRobotConfig") -> None:
        super().apply_control_overrides(spec, robot_config)
        if robot_config.command_mode.get("arm") != "osc_torque":
            return
        limits = {
            f"{robot_config.robot_namespace}fr3_joint{i}": 87.0 if i <= 4 else 12.0
            for i in range(1, 8)
        }
        converted = 0
        for actuator in spec.actuators:
            if actuator.target not in limits:
                continue
            limit = limits[actuator.target]
            actuator.inheritrange = 0
            actuator.gainprm[:] = 0
            actuator.gainprm[0] = 1
            actuator.biasprm[:] = 0
            actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
            actuator.ctrlrange = [-limit, limit]
            actuator.ctrllimited = True
            actuator.forcerange = [-limit, limit]
            actuator.forcelimited = True
            converted += 1
        if converted != 7:
            raise ValueError(f"Torque OSC requires seven arm actuators, found {converted}")

    @classmethod
    def create_robot_base_material(
        cls,
        robot_config: "MobileFrankaRobotConfig",
        spec: MjSpec,
        prefix: str,
        randomize_base_texture: bool,
    ) -> None:
        texture_dir = robot_config.get_robot_dir() / "assets" / "base_textures"
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
        robot_config: "MobileFrankaRobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        def add_slider_act(
            name: str, ctrlrange: float, gainprm: float, biasprm: list[float], gear_idx: int
        ):
            act = spec.add_actuator()
            act.name = f"{prefix}{name}"
            act.target = f"{prefix}base_site"
            act.refsite = f"{prefix}world"
            act.ctrlrange = np.array([-ctrlrange, ctrlrange])
            act.gainprm[0] = gainprm
            act.biasprm[: len(biasprm)] = biasprm
            act.trntype = mujoco.mjtTrn.mjTRN_SITE
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            gear = [0] * 6
            gear[gear_idx] = 1
            act.gear = gear
            return act

        robot_config = cast("MobileFrankaRobotConfig", robot_config)
        if (
            any(robot_config.base_yaw_pivot_offset_xy_m)
            and robot_config.command_mode["base"] != "holo_joint_planar_velocity"
        ):
            raise ValueError("Offset yaw-pivot calibration requires planar velocity control")
        pos = pos + [0.005] if len(pos) == 2 else pos

        material_name = cls.create_robot_base_material(
            robot_config, spec, prefix, randomize_textures
        )

        robot_body = spec.worldbody.add_body(
            name=f"{prefix}base",
            pos=pos,
            quat=quat,
        )
        robot_body.add_site(name=f"{prefix}base_site", pos=[0, 0, 0], quat=[1, 0, 0, 0])
        base_height = robot_config.base_size[2]

        init_rot = R.from_quat(quat, scalar_first=True)
        init_rpy = init_rot.as_euler("xyz")
        assert np.allclose(init_rpy[:2], [0, 0]), (
            f"Initial roll and pitch are not zero: {init_rpy[:2]}"
        )
        init_yaw = init_rpy[2]

        # Add base geometry (wooden platform)
        robot_body.add_geom(
            type=mjtGeom.mjGEOM_BOX,
            size=[x / 2 for x in robot_config.base_size],
            pos=[0, 0, base_height / 2],
            material=material_name,
            group=0,  # Visual group
        )
        attach_frame = robot_body.add_frame(pos=[0, 0, base_height])

        robot_spec = cls._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        robot_root_name = cls.robot_model_root_name()
        robot_root = robot_spec.body(robot_root_name)
        if robot_root is None:
            raise ValueError(f"Robot {robot_root_name=} not found in {robot_spec}")
        attach_frame.attach_body(robot_root, prefix, "")

        spec.worldbody.add_site(name=f"{prefix}world", pos=[0, 0, 0.005], quat=[1, 0, 0, 0])

        planar_joints = []
        planar_actuators = []
        for gear_idx, jnt_name in enumerate(["base_x", "base_y"]):
            act_name = jnt_name + "_act"
            params = robot_config.base_control_params[act_name]
            jnt_axis = np.zeros(3)
            jnt_axis[gear_idx] = 1
            jnt_axis = init_rot.inv().apply(jnt_axis)
            joint = robot_body.add_joint(
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                name=f"{prefix}{jnt_name}",
                axis=jnt_axis,
                range=[-params["ctrlrange"], params["ctrlrange"]],
                ref=pos[gear_idx],
            )
            planar_joints.append(joint)
            # use damping ratio if available, otherwise use kd (which should be negative in biasprm)
            actuator = add_slider_act(
                act_name,
                params["ctrlrange"],
                params["kp"],
                [0, -params["kp"], params.get("damping_ratio", -params.get("kd", 0.0))],
                gear_idx,
            )
            planar_actuators.append(actuator)

        theta_act_params = robot_config.base_control_params["base_theta_act"]
        yaw_joint = robot_body.add_joint(
            type=mujoco.mjtJoint.mjJNT_HINGE,
            name=f"{prefix}base_theta",
            axis=[0, 0, 1],
            pos=[*robot_config.base_yaw_pivot_offset_xy_m, 0.0],
            ref=init_yaw,
        )
        planar_joints.append(yaw_joint)
        theta_act = spec.add_actuator(
            name=f"{prefix}base_theta_act",
            target=f"{prefix}base_theta",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        )
        # The yaw target is an absolute heading in [-pi, pi]. Without an explicit
        # ctrlrange the actuator defaults to a [0, 0] limit, which makes the
        # position controller clip every heading command to 0 (the base can then
        # never turn toward a waypoint). Match the RBY1 holonomic base (+/- pi).
        theta_act.ctrlrange = np.array([-np.pi, np.pi])
        theta_act.gainprm[0] = theta_act_params["kp"]
        theta_act.biasprm[1] = -theta_act_params["kp"]
        # use damping ratio if available, otherwise use kd (which should be negative in biasprm)
        theta_act.biasprm[2] = theta_act_params.get(
            "damping_ratio", -theta_act_params.get("kd", 0.0)
        )
        planar_actuators.append(theta_act)

        if robot_config.command_mode["base"] == "holo_joint_planar_velocity":
            # Match Omron velocity actuators, including force/friction limits.
            for joint, act, limit, kv in zip(
                planar_joints, planar_actuators, [1.0, 1.0, 1.5], [1000.0, 1000.0, 1500.0], strict=True
            ):
                joint.frictionloss = 250.0
                act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                act.target = joint.name
                act.refsite = ""
                act.gear = [1, 0, 0, 0, 0, 0]
                act.gainprm[0] = kv
                act.biasprm[:3] = [0, 0, -kv]
                act.ctrlrange = [-limit, limit]
                act.ctrllimited = True
                act.forcerange = [-600, 600]
                act.forcelimited = True


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R
    import mujoco.viewer

    from molmo_spaces.configs.robot_configs import MobileFrankaRobotConfig
    from molmo_spaces.molmo_spaces_constants import get_procthor_10k_houses
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path,
    )

    houses = get_procthor_10k_houses(split="val")
    house_xml_path = houses["val"][0]["base"]
    install_scene_with_objects_and_grasps_from_path(house_xml_path)

    spec = MjSpec.from_file(house_xml_path)

    robot_config = MobileFrankaRobotConfig(base_size=[0.5, 0.5, 0.75])
    robot_config.init_qpos["base"] = [6.8, 9.75, np.radians(90.0)]

    MobileFrankaRobot.add_robot_to_scene(
        robot_config,
        spec,
        prefix=robot_config.robot_namespace,
        pos=robot_config.init_qpos["base"][:2],
        quat=R.from_euler("z", robot_config.init_qpos["base"][2]).as_quat(scalar_first=True),
    )
    MobileFrankaRobot.apply_control_overrides(spec, robot_config)

    model = spec.compile()
    data = MjData(model)
    view = robot_config.robot_view_factory(data, robot_config.robot_namespace)

    view.set_qpos_dict(robot_config.init_qpos)
    mujoco.mj_forward(model, data)
    for mg_id in view.move_group_ids():
        mg = view.get_move_group(mg_id)
        mg.ctrl = mg.noop_ctrl
    mujoco.mj_forward(model, data)

    mujoco.viewer.launch(model, data)
