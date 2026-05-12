"""BimanualFrankaConfig — standalone config module.

This gets copied into molmo_spaces/configs/ by install.sh so it's importable
and picklable (required by MolmoSpaces config serialization).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mujoco import MjData

from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.bimanual_franka import BimanualFrankaRobot
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory
from molmo_spaces.robots.robot_views.bimanual_franka_view import BimanualFrankaRobotView


class BimanualFrankaConfig(BaseRobotConfig):
    """Configuration for bimanual Franka robot (two FR3 arms with Robotiq 2F-85 grippers)."""

    robot_cls: type[BimanualFrankaRobot] | None = BimanualFrankaRobot
    robot_factory: Callable[[MjData, Any], Robot] | None = BimanualFrankaRobot
    robot_view_factory: RobotViewFactory | None = BimanualFrankaRobotView
    robot_namespace: str = "robot_0/"
    default_world_pose: list[float] = [0, 0, 0, 1, 0, 0, 0]
    name: str = "bimanual_franka"
    robot_xml_path: Path = Path("fr3_duo.xml")
    base_size: list[float] | None = [0.4, 0.5, 0.7]
    init_qpos: dict[str, list[float]] = {
        "left_arm": [0, -0.7853, 0, -2.35619, 0, 1.57079, 0.0],
        "right_arm": [0, -0.7853, 0, -2.35619, 0, 1.57079, 0.0],
        "left_gripper": [0.0, 0.0],
        "right_gripper": [0.0, 0.0],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = None
    command_mode: dict[str, str] = {
        "arm": "joint_position",
        "gripper": "joint_position",
    }
    gravcomp: bool = True

    def model_post_init(self, __context):
        super().model_post_init(__context)
