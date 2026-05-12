"""Kinematics solver for the bimanual Franka robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import BaseRobotConfig

# Map each gripper/arm to the joints that should be unlocked for IK.
# Only the arm joints move the TCP — gripper joints only open/close fingers.
_GRIPPER_TO_UNLOCKED = {
    "left_gripper": ["left_arm"],
    "right_gripper": ["right_arm"],
}

_ARM_TO_UNLOCKED = {
    "left_arm": ["left_arm"],
    "right_arm": ["right_arm"],
}


class BimanualFrankaKinematics(MlSpacesKinematics):
    """Kinematics solver for the bimanual Franka (two 7-DOF FR3 arms).

    Overrides IK to only unlock the arm kinematically connected to the
    requested gripper, preventing the solver from trying to move the
    other arm (which creates a poorly conditioned Jacobian).
    """

    def __init__(self, robot_config: "BaseRobotConfig") -> None:
        super().__init__(robot_config)

    def ik(
        self,
        move_group_id: str,
        pose: np.ndarray,
        unlocked_move_group_ids: list[str] | None,
        q0: dict[str, np.ndarray],
        base_pose: np.ndarray,
        rel_to_base: bool = False,
        **kwargs,
    ):
        # Filter unlocked groups to only the arm connected to the target
        if move_group_id in _GRIPPER_TO_UNLOCKED:
            unlocked_move_group_ids = _GRIPPER_TO_UNLOCKED[move_group_id]
        elif move_group_id in _ARM_TO_UNLOCKED:
            unlocked_move_group_ids = _ARM_TO_UNLOCKED[move_group_id]

        jp = super().ik(
            move_group_id, pose, unlocked_move_group_ids, q0, base_pose,
            rel_to_base=rel_to_base, **kwargs
        )

        # The caller (_tcp_to_jp_fn) extracts jp[mg_id] for ALL non-gripper
        # move groups (e.g. base, left_arm, right_arm).  We only solved for
        # one arm, so fill in q0 values for the other groups so the caller
        # doesn't hit a KeyError.
        if jp is not None:
            for mg_id, qpos in q0.items():
                if mg_id not in jp:
                    jp[mg_id] = qpos

        return jp
