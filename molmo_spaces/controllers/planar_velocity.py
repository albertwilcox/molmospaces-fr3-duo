"""Base-local planar velocity commands for the RoboCasa transfer."""

import numpy as np

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.robots.robot_views.abstract import MoveGroup


class PlanarVelocityController(Controller):
    """Normalized [forward, lateral, yaw] mapped to world-frame joint velocities."""

    def __init__(self, robot_move_group: MoveGroup) -> None:
        super().__init__(robot_move_group)
        self.reset()

    @property
    def target(self) -> np.ndarray:
        return self._target.copy()

    @property
    def stationary(self) -> bool:
        return self._stationary

    def set_target(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError(f"Expected finite planar velocity (3,), got {target}")
        self._target = np.clip(target, -1.0, 1.0)
        self._stationary = False
        yaw = self.robot_move_group.joint_pos[2]
        c, s = np.cos(yaw), np.sin(yaw)
        x, y, theta = self._target
        self._world_velocity = np.array([c * x - s * y, s * x + c * y, 1.5 * theta])

    def compute_ctrl_inputs(self) -> np.ndarray:
        return self._world_velocity.copy()

    def set_to_stationary(self) -> None:
        self._stationary = True
        self._target = np.zeros(3)
        self._world_velocity = np.zeros(3)

    def reset(self) -> None:
        self.set_to_stationary()
