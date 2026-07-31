"""Combined mobile pick-and-place-with-navigation task.

Reuses :class:`PickAndPlaceTask`'s success predicate (object supported by the
receptacle, no robot-object contact, receptacle not displaced) and layers on the
minimal navigation interface the A*/pure-pursuit navigation policy expects
(``nav_objs``, ``get_nav_object_priority``, ``set_planned_nav_goal*``), so the
same task instance can drive both the navigation and manipulation sub-policies of
:class:`MobilePickAndPlaceStateMachinePolicy`.
"""

import logging

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.env import BaseMujocoEnv
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask

log = logging.getLogger(__name__)


class MobilePickAndPlaceTask(PickAndPlaceTask):
    """Pick-and-place with a mobile base that must navigate between the pickup
    object and the place receptacle.

    Success is inherited verbatim from :class:`PickAndPlaceTask`; the additional
    surface here is purely the navigation interface consumed by the A* /
    pure-pursuit navigation sub-policy (which targets first the pickup object,
    then the receptacle).
    """

    def __init__(self, env: BaseMujocoEnv, exp_config: MlSpacesExpConfig) -> None:
        super().__init__(env, exp_config)

        # The A* planner builds its own occupancy grid from the model path, so no
        # precomputed occupancy map is required here; expose ``None`` for API
        # parity with the navigation task.
        self.occupancy_map = None

        # Planned pre-grasp / pre-place goal pose published by the navigation
        # policy (world frame). See NavToObjTask for the convention notes.
        self._planned_goal_xy: np.ndarray | None = None
        self._planned_goal_yaw: float | None = None

        # Default navigation target is the pickup object (phase 1).
        self._nav_target_name: str = self.config.task_config.pickup_obj_name
        self.nav_objs: list[list[MlSpacesObject]] = self._build_nav_objs(self._nav_target_name)

    # ------------------------------------------------------------------ #
    # Navigation target management                                        #
    # ------------------------------------------------------------------ #
    def _build_nav_objs(self, object_name: str) -> list[list[MlSpacesObject]]:
        """Build the per-batch navigation object list for ``object_name``."""
        return [
            [MlSpacesObject(data=self._env.mj_datas[i], object_name=object_name)]
            for i in range(self._env.n_batch)
        ]

    def set_nav_target(self, object_name: str) -> None:
        """Point navigation at ``object_name`` (pickup object or receptacle).

        Rebuilds ``nav_objs`` and clears any previously published goal pose so
        the navigation sub-policy re-plans from scratch for the new target.
        """
        self._nav_target_name = object_name
        self.nav_objs = self._build_nav_objs(object_name)
        self._planned_goal_xy = None
        self._planned_goal_yaw = None

    def reset(self):
        result = super().reset()
        # Re-arm navigation for the pickup object each episode.
        self.set_nav_target(self.config.task_config.pickup_obj_name)
        return result

    # ------------------------------------------------------------------ #
    # Navigation interface expected by the A* / pure-pursuit policy       #
    # ------------------------------------------------------------------ #
    def get_nav_object_priority(self, batch_index: int) -> list[MlSpacesObject]:
        """Return candidate navigation objects ordered by proximity (nearest
        first). A single target is used per phase, so the list has one element."""
        if len(self.nav_objs[batch_index]) == 1:
            return self.nav_objs[batch_index][:]

        robot_base_pos = self._env.robots[batch_index].robot_view.base.pose[:3, 3]
        priority = [
            (np.linalg.norm(obj.position[:2] - robot_base_pos[:2]), obj)
            for obj in self.nav_objs[batch_index]
        ]
        return [dist_obj[1] for dist_obj in sorted(priority, key=lambda x: x[0])]

    def get_nearest_nav_object(self, batch_index: int) -> MlSpacesObject | None:
        priority = self.get_nav_object_priority(batch_index)
        return priority[0] if priority else None

    def set_planned_nav_goal(self, position: np.ndarray, quaternion: np.ndarray) -> None:
        """Record the world-frame pre-grasp goal pose the nav policy committed to."""
        pos = np.asarray(position, dtype=float).reshape(-1)
        quat = np.asarray(quaternion, dtype=float).reshape(-1)
        self._planned_goal_xy = pos[:2].copy()
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        self._planned_goal_yaw = float(
            np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        )

    def set_planned_nav_goal_pose(self, xy: np.ndarray, yaw: float) -> None:
        """Record the planar pre-grasp goal pose directly (x, y, yaw)."""
        xy = np.asarray(xy, dtype=float).reshape(-1)
        self._planned_goal_xy = xy[:2].copy()
        self._planned_goal_yaw = float(yaw)
