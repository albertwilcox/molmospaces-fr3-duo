from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
from molmo_spaces.utils.mujoco_scene_utils import is_object_supported_by_body
from molmo_spaces.utils.pose import pos_quat_to_pose_mat


class PickAndPlaceTask(BaseMujocoTask):
    """Franka pick-and-place task implementation."""

    def get_task_description(self) -> str:
        pickup_name = self.config.task_config.referral_expressions["pickup_name"]
        place_name = self.config.task_config.referral_expressions["place_name"]
        return f"Pick up the {pickup_name} and place it in or on the {place_name}"

    def _create_sensor_suite_from_config(self, config: MlSpacesExpConfig) -> SensorSuite:
        """Create a sensor suite from configuration using the centralized get_core_sensors function."""
        from molmo_spaces.env.sensors import get_core_sensors

        sensors = get_core_sensors(config)
        return SensorSuite(sensors)

    def judge_success(self) -> bool:
        """Judge if the task was successful (for data generation)."""
        return self.get_info()[0]["success"]

    def get_reward(self) -> np.ndarray:
        """Calculate reward for each environment in the batch."""
        rewards = np.zeros(self._env.n_batch)

        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]
            task_config = self.config.task_config
            assert isinstance(task_config, PickAndPlaceTaskConfig)
            pickup_obj = create_mlspaces_body(data, task_config.pickup_obj_name)
            place_receptacle = create_mlspaces_body(data, task_config.place_receptacle_name)

            place_receptacle_aabb_center, _ = body_aabb(data.model, data, place_receptacle.body_id)

            pos_err = np.linalg.norm(pickup_obj.position - place_receptacle_aabb_center)
            rewards[i] = np.exp(-pos_err)

        return rewards

    def get_info(self) -> list[dict[str, Any]]:
        """Get additional metrics for each environment."""
        metrics = []

        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]
            task_config = self.config.task_config
            assert isinstance(task_config, PickAndPlaceTaskConfig)
            pickup_obj = create_mlspaces_body(data, task_config.pickup_obj_name)
            place_receptacle = create_mlspaces_body(data, task_config.place_receptacle_name)

            pickup_obj_aabb = body_aabb(data.model, data, pickup_obj.body_id)
            pickup_obj_aabb_min = pickup_obj_aabb[0] - pickup_obj_aabb[1] / 2
            pickup_obj_aabb_max = pickup_obj_aabb[0] + pickup_obj_aabb[1] / 2
            place_receptacle_aabb_center, _ = body_aabb(data.model, data, place_receptacle.body_id)

            # the pos err is the distance between the receptacle center and the pickup object aabb
            # pos err is 0 when the pickup object contains the receptacle center point
            pos_err = np.linalg.norm(
                np.maximum(0, pickup_obj_aabb_min - place_receptacle_aabb_center)
                + np.maximum(0, place_receptacle_aabb_center - pickup_obj_aabb_max)
            )

            # Success check
            # is the pickup object supported by the receptacle?

            # based on contact force only:
            supported_by_receptacle = is_object_supported_by_body(
                data,
                pickup_obj.body_id,
                place_receptacle.body_id,
                frac_weight_threshold=task_config.receptacle_supported_weight_frac,
            )

            if not supported_by_receptacle:
                # Use heuristic
                om = self._env.object_managers[i]
                objects_on_receptacle = om.objects_on_receptacle(
                    [om.get_object_by_name(task_config.pickup_obj_name)],
                    om.get_object_by_name(task_config.place_receptacle_name).geom_ids,
                )
                names_on_receptacle = {obj.name for obj in objects_on_receptacle}
                supported_by_receptacle = task_config.pickup_obj_name in names_on_receptacle

            # has the place receptacle moved too much?
            start_pose = pos_quat_to_pose_mat(
                task_config.place_receptacle_start_pose[0:3],
                task_config.place_receptacle_start_pose[3:7],
            )
            curr_pose = place_receptacle.pose
            displacement = np.linalg.inv(start_pose) @ curr_pose
            pos_displacement = displacement[:3, 3]
            rot_displacement = R.from_matrix(displacement[:3, :3]).magnitude()

            success = (
                supported_by_receptacle
                and np.linalg.norm(pos_displacement)
                <= task_config.max_place_receptacle_pos_displacement
                and rot_displacement <= task_config.max_place_receptacle_rot_displacement
            )

            metrics.append(
                {
                    "position_error": pos_err,
                    "success": success,
                    "supported_by_receptacle": supported_by_receptacle,
                    "receptacle_pos_displacement": pos_displacement,
                    "receptacle_rot_displacement": rot_displacement,
                    "episode_step": self.episode_step_count,
                }
            )

        return metrics

    def get_obs_scene(self):
        """
        This is for observations that are constant over all time steps of an env.
        """
        obs_scene = super().get_obs_scene()
        text = f"{self.config.task_type} {self.config.task_config.pickup_obj_name} {self.config.task_config.place_receptacle_name}"
        obs_extra = dict(
            text=text,
            object_name=self.config.task_config.pickup_obj_name,
            place_receptacle_name=self.config.task_config.place_receptacle_name,
        )
        obs_scene.update(obs_extra)
        return obs_scene
