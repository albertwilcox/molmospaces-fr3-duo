import logging
from typing import Any

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.env import BaseMujocoEnv
from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)


# Many scene object categories arrive as a single lowercase token with the word
# boundaries stripped (e.g. "alarmclock", "diningtable", "chestofdrawers"). The
# per-asset annotation "category" field is inconsistent (some assets store the
# spaced form, some the concatenated form), so we prettify with an explicit
# lookup of the known multi-word AI2-THOR / objathor object types the nav task
# can target. Anything not in the table (already-spaced names, genuine single
# words like "newspaper"/"basketball") is returned unchanged.
_OBJECT_NAME_PRETTY: dict[str, str] = {
    "alarmclock": "alarm clock",
    "baseballbat": "baseball bat",
    "butterknife": "butter knife",
    "cellphone": "cell phone",
    "chestofdrawers": "chest of drawers",
    "coffeemachine": "coffee machine",
    "coffeemaker": "coffee maker",
    "coffeetable": "coffee table",
    "compactdisk": "compact disk",
    "countertop": "counter top",
    "crapper": "toilet",
    "creditcard": "credit card",
    "desklamp": "desk lamp",
    "diningtable": "dining table",
    "dishsponge": "dish sponge",
    "dogbed": "dog bed",
    "floorlamp": "floor lamp",
    "garbagebag": "garbage bag",
    "garbagecan": "garbage can",
    "handtowel": "hand towel",
    "handtowelholder": "hand towel holder",
    "keychain": "key chain",
    "laundryhamper": "laundry hamper",
    "lightswitch": "light switch",
    "papertowelroll": "paper towel roll",
    "peppershaker": "pepper shaker",
    "remotecontrol": "remote control",
    "roomdecor": "room decor",
    "saltshaker": "salt shaker",
    "scrubbrush": "scrub brush",
    "shelvingunit": "shelving unit",
    "showercurtain": "shower curtain",
    "showerdoor": "shower door",
    "showerglass": "shower glass",
    "showerhead": "shower head",
    "sidetable": "side table",
    "sinkbasin": "sink basin",
    "soapbar": "soap bar",
    "soapbottle": "soap bottle",
    "soapdispenser": "soap dispenser",
    "spraybottle": "spray bottle",
    "stoveburner": "stove burner",
    "stoveknob": "stove knob",
    "tabletopdecor": "table top decor",
    "tablelamp": "table lamp",
    "teddybear": "teddy bear",
    "tennisracket": "tennis racket",
    "tissuebox": "tissue box",
    "tissuepaper": "tissue paper",
    "toiletpaper": "toilet paper",
    "toiletpaperhanger": "toilet paper hanger",
    "towelholder": "towel holder",
    "trashcan": "trash can",
    "tvstand": "tv stand",
    "vacuumcleaner": "vacuum cleaner",
    "wateringcan": "watering can",
    "winebottle": "wine bottle",
}


def prettify_object_name(name: str) -> str:
    """Insert word boundaries into concatenated object-type names.

    Looks up the collapsed (spaces removed) form in ``_OBJECT_NAME_PRETTY`` so
    that both "alarmclock" and an already-spaced "alarm clock" map to the same
    readable label. Unknown names are returned lowercased and stripped."""
    if not name:
        return name
    cleaned = " ".join(name.split()).strip().lower()
    collapsed = cleaned.replace(" ", "")
    return _OBJECT_NAME_PRETTY.get(collapsed, cleaned)


class NavToObjTask(BaseMujocoTask):
    """Navigation to object task implementation."""

    def __init__(self, env: BaseMujocoEnv, exp_config: MlSpacesExpConfig) -> None:
        super().__init__(env, exp_config)
        self.exp_config = exp_config

        # For eval mode: reconstruct candidate list from category if needed
        self._reconstruct_candidate_list_if_needed(env)

        self.nav_objs = self._get_nav_objects()

        # Planned pre-grasp goal pose published by the navigation policy (world
        # frame). Populated via :meth:`set_planned_nav_goal`; consumed by the
        # goal-pose-reaching success criterion (``succ_use_goal_pose``). ``None``
        # until the policy commits to a goal (falls back to distance success).
        self._planned_goal_xy: np.ndarray | None = None
        self._planned_goal_yaw: float | None = None

    def set_planned_nav_goal(self, position: np.ndarray, quaternion: np.ndarray) -> None:
        """Record the pre-grasp goal pose the navigation policy committed to.

        Called by the A* / pure-pursuit policy whenever it (re)selects a goal.
        ``position`` is a world-frame (x, y, z); ``quaternion`` is world-frame
        [w, x, y, z]. Only the planar (x, y) and yaw are retained — the success
        criterion is a planar base-pose match. Safe to call every step.

        NOTE: the pure-pursuit follower supersedes this with
        :meth:`set_planned_nav_goal_pose` using the true tracked plan endpoint and
        facing, which is in the base-frame convention that matches the robot pose.
        """
        pos = np.asarray(position, dtype=float).reshape(-1)
        quat = np.asarray(quaternion, dtype=float).reshape(-1)
        self._planned_goal_xy = pos[:2].copy()
        # Yaw about world +z from a [w, x, y, z] quaternion.
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        self._planned_goal_yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def set_planned_nav_goal_pose(self, xy: np.ndarray, yaw: float) -> None:
        """Record the planar pre-grasp goal pose directly (x, y, yaw).

        Preferred over :meth:`set_planned_nav_goal`: the pure-pursuit follower
        passes the exact plan endpoint it tracks and the ``_final_face_theta`` it
        aligns to, both in the same base-frame convention as the robot base pose,
        so the goal-reach and heading-error checks are convention-consistent.
        """
        xy = np.asarray(xy, dtype=float).reshape(-1)
        self._planned_goal_xy = xy[:2].copy()
        self._planned_goal_yaw = float(yaw)

    def _robot_base_xy_yaw(self, index: int) -> tuple[np.ndarray, float]:
        robot = self._env.robots[index]
        pose = robot.robot_view.base.pose
        xy = np.asarray(pose[:2, 3], dtype=float)
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        return xy, yaw

    def distance_to_goal(self, index: int) -> float:
        """Planar distance (m) from the robot base to the planned goal pose.

        Returns ``inf`` when no goal has been published, so callers treat an
        un-published goal as "not reached".
        """
        if self._planned_goal_xy is None:
            return float("inf")
        xy, _ = self._robot_base_xy_yaw(index)
        return float(np.linalg.norm(xy - self._planned_goal_xy))

    def goal_yaw_error(self, index: int) -> float:
        """Absolute wrapped heading error (rad) between the base and the goal."""
        if self._planned_goal_yaw is None:
            return float("inf")
        _, yaw = self._robot_base_xy_yaw(index)
        err = self._planned_goal_yaw - yaw
        return float(abs(np.arctan2(np.sin(err), np.cos(err))))

    def planned_goal_surface_distance(self, index: int) -> float:
        """Surface distance (m) from the planned goal pose to the target object.

        Mirrors :meth:`calculate_distance`'s surface convention but measured from
        the *goal* pose rather than the robot, so the validity guard can reject
        goals whose standoff exceeds ``max_pregrasp_standoff_m``. Returns ``inf``
        if no goal is published.
        """
        if self._planned_goal_xy is None:
            return float("inf")
        nearest_obj = self.get_nearest_nav_object(index)
        center_dist = float(np.linalg.norm(np.asarray(nearest_obj.position)[:2] - self._planned_goal_xy))
        if not self.config.task_config.succ_use_surface_distance:
            return center_dist
        effective_radius = float(np.max(np.asarray(nearest_obj.aabb_size)[:2]))
        return max(0.0, center_dist - effective_radius)

    def _reconstruct_candidate_list_if_needed(self, env: BaseMujocoEnv) -> None:
        """Reconstruct candidate list from category for eval mode.

        This is needed because saved benchmarks may contain pickup_obj_candidates
        that include objects of multiple categories. We need to filter them to only
        include objects of the target category.
        """
        # Check if we have a pickup_obj_name to work with
        if (
            not hasattr(self.config.task_config, "pickup_obj_name")
            or self.config.task_config.pickup_obj_name is None
        ):
            return

        # Check if candidates list exists and needs filtering
        if (
            not hasattr(self.config.task_config, "pickup_obj_candidates")
            or not self.config.task_config.pickup_obj_candidates
        ):
            return

        om = env.object_managers[env.current_batch_index]

        # Try to get category from config, or infer from pickup_obj_name
        target_category = None
        target_synset = None

        if (
            hasattr(self.config.task_config, "pickup_obj_category")
            and self.config.task_config.pickup_obj_category is not None
        ):
            # Category already saved in config (new data)
            target_category = self.config.task_config.pickup_obj_category
            target_synset = getattr(self.config.task_config, "pickup_obj_synset", None)
            log.info(f"[NavTask] Using saved category: {target_category} (synset: {target_synset})")
        else:
            # Infer category from pickup_obj_name (old data)
            try:
                target_category = om.category_from_name(self.config.task_config.pickup_obj_name)
                target_synset = om.get_annotation_synset(self.config.task_config.pickup_obj_name)
                # Save for future reference
                self.config.task_config.pickup_obj_category = target_category
                self.config.task_config.pickup_obj_synset = target_synset
                log.info(
                    f"[NavTask] Inferred category '{target_category}' (synset: {target_synset}) from pickup_obj_name"
                )
            except Exception as e:
                log.warning(f"[NavTask] Could not infer category from pickup_obj_name: {e}")
                return

        # Filter saved candidates to only include same category
        saved_candidates = self.config.task_config.pickup_obj_candidates
        filtered_candidates = []

        for obj_name in saved_candidates:
            try:
                obj_category = om.category_from_name(obj_name)
                obj_synset = om.get_annotation_synset(obj_name)

                # Match by category or synset
                if obj_category == target_category or (
                    target_synset and obj_synset == target_synset
                ):
                    filtered_candidates.append(obj_name)
            except Exception:
                # Skip objects that can't be categorized
                continue

        # Update config with filtered candidates if we found any
        if len(filtered_candidates) > 0:
            log.info(
                f"[NavTask] ✅ Filtered candidates: {len(saved_candidates)} → {len(filtered_candidates)} "
                f"(category: '{target_category}')"
            )
            log.info(f"[NavTask]    First 5: {filtered_candidates[:5]}")
            self.config.task_config.pickup_obj_candidates = filtered_candidates

            # Use first candidate as default pickup_obj_name if original is not in filtered list
            if self.config.task_config.pickup_obj_name not in filtered_candidates:
                old_name = self.config.task_config.pickup_obj_name
                self.config.task_config.pickup_obj_name = filtered_candidates[0]
                log.warning(
                    f"[NavTask] Original object '{old_name}' not in filtered candidates. "
                    f"Using '{filtered_candidates[0]}' as default."
                )
        else:
            log.warning(
                f"[NavTask] ⚠️  No objects in saved candidates match category '{target_category}'. "
                f"Keeping all {len(saved_candidates)} saved candidates."
            )

    def _get_nav_objects(self) -> list[list[MlSpacesObject]]:
        """Get all navigation objects of the target type for each batch.

        Returns:
            List of lists, where nav_objs[batch_idx] contains all candidate objects for that batch.
        """
        nav_objs_per_batch = []

        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]

            # If pickup_obj_candidates exists, create objects for all candidates
            if (
                hasattr(self.config.task_config, "pickup_obj_candidates")
                and self.config.task_config.pickup_obj_candidates is not None
                and len(self.config.task_config.pickup_obj_candidates) > 0
            ):
                objs = []
                for obj_name in self.config.task_config.pickup_obj_candidates:
                    try:
                        obj = MlSpacesObject(data=data, object_name=obj_name)
                        objs.append(obj)
                    except Exception as e:
                        print(f"Warning: Could not create MlSpacesObject for {obj_name}: {e}")

                nav_objs_per_batch.append(objs)
            else:
                # Backward compatibility: single object mode
                pickup_obj = MlSpacesObject(
                    data=data, object_name=self.config.task_config.pickup_obj_name
                )
                nav_objs_per_batch.append([pickup_obj])

        return nav_objs_per_batch

    def get_nav_object_priority(self, batch_index: int) -> list[MlSpacesObject]:
        """Get the nearest navigation object for the given batch.

        Args:
            batch_index: Index of the environment batch

        Returns:
            The MlSpacesObject instance that is nearest to the robot
        """
        robot_base_pos = self._env.robots[batch_index].robot_view.base.pose[:3, 3]

        if len(self.nav_objs[batch_index]) == 1:
            return self.nav_objs[batch_index][:]

        priority = [
            (np.linalg.norm(obj.position[:2] - robot_base_pos[:2]), obj)
            for obj in self.nav_objs[batch_index]
        ]

        return [dist_obj[1] for dist_obj in sorted(priority, key=lambda x: x[0])]

    def get_nearest_nav_object(self, batch_index: int) -> MlSpacesObject:
        """Get the nearest navigation object for the given batch.

        Args:
            batch_index: Index of the environment batch

        Returns:
            The MlSpacesObject instance that is nearest to the robot
        """
        priority = self.get_nav_object_priority(batch_index)
        return priority[0] if priority else None

    def get_task_description(self) -> str:
        """Get the task description for this navigation task."""
        pickup_obj_name = self.config.task_config.pickup_obj_name

        om = self.env.object_managers[self.env.current_batch_index]

        # Get natural name if available
        try:
            object_name = om.fallback_expression(pickup_obj_name)
        except Exception:
            # Fallback to raw name if natural name lookup fails
            object_name = pickup_obj_name.replace("_", " ").title()

        # Restore word boundaries stripped from concatenated category tokens
        # (e.g. "alarmclock" -> "alarm clock").
        object_name = prettify_object_name(object_name)

        return f"Navigate to the {object_name}"

    def _create_sensor_suite_from_config(self, exp_config: MlSpacesExpConfig) -> SensorSuite:
        """Create a sensor suite from configuration using the centralized get_nav_task_sensors function."""
        from molmo_spaces.env.sensors import get_nav_task_sensors

        sensors = get_nav_task_sensors(exp_config)
        return SensorSuite(sensors)

    def calculate_distance(self, index: int) -> float:
        """Calculate the distance to the NEAREST navigation object of the target type.

        Args:
            index: Index of the environment batch

        Returns:
            Distance in meters to the nearest object of the target type
        """
        robot = self._env.robots[index]
        robot_base_pose = robot.robot_view.base.pose
        robot_base_pos = robot_base_pose[:3, 3]

        # Get the nearest object dynamically
        nearest_obj = self.get_nearest_nav_object(index)

        center_dist = float(np.linalg.norm(nearest_obj.position[:2] - robot_base_pos[:2]))
        if not self.config.task_config.succ_use_surface_distance:
            return center_dist

        # Surface distance: subtract the object's in-plane reach so that being
        # right at a large object counts as "arrived" even if its centre is far.
        # ``aabb_size`` is the half-extent (local frame, initial pose); the larger
        # in-plane half-side is a robust effective radius (~0 for small objects, so
        # this is a no-op there). Clamp at 0 (never report negative distance).
        effective_radius = float(np.max(np.asarray(nearest_obj.aabb_size)[:2]))
        return max(0.0, center_dist - effective_radius)

    def check_object_visible(self, index: int) -> bool:
        """Check if the nearest navigation object is visible from the success
        camera(s). ORs over ``visibility_camera_names`` when set (e.g. the
        shoulder_left/right pair), otherwise uses ``visibility_camera_name``."""
        nearest_obj = self.get_nearest_nav_object(index)

        # Use the registry camera name(s) (e.g. 'shoulder_left' / 'nav_camera'),
        # not the MJCF name (e.g. 'robot_0/head_camera').
        camera_names = self.config.task_config.visibility_camera_names
        if not camera_names:
            camera_names = [self.config.task_config.visibility_camera_name]
        min_fraction = float(getattr(self.config.task_config, "min_visible_fraction", 0.0) or 0.0)
        for camera_name in camera_names:
            if self._env.check_visibility(camera_name, nearest_obj.name) > min_fraction:
                return True
        return False

    def get_reward(self) -> np.ndarray:
        """Calculate reward based on distance to target object.

        Returns:
            Array of rewards for each environment in the batch
        """
        rewards = []

        for i in range(self._env.n_batch):
            object_visible = (
                self.check_object_visible(i)
                if self.config.task_config.require_object_visible
                else True
            )

            if self.config.task_config.succ_use_goal_pose and self._planned_goal_xy is not None:
                # Goal-pose-reaching success (pre-grasp semantics): the robot must
                # arrive at the planned pre-grasp goal pose, and that goal must be a
                # valid pre-grasp (within ``max_pregrasp_standoff_m`` of the object
                # surface). Decoupled from object-centre distance so on-furniture
                # targets, whose closest navigable pose is the furniture edge, are
                # judged on positioning quality rather than an unreachable distance.
                tc = self.config.task_config
                goal_standoff = self.planned_goal_surface_distance(i)
                reached = (
                    self.distance_to_goal(i) <= tc.succ_goal_pos_threshold
                    and self.goal_yaw_error(i) <= tc.succ_goal_yaw_threshold
                    and goal_standoff <= tc.max_pregrasp_standoff_m
                )
                rewards.append(1.0 if (reached and object_visible) else 0.0)
                continue

            # Distance-to-object success (legacy / when no goal is published).
            distance = self.calculate_distance(i)
            if not object_visible:
                reward = 0.0
            else:
                # Linearly scale reward from 1 → 0 as distance goes from 0 → threshold
                reward = max(0.0, 1.0 - distance / self.config.task_config.succ_pos_threshold)
                # TODO this can be detrimental for RL training, as the reward goes up for locations
                #  with potentially vanishing visibility. We might want to make it maximum for
                #  distances under some smaller threshold than `succ_pos_threshold`

            rewards.append(reward)

        return np.array(rewards, dtype=np.float32)

    def judge_success(self) -> bool:
        """Judge whether the task is successfully completed.

        Returns:
            Boolean indicating success for the first environment
        """
        success = self.get_reward()[0] > 0.0
        if not success:
            distance = self.calculate_distance(0)
            object_visible = self.check_object_visible(0)
            if self.config.task_config.succ_use_goal_pose and self._planned_goal_xy is not None:
                log.info(
                    f"[Nav fail] goal_dist={self.distance_to_goal(0):.2f}m "
                    f"yaw_err={self.goal_yaw_error(0):.2f}rad "
                    f"goal_standoff={self.planned_goal_surface_distance(0):.2f}m "
                    f"obj_dist={distance:.2f}m visible={object_visible}"
                )
            else:
                log.info(f"[Nav fail] Distance: {distance:.2f}m, Object visible: {object_visible}")

        return success

    def get_info(self) -> list[dict[str, Any]]:
        """Get additional metrics for each environment."""
        metrics = []

        # Calculate rewards once for all environments
        rewards = self.get_reward()

        for i in range(self._env.n_batch):
            distance = self.calculate_distance(i)
            # Use pre-calculated reward
            success = rewards[i] > 0.0

            metrics.append(
                {
                    "position_error": distance,
                    "success": success,
                    "episode_step": self.episode_step_count,
                }
            )

        return metrics

    def get_obs_scene(self):
        """
        This is for observations that are constant over all time steps of an env.
        """
        # Get base observation from parent class (includes frozen_config handling)
        obs_scene = super().get_obs_scene()

        text = self.config.task_type + " " + self.config.task_config.pickup_obj_name
        obs_extra = dict(text=text, object_name=self.config.task_config.pickup_obj_name)
        obs_scene.update(obs_extra)

        return obs_scene
