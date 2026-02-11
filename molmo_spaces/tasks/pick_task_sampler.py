import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mujoco import MjSpec, mjtGeom

from molmo_spaces.env.arena.arena_utils import modify_mjmodel_thor_articulated
from molmo_spaces.env.data_views import (
    MlSpacesArticulationObject,
    MlSpacesObject,
    create_mlspaces_body,
)
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.env.object_manager import Context
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.tasks.task_sampler import (
    BaseMujocoTaskSampler,
)
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask, RobotPlacementError
from molmo_spaces.utils.asset_names import get_thor_name
from molmo_spaces.utils.grasp_sample import (
    get_noncolliding_grasp_mask,
    has_grasp_folder,
    has_valid_grasp_file,
    load_grasps_for_object,
)
from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom
from molmo_spaces.utils.pose import pos_quat_to_pose_mat, pose_mat_to_7d

if TYPE_CHECKING:
    from molmo_spaces.configs.base_pick_config import PickBaseConfig


log = logging.getLogger(__name__)


class RolloutFailure(Exception):
    """Exception for when scene setup fails."""

    pass


class PickTaskSampler(BaseMujocoTaskSampler):
    """
    Default task sampler for pick tasks with house iteration control.
    House order (`house_inds`) and samples per house are provided via config.
    """

    def __init__(self, config: "PickBaseConfig") -> None:
        super().__init__(config)
        self.candidate_objects: None | list[MlSpacesObject] = None
        self._task_counter = None  # Track tasks within the same house for variety
        self._grasp_failure_counts: dict[str, int] = {}  # Track grasp failures per object name

        # If pickup_types is None, default to empty list which matches any object type.
        # Objects are then filtered by grasp file availability in _get_scene_objects().
        if config.task_sampler_config.pickup_types is None:
            config.task_sampler_config.pickup_types = []

    def _remove_candidate_object(self, obj_name: str) -> None:
        """Remove an object from candidate_objects list."""
        if self.candidate_objects is not None:
            original_len = len(self.candidate_objects)
            self.candidate_objects = [obj for obj in self.candidate_objects if obj.name != obj_name]
            if len(self.candidate_objects) < original_len:
                log.info(
                    f"Removed {obj_name} from candidates, {len(self.candidate_objects)} remaining"
                )

    def report_grasp_failure(self, obj_name: str, max_failures: int = 2) -> None:
        """Report a grasp failure for an object. Remove from candidates if threshold exceeded.

        Args:
            obj_name: Name of the object that failed grasp finding
            max_failures: Remove object after this many failures (default 2)
        """
        self._grasp_failure_counts[obj_name] = self._grasp_failure_counts.get(obj_name, 0) + 1
        count = self._grasp_failure_counts[obj_name]
        if count > max_failures:
            self._remove_candidate_object(obj_name)
            log.info(f"Removed {obj_name} after {count} grasp failures (threshold: {max_failures})")

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        """Use this function to put task specific assets into the scene."""
        self.config.policy_config.policy_cls.add_auxiliary_objects(self.config, spec)

    def init_scene(self, env) -> None:
        # initialize randomizers here
        super().init_scene(env)

        log.info(
            f"Setting up scene for house {self.current_house_index}, task {self._task_counter}..."
        )
        model = env.mj_model
        data = env.mj_datas[0]
        modify_mjmodel_thor_articulated(model, data)

        # New house - reset counters
        self._task_counter = 0
        self._grasp_failure_counts = {}
        log.debug(f"New house {self.current_house_index} - resetting object tracking")

        # Shuffle order deterministically per house/task for variety
        candidate_objects = self._get_scene_objects(env)
        candidate_objects = self.balance_sample_names(candidate_objects)
        np.random.shuffle(candidate_objects)
        self.candidate_objects = candidate_objects

    def randomize_scene(self, env: CPUMujocoEnv, robot_view) -> None:
        """Setup scene state: robot joints, texture randomization, cameras."""
        # randomize scene here
        super().randomize_scene(env, robot_view)

        model = env.current_model
        data = env.current_data
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        # Set robot joints
        for group_name, qpos in self.config.robot_config.init_qpos.items():
            qpos = np.array(qpos)
            if (
                self.config.robot_config.init_qpos_noise_range is not None
                and group_name in self.config.robot_config.init_qpos_noise_range
            ):
                noise_mag = np.array(self.config.robot_config.init_qpos_noise_range[group_name])
                perturb = np.random.uniform(-noise_mag, noise_mag)
            else:
                perturb = np.zeros_like(qpos)
            robot_view.get_move_group(group_name).joint_pos = qpos + perturb

        # robot_color = None
        # robot_color = [.941, .322, .612,1.]  # example: red
        # if robot_color:
        #     # Get robot geometry ids
        #     robot_geoms = descendant_geoms(
        #         self.env._mj_model,
        #         self.env.current_robot.robot_view.base.root_body_id,
        #     )
        #     # Set color
        #     for geom_id in robot_geoms:
        #         model.geom_rgba[geom_id] = robot_color
        log.info("Scene setup completed.\n")

    def get_workspace_center(self, env: CPUMujocoEnv) -> np.ndarray:
        """Get workspace center for camera placement.

        For move-to-pose tasks, computes the average of:
        - Pickup object position
        - Place target position (or goal pose)
        - Robot gripper position
        """
        # Need current task config to get pickup object name
        if (
            not hasattr(self.config.task_config, "pickup_obj_name")
            or not self.config.task_config.pickup_obj_name
        ):
            # Fall back to default implementation
            return super().get_workspace_center(env)

        try:
            data = env.current_data
            om = env.object_managers[env.current_batch_index]
            pickup_obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)

            # Get place target position
            if self.config.task_sampler_config.place_target_name is not None:
                place_target = create_mlspaces_body(
                    data, self.config.task_sampler_config.place_target_name
                )
                place_target_position = place_target.position
            else:
                place_target_position = self.config.task_config.pickup_obj_goal_pose[:3]

            # Get gripper position
            robot_view = env.current_robot.robot_view
            gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
            ee_pose_rel_to_base = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_robot
            gripper_world_pose = robot_view.base.pose @ ee_pose_rel_to_base
            gripper_pos = gripper_world_pose[:3, 3]

            # Compute workspace center
            workspace_center = (pickup_obj.position + place_target_position + gripper_pos) / 3.0
            return workspace_center
        except Exception as e:
            log.debug(f"[CAMERA SETUP] Could not compute workspace center: {e}, using default")
            return super().get_workspace_center(env)

    def resolve_visibility_object(self, env: CPUMujocoEnv, key: str) -> list[str]:
        """Resolve special visibility object keys.

        Handles:
        - __task_objects__: Current pickup object
        - __gripper__: Robot gripper (via base class)
        """
        if key == "__task_objects__":
            if (
                hasattr(self.config.task_config, "pickup_obj_name")
                and self.config.task_config.pickup_obj_name
            ):
                return [self.config.task_config.pickup_obj_name]
            return []

        # Delegate to base class for other keys (e.g., __gripper__)
        return super().resolve_visibility_object(env, key)

    def _sample_task(self, env: CPUMujocoEnv) -> PickTask:
        """Sample a pick-and-place task configuration and create the task."""
        # Set current batch index to 0 (most common case for single-batch environments)
        # TODO(rose) at some point: handle multi-batch environments properly
        assert env.current_batch_index == 0
        assert self.candidate_objects is not None and len(self.candidate_objects) > 0

        om = env.object_managers[env.current_batch_index]

        keep_task_cfg = self.config.task_config.pickup_obj_name is not None

        sample_success = False
        max_attempts = len(self.candidate_objects)
        attempts = 0
        while not sample_success and len(self.candidate_objects) > 0 and attempts < max_attempts:
            attempts += 1

            if not keep_task_cfg:
                self.config.task_config.pickup_obj_name = None

            # Sample pickup object
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("sample_select_object")

            if self.config.task_config.pickup_obj_name is None:
                object_index = self._task_counter % len(self.candidate_objects)
                self.config.task_config.pickup_obj_name = self.candidate_objects[object_index].name
                log.info(
                    f"Attempting object {self.config.task_config.pickup_obj_name} {object_index}/{len(self.candidate_objects)}"
                )
            else:
                log.info(
                    f"Attempting object {self.config.task_config.pickup_obj_name} of {len(self.candidate_objects)}"
                )

            self._task_counter += 1  # update counter, so we don't re-try same object

            if self._datagen_profiler is not None:
                self._datagen_profiler.end("sample_select_object")

            # Setup cameras initially so they are available for visibility checks during placement
            # We will run setup_cameras again after placement to correct for final robot position
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("sample_cameras_initial")

            self.setup_cameras(env, deterministic_only=True)

            if self._datagen_profiler is not None:
                self._datagen_profiler.end("sample_cameras_initial")

            # Assuming a bench context is meaningful for the task type,
            # so we need to find the supporting bench for the context
            pickup_obj_name = self.config.task_config.pickup_obj_name
            pickup_obj_id = om.get_object_body_id(pickup_obj_name)

            supporting_geom_id = get_supporting_geom(env.current_data, pickup_obj_id)
            if supporting_geom_id is None or supporting_geom_id < 1:
                log.info(f"Failed to get a valid supporting geom_id for {pickup_obj_name}")
                # Remove this object - no supporting surface won't change on retry
                self._remove_candidate_object(pickup_obj_name)
                continue

            #  place robot accordingly
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("sample_place_robot")

            try:
                self._sample_and_place_robot(env)
            except RobotPlacementError as e:
                log.info(f"Robot placement failed for {pickup_obj_name}: {e}")
                # Report failure for this asset (may lead to dynamic blacklisting)
                asset_uid = self.get_asset_uid_from_object(env, pickup_obj_name)
                if asset_uid:
                    self.report_asset_failure(asset_uid, f"robot placement failed: {e}")
                # Remove this object from candidates - don't retry failed placements
                self._remove_candidate_object(pickup_obj_name)
                continue

            if self._datagen_profiler is not None:
                self._datagen_profiler.end("sample_place_robot")

            # Ensure robot is in final position before camera setup
            mujoco.mj_forward(env.current_model, env.current_data)

            # Check grasp feasibility before proceeding
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("sample_check_grasps")

            pickup_obj = om.get_object_by_name(pickup_obj_name)
            asset_uid = self.get_asset_uid_from_object(env, pickup_obj_name)
            if asset_uid:
                try:
                    _gripper, cached_grasps = load_grasps_for_object(asset_uid, 512)
                    if len(cached_grasps) > 0:
                        object_pose = pos_quat_to_pose_mat(pickup_obj.position, pickup_obj.quat)
                        grasp_poses_world = object_pose @ cached_grasps
                        noncolliding_mask = get_noncolliding_grasp_mask(
                            env.current_model, env.current_data, grasp_poses_world, 64
                        )
                        n_feasible = int(np.sum(noncolliding_mask))
                        if n_feasible == 0:
                            log.info(
                                f"No feasible grasps for {pickup_obj_name} (uid={asset_uid}): "
                                f"0/{len(grasp_poses_world)} non-colliding"
                            )
                            if self._datagen_profiler is not None:
                                self._datagen_profiler.end("sample_check_grasps")
                            self.report_grasp_failure(pickup_obj_name)
                            continue
                except ValueError:
                    log.info(f"No grasps found for {pickup_obj_name} (uid={asset_uid})")
                    if self._datagen_profiler is not None:
                        self._datagen_profiler.end("sample_check_grasps")
                    self.report_grasp_failure(pickup_obj_name)
                    continue

            if self._datagen_profiler is not None:
                self._datagen_profiler.end("sample_check_grasps")

            # Setup cameras after pickup object and robot placement
            # This allows cameras to use task-specific info (pickup object, workspace center)
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("sample_cameras_final")

            self.setup_cameras(env)

            if self._datagen_profiler is not None:
                self._datagen_profiler.end("sample_cameras_final")

            # Note: generating referral expressions for sampling
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("generate_context_expressions")

            bench_geom_body_id = env.mj_model.geom_bodyid[supporting_geom_id]
            context_objects = om.get_context_objects(
                pickup_obj_name, Context.BENCH, bench_geom_ids=[bench_geom_body_id]
            )

            context_names = {obj.name for obj in context_objects}
            if pickup_obj_name not in context_names:
                context_objects.append(om.get_object(pickup_obj_name))

            try:
                expression_priority = om.referral_expression_priority(
                    pickup_obj_name, context_objects
                )
                filtered_expression_priority = om.thresholded_expression_priority(
                    expression_priority
                )
            except NameError:
                pass

            if self._datagen_profiler is not None:
                self._datagen_profiler.end("generate_context_expressions")

            # if len(filtered_expression_priority) == 0:
            #     log.info(
            #         f"Skipped {pickup_obj_name} with no filtered expression priorities out of {expression_priority}"
            #     )
            #     # Remove this object - expression priority won't change on retry
            #     self._remove_candidate_object(pickup_obj_name)
            #     continue

            sample_success = True
            break

        if not sample_success:
            raise HouseInvalidForTask(
                f"Unable to sample a valid task after {attempts} attempts, {len(self.candidate_objects)} candidates remaining"
            )

        if self._datagen_profiler is not None:
            self._datagen_profiler.start("sample_context_expressions")

        self.config.task_config.referral_expressions["pickup_obj_name"] = om.sample_expression(
            filtered_expression_priority
        )
        self.config.task_config.referral_expressions_priority["pickup_obj_name"] = (
            expression_priority
        )

        if self._datagen_profiler is not None:
            self._datagen_profiler.end("sample_context_expressions")

        if self._datagen_profiler is not None:
            self._datagen_profiler.start("sample_task_create")

        task = PickTask(env, self.config)

        if self._datagen_profiler is not None:
            self._datagen_profiler.end("sample_task_create")

        return task

    def has_valid_grasp_file(self, pickup_obj, asset_uid):
        return has_valid_grasp_file(asset_uid)

    def _get_scene_objects(self, env: CPUMujocoEnv, mass_limit=100) -> list[MlSpacesObject]:
        """
        Get the list of candidate probjects in the scene for interactions.
        Filter by object types.

        Arguments:
            env: and environment
            mass_limit: don't choose objects with mass greater than limit
            oversample_obja: oversample objaverse assets by factor n
        """
        # Discover candidate pickup objects
        om = env.object_managers[env.current_batch_index]
        candidates = om.get_objects_of_type(self.config.task_sampler_config.pickup_types)
        log.info(f"Found {len(candidates)} candidate pickup objects in the scene")

        if not len(candidates) > 0:
            log.info("[TASK SAMPLING] ⚠️ No candidate pickup objects found in the scene")
            # print all the top-level objects in the scene for debugging
            om = env.object_managers[env.current_batch_index]
            all_objects = MlSpacesObject.get_top_level_bodies(model=self.env.mj_model)
            all_non_structural_or_excluded = [
                obj for obj in all_objects if not (om.is_structural(obj) or om.is_excluded(obj))
            ]
            for b in all_non_structural_or_excluded[:30]:
                name = self.env.mj_model.body(b).name
                pos = self.env.current_data.xpos[b]
                possible_types = om.get_possible_object_types(name or "")
                log.info(
                    f"  - #{b:02d} {name} (types={possible_types}) pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                )

            # log.info(f"[TASK SAMPLING] Scene objects (no candidates): {[obj.name for obj in all_objects]}")
            raise HouseInvalidForTask("No pickup candidates found in the scene")

        model = self._env.current_model

        # #Mass computation setup
        children = {}
        for bid in range(model.nbody):
            pid = model.body_parentid[bid]  # ID of parent
            children.setdefault(pid, []).append(bid)

        def get_children(body_id):
            return children.get(body_id, [])

        candidate_objects = []
        blacklisted_count = 0
        for pickup_obj in candidates:
            # Check if grasp files exist for this object
            asset_uid = None

            if not isinstance(pickup_obj, MlSpacesArticulationObject) and not om.has_free_joint(
                pickup_obj
            ):
                log.info(f"Skipping {pickup_obj.name} (uid={asset_uid}) - static in scene")
                continue

            scene_metadata = env.current_scene_metadata
            if scene_metadata is not None:
                asset_uid = (
                    scene_metadata.get("objects", {}).get(pickup_obj.name, {}).get("asset_id", None)
                )

            if asset_uid is None:
                asset_uid = get_thor_name(model, pickup_obj)

            # Check if asset is blacklisted (static or dynamic)
            if asset_uid and self.is_asset_blacklisted(asset_uid):
                log.debug(f"Skipping {pickup_obj.name} (uid={asset_uid}) - blacklisted")
                blacklisted_count += 1
                continue

            if not has_grasp_folder(asset_uid):
                log.info(f"Skipping {pickup_obj.name} (uid={asset_uid}) - no grasp file available")
                continue

            if not self.has_valid_grasp_file(pickup_obj, asset_uid):
                log.info(
                    f"Skipping {pickup_obj.name} (uid={asset_uid}) - grasp file exists but has no valid transforms"
                )
                continue

            # Mass computation
            masses = [model.body_mass[bid] for bid in get_children(pickup_obj.object_id)]
            if np.sum(masses) > mass_limit:
                continue

            candidate_objects.append(pickup_obj)

        if blacklisted_count > 0:
            log.info(f"Skipped {blacklisted_count} blacklisted objects")
        log.info(
            f"Filtered to {len(candidate_objects)} valid candidate pickup objects, {len(candidate_objects) / len(candidates) * 100:.1f} %"
        )

        return candidate_objects

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        """Sample a pickup object and receptacle, place robot using occupancy map, and return sampled params.

        Returns:
            dict with keys: pickup_obj_name, receptacle_name, placement_region, robot_base_pose
        Raises:
            RobotPlacementError
        """
        task_cfg = self.config.task_config
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_cfg.pickup_obj_name)
        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()
        log.debug(f"Selected pickup object: {self.config.task_config.pickup_obj_name}")
        log.debug(f"[TASK SAMPLING] Trying to place robot near '{pickup_obj.name}'")

        # randomize pickup object
        if (
            self.texture_randomizer is not None
            and self.config.task_sampler_config.randomize_textures
        ):
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("robot_randomize_pickup_obj")
            self.texture_randomizer.randomize_object(pickup_obj)
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("robot_randomize_pickup_obj")

        robot_view = env.current_robot.robot_view
        if isinstance(pickup_obj, MlSpacesObject):
            target_pos = pickup_obj.position
        else:
            raise ValueError(f"Invalid pickup object type: {type(pickup_obj)}")

        initial_robot_z = (
            target_pos[2]
            + self.config.task_sampler_config.robot_object_z_offset
            + np.random.uniform(
                self.config.task_sampler_config.robot_object_z_offset_random_min,
                self.config.task_sampler_config.robot_object_z_offset_random_max,
            )
        )

        # place robot near receptacle - this is the expensive call with collision/visibility checks
        if self._datagen_profiler is not None:
            self._datagen_profiler.start("robot_place_near")
        robot_placed = env.place_robot_near(
            robot_view=robot_view,
            target=pickup_obj,
            max_tries=10,  # Use config value or reasonable default
            sampling_radius_range=self.config.task_sampler_config.base_pose_sampling_radius_range,
            robot_safety_radius=self.config.task_sampler_config.robot_safety_radius,
            preserve_z=initial_robot_z,
            face_target=True,
            check_camera_visibility=self.config.task_sampler_config.check_robot_placement_visibility,
            visibility_resolver=self.get_visibility_resolver(env),
            excluded_positions=self.used_robot_positions[pickup_obj.name],
        )
        if self._datagen_profiler is not None:
            self._datagen_profiler.end("robot_place_near")

        if not robot_placed:
            log.info(f"[TASK SAMPLING] Failed to place robot near '{pickup_obj.name}'")
            raise RobotPlacementError(f"Failed to place robot near object: {pickup_obj.name}")

        # Add successful position to cache
        self.used_robot_positions[pickup_obj.name].append(robot_view.base.pose[:3, 3])

        # Get final robot pose for return data
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

        pickup_obj_goal_pose = pose_mat_to_7d(pickup_obj.pose)
        pickup_obj_goal_pose[2] += 0.05  # 5 cm
        task_cfg.pickup_obj_goal_pose = pickup_obj_goal_pose.tolist()

        log.info(f"Supporting receptacle: {self.config.task_config.receptacle_name}")

    def _place_target_near_object(
        self, env: CPUMujocoEnv, object_pos: np.ndarray, placement_region=None
    ) -> None:
        """Place the placement target on the same receptacle as the pickup object."""
        log.debug(
            f"[TARGET POSITIONING] Finding receptacle for pickup object '{self.config.task_sampler_config.pickup_obj_name}'"
        )

        # Get the pickup object using Object class
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(self.config.task_sampler_config.pickup_obj_name)

        support_name = None
        if placement_region is None:
            support_name = pickup_obj.get_support_below()
            if support_name is None:
                log.debug(
                    "[TARGET POSITIONING] ⚠️ No support found for pickup object, falling back to simple positioning"
                )

                # Fallback to original simple positioning if no support found
                offset_distance = 0.3
                offset_angle = np.random.uniform(0, 2 * np.pi)
                target_x = object_pos[0] + offset_distance * np.cos(offset_angle)
                target_y = object_pos[1] + offset_distance * np.sin(offset_angle)
                target_z = np.clip(object_pos[2], 0.7, 1.2)
                place_target = create_mlspaces_body(
                    env.current_data, self.config.task_sampler_config.place_target_name
                )
                place_target.position = [target_x, target_y, target_z]
                return

            log.debug(f"[TARGET POSITIONING] Pickup object is on receptacle: '{support_name}'")

            om = env.object_managers[env.current_batch_index]
            receptacle_obj = om.get_object_by_name(support_name)
            placement_region = receptacle_obj.compute_placement_region()

        xy_min = placement_region["xy_min"]
        xy_max = placement_region["xy_max"]
        top_z = placement_region["top_z"]

        log.debug(
            f"[TARGET POSITIONING] Receptacle placement region: xy_min=({xy_min[0]:.3f}, {xy_min[1]:.3f}), xy_max=({xy_max[0]:.3f}, {xy_max[1]:.3f}), top_z={top_z:.3f}"
        )

        # Apply minimum separation constraint from pickup object
        max_attempts = 50
        min_separation = self.config.task_sampler_config.min_object_separation
        pickup_xy = pickup_obj.position[:2]

        for attempt in range(max_attempts):
            # Sample a random position within the receptacle's placement region
            target_x = np.random.uniform(xy_min[0], xy_max[0])
            target_y = np.random.uniform(xy_min[1], xy_max[1])
            target_xy = np.array([target_x, target_y])

            # Check minimum separation from pickup object
            separation = np.linalg.norm(target_xy - pickup_xy)
            if separation >= min_separation:
                break

            log.debug(
                f"[TARGET POSITIONING]   Attempt {attempt + 1}: separation {separation:.3f}m < {min_separation:.3f}m, retrying..."
            )

        else:
            # If we couldn't find a position with minimum separation, use the last attempt
            log.debug(
                f"[TARGET POSITIONING] ⚠️ Could not achieve minimum separation of {min_separation:.3f}m after {max_attempts} attempts, using separation {separation:.3f}m"
            )

        # Position target slightly above the receptacle surface
        target_z = top_z + 0.01  # 1cm above surface to avoid z-fighting/embedding

        # Set placement target position
        place_target = create_mlspaces_body(
            env.current_data, self.config.task_sampler_config.place_target_name
        )
        place_target.position = [target_x, target_y, target_z]
        mujoco.mj_forward(env.current_model, env.current_data)

        distance_to_pickup = np.linalg.norm(
            np.array([target_x, target_y, target_z]) - pickup_obj.position
        )
        log.debug(
            f"[TARGET POSITIONING] Positioned target at ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
        )
        log.debug(f"[TARGET POSITIONING] Distance to pickup object: {distance_to_pickup:.3f}m")
        log.debug(
            f"[TARGET POSITIONING] XY separation: {separation:.3f}m (min: {min_separation:.3f}m)"
        )
        if support_name:
            log.debug(
                f"[TARGET POSITIONING] Target on receptacle '{support_name}' at height {target_z:.3f}m"
            )

    @staticmethod
    def add_placement_target(
        spec: MjSpec, pos=None, randomize=False, name="place_target"
    ) -> MjSpec:
        """
        Add a placement target (red cylinder) to the scene.

        Args:
            spec: MuJoCo MjSpec object
            pos: [x, y, z] position. If None, uses default or random
            randomize: Whether to randomize the position
            name: Name for the target body

        Returns:
            spec: Updated MjSpec with placement target added
        """

        if pos is None:
            if randomize:
                # Random position on table surface (approximate table bounds)
                pos = [
                    np.random.uniform(-0.4, 0.4),  # x: table width
                    np.random.uniform(0.2, 1.0),  # y: table depth
                    0.71,  # z: table height
                ]
            else:
                pos = [-0.1, 0.4, 0.71]  # Default position from XML

        # Create target body
        target_body = spec.worldbody.add_body(name=name, pos=pos, mocap=True)

        # Add red cylinder geometry
        target_body.add_geom(
            name=f"{name}_geom",  # Add geometry name for identification
            type=mjtGeom.mjGEOM_CYLINDER,
            size=[0.05, 0.001, 0],  # For cylinders in Python API: [radius, radius, half-height]
            rgba=[1, 0, 0, 1],  # Red color
            group=2,  # Visual group
        )

        return spec

    @staticmethod
    def add_pickup_target(
        spec: MjSpec, pos=None, randomize=False, name="obj_0", color=[0, 1, 0, 1]
    ) -> MjSpec:
        """
        Add a pickup target (cube) to the scene.

        Args:
            spec: MuJoCo MjSpec object
            pos: [x, y, z] position. If None, uses default or random
            randomize: Whether to randomize the position
            name: Name for the object body
            color: RGBA color for the cube

        Returns:
            spec: Updated MjSpec with pickup object added
        """
        if pos is None:
            if randomize:
                pos = [np.random.uniform(-0.4, 0.4), np.random.uniform(0.2, 1.0), 0.735]
            else:
                pos = [6.0, 3.0, 0.76]

        obj_body = spec.worldbody.add_body(name=name, pos=pos)
        obj_body.add_freejoint()
        obj_body.add_geom(
            name=f"{name}_geom",
            type=mjtGeom.mjGEOM_BOX,
            pos=[0, 0, 0.025],
            size=[0.025, 0.025, 0.025],
            rgba=color,
        )

        return spec
